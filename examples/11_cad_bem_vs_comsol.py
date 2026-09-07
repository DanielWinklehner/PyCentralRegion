"""
11_cad_bem_vs_comsol.py - BEM of the ORIGINAL cavity center (from the COMSOL
vacuum STEP) against the COMSOL RF field: the first milestone of the composite
central-region field.

Pipeline
  1. Import the vacuum STEP (quarter sector, grounded sector walls), crop to a
     cylinder r <= R_CUT, |z| <= Z_CUT and CUT the metal out of it: closed
     dee and ground solids with artificial cap faces on the crop surface
     (open truncation - the comparison stays inside r <= R_CMP).
  2. Surface-mesh with gmsh, graded by distance from the dee edge curves and
     the central ground features (posts, plug); caps and far plate flats
     coarse. The real 5 mm fillets and the reduced dee height at the tip are
     in the CAD and therefore in the mesh.
  3. Solve the Laplace Dirichlet problem (dee = 1 V, ground = 0) with
     gap_fields.solve_gap_field (OpenCL assembly / Jacobi GMRES).
  4. Evaluate E_BEM at the COMSOL grid points on several z planes by central
     differences of the single-layer potential, fix the (arbitrary) COMSOL
     amplitude by a least-squares scale over the annulus at the midplane,
     and compare: maps, annulus RMS per plane, across-gap profiles, gap
     voltage vs radius.

Usage:
    python 11_cad_bem_vs_comsol.py [--mesh-only] [--size-min 2.0] [--size-max 12]
Env: CONDA_PREFIX=<accel-dev-env> MPLBACKEND=Agg NUMBA_NUM_THREADS=10
Outputs in output/: cad_bem_quarter_mesh.npz, cad_bem_quarter_solution.npz,
cad_bem_vs_comsol.json, cad_bem_vs_comsol.png
"""
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

STEP = (r"D:\MIT Dropbox\Daniel Winklehner\Projects\IsoDAR\60 MeV Cyclotron"
        r"\MainAcceleration\IBA_PDR_Cavity_Design_from_COMSOL.step")
FIELD = ROOT / 'resources' / 'isodar_cavity_efield_center_2mm.npz'
OUT = ROOT / 'output'
R_CUT, Z_CUT = 250.0, 60.0          # crop cylinder [mm]
R_CMP = 240.0                       # comparison radius [mm]
R_ANN = (150.0, 230.0)              # annulus for the amplitude fit / RMS [mm]
Z_PLANES = (0.0, 10.0, 20.0, 28.0)  # COMSOL planes compared [mm]
PROBE_H = 0.25e-3                   # central-difference step [m]


# ============================================================================
# 1-2. CAD -> metal solids -> graded surface mesh
# ============================================================================
def build_mesh(size_min, size_max, dist_min=4.0, dist_max=50.0, r_fine_ground=170.0,
               verbose=True):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(STEP)
    gmsh.model.occ.synchronize()
    cyl = gmsh.model.occ.addCylinder(0, 0, -Z_CUT, 0, 0, 2 * Z_CUT, R_CUT)
    metal, _ = gmsh.model.occ.cut([(3, cyl)], [(3, 1)], removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()
    vols = [t for d, t in metal if d == 3]
    dee_vols, gnd_vols = [], []
    for t in vols:
        b = gmsh.model.getBoundingBox(3, t)
        (dee_vols if max(abs(b[2]), abs(b[5])) < Z_CUT - 1.0 else gnd_vols).append(t)
        if verbose:
            print(f"[mesh] volume {t}: x {b[0]:7.1f}..{b[3]:7.1f} y {b[1]:7.1f}..{b[4]:7.1f} "
                  f"z {b[2]:6.1f}..{b[5]:6.1f}  mass {gmsh.model.occ.getMass(3, t):.3e} mm3")
    if len(dee_vols) != 1:
        raise RuntimeError(f"expected exactly one dee volume, got {dee_vols}")
    print(f"[mesh] dee volume {dee_vols}, ground volumes {gnd_vols}")

    def faces_of(vlist):
        return sorted({f for d, f in gmsh.model.getBoundary(
            [(3, t) for t in vlist], oriented=False, recursive=False)})

    dee_faces, gnd_faces = faces_of(dee_vols), faces_of(gnd_vols)

    def curves_of(faces, r_max):
        cs = set()
        for f in faces:
            for d, c in gmsh.model.getBoundary([(2, f)], oriented=False, recursive=False):
                b = gmsh.model.getBoundingBox(1, c)
                if np.hypot(max(abs(b[0]), abs(b[3])), max(abs(b[1]), abs(b[4]))) < r_max:
                    cs.add(abs(c))
        return sorted(cs)

    curves = curves_of(dee_faces, R_CUT - 5.0) + curves_of(gnd_faces, r_fine_ground)
    print(f"[mesh] size field on {len(curves)} curves; dee faces {len(dee_faces)}, "
          f"ground faces {len(gnd_faces)}")
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", curves)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 300)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", size_min)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", size_max)
    gmsh.model.mesh.field.setNumber(2, "DistMin", dist_min)
    gmsh.model.mesh.field.setNumber(2, "DistMax", dist_max)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.6)
    gmsh.model.addPhysicalGroup(2, dee_faces, tag=1)
    gmsh.model.addPhysicalGroup(2, gnd_faces, tag=2)
    t0 = time.time()
    gmsh.model.mesh.generate(2)
    ntag, ncoord, _ = gmsh.model.mesh.getNodes()
    verts = ncoord.reshape(-1, 3)
    remap = {int(t): i for i, t in enumerate(ntag)}
    tris, pots = [], []
    for grp, pot in ((1, 1.0), (2, 0.0)):
        for f in gmsh.model.getEntitiesForPhysicalGroup(2, grp):
            etypes, etags, enodes = gmsh.model.mesh.getElements(2, f)
            for ty, nodes in zip(etypes, enodes):
                if ty == 2:
                    t = np.vectorize(remap.get)(np.asarray(nodes).reshape(-1, 3))
                    tris.append(t)
                    pots.append(np.full(len(t), pot))
    gmsh.finalize()
    tris = np.vstack(tris)
    pots = np.concatenate(pots)
    # drop unreferenced nodes
    used = np.unique(tris)
    idx = -np.ones(len(verts), dtype=int)
    idx[used] = np.arange(len(used))
    verts, tris = verts[used], idx[tris]
    areas = 0.5 * np.linalg.norm(np.cross(verts[tris[:, 1]] - verts[tris[:, 0]],
                                          verts[tris[:, 2]] - verts[tris[:, 0]]), axis=1)
    print(f"[mesh] {len(tris)} triangles ({int((pots > 0).sum())} dee, "
          f"{int((pots == 0).sum())} ground), {len(verts)} nodes, meshed in "
          f"{time.time() - t0:.0f} s; edge ~ {np.sqrt(areas.min() / 0.433):.2f}..{np.sqrt(areas.max() / 0.433):.1f} mm, "
          f"{int((areas < 1e-6).sum())} degenerate")
    return verts * 1e-3, tris, pots, areas * 1e-6


# ============================================================================
# 4. comparison helpers
# ============================================================================
def efield_at(sol, pts_m, h=PROBE_H, chunk=4000):
    """E [V/m per volt of dee potential] at (M, 3) points by central differences."""
    m = len(pts_m)
    probe = np.empty((6 * m, 3))
    for k in range(3):
        d = np.zeros(3); d[k] = h
        probe[2 * k * m:(2 * k + 1) * m] = pts_m + d
        probe[(2 * k + 1) * m:(2 * k + 2) * m] = pts_m - d
    phi = sol.potential(probe, chunk=chunk, verbose=False)
    e = np.empty((m, 3))
    for k in range(3):
        e[:, k] = -(phi[2 * k * m:(2 * k + 1) * m] - phi[(2 * k + 1) * m:(2 * k + 2) * m]) / (2 * h)
    return e


def arc_voltage(field_xy, x, y, r_m, az0, az1, n=1500):
    """-int E.dl along the arc r_m from az0 to az1 [deg] through a midplane field
    sampled on the (y, x) grid (nearest node). field_xy: (ny, nx, 3), NaN = metal."""
    th = np.radians(np.linspace(az0, az1, n))
    px, py = r_m * np.cos(th), r_m * np.sin(th)
    dx = x[1] - x[0]
    ix = np.clip(np.round(px / dx).astype(int), 0, len(x) - 1)
    iy = np.clip(np.round(py / dx).astype(int), 0, len(y) - 1)
    ex, ey = field_xy[iy, ix, 0], field_xy[iy, ix, 1]
    et = -ex * np.sin(th) + ey * np.cos(th)
    et = np.where(np.isfinite(et), et, 0.0)
    return -float(np.sum(0.5 * (et[1:] + et[:-1]) * np.diff(th))) * r_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh-only', action='store_true')
    ap.add_argument('--size-min', type=float, default=2.0)
    ap.add_argument('--size-max', type=float, default=12.0)
    ap.add_argument('--reuse-mesh', action='store_true')
    ap.add_argument('--reuse-solution', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    mesh_file = OUT / 'cad_bem_quarter_mesh.npz'
    sol_file = OUT / 'cad_bem_quarter_solution.npz'

    if args.reuse_mesh and mesh_file.exists():
        d = np.load(mesh_file)
        verts, tris, pots, areas = d['vertices'], d['triangles'], d['potentials'], d['areas']
        print(f"[mesh] reused {len(tris)} triangles from {mesh_file.name}")
    else:
        verts, tris, pots, areas = build_mesh(args.size_min, args.size_max)
        np.savez(mesh_file, vertices=verts, triangles=tris, potentials=pots, areas=areas,
                 r_cut=R_CUT * 1e-3, z_cut=Z_CUT * 1e-3, size_min=args.size_min,
                 size_max=args.size_max)
    if args.mesh_only:
        return

    from PyCentralRegion.gap_fields import ElectrodeModel, solve_gap_field, GapFieldSolution
    model = ElectrodeModel(vertices=verts, triangles=tris, potentials=pots, wedges=[],
                           params={'source': 'COMSOL vacuum STEP, cropped',
                                   'r_cut': R_CUT * 1e-3, 'z_cut': Z_CUT * 1e-3})
    if args.reuse_solution and sol_file.exists():
        import bempp_cl.api as bempp
        from PyCentralRegion.gap_fields import _bempp
        bempp = _bempp()
        grid = bempp.Grid(verts.T.copy(), tris.T.astype(np.uint32).copy())
        space = bempp.function_space(grid, "DP", 0)
        coef = np.load(sol_file)['neumann']
        sol = GapFieldSolution(model=model, space=space,
                               neumann=bempp.GridFunction(space, coefficients=coef),
                               gmres_info=0, solve_time_s=0.0)
        print("[solve] reused solution")
    else:
        t0 = time.time()
        sol = solve_gap_field(model, tol=1e-5, verbose=True)
        print(f"[solve] {sol.solver} on {sol.device_interface}: assembly {sol.assembly_time_s:.1f} s, "
              f"solve {sol.solve_time_s:.1f} s, total {time.time() - t0:.0f} s")
        np.savez(sol_file, neumann=np.asarray(sol.neumann.coefficients, dtype=float),
                 solver=sol.solver, device_interface=sol.device_interface,
                 assembly_time_s=sol.assembly_time_s, solve_time_s=sol.solve_time_s,
                 n_iterations=sol.n_iterations)

    # ---- sanity: interior potentials
    p_dee = np.array([[0.150 * np.cos(np.deg2rad(45)), 0.150 * np.sin(np.deg2rad(45)), 0.030]])  # inside dee plate
    p_gnd = np.array([[0.040, 0.040, 0.0]])                                                       # inside plug
    phi_chk = sol.potential(np.vstack([p_dee, p_gnd]))
    print(f"[check] phi inside dee plate {phi_chk[0]:.4f} (want 1), inside plug {phi_chk[1]:.4f} (want 0)")

    # ---- COMSOL field
    d = np.load(FIELD)
    x, y, z, Ec = d['x'], d['y'], d['z'], d['E']          # [z, y, x, comp], NaN = metal
    dx = x[1] - x[0]
    nx = int(np.round(R_CMP * 1e-3 / dx)) + 1
    x, y, Ec = x[:nx], y[:nx], Ec[:, :nx, :nx, :]
    gx, gy = np.meshgrid(x, y, indexing='xy')             # gx[iy, ix]
    rr = np.hypot(gx, gy)
    results = {'planes': {}, 'n_elements': int(len(tris))}

    # amplitude scale: least squares over the annulus at the midplane
    izs = {zp: int(np.argmin(np.abs(z - zp * 1e-3))) for zp in Z_PLANES}
    fields_b = {}
    t0 = time.time()
    for zp, iz in izs.items():
        ec = Ec[iz]
        ok = np.isfinite(ec[:, :, 0]) & (rr <= R_CMP * 1e-3)
        pts = np.column_stack([gx[ok], gy[ok], np.full(ok.sum(), z[iz])])
        eb = np.full(ec.shape, np.nan)
        eb[ok] = efield_at(sol, pts)
        fields_b[zp] = eb
        print(f"[eval] z = {zp:4.1f} mm: {ok.sum()} vacuum points evaluated "
              f"({time.time() - t0:.0f} s elapsed)")
    ec0, eb0 = Ec[izs[0.0]], fields_b[0.0]
    ann = (rr >= R_ANN[0] * 1e-3) & (rr <= R_ANN[1] * 1e-3) & np.isfinite(ec0[:, :, 0]) & np.isfinite(eb0[:, :, 0])
    scale = float(np.sum(ec0[ann] * eb0[ann]) / np.sum(eb0[ann] * eb0[ann]))
    # arc-integral scale: COMSOL gap voltage / BEM gap voltage, low gap, r = 150..230
    v_c, v_b = [], []
    for r_mm in (150, 175, 200, 230):
        v_c.append(arc_voltage(ec0, x, y, r_mm * 1e-3, 10.0, 45.0))
        v_b.append(arc_voltage(eb0, x, y, r_mm * 1e-3, 10.0, 45.0))
    scale_arc = float(np.mean(v_c) / np.mean(v_b))
    print(f"[scale] least-squares amplitude {scale:.4f}, arc-integral amplitude {scale_arc:.4f} "
          f"(BEM gap voltage {np.mean(v_b):.4f} V per volt of dee potential)")
    results['scale_lsq'] = scale
    results['scale_arc'] = scale_arc
    results['gap_voltage_bem_per_volt'] = [float(v) for v in v_b]
    results['gap_voltage_comsol'] = [float(v) for v in v_c]

    # per-plane statistics
    for zp, iz in izs.items():
        ec, eb = Ec[iz], fields_b[zp] * scale
        ok = np.isfinite(ec[:, :, 0]) & np.isfinite(eb[:, :, 0])
        diff = np.linalg.norm(eb - ec, axis=2)
        mag_c = np.linalg.norm(ec, axis=2)
        stats = {}
        for name, mask in (('annulus', ok & (rr >= R_ANN[0] * 1e-3) & (rr <= R_ANN[1] * 1e-3)),
                           ('tips', ok & (rr < R_ANN[0] * 1e-3)),
                           ('all', ok)):
            if mask.sum() == 0:
                continue
            rms_c = float(np.sqrt(np.mean(mag_c[mask] ** 2)))
            rms_d = float(np.sqrt(np.mean(diff[mask] ** 2)))
            # relative error where the field is significant (> 10% of the region max)
            sig = mask & (mag_c > 0.1 * np.nanmax(mag_c[mask]))
            rel_med = float(np.median(diff[sig] / mag_c[sig])) if sig.sum() else float('nan')
            stats[name] = {'n': int(mask.sum()), 'rms_E_comsol': rms_c, 'rms_diff': rms_d,
                           'rms_rel': rms_d / rms_c, 'median_rel_where_significant': rel_med}
        results['planes'][f"{zp:.0f}"] = stats
        print(f"[cmp] z = {zp:4.1f} mm: " + ", ".join(
            f"{k}: rms rel {v['rms_rel']:.3f}, median rel {v['median_rel_where_significant']:.3f} (n={v['n']})"
            for k, v in stats.items()))

    # across-gap profiles at r = 200 mm through the low-azimuth gap
    prof = {}
    th = np.radians(np.linspace(12, 36, 121))
    for zp, iz in izs.items():
        px, py = 0.200 * np.cos(th), 0.200 * np.sin(th)
        ix = np.clip(np.round(px / dx).astype(int), 0, nx - 1)
        iy = np.clip(np.round(py / dx).astype(int), 0, nx - 1)
        et_c = -Ec[iz][iy, ix, 0] * np.sin(th) + Ec[iz][iy, ix, 1] * np.cos(th)
        eb = fields_b[zp] * scale
        et_b = -eb[iy, ix, 0] * np.sin(th) + eb[iy, ix, 1] * np.cos(th)
        prof[zp] = (np.degrees(th), et_c, et_b)

    # gap voltage vs radius (midplane), COMSOL vs scaled BEM
    radii = np.array([100, 120, 150, 175, 200, 220, 235])
    vg_c = [arc_voltage(ec0, x, y, r * 1e-3, 10.0, 45.0) for r in radii]
    vg_b = [scale * arc_voltage(eb0, x, y, r * 1e-3, 10.0, 45.0) for r in radii]
    results['gap_voltage_vs_r'] = {'r_mm': radii.tolist(), 'comsol': [float(v) for v in vg_c],
                                   'bem_scaled': [float(v) for v in vg_b]}
    print("[cmp] gap voltage vs r [mm]: " + ", ".join(
        f"{r}: C {c:.3f} / B {b:.3f}" for r, c, b in zip(radii, vg_c, vg_b)))

    (OUT / 'cad_bem_vs_comsol.json').write_text(json.dumps(results, indent=2))

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    ext = (0, R_CMP, 0, R_CMP)
    mag_c0 = np.linalg.norm(ec0, axis=2)
    mag_b0 = np.linalg.norm(eb0 * scale, axis=2)
    vmax = np.nanpercentile(mag_c0, 99.5)
    im = axs[0, 0].imshow(mag_c0, origin='lower', extent=ext, vmin=0, vmax=vmax, cmap='magma')
    axs[0, 0].set_title('|E| COMSOL, z = 0 [mm axes]'); plt.colorbar(im, ax=axs[0, 0], shrink=0.8)
    im = axs[0, 1].imshow(mag_b0, origin='lower', extent=ext, vmin=0, vmax=vmax, cmap='magma')
    axs[0, 1].set_title(f'|E| BEM x {scale:.3f}, z = 0'); plt.colorbar(im, ax=axs[0, 1], shrink=0.8)
    rel = np.linalg.norm(eb0 * scale - ec0, axis=2) / vmax
    im = axs[0, 2].imshow(rel, origin='lower', extent=ext, vmin=0, vmax=0.2, cmap='viridis')
    axs[0, 2].set_title('|E_BEM - E_COMSOL| / max|E_COMSOL|, z = 0'); plt.colorbar(im, ax=axs[0, 2], shrink=0.8)
    for ax in axs[0]:
        for r in (R_ANN[0], R_ANN[1]):
            tt = np.linspace(0, np.pi / 2, 100)
            ax.plot(r * np.cos(tt), r * np.sin(tt), 'w:', lw=0.6)
    ax = axs[1, 0]
    for zp, (deg, ec_, eb_) in prof.items():
        l, = ax.plot(deg, ec_, '-', label=f'COMSOL z = {zp:.0f} mm')
        ax.plot(deg, eb_, '--', color=l.get_color(), label=f'BEM z = {zp:.0f} mm')
    ax.set_xlabel('azimuth [deg] at r = 200 mm'); ax.set_ylabel('E_azimuthal'); ax.grid(True, lw=0.3)
    ax.legend(fontsize=7); ax.set_title('across the low-azimuth gap at r = 200 mm')
    ax = axs[1, 1]
    zs = [float(k) for k in results['planes']]
    for region in ('annulus', 'tips'):
        ax.plot(zs, [results['planes'][k][region]['rms_rel'] for k in results['planes']], 'o-', label=f'{region}: rms rel')
        ax.plot(zs, [results['planes'][k][region]['median_rel_where_significant'] for k in results['planes']], 's--', label=f'{region}: median rel')
    ax.set_xlabel('z plane [mm]'); ax.set_ylabel('relative difference'); ax.grid(True, lw=0.3); ax.legend(fontsize=8)
    ax.set_title('BEM vs COMSOL per plane')
    ax = axs[1, 2]
    ax.plot(radii, vg_c, 'o-', label='COMSOL'); ax.plot(radii, vg_b, 's--', label='BEM scaled')
    ax.set_xlabel('r [mm]'); ax.set_ylabel('gap voltage (midplane arc)'); ax.grid(True, lw=0.3); ax.legend()
    ax.set_title('gap voltage vs radius')
    fig.suptitle(f'Original cavity center: BEM ({len(tris)} elements, {sol.solver}) vs COMSOL; '
                 f'amplitude LSQ {scale:.3f} / arc {scale_arc:.3f}', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'cad_bem_vs_comsol.png', dpi=110)
    print("[done] wrote output/cad_bem_vs_comsol.{json,png}")


if __name__ == '__main__':
    main()
