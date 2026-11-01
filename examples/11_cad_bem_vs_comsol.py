"""BEM field of a cavity centre built from its CAD, compared with a COMSOL field map of the same cavity.

Shows: the route from a vacuum CAD to a solved gap field and its validation. gmsh / OpenCASCADE imports the
vacuum STEP of one quarter sector (grounded sector walls), crops it to the cylinder r <= R_CUT, |z| <= Z_CUT and
cuts the metal out of it: closed dee and ground solids with artificial cap faces on the crop surface (an open
truncation, so the comparison stays inside r <= R_CMP). The surface mesh is graded by the distance from the dee
edges and the central ground features (fillets and the lower dee tip of the CAD are in the mesh), the Laplace
Dirichlet problem (dee 1 V, ground 0) is solved with gap_fields.solve_gap_field, and E_BEM is evaluated at the
COMSOL grid points on several z planes (central differences of the single-layer potential). The arbitrary COMSOL
amplitude is fixed by a least-squares scale over an annulus at the midplane; then |E| maps, rms differences per
plane, profiles across the gap and the gap voltage vs radius are compared.
Inputs: --step, the vacuum CAD of a quarter sector [mm] holding one dee between grounded sector walls; --field,
the COMSOL E-field on the same frame (npz: x, y, z [m] and E[z, y, x, 3], NaN in metal; default
<resources>/isodar_cavity_efield_center_2mm.npz). The constants below (crop, comparison annulus, the gap between
10 and 45 deg, two points known to be inside the metal) describe the 4-dee cavity the defaults belong to.
Runtime (CPU): about 1.5 minutes at the default mesh (56k elements: a 25 GB dense matrix, assembly 25 s, GMRES
35 s, the field evaluation 20 s; meshing takes seconds); --solver scipy keeps the GPU out of it, --mesh-only stops
after the mesh, --reuse-mesh starts from the saved one.
Outputs: <out>/cad_bem_quarter_mesh.npz, cad_bem_vs_comsol.json, cad_bem_vs_comsol.png.

Usage:
    python examples/11_cad_bem_vs_comsol.py --step CAVITY.step [--field FIELD.npz] [--resources DIR] [--out DIR]
        [--size-min MM] [--size-max MM] [--solver auto|scipy] [--mesh-only] [--reuse-mesh]
"""
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import _common
from PyCentralRegion.fields3d import arc_voltage
from PyCentralRegion.gap_fields import ElectrodeModel, solve_gap_field

FIELD_FILE = 'isodar_cavity_efield_center_2mm.npz'
R_CUT, Z_CUT = 250.0, 60.0          # crop cylinder [mm]
R_CMP = 240.0                       # comparison radius [mm]
R_ANN = (150.0, 230.0)              # annulus of the amplitude fit and the "annulus" statistics [mm]
Z_PLANES = (0.0, 10.0, 20.0, 28.0)  # COMSOL planes compared [mm]
GAP_AZ = (10.0, 45.0)               # arcs across the low-azimuth gap of the sector [deg]
PROFILE_R, PROFILE_AZ = 0.200, (12.0, 36.0)        # across-gap profile: radius [m], azimuth range [deg]
INSIDE_DEE = (0.150 * np.cos(np.radians(45.0)), 0.150 * np.sin(np.radians(45.0)), 0.030)    # in a dee plate [m]
INSIDE_GROUND = (0.040, 0.040, 0.0)                                                          # in the central plug [m]
PROBE_H = 0.25e-3                   # central-difference step [m]


# ============================================================================
# CAD -> metal solids -> graded surface mesh
# ============================================================================
def build_mesh(step, size_min, size_max, dist_min=4.0, dist_max=50.0, r_fine_ground=170.0):
    """Crop the vacuum STEP to the cylinder, cut the metal out of it and surface-mesh the solids, finer near
    the dee edge curves (and the ground curves inside ``r_fine_ground`` [mm]). Returns vertices [m], triangles,
    per-triangle potentials (dee 1, ground 0) and areas [m^2]. The one solid that does not reach the crop's
    top and bottom faces is the dee; everything else is ground."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(str(step))
    gmsh.model.occ.synchronize()
    cyl = gmsh.model.occ.addCylinder(0, 0, -Z_CUT, 0, 0, 2 * Z_CUT, R_CUT)
    metal, _ = gmsh.model.occ.cut([(3, cyl)], [(3, 1)], removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()
    dee_vols, gnd_vols = [], []
    for t in (t for d, t in metal if d == 3):
        b = gmsh.model.getBoundingBox(3, t)
        (dee_vols if max(abs(b[2]), abs(b[5])) < Z_CUT - 1.0 else gnd_vols).append(t)
        print(f"[mesh] volume {t}: x {b[0]:7.1f}..{b[3]:7.1f} y {b[1]:7.1f}..{b[4]:7.1f} "
              f"z {b[2]:6.1f}..{b[5]:6.1f}  mass {gmsh.model.occ.getMass(3, t):.3e} mm3")
    if len(dee_vols) != 1:
        raise RuntimeError(f"expected exactly one dee volume, got {dee_vols}")
    print(f"[mesh] dee volume {dee_vols}, ground volumes {gnd_vols}")

    def faces_of(vlist):
        return sorted({f for d, f in gmsh.model.getBoundary([(3, t) for t in vlist], oriented=False,
                                                            recursive=False)})

    def curves_of(faces, r_max):
        cs = set()
        for f in faces:
            for d, c in gmsh.model.getBoundary([(2, f)], oriented=False, recursive=False):
                b = gmsh.model.getBoundingBox(1, c)
                if np.hypot(max(abs(b[0]), abs(b[3])), max(abs(b[1]), abs(b[4]))) < r_max:
                    cs.add(abs(c))
        return sorted(cs)

    dee_faces, gnd_faces = faces_of(dee_vols), faces_of(gnd_vols)
    curves = curves_of(dee_faces, R_CUT - 5.0) + curves_of(gnd_faces, r_fine_ground)
    print(f"[mesh] size field on {len(curves)} curves; dee faces {len(dee_faces)}, ground faces {len(gnd_faces)}")
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
            etypes, _, enodes = gmsh.model.mesh.getElements(2, f)
            for ty, nodes in zip(etypes, enodes):
                if ty == 2:                                     # 3-node triangles
                    t = np.vectorize(remap.get)(np.asarray(nodes).reshape(-1, 3))
                    tris.append(t)
                    pots.append(np.full(len(t), pot))
    gmsh.finalize()
    tris, pots = np.vstack(tris), np.concatenate(pots)
    used = np.unique(tris)                                      # drop unreferenced nodes
    idx = -np.ones(len(verts), dtype=int)
    idx[used] = np.arange(len(used))
    verts, tris = verts[used], idx[tris]
    areas = 0.5 * np.linalg.norm(np.cross(verts[tris[:, 1]] - verts[tris[:, 0]],
                                          verts[tris[:, 2]] - verts[tris[:, 0]]), axis=1)
    print(f"[mesh] {len(tris)} triangles ({int((pots > 0).sum())} dee, {int((pots == 0).sum())} ground), "
          f"{len(verts)} nodes, meshed in {time.time() - t0:.0f} s; edge ~ {np.sqrt(areas.min() / 0.433):.2f}.."
          f"{np.sqrt(areas.max() / 0.433):.1f} mm, {int((areas < 1e-6).sum())} degenerate")
    return verts * 1e-3, tris, pots, areas * 1e-6


def nearest_node(field_xy, x, y):
    """A gridded midplane field (ny, nx, 3) as a callable of (N, 3) points (nearest node; NaN = metal)."""
    dx, dy = x[1] - x[0], y[1] - y[0]

    def field(pts):
        ix = np.clip(np.round((pts[:, 0] - x[0]) / dx).astype(int), 0, len(x) - 1)
        iy = np.clip(np.round((pts[:, 1] - y[0]) / dy).astype(int), 0, len(y) - 1)
        return field_xy[iy, ix]
    return field


def main():
    ap = _common.arguments(__doc__)
    ap.add_argument('--step', type=Path, required=True, help='vacuum CAD of the quarter sector (STEP, mm)')
    ap.add_argument('--field', type=Path, default=None,
                    help=f'COMSOL E-field npz on the same frame (default: <resources>/{FIELD_FILE})')
    ap.add_argument('--size-min', type=float, default=2.0, help='finest element size [mm] (default 2)')
    ap.add_argument('--size-max', type=float, default=12.0, help='coarsest element size [mm] (default 12)')
    ap.add_argument('--solver', default='auto', help="BEM solver: 'auto' (GPU when it fits) or 'scipy' (CPU)")
    ap.add_argument('--mesh-only', action='store_true', help='stop after the mesh')
    ap.add_argument('--reuse-mesh', action='store_true', help='start from <out>/cad_bem_quarter_mesh.npz')
    args = _common.parse(ap)
    step = _common.require(args.step)
    field_file = _common.require(args.field or args.resources / FIELD_FILE)
    mesh_file = args.out / 'cad_bem_quarter_mesh.npz'

    if args.reuse_mesh and mesh_file.exists():
        d = np.load(mesh_file)
        verts, tris, pots = d['vertices'], d['triangles'], d['potentials']
        print(f"[mesh] reused {len(tris)} triangles from {mesh_file.name}")
    else:
        verts, tris, pots, areas = build_mesh(step, args.size_min, args.size_max)
        np.savez(mesh_file, vertices=verts, triangles=tris, potentials=pots, areas=areas, r_cut=R_CUT * 1e-3,
                 z_cut=Z_CUT * 1e-3, size_min=args.size_min, size_max=args.size_max)
    if args.mesh_only:
        return

    model = ElectrodeModel(vertices=verts, triangles=tris, potentials=pots, wedges=[],
                           params={'source': f'{step.name}, cropped', 'r_cut': R_CUT * 1e-3, 'z_cut': Z_CUT * 1e-3})
    t0 = time.time()
    sol = solve_gap_field(model, tol=1e-5, solver=args.solver, verbose=True)
    print(f"[solve] {sol.solver} on {sol.device_interface}: assembly {sol.assembly_time_s:.1f} s, "
          f"solve {sol.solve_time_s:.1f} s, total {time.time() - t0:.0f} s")
    phi_chk = sol.potential(np.array([INSIDE_DEE, INSIDE_GROUND]))
    print(f"[check] potential inside the dee {phi_chk[0]:.4f} (want 1), inside the ground {phi_chk[1]:.4f} (want 0)")

    # ---- the COMSOL field, cut to the comparison square
    d = np.load(field_file)
    x, y, z, Ec = d['x'], d['y'], d['z'], d['E']          # E[z, y, x, component], NaN = metal
    nx = int(np.round(R_CMP * 1e-3 / (x[1] - x[0]))) + 1
    x, y, Ec = x[:nx], y[:nx], Ec[:, :nx, :nx, :]
    gx, gy = np.meshgrid(x, y, indexing='xy')             # gx[iy, ix]
    rr = np.hypot(gx, gy)
    results = {'step': str(step), 'field': str(field_file), 'n_elements': int(len(tris)),
               'solver': sol.solver, 'planes': {}}

    # ---- BEM at the COMSOL vacuum nodes of each plane
    izs = {zp: int(np.argmin(np.abs(z - zp * 1e-3))) for zp in Z_PLANES}
    fields_b = {}
    t0 = time.time()
    for zp, iz in izs.items():
        ok = np.isfinite(Ec[iz][:, :, 0]) & (rr <= R_CMP * 1e-3)
        eb = np.full(Ec[iz].shape, np.nan)
        eb[ok] = sol.efield(np.column_stack([gx[ok], gy[ok], np.full(ok.sum(), z[iz])]), h=PROBE_H)
        fields_b[zp] = eb
        print(f"[eval] z = {zp:4.1f} mm: {ok.sum()} vacuum points ({time.time() - t0:.0f} s elapsed)")

    # ---- amplitude: least squares over the annulus at the midplane; the gap-voltage ratio as a cross-check
    ec0, eb0 = Ec[izs[0.0]], fields_b[0.0]
    ann = ((rr >= R_ANN[0] * 1e-3) & (rr <= R_ANN[1] * 1e-3) & np.isfinite(ec0[:, :, 0])
           & np.isfinite(eb0[:, :, 0]))
    scale = float(np.sum(ec0[ann] * eb0[ann]) / np.sum(eb0[ann] * eb0[ann]))
    fc, fb = nearest_node(ec0, x, y), nearest_node(eb0, x, y)
    v_c = [arc_voltage(fc, r * 1e-3, *GAP_AZ) for r in (150, 175, 200, 230)]
    v_b = [arc_voltage(fb, r * 1e-3, *GAP_AZ) for r in (150, 175, 200, 230)]
    scale_arc = float(np.mean(v_c) / np.mean(v_b))
    print(f"[scale] least-squares amplitude {scale:.4f}, gap-voltage amplitude {scale_arc:.4f} "
          f"(BEM gap voltage {np.mean(v_b):.4f} V per volt on the dee)")
    results.update(scale_lsq=scale, scale_arc=scale_arc, gap_voltage_bem_per_volt=[float(v) for v in v_b],
                   gap_voltage_comsol=[float(v) for v in v_c])

    # ---- per-plane statistics: annulus, dee tips (inside the annulus), everything
    for zp, iz in izs.items():
        ec, eb = Ec[iz], fields_b[zp] * scale
        ok = np.isfinite(ec[:, :, 0]) & np.isfinite(eb[:, :, 0])
        diff = np.linalg.norm(eb - ec, axis=2)
        mag_c = np.linalg.norm(ec, axis=2)
        stats = {}
        for name, mask in (('annulus', ok & (rr >= R_ANN[0] * 1e-3) & (rr <= R_ANN[1] * 1e-3)),
                           ('tips', ok & (rr < R_ANN[0] * 1e-3)), ('all', ok)):
            if mask.sum() == 0:
                continue
            rms_c = float(np.sqrt(np.mean(mag_c[mask] ** 2)))
            rms_d = float(np.sqrt(np.mean(diff[mask] ** 2)))
            sig = mask & (mag_c > 0.1 * np.nanmax(mag_c[mask]))       # where the field is significant
            rel_med = float(np.median(diff[sig] / mag_c[sig])) if sig.sum() else float('nan')
            stats[name] = {'n': int(mask.sum()), 'rms_E_comsol': rms_c, 'rms_diff': rms_d,
                           'rms_rel': rms_d / rms_c, 'median_rel_where_significant': rel_med}
        results['planes'][f"{zp:.0f}"] = stats
        print(f"[cmp] z = {zp:4.1f} mm: " + ", ".join(
            f"{k}: rms rel {v['rms_rel']:.3f}, median rel {v['median_rel_where_significant']:.3f} (n={v['n']})"
            for k, v in stats.items()))

    # ---- azimuthal field across the gap at PROFILE_R, and the gap voltage vs radius (midplane)
    th = np.radians(np.linspace(*PROFILE_AZ, 121))
    pts = np.column_stack([PROFILE_R * np.cos(th), PROFILE_R * np.sin(th), np.zeros_like(th)])
    prof = {}
    for zp, iz in izs.items():
        e_c, e_b = nearest_node(Ec[iz], x, y)(pts), nearest_node(fields_b[zp] * scale, x, y)(pts)
        prof[zp] = (np.degrees(th), -e_c[:, 0] * np.sin(th) + e_c[:, 1] * np.cos(th),
                    -e_b[:, 0] * np.sin(th) + e_b[:, 1] * np.cos(th))
    radii = np.array([100, 120, 150, 175, 200, 220, 235])
    vg_c = [arc_voltage(fc, r * 1e-3, *GAP_AZ) for r in radii]
    vg_b = [scale * arc_voltage(fb, r * 1e-3, *GAP_AZ) for r in radii]
    results['gap_voltage_vs_r'] = {'r_mm': radii.tolist(), 'comsol': [float(v) for v in vg_c],
                                   'bem_scaled': [float(v) for v in vg_b]}
    print("[cmp] gap voltage vs r [mm]: " + ", ".join(f"{r}: C {c:.3f} / B {b:.3f}"
                                                     for r, c, b in zip(radii, vg_c, vg_b)))
    (args.out / 'cad_bem_vs_comsol.json').write_text(json.dumps(results, indent=2))
    plot(results, ec0, eb0 * scale, prof, radii, vg_c, vg_b, args.out / 'cad_bem_vs_comsol.png')
    print(f"[done] wrote {args.out / 'cad_bem_vs_comsol.json'} and .png")


def plot(results, ec0, eb0, prof, radii, vg_c, vg_b, png):
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    ext = (0, R_CMP, 0, R_CMP)
    mag_c0, mag_b0 = np.linalg.norm(ec0, axis=2), np.linalg.norm(eb0, axis=2)
    vmax = np.nanpercentile(mag_c0, 99.5)
    for ax, img, title, kw in (
            (axs[0, 0], mag_c0, '|E| COMSOL, z = 0 [mm axes]', dict(vmin=0, vmax=vmax, cmap='magma')),
            (axs[0, 1], mag_b0, f"|E| BEM x {results['scale_lsq']:.3f}, z = 0", dict(vmin=0, vmax=vmax, cmap='magma')),
            (axs[0, 2], np.linalg.norm(eb0 - ec0, axis=2) / vmax, '|E_BEM - E_COMSOL| / max|E_COMSOL|, z = 0',
             dict(vmin=0, vmax=0.2, cmap='viridis'))):
        im = ax.imshow(img, origin='lower', extent=ext, **kw)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
        tt = np.linspace(0, np.pi / 2, 100)
        for r in R_ANN:
            ax.plot(r * np.cos(tt), r * np.sin(tt), 'w:', lw=0.6)
    ax = axs[1, 0]
    for zp, (deg, ec_, eb_) in prof.items():
        line, = ax.plot(deg, ec_, '-', label=f'COMSOL z = {zp:.0f} mm')
        ax.plot(deg, eb_, '--', color=line.get_color(), label=f'BEM z = {zp:.0f} mm')
    ax.set(xlabel=f'azimuth [deg] at r = {PROFILE_R * 1e3:.0f} mm', ylabel='E_azimuthal',
           title=f'across the low-azimuth gap at r = {PROFILE_R * 1e3:.0f} mm')
    ax.legend(fontsize=7)
    ax = axs[1, 1]
    zs = [float(k) for k in results['planes']]
    for region in ('annulus', 'tips'):
        ax.plot(zs, [v[region]['rms_rel'] for v in results['planes'].values()], 'o-', label=f'{region}: rms rel')
        ax.plot(zs, [v[region]['median_rel_where_significant'] for v in results['planes'].values()], 's--',
                label=f'{region}: median rel')
    ax.set(xlabel='z plane [mm]', ylabel='relative difference', title='BEM vs COMSOL per plane')
    ax.legend(fontsize=8)
    ax = axs[1, 2]
    ax.plot(radii, vg_c, 'o-', label='COMSOL')
    ax.plot(radii, vg_b, 's--', label='BEM scaled')
    ax.set(xlabel='r [mm]', ylabel='gap voltage (midplane arc)', title='gap voltage vs radius')
    ax.legend()
    for ax in axs[1]:
        ax.grid(True, lw=0.3)
    fig.suptitle(f"Cavity centre: BEM ({results['n_elements']} elements, {results['solver']}) vs COMSOL; "
                 f"amplitude LSQ {results['scale_lsq']:.3f} / gap voltage {results['scale_arc']:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
