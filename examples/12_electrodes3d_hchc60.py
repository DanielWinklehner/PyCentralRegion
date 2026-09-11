"""
12_electrodes3d_hchc60.py - 3D electrodes for an optimized central region:
the HCHC-60 preliminary winner through electrodes3d.

Stages
  A. Rebuild the winner (HCHC-60 optimizer module + result pickle), build the
     2D scroll electrodes exactly as the BEM verification does.
  B. Tall-wall regression: the 3D builder with no vertical structure must
     reproduce the 2D mesher's field. Both are solved and compared on the
     midplane (r <= R_CMP).
  C. Real vertical structure (IBA height profiles: tapered dee aperture and
     height, hill gap, valley roof, crop): section plots of the solids, the
     midplane field against the tall-wall one (aperture softening), gap
     voltage vs radius, and a re-track of the winner on the 3D-derived
     midplane field with the 2D tracker: per-turn energies thin-gap /
     tall-wall BEM / 3D-aperture BEM.

Usage:
    python 12_electrodes3d_hchc60.py [--skip-regression] [--size-min 0.002]
Env: CONDA_PREFIX=<accel-dev-env> MPLBACKEND=Agg MPI4PY_RC_INITIALIZE=false
Outputs in output/: electrodes3d_hchc60_sections.png, electrodes3d_hchc60_fields.png,
electrodes3d_hchc60.json
"""
import os
import sys
import json
import time
import pickle
import argparse
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

CR_DIR = Path(os.environ.get('HCHC60_CR_DIR',
                             r"D:\MIT Dropbox\Daniel Winklehner\Projects\IsoDAR\60 MeV Cyclotron\CentralRegion"))
OUT = ROOT / 'output'
R_CMP = 0.20           # midplane comparison radius [m] (inside the 3D crop margin)
SPACING = 0.002        # comparison grid [m]
BUILD_KWARGS = dict(chain_ds=0.012, arc_ds=0.04, post_tip_gap=0.005, min_metal_width=0.001)
BEM_STEPS_PER_TURN = 2000
MAX_TURNS = 15
R_MAX_REGRESSION = 0.30   # tall-wall regression on a reduced-radius copy of the design [m]
# Gaps whose variable segment is straightened (angle / rotation -> 0) before the
# 3D build. Dee 3 (gaps 6, 7) of the 2026-09-05 winner: its segment ends below /
# right at the first orbit crossing, a kink the beam never uses (see
# rf_cavity.check_variable_segments, which flags such segments on every dee).
STRAIGHTEN_GAPS = (6, 7)
# IBA-style bars (stage D): walls along the dee inner edges and between the
# beam crossings on every gap chain, out to BAR_R_MAX, leaving BAR_MARGIN of
# open aperture on both sides of each crossing.
BAR_MARGIN = 0.008        # Euclidean clearance between a bar and the centroid path [m]
BAR_R_MAX = 0.20
BAR_WIDTH = 0.010         # chain bars
TIP_WIDTH = 0.003         # tip wall: the 2D tip rule leaves only traj_tip_clearance (10 mm here)
                          # between the dee inner edge and the centroid, a 10 mm wall would fill it
BEAM_HALFWIDTH = 0.0      # bunch half-width added to the clearances (0: centroid only)


def load_winner():
    spec = importlib.util.spec_from_file_location("hchc60_opt", CR_DIR / "HCHC-60_optimize_cavity_geometry.py")
    ex06 = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(CR_DIR)                       # the optimizer resolves its field cache relative to itself
    try:
        spec.loader.exec_module(ex06)
        result = pickle.load(open(CR_DIR / "output" / "optimized_cavity_geometry.pkl", "rb"))
    finally:
        os.chdir(cwd)
    return ex06, result


def build_design(ex06, result, gap_model):
    from PyCentralRegion.accelerated_orbit_finder import AcceleratedOrbitFinder
    from PyCentralRegion.rf_cavity import DeeSystem, snap_nodes_between_turns
    cwd = os.getcwd()
    os.chdir(CR_DIR)
    try:
        design, _, _, _ = ex06.build_system(rf_frequency=float(result.rf_frequency_mhz) * 1e6, quiet=True)
    finally:
        os.chdir(cwd)
    finder = AcceleratedOrbitFinder(design, target_energy_mev=ex06.TARGET_ENERGY_MEV,
                                    max_radius_m=ex06.MAX_RADIUS_M, algorithm='rk4_rel',
                                    steps_per_turn=BEM_STEPS_PER_TURN, verbose=False,
                                    gap_model=gap_model)
    geom = result.metadata['optimal_geometry']
    dee_system = DeeSystem(list(ex06.DEE_CENTER_ANGLES), ex06.DEE_OPENING_ANGLE, design.rf_cavities)
    if geom.get('opening_angle_deg') is not None:
        dee_system.apply_opening_angle(geom['opening_angle_deg'])
    rots = geom.get('segment_rotations_per_gap')
    for g, cav in enumerate(design.rf_cavities):
        cav.update_geometry(segment_angles=list(geom['segment_angles_per_gap'][g]),
                            segment_radii=list(geom['segment_radii_per_gap'][g]),
                            segment_rotations=(list(rots[g]) if rots else None))
    snap_nodes_between_turns(design.rf_cavities, result.trajectory_reference, verbose=False)
    return design, finder


def straighten_gaps(design, gaps):
    """Remove the kink of the variable segments of the given gaps (angles and
    rotations -> 0; the node radii stay, the chain is then collinear)."""
    for g in gaps:
        cav = design.rf_cavities[g]
        n = cav.n_variable_segments
        if n:
            cav.update_geometry(segment_angles=[0.0] * n, segment_rotations=[0.0] * n)


def shrink_design(design, r_max):
    """Cut every gap at r_max (geometry rebuilt): a smaller, identical-inside
    copy for the tall-wall regression, where the 3D crop must not touch it."""
    for cav in design.rf_cavities:
        cav.r_max = float(r_max)
        cav.update_geometry()


def midplane_grid(sol, r_cmp, spacing):
    """Midplane E on a Cartesian grid |x|,|y| <= r_cmp at the vacuum points."""
    xs = np.arange(-r_cmp, r_cmp + 0.5 * spacing, spacing)
    gx, gy = np.meshgrid(xs, xs, indexing='ij')
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    ok = (np.hypot(pts[:, 0], pts[:, 1]) <= r_cmp) & ~sol._metal_inside(pts)
    e = np.full((len(pts), 2), np.nan)
    e[ok] = sol.efield_midplane(pts[ok])
    return xs, e.reshape(len(xs), len(xs), 2), ok.reshape(len(xs), len(xs))


def gap_voltage_arcs(sol, radii, az0, az1, n=1200):
    """-int E.dl along midplane arcs from az0 to az1 [deg] at each radius [m]."""
    out = []
    for r in radii:
        th = np.radians(np.linspace(az0, az1, n))
        pts = np.column_stack([r * np.cos(th), r * np.sin(th)])
        e = sol.efield_midplane(pts)
        ok = ~sol._metal_inside(pts)
        et = np.where(ok, -e[:, 0] * np.sin(th) + e[:, 1] * np.cos(th), 0.0)
        out.append(-float(np.sum(0.5 * (et[1:] + et[:-1]) * np.diff(th))) * r)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-regression', action='store_true')
    ap.add_argument('--size-min', type=float, default=0.002)
    ap.add_argument('--size-max', type=float, default=0.012)
    ap.add_argument('--tip-width', type=float, default=TIP_WIDTH,
                    help='tip wall thickness [m]; a 2-3 mm sheet closes the aperture like a thick bar')
    ap.add_argument('--bar-margin', type=float, default=BAR_MARGIN,
                    help='Euclidean clearance bar <-> centroid path [m]')
    ap.add_argument('--beam-halfwidth', type=float, default=BEAM_HALFWIDTH,
                    help='bunch half-width [m] added to the bar clearance and to the metal test')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    from PyCentralRegion.gap_fields import build_gap_electrodes, solve_gap_field
    from PyCentralRegion.electrodes3d import (HeightProfiles, build_electrodes_3d, plot_sections,
                                              midplane_obstacles, check_trajectory_clearance)
    from PyPATools.field import TimedField
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ex06, result = load_winner()
    phase = float(result.bunch_phase_deg)
    rf_mhz = float(result.rf_frequency_mhz)
    print(f"[A] winner: {result.final_energy_mev:.3f} MeV in {result.n_turns} turns, "
          f"phase {phase:.2f} deg, f_rev {rf_mhz:.6f} MHz")
    design, finder = build_design(ex06, result, 'bem2d')
    from PyCentralRegion.rf_cavity import check_variable_segments
    print("[A] variable-segment check on the winner as optimized:")
    findings = check_variable_segments(design, result.trajectory_reference,
                                       tip_clearance=max(float(c.gap_width_at(c.r_min)) for c in design.rf_cavities))
    if STRAIGHTEN_GAPS:
        straighten_gaps(design, STRAIGHTEN_GAPS)
        print(f"[A] straightened the variable segments of gaps {list(STRAIGHTEN_GAPS)}; re-check:")
        check_variable_segments(design, result.trajectory_reference,
                                tip_clearance=max(float(c.gap_width_at(c.r_min)) for c in design.rf_cavities))
    model2d = build_gap_electrodes(design, trim_trajectory=result.trajectory_reference,
                                   verbose=False, **BUILD_KWARGS)
    print(f"[A] 2D electrodes: {model2d.n_elements} elements, scroll r = "
          f"{np.hypot(*np.asarray(model2d.wedges[-1].polygon).T).min() * 1000:.1f}.."
          f"{np.hypot(*np.asarray(model2d.wedges[-1].polygon).T).max() * 1000:.1f} mm")
    report = {'winner': {'final_energy_mev': result.final_energy_mev, 'n_turns': result.n_turns},
              'n_elements_2d': int(model2d.n_elements),
              'segment_findings': [{k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                    for k, v in f.items()} for f in findings],
              'straightened_gaps': list(STRAIGHTEN_GAPS)}
    radii = np.array([0.08, 0.10, 0.12, 0.15, 0.18])
    az_lo = float(design.rf_cavities[0].base_angle) - 20.0     # around the first entry gap (base 21.9 deg)
    az_hi = float(design.rf_cavities[0].base_angle) + 20.0

    # ---- B. tall-wall regression -------------------------------------------------
    if not args.skip_regression:
        # identical reduced-radius geometry for both solvers: the 3D crop sits
        # beyond the electrodes, so only the meshing differs
        design_r, _ = build_design(ex06, result, 'bem2d')
        if STRAIGHTEN_GAPS:
            straighten_gaps(design_r, STRAIGHTEN_GAPS)
        shrink_design(design_r, R_MAX_REGRESSION)
        model2d_r = build_gap_electrodes(design_r, trim_trajectory=result.trajectory_reference,
                                         verbose=False, **BUILD_KWARGS)
        t0 = time.time()
        sol2d = solve_gap_field(model2d_r, tol=1e-5, verbose=True)
        print(f"[B] 2D solve ({model2d_r.n_elements} elements, r_max {R_MAX_REGRESSION} m) "
              f"{time.time() - t0:.0f} s ({sol2d.solver}/{sol2d.device_interface})")
        height = model2d_r.params['height']
        model_tw = build_electrodes_3d(model2d_r, HeightProfiles.tall_wall(height, r_cut=R_MAX_REGRESSION + 0.02),
                                       size_min=args.size_min, size_max=args.size_max, verbose=True)
        t0 = time.time()
        sol_tw = solve_gap_field(model_tw, tol=1e-5, verbose=True)
        print(f"[B] tall-wall 3D solve {time.time() - t0:.0f} s")
        xs, e2, ok2 = midplane_grid(sol2d, R_CMP, SPACING)
        _, etw, oktw = midplane_grid(sol_tw, R_CMP, SPACING)
        ok = ok2 & oktw
        d = np.linalg.norm(etw - e2, axis=2)[ok]
        m2 = np.linalg.norm(e2, axis=2)[ok]
        sig = m2 > 0.1 * np.nanmax(m2)
        reg = {'n_elements_3d': int(model_tw.n_elements), 'n_points': int(ok.sum()),
               'rms_rel': float(np.sqrt(np.mean(d ** 2)) / np.sqrt(np.mean(m2 ** 2))),
               'median_rel_significant': float(np.median(d[sig] / m2[sig])),
               'gap_voltage_2d': gap_voltage_arcs(sol2d, radii, az_lo, az_hi).tolist(),
               'gap_voltage_tallwall3d': gap_voltage_arcs(sol_tw, radii, az_lo, az_hi).tolist()}
        print(f"[B] tall-wall 3D vs 2D on the midplane (r <= {R_CMP * 1000:.0f} mm): rms rel "
              f"{reg['rms_rel']:.4f}, median rel (significant) {reg['median_rel_significant']:.4f}")
        print("[B] gap voltage arcs [kV] 2D / 3D: " + ", ".join(
            f"r={r * 1000:.0f}: {a / 1e3:.2f}/{b / 1e3:.2f}" for r, a, b in
            zip(radii, reg['gap_voltage_2d'], reg['gap_voltage_tallwall3d'])))
        report['regression'] = reg
        e_ref, ok_ref = e2, ok2
    else:
        sol2d = None
        e_ref = ok_ref = None

    # ---- C. real vertical structure ------------------------------------------------
    prof = HeightProfiles.iba(r_cut=0.25, z_cut=0.06)
    model3d = build_electrodes_3d(model2d, prof, size_min=args.size_min, size_max=args.size_max,
                                  verbose=True)
    planes = [([0, 0, 1], [0, 0, 0.0], 0, 1, 'z = 0'),
              ([0, 0, 1], [0, 0, 0.015], 0, 1, 'z = +15 mm'),
              ([0, 0, 1], [0, 0, 0.040], 0, 1, 'z = +40 mm'),
              ([-np.sin(np.radians(45)), np.cos(np.radians(45)), 0], [0, 0, 0], 0, 2, 'vertical, dee0 centerline 45 deg: (x, z)'),
              ([-np.sin(np.radians(az_lo + 20)), np.cos(np.radians(az_lo + 20)), 0], [0, 0, 0], 0, 2,
               f'vertical, through gap 0 at {az_lo + 20:.1f} deg: (x, z)'),
              ([0, 1, 0], [0, 0, 0], 0, 2, 'vertical y = 0: (x, z)')]
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    lims = [((-260, 260), (-260, 260))] * 3 + [((-10, 260), (-70, 70))] * 3
    for ax, pl, lim in zip(axs.ravel(), planes, lims):
        plot_sections(model3d, [pl], ax=ax, lim=lim)
    tr = result.trajectory_reference * 1000
    for ax in axs[0]:
        ax.plot(tr[:, 0], tr[:, 1], 'g-', lw=0.4)
    fig.suptitle(f"HCHC-60 winner, 3D electrodes with IBA height profiles ({model3d.n_elements} elements); "
                 "blue = dees, red = ground, green = reference orbit", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'electrodes3d_hchc60_sections.png', dpi=110)

    t0 = time.time()
    sol3d = solve_gap_field(model3d, tol=1e-5, verbose=True)
    print(f"[C] 3D solve {time.time() - t0:.0f} s ({sol3d.solver}/{sol3d.device_interface})")
    xs, e3, ok3 = midplane_grid(sol3d, R_CMP, SPACING)
    c = {'n_elements_3d': int(model3d.n_elements),
         'gap_voltage_3d': gap_voltage_arcs(sol3d, radii, az_lo, az_hi).tolist()}
    print("[C] gap voltage arcs [kV] 3D-aperture: " + ", ".join(
        f"r={r * 1000:.0f}: {v / 1e3:.2f}" for r, v in zip(radii, c['gap_voltage_3d'])))
    if e_ref is not None:
        ok = ok_ref & ok3
        m2 = np.linalg.norm(e_ref, axis=2)
        m3 = np.linalg.norm(e3, axis=2)
        sig = ok & (m2 > 0.1 * np.nanmax(m2[ok]))
        c['peak_ratio_3d_over_2d_median'] = float(np.median(m3[sig] / m2[sig]))
        c['rms_rel_diff_3d_vs_2d'] = float(np.sqrt(np.mean((m3[ok] - m2[ok]) ** 2)) / np.sqrt(np.mean(m2[ok] ** 2)))
        print(f"[C] 3D-aperture vs tall-wall on the midplane: median |E| ratio (significant) "
              f"{c['peak_ratio_3d_over_2d_median']:.3f}, rms rel diff {c['rms_rel_diff_3d_vs_2d']:.3f}")
    report['aperture3d'] = c

    # ---- re-track: full 2D tall-wall field vs COMPOSITE (3D inside, 2D outside)
    # The 3D model only exists inside its crop (r <= 250 mm) and is trustworthy
    # to ~220 mm; the orbit leaves that by turn 5. So the 3D midplane field is
    # blended into the FULL 2D tall-wall field across R_SEAM (smoothstep), and
    # the winner is tracked through the pure 2D field and the composite with
    # the same solver stack - a like-for-like per-turn comparison.
    R_SEAM = (0.190, 0.220)
    t0 = time.time()
    sol2d_full = solve_gap_field(model2d, tol=1e-5, verbose=True)
    print(f"[C] full 2D tall-wall solve ({model2d.n_elements} elements) {time.time() - t0:.0f} s "
          f"({sol2d_full.solver}/{sol2d_full.device_interface})")
    t0 = time.time()
    static2d = sol2d_full.to_field(spacing=0.0015, verbose=False)
    print(f"[C] 2D field gridded in {time.time() - t0:.0f} s")
    gx2, gy2 = static2d._grid['x'], static2d._grid['y']
    ex2, ey2 = static2d._values['x'], static2d._values['y']
    ix = np.where(np.abs(gx2) <= R_SEAM[1] + 0.01)[0]
    iy = np.where(np.abs(gy2) <= R_SEAM[1] + 0.01)[0]
    xs_sub, ys_sub = gx2[ix], gy2[iy]
    sx, sy = np.meshgrid(xs_sub, ys_sub, indexing='ij')
    pts_sub = np.column_stack([sx.ravel(), sy.ravel(), np.zeros(sx.size)])
    from scipy.ndimage import binary_erosion
    from PyPATools.field import Field
    rr = np.hypot(sx, sy)
    tt = np.clip((rr - R_SEAM[0]) / (R_SEAM[1] - R_SEAM[0]), 0.0, 1.0)
    w2 = tt * tt * (3.0 - 2.0 * tt)                      # 0 inside (3D), 1 outside (2D)
    sub = np.ix_(ix, iy)

    def composite_field(sol, label):
        t0 = time.time()
        phi3 = sol.potential(pts_sub, chunk=4000).reshape(sx.shape)
        ex3 = -np.gradient(phi3, xs_sub, axis=0)
        ey3 = -np.gradient(phi3, ys_sub, axis=1)
        deep3 = binary_erosion(sol._metal_inside(pts_sub[:, :2]).reshape(sx.shape), iterations=3)
        ex3[deep3] = 0.0
        ey3[deep3] = 0.0
        ex_c, ey_c = ex2.copy(), ey2.copy()
        ex_c[sub] = (1.0 - w2) * ex3 + w2 * ex2[sub]
        ey_c[sub] = (1.0 - w2) * ey3 + w2 * ey2[sub]
        print(f"[C] composite '{label}' built in {time.time() - t0:.0f} s "
              f"(seam {R_SEAM[0] * 1000:.0f}-{R_SEAM[1] * 1000:.0f} mm)")
        return Field.from_arrays(grid={'x': gx2, 'y': gy2},
                                 values={'x': ex_c, 'y': ey_c, 'z': np.zeros_like(ex_c)},
                                 label=f"composite: {label} inside, tall-wall 2D outside")

    cav = design.rf_cavities[0]
    from PyPATools.species import IonSpecies
    cwd = os.getcwd()
    os.chdir(CR_DIR)
    try:
        beam = ex06.make_initial_beam(IonSpecies(ex06.SPECIES))
    finally:
        os.chdir(cwd)
    turns = {}

    def track(label, static, obstacles=None):
        """Re-track the hand-off centroid on ``static``. With ``obstacles`` (a
        midplane metal test of the 3D model) the particle is LOST on contact
        with metal and the run fails loudly - a centroid riding inside a wall
        sees E = 0 there and would otherwise report a perfect energy gain."""
        design.set_electric_field(TimedField(static, omega=float(cav.omega), phase=float(cav.bunch_phase_offset)))
        finder.obstacle_mask = obstacles
        t0 = time.time()
        try:
            res = finder.track_once(beam, bunch_phase_deg=phase, rf_freq_mhz=rf_mhz, max_turns=MAX_TURNS)
        finally:
            finder.obstacle_mask = None
        turns[label] = [float(ts.mean_energy_mev) for ts in res.turn_statistics]
        print(f"[C] track on {label}: {time.time() - t0:.0f} s, {len(turns[label])} turns, "
              f"final {turns[label][-1] if turns[label] else float('nan'):.3f} MeV")
        obs = res.metadata.get('obstacles')
        if obs and (obs['n_lost'] or obs['reference_contacts']):
            where = obs['hits'][0] if obs['hits'] else obs['first_reference_contact']
            raise RuntimeError(f"{label}: the centroid hit metal after {len(turns[label])} turns at "
                               f"({where[-2] * 1e3:.1f}, {where[-1] * 1e3:.1f}) mm, "
                               f"r = {np.hypot(where[-2], where[-1]) * 1e3:.1f} mm (step {where[-3]})")
        check_trajectory_clearance(obstacles, res.trajectory_reference, beam_halfwidth=args.beam_halfwidth,
                                   label=f"{label}: re-tracked centroid") if obstacles is not None else None
        return res

    track('bem_tallwall_2d', static2d)
    obstacles_open = midplane_obstacles(model3d, beam_halfwidth=args.beam_halfwidth, verbose=True)
    check_trajectory_clearance(obstacles_open, result.trajectory_reference, beam_halfwidth=args.beam_halfwidth,
                               label='winner centroid vs open-tip 3D model')
    track('composite_open_tips', composite_field(sol3d, 'open tips'), obstacles=obstacles_open)

    # ---- D. the same with IBA-style bars: tip walls + bars between crossings ---
    from PyCentralRegion.electrodes3d import auto_bars, bar_clearances
    bars = auto_bars(model2d, design, result.trajectory_reference, margin=args.bar_margin,
                     r_max=BAR_R_MAX, width=BAR_WIDTH, tip=True, ground=True,
                     tip_width=args.tip_width, beam_halfwidth=args.beam_halfwidth)
    clear = bar_clearances(model2d, bars, result.trajectory_reference)
    print(f"[D] {len(bars)} bars (clearance to the centroid path): " + ", ".join(
        f"{b.wedge}/{b.side}" + ("" if b.side == 'tip' else f" {b.r_from * 1000:.0f}-{b.r_to * 1000:.0f}")
        + f" ({c * 1e3:+.1f} mm)" for b, c in zip(bars, clear)))
    model3d_b = build_electrodes_3d(model2d, prof, bars=bars, size_min=args.size_min, size_max=args.size_max,
                                    verbose=True)
    # the centroid (and its +- beam_halfwidth envelope) must clear every wall
    # BEFORE the expensive solve; this raises with the first contact point
    obstacles_bars = midplane_obstacles(model3d_b, beam_halfwidth=args.beam_halfwidth, verbose=True)
    clr = check_trajectory_clearance(obstacles_bars, result.trajectory_reference,
                                     beam_halfwidth=args.beam_halfwidth, label='winner centroid vs barred 3D model')
    print(f"[D] centroid clearance check vs the barred model: {clr['n_contact']} of {clr['n_points']} points in metal")
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
    for ax, pl, lim in zip(axs, planes[:3], lims[:3]):
        plot_sections(model3d_b, [pl], ax=ax, lim=lim)
        ax.plot(tr[:, 0], tr[:, 1], 'g-', lw=0.4)
    fig.suptitle(f"with bars ({model3d_b.n_elements} elements): tip walls and bars between crossings", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'electrodes3d_hchc60_bars_sections.png', dpi=110)
    t0 = time.time()
    sol3d_b = solve_gap_field(model3d_b, tol=1e-5, verbose=True)
    print(f"[D] 3D solve with bars {time.time() - t0:.0f} s ({sol3d_b.solver}/{sol3d_b.device_interface})")
    vb = gap_voltage_arcs(sol3d_b, radii, az_lo, az_hi)
    print("[D] gap voltage arcs [kV] with bars: " + ", ".join(
        f"r={r * 1000:.0f}: {v / 1e3:.2f}" for r, v in zip(radii, vb)))
    report['bars'] = {'n_bars': len(bars), 'n_elements_3d': int(model3d_b.n_elements),
                      'gap_voltage_3d_bars': vb.tolist(), 'tip_width': args.tip_width,
                      'bar_margin': args.bar_margin, 'beam_halfwidth': args.beam_halfwidth,
                      'bars': [dict(vars(b), clearance_m=c) for b, c in zip(bars, clear)]}
    track('composite_with_bars', composite_field(sol3d_b, 'with bars'), obstacles=obstacles_bars)

    prev = {}
    vj = CR_DIR / 'output' / 'bem_gap_verification.json'
    if vj.exists():
        prev = json.load(open(vj))
    report['turns'] = {'thin_gap_verification': prev.get('turns_thin_mev'),
                       'bem_tallwall_verification_2026-09-05': prev.get('turns_bem_mev'), **turns}
    cols = ['bem_tallwall_2d', 'composite_open_tips', 'composite_with_bars']
    print("turn   thin [MeV]   tall-wall 2D   composite open tips   composite with bars")
    for i in range(max(len(v) for v in turns.values())):
        t = prev.get('turns_thin_mev', [None] * 99)[i] if prev else None
        fmt = lambda v: '      -' if v is None else f'{v:7.3f}'
        vals = [turns[c][i] if i < len(turns[c]) else None for c in cols]
        print(f"{i + 1:4d}   {fmt(t):>10s}   {fmt(vals[0]):>12s}   {fmt(vals[1]):>19s}   {fmt(vals[2]):>19s}")
    (OUT / 'electrodes3d_hchc60.json').write_text(json.dumps(report, indent=2))

    # fields figure
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.5))
    ext = (-R_CMP * 1000, R_CMP * 1000, -R_CMP * 1000, R_CMP * 1000)
    m3 = np.linalg.norm(e3, axis=2)
    vmax = np.nanpercentile(m3, 99.5)
    if e_ref is not None:
        m2 = np.linalg.norm(e_ref, axis=2)
        im = axs[0].imshow(m2.T / 1e6, origin='lower', extent=ext, vmin=0, vmax=vmax / 1e6, cmap='magma')
        axs[0].set_title('|E| tall-wall 2D [MV/m], z = 0'); plt.colorbar(im, ax=axs[0], shrink=0.8)
        im = axs[2].imshow((m3 - m2).T / 1e6, origin='lower', extent=ext, vmin=-0.3 * vmax / 1e6,
                           vmax=0.3 * vmax / 1e6, cmap='coolwarm')
        axs[2].set_title('|E| 3D-aperture - tall-wall [MV/m]'); plt.colorbar(im, ax=axs[2], shrink=0.8)
    im = axs[1].imshow(m3.T / 1e6, origin='lower', extent=ext, vmin=0, vmax=vmax / 1e6, cmap='magma')
    axs[1].set_title('|E| 3D-aperture BEM [MV/m], z = 0'); plt.colorbar(im, ax=axs[1], shrink=0.8)
    for ax in axs:
        ax.plot(tr[:, 0], tr[:, 1], 'c-', lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT / 'electrodes3d_hchc60_fields.png', dpi=110)
    print("[done] wrote output/electrodes3d_hchc60_{sections,fields}.png, .json")


if __name__ == '__main__':
    main()
