"""3D electrodes for an optimized central region: 2D footprints -> solids with vertical structure -> 3D BEM.

Shows: electrodes3d on example 06's winner. The electrode footprints of gap_fields.build_gap_electrodes (dee
and ground wedges, the grounded scroll that follows the winner's first turn) become closed metal solids with
the vertical structure of a cavity centre (HeightProfiles: dee aperture and height, hill gap, valley roof,
crop cylinder), are surface-meshed by gmsh and solved with the same Laplace BEM as the 2D (tall-wall) model.
  A. the winner rebuilt and matched to its orbit as in example 07, the variable-segment check
     (rf_cavity.check_variable_segments) and the 2D electrodes;
  B. tall-wall regression (--skip-regression skips it): with no vertical structure the 3D builder must
     reproduce the 2D mesher's field; both are solved on a copy cut inside the 3D crop and compared on the
     midplane;
  C. the vertical structure of PROFILES: section plots, the 3D solve, the midplane field and the gap voltage
     against the 2D model, and the winner re-tracked through the full 2D field and through a composite: the
     3D midplane field inside, the 2D one outside, fields3d.RadialBlendField across R_SEAM (an open crop is
     trustworthy to about one pole gap inside its radius), with the 3D metal as obstacles;
  D. aperture-closing bars (--skip-bars skips them): electrodes3d.auto_bars puts a wall on every gap-chain
     interval that clears the orbit by BAR_MARGIN and a thin wall along each dee tip; the orbit is checked
     against the barred metal before the solve, which is then solved and re-tracked like C.
On example 06's winner (4 / 16 mm mesh) the tall-wall 3D model reproduces the 2D field (median 2 %, gap voltage
1.5 %). The open dee tips spread the gap field over the aperture (median |E| 0.31 of the 2D field; 44 instead of
49.5 kV across gap 0 at 80 mm), and both composite tracks lose the particle in turn 1 on metal laid out around
the 2D orbit (open tips: the scroll; bars: a dee tip wall). A thin-gap winner is not yet a 3D design: the scroll
and the bars have to follow the orbit in the 3D field.
Inputs: <resources>/uCyclo_v3_midplane_field.comsol and an optimizer result pickle (default
<out>/optimized_cavity_geometry.pkl from example 06).
Runtime (CPU, --solver scipy): about 10 minutes with --size-min 0.004 --size-max 0.016 --skip-regression (2D model
27k elements, 3D 49k open / 56k with bars: dense matrices of 6 / 19 / 25 GB, 1.5-4 minutes per solve; the three
tracks take seconds); the regression adds about 2.5 minutes. The default 3 / 15 mm mesh has 62k / 73k 3D
elements (30 / 42 GB) and about 1.6 times the 3D solve time (N^2); 2 / 12 mm (100k / 122k elements, 80 / 119 GB)
does not fit a 128 GB machine.
Outputs: <out>/electrodes3d_sections.png, electrodes3d_bars_sections.png, electrodes3d_fields.png,
electrodes3d.json.

Usage:
    python examples/12_electrodes3d.py [RESULT_PKL] [--resources DIR] [--out DIR] [--skip-regression]
        [--skip-bars] [--size-min M] [--size-max M] [--solver auto|scipy] [--turns N] [--tip-width M]
        [--bar-margin M] [--beam-halfwidth M]
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PyPATools.field import TimedField

import _common
from PyCentralRegion.electrodes3d import (HeightProfiles, auto_bars, bar_clearances, build_electrodes_3d,
                                          check_trajectory_clearance, midplane_obstacles, plot_sections)
from PyCentralRegion.fields3d import RadialBlendField, arc_voltage
from PyCentralRegion.gap_fields import build_gap_electrodes, solve_gap_field, wedge_metal_mask
from PyCentralRegion.rf_cavity import check_variable_segments

ex06 = _common.load_example('06_optimize_cavity_geometry.py')
ex07 = _common.load_example('07_bem_gap_verification.py')

# Vertical structure of the cavity centre, full heights [m] vs radius: the profile measured on the vacuum CAD of
# a 4-dee cavity (dee aperture 26 -> 50 mm, dee height 66 -> 90 mm, hill gap 34 -> 48 mm, valley floors at
# +-55 mm), cropped to r <= 250 mm, |z| <= 60 mm. Describe your own cavity with a HeightProfiles of its own.
PROFILES = HeightProfiles.iba(r_cut=0.25, z_cut=0.06)
R_SEAM = (PROFILES.r_cut - 0.06, PROFILES.r_cut - 0.03)     # composite: 3D inside R_SEAM[0], 2D outside [m]
R_CMP = 0.20                  # midplane comparison radius [m]
SPACING = 0.002               # comparison grid [m]
FIELD_SPACING = ex07.FIELD_KWARGS['spacing']                # midplane field grids for tracking [m]
GAP_RADII = np.array([0.08, 0.10, 0.12, 0.15, 0.18])       # gap-voltage arcs across gap 0 [m]
GAP_ARC_DEG = 20.0            # arcs from gap 0's base angle -/+ this; the neighbouring gaps are a dee away
R_MAX_REGRESSION = 0.30       # the tall-wall regression runs on a copy cut at this radius [m]
# Bars (stage D): walls on the gap chains between the beam crossings out to BAR_R_MAX, leaving BAR_MARGIN of
# open aperture on both sides of every crossing, and a TIP_WIDTH wall along every dee tip.
BAR_MARGIN = 0.008            # Euclidean clearance between a bar and the orbit [m]
BAR_R_MAX = 0.20
BAR_WIDTH = 0.010             # chain bars [m]
TIP_WIDTH = 0.003             # tip walls [m]: the 2D tip rule leaves about one gap width between tip and orbit
BEAM_HALFWIDTH = 0.0          # bunch half-width added to the clearances [m] (0: the orbit itself)

XS = np.arange(-R_CMP, R_CMP + 0.5 * SPACING, SPACING)
GX, GY = np.meshgrid(XS, XS, indexing='ij')
GRID = np.column_stack([GX.ravel(), GY.ravel(), np.zeros(GX.size)])
IN_CMP = np.hypot(GRID[:, 0], GRID[:, 1]) <= R_CMP


def shrink(design, r_max):
    """Cut every gap at ``r_max`` [m] (geometry rebuilt): identical inside, and its electrodes end inside the 3D
    crop, so the tall-wall regression compares the two meshers on the same metal."""
    for cav in design.rf_cavities:
        cav.r_max = float(r_max)
        cav.update_geometry()


def compare(f_ref, f_new, vacuum):
    """Midplane E of ``f_new`` against ``f_ref`` on the comparison grid's vacuum points: rms and median relative
    difference, and the median |E_new| / |E_ref| where the field is significant (> 10 % of its maximum)."""
    ok = vacuum & IN_CMP
    e_ref, e_new = f_ref(GRID[ok])[:, :2], f_new(GRID[ok])[:, :2]
    m_ref, m_new = np.linalg.norm(e_ref, axis=1), np.linalg.norm(e_new, axis=1)
    d = np.linalg.norm(e_new - e_ref, axis=1)
    sig = m_ref > 0.1 * m_ref.max()
    return {'n_points': int(ok.sum()), 'rms_rel': float(np.sqrt(np.mean(d ** 2) / np.mean(m_ref ** 2))),
            'median_rel_significant': float(np.median(d[sig] / m_ref[sig])),
            'median_ratio_significant': float(np.median(m_new[sig] / m_ref[sig]))}


def gap_voltages(field, arc):
    """Gap voltage [V] across gap 0 at GAP_RADII: -int E.dl along midplane arcs."""
    return [arc_voltage(field, r, *arc) for r in GAP_RADII]


def track(finder, beam, result, static, label, turns, obstacles=None):
    """Re-track the winner's particle on the midplane pattern ``static``, modulated as cos(omega t + phase) (both
    re-synced from cavity 0 before the run). With ``obstacles`` (a midplane metal test of a 3D model) the particle
    is stopped where it enters metal; without the test it would fly on through a wall, where E = 0, and report a
    clean energy gain. Returns the energy per turn [MeV] and where the particle was lost ((x, y) [mm] or None)."""
    finder.design.set_electric_field(TimedField(static, omega=0.0))
    finder.obstacle_mask = obstacles
    t0 = time.time()
    try:
        res = finder.track_once(beam, bunch_phase_deg=float(result.bunch_phase_deg),
                                rf_freq_mhz=float(result.rf_frequency_mhz), max_turns=turns)
    finally:
        finder.obstacle_mask = None
    energies = [float(ts.mean_energy_mev) for ts in res.turn_statistics]
    obs = res.metadata.get('obstacles') or {}
    lost = None
    if obs.get('n_lost') or obs.get('reference_contacts'):
        hit = obs['hits'][0] if obs['hits'] else obs['first_reference_contact']
        lost = [round(hit[-2] * 1e3, 1), round(hit[-1] * 1e3, 1)]
    status = (f"LOST on metal in turn {len(energies) + 1} at ({lost[0]}, {lost[1]}) mm, r = {np.hypot(*lost):.1f} mm"
              if lost else f"final {energies[-1]:.3f} MeV" if energies else "no complete turn")
    print(f"[track] {label}: {len(energies)} turns, {status} ({time.time() - t0:.0f} s)")
    return energies, lost


def plot_sections_figure(model3d, traj, png, title, vertical_deg=()):
    """Horizontal sections at z = 0, 15, 40 mm, and vertical ones (x, z) through the given azimuths [deg]."""
    r_mm, z_mm = PROFILES.r_cut * 1e3 + 10.0, PROFILES.z_cut * 1e3 + 10.0
    planes = [([0, 0, 1], [0, 0, z], 0, 1, f'z = {z * 1e3:+.0f} mm') for z in (0.0, 0.015, 0.040)]
    planes += [([-np.sin(np.radians(a)), np.cos(np.radians(a)), 0], [0, 0, 0], 0, 2,
                f'vertical through {a:.1f} deg ({name}): (x, z)') for name, a in vertical_deg]
    fig, axs = plt.subplots((len(planes) + 2) // 3, 3, figsize=(20, 6.5 * ((len(planes) + 2) // 3)), squeeze=False)
    for i, (ax, pl) in enumerate(zip(axs.ravel(), planes)):
        plot_sections(model3d, [pl], ax=ax, lim=((-r_mm, r_mm), (-r_mm, r_mm)) if i < 3
                      else ((-10.0, r_mm), (-z_mm, z_mm)))
        if i < 3:
            ax.plot(traj[:, 0] * 1e3, traj[:, 1] * 1e3, 'g-', lw=0.4)
    fig.suptitle(f"{title} ({model3d.n_elements} elements); blue = dees, red = ground, green = orbit", fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def main():
    ap = _common.arguments(__doc__)
    ap.add_argument('result', nargs='?', type=Path,
                    help='optimizer result pickle (default: <out>/optimized_cavity_geometry.pkl)')
    ap.add_argument('--skip-regression', action='store_true', help='skip the tall-wall regression (stage B)')
    ap.add_argument('--skip-bars', action='store_true', help='skip the model with bars (stage D)')
    ap.add_argument('--size-min', type=float, default=0.003, help='finest 3D element size [m] (default 0.003)')
    ap.add_argument('--size-max', type=float, default=0.015, help='coarsest 3D element size [m] (default 0.015)')
    ap.add_argument('--solver', default='auto', help="BEM solver: 'auto' (GPU when the matrix fits) or 'scipy'")
    ap.add_argument('--turns', type=int, default=15, help='turns to re-track (default 15)')
    ap.add_argument('--tip-width', type=float, default=TIP_WIDTH, help=f'dee tip wall [m] (default {TIP_WIDTH})')
    ap.add_argument('--bar-margin', type=float, default=BAR_MARGIN,
                    help=f'clearance between a bar and the orbit [m] (default {BAR_MARGIN})')
    ap.add_argument('--beam-halfwidth', type=float, default=BEAM_HALFWIDTH,
                    help='bunch half-width added to the bar clearance and the metal tests [m] (default 0)')
    args = _common.parse(ap)
    field_path = _common.require(args.resources / ex06.FIELD_FILE)
    result_pkl = _common.require(args.result or args.out / 'optimized_cavity_geometry.pkl',
                                 'run example 06 first, or name a result pickle')
    with open(result_pkl, 'rb') as f:
        result = pickle.load(f)
    traj = result.trajectory_reference
    geom = result.metadata['optimal_geometry']
    rf_hz = float(result.rf_frequency_mhz) * 1e6
    seam_extent = ((-R_SEAM[1] - 0.01, R_SEAM[1] + 0.01),) * 2

    def solve(model, label):
        sol = solve_gap_field(model, tol=1e-5, solver=args.solver, verbose=False)
        print(f"[{label}] {model.n_elements} elements: assembly {sol.assembly_time_s:.0f} s, "
              f"{sol.solver} GMRES {sol.solve_time_s:.0f} s ({sol.n_iterations} iterations)")
        return sol

    def build3d(model2d, profiles, bars=()):
        t0 = time.time()
        model = build_electrodes_3d(model2d, profiles, bars=bars, size_min=args.size_min, size_max=args.size_max,
                                    verbose=False)
        print(f"[3D] {len(bars)} bars, {model.n_elements} elements, built in {time.time() - t0:.0f} s")
        return model

    # ---- A. the winner, matched to its orbit, and its 2D electrodes -------------------------------------------
    print(f"[A] winner: {result.final_energy_mev:.3f} MeV in {result.n_turns} turns, phase "
          f"{result.bunch_phase_deg:.2f} deg, f = {result.rf_frequency_mhz:.6f} MHz")
    design, finder = ex07.build(field_path, rf_hz, geom, 'bem2d', snap_trajectory=traj, verbose=False)
    cavities = design.rf_cavities
    findings = check_variable_segments(design, traj,
                                       tip_clearance=max(float(c.gap_width_at(c.r_min)) for c in cavities))
    model2d = build_gap_electrodes(design, trim_trajectory=traj, verbose=False, **ex07.BUILD_KWARGS)
    print(f"[A] 2D electrodes: {model2d.n_elements} elements")
    beam = ex06.make_initial_beam(design.species)
    gap0 = float(cavities[0].base_angle)
    arc = (gap0 - GAP_ARC_DEG, gap0 + GAP_ARC_DEG)
    report = {'result_pkl': str(result_pkl),
              'winner': {'final_energy_mev': float(result.final_energy_mev), 'n_turns': int(result.n_turns)},
              'n_elements_2d': int(model2d.n_elements), 'segment_findings': findings,
              'gap_radii_m': GAP_RADII.tolist(), 'gap_arc_deg': list(arc),
              'size_min_m': args.size_min, 'size_max_m': args.size_max}

    # ---- B. tall-wall regression --------------------------------------------------------------------------------
    if not args.skip_regression:
        design_r, _ = ex07.build(field_path, rf_hz, geom, 'bem2d', snap_trajectory=traj, verbose=False)
        shrink(design_r, R_MAX_REGRESSION)
        model2d_r = build_gap_electrodes(design_r, trim_trajectory=traj, verbose=False, **ex07.BUILD_KWARGS)
        model_tw = build3d(model2d_r, HeightProfiles.tall_wall(model2d_r.params['height'],
                                                               r_cut=R_MAX_REGRESSION + 0.02))
        f2d_r = solve(model2d_r, 'B: 2D').to_field(spacing=FIELD_SPACING, extent=seam_extent, verbose=False)
        f_tw = solve(model_tw, 'B: tall-wall 3D').to_field(spacing=FIELD_SPACING, extent=seam_extent, verbose=False)
        reg = compare(f2d_r, f_tw, ~wedge_metal_mask(model2d_r.wedges, GRID))
        reg.update(n_elements_2d=int(model2d_r.n_elements), n_elements_3d=int(model_tw.n_elements),
                   gap_voltage_2d=gap_voltages(f2d_r, arc), gap_voltage_tallwall3d=gap_voltages(f_tw, arc))
        print(f"[B] tall-wall 3D vs 2D on the midplane (r <= {R_CMP * 1e3:.0f} mm): rms rel {reg['rms_rel']:.4f}, "
              f"median rel (significant) {reg['median_rel_significant']:.4f}")
        print("[B] gap voltage [kV] 2D / tall-wall 3D: " + ", ".join(
            f"r = {r * 1e3:.0f}: {a / 1e3:.2f} / {b / 1e3:.2f}"
            for r, a, b in zip(GAP_RADII, reg['gap_voltage_2d'], reg['gap_voltage_tallwall3d'])))
        report['regression'] = reg

    # ---- C. vertical structure ------------------------------------------------------------------------------------
    f2d = solve(model2d, 'C: 2D, full radius').to_field(spacing=FIELD_SPACING, verbose=False)
    model3d = build3d(model2d, PROFILES)
    plot_sections_figure(model3d, traj, args.out / 'electrodes3d_sections.png', '3D electrodes, open dee tips',
                         vertical_deg=(('dee 0 centre line', float(ex06.DEE_CENTER_ANGLES[0])), ('gap 0', gap0),
                                       ('x axis', 0.0)))
    f3d = solve(model3d, 'C: 3D').to_field(spacing=FIELD_SPACING, extent=seam_extent, verbose=False)
    obstacles = midplane_obstacles(model3d, beam_halfwidth=args.beam_halfwidth)
    vacuum = ~wedge_metal_mask(model2d.wedges, GRID) & ~obstacles(GRID)
    c = compare(f2d, f3d, vacuum)
    c.update(n_elements_3d=int(model3d.n_elements), gap_voltage_2d=gap_voltages(f2d, arc),
             gap_voltage_3d=gap_voltages(f3d, arc))
    print(f"[C] 3D vs 2D in the gaps (r <= {R_CMP * 1e3:.0f} mm): median |E| ratio (significant) "
          f"{c['median_ratio_significant']:.3f}, rms rel diff {c['rms_rel']:.3f}")
    print("[C] gap voltage [kV] 2D / 3D: " + ", ".join(
        f"r = {r * 1e3:.0f}: {a / 1e3:.2f} / {b / 1e3:.2f}"
        for r, a, b in zip(GAP_RADII, c['gap_voltage_2d'], c['gap_voltage_3d'])))
    report['aperture3d'] = c
    check_trajectory_clearance(obstacles, traj, label='winner orbit vs the open-tip 3D model')
    tracks = {'2D tall-wall': track(finder, beam, result, f2d, '2D tall-wall', args.turns)}
    tracks['composite, open tips'] = track(
        finder, beam, result, RadialBlendField(f3d, f2d, *R_SEAM, label='3D open tips inside, 2D outside'),
        'composite, open tips', args.turns, obstacles=obstacles)

    # ---- D. bars between the beam crossings ----------------------------------------------------------------------
    f3d_b = None
    if not args.skip_bars:
        bars = auto_bars(model2d, design, traj, margin=args.bar_margin, r_max=BAR_R_MAX, width=BAR_WIDTH,
                         tip=True, ground=True, tip_width=args.tip_width, beam_halfwidth=args.beam_halfwidth)
        clear = bar_clearances(model2d, bars, traj)
        print(f"[D] {len(bars)} bars (clearance to the orbit): " + ", ".join(
            f"{b.wedge}/{b.side}" + ("" if b.side == 'tip' else f" {b.r_from * 1e3:.0f}-{b.r_to * 1e3:.0f} mm")
            + f" ({cl * 1e3:+.1f} mm)" for b, cl in zip(bars, clear)))
        model3d_b = build3d(model2d, PROFILES, bars)
        obstacles_b = midplane_obstacles(model3d_b, beam_halfwidth=args.beam_halfwidth)
        # the orbit must clear every wall BEFORE the solve; this raises with the first contact point
        check_trajectory_clearance(obstacles_b, traj, label='winner orbit vs the barred 3D model')
        plot_sections_figure(model3d_b, traj, args.out / 'electrodes3d_bars_sections.png',
                             '3D electrodes with tip walls and bars between the crossings')
        f3d_b = solve(model3d_b, 'D: 3D with bars').to_field(spacing=FIELD_SPACING, extent=seam_extent,
                                                             verbose=False)
        vb = gap_voltages(f3d_b, arc)
        print("[D] gap voltage [kV] with bars: " + ", ".join(f"r = {r * 1e3:.0f}: {v / 1e3:.2f}"
                                                              for r, v in zip(GAP_RADII, vb)))
        report['bars'] = {'n_bars': len(bars), 'n_elements_3d': int(model3d_b.n_elements), 'gap_voltage_3d': vb,
                          'tip_width_m': args.tip_width, 'bar_margin_m': args.bar_margin,
                          'beam_halfwidth_m': args.beam_halfwidth,
                          'bars': [dict(vars(b), clearance_m=cl) for b, cl in zip(bars, clear)]}
        tracks['composite, bars'] = track(
            finder, beam, result, RadialBlendField(f3d_b, f2d, *R_SEAM, label='3D with bars inside, 2D outside'),
            'composite, bars', args.turns, obstacles=obstacles_b)

    turns = {label: e for label, (e, _) in tracks.items()}
    report['tracks'] = {label: {'energy_per_turn_mev': e, 'lost_at_mm': lost} for label, (e, lost) in tracks.items()}
    print("\nturn" + "".join(f"{k:>24s}" for k in turns) + "   [MeV]")
    for i in range(max(len(v) for v in turns.values())):
        print(f"{i + 1:4d}" + "".join(f"{v[i]:24.4f}" if i < len(v) else f"{'-':>24s}" for v in turns.values()))
    (args.out / 'electrodes3d.json').write_text(json.dumps(report, indent=2, default=float))
    plot_fields(f2d, f3d, f3d_b, tracks, traj, args.out / 'electrodes3d_fields.png')
    print(f"[done] wrote {args.out / 'electrodes3d.json'} and the figures")


def plot_fields(f2d, f3d, f3d_b, tracks, traj, png):
    """Midplane |E| of the 2D, the open-tip 3D and the barred 3D model, and the energy per turn of every track."""
    n = len(XS)
    ext = (-R_CMP * 1e3, R_CMP * 1e3, -R_CMP * 1e3, R_CMP * 1e3)
    mags = [(label, np.where(IN_CMP, np.linalg.norm(f(GRID)[:, :2], axis=1), np.nan).reshape(n, n))
            for label, f in (('2D tall-wall', f2d), ('3D, open tips', f3d), ('3D, bars', f3d_b)) if f is not None]
    vmax = np.nanpercentile(mags[0][1], 99.5) / 1e6
    fig, axs = plt.subplots(2, 2, figsize=(15, 13))
    axs[1, 0].axis('off')                                   # unless the barred model fills it
    for ax, (label, m) in zip(axs.ravel(), mags):
        ax.axis('on')
        im = ax.imshow(m.T / 1e6, origin='lower', extent=ext, vmin=0, vmax=vmax, cmap='magma')
        ax.plot(traj[:, 0] * 1e3, traj[:, 1] * 1e3, 'c-', lw=0.4)
        ax.set(title=f'|E| [MV/m], z = 0: {label}', xlabel='x [mm]', ylabel='y [mm]', xlim=ext[:2], ylim=ext[2:])
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axs[1, 1]
    for label, (e, lost) in tracks.items():
        ax.plot(range(1, len(e) + 1), e, 'o-', ms=3,
                label=label + (f' (lost on metal in turn {len(e) + 1})' if lost else ''))
    ax.set(xlabel='turn', ylabel='energy [MeV]', title='Re-tracked winner: energy per turn')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
