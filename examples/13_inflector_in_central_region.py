"""
13_inflector_in_central_region.py - The spiral inflector inside the HCHC-60
central-region model: static field + housing + bunch in one simulation.

Since 2026-09-10 the spyral_inflector deck exports, in the machine frame,
  * the housing as a STEP solid (mm),
  * the static E-field of the inflector as a PyPATools Field pickle, and
  * the bunch at the plane through the ends of the spiral electrodes.
Everything downstream of that plane is simulated here:

  A. load the inflector (InflectorModel), section the housing at the midplane,
     overlay it with the CR winner's electrodes and the inflector field;
  B. reference orbit from the inflector's design exit state through the
     thin-gap model WITH the static field superposed and the housing as an
     obstacle (attach_inflector);
  C. 2D BEM electrodes with the housing as the grounded scroll
     (build_gap_electrodes(housing=...)): the housing wall is the dummy dee of
     the first gap, the dummy dees merge onto it where they reach it; solve
     (CPU) and re-track the reference on the superposed field;
  D. a bunch (the hand-off file if given - timed release - or a Gaussian bunch
     around the design exit) tracked through the same model: losses on the
     housing, energy per turn.

The CR winner was optimised for an older injection point (r 59.7 mm, az
0.7 deg); the inflector exit sits at r 70.4 mm, az 26.2 deg. The inflector is
rotated by --rotation (default: exit azimuth onto the winner's injection
azimuth) so that the geometry is roughly consistent; the radial mismatch stays.
This is the mechanics demonstration; the next CR optimisation takes the
inflector as fixed and moves the dees.

Usage:
    python 13_inflector_in_central_region.py [--no-solve] [--bunch H5] [--n-particles 300]
                                             [--turns 12] [--rotation DEG] [--solver scipy]
Env: CONDA_PREFIX=<accel-dev-env> MPLBACKEND=Agg
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'examples'))

SI_DIR = Path(os.environ.get('HCHC60_SI_DIR',
                             r"D:\MIT Dropbox\Daniel Winklehner\Projects\IsoDAR\60 MeV Cyclotron\Spiral_inflector"))
HOUSING_STEP = SI_DIR / 'Geometry' / 'rotation' / 'machine_frame_Rm84p4' / '002_Housing.step'   # mm, machine frame
EFIELD = SI_DIR / 'Results' / 'rotation' / 'exit_plane' / 'rotm84.4' / 'ef_itp_inner_best.pickle'  # deck frame (rotated)
OUT = ROOT / 'output'
# design exit of the rotated inflector (Results/rotation/exit_plane/rotm84.4/summary.json)
EXIT_R_MM, EXIT_AZ_DEG, EXIT_PR_OVER_P, EXIT_ENERGY_KEV = 70.381, 26.157, 0.2581, 68.44
WINNER_INJECTION_AZ_DEG = 0.67
# housing shell: 4 mm elements on the 4 mm wall, 6 mm median-plane rows (at 2 mm /
# 3 mm rows the housing alone was 26k elements and the GMRES needed 4000 iterations)
BUILD_KWARGS = dict(chain_ds=0.012, arc_ds=0.04, post_tip_gap=0.005, min_metal_width=0.001,
                    housing_ds=0.004, housing_wall_dz=0.006, spoke_max=0.020)
STEPS_PER_TURN = 2000


def exit_state(rotation_deg):
    """Position / velocity of the inflector's design particle at its exit, rotated."""
    from PyPATools.global_variables import CLIGHT
    az = np.radians(EXIT_AZ_DEG + rotation_deg)
    r_hat = np.array([np.cos(az), np.sin(az), 0.0])
    phi_hat = np.array([-np.sin(az), np.cos(az), 0.0])
    x = EXIT_R_MM * 1e-3 * r_hat
    d = EXIT_PR_OVER_P * r_hat + np.sqrt(1.0 - EXIT_PR_OVER_P ** 2) * phi_hat
    return x, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-solve', action='store_true', help='thin-gap model only (skip the BEM stage)')
    ap.add_argument('--bunch', default=None, help='hand-off h5 (plane-crossing) to track in stage D')
    ap.add_argument('--n-particles', type=int, default=300)
    ap.add_argument('--turns', type=int, default=12)
    ap.add_argument('--rotation', type=float, default=WINNER_INJECTION_AZ_DEG - EXIT_AZ_DEG,
                    help='frame rotation of the inflector (housing, field, bunch) [deg]')
    ap.add_argument('--solver', default='scipy', help="BEM solver ('scipy' keeps the GPU free)")
    ap.add_argument('--tip-clearance', type=float, default=0.012, help='beam-centre-to-dee-tip clearance [m]')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PyPATools.species import IonSpecies
    from PyPATools.global_variables import CLIGHT
    from PyCentralRegion.inflector import InflectorModel
    from PyCentralRegion.accelerated_orbit_finder import make_beam_from_state
    from PyCentralRegion.gap_fields import warn_if_trajectory_hits_post, plot_electrode_footprints
    from PyCentralRegion.electrodes3d import check_trajectory_clearance
    import importlib
    ex12 = importlib.import_module('12_electrodes3d_hchc60')
    report = {'args': vars(args)}

    # ---- A. the inflector ------------------------------------------------------
    t0 = time.time()
    # the field pickle is DECK frame (z-mirror needed), the STEP was exported in the
    # machine frame already (mirrored by the deck), a hand-off h5 says so itself
    infl = InflectorModel(bunch=args.bunch, efield=EFIELD, housing=HOUSING_STEP, rotation_deg=args.rotation,
                          field_flip_z=True, housing_flip_z=False, bunch_flip_z=False,
                          housing_scale=1e-3, name='housing3_Rm84p4')
    loops = infl.housing_loops(0.0)
    field2d = infl.midplane_field(0.0)
    print(f"[A] {infl}\n[A] housing section: {len(loops)} loop(s), "
          f"{sum(len(l) for l in loops)} points; field grid {infl.summary()['field_grid_mm']} ({time.time() - t0:.0f} s)")
    report['inflector'] = infl.summary()

    # ---- the CR winner ---------------------------------------------------------
    ex06, result = ex12.load_winner()
    design, finder = ex12.build_design(ex06, result, gap_model='thin')
    ex12.straighten_gaps(design, ex12.STRAIGHTEN_GAPS)
    finder.verbose = False
    phase, rf_mhz = float(result.bunch_phase_deg), float(result.rf_frequency_mhz)
    species = IonSpecies(ex06.SPECIES)
    x0, d0 = exit_state(args.rotation)
    gamma = 1.0 + EXIT_ENERGY_KEV * 1e3 / (species.mass_mev * 1e6)
    v0 = d0 * CLIGHT * np.sqrt(1.0 - 1.0 / gamma ** 2)
    ref_beam = make_beam_from_state(species, x0, v0)
    print(f"[A] winner: {result.final_energy_mev:.3f} MeV in {result.n_turns} turns, phase {phase:.1f} deg, "
          f"RF {rf_mhz:.3f} MHz; inflector exit at r = {np.hypot(*x0[:2]) * 1e3:.1f} mm, "
          f"az = {np.degrees(np.arctan2(x0[1], x0[0])):.1f} deg, {EXIT_ENERGY_KEV:.1f} keV")

    # ---- B. reference through the thin-gap model + inflector -----------------------
    finder.attach_inflector(infl, obstacle_extent=finder.r_max)
    t0 = time.time()
    res_thin = finder.track_once(ref_beam, bunch_phase_deg=phase, rf_freq_mhz=rf_mhz, max_turns=args.turns)
    tr_thin = np.asarray(res_thin.trajectory_reference)
    e_thin = [float(ts.mean_energy_mev) for ts in res_thin.turn_statistics]
    clr = infl.clearance(tr_thin, skip_deg=25.0)
    print(f"[B] thin gaps + inflector field: {res_thin.n_turns} turns, {res_thin.final_energy_mev:.3f} MeV "
          f"({time.time() - t0:.0f} s); housing clearance after the exit {clr['min_distance_m'] * 1e3:.1f} mm "
          f"at {clr['at_mm']}; obstacles {res_thin.metadata.get('obstacles')}")
    finder.detach_inflector()
    res_bare = finder.track_once(ref_beam, bunch_phase_deg=phase, rf_freq_mhz=rf_mhz, max_turns=args.turns)
    e_bare = [float(ts.mean_energy_mev) for ts in res_bare.turn_statistics]
    print(f"[B] thin gaps, no inflector field: {res_bare.n_turns} turns, {res_bare.final_energy_mev:.3f} MeV")
    finder.attach_inflector(infl, obstacle_extent=finder.r_max)
    report['B'] = {'thin_with_inflector': e_thin, 'thin_bare': e_bare, 'housing_clearance': clr,
                   'obstacles': res_thin.metadata.get('obstacles')}

    # ---- C. BEM with the housing as the scroll -----------------------------------
    e_bem, model2d, tr_bem, res_bem = None, None, None, None
    if not args.no_solve:
        t0 = time.time()
        design_b, finder_b = ex12.build_design(ex06, result, gap_model='bem2d')
        ex12.straighten_gaps(design_b, ex12.STRAIGHTEN_GAPS)
        finder_b.verbose = True
        finder_b.attach_inflector(infl, obstacle_extent=finder_b.r_max)
        build = dict(BUILD_KWARGS, trim_trajectory=tr_thin, traj_tip_clearance=args.tip_clearance,
                     housing=infl.housing_polygon(0.0))
        with warnings.catch_warnings(record=True) as wlog:
            warnings.simplefilter('always')
            finder_b.attach_bem_field(build_kwargs=build, solve_kwargs=dict(solver=args.solver, tol=1e-5),
                                      field_kwargs=dict(spacing=0.0015))
        model2d = finder_b.bem_solution.model
        info = model2d.params['housing']
        print(f"[C] BEM model: {model2d.n_elements} elements, housing union {info['union_faces']} face(s), "
              f"spokes merged {info['spokes']}, separate {info['unmerged_ground']}, holes {info['holes']} "
              f"({time.time() - t0:.0f} s); warnings: {[str(w.message)[:100] for w in wlog]}")
        print(f"[C] efield = {design_b.efield}")
        t0 = time.time()
        res_bem = finder_b.track_once(ref_beam, bunch_phase_deg=phase, rf_freq_mhz=rf_mhz, max_turns=args.turns)
        tr_bem = np.asarray(res_bem.trajectory_reference)
        e_bem = [float(ts.mean_energy_mev) for ts in res_bem.turn_statistics]
        clr_b = infl.clearance(tr_bem, skip_deg=25.0)
        post_clr = warn_if_trajectory_hits_post(model2d, tr_bem, label='BEM re-tracked orbit')
        print(f"[C] BEM + inflector: {res_bem.n_turns} turns, {res_bem.final_energy_mev:.3f} MeV ({time.time() - t0:.0f} s); "
              f"housing clearance {clr_b['min_distance_m'] * 1e3:.1f} mm, obstacles {res_bem.metadata.get('obstacles')}")
        report['C'] = {'n_elements': model2d.n_elements, 'housing': {k: v for k, v in info.items() if k != 'loops'},
                       'energies': e_bem, 'housing_clearance': clr_b, 'post_clearance_m': float(post_clr),
                       'obstacles': res_bem.metadata.get('obstacles'), 'warnings': [str(w.message) for w in wlog]}
        finder = finder_b

    # ---- D. bunch ----------------------------------------------------------------
    t0 = time.time()
    if args.bunch:
        bunch = infl.beam(n_particles=args.n_particles, seed=0)
        tag = f"hand-off bunch ({os.path.basename(args.bunch)})"
    else:
        # Gaussian bunch around the design exit: 1.5 mm, 3 deg, 1 % energy spread
        from PyPATools.particles import ParticleDistribution
        tag = 'Gaussian bunch around the design exit'
        rng = np.random.default_rng(0)
        n = args.n_particles
        xs = x0[None, :] + rng.normal(0.0, 0.0015, (n, 3)) * np.array([1.0, 1.0, 0.0])
        ang = rng.normal(0.0, np.radians(3.0), n)
        c, s = np.cos(ang), np.sin(ang)
        vs = np.column_stack([c * v0[0] - s * v0[1], s * v0[0] + c * v0[1], np.zeros(n)])
        vs *= (1.0 + rng.normal(0.0, 0.01, n))[:, None]
        bunch = ParticleDistribution(species=species, x_vec=xs, p_vec=np.zeros_like(xs))
        bunch.set_p_from_v_vec(vs)
    res_bunch = finder.track_once(bunch, bunch_phase_deg=phase, rf_freq_mhz=rf_mhz, max_turns=args.turns)
    obs = res_bunch.metadata.get('obstacles') or {}
    e_bunch = [float(ts.mean_energy_mev) for ts in res_bunch.turn_statistics]
    n_alive = [int(ts.n_active) for ts in res_bunch.turn_statistics]
    print(f"[D] {tag}: {bunch.numpart} particles, {res_bunch.n_turns} turns, {res_bunch.final_energy_mev:.3f} MeV, "
          f"lost on the housing {obs.get('n_lost')} ({time.time() - t0:.0f} s); alive per turn {n_alive}")
    report['D'] = {'tag': tag, 'n_particles': int(bunch.numpart), 'energies': e_bunch, 'alive': n_alive,
                   'obstacles': {k: v for k, v in obs.items() if k != 'hits'}}
    hits = np.array([(h[2], h[3]) for h in obs.get('hits', [])]) if obs.get('hits') else np.zeros((0, 2))

    # ---- figure --------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
    ax = axes[0]
    g = field2d.grid
    X, Y = np.meshgrid(g['x'], g['y'], indexing='ij')
    E = np.hypot(field2d.grid_values['x'], field2d.grid_values['y'])
    cf = ax.contourf(X * 1e3, Y * 1e3, np.log10(np.maximum(E, 1e2)), levels=np.linspace(2, 6.5, 19), cmap='viridis', alpha=0.85)
    fig.colorbar(cf, ax=ax, label='log10 |E_inflector| [V/m] at z = 0')
    if model2d is not None:
        for w in model2d.wedges:
            col = {'dee': 'tab:red', 'ground': 'tab:blue', 'post': 'k'}[w.kind]
            p = np.vstack([w.polygon, w.polygon[:1]])
            ax.plot(p[:, 0] * 1e3, p[:, 1] * 1e3, '-', color=col, lw=1.0 if w.kind != 'post' else 1.6)
            for h in w.holes:
                hh = np.vstack([h, h[:1]])
                ax.plot(hh[:, 0] * 1e3, hh[:, 1] * 1e3, '-', color='k', lw=1.0)
    for l in loops:
        ll = np.vstack([l, l[:1]])
        ax.plot(ll[:, 0] * 1e3, ll[:, 1] * 1e3, 'w-', lw=0.8)
    ax.plot(tr_thin[:, 0] * 1e3, tr_thin[:, 1] * 1e3, '-', color='orange', lw=0.7, label='reference, thin gaps + inflector')
    if tr_bem is not None:
        ax.plot(tr_bem[:, 0] * 1e3, tr_bem[:, 1] * 1e3, '-', color='lime', lw=0.7, label='reference, BEM (housing = scroll)')
    if len(hits):
        ax.plot(hits[:, 0] * 1e3, hits[:, 1] * 1e3, 'rx', ms=4, label=f'bunch lost on the housing ({len(hits)})')
    ax.plot(x0[0] * 1e3, x0[1] * 1e3, 'r*', ms=10, label='inflector design exit')
    ax.set_xlim(-160, 160)
    ax.set_ylim(-160, 160)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    ax.legend(loc='lower left', fontsize=8)
    ax.set_title(f"HCHC-60 CR winner (rotated inflector {args.rotation:+.1f} deg): housing (white/black), "
                 f"dees (red), dummy dees (blue)", fontsize=10)
    ax = axes[1]
    ax.plot(range(1, len(e_bare) + 1), e_bare, 'o-', label='thin gaps, no inflector field')
    ax.plot(range(1, len(e_thin) + 1), e_thin, 's-', label='thin gaps + inflector field')
    if e_bem is not None:
        ax.plot(range(1, len(e_bem) + 1), e_bem, '^-', label='BEM (housing = scroll) + inflector field')
    ax.plot(range(1, len(e_bunch) + 1), e_bunch, 'd--', label=f'{tag}: mean energy')
    ax.set_xlabel('turn')
    ax.set_ylabel('energy [MeV]')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('reference and bunch energies per turn')
    fig.tight_layout()
    png = OUT / 'inflector_in_central_region.png'
    fig.savefig(png, dpi=120)
    (OUT / 'inflector_in_central_region.json').write_text(json.dumps(report, indent=2, default=lambda o: str(o)))
    print(f"wrote {png} and the JSON report")


if __name__ == '__main__':
    main()
