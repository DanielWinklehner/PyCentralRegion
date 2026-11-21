"""Thin-gap kicks vs solved gap fields: re-track an optimized cavity geometry through its 2D BEM field.

Shows: example 06's winner (geometry, RF) rebuilt on fresh cavities and matched to its own orbit
(neutralise_inert_segments straightens segments the beam never crosses, snap_nodes_between_turns moves
the segment joints off the beam crossings), then the same hand-off particle tracked twice at 2000 steps
per turn: (1) with the thin-gap kick model the search used, and (2) through the solved electrostatic
field of closed dee/ground electrodes built from the optimized segment chains (gap_model='bem2d',
AcceleratedOrbitFinder.attach_bem_field: one Laplace solve, midplane E gridded and modulated as
E(x, y) cos(omega t + phase)). The central region is a grounded scroll that follows the winner's own
first turn (trim_trajectory). Differences are physics (in-plane fringe fields, finite transit time) and
are reported, not fitted away.
Inputs: <resources>/uCyclo_v3_midplane_field.comsol and an optimizer result pickle (default
<out>/optimized_cavity_geometry.pkl from example 06; example 08 writes optimized_cavity_geometry_bem.pkl).
Runtime: 1-2 minutes on the CPU, most of it the BEM solve (25-27k elements, GMRES 40-80 s depending on the
geometry), then the midplane gridding (10 s) and two tracks.
Outputs: <out>/bem_gap_verification.png, bem_gap_verification.json.

Usage:
    python examples/07_bem_gap_verification.py [RESULT_PKL] [--resources DIR] [--out DIR] [--turns N]
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import _common
from PyCentralRegion.accelerated_orbit_finder import AcceleratedOrbitFinder
from PyCentralRegion.gap_fields import plot_electrode_footprints, warn_if_trajectory_hits_post
from PyCentralRegion.machine import apply_seed_geometry
from PyCentralRegion.rf_cavity import DeeSystem, neutralise_inert_segments, snap_nodes_between_turns

ex06 = _common.load_example('06_optimize_cavity_geometry.py')

# Continuous gap integration must resolve the ~gap-width field bump along the
# orbit; at the top energy that is only ~2-3 steps at the thin-gap default of
# 500. Both models are tracked at this resolution for a like-for-like dt.
BEM_STEPS_PER_TURN = 2000

# BEM resolution (see gap_fields.build_gap_electrodes / to_field docstrings).
# Scroll central region: the winner's own tracked trajectory is added as
# trim_trajectory at runtime - electrode inner edges follow turn 1 and the
# grounded scroll hugs it from inside, clearing the middle for the inflector.
#   post_tip_gap        radial dee-tip-to-scroll gap (voltage window; the
#                       field there is ~ V / post_tip_gap)
#   traj_tip_clearance  beam-center-to-dee-tip clearance; default = the gap
#                       width at r_min (pass traj_tip_clearance=... to override)
BUILD_KWARGS = dict(chain_ds=0.012, arc_ds=0.04, post_tip_gap=0.005, min_metal_width=0.001)
FIELD_KWARGS = dict(spacing=0.0015)

# Match the gap lines to the winner's thin-gap orbit: straighten the segments it
# never crosses (a kink no crossing uses), then shift each gap's segment node
# radii midway between its turn crossings so the segment joints never coincide
# with a beam crossing. Applied to BOTH re-tracks.
SNAP_NODES_MID_TURN = True


def build(field_path, rf_freq_hz, geom, gap_model, snap_trajectory=None, steps_per_turn=BEM_STEPS_PER_TURN,
          verbose=True):
    """Example 06's system with an optimizer result's geometry applied (and, with ``snap_trajectory``,
    matched to that orbit), and a finder for ``gap_model`` ('thin' or 'bem2d') at ``steps_per_turn``
    (build_system wires thin gaps at 500 for the search)."""
    design, _, _, _ = ex06.build_system(field_path, rf_frequency=rf_freq_hz, quiet=True)
    finder = AcceleratedOrbitFinder(
        design, target_energy_mev=ex06.TARGET_ENERGY_MEV,
        max_radius_m=ex06.MAX_RADIUS_M, algorithm='rk4_rel',
        steps_per_turn=steps_per_turn, verbose=verbose, gap_model=gap_model)
    # the cavities came from a DeeSystem; recreate the handle on them
    apply_seed_geometry(DeeSystem(list(ex06.DEE_CENTER_ANGLES), ex06.DEE_OPENING_ANGLE, design.rf_cavities),
                        geom)
    if snap_trajectory is not None:
        talk = verbose and gap_model == 'thin'
        neutralise_inert_segments(design.rf_cavities, snap_trajectory, verbose=talk)
        snap_nodes_between_turns(design.rf_cavities, snap_trajectory, verbose=talk)
    return design, finder


def main():
    ap = _common.arguments(__doc__)
    ap.add_argument('result', nargs='?', type=Path,
                    help='optimizer result pickle (default: <out>/optimized_cavity_geometry.pkl)')
    ap.add_argument('--turns', type=int, default=15, help='turns to track (default 15)')
    args = _common.parse(ap)
    field_path = _common.require(args.resources / ex06.FIELD_FILE)
    result_pkl = _common.require(args.result or args.out / 'optimized_cavity_geometry.pkl',
                                 'run example 06 first, or name a result pickle')
    with open(result_pkl, 'rb') as f:
        result = pickle.load(f)
    geom = result.metadata['optimal_geometry']
    phase = float(result.bunch_phase_deg)
    rf_mhz = float(result.rf_frequency_mhz)
    print(f"Loaded winner: {result.final_energy_mev:.3f} MeV in {result.n_turns} "
          f"turns @ phase {phase:.2f} deg, f = {rf_mhz:.6f} MHz")

    snap_traj = result.trajectory_reference if SNAP_NODES_MID_TURN else None

    # ---- 1) thin-gap reference at matched resolution -------------------------
    print("\n--- thin-gap re-track ---")
    design_t, finder_t = build(field_path, rf_mhz * 1e6, geom, 'thin', snap_trajectory=snap_traj)
    beam = ex06.make_initial_beam(design_t.species)
    res_thin = finder_t.track_once(beam, bunch_phase_deg=phase, rf_freq_mhz=rf_mhz, max_turns=args.turns)
    e_thin = [ts.mean_energy_mev for ts in res_thin.turn_statistics]

    # ---- 2) BEM gap fields ----------------------------------------------------
    print("\n--- BEM solve + re-track ---")
    design_b, finder_b = build(field_path, rf_mhz * 1e6, geom, 'bem2d', snap_trajectory=snap_traj)
    t0 = time.time()
    # Scroll central region built from the winner's own reference trajectory:
    # trims follow turn 1, so the gaps are active at every crossing by
    # construction (the max_r_inner guard is implied).
    finder_b.attach_bem_field(build_kwargs={**BUILD_KWARGS, 'trim_trajectory': result.trajectory_reference},
                              field_kwargs=FIELD_KWARGS)
    print(f"attach_bem_field: {time.time() - t0:.1f} s "
          f"({finder_b.bem_solution.model.n_elements} elements)")
    res_bem = finder_b.track_once(beam, bunch_phase_deg=phase, rf_freq_mhz=rf_mhz, max_turns=args.turns)
    e_bem = [ts.mean_energy_mev for ts in res_bem.turn_statistics]

    # Did the BEM-tracked orbit stay clear of the scroll? (warn-only check)
    clear_m = warn_if_trajectory_hits_post(finder_b.bem_solution.model, res_bem.trajectory_reference,
                                           label="BEM re-tracked orbit")
    print(f"orbit-to-scroll clearance: {clear_m * 1000:.1f} mm")

    # ---- report ---------------------------------------------------------------
    h1_thin = res_thin.turn_metrics['r_center_h1']
    h1_bem = res_bem.turn_metrics['r_center_h1']
    print("\nturn   E_thin [MeV]   E_bem [MeV]   dE [%]    h1_thin [mm]  h1_bem [mm]")
    for i in range(max(len(e_thin), len(e_bem))):
        et = e_thin[i] if i < len(e_thin) else float('nan')
        eb = e_bem[i] if i < len(e_bem) else float('nan')
        ht = h1_thin[i] * 1000 if i < len(h1_thin) else float('nan')
        hb = h1_bem[i] * 1000 if i < len(h1_bem) else float('nan')
        print(f"{i + 1:4d}   {et:12.4f}   {eb:11.4f}   {100 * (eb - et) / et if et else 0:+6.2f}"
              f"    {ht:12.2f}  {hb:11.2f}")

    # ---- plots ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    ax = axes[0]
    ax.plot(range(1, len(e_thin) + 1), e_thin, 'o-', label='thin-gap kick')
    ax.plot(range(1, len(e_bem) + 1), e_bem, 's-', label='BEM 2D field')
    ax.set(xlabel='turn', ylabel='mean energy [MeV]', title='Energy per turn')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    tt, tb = res_thin.trajectory_reference, res_bem.trajectory_reference
    ax.plot(tt[:, 0], tt[:, 1], '-', lw=0.7, alpha=0.7, label='thin')
    ax.plot(tb[:, 0], tb[:, 1], '-', lw=0.7, alpha=0.7, label='bem2d')
    for w in finder_b.bem_solution.model.wedges:
        ax.plot(np.append(w.polygon[:, 0], w.polygon[0, 0]),
                np.append(w.polygon[:, 1], w.polygon[0, 1]), 'k-', lw=0.5, alpha=0.5)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Orbits + electrode footprints')

    plot_electrode_footprints(finder_b.bem_solution.model, ax=axes[2])
    fig.tight_layout()
    plot_file = args.out / 'bem_gap_verification.png'
    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved {plot_file}")

    summary = {
        'result_pkl': str(result_pkl), 'phase_deg': phase, 'rf_mhz': rf_mhz,
        'turns_thin_mev': e_thin, 'turns_bem_mev': e_bem,
        'final_thin_mev': e_thin[-1] if e_thin else None,
        'final_bem_mev': e_bem[-1] if e_bem else None,
        'n_elements': finder_b.bem_solution.model.n_elements,
        'solve_time_s': finder_b.bem_solution.solve_time_s,
        'orbit_scroll_clearance_m': clear_m,
    }
    (args.out / 'bem_gap_verification.json').write_text(json.dumps(summary, indent=2, default=float))
    print(f"Saved {args.out / 'bem_gap_verification.json'}")
    plt.show()


if __name__ == "__main__":
    main()
