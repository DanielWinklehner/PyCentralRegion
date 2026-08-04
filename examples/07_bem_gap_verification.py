"""
07_bem_gap_verification.py - Verify the optimized cavity geometry with REAL
2D electrostatic gap fields (gap_model='bem2d').

Loads the example-06 winner (output/optimized_cavity_geometry.pkl), rebuilds
the system, applies the optimal per-gap geometry (angles / radii / rotations /
opening angle) and RF parameters, then tracks the same initial beam twice:

  1. thin-gap kick model (the search physics), and
  2. continuous integration of the solved BEM gap field: closed dee/ground
     electrodes built from the optimized segment chains (continuity re-imposed
     across rotated segments), one static Laplace solve, midplane E pattern
     gridded and modulated as E(x,y) * cos(omega t + bunch phase).

Differences are physics (real in-plane fringe fields, finite-gap transit) -
they are REPORTED, not force-matched.

Runtime: the BEM attach (mesh + solve + grid) takes a few minutes at the
default resolution; tracking at BEM_STEPS_PER_TURN adds a bit more.
Run with the radiacuda2 python (see tests/smoke_gapfield.py for the env).
"""
import sys
import json
import time
import pickle
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'src'))

from PyCentralRegion.accelerated_orbit_finder import AcceleratedOrbitFinder
from PyCentralRegion.gap_fields import plot_electrode_footprints

RESULT_PKL = ROOT / 'output' / 'optimized_cavity_geometry.pkl'
OUTPUT_DIR = ROOT / 'output'

# Continuous gap integration must resolve the ~gap-width field bump along the
# orbit; at the top energy that is only ~2-3 steps at the thin-gap default of
# 500. Both models are tracked at this resolution for a like-for-like dt.
BEM_STEPS_PER_TURN = 2000
MAX_TURNS = 15

# BEM resolution (see gap_fields.build_gap_electrodes / to_field docstrings).
# SCROLL central region (correct final treatment): the winner's own tracked
# trajectory is added as trim_trajectory at runtime - electrode inner edges
# follow turn 1 and the grounded scroll hugs it from inside, clearing the
# middle for the spiral inflector.
#   post_tip_gap        radial dee-tip-to-scroll gap (voltage window; the
#                       field there is ~ V / post_tip_gap)
#   traj_tip_clearance  beam-center-to-dee-tip clearance; default = the gap
#                       width at r_min (uncomment to override)
BUILD_KWARGS = dict(chain_ds=0.012, arc_ds=0.04,
                    post_tip_gap=0.005,
                    min_metal_width=0.001)
                    # traj_tip_clearance=0.010,
                    # fillet_radius=0.002)
FIELD_KWARGS = dict(spacing=0.0015)

# Shift each gap's segment node radii midway between its turn crossings
# (from the winner's thin-gap trajectory) so the segment joints never
# coincide with a beam crossing. Applied to BOTH re-tracks.
SNAP_NODES_MID_TURN = True


def _load_example06():
    """Import example 06 as a module (for its build_system + config)."""
    spec = importlib.util.spec_from_file_location(
        "example06", HERE / "06_optimize_cavity_geometry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _apply_optimal_geometry(design, dee_system, geom):
    """Apply the saved optimal per-gap geometry to freshly built cavities."""
    opening = geom.get('opening_angle_deg')
    if opening is not None:
        dee_system.apply_opening_angle(opening)
    rots_per_gap = geom.get('segment_rotations_per_gap')
    for g, cav in enumerate(design.rf_cavities):
        cav.update_geometry(
            segment_angles=list(geom['segment_angles_per_gap'][g]),
            segment_radii=list(geom['segment_radii_per_gap'][g]),
            segment_rotations=(list(rots_per_gap[g]) if rots_per_gap else None))


def _build(ex06, rf_freq_hz, geom, gap_model, snap_trajectory=None):
    design, finder, _, _ = ex06.build_system(rf_frequency=rf_freq_hz, quiet=True)
    # Rebuild the finder with the requested gap model / stepping (build_system
    # hard-wires thin-gap at 500 spt for the optimizer).
    finder = AcceleratedOrbitFinder(
        design, target_energy_mev=ex06.TARGET_ENERGY_MEV,
        max_radius_m=ex06.MAX_RADIUS_M, algorithm='rk4_rel',
        steps_per_turn=BEM_STEPS_PER_TURN, verbose=True, gap_model=gap_model)
    # find the DeeSystem: cavities were built from one; recreate the handle
    from PyCentralRegion.rf_cavity import DeeSystem, snap_nodes_between_turns
    dee_system = DeeSystem(list(ex06.DEE_CENTER_ANGLES), ex06.DEE_OPENING_ANGLE,
                           design.rf_cavities)
    _apply_optimal_geometry(design, dee_system, geom)
    if snap_trajectory is not None:
        snap_nodes_between_turns(design.rf_cavities, snap_trajectory,
                                 verbose=(gap_model == 'thin'))
    return design, finder


def main():
    if not RESULT_PKL.exists():
        print(f"ERROR: {RESULT_PKL} not found - run example 06 first.")
        sys.exit(1)
    with open(RESULT_PKL, 'rb') as f:
        result = pickle.load(f)
    geom = result.metadata['optimal_geometry']
    phase = float(result.bunch_phase_deg)
    rf_mhz = float(result.rf_frequency_mhz)
    print(f"Loaded winner: {result.final_energy_mev:.3f} MeV in {result.n_turns} "
          f"turns @ phase {phase:.2f} deg, f = {rf_mhz:.6f} MHz")

    ex06 = _load_example06()
    from PyPATools.species import IonSpecies
    beam = ex06.make_initial_beam(IonSpecies(ex06.SPECIES))

    snap_traj = result.trajectory_reference if SNAP_NODES_MID_TURN else None

    # ---- 1) thin-gap reference at matched resolution -------------------------
    print("\n--- thin-gap re-track ---")
    design_t, finder_t = _build(ex06, rf_mhz * 1e6, geom, 'thin',
                                snap_trajectory=snap_traj)
    res_thin = finder_t.track_once(beam, bunch_phase_deg=phase, rf_freq_mhz=rf_mhz,
                                   max_turns=MAX_TURNS)
    e_thin = [ts.mean_energy_mev for ts in res_thin.turn_statistics]

    # ---- 2) BEM gap fields ----------------------------------------------------
    print("\n--- BEM solve + re-track ---")
    design_b, finder_b = _build(ex06, rf_mhz * 1e6, geom, 'bem2d',
                                snap_trajectory=snap_traj)
    t0 = time.time()
    # Scroll central region built from the winner's own reference trajectory:
    # trims follow turn 1, so the gaps are active at every crossing by
    # construction (the max_r_inner guard is implied).
    build_kwargs = {**BUILD_KWARGS,
                    'trim_trajectory': result.trajectory_reference}
    finder_b.attach_bem_field(build_kwargs=build_kwargs,
                              field_kwargs=FIELD_KWARGS)
    print(f"attach_bem_field: {time.time() - t0:.1f} s "
          f"({finder_b.bem_solution.model.n_elements} elements)")
    res_bem = finder_b.track_once(beam, bunch_phase_deg=phase, rf_freq_mhz=rf_mhz,
                                  max_turns=MAX_TURNS)
    e_bem = [ts.mean_energy_mev for ts in res_bem.turn_statistics]

    # Did the BEM-tracked orbit stay clear of the scroll? (warn-only check)
    from PyCentralRegion.gap_fields import warn_if_trajectory_hits_post
    clear_m = warn_if_trajectory_hits_post(
        finder_b.bem_solution.model, res_bem.trajectory_reference,
        label="BEM re-tracked orbit")
    print(f"orbit-to-scroll clearance: {clear_m * 1000:.1f} mm")

    # ---- report ---------------------------------------------------------------
    h1_thin = res_thin.turn_metrics.get('r_center_h1', res_thin.turn_metrics.get('r_center', []))
    h1_bem = res_bem.turn_metrics.get('r_center_h1', res_bem.turn_metrics.get('r_center', []))
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
    ax.set_xlabel('turn')
    ax.set_ylabel('mean energy [MeV]')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('Energy per turn')

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

    plt.tight_layout()
    plot_file = OUTPUT_DIR / 'bem_gap_verification.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {plot_file}")

    summary = {
        'phase_deg': phase, 'rf_mhz': rf_mhz,
        'turns_thin_mev': e_thin, 'turns_bem_mev': e_bem,
        'final_thin_mev': e_thin[-1] if e_thin else None,
        'final_bem_mev': e_bem[-1] if e_bem else None,
        'n_elements': finder_b.bem_solution.model.n_elements,
        'solve_time_s': finder_b.bem_solution.solve_time_s,
    }
    (OUTPUT_DIR / 'bem_gap_verification.json').write_text(
        json.dumps(summary, indent=2, default=float))
    print(f"Saved summary to {OUTPUT_DIR / 'bem_gap_verification.json'}")
    plt.show()


if __name__ == "__main__":
    main()
