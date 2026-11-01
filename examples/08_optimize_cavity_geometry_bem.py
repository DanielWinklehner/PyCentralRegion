"""Cavity geometry optimization with solved 2D gap fields in the loop (gap_model='bem2d').

Shows: example 06's staged flow (RF scan -> multi-start DFO-LS -> verify) on example 06's system, but
every objective evaluation whose geometry changed first rebuilds the closed dee/ground electrodes,
re-solves the Laplace problem and re-grids the midplane field before tracking (RF-only moves re-sync
omega/phase without a re-solve), so the search optimizes against the real in-plane fringe fields instead
of thin-gap kicks. The search uses a grounded circular central post sized per candidate (post radius =
inner truncation - post_tip_gap); unbuildable candidates (build/solve failures, an inner truncation above
the injection radius) get a graded penalty. The winner is then re-solved with the scroll central region
built from its own first turn and re-tracked.
Inputs: <resources>/uCyclo_v3_midplane_field.comsol.
Runtime: every geometry change re-meshes and re-solves a coarse BEM model (about 13k elements) before a
2000-step/turn track, about 5-10 s per evaluation on the CPU, so the default 150 evaluations per start
take about 20 minutes (the starts run in parallel); --workers 2 --maxfun 5 --rf-frequency 42.2234e6
takes about 1.5 minutes. Verify the winner at full BEM resolution with example 07 afterwards.
Outputs: <out>/optimized_cavity_geometry_bem.pkl, cavity_geometry_optimization_bem.png,
cavity_geometry_optimization_bem.csv.worker-<pid>.csv (every evaluation).

Usage:
    python examples/08_optimize_cavity_geometry_bem.py [--resources DIR] [--out DIR] [--workers N]
        [--maxfun N] [--rf-frequency HZ]
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import _common
from PyCentralRegion.accelerated_orbit_finder import AcceleratedOrbitFinder
from PyCentralRegion.cavity_optimizer import CavityGeometryOptimizer
from PyCentralRegion.gap_fields import plot_electrode_footprints, warn_if_trajectory_hits_post
from PyCentralRegion.rf_cavity import DeeSystem

ex06 = _common.load_example('06_optimize_cavity_geometry.py')

# Search-stage BEM resolution: coarser than the example-07 verification
# (chain_ds 0.012 / arc_ds 0.04 / spacing 0.0015 / tol 1e-5) - candidate
# RANKING is what matters during the search, not absolute field accuracy.
# The grounded central post is IN the search: sized per candidate from its
# auto truncation (post radius = truncation - post_tip_gap, floored by
# post_min_radius), so the optimizer sees the post's field and the
# max_r_inner buildability guard stays active. Keep these values identical
# to example 07 so verification sees the same structure.
BEM_BUILD_KWARGS = dict(chain_ds=0.02, arc_ds=0.06, post_tip_gap=0.005, post_min_radius=0.010)
BEM_SOLVE_KWARGS = dict(tol=1e-4)
BEM_FIELD_KWARGS = dict(spacing=0.003)

# bem2d tracking must resolve the ~gap-width field bump: >= ~2000 steps/turn.
STEPS_PER_TURN = 2000
SEARCH_MAX_TURNS = 8
FINAL_MAX_TURNS = 12

# Each worker holds its own field map AND transient dense BEM matrices
# (~n_elements^2 * 8 bytes during each solve) - keep the pool small.
N_WORKERS = max(1, min(6, (os.cpu_count() or 2) - 2))


def build_system_bem(field_path, rf_frequency=None, quiet=False, checkpoint_file=None):
    """Build (design, finder, geometry optimizer, rf_frequency) for bem2d.

    Module-level and deterministic so each multiprocessing worker can rebuild
    the identical system (same contract as example 06's build_system).
    """
    design, _, _, rf_frequency = ex06.build_system(field_path, rf_frequency=rf_frequency, quiet=quiet)
    finder = AcceleratedOrbitFinder(
        design, target_energy_mev=ex06.TARGET_ENERGY_MEV,
        max_radius_m=ex06.MAX_RADIUS_M, algorithm='rk4_rel',
        steps_per_turn=STEPS_PER_TURN, verbose=not quiet, gap_model='bem2d')
    dee_system = DeeSystem(list(ex06.DEE_CENTER_ANGLES), ex06.DEE_OPENING_ANGLE, design.rf_cavities)

    # Buildability guard: electrodes must exist at the injection radius, or
    # the first gap crossings see no field (per-eval graded penalty).
    beam = ex06.make_initial_beam(design.species)
    r_inject = float(np.hypot(beam.centroid[0], beam.centroid[1]))

    geo_optimizer = CavityGeometryOptimizer(
        orbit_finder=finder, n_segments=ex06.N_VARIABLE_SEGMENTS,
        max_angle_variable=10.0, max_r_variable=0.20, r_min_cavity=ex06.R_MIN,
        verbose=not quiet, checkpoint_file=checkpoint_file,
        dee_system=dee_system,
        optimize_opening_angle=ex06.OPTIMIZE_OPENING_ANGLE,
        opening_delta_max=ex06.OPENING_DELTA_MAX,
        rotatable_segments=ex06.ROTATABLE_SEGMENTS,
        rotation_max=ex06.ROTATION_MAX,
        bem_build_kwargs={**BEM_BUILD_KWARGS, 'max_r_inner': r_inject},
        bem_solve_kwargs=dict(BEM_SOLVE_KWARGS),
        bem_field_kwargs=dict(BEM_FIELD_KWARGS))
    return design, finder, geo_optimizer, rf_frequency


def build_worker_optimizer(field_path, rf_frequency):
    """Worker-side builder (module-level, picklable by reference)."""
    _, _, geo, _ = build_system_bem(field_path, rf_frequency=rf_frequency, quiet=True)
    return geo


def main():
    ap = _common.arguments(__doc__)
    ap.add_argument('--workers', type=int, default=N_WORKERS,
                    help=f'parallel worker processes (default {N_WORKERS}; 1 = serial)')
    ap.add_argument('--maxfun', type=int, default=150, help='DFO-LS evaluations per start (default 150)')
    ex06.add_rf_option(ap)
    args = _common.parse(ap)
    field_path = _common.require(args.resources / ex06.FIELD_FILE)
    output_dir = args.out
    print("=" * 70)
    print("CAVITY GEOMETRY + RF OPTIMIZATION - BEM 2D GAP FIELDS IN THE LOOP")
    print("=" * 70)
    checkpoint_file = output_dir / 'cavity_geometry_optimization_bem.csv'

    print("\n1. Building system (field load + isochronous-frequency scan)...")
    design, finder, geo_optimizer, rf_frequency = build_system_bem(
        field_path, rf_frequency=args.rf_frequency, checkpoint_file=str(checkpoint_file))
    print(f"   Species: {design.species.name}, target {ex06.TARGET_ENERGY_MEV} MeV")
    print(f"   RF base frequency: {rf_frequency / 1e6:.4f} MHz")
    print(f"   {len(design.rf_cavities)} gaps x {ex06.N_VARIABLE_SEGMENTS} "
          f"variable segment(s), bem2d @ {STEPS_PER_TURN} steps/turn")
    print(f"   Search BEM resolution: build {BEM_BUILD_KWARGS}, "
          f"solve {BEM_SOLVE_KWARGS}, field {BEM_FIELD_KWARGS}")
    print(f"\n   NOTE: every geometry move re-meshes and re-solves the BEM field (seconds each). Budget "
          f"{args.maxfun} evals/start x {args.workers} parallel starts.")

    initial_beam = ex06.make_initial_beam(design.species)
    result = geo_optimizer.optimize_staged(
        initial_beam,
        rf_optimize_params=['bunch_phase', 'rf_freq'],
        rf_bounds={'bunch_phase': (-180, 180),
                   'rf_freq': (rf_frequency * 0.95, rf_frequency * 1.05)},
        search_steps_per_turn=STEPS_PER_TURN,
        search_max_turns=SEARCH_MAX_TURNS,
        final_steps_per_turn=STEPS_PER_TURN,
        final_max_turns=FINAL_MAX_TURNS,
        maxfun=args.maxfun,
        r0_mode='offset',
        skip_turns=2,
        workers=args.workers,
        worker_builder=build_worker_optimizer,
        worker_builder_args=(field_path, rf_frequency),
    )

    # ========================================================================
    # Results summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS (bem2d in the loop)")
    print("=" * 70)
    print(f"\nFinal energy: {result.final_energy_mev:.3f} MeV in {result.n_turns} turns "
          f"(target {ex06.TARGET_ENERGY_MEV} MeV, success {result.success}, cost {result.cost:.2e})")
    print(f"Bunch phase: {result.bunch_phase_deg:.2f} deg, RF frequency: {result.rf_frequency_mhz:.6f} MHz")
    print("Optimal cavity geometry (per gap):")
    print("\n".join("  " + line for line in ex06.geometry_lines(result.metadata['optimal_geometry'])))

    # ========================================================================
    # Plots: orbit + electrodes, energy/turn, convergence, footprints
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    ax = axes[0, 0]
    traj = result.trajectory_reference
    ax.plot(traj[:, 0], traj[:, 1], 'b-', lw=0.8, alpha=0.8, label='orbit')
    if finder.bem_solution is not None:
        for w in finder.bem_solution.model.wedges:
            ax.plot(np.append(w.polygon[:, 0], w.polygon[0, 0]),
                    np.append(w.polygon[:, 1], w.polygon[0, 1]), 'k-', lw=0.5, alpha=0.5)
    ax.set(xlabel='x [m]', ylabel='y [m]', aspect='equal',
           title='Optimized orbit + electrode footprints (search resolution)')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    e_circ = [p.energy_mev for p in result.poincare_points_all[0]]
    ax.plot([p.turn for p in result.poincare_points_all[0]], e_circ, 'o-', lw=2, ms=4)
    ax.axhline(ex06.TARGET_ENERGY_MEV, color='r', ls='--', label=f'target {ex06.TARGET_ENERGY_MEV} MeV')
    ax.set(xlabel='turn', ylabel='energy [MeV]', title='Energy vs turn (BEM fields)')
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    history = ex06.read_checkpoints(checkpoint_file)
    for source, rows in (history.groupby('source') if len(history) else []):
        ax.semilogy(rows['cost'].to_numpy(), '-', lw=0.8, alpha=0.6)
    ax.set(xlabel='evaluation (per worker)', ylabel='cost',
           title=f"Convergence ({history['source'].nunique() if len(history) else 0} files, "
                 f"{len(history)} evals)")
    ax.grid(alpha=0.3)

    if finder.bem_solution is not None:
        plot_electrode_footprints(finder.bem_solution.model, ax=axes[1, 1])
        axes[1, 1].set_title('Final gap electrodes (search resolution)')
    fig.suptitle(f'BEM-in-the-loop cavity optimization: E_final = {result.final_energy_mev:.2f} MeV',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    plot_file = output_dir / 'cavity_geometry_optimization_bem.png'
    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved {plot_file}")

    # ========================================================================
    # Final evaluation with the scroll central region: the search used the
    # circular post; re-solve with the scroll built from the winner's own
    # trajectory and re-track (see gap_fields trim_trajectory).
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL EVAL: scroll central region (search used the circular post)")
    print("=" * 70)
    scroll_kwargs = {k: v for k, v in BEM_BUILD_KWARGS.items() if k not in ('post_min_radius', 'max_r_inner')}
    scroll_kwargs['trim_trajectory'] = result.trajectory_reference
    finder.attach_bem_field(build_kwargs=scroll_kwargs, field_kwargs=dict(BEM_FIELD_KWARGS))
    res_scroll = finder.track_once(initial_beam, bunch_phase_deg=float(result.bunch_phase_deg),
                                   rf_freq_mhz=float(result.rf_frequency_mhz), max_turns=FINAL_MAX_TURNS)
    clear_m = warn_if_trajectory_hits_post(finder.bem_solution.model, res_scroll.trajectory_reference,
                                           label="scroll-center re-tracked orbit")
    e_scr = [ts.mean_energy_mev for ts in res_scroll.turn_statistics]
    print("\nturn   E_circular-post [MeV]   E_scroll [MeV]")
    for i in range(max(len(e_circ), len(e_scr))):
        ec = e_circ[i] if i < len(e_circ) else float('nan')
        es = e_scr[i] if i < len(e_scr) else float('nan')
        print(f"{i + 1:4d}   {ec:20.4f}   {es:14.4f}")
    print(f"\nScroll-center final: {res_scroll.final_energy_mev:.3f} MeV in {res_scroll.n_turns} turns; "
          f"orbit-to-scroll clearance {clear_m * 1000:.1f} mm")
    result.metadata['scroll_final_energy_mev'] = float(res_scroll.final_energy_mev)
    result.metadata['scroll_final_turns'] = int(res_scroll.n_turns)
    result.metadata['scroll_orbit_clearance_m'] = float(clear_m)

    result_file = output_dir / 'optimized_cavity_geometry_bem.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved {result_file}")
    print(f"\nVerify the winner at full BEM resolution with: "
          f"python examples/07_bem_gap_verification.py {result_file}")
    plt.show()


if __name__ == "__main__":
    main()
