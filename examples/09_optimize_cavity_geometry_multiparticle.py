"""
09_optimize_cavity_geometry_multiparticle.py - Multiparticle Cavity Geometry
Optimization (real inflector-exit bunch).

Same staged flow as example 06 (RF scan -> multi-start DFO-LS -> verify), but
the objective tracks the REAL spiral-inflector exit bunch loaded from
resources/MuonBunchOutOfInflector.csv instead of a single reference particle.
With numpart > 1 the orbit finder's residual vector automatically grows two
blocks (see AcceleratedOrbitFinder.objective_residuals):

  - envelope:  per-turn radial beam spread (std_r / 5 mm), and
  - survival:  per-turn lost fraction (weighted like energy - losing the
               beam hurts as much as not accelerating it).

so the search optimizes bunch transmission and beam quality, not just the
centroid orbit. System configuration (field map, dee layout, variable
segments, pinch tie-breaker, ...) is INHERITED from example 06's
build_system - single source of truth; this file only swaps the beam and
the run bookkeeping.

The bunch is 2D-projected for this midplane study: z and vz are dropped
(vertical dynamics is 3D roadmap physics). The bunch CENTROID (mean position
and velocity) is prepended as particle 0, so trajectory_reference and the
per-turn Poincare metrics describe the centroid orbit.

COST: every objective evaluation tracks the full bunch, so expect roughly
linear scaling in particle count per evaluation. Use N_PARTICLES to
subsample for quick scans; the full 250-particle bunch is fine for a
production overnight run.

Usage:
    python 09_optimize_cavity_geometry_multiparticle.py
"""

import os
import sys
import csv
import pickle
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))

from PyCentralRegion.accelerated_orbit_finder import make_beam_from_state

# ============================================================================
# Configuration (system config is INHERITED from example 06 - single source)
# ============================================================================
BUNCH_CSV = HERE.parent / 'resources' / 'MuonBunchOutOfInflector.csv'

# None = use every particle in the CSV; an int subsamples (deterministic).
N_PARTICLES = None
SUBSAMPLE_SEED = 42

# Prepend the bunch centroid as particle 0 (the reference for
# trajectory_reference / turn metrics / Poincare plots).
PREPEND_CENTROID = True

# Least-squares residual weights. Defaults live in
# accelerated_orbit_finder.DEFAULT_LS_WEIGHTS = {'energy': 4.0, 'center': 1.0,
# 'smooth': 0.5, 'envelope': 0.5, 'survival': 4.0, 'phase': 0.5}; entries here
# override them. envelope/survival only act with numpart > 1.
LS_WEIGHTS = {
    # envelope raised 0.5 -> 2.0 (2026-08-04): the extraction studies
    # showed the caught-turn radial spread (~4 mm rms) drives the septum
    # foil shadow - weight the radial beam size accordingly.
    'envelope': 2.0,
    'survival': 4.0,
    # one-time price of the phase-slit collimator's removals (beam count
    # at the END; dynamic survival is measured post-collimator)
    'collimated': 4.0,
}

# Staged-run settings (mirroring example 06's main()).
SEARCH_STEPS_PER_TURN = 300
SEARCH_MAX_TURNS = 10
FINAL_STEPS_PER_TURN = 500
FINAL_MAX_TURNS = 15
MAXFUN_PER_START = 600
SKIP_TURNS = 4

# Each worker holds its own field map and tracks the full bunch per eval.
N_WORKERS = max(1, min(12, (os.cpu_count() or 2) - 2))


def _load_example06():
    """Import example 06 as a module (config + build_system)."""
    spec = importlib.util.spec_from_file_location(
        "example06", HERE / "06_optimize_cavity_geometry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_bunch(species):
    """The inflector-exit bunch as a 2D-projected ParticleDistribution.

    Reads x,y,z [m] and vx,vy,vz [m/s] from BUNCH_CSV, zeroes z/vz (midplane
    study), optionally subsamples to N_PARTICLES, and prepends the centroid
    as particle 0 (PREPEND_CENTROID).
    """
    data = np.genfromtxt(BUNCH_CSV, delimiter=',', skip_header=1)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"expected 6 columns (x,y,z,vx,vy,vz) in {BUNCH_CSV}, "
                         f"got shape {data.shape}")
    x = data[:, 0:3].copy()
    v = data[:, 3:6].copy()
    x[:, 2] = 0.0          # midplane projection: drop z / vz (3D roadmap)
    v[:, 2] = 0.0

    if N_PARTICLES is not None and N_PARTICLES < len(x):
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        idx = rng.choice(len(x), size=int(N_PARTICLES), replace=False)
        idx.sort()
        x, v = x[idx], v[idx]

    if PREPEND_CENTROID:
        x = np.vstack([x.mean(axis=0), x])
        v = np.vstack([v.mean(axis=0), v])

    return make_beam_from_state(species, x, v)


def build_worker_optimizer(rf_frequency):
    """Worker-side builder (module-level, picklable by reference)."""
    ex06 = _load_example06()
    _, _, geo, _ = ex06.build_system(rf_frequency=rf_frequency, quiet=True)
    return geo


def main():
    print("=" * 70)
    print("MULTIPARTICLE CAVITY GEOMETRY + RF OPTIMIZATION")
    print("(real inflector-exit bunch)")
    print("=" * 70)

    if not BUNCH_CSV.exists():
        print(f"\nERROR: bunch file not found: {BUNCH_CSV}")
        sys.exit(1)

    output_dir = HERE.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    checkpoint_file = output_dir / 'cavity_geometry_optimization_multi.csv'

    print("\n1. Building system (field load + isochronous-frequency scan)...")
    ex06 = _load_example06()
    if not ex06.FIELD_PATH.exists():
        print(f"\nERROR: Field file not found: {ex06.FIELD_PATH}")
        sys.exit(1)
    design, finder, geo_optimizer, rf_frequency = ex06.build_system(
        rf_frequency=ex06.RF_FREQUENCY_OVERRIDE,
        checkpoint_file=str(checkpoint_file))

    initial_beam = load_bunch(design.species)
    n_part = int(initial_beam.numpart)
    r_all = np.hypot(initial_beam.x_vec[:, 0], initial_beam.x_vec[:, 1])
    print(f"   Species: {design.species.name}, target "
          f"{ex06.TARGET_ENERGY_MEV} MeV")
    print(f"   RF base frequency: {rf_frequency / 1e6:.4f} MHz")
    print(f"   {len(design.rf_cavities)} gaps x {ex06.N_VARIABLE_SEGMENTS} "
          f"variable segment(s)")
    print(f"\n2. Loaded bunch: {n_part} particles"
          + (" (centroid prepended as reference)" if PREPEND_CENTROID else "")
          + (f", subsampled from CSV with seed {SUBSAMPLE_SEED}"
             if N_PARTICLES is not None else ""))
    print(f"   r = {r_all.mean() * 1000:.2f} +/- {r_all.std() * 1000:.2f} mm "
          f"(range {r_all.min() * 1000:.2f}-{r_all.max() * 1000:.2f} mm)")
    print(f"   mean energy: {initial_beam.mean_energy_mev * 1000:.1f} keV")
    print(f"   ls_weights overrides: {LS_WEIGHTS}")
    print(f"\n   NOTE: every evaluation tracks all {n_part} particles - "
          f"expect ~{n_part}x the single-particle eval cost of example 06.")

    # ========================================================================
    # Optimize geometry + RF parameters (staged, parallel)
    # ========================================================================
    print("\n" + "=" * 70)
    print("OPTIMIZING (RF scan -> multi-start DFO-LS -> verify), "
          f"{N_WORKERS} workers")
    print("=" * 70 + "\n")

    result = geo_optimizer.optimize_staged(
        initial_beam,
        # coll_azimuth / coll_aperture: central-region phase-slit
        # collimator (RadialSlitCollimator, reference-centered on the
        # prepended bunch centroid), optimized jointly with geometry +
        # RF. Its removals are priced ONCE by the 'collimated' LS
        # weight; the aperture starts wide open (x0 = upper bound).
        rf_optimize_params=['bunch_phase', 'rf_freq',
                            'coll_azimuth', 'coll_aperture'],
        rf_bounds={
            'bunch_phase': (-180, 180),
            'rf_freq': (rf_frequency * 0.95, rf_frequency * 1.05),
            'coll_azimuth': (0.0, 360.0),
            'coll_aperture': (2.0, 20.0),
        },
        ls_weights=dict(LS_WEIGHTS),
        search_steps_per_turn=SEARCH_STEPS_PER_TURN,
        search_max_turns=SEARCH_MAX_TURNS,
        final_steps_per_turn=FINAL_STEPS_PER_TURN,
        final_max_turns=FINAL_MAX_TURNS,
        maxfun=MAXFUN_PER_START,
        r0_mode='offset',
        skip_turns=SKIP_TURNS,
        workers=N_WORKERS,
        worker_builder=build_worker_optimizer,
        worker_builder_args=(rf_frequency,),
    )

    # ========================================================================
    # Results summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"\nSuccess: {result.success}")
    print(f"Final MEAN energy: {result.final_energy_mev:.3f} MeV "
          f"(target: {ex06.TARGET_ENERGY_MEV} MeV)")
    print(f"Number of turns: {result.n_turns}")
    print(f"Final cost: {result.cost:.2e}")

    stats = result.turn_statistics
    if stats:
        s0, s1 = stats[0], stats[-1]
        print(f"\nBunch, first -> last turn:")
        print(f"  surviving:      {s0.n_active} -> {s1.n_active} "
              f"of {n_part} ({100.0 * s1.n_active / max(n_part, 1):.1f}%)")
        print(f"  radial spread:  {s0.std_r * 1000:.2f} -> "
              f"{s1.std_r * 1000:.2f} mm (sigma)")
        print(f"  energy spread:  {s0.std_energy_mev * 1000:.1f} -> "
              f"{s1.std_energy_mev * 1000:.1f} keV (sigma)")

    print(f"\nOptimal RF parameters:")
    print(f"  Bunch phase: {result.bunch_phase_deg:.2f} degrees")
    print(f"  RF frequency: {result.rf_frequency_mhz:.6f} MHz")
    coll = result.metadata.get('collimator')
    if coll:
        print(f"\nOptimal collimator (phase slit, first turn):")
        print(f"  Azimuth: {coll['azimuth_deg']:.1f} deg, aperture: "
              f"{coll['aperture_mm']:.1f} mm "
              f"(center r = {coll['r_center_mm']} mm)")
        print(f"  Collimated: {coll['n_collimated']}/{n_part}")

    geom = result.metadata['optimal_geometry']
    print(f"\nOptimal cavity geometry (per gap):")
    for g, (angs, rads) in enumerate(zip(geom['segment_angles_per_gap'],
                                         geom['segment_radii_per_gap'])):
        segs = ", ".join(f"seg{i}: {a:+.2f}°/{r * 1000:.1f}mm"
                         for i, (a, r) in enumerate(zip(angs, rads)))
        print(f"  Gap {g}: {segs}")

    # ========================================================================
    # Visualization
    # ========================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 12))

    # Plot 1: geometry + centroid trajectory over the field map
    ax1 = plt.subplot(3, 4, 1)
    extent = 0.4
    x_plot = np.linspace(-extent, extent, 150)
    y_plot = np.linspace(-extent, extent, 150)
    X, Y = np.meshgrid(x_plot, y_plot, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(len(X.ravel()))])
    bfield = design.bfield(pts)
    bmag = np.sqrt((bfield ** 2).sum(axis=1)).reshape(150, 150)
    ax1.contourf(X, Y, bmag, levels=20, cmap='viridis', alpha=0.3)
    for cav in design.rf_cavities:
        for segment in cav.segments:
            ls = '-' if segment['type'] == 'variable' else '--'
            lw = 3 if segment['type'] == 'variable' else 2
            ax1.plot([segment['p1'][0], segment['p2'][0]],
                     [segment['p1'][1], segment['p2'][1]],
                     'r', linestyle=ls, linewidth=lw, alpha=0.7, zorder=5)
    traj = result.trajectory_reference
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1, alpha=0.8,
             label='Centroid orbit')
    ax1.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start',
             zorder=10)
    ax1.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=10, label='End',
             zorder=10)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Optimized Geometry + Centroid Trajectory')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: detailed cavity geometry
    ax2 = plt.subplot(3, 4, 2)
    ex06.plot_cavity_geometry(ax2, design.rf_cavities[:2],
                              "Cavity Geometry (Detail)")
    ax2.set_xlim(-0.2, 0.2)
    ax2.set_ylim(-0.1, 0.3)

    turns = [s.turn for s in stats]

    # Plot 3: mean energy +/- sigma vs turn
    ax3 = plt.subplot(3, 4, 3)
    ax3.errorbar(turns, [s.mean_energy_mev for s in stats],
                 yerr=[s.std_energy_mev for s in stats],
                 fmt='o-', linewidth=2, markersize=4, capsize=3,
                 label='mean +/- sigma')
    ax3.axhline(y=ex06.TARGET_ENERGY_MEV, color='r', linestyle='--',
                label=f'Target: {ex06.TARGET_ENERGY_MEV} MeV')
    ax3.set_xlabel('Turn Number')
    ax3.set_ylabel('Energy (MeV)')
    ax3.set_title('Bunch Energy vs Turn')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    # Plot 4: mean radius vs turn
    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(turns, [s.mean_r * 1000 for s in stats], 'o-', linewidth=2,
             markersize=4, color='green')
    ax4.set_xlabel('Turn Number')
    ax4.set_ylabel('Mean Radius (mm)')
    ax4.set_title('Mean Orbit Radius vs Turn')
    ax4.grid(True, alpha=0.3)

    # Plot 5: radial spread vs turn (the envelope residual's target)
    ax5 = plt.subplot(3, 4, 5)
    ax5.plot(turns, [s.std_r * 1000 for s in stats], 'o-', linewidth=2,
             markersize=4, color='purple')
    ax5.set_xlabel('Turn Number')
    ax5.set_ylabel('sigma_r (mm)')
    ax5.set_title('Radial Beam Spread vs Turn')
    ax5.grid(True, alpha=0.3)

    # Plot 6: survival vs turn (the survival residual's target)
    ax6 = plt.subplot(3, 4, 6)
    ax6.plot(turns, [100.0 * s.n_active / max(n_part, 1) for s in stats],
             'o-', linewidth=2, markersize=4, color='crimson')
    ax6.set_ylim(0, 105)
    ax6.set_xlabel('Turn Number')
    ax6.set_ylabel('Surviving (%)')
    ax6.set_title('Bunch Survival vs Turn')
    ax6.grid(True, alpha=0.3)

    # Plots 7/8: bunch phase space (r, vr), first vs last turn
    def poincare_scatter(ax, turn_no, color, title):
        r_vals, vr_vals = [], []
        for plist in result.poincare_points_all:
            for p in plist:
                if p.turn == turn_no:
                    r_vals.append(p.r * 1000)
                    vr_vals.append(p.vr)
                    break
        ax.plot(r_vals, vr_vals, 'o', markersize=3, alpha=0.5, color=color)
        ax.set_xlabel('Radius (mm)')
        ax.set_ylabel('Radial Velocity (m/s)')
        ax.set_title(f'{title} ({len(r_vals)} particles)')
        ax.grid(True, alpha=0.3)

    ax7 = plt.subplot(3, 4, 7)
    poincare_scatter(ax7, 0, 'blue', 'Bunch Phase Space, Turn 1')
    ax8 = plt.subplot(3, 4, 8)
    poincare_scatter(ax8, max(result.n_turns - 1, 0), 'red',
                     f'Bunch Phase Space, Turn {result.n_turns}')

    # Plot 9: energy spread vs turn
    ax9 = plt.subplot(3, 4, 9)
    ax9.plot(turns, [s.std_energy_mev * 1000 for s in stats], 'o-',
             linewidth=2, markersize=4, color='orange')
    ax9.set_xlabel('Turn Number')
    ax9.set_ylabel('sigma_E (keV)')
    ax9.set_title('Energy Spread vs Turn')
    ax9.grid(True, alpha=0.3)

    # Plot 10: optimization convergence (merged main checkpoint)
    ax10 = plt.subplot(3, 4, 10)
    if checkpoint_file.exists():
        iterations, costs = [], []
        with open(checkpoint_file, 'r') as f:
            for row in csv.DictReader(f):
                iterations.append(int(row['iteration']))
                costs.append(float(row['cost']))
        if costs:
            ax10.semilogy(iterations, costs, 'o-', linewidth=1, markersize=3,
                          alpha=0.7)
    ax10.set_xlabel('Iteration')
    ax10.set_ylabel('Cost Function')
    ax10.set_title('Optimization Convergence')
    ax10.grid(True, alpha=0.3)

    # Plot 11: centroid orbit centering (first-harmonic fit)
    ax11 = plt.subplot(3, 4, 11)
    if len(result.turn_metrics.get('r_center', [])) > 0:
        tn = np.arange(len(result.turn_metrics['r_center'])) + 1
        ax11.plot(tn, result.turn_metrics['r_center'] * 1000, 'o--',
                  linewidth=1.5, markersize=4, color='lightsteelblue',
                  label='centroid (incl. spiral ~dr/2pi)')
        ax11.plot(tn, result.turn_metrics['r_center_h1'] * 1000, 'o-',
                  linewidth=2, markersize=4, color='blue',
                  label='1st harmonic (true offset)')
        ax11.set_xlabel('Turn Number')
        ax11.set_ylabel('Orbit-Center Offset (mm)')
        ax11.set_title('Orbit Centering (reference)')
        ax11.legend(fontsize=7)
        ax11.grid(True, alpha=0.3)

    # Plot 12: RF crossing phase distribution (whole bunch)
    ax12 = plt.subplot(3, 4, 12)
    if len(result.rf_crossings) > 0:
        phases = [c.phase_deg for c in result.rf_crossings]
        ax12.hist(phases, bins=36, alpha=0.7, edgecolor='black')
        ax12.axvline(x=result.bunch_phase_deg, color='r', linestyle='--',
                     linewidth=2,
                     label=f'Optimized: {result.bunch_phase_deg:.1f} deg')
        ax12.set_xlabel('RF Phase (degrees)')
        ax12.set_ylabel('Count')
        ax12.set_title('RF Crossing Phases (all particles)')
        ax12.legend(fontsize=8)
        ax12.grid(True, alpha=0.3)

    fig.suptitle(f'Multiparticle Geometry Optimization: '
                 f'{design.species.name}, {n_part} particles, '
                 f'E_final={result.final_energy_mev:.2f} MeV, '
                 f'survival={100.0 * stats[-1].n_active / max(n_part, 1):.0f}%'
                 if stats else 'Multiparticle Geometry Optimization',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_file = output_dir / 'cavity_geometry_optimization_multi.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {plot_file}")

    # ========================================================================
    # Save results
    # ========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    result_file = output_dir / 'optimized_cavity_geometry_multi.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved optimized result to {result_file}")

    summary_file = output_dir / 'cavity_geometry_multi_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("MULTIPARTICLE CAVITY GEOMETRY + RF OPTIMIZATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Species: {design.species.name}\n")
        f.write(f"Bunch: {n_part} particles from {BUNCH_CSV.name}"
                + (" (centroid prepended)" if PREPEND_CENTROID else "") + "\n")
        f.write(f"Target energy: {ex06.TARGET_ENERGY_MEV} MeV\n")
        f.write(f"Final mean energy: {result.final_energy_mev:.3f} MeV\n")
        f.write(f"Number of turns: {result.n_turns}\n")
        if stats:
            f.write(f"Survival: {stats[-1].n_active}/{n_part} "
                    f"({100.0 * stats[-1].n_active / max(n_part, 1):.1f}%)\n")
            f.write(f"Final radial spread (sigma): "
                    f"{stats[-1].std_r * 1000:.2f} mm\n")
            f.write(f"Final energy spread (sigma): "
                    f"{stats[-1].std_energy_mev * 1000:.1f} keV\n")
        f.write(f"\nOptimal RF parameters:\n")
        f.write(f"  Bunch phase: {result.bunch_phase_deg:.2f} degrees\n")
        f.write(f"  RF frequency: {result.rf_frequency_mhz:.6f} MHz\n\n")
        f.write(f"Optimal cavity geometry "
                f"({ex06.N_VARIABLE_SEGMENTS} segments per gap):\n")
        for g, (angs, rads) in enumerate(zip(geom['segment_angles_per_gap'],
                                             geom['segment_radii_per_gap'])):
            segs = ", ".join(f"seg{i}: {a:+.2f} deg / {r * 1000:.1f} mm"
                             for i, (a, r) in enumerate(zip(angs, rads)))
            f.write(f"  Gap {g}: {segs}\n")
        f.write(f"\nOptimization:\n")
        f.write(f"  Method: {result.metadata['optimization_method']}\n")
        f.write(f"  Time: {result.metadata['optimization_time_s']:.1f} s\n")
        f.write(f"  Iterations: {result.metadata['total_iterations']}\n")
        f.write(f"  Final cost: {result.cost:.2e}\n")
    print(f"Saved summary to {summary_file}")

    print("\n" + "=" * 70)
    print("MULTIPARTICLE CAVITY GEOMETRY OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to {output_dir}/")
    plt.show()


if __name__ == "__main__":
    main()
