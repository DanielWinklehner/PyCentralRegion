"""Multi-particle cavity geometry optimization: example 06 for a real spiral-inflector exit bunch.

Shows: example 06's staged flow (RF scan -> multi-start DFO-LS -> verify) on example 06's system, but the
objective tracks a bunch read from a CSV (x y z [m], vx vy vz [m/s]) instead of one particle. With more
than one particle the finder's residual vector grows two blocks (AcceleratedOrbitFinder.objective_residuals):
envelope (per-turn radial spread, std_r / 5 mm) and survival (per-turn lost fraction, weighted like energy:
losing the beam hurts as much as not accelerating it). A central-region phase-slit collimator
(RadialSlitCollimator: azimuth and aperture) is optimized jointly, its removals priced once by the
'collimated' weight. The bunch is projected onto the midplane (z, vz dropped); the finder prepends the
bunch centroid as the reference particle and keeps it out of every beam statistic.
Inputs: <resources>/uCyclo_v3_midplane_field.comsol, <resources>/MuonBunchOutOfInflector.csv (250 muons
at the inflector exit).
Runtime: each evaluation tracks the whole bunch on the general (not the compiled single-particle) path:
--particles 8 --workers 2 --maxfun 10 --rf-frequency 42.2234e6 takes about 4 minutes (2.5 of them the 60
tracks of stage A); the full bunch with the default budget is an overnight run.
Outputs: <out>/optimized_cavity_geometry_multi.pkl (read by example 10), cavity_geometry_optimization_multi.png,
cavity_geometry_multi_summary.txt, cavity_geometry_optimization_multi.csv[.worker-<pid>.csv].

Usage:
    python examples/09_optimize_cavity_geometry_multiparticle.py [--resources DIR] [--out DIR]
        [--workers N] [--maxfun N] [--particles N] [--rf-frequency HZ]
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import _common
from PyCentralRegion.accelerated_orbit_finder import make_beam_from_state

ex06 = _common.load_example('06_optimize_cavity_geometry.py')

BUNCH_FILE = 'MuonBunchOutOfInflector.csv'
SUBSAMPLE_SEED = 42

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

# Staged-run settings (as in example 06)
SEARCH_STEPS_PER_TURN = 300
SEARCH_MAX_TURNS = 10
FINAL_STEPS_PER_TURN = 500
FINAL_MAX_TURNS = 15
SKIP_TURNS = 4

# Each worker holds its own field map and tracks the full bunch per eval.
N_WORKERS = max(1, min(12, (os.cpu_count() or 2) - 2))


def load_bunch(path, species, n_particles=None, seed=SUBSAMPLE_SEED):
    """The bunch in ``path`` (CSV with a header row; x y z [m], vx vy vz [m/s]) as a midplane
    ParticleDistribution: z and vz zeroed, optionally a deterministic subsample of ``n_particles``.
    Returns the physical bunch only - the orbit finder prepends the centroid reference particle."""
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"expected 6 columns (x,y,z,vx,vy,vz) in {path}, got shape {data.shape}")
    x, v = data[:, 0:3].copy(), data[:, 3:6].copy()
    x[:, 2] = 0.0          # midplane projection
    v[:, 2] = 0.0
    if n_particles is not None and n_particles < len(x):
        idx = np.sort(np.random.default_rng(seed).choice(len(x), size=int(n_particles), replace=False))
        x, v = x[idx], v[idx]
    return make_beam_from_state(species, x, v)


def build_worker_optimizer(field_path, rf_frequency):
    """Worker-side builder (module-level, picklable by reference)."""
    _, _, geo, _ = ex06.build_system(field_path, rf_frequency=rf_frequency, quiet=True)
    return geo


def main():
    ap = _common.arguments(__doc__)
    ap.add_argument('--workers', type=int, default=N_WORKERS,
                    help=f'parallel worker processes (default {N_WORKERS}; 1 = serial)')
    ap.add_argument('--maxfun', type=int, default=600, help='DFO-LS evaluations per start (default 600)')
    ap.add_argument('--particles', type=int, default=None,
                    help='deterministic subsample of the bunch (default: all particles)')
    ex06.add_rf_option(ap)
    args = _common.parse(ap)
    field_path = _common.require(args.resources / ex06.FIELD_FILE)
    bunch_path = _common.require(args.resources / BUNCH_FILE)
    output_dir = args.out
    print("=" * 70)
    print("MULTIPARTICLE CAVITY GEOMETRY + RF OPTIMIZATION (inflector-exit bunch)")
    print("=" * 70)
    checkpoint_file = output_dir / 'cavity_geometry_optimization_multi.csv'

    print("\n1. Building system (field load + isochronous-frequency scan)...")
    design, finder, geo_optimizer, rf_frequency = ex06.build_system(
        field_path, rf_frequency=args.rf_frequency, checkpoint_file=str(checkpoint_file))

    initial_beam = load_bunch(bunch_path, design.species, args.particles)
    n_part = int(initial_beam.numpart)
    r_all = np.hypot(initial_beam.x_vec[:, 0], initial_beam.x_vec[:, 1])
    print(f"   Species: {design.species.name}, target {ex06.TARGET_ENERGY_MEV} MeV")
    print(f"   RF base frequency: {rf_frequency / 1e6:.4f} MHz")
    print(f"   {len(design.rf_cavities)} gaps x {ex06.N_VARIABLE_SEGMENTS} variable segment(s)")
    print(f"\n2. Loaded bunch: {n_part} particles"
          + (f", subsampled with seed {SUBSAMPLE_SEED}" if args.particles else ""))
    print(f"   r = {r_all.mean() * 1000:.2f} +/- {r_all.std() * 1000:.2f} mm "
          f"(range {r_all.min() * 1000:.2f}-{r_all.max() * 1000:.2f} mm)")
    print(f"   mean energy: {initial_beam.mean_energy_mev * 1000:.1f} keV")
    print(f"   ls_weights overrides: {LS_WEIGHTS}")

    print("\n" + "=" * 70)
    print(f"OPTIMIZING (RF scan -> multi-start DFO-LS -> verify), {args.workers} workers, "
          f"maxfun {args.maxfun} per start")
    print("=" * 70 + "\n")
    result = geo_optimizer.optimize_staged(
        initial_beam,
        # coll_azimuth / coll_aperture: central-region phase-slit collimator
        # (reference-centered on the prepended bunch centroid), optimized
        # jointly with geometry + RF; the aperture starts wide open (x0 = upper bound).
        rf_optimize_params=['bunch_phase', 'rf_freq', 'coll_azimuth', 'coll_aperture'],
        rf_bounds={'bunch_phase': (-180, 180),
                   'rf_freq': (rf_frequency * 0.95, rf_frequency * 1.05),
                   'coll_azimuth': (0.0, 360.0),
                   'coll_aperture': (2.0, 20.0)},
        ls_weights=dict(LS_WEIGHTS),
        # Half the DFO-LS starts begin with the slit CLOSED at a seeded-test
        # optimum (the wide-open start faces a discrete-price-vs-diffuse-gain
        # barrier DFO-LS does not cross).
        collimator_seed=(346.0, 6.5),
        search_steps_per_turn=SEARCH_STEPS_PER_TURN,
        search_max_turns=SEARCH_MAX_TURNS,
        final_steps_per_turn=FINAL_STEPS_PER_TURN,
        final_max_turns=FINAL_MAX_TURNS,
        maxfun=args.maxfun,
        r0_mode='offset',
        skip_turns=SKIP_TURNS,
        workers=args.workers,
        worker_builder=build_worker_optimizer,
        worker_builder_args=(field_path, rf_frequency),
    )

    # ========================================================================
    # Results summary
    # ========================================================================
    stats = result.turn_statistics
    survival = 100.0 * stats[-1].n_active / max(n_part, 1) if stats else 0.0
    summary = [f"Species: {design.species.name}; bunch: {n_part} particles from {BUNCH_FILE}",
               f"Final mean energy: {result.final_energy_mev:.3f} MeV in {result.n_turns} turns "
               f"(target {ex06.TARGET_ENERGY_MEV} MeV, success {result.success}, cost {result.cost:.2e})"]
    if stats:
        s0, s1 = stats[0], stats[-1]
        summary += [f"Surviving: {s0.n_active} -> {s1.n_active} of {n_part} ({survival:.1f}%)",
                    f"Radial spread: {s0.std_r * 1000:.2f} -> {s1.std_r * 1000:.2f} mm (sigma)",
                    f"Energy spread: {s0.std_energy_mev * 1000:.1f} -> {s1.std_energy_mev * 1000:.1f} keV (sigma)"]
    summary.append(f"Bunch phase: {result.bunch_phase_deg:.2f} deg, RF frequency: "
                   f"{result.rf_frequency_mhz:.6f} MHz")
    coll = result.metadata.get('collimator')
    if coll:
        summary.append(f"Collimator (phase slit, first turn): azimuth {coll['azimuth_deg']:.1f} deg, aperture "
                       f"{coll['aperture_mm']:.1f} mm (center r = {coll['r_center_mm']} mm), "
                       f"collimated {coll['n_collimated']}/{n_part}")
    summary.append(f"Optimal cavity geometry ({ex06.N_VARIABLE_SEGMENTS} segment(s) per gap):")
    summary += ["  " + line for line in ex06.geometry_lines(result.metadata['optimal_geometry'])]
    print("\n" + "=" * 70 + "\nOPTIMIZATION RESULTS\n" + "=" * 70)
    print("\n".join(summary))

    # ========================================================================
    # Visualization
    # ========================================================================
    fig = plt.figure(figsize=(20, 12))
    turns = [s.turn for s in stats]

    # Plot 1: geometry + centroid trajectory over the field map
    ax1 = plt.subplot(3, 4, 1)
    xy = np.linspace(-0.4, 0.4, 150)
    X, Y = np.meshgrid(xy, xy, indexing='ij')
    bmag = np.linalg.norm(design.bfield(np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])),
                          axis=1).reshape(X.shape)
    ax1.contourf(X, Y, bmag, levels=20, cmap='viridis', alpha=0.3)
    for cav in design.rf_cavities:
        for segment in cav.segments:
            variable = segment['type'] == 'variable'
            ax1.plot([segment['p1'][0], segment['p2'][0]], [segment['p1'][1], segment['p2'][1]], 'r',
                     linestyle='-' if variable else '--', linewidth=3 if variable else 2, alpha=0.7, zorder=5)
    traj = result.trajectory_reference
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1, alpha=0.8, label='Centroid orbit')
    ax1.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start', zorder=10)
    ax1.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=10, label='End', zorder=10)
    ax1.set(xlabel='x (m)', ylabel='y (m)', title='Optimized Geometry + Centroid Trajectory', aspect='equal')
    ax1.legend(fontsize=8)

    # Plot 2: detailed cavity geometry
    ax2 = plt.subplot(3, 4, 2)
    ex06.plot_cavity_geometry(ax2, design.rf_cavities[:2], "Cavity Geometry (Detail)")
    ax2.set_xlim(-0.2, 0.2)
    ax2.set_ylim(-0.1, 0.3)

    # Plots 3-6, 9: per-turn statistics of the live particles
    ax3 = plt.subplot(3, 4, 3)
    ax3.errorbar(turns, [s.mean_energy_mev for s in stats], yerr=[s.std_energy_mev for s in stats],
                 fmt='o-', linewidth=2, markersize=4, capsize=3, label='mean +/- sigma')
    ax3.axhline(y=ex06.TARGET_ENERGY_MEV, color='r', linestyle='--',
                label=f'Target: {ex06.TARGET_ENERGY_MEV} MeV')
    ax3.set(xlabel='Turn Number', ylabel='Energy (MeV)', title='Bunch Energy vs Turn')
    ax3.legend(fontsize=8)
    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(turns, [s.mean_r * 1000 for s in stats], 'o-', linewidth=2, markersize=4, color='green')
    ax4.set(xlabel='Turn Number', ylabel='Mean Radius (mm)', title='Mean Orbit Radius vs Turn')
    ax5 = plt.subplot(3, 4, 5)
    ax5.plot(turns, [s.std_r * 1000 for s in stats], 'o-', linewidth=2, markersize=4, color='purple')
    ax5.set(xlabel='Turn Number', ylabel='sigma_r (mm)', title='Radial Beam Spread vs Turn')
    ax6 = plt.subplot(3, 4, 6)
    ax6.plot(turns, [100.0 * s.n_active / max(n_part, 1) for s in stats], 'o-', linewidth=2,
             markersize=4, color='crimson')
    ax6.set(xlabel='Turn Number', ylabel='Surviving (%)', title='Bunch Survival vs Turn', ylim=(0, 105))
    ax9 = plt.subplot(3, 4, 9)
    ax9.plot(turns, [s.std_energy_mev * 1000 for s in stats], 'o-', linewidth=2, markersize=4,
             color='orange')
    ax9.set(xlabel='Turn Number', ylabel='sigma_E (keV)', title='Energy Spread vs Turn')

    # Plots 7, 8, 12: the bunch after the first step and at the end (verification run)
    full_beam = result.metadata.get('full_beam')
    ax7, ax8, ax12 = plt.subplot(3, 4, 7), plt.subplot(3, 4, 8), plt.subplot(3, 4, 12)
    if full_beam is not None and len(full_beam) > 1:
        r0, vr0, _ = _common.bunch_snapshot(full_beam, 0, design.species)
        r1, vr1, e1 = _common.bunch_snapshot(full_beam, -1, design.species)
        ax7.plot(r0, vr0, 'o', markersize=3, alpha=0.5, color='blue')
        ax7.set_title(f'Bunch after the First Step ({len(r0)} particles)')
        ax8.plot(r1, vr1, 'o', markersize=3, alpha=0.5, color='red')
        ax8.set_title(f'Bunch at the End ({len(r1)} tracked)')
        ax12.hist(e1, bins=30, alpha=0.7, edgecolor='black')
    for ax in (ax7, ax8):
        ax.set(xlabel='Radius (mm)', ylabel='Radial Velocity (m/s)')
    ax12.set(xlabel='Energy (MeV)', ylabel='Count', title='Energy Distribution at the End')

    # Plot 10: optimization history, all workers
    ax10 = plt.subplot(3, 4, 10)
    history = ex06.read_checkpoints(checkpoint_file)
    if len(history):
        ax10.semilogy(np.arange(len(history)), history['cost'], '.', markersize=3, alpha=0.7)
    ax10.set(xlabel='Evaluation (all workers)', ylabel='Cost Function', title='Optimization History')

    # Plot 11: centroid orbit centering (first-harmonic fit)
    ax11 = plt.subplot(3, 4, 11)
    tn = np.arange(1, len(result.turn_metrics['r_center']) + 1)
    ax11.plot(tn, result.turn_metrics['r_center'] * 1000, 'o--', linewidth=1.5, markersize=4,
              color='lightsteelblue', label='centroid (incl. spiral ~dr/2pi)')
    ax11.plot(tn, result.turn_metrics['r_center_h1'] * 1000, 'o-', linewidth=2, markersize=4,
              color='blue', label='1st harmonic (true offset)')
    ax11.set(xlabel='Turn Number', ylabel='Orbit-Center Offset (mm)', title='Orbit Centering (reference)')
    ax11.legend(fontsize=7)

    for ax in fig.axes:
        if ax.get_label() != '<colorbar>':
            ax.grid(True, alpha=0.3)
    fig.suptitle(f'Multiparticle Geometry Optimization: {design.species.name}, {n_part} particles, '
                 f'E_final={result.final_energy_mev:.2f} MeV, survival={survival:.0f}%',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    plot_file = output_dir / 'cavity_geometry_optimization_multi.png'
    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved {plot_file}")

    result_file = output_dir / 'optimized_cavity_geometry_multi.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
    (output_dir / 'cavity_geometry_multi_summary.txt').write_text("\n".join(summary) + "\n")
    print(f"Saved {result_file} and cavity_geometry_multi_summary.txt")
    plt.show()


if __name__ == "__main__":
    main()
