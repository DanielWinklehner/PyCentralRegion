"""Cavity geometry + RF optimization: per-gap segment geometry of a dee system for a hand-off particle.

Shows: create_dee_system (four 45-deg dees, h = 4, one variable segment per gap that tilts, moves and
rotates), the isochronous RF frequency from SEOs (SEOFinder.mean_frequency), and
CavityGeometryOptimizer.optimize_staged: stage A, a bunch phase x RF frequency scan with straight gaps;
stage B, DFO-LS on all per-gap segment parameters + opening angle + RF (one start per worker, seeded from
distinct stage-A basins); stage C, the winner tracked at full resolution. The objective is the
least-squares residual of energy, orbit centring and turn-separation smoothness, plus a tie-breaker that
pulls each gap's segment node to a target radius (pinch_target_r_m). Examples 07-10 and 12 build on
this system (build_system); 07 and 12 re-track its result pickle.
Inputs: <resources>/uCyclo_v3_midplane_field.comsol (COMSOL midplane export of a muon cyclotron).
Runtime: about 2 minutes with the defaults (12 workers on 28 cores). The isochronous-frequency scan takes
about 45 s (--rf-frequency 42.2234e6 skips it); a single-particle evaluation runs on the compiled path in
tens of ms, so one start of the default budget (600 evaluations, 27 parameters) takes 30-45 s, and the starts
run in parallel (one per worker): --workers 4 --rf-frequency 42.2234e6 takes about 35 s. A --maxfun below the
28 points DFO-LS samples for its first model leaves every start where stage A put it (straight gaps).
Outputs: <out>/optimized_cavity_geometry.pkl (read by 07 and 12), cavity_geometry_optimization.png,
cavity_geometry_summary.txt, cavity_geometry_optimization.csv[.worker-<pid>.csv] (every evaluation).

Usage:
    python examples/06_optimize_cavity_geometry.py [--resources DIR] [--out DIR] [--workers N]
        [--maxfun N] [--rf-frequency HZ]
"""
import contextlib
import glob
import io
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyPATools.field import Field

import _common
from PyCentralRegion.central_region import CentralRegion
from PyCentralRegion.seo_finder import SEOFinder
from PyCentralRegion.rf_cavity import create_dee_system
from PyCentralRegion.accelerated_orbit_finder import (AcceleratedOrbitFinder,
                                                      make_beam_from_cylindrical)
from PyCentralRegion.cavity_optimizer import CavityGeometryOptimizer

# ============================================================================
# System configuration (module level so multiprocessing workers can rebuild
# the identical system via build_worker_optimizer)
# ============================================================================
FIELD_FILE = 'uCyclo_v3_midplane_field.comsol'
SPECIES = 'muon'
TARGET_ENERGY_MEV = 5.0
MAX_RADIUS_M = 0.4

R_MIN = 0.005                      # 5 mm inner cavity radius
R_MAX = 0.40                       # 400 mm outer cavity radius
CAVITY_VOLTAGE = 50000.0           # V per gap
HARMONIC = 4

# Dee system: each cavity is (center angle, opening angle); gaps at
# center +- opening/2. Two-dee system: DEE_CENTER_ANGLES = [90.0, 270.0].
DEE_CENTER_ANGLES = [45.0, 135.0, 225.0, 315.0]
DEE_OPENING_ANGLE = 180.0 / HARMONIC   # classical synchronous dee angle [deg]
# Opt-in: add ONE shared opening-angle delta to the optimization parameters.
OPTIMIZE_OPENING_ANGLE = True
OPENING_DELTA_MAX = 10.0               # bounds [deg] when optimizable

# 1 variable segment per gap; increase to 2 when going multiparticle/3D.
N_VARIABLE_SEGMENTS = 1
SEGMENT_ANGLES_INITIAL = [0.0]
SEGMENT_RADII_INITIAL = [0.1]

# Rotatable segments: each variable segment additionally rotates about its own
# midpoint (need not point at the origin), decoupling the crossing azimuth (RF
# phase) from the kick direction. Authority test: tilts alone floor at ~7 mm
# orbit-center offset; tilts+rotations reach ~1 mm at full acceleration.
# Params: 8 gaps x (angle + radius + rotation) + the shared opening-angle delta + 2 RF = 27.
ROTATABLE_SEGMENTS = True
ROTATION_MAX = 20.0                    # per-segment rotation bounds [deg]

ISO_RADII_MM = [100, 150, 200, 250, 300]   # radii for the isochronous-freq average

# Each worker holds its own copy of the field map, so memory (not CPU) is the
# practical cap; 12 is a good default on a 28-core / large-RAM machine.
N_WORKERS = max(1, min(12, (os.cpu_count() or 2) - 2))


def build_system(field_path, rf_frequency=None, quiet=False, checkpoint_file=None):
    """Build (design, finder, geometry optimizer, rf_frequency).

    Module-level and deterministic so each multiprocessing worker can rebuild
    the identical system. With quiet=True all construction chatter is
    suppressed (used in workers). The RF base frequency, unless given, is the
    ISOCHRONOUS-region frequency (mean over larger-radius SEOs), not the
    injection orbit's: the centre field has no flutter, so the innermost orbit
    is far off-isochronous.
    """
    sink = io.StringIO() if quiet else None
    with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
        design = CentralRegion(name="GeometryOptimizedCyclotron", dimensionality='2D')
        design.set_species(SPECIES)
        design.set_magnetic_field(field_path, interpolator_backend='fast')
        design.set_electric_field(Field.zero())

        if rf_frequency is None:
            seo_finder = SEOFinder(design, n_turns=20, steps_per_turn=500,
                                   closure_tol_mm=0.5, algorithm='rk4_rel',
                                   verbose=False)
            rf_frequency = seo_finder.mean_frequency(ISO_RADII_MM)

        dee_system = create_dee_system(
            r_min=R_MIN, r_max=R_MAX, center_angles=list(DEE_CENTER_ANGLES),
            opening_angle=DEE_OPENING_ANGLE, voltage=CAVITY_VOLTAGE,
            frequency=rf_frequency, phases=[0.0] * len(DEE_CENTER_ANGLES),
            harmonic=HARMONIC, gap_width=0.01,
            n_variable_segments=N_VARIABLE_SEGMENTS,
            segment_angles=list(SEGMENT_ANGLES_INITIAL),
            segment_radii=list(SEGMENT_RADII_INITIAL))
        design.clear_rf_cavities()
        for cav in dee_system.gaps:
            design.add_rf_cavity(cav)

        finder = AcceleratedOrbitFinder(
            design, target_energy_mev=TARGET_ENERGY_MEV, max_radius_m=MAX_RADIUS_M,
            algorithm='rk4_rel', steps_per_turn=500, verbose=False)

        geo_optimizer = CavityGeometryOptimizer(
            orbit_finder=finder, n_segments=N_VARIABLE_SEGMENTS,
            max_angle_variable=10.0, max_r_variable=0.20, r_min_cavity=R_MIN,
            verbose=not quiet, checkpoint_file=checkpoint_file,
            dee_system=dee_system,
            optimize_opening_angle=OPTIMIZE_OPENING_ANGLE,
            opening_delta_max=OPENING_DELTA_MAX,
            rotatable_segments=ROTATABLE_SEGMENTS,
            rotation_max=ROTATION_MAX,
            # tie-breaker: each gap's segment node pulled towards these radii [m]
            # (the 2 mm metal width must match the BEM build of example 07)
            pinch_target_r_m=[0.016, 0.013, 0.030, 0.036, 0.047, 0.052, 0.063, 0.012],
            pinch_metal_width_m=0.002,
            pinch_weight=50.0
        )

    return design, finder, geo_optimizer, rf_frequency


def build_worker_optimizer(field_path, rf_frequency):
    """Worker-side builder (module-level, picklable by reference)."""
    _, _, geo, _ = build_system(field_path, rf_frequency=rf_frequency, quiet=True)
    return geo


def add_rf_option(ap):
    """``--rf-frequency``: skip the isochronous-frequency scan of build_system (about 45 s)."""
    ap.add_argument('--rf-frequency', type=float, default=None,
                    help='RF base frequency [Hz] (default: mean SEO frequency over '
                         f'{ISO_RADII_MM} mm; 42.2234e6 for the default field map)')


def make_initial_beam(species):
    """Initial single particle at a spiral-inflector hand-off point.

    Cylindrical lab coordinates: position (r, theta, z), momentum
    (p_r, p_theta, p_z) in beta*gamma. Shared with the downstream examples
    (07, 08, 12), which re-track the identical injection point.
    """
    return make_beam_from_cylindrical(
        species,
        r=27.7e-3, theta_deg=-7.6, z=0.07e-3,
        p_r=-0.0031, p_theta=0.023, p_z=0.00026,
    )


def read_checkpoints(checkpoint_file):
    """Every evaluation logged by the optimizer: the checkpoint CSV and, in parallel runs, the
    per-worker files <checkpoint>.worker-<pid>.csv, concatenated (column 'source' = file)."""
    frames = [pd.read_csv(fp).assign(source=os.path.basename(fp))
              for fp in [str(checkpoint_file)] + sorted(glob.glob(f"{checkpoint_file}.worker-*.csv"))
              if os.path.exists(fp) and os.path.getsize(fp) > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def geometry_lines(geom):
    """One line per gap: segment angle [deg] / node radius [mm] of each variable segment."""
    return [f"Gap {g}: " + ", ".join(f"seg{i}: {a:+.2f} deg / {r * 1000:.1f} mm"
                                     for i, (a, r) in enumerate(zip(angs, rads)))
            for g, (angs, rads) in enumerate(zip(geom['segment_angles_per_gap'],
                                                 geom['segment_radii_per_gap']))]


def plot_cavity_geometry(ax, cavities, title="Cavity Geometry"):
    """Plot cavity segment geometry."""

    colors = plt.cm.tab10(np.linspace(0, 1, len(cavities)))

    for cav_id, (cavity, color) in enumerate(zip(cavities, colors)):
        # Nominal RADIAL reference (base angle, no tilt/rotation) so the
        # segment-angle and rotation offsets are visible against it.
        th = np.deg2rad(cavity.base_angle)
        r_ref = max(s['r_max'] for s in cavity.segments if s['type'] == 'variable') \
            if any(s['type'] == 'variable' for s in cavity.segments) else cavity.r_max
        ax.plot([cavity.r_min * np.cos(th), r_ref * np.cos(th)],
                [cavity.r_min * np.sin(th), r_ref * np.sin(th)],
                color='gray', linestyle=':', linewidth=1, alpha=0.6, zorder=1)

        # Plot each segment
        for seg_id, segment in enumerate(cavity.segments):
            p1 = segment['p1']
            p2 = segment['p2']

            linestyle = '-' if segment['type'] == 'variable' else '--'
            linewidth = 2 if segment['type'] == 'variable' else 1
            alpha = 0.8 if segment['type'] == 'variable' else 0.5

            label = f"Cavity {cav_id}" if seg_id == 0 else None

            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=color, linestyle=linestyle, linewidth=linewidth,
                    alpha=alpha, label=label)

            # Mark segment boundaries
            if segment['type'] == 'variable':
                ax.plot(p2[0], p2[1], 'o', color=color, markersize=4)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)


def main():
    ap = _common.arguments(__doc__)
    ap.add_argument('--workers', type=int, default=N_WORKERS,
                    help=f'parallel worker processes (default {N_WORKERS}; 1 = serial)')
    ap.add_argument('--maxfun', type=int, default=600,
                    help='DFO-LS evaluations per start (default 600)')
    add_rf_option(ap)
    args = _common.parse(ap)
    field_path = _common.require(args.resources / FIELD_FILE)
    output_dir = args.out
    print("=" * 70)
    print("CAVITY GEOMETRY + RF PARAMETER OPTIMIZATION")
    print("=" * 70)

    checkpoint_file = output_dir / 'cavity_geometry_optimization.csv'

    print("\n1. Building system (field load + isochronous-frequency scan)...")
    design, finder, geo_optimizer, rf_frequency = build_system(
        field_path, rf_frequency=args.rf_frequency, checkpoint_file=str(checkpoint_file))

    print(f"   Species: {design.species.name}, target {TARGET_ENERGY_MEV} MeV")
    print(f"   RF base frequency: {rf_frequency / 1e6:.4f} MHz "
          f"({'given' if args.rf_frequency else f'isochronous avg over {ISO_RADII_MM} mm'})")
    print(f"   {len(design.rf_cavities)} gaps x {N_VARIABLE_SEGMENTS} variable segment(s)")

    print(f"\nStarting staged optimization (RF scan -> DFO-LS -> verify), "
          f"{args.workers} workers, maxfun {args.maxfun} per start...\n")
    initial_beam = make_initial_beam(design.species)

    # Stage A: coarse (phase x freq) scan with straight geometry, farmed over
    # the worker pool; Stage B: multi-start DFO-LS (one start per worker, seeded
    # from distinct stage-A basins) on all parameters at search resolution
    # (300 spt = the measured reliability floor); Stage C: verify the winner at
    # full resolution. Per-eval checkpoints land in per-worker CSVs.
    result = geo_optimizer.optimize_staged(
        initial_beam,
        rf_optimize_params=['bunch_phase', 'rf_freq'],
        rf_bounds={'bunch_phase': (-180, 180),
                   'rf_freq': (rf_frequency * 0.95, rf_frequency * 1.05)},
        search_steps_per_turn=300,
        search_max_turns=10,
        final_steps_per_turn=500,
        final_max_turns=15,
        maxfun=args.maxfun,
        r0_mode='offset',
        # First turns are inherently lopsided (modified gaps + off-orbit
        # injection); exempt them from the centering/smoothness residuals.
        skip_turns=4,
        workers=args.workers,
        worker_builder=build_worker_optimizer,
        worker_builder_args=(field_path, rf_frequency),
    )

    # ========================================================================
    # Results summary
    # ========================================================================
    geom = result.metadata['optimal_geometry']
    summary = [f"Species: {SPECIES}, target {TARGET_ENERGY_MEV} MeV",
               f"Final energy: {result.final_energy_mev:.3f} MeV in {result.n_turns} turns "
               f"(success {result.success}, cost {result.cost:.2e})",
               f"Bunch phase: {result.bunch_phase_deg:.2f} deg, RF frequency: "
               f"{result.rf_frequency_mhz:.6f} MHz",
               f"Opening angle: {geom.get('opening_angle_deg')}",
               f"Optimal cavity geometry ({N_VARIABLE_SEGMENTS} variable segment(s) per gap):",
               *("  " + line for line in geometry_lines(geom)),
               f"Optimization: {result.metadata.get('optimization_method')}, "
               f"{result.metadata.get('optimization_time_s', float('nan')):.1f} s, "
               f"{result.metadata.get('total_iterations')} evaluations"]
    print("\n" + "=" * 70 + "\nOPTIMIZATION RESULTS\n" + "=" * 70)
    print("\n".join(summary))

    # ========================================================================
    # Visualization
    # ========================================================================
    fig = plt.figure(figsize=(20, 12))
    ref = result.poincare_points_all[0]          # the single particle at the Poincare section
    turns = [p.turn for p in ref]
    energies = [p.energy_mev for p in ref]
    radii = [p.r * 1000 for p in ref]

    # Plot 1: Optimized cavity geometry with trajectory over |B|
    ax1 = plt.subplot(3, 4, 1)
    xy = np.linspace(-0.4, 0.4, 150)
    X, Y = np.meshgrid(xy, xy, indexing='ij')
    bmag = np.linalg.norm(design.bfield(np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])),
                          axis=1).reshape(X.shape)
    ax1.contourf(X, Y, bmag, levels=20, cmap='viridis', alpha=0.3)
    for cav in design.rf_cavities:
        for segment in cav.segments:
            variable = segment['type'] == 'variable'
            ax1.plot([segment['p1'][0], segment['p2'][0]], [segment['p1'][1], segment['p2'][1]],
                     color='r', linestyle='-' if variable else '--', linewidth=3 if variable else 2,
                     alpha=0.7, zorder=5)
    traj = result.trajectory_reference
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1, alpha=0.8, label='Orbit')
    ax1.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start', zorder=10)
    ax1.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=10, label='End', zorder=10)
    ax1.set(xlabel='x (m)', ylabel='y (m)', title='Optimized Cavity Geometry + Trajectory',
            aspect='equal')
    ax1.legend(fontsize=8)

    # Plot 2: Detailed cavity geometry (zoomed)
    ax2 = plt.subplot(3, 4, 2)
    plot_cavity_geometry(ax2, design.rf_cavities[:2], "Cavity Geometry (Detail)")
    ax2.set_xlim(-0.2, 0.2)
    ax2.set_ylim(-0.1, 0.3)

    # Plots 3-8: the particle turn by turn
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(turns, energies, 'o-', linewidth=2, markersize=4)
    ax3.axhline(y=TARGET_ENERGY_MEV, color='r', linestyle='--', label=f'Target: {TARGET_ENERGY_MEV} MeV')
    ax3.set(xlabel='Turn Number', ylabel='Energy (MeV)', title='Energy Gain vs Turn')
    ax3.legend()
    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(turns, radii, 'o-', linewidth=2, markersize=4, color='green')
    ax4.set(xlabel='Turn Number', ylabel='Radius (mm)', title='Radius vs Turn')
    ax5 = plt.subplot(3, 4, 5)
    ax5.plot(result.turn_metrics['dr'] * 1000, 'o-', linewidth=2, markersize=4, color='purple')
    ax5.set(xlabel='Turn Number', ylabel='dr/dturn (mm)', title='Turn Separation')
    ax6 = plt.subplot(3, 4, 6)
    scatter = ax6.scatter(radii, [p.vr for p in ref], c=turns, cmap='viridis', s=20)
    plt.colorbar(scatter, ax=ax6, label='Turn')
    ax6.set(xlabel='Radius (mm)', ylabel='Radial Velocity (m/s)', title='Poincare Section (r vs vr)')
    ax7 = plt.subplot(3, 4, 7)
    ax7.plot(turns, [p.phase_deg for p in ref], 'o', markersize=4, alpha=0.6)
    ax7.set(xlabel='Turn Number', ylabel='RF Phase (degrees)', title='Synchrotron Oscillation')
    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(energies, radii, 'o-', linewidth=2, markersize=4, color='brown')
    ax8.set(xlabel='Energy (MeV)', ylabel='Radius (mm)', title='Radius vs Energy')

    # Plot 9: Segment geometry per gap, as text
    ax9 = plt.subplot(3, 4, 9)
    ax9.text(0.02, 0.95, "Optimized segment geometry (angle / node radius):",
             transform=ax9.transAxes, fontweight='bold', fontsize=9)
    for k, line in enumerate(geometry_lines(geom)):
        ax9.text(0.02, 0.86 - 0.09 * k, line, transform=ax9.transAxes, fontsize=7)
    ax9.axis('off')
    ax9.set_title('Geometry Parameters')

    # Plots 10 and 12: every evaluation, all workers in sequence
    history = read_checkpoints(checkpoint_file)
    ax10 = plt.subplot(3, 4, 10)
    ax12 = plt.subplot(3, 4, 12)
    if len(history):
        ax10.semilogy(np.arange(len(history)), history['cost'], '.', markersize=3, alpha=0.7)
        for c in [c for c in history.columns if c.startswith('gap') and c.endswith('_angle_deg')]:
            ax12.plot(np.arange(len(history)), history[c], linewidth=1, alpha=0.6,
                      label=c.replace('_angle_deg', ''))
        if ax12.lines and len(ax12.lines) <= 8:
            ax12.legend(fontsize=6)
    ax10.set(xlabel='Evaluation (all workers)', ylabel='Cost Function', title='Optimization History')
    ax12.set(xlabel='Evaluation (all workers)', ylabel='Angle (degrees)',
             title='Segment Angles Tried (all gaps)')

    # Plot 11: Orbit centering - centroid (spiral-contaminated) vs
    # first-harmonic fit (true orbit-center offset; what the optimizer uses)
    ax11 = plt.subplot(3, 4, 11)
    turn_nums = np.arange(1, len(result.turn_metrics['r_center']) + 1)
    ax11.plot(turn_nums, result.turn_metrics['r_center'] * 1000, 'o--', linewidth=1.5, markersize=4,
              color='lightsteelblue', label='centroid (incl. spiral ~dr/2pi)')
    ax11.plot(turn_nums, result.turn_metrics['r_center_h1'] * 1000, 'o-', linewidth=2, markersize=4,
              color='blue', label='1st harmonic (true offset)')
    ax11.set(xlabel='Turn Number', ylabel='Orbit-Center Offset (mm)', title='Orbit Centering')
    ax11.legend(fontsize=7)

    for ax in fig.axes:
        if ax is not ax9 and ax.get_label() != '<colorbar>':
            ax.grid(True, alpha=0.3)
    fig.suptitle(f'Cavity Geometry Optimization: {SPECIES}, E_final={result.final_energy_mev:.2f} MeV, '
                 f'{N_VARIABLE_SEGMENTS} variable segment(s) per gap', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plot_file = output_dir / 'cavity_geometry_optimization.png'
    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved {plot_file}")

    # ========================================================================
    # Save results
    # ========================================================================
    result_file = output_dir / 'optimized_cavity_geometry.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
    (output_dir / 'cavity_geometry_summary.txt').write_text("\n".join(summary) + "\n")
    print(f"Saved {result_file} and cavity_geometry_summary.txt")
    plt.show()


if __name__ == "__main__":
    main()
