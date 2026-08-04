"""
06_optimize_cavity_geometry.py - Optimize Cavity Geometry + RF Parameters

Demonstrates staged, parallel optimization of per-gap RF cavity segment
geometry together with RF parameters (bunch phase, frequency) for a
user-supplied initial beam (spiral-inflector hand-off).

Usage:
    python 06_optimize_cavity_geometry.py
"""

import os
import io
import sys
import contextlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from PyPATools.field import Field

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

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
FIELD_PATH = Path(__file__).parent.parent / 'resources' /  'uCyclo_v3_midplane_field.comsol'  # 'uCyclo_v2_Midplane2D_400mm_0.5mm.comsol'
SPECIES = 'muon'
TARGET_ENERGY_MEV = 5.0
MAX_RADIUS_M = 0.4

R_MIN = 0.005                      # 5 mm inner cavity radius
R_MAX = 0.40                       # 400 mm outer cavity radius
CAVITY_VOLTAGE = 50000.0           # 60 kV originally
HARMONIC = 4                       # 4 originally

# Dee system: each cavity is (center angle, opening angle); gaps at
# center +- opening/2. Two-dee system: DEE_CENTER_ANGLES = [90.0, 270.0].
DEE_CENTER_ANGLES = [45.0, 135.0, 225.0, 315.0]  # [45.0, 135.0, 225.0, 315.0] originally
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
# Params: 8 gaps x (angle + radius + rotation) + 2 RF = 26.
ROTATABLE_SEGMENTS = True
ROTATION_MAX = 20.0                    # per-segment rotation bounds [deg]

ISO_RADII_MM = [100, 150, 200, 250, 300]   # radii for the isochronous-freq average
RF_FREQUENCY_OVERRIDE = None               # e.g. 42.5e6 to set manually [Hz]

# Each worker holds its own copy of the field map, so memory (not CPU) is the
# practical cap; 12 is a good default on a 28-core / large-RAM machine.
N_WORKERS = max(1, min(12, (os.cpu_count() or 2) - 2))


def build_system(rf_frequency=None, quiet=False, checkpoint_file=None):
    """Build (design, finder, geometry optimizer, rf_frequency).

    Module-level and deterministic so each multiprocessing worker can rebuild
    the identical system. With quiet=True all construction chatter is
    suppressed (used in workers).
    """
    sink = io.StringIO() if quiet else None
    with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
        design = CentralRegion(name="GeometryOptimizedCyclotron", dimensionality='2D')
        design.set_species(SPECIES)
        design.set_magnetic_field(FIELD_PATH, interpolator_backend='fast')
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
            pinch_target_r_m=[0.016, 0.013, 0.030, 0.036, 0.047, 0.052, 0.063, 0.012],
            pinch_metal_width_m=0.002,  # match your BEM build
            pinch_weight=50.0
        )

    return design, finder, geo_optimizer, rf_frequency


def build_worker_optimizer(rf_frequency):
    """Worker-side builder (module-level, picklable by reference)."""
    _, _, geo, _ = build_system(rf_frequency=rf_frequency, quiet=True)
    return geo


def make_initial_beam(species):
    """Initial single-particle beam from the spiral-inflector hand-off.

    Cylindrical lab coordinates: position (r, theta, z), momentum
    (p_r, p_theta, p_z) in beta*gamma. Module-level so downstream scripts
    (e.g. 07_bem_gap_verification.py) reuse the identical injection point.
    """
    return make_beam_from_cylindrical(
        species,
        r=27.7e-3, theta_deg=-7.6, z=0.07e-3,
        p_r=-0.0031, p_theta=0.023, p_z=0.00026,
    )


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
    print("=" * 70)
    print("CAVITY GEOMETRY + RF PARAMETER OPTIMIZATION")
    print("=" * 70)

    # ========================================================================
    # Setup: build the whole system (design + finder + geometry optimizer)
    # ========================================================================
    # The RF base frequency is the ISOCHRONOUS-region frequency (average over
    # larger-radius SEOs), NOT the injection orbit's - the center field has no
    # flutter, so the innermost orbit is far off-isochronous. Set
    # RF_FREQUENCY_OVERRIDE at the top of this file to specify it manually.

    if not FIELD_PATH.exists():
        print(f"\nERROR: Field file not found: {FIELD_PATH}")
        sys.exit(1)

    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    checkpoint_file = output_dir / 'cavity_geometry_optimization.csv'

    print("\n1. Building system (field load + isochronous-frequency scan)...")
    design, finder, geo_optimizer, rf_frequency = build_system(
        rf_frequency=RF_FREQUENCY_OVERRIDE, checkpoint_file=str(checkpoint_file))

    species = SPECIES
    target_energy_mev = TARGET_ENERGY_MEV
    r_max = R_MAX
    n_variable_segments = N_VARIABLE_SEGMENTS

    print(f"   Species: {design.species.name}, target {target_energy_mev} MeV")
    print(f"   RF base frequency: {rf_frequency / 1e6:.4f} MHz "
          f"({'manual' if RF_FREQUENCY_OVERRIDE else f'isochronous avg over {ISO_RADII_MM} mm'})")
    print(f"   {len(design.rf_cavities)} gaps x {n_variable_segments} variable segment(s)")

    # ========================================================================
    # Optimize geometry + RF parameters
    # ========================================================================

    print("\n" + "=" * 70)
    print("OPTIMIZING CAVITY GEOMETRY + RF PARAMETERS")
    print("=" * 70)

    # Optimize geometry, bunch phase, and RF frequency
    rf_optimize_params = ['bunch_phase', 'rf_freq']

    rf_bounds = {
        'bunch_phase': (-180, 180),
        'rf_freq': (rf_frequency * 0.95, rf_frequency * 1.05),
    }

    print(f"\nStarting staged optimization (RF scan -> DFO-LS -> verify), "
          f"{N_WORKERS} workers...\n")

    # Initial single-particle beam from the spiral-inflector hand-off (see
    # make_initial_beam - shared with the BEM verification script).
    initial_beam = make_initial_beam(design.species)

    # Stage A: coarse (phase x freq) scan with straight geometry, farmed over
    # the worker pool; Stage B: multi-start DFO-LS (one start per worker, seeded
    # from distinct stage-A basins) on all 18 params at search resolution
    # (300 spt / 8 turns = the measured reliability floor); Stage C: verify the
    # winner at full resolution. Per-eval checkpoints land in per-worker CSVs
    # (<checkpoint>.worker-<pid>.csv).
    result = geo_optimizer.optimize_staged(
        initial_beam,
        rf_optimize_params=rf_optimize_params,
        rf_bounds=rf_bounds,
        search_steps_per_turn=300,
        search_max_turns=10,
        final_steps_per_turn=500,
        final_max_turns=15,
        maxfun=600,  # DFO-LS budget PER START (26 params; ~70 min wall, parallel starts)
        r0_mode='offset',
        # First turns are inherently lopsided (modified gaps + off-orbit
        # injection); exempt them from the centering/smoothness residuals.
        skip_turns=4,
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
    print(f"Final energy: {result.final_energy_mev:.3f} MeV (target: {target_energy_mev} MeV)")
    print(f"Number of turns: {result.n_turns}")
    print(f"Final cost: {result.cost:.2e}")

    print(f"\nOptimal RF parameters:")
    print(f"  Bunch phase: {result.bunch_phase_deg:.2f} degrees")
    print(f"  RF frequency: {result.rf_frequency_mhz:.6f} MHz")

    print(f"\nOptimal cavity geometry (per gap):")
    geom = result.metadata['optimal_geometry']
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

    # Plot 1: Optimized cavity geometry with trajectory
    ax1 = plt.subplot(3, 4, 1)

    # Field map
    extent = 0.4
    x_plot = np.linspace(-extent, extent, 150)
    y_plot = np.linspace(-extent, extent, 150)
    X, Y = np.meshgrid(x_plot, y_plot, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(len(X.ravel()))])
    bfield = design.bfield(pts)
    bmag = np.sqrt(bfield[:, 0] ** 2 + bfield[:, 1] ** 2 + bfield[:, 2] ** 2).reshape(150, 150)

    contour = ax1.contourf(X, Y, bmag, levels=20, cmap='viridis', alpha=0.3)

    # Optimized cavities
    for cav in design.rf_cavities:
        for segment in cav.segments:
            linestyle = '-' if segment['type'] == 'variable' else '--'
            linewidth = 3 if segment['type'] == 'variable' else 2
            ax1.plot([segment['p1'][0], segment['p2'][0]],
                     [segment['p1'][1], segment['p2'][1]],
                     'r-', linestyle=linestyle, linewidth=linewidth, alpha=0.7, zorder=5)

    # Trajectory
    traj = result.trajectory_reference
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1, alpha=0.8, label='Orbit')
    ax1.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start', zorder=10)
    ax1.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=10, label='End', zorder=10)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Optimized Cavity Geometry + Trajectory')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Detailed cavity geometry (zoomed)
    ax2 = plt.subplot(3, 4, 2)
    plot_cavity_geometry(ax2, design.rf_cavities[:2], "Cavity Geometry (Detail)")
    ax2.set_xlim(-0.2, 0.2)
    ax2.set_ylim(-0.1, 0.3)

    # Plot 3: Energy vs Turn
    ax3 = plt.subplot(3, 4, 3)
    turns = [p.turn for p in result.poincare_points_all[0]]
    energies = [p.energy_mev for p in result.poincare_points_all[0]]

    ax3.plot(turns, energies, 'o-', linewidth=2, markersize=4)
    ax3.axhline(y=target_energy_mev, color='r', linestyle='--',
                label=f'Target: {target_energy_mev} MeV')
    ax3.set_xlabel('Turn Number')
    ax3.set_ylabel('Energy (MeV)')
    ax3.set_title('Energy Gain vs Turn')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Radius vs Turn
    ax4 = plt.subplot(3, 4, 4)
    radii = [p.r * 1000 for p in result.poincare_points_all[0]]

    ax4.plot(turns, radii, 'o-', linewidth=2, markersize=4, color='green')
    ax4.set_xlabel('Turn Number')
    ax4.set_ylabel('Radius (mm)')
    ax4.set_title('Radius vs Turn')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Turn Separation
    ax5 = plt.subplot(3, 4, 5)
    if len(result.turn_metrics['dr']) > 0:
        dr_mm = result.turn_metrics['dr'] * 1000
        ax5.plot(range(len(dr_mm)), dr_mm, 'o-', linewidth=2, markersize=4, color='purple')
        ax5.set_xlabel('Turn Number')
        ax5.set_ylabel('dr/dturn (mm)')
        ax5.set_title('Turn Separation')
        ax5.grid(True, alpha=0.3)

    # Plot 6: Poincare section (r vs vr)
    ax6 = plt.subplot(3, 4, 6)
    vr_vals = [p.vr for p in result.poincare_points_all[0]]

    scatter = ax6.scatter(radii, vr_vals, c=turns, cmap='viridis', s=20)
    plt.colorbar(scatter, ax=ax6, label='Turn')
    ax6.set_xlabel('Radius (mm)')
    ax6.set_ylabel('Radial Velocity (m/s)')
    ax6.set_title('Poincare Section (r vs vr)')
    ax6.grid(True, alpha=0.3)

    # Plot 7: RF Phase vs Turn
    ax7 = plt.subplot(3, 4, 7)
    phases = [p.phase_deg for p in result.poincare_points_all[0]]

    ax7.plot(turns, phases, 'o', markersize=4, alpha=0.6)
    ax7.set_xlabel('Turn Number')
    ax7.set_ylabel('RF Phase (degrees)')
    ax7.set_title('Synchrotron Oscillation')
    ax7.grid(True, alpha=0.3)

    # Plot 8: Radius vs Energy
    ax8 = plt.subplot(3, 4, 8)
    ax8.plot(energies, radii, 'o-', linewidth=2, markersize=4, color='brown')
    ax8.set_xlabel('Energy (MeV)')
    ax8.set_ylabel('Radius (mm)')
    ax8.set_title('Radius vs Energy')
    ax8.grid(True, alpha=0.3)

    # Plot 9: Segment geometry diagram (per gap)
    ax9 = plt.subplot(3, 4, 9)

    ax9.text(0.02, 0.95, "Optimized Segment Geometry (per gap):",
             transform=ax9.transAxes, fontweight='bold', fontsize=9)

    y_pos = 0.86
    for g, (angs, rads) in enumerate(zip(geom['segment_angles_per_gap'],
                                         geom['segment_radii_per_gap'])):
        text = f"G{g}: " + " ".join(f"{a:+.1f}°/{r * 1000:.0f}mm"
                                    for a, r in zip(angs, rads))
        ax9.text(0.02, y_pos, text, transform=ax9.transAxes, fontsize=7)
        y_pos -= 0.09

    ax9.text(0.02, y_pos, f"Fixed outer: r={r_max * 1000:.0f}mm",
             transform=ax9.transAxes, fontsize=8, style='italic')

    ax9.axis('off')
    ax9.set_title('Geometry Parameters')

    # Plot 10: Optimization convergence
    ax10 = plt.subplot(3, 4, 10)
    if checkpoint_file.exists():
        import csv
        iterations = []
        costs = []
        with open(checkpoint_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                iterations.append(int(row['iteration']))
                costs.append(float(row['cost']))

        ax10.semilogy(iterations, costs, 'o-', linewidth=1, markersize=3, alpha=0.7)
        ax10.set_xlabel('Iteration')
        ax10.set_ylabel('Cost Function')
        ax10.set_title('Optimization Convergence')
        ax10.grid(True, alpha=0.3)

    # Plot 11: Orbit centering - centroid (legacy, spiral-contaminated) vs
    # first-harmonic fit (true orbit-center offset; what the optimizer uses)
    ax11 = plt.subplot(3, 4, 11)
    if len(result.turn_metrics['r_center']) > 0:
        turn_nums = np.arange(len(result.turn_metrics['r_center'])) + 1
        ax11.plot(turn_nums, result.turn_metrics['r_center'] * 1000,
                  'o--', linewidth=1.5, markersize=4, color='lightsteelblue',
                  label='centroid (incl. spiral ~dr/2π)')
        ax11.plot(turn_nums, result.turn_metrics['r_center_h1'] * 1000,
                  'o-', linewidth=2, markersize=4, color='blue',
                  label='1st harmonic (true offset)')
        ax11.set_xlabel('Turn Number')
        ax11.set_ylabel('Orbit-Center Offset (mm)')
        ax11.set_title('Orbit Centering')
        ax11.legend(fontsize=7)
        ax11.grid(True, alpha=0.3)

    # Plot 12: Cavity segment-angle evolution (per gap, from checkpoint)
    ax12 = plt.subplot(3, 4, 12)
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            reader = csv.DictReader(f)
            angle_cols = [c for c in reader.fieldnames if c.endswith('_angle_deg')]
            history = {c: [] for c in angle_cols}
            for row in reader:
                for c in angle_cols:
                    history[c].append(float(row[c]))

        for c in angle_cols:
            ax12.plot(history[c], linewidth=1, alpha=0.6,
                      label=c.replace('_angle_deg', ''))
        ax12.set_xlabel('Iteration')
        ax12.set_ylabel('Angle (degrees)')
        ax12.set_title('Segment Angle Evolution (all gaps)')
        if len(angle_cols) <= 8:
            ax12.legend(fontsize=6)
        ax12.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'Cavity Geometry Optimization: {species}, '
                 f'E_final={result.final_energy_mev:.2f} MeV, '
                 f'{n_variable_segments} variable segments per cavity',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    plot_file = output_dir / 'cavity_geometry_optimization.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {plot_file}")

    plt.show()

    # ========================================================================
    # Save results
    # ========================================================================

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Save orbit data
    import pickle
    result_file = output_dir / 'optimized_cavity_geometry.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved optimized result to {result_file}")

    # Save summary text file
    summary_file = output_dir / 'cavity_geometry_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("CAVITY GEOMETRY + RF OPTIMIZATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Species: {species}\n")
        f.write(f"Target energy: {target_energy_mev} MeV\n")
        f.write(f"Final energy: {result.final_energy_mev:.3f} MeV\n")
        f.write(f"Number of turns: {result.n_turns}\n\n")

        f.write(f"Optimal RF parameters:\n")
        f.write(f"  Bunch phase: {result.bunch_phase_deg:.2f} degrees\n")
        f.write(f"  RF frequency: {result.rf_frequency_mhz:.6f} MHz\n\n")

        f.write(f"Optimal cavity geometry ({n_variable_segments} segments per gap):\n")
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
    print("CAVITY GEOMETRY OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()