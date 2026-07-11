"""
02_optimize_acceleration.py - Optimize Accelerated Cyclotron OrbitsDemonstrates RF parameter optimization for particle acceleration.
Finds optimal bunch phase and RF frequency for reaching target energy.Updated to use unified AcceleratedOrbitFinder (n_particles=1 for single particle).Usage:
python 02_optimize_acceleration.py
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PyPATools.species import IonSpecies
from PyPATools.particles import ParticleDistribution
from PyPATools.field import Field
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from PyCentralRegion.central_region import CentralRegion
from PyCentralRegion.seo_finder import SEOFinder
from PyCentralRegion.rf_cavity import create_four_cavity_system
from PyCentralRegion.accelerated_orbit_finder import (AcceleratedOrbitFinder,
                                                      make_single_particle_beam)

def main():
    print("=" * 70)
    print("ACCELERATED ORBIT OPTIMIZATION (SINGLE PARTICLE)")
    print("=" * 70)# ========================================================================
    # Setup: Create design and load field
    # ========================================================================

    print("\n1. Creating cyclotron design...")
    design = CentralRegion(name="AcceleratingCyclotron", dimensionality='2D')

    # Choose particle species
    species = 'muon'  # or 'H2_1+'
    target_energy_mev = 5.0  # Target final energy

    design.set_species(species)
    print(f"   Species: {design.species.name}")
    print(f"   Mass: {design.species.mass_mev:.3f} MeV/c^2")
    print(f"   Charge: {design.species.q}e")
    print(f"   Target energy: {target_energy_mev} MeV")

    # Load magnetic field
    field_path = Path(__file__).parent.parent / 'resources' / 'midplane_field_0.5mm.comsol'

    if not field_path.exists():
        print(f"\nERROR: Field file not found: {field_path}")
        print("Please ensure midplane_field_0.5mm.comsol is in resources/")
        sys.exit(1)

    print(f"\n2. Loading magnetic field from {field_path.name}...")
    design.set_magnetic_field(field_path, interpolator_backend='fast')
    design.set_electric_field(Field.zero())

    # ========================================================================
    # Find injection orbit (SEO at injection radius)
    # ========================================================================

    print("\n" + "=" * 70)
    print("FINDING INJECTION ORBIT (SEO)")
    print("=" * 70)

    injection_radius_mm = 100.0  # Start at 100 mm radius

    seo_finder = SEOFinder(
        design,
        n_turns=20,
        steps_per_turn=500,
        closure_tol_mm=0.5,
        algorithm='rk4_rel',
        verbose=True
    )

    injection_seo = seo_finder.find_seo_newton(injection_radius_mm)

    if not injection_seo.is_closed:
        print("\nWARNING: Injection orbit not well closed, but continuing...")

    print(f"\nInjection orbit:")
    print(f"  Radius: {injection_seo.radius_mm:.2f} mm")
    print(f"  Energy: {injection_seo.energy_kev / 1000:.3f} MeV")
    print(f"  Frequency: {injection_seo.frequency_hz / 1e6:.3f} MHz")

    rf_frequency = injection_seo.frequency_hz
    injection_v_total = float(np.linalg.norm(injection_seo.v0))

    # ========================================================================
    # Setup RF cavities
    # ========================================================================

    print("\n" + "=" * 70)
    print("SETTING UP RF CAVITIES")
    print("=" * 70)

    # Create 4 double-gap cavities (8 gaps total)
    # Harmonic 4: gaps separated by 90 degrees
    r_min = 0.05  # 50 mm inner radius
    r_max = 0.40  # 400 mm outer radius
    cavity_angles = [22.5, 112.5, 202.5, 292.5]  # First gap of each double-gap cavity
    cavity_phases = [0.0, 0.0, 0.0, 0.0]
    cavity_voltage = 60000.0  # 60 kV per gap

    print(f"\nCreating 4 double-gap cavities (h=4, radial):")
    print(f"  Inner radius: {r_min * 1000:.0f} mm")
    print(f"  Outer radius: {r_max * 1000:.0f} mm")
    print(f"  Angles: {cavity_angles} deg")

    rf_cavities = create_four_cavity_system(
        r_min=r_min,
        r_max=r_max,
        angles=cavity_angles,
        voltage=cavity_voltage,
        frequency=rf_frequency,
        phases=cavity_phases,
        harmonic=4,
        gap_width=0.01,
    )

    print(f"\nCreated {len(rf_cavities)} RF gaps:")
    for i, cav in enumerate(rf_cavities):
        print(f"  Gap {i}: phase={cav.phase_deg:.1f} deg")

    # Add cavities to design
    design.clear_rf_cavities()
    for cav in rf_cavities:
        design.add_rf_cavity(cav)

    # ========================================================================
    # Optimize acceleration parameters
    # ========================================================================

    print("\n" + "=" * 70)
    print("OPTIMIZING ACCELERATION PARAMETERS")
    print("=" * 70)

    # Create output directory for checkpoint
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    checkpoint_file = output_dir / 'acceleration_optimization.csv'

    # Create optimizer (n_particles=1 for single particle mode)
    max_radius_m = 0.35  # 350 mm max radius

    finder = AcceleratedOrbitFinder(
        design,
        target_energy_mev=target_energy_mev,
        max_radius_m=max_radius_m,
        algorithm='rk4_rel',
        steps_per_turn=500,
        verbose=True,
        checkpoint_file=str(checkpoint_file)
    )

    # Optimization parameters
    optimize_params = ['bunch_phase', 'rf_freq', 'r0', 'vr0']

    # Custom bounds (optional)
    bounds = {
        'bunch_phase': (-180, 180),  # degrees
        'rf_freq': (rf_frequency * 0.9, rf_frequency * 1.1),  # +/- 10%
    }

    # Run optimization
    print("\nStarting optimization...")
    print("(This may take several minutes...)\n")

    # Initial beam (single particle). r0_mode='absolute' so the optimized r0/vr0
    # tune the absolute injection point of the reference particle.
    initial_beam = make_single_particle_beam(
        design.species, r=injection_radius_mm / 1000.0, theta_deg=0.0,
        vr=0.0, v_total=injection_v_total)

    result = finder.optimize(
        initial_beam,
        max_turns=11,
        optimize_params=optimize_params,
        method='nelder_mead',
        bounds=bounds,
        maxiter=25,
        r0_mode='absolute',
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

    print(f"\nOptimal parameters:")
    print(f"  Bunch phase: {result.bunch_phase_deg:.2f} degrees")
    print(f"  RF frequency: {result.rf_frequency_mhz:.6f} MHz")
    print(f"  Initial radius: {result.initial_r_mm:.2f} mm")

    print(f"\nTurn metrics:")
    if len(result.turn_metrics['r_center']) > 0:
        print(f"  Initial radius: {result.turn_metrics['r_center'][0] * 1000:.2f} mm")
        print(f"  Final radius: {result.turn_metrics['r_center'][-1] * 1000:.2f} mm")
        print(f"  Max breathing: {np.max(result.turn_metrics['r_spread']) * 1000:.2f} mm")
        if len(result.turn_metrics['dr']) > 0:
            print(f"  Mean turn separation: {np.mean(result.turn_metrics['dr']) * 1000:.2f} mm")
            print(f"  Turn separation std: {np.std(result.turn_metrics['dr']) * 1000:.2f} mm")

    print(f"\nRF cavity performance:")
    total_crossings = sum(cav.n_crossings for cav in design.rf_cavities)
    total_energy = sum(cav.total_energy_gain for cav in design.rf_cavities)
    print(f"  Total cavity crossings: {total_crossings}")
    print(f"  Total energy gain: {total_energy / 1.602176634e-13:.3f} MeV")
    print(f"  Average energy per crossing: {total_energy / total_crossings / 1.602176634e-16:.2f} keV"
          if total_crossings > 0 else "  N/A")

    # ========================================================================
    # Visualization
    # ========================================================================

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 12))

    # Plot 1: Trajectory with field and RF cavities
    ax1 = plt.subplot(3, 4, 1)

    print("  Plotting trajectory...")

    # Field map
    extent = 0.4
    x_plot = np.linspace(-extent, extent, 150)
    y_plot = np.linspace(-extent, extent, 150)
    X, Y = np.meshgrid(x_plot, y_plot, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(len(X.ravel()))])
    bfield = design.bfield(pts)
    bmag = np.sqrt(bfield[:, 0] ** 2 + bfield[:, 1] ** 2 + bfield[:, 2] ** 2).reshape(150, 150)

    contour = ax1.contourf(X, Y, bmag, levels=20, cmap='viridis', alpha=0.3)

    # RF cavities
    for i, cav in enumerate(design.rf_cavities):
        ax1.plot([cav.p1[0], cav.p2[0]], [cav.p1[1], cav.p2[1]],
                 'r-', linewidth=3, alpha=0.7, zorder=5)

    # Trajectory
    traj = result.trajectory_reference
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1, alpha=0.8, label='Orbit')
    ax1.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start', zorder=10)
    ax1.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=10, label='End', zorder=10)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Accelerated Orbit')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Energy vs Turn
    ax2 = plt.subplot(3, 4, 2)
    turns = [p.turn for p in result.poincare_points_all[0]]
    energies = [p.energy_mev for p in result.poincare_points_all[0]]

    ax2.plot(turns, energies, 'o-', linewidth=2, markersize=4)
    ax2.axhline(y=target_energy_mev, color='r', linestyle='--',
                label=f'Target: {target_energy_mev} MeV')
    ax2.set_xlabel('Turn Number')
    ax2.set_ylabel('Energy (MeV)')
    ax2.set_title('Energy Gain vs Turn')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Radius vs Turn
    ax3 = plt.subplot(3, 4, 3)
    radii = [p.r * 1000 for p in result.poincare_points_all[0]]

    ax3.plot(turns, radii, 'o-', linewidth=2, markersize=4, color='green')
    ax3.set_xlabel('Turn Number')
    ax3.set_ylabel('Radius (mm)')
    ax3.set_title('Radius vs Turn')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Turn Separation
    ax4 = plt.subplot(3, 4, 4)
    if len(result.turn_metrics['dr']) > 0:
        dr_mm = result.turn_metrics['dr'] * 1000
        ax4.plot(range(len(dr_mm)), dr_mm, 'o-', linewidth=2, markersize=4, color='purple')
        ax4.set_xlabel('Turn Number')
        ax4.set_ylabel('dr/dturn (mm)')
        ax4.set_title('Turn Separation')
        ax4.grid(True, alpha=0.3)

    # Plot 5: Breathing (radial oscillation amplitude)
    ax5 = plt.subplot(3, 4, 5)
    if len(result.turn_metrics['r_spread']) > 0:
        r_spread_mm = result.turn_metrics['r_spread'] * 1000
        ax5.plot(range(len(r_spread_mm)), r_spread_mm, 'o-',
                 linewidth=2, markersize=4, color='orange')
        ax5.set_xlabel('Turn Number')
        ax5.set_ylabel('Radial Spread (mm)')
        ax5.set_title('Betatron Oscillation Amplitude')
        ax5.grid(True, alpha=0.3)

    # Plot 6: Poincare section (r vs vr)
    ax6 = plt.subplot(3, 4, 6)
    vr_vals = [p.vr for p in result.poincare_points_all[0]]

    # Color by turn number
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

    # Plot 8: Energy gain per cavity crossing
    ax8 = plt.subplot(3, 4, 8)
    if len(result.rf_crossings) > 0:
        crossing_turns = [c.turn for c in result.rf_crossings]
        energy_gains = [c.energy_gain_kev for c in result.rf_crossings]
        cavity_ids = [c.cavity_id for c in result.rf_crossings]

        scatter = ax8.scatter(crossing_turns, energy_gains, c=cavity_ids,
                              cmap='tab10', s=10, alpha=0.6)
        plt.colorbar(scatter, ax=ax8, label='Cavity ID')
        ax8.set_xlabel('Turn Number')
        ax8.set_ylabel('Energy Gain (keV)')
        ax8.set_title('Energy Gain per RF Crossing')
        ax8.grid(True, alpha=0.3)

    # Plot 9: Radius vs Energy
    ax9 = plt.subplot(3, 4, 9)
    ax9.plot(energies, radii, 'o-', linewidth=2, markersize=4, color='brown')
    ax9.set_xlabel('Energy (MeV)')
    ax9.set_ylabel('Radius (mm)')
    ax9.set_title('Radius vs Energy')
    ax9.grid(True, alpha=0.3)

    # Plot 10: Optimization convergence (if checkpoint exists)
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

    # Plot 11: RF crossing phases distribution
    ax11 = plt.subplot(3, 4, 11)
    if len(result.rf_crossings) > 0:
        crossing_phases = [c.phase_deg for c in result.rf_crossings]
        ax11.hist(crossing_phases, bins=36, alpha=0.7, edgecolor='black')
        ax11.set_xlabel('Crossing Phase (degrees)')
        ax11.set_ylabel('Count')
        ax11.set_title('RF Crossing Phase Distribution')
        ax11.grid(True, alpha=0.3)

    # Plot 12: Orbit centering (mean radius per turn)
    ax12 = plt.subplot(3, 4, 12)
    if len(result.turn_metrics['r_center']) > 0:
        turn_nums = np.arange(len(result.turn_metrics['r_center'])) + 1
        ax12.plot(turn_nums, result.turn_metrics['r_center'] * 1000,
                 'o-', linewidth=2, markersize=4, color='blue', label='Actual')
        ax12.set_xlabel('Turn Number')
        ax12.set_ylabel('Mean Radius (mm)')
        ax12.set_title('Orbit Centering')
        ax12.legend()
        ax12.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'Accelerated Orbit Optimization: {species}, '
                 f'E_final={result.final_energy_mev:.2f} MeV, '
                 f'bunch_phase={result.bunch_phase_deg:.1f}°',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    plot_file = output_dir / 'accelerated_orbit_analysis.png'
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
    result_file = output_dir / 'optimized_orbit.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved optimized orbit to {result_file}")

    # Save summary text file
    summary_file = output_dir / 'optimization_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("ACCELERATED ORBIT OPTIMIZATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Species: {species}\n")
        f.write(f"Target energy: {target_energy_mev} MeV\n")
        f.write(f"Final energy: {result.final_energy_mev:.3f} MeV\n")
        f.write(f"Number of turns: {result.n_turns}\n\n")
        f.write(f"Optimal parameters:\n")
        f.write(f"  Bunch phase: {result.bunch_phase_deg:.2f} degrees\n")
        f.write(f"  RF frequency: {result.rf_frequency_mhz:.6f} MHz\n")
        f.write(f"  Initial radius: {result.initial_r_mm:.2f} mm\n")
        f.write(f"  Initial rad. velocity: {result.initial_vr_m_s} m/s\n\n")
        f.write(f"Optimization:\n")
        f.write(f"  Method: {result.metadata['optimization_method']}\n")
        f.write(f"  Time: {result.metadata['optimization_time_s']:.1f} s\n")
        f.write(f"  Iterations: {result.metadata['total_iterations']}\n")
        f.write(f"  Final cost: {result.cost:.2e}\n")

    print(f"Saved summary to {summary_file}")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()