"""
05_optimize_multiparticle.py - Multi-Particle Beam Optimization

Optimizes RF parameters for multi-particle beam quality.
Minimizes beam envelope oscillations and maximizes energy.

Usage:
    python 05_optimize_multiparticle.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from PyCentralRegion.central_region import CentralRegion
from PyCentralRegion.seo_finder import SEOFinder, StaticOrbit
from PyCentralRegion.rf_cavity import create_four_cavity_system
from PyCentralRegion.accelerated_orbit_finder_multiparticle import AcceleratedOrbitFinderMulti
from PyPATools.field import Field


def plot_optimization_history(checkpoint_file, output_dir):
    """Plot optimization convergence from checkpoint file."""
    if not Path(checkpoint_file).exists():
        print(f"Checkpoint file not found: {checkpoint_file}")
        return

    df = pd.read_csv(checkpoint_file)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Cost vs iteration
    ax = axes[0, 0]
    ax.plot(df['iteration'], df['cost'], 'o-', linewidth=1, markersize=3, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title('Optimization Convergence')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Final energy vs iteration
    ax = axes[0, 1]
    ax.plot(df['iteration'], df['final_energy_mev'], 'o-', linewidth=1, markersize=3, alpha=0.7, color='green')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Final Energy (MeV)')
    ax.set_title('Energy Evolution')
    ax.grid(True, alpha=0.3)

    # Envelope oscillation vs iteration
    ax = axes[0, 2]
    ax.plot(df['iteration'], df['envelope_oscillation_mm'], 'o-', linewidth=1, markersize=3, alpha=0.7, color='purple')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Envelope Oscillation (mm)')
    ax.set_title('Beam Size Oscillation')
    ax.grid(True, alpha=0.3)

    # Bunch phase vs iteration
    ax = axes[1, 0]
    ax.plot(df['iteration'], df['bunch_phase_deg'], 'o', markersize=3, alpha=0.7, color='orange')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Bunch Phase (deg)')
    ax.set_title('Phase Parameter')
    ax.grid(True, alpha=0.3)

    # RF frequency vs iteration
    ax = axes[1, 1]
    ax.plot(df['iteration'], df['rf_freq_mhz'], 'o', markersize=3, alpha=0.7, color='red')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('RF Frequency (MHz)')
    ax.set_title('Frequency Parameter')
    ax.grid(True, alpha=0.3)

    # Final radial spread vs iteration
    ax = axes[1, 2]
    ax.plot(df['iteration'], df['final_std_r_mm'], 'o', markersize=3, alpha=0.7, color='brown')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Final Radial Spread (mm)')
    ax.set_title('Final Beam Size')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / 'optimization_history.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved optimization history to {plot_file}")
    plt.close()


def main():
    print("=" * 70)
    print("MULTI-PARTICLE BEAM OPTIMIZATION")
    print("=" * 70)

    # ========================================================================
    # Setup: Create design and load field
    # ========================================================================

    print("\n1. Creating cyclotron design...")
    design = CentralRegion(name="OptimizedMultiParticleCyclotron", dimensionality='2D')

    # Species
    species = 'muon'
    target_energy_mev = 5.0

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
    # Find injection orbit (SEO)
    # ========================================================================

    # print("\n" + "=" * 70)
    # print("FINDING INJECTION ORBIT (SEO)")
    # print("=" * 70)

    # injection_radius_mm = 100.0
    #
    # seo_finder = SEOFinder(
    #     design,
    #     n_turns=20,
    #     steps_per_turn=500,
    #     closure_tol_mm=0.5,
    #     algorithm='RK4',
    #     verbose=True
    # )

    # injection_seo = seo_finder.find_seo_at_radius(injection_radius_mm, n_iterations=5)

    injection_radius_mm = 100.18
    rf_frequency_initial = 42.598633e6  # Hz
    bunch_phase_deg = 20.40
    initial_v_tangential = 26365144.5  # m/s

    injection_seo = StaticOrbit(radius_mm=100.18,
                                energy_kev=411,
                                b_field_avg=0.3,
                                r0=np.array([injection_radius_mm*1e-3, 0.0, 0.0]),
                                v0=np.array([0.0, initial_v_tangential, 0.0]),
                                frequency_hz=rf_frequency_initial,
                                poincare_points=None,
                                )

    # if not injection_seo.is_closed:
    #     print("\nWARNING: Injection orbit not well closed, but continuing...")
    #
    # print(f"\nInjection orbit:")
    # print(f"  Radius: {injection_seo.radius_mm:.2f} mm")
    # print(f"  Energy: {injection_seo.energy_kev / 1000:.3f} MeV")
    # print(f"  Frequency: {injection_seo.frequency_hz / 1e6:.3f} MHz")

    # ========================================================================
    # Setup RF cavities
    # ========================================================================

    print("\n" + "=" * 70)
    print("SETTING UP RF CAVITIES")
    print("=" * 70)

    r_min = 0.05  # 50 mm
    r_max = 0.40  # 400 mm
    cavity_angles = [22.5, 112.5, 202.5, 292.5]
    cavity_phases = [0.0, 0.0, 0.0, 0.0]
    cavity_voltage = 60000.0  # 60 kV per gap

    # Initial guess for RF frequency (will be optimized)
    # rf_frequency_initial = injection_seo.frequency_hz * 4  # h=4 harmonic

    print(f"\nCreating 4 double-gap cavities (h=4, radial):")
    print(f"  Inner radius: {r_min * 1000:.0f} mm")
    print(f"  Outer radius: {r_max * 1000:.0f} mm")
    print(f"  Angles: {cavity_angles} deg")
    print(f"  Initial RF frequency: {rf_frequency_initial / 1e6:.6f} MHz")

    rf_cavities = create_four_cavity_system(
        r_min=r_min,
        r_max=r_max,
        angles=cavity_angles,
        voltage=cavity_voltage,
        frequency=rf_frequency_initial,
        phases=cavity_phases,
        harmonic=4
    )

    design.clear_rf_cavities()
    for cav in rf_cavities:
        design.add_rf_cavity(cav)

    print(f"\nCreated {len(rf_cavities)} RF gaps")

    # ========================================================================
    # Multi-particle optimization
    # ========================================================================

    print("\n" + "=" * 70)
    print("MULTI-PARTICLE OPTIMIZATION")
    print("=" * 70)

    # Optimization settings
    n_particles = 100  # Start small for testing
    r_spread_mm = 0.4
    vr_spread_m_s = 1e4

    # Output directory
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    checkpoint_file = output_dir / 'multiparticle_optimization.csv'

    print(f"\nOptimization settings:")
    print(f"  Particles: {n_particles}")
    print(f"  Initial spread: r={r_spread_mm} mm (1σ), vr={vr_spread_m_s/1e3:.1f} km/s (1σ)")
    print(f"  Checkpoint file: {checkpoint_file}")

    # Create optimizer
    finder = AcceleratedOrbitFinderMulti(
        design,
        target_energy_mev=target_energy_mev,
        n_particles=n_particles,
        max_radius_m=0.4,
        algorithm='rk4_rel',
        steps_per_turn=500,
        dump_frequency=5,
        verbose=True,
        checkpoint_file=str(checkpoint_file)
    )

    # Cost function weights
    weights = {
        'energy': 0.1,         # Maximize energy
        'spread': 1000.0,      # Minimize envelope oscillation
        'center': 100.0,       # Minimize centering error
        'smooth': 10000.0      # Smooth turn-by-turn progression
    }

    print(f"\nCost function weights:")
    print(f"  Energy (maximize): {weights['energy']}")
    print(f"  Envelope oscillation (minimize): {weights['spread']}")
    print(f"  Centering (minimize): {weights['center']}")
    print(f"  Turn smoothness (minimize): {weights['smooth']}")

    # Run optimization
    print("\nStarting optimization...")
    result, full_beam = finder.optimize(
        initial_seo=injection_seo,
        initial_phase=bunch_phase_deg,
        max_turns=12,
        r_spread_mm=r_spread_mm,
        vr_spread_m_s=vr_spread_m_s,
        optimize_params=['bunch_phase', 'rf_freq', 'r0', 'vr0'],
        method='nelder_mead',
        weights=weights,
        maxiter=100
    )

    # ========================================================================
    # Results summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"\nOptimized parameters:")
    print(f"  Bunch phase: {result.bunch_phase_deg:.2f} deg")
    print(f"  RF frequency: {result.rf_frequency_mhz:.6f} MHz")
    print(f"  Initial radius: {result.initial_r_mm:.2f} mm")

    print(f"\nBeam performance:")
    print(f"  Success: {result.success}")
    print(f"  Final energy: {result.final_energy_mev:.3f} MeV (target: {target_energy_mev} MeV)")
    print(f"  Number of turns: {result.n_turns}")
    print(f"  Number of particles: {result.n_particles}")

    if len(result.turn_statistics) > 0:
        print(f"\nInitial beam:")
        print(f"  Mean radius: {result.turn_statistics[0].mean_r * 1000:.2f} mm")
        print(f"  Radial spread (σ): {result.turn_statistics[0].std_r * 1000:.3f} mm")
        print(f"  Mean energy: {result.turn_statistics[0].mean_energy_mev:.3f} MeV")

        print(f"\nFinal beam:")
        print(f"  Mean radius: {result.turn_statistics[-1].mean_r * 1000:.2f} mm")
        print(f"  Radial spread (σ): {result.turn_statistics[-1].std_r * 1000:.3f} mm")
        print(f"  Mean energy: {result.turn_statistics[-1].mean_energy_mev:.3f} MeV")
        print(f"  Energy spread (σ): {result.turn_statistics[-1].std_energy_mev * 1000:.2f} keV")

    print(f"\nBeam quality:")
    print(f"  Envelope oscillation (σ): {result.metadata['envelope_oscillation_mm']:.3f} mm")
    print(f"  Final cost: {result.cost:.2e}")

    # ========================================================================
    # Visualization
    # ========================================================================

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    # Plot 1: Optimization history
    plot_optimization_history(checkpoint_file, output_dir)

    # Plot 2: Final beam results
    fig = plt.figure(figsize=(20, 12))

    # Trajectory with field
    ax1 = plt.subplot(3, 4, 1)
    extent = 0.4
    x_plot = np.linspace(-extent, extent, 150)
    y_plot = np.linspace(-extent, extent, 150)
    X, Y = np.meshgrid(x_plot, y_plot, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(len(X.ravel()))])
    bfield = design.bfield(pts)
    bmag = np.sqrt(bfield[:, 0] ** 2 + bfield[:, 1] ** 2 + bfield[:, 2] ** 2).reshape(150, 150)
    contour = ax1.contourf(X, Y, bmag, levels=20, cmap='viridis', alpha=0.3)

    # RF cavities
    for cav in design.rf_cavities:
        ax1.plot([cav.p1[0], cav.p2[0]], [cav.p1[1], cav.p2[1]],
                 'r-', linewidth=3, alpha=0.7, zorder=5)

    # Reference trajectory
    traj = result.trajectory_reference
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1, alpha=0.8, label='Reference')
    ax1.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start', zorder=10)
    ax1.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=10, label='End', zorder=10)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Reference Particle Trajectory')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Mean Energy vs Turn
    ax2 = plt.subplot(3, 4, 2)
    turns = [s.turn for s in result.turn_statistics]
    energies = [s.mean_energy_mev for s in result.turn_statistics]
    energy_std = [s.std_energy_mev for s in result.turn_statistics]
    ax2.errorbar(turns, energies, yerr=energy_std, fmt='o-', linewidth=2,
                 markersize=4, capsize=3, label='Mean +/- σ')
    ax2.axhline(y=target_energy_mev, color='r', linestyle='--',
                label=f'Target: {target_energy_mev} MeV')
    ax2.set_xlabel('Turn Number')
    ax2.set_ylabel('Energy (MeV)')
    ax2.set_title('Beam Energy Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Mean Radius vs Turn
    ax3 = plt.subplot(3, 4, 3)
    radii_mean = [s.mean_r * 1000 for s in result.turn_statistics]
    ax3.plot(turns, radii_mean, 'o-', linewidth=2, markersize=4, color='green')
    ax3.set_xlabel('Turn Number')
    ax3.set_ylabel('Mean Radius (mm)')
    ax3.set_title('Mean Orbit Radius')
    ax3.grid(True, alpha=0.3)

    # Radial Spread vs Turn
    ax4 = plt.subplot(3, 4, 4)
    radii_std = [s.std_r * 1000 for s in result.turn_statistics]
    ax4.plot(turns, radii_std, 'o-', linewidth=2, markersize=4, color='purple')
    ax4.set_xlabel('Turn Number')
    ax4.set_ylabel('Radial Spread σ_r (mm)')
    ax4.set_title('Beam Size Evolution (per turn)')
    ax4.grid(True, alpha=0.3)

    # Envelope oscillation (step-by-step)
    ax5 = plt.subplot(3, 4, 5)
    time_steps = np.arange(len(result.std_r_per_step))
    ax5.plot(time_steps, result.std_r_per_step * 1000, '-', linewidth=1, alpha=0.7, color='blue')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Radial Spread σ_r (mm)')
    ax5.set_title(f'Envelope Evolution (σ={result.metadata["envelope_oscillation_mm"]:.3f} mm)')
    ax5.grid(True, alpha=0.3)

    # Poincaré Section - Initial
    ax6 = plt.subplot(3, 4, 6)
    if len(result.poincare_points_all[0]) > 0:
        for i, poincare_list in enumerate(result.poincare_points_all[:100]):
            if len(poincare_list) > 0:
                r_vals = [p.r * 1000 for p in poincare_list if p.turn == 0]
                vr_vals = [p.vr for p in poincare_list if p.turn == 0]
                if len(r_vals) > 0:
                    ax6.plot(r_vals[0], vr_vals[0], 'o', markersize=3, alpha=0.5, color='blue')
    ax6.set_xlabel('Radius (mm)')
    ax6.set_ylabel('Radial Velocity (m/s)')
    ax6.set_title('Initial Distribution (Turn 0)')
    ax6.grid(True, alpha=0.3)

    # Poincaré Section - Final
    ax7 = plt.subplot(3, 4, 7)
    final_turn = result.n_turns - 1
    if final_turn >= 0:
        for i, poincare_list in enumerate(result.poincare_points_all[:100]):
            if len(poincare_list) > 0:
                r_vals = [p.r * 1000 for p in poincare_list if p.turn == final_turn]
                vr_vals = [p.vr for p in poincare_list if p.turn == final_turn]
                if len(r_vals) > 0:
                    ax7.plot(r_vals[0], vr_vals[0], 'o', markersize=3, alpha=0.5, color='red')
    ax7.set_xlabel('Radius (mm)')
    ax7.set_ylabel('Radial Velocity (m/s)')
    ax7.set_title(f'Final Distribution (Turn {final_turn})')
    ax7.grid(True, alpha=0.3)

    # Reference Particle Phase Space
    ax8 = plt.subplot(3, 4, 8)
    if len(result.poincare_points_all[0]) > 0:
        ref_poincare = result.poincare_points_all[0]
        ref_r = [p.r * 1000 for p in ref_poincare]
        ref_vr = [p.vr for p in ref_poincare]
        ref_turns = [p.turn for p in ref_poincare]
        scatter = ax8.scatter(ref_r, ref_vr, c=ref_turns, cmap='viridis', s=20)
        plt.colorbar(scatter, ax=ax8, label='Turn')
    ax8.set_xlabel('Radius (mm)')
    ax8.set_ylabel('Radial Velocity (m/s)')
    ax8.set_title('Reference Particle Phase Space')
    ax8.grid(True, alpha=0.3)

    # Energy Spread vs Turn
    ax9 = plt.subplot(3, 4, 9)
    energy_spread_kev = [s.std_energy_mev * 1000 for s in result.turn_statistics]
    ax9.plot(turns, energy_spread_kev, 'o-', linewidth=2, markersize=4, color='orange')
    ax9.set_xlabel('Turn Number')
    ax9.set_ylabel('Energy Spread σ_E (keV)')
    ax9.set_title('Energy Spread Evolution')
    ax9.grid(True, alpha=0.3)

    # RF Phase Distribution
    ax10 = plt.subplot(3, 4, 10)
    if len(result.rf_crossings) > 0:
        phases = [c.phase_deg for c in result.rf_crossings]
        ax10.hist(phases, bins=36, alpha=0.7, edgecolor='black')
        ax10.axvline(x=result.bunch_phase_deg, color='r', linestyle='--', linewidth=2,
                     label=f'Optimized: {result.bunch_phase_deg:.1f}°')
        ax10.set_xlabel('RF Phase (degrees)')
        ax10.set_ylabel('Count')
        ax10.set_title('RF Crossing Phase Distribution')
        ax10.legend()
        ax10.grid(True, alpha=0.3)

    # Energy Gain Distribution
    ax11 = plt.subplot(3, 4, 11)
    if len(result.rf_crossings) > 0:
        gains = [c.energy_gain_kev for c in result.rf_crossings]
        ax11.hist(gains, bins=50, alpha=0.7, edgecolor='black')
        ax11.set_xlabel('Energy Gain (keV)')
        ax11.set_ylabel('Count')
        ax11.set_title('RF Energy Gain Distribution')
        ax11.grid(True, alpha=0.3)

    # Beam Quality Factor
    ax12 = plt.subplot(3, 4, 12)
    quality = [s.std_r / s.mean_r * 100 for s in result.turn_statistics]
    ax12.plot(turns, quality, 'o-', linewidth=2, markersize=4, color='brown')
    ax12.set_xlabel('Turn Number')
    ax12.set_ylabel('Relative Spread (%)')
    ax12.set_title('Beam Quality (σ_r / <r>)')
    ax12.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'Multi-Particle Optimization: {species}, {n_particles} particles, '
                 f'E_final={result.final_energy_mev:.2f} MeV, '
                 f'Envelope σ={result.metadata["envelope_oscillation_mm"]:.2f} mm',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    plot_file = output_dir / 'multiparticle_optimization_result.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved result plot to {plot_file}")

    plt.show()

    # Plot full trajectories
    fig = plt.figure(figsize=(12, 12))

    # Field map
    extent = 0.4
    x_plot = np.linspace(-extent, extent, 150)
    y_plot = np.linspace(-extent, extent, 150)
    X, Y = np.meshgrid(x_plot, y_plot, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(len(X.ravel()))])
    bfield = design.bfield(pts)
    bmag = np.sqrt(bfield[:, 0] ** 2 + bfield[:, 1] ** 2 + bfield[:, 2] ** 2).reshape(150, 150)

    contour = plt.contourf(X, Y, bmag, levels=20, cmap='viridis', alpha=0.3)

    for i in range(full_beam.shape[1]):
        plt.plot(full_beam[:, i, 0], full_beam[:, i, 1], color="blue")

    plt.show()

    # ========================================================================
    # Save results
    # ========================================================================

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    import pickle
    result_file = output_dir / 'multiparticle_optimization_result.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved result to {result_file}")

    print("\n" + "=" * 70)
    print("MULTI-PARTICLE OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
