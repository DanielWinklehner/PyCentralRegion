"""
04_multiparticle_tracking.py - Multi-Particle Beam Tracking

Demonstrates multi-particle tracking with optimized RF parameters.
Uses results from single-particle optimization as starting point.

Usage:
    python 04_multiparticle_tracking.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from PyCentralRegion.central_region import CentralRegion
from PyCentralRegion.seo_finder import SEOFinder
from PyCentralRegion.rf_cavity import create_four_cavity_system
from PyCentralRegion.accelerated_orbit_finder_multiparticle import AcceleratedOrbitFinderMulti
from PyPATools.field import Field


def main():
    print("=" * 70)
    print("MULTI-PARTICLE BEAM TRACKING")
    print("=" * 70)

    # ========================================================================
    # Setup: Create design and load field
    # ========================================================================

    print("\n1. Creating cyclotron design...")
    design = CentralRegion(name="MultiParticleCyclotron", dimensionality='2D')

    # Species
    species = 'muon'
    target_energy_mev = 5.0

    design.set_species(species)
    print(f"   Species: {design.species.name}")
    print(f"   Mass: {design.species.mass_mev:.3f} MeV/c^2")
    print(f"   Charge: {design.species.q}e")
    print(f"   Target energy: {target_energy_mev} MeV")

    # Load magnetic field with fast interpolator
    field_path = Path(__file__).parent.parent / 'resources' / 'midplane_field_0.5mm.comsol'

    if not field_path.exists():
        print(f"\nERROR: Field file not found: {field_path}")
        print("Please ensure midplane_field_0.5mm.comsol is in resources/")
        sys.exit(1)

    print(f"\n2. Loading magnetic field from {field_path.name}...")
    design.set_magnetic_field(field_path, interpolator_backend='fast')
    design.set_electric_field(Field.zero())

    # ========================================================================
    # Find injection orbit (SEO at optimized radius)
    # ========================================================================

    print("\n" + "=" * 70)
    print("FINDING INJECTION ORBIT (SEO)")
    print("=" * 70)

    # Use optimized radius from single-particle run
    injection_radius_mm = 100.18

    # seo_finder = SEOFinder(
    #     design,
    #     n_turns=20,
    #     steps_per_turn=500,
    #     closure_tol_mm=0.5,
    #     algorithm='RK4',
    #     verbose=True
    # )
    #
    # injection_seo = seo_finder.find_seo_at_radius(injection_radius_mm, n_iterations=5)
    #
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

    r_min = 0.05  # 50 mm inner radius
    r_max = 0.40  # 400 mm outer radius
    cavity_angles = [22.5, 112.5, 202.5, 292.5]
    cavity_phases = [0.0, 0.0, 0.0, 0.0]
    cavity_voltage = 60000.0  # 60 kV per gap

    # Use optimized RF frequency from single-particle run
    rf_frequency = 42.598633e6  # Hz

    print(f"\nCreating 4 double-gap cavities (h=4, radial):")
    print(f"  Inner radius: {r_min * 1000:.0f} mm")
    print(f"  Outer radius: {r_max * 1000:.0f} mm")
    print(f"  Angles: {cavity_angles} deg")
    print(f"  RF frequency: {rf_frequency / 1e6:.6f} MHz")

    rf_cavities = create_four_cavity_system(
        r_min=r_min,
        r_max=r_max,
        angles=cavity_angles,
        voltage=cavity_voltage,
        frequency=rf_frequency,
        phases=cavity_phases,
        harmonic=4
    )

    # Add cavities to design
    design.clear_rf_cavities()
    for cav in rf_cavities:
        design.add_rf_cavity(cav)

    print(f"\nCreated {len(rf_cavities)} RF gaps")

    # ========================================================================
    # Multi-particle tracking with optimized parameters
    # ========================================================================

    print("\n" + "=" * 70)
    print("MULTI-PARTICLE TRACKING")
    print("=" * 70)

    # Create multi-particle finder
    n_particles = 100

    finder = AcceleratedOrbitFinderMulti(
        design,
        target_energy_mev=target_energy_mev,
        n_particles=n_particles,
        max_radius_m=0.35,
        algorithm='rk4_rel',
        steps_per_turn=500,
        dump_frequency=5,
        verbose=True
    )

    # Use optimized parameters from single-particle run
    bunch_phase_deg = 20.40

    # Extract values from SEO
    # initial_r_mm = injection_seo.radius_mm
    # initial_v_tangential = np.linalg.norm(injection_seo.v0)

    initial_v_tangential = 26365144.5  # m/s

    print(f"\nTracking {n_particles} particles with optimized parameters:")
    print(f"  Bunch phase: {bunch_phase_deg:.2f} degrees")
    print(f"  RF frequency: {rf_frequency / 1e6:.6f} MHz")
    print(f"  Initial radius: {injection_radius_mm:.2f} mm")
    print(f"  Initial tangential velocity: {initial_v_tangential / 1e6:.2f} Mm/s")
    print(f"  Initial spread: 2.0 mm (1σ)")
    print(f"  Radial velocity spread: 10 km/s (1σ)")

    # Track
    result, r_plot = finder.track_once(
        initial_r_mm=injection_radius_mm,
        initial_v_tangential_m_s=initial_v_tangential,
        bunch_phase_deg=bunch_phase_deg,
        rf_freq_mhz=rf_frequency / 1e6,
        max_turns=11,
        r_spread_mm=2.0,
        vr_spread_m_s=1e4
    )

    # ========================================================================
    # Results summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("TRACKING RESULTS")
    print("=" * 70)

    print(f"\nSuccess: {result.success}")
    print(f"Final energy: {result.final_energy_mev:.3f} MeV (target: {target_energy_mev} MeV)")
    print(f"Number of turns: {result.n_turns}")
    print(f"Number of particles: {result.n_particles}")

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

    print(f"\nRF cavity performance:")
    total_crossings = sum(cav.n_crossings for cav in design.rf_cavities)
    print(f"  Total cavity crossings: {total_crossings}")
    print(f"  Average crossings per particle: {total_crossings / n_particles:.1f}")

    # ========================================================================
    # Visualization
    # ========================================================================

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 12))

    # Plot 1: Trajectory with field and RF cavities
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

    # Plot 2: Mean Energy vs Turn
    ax2 = plt.subplot(3, 4, 2)
    turns = [s.turn for s in result.turn_statistics]
    energies = [s.mean_energy_mev for s in result.turn_statistics]
    energy_std = [s.std_energy_mev for s in result.turn_statistics]

    ax2.errorbar(turns, energies, yerr=energy_std, fmt='o-', linewidth=2,
                 markersize=4, capsize=3, label='Mean ± σ')
    ax2.axhline(y=target_energy_mev, color='r', linestyle='--',
                label=f'Target: {target_energy_mev} MeV')
    ax2.set_xlabel('Turn Number')
    ax2.set_ylabel('Energy (MeV)')
    ax2.set_title('Beam Energy Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Mean Radius vs Turn
    ax3 = plt.subplot(3, 4, 3)
    radii_mean = [s.mean_r * 1000 for s in result.turn_statistics]

    ax3.plot(turns, radii_mean, 'o-', linewidth=2, markersize=4, color='green')
    ax3.set_xlabel('Turn Number')
    ax3.set_ylabel('Mean Radius (mm)')
    ax3.set_title('Mean Orbit Radius')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Radial Spread (Beam Size) vs Turn
    ax4 = plt.subplot(3, 4, 4)
    radii_std = [s.std_r * 1000 for s in result.turn_statistics]

    ax4.plot(turns, radii_std, 'o-', linewidth=2, markersize=4, color='purple')
    ax4.set_xlabel('Turn Number')
    ax4.set_ylabel('Radial Spread σ_r (mm)')
    ax4.set_title('Beam Size Evolution')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Poincaré Section - All Particles (Turn 0)
    ax5 = plt.subplot(3, 4, 5)
    if len(result.poincare_points_all[0]) > 0:
        for i, poincare_list in enumerate(result.poincare_points_all[:100]):  # First 100
            if len(poincare_list) > 0:
                r_vals = [p.r * 1000 for p in poincare_list if p.turn == 0]
                vr_vals = [p.vr for p in poincare_list if p.turn == 0]
                if len(r_vals) > 0:
                    ax5.plot(r_vals[0], vr_vals[0], 'o', markersize=3, alpha=0.5, color='blue')

    ax5.set_xlabel('Radius (mm)')
    ax5.set_ylabel('Radial Velocity (m/s)')
    ax5.set_title('Initial Distribution (Turn 0)')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Poincaré Section - All Particles (Final Turn)
    ax6 = plt.subplot(3, 4, 6)
    final_turn = result.n_turns - 1
    if final_turn >= 0:
        for i, poincare_list in enumerate(result.poincare_points_all[:100]):
            if len(poincare_list) > 0:
                r_vals = [p.r * 1000 for p in poincare_list if p.turn == final_turn]
                vr_vals = [p.vr for p in poincare_list if p.turn == final_turn]
                if len(r_vals) > 0:
                    ax6.plot(r_vals[0], vr_vals[0], 'o', markersize=3, alpha=0.5, color='red')

    ax6.set_xlabel('Radius (mm)')
    ax6.set_ylabel('Radial Velocity (m/s)')
    ax6.set_title(f'Final Distribution (Turn {final_turn})')
    ax6.grid(True, alpha=0.3)

    # Plot 7: Phase Space Evolution - Reference Particle
    ax7 = plt.subplot(3, 4, 7)
    if len(result.poincare_points_all[0]) > 0:
        ref_poincare = result.poincare_points_all[0]
        ref_r = [p.r * 1000 for p in ref_poincare]
        ref_vr = [p.vr for p in ref_poincare]
        ref_turns = [p.turn for p in ref_poincare]

        scatter = ax7.scatter(ref_r, ref_vr, c=ref_turns, cmap='viridis', s=20)
        plt.colorbar(scatter, ax=ax7, label='Turn')

    ax7.set_xlabel('Radius (mm)')
    ax7.set_ylabel('Radial Velocity (m/s)')
    ax7.set_title('Reference Particle Phase Space')
    ax7.grid(True, alpha=0.3)

    # Plot 8: Energy Spread vs Turn
    ax8 = plt.subplot(3, 4, 8)
    energy_spread_kev = [s.std_energy_mev * 1000 for s in result.turn_statistics]

    ax8.plot(turns, energy_spread_kev, 'o-', linewidth=2, markersize=4, color='orange')
    ax8.set_xlabel('Turn Number')
    ax8.set_ylabel('Energy Spread σ_E (keV)')
    ax8.set_title('Energy Spread Evolution')
    ax8.grid(True, alpha=0.3)

    # Plot 9: RF Phase Distribution at Crossings
    ax9 = plt.subplot(3, 4, 9)
    if len(result.rf_crossings) > 0:
        phases = [c.phase_deg for c in result.rf_crossings]
        ax9.hist(phases, bins=36, alpha=0.7, edgecolor='black')
        ax9.set_xlabel('RF Phase (degrees)')
        ax9.set_ylabel('Count')
        ax9.set_title('RF Crossing Phase Distribution')
        ax9.grid(True, alpha=0.3)

    # Plot 10: Energy Gain Distribution
    ax10 = plt.subplot(3, 4, 10)
    if len(result.rf_crossings) > 0:
        gains = [c.energy_gain_kev for c in result.rf_crossings]
        ax10.hist(gains, bins=50, alpha=0.7, edgecolor='black')
        ax10.set_xlabel('Energy Gain (keV)')
        ax10.set_ylabel('Count')
        ax10.set_title('RF Energy Gain Distribution')
        ax10.grid(True, alpha=0.3)

    # Plot 11: Radius vs Energy (all particles, final turn)
    ax11 = plt.subplot(3, 4, 11)
    if final_turn >= 0:
        final_r = []
        final_e = []
        for poincare_list in result.poincare_points_all:
            for p in poincare_list:
                if p.turn == final_turn:
                    final_r.append(p.r * 1000)
                    final_e.append(p.energy_mev)

        if len(final_r) > 0:
            ax11.plot(final_e, final_r, 'o', markersize=3, alpha=0.5)

    ax11.set_xlabel('Energy (MeV)')
    ax11.set_ylabel('Radius (mm)')
    ax11.set_title(f'Energy vs Radius (Turn {final_turn})')
    ax11.grid(True, alpha=0.3)

    # Plot 12: Beam Quality Factor
    ax12 = plt.subplot(3, 4, 12)
    # Normalized emittance proxy: σ_r / <r>
    quality = [s.std_r / s.mean_r * 100 for s in result.turn_statistics]

    ax12.plot(turns, quality, 'o-', linewidth=2, markersize=4, color='brown')
    ax12.set_xlabel('Turn Number')
    ax12.set_ylabel('Relative Spread (%)')
    ax12.set_title('Beam Quality (σ_r / <r>)')
    ax12.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'Multi-Particle Beam Tracking: {species}, {n_particles} particles, '
                 f'E_final={result.final_energy_mev:.2f} MeV',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    plot_file = output_dir / 'multiparticle_tracking.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {plot_file}")

    plt.show()

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

    for i in range(100):
        plt.plot(r_plot[:, i, 0], r_plot[:, i, 1], color="blue")

    plt.show()

    # ========================================================================
    # Save results
    # ========================================================================

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    import pickle
    result_file = output_dir / 'multiparticle_result.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved result to {result_file}")

    print("\n" + "=" * 70)
    print("MULTI-PARTICLE TRACKING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
