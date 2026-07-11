"""
01_find_static_orbits.py - Find Static Equilibrium Orbits

Demonstrates SEO finding using Poincare section method.
Defines radii, calculates ideal energies from B-field sampling.

Usage:
    python 01_find_static_orbits.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import from src/
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from PyCentralRegion.central_region import CentralRegion
from PyCentralRegion.seo_finder import SEOFinder, save_seo_database, analyze_isochronism


def main():
    print("=" * 70)
    print("STATIC EQUILIBRIUM ORBIT FINDER - POINCARE METHOD")
    print("=" * 70)

    # ========================================================================
    # Setup: Create design and load field
    # ========================================================================

    print("\n1. Creating cyclotron design...")
    design = CentralRegion(name="CompactCyclotron", dimensionality='2D')

    # Set ion species (H2+ molecular hydrogen ion)
    design.set_species('muon')
    print(f"   Species: {design.species.name}")
    print(f"   Mass: {design.species.mass_mev:.3f} MeV/c^2")
    print(f"   Charge: {design.species.q}e")

    # Load magnetic field
    field_path = Path(__file__).parent.parent / 'resources' / 'midplane_field_0.5mm.comsol'

    if not field_path.exists():
        print(f"\nERROR: Field file not found: {field_path}")
        print("Please ensure midplane_field_0.5mm.comsol is in resources/")
        sys.exit(1)

    print(f"\n2. Loading magnetic field from {field_path.name}...")
    design.set_magnetic_field(field_path, interpolator_backend='fast')

    # Sample field at origin to check
    pts = np.array([[0.0, 0.0, 0.0]])
    b0 = design.bfield(pts)
    print(f"   Field at origin: Bz = {b0[0, 2]:.4f} T")

    # Validation
    design.is_valid(verbose=True)

    # ========================================================================
    # Preview: Sample field at different radii
    # ========================================================================

    print("\n" + "=" * 70)
    print("FIELD PREVIEW")
    print("=" * 70)

    preview_radii = [50, 100, 150, 200, 250, 300]  # mm

    print(f"\n{'Radius [mm]':<12} {'B_avg [T]':<12} {'Flutter [%]':<12} {'Ideal E [MeV]':<12}")
    print("-" * 70)

    # Create temporary finder for field sampling
    temp_finder = SEOFinder(design, verbose=False)

    for r_mm in preview_radii:
        field_info = temp_finder.calculate_avg_field(r_mm / 1000.0)
        energy = temp_finder.calculate_ideal_energy(r_mm / 1000.0, field_info['B_avg'])
        print(f"{r_mm:<12.1f} {field_info['B_avg']:<12.4f} "
              f"{field_info['flutter'] * 100:<12.2f} {energy / 1000:<12.3f}")

    # ========================================================================
    # Find SEOs at specified radii
    # ========================================================================

    print("\n" + "=" * 70)
    print("FINDING STATIC EQUILIBRIUM ORBITS")
    print("=" * 70)

    # Define radii to scan (adjust based on your cyclotron size)
    radii_mm = np.linspace(50, 300, 6)

    print(f"\nRadii: {radii_mm}")
    print(f"Number of radii: {len(radii_mm)}")

    # Create SEO finder
    finder = SEOFinder(
        design,
        n_turns=10,  # Track 10 turns
        steps_per_turn=500,  # 500 steps per turn
        n_theta_samples=36,  # Sample field at 36 angles
        closure_tol_mm=0.5,  # 1 mm closure tolerance
        algorithm='rk4_rel',
        verbose=True
    )

    # Find orbits
    orbits = finder.find_seos_at_radii(radii_mm, n_iterations=10)

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'R [mm]':<10} {'E [MeV]':<10} {'B [T]':<10} {'f [MHz]':<10} "
          f"{'Err [mm]':<10} {'Closed':<10}")
    print("-" * 70)

    for orbit in orbits:
        closed_str = "YES" if orbit.is_closed else "NO"
        print(f"{orbit.radius_mm:<10.1f} {orbit.energy_kev / 1000:<10.3f} "
              f"{orbit.b_field_avg:<10.4f} {orbit.frequency_hz / 1e6:<10.3f} "
              f"{orbit.closure_error_mm:<10.3f} {closed_str:<10}")

    # Isochronism analysis
    iso_analysis = analyze_isochronism(orbits)

    print(f"\n{'=' * 70}")
    print("ISOCHRONISM ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Average frequency: {iso_analysis.get('freq_avg_mhz', 0):.3f} MHz")
    print(f"Frequency std dev: {iso_analysis.get('freq_std_mhz', 0):.3f} MHz")
    print(f"Frequency variation: {iso_analysis.get('freq_variation_percent', 0):.2f}%")
    print(f"Is isochronous: {iso_analysis['is_isochronous']} (<1% variation)")

    # ========================================================================
    # Save database
    # ========================================================================

    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)

    db_file = output_dir / 'seo_database.pkl'
    save_seo_database(orbits, str(db_file))

    # ========================================================================
    # Visualization
    # ========================================================================

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    fig = plt.figure(figsize=(18, 12))

    # Plot 1: All orbits overlaid on field map
    ax1 = plt.subplot(3, 3, 1)

    print("  Plotting field map...")
    extent = 0.4  # 350 mm
    x_plot = np.linspace(-extent, extent, 150)
    y_plot = np.linspace(-extent, extent, 150)
    X, Y = np.meshgrid(x_plot, y_plot, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(len(X.ravel()))])
    bfield = design.bfield(pts)
    bmag = np.sqrt(bfield[:, 0] ** 2 + bfield[:, 1] ** 2 + bfield[:, 2] ** 2).reshape(150, 150)

    contour = ax1.contourf(X, Y, bmag, levels=20, cmap='viridis', alpha=0.4)
    plt.colorbar(contour, ax=ax1, label='|B| (T)')

    # Plot orbits
    colors = plt.cm.plasma(np.linspace(0, 1, len(orbits)))
    for orbit, color in zip(orbits, colors):
        if orbit.trajectory is not None:
            label = f"{orbit.radius_mm:.0f} mm"
            linestyle = '-' if orbit.is_closed else '--'
            ax1.plot(orbit.trajectory[:, 0], orbit.trajectory[:, 1],
                     color=color, linestyle=linestyle, linewidth=1.5,
                     label=label, alpha=0.7)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Static Equilibrium Orbits')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=7, loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Energy vs Radius
    ax2 = plt.subplot(3, 3, 2)
    radii = [o.radius_mm for o in orbits]
    energies = [o.energy_kev / 1000.0 for o in orbits]
    closed_mask = [o.is_closed for o in orbits]

    ax2.plot(radii, energies, 'o-', linewidth=2, markersize=8, label='SEO')

    for i, closed in enumerate(closed_mask):
        if not closed:
            ax2.plot(radii[i], energies[i], 'rx', markersize=12,
                     markeredgewidth=3)

    ax2.set_xlabel('Radius (mm)')
    ax2.set_ylabel('Energy (MeV)')
    ax2.set_title('Energy vs Radius')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Frequency vs Energy (Isochronism)
    ax3 = plt.subplot(3, 3, 3)
    frequencies = [o.frequency_hz / 1e6 for o in orbits]

    ax3.plot(energies, frequencies, 's-', linewidth=2, markersize=8,
             color='green', label='Measured')

    for i, closed in enumerate(closed_mask):
        if not closed:
            ax3.plot(energies[i], frequencies[i], 'rx',
                     markersize=12, markeredgewidth=3)

    # Add horizontal line at average frequency
    if len(iso_analysis['frequencies_mhz']) > 0:
        f_avg = iso_analysis['freq_avg_mhz']
        ax3.axhline(y=f_avg, color='red', linestyle='--', alpha=0.5,
                    label=f'Avg: {f_avg:.2f} MHz')

    ax3.set_xlabel('Energy (MeV)')
    ax3.set_ylabel('Orbital Frequency (MHz)')
    ax3.set_title(f"Isochronism: {iso_analysis['freq_variation_percent']:.2f}% variation")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: B-field vs Radius
    ax4 = plt.subplot(3, 3, 4)
    b_fields = [o.b_field_avg for o in orbits]

    ax4.plot(radii, b_fields, 'd-', linewidth=2, markersize=8, color='purple')
    ax4.set_xlabel('Radius (mm)')
    ax4.set_ylabel('Average B-field (T)')
    ax4.set_title('B-field vs Radius')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Closure errors
    ax5 = plt.subplot(3, 3, 5)
    closure_errors = [o.closure_error_mm for o in orbits]

    ax5.semilogy(radii, closure_errors, 'o-', linewidth=2, markersize=8,
                 color='red')
    ax5.axhline(y=finder.closure_tol_mm, color='k', linestyle='--',
                alpha=0.5, label=f'Tolerance: {finder.closure_tol_mm} mm')
    ax5.set_xlabel('Radius (mm)')
    ax5.set_ylabel('Closure Error (mm)')
    ax5.set_title('Orbit Closure Quality')
    ax5.grid(True, alpha=0.3, which='both')
    ax5.legend()

    # Plot 6: Poincare section for middle orbit
    ax6 = plt.subplot(3, 3, 6)
    mid_orbit = orbits[len(orbits) // 2]
    finder.plot_poincare_section(mid_orbit, ax=ax6)

    # Plot 7-9: Individual Poincare sections for 3 selected orbits
    selected_indices = [0, len(orbits) // 2, -1]

    for plot_idx, orbit_idx in enumerate(selected_indices):
        ax = plt.subplot(3, 3, 7 + plot_idx)
        if orbit_idx < len(orbits):
            finder.plot_poincare_section(orbits[orbit_idx], ax=ax)

    # Overall title
    fig.suptitle(f'Static Equilibrium Orbit Analysis (Poincare Method)\n'
                 f'{design.name} ({design.species.name})',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    plot_file = output_dir / 'seo_poincare_analysis.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {plot_file}")

    plt.show()

    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)
    print(f"Results saved to {output_dir}/")
    print(f"  - SEO database: seo_database.pkl")
    print(f"  - Analysis plot: seo_poincare_analysis.png")


if __name__ == "__main__":
    main()
