"""Multi-particle tracking of a Gaussian bunch through the thin-gap RF system of example 02.

Shows: make_gaussian_beam (100 particles, 2 mm / 10 km/s rms radial spreads) and one
AcceleratedOrbitFinder.track_once run at RF parameters of the kind example 02 finds (bunch phase, RF
frequency, injection radius); per-turn beam statistics (mean and rms radius and energy, energy gain), the
bunch in (r, vr) after the first step and at the end, the final energy distribution, and every particle's
path (save_full_beam -> result.metadata['full_beam']). Example 05 reuses build_design and plot_beam.
Inputs: <resources>/midplane_field_0.5mm.comsol (COMSOL midplane export of a muon cyclotron, 0.5 mm grid).
Runtime: about 15 s.
Outputs: <out>/multiparticle_tracking.png, multiparticle_tracking_paths.png, multiparticle_result.pkl.

Usage:
    python examples/04_multiparticle_tracking_2D.py [--resources DIR] [--out DIR]
"""
import pickle

import numpy as np
import matplotlib.pyplot as plt
from PyPATools.field import Field

import _common
from PyCentralRegion.central_region import CentralRegion
from PyCentralRegion.rf_cavity import create_four_cavity_system
from PyCentralRegion.accelerated_orbit_finder import AcceleratedOrbitFinder, make_gaussian_beam

FIELD_FILE = 'midplane_field_0.5mm.comsol'
SPECIES = 'muon'
TARGET_ENERGY_MEV = 5.0
# RF parameters and injection point of a single-particle optimum of example 02 (bunch phase, RF
# frequency, injection radius; the speed is that of the 100 mm SEO)
INJECTION_RADIUS_MM = 100.18
RF_FREQUENCY = 42.598633e6          # Hz, revolution frequency (the gaps run at h = 4)
BUNCH_PHASE_DEG = 20.40
V_TANGENTIAL = 26365144.5           # m/s
CAVITY_VOLTAGE = 60e3               # V per gap


def build_design(field_path, name):
    """Muon design on the midplane map with example 02's RF system: four double-gap thin-gap cavities
    (8 radial gaps from 50 to 400 mm, first gaps at 22.5 + k 90 deg, h = 4) at RF_FREQUENCY."""
    design = CentralRegion(name=name, dimensionality='2D')
    design.set_species(SPECIES)
    print(f"Species: {design.species.name}, {design.species.mass_mev:.3f} MeV/c^2, "
          f"charge {design.species.q}e; target {TARGET_ENERGY_MEV} MeV")
    print(f"Loading the magnetic field from {field_path.name}...")
    design.set_magnetic_field(field_path, interpolator_backend='fast')
    design.set_electric_field(Field.zero())
    design.clear_rf_cavities()
    for cav in create_four_cavity_system(r_min=0.05, r_max=0.40, angles=[22.5, 112.5, 202.5, 292.5],
                                         voltage=CAVITY_VOLTAGE, frequency=RF_FREQUENCY,
                                         phases=[0.0] * 4, harmonic=4):
        design.add_rf_cavity(cav)
    print(f"{len(design.rf_cavities)} RF gaps of {CAVITY_VOLTAGE / 1e3:.0f} kV, "
          f"RF {RF_FREQUENCY / 1e6:.6f} MHz x 4")
    return design


def print_beam(result, n_launched):
    """Per-turn beam statistics of a tracking result: first and last turn, survivors, gain per turn."""
    stats = result.turn_statistics
    print(f"\nSuccess: {result.success}; final energy {result.final_energy_mev:.3f} MeV "
          f"(target {TARGET_ENERGY_MEV} MeV) after {result.n_turns} turns")
    if not stats:
        return
    for label, s in (('Initial', stats[0]), ('Final', stats[-1])):
        print(f"{label} beam: <r> {s.mean_r * 1e3:.2f} mm, sigma_r {s.std_r * 1e3:.3f} mm, "
              f"<E> {s.mean_energy_mev:.3f} MeV, sigma_E {s.std_energy_mev * 1e3:.2f} keV")
    print(f"Survivors: {stats[-1].n_active} of {n_launched}; mean energy gain "
          f"{1e3 * (stats[-1].mean_energy_mev - stats[0].mean_energy_mev) / max(len(stats) - 1, 1):.0f}"
          f" keV per turn")


def plot_beam(design, result, title, out_png):
    """12-panel figure of a multi-particle result (out_png) and every particle's path over the field
    map (<out_png stem>_paths.png). Bunch phase space comes from result.metadata['full_beam']."""
    stats = result.turn_statistics
    turns = [s.turn for s in stats]
    energies = np.array([s.mean_energy_mev for s in stats])
    full_beam = result.metadata['full_beam']           # (steps, particles, x y z vx vy vz)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    ax = axes.ravel()

    # 1: reference trajectory over |B| and the gaps
    xy = np.linspace(-0.4, 0.4, 150)
    X, Y = np.meshgrid(xy, xy, indexing='ij')
    b = design.bfield(np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]))
    bmag = np.linalg.norm(b, axis=1).reshape(X.shape)
    ax[0].contourf(X, Y, bmag, levels=20, cmap='viridis', alpha=0.3)
    for cav in design.rf_cavities:
        ax[0].plot([cav.p1[0], cav.p2[0]], [cav.p1[1], cav.p2[1]], 'r-', lw=3, alpha=0.7, zorder=5)
    traj = result.trajectory_reference
    ax[0].plot(traj[:, 0], traj[:, 1], 'b-', lw=1, alpha=0.8, label='Reference')
    ax[0].plot(traj[0, 0], traj[0, 1], 'go', ms=10, label='Start', zorder=10)
    ax[0].plot(traj[-1, 0], traj[-1, 1], 'rs', ms=10, label='End', zorder=10)
    ax[0].set(xlabel='x (m)', ylabel='y (m)', title='Reference Particle Trajectory', aspect='equal')
    ax[0].legend(fontsize=8)

    # 2-4, 8, 9, 12: per-turn statistics of the live particles
    ax[1].errorbar(turns, energies, yerr=[s.std_energy_mev for s in stats], fmt='o-', capsize=3,
                   label='Mean +- sigma')
    ax[1].axhline(TARGET_ENERGY_MEV, color='r', ls='--', label=f'Target: {TARGET_ENERGY_MEV} MeV')
    ax[1].set(xlabel='Turn Number', ylabel='Energy (MeV)', title='Beam Energy Evolution')
    ax[1].legend()
    ax[2].plot(turns, [s.mean_r * 1e3 for s in stats], 'o-', color='green')
    ax[2].set(xlabel='Turn Number', ylabel='Mean Radius (mm)', title='Mean Orbit Radius')
    ax[3].plot(turns, [s.std_r * 1e3 for s in stats], 'o-', color='purple')
    ax[3].set(xlabel='Turn Number', ylabel='sigma_r (mm)', title='Beam Size Evolution')
    ax[7].plot(turns, [s.std_energy_mev * 1e3 for s in stats], 'o-', color='orange')
    ax[7].set(xlabel='Turn Number', ylabel='sigma_E (keV)', title='Energy Spread Evolution')
    ax[8].plot(turns[1:], 1e3 * np.diff(energies), 'o-', color='teal')
    ax[8].set(xlabel='Turn Number', ylabel='Energy Gain (keV / turn)', title='Mean Energy Gain per Turn')
    ax[11].plot(turns, [s.std_r / s.mean_r * 100 for s in stats], 'o-', color='brown')
    ax[11].set(xlabel='Turn Number', ylabel='Relative Spread (%)', title='Beam Quality (sigma_r / <r>)')

    # 5, 6, 10, 11: the bunch after the first step and at the last step
    r0, vr0, _ = _common.bunch_snapshot(full_beam, 0, design.species)
    r1, vr1, e1 = _common.bunch_snapshot(full_beam, -1, design.species)
    ax[4].plot(r0, vr0, 'o', ms=3, alpha=0.5, color='blue')
    ax[4].set(xlabel='Radius (mm)', ylabel='Radial Velocity (m/s)',
              title=f'Bunch after the First Step ({len(r0)} particles)')
    ax[5].plot(r1, vr1, 'o', ms=3, alpha=0.5, color='red')
    ax[5].set(xlabel='Radius (mm)', ylabel='Radial Velocity (m/s)',
              title=f'Bunch at the End ({len(r1)} tracked)')
    ax[9].hist(e1, bins=30, alpha=0.7, edgecolor='black')
    ax[9].set(xlabel='Energy (MeV)', ylabel='Count', title='Energy Distribution at the End')
    ax[10].plot(e1, r1, 'o', ms=3, alpha=0.5)
    ax[10].set(xlabel='Energy (MeV)', ylabel='Radius (mm)', title='Radius vs Energy at the End')

    # 7: reference particle at the Poincare section, turn by turn
    ref = result.poincare_points_all[0]
    if ref:
        sc = ax[6].scatter([p.r * 1e3 for p in ref], [p.vr for p in ref], c=[p.turn for p in ref],
                           cmap='viridis', s=20)
        fig.colorbar(sc, ax=ax[6], label='Turn')
    ax[6].set(xlabel='Radius (mm)', ylabel='Radial Velocity (m/s)', title='Reference Particle Phase Space')

    for a in ax:
        a.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved {out_png}")

    # every particle's path; a particle stays where it was lost or left the tracking radius
    fig_paths, axp = plt.subplots(figsize=(12, 12))
    axp.contourf(X, Y, bmag, levels=20, cmap='viridis', alpha=0.3)
    for i in range(full_beam.shape[1]):
        axp.plot(full_beam[:, i, 0], full_beam[:, i, 1], color='blue', lw=0.3)
    axp.set(xlabel='x (m)', ylabel='y (m)', title=f'{full_beam.shape[1]} particle paths', aspect='equal')
    paths_png = out_png.with_name(out_png.stem + '_paths.png')
    fig_paths.savefig(paths_png, dpi=120, bbox_inches='tight')
    print(f"Saved {paths_png}")


def main():
    args = _common.parse(_common.arguments(__doc__))
    design = build_design(_common.require(args.resources / FIELD_FILE), 'MultiParticleCyclotron')
    n_particles = 100
    finder = AcceleratedOrbitFinder(design, target_energy_mev=TARGET_ENERGY_MEV, max_radius_m=0.35,
                                    algorithm='rk4_rel', steps_per_turn=500, verbose=True)

    # Gaussian bunch on the +x axis; the finder prepends the bunch centroid as the
    # reference particle (reference_centroid) and keeps it out of the beam statistics
    print(f"\nTracking {n_particles} particles: bunch phase {BUNCH_PHASE_DEG:.2f} deg, "
          f"r0 {INJECTION_RADIUS_MM:.2f} mm, v_t {V_TANGENTIAL / 1e6:.2f} Mm/s, "
          f"sigma_r 2 mm, sigma_vr 10 km/s")
    np.random.seed(0)               # PyPATools draws from numpy's global generator: same bunch every run
    beam = make_gaussian_beam(design.species, r_mean=INJECTION_RADIUS_MM * 1e-3,
                              v_tangential=V_TANGENTIAL, n_particles=n_particles,
                              r_spread=2.0e-3, vr_spread=1e4)
    result = finder.track_once(beam, bunch_phase_deg=BUNCH_PHASE_DEG,
                               rf_freq_mhz=RF_FREQUENCY / 1e6, max_turns=11, save_full_beam=True)
    print_beam(result, n_particles)

    plot_beam(design, result, f'Multi-Particle Beam Tracking: {SPECIES}, {n_particles} particles, '
              f'E_final={result.final_energy_mev:.2f} MeV', args.out / 'multiparticle_tracking.png')
    result_file = args.out / 'multiparticle_result.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved {result_file}")
    plt.show()


if __name__ == "__main__":
    main()
