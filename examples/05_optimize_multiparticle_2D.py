"""Multi-particle optimization: bunch phase and injection offset for a bunch, not just one particle.

Shows: AcceleratedOrbitFinder.optimize on a Gaussian bunch (100 particles, 0.4 mm / 10 km/s rms radial
spreads) in example 04's system, at example 04's RF frequency: Nelder-Mead over the bunch phase and the
bunch's radial offset r0, with the multi-particle cost (energy short of the target, orbit-centre offset,
turn-separation smoothness and the envelope oscillation std(sigma_r(t))), a CSV checkpoint of every
evaluation and its convergence plot, then the optimum's beam diagnostics (example 04's figure). The RF
frequency stays fixed: Nelder-Mead's first simplex step is 5 % of each start value, far outside an
isochronous cyclotron's frequency window.
Inputs: <resources>/midplane_field_0.5mm.comsol (COMSOL midplane export of a muon cyclotron, 0.5 mm grid).
Runtime: about 7 s per evaluation (12 turns, 100 particles); the default --maxiter 20 (about 35
evaluations) takes about 5 minutes, --maxiter 3 about a minute.
Outputs: <out>/multiparticle_optimization.csv, optimization_history.png,
multiparticle_optimization_result.png (+ _paths.png), multiparticle_optimization_result.pkl.

Usage:
    python examples/05_optimize_multiparticle_2D.py [--resources DIR] [--out DIR] [--maxiter N]
"""
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import _common
from PyCentralRegion.accelerated_orbit_finder import AcceleratedOrbitFinder, make_gaussian_beam

ex04 = _common.load_example('04_multiparticle_tracking_2D.py')


def plot_history(checkpoint_file, out_png):
    """Cost, final energy, envelope oscillation, bunch phase, RF frequency and final sigma_r per
    evaluation, from the optimizer's checkpoint CSV."""
    df = pd.read_csv(checkpoint_file)
    panels = [('cost', 'Cost', 'Optimization Convergence'),
              ('final_energy_mev', 'Final Energy (MeV)', 'Energy'),
              ('envelope_oscillation_mm', 'Envelope Oscillation (mm)', 'Beam Size Oscillation'),
              ('bunch_phase_deg', 'Bunch Phase (deg)', 'Phase Parameter'),
              ('rf_freq_mhz', 'RF Frequency (MHz)', 'Frequency Parameter'),
              ('final_std_r_mm', 'Final Radial Spread (mm)', 'Final Beam Size')]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (column, ylabel, title) in zip(axes.ravel(), panels):
        ax.plot(df['iteration'], df[column], 'o-' if column == 'cost' else 'o', ms=3, alpha=0.7)
        ax.set(xlabel='Evaluation', ylabel=ylabel, title=title)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved {out_png}")


def main():
    ap = _common.arguments(__doc__)
    ap.add_argument('--maxiter', type=int, default=20, help='Nelder-Mead iterations (default 20)')
    args = _common.parse(ap)
    design = ex04.build_design(_common.require(args.resources / ex04.FIELD_FILE),
                               'OptimizedMultiParticleCyclotron')

    n_particles, r_spread, vr_spread = 100, 0.4e-3, 1e4
    checkpoint_file = args.out / 'multiparticle_optimization.csv'
    finder = AcceleratedOrbitFinder(design, target_energy_mev=ex04.TARGET_ENERGY_MEV, max_radius_m=0.4,
                                    algorithm='rk4_rel', steps_per_turn=500, verbose=True,
                                    checkpoint_file=str(checkpoint_file))
    np.random.seed(0)               # PyPATools draws from numpy's global generator: same bunch every run
    beam = make_gaussian_beam(design.species, r_mean=ex04.INJECTION_RADIUS_MM * 1e-3,
                              v_tangential=ex04.V_TANGENTIAL, n_particles=n_particles,
                              r_spread=r_spread, vr_spread=vr_spread)

    # The finder's multi-particle cost terms: 'energy' per MeV short of the target, 'center' per m of
    # orbit-centre offset (mean over turns), 'smooth' on the dimensionless log-curvature of the turn
    # separation, 'spread' per m of envelope oscillation std(sigma_r(t)). Against the finder's defaults
    # (energy 5, smooth 100) energy is raised and smoothness lowered: with those, a bunch that stops
    # accelerating after two turns (no turn-separation term) costs less than one that reaches the target.
    weights = {'energy': 50.0, 'spread': 100.0, 'center': 1000.0, 'smooth': 10.0}
    print(f"\nOptimizing {n_particles} particles (sigma_r {r_spread * 1e3} mm, sigma_vr "
          f"{vr_spread / 1e3:.0f} km/s), weights {weights}, maxiter {args.maxiter}")
    result = finder.optimize(beam, max_turns=12, optimize_params=['bunch_phase', 'r0'],
                             method='nelder_mead', weights=weights, maxiter=args.maxiter)

    print(f"\nOptimum: bunch phase {result.bunch_phase_deg:.2f} deg, RF {result.rf_frequency_mhz:.6f} MHz, "
          f"r0 {result.initial_r_mm:.2f} mm")
    ex04.print_beam(result, n_particles)
    print(f"Envelope oscillation (std of sigma_r): {result.metadata['envelope_oscillation_mm']:.3f} mm; "
          f"final cost {result.cost:.3e}")

    plot_history(checkpoint_file, args.out / 'optimization_history.png')
    ex04.plot_beam(design, result,
                   f'Multi-Particle Optimization: {ex04.SPECIES}, {n_particles} particles, '
                   f'E_final={result.final_energy_mev:.2f} MeV, envelope '
                   f'{result.metadata["envelope_oscillation_mm"]:.2f} mm',
                   args.out / 'multiparticle_optimization_result.png')
    result_file = args.out / 'multiparticle_optimization_result.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved {result_file}")
    plt.show()


if __name__ == "__main__":
    main()
