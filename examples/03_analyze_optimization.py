"""
03_analyze_optimization.py - Analyze Optimization Checkpoint Data

Post-processing tool to visualize and analyze optimization progress
from checkpoint CSV files generated during accelerated orbit optimization.

Usage:
    python 03_analyze_optimization.py [checkpoint_file.csv]

If no file specified, looks for 'output/acceleration_optimization.csv'
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def load_checkpoint(filename):
    """Load checkpoint CSV file."""
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} iterations from {filename}")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading file: {e}")
        sys.exit(1)


def analyze_optimization(df):
    """Analyze optimization data and print statistics."""

    print("\n" + "=" * 70)
    print("OPTIMIZATION ANALYSIS")
    print("=" * 70)

    # Overall statistics
    total_iterations = len(df)
    successful_runs = df[df['success'] == True]
    failed_runs = df[df['success'] == False]

    print(f"\nTotal iterations: {total_iterations}")
    print(f"Successful runs: {len(successful_runs)} ({len(successful_runs) / total_iterations * 100:.1f}%)")
    print(f"Failed runs: {len(failed_runs)} ({len(failed_runs) / total_iterations * 100:.1f}%)")

    if len(successful_runs) > 0:
        # Best solution
        best_idx = successful_runs['cost'].idxmin()
        best_run = successful_runs.loc[best_idx]

        print(best_run.keys())

        print(f"\n--- BEST SOLUTION ---")
        print(f"Iteration: {int(best_run['iteration'])}")
        print(f"Cost: {best_run['cost']:.2e}")
        print(f"Final energy: {best_run['final_energy_mev']:.3f} MeV")
        print(f"Turns: {int(best_run['n_turns'])}")
        print(f"Bunch phase: {best_run['bunch_phase_deg']:.2f} degrees")
        print(f"RF frequency: {best_run['rf_freq_mhz']:.6f} MHz")

        # Parameter ranges explored
        print(f"\n--- PARAMETER RANGES EXPLORED ---")
        print(f"Bunch phase: [{successful_runs['bunch_phase_deg'].min():.1f}, "
              f"{successful_runs['bunch_phase_deg'].max():.1f}] degrees")
        print(f"RF frequency: [{successful_runs['rf_freq_mhz'].min():.6f}, "
              f"{successful_runs['rf_freq_mhz'].max():.6f}] MHz")

        # Energy statistics
        print(f"\n--- ENERGY STATISTICS ---")
        print(f"Mean final energy: {successful_runs['final_energy_mev'].mean():.3f} MeV")
        print(f"Std dev: {successful_runs['final_energy_mev'].std():.3f} MeV")
        print(f"Min: {successful_runs['final_energy_mev'].min():.3f} MeV")
        print(f"Max: {successful_runs['final_energy_mev'].max():.3f} MeV")

        # Turn statistics
        print(f"\n--- TURN STATISTICS ---")
        print(f"Mean turns: {successful_runs['n_turns'].mean():.1f}")
        print(f"Min turns: {int(successful_runs['n_turns'].min())}")
        print(f"Max turns: {int(successful_runs['n_turns'].max())}")

        # Convergence analysis
        cost_values = successful_runs['cost'].values
        if len(cost_values) > 10:
            # Find when cost dropped below certain thresholds
            min_cost = cost_values.min()
            thresholds = [min_cost * 10, min_cost * 2, min_cost * 1.1]

            print(f"\n--- CONVERGENCE ANALYSIS ---")
            print(f"Best cost: {min_cost:.2e}")

            for thresh in thresholds:
                below_thresh = successful_runs[successful_runs['cost'] <= thresh]
                if len(below_thresh) > 0:
                    first_iter = below_thresh['iteration'].min()
                    print(f"  Cost < {thresh:.2e}: iteration {int(first_iter)}")

    else:
        print("\nWARNING: No successful runs found!")

    return successful_runs if len(successful_runs) > 0 else None


def plot_optimization_progress(df, successful_runs, output_dir):
    """Create comprehensive visualization of optimization progress."""

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Cost vs Iteration
    ax1 = plt.subplot(3, 3, 1)

    if successful_runs is not None and len(successful_runs) > 0:
        ax1.semilogy(successful_runs['iteration'], successful_runs['cost'],
                     'o', markersize=3, alpha=0.5, label='Successful')

    failed = df[df['success'] == False]
    if len(failed) > 0:
        ax1.semilogy(failed['iteration'], failed['cost'],
                     'rx', markersize=4, alpha=0.5, label='Failed')

    # Running minimum
    if successful_runs is not None and len(successful_runs) > 0:
        cummin = successful_runs.sort_values('iteration')['cost'].cummin()
        ax1.semilogy(successful_runs.sort_values('iteration')['iteration'],
                     cummin, 'b-', linewidth=2, label='Best so far')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    ax1.set_title('Optimization Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final Energy vs Iteration
    ax2 = plt.subplot(3, 3, 2)

    if successful_runs is not None and len(successful_runs) > 0:
        scatter = ax2.scatter(successful_runs['iteration'],
                              successful_runs['final_energy_mev'],
                              c=successful_runs['cost'], cmap='viridis',
                              s=20, alpha=0.6)
        plt.colorbar(scatter, ax=ax2, label='Cost')

        # Target line if identifiable
        target_energy = successful_runs['final_energy_mev'].max()
        ax2.axhline(y=target_energy, color='r', linestyle='--',
                    alpha=0.5, label=f'Target: {target_energy:.1f} MeV')

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Final Energy (MeV)')
    ax2.set_title('Energy Achievement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Number of Turns vs Iteration
    ax3 = plt.subplot(3, 3, 3)

    if successful_runs is not None and len(successful_runs) > 0:
        scatter = ax3.scatter(successful_runs['iteration'],
                              successful_runs['n_turns'],
                              c=successful_runs['cost'], cmap='viridis',
                              s=20, alpha=0.6)
        plt.colorbar(scatter, ax=ax3, label='Cost')

    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Number of Turns')
    ax3.set_title('Turns Completed')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Bunch Phase Parameter Space
    ax4 = plt.subplot(3, 3, 4)

    if successful_runs is not None and len(successful_runs) > 0:
        scatter = ax4.scatter(successful_runs['bunch_phase_deg'],
                              successful_runs['final_energy_mev'],
                              c=successful_runs['cost'], cmap='viridis',
                              s=30, alpha=0.7)
        plt.colorbar(scatter, ax=ax4, label='Cost')

        # Mark best solution
        best_idx = successful_runs['cost'].idxmin()
        best = successful_runs.loc[best_idx]
        ax4.plot(best['bunch_phase_deg'], best['final_energy_mev'],
                 'r*', markersize=20, label='Best', zorder=10)

    ax4.set_xlabel('Bunch Phase (degrees)')
    ax4.set_ylabel('Final Energy (MeV)')
    ax4.set_title('Bunch Phase Parameter Space')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: RF Frequency Parameter Space
    ax5 = plt.subplot(3, 3, 5)

    if successful_runs is not None and len(successful_runs) > 0:
        scatter = ax5.scatter(successful_runs['rf_freq_mhz'],
                              successful_runs['final_energy_mev'],
                              c=successful_runs['cost'], cmap='viridis',
                              s=30, alpha=0.7)
        plt.colorbar(scatter, ax=ax5, label='Cost')

        # Mark best
        ax5.plot(best['rf_freq_mhz'], best['final_energy_mev'],
                 'r*', markersize=20, label='Best', zorder=10)

    ax5.set_xlabel('RF Frequency (MHz)')
    ax5.set_ylabel('Final Energy (MeV)')
    ax5.set_title('RF Frequency Parameter Space')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: 2D Parameter Space (Bunch Phase vs RF Freq)
    ax6 = plt.subplot(3, 3, 6)

    if successful_runs is not None and len(successful_runs) > 0:
        scatter = ax6.scatter(successful_runs['bunch_phase_deg'],
                              successful_runs['rf_freq_mhz'],
                              c=successful_runs['cost'], cmap='viridis',
                              s=30, alpha=0.7)
        plt.colorbar(scatter, ax=ax6, label='Cost')

        # Mark best
        ax6.plot(best['bunch_phase_deg'], best['rf_freq_mhz'],
                 'r*', markersize=20, label='Best', zorder=10)

    ax6.set_xlabel('Bunch Phase (degrees)')
    ax6.set_ylabel('RF Frequency (MHz)')
    ax6.set_title('2D Parameter Space')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Plot 7: Cost Distribution
    ax7 = plt.subplot(3, 3, 7)

    if successful_runs is not None and len(successful_runs) > 0:
        costs = successful_runs['cost'].values
        ax7.hist(np.log10(costs), bins=30, alpha=0.7, edgecolor='black')
        ax7.axvline(x=np.log10(costs.min()), color='r', linestyle='--',
                    linewidth=2, label=f'Best: {costs.min():.2e}')

    ax7.set_xlabel('log10(Cost)')
    ax7.set_ylabel('Count')
    ax7.set_title('Cost Distribution')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Plot 8: Energy Distribution
    ax8 = plt.subplot(3, 3, 8)

    if successful_runs is not None and len(successful_runs) > 0:
        energies = successful_runs['final_energy_mev'].values
        ax8.hist(energies, bins=30, alpha=0.7, edgecolor='black')
        ax8.axvline(x=energies[successful_runs['cost'].idxmin()],
                    color='r', linestyle='--', linewidth=2, label='Best solution')

    ax8.set_xlabel('Final Energy (MeV)')
    ax8.set_ylabel('Count')
    ax8.set_title('Final Energy Distribution')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Plot 9: Success Rate vs Time
    ax9 = plt.subplot(3, 3, 9)

    # Rolling success rate
    window = max(10, len(df) // 20)
    df_sorted = df.sort_values('iteration')
    rolling_success = df_sorted['success'].rolling(window=window, center=True).mean() * 100

    ax9.plot(df_sorted['iteration'], rolling_success, linewidth=2)
    ax9.set_xlabel('Iteration')
    ax9.set_ylabel('Success Rate (%)')
    ax9.set_title(f'Rolling Success Rate (window={window})')
    ax9.grid(True, alpha=0.3)
    ax9.set_ylim([0, 105])

    # Overall title
    if successful_runs is not None and len(successful_runs) > 0:
        fig.suptitle(f'Optimization Analysis: {len(df)} iterations, '
                     f'Best cost={successful_runs["cost"].min():.2e}, '
                     f'E_final={best["final_energy_mev"]:.2f} MeV',
                     fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f'Optimization Analysis: {len(df)} iterations (No successful runs)',
                     fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    plot_file = output_dir / 'optimization_analysis.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Saved analysis plot to {plot_file}")

    plt.show()


def export_best_solutions(successful_runs, output_dir, n_best=10):
    """Export top N solutions to CSV."""

    if successful_runs is None or len(successful_runs) == 0:
        print("No successful runs to export")
        return

    # Sort by cost
    best_solutions = successful_runs.nsmallest(n_best, 'cost')

    # Export
    export_file = output_dir / 'best_solutions.csv'
    best_solutions.to_csv(export_file, index=False)

    print(f"\nExported top {len(best_solutions)} solutions to {export_file}")

    # Print summary table
    print(f"\n{'Rank':<6} {'Cost':<12} {'Energy (MeV)':<12} {'Turns':<8} "
          f"{'Phase (deg)':<12} {'Freq (MHz)':<12}")
    print("-" * 70)

    for i, (idx, row) in enumerate(best_solutions.iterrows(), 1):
        print(f"{i:<6} {row['cost']:<12.2e} {row['final_energy_mev']:<12.3f} "
              f"{int(row['n_turns']):<8} {row['bunch_phase_deg']:<12.2f} "
              f"{row['rf_freq_mhz']:<12.6f}")


def main():
    print("=" * 70)
    print("OPTIMIZATION CHECKPOINT ANALYSIS")
    print("=" * 70)

    # Get checkpoint file
    if len(sys.argv) > 1:
        checkpoint_file = Path(sys.argv[1])
    else:
        checkpoint_file = Path(__file__).parent.parent / 'output' / 'acceleration_optimization.csv'

    if not checkpoint_file.exists():
        print(f"\nERROR: Checkpoint file not found: {checkpoint_file}")
        print("\nUsage: python 03_analyze_optimization.py [checkpoint_file.csv]")
        print("   or: Run after 02_optimize_acceleration.py to use default location")
        sys.exit(1)

    print(f"\nAnalyzing: {checkpoint_file}")

    # Load data
    df = load_checkpoint(checkpoint_file)

    # Analyze
    successful_runs = analyze_optimization(df)

    # Output directory
    output_dir = checkpoint_file.parent

    # Export best solutions
    if successful_runs is not None:
        export_best_solutions(successful_runs, output_dir, n_best=10)

    # Plot
    plot_optimization_progress(df, successful_runs, output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
