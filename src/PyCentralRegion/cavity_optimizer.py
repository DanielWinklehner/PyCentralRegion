"""
cavity_optimizer.py - RF Cavity Geometry Optimization

Optimizes cavity segment geometry (angles and radii) along with RF parameters
(bunch phase, frequency) for improved acceleration efficiency.

Part of: PyCentralRegion module
Dependencies: accelerated_orbit_finder, tracking

Usage:
    from cavity_optimizer import CavityGeometryOptimizer

    # Create orbit finder
    finder = AcceleratedOrbitFinder(design, target_energy_mev=5.0, n_particles=1)

    # Create geometry optimizer
    geo_optimizer = CavityGeometryOptimizer(
        orbit_finder=finder,
        n_segments=2,
        max_angle_variable=15.0,  # degrees
        max_r_variable=0.25,      # m
        r_min_cavity=0.05         # m
    )

    # Optimize geometry + RF parameters
    result = geo_optimizer.optimize(
        initial_seo=seo,
        rf_optimize_params=['bunch_phase', 'rf_freq'],
        maxiter=50
    )
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from scipy.optimize import differential_evolution, minimize
import time
import csv
from pathlib import Path

from .accelerated_orbit_finder import AcceleratedOrbitFinder, OptimizedOrbit


class CavityGeometryOptimizer:
    """
    Optimize RF cavity geometry along with RF parameters.

    Optimizes:
    - Angular excursions of variable segments [degrees]
    - Radial positions of segment nodes [m]
    - Bunch phase [degrees]
    - RF frequency [Hz]
    - Initial conditions (r0, vr0) if requested

    Parameters
    ----------
    orbit_finder : AcceleratedOrbitFinder
        Configured orbit finder (single or multi-particle)
    n_segments : int
        Number of variable segments per cavity
    max_angle_variable : float
        Maximum angular excursion [degrees]
    max_r_variable : float
        Maximum radial extent for variable segments [m]
    r_min_cavity : float
        Minimum cavity radius [m]
    verbose : bool
        Print progress
    checkpoint_file : str, optional
        CSV file for checkpointing
    """

    def __init__(self,
                 orbit_finder: AcceleratedOrbitFinder,
                 n_segments: int,
                 max_angle_variable: float,
                 max_r_variable: float,
                 r_min_cavity: float,
                 verbose: bool = True,
                 checkpoint_file: Optional[str] = None):

        self.orbit_finder = orbit_finder
        self.n_segments = n_segments
        self.max_angle = max_angle_variable
        self.max_r = max_r_variable
        self.r_min = r_min_cavity
        self.verbose = verbose
        self.checkpoint_file = checkpoint_file

        self.iteration = 0
        self.best_cost = np.inf
        self.best_geometry = None

        if self.checkpoint_file:
            self._init_checkpoint_file()

    def _init_checkpoint_file(self):
        """Initialize CSV checkpoint file with geometry columns."""
        with open(self.checkpoint_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Build header
            header = ['iteration']

            # Geometry parameters
            for i in range(self.n_segments):
                header.append(f'seg{i}_angle_deg')
            for i in range(self.n_segments):
                header.append(f'seg{i}_radius_mm')

            # RF parameters
            header.extend(['bunch_phase_deg', 'rf_freq_mhz', 'r0_mm', 'vr0_m_s'])

            # Results
            header.extend(['final_energy_mev', 'n_turns', 'cost', 'success', 'timestamp'])

            # Multi-particle specific
            if self.orbit_finder.is_multiparticle:
                header.extend(['final_std_r_mm', 'envelope_oscillation_mm'])

            writer.writerow(header)

    def _write_checkpoint(self, geometry_params, rf_params, cost, energy, n_turns, success,
                          std_r_mm=None, envelope_osc_mm=None):
        """Append iteration to checkpoint file."""
        if not self.checkpoint_file:
            return

        with open(self.checkpoint_file, 'a', newline='') as f:
            writer = csv.writer(f)

            row = [self.iteration]

            # Geometry (angles and radii)
            segment_angles, segment_radii = geometry_params
            row.extend(segment_angles)
            row.extend([r * 1000 for r in segment_radii])  # Convert to mm

            # RF parameters
            bunch_phase = rf_params[0]
            rf_freq = rf_params[1]
            r0 = rf_params[2] * 1000 if len(rf_params) > 2 else 0
            vr0 = rf_params[3] if len(rf_params) > 3 else 0

            row.extend([bunch_phase, rf_freq / 1e6, r0, vr0])

            # Results
            row.extend([energy, n_turns, cost, success, time.time()])

            # Multi-particle
            if self.orbit_finder.is_multiparticle:
                row.extend([std_r_mm or 0.0, envelope_osc_mm or 0.0])

            writer.writerow(row)

    def _unpack_params(self, params: np.ndarray, rf_param_names: List[str]) -> Tuple:
        """
        Unpack optimization parameters into geometry and RF components.

        Parameters
        ----------
        params : array
            [angle_0, ..., angle_N, r_0, ..., r_N, bunch_phase, rf_freq, ...]
        rf_param_names : list
            Names of RF parameters being optimized

        Returns
        -------
        segment_angles : list [degrees]
        segment_radii : list [m]
        rf_params : array
        """

        n_seg = self.n_segments

        # Geometry parameters
        segment_angles = params[:n_seg].tolist()
        segment_radii = params[n_seg:2 * n_seg].tolist()

        # RF parameters (rest of params array)
        rf_params = params[2 * n_seg:]

        return segment_angles, segment_radii, rf_params

    def _validate_geometry(self, segment_radii: List[float]) -> bool:
        """
        Check if geometry is valid (monotonic radii).

        Returns
        -------
        valid : bool
        """

        # Check monotonicity: r_min < r0 < r1 < ... < r_max
        r_max = self.orbit_finder.design.rf_cavities[0].r_max
        all_radii = [self.r_min] + segment_radii + [r_max]

        for i in range(len(all_radii) - 1):
            if all_radii[i] >= all_radii[i + 1]:
                return False

        return True

    def objective_function_with_geometry(self, params, initial_seo, dt, max_turns,
                                         r_spread, vr_spread, weights, rf_param_names):
        """
        Objective function that updates cavity geometry then tracks.

        Wraps AcceleratedOrbitFinder.objective_function with geometry updates.
        """

        self.iteration += 1

        # Unpack parameters
        segment_angles, segment_radii, rf_params = self._unpack_params(params, rf_param_names)

        # Validate geometry
        if not self._validate_geometry(segment_radii):
            if self.verbose:
                print(f"    Iter {self.iteration}: Invalid geometry (non-monotonic radii)")
            self._write_checkpoint(
                (segment_angles, segment_radii), rf_params,
                1e10, 0.0, 0, False
            )
            return 1e10

        # Update all cavity geometries
        try:
            for cavity in self.orbit_finder.design.rf_cavities:
                cavity.update_geometry(
                    segment_angles=segment_angles,
                    segment_radii=segment_radii
                )
        except Exception as e:
            if self.verbose:
                print(f"    Iter {self.iteration}: Geometry update failed: {e}")
            self._write_checkpoint(
                (segment_angles, segment_radii), rf_params,
                1e10, 0.0, 0, False
            )
            return 1e10

        # Call underlying orbit finder's objective function
        cost = self.orbit_finder.objective_function(
            rf_params, initial_seo, dt, max_turns,
            r_spread, vr_spread, weights
        )

        # Extract results for checkpoint
        # (Note: orbit_finder already writes its own checkpoint, but we want geometry too)
        # We'll extract from the last tracking result if available

        # Track best geometry
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_geometry = {
                'segment_angles': segment_angles.copy(),
                'segment_radii': segment_radii.copy(),
                'rf_params': rf_params.copy()
            }

            if self.verbose:
                print(f"    Iter {self.iteration}: NEW BEST GEOMETRY - cost={cost:.2e}")
                print(f"      Angles: {[f'{a:.2f}' for a in segment_angles]} deg")
                print(f"      Radii:  {[f'{r * 1000:.1f}' for r in segment_radii]} mm")

        return cost

    def optimize(self,
                 initial_seo,
                 max_turns: int = 500,
                 r_spread_mm: float = 2.0,
                 vr_spread_m_s: float = 1e4,
                 rf_optimize_params: List[str] = ['bunch_phase', 'rf_freq'],
                 rf_bounds: Optional[Dict] = None,
                 method: str = 'differential_evolution',
                 maxiter: int = 100,
                 weights: Optional[Dict] = None) -> OptimizedOrbit:
        """
        Optimize cavity geometry and RF parameters.

        Parameters
        ----------
        initial_seo : StaticOrbit
            Starting orbit
        max_turns : int
            Maximum turns to track
        r_spread_mm : float
            Radial spread [mm] (multi-particle only)
        vr_spread_m_s : float
            Radial velocity spread [m/s] (multi-particle only)
        rf_optimize_params : list
            RF parameters to optimize: 'bunch_phase', 'rf_freq', 'r0', 'vr0'
        rf_bounds : dict, optional
            Custom bounds for RF parameters
        method : str
            'differential_evolution' or 'nelder_mead'
        maxiter : int
            Maximum iterations
        weights : dict, optional
            Cost function weights

        Returns
        -------
        result : OptimizedOrbit
            Optimized orbit with best geometry
        """

        if self.verbose:
            print("\n" + "=" * 70)
            print("CAVITY GEOMETRY + RF PARAMETER OPTIMIZATION")
            print("=" * 70)
            print(f"Target energy: {self.orbit_finder.target_energy_mev} MeV")
            print(f"Particles: {self.orbit_finder.n_particles}")
            print(f"Variable segments: {self.n_segments}")
            print(f"RF parameters: {rf_optimize_params}")

        # Setup bounds
        if rf_bounds is None:
            rf_bounds = {}

        param_bounds = []
        param_names = []
        x0 = []

        # Geometry parameters
        # Angular excursions
        for i in range(self.n_segments):
            param_names.append(f'seg{i}_angle')
            param_bounds.append((-self.max_angle, self.max_angle))
            x0.append(0.0)  # Start with straight cavities

        # Radial positions
        r_max = self.orbit_finder.design.rf_cavities[0].r_max
        r_spacing = (self.max_r - self.r_min) / (self.n_segments + 1)

        for i in range(self.n_segments):
            param_names.append(f'seg{i}_radius')
            r_nominal = self.r_min + (i + 1) * r_spacing
            param_bounds.append((self.r_min + 0.01, min(self.max_r, r_max - 0.01)))
            x0.append(r_nominal)

        # RF parameters
        if 'bunch_phase' in rf_optimize_params:
            param_names.append('bunch_phase')
            param_bounds.append(rf_bounds.get('bunch_phase', (-180, 180)))
            x0.append(20.0)

        if 'rf_freq' in rf_optimize_params:
            param_names.append('rf_freq')
            f_seo = initial_seo.frequency_hz
            param_bounds.append(rf_bounds.get('rf_freq', (f_seo * 0.9, f_seo * 1.1)))
            x0.append(f_seo)

        if 'r0' in rf_optimize_params:
            param_names.append('r0')
            r_seo = initial_seo.r0[0]
            param_bounds.append(rf_bounds.get('r0', (r_seo - 0.010, r_seo + 0.010)))
            x0.append(r_seo)

        if 'vr0' in rf_optimize_params:
            param_names.append('vr0')
            param_bounds.append(rf_bounds.get('vr0', (-5e5, 5e5)))
            x0.append(0.0)

        if self.verbose:
            print(f"\nOptimization parameters ({len(param_names)} total):")
            for name, bnd, x in zip(param_names, param_bounds, x0):
                print(f"  {name}: bounds={bnd}, initial={x}")
            print()

        # Setup
        self.iteration = 0
        self.best_cost = np.inf
        self.best_geometry = None

        dt = self.orbit_finder._estimate_timestep(initial_seo.frequency_hz)
        r_spread = r_spread_mm / 1000.0 if self.orbit_finder.is_multiparticle else 0.0
        vr_spread = vr_spread_m_s if self.orbit_finder.is_multiparticle else 0.0

        # Default weights
        if weights is None:
            if self.orbit_finder.is_multiparticle:
                weights = {
                    'energy': 5.0,
                    'spread': 100.0,
                    'center': 1000.0,
                    'smooth': 1000.0
                }
            else:
                weights = {
                    'energy': 5.0,
                    'center': 1000.0,
                    'smooth': 1000.0
                }

        # Optimize
        start_time = time.time()

        if method == 'differential_evolution':
            result = differential_evolution(
                self.objective_function_with_geometry,
                param_bounds,
                args=(initial_seo, dt, max_turns, r_spread, vr_spread, weights, rf_optimize_params),
                maxiter=maxiter,
                workers=1,
                updating='deferred',
                disp=False
            )
            optimal_params = result.x
            final_cost = result.fun

        elif method == 'nelder_mead':
            result = minimize(
                self.objective_function_with_geometry,
                x0,
                args=(initial_seo, dt, max_turns, r_spread, vr_spread, weights, rf_optimize_params),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'disp': False}
            )
            optimal_params = result.x
            final_cost = result.fun

        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"OPTIMIZATION COMPLETE")
            print(f"{'=' * 70}")
            print(f"Time elapsed: {elapsed:.1f} s")
            print(f"Iterations: {self.iteration}")
            print(f"Final cost: {final_cost:.2e}")

        # Unpack optimal parameters
        segment_angles, segment_radii, rf_params = self._unpack_params(
            optimal_params, rf_optimize_params
        )

        if self.verbose:
            print(f"\nOptimal geometry:")
            for i, (ang, rad) in enumerate(zip(segment_angles, segment_radii)):
                print(f"  Segment {i}: angle={ang:.2f}°, radius={rad * 1000:.1f} mm")

            print(f"\nOptimal RF parameters:")
            for i, name in enumerate(rf_optimize_params):
                val = rf_params[i]
                if name == 'rf_freq':
                    print(f"  {name}: {val / 1e6:.6f} MHz")
                elif name == 'bunch_phase':
                    print(f"  {name}: {val:.2f} deg")
                elif name == 'r0':
                    print(f"  {name}: {val * 1000:.3f} mm")
                elif name == 'vr0':
                    print(f"  {name}: {val:.1f} m/s")

        # Final tracking with optimal geometry
        for cavity in self.orbit_finder.design.rf_cavities:
            cavity.update_geometry(
                segment_angles=segment_angles,
                segment_radii=segment_radii
            )

        # Set RF parameters
        self.orbit_finder.design.set_bunch_phase(rf_params[0])
        self.orbit_finder.design.set_rf_frequency(rf_params[1])

        # Create initial distribution
        r_mean = rf_params[2] if len(rf_params) > 2 else initial_seo.r0[0]
        vr_mean = rf_params[3] if len(rf_params) > 3 else 0.0
        v_tangential = np.linalg.norm(initial_seo.v0)

        pd_init_final = self.orbit_finder.create_initial_distribution(
            r_mean=r_mean,
            v_tangential=v_tangential,
            v_perp=vr_mean,
            r_spread=r_spread,
            vr_spread=vr_spread
        )

        if self.verbose:
            print("\nFinal tracking with optimal parameters...")

        # Track
        result_tracking = self.orbit_finder.track_with_rf(
            pd_init_final, dt, max_turns, save_full_beam=True
        )
        (success, turn_stats, rf_cross, traj_ref, poincare_all,
         std_r_steps, turn_ids, full_beam) = result_tracking

        from .diagnostics import calculate_turn_metrics
        metrics = calculate_turn_metrics(traj_ref, turn_ids)

        # Create result
        optimized_orbit = OptimizedOrbit(
            success=success,
            final_energy_mev=turn_stats[-1].mean_energy_mev if len(turn_stats) > 0 else 0.0,
            n_turns=len(turn_stats),
            n_particles=self.orbit_finder.n_particles,
            bunch_phase_deg=rf_params[0],
            rf_frequency_mhz=rf_params[1] / 1e6,
            initial_r_mm=r_mean * 1000,
            initial_vr_m_s=vr_mean,
            trajectory_reference=traj_ref,
            poincare_points_all=poincare_all,
            rf_crossings=rf_cross,
            turn_statistics=turn_stats,
            turn_metrics=metrics,
            std_r_per_step=std_r_steps,
            cost=final_cost,
            metadata={
                'initial_seo': initial_seo,
                'optimization_method': method,
                'optimization_time_s': elapsed,
                'total_iterations': self.iteration,
                'param_names': param_names,
                'param_bounds': param_bounds,
                'weights': weights,
                'n_particles': self.orbit_finder.n_particles,
                'r_spread_mm': r_spread_mm if self.orbit_finder.is_multiparticle else 0.0,
                'vr_spread_m_s': vr_spread_m_s if self.orbit_finder.is_multiparticle else 0.0,
                'envelope_oscillation_mm': np.std(std_r_steps) * 1000 if self.orbit_finder.is_multiparticle else 0.0,
                'full_beam': full_beam,
                'optimal_geometry': {
                    'segment_angles': segment_angles,
                    'segment_radii': segment_radii,
                    'n_segments': self.n_segments
                }
            }
        )

        if self.verbose:
            print(f"\nFinal results:")
            print(f"  Final energy: {optimized_orbit.final_energy_mev:.3f} MeV")
            print(f"  Turns: {optimized_orbit.n_turns}")
            if self.orbit_finder.is_multiparticle and len(turn_stats) > 0:
                print(f"  Final radial spread: {turn_stats[-1].std_r * 1000:.3f} mm")
                print(f"  Envelope oscillation: {np.std(std_r_steps) * 1000:.3f} mm")

        return optimized_orbit


if __name__ == "__main__":
    print("cavity_optimizer.py - RF Cavity Geometry Optimization")
    print("=" * 70)
    print("Optimizes cavity segment geometry + RF parameters for improved acceleration.")
    print("\nUsage:")
    print("  from cavity_optimizer import CavityGeometryOptimizer")
    print("  geo_opt = CavityGeometryOptimizer(finder, n_segments=2, ...)")
    print("  result = geo_opt.optimize(initial_seo)")