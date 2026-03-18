"""
accelerated_orbit_finder.py - Unified Accelerated Orbit Optimizer

Single class handles both single-particle (n=1) and multi-particle optimization.
Uses TrackingEngine and centralized diagnostics.

Part of: PyCentralRegion module
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from scipy.optimize import differential_evolution, minimize
import time
import csv

from .tracking import TrackingEngine
from .diagnostics import (PoincareAnalyzer, calculate_turn_metrics,
                          BeamStatisticsCollector, TurnStatistics)
from PyPATools.particles import ParticleDistribution
from PyPATools.global_variables import CLIGHT


@dataclass
class RFCrossingData:
    """Single RF crossing data."""
    turn: int
    cavity_id: int
    particle_id: int
    energy_before_kev: float
    energy_after_kev: float
    energy_gain_kev: float
    phase_deg: float
    time: float


@dataclass
class OptimizedOrbit:
    """
    Result from acceleration optimization (single or multi-particle).

    For single particle: turn_statistics has 1 particle stats
    For multi-particle: turn_statistics has beam ensemble stats
    """
    success: bool
    final_energy_mev: float
    n_turns: int
    n_particles: int
    bunch_phase_deg: float
    rf_frequency_mhz: float
    initial_r_mm: float
    initial_vr_m_s: float
    trajectory_reference: np.ndarray  # Reference particle or centroid
    poincare_points_all: List[List]  # Per-particle Poincaré points
    rf_crossings: List[RFCrossingData]
    turn_statistics: List[TurnStatistics]  # Beam stats per turn
    turn_metrics: dict  # Reference trajectory metrics
    std_r_per_step: np.ndarray  # Radial spread at each step (multi-particle only)
    cost: float
    metadata: dict = field(default_factory=dict)


class AcceleratedOrbitFinder:
    """
    Unified optimizer for accelerated orbits (single or multi-particle).

    Automatically adapts cost function based on n_particles:
    - n=1: Optimize energy, centering, turn smoothness
    - n>1: Also optimize beam envelope and spread

    Parameters
    ----------
    design : CentralRegion
        Design with fields and RF cavities
    target_energy_mev : float
        Target final energy [MeV]
    n_particles : int
        Number of particles (default: 1 for single-particle)
    max_radius_m : float
        Maximum radius [m]
    algorithm : str
        Pusher algorithm
    steps_per_turn : int
        Steps per turn
    verbose : bool
        Print progress
    checkpoint_file : str, optional
        CSV checkpoint file
    """

    def __init__(self,
                 design,
                 target_energy_mev: float,
                 n_particles: int = 1,
                 max_radius_m: float = 0.4,
                 algorithm: str = 'rk4_rel',
                 steps_per_turn: int = 500,
                 verbose: bool = True,
                 checkpoint_file: Optional[str] = None):

        self.design = design
        self.target_energy_mev = target_energy_mev
        self.n_particles = n_particles
        self.r_max = max_radius_m
        self.algorithm = algorithm
        self.steps_per_turn = steps_per_turn
        self.verbose = verbose
        self.checkpoint_file = checkpoint_file

        # Validate
        if not design.is_valid(verbose=False):
            raise ValueError("Design must have bfield, species, and RF cavities")

        if len(design.rf_cavities) == 0:
            raise ValueError("Design must have at least one RF cavity")

        # Determine if this is multi-particle mode
        self.is_multiparticle = (n_particles > 1)

        # Create tracking engine
        self.engine = TrackingEngine(
            design,
            algorithm=algorithm,
            dimensionality='2D',
            use_rf=True,
            max_radius_m=max_radius_m,
            verbose=False
        )

        # Optimization tracking
        self.iteration = 0
        self.best_cost = np.inf
        self.best_params = None

        if self.checkpoint_file:
            self._init_checkpoint_file()

        if self.verbose:
            mode = "multi-particle" if self.is_multiparticle else "single-particle"
            print(f"Initialized AcceleratedOrbitFinder in {mode} mode (n={n_particles})")

    def _init_checkpoint_file(self):
        """Initialize CSV checkpoint file."""
        with open(self.checkpoint_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'iteration', 'bunch_phase_deg', 'rf_freq_mhz', 'r0_mm', 'vr0_m_s',
                'final_energy_mev', 'n_turns', 'cost', 'success', 'timestamp'
            ]

            # Add multi-particle specific columns
            if self.is_multiparticle:
                header.extend(['final_std_r_mm', 'envelope_oscillation_mm'])

            writer.writerow(header)

    def _write_checkpoint(self, params, cost, energy, n_turns, success,
                          std_r_mm=None, envelope_osc_mm=None):
        """Append to checkpoint file."""
        if not self.checkpoint_file:
            return

        with open(self.checkpoint_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                self.iteration,
                params[0],  # bunch_phase
                params[1],  # rf_freq
                params[2] * 1000 if len(params) > 2 else 0,
                params[3] if len(params) > 3 else 0,
                energy,
                n_turns,
                cost,
                success,
                time.time()
            ]

            if self.is_multiparticle:
                row.extend([std_r_mm or 0.0, envelope_osc_mm or 0.0])

            writer.writerow(row)

    def _estimate_timestep(self, frequency_hz: float) -> float:
        """Estimate timestep from RF frequency."""
        period = 1.0 / frequency_hz
        return period / self.steps_per_turn

    def create_initial_distribution(self,
                                    r_mean: float,
                                    v_tangential: float,
                                    v_perp: float = 0.0,
                                    r_spread: float = 0.0,
                                    vr_spread: float = 0.0) -> ParticleDistribution:
        """
        Create initial particle distribution.

        For single particle (n=1): spreads are ignored
        For multi-particle: uses Gaussian distribution

        Parameters
        ----------
        r_mean : float
            Mean radius [m]
        v_tangential : float
            Tangential velocity [m/s]
        v_perp : float
            Perpendicular velocity [m/s]
        r_spread : float
            Radial spread (1σ) [m]
        vr_spread : float
            Radial velocity spread (1σ) [m/s]
        """

        if self.n_particles == 1:
            # Single particle - no spread
            pd = ParticleDistribution(species=self.design.species)
            pd.x_vec = np.array([[r_mean, 0.0, 0.0]])

            p_tang = v_tangential / np.sqrt(CLIGHT ** 2 - v_tangential ** 2)
            p_perp = v_perp / np.sqrt(CLIGHT ** 2 - v_perp ** 2) if v_perp != 0 else 0.0
            pd.p_vec = np.array([[p_perp, p_tang, 0.0]])

        else:
            # Multi-particle - Gaussian distribution
            corr_matrix = np.eye(6)
            pd = ParticleDistribution.generate_distribution(
                self.design.species,
                type=['gaussian', 'gaussian', 'gaussian'],
                s_direction='z',
                n_particles=self.n_particles,
                correlation_matrix=corr_matrix,
                sigma_x=r_spread,
                sigma_px=vr_spread / np.sqrt(CLIGHT ** 2 - vr_spread ** 2),
                sigma_y=1e-20,
                sigma_py=1e-20,
                sigma_z=1e-20,
                sigma_pz=1e-20,
                cutoff_x=3,
                cutoff_px=3
            )

            pd.set_centroid(r_mean, 0.0, 0.0)
            pd.add_mean_momentum(
                v_perp / np.sqrt(CLIGHT ** 2 - v_perp ** 2) if v_perp != 0 else 0.0,
                v_tangential / np.sqrt(CLIGHT ** 2 - v_tangential ** 2),
                0.0
            )

        return pd

    def track_with_rf(self,
                      pd_init: ParticleDistribution,
                      dt: float,
                      max_turns: int,
                      save_full_beam: bool = False) -> Tuple:
        """
        Track particle(s) with RF and collect diagnostics.

        Works for both single-particle (n=1) and multi-particle (n>1).

        Returns
        -------
        success : bool
        turn_statistics : list of TurnStatistics
        rf_crossings : list
        trajectory_ref : np.ndarray
        poincare_all : list of lists
        std_r_per_step : np.ndarray
        turn_ids : list
        full_beam : np.ndarray or None
        """

        # Data collectors
        poincare_analyzers = [PoincareAnalyzer(section_angle=0.0)
                              for _ in range(self.n_particles)]
        beam_stats_collector = BeamStatisticsCollector(
            self.design.species,
            save_frequency=1
        )

        rf_crossings = []
        trajectory_storage = []
        std_r_storage = []
        turn_ids = []
        turn_counter = [0]
        energy_reached = [False]

        if save_full_beam:
            n_steps = max_turns * self.steps_per_turn
            full_beam = np.full((n_steps, self.n_particles, 6), np.nan)
        else:
            full_beam = None

        def callback(step, r_array, v_array, active, t):
            """Collect all diagnostics."""

            if not np.any(active):
                return False

            # Store reference trajectory (centroid or particle 0)
            if self.is_multiparticle:
                trajectory_storage.append(np.mean(r_array[active], axis=0))
            else:
                trajectory_storage.append(r_array[0].copy())

            # Calculate radial spread (multi-particle only)
            if self.is_multiparticle:
                radii = np.sqrt(r_array[active, 0] ** 2 + r_array[active, 1] ** 2)
                std_r_storage.append(np.std(radii))
            else:
                std_r_storage.append(0.0)

            # Save full beam state if requested
            if save_full_beam and full_beam is not None:
                full_beam[step, :, :3] = r_array
                full_beam[step, :, 3:] = v_array

            # Check Poincaré crossings for reference particle (particle 0)
            if active[0]:
                if step == 0:
                    r_prev = r_array[0]
                else:
                    r_prev = callback.r_prev

                crossed, t_frac = poincare_analyzers[0].check_crossing(r_prev, r_array[0])

                if crossed:
                    turn_ids.append(step)

                    # Record crossing for particle 0
                    r_cross = r_prev + t_frac * (r_array[0] - r_prev) if t_frac else r_array[0]
                    v_cross = v_array[0]

                    cav = self.design.rf_cavities[0]
                    phase_rad = np.fmod(cav.omega * t + cav.get_total_phase_rad(), 2.0 * np.pi)
                    phase_deg = np.rad2deg(phase_rad)

                    poincare_analyzers[0].record_crossing(
                        turn=turn_counter[0],
                        r=r_cross,
                        v=v_cross,
                        time=t,
                        species=self.design.species,
                        phase_deg=phase_deg
                    )

                    # Collect beam statistics at this crossing
                    beam_stats_collector.record(step, r_array[active], v_array[active], t)
                    beam_stats_collector.increment_turn()

                    turn_counter[0] += 1

                    # Check termination
                    if poincare_analyzers[0].crossings[-1].energy_mev >= self.target_energy_mev:
                        energy_reached[0] = True
                        if self.verbose:
                            print(f"    Reached target energy at turn {turn_counter[0]}")
                        return True

                    if turn_counter[0] >= max_turns:
                        if self.verbose:
                            print(f"    Reached max turns: {turn_counter[0]}")
                        return True

                callback.r_prev = r_array[0].copy()

            return False

        callback.r_prev = pd_init.x_vec[0].copy()

        # Track
        n_steps = max_turns * self.steps_per_turn

        try:
            result = self.engine.track_multiparticle(
                pd_init,
                dt=dt,
                n_steps=n_steps,
                callback=callback,
                callback_frequency=1,
                show_progress=False
            )
        except Exception as e:
            if self.verbose:
                print(f"    Tracking exception: {e}")
            return (False, [], [], np.array([]), [[] for _ in range(self.n_particles)],
                    np.array([]), [], None)

        # Collect results
        trajectory_ref = np.array(trajectory_storage) if trajectory_storage else np.array([])
        std_r_per_step = np.array(std_r_storage)
        turn_statistics = beam_stats_collector.get_statistics()

        # For single particle, also need to populate poincare_all with just particle 0
        poincare_all = [[p for p in poincare_analyzers[0].crossings]]

        success = result.success or energy_reached[0]

        return (success, turn_statistics, rf_crossings, trajectory_ref,
                poincare_all, std_r_per_step, turn_ids, full_beam)

    def objective_function(self, params, initial_seo, dt, max_turns,
                           r_spread, vr_spread, weights):
        """
        Unified objective function for single and multi-particle.

        Automatically adapts based on self.n_particles.
        """

        self.iteration += 1

        # Extract parameters
        bunch_phase_deg = params[0]
        rf_freq_hz = params[1]

        r_mean = params[2] if len(params) > 2 else initial_seo.r0[0]
        vr_mean = params[3] if len(params) > 3 else 0.0

        # Set RF parameters
        self.design.set_bunch_phase(bunch_phase_deg)
        self.design.set_rf_frequency(rf_freq_hz)

        # Create distribution
        v_tangential = np.linalg.norm(initial_seo.v0)

        try:
            pd_init = self.create_initial_distribution(
                r_mean=r_mean,
                v_tangential=v_tangential,
                v_perp=vr_mean,
                r_spread=r_spread,
                vr_spread=vr_spread
            )

        except Exception as e:
            if self.verbose:
                print(f"    Iter {self.iteration}: Distribution creation failed: {e}")
            self._write_checkpoint(params, 1e10, 0.0, 0, False)
            return 1e10

        # Track
        try:
            result = self.track_with_rf(pd_init, dt, max_turns, save_full_beam=False)
            (success, turn_stats, rf_cross, traj_ref, poincare_all,
             std_r_steps, turn_ids, _) = result
        except Exception as e:
            if self.verbose:
                print(f"    Iter {self.iteration}: Tracking failed: {e}")
            self._write_checkpoint(params, 1e10, 0.0, 0, False)
            return 1e10

        # Check for failure
        if not success or len(turn_stats) == 0:
            cost = 1e8
            if self.verbose:
                print(f"    Iter {self.iteration}: Failed, cost={cost:.2e}")
            self._write_checkpoint(params, cost, 0.0, 0, False)
            return cost

        # Calculate metrics
        metrics = calculate_turn_metrics(traj_ref, turn_ids)
        final_energy = turn_stats[-1].mean_energy_mev
        n_turns = len(turn_stats)

        # Cost function - adapts automatically
        cost = 0.0

        # Universal terms (single and multi-particle)
        w_energy = weights.get('energy', 5.0)
        cost -= w_energy * final_energy  # Maximize energy

        w_center = weights.get('center', 5000.0)
        if len(metrics['r_center']) > 0:
            cost += w_center * np.mean(metrics['r_center'])  # Minimize centering

        w_smooth = weights.get('smooth', 5000.0)
        if len(metrics['dr']) > 1:
            cost += w_smooth * np.std(metrics['dr']) ** 2  # Smooth turns

        # Multi-particle specific terms
        if self.is_multiparticle:
            # Envelope oscillation
            envelope_osc = np.std(std_r_steps)
            w_envelope = weights.get('spread', 100.0)
            cost += w_envelope * envelope_osc

            # Final beam spread
            final_std_r = turn_stats[-1].std_r
        else:
            envelope_osc = 0.0
            final_std_r = 0.0

        # Checkpoint
        self._write_checkpoint(
            params, cost, final_energy, n_turns, True,
            final_std_r * 1000, envelope_osc * 1000
        )

        # Track best
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_params = params.copy()
            msg = (f"    Iter {self.iteration}: NEW BEST - cost={cost:.2e}, "
                   f"E={final_energy:.3f} MeV, turns={n_turns}, "
                   f"phase={bunch_phase_deg:.1f}°, f={rf_freq_hz / 1e6:.3f} MHz")
            if self.is_multiparticle:
                msg += f", env_osc={envelope_osc * 1000:.2f} mm"
            if self.verbose:
                print(msg)
        else:
            if self.verbose and self.iteration % 10 == 0:
                print(f"    Iter {self.iteration}: cost={cost:.2e}, "
                      f"E={final_energy:.3f} MeV, turns={n_turns}")

        return cost

    def optimize(self,
                 initial_seo,
                 max_turns: int = 500,
                 r_spread_mm: float = 2.0,
                 vr_spread_m_s: float = 1e4,
                 optimize_params: List[str] = ['bunch_phase', 'rf_freq'],
                 method: str = 'differential_evolution',
                 bounds: Optional[dict] = None,
                 weights: Optional[dict] = None,
                 maxiter: int = 100) -> OptimizedOrbit:
        """
        Optimize RF parameters for acceleration.

        Works for both single-particle (n=1) and multi-particle (n>1).
        For single particle, spread parameters are ignored.

        Parameters
        ----------
        initial_seo : StaticOrbit
            Starting point
        max_turns : int
            Maximum turns
        r_spread_mm : float
            Initial radial spread [mm] (multi-particle only)
        vr_spread_m_s : float
            Radial velocity spread [m/s] (multi-particle only)
        optimize_params : list
            Parameters to optimize
        method : str
            Optimization method
        bounds : dict, optional
            Custom bounds
        weights : dict, optional
            Cost function weights
        maxiter : int
            Maximum iterations

        Returns
        -------
        result : OptimizedOrbit
        """

        if self.verbose:
            print("\n" + "=" * 70)
            mode_str = f"MULTI-PARTICLE ({self.n_particles})" if self.is_multiparticle else "SINGLE-PARTICLE"
            print(f"{mode_str} ACCELERATED ORBIT OPTIMIZATION")
            print("=" * 70)
            print(f"Target energy: {self.target_energy_mev} MeV")
            print(f"Initial energy: {initial_seo.energy_kev / 1000:.3f} MeV")
            print(f"Max turns: {max_turns}")
            if self.is_multiparticle:
                print(f"Particles: {self.n_particles}")
                print(f"Spreads: r={r_spread_mm} mm, vr={vr_spread_m_s / 1e3:.1f} km/s")
            print(f"Optimizing: {optimize_params}")
            print(f"Method: {method}")

        # Default weights
        if weights is None:
            if self.is_multiparticle:
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

        # Timestep
        dt = self._estimate_timestep(initial_seo.frequency_hz)

        if self.verbose:
            print(f"Timestep: {dt * 1e12:.2f} ps")
            print(f"\nCost function weights:")
            for key, val in weights.items():
                print(f"  {key}: {val}")

        # Setup parameter bounds
        if bounds is None:
            bounds = {}

        param_bounds = []
        param_names = []
        x0 = []

        if 'bunch_phase' in optimize_params:
            param_names.append('bunch_phase')
            param_bounds.append(bounds.get('bunch_phase', (-180, 180)))
            x0.append(20.0)

        if 'rf_freq' in optimize_params:
            param_names.append('rf_freq')
            f_seo = initial_seo.frequency_hz
            param_bounds.append(bounds.get('rf_freq', (f_seo * 0.95, f_seo * 1.05)))
            x0.append(f_seo)

        if 'r0' in optimize_params:
            param_names.append('r0')
            r_seo = initial_seo.r0[0]
            param_bounds.append(bounds.get('r0', (r_seo - 0.010, r_seo + 0.010)))
            x0.append(r_seo)

        if 'vr0' in optimize_params:
            param_names.append('vr0')
            param_bounds.append(bounds.get('vr0', (-5e5, 5e5)))
            x0.append(0.0)

        if self.verbose:
            print(f"\nOptimization parameters:")
            for name, bnd, x in zip(param_names, param_bounds, x0):
                print(f"  {name}: bounds={bnd}, initial={x}")
            print()

        # Reset
        self.iteration = 0
        self.best_cost = np.inf
        self.best_params = None

        # Convert spreads to SI
        r_spread = r_spread_mm / 1000.0 if self.is_multiparticle else 0.0
        vr_spread = vr_spread_m_s if self.is_multiparticle else 0.0

        # Optimize
        start_time = time.time()

        if method == 'differential_evolution':
            result = differential_evolution(
                self.objective_function,
                param_bounds,
                args=(initial_seo, dt, max_turns, r_spread, vr_spread, weights),
                maxiter=maxiter,
                workers=1,
                updating='deferred',
                disp=False
            )
            optimal_params = result.x
            final_cost = result.fun

        elif method == 'nelder_mead':
            result = minimize(
                self.objective_function,
                x0,
                args=(initial_seo, dt, max_turns, r_spread, vr_spread, weights),
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
            print(f"\nOptimal parameters:")
            for name, val in zip(param_names, optimal_params):
                if name == 'rf_freq':
                    print(f"  {name}: {val / 1e6:.6f} MHz")
                elif name == 'bunch_phase':
                    print(f"  {name}: {val:.2f} deg")
                elif name == 'r0':
                    print(f"  {name}: {val * 1000:.3f} mm")
                elif name == 'vr0':
                    print(f"  {name}: {val:.1f} m/s")

        # Final tracking with optimal parameters
        self.design.set_bunch_phase(optimal_params[0])
        self.design.set_rf_frequency(optimal_params[1])

        r_mean_final = optimal_params[2] if len(optimal_params) > 2 else initial_seo.r0[0]
        vr_mean_final = optimal_params[3] if len(optimal_params) > 3 else 0.0
        v_tangential = np.linalg.norm(initial_seo.v0)

        pd_init_final = self.create_initial_distribution(
            r_mean=r_mean_final,
            v_tangential=v_tangential,
            v_perp=vr_mean_final,
            r_spread=r_spread,
            vr_spread=vr_spread
        )

        if self.verbose:
            print("\nFinal tracking with optimal parameters...")

        result_tracking = self.track_with_rf(
            pd_init_final, dt, max_turns, save_full_beam=True
        )
        (success, turn_stats, rf_cross, traj_ref, poincare_all,
         std_r_steps, turn_ids, full_beam) = result_tracking

        metrics = calculate_turn_metrics(traj_ref, turn_ids)

        # Create result object
        optimized_orbit = OptimizedOrbit(
            success=success,
            final_energy_mev=turn_stats[-1].mean_energy_mev if len(turn_stats) > 0 else 0.0,
            n_turns=len(turn_stats),
            n_particles=self.n_particles,
            bunch_phase_deg=optimal_params[0],
            rf_frequency_mhz=optimal_params[1] / 1e6,
            initial_r_mm=r_mean_final * 1000,
            initial_vr_m_s=vr_mean_final,
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
                'n_particles': self.n_particles,
                'r_spread_mm': r_spread_mm if self.is_multiparticle else 0.0,
                'vr_spread_m_s': vr_spread_m_s if self.is_multiparticle else 0.0,
                'envelope_oscillation_mm': np.std(std_r_steps) * 1000 if self.is_multiparticle else 0.0,
                'full_beam': full_beam
            }
        )

        if self.verbose:
            print(f"\nFinal results:")
            print(f"  Final energy: {optimized_orbit.final_energy_mev:.3f} MeV")
            print(f"  Turns: {optimized_orbit.n_turns}")
            if self.is_multiparticle and len(turn_stats) > 0:
                print(f"  Final radial spread: {turn_stats[-1].std_r * 1000:.3f} mm")
                print(f"  Envelope oscillation: {np.std(std_r_steps) * 1000:.3f} mm")

        return optimized_orbit

    def track_once(self,
                   initial_r_mm: float,
                   initial_v_tangential_m_s: float,
                   bunch_phase_deg: float,
                   rf_freq_mhz: float,
                   max_turns: int = 500,
                   r_spread_mm: float = 2.0,
                   vr_spread_m_s: float = 1e4,
                   save_full_beam: bool = False) -> OptimizedOrbit:
        """
        Single tracking run without optimization (for testing).

        Works for both single-particle and multi-particle.

        Parameters
        ----------
        initial_r_mm : float
            Initial mean radius [mm]
        initial_v_tangential_m_s : float
            Initial tangential velocity [m/s]
        bunch_phase_deg : float
            Bunch phase [degrees]
        rf_freq_mhz : float
            RF frequency [MHz]
        max_turns : int
            Maximum turns
        r_spread_mm : float
            Radial spread [mm] (multi-particle only)
        vr_spread_m_s : float
            Radial velocity spread [m/s] (multi-particle only)
        save_full_beam : bool
            Save full beam trajectory

        Returns
        -------
        result : OptimizedOrbit
        """

        if self.verbose:
            print("\n" + "=" * 70)
            mode_str = f"MULTI-PARTICLE ({self.n_particles})" if self.is_multiparticle else "SINGLE-PARTICLE"
            print(f"{mode_str} TRACKING (SINGLE RUN)")
            print("=" * 70)
            print(f"Initial radius: {initial_r_mm:.2f} mm")
            print(f"Tangential velocity: {initial_v_tangential_m_s / 1e6:.2f} Mm/s")
            print(f"Bunch phase: {bunch_phase_deg:.2f} deg")
            print(f"RF frequency: {rf_freq_mhz:.6f} MHz")
            if self.is_multiparticle:
                print(f"Particles: {self.n_particles}")
                print(f"Spreads: r={r_spread_mm} mm, vr={vr_spread_m_s / 1e3:.1f} km/s")

        # Set RF parameters
        rf_freq_hz = rf_freq_mhz * 1e6
        self.design.set_bunch_phase(bunch_phase_deg)
        self.design.set_rf_frequency(rf_freq_hz)

        # Create distribution
        r_mean = initial_r_mm / 1000.0
        r_spread = r_spread_mm / 1000.0 if self.is_multiparticle else 0.0
        vr_spread = vr_spread_m_s if self.is_multiparticle else 0.0

        pd_init = self.create_initial_distribution(
            r_mean=r_mean,
            v_tangential=initial_v_tangential_m_s,
            v_perp=0.0,
            r_spread=r_spread,
            vr_spread=vr_spread
        )

        # Timestep
        dt = self._estimate_timestep(rf_freq_hz)

        # Track
        result = self.track_with_rf(pd_init, dt, max_turns, save_full_beam=save_full_beam)
        (success, turn_stats, rf_cross, traj_ref, poincare_all,
         std_r_steps, turn_ids, full_beam) = result

        metrics = calculate_turn_metrics(traj_ref, turn_ids)

        # Create result
        final_energy = turn_stats[-1].mean_energy_mev if len(turn_stats) > 0 else 0.0

        result_obj = OptimizedOrbit(
            success=success,
            final_energy_mev=final_energy,
            n_turns=len(turn_stats),
            n_particles=self.n_particles,
            bunch_phase_deg=bunch_phase_deg,
            rf_frequency_mhz=rf_freq_mhz,
            initial_r_mm=initial_r_mm,
            initial_vr_m_s=0.0,
            trajectory_reference=traj_ref,
            poincare_points_all=poincare_all,
            rf_crossings=rf_cross,
            turn_statistics=turn_stats,
            turn_metrics=metrics,
            std_r_per_step=std_r_steps,
            cost=0.0,
            metadata={
                'mode': 'single_run',
                'n_particles': self.n_particles,
                'r_spread_mm': r_spread_mm if self.is_multiparticle else 0.0,
                'vr_spread_m_s': vr_spread_m_s if self.is_multiparticle else 0.0,
                'initial_v_tangential_m_s': initial_v_tangential_m_s,
                'envelope_oscillation_mm': np.std(std_r_steps) * 1000 if self.is_multiparticle else 0.0,
                'full_beam': full_beam
            }
        )

        if self.verbose:
            print(f"\nResults:")
            print(f"  Final energy: {final_energy:.3f} MeV")
            print(f"  Turns completed: {len(turn_stats)}")
            if self.is_multiparticle and len(turn_stats) > 0:
                print(f"  Final radial spread: {turn_stats[-1].std_r * 1000:.3f} mm")
                print(f"  Envelope oscillation: {np.std(std_r_steps) * 1000:.3f} mm")

        return result_obj

    if __name__ == "__main__":
        print("accelerated_orbit_finder.py - Unified Accelerated Orbit Optimizer")
        print("=" * 70)
        print("Single class handles both single-particle and multi-particle optimization.")
        print("\nUsage:")
        print("  # Single particle")
        print("  finder = AcceleratedOrbitFinder(design, target_energy_mev=5.0, n_particles=1)")
        print("  result = finder.optimize(initial_seo)")
        print("\n  # Multi-particle")
        print("  finder = AcceleratedOrbitFinder(design, target_energy_mev=5.0, n_particles=100)")
        print("  result = finder.optimize(initial_seo, r_spread_mm=2.0, vr_spread_m_s=1e4)")