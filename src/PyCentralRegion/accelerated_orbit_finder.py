"""
accelerated_orbit_finder.py - Accelerated Orbit Optimization

Finds optimal RF parameters for acceleration in cyclotrons.
Optimizes bunch phase, RF frequency, and optionally initial conditions.

Part of: PyCentralRegion module
Dependencies: PyPATools, scipy, numpy

Usage:
    from accelerated_orbit_finder import AcceleratedOrbitFinder

    finder = AcceleratedOrbitFinder(design, target_energy_mev=5.0)
    result = finder.optimize(initial_seo, max_turns=500)
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from scipy.optimize import differential_evolution, minimize
from PyPATools.pusher import Pusher
from PyPATools.field import Field
import warnings
import csv
from pathlib import Path
import time
import matplotlib.pyplot as plt


@dataclass
class PoincarePoint:
    """Single Poincaré crossing during acceleration."""
    turn: int
    r: float  # m
    vr: float  # m/s
    energy_mev: float
    phase_deg: float  # RF phase at crossing
    time: float  # s


@dataclass
class RFCrossingData:
    """Data from single RF cavity crossing."""
    turn: int
    cavity_id: int
    energy_before_kev: float
    energy_after_kev: float
    energy_gain_kev: float
    phase_deg: float
    time: float


@dataclass
class OptimizedOrbit:
    """
    Result from accelerated orbit optimization.

    Attributes
    ----------
    success : bool
        Whether optimization succeeded
    final_energy_mev : float
        Final particle energy [MeV]
    n_turns : int
        Number of turns completed
    bunch_phase_deg : float
        Optimal bunch phase [degrees]
    rf_frequency_mhz : float
        Optimal RF frequency [MHz]
    initial_r_mm : float
        Initial radius [mm]
    initial_vr_m_s : float
        Initial radial velocity [m/s]
    trajectory : np.ndarray
        Full trajectory (N x 3) [m]
    poincare_points : list
        List of PoincarePoint objects
    rf_crossings : list
        List of RFCrossingData objects
    turn_metrics : dict
        Turn-by-turn analysis
    cost : float
        Final objective function value
    metadata : dict
        Additional information
    """
    success: bool
    final_energy_mev: float
    n_turns: int
    bunch_phase_deg: float
    rf_frequency_mhz: float
    initial_r_mm: float
    initial_vr_m_s: float
    initial_vtheta_m_s: float
    trajectory: np.ndarray
    poincare_points: List[PoincarePoint]
    rf_crossings: List[RFCrossingData]
    turn_metrics: dict
    cost: float
    metadata: dict = field(default_factory=dict)


class AcceleratedOrbitFinder:
    """
    Optimizer for accelerated cyclotron orbits.

    Parameters
    ----------
    design : CentralRegion
        Cyclotron design with fields and RF cavities
    target_energy_mev : float
        Target final energy [MeV]
    max_radius_m : float, optional
        Maximum allowed radius [m] (default: 0.4)
    algorithm : str
        Pusher algorithm (default: 'RK4')
    steps_per_turn : int
        Time steps per turn (default: 500)
    verbose : bool
        Print progress (default: True)
    checkpoint_file : str, optional
        CSV file for checkpointing
    """

    def __init__(self,
                 design,
                 target_energy_mev: float,
                 max_radius_m: float = 0.4,
                 algorithm: str = 'RK4',
                 steps_per_turn: int = 500,
                 verbose: bool = True,
                 checkpoint_file: Optional[str] = None):

        self.design = design
        self.target_energy_mev = target_energy_mev
        self.target_energy_j = target_energy_mev * 1.602176634e-13
        self.r_max = max_radius_m
        self.algorithm = algorithm
        self.steps_per_turn = steps_per_turn
        self.verbose = verbose
        self.checkpoint_file = checkpoint_file

        # Validate design
        if not design.is_valid(verbose=False):
            raise ValueError("Design must have bfield, species, and RF cavities")

        if len(design.rf_cavities) == 0:
            raise ValueError("Design must have at least one RF cavity")

        # Create pusher
        self.pusher = Pusher(design.species, algorithm=algorithm)

        # Cache
        self._zero_efield = Field.zero()

        # Optimization tracking
        self.iteration = 0
        self.best_cost = np.inf
        self.best_params = None

        # Initialize checkpoint file
        if self.checkpoint_file:
            self._init_checkpoint_file()

    def _init_checkpoint_file(self):
        """Initialize CSV checkpoint file with headers."""
        with open(self.checkpoint_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'bunch_phase_deg', 'rf_freq_mhz', 'r0_mm', 'vr0_m_s',
                'final_energy_mev', 'n_turns', 'cost', 'success', 'timestamp'
            ])

    def _write_checkpoint(self, params, cost, energy, n_turns, success):
        """Append iteration to checkpoint file."""
        if not self.checkpoint_file:
            return

        with open(self.checkpoint_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.iteration,
                params[0],  # bunch_phase
                params[1],  # rf_freq
                params[2] * 1000 if len(params) > 2 else 0,  # r0
                params[3] if len(params) > 3 else 0,  # vr0
                energy,
                n_turns,
                cost,
                success,
                time.time()
            ])

    def calculate_expected_trajectory(self,
                                      e_inject_mev: float,
                                      de_per_turn_mev: float,
                                      n_turns: int) -> np.ndarray:
        """
        Calculate expected radius vs turn for isochronous cyclotron.

        Uses p = q*B(r)*r to find r(E) from field map.

        Returns
        -------
        r_expected : np.ndarray
            Expected radius per turn [m]
        """
        energies = e_inject_mev + de_per_turn_mev * np.arange(n_turns)
        radii = np.zeros(n_turns)

        q = abs(self.design.species.charge)
        mass_mev = self.design.species.mass_mev

        for i, E in enumerate(energies):
            # Calculate momentum
            gamma = E / mass_mev + 1.0
            p = np.sqrt(gamma ** 2 - 1.0) * mass_mev * 1.602176634e-13 / 299792458.0  # kg*m/s

            # Find radius where p = q*B*r (iterative)
            # Start with previous radius (or reasonable guess)
            r_guess = radii[i - 1] if i > 0 else 0.1

            for _ in range(10):  # Newton iteration
                pts = np.array([[r_guess, 0.0, 0.0]])
                bz = self.design.bfield(pts)[0, 2]

                r_new = p / (q * bz) if bz > 0 else r_guess
                if abs(r_new - r_guess) < 1e-6:
                    break
                r_guess = r_new

            radii[i] = r_guess

        return radii

    def track_with_rf(self,
                      r0: np.ndarray,
                      v0: np.ndarray,
                      dt: float,
                      max_turns: int) -> Tuple[bool, List, List, np.ndarray, np.ndarray, List]:
        """
        Track particle with RF cavities and record data.

        Returns
        -------
        success : bool
            True if reached target or max turns without loss
        poincare_points : list
            Poincaré crossings
        rf_crossings : list
            RF cavity crossings
        trajectory : np.ndarray
            Full position trajectory
        """
        nsteps = max_turns * self.steps_per_turn

        # Storage
        r_traj = np.zeros((nsteps, 3))
        v_traj = np.zeros((nsteps, 3))
        poincare_points = []
        rf_crossings = []

        # Initialize
        r = r0.copy()
        v = v0.copy()
        t = 0.0
        turn = 0

        mass = self.design.species.mass_kg
        charge = self.design.species.charge

        # Boris half-step initialization
        if self.pusher.algorithm == 'boris':
            ef = self._zero_efield(r.reshape(1, 3))
            bf = self.design.bfield(r.reshape(1, 3))
            _, v = self.pusher.push(r, v, ef, bf, -0.5 * dt)

        # Track
        turn_ids = []

        for step in range(nsteps):
            # print(r, v)
            r_prev = r.copy()
            v_prev = v.copy()

            # Get fields
            # ef = self.pusher._ensure_field_array(self._zero_efield(r.reshape(1, 3)))
            # bf = self.pusher._ensure_field_array(self.design.bfield(r.reshape(1, 3)))

            # Push
            r, v = self.pusher.push(r, v, self._zero_efield, self.design.bfield, dt)
            t += dt

            # Check RF cavities
            for cav_id, cavity in enumerate(self.design.rf_cavities):
                v, r, crossed, de_mev, phase = cavity.apply_kick_if_crossing(
                    r_prev, r, v, t, dt, self.design, self.pusher
                )

                if crossed:
                    # TODO: FIXME needs to be relativistic or remove?
                    E_before = 0.5 * mass * np.dot(v_prev, v_prev) / 1.602176634e-16  # keV
                    E_after = 0.5 * mass * np.dot(v, v) / 1.602176634e-16

                    rf_crossings.append(RFCrossingData(
                        turn=turn,
                        cavity_id=cav_id,
                        energy_before_kev=E_before,
                        energy_after_kev=E_after,
                        energy_gain_kev=de_mev * 1000.0,
                        phase_deg=phase,
                        time=t
                    ))

            # Store
            r_traj[step] = r
            v_traj[step] = v

            # Check Poincaré crossing (y=0, moving upward)
            if r_prev[1] <= 0.0 < r[1]:
                # TODO: FIXME needs to be relativistic
                E_mev = 0.5 * mass * np.dot(v, v) / 1.602176634e-13
                turn_ids.append(step)  # Record step number
                # Get RF phase at crossing
                cav = self.design.rf_cavities[0]  # Reference cavity
                phase_rad = np.fmod(cav.omega * t + cav.get_total_phase_rad(), 2.0 * np.pi)
                phase_deg = np.rad2deg(phase_rad)

                poincare_points.append(PoincarePoint(
                    turn=turn,
                    r=r[0],
                    vr=v[0],
                    energy_mev=E_mev,
                    phase_deg=phase_deg,
                    time=t
                ))

                turn += 1

                # Check termination conditions
                if E_mev >= self.target_energy_mev:
                    if self.verbose:
                        print(f"    Reached target energy: {E_mev:.3f} MeV at turn {turn}")
                    r_traj = r_traj[:step + 1]
                    return True, poincare_points, rf_crossings, r_traj, v_traj, turn_ids

                if turn >= max_turns:
                    if self.verbose:
                        print(f"    Reached max turns: {turn}")
                    r_traj = r_traj[:step + 1]
                    return True, poincare_points, rf_crossings, r_traj, v_traj, turn_ids

                # Check boundaries
                radius = np.sqrt(r[0] ** 2 + r[1] ** 2)
                if radius > self.r_max:
                    if self.verbose:
                        print(
                            f"    Particle lost: r={radius * 1000:.1f} mm > r_max={self.r_max * 1000:.1f} mm at turn {turn}")
                    r_traj = r_traj[:step + 1]
                    return False, poincare_points, rf_crossings, r_traj, v_traj, turn_ids

            # Boris final half-step
            if self.pusher.algorithm == 'boris':
                ef = self._zero_efield(r.reshape(1, 3))
                bf = self.design.bfield(r.reshape(1, 3))
                _, v = self.pusher.push(r, v, ef, bf, 0.5 * dt)
                v_traj[-1] = v

        return True, poincare_points, rf_crossings, r_traj, v_traj, turn_ids

    def calculate_turn_metrics(self,
                               traj: np.ndarray,
                               v_traj: np.ndarray,
                               turn_ids: List[int]) -> dict:
        """
        Calculate turn-by-turn orbit quality metrics from full trajectory.

        Parameters
        ----------
        traj : np.ndarray
            Position trajectory (nsteps x 3) [m]
        v_traj : np.ndarray
            Velocity trajectory (nsteps x 3) [m/s]
        turn_ids : list of int
            Step indices where positive x-axis was crossed (length = n_turns)

        Returns
        -------
        metrics : dict
            'r_center' : mean orbit center radius per turn [m]
            'r_spread' : std dev of radius per turn [m]
            'dr' : turn separation [m]
            'energy' : energy at end of turn [MeV]
            'x_center' : mean x position per turn [m]
            'y_center' : mean y position per turn [m]
        """
        if len(turn_ids) == 0:
            return {
                'r_center': np.array([]),
                'r_spread': np.array([]),
                'dr': np.array([]),
                'energy': np.array([]),
                'x_center': np.array([]),
                'y_center': np.array([])
            }

        n_turns = len(turn_ids)

        r_center = np.zeros(n_turns)
        r_spread = np.zeros(n_turns)
        energy_turn = np.zeros(n_turns)
        x_center = np.zeros(n_turns)
        y_center = np.zeros(n_turns)
        r_avg = np.zeros(n_turns)

        mass = self.design.species.mass_kg

        for i in range(n_turns):
            # Get trajectory segment for this turn
            if i == 0:
                start_idx = 0
            else:
                start_idx = turn_ids[i - 1]

            end_idx = turn_ids[i]
            # end_idx = turn_ids[i] if i < n_turns - 1 else len(traj)

            # Extract segment
            traj_segment = traj[start_idx:end_idx]

            if len(traj_segment) < 2:
                continue

            # Calculate orbit center: mean(x), mean(y)
            x_mean = np.mean(traj_segment[:, 0])
            y_mean = np.mean(traj_segment[:, 1])

            x_center[i] = x_mean
            y_center[i] = y_mean

            # Center radius
            r_center[i] = np.sqrt(x_mean ** 2 + y_mean ** 2)

            # Calculate radii of all points in segment
            radii = np.sqrt(traj_segment[:, 0] ** 2 + traj_segment[:, 1] ** 2)

            # Radial spread (betatron amplitude)
            r_spread[i] = np.std(radii)
            r_avg[i] = np.mean(radii)

            # Energy at end of turn TODO: redundant (calculated outside alrady)
            v_end = v_traj[end_idx - 1] if end_idx <= len(v_traj) else v_traj[-1]
            v_mag_sq = np.dot(v_end, v_end)
            ekin = 0.5 * mass * v_mag_sq
            energy_turn[i] = ekin / 1.602176634e-13  # Convert to MeV

        # Turn separation
        dr = np.diff(r_avg)

        return {
            'r_center': r_center,
            'r_spread': r_spread,
            'r_avg': r_avg,
            'dr': dr,
            'energy': energy_turn,
            'x_center': x_center,
            'y_center': y_center
        }

    def objective_function(self, params, r0_seo, v0_seo, dt, max_turns, r_expected):
        """
        Objective function for optimization.

        Parameters
        ----------
        params : list
            [bunch_phase_deg, rf_freq_hz] or
            [bunch_phase_deg, rf_freq_hz, r0, vr0]
        r0_seo : np.ndarray
            Initial position from SEO
        v0_seo : np.ndarray
            Initial velocity from SEO
        dt : float
            Timestep
        max_turns : int
            Maximum turns
        r_expected : np.ndarray
            Expected radius trajectory

        Returns
        -------
        cost : float
            Objective function value (lower is better)
        """
        self.iteration += 1

        # Extract parameters
        bunch_phase_deg = params[0]
        rf_freq_hz = params[1]

        if len(params) > 2:
            r0 = np.array([params[2], 0.0, 0.0])
            v_mag = np.linalg.norm(v0_seo)
            vr = params[3]
            vtheta = np.sqrt(v_mag ** 2.0 - vr ** 2.0) if v_mag ** 2.0 > vr ** 2.0 else 0.0
            v0 = np.array([vr, vtheta, 0.0])
        else:
            r0 = r0_seo
            v0 = v0_seo

        # Set design parameters
        self.design.set_bunch_phase(bunch_phase_deg)
        self.design.set_rf_frequency(rf_freq_hz)

        # Track
        try:
            success, poincare, rf_crossings, r_traj, v_traj, turn_ids = self.track_with_rf(
                r0, v0, dt, max_turns
            )
        except Exception as e:
            if self.verbose:
                print(f"    Iteration {self.iteration}: Tracking failed: {e}")
            return 1e10

        # Check for loss
        if not success:
            cost = 1e8
            if self.verbose:
                print(f"    Iteration {self.iteration}: Particle lost, cost={cost:.2e}")
            self._write_checkpoint(params, cost, 0.0, 0, False)
            return cost

        # Calculate metrics
        if len(poincare) == 0:
            cost = 1e9
            if self.verbose:
                print(f"    Iteration {self.iteration}: No Poincare points, cost={cost:.2e}")
            self._write_checkpoint(params, cost, 0.0, 0, False)
            return cost

        metrics = self.calculate_turn_metrics(r_traj, v_traj, turn_ids)

        n_turns = len(metrics['r_center'])
        final_energy = poincare[-1].energy_mev

        # Cost function components
        cost = 0.0

        # 1. Energy target penalty
        w_energy = 5.0
        cost -= w_energy * final_energy

        # 2. Orbit centering penalty (compare to expected trajectory)
        if n_turns > 0 and len(r_expected) > 0:
            centering_error = np.mean(metrics['r_center'])
            w_centering = 1000.0
            cost += w_centering * centering_error

        # # 3. Breathing penalty (radial oscillation amplitude)
        # r_spread_max = np.max(metrics['r_spread']) if len(metrics['r_spread']) > 0 else 0.0
        # breathing_threshold = 0.005  # 5 mm
        # if r_spread_max > breathing_threshold:
        #     w_breathing = 5000.0
        #     cost += w_breathing * (r_spread_max - breathing_threshold) ** 2

        # 4. Turn separation smoothness penalty
        if len(metrics['dr']) > 1:
            dr_variation = np.std(metrics['dr'])
            w_smooth = 1000.0
            cost += w_smooth * dr_variation ** 2

        # # 5. Penalty for not reaching enough turns
        # if n_turns < max_turns * 0.5 and final_energy < self.target_energy_mev * 0.9:
        #     w_turns = 100.0
        #     cost += w_turns * (max_turns * 0.5 - n_turns)

        # Checkpoint
        self._write_checkpoint(params, cost, final_energy, n_turns, True)

        # Track best
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_params = params.copy()
            if self.verbose:
                print(f"    Iteration {self.iteration}: NEW BEST - cost={cost:.2e}, "
                      f"E={final_energy:.3f} MeV, turns={n_turns}, "
                      f"phase={bunch_phase_deg:.1f}deg, f={rf_freq_hz / 1e6:.3f} MHz")
        else:
            if self.verbose and self.iteration % 10 == 0:
                print(f"    Iteration {self.iteration}: cost={cost:.2e}, "
                      f"E={final_energy:.3f} MeV, turns={n_turns}")

        return cost

    def optimize(self,
                 initial_seo,
                 max_turns: int = 500,
                 optimize_params: List[str] = ['bunch_phase', 'rf_freq'],
                 method: str = 'differential_evolution',
                 bounds: Optional[dict] = None,
                 maxiter: int = 100) -> OptimizedOrbit:
        """
        Optimize accelerated orbit parameters.

        Parameters
        ----------
        initial_seo : StaticOrbit
            Starting point from SEO finder
        max_turns : int
            Maximum turns to track
        optimize_params : list
            Parameters to optimize: 'bunch_phase', 'rf_freq', 'r0', 'vr0'
        method : str
            'differential_evolution' or 'nelder_mead'
        bounds : dict, optional
            Custom bounds for parameters
        maxiter : int
            Maximum optimization iterations

        Returns
        -------
        result : OptimizedOrbit
            Optimization result
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("ACCELERATED ORBIT OPTIMIZATION")
            print("=" * 70)
            print(f"Target energy: {self.target_energy_mev} MeV")
            print(f"Initial energy: {initial_seo.energy_kev / 1000:.3f} MeV")
            print(f"Max turns: {max_turns}")
            print(f"Optimizing: {optimize_params}")
            print(f"Method: {method}")

        # Setup initial conditions from SEO
        r0_seo = initial_seo.r0
        v0_seo = initial_seo.v0

        # Estimate energy gain per turn
        total_voltage = sum(cav.voltage for cav in self.design.rf_cavities)
        charge = abs(self.design.species.charge)
        de_per_turn = total_voltage * charge / 1.602176634e-13  # MeV

        if self.verbose:
            print(f"Estimated dE/turn: {de_per_turn:.3f} MeV")

        # Calculate expected trajectory
        r_expected = self.calculate_expected_trajectory(
            initial_seo.energy_kev / 1000.0,
            de_per_turn,
            max_turns
        )

        # Timestep
        dt = self._estimate_timestep(initial_seo.frequency_hz)

        if self.verbose:
            print(f"Timestep: {dt * 1e12:.2f} ps")

        # Setup parameter bounds
        if bounds is None:
            bounds = {}

        param_bounds = []
        param_names = []
        x0 = []

        # Bunch phase
        if 'bunch_phase' in optimize_params:
            param_names.append('bunch_phase')
            param_bounds.append(bounds.get('bunch_phase', (-180, 180)))
            x0.append(20.0)  # Typical initial guess

        # RF frequency
        if 'rf_freq' in optimize_params:
            param_names.append('rf_freq')
            f_seo = initial_seo.frequency_hz
            param_bounds.append(bounds.get('rf_freq', (f_seo * 0.95, f_seo * 1.05)))
            x0.append(f_seo)

        # Initial radius
        if 'r0' in optimize_params:
            param_names.append('r0')
            r_seo = np.linalg.norm(r0_seo[:2])
            param_bounds.append(bounds.get('r0', (r_seo - 0.005, r_seo + 0.005)))
            x0.append(r_seo)

        # Initial radial velocity
        if 'vr0' in optimize_params:
            param_names.append('vr0')
            param_bounds.append(bounds.get('vr0', (-5e5, 5e5)))
            x0.append(0.0)

        if self.verbose:
            print(f"\nOptimization parameters:")
            for name, bnd, x in zip(param_names, param_bounds, x0):
                print(f"  {name}: bounds={bnd}, initial={x}")
            print()

        # Reset iteration counter
        self.iteration = 0
        self.best_cost = np.inf
        self.best_params = None

        # Optimize
        start_time = time.time()

        if method == 'differential_evolution':
            result = differential_evolution(
                self.objective_function,
                param_bounds,
                args=(r0_seo, v0_seo, dt, max_turns, r_expected),
                maxiter=maxiter,
                workers=1,  # Can increase for parallelization
                updating='deferred',
                disp=False
            )
            optimal_params = result.x
            final_cost = result.fun

        elif method == 'nelder_mead':
            result = minimize(
                self.objective_function,
                x0,
                args=(r0_seo, v0_seo, dt, max_turns, r_expected),
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
                else:
                    print(f"  {name}: {val:.3f}")

        # Final tracking with optimal parameters
        self.design.set_bunch_phase(optimal_params[0])
        self.design.set_rf_frequency(optimal_params[1])

        if len(optimal_params) > 2:
            r0_final = np.array([optimal_params[2], 0.0, 0.0])
            v_mag = np.linalg.norm(v0_seo)
            vr = optimal_params[3]
            vtheta = np.sqrt(v_mag ** 2 - vr ** 2) if v_mag ** 2 > vr ** 2 else 0.0
            v0_final = np.array([vr, vtheta, 0.0])
        else:
            r0_final = r0_seo
            v0_final = v0_seo

        success, poincare, rf_crossings, r_traj, v_traj, turn_ids = self.track_with_rf(
            r0_final, v0_final, dt, max_turns
        )

        metrics = self.calculate_turn_metrics(r_traj, v_traj, turn_ids)

        # Create result object
        optimized_orbit = OptimizedOrbit(
            success=success,
            final_energy_mev=poincare[-1].energy_mev if len(poincare) > 0 else 0.0,
            n_turns=len(metrics['r_center']),
            bunch_phase_deg=optimal_params[0],
            rf_frequency_mhz=optimal_params[1] / 1e6,
            initial_r_mm=r0_final[0] * 1000,
            initial_vr_m_s=v0_final[0],
            initial_vtheta_m_s=v0_final[1],
            trajectory=r_traj,
            poincare_points=poincare,
            rf_crossings=rf_crossings,
            turn_metrics=metrics,
            cost=final_cost,
            metadata={
                'initial_seo': initial_seo,
                'optimization_method': method,
                'optimization_time_s': elapsed,
                'total_iterations': self.iteration,
                'param_names': param_names,
                'param_bounds': param_bounds
            }
        )

        return optimized_orbit

    def _estimate_timestep(self, frequency_hz: float) -> float:
        """Estimate timestep from RF frequency."""
        period = 1.0 / frequency_hz
        dt = period / self.steps_per_turn
        return dt

if __name__ == "__main__":
    print("accelerated_orbit_finder.py - Accelerated Orbit Optimization")
    print("=" * 70)
    print("This module requires a CentralRegion design with RF cavities.")
    print("See examples/02_optimize_acceleration.py for usage.")
