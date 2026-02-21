"""
accelerated_orbit_finder_multiparticle.py - Multi-Particle Accelerated Orbit Optimization

Extends single-particle optimization to track particle distributions.
Optimizes bunch phase, RF frequency, and beam quality metrics.

Part of: PyCentralRegion module
Dependencies: PyPATools, scipy, numpy

Usage:
    from accelerated_orbit_finder_multiparticle import AcceleratedOrbitFinderMulti

    finder = AcceleratedOrbitFinderMulti(design, target_energy_mev=5.0, n_particles=1000)

    # Single run
    result = finder.track_once(initial_r_mm=100, initial_v_tangential_m_s=2.6e7,
                               bunch_phase_deg=-25, rf_freq_mhz=168.0)

    # Optimization
    result = finder.optimize(initial_seo, max_turns=100)
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from scipy.optimize import differential_evolution, minimize
from PyPATools.pusher import Pusher
from PyPATools.field import Field
from PyPATools.particles import ParticleDistribution
from PyPATools.global_variables import CLIGHT
import warnings
import csv
from pathlib import Path
import time
from tqdm import tqdm
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
    particle_id: int
    energy_before_kev: float
    energy_after_kev: float
    energy_gain_kev: float
    phase_deg: float
    time: float


@dataclass
class TurnStatistics:
    """Statistics for one turn across all particles."""
    turn: int
    mean_r: float  # m
    std_r: float  # m
    mean_energy_mev: float
    std_energy_mev: float
    mean_x: float
    mean_y: float
    emittance_r: float  # Placeholder for future


@dataclass
class OptimizedOrbit:
    """
    Result from multi-particle accelerated orbit optimization.

    Includes beam statistics and individual particle data.
    """
    success: bool
    final_energy_mev: float
    n_turns: int
    bunch_phase_deg: float
    rf_frequency_mhz: float
    initial_r_mm: float
    initial_vr_m_s: float
    n_particles: int
    trajectory_reference: np.ndarray  # Reference particle full trajectory
    turn_statistics: List[TurnStatistics]
    poincare_points_all: List[List[PoincarePoint]]  # Per particle
    rf_crossings: List[RFCrossingData]
    std_r_per_step: np.ndarray  # Radial spread at every step
    cost: float
    metadata: dict = field(default_factory=dict)


class AcceleratedOrbitFinderMulti:
    """
    Multi-particle optimizer for accelerated cyclotron orbits.

    Parameters
    ----------
    design : CentralRegion
        Cyclotron design with fields and RF cavities
    target_energy_mev : float
        Target final energy [MeV]
    n_particles : int
        Number of particles to track (default: 100)
    max_radius_m : float
        Maximum allowed radius [m] (default: 0.4)
    algorithm : str
        Pusher algorithm (default: 'RK4')
    steps_per_turn : int
        Time steps per turn (default: 500)
    dump_frequency : int
        Save full distribution every N turns (default: 10)
    verbose : bool
        Print progress (default: True)
    checkpoint_file : str, optional
        CSV file for checkpointing
    """

    def __init__(self,
                 design,
                 target_energy_mev: float,
                 n_particles: int = 100,
                 max_radius_m: float = 0.4,
                 algorithm: str = 'RK4',
                 steps_per_turn: int = 500,
                 dump_frequency: int = 10,
                 verbose: bool = True,
                 checkpoint_file: Optional[str] = None):

        self.design = design
        self.target_energy_mev = target_energy_mev
        self.n_particles = n_particles
        self.r_max = max_radius_m
        self.algorithm = algorithm
        self.steps_per_turn = steps_per_turn
        self.dump_frequency = dump_frequency
        self.verbose = verbose
        self.checkpoint_file = checkpoint_file

        # Validate design
        if not design.is_valid(verbose=False):
            raise ValueError("Design must have bfield, species, and RF cavities")

        if len(design.rf_cavities) == 0:
            raise ValueError("Design must have at least one RF cavity")

        # Create pusher
        self.pusher = Pusher(design.species, algorithm=algorithm)

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
                'final_energy_mev', 'final_std_r_mm', 'envelope_oscillation_mm',
                'n_turns', 'cost', 'success', 'timestamp'
            ])

    def _write_checkpoint(self, params, cost, energy, std_r, envelope_osc, n_turns, success):
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
                std_r * 1000,  # mm
                envelope_osc * 1000,  # mm
                n_turns,
                cost,
                success,
                time.time()
            ])

    def create_initial_distribution(self,
                                    r_mean: float,
                                    r_spread: float,
                                    v_tangential: float,
                                    v_perp: float,
                                    vr_spread: float,
                                    distribution_type: str = 'gaussian') -> ParticleDistribution:
        """
        Create initial particle distribution.

        All particles start at θ=0 (positive x-axis) with variations in r and vr.

        Parameters
        ----------
        r_mean : float
            Mean radius [m]
        r_spread : float
            Radial spread (1σ) [m]
        v_tangential : float
            Tangential velocity [m/s]
        v_perp : float
            Perpendicular velocity [m/s]
        vr_spread : float
            Radial velocity spread (1σ) [m/s]
        distribution_type : str
            'gaussian', 'uniform', or 'waterbag'

        Returns
        -------
        pd : ParticleDistribution
            Initial distribution
        """
        # if distribution_type == 'gaussian':
        #     r_values = np.random.normal(r_mean, r_spread, self.n_particles)
        #     vr_values = np.random.normal(0.0, vr_spread, self.n_particles)
        # elif distribution_type == 'uniform':
        #     r_values = np.random.uniform(r_mean - r_spread, r_mean + r_spread, self.n_particles)
        #     vr_values = np.random.uniform(-vr_spread, vr_spread, self.n_particles)
        # elif distribution_type == 'waterbag':
        #     # Uniform in 4D phase space (r, vr)
        #     # Generate on unit circle, scale
        #     theta_4d = np.random.uniform(0, 2 * np.pi, self.n_particles)
        #     rho = np.sqrt(np.random.uniform(0, 1, self.n_particles))
        #     r_values = r_mean + r_spread * rho * np.cos(theta_4d)
        #     vr_values = vr_spread * rho * np.sin(theta_4d)
        # else:
        #     raise ValueError(f"Unknown distribution type: {distribution_type}")

        if self.verbose:
            print(f"Creating initial {distribution_type} distribution.")

        # Create ParticleDistribution
        corr_matrix = np.zeros([6, 6])
        for i in range(6):
            corr_matrix[i, i] = 1

        pd = ParticleDistribution.generate_distribution(self.design.species,
                                                        type=['gaussian', 'gaussian', 'gaussian'],
                                                        s_direction='z',
                                                        n_particles=self.n_particles,
                                                        correlation_matrix=corr_matrix,
                                                        sigma_x=r_spread,
                                                        sigma_px=vr_spread/CLIGHT, # TODO: assuming gamma = 1 at start
                                                        sigma_y=1e-20,
                                                        sigma_py=1e-20,
                                                        sigma_z=1e-20,
                                                        sigma_pz=1e-20,
                                                        cutoff_x = 3,
                                                        cutoff_px = 3)

        pd.set_centroid(r_mean, 0.0, 0.0)
        pd.add_mean_momentum(0.0,
                             v_tangential / np.sqrt(CLIGHT ** 2 - v_tangential ** 2),
                             v_perp / np.sqrt(CLIGHT ** 2 - v_perp ** 2))

        # # Positions: all at θ=0
        # pd.x = r_values
        # pd.y = np.zeros(self.n_particles)
        # pd.z = np.zeros(self.n_particles)
        #
        # # Velocities: tangential + radial perturbation
        # pd.set_p_from_v(vr_values,
        #                 np.full(self.n_particles, v_tangential),
        #                 np.zeros(self.n_particles))

        return pd

    def track_with_rf_multiparticle(self,
                                    initial_distribution: ParticleDistribution,
                                    dt: float,
                                    max_turns: int,
                                    show_progress: bool = True,
                                    save_full: bool = False) -> Tuple[
        bool, List, List, np.ndarray, List, np.ndarray, List, np.ndarray]:
        """
        Track multiple particles with RF cavities.

        Returns
        -------
        success : bool
            True if reached target or max turns
        turn_statistics : list
            TurnStatistics per turn
        rf_crossings : list
            All RF crossings
        trajectory_ref : np.ndarray
            Full trajectory of reference particle (particle 0)
        poincare_all : list
            List of Poincaré points per particle
        std_r_per_step : np.ndarray
            Radial spread at every step
        """
        nsteps = max_turns * self.steps_per_turn

        # t_start = time.time()

        # Initialize particle arrays
        r_array = initial_distribution.x_vec
        v_array = initial_distribution.v_vec

        # Fix particle 0 as reference particle
        r_array[0, :] = initial_distribution.centroid
        v_array[0, :] = np.mean(initial_distribution.v_vec, axis=0)

        n_particles = len(r_array)

        # Storage
        trajectory_ref = np.zeros((nsteps, 3))  # Only reference particle
        turn_statistics = []
        poincare_all = [[] for _ in range(n_particles)]
        rf_crossings = []
        std_r_per_step = np.zeros(nsteps)  # NEW: Radial spread every step
        if save_full:
            full_beam = np.nan * np.ones((nsteps, self.n_particles, 6))
        else:
            full_beam = None

        # Previous positions for crossing detection
        r_prev = r_array.copy()

        t = 0.0
        turn = 0
        active = np.ones(n_particles, dtype=bool)

        # Boris half-step initialization
        if self.pusher.algorithm.lower() == 'boris':
            for i in range(n_particles):
                if active[i]:
                    ef = self.design.efield(r_array[i].reshape(1, 3))
                    bf = self.design.bfield(r_array[i].reshape(1, 3))
                    _, v_array[i] = self.pusher.push(r_array[i], v_array[i], ef, bf, -0.5 * dt)

        # Track
        pbar = tqdm(total=nsteps,
                    desc=f"Tracking {n_particles} particles",
                    disable=not show_progress,
                    ncols=120)

        turn_ids = []  # number of steps at which turns end

        for step in range(nsteps):
            # Push all particles
            # t0 = time.perf_counter()
            r_array, v_array = self.pusher.push_batch(r_array, v_array,
                                                      self.design.efield, self.design.bfield,
                                                      dt)
            t += dt

            # Check RF cavities
            for cav_id, cavity in enumerate(self.design.rf_cavities):
                # Check which particles crossed this cavity

                # t0_rf = time.perf_counter()
                crossed_mask, t_cross_array = cavity.check_crossings_batch(r_prev, r_array)
                # t_rf_cross += time.perf_counter() - t0_rf

                if np.any(crossed_mask):
                    # Apply kicks to crossed particles
                    # t0_rf = time.perf_counter()
                    v_array, r_array, energy_gains, phases = cavity.apply_kicks_batch(
                        r_prev, r_array, v_array, crossed_mask, t_cross_array,
                        t, dt, self.design, self.pusher
                    )

                    p_vec = v_array / np.sqrt(CLIGHT ** 2 - v_array ** 2)
                    self.design.beam.x_vec_p_vec = (r_array, p_vec)  # Reset to full beam (was using it for crossing particles inside RF Cavity)
                    # t_rf_kick += time.perf_counter() - t0_rf

                    # Record crossings
                    for i in np.where(crossed_mask)[0]:
                        E_after = self.design.beam.mean_energy_mev * 1000  # keV
                        rf_crossings.append(RFCrossingData(
                            turn=turn,
                            cavity_id=cav_id,
                            particle_id=i,
                            energy_before_kev=E_after - energy_gains[i] * 1000,
                            energy_after_kev=E_after,
                            energy_gain_kev=energy_gains[i] * 1000,
                            phase_deg=phases[i],
                            time=t
                        ))

            # t_rf += time.perf_counter() - t0

            # Check for reference particle crossing the +x-axis
            # TODO: this needs to be cleaner (catching edge cases) and modulized,
            #  probably all tracking should go into a tracking.py file
            if r_prev[0, 1] <= 0.0 < r_array[0, 1]:
                turn += 1
                turn_ids.append(step)

                p_vec_active = v_array[active] / np.sqrt(CLIGHT ** 2 - v_array[active] ** 2)
                self.design.beam.x_vec_p_vec = (r_array[active], p_vec_active)
                radii = np.sqrt(self.design.beam.x ** 2 + self.design.beam.y ** 2)

                turn_statistics.append(TurnStatistics(
                        turn=turn,
                        mean_r=np.mean(radii),
                        std_r=np.std(radii),
                        mean_energy_mev=self.design.beam.mean_energy_mev,
                        std_energy_mev=self.design.beam.rms_energy_spread_mev,
                        mean_x=np.mean(self.design.beam.x),
                        mean_y=np.mean(self.design.beam.y),
                        emittance_r=0.0  # Placeholder
                    ))


            # t0 = time.perf_counter()
            # Check Poincaré crossings (y=0, moving upward)
            # crossed_particles = []
            # for i in range(n_particles):
            #     if not active[i]:
            #         continue
            #
            #     if r_prev[i, 1] <= 0.0 < r_array[i, 1]:
            #         crossed_particles.append(i)
            #
            #         # Get energy for this particle
            #         self.design.beam.set_p_from_v_vec(v_array)
            #             # np.array([v_array[i, 0]]),
            #             # np.array([v_array[i, 1]]),
            #             # np.array([v_array[i, 2]]))
            #         E_mev = self.design.beam.mean_energy_mev
            #
            #         # Get RF phase
            #         cav = self.design.rf_cavities[0]
            #         phase_rad = np.fmod(cav.omega * t + cav.get_total_phase_rad(), 2.0 * np.pi)
            #         phase_deg = np.rad2deg(phase_rad)
            #
            #         poincare_all[i].append(PoincarePoint(
            #             turn=turn,
            #             r=r_array[i, 0],
            #             vr=v_array[i, 0],
            #             energy_mev=E_mev,
            #             phase_deg=phase_deg,
            #             time=t
            #         ))

            # t_poincare += time.perf_counter() - t0

            # # If reference particle crossed, compute turn statistics
            # if len(crossed_particles) > 0 and crossed_particles[0] == 0:
            #     # Compute statistics for all active particles at this crossing
            #     active_r = r_array[active]
            #     active_v = v_array[active]
            #
            #     radii = np.sqrt(active_r[:, 0] ** 2 + active_r[:, 1] ** 2)
            #
            #     # Energy statistics (relativistic)
            #     v_mag = np.linalg.norm(active_v, axis=1)
            #     beta = v_mag / CLIGHT
            #     gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
            #     energies_mev = (gamma - 1.0) * self.design.species.mass_mev
            #
            #     turn += 1
            #
            #     turn_statistics.append(TurnStatistics(
            #         turn=turn,
            #         mean_r=np.mean(radii),
            #         std_r=np.std(radii),
            #         mean_energy_mev=np.mean(energies_mev),
            #         std_energy_mev=np.std(energies_mev),
            #         mean_x=np.mean(active_r[:, 0]),
            #         mean_y=np.mean(active_r[:, 1]),
            #         emittance_r=0.0  # Placeholder
            #     ))

                # Check termination
                if turn_statistics[-1].mean_energy_mev >= self.target_energy_mev:
                    if self.verbose and show_progress:
                        print(f"    Reached target energy: {turn_statistics[-1].mean_energy_mev:.3f} MeV")
                    trajectory_ref = trajectory_ref[:step + 1]
                    std_r_per_step = std_r_per_step[:step + 1]
                    pbar.close()

                    # elapsed = time.time() - t_start
                    # print(f"\n  Timing breakdown:")
                    # print(f"    Push: {t_push:.2f} s ({t_push / elapsed * 100:.1f}%)")
                    # print(f"    RF:   {t_rf:.2f} s ({t_rf / elapsed * 100:.1f}%)")
                    # print(f"    -- cross: {t_rf_cross:.2f} s ({t_rf_cross / elapsed * 100:.1f}%)")
                    # print(f"    -- kick: {t_rf_kick:.2f} s ({t_rf_kick / elapsed * 100:.1f}%)")
                    # print(f"    Poincare: {t_poincare:.2f} s ({t_poincare / elapsed * 100:.1f}%)")

                    return (True, turn_statistics, rf_crossings, trajectory_ref,
                            poincare_all, std_r_per_step, turn_ids, full_beam)

                if turn >= max_turns:
                    if self.verbose and show_progress:
                        print(f"    Reached max turns: {turn}")
                    trajectory_ref = trajectory_ref[:step + 1]
                    std_r_per_step = std_r_per_step[:step + 1]
                    pbar.close()

                    # elapsed = time.time() - t_start
                    # print(f"\n  Timing breakdown:")
                    # print(f"    Push: {t_push:.2f} s ({t_push / elapsed * 100:.1f}%)")
                    # print(f"    RF:   {t_rf:.2f} s ({t_rf / elapsed * 100:.1f}%)")
                    # print(f"    -- cross: {t_rf_cross:.2f} s ({t_rf_cross / elapsed * 100:.1f}%)")
                    # print(f"    -- kick: {t_rf_kick:.2f} s ({t_rf_kick / elapsed * 100:.1f}%)")
                    # print(f"    Poincare: {t_poincare:.2f} s ({t_poincare / elapsed * 100:.1f}%)")

                    return (True, turn_statistics, rf_crossings, trajectory_ref,
                            poincare_all, std_r_per_step, turn_ids, full_beam)

            # Check boundaries
            for i in range(n_particles):
                if active[i]:
                    radius = np.sqrt(r_array[i, 0] ** 2 + r_array[i, 1] ** 2)
                    if radius > self.r_max:
                        active[i] = False
                        if self.verbose and show_progress and i < 5:  # Only print first few
                            print(f"    Particle {i} lost at r={radius * 1000:.1f} mm")

            # Calculate radial spread at this step
            active_r = r_array[active]
            trajectory_ref[step] = active_r[0]  # store centroid as reference
            radii = np.sqrt(active_r[:, 0] ** 2 + active_r[:, 1] ** 2)
            std_r_per_step[step] = np.std(radii) if len(radii) > 1 else 0.0

            # If all particles lost
            if not np.any(active):
                if self.verbose and show_progress:
                    print(f"    All particles lost at turn {turn}")
                trajectory_ref = trajectory_ref[:step + 1]
                std_r_per_step = std_r_per_step[:step + 1]
                pbar.close()

                # elapsed = time.time() - t_start
                # print(f"\n  Timing breakdown:")
                # print(f"    Push: {t_push:.2f} s ({t_push / elapsed * 100:.1f}%)")
                # print(f"    RF:   {t_rf:.2f} s ({t_rf / elapsed * 100:.1f}%)")
                # print(f"    -- cross: {t_rf_cross:.2f} s ({t_rf_cross / elapsed * 100:.1f}%)")
                # print(f"    -- kick: {t_rf_kick:.2f} s ({t_rf_kick / elapsed * 100:.1f}%)")
                # print(f"    Poincare: {t_poincare:.2f} s ({t_poincare / elapsed * 100:.1f}%)")

                return (False, turn_statistics, rf_crossings, trajectory_ref,
                        poincare_all, std_r_per_step, turn_ids, full_beam)

            # Update previous positions
            r_prev = r_array.copy()

            # Update progress bar
            pbar.update(1)
            if turn_statistics:
                pbar.set_postfix({'Energy': f"{turn_statistics[-1].mean_energy_mev:.2f} MeV"})

            if save_full:
                full_beam[step, :, :3] = r_array[:]
                full_beam[step, :, 3:] = v_array[:]

            # --- END nsteps loop --- #

        # Boris final half-step
        if self.pusher.algorithm.lower() == 'boris':
            for i in range(n_particles):
                if active[i]:
                    ef = self.design.efield(r_array[i].reshape(1, 3))
                    bf = self.design.bfield(r_array[i].reshape(1, 3))
                    _, v_array[i] = self.pusher.push(r_array[i], v_array[i], ef, bf, 0.5 * dt)

        pbar.close()

        # elapsed = time.time() - t_start
        # print(f"\n  Timing breakdown:")
        # print(f"    Push: {t_push:.2f} s ({t_push / elapsed * 100:.1f}%)")
        # print(f"    RF:   {t_rf:.2f} s ({t_rf / elapsed * 100:.1f}%)")
        # print(f"    -- cross: {t_rf_cross:.2f} s ({t_rf_cross / elapsed * 100:.1f}%)")
        # print(f"    -- kick: {t_rf_kick:.2f} s ({t_rf_kick / elapsed * 100:.1f}%)")
        # print(f"    Poincare: {t_poincare:.2f} s ({t_poincare / elapsed * 100:.1f}%)")

        return (True, turn_statistics, rf_crossings, trajectory_ref,
                poincare_all, std_r_per_step, turn_ids, full_beam)

    def track_once(self,
                   initial_r_mm: float,
                   initial_v_tangential_m_s: float,
                   bunch_phase_deg: float,
                   rf_freq_mhz: float,
                   max_turns: int = 500,
                   r_spread_mm: float = 2.0,
                   vr_spread_m_s: float = 1e4) -> OptimizedOrbit:
        """
        Single tracking run without optimization.

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
            Maximum turns to track
        r_spread_mm : float
            Initial radial spread (1σ) [mm]
        vr_spread_m_s : float
            Initial radial velocity spread (1σ) [m/s]

        Returns
        -------
        result : OptimizedOrbit
            Tracking result
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("MULTI-PARTICLE TRACKING (SINGLE RUN)")
            print("=" * 70)
            print(f"Initial radius: {initial_r_mm:.2f} mm")
            print(f"Tangential velocity: {initial_v_tangential_m_s / 1e6:.2f} Mm/s")
            print(f"Bunch phase: {bunch_phase_deg:.2f} deg")
            print(f"RF frequency: {rf_freq_mhz:.6f} MHz")
            print(f"Particles: {self.n_particles}")

        # Set RF parameters
        rf_freq_hz = rf_freq_mhz * 1e6
        self.design.set_bunch_phase(bunch_phase_deg)
        self.design.set_rf_frequency(rf_freq_hz)

        # Create initial distribution
        r_mean = initial_r_mm / 1000.0

        initial_dist = self.create_initial_distribution(
            r_mean=r_mean,
            r_spread=r_spread_mm / 1000.0,
            v_tangential=initial_v_tangential_m_s,
            vr_spread=vr_spread_m_s,
            distribution_type='gaussian'
        )

        # Timestep (estimate from RF frequency)
        dt = self._estimate_timestep(rf_freq_hz)

        # Track
        success, turn_stats, rf_cross, traj_ref, poincare_all, std_r_steps = self.track_with_rf_multiparticle(
            initial_dist, dt, max_turns, show_progress=True
        )

        # Create result
        final_energy = turn_stats[-1].mean_energy_mev if len(turn_stats) > 0 else 0.0

        result = OptimizedOrbit(
            success=success,
            final_energy_mev=final_energy,
            n_turns=len(turn_stats),
            bunch_phase_deg=bunch_phase_deg,
            rf_frequency_mhz=rf_freq_mhz,
            initial_r_mm=initial_r_mm,
            initial_vr_m_s=0.0,
            n_particles=self.n_particles,
            trajectory_reference=traj_ref,
            turn_statistics=turn_stats,
            poincare_points_all=poincare_all,
            rf_crossings=rf_cross,
            std_r_per_step=std_r_steps,
            cost=0.0,
            metadata={
                'mode': 'single_run',
                'r_spread_mm': r_spread_mm,
                'vr_spread_m_s': vr_spread_m_s,
                'initial_v_tangential_m_s': initial_v_tangential_m_s
            }
        )

        if self.verbose:
            print(f"\nFinal energy: {final_energy:.3f} MeV")
            if len(turn_stats) > 0:
                print(f"Final radial spread: {turn_stats[-1].std_r * 1000:.3f} mm")
            print(f"Turns completed: {len(turn_stats)}")
            print(f"Envelope oscillation (std): {np.std(std_r_steps) * 1000:.3f} mm")

        return result


    def calculate_turn_metrics(self,
                               traj: np.ndarray,
                               # v_traj: np.ndarray,
                               turn_ids: List[int]) -> dict:
        """
        Calculate turn-by-turn orbit quality metrics from full trajectory.

        Parameters
        ----------
        traj : np.ndarray
            Position trajectory (nsteps x 3) [m]
        # v_traj : np.ndarray
        #     Velocity trajectory (nsteps x 3) [m/s]
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

            # # Energy at end of turn TODO: redundant (calculated outside alrady)
            # v_end = v_traj[end_idx - 1] if end_idx <= len(v_traj) else v_traj[-1]
            # v_mag_sq = np.dot(v_end, v_end)
            # ekin = 0.5 * mass * v_mag_sq
            # energy_turn[i] = ekin / 1.602176634e-13  # Convert to MeV

        # Turn separation
        dr = np.diff(r_avg)

        return {
            'r_center': r_center,
            'r_spread': r_spread,
            'r_avg': r_avg,
            'dr': dr,
            'energy': None,
            'x_center': x_center,
            'y_center': y_center
        }


    def objective_function(self, params, r_mean, v_tangential, r_spread, vr_spread, dt, max_turns, weights):
        """
        Objective function for multi-particle optimization.

        Parameters
        ----------
        params : list
            [bunch_phase_deg, rf_freq_hz] or
            [bunch_phase_deg, rf_freq_hz, r0, vr0]
        r_mean : float
            Mean initial radius [m]
        v_tangential : float
            Initial tangential velocity [m/s]
        r_spread : float
            Radial spread [m]
        vr_spread : float
            Radial velocity spread [m/s]
        dt : float
            Timestep
        max_turns : int
            Maximum turns
        weights : dict
            Cost function weights

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
            r_mean = params[2]
            vr_mean = params[3] if len(params) > 3 else 0.0
        else:
            vr_mean = 0.0

        # TODO: Include vr_mean!!!

        # Set design parameters
        self.design.set_bunch_phase(bunch_phase_deg)
        self.design.set_rf_frequency(rf_freq_hz)

        # Create distribution
        try:
            initial_dist = self.create_initial_distribution(
                r_mean=r_mean,
                r_spread=r_spread,
                v_tangential=v_tangential,
                v_perp=vr_mean,
                vr_spread=vr_spread,
                distribution_type='gaussian'
            )

        except Exception as e:
            if self.verbose:
                print(f"    Iteration {self.iteration}: Distribution creation failed: {e}")
            return 1e10

        # Track
        try:
            result = self.track_with_rf_multiparticle(initial_dist, dt, max_turns, show_progress=True)
            success, turn_stats, rf_cross, traj_ref, poincare_all, std_r_steps, turn_ids, _ = result
        except Exception as e:
            if self.verbose:
                print(f"    Iteration {self.iteration}: Tracking failed: {e}")
            return 1e10

        # Check for loss
        if not success:
            cost = 1e8
            if self.verbose:
                print(f"    Iteration {self.iteration}: Particles lost, cost={cost:.2e}")
            self._write_checkpoint(params, cost, 0.0, 0.0, 0.0, 0, False)
            return cost

        # Check for insufficient data
        if len(turn_stats) == 0:
            cost = 1e9
            if self.verbose:
                print(f"    Iteration {self.iteration}: No turn statistics, cost={cost:.2e}")
            self._write_checkpoint(params, cost, 0.0, 0.0, 0.0, 0, False)
            return cost

        # Extract metrics
        metrics = self.calculate_turn_metrics(traj_ref, turn_ids)
        final_energy = turn_stats[-1].mean_energy_mev
        final_std_r = turn_stats[-1].std_r

        # mean_r_per_turn = np.array([s.mean_r for s in turn_stats])

        # Envelope oscillation
        envelope_oscillation = np.std(std_r_steps)

        # Turn separation smoothness
        # dr = np.diff(r_mean)
        #
        #
        # if len(mean_r_per_turn) > 1:
        #     dr = np.diff(mean_r_per_turn)
        #     turn_smoothness = np.std(dr)
        # else:
        #     turn_smoothness = 0.0

        # Cost function
        w_energy = weights.get('energy', 5.0)
        w_envelope = weights.get('spread', 100.0)
        w_center = weights.get('center', 1000.0)
        w_smooth = weights.get('smooth', 1000.0)

        cost = 0.0
        cost -= w_energy * final_energy  # Maximize energy
        cost += w_envelope * envelope_oscillation  # Minimize beam size oscillation
        cost += w_center * np.mean(metrics['r_center'])  # Minimize centering
        cost += w_smooth * np.std(metrics['dr']) ** 2  # Smooth turn progression

        print(f"Iteration {self.iteration}, cost breakdown: ekin: {w_energy * final_energy}, "
              f"beam size oscillation: {w_envelope * envelope_oscillation}, "
              f"orbit centering: {w_center * np.mean(metrics['r_center'])}, "
              f"turn separation: {w_smooth * np.std(metrics['dr']) ** 2}")

        # Checkpoint
        self._write_checkpoint(params, cost, final_energy, final_std_r, envelope_oscillation, len(turn_stats), True)

        # Track best
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_params = params.copy()
            if self.verbose:
                print(f"    Iteration {self.iteration}: NEW BEST - cost={cost:.2e}, "
                      f"E={final_energy:.3f} MeV, turns={len(turn_stats)}, "
                      f"env_osc={envelope_oscillation * 1000:.3f} mm, "
                      f"phase={bunch_phase_deg:.1f}deg, f={rf_freq_hz / 1e6:.3f} MHz")
        else:
            if self.verbose and self.iteration % 10 == 0:
                print(f"    Iteration {self.iteration}: cost={cost:.2e}, "
                      f"E={final_energy:.3f} MeV, turns={len(turn_stats)}")

        return cost

    def optimize(self,
                 initial_seo,
                 initial_phase,
                 max_turns: int = 500,
                 r_spread_mm: float = 2.0,
                 vr_spread_m_s: float = 1e4,
                 optimize_params: List[str] = ['bunch_phase', 'rf_freq'],
                 method: str = 'differential_evolution',
                 bounds: Optional[dict] = None,
                 weights: Optional[dict] = None,
                 maxiter: int = 100) -> Tuple[OptimizedOrbit, np.ndarray]:
        """
        Optimize accelerated orbit parameters for multi-particle beam.

        Parameters
        ----------
        initial_seo : StaticOrbit
            Starting point from SEO finder
        initial_phase : float
        max_turns : int
            Maximum turns to track
        r_spread_mm : float
            Initial radial spread (1σ) [mm]
        vr_spread_m_s : float
            Initial radial velocity spread (1σ) [m/s]
        optimize_params : list
            Parameters to optimize: 'bunch_phase', 'rf_freq', 'r0', 'vr0'
        method : str
            'differential_evolution' or 'nelder_mead'
        bounds : dict, optional
            Custom bounds for parameters
        weights : dict, optional
            Cost function weights: 'energy', 'spread', 'center', 'smooth'
        maxiter : int
            Maximum optimization iterations

        Returns
        -------
        result : OptimizedOrbit
            Optimization result
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("MULTI-PARTICLE ACCELERATED ORBIT OPTIMIZATION")
            print("=" * 70)
            print(f"Target energy: {self.target_energy_mev} MeV")
            print(f"Initial energy: {initial_seo.energy_kev / 1000:.3f} MeV")
            print(f"Max turns: {max_turns}")
            print(f"Number of particles: {self.n_particles}")
            print(f"Initial spread: r={r_spread_mm} mm, vr={vr_spread_m_s/1e3:.1f} km/s")
            print(f"Optimizing: {optimize_params}")
            print(f"Method: {method}")

        # Setup initial conditions from SEO
        r0_seo = initial_seo.r0
        v0_seo = initial_seo.v0
        r_mean = np.linalg.norm(r0_seo[0])
        v_tangential = np.linalg.norm(v0_seo[1])

        # Timestep
        dt = self._estimate_timestep(initial_seo.frequency_hz)

        if self.verbose:
            print(f"Timestep: {dt * 1e12:.2f} ps")

        # Setup weights
        if weights is None:
            weights = {
                'energy': 5.0,
                'spread': 100.0,
                'center': 1000.0,
                'smooth': 1000.0
            }

        if self.verbose:
            print(f"\nCost function weights:")
            print(f"  Energy (maximize): {weights['energy']}")
            print(f"  Envelope oscillation (minimize): {weights['spread']}")
            print(f"  Centering (minimize): {weights['center']}")
            print(f"  Turn smoothness (minimize): {weights['smooth']}")

        # Setup parameter bounds
        if bounds is None:
            bounds = {}

        param_bounds = []
        param_names = []
        x0 = []

        # Bunch phase
        if 'bunch_phase' in optimize_params:
            param_names.append('bunch_phase')
            param_bounds.append(bounds.get('bunch_phase', (10, 30)))
            x0.append(initial_phase)  # Typical initial guess

        # RF frequency
        if 'rf_freq' in optimize_params:
            param_names.append('rf_freq')
            f_seo = initial_seo.frequency_hz
            param_bounds.append(bounds.get('rf_freq', (f_seo * 0.95, f_seo * 1.05)))
            x0.append(f_seo)

        # Initial radius
        if 'r0' in optimize_params:
            param_names.append('r0')
            param_bounds.append(bounds.get('r0', (r_mean - 0.010, r_mean + 0.010)))
            x0.append(r_mean)

        # Initial radial velocity
        if 'vr0' in optimize_params:
            param_names.append('vr0')
            param_bounds.append(bounds.get('vr0', (-5e4, 5e4)))
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

        # Convert spreads to meters
        r_spread = r_spread_mm / 1000.0
        vr_spread = vr_spread_m_s

        # Optimize
        start_time = time.time()

        if method == 'differential_evolution':
            result = differential_evolution(
                self.objective_function,
                param_bounds,
                args=(r_mean, v_tangential, r_spread, vr_spread, dt, max_turns, weights),
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
                args=(r_mean, v_tangential, r_spread, vr_spread, dt, max_turns, weights),
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
            r_mean_final = optimal_params[2]
        else:
            r_mean_final = r_mean

        if len(optimal_params) > 3:
            vr_mean_final = optimal_params[3]
        else:
            vr_mean_final = 0.0

        # Create final distribution
        initial_dist = self.create_initial_distribution(
            r_mean=r_mean_final,
            r_spread=r_spread,
            v_tangential=v_tangential,
            v_perp=vr_mean_final,
            vr_spread=vr_spread,
            distribution_type='gaussian'
        )

        # Track with progress bar
        if self.verbose:
            print("\nFinal tracking with optimal parameters...")

        result = self.track_with_rf_multiparticle(initial_dist, dt, max_turns, show_progress=True, save_full=True)
        success, turn_stats, rf_cross, traj_ref, poincare_all, std_r_steps, turn_ids, full_beam = result

        # Create result object
        optimized_orbit = OptimizedOrbit(
            success=success,
            final_energy_mev=turn_stats[-1].mean_energy_mev if len(turn_stats) > 0 else 0.0,
            n_turns=len(turn_stats),
            bunch_phase_deg=optimal_params[0],
            rf_frequency_mhz=optimal_params[1] / 1e6,
            initial_r_mm=r_mean_final * 1000,
            initial_vr_m_s=vr_mean_final,
            n_particles=self.n_particles,
            trajectory_reference=traj_ref,
            turn_statistics=turn_stats,
            poincare_points_all=poincare_all,
            rf_crossings=rf_cross,
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
                'r_spread_mm': r_spread_mm,
                'vr_spread_m_s': vr_spread_m_s,
                'envelope_oscillation_mm': np.std(std_r_steps) * 1000
            }
        )

        if self.verbose:
            print(f"\nOptimization summary:")
            print(f"  Final energy: {optimized_orbit.final_energy_mev:.3f} MeV")
            print(f"  Turns: {optimized_orbit.n_turns}")
            print(f"  Final radial spread: {turn_stats[-1].std_r * 1000:.3f} mm")
            print(f"  Envelope oscillation: {np.std(std_r_steps) * 1000:.3f} mm")

        return optimized_orbit, full_beam

    def _estimate_timestep(self, frequency_hz: float) -> float:
        """Estimate timestep from RF frequency."""
        period = 1.0 / frequency_hz
        dt = period / self.steps_per_turn
        return dt


if __name__ == "__main__":
    print("accelerated_orbit_finder_multiparticle.py - Multi-Particle Optimization")
    print("=" * 70)
    print("This module requires a CentralRegion design with RF cavities.")
    print("See examples for usage.")