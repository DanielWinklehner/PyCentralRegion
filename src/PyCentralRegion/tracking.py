"""
tracking.py - Core Tracking Engine with Strategy PatternZero-branch inner loop design: strategy selected at initialization.
Handles both single and multi-particle tracking in 2D/3D with/without RF.Part of: PyCentralRegion module
"""
import numpy as np
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass
from PyPATools.pusher import Pusher
from PyPATools.particles import ParticleDistribution
from PyPATools.global_variables import CLIGHT
from tqdm import tqdm


@dataclass
class TrackingResult:
    """Container for tracking results."""
    success: bool
    n_steps: int
    r_final: np.ndarray  # Final positions (n_particles, 3)
    v_final: np.ndarray  # Final velocities (n_particles, 3)
    active: np.ndarray  # Active particle mask (n_particles,)
    metadata: dict


class TrackingEngine:
    """
    Core tracking engine with strategy pattern.Strategy is selected at initialization based on dimensionality and RF usage.
    This eliminates ALL branches from the inner tracking loop.

    Parameters
    ----------
    design : CentralRegion
        Design with fields and RF cavities
    algorithm : str
        Pusher algorithm ('boris', 'rk4', 'rk4_rel', etc.)
    dimensionality : str
        '2D' or '3D'
    use_rf : bool
        Whether to apply RF kicks
    max_radius_m : float
        Maximum allowed radius for particle loss
    verbose : bool
        Print progress
    """


    def __init__(self,
                 design,
                 algorithm: str = 'rk4_rel',
                 dimensionality: str = '2D',
                 use_rf: bool = False,
                 max_radius_m: float = 0.5,
                 verbose: bool = True):
        self.design = design
        self.algorithm = algorithm
        self.dim = dimensionality
        self.use_rf = use_rf
        self.r_max = max_radius_m
        self.verbose = verbose

        # Create pusher
        self.pusher = Pusher(design.species, algorithm=algorithm)

        # Temporary ParticleDistribution for calculations
        self.pd_temp = ParticleDistribution(species=design.species)

        # Select strategy
        self._select_strategy()


    def _select_strategy(self):
        """Select tracking strategy based on configuration."""

        if self.dim == '2D' and self.use_rf:
            self._track_step = self._track_step_2d_rf
            self._check_boundary = self._check_boundary_2d

        elif self.dim == '2D' and not self.use_rf:
            self._track_step = self._track_step_2d_norf
            self._check_boundary = self._check_boundary_2d

        elif self.dim == '3D' and self.use_rf:
            self._track_step = self._track_step_3d_rf
            self._check_boundary = self._check_boundary_3d

        else:  # 3D without RF
            self._track_step = self._track_step_3d_norf
            self._check_boundary = self._check_boundary_3d


    def track_multiparticle(self,
                            pd_init: ParticleDistribution,
                            dt: float,
                            n_steps: int,
                            callback: Optional[Callable] = None,
                            callback_frequency: int = 1,
                            show_progress: bool = True) -> TrackingResult:
        """
        Track multiple particles with selected strategy.

        Parameters
        ----------
        pd_init : ParticleDistribution
            Initial particle distribution
        dt : float
            Timestep [s]
        n_steps : int
            Number of steps
        callback : callable, optional
            Function called every callback_frequency steps: callback(step, r, v, active, t)
        callback_frequency : int
            How often to call callback
        show_progress : bool
            Show progress bar

        Returns
        -------
        result : TrackingResult
            Final state and metadata
        """

        n_particles = pd_init.numpart

        # Initialize arrays
        r_array = pd_init.x_vec.copy()  # (n_particles, 3)
        v_array = pd_init.v_vec.copy()
        active = np.ones(n_particles, dtype=bool)

        t = 0.0

        # Progress bar
        pbar = tqdm(total=n_steps, disable=not show_progress, ncols=120,
                    desc="Tracking")

        # Boris initialization (half-step back)
        if self.pusher.algorithm.lower() == 'boris':
            r_array[active], v_array[active] = self.pusher.push_batch(
                r_array[active], v_array[active],
                self.design.efield, self.design.bfield, -0.5 * dt
            )

        # Main tracking loop - ONE function call, strategy already selected
        for step in range(n_steps):

            # Execute one step with selected strategy (NO BRANCHING)
            r_array, v_array, active = self._track_step(
                r_array, v_array, active, dt, t
            )

            t += dt

            # Callback (e.g., for diagnostics, Poincaré sections)
            if callback is not None and step % callback_frequency == 0:
                terminate = callback(step, r_array, v_array, active, t)

                if terminate:
                    pbar.close()
                    return TrackingResult(
                        success=True,
                        n_steps=step,
                        r_final=r_array,
                        v_final=v_array,
                        active=active,
                        metadata={'termination': 'turns_or_energy_reached', 'time': t}
                    )

            # Check if all particles lost
            if not np.any(active):
                if self.verbose:
                    print(f"\nAll particles lost at step {step}")
                pbar.close()
                return TrackingResult(
                    success=False,
                    n_steps=step,
                    r_final=r_array,
                    v_final=v_array,
                    active=active,
                    metadata={'termination': 'all_lost', 'time': t}
                )

            pbar.update(1)

        # Boris finalization (half-step forward)
        if self.pusher.algorithm.lower() == 'boris':
            r_array[active], v_array[active] = self.pusher.push_batch(
                r_array[active], v_array[active],
                self.design.efield, self.design.bfield, 0.5 * dt
            )

        pbar.close()

        return TrackingResult(
            success=True,
            n_steps=n_steps,
            r_final=r_array,
            v_final=v_array,
            active=active,
            metadata={'time': t}
        )


    # ========================================================================
    # Strategy implementations - NO BRANCHING INSIDE
    # ========================================================================

    def _track_step_2d_rf(self,
                          r_array: np.ndarray,
                          v_array: np.ndarray,
                          active: np.ndarray,
                          dt: float,
                          t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """2D tracking with RF cavities - optimized inner loop."""

        # Store previous positions for crossing checks
        r_prev = r_array.copy()

        # Push active particles
        r_array[active], v_array[active] = self.pusher.push_batch(
            r_array[active], v_array[active],
            self.design.efield, self.design.bfield, dt
        )

        # Apply RF kicks
        for cavity in self.design.rf_cavities:
            crossed_mask, t_cross, segment_ids = cavity.check_crossings_batch(
                r_prev[active], r_array[active]
            )

            if np.any(crossed_mask):
                # Map back to full array indices
                active_indices = np.where(active)[0]
                global_crossed = np.zeros(len(r_array), dtype=bool)
                global_crossed[active_indices[crossed_mask]] = True

                v_array, r_array, _, _ = cavity.apply_kicks_batch(
                    r_prev, r_array, v_array, global_crossed, t_cross,
                    t, dt, self.design, self.pusher
                )

        # Check boundaries (2D: radial loss)
        active = self._check_boundary(r_array, active)

        return r_array, v_array, active


    def _track_step_2d_norf(self,
                            r_array: np.ndarray,
                            v_array: np.ndarray,
                            active: np.ndarray,
                            dt: float,
                            t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """2D tracking without RF - pure field tracking."""

        # Push active particles (no RF logic at all)
        r_array[active], v_array[active] = self.pusher.push_batch(
            r_array[active], v_array[active],
            self.design.efield, self.design.bfield, dt
        )

        # Check boundaries
        active = self._check_boundary(r_array, active)

        return r_array, v_array, active


    def _track_step_3d_rf(self,
                          r_array: np.ndarray,
                          v_array: np.ndarray,
                          active: np.ndarray,
                          dt: float,
                          t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """3D tracking with RF - placeholder for future."""

        # TODO: Implement 3D tracking with RF
        # Same structure as 2D but with 3D boundary checks

        r_prev = r_array.copy()

        r_array[active], v_array[active] = self.pusher.push_batch(
            r_array[active], v_array[active],
            self.design.efield, self.design.bfield, dt
        )

        # RF logic (same as 2D)
        for cavity in self.design.rf_cavities:
            crossed_mask, t_cross, segment_ids = cavity.check_crossings_batch(
                r_prev[active], r_array[active]
            )

            if np.any(crossed_mask):
                active_indices = np.where(active)[0]
                global_crossed = np.zeros(len(r_array), dtype=bool)
                global_crossed[active_indices[crossed_mask]] = True

                v_array, r_array, _, _ = cavity.apply_kicks_batch(
                    r_prev, r_array, v_array, global_crossed, t_cross,
                    t, dt, self.design, self.pusher
                )

        # 3D boundaries
        active = self._check_boundary(r_array, active)

        return r_array, v_array, active


    def _track_step_3d_norf(self,
                            r_array: np.ndarray,
                            v_array: np.ndarray,
                            active: np.ndarray,
                            dt: float,
                            t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """3D tracking without RF - placeholder for future."""

        r_array[active], v_array[active] = self.pusher.push_batch(
            r_array[active], v_array[active],
            self.design.efield, self.design.bfield, dt
        )

        active = self._check_boundary(r_array, active)

        return r_array, v_array, active


    # ========================================================================
    # Boundary checks
    # ========================================================================

    def _check_boundary_2d(self, r_array: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Check radial boundary in 2D."""
        radii = np.sqrt(r_array[:, 0] ** 2 + r_array[:, 1] ** 2)
        lost = radii > self.r_max
        active[lost] = False
        return active


    def _check_boundary_3d(self, r_array: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Check boundaries in 3D (radial + vertical)."""
        # Radial check
        radii = np.sqrt(r_array[:, 0] ** 2 + r_array[:, 1] ** 2)
        lost_r = radii > self.r_max

        # Vertical check (example: |z| < 0.1 m)
        z_max = 0.1  # TODO: make configurable
        lost_z = np.abs(r_array[:, 2]) > z_max

        lost = lost_r | lost_z
        active[lost] = False
        return active

# ========================================================================
# Convenience function for single particle tracking
# ========================================================================
def track_single_particle(design,
                          r0: np.ndarray,
                          v0: np.ndarray,
                          dt: float,
                          n_steps: int,
                          algorithm: str = 'rk4_rel',
                          use_rf: bool = False,
                          callback: Optional[Callable] = None) -> TrackingResult:
    """
    Convenience wrapper for single particle tracking.Parameters
    ----------
    design : CentralRegion
        Design with fields
    r0 : np.ndarray
        Initial position [m]
    v0 : np.ndarray
        Initial velocity [m/s]
    dt : float
        Timestep [s]
    n_steps : int
        Number of steps
    algorithm : str
        Pusher algorithm
    use_rf : bool
        Apply RF kicks
    callback : callable, optional
        Called each step: callback(step, r, v, active, t)

    Returns
    -------
    result : TrackingResult
    """

    # Create single-particle distribution
    pd_init = ParticleDistribution(species=design.species)
    pd_init.x_vec = r0.reshape(1, 3)
    p_vec = v0 / np.sqrt(CLIGHT ** 2 - np.linalg.norm(v0) ** 2)
    pd_init.p_vec = p_vec.reshape(1, 3)

    # Create engine and track
    engine = TrackingEngine(
        design,
        algorithm=algorithm,
        dimensionality='2D',
        use_rf=use_rf,
        verbose=False
    )

    result = engine.track_multiparticle(
        pd_init,
        dt=dt,
        n_steps=n_steps,
        callback=callback,
        show_progress=False
    )

    return result
