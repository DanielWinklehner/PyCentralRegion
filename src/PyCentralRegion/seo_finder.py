"""
seo_finder.py - Static Equilibrium Orbit (SEO) Finder for Cyclotrons

Finds closed orbits using physically-motivated approach:
1. Define list of radii
2. Calculate average B-field on each radius
3. Calculate ideal particle energy from p = qBR
4. Launch ensemble of particles with (r, pr) variations
5. Use Poincare sections (theta=0 crossings) to check closure

Part of: PyCentralRegion module
Dependencies: PyPATools, scipy, numpy

Usage:
    from central_region import CentralRegion
    from seo_finder import SEOFinder

    design = CentralRegion.from_file('my_design.pkl')
    finder = SEOFinder(design)

    # Find SEOs at specific radii
    seos = finder.find_seos_at_radii(radii_mm=[50, 100, 150, 200])
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass, field
from PyPATools.pusher import Pusher
from PyPATools.field import Field
from PyPATools.particles import ParticleDistribution
from PyPATools.global_variables import CLIGHT
import warnings
import matplotlib.pyplot as plt
import pickle


@dataclass
class PoincarePoint:
    """
    Single point on Poincare section (theta=0 crossing).

    Attributes
    ----------
    turn : int
        Turn number
    r : float
        Radius [m]
    pr : float
        Radial momentum [kg*m/s]
    z : float
        Vertical position [m]
    pz : float
        Vertical momentum [kg*m/s]
    time : float
        Time [s]
    """
    turn: int
    r: float
    vr: float
    z: float
    vz: float
    time: float


@dataclass
class StaticOrbit:
    """
    Container for Static Equilibrium Orbit data.

    Attributes
    ----------
    radius_mm : float
        Nominal radius [mm]
    energy_kev : float
        Kinetic energy [keV]
    b_field_avg : float
        Average B-field at radius [T]
    r0 : np.ndarray
        Initial position [m] (x, y, z)
    v0 : np.ndarray
        Initial velocity [m/s] (vx, vy, vz)
    poincare_points : list
        List of PoincarePoint objects
    trajectory : np.ndarray
        Full orbit trajectory (if stored)
    is_closed : bool
        Whether orbit meets closure criterion
    closure_error_mm : float
        RMS closure error [mm]
    frequency_hz : float
        Orbital frequency [Hz]
    tune : float
        Betatron tune (radial oscillation frequency / orbital frequency)
    metadata : dict
        Additional information
    """
    radius_mm: float
    energy_kev: float
    b_field_avg: float
    r0: np.ndarray
    v0: np.ndarray
    poincare_points: List[PoincarePoint]
    trajectory: Optional[np.ndarray] = None
    is_closed: bool = False
    closure_error_mm: float = np.inf
    frequency_hz: float = 0.0
    tune: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __str__(self):
        return (f"StaticOrbit: R={self.radius_mm:.1f} mm, "
                f"E={self.energy_kev:.1f} keV, "
                f"B={self.b_field_avg:.4f} T, "
                f"f={self.frequency_hz / 1e6:.2f} MHz, "
                f"closed={self.is_closed}")


class SEOFinder:
    """
    Static Equilibrium Orbit finder using Poincare sections.

    Physical approach:
    1. Sample B-field on circle at given radius
    2. Calculate ideal energy from p = qBR
    3. Launch particle ensemble with (r, pr) variations
    4. Record Poincare sections at theta=0
    5. Check convergence of fixed point

    Parameters
    ----------
    design : CentralRegion
        Cyclotron design container
    n_turns : int
        Number of turns to track (default: 10)
    steps_per_turn : int
        Time steps per turn (default: 200)
    n_theta_samples : int
        Number of azimuthal samples for field averaging (default: 36)
    poincare_angle : float
        Angle for Poincare section [degrees] (default: 0.0)
    closure_tol_mm : float
        Position closure tolerance [mm] (default: 1.0)
    n_ensemble : int
        Number of particles in ensemble (default: 10)
    ensemble_spread_mm : float
        Radial spread of ensemble [mm] (default: 2.0)
    ensemble_spread_percent : float
        Momentum spread of ensemble [%] (default: 1.0)
    algorithm : str
        Pusher algorithm: 'boris', 'rk4', 'yoshida' (default: 'boris')
    verbose : bool
        Print progress messages (default: True)

    Examples
    --------
    > design = CentralRegion.from_file('cyclotron.pkl')
    > finder = SEOFinder(design, n_turns=10)
    > radii = np.linspace(50, 300, 10)  # mm
    > seos = finder.find_seos_at_radii(radii)
    """

    def __init__(self,
                 design,
                 n_turns: int = 10,
                 steps_per_turn: int = 200,
                 n_theta_samples: int = 360,
                 poincare_angle: float = 0.0,
                 closure_tol_mm: float = 1.0,
                 algorithm: str = 'boris',
                 verbose: bool = True):

        self.design = design
        self.n_turns = n_turns
        self.steps_per_turn = steps_per_turn
        self.n_theta_samples = n_theta_samples
        self.poincare_angle = np.deg2rad(poincare_angle)
        self.closure_tol_mm = closure_tol_mm
        self.algorithm = algorithm
        self.verbose = verbose
        self.pd = ParticleDistribution(species=design.species)

        # Validate design
        if not design.is_valid(verbose=False):
            raise ValueError("Design validation failed. Need at least bfield and species.")

        # Create pusher
        self.pusher = Pusher(design.species, algorithm=algorithm)

        # Cache for field calls
        self._zero_efield = Field.zero()

    def calculate_avg_field(self, radius_m: float, z: float = 0.0) -> Dict:
        """
        Calculate average B-field on circle at given radius.

        Samples field at n_theta_samples points around circle.

        Parameters
        ----------
        radius_m : float
            Radius [m]
        z : float
            Vertical position [m] (default: 0.0 for midplane)

        Returns
        -------
        field_info : dict
            'B_avg' : Average |B| [T]
            'Bz_avg' : Average Bz [T]
            'Br_avg' : Average radial component [T]
            'B_theta' : Array of |B| at each theta
            'flutter' : (max-min)/avg field variation
        """
        theta_array = np.linspace(0, 2 * np.pi, self.n_theta_samples, endpoint=False)

        x_array = radius_m * np.cos(theta_array)
        y_array = radius_m * np.sin(theta_array)
        z_array = np.full_like(x_array, z)

        pts = np.column_stack([x_array, y_array, z_array])

        # Get field
        bz = self.design.bfield(pts)[:, 2]

        # Calculate components
        # B_mag = np.mean(Bz)

        # Radial component (pointing outward from origin)
        # cos_theta = x_array / radius_m
        # sin_theta = y_array / radius_m
        # Br = Bx * cos_theta + By * sin_theta

        # Averages
        b_avg = np.mean(bz)
        # Bz_avg = np.mean(Bz)
        # Br_avg = np.mean(Br)

        # Flutter
        if b_avg > 0:
            flutter = (np.max(bz) - np.min(bz)) / b_avg
        else:
            flutter = 0.0

        return {
            'B_avg': b_avg,
            'Bz_avg': b_avg,
            'Br_avg': None,
            'B_theta': None,
            'flutter': flutter,
            'theta': theta_array
        }

    def calculate_ideal_energy(self, radius_m: float, B_field: float) -> float:
        """
        Calculate ideal kinetic energy for closed orbit at given radius.

        Uses cyclotron relation: p = q * B * R

        Parameters
        ----------
        radius_m : float
            Orbital radius [m]
        B_field : float
            Magnetic field [T]

        Returns
        -------
        energy_kev : float
            Kinetic energy [keV]
        """
        # q = abs(self.design.species.charge)
        # m = self.design.species.mass_kg
        #
        # # Momentum
        # p = q * B_field * radius_m  # kg*m/s
        #
        # # Relativistic energy: E^2 = (pc)^2 + (mc^2)^2
        # E_total = np.sqrt((p * self.c) ** 2 + (m * CLIGHT ** 2) ** 2)
        #
        # # Kinetic energy
        # E_kinetic = E_total - m * CLIGHT ** 2  # J
        #
        # # Convert to keV
        # energy_kev = E_kinetic / 1.602176634e-16

        return 1000.0 * self.pd.set_z_momentum_from_b_rho(B_field * radius_m)

    def _calculate_velocity_from_energy(self, energy_kev: float) -> float:
        """Calculate particle speed from kinetic energy (relativistic)."""
        energy_mev = energy_kev / 1000.0
        mass_mev = self.design.species.mass_mev

        gamma = energy_mev / mass_mev + 1.0
        beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
        velocity = beta * self.c

        return velocity

    def _estimate_timestep(self, radius_m: float, B_field: float) -> float:
        """
        Estimate timestep for tracking.

        Uses cyclotron period: T = 2*pi*m / (q*B)
        """
        q = abs(self.design.species.charge)
        m = self.design.species.mass_kg

        if B_field < 1e-6:
            return 1e-10

        # Cyclotron frequency
        omega = q * B_field / m
        T = 2.0 * np.pi / omega

        # Timestep
        dt = T / self.steps_per_turn

        return dt

    def _check_poincare_crossing(self,
                                 r_old: np.ndarray,
                                 r_new: np.ndarray) -> Tuple[bool, Optional[float]]:
        """
        Check if particle crosses Poincare section (theta = poincare_angle).

        Returns
        -------
        crossed : bool
            True if crossed in positive theta direction
        t_frac : float or None
            Fractional timestep where crossing occurred (0-1)
        """
        # Calculate angles
        theta_old = np.arctan2(r_old[1], r_old[0])
        theta_new = np.arctan2(r_new[1], r_new[0])

        # Unwrap angle (handle 2*pi discontinuity)
        dtheta = theta_new - theta_old
        if dtheta > np.pi:
            dtheta -= 2.0 * np.pi
        elif dtheta < -np.pi:
            dtheta += 2.0 * np.pi

        # Check if we crossed poincare_angle
        # We want positive crossing (counterclockwise motion)
        if dtheta > 0:
            # Check if poincare_angle is between theta_old and theta_new
            angle_diff_old = self.poincare_angle - theta_old
            angle_diff_new = self.poincare_angle - theta_new

            # Normalize to [-pi, pi]
            angle_diff_old = np.arctan2(np.sin(angle_diff_old), np.cos(angle_diff_old))
            angle_diff_new = np.arctan2(np.sin(angle_diff_new), np.cos(angle_diff_new))

            # Crossing occurs if signs differ and we're moving forward
            if angle_diff_old * angle_diff_new < 0 and angle_diff_old > 0:
                # Linear interpolation for crossing point
                t_frac = angle_diff_old / (angle_diff_old - angle_diff_new)
                return True, t_frac

        return False, None

    @staticmethod
    def _check_poincare_crossing_simple(r_old: np.ndarray,
                                        r_new: np.ndarray) -> Tuple[bool, Optional[float]]:
        """
        Check if particle crosses Poincare section (= +x-axis).

        Returns
        -------
        crossed : bool
            True if crossed in positive theta direction
        t_frac : float or None
            Fractional timestep where crossing occurred (0-1)
        """
        if r_old[1] <= 0.0 < r_new[1]:
            t_frac = r_old[1] / (r_old[1] - r_new[1])
            return True, t_frac

        return False, None

    def track_with_poincare(self,
                            r0: np.ndarray,
                            v0: np.ndarray,
                            dt: float,
                            n_turns: int) -> Tuple[List[PoincarePoint], np.ndarray, np.ndarray]:
        """
        Track particle and record Poincare sections.

        Parameters
        ----------
        r0 : np.ndarray
            Initial position [m]
        v0 : np.ndarray
            Initial velocity [m/s]
        dt : float
            Timestep [s]
        n_turns : int
            Number of turns to track

        Returns
        -------
        poincare_points : list
            List of PoincarePoint objects
        r_traj : np.ndarray
            Full trajectory positions
        v_traj : np.ndarray
            Full trajectory velocities
        """
        nsteps = n_turns * self.steps_per_turn

        # Storage
        r_traj = np.zeros((nsteps, 3))
        v_traj = np.zeros((nsteps, 3))
        poincare_points = []

        # Initialize
        r = r0.copy()
        v = v0.copy()
        t = 0.0
        turn = 0

        # Boris: half-step back initialization
        if self.pusher.algorithm == 'boris':
            ef = self.pusher._ensure_field_array(
                self._zero_efield(r.reshape(1, 3))
            )
            bf = self.pusher._ensure_field_array(
                self.design.bfield(r.reshape(1, 3))
            )
            _, v = self.pusher.push(r, v, ef, bf, -0.5 * dt)

        # Track
        for step in range(nsteps):
            r_prev = r.copy()
            v_prev = v.copy()

            # Get fields
            # ef = self.pusher._ensure_field_array(
            #     self._zero_efield(r.reshape(1, 3))
            # )
            # bf = self.pusher._ensure_field_array(
            #     self.design.bfield(r.reshape(1, 3))
            # )

            # Push
            r, v = self.pusher.push(r, v, self._zero_efield, self.design.bfield, dt)
            t += dt

            # Store
            r_traj[step] = r
            v_traj[step] = v

            # Check Poincare crossing
            crossed, t_frac = self._check_poincare_crossing_simple(r_prev, r)

            if crossed:
                # Interpolate position and velocity at crossing
                if t_frac is not None:
                    r_cross, v_cross = self.pusher.push(r_prev, v_prev,
                                                        self._zero_efield, self.design.bfield, t_frac*dt)
                else:
                    r_cross = r
                    v_cross = v

                # Record Poincare point
                poincare_points.append(PoincarePoint(
                    turn=turn,
                    r=r_cross[0],
                    vr=v_cross[0],  # Since our crossing point is the x-axis
                    z=r_cross[2],
                    vz=v_cross[2],
                    time=t - dt + t_frac * dt if t_frac else t
                ))

                # print(f"Crossed 'x-axis' at y = {1000.0 * r_cross[1]} mm")

                turn += 1

        # Boris: final half-step
        if self.pusher.algorithm == 'boris':
            ef = self.pusher._ensure_field_array(
                self._zero_efield(r.reshape(1, 3))
            )
            bf = self.pusher._ensure_field_array(
                self.design.bfield(r.reshape(1, 3))
            )
            _, v = self.pusher.push(r, v, ef, bf, 0.5 * dt)
            v_traj[-1] = v

        return poincare_points, r_traj, v_traj

    def find_seo_at_radius(self, radius_mm: float, n_iterations: int = 3) -> StaticOrbit:
        """
        Find static equilibrium orbit at given radius using iterative refinement.

        Parameters
        ----------
        radius_mm : float
            Radius [mm]
        n_iterations : int
            Number of refinement iterations (default: 3)

        Returns
        -------
        orbit : StaticOrbit
            Orbit information with Poincare analysis
        """
        radius_m = radius_mm / 1000.0

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"Finding SEO at R = {radius_mm:.1f} mm ({n_iterations} iterations)")
            print(f"{'=' * 70}")

        # Step 1: Calculate average field at this radius
        field_info = self.calculate_avg_field(radius_m)
        B_avg = field_info['B_avg']

        if self.verbose:
            print(f"Average field: B = {B_avg:.4f} T")
            print(f"Field flutter: {field_info['flutter'] * 100:.2f}%")

        # Step 2: Calculate ideal energy
        energy_kev = self.calculate_ideal_energy(radius_m, B_avg)

        if self.verbose:
            print(f"Ideal energy: E = {energy_kev:.2f} keV ({energy_kev / 1000:.3f} MeV)")

        # Step 3: Estimate timestep
        dt = self._estimate_timestep(radius_m, B_avg)
        period_est = dt * self.steps_per_turn

        if self.verbose:
            print(f"Timestep: {dt * 1e12:.3f} ps")
            print(f"Est. period: {period_est * 1e9:.3f} ns ({1.0 / period_est / 1e6:.2f} MHz)")

        # Set up ParticleDistribution (velocities are set from B-rho above, y, z are 0 by default)
        self.pd.x = np.array([radius_m])

        for iteration in range(n_iterations):
            if self.verbose:
                print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
                print(f"Starting point: R={self.pd.x[0]  * 1000:.3f} mm, E={1000.0 * self.pd.mean_energy_mev:.2f} keV")

            # Coordinate system: PyPATools uses z as longitudinal (linac convention)
            # but cyclotrons use y as azimuthal direction. Swap y<->z here.
            poincare, r_traj, v_traj = self.track_with_poincare(
                np.array([self.pd.x[0], self.pd.y[0], self.pd.z[0]]),
                np.array([self.pd.vx[0], self.pd.vz[0], self.pd.vy[0]]),
                dt, self.n_turns)

            all_radii = []
            all_vr = []

            for p in poincare:
                all_radii.append(p.r)
                all_vr.append(p.vr)

            if len(all_radii) < 5:
                if self.verbose:
                    print("WARNING: Not enough Poincare points, using current guess")
                break

            # Calculate centroid (fixed point of Poincare map)
            r_center = np.mean(all_radii)
            vr_center = np.mean(all_vr)

            if self.verbose:
                print(f"Poincare contour center: R={r_center * 1000:.3f} mm, vr={vr_center:.3e} m/s")

            # Calculate convergence
            radius_shift = abs(r_center - self.pd.x[0]) * 1000  # mm

            if self.verbose:
                print(f"Radius shift: {radius_shift:.4f} mm")

            # Update for next iteration
            self.pd.x = np.array([r_center])

            # Because we start on the x-axis, pr = px, p_long = py. But we also need to keep
            # in mind that ParticleDistribution uses z as s and here we use y as s.
            # In 2D, we do self.pd.pz = np.sqrt(mean momentum^2 - pr^2) and px = pr, py = 0
            vz = np.array([np.sqrt(self.pd.v_mean_m_per_s**2 - vr_center**2.0)])
            vx = np.array([vr_center])

            self.pd.set_p_from_v(vx, np.zeros(1), vz)

            # Check convergence
            closure_error_m = np.std(all_radii)
            closure_error_mm = closure_error_m * 1000.0

            is_closed = closure_error_mm < self.closure_tol_mm

            if iteration > 0 and is_closed:
            # if iteration > 0 and radius_shift < 0.1:  # 0.1 mm convergence
                if self.verbose:
                    print(f"Converged! (closure_error < {self.closure_tol_mm} mm)")
                break

        # Final tracking with best parameters (25 turns for full contour)
        if self.verbose:
            print(f"\nFinal tracking with 25 turns for full contour mapping...")

        # Coordinate system: PyPATools uses z as longitudinal (linac convention)
        # but cyclotrons use y as azimuthal direction. Swap y<->z here.
        poincare, r_traj, v_traj = self.track_with_poincare(
            np.array([self.pd.x[0], self.pd.y[0], self.pd.z[0]]),
            np.array([self.pd.vx[0], self.pd.vz[0], self.pd.vy[0]]),  # Need to switch from linac coords to cyclotron coords
            dt, 25)

        best_poincare = poincare
        best_traj_r, best_traj_v = r_traj, v_traj

        all_radii = []
        all_vr = []

        for p in poincare:
            all_radii.append(p.r)
            all_vr.append(p.vr)

        closure_error_m = np.std(all_radii)
        closure_error_mm = closure_error_m * 1000.0

        is_closed = closure_error_mm < self.closure_tol_mm

        # Calculate frequency from Poincare crossings
        if len(best_poincare) >= 2:
            times = np.array([p.time for p in best_poincare])
            periods = np.diff(times)
            avg_period = np.mean(periods)
            frequency = 1.0 / avg_period if avg_period > 0 else 0.0
        else:
            frequency = 0.0
            avg_period = 0.0

        # Calculate tune (number of betatron oscillations per turn)
        tune = 0.0
        if len(best_poincare) >= 20:
            # FFT of radial oscillation to find betatron frequency
            radii = np.array([p.r for p in best_poincare])
            radii_detrended = radii - np.mean(radii)

            # Simple zero-crossing count method
            crossings = np.where(np.diff(np.sign(radii_detrended)))[0]
            if len(crossings) >= 4:
                # Half-period per crossing
                betatron_periods = len(best_poincare) / (len(crossings) / 2.0)
                tune = 1.0 / betatron_periods

        if self.verbose:
            print(f"\nFinal Results:")
            print(f"  Refined radius: {self.pd.x[0] * 1000:.3f} mm")
            print(f"  Refined energy: {self.pd.mean_energy_mev * 1000:.2f} keV")
            print(f"  Poincare crossings: {len(best_poincare)}")
            print(f"  Closure error (std): {closure_error_mm:.3f} mm")
            print(f"  Closed: {is_closed}")
            print(f"  Frequency: {frequency / 1e6:.3f} MHz")
            print(f"  Period: {avg_period * 1e9:.3f} ns")
            print(f"  Tune (estimate): {tune:.4f}")

        # Create orbit object
        orbit = StaticOrbit(
            radius_mm=radius_mm,  # Keep nominal radius
            energy_kev=self.pd.mean_energy_mev * 1000,  # Use refined energy
            b_field_avg=B_avg,
            r0=np.array([self.pd.x[0], self.pd.y[0], self.pd.z[0]]),
            v0=np.array([self.pd.vx[0], self.pd.vz[0], self.pd.vy[0]]),
            poincare_points=best_poincare,
            trajectory=best_traj_r,
            is_closed=is_closed,
            closure_error_mm=closure_error_mm,
            frequency_hz=frequency,
            tune=tune,
            metadata={
                'field_info': field_info,
                'timestep_s': dt,
                'n_iterations': n_iterations,
                'all_poincare': [poincare],
                'refined_radius_m': self.pd.x[0],
                'poincare_center_r': r_center,
                'poincare_center_vr': vr_center
            }
        )

        return orbit

    def find_seos_at_radii(self, radii_mm: List[float], n_iterations: int = 5) -> List[StaticOrbit]:
        """
        Find SEOs at multiple radii with iterative refinement.

        Parameters
        ----------
        radii_mm : list
            List of radii [mm]
        n_iterations : int
            Number of refinement iterations per radius (default: 3)

        Returns
        -------
        orbits : list
            List of StaticOrbit objects
        """
        orbits = []

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"SCANNING {len(radii_mm)} RADII with {n_iterations} iterations each")
            print(f"{'=' * 70}")

        for i, radius in enumerate(radii_mm):
            if self.verbose:
                print(f"\n[{i + 1}/{len(radii_mm)}]", end=" ")

            orbit = self.find_seo_at_radius(radius, n_iterations=n_iterations)
            orbits.append(orbit)

        if self.verbose:
            n_closed = sum(1 for o in orbits if o.is_closed)
            print(f"\n{'=' * 70}")
            print(f"SCAN COMPLETE: {n_closed}/{len(orbits)} closed orbits")
            print(f"{'=' * 70}")

        return orbits

    @staticmethod
    def plot_poincare_section(orbit: StaticOrbit, ax=None):
        """
        Plot Poincare section (r vs pr).

        Parameters
        ----------
        orbit : StaticOrbit
            Orbit with Poincare points
        ax : matplotlib.Axes, optional
            Axes to plot on
        """


        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Plot best particle
        r_vals = [p.r * 1000 for p in orbit.poincare_points]  # mm
        vr_vals = [p.vr for p in orbit.poincare_points]

        ax.plot(r_vals, vr_vals, 'o-', markersize=6, linewidth=1,
                label='Best particle', zorder=3)
        ax.set_xlabel('Radius r (mm)')
        ax.set_ylabel('Radial Velocity vr (m/s)')
        ax.set_title(f'Poincare Section: R={orbit.radius_mm:.1f} mm, '
                     f'E={orbit.energy_kev:.1f} keV')
        ax.grid(True, alpha=0.3)
        ax.legend()

        return ax


# ============================================================================
# Utility functions
# ============================================================================

def save_seo_database(orbits: List[StaticOrbit], filename: str):
    """Save list of orbits to file."""

    with open(filename, 'wb') as f:
        pickle.dump(orbits, f)
    print(f"Saved {len(orbits)} orbits to {filename}")


def load_seo_database(filename: str) -> List[StaticOrbit]:
    """Load orbits from file."""

    with open(filename, 'rb') as f:
        orbits = pickle.load(f)
    print(f"Loaded {len(orbits)} orbits from {filename}")
    return orbits


def analyze_isochronism(orbits: List[StaticOrbit]) -> Dict:
    """
    Analyze isochronism of cyclotron from SEO data.

    Parameters
    ----------
    orbits : list
        List of StaticOrbit objects

    Returns
    -------
    analysis : dict
        'frequencies_mhz' : array of frequencies
        'energies_mev' : array of energies
        'freq_variation_percent' : frequency spread
        'is_isochronous' : bool (< 1% variation)
    """
    freqs = np.array([o.frequency_hz / 1e6 for o in orbits if o.frequency_hz > 0])
    energies = np.array([o.energy_kev / 1000.0 for o in orbits if o.frequency_hz > 0])

    if len(freqs) == 0:
        return {
            'frequencies_mhz': np.array([]),
            'energies_mev': np.array([]),
            'freq_variation_percent': np.inf,
            'is_isochronous': False
        }

    freq_avg = np.mean(freqs)
    freq_std = np.std(freqs)
    freq_variation = (freq_std / freq_avg * 100.0) if freq_avg > 0 else np.inf

    is_isochronous = freq_variation < 1.0  # Less than 1% variation

    return {
        'frequencies_mhz': freqs,
        'energies_mev': energies,
        'freq_avg_mhz': freq_avg,
        'freq_std_mhz': freq_std,
        'freq_variation_percent': freq_variation,
        'is_isochronous': is_isochronous
    }


if __name__ == "__main__":
    print("seo_finder.py - Static Equilibrium Orbit Finder (Poincare Method)")
    print("=" * 70)
    print("This module requires a CentralRegion design to run.")
    print("See examples/01_find_static_orbits.py for usage.")
    print()
    print("Key features:")
    print("  - Radius-based approach: specify radii, calculate ideal energies")
    print("  - Poincare sections for convergence analysis")
    print("  - Ensemble tracking with (r, pr) variations")
    print("  - Automatic field averaging on circles")
    print("  - Isochronism analysis")
