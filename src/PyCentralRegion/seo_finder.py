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
from PyPATools.trackers import Tracker, Recorder
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
    bz_avg: float


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


class _PoincareTrajRecorder(Recorder):
    """Reproduces SEOFinder.track_with_poincare exactly as a Tracker recorder:
    stores the full trajectory, accumulates the path-averaged Bz, and records a
    PoincarePoint at every +x-axis (y: -ve -> +ve) crossing via re-push
    interpolation. Numerics are identical to the legacy hand-rolled loop."""

    def __init__(self, finder, r0, v0, nsteps, dt):
        self.pusher = finder.pusher
        self.ef = finder._zero_efield
        self.bf = finder.design.bfield
        self.dt = dt
        self.r_traj = np.zeros((nsteps, 3))
        self.v_traj = np.zeros((nsteps, 3))
        self.points = [PoincarePoint(turn=0, r=r0[0], vr=v0[0],
                                     z=r0[2], vz=v0[2], time=0.0, bz_avg=0.0)]
        self.turn = 0
        self.bz_accum = 0.0
        self.steps_taken = 0

    def record(self, step, r_prev, v_prev, r, v, active, t):
        self.steps_taken += 1
        rp = r[0]
        self.r_traj[step] = rp
        self.v_traj[step] = v[0]

        ro = r_prev[0]
        if ro[1] < 0.0 < rp[1]:                       # +x-axis crossing (legacy rule)
            t_frac = ro[1] / (ro[1] - rp[1])
            r_cross, v_cross = self.pusher.push(ro, v_prev[0], self.ef, self.bf,
                                                t_frac * self.dt)
            self.bz_accum += self.bf(r_cross.reshape(1, 3))[0][2]
            self.points.append(PoincarePoint(
                turn=self.turn, r=r_cross[0], vr=v_cross[0],
                z=r_cross[2], vz=v_cross[2],
                time=t - self.dt + t_frac * self.dt,
                bz_avg=self.bz_accum / self.steps_taken))
            self.turn += 1
            self.bz_accum = 0.0
            self.steps_taken = 0
        else:
            self.bz_accum += self.bf(rp.reshape(1, 3))[0][2]
        return None

    def finalize(self, r, v, active, t):
        self.v_traj[-1] = v[0]


class _AngleCrossRecorder(Recorder):
    """Stops at the first increasing crossing of polar angle theta_target and
    stores the interpolated (r_cross, v_cross, t_cross). Reproduces the legacy
    SEOFinder._track_to_angle crossing logic exactly."""

    stop_reason = "angle_crossing"

    def __init__(self, finder, theta_target, dt):
        self.pusher = finder.pusher
        self.ef = finder._zero_efield
        self.bf = finder.design.bfield
        self.boris = (self.pusher.algorithm == 'boris')
        self.theta = theta_target
        self.dt = dt
        self.result = None

    def _offset(self, rr):
        d = np.arctan2(rr[1], rr[0]) - self.theta
        return np.arctan2(np.sin(d), np.cos(d))      # wrap to (-pi, pi]

    def record(self, step, r_prev, v_prev, r, v, active, t):
        d_prev = self._offset(r_prev[0])
        d_new = self._offset(r[0])
        if d_prev < 0.0 <= d_new and (d_new - d_prev) < np.pi:
            t_frac = -d_prev / (d_new - d_prev) if d_new != d_prev else 1.0
            r_c, v_c = self.pusher.push(r_prev[0], v_prev[0], self.ef, self.bf,
                                        t_frac * self.dt)
            if self.boris:                            # de-stagger velocity at crossing
                _, v_c = self.pusher.push(r_c, v_c, self.ef, self.bf, 0.5 * self.dt)
            self.result = (r_c, v_c, t - self.dt + t_frac * self.dt)
            return True
        return None


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
                 solver: str = 'newton',
                 symmetry_half_angle_deg: float = 45.0,
                 verbose: bool = True,
                 polar_seed: bool = True):

        self.design = design
        self.n_turns = n_turns
        self.steps_per_turn = steps_per_turn
        self.n_theta_samples = n_theta_samples
        self.poincare_angle = np.deg2rad(poincare_angle)
        self.closure_tol_mm = closure_tol_mm
        self.algorithm = algorithm
        # SEO solver: 'newton' (full-turn 2D Newton fixed point of the Cartesian
        # tracker), 'symmetric' (mirror-plane shooting; assumes the tracked field
        # is symmetric about theta=0 and theta=symmetry_half_angle), 'polar'
        # (Gordon closed-orbit integrator in polar coordinates on the regular-grid
        # map, see closed_orbit.py -- ~100x faster than 'newton', identical
        # orbit to ~1e-6), or 'centroid' (legacy averaging).
        self.solver = solver
        # Seed the Cartesian Newton solver with the polar closed orbit. Without
        # it the Newton iteration, started from the circle radius, can land on a
        # degenerate off-centre orbit where nu_r dips towards 1 (pole edge).
        self.polar_seed = polar_seed
        self._eo_field = None          # lazily built CartesianMidplane
        self.symmetry_half_angle = np.deg2rad(symmetry_half_angle_deg)
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

        return 1000.0 * self.pd.set_z_momentum_from_b_rho(abs(B_field) * radius_m)

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

        if abs(B_field) < 1e-6:
            return 1e-10

        # Cyclotron frequency
        omega = q * B_field / m
        T = 2.0 * np.pi / omega

        # Timestep
        dt = T / self.steps_per_turn

        return abs(dt)

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
        # Strict '< 0' on the old point: the orbit is launched exactly on the section
        # (y = 0), so a '<= 0' test would register a spurious crossing on the very first
        # step. Excluding it makes poincare[1] the first *real* turn, so its bz_avg is a
        # true path average over one revolution (not a single-azimuth sample).
        if r_old[1] < 0.0 < r_new[1]:
            t_frac = r_old[1] / (r_old[1] - r_new[1])
            return True, t_frac

        return False, None

    def _make_pd(self, pos, vel) -> ParticleDistribution:
        """Build a single-particle ParticleDistribution (lab frame) for the Tracker."""
        pos = np.asarray(pos, dtype=float).reshape(1, 3)
        pd = ParticleDistribution(species=self.design.species,
                                  x_vec=pos.copy(), p_vec=np.zeros((1, 3)))
        pd.set_p_from_v_vec(np.asarray(vel, dtype=float).reshape(1, 3))
        return pd

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

        # Centralized tracking: the integration loop, Boris staggering and the
        # alive mask all live in PyPATools.trackers.Tracker. The Poincare section
        # detection, path-averaged Bz and trajectory storage are provided by a
        # recorder hook that reproduces the legacy loop exactly.
        pd = self._make_pd(r0, v0)
        recorder = _PoincareTrajRecorder(self, np.asarray(r0), np.asarray(v0), nsteps, dt)
        tracker = Tracker(self.pusher, self._zero_efield, self.design.bfield,
                          recorders=[recorder])
        tracker.run(pd, dt, nsteps, record_every=1, show_progress=False, sync_back=False)

        return recorder.points, recorder.r_traj, recorder.v_traj

    # ==================================================================
    # Gordon-seeded fixed-point (Newton) equilibrium-orbit solver.
    # All tracking goes through the PyPATools pusher (self.pusher); this
    # finds the EXACT closed orbit (the fixed point of the one-turn map),
    # so there is no residual betatron oscillation -- unlike the legacy
    # centroid-averaging finder in find_seo_at_radius.
    # ==================================================================
    def _polar_field(self):
        """CartesianMidplane view of the design's regular-grid midplane map
        (None when the field is not a regular 2D grid map)."""
        if self._eo_field is None:
            from .closed_orbit import CartesianMidplane
            try:
                self._eo_field = CartesianMidplane.from_field(self.design.bfield)
            except (ValueError, AttributeError, TypeError, KeyError):
                self._eo_field = False
        return self._eo_field or None

    def _polar_closed_orbit(self, energy_kev: float, radius_m: float):
        """Closed orbit at fixed energy from the polar integrator.

        Returns the closed_orbits() dict (single entry) or None when the map is
        not a regular grid or the orbit does not close.
        """
        fld = self._polar_field()
        if fld is None:
            return None
        from .closed_orbit import closed_orbits, azimuthal_stats
        # Seed at the scalloped hill radius r (1 + a_N / (N^2 - 1)), not at the
        # nominal radius: near the pole edge a seed 5-8 cm inside the centred
        # orbit converges onto a DISPLACED closed orbit of the nu_r = 1 family
        # (tune 0, ~30 kHz off). Such orbits are flagged and rejected.
        st = azimuthal_stats(fld, [radius_m])
        x = float(st['a_dom'][0]) / max(st['n_dom'] ** 2 - 1.0, 1.0)
        for r_seed in (radius_m * (1.0 + x), radius_m):
            o = closed_orbits(fld, [energy_kev * 1e-3], self.design.species, [r_seed],
                              n_steps=3600)
            if o['converged'][0] and not o['displaced'][0]:
                return o
        return None

    def _v_from_energy_kev(self, energy_kev: float) -> float:
        """Particle speed [m/s] from kinetic energy [keV] (relativistic)."""
        gamma = energy_kev / 1000.0 / self.design.species.mass_mev + 1.0
        beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
        return beta * CLIGHT

    @staticmethod
    def _launch_state(r: float, vr: float, v_total: float):
        """Tracker (lab) (pos, vel) at theta=0 (on +x axis) with radial velocity vr."""
        v_az = np.sqrt(max(v_total ** 2 - vr ** 2, 0.0))
        return np.array([r, 0.0, 0.0]), np.array([vr, v_az, 0.0])

    @staticmethod
    def _radial_velocity(r_vec, v_vec) -> float:
        """Radial velocity component v . r_hat in the median plane."""
        rho = np.hypot(r_vec[0], r_vec[1])
        return 0.0 if rho == 0.0 else (v_vec[0] * r_vec[0] + v_vec[1] * r_vec[1]) / rho

    def _track_to_angle(self, pos, vel, dt, theta_target, max_steps):
        """Track (PyPATools pusher) to the next increasing crossing of polar angle theta_target.

        Returns (r_cross, v_cross, t_cross) in the tracker (lab) frame, or None.
        """
        pd = self._make_pd(pos, vel)
        recorder = _AngleCrossRecorder(self, theta_target, dt)
        tracker = Tracker(self.pusher, self._zero_efield, self.design.bfield,
                          recorders=[recorder])
        tracker.run(pd, dt, int(max_steps), record_every=1, show_progress=False,
                    sync_back=False)
        return recorder.result

    def _one_turn(self, r, vr, v_total, dt, max_steps):
        """One-turn map at theta=0: (r, vr) -> (r', vr', turn_time). None if it fails."""
        pos, vel = self._launch_state(r, vr, v_total)
        res = self._track_to_angle(pos, vel, dt, 0.0, max_steps)
        if res is None:
            return None
        r_c, v_c, t = res
        return r_c[0], v_c[0], t   # at theta=0: radial = x, vr = vx

    def _one_turn_jacobian(self, r, vr, v_total, dt, max_steps):
        """2x2 finite-difference Jacobian of the one-turn map d(r1,vr1)/d(r0,vr0)."""
        base = self._one_turn(r, vr, v_total, dt, max_steps)
        if base is None:
            return None
        r1, vr1, t_turn = base
        d_r = 1.0e-5                          # 10 um
        d_vr = max(1.0e-3 * v_total, 1.0)
        pr = self._one_turn(r + d_r, vr, v_total, dt, max_steps)
        pv = self._one_turn(r, vr + d_vr, v_total, dt, max_steps)
        if pr is None or pv is None:
            return None
        col_r = np.array([pr[0] - r1, pr[1] - vr1]) / d_r
        col_v = np.array([pv[0] - r1, pv[1] - vr1]) / d_vr
        return np.column_stack([col_r, col_v]), r1, vr1, t_turn

    def _newton_full_turn(self, r0, v_total, dt, max_steps, max_iter=15, vr0=0.0):
        """2D Newton on (r, vr) for the one-turn fixed point.

        Returns (r, vr, t_turn, M, residual_m) or None.
        """
        x = np.array([r0, vr0])               # default seed vr = 0 (symmetry-plane crossing)
        M = r1 = vr1 = t_turn = None
        for _ in range(max_iter):
            jac = self._one_turn_jacobian(x[0], x[1], v_total, dt, max_steps)
            if jac is None:
                return None
            M, r1, vr1, t_turn = jac
            F = np.array([r1 - x[0], vr1 - x[1]])
            if np.hypot(F[0], F[1] * t_turn) < 1.0e-7:   # length-scaled closure
                break
            try:
                x = x - np.linalg.solve(M - np.eye(2), F)
            except np.linalg.LinAlgError:
                break
        residual = np.hypot(r1 - x[0], (vr1 - x[1]) * t_turn) if r1 is not None else np.inf
        return x[0], x[1], t_turn, M, residual

    def _shoot_symmetric(self, r0, v_total, dt, theta_mirror, max_steps, max_iter=40):
        """1D secant on r: launch vr=0 at theta=0, find r so vr=0 at theta_mirror.

        Returns (r, t_segment) -- t_segment is the time to reach theta_mirror -- or None.
        """
        def vr_at_mirror(r):
            pos, vel = self._launch_state(r, 0.0, v_total)
            res = self._track_to_angle(pos, vel, dt, theta_mirror, max_steps)
            if res is None:
                return None
            r_c, v_c, t = res
            return self._radial_velocity(r_c, v_c), t

        ra, rb = r0 - 1.0e-4, r0 + 1.0e-4
        fa, fb = vr_at_mirror(ra), vr_at_mirror(rb)
        if fa is None or fb is None:
            return None
        fa, fb = fa[0], fb[0]
        t_seg = None
        for _ in range(max_iter):
            if fb == fa:
                break
            rc = rb - fb * (rb - ra) / (fb - fa)
            res = vr_at_mirror(rc)
            if res is None:
                return None
            fc, t_seg = res
            ra, fa, rb, fb = rb, fb, rc, fc
            if abs(fc) < 1.0e-3 and abs(rb - ra) < 1.0e-8:
                break
        return rb, t_seg

    def find_seo_newton(self, radius_mm: float, energy_seed_kev: Optional[float] = None,
                        solver: Optional[str] = None, do_final_tracking: bool = True) -> StaticOrbit:
        """Find the SEO at a radius via fixed-point (Newton) solving.

        Parameters
        ----------
        radius_mm : float
            Target radius [mm].
        energy_seed_kev : float, optional
            Seed kinetic energy [keV] (e.g. from the Gordon method). If None, the
            energy is seeded from the rigidity p = q*<B>*R at this radius.
        solver : str, optional
            'newton', 'symmetric' or 'polar' (defaults to self.solver).
        do_final_tracking : bool
            Track 25 turns from the converged state for the stored trajectory.
            With solver='polar' and do_final_tracking=False no tracking is done
            at all (closure/frequency/tune come from the polar integrator).
        """
        solver = solver or self.solver
        radius_m = radius_mm / 1000.0

        field_info = self.calculate_avg_field(radius_m)
        B_avg = field_info['B_avg']
        energy_kev = (energy_seed_kev if energy_seed_kev is not None
                      else self.calculate_ideal_energy(radius_m, B_avg))
        v_total = self._v_from_energy_kev(energy_kev)
        dt = self._estimate_timestep(radius_m, B_avg)

        if self.verbose:
            print(f"\n{'=' * 70}")
            print(f"Newton SEO at R={radius_mm:.1f} mm (solver={solver}), seed E={energy_kev:.2f} keV"
                  + ("  [Gordon-seeded]" if energy_seed_kev is not None else ""))
            print(f"{'=' * 70}")

        polar = None
        if solver == 'polar' or (solver == 'newton' and self.polar_seed):
            polar = self._polar_closed_orbit(energy_kev, radius_m)
            if polar is None and solver == 'polar':
                raise RuntimeError(f"Polar closed-orbit solve failed at R={radius_mm:.1f} mm "
                                   "(no regular-grid 2D map, or no closed orbit)")

        if solver == 'polar':
            r_star = float(polar['r0'][0])
            vr_star = float(v_total * polar['pr0'][0] / polar['brho'][0])
            t_turn = float(polar['T_rev'][0])
            M = None
            nu_r = float(polar['nu_r'][0])
            tune = (abs(nu_r - round(nu_r)) if np.isfinite(nu_r) else 0.0)
            frequency = 1.0 / t_turn
            if not do_final_tracking:
                pos, vel = self._launch_state(r_star, vr_star, v_total)
                if self.verbose:
                    print(f"  polar: r* = {r_star * 1000:.4f} mm, f = {frequency / 1e6:.4f} MHz, "
                          f"nu_r = {nu_r:.4f}, nu_z = {float(polar['nu_z'][0]):.4f}")
                return StaticOrbit(
                    radius_mm=radius_mm, energy_kev=energy_kev,
                    b_field_avg=float(polar['B_avg_orbit'][0]),
                    r0=pos, v0=vel, poincare_points=[], trajectory=None,
                    is_closed=True, closure_error_mm=float(polar['residual_m'][0]) * 1e3,
                    frequency_hz=frequency, tune=tune,
                    metadata={'solver': 'polar', 'field_info': field_info,
                              'turn_time_s': t_turn, 'transfer_matrix': None,
                              'energy_seeded': energy_seed_kev is not None,
                              'refined_radius_m': r_star, 'refined_vr_m_s': vr_star,
                              'nu_r': nu_r, 'nu_z': float(polar['nu_z'][0]),
                              'nu_z_sq': float(polar['nu_z_sq'][0]),
                              'r_mean_m': float(polar['r_mean'][0]),
                              'r_max_m': float(polar['r_max'][0])})
        elif solver == 'symmetric':
            n_seg = max(1, int(round(2.0 * np.pi / self.symmetry_half_angle)))
            max_steps = int(self.steps_per_turn / n_seg * 2 + 20)
            sol = self._shoot_symmetric(radius_m, v_total, dt, self.symmetry_half_angle, max_steps)
            if sol is None:
                raise RuntimeError(f"Symmetric SEO shoot failed at R={radius_mm:.1f} mm")
            r_star, t_seg = sol
            vr_star = 0.0
            t_turn = n_seg * t_seg
            jac = self._one_turn_jacobian(r_star, vr_star, v_total, dt, int(self.steps_per_turn * 1.5))
            M = jac[0] if jac is not None else None
        else:  # 'newton'
            max_steps = int(self.steps_per_turn * 1.5)
            r_seed, vr_seed = radius_m, 0.0
            if polar is not None:
                r_seed = float(polar['r0'][0])
                vr_seed = float(v_total * polar['pr0'][0] / polar['brho'][0])
            sol = self._newton_full_turn(r_seed, v_total, dt, max_steps, vr0=vr_seed)
            if sol is None:
                raise RuntimeError(f"Newton SEO solve failed at R={radius_mm:.1f} mm")
            r_star, vr_star, t_turn, M, _ = sol

        if solver != 'polar':
            tune = (np.arccos(np.clip(np.trace(M) / 2.0, -1.0, 1.0)) / (2.0 * np.pi)
                    if M is not None else 0.0)
        frequency = 1.0 / t_turn if t_turn else 0.0

        # Final tracking from the converged state (clean closed orbit -> tiny std).
        n_final = 25 if do_final_tracking else max(self.n_turns, 2)
        pos, vel = self._launch_state(r_star, vr_star, v_total)
        poincare, r_traj, v_traj = self.track_with_poincare(pos, vel, dt, n_final)

        all_radii = [p.r for p in poincare]
        closure_error_mm = float(np.std(all_radii) * 1000.0)
        is_closed = closure_error_mm < self.closure_tol_mm

        if self.verbose:
            print(f"  r* = {r_star * 1000:.4f} mm, vr* = {vr_star:.3e} m/s")
            print(f"  f = {frequency / 1e6:.4f} MHz, nu_r = {tune:.4f}, "
                  f"closure(std) = {closure_error_mm:.4e} mm, closed = {is_closed}")

        return StaticOrbit(
            radius_mm=radius_mm,
            energy_kev=energy_kev,
            # Path-averaged field over the first full turn (poincare[1] is now the first
            # real theta=0 crossing, not the launch point).
            b_field_avg=poincare[1].bz_avg if len(poincare) > 1 else B_avg,
            r0=pos.copy(),
            v0=vel.copy(),
            poincare_points=poincare,
            trajectory=r_traj,
            is_closed=is_closed,
            closure_error_mm=closure_error_mm,
            frequency_hz=frequency,
            tune=tune,
            metadata={
                'solver': solver,
                'field_info': field_info,
                'timestep_s': dt,
                'turn_time_s': t_turn,
                'transfer_matrix': M,
                'energy_seeded': energy_seed_kev is not None,
                'refined_radius_m': r_star,
                'refined_vr_m_s': vr_star,
                'polar_seeded': polar is not None,
            },
        )

    def find_seo_at_radius(self, radius_mm: float, n_iterations: int = 3, do_final_tracking=True) -> StaticOrbit:
        """
        Find static equilibrium orbit at given radius using iterative refinement.

        Parameters
        ----------
        radius_mm : float
            Radius [mm]
        n_iterations : int
            Number of refinement iterations (default: 3)
        do_final_tracking : bool
            Whether to perform a final 25 turn tracking for refined orbit analysis
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

        if do_final_tracking:
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

        if self.verbose and do_final_tracking:
            # Calculate centroid (fixed point of Poincare map)
            r_center = np.mean(all_radii)
            vr_center = np.mean(all_vr)

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
            b_field_avg=best_poincare[1].bz_avg,
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

    def find_seos_at_radii(self, radii_mm: List[float], n_iterations: int = 5, do_final_tracking=True,
                           solver: Optional[str] = None,
                           energy_seeds_kev: Optional[List[float]] = None) -> List[StaticOrbit]:
        """
        Find SEOs at multiple radii with iterative refinement.

        Parameters
        ----------
        radii_mm : list
            List of radii [mm]
        n_iterations : int
            Number of refinement iterations per radius (default: 3)
        do_final_tracking : bool
            Whether to perform a final 25 turn tracking for refined orbit analysis
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

        solver = solver or self.solver

        for i, radius in enumerate(radii_mm):
            if self.verbose:
                print(f"\n[{i + 1}/{len(radii_mm)}]", end=" ")

            if solver == 'centroid':
                orbit = self.find_seo_at_radius(radius, n_iterations=n_iterations,
                                                do_final_tracking=do_final_tracking)
            else:
                seed = energy_seeds_kev[i] if energy_seeds_kev is not None else None
                orbit = self.find_seo_newton(radius, energy_seed_kev=seed, solver=solver,
                                             do_final_tracking=do_final_tracking)
            orbits.append(orbit)

        if self.verbose:
            n_closed = sum(1 for o in orbits if o.is_closed)
            print(f"\n{'=' * 70}")
            print(f"SCAN COMPLETE: {n_closed}/{len(orbits)} closed orbits")
            print(f"{'=' * 70}")

        return orbits

    def mean_frequency(self, radii_mm: List[float], solver: Optional[str] = None) -> float:
        """Average orbital frequency [Hz] over SEOs at the given radii.

        Use this to set the RF base frequency to the ISOCHRONOUS region of the
        field (larger radii, where flutter provides focusing) rather than the
        injection orbit near the center, whose frequency is far off-isochronous.
        """
        orbits = self.find_seos_at_radii(list(radii_mm), solver=solver,
                                         do_final_tracking=False)
        freqs = [o.frequency_hz for o in orbits if o.frequency_hz > 0]
        if not freqs:
            raise RuntimeError("mean_frequency: no valid SEOs found")
        return float(np.mean(freqs))

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
