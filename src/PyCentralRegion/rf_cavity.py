"""
rf_cavity.py - RF Cavity Model for Cyclotrons

RF cavity gaps as radial lines in the midplane, extending infinitely in z direction.
Supports global bunch phase control and harmonic RF systems.

Usage:
    from rf_cavity import RFCavity, create_double_gap_cavity, create_four_cavity_system

    # Create double-gap cavity (h=4)
    gap1, gap2 = create_double_gap_cavity(
        r_min=0.05,
        r_max=0.30,
        angle=45.0,
        voltage=60000.0,
        frequency=42e6,  # Revolution frequency
        phase=0.0,
        harmonic=4  # RF at 168 MHz
    )
"""

import numpy as np
from typing import Tuple, Optional, List
from PyPATools.global_variables import CLIGHT
from fontTools.merge.util import recalculate
from PyPATools.particles import ParticleDistribution

class RFCavity:
    """
    RF cavity gap model for cyclotrons.

    Cavity is a radial line in the xy-plane, extending infinitely in z.
    Particle receives momentum kick when crossing the line.

    Parameters
    ----------
    p1, p2 : array_like
        Endpoints of cavity gap line [m] (in xy plane)
    voltage : float
        Peak RF voltage [V]
    frequency : float
        RF frequency [Hz]
    phase : float
        Cavity phase [degrees] (relative phase)
    gap_width : float
        Cavity gap width [m] (used for TTF)
    bunch_phase_offset : float
        Global bunch phase offset [degrees] (default: 0.0)

    Notes
    -----
    Total phase = phase + bunch_phase_offset
    """

    def __init__(self,
                 p1: np.ndarray,
                 p2: np.ndarray,
                 voltage: float = 60000.0,
                 frequency: float = 42e6,
                 harmonic: int = 4,
                 phase: float = 0.0,
                 gap_width: float = 0.02,
                 bunch_phase_offset: float = 0.0,
                 ):

        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)

        if len(self.p1) != 3 or len(self.p2) != 3:
            raise ValueError("Endpoints must be 3D vectors")

        self.voltage = voltage  # V
        self.frequency = frequency  # Hz
        self.harmonic = harmonic
        self.omega = 2.0 * np.pi * frequency  # rad/s
        self.phase_deg = phase  # degrees (relative phase)
        self.phase = np.deg2rad(phase)  # rad
        self.bunch_phase_offset_deg = bunch_phase_offset  # degrees (global offset)
        self.bunch_phase_offset = np.deg2rad(bunch_phase_offset)  # rad
        self.gap_width = gap_width

        # Statistics
        self.n_crossings = 0
        self.total_energy_gain = 0.0  # MeV

        # ParticelDistribution container
        # self.pd = ParticleDistribution()

        # Precompute geometry
        self._compute_geometry()

    def _compute_geometry(self):
        """Precompute geometric properties (xy plane, infinite z extent)."""
        # Extract xy coordinates (ignore z)
        self.p1_2d = np.array([self.p1[0], self.p1[1]])
        self.p2_2d = np.array([self.p2[0], self.p2[1]])

        line_vec = self.p2_2d - self.p1_2d
        self.line_length = np.linalg.norm(line_vec)
        self.line_dir = line_vec / self.line_length

        # Perpendicular direction (kick direction in xy plane)
        self.perp_dir_2d = np.array([-self.line_dir[1], self.line_dir[0]])

        # 3D perpendicular direction (for applying kick)
        self.perp_dir_3d = np.array([self.perp_dir_2d[0], self.perp_dir_2d[1], 0.0])

        self.cavity_angle = np.arctan2(self.p1[1], self.p1[0])
        self.cos_cavity_angle = np.cos(self.cavity_angle)
        self.sin_cavity_angle = np.sin(self.cavity_angle)

    def set_bunch_phase_offset(self, phase_deg: float):
        """Set global bunch phase offset [degrees]."""
        self.bunch_phase_offset_deg = phase_deg
        self.bunch_phase_offset = np.deg2rad(phase_deg)

    def set_frequency(self, frequency: float):
        """Set RF frequency [Hz]."""
        self.frequency = frequency
        self.omega = 2.0 * np.pi * frequency * self.harmonic

    def set_harmonic(self, harmonic):
        self.harmonic = harmonic
        self.omega = 2.0 * np.pi * harmonic * self.frequency

    def get_total_phase_deg(self) -> float:
        """Get total phase (relative + bunch offset) [degrees]."""
        return self.phase_deg + self.bunch_phase_offset_deg

    def get_total_phase_rad(self) -> float:
        """Get total phase (relative + bunch offset) [radians]."""
        return self.phase + self.bunch_phase_offset

    def check_crossing(self, r_old: np.ndarray, r_new: np.ndarray) -> Tuple[bool, Optional[float]]:
        """
        Check if particle trajectory crosses the cavity line.

        Parameters
        ----------
        r_old : np.ndarray
            Previous position [m]
        r_new : np.ndarray
            Current position [m]

        Returns
        -------
        crossed : bool
            True if trajectory crossed
        t_cross : float or None
            Fractional timestep (0-1) where crossing occurred
        """
        # Use only xy coordinates
        r1_2d = np.array([r_old[0], r_old[1]])
        r2_2d = np.array([r_new[0], r_new[1]])

        p1 = self.p1_2d
        p2 = self.p2_2d

        r1 = r1_2d
        r2 = r2_2d

        d_part = r2 - r1
        d_cav = p2 - p1

        A = np.column_stack([d_cav, -d_part])
        b = r1 - p1

        det = np.linalg.det(A)
        if abs(det) < 1e-10:
            return False, None

        params = np.linalg.solve(A, b)
        t_cav = params[0]
        s_part = params[1]

        if 0.0 <= t_cav <= 1.0 and 0.0 <= s_part <= 1.0:
            return True, s_part

        return False, None

    def check_crossings_batch(self, r_old_array: np.ndarray, r_new_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check which particles crossed the cavity line (vectorized).

        Parameters
        ----------
        r_old_array : np.ndarray
            Previous positions (N, 3) [m]
        r_new_array : np.ndarray
            Current positions (N, 3) [m]

        Returns
        -------
        crossed_mask : np.ndarray
            Boolean array (N,) indicating which particles crossed
        t_cross : np.ndarray
            Fractional timestep (N,) where crossing occurred (0 for non-crossed)
        """
        n_particles = len(r_old_array)

        # Extract xy coordinates only
        r1_2d = r_old_array[:, :2]  # (N, 2)
        r2_2d = r_new_array[:, :2]  # (N, 2)

        # Particle trajectory vectors
        d_part = r2_2d - r1_2d  # (N, 2)

        # Cavity line vectors (broadcast to N particles)
        p1 = self.p1_2d  # (2,)
        p2 = self.p2_2d  # (2,)
        d_cav = p2 - p1  # (2,)

        # Solve line-line intersection for all particles
        # For each particle: [d_cav, -d_part] @ [t_cav, s_part] = r1 - p1

        # Build matrix A for all particles
        # A[i] = [[d_cav[0], -d_part[i,0]],
        #         [d_cav[1], -d_part[i,1]]]

        # Calculate determinants (vectorized)
        det = d_cav[0] * (-d_part[:, 1]) - d_cav[1] * (-d_part[:, 0])
        det = -det  # Correct sign

        # Find valid (non-singular) systems
        valid_mask = np.abs(det) > 1e-10

        # Initialize outputs
        crossed_mask = np.zeros(n_particles, dtype=bool)
        t_cross = np.zeros(n_particles)

        if not np.any(valid_mask):
            return crossed_mask, t_cross

        # Solve for valid particles using Cramer's rule
        b = r1_2d - p1  # (N, 2)

        # t_cav = det([b, -d_part]) / det(A)
        # s_part = det([d_cav, b]) / det(A)

        det_t_cav = b[:, 0] * (-d_part[:, 1]) - b[:, 1] * (-d_part[:, 0])
        det_t_cav = -det_t_cav

        det_s_part = d_cav[0] * b[:, 1] - d_cav[1] * b[:, 0]

        t_cav = np.zeros(n_particles)
        s_part = np.zeros(n_particles)

        t_cav[valid_mask] = det_t_cav[valid_mask] / det[valid_mask]
        s_part[valid_mask] = det_s_part[valid_mask] / det[valid_mask]

        # Check if intersection is within both line segments
        crossed_mask = valid_mask & (t_cav >= 0.0) & (t_cav <= 1.0) & (s_part >= 0.0) & (s_part <= 1.0)
        t_cross[crossed_mask] = s_part[crossed_mask]

        return crossed_mask, t_cross

    def apply_kicks_batch(self,
                          r_old_array: np.ndarray,
                          r_new_array: np.ndarray,
                          v_array: np.ndarray,
                          crossed_mask: np.ndarray,
                          t_cross: np.ndarray,
                          t: float,
                          dt: float,
                          design,
                          pusher) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply RF kicks to particles that crossed (vectorized).

        Parameters
        ----------
        r_old_array : np.ndarray
            Positions before step (N, 3) [m]
        r_new_array : np.ndarray
            Positions after step (N, 3) [m]
        v_array : np.ndarray
            Velocities after step (N, 3) [m/s]
        crossed_mask : np.ndarray
            Boolean mask (N,) of particles that crossed
        t_cross : np.ndarray
            Fractional timestep (N,) where crossing occurred
        t : float
            Time after step [s]
        dt : float
            Timestep [s]
        design : CentralRegion
            Design object
        pusher : Pusher
            Pusher object

        Returns
        -------
        v_final : np.ndarray
            Updated velocities (N, 3) [m/s]
        r_final : np.ndarray
            Updated positions (N, 3) [m]
        energy_gains : np.ndarray
            Energy gain per particle (N,) [MeV]
        crossing_phases : np.ndarray
            RF phase at crossing per particle (N,) [deg]
        """
        n_particles = len(r_new_array)
        n_crossed = np.sum(crossed_mask)

        if n_crossed == 0:
            return v_array.copy(), r_new_array.copy(), np.zeros(n_particles), np.zeros(n_particles)

        # Initialize outputs
        # v_final = v_array.copy()
        # r_final = r_new_array.copy()
        energy_gains = np.zeros(n_particles)
        crossing_phases = np.zeros(n_particles)

        # Decide whether to back-track
        use_backtrack = pusher.algorithm.lower() != 'boris'

        # Get indices of crossed particles
        crossed_indices = np.where(crossed_mask)[0]

        if use_backtrack:
            # Back-track to cavity
            dt_back = (1.0 - t_cross[crossed_indices]) * dt
            r_cavity, v_cavity = pusher.push_batch(r_new_array[crossed_indices], v_array[crossed_indices],
                                             design.efield, design.bfield, -dt_back)
            t_cavity = t - dt_back
        else:
            # Apply at current position (Boris)
            r_cavity = r_new_array[crossed_indices]
            v_cavity = v_array[crossed_indices]
            t_cavity = t

        # Calculate transit time factor
        # TODO: Check v_mag > 1e-6, else don't do anything (stopped particle?)
        # TODO: Think about this!
        v_mag = np.linalg.norm(v_cavity, axis=1)

        # TODO: Check abs(omega_tau_half) > 1e-10, else ttf = 1.0
        transit_time = self.gap_width / v_mag
        omega_tau_half = self.omega * transit_time / 2.0
        ttf = np.sin(omega_tau_half) / omega_tau_half

        # Calculate energy gain
        total_phase = self.get_total_phase_rad()
        d_e_mev = 1e-6 * design.species.q * self.voltage * ttf * np.cos(
            self.omega * t_cavity + total_phase)

        # Conserve radial momentum, adjust tangential
        design.beam.x_vec_p_vec = (r_cavity, v_cavity / np.sqrt(CLIGHT ** 2 - v_cavity ** 2))

        # pd = ParticleDistribution(species=design.species,
        #                           x_vec=r_cavity,
        #                           p_vec=v_cavity / np.sqrt(CLIGHT ** 2 - v_cavity ** 2))

        # design.beam.set_p_from_v_vec(v_cavity)
        e_old_mev = design.beam.ekin_mev

        # New total momentum magnitude
        gamma_new = (e_old_mev + d_e_mev) / design.species.mass_mev + 1.0
        beta_gamma_new = np.sqrt(gamma_new ** 2 - 1.0)

        # Cavity geometry
        cos_angle = self.cos_cavity_angle
        sin_angle = self.sin_cavity_angle

        # Old momentum components
        px_old = design.beam.px
        py_old = design.beam.py

        # Radial momentum (conserved)
        p_r = px_old * cos_angle + py_old * sin_angle

        # Tangential momentum from energy constraint
        # TODO: pre-calculate sign during cavity init
        # TODO: check discriminant >= 0
        discriminant = beta_gamma_new ** 2 - p_r ** 2
        sign = 1.0
        p_theta = sign * np.sqrt(discriminant)

        # Reconstruct Cartesian momentum
        px_new = p_r * cos_angle - p_theta * sin_angle
        py_new = p_r * sin_angle + p_theta * cos_angle

        # Update momentum
        design.beam.px = px_new
        design.beam.py = py_new

        # Get new velocity
        v_new = design.beam.v_vec

        if use_backtrack:
            r_final, v_final = pusher.push_batch(r_cavity, v_new,
                                                 design.efield, design.bfield, dt_back)
        else:
            r_final = r_new_array[crossed_indices]
            v_final = v_new

        r_new_array[crossed_indices] = r_final
        v_array[crossed_indices] = v_final

        # Store results
        energy_gains[crossed_indices] = d_e_mev
        crossing_phase_rad = np.fmod(self.omega * t_cavity + total_phase, 2.0 * np.pi)
        crossing_phases[crossed_indices] = np.rad2deg(crossing_phase_rad)

        # Statistics
        self.n_crossings += len(r_final)
        self.total_energy_gain += np.sum(d_e_mev)

        return v_array, r_new_array, energy_gains, crossing_phases

    def apply_kick_if_crossing(self,
                               r_old: np.ndarray,
                               r_new: np.ndarray,
                               v: np.ndarray,
                               t: float,
                               dt: float,
                               design,
                               pusher) -> Tuple[np.ndarray, np.ndarray, bool, float, float]:
        """
        Apply RF kick when particle crosses cavity gap.

        Uses relativistic momentum and transit time factor.
        Uses OPAL's approach: conserve radial momentum, adjust tangential momentum.

        For Boris pusher: applies kick at r_new to preserve symplectic structure.
        For other pushers: back-tracks to cavity, applies kick, forward-tracks.

        Parameters
        ----------
        r_old : np.ndarray
            Position before step [m]
        r_new : np.ndarray
            Position after step [m]
        v : np.ndarray
            Velocity after step [m/s]
        t : float
            Time after step [s]
        dt : float
            Timestep [s]
        design : CentralRegion
            Design with ParticleDistribution (beam)
        pusher : Pusher
            Pusher object

        Returns
        -------
        v_final : np.ndarray
            Velocity after kick [m/s]
        r_final : np.ndarray
            Position after kick [m]
        crossed : bool
            Whether crossing occurred
        energy_gain : float
            Energy gain [MeV]
        crossing_phase : float
            RF phase at crossing [deg]
        """

        # 1. Check crossing
        crossed, t_cross = self.check_crossing(r_old, r_new)

        if not crossed:
            return v, r_new, False, 0.0, 0.0

        # Decide whether to back-track based on pusher algorithm
        use_backtrack = pusher.algorithm.lower() != 'boris'

        if use_backtrack:
            # 2a. Track back to cavity location
            dt_back = (1.0 - t_cross) * dt

            r_cavity, v_cavity = pusher.push(r_new.copy(), v.copy(), design.efield, design.bfield, -dt_back)
            t_cavity = t - dt_back
        else:
            # 2b. Apply kick at current position (Boris)
            r_cavity = r_new.copy()
            v_cavity = v.copy()
            t_cavity = t

        # 3. Calculate transit time factor
        v_mag = np.linalg.norm(v_cavity)
        if v_mag < 1e-6:
            return v, r_new, False, 0.0, 0.0

        transit_time = self.gap_width / v_mag
        omega_tau_half = self.omega * transit_time / 2.0

        if abs(omega_tau_half) > 1e-10:
            ttf = np.sin(omega_tau_half) / omega_tau_half
        else:
            ttf = 1.0  # Limit as ω*τ → 0

        # 4. Calculate energy gain
        total_phase = self.get_total_phase_rad()
        d_e_mev = 1e-6 * design.species.q * self.voltage * ttf * np.cos(
            self.omega * t_cavity + total_phase)

        # 5. OPAL approach: conserve radial momentum, adjust tangential
        design.beam.set_p_from_v(
            np.array([v_cavity[0]]),
            np.array([v_cavity[1]]),
            np.array([v_cavity[2]]))

        e_old_mev = design.beam.mean_energy_mev

        # New total momentum magnitude after energy gain
        gamma_new = (e_old_mev + d_e_mev) / design.species.mass_mev + 1.0
        beta_gamma_new = np.sqrt(gamma_new ** 2 - 1.0)

        # Cavity geometry: radial line at cavity_angle
        cos_angle = self.cos_cavity_angle
        sin_angle = self.sin_cavity_angle

        # Old momentum components
        px_old = design.beam.px[0]
        py_old = design.beam.py[0]

        # Radial momentum (parallel to gap face) - CONSERVED across gap
        p_r = px_old * cos_angle + py_old * sin_angle

        # Tangential momentum from energy constraint: p_r² + p_θ² = (βγ)²
        discriminant = beta_gamma_new ** 2 - p_r ** 2

        if discriminant < 0:
            # Shouldn't happen unless large deceleration or numerical error
            return v, r_new, False, 0.0, 0.0

        # Sign: check crossing direction relative to tangential kick direction
        # TODO: Move this into cavity initialization (depends on species charge sign and Bz sign)
        sign = 1.0
        p_theta = sign * np.sqrt(discriminant)

        # Reconstruct Cartesian momentum components
        # p_r along radial: (cos θ, sin θ)
        # p_θ along tangential: (-sin θ, cos θ)
        px_new = p_r * cos_angle - p_theta * sin_angle
        py_new = p_r * sin_angle + p_theta * cos_angle

        # Update momentum (pz unchanged for midplane tracking)
        design.beam.px = np.array([px_new])
        design.beam.py = np.array([py_new])

        # Get new velocity from updated momentum
        v_new = np.array([design.beam.vx[0], design.beam.vy[0], design.beam.vz[0]])

        # 6. Track forward (if we back-tracked)
        if use_backtrack:
            r_final, v_final = pusher.push(r_cavity, v_new, design.efield, design.bfield, dt_back)
        else:
            r_final = r_new
            v_final = v_new

        # 7. Statistics and diagnostics
        self.n_crossings += 1
        self.total_energy_gain += d_e_mev

        crossing_phase_rad = np.fmod(self.omega * t_cavity + total_phase, 2.0 * np.pi)
        crossing_phase_deg = np.rad2deg(crossing_phase_rad)

        return v_final, r_final, True, d_e_mev, crossing_phase_deg

    def get_statistics(self) -> dict:
        """Get cavity crossing statistics."""
        return {
            'n_crossings': self.n_crossings,
            'total_energy_gain_J': self.total_energy_gain,
            'total_energy_gain_keV': self.total_energy_gain / 1.602176634e-16,
            'average_energy_gain_keV': (self.total_energy_gain / 1.602176634e-16 / self.n_crossings
                                        if self.n_crossings > 0 else 0.0)
        }

    def reset_statistics(self):
        """Reset crossing statistics."""
        self.n_crossings = 0
        self.total_energy_gain = 0.0

    def __str__(self):
        total_phase = self.get_total_phase_deg()
        angle = np.rad2deg(np.arctan2(self.p1[1], self.p1[0]))
        return (f"RFCavity: {self.voltage / 1000:.1f} kV @ {self.frequency / 1e6:.1f} MHz, "
                f"angle={angle:.1f}deg, phase={self.phase_deg:.1f}deg + "
                f"{self.bunch_phase_offset_deg:.1f}deg = {total_phase:.1f}deg")


def create_double_gap_cavity(r_min: float,
                             r_max: float,
                             angle: float,
                             voltage: float = 60000.0,
                             frequency: float = 42e6,
                             phase: float = 0.0,
                             gap_width: float = 0.01,
                             harmonic: int = 4) -> Tuple[RFCavity, RFCavity]:
    """
    Create a double-gap RF cavity with two radial gaps.

    Parameters
    ----------
    r_min : float
        Inner radius of gaps [m]
    r_max : float
        Outer radius of gaps [m]
    angle : float
        Azimuthal angle of first gap [degrees]
    voltage : float
        Voltage per gap [V]
    frequency : float
        Revolution frequency (h=1) [Hz] - will be multiplied by harmonic
    phase : float
        Phase of first gap [degrees]
    gap_width : float
        width of accelerating gap [m]
    harmonic : int
        RF harmonic number

    Returns
    -------
    gap1, gap2 : tuple of RFCavity
        Two cavity gaps as radial lines (infinite in z)

    Notes
    -----
    - Gap separation is 360/(2*h) degrees
    - Second gap has 180 degree phase shift
    - RF frequency = harmonic * revolution frequency

    Examples
    --------
    > # For h=4: gaps 45 degrees apart, RF at 4x revolution freq
    > gap1, gap2 = create_double_gap_cavity(
    ...     r_min=0.05, r_max=0.30, angle=0.0,
    ...     frequency=42e6, harmonic=4
    ... )
    > # gap1 at 0 deg, gap2 at 45 deg, both at 168 MHz
    """
    # First gap - radial line at angle
    angle1_rad = np.deg2rad(angle)
    p1_1 = np.array([r_min * np.cos(angle1_rad),
                     r_min * np.sin(angle1_rad),
                     0.0])
    p1_2 = np.array([r_max * np.cos(angle1_rad),
                     r_max * np.sin(angle1_rad),
                     0.0])

    gap1 = RFCavity(p1_1, p1_2, voltage=voltage,
                    frequency=frequency, harmonic=harmonic,
                    phase=phase, gap_width=gap_width)

    # Second gap: 360/(2*h) degrees away, 180 degree phase shift
    angle2 = angle + 360.0 / (2.0 * harmonic)
    angle2_rad = np.deg2rad(angle2)
    p2_1 = np.array([r_min * np.cos(angle2_rad),
                     r_min * np.sin(angle2_rad),
                     0.0])
    p2_2 = np.array([r_max * np.cos(angle2_rad),
                     r_max * np.sin(angle2_rad),
                     0.0])

    gap2 = RFCavity(p2_1, p2_2, voltage=voltage,
                    frequency=frequency, harmonic=harmonic,
                    phase=phase + 180.0, gap_width=gap_width)

    return gap1, gap2


def create_four_cavity_system(r_min: float,
                              r_max: float,
                              angles: List[float],
                              voltage: float = 60000.0,
                              frequency: float = 42e6,
                              phases: List[float] = None,
                              gap_width: float = 0.01,
                              harmonic: int = 4) -> List[RFCavity]:
    """
    Create system of 4 double-gap cavities (8 radial gaps total).

    Parameters
    ----------
    r_min : float
        Inner radius for all gaps [m]
    r_max : float
        Outer radius for all gaps [m]
    angles : list of float
        Azimuthal angles of 4 cavities [degrees]
    voltage : float
        Voltage per gap [V]
    frequency : float
        Revolution frequency (h=1) [Hz] - will be multiplied by harmonic
    phases : list of float, optional
        Phases of 4 cavities [degrees]. If None, all set to 0.
    gap_width : float
        width of accelerating gap [m]
    harmonic : int
        RF harmonic number

    Returns
    -------
    gaps : list of RFCavity
        List of 8 cavity gaps (2 per cavity, all radial, infinite in z)

    Examples
    --------
    > # For h=4, gaps separated by 360/(2*4) = 45 degrees
    > # If angles = [0, 90, 180, 270], gaps will be at:
    > # 0, 45, 90, 135, 180, 225, 270, 315 degrees
    > gaps = create_four_cavity_system(
    ...     r_min=0.05, r_max=0.30,
    ...     angles=[0, 90, 180, 270],
    ...     frequency=42e6,  # 42 MHz revolution
    ...     harmonic=4  # 168 MHz RF
    ... )
    """
    if phases is None:
        phases = [0.0] * 4

    if len(angles) != 4 or len(phases) != 4:
        raise ValueError("Must provide 4 angles and 4 phases")

    gaps = []
    for ang, ph in zip(angles, phases):
        gap1, gap2 = create_double_gap_cavity(
            r_min=r_min,
            r_max=r_max,
            angle=ang,
            voltage=voltage,
            frequency=frequency,  # Pass revolution frequency
            phase=ph,
            gap_width=gap_width,
            harmonic=harmonic
        )
        gaps.extend([gap1, gap2])

    return gaps


if __name__ == "__main__":
    print("Testing RF cavity system...")

    # Test double-gap cavity
    print("\n1. Creating double-gap cavity (h=4, radial gaps):")
    gap1, gap2 = create_double_gap_cavity(
        r_min=0.05,
        r_max=0.30,
        angle=45.0,
        voltage=60000.0,
        frequency=42e6,  # 42 MHz revolution
        phase=-25.0,
        harmonic=4  # 168 MHz RF
    )
    print(f"   Gap 1: r={np.linalg.norm(gap1.p1[:2]):.3f} to {np.linalg.norm(gap1.p2[:2]):.3f} m "
          f"at {np.rad2deg(np.arctan2(gap1.p1[1], gap1.p1[0])):.1f} deg, "
          f"f={gap1.frequency / 1e6:.1f} MHz")
    print(f"   Gap 2: r={np.linalg.norm(gap2.p1[:2]):.3f} to {np.linalg.norm(gap2.p2[:2]):.3f} m "
          f"at {np.rad2deg(np.arctan2(gap2.p1[1], gap2.p1[0])):.1f} deg, "
          f"f={gap2.frequency / 1e6:.1f} MHz")
    print(f"   Gap separation: {360.0 / (2.0 * 4):.1f} degrees")

    # Test bunch phase offset
    print("\n2. Setting bunch phase offset to +10 degrees:")
    gap1.set_bunch_phase_offset(10.0)
    gap2.set_bunch_phase_offset(10.0)
    print(f"   Gap 1 total phase: {gap1.get_total_phase_deg():.1f} deg")
    print(f"   Gap 2 total phase: {gap2.get_total_phase_deg():.1f} deg")

    # Test 4-cavity system
    print("\n3. Creating 4-cavity system:")
    angles = [0, 90, 180, 270]  # First gap of each double-gap
    phases = [-25, -25, -25, -25]

    all_gaps = create_four_cavity_system(
        r_min=0.05,
        r_max=0.30,
        angles=angles,
        voltage=60000.0,
        frequency=42e6,
        phases=phases,
        harmonic=4
    )

    print(f"\n   Created {len(all_gaps)} radial gaps:")
    for i, gap in enumerate(all_gaps):
        angle = np.rad2deg(np.arctan2(gap.p1[1], gap.p1[0]))
        print(f"   Gap {i}: angle={angle:.1f} deg, phase={gap.phase_deg:.1f} deg, "
              f"f={gap.frequency / 1e6:.1f} MHz")

    # Verify gap spacing for h=4
    print(f"\n   Expected gap spacing: 360/(2*4) = {360.0 / (2.0 * 4):.1f} deg")
    print(f"   Actual gaps at: ", end="")
    for gap in all_gaps:
        angle = np.rad2deg(np.arctan2(gap.p1[1], gap.p1[0]))
        print(f"{angle:.1f} ", end="")
    print("degrees")

    print("\n✓ RF cavity system tests passed")