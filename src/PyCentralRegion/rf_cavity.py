"""
rf_cavity.py - RF Cavity Model with Variable Geometry Segments

RF cavities can now have multiple segments with different angular orientations.
Enables optimization of cavity shape for improved acceleration efficiency.

Part of: PyCentralRegion module
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from PyPATools.global_variables import CLIGHT
from PyPATools.particles import ParticleDistribution


class RFCavity:
    """
    RF cavity gap with variable geometry segments.

    Cavity consists of:
    1. N "variable segments" with angular excursions
    2. One "fixed segment" (radial line) to final radius

    All segments share the same RF phase.

    Parameters
    ----------
    r_min : float
        Inner radius [m]
    r_max : float
        Outer radius [m]
    base_angle : float
        Base azimuthal angle [degrees]
    voltage : float
        Peak RF voltage [V]
    frequency : float
        RF frequency [Hz]
    harmonic : int
        RF harmonic number
    phase : float
        Cavity phase [degrees]
    gap_width : float
        Gap width [m]
    bunch_phase_offset : float
        Global bunch phase offset [degrees]
    n_variable_segments : int
        Number of variable geometry segments (default: 0)
    segment_angles : list of float, optional
        Angular excursion for each segment [degrees]
        Positive = counterclockwise, negative = clockwise
    segment_radii : list of float, optional
        Outer radius for each segment [m]
        Must be monotonically increasing: r_min < r0 < r1 < ... < r_max

    Examples
    --------
    # Traditional radial cavity
    cav = RFCavity(r_min=0.05, r_max=0.30, base_angle=45.0, ...)

    # Cavity with 2 variable segments
    cav = RFCavity(
        r_min=0.05, r_max=0.30, base_angle=45.0,
        n_variable_segments=2,
        segment_angles=[5.0, -3.0],      # Segment 0: +5°, Segment 1: -3°
        segment_radii=[0.15, 0.22],      # Segment 0 ends at 0.15m, Segment 1 at 0.22m
        ...
    )
    # This creates:
    # - Segment 0: r=0.05 to r=0.15, angle offset +5°
    # - Segment 1: r=0.15 to r=0.22, angle offset -3°
    # - Fixed segment: r=0.22 to r=0.30, radial
    """

    def __init__(self,
                 r_min: float,
                 r_max: float,
                 base_angle: float,
                 voltage: float = 60000.0,
                 frequency: float = 42e6,
                 harmonic: int = 4,
                 phase: float = 0.0,
                 gap_width: float = 0.02,
                 bunch_phase_offset: float = 0.0,
                 n_variable_segments: int = 0,
                 segment_angles: Optional[List[float]] = None,
                 segment_radii: Optional[List[float]] = None):

        self.r_min = r_min
        self.r_max = r_max
        self.base_angle = base_angle
        self.voltage = voltage
        self.frequency = frequency
        self.harmonic = harmonic
        self.omega = 2.0 * np.pi * frequency * harmonic
        self.phase_deg = phase
        self.phase = np.deg2rad(phase)
        self.bunch_phase_offset_deg = bunch_phase_offset
        self.bunch_phase_offset = np.deg2rad(bunch_phase_offset)
        self.gap_width = gap_width

        # Variable segments
        self.n_variable_segments = n_variable_segments
        self.segment_angles = segment_angles if segment_angles is not None else []
        self.segment_radii = segment_radii if segment_radii is not None else []

        # Validate
        if n_variable_segments > 0:
            if len(self.segment_angles) != n_variable_segments:
                raise ValueError(f"segment_angles must have length {n_variable_segments}")
            if len(self.segment_radii) != n_variable_segments:
                raise ValueError(f"segment_radii must have length {n_variable_segments}")

            # Check monotonicity
            all_radii = [r_min] + list(self.segment_radii) + [r_max]
            if not all(all_radii[i] < all_radii[i + 1] for i in range(len(all_radii) - 1)):
                raise ValueError(f"Radii must be monotonically increasing: {all_radii}")

        # Statistics
        self.n_crossings = 0
        self.total_energy_gain = 0.0

        # Build cavity geometry
        self._build_geometry()

    def _build_geometry(self):
        """Build piecewise cavity geometry from segments."""

        base_angle_rad = np.deg2rad(self.base_angle)

        # Build segment list
        self.segments = []

        r_current = self.r_min
        angle_current = base_angle_rad

        # Variable segments
        for i in range(self.n_variable_segments):
            r_outer = self.segment_radii[i]
            angle_excursion = np.deg2rad(self.segment_angles[i])

            # Start point
            p1 = np.array([
                r_current * np.cos(angle_current),
                r_current * np.sin(angle_current),
                0.0
            ])

            # End point (with angular excursion)
            angle_end = angle_current + angle_excursion
            p2 = np.array([
                r_outer * np.cos(angle_end),
                r_outer * np.sin(angle_end),
                0.0
            ])

            self.segments.append({
                'p1': p1,
                'p2': p2,
                'r_min': r_current,
                'r_max': r_outer,
                'type': 'variable'
            })

            # Update for next segment
            r_current = r_outer
            angle_current = angle_end

        # Fixed radial segment (from last variable segment to r_max)
        p1 = np.array([
            r_current * np.cos(angle_current),
            r_current * np.sin(angle_current),
            0.0
        ])
        p2 = np.array([
            self.r_max * np.cos(angle_current),
            self.r_max * np.sin(angle_current),
            0.0
        ])

        self.segments.append({
            'p1': p1,
            'p2': p2,
            'r_min': r_current,
            'r_max': self.r_max,
            'type': 'fixed'
        })

        # Store full cavity endpoints for compatibility
        self.p1 = self.segments[0]['p1']
        self.p2 = self.segments[-1]['p2']

        # Precompute geometry for each segment
        for seg in self.segments:
            self._precompute_segment_geometry(seg)

    def _precompute_segment_geometry(self, segment: dict):
        """Precompute geometric properties for a segment."""

        p1_2d = segment['p1'][:2]
        p2_2d = segment['p2'][:2]

        line_vec = p2_2d - p1_2d
        segment['length'] = np.linalg.norm(line_vec)
        segment['direction'] = line_vec / segment['length']

        # Perpendicular direction (for kick)
        segment['perp_2d'] = np.array([-segment['direction'][1], segment['direction'][0]])
        segment['perp_3d'] = np.array([segment['perp_2d'][0], segment['perp_2d'][1], 0.0])

    def update_geometry(self, segment_angles: Optional[List[float]] = None,
                        segment_radii: Optional[List[float]] = None):
        """
        Update cavity geometry (for optimization).

        Parameters
        ----------
        segment_angles : list of float, optional
            New angular excursions [degrees]
        segment_radii : list of float, optional
            New outer radii [m]
        """

        if segment_angles is not None:
            if len(segment_angles) != self.n_variable_segments:
                raise ValueError(f"segment_angles must have length {self.n_variable_segments}")
            self.segment_angles = segment_angles

        if segment_radii is not None:
            if len(segment_radii) != self.n_variable_segments:
                raise ValueError(f"segment_radii must have length {self.n_variable_segments}")
            self.segment_radii = segment_radii

        # Rebuild geometry
        self._build_geometry()

    def check_crossing(self, r_old: np.ndarray, r_new: np.ndarray) -> Tuple[bool, Optional[float], Optional[int]]:
        """
        Check if particle crosses any segment of the cavity.

        Returns
        -------
        crossed : bool
        t_frac : float or None
            Fractional timestep of crossing
        segment_id : int or None
            Which segment was crossed
        """

        r1_2d = r_old[:2]
        r2_2d = r_new[:2]

        # Check each segment
        for seg_id, seg in enumerate(self.segments):
            p1 = seg['p1'][:2]
            p2 = seg['p2'][:2]

            d_part = r2_2d - r1_2d
            d_cav = p2 - p1

            # Line-line intersection
            A = np.column_stack([d_cav, -d_part])
            b = r1_2d - p1

            det = np.linalg.det(A)
            if abs(det) < 1e-10:
                continue

            params = np.linalg.solve(A, b)
            t_cav = params[0]
            s_part = params[1]

            if 0.0 <= t_cav <= 1.0 and 0.0 <= s_part <= 1.0:
                return True, s_part, seg_id

        return False, None, None

    def check_crossings_batch(self, r_old_array: np.ndarray,
                              r_new_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized crossing check for multiple particles.

        Returns
        -------
        crossed_mask : np.ndarray (N,)
            Boolean mask
        t_cross : np.ndarray (N,)
            Fractional timesteps
        segment_ids : np.ndarray (N,)
            Which segment each particle crossed (-1 for no crossing)
        """

        n_particles = len(r_old_array)
        crossed_mask = np.zeros(n_particles, dtype=bool)
        t_cross = np.zeros(n_particles)
        segment_ids = np.full(n_particles, -1, dtype=int)

        r1_2d = r_old_array[:, :2]
        r2_2d = r_new_array[:, :2]
        d_part = r2_2d - r1_2d

        # Check each segment
        for seg_id, seg in enumerate(self.segments):
            p1 = seg['p1'][:2]
            p2 = seg['p2'][:2]
            d_cav = p2 - p1

            # Vectorized intersection
            det = d_cav[0] * (-d_part[:, 1]) - d_cav[1] * (-d_part[:, 0])
            det = -det

            valid_mask = np.abs(det) > 1e-10

            if not np.any(valid_mask):
                continue

            b = r1_2d - p1
            det_t_cav = b[:, 0] * (-d_part[:, 1]) - b[:, 1] * (-d_part[:, 0])
            det_t_cav = -det_t_cav
            det_s_part = d_cav[0] * b[:, 1] - d_cav[1] * b[:, 0]

            t_cav = np.zeros(n_particles)
            s_part = np.zeros(n_particles)

            t_cav[valid_mask] = det_t_cav[valid_mask] / det[valid_mask]
            s_part[valid_mask] = det_s_part[valid_mask] / det[valid_mask]

            # Check bounds
            this_crossed = (valid_mask & (t_cav >= 0.0) & (t_cav <= 1.0) &
                            (s_part >= 0.0) & (s_part <= 1.0))

            # Only record first crossing per particle
            new_crossings = this_crossed & ~crossed_mask
            crossed_mask[new_crossings] = True
            t_cross[new_crossings] = s_part[new_crossings]
            segment_ids[new_crossings] = seg_id

        return crossed_mask, t_cross, segment_ids

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
        """Apply RF kicks (uses first segment's perpendicular direction for now)."""

        n_particles = len(r_new_array)
        n_crossed = np.sum(crossed_mask)

        if n_crossed == 0:
            return v_array.copy(), r_new_array.copy(), np.zeros(n_particles), np.zeros(n_particles)

        energy_gains = np.zeros(n_particles)
        crossing_phases = np.zeros(n_particles)

        use_backtrack = pusher.algorithm.lower() != 'boris'
        crossed_indices = np.where(crossed_mask)[0]

        if use_backtrack:
            dt_back = (1.0 - t_cross[crossed_indices]) * dt
            r_cavity, v_cavity = pusher.push_batch(
                r_new_array[crossed_indices], v_array[crossed_indices],
                design.efield, design.bfield, -dt_back
            )
            t_cavity = t - dt_back
        else:
            r_cavity = r_new_array[crossed_indices]
            v_cavity = v_array[crossed_indices]
            t_cavity = t

        # TTF calculation
        v_mag = np.linalg.norm(v_cavity, axis=1)
        transit_time = self.gap_width / v_mag
        omega_tau_half = self.omega * transit_time / 2.0
        ttf = np.sin(omega_tau_half) / omega_tau_half

        # Energy gain
        total_phase = self.get_total_phase_rad()
        d_e_mev = 1e-6 * design.species.q * self.voltage * ttf * np.cos(
            self.omega * t_cavity + total_phase
        )

        # Update momentum (conserve radial, adjust tangential)
        design.beam.x_vec_p_vec = (r_cavity, v_cavity / np.sqrt(CLIGHT ** 2 - v_cavity ** 2))
        e_old_mev = design.beam.ekin_mev

        gamma_new = (e_old_mev + d_e_mev) / design.species.mass_mev + 1.0
        beta_gamma_new = np.sqrt(gamma_new ** 2 - 1.0)

        # Use base angle for momentum calculation
        cos_angle = np.cos(np.deg2rad(self.base_angle))
        sin_angle = np.sin(np.deg2rad(self.base_angle))

        px_old = design.beam.px
        py_old = design.beam.py

        p_r = px_old * cos_angle + py_old * sin_angle
        discriminant = beta_gamma_new ** 2 - p_r ** 2

        sign = 1.0
        p_theta = sign * np.sqrt(np.maximum(discriminant, 0.0))

        px_new = p_r * cos_angle - p_theta * sin_angle
        py_new = p_r * sin_angle + p_theta * cos_angle

        design.beam.px = px_new
        design.beam.py = py_new

        v_new = design.beam.v_vec

        if use_backtrack:
            r_final, v_final = pusher.push_batch(
                r_cavity, v_new, design.efield, design.bfield, dt_back
            )
        else:
            r_final = r_new_array[crossed_indices]
            v_final = v_new

        r_new_array[crossed_indices] = r_final
        v_array[crossed_indices] = v_final

        energy_gains[crossed_indices] = d_e_mev
        crossing_phase_rad = np.fmod(self.omega * t_cavity + total_phase, 2.0 * np.pi)
        crossing_phases[crossed_indices] = np.rad2deg(crossing_phase_rad)

        self.n_crossings += len(r_final)
        self.total_energy_gain += np.sum(d_e_mev)

        return v_array, r_new_array, energy_gains, crossing_phases

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
        if self.n_variable_segments == 0:
            return (f"RFCavity: {self.voltage / 1000:.1f} kV @ {self.frequency / 1e6:.1f} MHz, "
                    f"angle={self.base_angle:.1f}°, phase={total_phase:.1f}°")
        else:
            return (f"RFCavity: {self.voltage / 1000:.1f} kV @ {self.frequency / 1e6:.1f} MHz, "
                    f"base_angle={self.base_angle:.1f}°, {self.n_variable_segments} variable segments, "
                    f"phase={total_phase:.1f}°")


# Helper functions remain backward compatible
def create_double_gap_cavity(r_min: float,
                             r_max: float,
                             angle: float,
                             voltage: float = 60000.0,
                             frequency: float = 42e6,
                             phase: float = 0.0,
                             gap_width: float = 0.01,
                             harmonic: int = 4,
                             n_variable_segments: int = 0,
                             segment_angles: Optional[List[float]] = None,
                             segment_radii: Optional[List[float]] = None) -> Tuple[RFCavity, RFCavity]:
    """Create double-gap cavity (now with optional variable segments)."""

    gap1 = RFCavity(
        r_min, r_max, angle, voltage, frequency, harmonic, phase, gap_width,
        n_variable_segments=n_variable_segments,
        segment_angles=segment_angles,
        segment_radii=segment_radii
    )

    angle2 = angle + 360.0 / (2.0 * harmonic)
    gap2 = RFCavity(
        r_min, r_max, angle2, voltage, frequency, harmonic, phase + 180.0, gap_width,
        n_variable_segments=n_variable_segments,
        segment_angles=segment_angles,
        segment_radii=segment_radii
    )

    return gap1, gap2


def create_four_cavity_system(r_min: float,
                              r_max: float,
                              angles: List[float],
                              voltage: float = 60000.0,
                              frequency: float = 42e6,
                              phases: List[float] = None,
                              gap_width: float = 0.01,
                              harmonic: int = 4,
                              n_variable_segments: int = 0,
                              segment_angles: Optional[List[float]] = None,
                              segment_radii: Optional[List[float]] = None) -> List[RFCavity]:
    """Create 4-cavity system (now with optional variable segments)."""

    if phases is None:
        phases = [0.0] * 4

    if len(angles) != 4 or len(phases) != 4:
        raise ValueError("Must provide 4 angles and 4 phases")

    gaps = []
    for ang, ph in zip(angles, phases):
        gap1, gap2 = create_double_gap_cavity(
            r_min, r_max, ang, voltage, frequency, ph, gap_width, harmonic,
            n_variable_segments, segment_angles, segment_radii
        )
        gaps.extend([gap1, gap2])

    return gaps


if __name__ == "__main__":
    print("Testing variable segment RF cavity...")

    # Test traditional radial cavity
    print("\n1. Traditional radial cavity:")
    cav = RFCavity(r_min=0.05, r_max=0.30, base_angle=45.0)
    print(f"   {cav}")
    print(f"   Segments: {len(cav.segments)}")

    # Test cavity with variable segments
    print("\n2. Cavity with 2 variable segments:")
    cav_var = RFCavity(
        r_min=0.05, r_max=0.30, base_angle=45.0,
        n_variable_segments=2,
        segment_angles=[5.0, -3.0],
        segment_radii=[0.15, 0.22]
    )
    print(f"   {cav_var}")
    print(f"   Segments: {len(cav_var.segments)}")
    for i, seg in enumerate(cav_var.segments):
        print(f"     Segment {i} ({seg['type']}): r={seg['r_min'] * 1000:.0f}-{seg['r_max'] * 1000:.0f} mm")

    # Test geometry update
    print("\n3. Updating geometry:")
    cav_var.update_geometry(segment_angles=[10.0, -5.0], segment_radii=[0.12, 0.25])
    print(f"   {cav_var}")
    for i, seg in enumerate(cav_var.segments):
        print(f"     Segment {i} ({seg['type']}): r={seg['r_min'] * 1000:.0f}-{seg['r_max'] * 1000:.0f} mm")

    print("\n✓ Variable segment RF cavity tests passed")