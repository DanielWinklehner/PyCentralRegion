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
        Nominal gap width [m] (the full width at large radii)
    gap_width_inner : float, optional
        Gap width [m] at the cavity inner radius ``r_min``. Together with
        ``gap_taper_radius`` this defines a linear TAPER: the gap channel
        narrows from ``gap_width`` at ``gap_taper_radius`` down to
        ``gap_width_inner`` at ``r_min`` (like real central regions, where
        narrower inner gaps leave room for metal between converging gaps).
        Default None = no taper (constant ``gap_width``).
    gap_taper_radius : float, optional
        Transition radius [m] from tapered to straight (nominal) gap width.
        Required together with ``gap_width_inner``.
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
                 segment_radii: Optional[List[float]] = None,
                 segment_rotations: Optional[List[float]] = None,
                 gap_width_inner: Optional[float] = None,
                 gap_taper_radius: Optional[float] = None):

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

        # Optional linear taper of the gap CHANNEL width toward the center.
        if (gap_width_inner is None) != (gap_taper_radius is None):
            raise ValueError("gap_width_inner and gap_taper_radius must be "
                             "given together (or both omitted)")
        if gap_width_inner is not None:
            if gap_width_inner <= 0.0:
                raise ValueError(f"gap_width_inner must be > 0 (got {gap_width_inner})")
            if not r_min < gap_taper_radius <= r_max:
                raise ValueError(f"gap_taper_radius must be in (r_min, r_max] "
                                 f"(got {gap_taper_radius} for r_min={r_min}, "
                                 f"r_max={r_max})")
        self.gap_width_inner = gap_width_inner
        self.gap_taper_radius = gap_taper_radius

        # Variable segments. `segment_rotations` rotates each variable segment
        # about its own MIDPOINT (deg, CCW) after the chain is laid out - the
        # segment need not point at the origin. This decouples the crossing
        # azimuth (RF phase) from the kick direction (segment normal). In the
        # thin-gap model the resulting disconnect between neighboring segments
        # is harmless (each is an independent crossing line); electrode
        # continuity is re-imposed at the field-solving stage.
        self.n_variable_segments = n_variable_segments
        self.segment_angles = segment_angles if segment_angles is not None else []
        self.segment_radii = segment_radii if segment_radii is not None else []
        self.segment_rotations = (segment_rotations if segment_rotations is not None
                                  else [0.0] * n_variable_segments)

        # Validate
        if n_variable_segments > 0:
            if len(self.segment_angles) != n_variable_segments:
                raise ValueError(f"segment_angles must have length {n_variable_segments}")
            if len(self.segment_radii) != n_variable_segments:
                raise ValueError(f"segment_radii must have length {n_variable_segments}")
            if len(self.segment_rotations) != n_variable_segments:
                raise ValueError(f"segment_rotations must have length {n_variable_segments}")

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

            # Optional rotation about the segment midpoint (decouples kick
            # direction from crossing azimuth). The CHAIN continues from the
            # NOMINAL (unrotated) endpoint so rotations don't propagate.
            rot = np.deg2rad(self.segment_rotations[i]) if self.segment_rotations else 0.0
            if rot != 0.0:
                mid = 0.5 * (p1 + p2)
                c, s = np.cos(rot), np.sin(rot)
                R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                p1_stored = mid + R @ (p1 - mid)
                p2_stored = mid + R @ (p2 - mid)
            else:
                p1_stored, p2_stored = p1, p2

            self.segments.append({
                'p1': p1_stored,
                'p2': p2_stored,
                'r_min': r_current,
                'r_max': r_outer,
                'type': 'variable'
            })

            # Update for next segment
            r_current = r_outer
            angle_current = angle_end

        # Fixed radial segment out to r_max, anchored at the NOMINAL base angle:
        # it must NOT inherit the variable segments' angular excursions. The
        # variable segments shape the INNER region only; letting the excursion
        # re-azimuth the whole outer gap line (a) moves the dee edge at large
        # radii with an inner-region parameter (not buildable as intended) and
        # (b) makes a common-mode segment angle degenerate with bunch_phase
        # (a pure global RF-phase trim). The disconnect from the variable tip is
        # the same thin-gap idiom as the rotations - electrode continuity is
        # re-imposed at the field-solving stage.
        p1 = np.array([
            r_current * np.cos(base_angle_rad),
            r_current * np.sin(base_angle_rad),
            0.0
        ])
        p2 = np.array([
            self.r_max * np.cos(base_angle_rad),
            self.r_max * np.sin(base_angle_rad),
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
                        segment_radii: Optional[List[float]] = None,
                        base_angle: Optional[float] = None,
                        segment_rotations: Optional[List[float]] = None):
        """
        Update cavity geometry (for optimization).

        Parameters
        ----------
        segment_angles : list of float, optional
            New angular excursions [degrees]
        segment_radii : list of float, optional
            New outer radii [m]
        base_angle : float, optional
            New azimuthal position of the gap [degrees] (used when the dee
            opening angle is an optimization parameter).
        segment_rotations : list of float, optional
            New per-segment midpoint rotations [degrees].
        """

        if segment_angles is not None:
            if len(segment_angles) != self.n_variable_segments:
                raise ValueError(f"segment_angles must have length {self.n_variable_segments}")
            self.segment_angles = segment_angles

        if segment_radii is not None:
            if len(segment_radii) != self.n_variable_segments:
                raise ValueError(f"segment_radii must have length {self.n_variable_segments}")
            self.segment_radii = segment_radii

        if base_angle is not None:
            self.base_angle = base_angle

        if segment_rotations is not None:
            if len(segment_rotations) != self.n_variable_segments:
                raise ValueError(f"segment_rotations must have length {self.n_variable_segments}")
            self.segment_rotations = segment_rotations

        # Rebuild geometry
        self._build_geometry()

    def gap_width_at(self, r):
        """Local gap channel width [m] at radius ``r`` (scalar or array).

        Linear taper from ``gap_width_inner`` at ``r_min`` to the nominal
        ``gap_width`` at ``gap_taper_radius``, constant nominal beyond;
        constant ``gap_width`` everywhere when no taper is configured.
        """
        if self.gap_width_inner is None:
            if np.isscalar(r):
                return self.gap_width
            return np.full(np.shape(r), self.gap_width)
        frac = np.clip((np.asarray(r, dtype=float) - self.r_min)
                       / (self.gap_taper_radius - self.r_min), 0.0, 1.0)
        w = self.gap_width_inner + frac * (self.gap_width - self.gap_width_inner)
        return float(w) if np.isscalar(r) else w

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
            # Cramer's rule for A = [d_cav, -d_part]: det(A) = -det (the sign
            # was folded into `det` above), so s_part = det_s / det(A) needs the
            # minus sign. Without it the crossing was found on the chord AFTER
            # the true crossing (s_part = -s_true), i.e. one step late, and a
            # chord ending exactly on the gap line produced a missed or a
            # doubled kick (seen as 7 or 9 kicks per turn in the HCHC-60
            # accelerated orbit, 2026-09-03).
            s_part[valid_mask] = -det_s_part[valid_mask] / det[valid_mask]

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
                          segment_ids: np.ndarray,
                          t: float,
                          dt: float,
                          design,
                          pusher) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply RF kicks perpendicular to the actually-crossed gap segment.

        Physics model (thin gap): the gap accelerates along its NORMAL (the
        segment's perpendicular), leaving the momentum component parallel to the
        gap line unchanged. The conserved/added split therefore uses the crossed
        segment's own orientation (``segment_ids``), so varying the segment
        geometry changes the kick - not just where the particle crosses.

        All beta*gamma <-> velocity conversions are done locally from the TOTAL
        speed |v| (norm-based), avoiding both the component-wise error in the
        generic helpers and any mutation of the shared ``design.beam`` scratch
        state. ``t_cross`` and ``segment_ids`` are full-length (aligned with the
        global particle arrays); ``r_old_array`` is unused (kept for signature
        stability).
        """
        c = CLIGHT
        n_particles = len(r_new_array)
        n_crossed = int(np.sum(crossed_mask))

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

        # Old energy from the TOTAL speed (norm-based, relativistic).
        speed = np.linalg.norm(v_cavity, axis=1)
        gamma = 1.0 / np.sqrt(1.0 - (speed / c) ** 2)
        e_old_mev = (gamma - 1.0) * design.species.mass_mev

        # Transit-time factor (local gap width at the crossing radius when
        # the gap is tapered).
        r_crossing = np.linalg.norm(r_cavity[:, :2], axis=1)
        transit_time = self.gap_width_at(r_crossing) / speed
        omega_tau_half = self.omega * transit_time / 2.0
        ttf = np.sin(omega_tau_half) / omega_tau_half

        # Energy gain at the crossing.
        total_phase = self.get_total_phase_rad()
        d_e_mev = 1e-6 * design.species.q * self.voltage * ttf * np.cos(
            self.omega * t_cavity + total_phase
        )

        # New total relativistic momentum magnitude |u_new| = beta*gamma.
        gamma_new = (e_old_mev + d_e_mev) / design.species.mass_mev + 1.0
        bg_new = np.sqrt(np.maximum(gamma_new ** 2 - 1.0, 0.0))

        # Relativistic momentum vector u = gamma * v / c (norm-based).
        u = (gamma / c)[:, None] * v_cavity

        # Per-particle gap orientation from the segment actually crossed.
        seg = segment_ids[crossed_indices]
        dir_along = np.array([self.segments[s]['direction'] for s in seg])   # (n_crossed, 2)
        perp = np.array([self.segments[s]['perp_2d'] for s in seg])          # (n_crossed, 2)

        u_xy = u[:, :2]
        u_z = u[:, 2]
        u_along = np.sum(u_xy * dir_along, axis=1)   # parallel to gap line -> conserved
        u_perp_old = np.sum(u_xy * perp, axis=1)     # along gap normal -> kicked

        # Re-solve the perpendicular (kicked) component for the new energy,
        # keeping the along-gap and vertical components fixed.
        u_perp_new_sq = bg_new ** 2 - u_along ** 2 - u_z ** 2
        u_perp_new = np.sign(u_perp_old) * np.sqrt(np.maximum(u_perp_new_sq, 0.0))

        u_xy_new = u_along[:, None] * dir_along + u_perp_new[:, None] * perp
        u_new = np.column_stack([u_xy_new, u_z])

        # Back to velocity: v = c * u / sqrt(1 + |u|^2) (norm-based).
        u_new_mag2 = np.sum(u_new ** 2, axis=1)
        v_new = c * u_new / np.sqrt(1.0 + u_new_mag2)[:, None]

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

        self.n_crossings += n_crossed
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


# ============================================================================
# Dee-system parametrization: a "dee" (cavity) is described by its CENTER
# azimuth and OPENING angle; the two accelerating gaps sit at
# center +- opening/2 with a 180 deg internal phase flip. The gap list is
# ordered [dee0_entry, dee0_exit, dee1_entry, dee1_exit, ...].
# ============================================================================
class DeeSystem:
    """Descriptor of a dee-based RF system (center + opening parametrization).

    Holds the construction parameters so the OPENING ANGLE can later be changed
    (or optimized): ``apply_opening_angle`` recomputes every gap's azimuthal
    position from the stored centers without touching segment geometry.
    """

    def __init__(self, center_angles: List[float], opening_angle: float,
                 gaps: List[RFCavity]):
        self.center_angles = list(center_angles)
        self.opening_angle = float(opening_angle)
        self.gaps = gaps

    @property
    def n_dees(self) -> int:
        return len(self.center_angles)

    def gap_angles(self, opening_angle: Optional[float] = None) -> List[float]:
        """Gap azimuths [deg] for the given (or stored) opening angle."""
        op = self.opening_angle if opening_angle is None else float(opening_angle)
        out = []
        for c in self.center_angles:
            out.extend([c - op / 2.0, c + op / 2.0])
        return out

    def apply_opening_angle(self, opening_angle: float):
        """Reposition all gaps for a new opening angle (segments unchanged)."""
        for gap, ang in zip(self.gaps, self.gap_angles(opening_angle)):
            gap.update_geometry(base_angle=ang)


def create_dee_system(r_min: float,
                      r_max: float,
                      center_angles: List[float],
                      opening_angle: Optional[float] = None,
                      voltage: float = 60000.0,
                      frequency: float = 42e6,
                      phases: Optional[List[float]] = None,
                      gap_width: float = 0.01,
                      harmonic: int = 4,
                      n_variable_segments: int = 0,
                      segment_angles: Optional[List[float]] = None,
                      segment_radii: Optional[List[float]] = None,
                      gap_width_inner: Optional[float] = None,
                      gap_taper_radius: Optional[float] = None) -> DeeSystem:
    """Create an N-dee RF system from (center angle, opening angle).

    Parameters
    ----------
    center_angles : list of float
        Azimuthal CENTER of each dee [deg]. Two-dee system: e.g. [90, 270].
    opening_angle : float, optional
        Angular width of each dee [deg]; the gaps sit at center +- opening/2.
        Default: 180/harmonic (the classical synchronous dee angle).
    phases : list of float, optional
        RF phase of each dee's ENTRY gap [deg]; the exit gap is +180 deg.
    gap_width_inner, gap_taper_radius : float, optional
        Linear gap-width taper toward the center (see RFCavity): width is
        ``gap_width_inner`` at ``r_min``, growing to the nominal ``gap_width``
        at ``gap_taper_radius``, constant beyond.

    Returns
    -------
    DeeSystem (gaps in ``.gaps``, ordered entry/exit per dee).
    """
    if opening_angle is None:
        opening_angle = 180.0 / harmonic
    if phases is None:
        phases = [0.0] * len(center_angles)
    if len(phases) != len(center_angles):
        raise ValueError("phases must match center_angles in length")

    gaps = []
    for c, ph in zip(center_angles, phases):
        for ang, gph in ((c - opening_angle / 2.0, ph),
                         (c + opening_angle / 2.0, ph + 180.0)):
            gaps.append(RFCavity(
                r_min, r_max, ang, voltage, frequency, harmonic, gph, gap_width,
                n_variable_segments=n_variable_segments,
                segment_angles=list(segment_angles) if segment_angles else None,
                segment_radii=list(segment_radii) if segment_radii else None,
                gap_width_inner=gap_width_inner,
                gap_taper_radius=gap_taper_radius,
            ))
    return DeeSystem(center_angles, opening_angle, gaps)


# Helper functions remain backward compatible (implemented on the dee scheme).
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
                             segment_radii: Optional[List[float]] = None,
                             gap_width_inner: Optional[float] = None,
                             gap_taper_radius: Optional[float] = None) -> Tuple[RFCavity, RFCavity]:
    """Create double-gap cavity: first gap at ``angle``, opening 180/harmonic."""
    opening = 360.0 / (2.0 * harmonic)
    dee = create_dee_system(r_min, r_max, [angle + opening / 2.0], opening,
                            voltage, frequency, [phase], gap_width, harmonic,
                            n_variable_segments, segment_angles, segment_radii,
                            gap_width_inner, gap_taper_radius)
    return dee.gaps[0], dee.gaps[1]


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
                              segment_radii: Optional[List[float]] = None,
                              gap_width_inner: Optional[float] = None,
                              gap_taper_radius: Optional[float] = None) -> List[RFCavity]:
    """Create 4-cavity system (``angles`` are the FIRST-gap azimuths)."""

    if phases is None:
        phases = [0.0] * 4

    if len(angles) != 4 or len(phases) != 4:
        raise ValueError("Must provide 4 angles and 4 phases")

    opening = 360.0 / (2.0 * harmonic)
    dees = create_dee_system(r_min, r_max,
                             [a + opening / 2.0 for a in angles], opening,
                             voltage, frequency, list(phases), gap_width, harmonic,
                             n_variable_segments, segment_angles, segment_radii,
                             gap_width_inner, gap_taper_radius)
    return dees.gaps


def snap_nodes_between_turns(design_or_cavities, trajectory,
                             verbose: bool = True):
    """Shift each gap's segment node radii midway between its turn crossings.

    Uses a reference tracked trajectory (e.g. the thin-gap winner's
    ``trajectory_reference``): for every gap, the radii where the trajectory
    actually crosses its segments are extracted, and each variable-segment
    node radius is moved to the midpoint of the two crossings that bracket
    it - the segment joints (kick-direction changes, electrode bridge jogs)
    then never coincide with a beam crossing. Several nodes falling in the
    same bracket are distributed evenly inside it; nodes below the first or
    above the last crossing stay put. Geometry is updated IN PLACE.

    Returns a list of (label, old_radii, new_radii, crossing_radii) tuples.
    """
    cavities = getattr(design_or_cavities, 'rf_cavities', design_or_cavities)
    xy = np.asarray(trajectory, dtype=float)[:, :2]
    report = []
    for gi, cav in enumerate(cavities):
        if cav.n_variable_segments == 0:
            continue
        crossed, t_cross, _ = cav.check_crossings_batch(xy[:-1], xy[1:])
        idx = np.where(crossed)[0]
        if len(idx) < 2:
            continue
        pts = xy[idx] + t_cross[idx, None] * (xy[idx + 1] - xy[idx])
        crossings = np.sort(np.hypot(pts[:, 0], pts[:, 1]))
        # merge near-duplicate (grazing) crossings
        keep = [crossings[0]]
        for r in crossings[1:]:
            if r - keep[-1] > 1e-3:
                keep.append(r)
        crossings = np.asarray(keep)

        old = list(cav.segment_radii)
        new = list(old)
        # group node indices by the crossing bracket they fall into
        brackets = {}
        for i, r in enumerate(old):
            k = int(np.searchsorted(crossings, r))
            if 0 < k < len(crossings):
                brackets.setdefault(k, []).append(i)
        for k, nodes in brackets.items():
            lo, hi = crossings[k - 1], crossings[k]
            for j, i in enumerate(sorted(nodes, key=lambda i: old[i])):
                new[i] = lo + (hi - lo) * (j + 1) / (len(nodes) + 1)
        # keep strict monotonicity inside (r_min, r_max)
        prev = cav.r_min
        for i in range(len(new)):
            new[i] = min(max(new[i], prev + 1e-3), cav.r_max - 1e-3)
            prev = new[i]
        cav.update_geometry(segment_radii=new)
        report.append((f"gap{gi}", old, new, crossings))
        if verbose:
            moves = ", ".join(f"{o*1000:.1f}->{n*1000:.1f}"
                              for o, n in zip(old, new))
            print(f"[snap_nodes] gap{gi} ({cav.base_angle:.1f} deg): "
                  f"{len(crossings)} crossings, node radii [mm]: {moves}")
    return report


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