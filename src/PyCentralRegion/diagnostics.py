"""
diagnostics.py - Centralized Diagnostic ToolsHandles Poincaré analysis, turn metrics, isochronism, beam statistics.
Used by SEOFinder and acceleration optimizers.Part of: PyCentralRegion module
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PyPATools.particles import ParticleDistribution
from scipy import constants

CLIGHT = constants.speed_of_light


@dataclass
class PoincarePoint:
    """Single Poincaré section crossing."""
    turn: int
    r: float  # m
    vr: float  # m/s
    z: float  # m
    vz: float  # m/s
    energy_mev: float
    phase_deg: Optional[float] = None  # RF phase if applicable
    time: float = 0.0

@ dataclass
class TurnStatistics:
    """Beam statistics for one turn."""
    turn: int
    mean_r: float  # m
    std_r: float  # m
    mean_energy_mev: float
    std_energy_mev: float
    mean_x: float
    mean_y: float
    emittance_r: float = 0.0
    n_active: int = 0  # surviving particles at this turn


class PoincareAnalyzer:
    """
    Analyzes Poincaré sections for orbit closure and tune.

    Parameters
    ----------
    section_angle : float
        Angle [rad] defining Poincaré section (default: 0 = +x axis)
    arm_angle : float
        Accumulated azimuth [rad] the particle must travel before the FIRST
        crossing may be logged (default 0 = disabled). Set it to ~pi when the
        particle is launched ON the section: the crossing test only sees
        consecutive tracked positions, so a launch a fraction of a step behind
        the section registers a crossing within the first few steps and turn 0
        then spans almost no azimuth. Arming makes "turn N" mean N genuine
        revolutions from launch whatever the injection geometry.
    """


    def __init__(self, section_angle: float = 0.0, arm_angle: float = 0.0):
        self.section_angle = section_angle
        self.arm_angle = float(arm_angle)
        self.crossings = []
        self._azimuth_travelled = 0.0
        self._armed = self.arm_angle <= 0.0


    def check_crossing(self,
                       r_old: np.ndarray,
                       r_new: np.ndarray) -> Tuple[bool, Optional[float]]:
        """
        Check if trajectory crosses Poincaré section.

        For section_angle = 0: checks y crossing from negative to positive.

        Returns
        -------
        crossed : bool
        t_frac : float or None
            Fractional timestep of crossing (0-1)
        """

        if not self._armed:
            # Signed accumulation (motion is counter-clockwise); a brief
            # backward excursion must not arm the detector early.
            dtheta = (np.arctan2(r_new[1], r_new[0])
                      - np.arctan2(r_old[1], r_old[0]))
            if dtheta > np.pi:
                dtheta -= 2.0 * np.pi
            elif dtheta < -np.pi:
                dtheta += 2.0 * np.pi
            self._azimuth_travelled += dtheta
            if self._azimuth_travelled < self.arm_angle:
                return False, None
            self._armed = True

        if self.section_angle == 0.0:
            # Optimized for +x axis: y crosses zero upward
            if r_old[1] <= 0.0 < r_new[1]:
                t_frac = -r_old[1] / (r_new[1] - r_old[1])
                return True, t_frac
            return False, None

        else:
            # General angle
            theta_old = np.arctan2(r_old[1], r_old[0])
            theta_new = np.arctan2(r_new[1], r_new[0])

            # Unwrap
            dtheta = theta_new - theta_old
            if dtheta > np.pi:
                dtheta -= 2.0 * np.pi
            elif dtheta < -np.pi:
                dtheta += 2.0 * np.pi

            if dtheta > 0:  # Forward motion
                angle_diff_old = self.section_angle - theta_old
                angle_diff_new = self.section_angle - theta_new

                angle_diff_old = np.arctan2(np.sin(angle_diff_old), np.cos(angle_diff_old))
                angle_diff_new = np.arctan2(np.sin(angle_diff_new), np.cos(angle_diff_new))

                if angle_diff_old * angle_diff_new < 0 and angle_diff_old > 0:
                    t_frac = angle_diff_old / (angle_diff_old - angle_diff_new)
                    return True, t_frac

            return False, None


    def record_crossing(self,
                        turn: int,
                        r: np.ndarray,
                        v: np.ndarray,
                        time: float,
                        species,
                        phase_deg: Optional[float] = None):
        """Record a Poincaré crossing."""

        # Calculate energy
        v_mag = np.linalg.norm(v)
        gamma = 1.0 / np.sqrt(1.0 - (v_mag / CLIGHT) ** 2)
        energy_mev = (gamma - 1.0) * species.mass_mev

        point = PoincarePoint(
            turn=turn,
            r=r[0],  # Radial position at crossing
            vr=v[0],  # Radial velocity
            z=r[2],
            vz=v[2],
            energy_mev=energy_mev,
            phase_deg=phase_deg,
            time=time
        )

        self.crossings.append(point)


    def analyze_closure(self, tolerance_mm: float = 1.0) -> Dict:
        """
        Analyze orbit closure from Poincaré points.

        Returns
        -------
        analysis : dict
            'is_closed' : bool
            'closure_error_mm' : float (RMS)
            'frequency_hz' : float
            'tune' : float (betatron tune)
        """

        if len(self.crossings) < 2:
            return {
                'is_closed': False,
                'closure_error_mm': np.inf,
                'frequency_hz': 0.0,
                'tune': 0.0
            }

        # Closure error (radial spread)
        radii = np.array([p.r for p in self.crossings])
        closure_error_m = np.std(radii)
        closure_error_mm = closure_error_m * 1000.0

        is_closed = closure_error_mm < tolerance_mm

        # Frequency
        times = np.array([p.time for p in self.crossings])
        periods = np.diff(times)
        avg_period = np.mean(periods)
        frequency_hz = 1.0 / avg_period if avg_period > 0 else 0.0

        # Tune (betatron frequency / orbital frequency)
        tune = 0.0
        if len(self.crossings) >= 20:
            radii_detrended = radii - np.mean(radii)
            crossings_zero = np.where(np.diff(np.sign(radii_detrended)))[0]
            if len(crossings_zero) >= 4:
                betatron_periods = len(self.crossings) / (len(crossings_zero) / 2.0)
                tune = 1.0 / betatron_periods

        return {
            'is_closed': is_closed,
            'closure_error_mm': closure_error_mm,
            'frequency_hz': frequency_hz,
            'tune': tune,
            'n_crossings': len(self.crossings)
        }


def calculate_turn_metrics(traj: np.ndarray,
                           turn_ids: List[int]) -> Dict:
    """
    Calculate turn-by-turn orbit quality metrics.Parameters
    ----------
    traj : np.ndarray
        Position trajectory (nsteps, 3) [m]
    turn_ids : list of int
        Step indices where turns end

    Returns
    -------
    metrics : dict
        'r_center' : orbit-center offset per turn, CENTROID metric [m]
                     (contaminated by ~dr/2pi for accelerated orbits)
        'r_center_h1' : orbit-center offset per turn, first-harmonic fit with
                        the acceleration spiral removed [m] (preferred)
        'r_spread' : radial spread per turn [m]
        'r_avg' : average radius per turn [m]
        'dr' : turn separation [m]
        'x_center' : mean x per turn [m]
        'y_center' : mean y per turn [m]
    """


    if len(turn_ids) == 0:
        return {
            'r_center': np.array([]),
            'r_center_h1': np.array([]),
            'r_spread': np.array([]),
            'r_avg': np.array([]),
            'dr': np.array([]),
            'x_center': np.array([]),
            'y_center': np.array([])
        }

    n_turns = len(turn_ids)
    r_center = np.zeros(n_turns)
    r_center_h1 = np.zeros(n_turns)
    r_spread = np.zeros(n_turns)
    r_avg = np.zeros(n_turns)
    x_center = np.zeros(n_turns)
    y_center = np.zeros(n_turns)

    for i in range(n_turns):
        start_idx = 0 if i == 0 else turn_ids[i - 1]
        end_idx = turn_ids[i]

        traj_segment = traj[start_idx:end_idx]

        if len(traj_segment) < 2:
            continue

        # Orbit center - CENTROID metric (legacy). NOTE: for an ACCELERATED
        # orbit this conflates the true orbit-center offset with the
        # acceleration spiral: a perfectly centered spiral turn with radius
        # gain dr has its centroid offset by ~dr/(2*pi). Kept for plots and
        # backward compatibility; prefer r_center_h1 for optimization.
        x_center[i] = np.mean(traj_segment[:, 0])
        y_center[i] = np.mean(traj_segment[:, 1])
        r_center[i] = np.sqrt(x_center[i] ** 2 + y_center[i] ** 2)

        # Radial statistics
        radii = np.sqrt(traj_segment[:, 0] ** 2 + traj_segment[:, 1] ** 2)
        r_spread[i] = np.std(radii)
        r_avg[i] = np.mean(radii)

        # Orbit center - FIRST-HARMONIC metric (spiral-corrected): fit
        #   r(theta) = r0 + a*theta + c1*cos(theta) + s1*sin(theta)
        # over the turn. The linear term absorbs the acceleration spiral; the
        # first-harmonic amplitude hypot(c1, s1) is the actual orbit-center
        # offset (for an off-center circle r ~ R + dx*cos + dy*sin).
        if len(traj_segment) >= 8:
            theta = np.unwrap(np.arctan2(traj_segment[:, 1], traj_segment[:, 0]))
            theta = theta - theta[0]
            A = np.column_stack([np.ones_like(theta), theta,
                                 np.cos(theta), np.sin(theta)])
            try:
                coef, *_ = np.linalg.lstsq(A, radii, rcond=None)
                r_center_h1[i] = float(np.hypot(coef[2], coef[3]))
            except np.linalg.LinAlgError:
                r_center_h1[i] = r_center[i]
        else:
            r_center_h1[i] = r_center[i]

    dr = np.diff(r_avg)

    return {
        'r_center': r_center,
        'r_center_h1': r_center_h1,
        'r_spread': r_spread,
        'r_avg': r_avg,
        'dr': dr,
        'x_center': x_center,
        'y_center': y_center
    }


def analyze_isochronism(orbits: List) -> Dict:
    """
    Analyze isochronism from list of static orbits.Parameters
    ----------
    orbits : list
        List of StaticOrbit objects with frequency_hz

    Returns
    -------
    analysis : dict
        'frequencies_mhz' : array
        'energies_mev' : array
        'freq_avg_mhz' : float
        'freq_std_mhz' : float
        'freq_variation_percent' : float
        'is_isochronous' : bool (< 1% variation)
    """


    freqs = np.array([o.frequency_hz / 1e6 for o in orbits if o.frequency_hz > 0])
    energies = np.array([o.energy_kev / 1000.0 for o in orbits if o.frequency_hz > 0])

    if len(freqs) == 0:
        return {
            'frequencies_mhz': np.array([]),
            'energies_mev': np.array([]),
            'freq_avg_mhz': 0.0,
            'freq_std_mhz': 0.0,
            'freq_variation_percent': np.inf,
            'is_isochronous': False
        }

    freq_avg = np.mean(freqs)
    freq_std = np.std(freqs)
    freq_variation = (freq_std / freq_avg * 100.0) if freq_avg > 0 else np.inf

    is_isochronous = freq_variation < 1.0

    return {
        'frequencies_mhz': freqs,
        'energies_mev': energies,
        'freq_avg_mhz': freq_avg,
        'freq_std_mhz': freq_std,
        'freq_variation_percent': freq_variation,
        'is_isochronous': is_isochronous
    }


class BeamStatisticsCollector:
    """
    Collects beam statistics during multi-particle tracking.Usage:
        collector = BeamStatisticsCollector(species, save_frequency=10)

        # In tracking loop:
        def callback(step, r, v, active, t):
            collector.record(step, r[active], v[active], t)

        # After tracking:
        stats = collector.get_statistics()
    """


    def __init__(self, species, save_frequency: int = 1):
        self.species = species
        self.save_frequency = save_frequency
        self.turn_stats = []
        self.current_turn = 0


    def record(self, step: int, r_active: np.ndarray, v_active: np.ndarray, time: float):
        """Record beam statistics at this step."""

        if step % self.save_frequency != 0:
            return

        radii = np.sqrt(r_active[:, 0] ** 2 + r_active[:, 1] ** 2)

        # Energy
        v_mag = np.linalg.norm(v_active, axis=1)
        gamma = 1.0 / np.sqrt(1.0 - (v_mag / 299792458.0) ** 2)
        energies_mev = (gamma - 1.0) * self.species.mass_mev

        stats = TurnStatistics(
            turn=self.current_turn,
            mean_r=np.mean(radii),
            std_r=np.std(radii),
            mean_energy_mev=np.mean(energies_mev),
            std_energy_mev=np.std(energies_mev),
            mean_x=np.mean(r_active[:, 0]),
            mean_y=np.mean(r_active[:, 1]),
            emittance_r=0.0,  # TODO: calculate properly
            n_active=int(len(r_active))
        )

        self.turn_stats.append(stats)


    def increment_turn(self):
        """Increment turn counter."""
        self.current_turn += 1


    def get_statistics(self) -> List[TurnStatistics]:
        """Return collected statistics."""
        return self.turn_stats
