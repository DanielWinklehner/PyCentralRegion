"""The rules of the virtual probes, shared by the tracker's probes (``accelerated_orbit_finder.ProbeRecorder``, step by
step during a run) and the probes built afterwards from recorded tracks (``analysis.metrics.probe_crossings``).

A probe is the vertical half-plane from the machine axis at an azimuth [deg, atan2(y, x)], as OPAL's PROBE. The beam's
sense of rotation is the sign of its summed angular momentum (``beam_sense``). A particle starts at pass 1 when it is
already past a probe in that sense, else at pass 0 (``start_pass``), and every crossing of the half-plane increments
it, so pass p is the crossing made during revolution p from the particle's own start. A crossing is a sign change of
the signed distance from the probe's plane between two live records: it lies ``crossing_fraction`` into the record
interval, its position (and velocity) is interpolated linearly there (``interpolate``), and its ``crossing_direction``
is +1 with the beam, -1 against it. Each caller computes the signed distances and keeps the crossings on the
half-plane itself: the tracker with matrix products, the analysis component by component, which round differently.
The hand-off's snapshot launch (``handoff``) turns its particles in the sense of ``beam_sense`` too. numpy only.
"""
import numpy as np


def beam_sense(r, v):
    """+1.0 when the beam circulates counter-clockwise seen from +z, else -1.0: the sign of the summed angular momentum
    x v_y - y v_x of positions r (n, 2+) [m] and velocities v (n, 2+) (NaN entries left out; a sum of 0 is +1)."""
    return 1.0 if np.nansum(r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0]) >= 0.0 else -1.0


def start_pass(phi, angle_deg, sense):
    """The pass number (int32) a particle starts with at a probe: 1 when its azimuth phi [rad] is ahead of the probe at
    angle_deg [deg] in the direction of motion (sense +1: counter-clockwise), the difference wrapped to (-pi, pi], else
    0. phi and angle_deg broadcast."""
    return (sense * np.angle(np.exp(1j * (phi - np.deg2rad(angle_deg)))) > 0.0).astype(np.int32)


def crossing_fraction(s0, s1):
    """Where a crossing lies in its record interval, from the signed distances s0 before and s1 after it: the fraction
    s0 / (s0 - s1) of the interval."""
    return s0 / (s0 - s1)


def interpolate(x0, x1, f):
    """The rows x0 + f (x1 - x0) at the crossing fractions f (m,) between the records before (x0) and after (x1), e.g.
    positions or velocities (m, 3)."""
    return x0 + f[:, None] * (x1 - x0)


def crossing_direction(s1, sense):
    """+1 (int8) for a crossing with the beam's sense of rotation, -1 against it: the side of the plane it ends on, s1
    the signed distance after the crossing (> 0 on the counter-clockwise side; 0 counts as with the beam)."""
    return np.where(sense * s1 >= 0.0, 1, -1).astype(np.int8)
