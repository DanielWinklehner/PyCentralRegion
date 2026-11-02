"""The machine frame: rotations about the machine (z) axis and azimuths, angles in degrees.

``rotation_z`` builds the rotation matrix one way (np.cos and np.sin of np.radians(deg)), so the electrode build, the
rotated gap segments (``rf_cavity``), the hand-off bunch and the spiral-inflector field turn by the same floats. A
caller keeps its own product form (``pts @ R.T`` for rows of points, ``R @ p`` for one vector): the forms round
differently. ``wrap_deg`` folds an azimuth or a phase into [-180, 180). numpy only.
"""
import numpy as np


def rotation_z(deg, dim=3):
    """The rotation by ``deg`` [deg, +ccw] about the machine axis: the (3, 3) matrix, or with ``dim`` 2 its (2, 2) x-y
    block."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    if dim == 3:
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if dim == 2:
        return np.array([[c, -s], [s, c]])
    raise ValueError(f"rotation_z: dim is 2 or 3, got {dim!r}")


def wrap_deg(a):
    """Angles [deg] wrapped to [-180, 180)."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0
