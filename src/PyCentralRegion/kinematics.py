"""The speed of light and the kinetic energy of tracked ions, one way.

``kinetic_energy_mev`` gives the energies the tracker writes (final states, snapshots, tracks, probe files) and the
ones the analysis computes from recorded velocities (``analysis.runs.virtual_probe``), so both use the same floats.
The compiled kernels (``fast_track``, ``fast_track3d``) keep their own inline formulas for numba. numpy only.
"""
import numpy as np

CLIGHT = 299792458.0             # speed of light [m/s]


def kinetic_energy_mev(v, mass_mev):
    """Kinetic energy (gamma - 1) m [MeV] of velocities v (n, 3) [m/s], mass_mev the rest mass [MeV / c^2]."""
    beta2 = np.sum(v * v, axis=1) / CLIGHT ** 2
    return (1.0 / np.sqrt(1.0 - beta2) - 1.0) * mass_mev
