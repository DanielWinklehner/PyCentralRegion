"""machine: the supported layout (four dees with radial gap lines), the inflector turn, the seed's reference orbit."""
import dataclasses
import types

import numpy as np
import pytest

from PyCentralRegion import config as C
from PyCentralRegion.machine import check_supported, rotate_z, seed_orbit


def test_supported_layout():
    mc = C.section_class('machine')()
    check_supported(mc)
    with pytest.raises(NotImplementedError, match='n_dees = 3'):
        check_supported(dataclasses.replace(mc, n_dees=3, dee_center_angles_deg=(30.0, 150.0, 270.0)))
    with pytest.raises(NotImplementedError, match='spiral'):
        check_supported(dataclasses.replace(mc, dee_shape='spiral'))
    with pytest.raises(C.ConfigError, match='3 angles'):
        check_supported(dataclasses.replace(mc, dee_center_angles_deg=(45.0, 135.0, 225.0)))


def test_supported_layout_from_a_run_file(tmp_path):
    (tmp_path / 'r.toml').write_text('[machine]\nn_dees = 2\ndee_center_angles_deg = [0.0, 180.0]\n')
    with pytest.raises(NotImplementedError):
        check_supported(C.section(C.load(tmp_path / 'r.toml'), 'machine'))


def test_rotate_z():
    assert rotate_z([1.0, 0.0, 0.3], 90.0) == pytest.approx([0.0, 1.0, 0.3], abs=1e-15)
    v = np.array([0.07, -0.02, 0.001])
    assert rotate_z(rotate_z(v, 4.726), -4.726) == pytest.approx(v, abs=1e-15)
    assert rotate_z(v, 4.726) is not v                                          # the input is left alone


def test_seed_without_orbit_refused():
    with pytest.raises(C.ConfigError, match='reference orbit'):
        seed_orbit(types.SimpleNamespace())
    traj = np.zeros((5, 3))
    assert seed_orbit(types.SimpleNamespace(trajectory_reference=traj)).shape == (5, 3)
