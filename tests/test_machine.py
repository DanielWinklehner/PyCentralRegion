"""machine: the supported layout (four dees with radial gap lines), the inflector turn, the seed's reference orbit;
frames: the rotation about the machine axis (also the turn of a rotated gap segment) and the azimuth wrap."""
import dataclasses
import types

import numpy as np
import pytest

from PyCentralRegion import config as C
from PyCentralRegion.frames import rotation_z, wrap_deg
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


def test_rotation_z():
    """The (3, 3) turn about the machine axis (+ccw) and, with dim 2, its x-y block, by the same floats."""
    R3, R2 = rotation_z(4.726), rotation_z(4.726, 2)
    c, s = np.cos(np.radians(4.726)), np.sin(np.radians(4.726))
    np.testing.assert_array_equal(R3, [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(R2, R3[:2, :2])
    np.testing.assert_allclose(R3 @ R3.T, np.eye(3), rtol=0.0, atol=1e-15)
    assert rotation_z(90.0) @ np.array([1.0, 0.0, 0.3]) == pytest.approx([0.0, 1.0, 0.3], abs=1e-15)
    assert rotation_z(90.0, 2) @ np.array([1.0, 0.0]) == pytest.approx([0.0, 1.0], abs=1e-15)
    pts = np.array([[0.07, -0.02, 0.001], [-0.03, 0.05, -0.002]])
    np.testing.assert_allclose(pts @ R3.T @ rotation_z(-4.726).T, pts, rtol=0.0, atol=1e-15)
    assert pts[0] @ R3.T == pytest.approx(rotate_z(pts[0], 4.726), abs=1e-15)     # machine.rotate_z turns alike
    with pytest.raises(ValueError, match='dim'):
        rotation_z(10.0, 4)


def test_gap_segment_rotation():
    """A rotated gap segment (rf_cavity) turns about its own midpoint by rotation_z, R @ (p - mid), so the gap lines
    of the build turn by the same floats; the chain goes on from the unrotated ends, the fixed radial segment stays."""
    pytest.importorskip('PyPATools')
    from PyCentralRegion.rf_cavity import RFCavity
    kw = dict(n_variable_segments=2, segment_angles=[12.0, -7.0], segment_radii=[0.05, 0.11])
    flat = RFCavity(0.012, 0.25, 37.0, **kw)
    cav = RFCavity(0.012, 0.25, 37.0, segment_rotations=[4.726, -20.0], **kw)
    for a, b, rot in zip(flat.segments, cav.segments, (4.726, -20.0)):
        mid = 0.5 * (a['p1'] + a['p2'])
        np.testing.assert_array_equal(b['p1'], mid + rotation_z(rot) @ (a['p1'] - mid))
        np.testing.assert_array_equal(b['p2'], mid + rotation_z(rot) @ (a['p2'] - mid))
    assert [s['type'] for s in cav.segments] == ['variable', 'variable', 'fixed']
    np.testing.assert_array_equal(np.r_[cav.segments[2]['p1'], cav.segments[2]['p2']],
                                  np.r_[flat.segments[2]['p1'], flat.segments[2]['p2']])


def test_wrap_deg():
    a = [-540.0, -180.0, -179.5, 0.0, 179.5, 180.0, 359.0, 720.25]
    np.testing.assert_array_equal(wrap_deg(a), [-180.0, -180.0, -179.5, 0.0, 179.5, -180.0, -1.0, 0.25])
    assert float(wrap_deg(190.0)) == -170.0 and wrap_deg(np.zeros((2, 3))).shape == (2, 3)


def test_seed_without_orbit_refused():
    with pytest.raises(C.ConfigError, match='reference orbit'):
        seed_orbit(types.SimpleNamespace())
    traj = np.zeros((5, 3))
    assert seed_orbit(types.SimpleNamespace(trajectory_reference=traj)).shape == (5, 3)
