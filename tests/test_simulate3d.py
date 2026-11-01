"""simulate3d: the radial slit (its jaw raster, its place on the fast path, the placement rules, the pass numbering it
is placed by) and the inflector-rotation guard. No field cache needed: synthetic rasters and a stub tracker."""
import types

import numpy as np
import pytest

pytest.importorskip('PyPATools')

from PyCentralRegion import config as C                     # noqa: E402
from PyCentralRegion import simulate3d as S                 # noqa: E402

PHASE = -15.0


def _slit(**kw):
    return C.SlitConfig(**{'half_width_mm': 3.0, 'azimuth_deg': 315.0, 'jaw_length_mm': 6.0, 'thickness_mm': 2.0, **kw})


def _point(u, w=0.0, z=0.0, phi_deg=315.0):
    """A point at u along the ray at phi_deg, w across it (counter-clockwise), height z."""
    phi = np.deg2rad(phi_deg)
    return np.array([[u * np.cos(phi) - w * np.sin(phi), u * np.sin(phi) + w * np.cos(phi), z]])


def _stub(sl, passes, steps_per_turn=1000):
    """What BunchTracker.slit_for reads: [slit], the reference crossings per phase, z_max and steps per turn."""
    return types.SimpleNamespace(slit=sl, slit_passes={} if passes is None else {PHASE: passes},
                                 mc=types.SimpleNamespace(z_max_m=0.03),
                                 tc=types.SimpleNamespace(steps_per_turn=steps_per_turn))


def test_slit_raster_jaws():
    """Jaws [r_c - hw - L, r_c - hw] and [r_c + hw, r_c + hw + L] on the ray, |w| <= t / 2, full height; the edges
    are good to the raster's 0.1 mm (points 0.15 mm off an edge fall on the right side)."""
    test, jaws = S.slit_raster(_slit(), 0.150, 0.03)
    assert np.allclose(jaws, [[0.141, 0.147], [0.153, 0.159]], rtol=0.0, atol=1e-12)

    def inside(u, w=0.0, z=0.0):
        return bool(test(_point(u, w, z))[0])
    for u in (0.150, 0.14715, 0.15285, 0.14085, 0.15915):          # the opening, and beyond the jaws
        assert not inside(u), u
    for u in (0.14115, 0.1445, 0.14685, 0.15315, 0.156, 0.15885):   # the jaws, at every height
        assert inside(u) and inside(u, z=0.029) and inside(u, z=-0.029), u
    assert inside(0.156, w=0.00085) and inside(0.156, w=-0.00085)
    assert not inside(0.156, w=0.00115) and not inside(0.156, w=-0.00115)
    assert not inside(-0.156)                                       # the opposite ray is free


def test_slit_is_the_fast_paths_second_raster():
    """Combined with the model's metal raster, the slit reaches the compiled 3D kernel as its second raster."""
    from PyCentralRegion.electrodes3d import combine_obstacles
    from PyCentralRegion.fast_track3d import _raster_parts
    xs, zs = np.arange(-0.2, 0.2 + 1e-3, 2e-3), np.arange(-0.03, 0.0305, 1e-2)
    metal = S._raster_inside(np.zeros((len(xs), len(xs), len(zs)), dtype=bool), xs, zs)
    slit, _ = S.slit_raster(_slit(), 0.150, 0.03)
    parts = _raster_parts(combine_obstacles(metal, slit))
    assert len(parts) == 2
    _, grid, ox, oy, oz, hx, hy, hz = parts[1]
    assert (hx, hy, hz) == pytest.approx((S.SLIT_H_M, S.SLIT_H_M, 0.03))
    assert int(grid.sum()) == 3 * int(slit.grid[:, :, 0].sum()) > 0
    both = combine_obstacles(metal, slit)
    assert both(_point(0.156))[0] and not both(_point(0.150))[0]


def test_slit_placement():
    """Centre = the reference radius at pass_index plus offset_mm (or r_center_mm with -1); passes that cross a jaw
    are reported; a missing pass, a missing reference run and a jaw thinner than a step are refused."""
    passes = {1: 0.100, 2: 0.126, 3: 0.150, 4: 0.156}
    test, rec = S.BunchTracker.slit_for(_stub(_slit(offset_mm=0.5), passes), PHASE)
    assert rec['r_center_mm'] == pytest.approx(150.5)
    assert np.allclose(rec['jaws_mm'], [[141.5, 147.5], [153.5, 159.5]], rtol=0.0, atol=1e-9)
    assert rec['reference_on_jaw_mm'] == {'4': pytest.approx(156.0)}            # pass 4 runs into the outer jaw
    assert rec['columns'] == int(test.grid[:, :, 0].sum()) and not test(_point(0.1505))[0]
    _, rec = S.BunchTracker.slit_for(_stub(_slit(pass_index=-1, r_center_mm=112.0), passes), PHASE)
    assert rec['r_center_mm'] == pytest.approx(112.0) and rec['reference_on_jaw_mm'] == {}
    assert S.BunchTracker.slit_for(_stub(_slit(half_width_mm=0.0), None), PHASE) == (None, None)
    with pytest.raises(C.ConfigError, match='pass_index = 5'):
        S.BunchTracker.slit_for(_stub(_slit(pass_index=5), passes), PHASE)
    with pytest.raises(RuntimeError, match='reference'):
        S.BunchTracker.slit_for(_stub(_slit(), None), PHASE)
    with pytest.raises(C.ConfigError, match='step over'):                     # 2 pi 0.159 m / 300 = 3.3 mm > 2 mm
        S.BunchTracker.slit_for(_stub(_slit(), passes, steps_per_turn=300), PHASE)


def test_reference_passes_on_the_slit_ray():
    """The slit is placed by the probes' pass numbering: a particle starting just past the ray (in its direction of
    motion) meets it first on pass 1, a revolution later."""
    from PyCentralRegion.accelerated_orbit_finder import ProbeRecorder
    n = 4000
    travel = np.linspace(0.0, 3.5 * 2.0 * np.pi, n)                  # 3.5 turns counter-clockwise from 320 deg
    th, r = np.deg2rad(320.0) + travel, 0.07 + 0.01 * travel / (2.0 * np.pi)
    xyz = np.column_stack([r * np.cos(th), r * np.sin(th), np.zeros(n)])
    v = np.gradient(xyz, axis=0)
    pr = ProbeRecorder([315.0])
    for k in range(n):
        pr.record(k, xyz[k:k + 1], v[k:k + 1], np.array([True]), float(k))
    d = pr.data()[0]
    assert list(d['pass']) == [1, 2, 3] and list(d['direction']) == [1, 1, 1]
    expect = 0.07 + 0.01 * np.array([355.0, 715.0, 1075.0]) / 360.0
    assert np.hypot(d['xyz'][:, 0], d['xyz'][:, 1]) == pytest.approx(expect, abs=2e-6)


def test_si_rotation_guard():
    model = types.SimpleNamespace(params={'si_rotation_deg': 4.726})
    S.check_si_rotation(model, 4.726)
    S.check_si_rotation(types.SimpleNamespace(params={}), 7.0)        # a model that does not record it
    with pytest.raises(C.ConfigError, match='turned by 4.726'):
        S.check_si_rotation(model, 0.0, 'fields.pkl')
