"""[btrim]: the first-harmonic trim is a vacuum field (divergence-free exactly, curl-free to O(z^2)), its radial profile
is the documented C2 plateau, inconsistent edges are refused, add_trim adds it where it reaches, scaled, its
Cartesian components are the same harmonic as the polar form, and on the median plane the keys mean what the
defaults file says (b_c on cos(theta), b_s on sin(theta), the polar maximum at phase_deg)."""
import dataclasses
import math

import numpy as np
import pytest

from PyCentralRegion.btrim import BTrimConfig, add_trim, profile, resolved, trim_field
from PyCentralRegion.config import ConfigError

BT = BTrimConfig(amplitude_t=2e-3, phase_deg=30.0, r_min_m=0.05, r_max_m=0.135, edge_m=0.01)


def _field(x, y, z):
    return np.array(trim_field(x, y, z, BT))


def _jacobian(p, h=1e-6):
    """J[n, i, j] = dB_i / dx_j at the points p (n, 3), central differences."""
    jac = np.zeros((len(p), 3, 3))
    for j in range(3):
        dp = np.zeros(3)
        dp[j] = h
        jac[:, :, j] = (_field(*(p + dp).T) - _field(*(p - dp).T)).T / (2.0 * h)
    return jac


def _points(z, n=300):
    rng = np.random.default_rng(1)
    r, th = rng.uniform(0.042, 0.143, n), rng.uniform(0.0, 2.0 * np.pi, n)    # plateau and both edges
    return np.column_stack([r * np.cos(th), r * np.sin(th), np.full(n, z)])


def test_divergence_free():
    jac = _jacobian(_points(0.012))
    div = np.trace(jac, axis1=1, axis2=2)
    assert np.all(np.abs(div) <= 1e-6 * np.abs(jac).max(axis=(1, 2)))


def test_curl_free_to_second_order():
    def worst_curl(z):
        j = _jacobian(_points(z))
        curl = np.stack([j[:, 2, 1] - j[:, 1, 2], j[:, 0, 2] - j[:, 2, 0], j[:, 1, 0] - j[:, 0, 1]], axis=1)
        return np.abs(curl).max(), np.abs(j).max()
    c0, scale = worst_curl(0.0)
    c1, c2 = worst_curl(0.005)[0], worst_curl(0.010)[0]
    assert c0 <= 1e-7 * scale                     # curl-free on the median plane
    assert c2 / c1 == pytest.approx(4.0, rel=0.02)  # and the residual grows as z^2


def test_profile_plateau_and_edges():
    a, e, r0, r1 = BT.amplitude_t, BT.edge_m, BT.r_min_m, BT.r_max_m
    b, _, _ = profile(np.array([0.0, r0 - e, r0 - e / 2, r0, 0.09, r1, r1 + e / 2, r1 + e, 0.3]), BT)
    assert b == pytest.approx([0.0, 0.0, a / 2, a, a, a, a / 2, 0.0, 0.0], abs=1e-15)
    # b' and b'' are the derivatives of b: integrated back (trapezoid) they give b and b'
    r = np.linspace(0.03, 0.16, 5201)
    b, bp, bpp = profile(r, BT)

    def integral(f):
        return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(r))])
    assert integral(bp) == pytest.approx(b - b[0], abs=1e-4 * a)
    assert integral(bpp) == pytest.approx(bp - bp[0], abs=1e-3 * np.abs(bp).max())
    joints = np.array([r0 - e, r0, r1, r1 + e])
    for side in (-1e-9, 1e-9):                     # C2: b' and b'' vanish on both sides of every joint
        _, bp_j, bpp_j = profile(joints + side, BT)
        assert np.abs(bp_j).max() < 1e-9 and np.abs(bpp_j).max() < 1e-3


@pytest.mark.parametrize('bad', [{'edge_m': 0.0}, {'r_min_m': 0.005}, {'r_max_m': 0.05}])
def test_inconsistent_profile_refused(bad):
    with pytest.raises(ConfigError, match='btrim'):
        profile(0.1, dataclasses.replace(BT, **bad))


def test_add_trim_on_a_crop():
    g, gz = np.arange(-0.2, 0.2001, 0.005), np.array([-0.01, 0.0, 0.01])
    crop = {'grid': {'x': g, 'y': g, 'z': gz}, 'values': {k: np.ones((len(g), len(g), len(gz))) for k in 'xyz'}}
    add_trim(crop, BT, scaling=-2.0, verbose=False)
    want = trim_field(*np.meshgrid(g, g, gz, indexing='ij'), BT)
    for k, w in zip('xyz', want):
        assert crop['values'][k] == pytest.approx(1.0 + w / -2.0, abs=1e-15)    # the Field's scaling restores it
    untouched = {k: v.copy() for k, v in crop['values'].items()}
    add_trim(crop, dataclasses.replace(BT, amplitude_t=0.0), verbose=False)
    assert all(np.array_equal(crop['values'][k], untouched[k]) for k in 'xyz')


def _cartesian(bt, scale=1.0):
    """bt's harmonic as Cartesian components times scale, with no polar part."""
    ph = math.radians(bt.phase_deg)
    return dataclasses.replace(bt, amplitude_t=0.0, amplitude_cos_t=scale * bt.amplitude_t * math.cos(ph),
                               amplitude_sin_t=scale * bt.amplitude_t * math.sin(ph))


def test_cartesian_components_are_the_polar_harmonic():
    """b_c cos(theta) + b_s sin(theta) is the trim of amplitude hypot(b_c, b_s) at atan2(b_s, b_c); the components add
    to a polar part; a trim without them is left as it is (a polar trim's field stays bit for bit)."""
    p = _points(0.008)
    want = np.array(trim_field(*p.T, BT))
    tol = 1e-12 * np.abs(want).max()
    cart = _cartesian(BT)
    assert np.array(trim_field(*p.T, cart)) == pytest.approx(want, abs=tol)
    assert resolved(BT) is BT
    r = resolved(cart)
    assert (r.amplitude_t, r.phase_deg, r.amplitude_cos_t, r.amplitude_sin_t) == pytest.approx(
        (BT.amplitude_t, BT.phase_deg, 0.0, 0.0), abs=1e-12)
    assert profile(0.09, cart)[0] == pytest.approx(BT.amplitude_t, abs=1e-15)
    # half of it polar and half as components is the same trim again
    mixed = dataclasses.replace(_cartesian(BT, 0.5), amplitude_t=0.5 * BT.amplitude_t, phase_deg=BT.phase_deg)
    assert np.array(trim_field(*p.T, mixed)) == pytest.approx(want, abs=tol)
    # the polar seam: -180 and +180 deg are one trim, (-A, 0) as components
    west = resolved(dataclasses.replace(BT, amplitude_t=0.0, amplitude_cos_t=-BT.amplitude_t))
    assert (west.amplitude_t, abs(west.phase_deg)) == (BT.amplitude_t, 180.0)
    # components that cancel the polar part leave no trim at all
    ph = math.radians(BT.phase_deg)
    cancel = dataclasses.replace(BT, amplitude_cos_t=-BT.amplitude_t * math.cos(ph),
                                 amplitude_sin_t=-BT.amplitude_t * math.sin(ph))
    g, gz = np.arange(-0.2, 0.2001, 0.005), np.array([-0.01, 0.0, 0.01])

    def crop_with(bt):
        crop = {'grid': {'x': g, 'y': g, 'z': gz}, 'values': {k: np.ones((len(g), len(g), len(gz))) for k in 'xyz'}}
        return add_trim(crop, bt, scaling=-2.0, verbose=False)['values']
    assert resolved(cancel).amplitude_t == 0.0
    assert all(np.array_equal(v, np.ones_like(v)) for v in crop_with(cancel).values())
    polar, comp = crop_with(BT), crop_with(cart)
    assert all(comp[k] == pytest.approx(polar[k], abs=1e-12 * BT.amplitude_t) for k in 'xyz')


def test_keys_on_the_median_plane():
    """B_z(z = 0) = b(r) [b_c cos(theta) + b_s sin(theta)] with b the profile of unit amplitude (components of either
    sign), and a polar trim's maximum lies at theta = phase_deg."""
    p = _points(0.0)
    r, th = np.hypot(p[:, 0], p[:, 1]), np.arctan2(p[:, 1], p[:, 0])
    unit = profile(r, dataclasses.replace(BT, amplitude_t=1.0))[0]
    bt = dataclasses.replace(BT, amplitude_t=0.0, amplitude_cos_t=1e-3, amplitude_sin_t=-1.5e-3)
    assert np.array(trim_field(*p.T, bt))[2] == pytest.approx(unit * (1e-3 * np.cos(th) - 1.5e-3 * np.sin(th)),
                                                              abs=1e-15)
    ring = np.radians(np.arange(0.0, 360.0, 0.25))
    on = np.column_stack([0.09 * np.cos(ring), 0.09 * np.sin(ring), np.zeros_like(ring)])
    peak = math.degrees(ring[np.argmax(np.array(trim_field(*on.T, BT))[2])])
    assert peak == pytest.approx(BT.phase_deg, abs=0.25)
