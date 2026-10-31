"""First-harmonic trim of the magnet field, a design knob for centring the first turns.

A trim (coils, or a shimmed pole) adding a first harmonic b(r) cos(theta - theta_0) to the median-plane B_z shifts the
orbit centres of the turns it acts on. Whether the beam gains from that is a cheap question - the electrodes and the
BEM field stay as they are - so the trim is a [btrim] section the optimizer can vary.

The trim is a vacuum field. From the scalar potential Phi = z f - z^3/6 lap_perp f with f = b(r) cos(theta - theta_0):

    B_z     = [b - z^2/2 (b'' + b'/r - b/r^2)] cos(theta - theta_0)
    B_r     = z b' cos(theta - theta_0)
    B_theta = -z (b / r) sin(theta - theta_0)

divergence-free exactly and curl-free to O(z^2), the order the median-plane expansion carries. b(r) is
``amplitude_t`` on [r_min_m, r_max_m] and rises / falls to 0 over ``edge_m`` on either side with the quintic
smoothstep (C2, so b'' is continuous). A first harmonic has to vanish on the axis, so r_min_m >= edge_m.

``add_trim`` adds it to the node values of the 3D map crop before the Field is made (``machine.magnet_field_3d``):
the tracked field stays one gridded map, which the compiled 3D tracker needs.
"""
import dataclasses

import numpy as np

from .config import ConfigError


@dataclasses.dataclass
class BTrimConfig:
    amplitude_t: float = 0.0         # first-harmonic amplitude of the median-plane B_z [T]; 0 = no trim
    phase_deg: float = 0.0           # azimuth of the maximum [deg]
    r_min_m: float = 0.05            # full amplitude between r_min_m and r_max_m ...
    r_max_m: float = 0.135
    edge_m: float = 0.01             # ... falling to 0 over edge_m on either side


def _smoothstep(t):
    """Quintic smoothstep s(t) and its first two derivatives, t clipped to [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    return (t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t), 30.0 * t * t * (1.0 - t) ** 2,
            60.0 * t * (1.0 - t) * (1.0 - 2.0 * t))


def profile(r, bt):
    """b(r), b'(r), b''(r) [T, T/m, T/m^2] of the trim's radial profile."""
    if bt.edge_m <= 0 or bt.r_min_m < bt.edge_m or bt.r_max_m <= bt.r_min_m:
        raise ConfigError(f"[btrim] needs 0 < edge_m <= r_min_m < r_max_m (got edge {bt.edge_m}, r_min {bt.r_min_m}, "
                          f"r_max {bt.r_max_m})")
    r = np.asarray(r, dtype=float)
    e = bt.edge_m
    s1, d1, dd1 = _smoothstep((r - (bt.r_min_m - e)) / e)
    s2, d2, dd2 = _smoothstep((r - bt.r_max_m) / e)
    w = s1 * (1.0 - s2)
    wp = (d1 * (1.0 - s2) - s1 * d2) / e
    wpp = (dd1 * (1.0 - s2) - 2.0 * d1 * d2 - s1 * dd2) / e ** 2
    a = float(bt.amplitude_t)
    return a * w, a * wp, a * wpp


def trim_field(x, y, z, bt):
    """The trim's (Bx, By, Bz) [T] at points (arrays of one shape)."""
    r = np.hypot(x, y)
    rs = np.maximum(r, 1e-12)
    th = np.arctan2(y, x)
    b, bp, bpp = profile(r, bt)
    c, s = np.cos(th - np.radians(bt.phase_deg)), np.sin(th - np.radians(bt.phase_deg))
    bz = (b - 0.5 * z * z * (bpp + bp / rs - b / rs ** 2)) * c
    br = z * bp * c
    bth = -z * (b / rs) * s
    ct, st = np.cos(th), np.sin(th)
    return br * ct - bth * st, br * st + bth * ct, bz


def add_trim(crop, bt, scaling=1.0, verbose=True):
    """Add the trim of [btrim] to a 3D map crop (``fields3d.crop_comsol_3d`` layout) in place, divided by the Field
    ``scaling`` the crop will get, so the tracked field carries the trim at ``amplitude_t``. Only the nodes inside
    r_max_m + edge_m are touched. Returns the crop."""
    if bt.amplitude_t == 0.0:
        return crop
    gx, gy, gz = (np.asarray(crop['grid'][k], dtype=float) for k in 'xyz')
    reach = bt.r_max_m + bt.edge_m
    ix = np.flatnonzero(np.abs(gx) <= reach)
    iy = np.flatnonzero(np.abs(gy) <= reach)
    X, Y, Z = np.meshgrid(gx[ix], gy[iy], gz, indexing='ij')
    bx, by, bz = trim_field(X, Y, Z, bt)
    block = (slice(ix[0], ix[-1] + 1), slice(iy[0], iy[-1] + 1), slice(None))
    for k, v in zip('xyz', (bx, by, bz)):
        crop['values'][k][block] += v / float(scaling)
    if verbose:
        print(f"[btrim] first harmonic {bt.amplitude_t * 1e4:.2f} G at {bt.phase_deg:+.1f} deg on "
              f"r = {bt.r_min_m * 1e3:.0f}..{bt.r_max_m * 1e3:.0f} mm (edges {bt.edge_m * 1e3:.0f} mm) added to the map",
              flush=True)
    return crop
