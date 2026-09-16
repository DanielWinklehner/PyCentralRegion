"""
fast_track3d.py - Compiled (numba) MULTI-PARTICLE 3D tracking for the central
region: the fast path of ``AcceleratedOrbitFinder.track_with_rf`` in 3D.

The general path drives ``PyPATools.trackers.Tracker``: a Python loop that per
step calls compiled kernels (field interpolation, pusher) and Python hooks
(timed release, space-charge kick, terminators, the finder's callback). Its
interpreter and fork/join overhead is 1-2 ms per step whatever the particle
count. This module runs the same physics for a CHUNK of steps inside one
``njit`` kernel and hands the recorded steps to the very same callback
afterwards, so every diagnostic (Poincare crossings, per-turn statistics, the
sparse beam record, the reference trajectory) is produced by unchanged code -
exactly what ``fast_track.py`` does for one particle in 2D.

What the kernel reproduces per step, in ``Tracker.run`` order:
  1. ``TimedField.set_time(t + dt/2)``  - the RF phase frozen at the step midpoint;
  2. ``Pusher._boris_rel_step_batch`` on the ACTIVE particles only:
     ``r_half = r + v dt/2``, ONE E and ONE B evaluation at ``r_half``,
     ``boris_rel_dkd_batch`` kick of ``u = gamma v`` over dt, ``r = r_half + v_new dt/2``;
  3. interactions in ``engine.extra_interactions`` order: ``TimedRelease``
     (birth_step == step + 1 becomes active at the END of step ``step``), then
     ``SpaceChargeKick`` (the per-particle E_sc/B_sc impulse, every step);
  4. terminators: ``RadialVerticalTerminator(r_max, z_max)`` over ALL rows, then
     ``MetalTerminator`` over the active ones in ascending index order, the
     virtual reference particle exempt (its contacts logged in ``ref_hits``);
  5. ``t += dt``; the finder's callback - replayed in Python afterwards.

The E-field stack it understands is the one the 3D bunch deck installs:
``StaticPlusRFField(rf=TimedField(RadialBlendField(bem3d, cavity map [, RadialTaperField]),
omega, phase), statics=[inflector 3D field])`` over gridded ``Field``s with the
numba interpolator backend, plus the 3D magnet map as B.

There are two modes, because the two compiled kernels the Python path leans on -
``interpolators._interp3d_batch3`` and ``pusher.boris_rel_dkd_batch`` - are both
``parallel=True``, and a parfor contracts its arithmetic differently from EVERY
scalar spelling of the same source. ``_interp3d_batch3`` disagrees by an ulp
with ``_interp3d_single`` and with the per-component ``_interp3d_batch`` on a few
per cent of points; ``boris_rel_dkd_batch`` disagrees with its own per-particle
body on about one input in 1e5. A per-particle loop therefore CANNOT be
bit-identical to the Python path, however faithfully it is written:

  ``interp='batch'`` (exact)  the kernel calls ``_interp3d_batch3`` and
      ``boris_rel_dkd_batch`` themselves, on the same point sets and in the same
      order as the Python path, and reproduces it BIT FOR BIT. It keeps their
      thread fork/join, so it is the verification mode rather than the fast one.
  ``interp='point'`` (default, fast)  interpolation and push are inlined per
      particle (the verbatim bodies of the two kernels' loops, over the same leaf
      functions, with a per-particle interpolation cell hint that returns the
      same cell as the plain search). Agreement with the Python path is then a
      few ulp per evaluation, which after two turns is |dr| ~ 3e-16 m.

The space-charge SOLVE stays in Python: the driver alternates kernel chunks with
``SpaceChargeKick.apply`` at the steps where a solve (or a gather for particles
released since the last solve) is due, so the hook's own unchanged code decides
the fields and the kernel only applies the impulse in between.

Anything outside this configuration raises ``FastPathUnavailable`` and the
finder takes the general path. Opt-in via ``finder.fast3d = True``; OFF by
default. ``finder.fast3d_interp = 'batch'`` selects the exact mode.
``result.metadata['fast_path']`` says which path ran.

Part of: PyCentralRegion (2026-09-16)
"""
import time

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:                                   # pragma: no cover
    HAS_NUMBA = False
    prange = range

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]

from PyPATools.global_variables import CLIGHT
from PyPATools.field import Field, TimedField

if HAS_NUMBA:
    from PyPATools.field_src.interpolators import (_searchsorted_numba, _searchsorted_with_hint,
                                                   _trilinear_cell, _interp3d_batch3)
    from PyPATools.pusher import boris_rel_dkd_batch


class FastPathUnavailable(Exception):
    """The configuration is outside what the kernel reproduces."""


# Recorded (r, v, active, t) per step of a chunk; the chunk length is chosen so
# these stay around this many bytes (the spec asks for a few MB).
MAX_RECORD_BYTES = 12_000_000
MAX_CHUNK_STEPS = 1024

_EMPTY1 = np.zeros(2, dtype=np.float64)
_EMPTY3 = np.zeros((2, 2, 2), dtype=np.float64)
_EMPTY3U8 = np.zeros((2, 2, 2), dtype=np.uint8)
_ZERO3 = np.zeros(3, dtype=np.float64)

# A packed field: (mode, gx, gy, gz, vx, vy, vz, scaling, fill_value, const)
# mode 0 = zero, 1 = 0D constant, 2 = gridded 3D (trilinear, fill outside).
_ZERO_FIELD = (0, _EMPTY1, _EMPTY1, _EMPTY1, _EMPTY3, _EMPTY3, _EMPTY3, 1.0, 0.0, _ZERO3)
# A packed raster: (use, grid uint8, ox, oy, oz, hx, hy, hz)
_NO_RASTER = (0, _EMPTY3U8, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)


# ============================================================================
# numba leaves
# ============================================================================
@njit(cache=True, fastmath=True, inline='always')
def _interp3_point(x, y, z, grid_x, grid_y, grid_z, vx, vy, vz, fill_value):
    """One point of ``interpolators._interp3d_batch3``: one cell search and one
    set of weights, three gathers through the same ``_trilinear_cell``.

    Returns the three components as a tuple - an ``out`` array would be a heap
    allocation (or an aliasing memory write) per field evaluation per particle
    per step, which is most of the kernel's cost.
    """
    nx, ny, nz = len(grid_x), len(grid_y), len(grid_z)
    if (x < grid_x[0] or x > grid_x[nx - 1] or
            y < grid_y[0] or y > grid_y[ny - 1] or
            z < grid_z[0] or z > grid_z[nz - 1]):
        return fill_value, fill_value, fill_value
    i = _searchsorted_numba(grid_x, x)
    j = _searchsorted_numba(grid_y, y)
    k = _searchsorted_numba(grid_z, z)
    i = max(0, min(i, nx - 2))
    j = max(0, min(j, ny - 2))
    k = max(0, min(k, nz - 2))
    tx = (x - grid_x[i]) / (grid_x[i + 1] - grid_x[i])
    ty = (y - grid_y[j]) / (grid_y[j + 1] - grid_y[j])
    tz = (z - grid_z[k]) / (grid_z[k + 1] - grid_z[k])
    tx = max(0.0, min(tx, 1.0))
    ty = max(0.0, min(ty, 1.0))
    tz = max(0.0, min(tz, 1.0))
    return (_trilinear_cell(vx, i, j, k, tx, ty, tz),
            _trilinear_cell(vy, i, j, k, tx, ty, tz),
            _trilinear_cell(vz, i, j, k, tx, ty, tz))


@njit(cache=True, fastmath=True, inline='always')
def _interp3_point_h(x, y, z, grid_x, grid_y, grid_z, vx, vy, vz, fill_value, hints, p, off):
    """``_interp3_point`` with a per-particle cell hint.

    ``interpolators._searchsorted_with_hint`` returns exactly the index the
    plain binary search returns on a strictly increasing grid (``_pack_field``
    checks that), so this is bit-identical to ``_interp3_point`` - it only skips
    the search when the particle is still in, or next to, the cell it was in
    last step, which at 2000 steps per turn it almost always is.
    """
    nx, ny, nz = len(grid_x), len(grid_y), len(grid_z)
    if (x < grid_x[0] or x > grid_x[nx - 1] or
            y < grid_y[0] or y > grid_y[ny - 1] or
            z < grid_z[0] or z > grid_z[nz - 1]):
        return fill_value, fill_value, fill_value
    i = _searchsorted_with_hint(grid_x, x, hints[p, off])
    j = _searchsorted_with_hint(grid_y, y, hints[p, off + 1])
    k = _searchsorted_with_hint(grid_z, z, hints[p, off + 2])
    i = max(0, min(i, nx - 2))
    j = max(0, min(j, ny - 2))
    k = max(0, min(k, nz - 2))
    hints[p, off] = i
    hints[p, off + 1] = j
    hints[p, off + 2] = k
    tx = (x - grid_x[i]) / (grid_x[i + 1] - grid_x[i])
    ty = (y - grid_y[j]) / (grid_y[j + 1] - grid_y[j])
    tz = (z - grid_z[k]) / (grid_z[k + 1] - grid_z[k])
    tx = max(0.0, min(tx, 1.0))
    ty = max(0.0, min(ty, 1.0))
    tz = max(0.0, min(tz, 1.0))
    return (_trilinear_cell(vx, i, j, k, tx, ty, tz),
            _trilinear_cell(vy, i, j, k, tx, ty, tz),
            _trilinear_cell(vz, i, j, k, tx, ty, tz))


@njit(cache=True, fastmath=True, inline='always')
def _eval_field(f, x, y, z, hints, p, off):
    """A packed field at one point, with ``Field._get_field_*``'s scaling."""
    mode = f[0]
    if mode == 0:
        return 0.0, 0.0, 0.0
    s = f[7]
    if mode == 1:
        c = f[9]
        return s * c[0], s * c[1], s * c[2]
    ex, ey, ez = _interp3_point_h(x, y, z, f[1], f[2], f[3], f[4], f[5], f[6], f[8],
                                  hints, p, off)
    return s * ex, s * ey, s * ez


@njit(cache=True)
def _eval_field_batch(f, xa, ya, za, m, out):
    """A packed field at ``m`` points through the SAME ``_interp3d_batch3`` the
    Python path uses (``Field._get_field_3d`` via ``interp3d_components``).

    ``interp3d_components`` refuses fewer than two points, so a 1-point query
    takes the per-component path - which is what ``_interp3_point`` reproduces
    exactly - and that is what is done here too.
    """
    s = f[7]
    if f[0] == 0:
        for i in range(m):
            out[i, 0] = 0.0
            out[i, 1] = 0.0
            out[i, 2] = 0.0
        return
    if f[0] == 1:
        c = f[9]
        for i in range(m):
            out[i, 0] = s * c[0]
            out[i, 1] = s * c[1]
            out[i, 2] = s * c[2]
        return
    if m < 2:
        for i in range(m):
            tx, ty, tz = _interp3_point(xa[i], ya[i], za[i], f[1], f[2], f[3],
                                        f[4], f[5], f[6], f[8])
            out[i, 0] = s * tx
            out[i, 1] = s * ty
            out[i, 2] = s * tz
        return
    res = _interp3d_batch3(xa[:m], ya[:m], za[:m], f[1], f[2], f[3], f[4], f[5], f[6], f[8])
    for i in range(m):
        out[i, 0] = s * res[i, 0]
        out[i, 1] = s * res[i, 1]
        out[i, 2] = s * res[i, 2]


@njit(cache=True, fastmath=True)
def _boris_point(rhx, rhy, rhz, vx, vy, vz, ex, ey, ez, bx, by, bz, dt, q_over_m):
    """One particle of ``pusher.boris_rel_dkd_batch`` (drift-kick-drift, u = gamma v).
    Returns (r_new, v_new) as six components."""
    c2 = CLIGHT * CLIGHT
    b2 = (vx * vx + vy * vy + vz * vz) / c2
    if b2 >= 0.9999:
        b2 = 0.9999
    g = 1.0 / np.sqrt(1.0 - b2)
    he = 0.5 * q_over_m * dt
    ux = g * vx + he * ex
    uy = g * vy + he * ey
    uz = g * vz + he * ez
    gm = np.sqrt(1.0 + (ux * ux + uy * uy + uz * uz) / c2)
    f = he / gm
    tx, ty, tz = f * bx, f * by, f * bz
    t2 = tx * tx + ty * ty + tz * tz
    sf = 2.0 / (1.0 + t2)
    sx, sy, sz = sf * tx, sf * ty, sf * tz
    px = ux + (uy * tz - uz * ty)
    py = uy + (uz * tx - ux * tz)
    pz = uz + (ux * ty - uy * tx)
    ux = ux + (py * sz - pz * sy)
    uy = uy + (pz * sx - px * sz)
    uz = uz + (px * sy - py * sx)
    ux += he * ex
    uy += he * ey
    uz += he * ez
    gn = np.sqrt(1.0 + (ux * ux + uy * uy + uz * uz) / c2)
    vnx, vny, vnz = ux / gn, uy / gn, uz / gn
    hd = 0.5 * dt
    return rhx + hd * vnx, rhy + hd * vny, rhz + hd * vnz, vnx, vny, vnz


# NO fastmath here: ``tracking.SpaceChargeKick._kick`` is NumPy, which never
# reassociates. Same operations, same order.
@njit(cache=True)
def _sc_kick_point(vx, vy, vz, ex, ey, ez, bx, by, bz, any_b, qom_dt):
    """One particle of ``SpaceChargeKick._kick``: u += (q/m)(E + v x B) dt."""
    c2 = CLIGHT ** 2
    gamma = 1.0 / np.sqrt(1.0 - ((vx * vx + vy * vy) + vz * vz) / c2)
    ux = gamma * vx
    uy = gamma * vy
    uz = gamma * vz
    fx = ex
    fy = ey
    fz = ez
    if any_b:
        fx = ex + (vy * bz - vz * by)
        fy = ey + (vz * bx - vx * bz)
        fz = ez + (vx * by - vy * bx)
    ux = ux + qom_dt * fx
    uy = uy + qom_dt * fy
    uz = uz + qom_dt * fz
    gn = np.sqrt(1.0 + ((ux * ux + uy * uy) + uz * uz) / c2)
    return ux / gn, uy / gn, uz / gn


@njit(cache=True, inline='always')
def _in_raster(ra, x, y, z):
    """``Raster3D.__call__`` / the deck's stacked-metal closure: nearest node,
    off-grid = vacuum."""
    if ra[0] == 0:
        return False
    g = ra[1]
    i = int(np.rint((x - ra[2]) / ra[5]))
    j = int(np.rint((y - ra[3]) / ra[6]))
    k = int(np.rint((z - ra[4]) / ra[7]))
    if i < 0 or i >= g.shape[0] or j < 0 or j >= g.shape[1] or k < 0 or k >= g.shape[2]:
        return False
    return g[i, j, k] != 0


@njit(cache=True, inline='always')
def _smoothstep(rr, r0, r1):
    """``RadialBlendField.weights`` / ``RadialTaperField.factor``'s smoothstep."""
    t = (rr - r0) / (r1 - r0)
    t = max(t, 0.0)
    t = min(t, 1.0)
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, inline='always')
def _taper_factor(tmode, rr, taper_s, taper_r0, taper_r1):
    if tmode == 0:
        return 1.0
    if tmode == 1:
        return taper_s
    return taper_s + (1.0 - taper_s) * _smoothstep(rr, taper_r0, taper_r1)


# ============================================================================
# the chunk kernel
# ============================================================================
def _chunk3d_impl(r, v, active, birth_step, step0, n_sub, skip_ab_first, stop_after_a_last,
                  t_start, dt, q_over_m, use_batch,
                  bfld, e_in, e_out, e_st, blend_on, blend_r0, blend_r1,
                  taper_on, taper_s, taper_r0, taper_r1, omega, phase,
                  sc_on, sc_E, sc_B, sc_kick_mask, sc_has_field, sc_qom, sc_kick_z,
                  r_max, z_max, metal_on, metal_every, exempt_ref, ref_particle, ras_a, ras_b,
                  hints, r_hist, v_hist, active_hist, t_hist, ref_hit_hist, kick_hist, hit_p, hit_s):
    """Advance ``r``, ``v``, ``active`` (all full-length, mutated in place) over
    steps ``step0 .. step0 + n_sub - 1``.

    ``skip_ab_first``   the first step's push/release AND space-charge kick were
                        already done (the driver ran the Python solve): start at
                        the terminators.
    ``stop_after_a_last`` the last step stops right after push/release, so the
                        driver can run the Python solve for it.

    Returns (n_c, all_lost, n_kick_steps, n_hits, t_out); ``n_c`` steps completed
    the terminator/record phase and occupy ``r_hist[:n_c]``, i.e. steps
    ``step0 .. step0 + n_c - 1``.
    """
    n = r.shape[0]
    idx = np.empty(n, dtype=np.int64)
    sel = np.empty(n, dtype=np.int64)
    sub_i = np.empty(n, dtype=np.int64)
    sub_o = np.empty(n, dtype=np.int64)
    r_half = np.empty((n, 3))
    wbuf = np.empty(n)
    rrbuf = np.empty(n)
    xa = np.empty(n)
    ya = np.empty(n)
    za = np.empty(n)
    e_acc = np.empty((n, 3))
    e_tmp = np.empty((n, 3))
    b_acc = np.empty((n, 3))
    v_act = np.empty((n, 3))
    dt_arr = np.empty(n)
    hitmask = np.zeros(n, dtype=np.bool_)
    hdt = 0.5 * dt
    t = t_start
    n_c = 0
    n_kick = 0
    n_hits = 0
    all_lost = 0

    for m in range(n_sub):
        step = step0 + m
        do_ab = not (m == 0 and skip_ab_first)
        do_c = not (m == n_sub - 1 and stop_after_a_last)
        kicked = False

        if do_ab:
            # ---------------- 1 + 2: RF time at the midpoint, one push -------
            cos_rf = np.cos(omega * (t + hdt) + phase)
            na = 0
            for p in range(n):
                if active[p]:
                    idx[na] = p
                    na += 1
            # ONE parallel region: r_half, and the RadialBlendField weight
            for a in prange(na):
                p = idx[a]
                r_half[a, 0] = r[p, 0] + hdt * v[p, 0]
                r_half[a, 1] = r[p, 1] + hdt * v[p, 1]
                r_half[a, 2] = r[p, 2] + hdt * v[p, 2]
                xa[a] = r_half[a, 0]
                ya[a] = r_half[a, 1]
                za[a] = r_half[a, 2]
                if blend_on:
                    wbuf[a] = _smoothstep(np.hypot(r_half[a, 0], r_half[a, 1]),
                                          blend_r0, blend_r1)

            # RadialBlendField: the batch decides which parts are evaluated
            branch = 4
            if blend_on:
                wmin = 2.0
                wmax = -1.0
                for a in range(na):
                    w = wbuf[a]
                    if w < wmin:
                        wmin = w
                    if w > wmax:
                        wmax = w
                if na == 0:
                    branch = 0
                elif wmin >= 1.0:
                    branch = 2            # outer only, whole batch
                elif wmax <= 0.0:
                    branch = 1            # inner only, whole batch
                else:
                    branch = 3            # per-particle blend

            # the inner / outer point sets, exactly as RadialBlendField slices them
            n_i = 0
            n_o = 0
            if branch == 1 or branch == 4:
                for a in range(na):
                    sub_i[n_i] = a
                    n_i += 1
            elif branch == 2:
                for a in range(na):
                    sub_o[n_o] = a
                    n_o += 1
            elif branch == 3:
                for a in range(na):
                    if wbuf[a] < 1.0:
                        sub_i[n_i] = a
                        n_i += 1
                    if wbuf[a] > 0.0:
                        sub_o[n_o] = a
                        n_o += 1

            # RadialTaperField wraps the OUTER field and makes the same kind of
            # batch decision on the subset the blend hands it.
            tmode = 0
            if taper_on and n_o > 0:
                rsmin = 1.0e300
                rsmax = -1.0e300
                for a in range(n_o):
                    rr = np.hypot(r_half[sub_o[a], 0], r_half[sub_o[a], 1])
                    rrbuf[sub_o[a]] = rr
                    if rr < rsmin:
                        rsmin = rr
                    if rr > rsmax:
                        rsmax = rr
                if rsmax <= taper_r0:
                    tmode = 1             # x inner_scale, whole subset
                elif rsmin >= taper_r1:
                    tmode = 0             # bare field
                else:
                    tmode = 2             # per-particle factor

            if use_batch:
                # --- E: the blend, through the Python path's own batch kernel
                for a in range(na):
                    e_acc[a, 0] = 0.0
                    e_acc[a, 1] = 0.0
                    e_acc[a, 2] = 0.0
                if n_i > 0:
                    for a in range(n_i):
                        xa[a] = r_half[sub_i[a], 0]
                        ya[a] = r_half[sub_i[a], 1]
                        za[a] = r_half[sub_i[a], 2]
                    _eval_field_batch(e_in, xa, ya, za, n_i, e_tmp)
                    if branch == 3:
                        for a in range(n_i):
                            w = 1.0 - wbuf[sub_i[a]]
                            e_acc[sub_i[a], 0] += w * e_tmp[a, 0]
                            e_acc[sub_i[a], 1] += w * e_tmp[a, 1]
                            e_acc[sub_i[a], 2] += w * e_tmp[a, 2]
                    else:
                        for a in range(n_i):
                            e_acc[sub_i[a], 0] = e_tmp[a, 0]
                            e_acc[sub_i[a], 1] = e_tmp[a, 1]
                            e_acc[sub_i[a], 2] = e_tmp[a, 2]
                if n_o > 0:
                    for a in range(n_o):
                        xa[a] = r_half[sub_o[a], 0]
                        ya[a] = r_half[sub_o[a], 1]
                        za[a] = r_half[sub_o[a], 2]
                    _eval_field_batch(e_out, xa, ya, za, n_o, e_tmp)
                    if branch == 3:
                        for a in range(n_o):
                            g = sub_o[a]
                            fac = _taper_factor(tmode, rrbuf[g], taper_s, taper_r0, taper_r1)
                            w = wbuf[g]
                            e_acc[g, 0] += w * (e_tmp[a, 0] * fac)
                            e_acc[g, 1] += w * (e_tmp[a, 1] * fac)
                            e_acc[g, 2] += w * (e_tmp[a, 2] * fac)
                    else:
                        for a in range(n_o):
                            g = sub_o[a]
                            fac = _taper_factor(tmode, rrbuf[g], taper_s, taper_r0, taper_r1)
                            e_acc[g, 0] = e_tmp[a, 0] * fac
                            e_acc[g, 1] = e_tmp[a, 1] * fac
                            e_acc[g, 2] = e_tmp[a, 2] * fac
                for a in range(na):
                    xa[a] = r_half[a, 0]
                    ya[a] = r_half[a, 1]
                    za[a] = r_half[a, 2]
                for a in range(na):
                    e_acc[a, 0] = e_acc[a, 0] * cos_rf
                    e_acc[a, 1] = e_acc[a, 1] * cos_rf
                    e_acc[a, 2] = e_acc[a, 2] * cos_rf
                if e_st[0] != 0:
                    _eval_field_batch(e_st, xa, ya, za, na, e_tmp)
                    for a in range(na):
                        e_acc[a, 0] = e_acc[a, 0] + e_tmp[a, 0]
                        e_acc[a, 1] = e_acc[a, 1] + e_tmp[a, 1]
                        e_acc[a, 2] = e_acc[a, 2] + e_tmp[a, 2]
                _eval_field_batch(bfld, xa, ya, za, na, b_acc)
                # and the push through the Python path's OWN batch kernel: its
                # parfor contracts a little differently from any scalar spelling
                # of the same source (about one row in 1e5 differs by an ulp), so
                # calling it is the only way to be bit-identical.
                for a in range(na):
                    p = idx[a]
                    v_act[a, 0] = v[p, 0]
                    v_act[a, 1] = v[p, 1]
                    v_act[a, 2] = v[p, 2]
                    dt_arr[a] = dt
                r_new, v_new = boris_rel_dkd_batch(r_half[:na], v_act[:na], e_acc[:na],
                                                   b_acc[:na], dt_arr[:na], q_over_m)
                for a in range(na):
                    p = idx[a]
                    r[p, 0] = r_new[a, 0]
                    r[p, 1] = r_new[a, 1]
                    r[p, 2] = r_new[a, 2]
                    v[p, 0] = v_new[a, 0]
                    v[p, 1] = v_new[a, 1]
                    v[p, 2] = v_new[a, 2]
            else:
                for a in prange(na):
                    p = idx[a]
                    x = r_half[a, 0]
                    y = r_half[a, 1]
                    z = r_half[a, 2]
                    ex = 0.0
                    ey = 0.0
                    ez = 0.0
                    if branch == 1 or branch == 4:
                        ex, ey, ez = _eval_field(e_in, x, y, z, hints, p, 0)
                    elif branch == 2:
                        gx0, gy0, gz0 = _eval_field(e_out, x, y, z, hints, p, 3)
                        fac = _taper_factor(tmode, rrbuf[a], taper_s, taper_r0, taper_r1)
                        ex = gx0 * fac
                        ey = gy0 * fac
                        ez = gz0 * fac
                    elif branch == 3:
                        w = wbuf[a]
                        if w < 1.0:
                            gx0, gy0, gz0 = _eval_field(e_in, x, y, z, hints, p, 0)
                            ex = ex + (1.0 - w) * gx0
                            ey = ey + (1.0 - w) * gy0
                            ez = ez + (1.0 - w) * gz0
                        if w > 0.0:
                            gx0, gy0, gz0 = _eval_field(e_out, x, y, z, hints, p, 3)
                            fac = _taper_factor(tmode, rrbuf[a], taper_s, taper_r0, taper_r1)
                            ex = ex + w * (gx0 * fac)
                            ey = ey + w * (gy0 * fac)
                            ez = ez + w * (gz0 * fac)
                    # TimedField / StaticPlusRFField: pattern x cos, then the statics
                    ex = ex * cos_rf
                    ey = ey * cos_rf
                    ez = ez * cos_rf
                    if e_st[0] != 0:
                        sx0, sy0, sz0 = _eval_field(e_st, x, y, z, hints, p, 6)
                        ex = ex + sx0
                        ey = ey + sy0
                        ez = ez + sz0
                    bx0, by0, bz0 = _eval_field(bfld, x, y, z, hints, p, 9)
                    nrx, nry, nrz, nvx, nvy, nvz = _boris_point(
                        x, y, z, v[p, 0], v[p, 1], v[p, 2], ex, ey, ez,
                        bx0, by0, bz0, dt, q_over_m)
                    r[p, 0] = nrx
                    r[p, 1] = nry
                    r[p, 2] = nrz
                    v[p, 0] = nvx
                    v[p, 1] = nvy
                    v[p, 2] = nvz

            # ---------------- 3a: TimedRelease -------------------------------
            for p in range(n):
                if birth_step[p] == step + 1:
                    active[p] = True

            # ---------------- 3b: SpaceChargeKick, stored field --------------
            # ``stop_after_a_last`` splits the step between the release and this
            # kick: the driver runs the Python solve for it and the kick comes
            # from SpaceChargeKick.apply, so the kernel must NOT kick here too.
            if sc_on and do_c:
                nsel = 0
                any_b = False
                for p in range(n):
                    if active[p] and sc_kick_mask[p] and sc_has_field[p]:
                        ex = sc_E[p, 0]
                        ey = sc_E[p, 1]
                        ez = sc_E[p, 2] if sc_kick_z else 0.0
                        bx = sc_B[p, 0]
                        by = sc_B[p, 1]
                        bz = sc_B[p, 2]
                        if (ex != 0.0 or ey != 0.0 or ez != 0.0
                                or bx != 0.0 or by != 0.0 or bz != 0.0):
                            sel[nsel] = p
                            nsel += 1
                            if bx != 0.0 or by != 0.0 or bz != 0.0:
                                any_b = True
                if nsel > 0:
                    qom_dt = sc_qom * dt
                    for a in range(nsel):
                        p = sel[a]
                        ez2 = sc_E[p, 2] if sc_kick_z else 0.0
                        nvx, nvy, nvz = _sc_kick_point(
                            v[p, 0], v[p, 1], v[p, 2], sc_E[p, 0], sc_E[p, 1], ez2,
                            sc_B[p, 0], sc_B[p, 1], sc_B[p, 2], any_b, qom_dt)
                        v[p, 0] = nvx
                        v[p, 1] = nvy
                        v[p, 2] = nvz
                    n_kick += 1
                    kicked = True

        if not do_c:
            break

        # ---------------- 4: terminators -------------------------------------
        # ONE parallel region: the radial / vertical loss (which for an already
        # inactive row is a no-op, so restricting it to the active ones is the
        # same mask) and the metal test that follows it.
        do_metal = metal_on and (step % metal_every) == 0
        for p in prange(n):
            hitmask[p] = False
            if not active[p]:
                continue
            rad = np.sqrt(r[p, 0] * r[p, 0] + r[p, 1] * r[p, 1])
            if rad > r_max or abs(r[p, 2]) > z_max:
                active[p] = False
                continue
            if do_metal:
                hitmask[p] = (_in_raster(ras_a, r[p, 0], r[p, 1], r[p, 2])
                              or _in_raster(ras_b, r[p, 0], r[p, 1], r[p, 2]))
        ref_hit = False
        if do_metal:
            for p in range(n):
                if not hitmask[p]:
                    continue
                if exempt_ref and p == ref_particle:
                    ref_hit = True
                    continue
                active[p] = False
                if n_hits < hit_p.shape[0]:
                    hit_p[n_hits] = p
                    hit_s[n_hits] = step
                    n_hits += 1

        # ---------------- 5: clock, record ------------------------------------
        t += dt
        for p in range(n):
            r_hist[m, p, 0] = r[p, 0]
            r_hist[m, p, 1] = r[p, 1]
            r_hist[m, p, 2] = r[p, 2]
            v_hist[m, p, 0] = v[p, 0]
            v_hist[m, p, 1] = v[p, 1]
            v_hist[m, p, 2] = v[p, 2]
            active_hist[m, p] = active[p]
        t_hist[m] = t
        ref_hit_hist[m] = ref_hit
        kick_hist[m] = kicked
        n_c = m + 1

        n_alive = 0
        for p in range(n):
            if active[p]:
                n_alive += 1
        if n_alive == 0:
            all_lost = 1
            break

    return n_c, all_lost, n_kick, n_hits, t


# Two compilations of the SAME body. ``prange`` degrades to ``range`` without
# parallel=True, so the serial one is a plain loop: with a few hundred particles
# the thread fork/join of four parallel regions per step (tens of microseconds,
# 32 threads on this box) costs more than the work itself, and the whole point of
# the kernel is to get rid of exactly that overhead. PAR_MIN_PARTICLES is where
# the parallel one starts to win (measured, see FAST3D_KERNEL_REPORT.md).
_chunk3d_serial = njit(cache=True)(_chunk3d_impl)
# cache=False on purpose. With cache=True numba 0.64 hands back a SERIALIZED
# build of a parallel=True function on every load after the first: a fresh
# compile scales with the thread count, the cached one runs single-threaded and
# is indistinguishable from _chunk3d_serial (measured, 2026-09-16). The parallel
# variant is only chosen for large bunches, whose runs dwarf its ~10 s compile.
_chunk3d_par = njit(cache=False, parallel=True)(_chunk3d_impl)
# Below this the thread fork/join of two parallel regions per step (~25 us each
# on the laptop) costs more than the work: measured best-of-3 on the smoke model,
# 120 particles serial 4.7x vs parallel 3.6x, 480 particles serial 3.2x vs
# parallel 5.1x.
PAR_MIN_PARTICLES = 256
# The inner loop is trilinear interpolation on 300-500 MB of field map, i.e.
# DRAM latency bound: it stops scaling around 8 threads and is slower again at
# 32 (4.1x vs 3.4x at 4000 particles). finder.fast3d_threads overrides; None
# leaves numba's thread count alone.
DEFAULT_PAR_THREADS = 8


# ============================================================================
# self-check: how closely the leaves reproduce the batch kernels
# ============================================================================
_SELFTEST = None


@njit(cache=True)
def _hypot_point(x, y):
    return np.hypot(x, y)


def selftest(n: int = 4096, seed: int = 20260916) -> dict:
    """Compare the per-point leaves with the batch kernels they stand in for.

    Neither ``_interp3_point`` nor ``_boris_point`` can be exact against the
    ``parallel=True`` kernels they stand in for: a parfor contracts differently
    from every scalar spelling of the same source (``_interp3d_batch3`` also
    disagrees with ``_interp3d_single`` and with the per-component
    ``_interp3d_batch``, and ``boris_rel_dkd_batch`` differs from its own scalar
    body on about one input in 1e5). The residuals are measured here; ``interp
    ='batch'`` calls both batch kernels instead and is bit-identical.

    What MUST be exact, and is checked: the cell-hint search returning the same
    index as the plain binary search, the per-point interpolation matching
    ``_interp3d_single``, and numba's ``np.hypot`` matching NumPy's.
    """
    from PyPATools.field_src.interpolators import _interp3d_batch3 as _b3, _interp3d_single
    from PyPATools.pusher import boris_rel_dkd_batch

    rng = np.random.default_rng(seed)
    gx = np.linspace(-0.2, 0.2, 41)
    gy = np.linspace(-0.2, 0.2, 41)
    gz = np.linspace(-0.03, 0.03, 13)
    vx = rng.normal(size=(41, 41, 13))
    vy = rng.normal(size=(41, 41, 13))
    vz = rng.normal(size=(41, 41, 13))
    pts = rng.uniform(-0.25, 0.25, size=(n, 3))
    pts[:, 2] = rng.uniform(-0.04, 0.04, size=n)
    x = np.ascontiguousarray(pts[:, 0])
    y = np.ascontiguousarray(pts[:, 1])
    z = np.ascontiguousarray(pts[:, 2])
    ref = _b3(x, y, z, gx, gy, gz, vx, vy, vz, 0.0)
    mine = np.empty_like(ref)
    single = np.empty_like(ref)
    hinted = np.empty_like(ref)
    hints = np.zeros((n, 12), dtype=np.int64)
    for i in range(n):
        mine[i] = _interp3_point(x[i], y[i], z[i], gx, gy, gz, vx, vy, vz, 0.0)
        hinted[i] = _interp3_point_h(x[i], y[i], z[i], gx, gy, gz, vx, vy, vz, 0.0,
                                     hints, i % 2, 0)
        single[i, 0] = _interp3d_single(x[i], y[i], z[i], gx, gy, gz, vx, 0.0)
        single[i, 1] = _interp3d_single(x[i], y[i], z[i], gx, gy, gz, vy, 0.0)
        single[i, 2] = _interp3d_single(x[i], y[i], z[i], gx, gy, gz, vz, 0.0)
    scale = np.maximum(np.abs(ref), 1e-300)
    interp_ok = bool(np.array_equal(ref, mine))
    interp_rel = float(np.max(np.abs(ref - mine) / scale))
    single_ok = bool(np.array_equal(mine, single))
    hint_ok = bool(np.array_equal(mine, hinted))
    b3_vs_single = float(np.max(np.abs(ref - single) / scale))

    rh = rng.uniform(-0.2, 0.2, size=(n, 3))
    vv = rng.normal(scale=1.0e7, size=(n, 3))
    ee = rng.normal(scale=1.0e6, size=(n, 3))
    bb = rng.normal(scale=1.0, size=(n, 3))
    dts = np.full(n, 6.0e-11)
    qom = 47875698.77970449
    r_ref, v_ref = boris_rel_dkd_batch(rh, vv, ee, bb, dts, qom)
    r_mine = np.empty_like(r_ref)
    v_mine = np.empty_like(v_ref)
    for i in range(n):
        ob = _boris_point(rh[i, 0], rh[i, 1], rh[i, 2], vv[i, 0], vv[i, 1], vv[i, 2],
                          ee[i, 0], ee[i, 1], ee[i, 2], bb[i, 0], bb[i, 1], bb[i, 2],
                          dts[i], qom)
        r_mine[i] = ob[:3]
        v_mine[i] = ob[3:]
    boris_ok = bool(np.array_equal(r_ref, r_mine) and np.array_equal(v_ref, v_mine))
    boris_max = float(max(np.max(np.abs(r_ref - r_mine)), np.max(np.abs(v_ref - v_mine))))
    boris_rows = int(np.sum((np.abs(r_ref - r_mine).max(axis=1) > 0)
                            | (np.abs(v_ref - v_mine).max(axis=1) > 0)))

    # np.hypot: the blend / taper radius. NumPy's ufunc and numba's scalar
    # implementation need not be the same function.
    hx = rng.uniform(-0.6, 0.6, size=n)
    hy = rng.uniform(-0.6, 0.6, size=n)
    h_ref = np.hypot(hx, hy)
    h_mine = np.array([_hypot_point(hx[i], hy[i]) for i in range(n)])
    hypot_ok = bool(np.array_equal(h_ref, h_mine))

    return {'n_samples': int(n),
            'boris_point_exact_vs_batch': boris_ok, 'boris_point_max_abs': boris_max,
            'boris_point_rows_differing': boris_rows,
            'interp_point_exact_vs_batch3': interp_ok,
            'interp_point_max_rel_vs_batch3': interp_rel,
            'interp_point_exact_vs_single': single_ok,
            'batch3_max_rel_vs_single': b3_vs_single,
            'hinted_exact_vs_plain': hint_ok, 'hypot_exact': hypot_ok}


def _require_selftest():
    global _SELFTEST
    if _SELFTEST is None:
        _SELFTEST = selftest()
    s = _SELFTEST
    # boris_point / interp3_point are ALLOWED to differ from the batch kernels in
    # the last ulp (see the module docstring); interp='batch' calls those kernels
    # instead. What must hold exactly is the cell-hint equivalence and hypot.
    if not s['interp_point_exact_vs_single']:
        raise FastPathUnavailable(
            "the per-point interpolation does not reproduce _interp3d_single exactly")
    if not s['hinted_exact_vs_plain']:
        raise FastPathUnavailable(
            "the cell-hint interpolation does not match the plain binary search")
    if not s['hypot_exact']:
        raise FastPathUnavailable("numba's np.hypot differs from NumPy's (blend radius)")
    return s


# ============================================================================
# packing the finder's configuration
# ============================================================================
def _pack_field(f, what: str):
    """A gridded / constant / absent ``Field`` as the kernel's packed tuple."""
    if f is None:
        return _ZERO_FIELD
    if not isinstance(f, Field):
        raise FastPathUnavailable(f"{what}: {type(f).__name__} is not a plain Field")
    if getattr(f, 'label', '') == 'Zero Field':
        return _ZERO_FIELD
    scale = float(getattr(f, '_scaling', 1.0))
    if f.dim == 0:
        c = np.ascontiguousarray([float(f._field[k]) for k in 'xyz'], dtype=np.float64)
        return (1, _EMPTY1, _EMPTY1, _EMPTY1, _EMPTY3, _EMPTY3, _EMPTY3, scale, 0.0, c)
    if f.dim != 3:
        raise FastPathUnavailable(f"{what}: dim {f.dim} field (only 3D grids and constants)")
    ips = [f._field[k] for k in 'xyz']
    for ip in ips:
        if not (hasattr(ip, '_grid') and hasattr(ip, '_values')):
            raise FastPathUnavailable(
                f"{what}: interpolator {type(ip).__name__} has no raw grid (use the numba backend)")
        if getattr(ip, 'method', 'linear') != 'linear':
            raise FastPathUnavailable(f"{what}: {ip.method} interpolation")
        if bool(getattr(ip, 'bounds_error', False)):
            raise FastPathUnavailable(f"{what}: bounds_error interpolator")
        if not np.isfinite(float(getattr(ip, 'fill_value', 0.0))):
            raise FastPathUnavailable(f"{what}: fill value {ip.fill_value} (only finite)")
    # interp3d_components' own grid-compatibility test, so the Python path is on
    # the fused kernel too (that is what this module reproduces)
    g0 = ips[0]._grid
    fill = float(ips[0].fill_value)
    for ip in ips[1:]:
        for d in range(3):
            a, b = g0[d], ip._grid[d]
            if a is not b and (a.shape != b.shape or a[0] != b[0] or a[-1] != b[-1]):
                raise FastPathUnavailable(f"{what}: components on different grids")
        if float(ip.fill_value) != fill:
            raise FastPathUnavailable(f"{what}: components with different fill values")
    gx, gy, gz = (np.ascontiguousarray(g, dtype=np.float64) for g in g0)
    vx, vy, vz = (np.ascontiguousarray(ip._values, dtype=np.float64) for ip in ips)
    if vx.shape != (len(gx), len(gy), len(gz)):
        raise FastPathUnavailable(f"{what}: values {vx.shape} do not match the grid")
    # the cell-hint search is only equivalent to the plain binary search on a
    # STRICTLY increasing axis (a repeated node makes the two pick different
    # cells, and a zero-width cell divides by zero in the weights anyway)
    for nm, g in zip('xyz', (gx, gy, gz)):
        if len(g) < 2 or not np.all(np.diff(g) > 0.0):
            raise FastPathUnavailable(f"{what}: {nm} axis is not strictly increasing")
    return (2, gx, gy, gz, vx, vy, vz, scale, fill, _ZERO3)


def _pack_efield(ef):
    """The design's E-field stack as inner / outer / static parts plus the blend,
    taper and modulation parameters.

    Understood: ``StaticPlusRFField`` / ``TimedField`` wrapping either a plain
    gridded ``Field`` or a ``RadialBlendField`` (whose outer part may be wrapped
    in a ``RadialTaperField``), plus at most one static gridded ``Field``.
    """
    from .fields3d import RadialBlendField, RadialTaperField
    from .inflector import StaticPlusRFField

    static = None
    pattern = None
    omega = 0.0
    phase = 0.0
    if ef is None:
        pass
    elif isinstance(ef, StaticPlusRFField):
        if len(ef.statics) > 1:
            raise FastPathUnavailable("more than one static E-field")
        static = ef.statics[0] if ef.statics else None
        if ef.rf is not None:
            pattern, omega, phase = ef.static, float(ef.omega), float(ef.phase)
    elif isinstance(ef, TimedField):
        pattern, omega, phase = ef.static, float(ef.omega), float(ef.phase)
    elif isinstance(ef, Field):
        static = ef
    else:
        raise FastPathUnavailable(f"E-field of type {type(ef).__name__}")

    blend_on = False
    r0 = r1 = 0.0
    taper_on = False
    taper_s, tr0, tr1 = 1.0, 0.0, 1.0
    inner = pattern
    outer = None
    if isinstance(pattern, RadialBlendField):
        blend_on = True
        inner, outer = pattern.inner, pattern.outer
        r0, r1 = float(pattern.r0), float(pattern.r1)
        if isinstance(outer, RadialTaperField):
            taper_on = True
            taper_s = float(outer.inner_scale)
            tr0, tr1 = float(outer.r0), float(outer.r1)
            outer = outer.field
        if isinstance(inner, RadialTaperField):
            raise FastPathUnavailable("RadialTaperField on the inner (BEM) part of the blend")
    elif pattern is not None and not isinstance(pattern, Field):
        raise FastPathUnavailable(f"RF pattern of type {type(pattern).__name__}")

    return dict(e_in=_pack_field(inner, 'RF pattern (inner)'),
                e_out=_pack_field(outer, 'RF pattern (outer)'),
                e_st=_pack_field(static, 'static E-field'),
                blend_on=bool(blend_on), blend_r0=r0, blend_r1=r1,
                taper_on=bool(taper_on), taper_s=taper_s, taper_r0=tr0, taper_r1=tr1,
                omega=omega, phase=phase)


def _raster_parts(mask):
    """The nearest-node boolean tests behind ``combine_obstacles``."""
    if mask is None:
        return []
    parts = list(getattr(mask, 'parts', None) or [mask])
    out = []
    for p in parts:
        grid = getattr(p, 'grid', None)
        if grid is None:
            raise FastPathUnavailable(f"obstacle test {type(p).__name__} is not a raster")
        grid = np.asarray(grid)
        if grid.ndim != 3:
            raise FastPathUnavailable("obstacle raster is not a 3D volume (midplane-only test)")
        if int(getattr(p, 'ndim', 2)) != 3:
            raise FastPathUnavailable("obstacle raster does not carry ndim = 3")
        if hasattr(p, 'origin') and hasattr(p, 'h'):          # fields3d.Raster3D
            origin = np.asarray(p.origin, dtype=float)
            h = np.asarray(p.h, dtype=float)
        elif hasattr(p, 'xs') and hasattr(p, 'zs'):           # the deck's stacked-metal closure
            xs = np.asarray(p.xs, dtype=float)
            zs = np.asarray(p.zs, dtype=float)
            ext = float(xs[-1])
            hxy = float(getattr(p, 'spacing', xs[1] - xs[0]))
            origin = np.array([-ext, -ext, float(zs[0])])
            h = np.array([hxy, hxy, float(zs[1] - zs[0])])
        else:
            raise FastPathUnavailable(
                f"obstacle raster {type(p).__name__} exposes no grid origin/spacing")
        out.append((1, np.ascontiguousarray(grid.astype(np.uint8)),
                    float(origin[0]), float(origin[1]), float(origin[2]),
                    float(h[0]), float(h[1]), float(h[2])))
    return out


def build_kernel_args3d(finder, pd_init, dt: float, n_steps: int, t0: float,
                        birth_step, release, n_ref: int, interp: str = 'point') -> dict:
    """Everything the kernel needs for this run, or ``FastPathUnavailable``."""
    if not HAS_NUMBA:
        raise FastPathUnavailable("numba not installed")
    if interp not in ('point', 'batch'):
        raise FastPathUnavailable(f"interp {interp!r} (use 'point' or 'batch')")
    _require_selftest()
    engine = finder.engine
    if engine.dim != '3D':
        raise FastPathUnavailable(f"{engine.dim} engine (fast_track.py covers 2D)")
    if str(getattr(engine, 'algorithm', '')).lower() != 'boris_rel':
        raise FastPathUnavailable(f"algorithm {engine.algorithm} (only boris_rel)")
    if engine.use_rf and engine.gap_model != 'bem2d':
        raise FastPathUnavailable(
            f"gap_model {engine.gap_model} (thin-gap kicks are not reproduced)")
    if engine.use_rf and engine.gap_model == 'bem2d':
        engine._sync_bem_field()

    # interactions: TimedRelease (optional) then SpaceChargeKick (optional), that order
    sc = None
    seen_release = False
    for ix in engine.extra_interactions:
        name = type(ix).__name__
        if name == 'TimedRelease':
            if seen_release or sc is not None:
                raise FastPathUnavailable("unexpected interaction order")
            seen_release = True
        elif name == 'SpaceChargeKick':
            if sc is not None:
                raise FastPathUnavailable("more than one SpaceChargeKick")
            sc = ix
        else:
            raise FastPathUnavailable(f"interaction {name}")
    if (release is not None) != seen_release:
        raise FastPathUnavailable("timed release hook and birth_step disagree")
    if sc is not None:
        if getattr(sc, 'charges', None) is None or len(sc.charges) != int(pd_init.numpart):
            raise FastPathUnavailable("SpaceChargeKick.prepare was not run for this beam")
        if not hasattr(sc.solver, 'gather'):
            raise FastPathUnavailable("space-charge solver without gather()")
        if getattr(sc, 'move', False):
            # HCHC60_SC_MOVE rotates the stored per-particle E_sc/B_sc with the
            # bunch's mean direction on EVERY step between solves, so the kernel's
            # "the solve's field is frozen until the next solve" no longer holds
            # and every step would have to be a chunk boundary. Reproducing the
            # rotation inside the kernel is a contained follow-up (the matrix is
            # one Rodrigues formula over the mean velocity of the solve's own
            # particle set); until then these runs take the general path.
            raise FastPathUnavailable(
                "SpaceChargeKick.move (HCHC60_SC_MOVE=1): the stored self-field is "
                "rotated every step, which the chunked kernel does not reproduce")

    # terminators: at most one MetalTerminator, no collimator
    metal = None
    for tm in engine.extra_terminators:
        name = type(tm).__name__
        if name == 'MetalTerminator':
            if metal is not None:
                raise FastPathUnavailable("more than one obstacle terminator")
            metal = tm
        else:
            raise FastPathUnavailable(f"terminator {name}")
    rasters = _raster_parts(metal.inside if metal is not None else None)
    if len(rasters) > 2:
        raise FastPathUnavailable(f"{len(rasters)} obstacle rasters (the kernel carries two)")

    design = engine.design
    e = _pack_efield(design.efield)
    b = _pack_field(design.bfield, 'B-field')

    n = int(pd_init.numpart)
    r0 = np.ascontiguousarray(np.array(pd_init.x_vec, dtype=float))
    v0 = np.ascontiguousarray(np.array(pd_init.v_vec, dtype=float))
    alive = np.ascontiguousarray(np.array(pd_init.alive, dtype=bool))
    if r0.shape != (n, 3) or v0.shape != (n, 3) or alive.shape != (n,):
        raise FastPathUnavailable("beam arrays are not (N, 3) / (N,)")
    bstep = np.ascontiguousarray(np.asarray(birth_step, dtype=np.int64).reshape(n))

    return dict(
        n=n, r0=r0, v0=v0, alive=alive, birth_step=bstep,
        dt=float(dt), n_steps=int(n_steps), t0=float(t0),
        q_over_m=float(engine.pusher.q_over_m), use_batch=(interp == 'batch'),
        bfld=b, e_in=e['e_in'], e_out=e['e_out'], e_st=e['e_st'],
        blend_on=e['blend_on'], blend_r0=e['blend_r0'], blend_r1=e['blend_r1'],
        taper_on=e['taper_on'], taper_s=e['taper_s'], taper_r0=e['taper_r0'],
        taper_r1=e['taper_r1'], omega=e['omega'], phase=e['phase'],
        r_max=float(engine.r_max), z_max=float(engine.z_max),
        metal=metal, metal_on=metal is not None,
        metal_every=int(getattr(metal, 'every', 1)) if metal is not None else 1,
        exempt_ref=bool(getattr(metal, 'exempt_ref', False)) if metal is not None else False,
        ref_particle=int(getattr(metal, 'ref_particle', 0)) if metal is not None else 0,
        index_offset=int(getattr(metal, 'index_offset', 0)) if metal is not None else 0,
        ras_a=rasters[0] if len(rasters) > 0 else _NO_RASTER,
        ras_b=rasters[1] if len(rasters) > 1 else _NO_RASTER,
        sc=sc, release=release, n_ref=int(n_ref), interp=interp,
        # None = decide from the particle count (PAR_MIN_PARTICLES); True/False forces it
        parallel=getattr(finder, 'fast3d_parallel', None),
        # None = leave numba's thread count alone. The kernel's inner loop is
        # trilinear interpolation on hundreds of MB of field map, i.e. DRAM
        # latency bound: on the laptop it peaks around 8 threads and is slower
        # again at 32 (see FAST3D_KERNEL_REPORT.md), so a box with many logical
        # cores wants this capped.
        threads=getattr(finder, 'fast3d_threads', DEFAULT_PAR_THREADS),
    )


# ============================================================================
# the driver: kernel chunks, the Python space-charge solve, the callback replay
# ============================================================================
def _next_sc_step(args, sc, active, from_step, n_steps):
    """The first step >= ``from_step`` at which ``SpaceChargeKick.apply`` has
    Python work to do (a solve, or a gather for particles released there)."""
    if sc is None or sc.noop:
        return n_steps
    every = int(sc.every)
    out = ((from_step + every - 1) // every) * every
    rel = args['_release_steps']
    k = int(np.searchsorted(rel, from_step))
    if k < len(rel):
        out = min(out, int(rel[k]))
    if sc.n_solves > 0 and np.any(active & ~sc.has_field & sc.deposit_mask):
        out = min(out, from_step)
    return min(out, n_steps)


def run_fast3d(finder, args: dict, callback, dt: float, t0: float):
    """Run the chunked kernel and replay every step through ``callback``.

    Returns a ``tracking.TrackingResult`` with the same contract as
    ``TrackingEngine.track_multiparticle``. ``finder.fast3d_threads`` caps
    numba's thread count for the kernel only and is restored afterwards.
    """
    n_threads = args.get('threads')
    want = before = None
    parallel = args.get('parallel')
    parallel = bool(args['n'] >= PAR_MIN_PARTICLES) if parallel is None else bool(parallel)
    if n_threads and parallel:
        import numba
        want = max(1, min(int(n_threads), int(numba.config.NUMBA_NUM_THREADS)))
        before = numba.get_num_threads()
        if want == before:
            want = before = None
    try:
        return _run_fast3d(finder, args, callback, dt, t0, want, before)
    finally:
        if before is not None:
            import numba
            numba.set_num_threads(before)


def _run_fast3d(finder, args: dict, callback, dt: float, t0: float,
                thr_kernel=None, thr_outer=None):
    from .tracking import TrackingResult
    numba_mod = None
    if thr_kernel is not None:
        import numba as numba_mod

    t_start = time.time()
    n = args['n']
    n_steps = args['n_steps']
    sc = args['sc']
    metal = args['metal']
    release = args['release']
    index_offset = args['index_offset']
    if metal is not None:
        metal.reset()

    bstep = args['birth_step']
    rel = np.unique(bstep[bstep > 0]) - 1
    args['_release_steps'] = np.ascontiguousarray(rel.astype(np.int64))

    r = args['r0'].copy()
    v = args['v0'].copy()
    active = args['alive'].copy()

    sc_on = sc is not None and not sc.noop
    if sc_on:
        sc_E, sc_B = sc.E, sc.B
        sc_kick_mask, sc_has_field = sc.kick_mask, sc.has_field
        sc_qom, sc_kick_z = float(sc.q_over_m), bool(sc.kick_z)
    else:
        sc_E = sc_B = np.zeros((n, 3))
        sc_kick_mask = sc_has_field = np.zeros(n, dtype=bool)
        sc_qom, sc_kick_z = 0.0, False

    parallel = args.get('parallel')
    parallel = bool(n >= PAR_MIN_PARTICLES) if parallel is None else bool(parallel)
    kernel = _chunk3d_par if parallel else _chunk3d_serial
    per_step = max(1, n * 57)
    max_chunk = max(1, min(MAX_CHUNK_STEPS, MAX_RECORD_BYTES // per_step))
    r_hist = v_hist = active_hist = t_hist = ref_hist = kick_hist = None
    cap = 0
    # per-particle interpolation cell hints, four fields x three axes, kept
    # ACROSS chunks (a particle stays in its cell for many steps)
    hints = np.zeros((n, 12), dtype=np.int64)

    step = 0
    t = float(t0)
    mid = False
    stopped = False
    all_lost = False
    steps_done = 0
    final = None
    n_chunks = 0
    n_kick_steps = 0
    t_kernel = 0.0
    t_replay = 0.0

    while step < n_steps:
        if mid:
            sc.apply(step, r, v, r, v, active, t, dt)
            mid = False
            skip_ab = True
        else:
            skip_ab = False

        # The first step the kernel will START (phase A of `step`, unless its A
        # and B were already done) is the earliest one that can still need the
        # Python solve - step 0 is a solve step and must not be skipped.
        nb = _next_sc_step(args, sc, active, step + 1 if skip_ab else step, n_steps)
        if nb >= n_steps:
            n_sub = n_steps - step
            stop_a = False
        else:
            n_sub = nb - step + 1
            stop_a = True
        if n_sub > max_chunk:
            n_sub = max_chunk
            stop_a = False

        if n_sub > cap:
            cap = int(n_sub)
            r_hist = np.empty((cap, n, 3))
            v_hist = np.empty((cap, n, 3))
            active_hist = np.empty((cap, n), dtype=np.bool_)
            t_hist = np.empty(cap)
            ref_hist = np.empty(cap, dtype=np.bool_)
            kick_hist = np.empty(cap, dtype=np.bool_)
        hit_p = np.empty(n, dtype=np.int64)
        hit_s = np.empty(n, dtype=np.int64)

        _tk = time.perf_counter()
        if numba_mod is not None:
            numba_mod.set_num_threads(thr_kernel)
        n_c, lost_flag, n_kick, n_hits, t = kernel(
            r, v, active, bstep, step, int(n_sub), bool(skip_ab), bool(stop_a),
            t, float(dt), args['q_over_m'], args['use_batch'],
            args['bfld'], args['e_in'], args['e_out'], args['e_st'],
            args['blend_on'], args['blend_r0'], args['blend_r1'],
            args['taper_on'], args['taper_s'], args['taper_r0'], args['taper_r1'],
            args['omega'], args['phase'],
            sc_on, sc_E, sc_B, sc_kick_mask, sc_has_field, sc_qom, sc_kick_z,
            args['r_max'], args['z_max'], args['metal_on'], args['metal_every'],
            args['exempt_ref'], args['ref_particle'], args['ras_a'], args['ras_b'],
            hints, r_hist, v_hist, active_hist, t_hist, ref_hist, kick_hist, hit_p, hit_s)
        if numba_mod is not None:
            # back to the process default so the space-charge solve between
            # chunks (poisson_fft has parallel=True kernels of its own) is not
            # throttled by the cap the tracking kernel wants
            numba_mod.set_num_threads(thr_outer)
        t_kernel += time.perf_counter() - _tk
        n_chunks += 1

        # replay: the metal hits of the step, then the unchanged callback
        _tr = time.perf_counter()
        h = 0
        stop_m = -1
        for m in range(int(n_c)):
            s = step + m
            # the kick of a step the callback then stops on still happened
            if kick_hist[m]:
                n_kick_steps += 1
            if metal is not None:
                while h < n_hits and hit_s[h] == s:
                    p = int(hit_p[h])
                    metal.hits.append((p - index_offset, int(s),
                                       float(r_hist[m, p, 0]), float(r_hist[m, p, 1])))
                    metal.hits_z.append(float(r_hist[m, p, 2]))
                    h += 1
                if ref_hist[m]:
                    pr = args['ref_particle']
                    metal.ref_hits.append((int(s), float(r_hist[m, pr, 0]),
                                           float(r_hist[m, pr, 1])))
            if callback(s, r_hist[m], v_hist[m], active_hist[m], t_hist[m]):
                stopped = True
                stop_m = m
                break
        t_replay += time.perf_counter() - _tr

        if stopped:
            m = stop_m
        elif lost_flag:
            all_lost = True
            m = int(n_c) - 1
        else:
            m = None

        if m is not None:
            steps_done = step + m + 1
            final = (r_hist[m].copy(), v_hist[m].copy(), active_hist[m].copy(),
                     float(t_hist[m]))
            break

        if stop_a:
            step = step + int(n_sub) - 1
            mid = True
        else:
            step = step + int(n_sub)
        steps_done = step

    if sc_on:
        sc.n_kick_steps += n_kick_steps
    if release is not None:
        release.released = int(np.sum(release.birth_step <= min(steps_done, n_steps)))

    if final is None:
        final = (r.copy(), v.copy(), active.copy(), float(t))
        steps_done = n_steps
    r_f, v_f, a_f, t_f = final

    finder.last_fast_path = {'used': True, 'path': '3d', 'interp': args['interp'],
                             'parallel': parallel, 'threads': args.get('threads'),
                             'steps': int(steps_done),
                             'n_particles': int(n),
                             'chunks': int(n_chunks), 'max_chunk': int(max_chunk),
                             'sc_solves': int(sc.n_solves) if sc_on else 0,
                             'kernel_seconds': float(t_kernel),
                             'replay_seconds': float(t_replay),
                             'seconds': time.time() - t_start}
    if all_lost:
        return TrackingResult(False, int(steps_done), r_f, v_f, a_f,
                              {'termination': 'all_lost', 'time': t_f})
    if stopped:
        return TrackingResult(True, int(steps_done), r_f, v_f, a_f,
                              {'termination': 'turns_or_energy_reached', 'time': t_f})
    return TrackingResult(True, int(steps_done), r_f, v_f, a_f, {'time': t_f})


if __name__ == '__main__':                            # pragma: no cover
    for k, val in selftest().items():
        print(f"{k:28s} {val}")
