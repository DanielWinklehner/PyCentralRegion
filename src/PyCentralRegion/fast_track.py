"""
fast_track.py - Compiled (numba) single-particle tracking for the 2D
central-region model: the fast path of ``AcceleratedOrbitFinder.track_with_rf``.

The general path drives ``PyPATools.trackers.Tracker`` with Python hooks
(RF kicks, terminators, the finder's recording callback) and spends about a
millisecond of interpreter time per step on ONE particle. This module runs the
same physics in one ``njit`` kernel and hands the recorded steps to the very
same callback afterwards, so every diagnostic (Poincare crossings, turn
statistics, reference trajectory, full-beam record) is produced by unchanged
code and the two paths agree to round-off (tests/test_fast_track.py).

What the kernel reproduces, step by step, from ``Tracker.run`` /
``Pusher._rk4_step_batch`` / ``RFCavityInteraction`` / ``RFCavity``:
  * RF time frozen at the step midpoint for the four RK4 stages (TimedField
    convention), relativistic RK4 (``rk4_rel_dbetagamma_dt``) with the
    0.9999 c clamp, bilinear field lookups (``_interp2d_single``);
  * thin-gap kicks in cavity order: Cramer crossing test on the step chord,
    RK4 backtrack to the crossing, kick along the crossed segment's normal
    with the transit-time factor, the radial voltage profile (tabulated) and
    the crossing phase ``omega * (t_step + dt - dt_back) + phase`` exactly as
    ``apply_kicks_batch`` computes it, RK4 forward to the step end;
  * radial boundary and the midplane obstacle raster as terminators;
  * the Poincare section crossing count with the same arming rule, so the
    kernel stops where the callback would (max turns / target energy).

Supported: one physical particle (no virtual reference, no staggered
launch), ``rk4_rel``, gridded 2D or constant B, E as zero / a gridded 2D
static field / a ``TimedField`` pattern / their ``StaticPlusRFField`` sum,
thin-gap or bem2d gap model, at most one raster obstacle test, no collimator.
Anything else raises ``FastPathUnavailable`` and the finder takes the general
path.
"""
from typing import Optional, Tuple

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:                                   # pragma: no cover
    HAS_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]

from PyPATools.global_variables import CLIGHT
from PyPATools.field import Field, TimedField
from PyPATools.field_src.interpolators import _interp2d_single


class FastPathUnavailable(Exception):
    """The configuration is outside what the kernel reproduces."""


# ============================================================================
# numba kernels
# ============================================================================
@njit(cache=True, fastmath=True)
def _field2d(x, y, mode, gx, gy, vx, vy, vz, scale, const, out):
    """E or B at (x, y): mode 0 zero, 1 constant, 2 gridded (bilinear, 0 outside)."""
    if mode == 0:
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
    elif mode == 1:
        out[0] = scale * const[0]
        out[1] = scale * const[1]
        out[2] = scale * const[2]
    else:
        out[0] = scale * _interp2d_single(x, y, gx, gy, vx, 0.0)
        out[1] = scale * _interp2d_single(x, y, gx, gy, vy, 0.0)
        out[2] = scale * _interp2d_single(x, y, gx, gy, vz, 0.0)


@njit(cache=True, fastmath=True)
def _accel(v, e, b, q_over_m, out):
    """d(gamma v)/dt as rk4_rel_dbetagamma_dt."""
    v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    gamma = 1.0 / np.sqrt(1.0 - v2 / (CLIGHT * CLIGHT))
    fx = q_over_m * (e[0] + v[1] * b[2] - v[2] * b[1])
    fy = q_over_m * (e[1] + v[2] * b[0] - v[0] * b[2])
    fz = q_over_m * (e[2] + v[0] * b[1] - v[1] * b[0])
    vdotf = v[0] * fx + v[1] * fy + v[2] * fz
    corr = vdotf / (gamma * CLIGHT * CLIGHT)
    out[0] = fx / gamma - corr * v[0]
    out[1] = fy / gamma - corr * v[1]
    out[2] = fz / gamma - corr * v[2]


@njit(cache=True, fastmath=True)
def _rk4(r, v, dt, cos_rf, q_over_m,
         bm, bgx, bgy, bvx, bvy, bvz, bsc, bconst,
         sm, sgx, sgy, svx, svy, svz, ssc, sconst,
         rm, rgx, rgy, rvx, rvy, rvz, rsc, rconst,
         r_new, v_new):
    """One relativistic RK4 step with fields re-evaluated at the four stages
    (RF pattern frozen at cos_rf), as Pusher._rk4_step_batch."""
    e = np.empty(3)
    es = np.empty(3)
    b = np.empty(3)
    k_v = np.empty((4, 3))
    k_r = np.empty((4, 3))
    rr = np.empty(3)
    vv = np.empty(3)
    for i in range(3):
        rr[i] = r[i]
        vv[i] = v[i]
    for stage in range(4):
        _field2d(rr[0], rr[1], bm, bgx, bgy, bvx, bvy, bvz, bsc, bconst, b)
        _field2d(rr[0], rr[1], sm, sgx, sgy, svx, svy, svz, ssc, sconst, es)
        _field2d(rr[0], rr[1], rm, rgx, rgy, rvx, rvy, rvz, rsc, rconst, e)
        for i in range(3):
            e[i] = es[i] + e[i] * cos_rf
        _accel(vv, e, b, q_over_m, k_v[stage])
        for i in range(3):
            k_r[stage, i] = vv[i]
        if stage < 3:
            f = 0.5 * dt if stage < 2 else dt
            for i in range(3):
                rr[i] = r[i] + f * k_r[stage, i]
                vv[i] = v[i] + f * k_v[stage, i]
    for i in range(3):
        r_new[i] = r[i] + (dt / 6.0) * (k_r[0, i] + 2.0 * k_r[1, i] + 2.0 * k_r[2, i] + k_r[3, i])
        v_new[i] = v[i] + (dt / 6.0) * (k_v[0, i] + 2.0 * k_v[1, i] + 2.0 * k_v[2, i] + k_v[3, i])
    vmag = np.sqrt(v_new[0] * v_new[0] + v_new[1] * v_new[1] + v_new[2] * v_new[2])
    if vmag >= 0.9999 * CLIGHT:
        f = 0.9999 * CLIGHT / vmag
        for i in range(3):
            v_new[i] = v_new[i] * f


@njit(cache=True, fastmath=True)
def _track_kernel(r0, v0, dt, n_steps, t0, q_over_m, q_state, mass_mev,
                  bm, bgx, bgy, bvx, bvy, bvz, bsc, bconst,
                  sm, sgx, sgy, svx, svy, svz, ssc, sconst,
                  rm, rgx, rgy, rvx, rvy, rvz, rsc, rconst, omega_rf, phase_rf,
                  use_kicks, cav_seg_start, seg_p1, seg_p2, seg_dir, seg_perp,
                  cav_omega, cav_phase, cav_voltage, cav_gap_w, cav_gap_w_inner, cav_taper_r,
                  cav_r_min, vp_start, vp_r, vp_s,
                  r_max, use_raster, raster, raster_ext, raster_h,
                  section_angle, arm_angle, max_turns, target_energy_mev,
                  r_hist, v_hist, active_hist):
    """Returns (n_done, lost_step, lost_reason, n_cross, stop_reason).
    lost_reason: 0 none, 1 radial boundary, 2 obstacle.
    stop_reason: 0 ran out of steps, 1 lost, 2 turns reached, 3 energy reached."""
    r = np.empty(3)
    v = np.empty(3)
    r_prev = np.empty(3)
    v_prev = np.empty(3)
    r_new = np.empty(3)
    v_new = np.empty(3)
    r_c = np.empty(3)
    v_c = np.empty(3)
    r_cb = np.empty(3)          # the callback's previous post-step position
    for i in range(3):
        r[i] = r0[i]
        v[i] = v0[i]
    t = t0
    n_cav = len(cav_seg_start) - 1
    n_cross = 0
    armed = arm_angle <= 0.0
    az_travelled = 0.0
    lost_step = -1
    lost_reason = 0
    stop_reason = 0
    n_done = 0
    for step in range(n_steps):
        for i in range(3):
            r_prev[i] = r[i]
            v_prev[i] = v[i]
        t_mid = t + 0.5 * dt
        cos_rf = np.cos(omega_rf * t_mid + phase_rf) if rm != 0 else 0.0
        _rk4(r, v, dt, cos_rf, q_over_m, bm, bgx, bgy, bvx, bvy, bvz, bsc, bconst,
             sm, sgx, sgy, svx, svy, svz, ssc, sconst, rm, rgx, rgy, rvx, rvy, rvz, rsc, rconst,
             r_new, v_new)
        for i in range(3):
            r[i] = r_new[i]
            v[i] = v_new[i]
        # ---- thin-gap kicks, cavity by cavity, on the chord r_prev -> r
        if use_kicks:
            for c in range(n_cav):
                crossed = False
                s_part = 0.0
                seg_hit = -1
                dpx = r[0] - r_prev[0]
                dpy = r[1] - r_prev[1]
                for s in range(cav_seg_start[c], cav_seg_start[c + 1]):
                    dcx = seg_p2[s, 0] - seg_p1[s, 0]
                    dcy = seg_p2[s, 1] - seg_p1[s, 1]
                    det = dcx * (-dpy) - dcy * (-dpx)
                    det = -det
                    if abs(det) <= 1e-10:
                        continue
                    bx = r_prev[0] - seg_p1[s, 0]
                    by = r_prev[1] - seg_p1[s, 1]
                    det_t = bx * (-dpy) - by * (-dpx)
                    det_t = -det_t
                    det_s = dcx * by - dcy * bx
                    t_cav = det_t / det
                    sp = -det_s / det
                    if t_cav >= 0.0 and t_cav <= 1.0 and sp >= 0.0 and sp <= 1.0:
                        crossed = True
                        s_part = sp
                        seg_hit = s
                        break
                if not crossed:
                    continue
                dt_back = (1.0 - s_part) * dt
                _rk4(r, v, -dt_back, cos_rf, q_over_m, bm, bgx, bgy, bvx, bvy, bvz, bsc, bconst,
                     sm, sgx, sgy, svx, svy, svz, ssc, sconst, rm, rgx, rgy, rvx, rvy, rvz, rsc, rconst,
                     r_c, v_c)
                t_cavity = t + dt - dt_back   # crossing time (t = step start)
                speed = np.sqrt(v_c[0] * v_c[0] + v_c[1] * v_c[1] + v_c[2] * v_c[2])
                gamma = 1.0 / np.sqrt(1.0 - (speed / CLIGHT) ** 2)
                e_old = (gamma - 1.0) * mass_mev
                r_cross = np.sqrt(r_c[0] * r_c[0] + r_c[1] * r_c[1])
                gw = cav_gap_w[c]
                if cav_gap_w_inner[c] > 0.0:
                    frac = (r_cross - cav_r_min[c]) / (cav_taper_r[c] - cav_r_min[c])
                    frac = max(0.0, min(frac, 1.0))
                    gw = cav_gap_w_inner[c] + frac * (cav_gap_w[c] - cav_gap_w_inner[c])
                x = cav_omega[c] * (gw / speed) / 2.0
                ttf = np.sin(x) / x
                v_gap = cav_voltage[c]
                if vp_start[c + 1] > vp_start[c]:
                    v_gap = v_gap * np.interp(r_cross, vp_r[vp_start[c]:vp_start[c + 1]],
                                              vp_s[vp_start[c]:vp_start[c + 1]])
                d_e = 1e-6 * q_state * v_gap * ttf * np.cos(cav_omega[c] * t_cavity + cav_phase[c])
                gamma_new = (e_old + d_e) / mass_mev + 1.0
                bg_new = np.sqrt(max(gamma_new * gamma_new - 1.0, 0.0))
                ux = (gamma / CLIGHT) * v_c[0]
                uy = (gamma / CLIGHT) * v_c[1]
                uz = (gamma / CLIGHT) * v_c[2]
                u_along = ux * seg_dir[seg_hit, 0] + uy * seg_dir[seg_hit, 1]
                u_perp = ux * seg_perp[seg_hit, 0] + uy * seg_perp[seg_hit, 1]
                u_perp_new = np.sign(u_perp) * np.sqrt(max(bg_new * bg_new - u_along * u_along - uz * uz, 0.0))
                unx = u_along * seg_dir[seg_hit, 0] + u_perp_new * seg_perp[seg_hit, 0]
                uny = u_along * seg_dir[seg_hit, 1] + u_perp_new * seg_perp[seg_hit, 1]
                un2 = unx * unx + uny * uny + uz * uz
                f = CLIGHT / np.sqrt(1.0 + un2)
                v_c[0] = f * unx
                v_c[1] = f * uny
                v_c[2] = f * uz
                _rk4(r_c, v_c, dt_back, cos_rf, q_over_m, bm, bgx, bgy, bvx, bvy, bvz, bsc, bconst,
                     sm, sgx, sgy, svx, svy, svz, ssc, sconst, rm, rgx, rgy, rvx, rvy, rvz, rsc, rconst,
                     r_new, v_new)
                for i in range(3):
                    r[i] = r_new[i]
                    v[i] = v_new[i]
        # ---- terminators
        active = True
        if np.sqrt(r[0] * r[0] + r[1] * r[1]) > r_max:
            active = False
            lost_reason = 1
        if active and use_raster:
            ii = int(np.rint((r[0] + raster_ext) / raster_h))
            jj = int(np.rint((r[1] + raster_ext) / raster_h))
            nn = raster.shape[0]
            if ii >= 0 and ii < nn and jj >= 0 and jj < raster.shape[1]:
                if raster[ii, jj] != 0:
                    active = False
                    lost_reason = 2
        t += dt
        for i in range(3):
            r_hist[step, i] = r[i]
            v_hist[step, i] = v[i]
        active_hist[step] = active
        n_done = step + 1
        if not active:
            lost_step = step
            stop_reason = 1
            break
        # ---- Poincare section crossing, as the finder's callback sees it
        if step == 0:
            for i in range(3):
                r_cb[i] = r[i]
            continue
        crossed = False
        if not armed:
            dth = np.arctan2(r[1], r[0]) - np.arctan2(r_cb[1], r_cb[0])
            if dth > np.pi:
                dth -= 2.0 * np.pi
            elif dth < -np.pi:
                dth += 2.0 * np.pi
            az_travelled += dth
            if az_travelled >= arm_angle:
                armed = True
        if armed:
            if section_angle == 0.0:
                if r_cb[1] <= 0.0 and r[1] > 0.0:
                    crossed = True
            else:
                th_o = np.arctan2(r_cb[1], r_cb[0])
                th_n = np.arctan2(r[1], r[0])
                dth = th_n - th_o
                if dth > np.pi:
                    dth -= 2.0 * np.pi
                elif dth < -np.pi:
                    dth += 2.0 * np.pi
                if dth > 0.0:
                    ao = section_angle - th_o
                    an = section_angle - th_n
                    ao = np.arctan2(np.sin(ao), np.cos(ao))
                    an = np.arctan2(np.sin(an), np.cos(an))
                    if ao * an < 0.0 and ao > 0.0:
                        crossed = True
        for i in range(3):
            r_cb[i] = r[i]
        if crossed:
            n_cross += 1
            vm2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
            e_mev = (1.0 / np.sqrt(1.0 - vm2 / (CLIGHT * CLIGHT)) - 1.0) * mass_mev
            if e_mev >= target_energy_mev:
                stop_reason = 3
                break
            if n_cross >= max_turns:
                stop_reason = 2
                break
    return n_done, lost_step, lost_reason, n_cross, stop_reason


# ============================================================================
# Packing the finder's configuration for the kernel
# ============================================================================
_EMPTY2 = np.zeros((2, 2))
_EMPTY1 = np.zeros(2)


def _pack_field(f, what: str):
    """(mode, gx, gy, vx, vy, vz, scale, const) of a Field usable by the kernel."""
    if f is None:
        return (0, _EMPTY1, _EMPTY1, _EMPTY2, _EMPTY2, _EMPTY2, 1.0, np.zeros(3))
    if not isinstance(f, Field):
        raise FastPathUnavailable(f"{what}: {type(f).__name__} is not a plain Field")
    label = getattr(f, 'label', '')
    if label == 'Zero Field':
        return (0, _EMPTY1, _EMPTY1, _EMPTY2, _EMPTY2, _EMPTY2, 1.0, np.zeros(3))
    scale = float(getattr(f, '_scaling', 1.0))
    if f.dim == 0:
        c = np.array([float(f._field[k]) for k in 'xyz'])
        return (1, _EMPTY1, _EMPTY1, _EMPTY2, _EMPTY2, _EMPTY2, scale, c)
    if f.dim != 2:
        raise FastPathUnavailable(f"{what}: dim {f.dim} field (only 2D grids)")
    ips = [f._field[k] for k in 'xyz']
    for ip in ips:
        if not (hasattr(ip, '_grid') and hasattr(ip, '_values')):
            raise FastPathUnavailable(f"{what}: interpolator {type(ip).__name__} has no raw grid (use the numba backend)")
        if getattr(ip, 'method', 'linear') != 'linear':
            raise FastPathUnavailable(f"{what}: {ip.method} interpolation")
        if abs(float(getattr(ip, 'fill_value', 0.0))) > 0:
            raise FastPathUnavailable(f"{what}: fill value {ip.fill_value} (only 0)")
    gx = np.ascontiguousarray(ips[0]._grid[0], dtype=float)
    gy = np.ascontiguousarray(ips[0]._grid[1], dtype=float)
    vals = []
    for ip in ips:
        if len(ip._grid[0]) != len(gx) or len(ip._grid[1]) != len(gy):
            raise FastPathUnavailable(f"{what}: components on different grids")
        vals.append(np.ascontiguousarray(ip._values, dtype=float))
    return (2, gx, gy, vals[0], vals[1], vals[2], scale, np.zeros(3))


def _efield_parts(ef):
    """(static Field or None, rf pattern Field or None, omega, phase) of the
    design's E-field. omega / phase come from the OUTER timed object: that is
    what the engine re-syncs from cavity 0 before a run (an inner TimedField
    wrapped by StaticPlusRFField keeps its construction values)."""
    from .inflector import StaticPlusRFField
    if ef is None:
        return None, None, 0.0, 0.0
    if isinstance(ef, StaticPlusRFField):
        if len(ef.statics) > 1:
            raise FastPathUnavailable("more than one static E-field")
        static = ef.statics[0] if ef.statics else None
        if ef.rf is None:
            return static, None, 0.0, 0.0
        return static, ef.rf.static, float(ef.omega), float(ef.phase)
    if isinstance(ef, TimedField):
        return None, ef.static, float(ef.omega), float(ef.phase)
    if isinstance(ef, Field):
        return ef, None, 0.0, 0.0
    raise FastPathUnavailable(f"E-field of type {type(ef).__name__}")


def _pack_raster(mask):
    """(use, grid uint8, ext, h) of a raster obstacle test (or a combination of rasters)."""
    if mask is None:
        return (False, np.zeros((2, 2), dtype=np.uint8), 0.0, 1.0)
    parts = getattr(mask, 'parts', None) or [mask]
    grids = []
    for p in parts:
        if not hasattr(p, 'grid') or not hasattr(p, 'xs'):
            raise FastPathUnavailable("obstacle test is not a raster (electrodes3d.raster_obstacles)")
        grids.append(p)
    g0 = grids[0]
    grid = np.asarray(g0.grid, dtype=bool).copy()
    for p in grids[1:]:
        if p.grid.shape != grid.shape or abs(float(p.xs[0]) - float(g0.xs[0])) > 1e-12:
            raise FastPathUnavailable("obstacle rasters on different grids")
        grid |= np.asarray(p.grid, dtype=bool)
    ext = -float(g0.xs[0])
    h = float(g0.spacing)
    return (True, np.ascontiguousarray(grid.astype(np.uint8)), ext, h)


def _pack_cavities(cavities, use_kicks):
    seg_start = [0]
    p1, p2, d, p = [], [], [], []
    omega, phase, volt, gw, gwi, taper, rmin = [], [], [], [], [], [], []
    vp_start = [0]
    vp_r, vp_s = [], []
    for cav in cavities:
        for seg in cav.segments:
            p1.append(np.asarray(seg['p1'][:2], dtype=float))
            p2.append(np.asarray(seg['p2'][:2], dtype=float))
            d.append(np.asarray(seg['direction'][:2], dtype=float))
            p.append(np.asarray(seg['perp_2d'][:2], dtype=float))
        seg_start.append(len(p1))
        omega.append(float(cav.omega))
        phase.append(float(cav.get_total_phase_rad()))
        volt.append(float(cav.voltage))
        gw.append(float(cav.gap_width))
        gwi.append(float(cav.gap_width_inner) if cav.gap_width_inner is not None else -1.0)
        taper.append(float(cav.gap_taper_radius) if cav.gap_taper_radius is not None else 0.0)
        rmin.append(float(cav.r_min))
        scale = getattr(cav, '_voltage_scale', None)
        if scale is not None:
            if not (hasattr(scale, 'r') and hasattr(scale, 'scale_tab')):
                raise FastPathUnavailable("non-tabulated voltage profile")
            vp_r.extend(np.asarray(scale.r, dtype=float).tolist())
            vp_s.extend(np.asarray(scale.scale_tab, dtype=float).tolist())
        vp_start.append(len(vp_r))
    if not p1:
        p1 = p2 = d = p = [np.zeros(2)]
    return dict(use_kicks=bool(use_kicks), cav_seg_start=np.asarray(seg_start, dtype=np.int64),
                seg_p1=np.ascontiguousarray(np.vstack(p1)), seg_p2=np.ascontiguousarray(np.vstack(p2)),
                seg_dir=np.ascontiguousarray(np.vstack(d)), seg_perp=np.ascontiguousarray(np.vstack(p)),
                cav_omega=np.asarray(omega, dtype=float), cav_phase=np.asarray(phase, dtype=float),
                cav_voltage=np.asarray(volt, dtype=float), cav_gap_w=np.asarray(gw, dtype=float),
                cav_gap_w_inner=np.asarray(gwi, dtype=float), cav_taper_r=np.asarray(taper, dtype=float),
                cav_r_min=np.asarray(rmin, dtype=float), vp_start=np.asarray(vp_start, dtype=np.int64),
                vp_r=np.asarray(vp_r if vp_r else [0.0, 1.0], dtype=float),
                vp_s=np.asarray(vp_s if vp_s else [1.0, 1.0], dtype=float))


def build_kernel_args(finder, pd_init, dt: float, n_steps: int, t0: float,
                      section_angle: float, max_turns: int) -> dict:
    """Everything the kernel needs for this run, or FastPathUnavailable."""
    if not HAS_NUMBA:
        raise FastPathUnavailable("numba not installed")
    engine = finder.engine
    if int(pd_init.numpart) != 1:
        raise FastPathUnavailable("more than one particle")
    if not bool(np.asarray(pd_init.alive)[0]):
        raise FastPathUnavailable("particle not alive at launch")
    if str(getattr(engine, 'algorithm', '')).lower() != 'rk4_rel':
        raise FastPathUnavailable(f"algorithm {engine.algorithm}")
    if engine.dim != '2D':
        raise FastPathUnavailable("3D engine")
    for tm in engine.extra_terminators:
        if type(tm).__name__ != 'MetalTerminator':
            raise FastPathUnavailable(f"terminator {type(tm).__name__}")
    if engine.extra_interactions:
        raise FastPathUnavailable("extra interactions (staggered launch)")
    design = engine.design
    use_kicks = bool(engine.use_rf and engine.gap_model == 'thin')
    if engine.use_rf and engine.gap_model == 'bem2d':
        engine._sync_bem_field()
    static, rf_pattern, omega_rf, phase_rf = _efield_parts(design.efield)
    bpack = _pack_field(design.bfield, 'B-field')
    spack = _pack_field(static, 'static E-field')
    rpack = _pack_field(rf_pattern, 'RF E-field pattern')
    if rf_pattern is not None and not isinstance(rf_pattern, Field):
        raise FastPathUnavailable(f"RF pattern of type {type(rf_pattern).__name__}")
    masks = [tm.inside for tm in engine.extra_terminators]
    raster = _pack_raster(masks[0] if masks else None)
    if len(masks) > 1:
        raise FastPathUnavailable("more than one obstacle terminator")
    cav = _pack_cavities(design.rf_cavities, use_kicks)
    pusher = engine.pusher
    return dict(r0=np.ascontiguousarray(np.asarray(pd_init.x_vec[0], dtype=float)),
                v0=np.ascontiguousarray(np.asarray(pd_init.v_vec[0], dtype=float)),
                dt=float(dt), n_steps=int(n_steps), t0=float(t0),
                q_over_m=float(pusher.q_over_m), q_state=float(design.species.q),
                mass_mev=float(design.species.mass_mev),
                b=bpack, s=spack, rf=rpack, omega_rf=omega_rf, phase_rf=phase_rf,
                cav=cav, r_max=float(engine.r_max), raster=raster,
                section_angle=float(section_angle), arm_angle=float(np.pi),
                max_turns=int(max_turns), target_energy_mev=float(finder.target_energy_mev))


def run_kernel(args: dict):
    """Run the kernel; returns (r_hist, v_hist, active_hist, n_done, lost_step,
    lost_reason, n_cross, stop_reason)."""
    n = args['n_steps']
    r_hist = np.empty((n, 3))
    v_hist = np.empty((n, 3))
    active_hist = np.zeros(n, dtype=np.bool_)
    b, s, rf, cav, ra = args['b'], args['s'], args['rf'], args['cav'], args['raster']
    out = _track_kernel(
        args['r0'], args['v0'], args['dt'], n, args['t0'], args['q_over_m'], args['q_state'], args['mass_mev'],
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        rf[0], rf[1], rf[2], rf[3], rf[4], rf[5], rf[6], rf[7], args['omega_rf'], args['phase_rf'],
        cav['use_kicks'], cav['cav_seg_start'], cav['seg_p1'], cav['seg_p2'], cav['seg_dir'], cav['seg_perp'],
        cav['cav_omega'], cav['cav_phase'], cav['cav_voltage'], cav['cav_gap_w'], cav['cav_gap_w_inner'],
        cav['cav_taper_r'], cav['cav_r_min'], cav['vp_start'], cav['vp_r'], cav['vp_s'],
        args['r_max'], ra[0], ra[1], ra[2], ra[3],
        args['section_angle'], args['arm_angle'], args['max_turns'], args['target_energy_mev'],
        r_hist, v_hist, active_hist)
    n_done, lost_step, lost_reason, n_cross, stop_reason = out
    return r_hist, v_hist, active_hist, int(n_done), int(lost_step), int(lost_reason), int(n_cross), int(stop_reason)
