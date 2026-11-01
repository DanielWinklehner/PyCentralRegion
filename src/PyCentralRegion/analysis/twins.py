"""The space-charge vortex in guiding-centre coordinates, from twin runs that differ only in space charge.

A particle's instantaneous bunch-frame position mixes its guiding centre (energy -> radius, arrival phase) with its
orbit-centre offset, which turns once per revolution in the bunch frame and precesses at 2 pi (nu_r - 1) per turn. The
vortex drift (dr/dt = +E_l / B, dl/dt = -E_r / B) acts on the guiding centres, and space charge also lowers nu_r and so
changes the precession of the offsets: in the instantaneous core map of ``metrics.one_turn_map`` the offsets dilute
the vortex and their shifted precession cancels the rest. Averaging each particle's radius and arrival time over J
equally spaced azimuths of a turn removes the offset to first order and leaves the guiding-centre coordinates

    g_r = <R> - <R>_bunch,   g_l = omega <R>_bunch (<t>_bunch - <t>)      (ahead > 0)

per particle and turn window n (azimuths psi0 + 2 pi [n, n + 1) along the motion). On the particles that survive in
both runs, the one-turn map of the guiding-centre core, g(n + 1) = M g(n) + m0, gives the rotation
atan2(M_lr - M_rl, tr M) (``metrics.map_rotation``), b = M_rl and c = M_lr; the space-charge part is SC on minus SC
off, and a vortex has b > 0, c < 0 and a negative rotation.

- Cumulative rotation: the running sum of the per-map SC rotations from window 0, the estimator that recovered the
  truth on synthetic twins. Cross-check: the eigen-angle of the paired map g_on(n) = A(n) g_off(n) + a0 (its polar
  angle is inflated by shear and r-l decorrelation; it is not used).
- Time base: map n sees the drift through a triangle over [t_n - 1/2, t_(n+1) + 1/2] revolutions that peaks at
  (t_n + t_(n+1)) / 2. The Gaussian reference (the SC-on score's b_pred, c_pred, rot_pred of the time-turns and of
  turn0_part) is averaged with that kernel; it is the central field gradient of a Gaussian bunch, more than a 2-sigma
  core fit sees (the like-for-like reference is the drift in the real space-charge field, not computed here).
- Errors: paired bootstrap resamples of the common survivors, statistical only; the window phase (twin_psi0_deg)
  adds about 1 deg per turn systematically.
"""
import numpy as np

from .metrics import (alive_mask, core_covariance, core_variance_factor, ghost_first_step, map_rotation,
                      rotation_sense, run_constants, survivors)
from .objectives import ScoreConfig


def guiding_centres(xyz, idx, steps, alive, sense, f_rev, spt, n_steps, n_az, psi0_deg):
    """Mean radius <R> [mm] and mean time <t> [s] of the particles idx in every complete turn window: over n_az
    azimuths psi0 + 2 pi (n + (j + 1/2) / n_az), psi the particle's azimuth along the motion (unwrapped, monotonic),
    interpolated linearly in psi between its live records; window 0 is the first one after its start. A particle born
    just past psi0 starts its count one turn late, so the rows are aligned on the bunch's window-0 time. xyz (S, N, 3)
    [m]; returns R, T (n, n_windows), NaN where a window is incomplete."""
    t = np.asarray(steps) * (1.0 / f_rev / spt)
    n_win = int(n_steps / spt) + 2
    R, T = np.full((len(idx), n_win), np.nan), np.full((len(idx), n_win), np.nan)
    psi_s = np.deg2rad(psi0_deg)
    frac = (np.arange(n_az) + 0.5) / n_az * 2 * np.pi
    for m, i in enumerate(idx):
        ok = alive[:, i]
        if ok.sum() < 2:
            continue
        p = xyz[ok, i]
        psi = np.maximum.accumulate(np.unwrap(sense * np.arctan2(p[:, 1], p[:, 0])))
        base = psi_s + 2 * np.pi * np.ceil((psi[0] - psi_s) / (2 * np.pi))
        nf = min(int(np.floor((psi[-1] - base) / (2 * np.pi))), n_win)
        if nf <= 0:
            continue
        q = base + 2 * np.pi * np.arange(nf)[:, None] + frac[None, :]
        R[m, :nf] = np.interp(q, psi, 1e3 * np.hypot(p[:, 0], p[:, 1])).mean(axis=1)
        T[m, :nf] = np.interp(q, psi, t[ok]).mean(axis=1)
    k = np.rint((T[:, 0] - np.nanmedian(T[:, 0])) * f_rev)
    k = np.where(np.isfinite(k), k, 0).astype(int)
    for m in np.flatnonzero(k):
        for A in (R, T):
            A[m] = np.roll(A[m], k[m])
            if k[m] > 0:
                A[m, :k[m]] = np.nan
            else:
                A[m, k[m]:] = np.nan
    return R, T


def gc_frame(R, T, f_rev):
    """Guiding-centre bunch coordinates g (n, n_windows, 2) [mm] (g_r, g_l ahead > 0) from <R> [mm] and <t> [s]."""
    Rb, Tb = np.nanmean(R, axis=0), np.nanmean(T, axis=0)
    return np.stack([R - Rb[None, :], 2 * np.pi * f_rev * Rb[None, :] * (Tb[None, :] - T)], axis=2)


def gc_core(g, q, sigma):
    """The core of the guiding centres g (n, 2) of one window: the mask of the particles within sigma of the mean in
    the Gaussian-equivalent core covariance (``metrics.core_covariance`` about the core's own mean) and that
    covariance."""
    ok = np.all(np.isfinite(g), axis=1)
    x = g[ok] - g[ok].mean(axis=0)
    Cc, d2 = core_covariance(x[None], q, about_core=True)
    mask = np.zeros(len(g), bool)
    mask[np.flatnonzero(ok)] = d2[0] <= sigma ** 2
    return mask, Cc[0] / core_variance_factor(q)


def fit_map(X, Y):
    """The linear part M of Y = M X + y0 (least squares with an offset)."""
    return np.linalg.lstsq(np.column_stack([X, np.ones(len(X))]), Y, rcond=None)[0][:2].T


def eig_angle(A):
    """Similarity-invariant rotation angle [deg] of a 2x2 map: the argument of its complex eigenvalues, signed like
    ``metrics.map_rotation``; NaN for real eigenvalues."""
    tr, det = float(np.trace(A)), float(np.linalg.det(A))
    if det <= 0 or abs(tr) >= 2 * np.sqrt(det):
        return float('nan')
    return float(np.sign(A[1, 0] - A[0, 1]) * np.degrees(np.arccos(tr / (2 * np.sqrt(det)))))


def turn_map(g, n, q, sigma):
    """The one-turn map of the guiding-centre core of window n -> n + 1 and the core covariance of window n."""
    cm, Cc = gc_core(g[:, n], q, sigma)
    ok = cm & np.all(np.isfinite(g[:, n + 1]), axis=1)
    return fit_map(g[ok, n], g[ok, n + 1]), Cc


def paired_map(g_on, g_off, n, q, sigma):
    """g_on(n) = A g_off(n) + a0 on the SC-off core of window n."""
    cm, _ = gc_core(g_off[:, n], q, sigma)
    ok = cm & np.all(np.isfinite(g_on[:, n]), axis=1)
    return fit_map(g_off[ok, n], g_on[ok, n])


def sc_terms(g_on, g_off, q, sigma):
    """The SC parts (on - off) of b, c and the rotation of every one-turn map, their running sum from window 0
    (cum_sum), and the eigen-angle of the paired map per window (cum_eig)."""
    nT = g_on.shape[1]
    Mon = [turn_map(g_on, n, q, sigma)[0] for n in range(nT - 1)]
    Moff = [turn_map(g_off, n, q, sigma)[0] for n in range(nT - 1)]
    t = {'b': np.array([a[0, 1] - b[0, 1] for a, b in zip(Mon, Moff)]),
         'c': np.array([a[1, 0] - b[1, 0] for a, b in zip(Mon, Moff)]),
         'rot': np.array([map_rotation(a) - map_rotation(b) for a, b in zip(Mon, Moff)])}
    t['cum_sum'] = np.r_[0.0, np.cumsum(t['rot'])]
    t['cum_eig'] = np.array([eig_angle(paired_map(g_on, g_off, n, q, sigma)) for n in range(nT)])
    return t


def kernel_weights(t0, t1, ks):
    """Weights of the time-turns [k, k + 1) revolutions in the guiding-centre map between the windows centred at t0
    and t1: g(n + 1) - g(n) integrates the drift with a triangle over [t0 - 1/2, t1 + 1/2] peaking at (t0 + t1) / 2."""
    a, m, e = t0 - 0.5, 0.5 * (t0 + t1), t1 + 0.5

    def cdf(x):
        x = min(max(x, a), e)
        return (x - a) ** 2 / ((m - a) * (e - a)) if x <= m else 1.0 - (e - x) ** 2 / ((e - m) * (e - a))
    return {k: cdf(k + 1) - cdf(k) for k in ks}


def gauss_reference(score, t0, t1):
    """The score's Gaussian prediction (b_pred, c_pred, rot_pred per time-turn; time-turn 0 from turn0_part, which
    starts once the last survivor is born) averaged with the map's kernel; the coverage (kernel weight with a value)
    and the weights."""
    pt = {int(k): v for k, v in (score.get('turns') or {}).items() if str(k).isdigit() and 'b_pred' in v}
    if (score.get('turn0_part') or {}).get('b_pred') is not None:
        pt[0] = score['turn0_part']
    w = kernel_weights(t0, t1, sorted(pt))
    cov = sum(w.values())
    if cov <= 0:
        return {}, 0.0, {}
    return ({key: float(sum(w[k] * pt[k][key] for k in pt) / cov) for key in ('b_pred', 'c_pred', 'rot_pred')},
            float(cov), {str(k): round(v, 4) for k, v in w.items() if v > 0})


def _prepare(run, cfg):
    k = run_constants(run, cfg)
    steps = np.asarray(run.steps)
    gstep = ghost_first_step(run.ids, run.ghost_ids, run.ghost_step)
    alive = alive_mask(steps, run.xyz, run.birth_step, run.stop_step, gstep)
    surv = survivors(run.xyz[-1], run.stop_step, run.n_steps, gstep, k['r_max'], k['z_max'])
    return k, steps, alive, surv, rotation_sense(run.xyz, alive)


def vortex_twin(run_on, run_off, score_on=None, cfg=None):
    """The space-charge vortex of run_on measured against its SC-off twin run_off (``runs.RunData`` tracking the same
    particles), in guiding-centre coordinates (module docstring); score_on, the SC-on run's score, adds the
    time-matched Gaussian reference. Returns tag_on, tag_off, J, psi0_deg, n_both, sense, boot, window_centres_rev,
    turns (per window n: t_rev; on / off: R_mean, sig_r, sig_l, ratio, corr of the core and, but for the last window,
    the map M, rot, b, c; the paired map A, det_A, cum_eig, cum_sum with errors; sc (b, c, rot: on - off) with sc_err
    and pred, pred_coverage, pred_weights) and the totals rot_sc_total (cum_sum after the last map), its error and
    rot_pred_total (the kernel-averaged Gaussian rotations summed over the maps)."""
    cfg = cfg or ScoreConfig()
    if not np.array_equal(run_on.ids, run_off.ids):
        raise ValueError(f"{run_on.tag} and {run_off.tag} track different particles: not twins")
    q, sig, J, psi0 = cfg.core_q, cfg.map_core_sigma, int(cfg.twin_azimuths), float(cfg.twin_psi0_deg)
    k_on, st_on, al_on, sv_on, s_on = _prepare(run_on, cfg)
    k_off, st_off, al_off, sv_off, s_off = _prepare(run_off, cfg)
    both = np.flatnonzero(sv_on & sv_off)
    R_on, T_on = guiding_centres(run_on.xyz, both, st_on, al_on, s_on, k_on['f_rev'], run_on.steps_per_turn,
                                 run_on.n_steps, J, psi0)
    R_off, T_off = guiding_centres(run_off.xyz, both, st_off, al_off, s_off, k_off['f_rev'], run_off.steps_per_turn,
                                   run_off.n_steps, J, psi0)
    full = [np.flatnonzero(np.isfinite(R).mean(axis=0) > cfg.twin_min_share) for R in (R_on, R_off)]
    if not all(len(f) >= 2 for f in full):
        raise ValueError(f"fewer than two complete turn windows in {run_on.tag} / {run_off.tag}")
    nT = min(f.max() for f in full) + 1
    g_on, g_off = gc_frame(R_on[:, :nT], T_on[:, :nT], k_on['f_rev']), gc_frame(R_off[:, :nT], T_off[:, :nT],
                                                                                 k_off['f_rev'])
    t_win = np.nanmedian(T_on[:, :nT], axis=0) * k_on['f_rev']
    nom = sc_terms(g_on, g_off, q, sig)
    err = {}
    if cfg.twin_boot:
        rng = np.random.default_rng(cfg.twin_boot_seed)
        bs = [sc_terms(g_on[i], g_off[i], q, sig)
              for i in (rng.integers(0, len(g_on), len(g_on)) for _ in range(int(cfg.twin_boot)))]
        err = {key: np.nanstd([b[key] for b in bs], axis=0) for key in nom}
    res = {'tag_on': run_on.tag, 'tag_off': run_off.tag, 'J': J, 'psi0_deg': psi0, 'n_both': int(len(both)),
           'sense': [float(s_on), float(s_off)], 'boot': int(cfg.twin_boot),
           'window_centres_rev': t_win.tolist(), 'turns': []}
    for n in range(nT):
        row = {'n': n, 't_rev': float(t_win[n])}
        for lab, g, R in (('on', g_on, R_on), ('off', g_off, R_off)):
            _, Cc = gc_core(g[:, n], q, sig)
            w = np.linalg.eigvalsh(Cc)
            row[lab] = {'R_mean': float(np.nanmean(R[:, n])), 'sig_r': float(np.sqrt(Cc[0, 0])),
                        'sig_l': float(np.sqrt(Cc[1, 1])), 'ratio': float(np.sqrt(w[1] / w[0])),
                        'corr': float(Cc[0, 1] / np.sqrt(Cc[0, 0] * Cc[1, 1]))}
            if n + 1 < nT:
                M = turn_map(g, n, q, sig)[0]
                row[lab].update(M=M.tolist(), rot=map_rotation(M), b=float(M[0, 1]), c=float(M[1, 0]))
        A = paired_map(g_on, g_off, n, q, sig)
        row.update(A=A.tolist(), det_A=float(np.linalg.det(A)), cum_eig=float(nom['cum_eig'][n]),
                   cum_sum=float(nom['cum_sum'][n]))
        if err:
            row.update(cum_eig_err=float(err['cum_eig'][n]), cum_sum_err=float(err['cum_sum'][n]))
        if n + 1 < nT:
            row['sc'] = {key: float(nom[key][n]) for key in ('b', 'c', 'rot')}
            if err:
                row['sc_err'] = {key: float(err[key][n]) for key in ('b', 'c', 'rot')}
            if score_on is not None:
                row['pred'], row['pred_coverage'], row['pred_weights'] = gauss_reference(score_on, t_win[n],
                                                                                         t_win[n + 1])
        res['turns'].append(row)
    preds = [r['pred']['rot_pred'] for r in res['turns'] if r.get('pred')]
    res.update(rot_sc_total=float(nom['cum_sum'][-1]),
               rot_sc_total_err=float(err['cum_sum'][-1]) if err else float('nan'),
               rot_pred_total=float(np.sum(preds)) if preds else float('nan'))
    return res
