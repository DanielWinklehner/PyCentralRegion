"""Beam-quality metrics of 3D bunch-tracking runs.

Pure functions on arrays (recorded tracks, virtual-probe crossings, energies, scan rows) and ``score_run``, which
applies them to one run (``runs.RunData``) and returns its score: a flat dict of scalars (``objectives.METRICS`` lists
the ones an optimizer may use) plus a few nested tables (per turn, per probe, per release-phase bin, per slit).

The metric set serves the strategy "vortex motion keeps the bunch compact, collimators remove the halo": current and
acceptance, compression, the bunch shape and halo per turn, the measured against the predicted vortex rotation, the
per-particle orbit-centre fan, the turn structure at virtual probes, and the trade-off of one early slit. Algorithms and
key names are those of the 3D campaign's run scoring (2026-11-01), so its tables stay comparable; its constants are
``ScoreConfig`` fields. Added to it: the best-placed energy window, the release-phase budget (final energy, phase lag
and turn of loss per release-phase bin), the radial width split into its energy part and the rest, the last-turn
aspect ratio without the centre fan, survival against the hand-off phase band, |z| and |z'|, the compression slope per
turn (an early proxy of the final one), the fan's radial projection, and the best slits in the conventions of the
tracker's [slit] section. The vortex itself needs an SC-off twin of the run (``twins``).

Conventions: track positions are [m], every length in a score [mm], phases RF degrees. The bunch frame at an instant
is centred on the survivors' centroid: u_r along the centroid's position vector (radially out), u_l along the motion
(l > 0 ahead). A particle's release phase is its hand-off RF phase (release time x RF, relative to the bunch centre),
wrapped to +-180 deg; its phase lag is h x its azimuth behind the survivors' mean azimuth (+ = behind).
"""
import time
import warnings
from collections import namedtuple

import numpy as np

from ..config import MachineConfig, get_dotted
from .objectives import METRICS, ScoreConfig

EPS0 = 8.8541878128e-12          # vacuum permittivity [F/m]
CLIGHT = 299792458.0             # [m/s]
MAD_SIGMA = 1.4826               # sigma of a Gaussian / its median absolute deviation


def wrap_deg(a):
    """Angles [deg] wrapped to [-180, 180)."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def robust_sigma(x, axis=None):
    """Gaussian-equivalent sigma from the median absolute deviation: insensitive to tails and halo."""
    x = np.asarray(x, dtype=float)
    return MAD_SIGMA * np.median(np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis)


def _mean(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()) if v.size else float('nan')


def poly_fit_values(x, y, degree):
    """Least-squares polynomial of the given degree in x fitted to y (n,) or to each column of y (n, m): the fitted
    values. x is scaled to |x| <= 1 for the conditioning of the Vandermonde matrix."""
    x = np.asarray(x, dtype=float)
    V = np.vander(x / (np.max(np.abs(x)) or 1.0), degree + 1)
    return V @ np.linalg.lstsq(V, y, rcond=None)[0]


# ---------------------------------------------------------------------------------------------------------------------
# who is alive, who survives, when the others were lost
def ghost_first_step(ids, ghost_ids, ghost_step):
    """(N,) the first ghost step of every tracked particle, inf for the ones never ghosted. In a ghost run a particle
    that enters a would-be obstacle turns into a zero-charge test particle and is tracked on; scoring counts it as
    lost from that step."""
    out = np.full(len(ids), np.inf)
    if ghost_ids is None or len(ghost_ids) == 0 or len(ids) == 0:
        return out
    ids = np.asarray(ids)
    order = np.argsort(ids, kind='stable')
    pos = np.clip(np.searchsorted(ids[order], ghost_ids), 0, len(ids) - 1)
    hit = ids[order][pos] == np.asarray(ghost_ids)
    np.minimum.at(out, order[pos[hit]], np.asarray(ghost_step, dtype=float)[hit])
    return out


def alive_mask(steps, xyz, birth_step, stop_step, ghost_step):
    """(S, N): the record is a live position - after birth, before the metal hit and the ghost step, and finite."""
    s = np.asarray(steps)[:, None]
    return (s >= birth_step[None, :]) & (s < stop_step[None, :]) & (s < ghost_step[None, :]) \
        & np.all(np.isfinite(xyz), axis=2)


def survivors(xyz_last, stop_step, n_steps, ghost_step, r_max, z_max):
    """(N,): survived to the end of the run. The tracker records only metal hits (stop_step = n_steps + 1 when there
    was none); a particle lost at the radius or height limit keeps its slot where it left the volume, so the last
    record must also be finite and inside r < r_max, |z| < z_max. ``stop >= stop.max()`` would count those wall
    losses as survivors. Ghosted particles do not survive."""
    last = np.asarray(xyz_last)
    return (stop_step > n_steps) & np.isinf(ghost_step) & np.all(np.isfinite(last), axis=1) \
        & (np.abs(last[:, 2]) < z_max) & (np.hypot(last[:, 0], last[:, 1]) < r_max)


def rotation_sense(xyz, alive):
    """+1 when the beam circulates counter-clockwise seen from +z (sum of L_z at the first live records), else -1."""
    N = alive.shape[1]
    first = np.argmax(alive, axis=0)
    r0 = xyz[first, np.arange(N)]
    v0 = xyz[np.minimum(first + 1, len(xyz) - 1), np.arange(N)] - r0
    return 1.0 if np.nansum(r0[:, 0] * v0[:, 1] - r0[:, 1] * v0[:, 0]) >= 0 else -1.0


def loss_steps(steps, xyz, birth_step, stop_step, ghost_step, n_steps, surv, r_max, z_max):
    """(N,) the step at which each tracked particle was lost, inf for the survivors: the metal hit or ghost step, else
    the first record after birth outside the loss limits (r >= r_max, |z| >= z_max) or not finite."""
    loss = np.minimum(np.asarray(stop_step, dtype=float), ghost_step)
    loss[(loss > n_steps) | surv] = np.inf
    wall = np.flatnonzero(np.isinf(loss) & ~surv)
    if len(wall):
        steps = np.asarray(steps)
        p = xyz[:, wall]
        out = ~np.all(np.isfinite(p), axis=2) | (np.hypot(p[..., 0], p[..., 1]) >= r_max) | (np.abs(p[..., 2]) >= z_max)
        out &= steps[:, None] >= np.asarray(birth_step)[wall][None, :]
        loss[wall] = np.where(out.any(axis=0), steps[np.argmax(out, axis=0)], steps[-1])
    return loss


def loss_turn_shares(lost_turn, edges):
    """Share of the particles lost in each class of the turn of loss (turns after release): [0, e0), [e0, e1), ...,
    [e_last, inf). lost_turn is NaN for the survivors."""
    b = np.r_[0.0, np.asarray(edges, dtype=float), np.inf]
    return [float(np.mean((lost_turn >= lo) & (lost_turn < hi))) for lo, hi in zip(b[:-1], b[1:])]


# ---------------------------------------------------------------------------------------------------------------------
# energy
def energy_window_median(e, rel):
    """Particles within +-rel of the median energy."""
    e = np.asarray(e, dtype=float)
    med = np.median(e)
    return int(np.sum(np.abs(e - med) <= rel * med))


def energy_window_best(e, rel):
    """The best-placed +-rel window: the most particles any window [c (1 - rel), c (1 + rel)] holds, and its centre c.
    An optimal window can be moved up until its lower edge sits on a particle without losing any, so trying every
    energy as the lower edge is exact (the design deck's scan of c on a 4000-point grid finds at most this). Unlike
    the median-centred window it does not fall into the trough of a two-peaked spectrum."""
    e = np.sort(np.asarray(e, dtype=float))
    if e.size == 0:
        return 0, float('nan')
    lo = np.searchsorted(e, e, 'left')
    hi = np.searchsorted(e, e * (1 + rel) / (1 - rel), 'right')
    k = int(np.argmax(hi - lo))
    c = 0.5 * (e[hi[k] - 1] / (1 + rel) + e[lo[k]] / (1 - rel))      # middle of the centres that hold these particles
    return int(hi[k] - lo[k]), float(c)


def energy_metrics(e, n_tracks, i0, last_xyz, windows):
    """Final energies e (n,) [MeV] of the survivors and their last records (n, 3) [m]: E_med, E_mean, E_min, dE_E (rms)
    and dE_E_robust (MAD) relative spreads, R_last [mm] the mean radius at the last record and sigR_E = R_last / 2 x
    dE_E_robust, the radial width the energy spread alone makes (R ~ p ~ sqrt(E)). Per window rel: the current within
    +-rel of the median, I_<p>pct, and in the best-placed window, I_<p>pct_best (centre E_<p>pct_best), as i0 x the
    count / n_tracks (p = 100 rel)."""
    med = float(np.median(e))
    R_last = float(np.mean(np.hypot(last_xyz[:, 0], last_xyz[:, 1])) * 1e3)
    dEr = float(robust_sigma(e) / med)
    out = {'E_med': med, 'E_mean': float(e.mean()), 'E_min': float(e.min()), 'dE_E': float(e.std() / e.mean()),
           'dE_E_robust': dEr, 'R_last': R_last, 'sigR_E': 0.5 * R_last * dEr}
    for rel in windows:
        p = f"{100 * rel:g}"
        n_best, c_best = energy_window_best(e, rel)
        out[f'I_{p}pct'] = float(i0 * energy_window_median(e, rel) / n_tracks)
        out[f'I_{p}pct_best'] = float(i0 * n_best / n_tracks)
        out[f'E_{p}pct_best'] = c_best
    return out


def radial_width_split(r_mm, E, lag):
    """The survivors' radius r [mm] at the last record split into the part their final energy explains (linear fit
    r(E), slope dR_dE [mm/MeV]) and the rest (orbit centres and betatron amplitudes), rms each; and the correlation of
    r and of E with the final phase lag (an r-l tilt of the bunch)."""
    b = np.polyfit(E, r_mm, 1)
    r_E = np.polyval(b, E)
    return {'r_rms_last': float(r_mm.std()), 'r_rms_E': float(r_E.std()), 'r_rms_rest': float((r_mm - r_E).std()),
            'dR_dE': float(b[0]), 'corr_r_lag': float(np.corrcoef(r_mm, lag)[0, 1]),
            'corr_E_lag': float(np.corrcoef(E, lag)[0, 1])}


# ---------------------------------------------------------------------------------------------------------------------
# acceptance and compression in hand-off release phase
def phase_bins(ph, bin_deg, range_deg):
    """Release-phase bin edges spanning +-range_deg and the bin of every phase (outside phases go to the end bins)."""
    n = int(round(2 * range_deg / bin_deg))
    edges = -range_deg + bin_deg * np.arange(n + 1)
    return edges, np.clip(np.digitize(ph, edges) - 1, 0, n - 1)


def _longest_run(ok):
    """(i0, i1) of the longest run of consecutive True (the first of equal ones); (0, 0) when there is none."""
    best, start = (0, 0), None
    for i, o in enumerate(np.r_[ok, False]):
        if o and start is None:
            start = i
        elif not o and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def acceptance(ph, surv, ph_all, cfg):
    """Survival against release phase ph (N,) of the tracked particles: survival per bin (survival_curve, bins with
    >= min_bin_count particles); per level L the widest window of consecutive bins with survival >= L (acc<L>_width,
    _lo, _hi [deg]) and the share of the whole hand-off bunch ph_all inside it (acc<L>_ceiling: the most such a window
    can transmit); W_eff, the survival curve integrated over phase (the width of an equivalent 100 % flat top); and
    the survival in bands of |phase| (S30 = |phase| <= 30, S30_40 = 30 < |phase| <= 40 for bands (30, 40))."""
    edges, b = phase_bins(ph, cfg.phase_bin_deg, cfg.phase_range_deg)
    nb = np.bincount(b, minlength=len(edges) - 1)
    sb = np.bincount(b, weights=surv, minlength=len(edges) - 1) / np.maximum(nb, 1)
    full = nb >= cfg.min_bin_count
    out = {}
    for lev in cfg.acceptance_levels:
        i0, i1 = _longest_run(full & (sb >= lev))
        lo, hi = edges[i0], edges[i1]
        t = f"{int(round(100 * lev))}"
        out.update({f'acc{t}_width': float(hi - lo), f'acc{t}_lo': float(lo) if hi > lo else None,
                    f'acc{t}_hi': float(hi) if hi > lo else None,
                    f'acc{t}_ceiling': float(np.mean((ph_all >= lo) & (ph_all < hi))) if hi > lo else 0.0})
    out['W_eff'] = float(np.sum(np.where(full, sb, 0.0)) * cfg.phase_bin_deg)
    centres = 0.5 * (edges[1:] + edges[:-1])
    out['survival_curve'] = {f"{c:g}": float(s) for c, s, f in zip(centres, sb, full) if f}
    lo = None
    for hi in cfg.survival_bands_deg:
        a = np.abs(ph)
        m = (a <= hi) if lo is None else (a > lo) & (a <= hi)
        out['S' + (f"{hi:g}" if lo is None else f"{lo:g}_{hi:g}")] = _mean(surv[m])
        lo = hi
    return out


def final_phase_lag(p1, p0, sel, harmonic):
    """(N,) final phase lag [RF deg, + = behind] at the last record p1 (N, 3): h x the azimuth behind the circular-mean
    azimuth of the particles in sel; the direction of motion from that mean one record earlier (p0)."""
    th1, th0 = np.arctan2(p1[:, 1], p1[:, 0]), np.arctan2(p0[:, 1], p0[:, 0])
    c1, c0 = np.angle(np.mean(np.exp(1j * th1[sel]))), np.angle(np.mean(np.exp(1j * th0[sel])))
    return -harmonic * np.degrees(np.angle(np.exp(1j * (th1 - c1)))) * np.sign(np.angle(np.exp(1j * (c1 - c0))))


def turn_lag(X, harmonic, sense):
    """(n,) phase lag [RF deg, + = behind] of n particles over one turn: X (K, n, 2+) [any length unit] their
    positions at K instants spread over the turn; at each instant h x the azimuth behind the particles' circular-mean
    azimuth, averaged over the instants. The turn average removes the once-per-turn azimuth wobble an orbit-centre
    offset makes, so the lag is the particle's RF phase relative to the bunch."""
    th = np.arctan2(X[..., 1], X[..., 0])
    c = np.angle(np.mean(np.exp(1j * th), axis=1))
    return -harmonic * sense * np.degrees(np.angle(np.exp(1j * (th - c[:, None])))).mean(axis=0)


def compression_slope(ph, lag, clip_sigma=2.5):
    """d(lag) / d(release phase) of the survivors, a straight line through those within clip_sigma of the median
    phase: 1 = phase conserved, < 1 = the bunch compressed in phase."""
    core = np.abs(ph - np.median(ph)) < clip_sigma * ph.std()
    return float(np.polyfit(ph[core], lag[core], 1)[0])


def release_budget(ph, sel, E, lag, lost_turn, cfg):
    """What becomes of each release phase ph (N,) of the tracked particles. release_bins: per bin with >=
    min_bin_count of them (phases beyond +-phase_range_deg are left out, not put in the end bins), its share of them,
    the share of the bin lost in each loss-turn class (loss_turn_edges; lost_turn NaN for the survivors, so the bin's
    survival is 1 - sum(lost)) and, when at least min_bin_survivors of its survivors are in sel, their mean and rms
    final energy and phase lag. E_peak_phase, E_peak: the maximum of a quadratic E(release) fitted to the core
    survivors in sel (phase conserved through the gaps makes E(release) a cosine; NaN when the fit has no maximum)."""
    edges, b = phase_bins(ph, cfg.phase_bin_deg, cfg.phase_range_deg)
    inside = (ph >= edges[0]) & (ph < edges[-1])
    tb = np.r_[0.0, np.asarray(cfg.loss_turn_edges, dtype=float), np.inf]
    rows = {}
    for i in range(len(edges) - 1):
        m = (b == i) & inside
        if m.sum() < cfg.min_bin_count:
            continue
        ms = m & sel
        row = {'n': int(m.sum()), 'share': float(m.mean()),
               'lost': [float(np.mean((lost_turn[m] >= lo) & (lost_turn[m] < hi))) for lo, hi in zip(tb[:-1], tb[1:])]}
        if ms.sum() >= cfg.min_bin_survivors:
            row.update(E_mean=float(E[ms].mean()), E_std=float(E[ms].std()), lag_mean=float(lag[ms].mean()),
                       lag_std=float(lag[ms].std()))
        rows[f"{0.5 * (edges[i] + edges[i + 1]):g}"] = row
    ps, es = ph[sel], E[sel]
    core = np.abs(ps - np.median(ps)) < cfg.compression_clip_sigma * ps.std()
    k = np.polyfit(ps[core], es[core], 2)
    peak = -k[1] / (2 * k[0]) if k[0] < 0 else np.nan
    return {'release_bins': rows, 'E_peak_phase': float(peak),
            'E_peak': float(np.polyval(k, peak)) if np.isfinite(peak) else float('nan')}


def handoff_survival(ph, z_mm, zp_mrad, surv, cfg):
    """Is the loss phase acceptance or vertical? band<q>_lo/_hi: the central release-phase band holding the share q of
    the survivors, band<q>_share the share of the tracked bunch inside it and band<q>_survival the survival there;
    surv_vs_z / surv_vs_zp: survival per quantile bin of the hand-off |z| [mm] and |z'| [mrad]."""
    q = cfg.phase_band_q
    lo, hi = np.quantile(ph[surv], [(1 - q) / 2, (1 + q) / 2])
    m = (ph >= lo) & (ph <= hi)
    t = f"{int(round(100 * q))}"
    out = {f'band{t}_lo': float(lo), f'band{t}_hi': float(hi), f'band{t}_share': float(m.mean()),
           f'band{t}_survival': _mean(surv[m])}
    for name, v in (('z', np.abs(z_mm)), ('zp', np.abs(zp_mrad))):
        e = np.quantile(v, np.linspace(0.0, 1.0, cfg.handoff_quantiles + 1))
        out[f'surv_vs_{name}'] = {'edges': e.tolist(),
                                  'survival': [_mean(surv[(v >= a) & (v <= b)]) for a, b in zip(e[:-1], e[1:])]}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# bunch shape in the bunch frame
def core_variance_factor(q):
    """Variance inside the Mahalanobis ellipse that holds the share q of a 2D Gaussian over its full variance,
    (1 - (1 - q)(1 - ln(1 - q))) / q (0.8423 for q = 0.95): the squared Mahalanobis radius is chi^2 with 2 degrees of
    freedom. A trimmed core covariance divided by it is the covariance of the Gaussian the core belongs to."""
    q = float(q)
    return (1.0 - (1.0 - q) * (1.0 - np.log(1.0 - q))) / q


def bunch_coordinates(P, sense):
    """P (K, n, 2) [mm]: n particles at K instants -> u (K, n, 2), their bunch-frame coordinates at each instant's
    centroid: u_r along the centroid's position vector, u_l along the motion of sense (+1 counter-clockwise)."""
    c = P.mean(axis=1)
    er = c / np.linalg.norm(c, axis=1)[:, None]
    el = sense * np.column_stack([-er[:, 1], er[:, 0]])
    d = P - c[:, None, :]
    return np.stack([np.einsum('knj,kj->kn', d, er), np.einsum('knj,kj->kn', d, el)], axis=2)


def core_covariance(u, q, about_core=False):
    """The core of a bunch at K instants, u (K, n, 2) centred on the centroid: Cc (K, 2, 2), the covariance of the
    share q of the particles closest in Mahalanobis distance w.r.t. the full covariance, so a halo does not inflate
    it, about the centroid (the bunch frame's origin) or with about_core about the core's own mean (the guiding-centre
    twin's convention: a skewed tail then does not widen the core); and d2 (K, n), every particle's squared
    Mahalanobis distance from the centroid w.r.t. the Gaussian-equivalent core covariance Cc / core_variance_factor(q),
    i.e. in sigmas of the core (d2 > 9: beyond 3 sigma)."""
    C = np.einsum('kni,knj->kij', u, u) / u.shape[1]
    m2 = np.einsum('kni,kij,knj->kn', u, np.linalg.inv(C), u)
    w = (m2 <= np.quantile(m2, q, axis=1)[:, None]).astype(float)
    v = u - (np.einsum('kn,kni->ki', w, u) / w.sum(axis=1)[:, None])[:, None, :] if about_core else u
    Cc = np.einsum('kn,kni,knj->kij', w, v, v) / w.sum(axis=1)[:, None, None]
    return Cc, np.einsum('kni,kij,knj->kn', u, np.linalg.inv(Cc / core_variance_factor(q)), u)


def shape_stats(Cc, d2, halo_sigma=3.0):
    """Core shape over K instants: ratio, the aspect ratio sigma_major / sigma_minor as the exponential of the mean of
    ln sqrt(l_2 / l_1) (eigenvalues l_1 <= l_2: a tilted ellipse is not round; the mean log is the post-gap
    optimizer's roundness R, so ratio = exp(R)); sig_r, sig_l and the principal sig_1 <= sig_2 [mm]; tilt [deg] of the
    major axis from r (circular mean of the doubled angle, l > 0 ahead); s_round = (l_1 l_2)^(1/4), the r-l size a
    pure rotation conserves; and halo<s>, the share beyond halo_sigma of the Gaussian-equivalent core."""
    lam = np.linalg.eigvalsh(Cc)
    t2 = np.arctan2(2 * Cc[:, 0, 1], Cc[:, 0, 0] - Cc[:, 1, 1])
    return {'ratio': float(np.exp(np.mean(0.5 * np.log(lam[:, 1] / lam[:, 0])))),
            'sig_r': float(np.sqrt(Cc[:, 0, 0]).mean()), 'sig_l': float(np.sqrt(Cc[:, 1, 1]).mean()),
            'sig_1': float(np.sqrt(lam[:, 0]).mean()), 'sig_2': float(np.sqrt(lam[:, 1]).mean()),
            'tilt': float(0.5 * np.degrees(np.angle(np.mean(np.exp(1j * t2))))),
            's_round': float(np.mean((lam[:, 0] * lam[:, 1]) ** 0.25)),
            f'halo{halo_sigma:g}': float(np.mean(d2 > halo_sigma ** 2))}


def roundness(P, sense=1.0, q=0.95):
    """The post-gap optimizer's roundness of a bunch at K instants, P (K, n, 2) [mm]: R = the mean over the instants
    of ln sqrt(l_2 / l_1) of the q core in the bunch frame, ratio = exp(R), sig_r, sig_l [mm] (shape_stats on the same
    core, so the optimizer and the scores share one implementation)."""
    u = bunch_coordinates(np.asarray(P, dtype=float), sense)
    st = shape_stats(*core_covariance(u, q))
    return {'R': float(np.log(st['ratio'])), 'ratio': st['ratio'], 'sig_r': st['sig_r'], 'sig_l': st['sig_l']}


def ratio_without_fan(u, ph, q, degree):
    """The core aspect ratio (as shape_stats) after removing from u_r the part a polynomial in release phase ph (n,)
    explains, and the share of the u_r variance that part holds (mean over the K instants of u (K, n, 2)): how round
    the bunch would be without the orbit-centre fan of phase-dependent energy gains."""
    fit = poly_fit_values(ph, u[..., 0].T, degree).T
    v = u.copy()
    v[..., 0] -= fit - fit.mean(axis=1, keepdims=True)
    lam = np.linalg.eigvalsh(core_covariance(v, q)[0])
    return (float(np.exp(np.mean(0.5 * np.log(lam[:, 1] / lam[:, 0])))),
            float(np.mean(np.var(fit, axis=1) / np.var(u[..., 0], axis=1))))


# ---------------------------------------------------------------------------------------------------------------------
# vortex: measured one-turn map of the core against the bunch's own field
def gauss_gradients(s1, s2, s3):
    """E_i / x_i [V/m^2 per C] at the centre of a 3D Gaussian bunch of rms sizes s1, s2, s3 [m], for i = 1, 2:
    (1 / 4 pi^1.5 eps0) int_0^inf 2 dt / ((a_i + t) prod_j (a_j + t)^(1/2)), a = 2 s^2, which is Carlson's elliptic
    integral R_D(a_j, a_k, a_i) / (3 pi^1.5 eps0)."""
    from scipy.special import elliprd
    a = 2.0 * np.array([s1, s2, s3], dtype=float) ** 2
    k = 3.0 * np.pi ** 1.5 * EPS0
    return [float(elliprd(a[1], a[2], a[0]) / k), float(elliprd(a[0], a[2], a[1]) / k)]


def vortex_prediction(Cc, sig_z_mm, charge_c, b0_t, f_rev_hz, q):
    """The one-turn map of the core that the bunch's own field predicts. A Gaussian bunch with the core's
    Gaussian-equivalent sizes (Cc / core_variance_factor(q), mean over the instants [mm]; sig_z [mm]) and charge
    charge_c has the linear field E = G u at its core, G the gradient tensor turned from the principal axes to
    (r, l). In the bunch frame of a cyclotron a tangential field changes a particle's energy and so its orbit radius,
    a radial field its angular velocity: dr/dt = +E_l / B, dl/dt = -E_r / B (leading particles move out, outer ones
    fall behind). Over a revolution T = 1 / f_rev: d u_r = b u_l with b_pred = G_ll T / B, d u_l = c u_r with
    c_pred = -G_rr T / B, and the core turns by rot_pred = -sqrt(G_1 G_2) T / B [deg]: a vortex turns the bunch
    clockwise in (r, l). It is the CENTRAL gradient: an index of the vortex strength, above what the core feels."""
    w, V = np.linalg.eigh((Cc / core_variance_factor(q)).mean(axis=0))
    g1, g2 = gauss_gradients(np.sqrt(w[0]) * 1e-3, np.sqrt(w[1]) * 1e-3, sig_z_mm * 1e-3)
    G = charge_c * (V @ np.diag([g1, g2]) @ V.T)
    k = 1.0 / b0_t / f_rev_hz
    return {'b_pred': float(G[1, 1] * k), 'c_pred': float(-G[0, 0] * k),
            'rot_pred': float(-np.degrees(charge_c * np.sqrt(g1 * g2) * k))}


def map_rotation(M):
    """Rotation angle [deg] of a linear 2x2 map in (r, l): atan2(M_lr - M_rl, tr M) (a vortex gives < 0)."""
    return float(np.degrees(np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])))


def one_turn_map(u0, u1, sel):
    """The measured linear one-turn map of the core, u' = M u: u0, u1 (K, n, 2) the same particles at the same K
    instants of two consecutive turns, least squares over the selected (K, n) entries. Returns M, b_meas = M_rl
    (d u_r per u_l), c_meas = M_lr and its rotation rot_meas (map_rotation). In the instantaneous bunch frame the
    orbit-centre offsets dilute the vortex and their space-charge-shifted precession cancels the rest, so this map
    does not see it; ``twins.vortex_twin`` measures it in guiding-centre coordinates."""
    M = np.linalg.lstsq(u0[sel], u1[sel], rcond=None)[0].T
    return {'M': M.tolist(), 'b_meas': float(M[0, 1]), 'c_meas': float(M[1, 0]), 'rot_meas': map_rotation(M)}


def isochronous_field(f_rev_hz, mass_mev, charge_state=1.0):
    """The central field of an isochronous cyclotron, 2 pi f_rev m / q [T] (mass [MeV / c^2], charge state Z)."""
    if mass_mev is None or not np.isfinite(f_rev_hz):
        return float('nan')
    return float(2.0 * np.pi * f_rev_hz * float(mass_mev) * 1e6 / (CLIGHT ** 2 * float(charge_state)))


# ---------------------------------------------------------------------------------------------------------------------
# per-particle orbit centres
def turn_centres(xy, alive, sense, fill=0.8):
    """Orbit centre of every particle in every complete revolution counted from its own start: per revolution the
    least-squares fit r(theta) = a + b (theta - theta_mid) + cx cos(theta) + cy sin(theta) of its live records (the
    radius grows by b per radian as it gains energy). xy (S, n, 2) [m], alive (S, n); a revolution needs fill x the
    records of the fullest one. Returns (n, K, 2) [mm], NaN where a revolution is incomplete."""
    rec, n = alive.shape
    first = np.argmax(alive, axis=0)
    ang = np.where(alive, np.arctan2(xy[..., 1], xy[..., 0]), np.nan)
    pre = np.arange(rec)[:, None] < first[None, :]
    ang = np.where(pre, ang[first, np.arange(n)][None, :], ang)
    ang = np.where(np.isnan(ang), 0.0, ang)                                  # dead records are masked below
    th = np.maximum.accumulate(sense * np.unwrap(ang, axis=0), axis=0)
    rel = th - th[first, np.arange(n)][None, :]
    k = np.floor(rel / (2 * np.pi)).astype(np.int64)
    ok = alive & ~pre & (k >= 0)
    K = int(k[ok].max()) + 1
    g = (np.arange(n)[None, :] * K + k)[ok]
    phys = sense * th[ok]
    cols = [np.ones(g.size), rel[ok] - 2 * np.pi * (k[ok] + 0.5), np.cos(phys), np.sin(phys)]
    r = np.hypot(xy[..., 0], xy[..., 1])[ok]
    cnt = np.bincount(g, minlength=n * K)
    ATA = np.empty((n * K, 4, 4))
    for i in range(4):
        for j in range(i, 4):
            ATA[:, i, j] = ATA[:, j, i] = np.bincount(g, weights=cols[i] * cols[j], minlength=n * K)
    ATr = np.stack([np.bincount(g, weights=cols[i] * r, minlength=n * K) for i in range(4)], axis=1)
    full = cnt >= fill * cnt.max()
    sol = np.full((n * K, 4), np.nan)
    sol[full] = np.linalg.solve(ATA[full], ATr[full][..., None])[..., 0]
    return sol[:, 2:].reshape(n, K, 2) * 1e3


def centre_fan(cen, ph=None, degree=3, fill=0.8):
    """Per revolution n = 1, 2, ... of the per-particle orbit centres cen (n, K, 2) [mm] of ``turn_centres``, which
    counts the revolutions from each particle's own start: revolution n of a bunch released around step 0 is about
    the time-turn n - 1 of score_run's 'turns' (so fan key _t3 is a turn earlier than ratio_t3). Per revolution: coh,
    the coherent offset |mean centre|, and coh_xy, the mean centre (x, y); spread, the rms spread about it; c_med, the
    median |centre|; proj and proj_max, the rms of the spread projected on the radial direction, averaged over the
    azimuths (spread / sqrt 2) and at the worst azimuth (the major axis of the fan ellipse): an orbit-centre offset c
    moves a particle's radius at azimuth theta by c . e_r(theta), so this is what the fan adds to the turn width at a
    probe. With the release phases ph (n,): expl_share, the share of the spread a polynomial of the given degree in
    release phase explains (the 'fan' that phase-dependent energy gains in the first gaps make), and early_late, the
    distance between the mean centres of the earliest and the latest third. Turns where fewer than fill x n particles
    have a centre are left out."""
    fan = {}
    for t in range(cen.shape[1]):
        c = cen[:, t]
        okc = np.all(np.isfinite(c), axis=1)
        if okc.sum() < fill * len(cen):
            continue
        c = c[okc]
        lam = np.linalg.eigvalsh(np.cov(c.T, bias=True))
        tot = float(lam.sum())
        cm = c.mean(axis=0)
        row = {'coh': float(np.hypot(*cm)), 'coh_xy': cm.tolist(), 'spread': float(np.sqrt(tot)),
               'c_med': float(np.median(np.hypot(c[:, 0], c[:, 1]))), 'proj': float(np.sqrt(tot / 2)),
               'proj_max': float(np.sqrt(lam[-1]))}
        if ph is not None:
            pp = ph[okc]
            fit = poly_fit_values(pp, c, degree)
            q3 = np.quantile(pp, [1 / 3, 2 / 3])
            row.update(expl_share=float(np.sum(np.var(fit, axis=0)) / tot),
                       early_late=float(np.hypot(*(c[pp >= q3[1]].mean(axis=0) - c[pp <= q3[0]].mean(axis=0)))))
        fan[t + 1] = row
    return fan


# ---------------------------------------------------------------------------------------------------------------------
# virtual probes and turn structure
def probe_crossings(xyz, alive, steps, dt_step, sense, angle_deg, forward_only=True, t0=0.0):
    """Crossings of the vertical half-plane at azimuth angle_deg, with the tracker's probe rules applied to recorded
    tracks: a crossing is a sign change of the signed distance between two live records, its position linearly
    interpolated; the pass number counts from the particle's own start (pass 1 when it starts already past the probe)
    and every crossing, forward or back, increments it. Returns j (track index), pass, R, z [mm], t [s] (t0 + the time
    since step 0) of the forward crossings; with forward_only False of all crossings, plus direction (+1 / -1), xyz [m],
    v [m/s] (chord over the record interval) and step (the record after the crossing)."""
    th = np.deg2rad(angle_deg)
    nrm, along = np.array([-np.sin(th), np.cos(th)]), np.array([np.cos(th), np.sin(th)])
    s = xyz[..., 0] * nrm[0] + xyz[..., 1] * nrm[1]
    N = alive.shape[1]
    first = np.argmax(alive, axis=0)
    r0 = xyz[first, np.arange(N), :2]
    passes = (sense * np.angle(np.exp(1j * (np.arctan2(r0[:, 1], r0[:, 0]) - th))) > 0.0).astype(np.int32)
    k, j = np.nonzero(alive[:-1] & alive[1:] & ((s[:-1] < 0.0) != (s[1:] < 0.0)))
    f = s[k, j] / (s[k, j] - s[k + 1, j])
    rc = xyz[k, j] + f[:, None] * (xyz[k + 1, j] - xyz[k, j])
    keep = rc[:, 0] * along[0] + rc[:, 1] * along[1] >= 0.0
    k, j, f, rc = k[keep], j[keep], f[keep], rc[keep]
    o = np.lexsort((k, j))
    k, j, f, rc = k[o], j[o], f[o], rc[o]
    start = np.r_[0, np.flatnonzero(np.diff(j)) + 1]
    rank = np.arange(len(j)) - np.repeat(start, np.diff(np.r_[start, len(j)]))
    direction = np.where(sense * s[k + 1, j] >= 0.0, 1, -1).astype(np.int8)
    ds = np.diff(steps)[k]
    out = {'j': j, 'pass': passes[j] + rank, 'R': 1e3 * np.hypot(rc[:, 0], rc[:, 1]), 'z': 1e3 * rc[:, 2],
           't': t0 + (steps[k] + f * ds) * dt_step}
    if forward_only:
        fw = direction > 0
        return {key: v[fw] for key, v in out.items()}
    out.update(direction=direction, xyz=rc, v=(xyz[k + 1, j] - xyz[k, j]) / (ds * dt_step)[:, None],
               step=np.asarray(steps)[k + 1])
    return out


def reference_crossings(ref_xyz, angle_deg, sense):
    """{pass: R [mm]} of the forward crossings of the half-plane at angle_deg by one orbit ref_xyz (S, 3+) [m] (the
    reference particle's, recorded every step), with the probes' pass numbering: what the tracker's [slit] centres a
    slit on."""
    xyz = np.asarray(ref_xyz, dtype=float)[:, None, :3]
    pr = probe_crossings(xyz, np.all(np.isfinite(xyz), axis=2), np.arange(len(xyz)), 1.0, sense, angle_deg)
    return {int(p): float(R) for p, R in zip(pr['pass'], pr['R'])}


def turn_structure(pr, mask, n_need, cfg):
    """Turn structure at one probe from the forward crossings pr (probe_crossings) of the particles in mask: the last
    probe_passes complete passes (crossed by >= pass_complete x n_need of them, consecutive). Per pass the centre R
    (median) and the robust sigma [mm]; dR between neighbours; sep, the mean separation 2 dR / (sigma_a + sigma_b);
    valley, the share of the crossings inside the window (half a separation beyond the outer passes) that fall in the
    middle third between two turn centres; vp, the valley / peak ratio of the radial histogram (hist_bin_mm, a
    hist_smooth_bins moving average); halo, the share of a pass beyond probe_halo_sigma of its own centre. j_w and
    mid_w give the crossings in the window (track index, in a valley) for the all-probe valley share. None when there
    are not enough complete passes."""
    m = mask[pr['j']]
    j, p, R = pr['j'][m], pr['pass'][m], pr['R'][m]
    npass = int(cfg.probe_passes)
    if len(p) == 0 or npass < 2:
        return None
    cnt = np.bincount(p - p.min())
    comp = np.flatnonzero(cnt >= cfg.pass_complete * n_need) + p.min()
    if len(comp) < npass or comp[-1] - comp[-npass] != npass - 1:
        return None
    P = comp[-npass:]
    Rc = np.array([np.median(R[p == q]) for q in P])
    sg = np.array([robust_sigma(R[p == q]) for q in P])
    dR = np.diff(Rc)
    lo, hi = Rc[0] - dR[0] / 2, Rc[-1] + dR[-1] / 2
    w = (R >= lo) & (R <= hi)
    mid = np.zeros(len(R), bool)
    for g in range(npass - 1):
        mid |= (R > Rc[g] + dR[g] / 3) & (R < Rc[g + 1] - dR[g] / 3)
    bw = cfg.hist_bin_mm
    hist, e = np.histogram(R[w], bins=np.arange(np.floor(lo / bw) * bw, np.ceil(hi / bw) * bw + 0.5 * bw, bw))
    hs = np.convolve(hist, np.ones(cfg.hist_smooth_bins) / cfg.hist_smooth_bins, mode='same')
    cen = 0.5 * (e[1:] + e[:-1])

    def extreme(sel, fn):
        return fn(hs[sel]) if np.any(sel) else np.nan

    pk = [extreme(np.abs(cen - c) <= max(s, bw), np.max) for c, s in zip(Rc, sg)]
    vp = [extreme((cen > Rc[g] + dR[g] / 3) & (cen < Rc[g + 1] - dR[g] / 3), np.min) / (0.5 * (pk[g] + pk[g + 1]))
          for g in range(npass - 1)]
    own = np.isin(p, P)
    i_own = np.searchsorted(P, p[own])
    halo = np.mean(np.abs(R[own] - Rc[i_own]) > cfg.probe_halo_sigma * sg[i_own])
    return {'passes': P.tolist(), 'R': Rc.tolist(), 'sigma': sg.tolist(), 'dR': dR.tolist(),
            'sep': float(np.mean(2 * dR / (sg[:-1] + sg[1:]))), 'valley': float(np.mean(mid[w])),
            'vp': float(np.mean(vp)), 'halo': float(halo), 'j_w': j[w], 'mid_w': mid[w]}


def recorded_probe_turns(pr, cfg):
    """The turn structure (turn_structure) at a probe the tracker recorded for every particle (runs.load_probe), over
    its own survivors: the check of the virtual probe at that azimuth, which sees only the tracked sample."""
    s = pr['survived']
    ts = turn_structure(pr, s, int(s.sum()), cfg)
    return None if ts is None else {k: v for k, v in ts.items() if k not in ('j_w', 'mid_w')}


def compare_probes(virtual, recorded):
    """A virtual probe (runs.virtual_probe) against the tracker's probe file at the same azimuth (dict(np.load(f))),
    particle by particle: the forward crossings with the same bunch particle and pass. Returns n_matched, n_virtual
    and over the matched crossings dR_median, dR_max, dz_max [mm] and dE_E_median, dE_E_q99 (a virtual crossing's
    energy is that of the chord velocity between two records, so dE/E measures the record interval, not an error)."""
    fv, fr = np.asarray(virtual['direction']) > 0, np.asarray(recorded['direction']) > 0
    pid = np.asarray(virtual['particle_ids'], dtype=np.int64)[np.asarray(virtual['id'])[fv]]
    tv, tr = np.asarray(virtual['turn'])[fv].astype(np.int64), np.asarray(recorded['turn'])[fr].astype(np.int64)
    span = int(max(tv.max(initial=0), tr.max(initial=0))) + 2
    _, iv, ir = np.intersect1d(pid * span + tv + 1, np.asarray(recorded['id'])[fr].astype(np.int64) * span + tr + 1,
                               return_indices=True)     # a particle's passes are distinct: the keys are unique
    if not len(iv):
        return {'n_matched': 0, 'n_virtual': int(fv.sum())}
    xv, xr = np.asarray(virtual['xyz'])[fv][iv], np.asarray(recorded['xyz'])[fr][ir]
    dR = 1e3 * np.abs(np.hypot(xv[:, 0], xv[:, 1]) - np.hypot(xr[:, 0], xr[:, 1]))
    dE = np.asarray(virtual['E_mev'])[fv][iv] / np.asarray(recorded['E_mev'])[fr][ir] - 1.0
    return {'n_matched': int(len(iv)), 'n_virtual': int(fv.sum()), 'dR_median': float(np.median(dR)),
            'dR_max': float(dR.max()), 'dz_max': float(1e3 * np.max(np.abs(xv[:, 2] - xr[:, 2]))),
            'dE_E_median': float(np.median(dE)), 'dE_E_q99': float(np.quantile(np.abs(dE), 0.99))}


def slit_tradeoff(prs, surv, v, n, cfg):
    """Per budget b (slit_budgets: the share of the survivors a collimator takes) the single two-sided radial slit and
    the single vertical slit (|z| cut) at one of slit_passes at any probe that leaves the smallest final valley share
    sum(v) / sum(n) over the kept survivors - v, n (N,) a survivor's valley and window crossings at all probes. Turn
    centres are held fixed and there is no space-charge feedback. A radial budget is split between the inner and the
    outer jaw in slit_splits ways; a pass needs slit_min_share of the survivors to carry a slit. A radial slit: az,
    pass, its opening R_lo..R_hi [mm], taken, and jaw_mm, how far the cut crossings reach beyond the opening (the jaw
    length that stops all of them)."""
    V, Nn, n_surv = v.sum(), n.sum(), surv.sum()
    out = {}
    for b in cfg.slit_budgets:
        rad, ver = (V / Nn, None), (V / Nn, None)
        k = int(round(b * n_surv))
        for ang, pr in prs.items():
            for p in cfg.slit_passes:
                m = (pr['pass'] == p) & surv[pr['j']]
                if m.sum() < cfg.slit_min_share * n_surv:
                    continue
                j, R, z = pr['j'][m], pr['R'][m], np.abs(pr['z'][m])
                Rs = np.sort(R)
                for a in np.linspace(0.0, 1.0, cfg.slit_splits):
                    k_lo = int(round(a * k))
                    lo, hi = Rs[min(k_lo, len(Rs) - 1)], Rs[max(len(Rs) - 1 - (k - k_lo), 0)]
                    cut = j[(R < lo) | (R > hi)]
                    val = (V - v[cut].sum()) / (Nn - n[cut].sum())
                    if val < rad[0]:
                        jaw = max(lo - Rs[0], Rs[-1] - hi, 0.0)
                        rad = (val, {'az': float(ang), 'pass': int(p), 'R_lo': float(lo), 'R_hi': float(hi),
                                     'taken': float(len(cut) / n_surv), 'jaw_mm': float(jaw)})
                thr = np.sort(z)[max(len(z) - 1 - k, 0)]
                cut = j[z > thr]
                val = (V - v[cut].sum()) / (Nn - n[cut].sum())
                if val < ver[0]:
                    ver = (val, {'az': float(ang), 'pass': int(p), 'z_max': float(thr),
                                 'taken': float(len(cut) / n_surv)})
        out[f"{b:g}"] = {'radial': {'valley': float(rad[0]), **(rad[1] or {})},
                         'vertical': {'valley': float(ver[0]), **(ver[1] or {})}}
    return out


def slit_settings(radial, ref_R=None):
    """A best radial slit of slit_tradeoff in the conventions of the tracker's [slit] section (config.SlitConfig):
    azimuth_deg, pass_index, offset_mm (the opening's centre minus the reference particle's radius at that crossing,
    ref_R [mm]), half_width_mm, r_center_mm and jaw_length_mm, the jaw that stops every cut crossing of that pass (the
    trade-off sees one pass only: a jaw reaching a neighbouring turn cuts it too, which the tracker's slit check
    reports). Without the reference radius the centre is absolute: pass_index -1 with r_center_mm. None when no slit
    lowers the valley share. A seed for a slit campaign, not a design: centres held, no space-charge feedback."""
    if 'az' not in radial:
        return None
    rc, hw = 0.5 * (radial['R_lo'] + radial['R_hi']), 0.5 * (radial['R_hi'] - radial['R_lo'])
    return {'azimuth_deg': radial['az'], 'pass_index': radial['pass'] if ref_R is not None else -1,
            'offset_mm': float(rc - ref_R) if ref_R is not None else 0.0, 'half_width_mm': float(hw),
            'r_center_mm': float(rc), 'jaw_length_mm': radial['jaw_mm']}


# ---------------------------------------------------------------------------------------------------------------------
# operating point and campaign statistics
def pick_operating_point(rows, cfg=None):
    """The operating point of a dee-voltage x RF-phase bunch grid. rows: dicts with v (voltage scale), phase,
    transmission, e_final [MeV] and de_e (dE/E). Among the points whose bunch reaches pick_e_min_mev with a voltage
    scale <= pick_v_max (the RF amplifier's ceiling: wall power grows as V^2), the highest transmission; within
    pick_t_tol (absolute) of it the smallest dE/E wins. Returns winner, best_t, uncapped_best (the best transmission
    without the voltage cap) and n_ok; ValueError when no capped point reaches the energy."""
    cfg = cfg or ScoreConfig()
    reach = [r for r in rows if r['e_final'] >= cfg.pick_e_min_mev and r['transmission'] > 0]
    ok = [r for r in reach if r['v'] <= cfg.pick_v_max + 1e-9]
    if not ok:
        raise ValueError(f"no grid point reaches {cfg.pick_e_min_mev} MeV with a voltage scale <= {cfg.pick_v_max}")
    t_best = max(r['transmission'] for r in ok)
    band = [r for r in ok if r['transmission'] >= t_best - cfg.pick_t_tol]
    return {'winner': min(band, key=lambda r: r['de_e']), 'best_t': t_best,
            'uncapped_best': max(reach, key=lambda r: r['transmission']), 'n_ok': len(ok)}


def spearman(a, b, min_n=8):
    """Spearman rank correlation of two metrics over runs (the pairs where both are finite) and the number of pairs;
    NaN with fewer than min_n. The campaign used it to pick early-turn proxies of the final targets."""
    from scipy.stats import spearmanr
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n:
        return float('nan'), int(m.sum())
    return float(spearmanr(a[m], b[m])[0]), int(m.sum())


def score_column(scores, path):
    """One metric over many scores (dotted path, e.g. 'turns.3.sig_r'): a float array, NaN where it is missing."""
    out = []
    for s in scores:
        v = get_dotted(s, path)
        out.append(float(v) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else np.nan)
    return np.array(out)


def proxy_correlations(scores, candidates, targets, min_n=8):
    """{candidate: {target: (Spearman rho, pairs)}} over scored runs: how well a cheap early metric (e.g. sig_r_t3)
    ranks the runs like a final target (e.g. valley_all). Scores of skipped runs are ignored."""
    scores = [s for s in scores if 'skip' not in s and 'error' not in s]
    cols = {p: score_column(scores, p) for p in set(candidates) | set(targets)}
    return {c: {t: spearman(cols[c], cols[t], min_n) for t in targets} for c in candidates}


def campaign_summary(scores, paths):
    """{metric: (median, 25 %, 75 % quantile, n)} over scored runs (finite values only)."""
    out = {}
    for p in paths:
        v = score_column([s for s in scores if 'skip' not in s and 'error' not in s], p)
        v = v[np.isfinite(v)]
        out[p] = (float(np.median(v)), float(np.quantile(v, 0.25)), float(np.quantile(v, 0.75)), int(v.size)) \
            if v.size else (float('nan'),) * 3 + (0,)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# one run
def machine_value(run, key):
    """A [machine] setting of the run: its recorded table (``runs.RunData.machine``: the config it was tracked with),
    else the ``config.MachineConfig`` default."""
    return run.machine[key] if key in run.machine else getattr(MachineConfig(), key)


def tracked_species(run):
    """(mass [MeV / c^2], charge state Z) of the tracked ions. The tracker pushes, and records E_mev with, the
    PyPATools preset of its [machine] species (``machine.make_design``; a hand-off file's species_mass_mev only turns
    its momenta into velocities), so: that preset when the run records its [machine] table, else the hand-off file's
    species_mass_mev and species_charge_state, else the preset of its pypatools_species_name or of the default
    [machine] species; NaN mass when none is known."""
    meta = run.handoff_meta or {}
    if 'species' not in run.machine and meta.get('species_mass_mev') is not None:
        return float(meta['species_mass_mev']), float(meta.get('species_charge_state', 1.0))
    try:
        from PyPATools.species import PRESETS, IonSpecies
    except ImportError:
        return float('nan'), 1.0
    name = str(run.machine.get('species') or meta.get('pypatools_species_name') or MachineConfig().species)
    if name not in PRESETS:                          # IonSpecies exits the interpreter on a name it does not know
        return float('nan'), 1.0
    sp = IonSpecies(name)
    return float(sp.mass_mev), float(sp.q)


def run_constants(run, cfg):
    """What scoring needs from the run besides its tracks: from its json, else from its [machine] table
    (``machine_value``). The loss limits (json['limits'], else machine r_max_m / z_max_m), the harmonic
    (json['harmonic'], else rf / f_rev, else machine harmonic), the RF frequency (json rf_mhz, else h x f_rev_mhz, else
    the hand-off file's RF: the bunch was formed for it) and the revolution frequency, the tracked species
    (``tracked_species``), |B| at the bunch (mean |B_z| of the run's field check, else cfg.b0_t, else the isochronous
    field of the species), the injected current (the run's space-charge current, else the hand-off file's transmitted
    current: an SC-off run has none of its own), the bunch charge and whether space charge was on."""
    js = run.info
    fj, scj, lim = js.get('final') or {}, js.get('space_charge') or {}, js.get('limits') or {}
    rf, frev = js.get('rf_mhz'), js.get('f_rev_mhz')
    h = int(js['harmonic']) if js.get('harmonic') else (int(round(rf / frev)) if rf and frev else
                                                        int(machine_value(run, 'harmonic')))
    f_rf = 1e6 * float(rf) if rf else (1e6 * float(frev) * h if frev else
                                       float(run.handoff_meta.get('rf_frequency_hz', np.nan)))
    f_rev = f_rf / h
    mass, zq = tracked_species(run)
    i0_from = 'run' if scj.get('current_ma') else 'handoff'
    i0 = float(scj['current_ma']) if i0_from == 'run' else float(run.handoff_meta.get('transmitted_current_ma', np.nan))
    bchk = (js.get('fields') or {}).get('bfield_check') or {}
    if bchk:
        b0 = float(np.mean([abs(v['bz3_mean']) for v in bchk.values()]))
    elif cfg.b0_t > 0:
        b0 = float(cfg.b0_t)
    else:
        b0 = isochronous_field(f_rev, mass, zq)
    q = (fj.get('space_charge') or {}).get('bunch_charge_c') or (i0 * 1e-3 / f_rf if np.isfinite(i0) else np.nan)
    return {'r_max': float(lim.get('r_max_m', machine_value(run, 'r_max_m'))),
            'z_max': float(lim.get('z_max_m', machine_value(run, 'z_max_m'))), 'harmonic': h, 'f_rf': f_rf,
            'f_rev': f_rev, 'mass_mev': mass, 'charge_state': zq, 'B0': b0, 'I0': i0, 'I0_from': i0_from,
            'Q': float(q), 'sc_on': bool(fj.get('sc', scj.get('on', False))) and bool(run.sc_recorded)}


RunState = namedtuple('RunState', 'k steps gstep alive surv sense')


def run_state(run, cfg):
    """A run's constants and who is alive when, the common start of ``score_run``, ``runs.virtual_probe`` and
    ``twins.vortex_twin``: k (``run_constants``), the recorded steps (S,), every track's first ghost step (N,), alive
    (S, N) (``alive_mask``), the survivors (N,) (``survivors``) and the rotation sense."""
    k = run_constants(run, cfg)
    steps = np.asarray(run.steps)
    gstep = ghost_first_step(run.ids, run.ghost_ids, run.ghost_step)
    alive = alive_mask(steps, run.xyz, run.birth_step, run.stop_step, gstep)
    surv = survivors(run.xyz[-1], run.stop_step, run.n_steps, gstep, k['r_max'], k['z_max'])
    return RunState(k, steps, gstep, alive, surv, rotation_sense(run.xyz, alive))


def turn_instants(steps, alive_surv, n_steps, spt, n_snap):
    """Record indices of n_snap instants per complete revolution k = 1 .. n_steps // spt - 1 (centred in 1/n_snap
    slices of steps [k spt, (k + 1) spt)), kept only when every survivor (alive_surv (S, nS)) is alive at all of them
    and each is within one record interval of its target; 'last': n_snap records spread over the last spt steps."""
    tol = np.median(np.diff(steps)) if len(steps) > 1 else 0.0
    inst = {}
    for kk in range(1, n_steps // spt):
        tgt = kk * spt + (np.arange(n_snap) + 0.5) * spt / n_snap
        ri = np.clip(np.searchsorted(steps, tgt), 0, len(steps) - 1)
        if np.any(np.abs(steps[ri] - tgt) > tol) or not np.all(alive_surv[ri]):
            continue
        inst[kk] = ri
    k_last = np.flatnonzero(steps >= n_steps - spt)
    inst['last'] = k_last[np.linspace(0, len(k_last) - 1, n_snap).round().astype(int)]
    return inst


def score_run(run, cfg=None):
    """Score one run (``runs.RunData``) with the settings cfg (``ScoreConfig``). Returns a dict; a run with fewer than
    min_tracks tracks or min_survivors tracked survivors gives {'tag', 'skip': reason}.

    Survivors are the tracked particles that reach the end (``survivors``); in a ghost run a ghosted particle is lost
    at its ghost step. Every per-turn metric is evaluated on these final survivors. The keys, by block:
      run       tag, beam, bmap, sc, ghost_run, phase, n_tracks, n_surv, T (the json's full-beam transmission; the
                tracked real survivors in a ghost run), T_trk, I0 (injected current [mA]), I_final = T x I0, n_steps,
                spt, sense
      energy    E_med, E_mean, E_min, dE_E, dE_E_robust, R_last, sigR_E, I_<p>pct, I_<p>pct_best, E_<p>pct_best
                (energy_metrics); r_rms_last, r_rms_E, r_rms_rest, dR_dE, corr_r_lag, corr_E_lag (radial_width_split);
                losses = share of the tracked bunch per loss-turn class
      release   (with the hand-off phases) acc<L>_width/_lo/_hi/_ceiling, W_eff, survival_curve, S30, S30_40
                (acceptance); slope, lag_rms, release_rms_surv (compression from the final lag); release_bins,
                E_peak_phase, E_peak (release_budget); band<q>_*, surv_vs_z, surv_vs_zp (handoff_survival)
      shape     turns: per time-turn k (n_snapshots instants of steps [k spt, (k + 1) spt), all survivors alive;
                step 0 is the first release) and 'last' (the last spt steps): shape_stats, sig_z (robust), alive_frac,
                with space charge b_pred, c_pred, rot_pred
                (vortex_prediction), M, b_meas, c_meas, rot_meas towards turn k + 1 (one_turn_map, core within
                map_core_sigma), and with the hand-off phases compress and lag_rms (the turn-mean lag, turn_lag);
                turn0_part: the shape and prediction over time-turn 0 once the last survivor is born (t_from [turns]);
                flat: ratio, sig_r, sig_l, s_round_last, halo3, tilt_last, sig_z_last (last turn), ratio_t1, sig_r_t1,
                sig_l_t1, s_round_t1, tilt_t1 (first full turn), ratio / sig_r / sig_l / s_round / compress _t<t> for
                the proxy turns, rot_meas_sum, rot_pred_sum, b/c_meas_mean, b/c_pred_mean, n_trans, halo_born_t<t>;
                ratio_no_fan, r_release_share (ratio_without_fan)
      fan       fan: per revolution n from each particle's own start (about time-turn n - 1) coh, coh_xy, spread,
                c_med, proj, proj_max, expl_share, early_late (centre_fan); fan_spread / fan_share / fan_proj / coh
                _t<t> (revolution t of the proxy turns) and fan_spread / fan_share / fan_proj / fan_proj_max / coh
                _last, fan_last_turn (the last revolution most survivors complete)
      turns     probes: per azimuth passes, R, sigma, dR, sep, valley, vp, halo (turn_structure); turn_sep,
                turn_valley, turn_vp, turn_halo (means over the probes), turn_valley_max / _min / _min_az, turn_dR,
                turn_sigR (last pass) / _min / _max; valley_all (all probes' crossings pooled), valley_from_halo
                (share of them made by last-turn halo particles), valley_no_halo
      slits     slits: per budget the best radial and vertical slit (slit_tradeoff); valley_rad<b>, valley_ver<b>,
                I_rad<b> = I_final x (1 - taken); best_slit: per budget percentage the radial slit as [slit] settings
                (slit_settings) with its valley and I_final, e.g. best_slit.20.offset_mm
    plus every registered user metric (objectives.register_metric), notes (why a block is missing) and seconds.
    """
    cfg = cfg or ScoreConfig()
    t_start = time.time()
    N = run.n_tracks
    if N < cfg.min_tracks:
        return {'tag': run.tag, 'skip': f"{N} tracks (min_tracks {cfg.min_tracks})"}
    k, steps, gstep, alive, surv, sense = run_state(run, cfg)
    xyz, n_steps, spt = run.xyz, int(run.n_steps), int(run.steps_per_turn)
    last = xyz[-1]
    nS = int(surv.sum())
    if nS < cfg.min_survivors:
        return {'tag': run.tag, 'skip': f"{nS} tracked survivors inside r {k['r_max']} m (T json "
                                        f"{run.pkl_final.get('transmission')}, {n_steps / spt:.1f} turns)"}
    fj = run.info.get('final') or {}
    T = nS / N if run.ghost_run or 'transmission' not in fj else float(fj['transmission'])
    I0, h = k['I0'], k['harmonic']
    res = {'tag': run.tag, 'beam': run.beam, 'bmap': str(run.info.get('bmap', '')), 'sc': k['sc_on'],
           'ghost_run': run.ghost_run, 'phase': fj.get('phase'), 'n_tracks': N, 'n_surv': nS, 'T': T, 'T_trk': nS / N,
           'I0': I0, 'I_final': T * I0, 'n_steps': n_steps, 'spt': spt, 'sense': sense}

    # --- energy, final phase lag, radial width, losses
    E = np.asarray(run.E_mev, dtype=float)
    res.update(energy_metrics(E[surv], N, I0, last[surv], cfg.energy_windows))
    s = surv & np.all(np.isfinite(xyz[-2]), axis=1)
    lag = final_phase_lag(last, xyz[-2], s, h)
    res.update(radial_width_split(1e3 * np.hypot(last[s, 0], last[s, 1]), E[s], lag[s]))
    loss = loss_steps(steps, xyz, run.birth_step, run.stop_step, gstep, n_steps, surv, k['r_max'], k['z_max'])
    lost_turn = np.where(surv, np.nan, (loss - run.birth_step) / spt)
    res['losses'] = {'turn_edges': [float(x) for x in cfg.loss_turn_edges],
                     'share': loss_turn_shares(lost_turn, cfg.loss_turn_edges)}

    # --- acceptance and compression in release phase
    ph = run.of_tracks(run.release_phase_deg)
    if ph is not None:
        res.update(acceptance(ph, surv, run.release_phase_deg, cfg))
        ps, ls = ph[s], lag[s]
        res.update(slope=compression_slope(ps, ls, cfg.compression_clip_sigma), lag_rms=float(ls.std()),
                   release_rms_surv=float(ps.std()))
        res.update(release_budget(ph, s, E, lag, lost_turn, cfg))
        z0, zp0 = run.of_tracks(run.handoff_z_m), run.of_tracks(run.handoff_zp_rad)
        if z0 is not None and zp0 is not None:
            res.update(handoff_survival(ph, 1e3 * z0, 1e3 * zp0, surv, cfg))

    # --- per-turn bunch frame, vortex estimate, turn-mean phase lag
    si = np.flatnonzero(surv)
    inst = turn_instants(steps, alive[:, si], n_steps, spt, cfg.n_snapshots)
    halo_key, h2 = f"halo{cfg.halo_sigma:g}", cfg.halo_sigma ** 2
    vortex = k['sc_on'] and all(np.isfinite([k['Q'], k['B0'], k['f_rev']]))

    def snapshot(ri):
        X = xyz[ri][:, surv]
        u = bunch_coordinates(X[..., :2] * 1e3, sense)
        Cc, d2 = core_covariance(u, cfg.core_q)
        st = shape_stats(Cc, d2, cfg.halo_sigma)
        st['sig_z'] = float(np.mean(robust_sigma(X[..., 2] * 1e3, axis=1)))
        st['alive_frac'] = float(alive[ri].mean())
        if vortex:
            st.update(vortex_prediction(Cc, st['sig_z'], k['Q'] * st['alive_frac'], k['B0'], k['f_rev'], cfg.core_q))
        return u, d2, st

    turns, frames = {}, {}
    for kk, ri in inst.items():
        u, d2, st = snapshot(ri)
        if ph is not None:
            lag_k = turn_lag(xyz[ri][:, surv], h, sense)
            st.update(compress=compression_slope(ph[surv], lag_k, cfg.compression_clip_sigma),
                      lag_rms=float(lag_k.std()))
        frames[kk], turns[kk] = (u, d2), st
    # time-turn 0 once the last survivor is born: shape and prediction only (no one-turn map, not in 'turns'), the
    # first point of the time-matched Gaussian reference of the guiding-centre vortex (twins.vortex_twin)
    k0 = np.flatnonzero((steps < spt) & np.all(alive[:, si], axis=1))
    if len(k0) >= cfg.n_snapshots:
        res['turn0_part'] = snapshot(k0[np.linspace(0, len(k0) - 1, cfg.n_snapshots).round().astype(int)])[2] | {
            't_from': float(steps[k0[0]] / spt)}
    ks = sorted(kk for kk in inst if kk != 'last')
    for kk in ks:
        if kk + 1 in frames:
            (u0, d0), (u1, _) = frames[kk], frames[kk + 1]
            turns[kk].update(one_turn_map(u0, u1, d0 <= cfg.map_core_sigma ** 2))
    res['turns'] = {str(kk): v for kk, v in turns.items()}
    lt = turns['last']
    res.update({'ratio': lt['ratio'], 'sig_r': lt['sig_r'], 'sig_l': lt['sig_l'], 's_round_last': lt['s_round'],
                halo_key: lt[halo_key], 'tilt_last': lt['tilt'], 'sig_z_last': lt['sig_z']})
    h_last = np.median(frames['last'][1], axis=0) > h2                  # last-turn halo (median over the instants)
    if ks:
        t1 = turns[ks[0]]
        res.update(s_round_t1=t1['s_round'], ratio_t1=t1['ratio'], sig_l_t1=t1['sig_l'], sig_r_t1=t1['sig_r'],
                   tilt_t1=t1['tilt'])
        for t in cfg.proxy_turns:
            for key in ('ratio', 'sig_r', 'sig_l', 's_round', 'compress'):
                res[f'{key}_t{t}'] = turns.get(t, {}).get(key, np.nan)
        trans = [kk for kk in ks if 'rot_meas' in turns[kk]]
        col = (lambda key: [turns[kk].get(key, np.nan) for kk in trans])
        res.update(rot_meas_sum=float(np.sum(col('rot_meas'))) if trans else np.nan,
                   rot_pred_sum=float(np.sum(col('rot_pred'))) if trans else np.nan,
                   c_meas_mean=_mean(col('c_meas')), c_pred_mean=_mean(col('c_pred')),
                   b_meas_mean=_mean(col('b_meas')), b_pred_mean=_mean(col('b_pred')), n_trans=len(trans))
        for t in cfg.halo_born_turns:
            if t in frames and h_last.any():
                res[f'halo_born_t{t}'] = float(np.mean((np.median(frames[t][1], axis=0) > h2)[h_last]))
    if ph is not None:
        res['ratio_no_fan'], res['r_release_share'] = ratio_without_fan(frames['last'][0], ph[surv], cfg.core_q,
                                                                        cfg.fan_degree)

    # --- centre fan
    cen = turn_centres(xyz[:, si, :2], alive[:, si], sense, cfg.turn_fill)
    fan = centre_fan(cen, None if ph is None else ph[si], cfg.fan_degree, cfg.turn_fill)
    res['fan'] = {str(t): row for t, row in fan.items()}
    for t in cfg.proxy_turns:
        if t in fan:
            res.update({f'fan_spread_t{t}': fan[t]['spread'], f'fan_share_t{t}': fan[t].get('expl_share', np.nan),
                        f'fan_proj_t{t}': fan[t]['proj'], f'coh_t{t}': fan[t]['coh']})
    if fan:
        tl = max(fan)
        res.update(fan_spread_last=fan[tl]['spread'], fan_share_last=fan[tl].get('expl_share', np.nan),
                   fan_proj_last=fan[tl]['proj'], fan_proj_max_last=fan[tl]['proj_max'], coh_last=fan[tl]['coh'],
                   fan_last_turn=tl)

    # --- virtual probes: turn structure, valleys, collimation
    dt_step = 1.0 / k['f_rev'] / spt
    angles = [float(a) for a in cfg.probe_angles_deg]
    prs = {a: probe_crossings(xyz, alive, steps, dt_step, sense, a) for a in angles}
    tsr = {a: turn_structure(prs[a], surv, nS, cfg) for a in angles}
    good = [a for a in angles if tsr[a] is not None]
    res['probes'] = {f"{a:g}": {kk: vv for kk, vv in tsr[a].items() if kk not in ('j_w', 'mid_w')} for a in good}
    if good:
        for key in ('sep', 'valley', 'vp', 'halo'):
            res[f'turn_{key}'] = float(np.mean([tsr[a][key] for a in good]))
        res.update(turn_valley_max=float(max(tsr[a]['valley'] for a in good)),
                   turn_valley_min=float(min(tsr[a]['valley'] for a in good)),
                   turn_valley_min_az=float(min(good, key=lambda a: tsr[a]['valley'])),
                   turn_dR=float(np.mean([tsr[a]['dR'][-1] for a in good])),
                   turn_sigR=float(np.mean([tsr[a]['sigma'][-1] for a in good])),
                   turn_sigR_min=float(min(tsr[a]['sigma'][-1] for a in good)),
                   turn_sigR_max=float(max(tsr[a]['sigma'][-1] for a in good)))
        # per survivor: its crossings in the probe windows (n) and how many of them sit in a valley (v)
        jw = np.concatenate([tsr[a]['j_w'] for a in good])
        n_c = np.bincount(jw, minlength=N).astype(float)
        v_c = np.bincount(jw, weights=np.concatenate([tsr[a]['mid_w'] for a in good]), minlength=N)
        halo_any = np.zeros(N, bool)
        halo_any[surv] = h_last
        res['valley_all'] = float(v_c.sum() / n_c.sum())
        res['valley_from_halo'] = float(v_c[halo_any].sum() / max(v_c.sum(), 1.0))
        res['valley_no_halo'] = float((v_c.sum() - v_c[halo_any].sum()) / (n_c.sum() - n_c[halo_any].sum()))
        res['slits'] = slit_tradeoff(prs, surv, v_c, n_c, cfg)
        refx = {}
        if run.ref_xyz is not None:
            refx = {a: reference_crossings(run.ref_xyz, a, sense) for a in {sl['radial'].get('az')
                                                                            for sl in res['slits'].values()} - {None}}
        res['best_slit'] = {}
        for b, sl in res['slits'].items():
            tb = f"{int(round(100 * float(b)))}"
            res[f'valley_rad{tb}'] = sl['radial']['valley']
            res[f'valley_ver{tb}'] = sl['vertical']['valley']
            res[f'I_rad{tb}'] = res['I_final'] * (1.0 - sl['radial'].get('taken', 0.0))
            seed = slit_settings(sl['radial'], refx.get(sl['radial'].get('az'), {}).get(sl['radial'].get('pass')))
            if seed is not None:
                res['best_slit'][tb] = seed | {'valley': sl['radial']['valley'], 'I_final': res[f'I_rad{tb}']}
    notes = list(run.notes)
    if k['I0_from'] == 'handoff':
        notes.append("I0 is the hand-off file's transmitted current (the run records no space-charge current of its "
                     "own)" if np.isfinite(I0) else "the injected current is unknown (no space-charge current in the "
                     "run, no transmitted_current_ma in the hand-off file): I_final and the I_* currents are NaN")
    if k['sc_on'] and not vortex:
        notes.append(f"space charge on, but no vortex prediction: bunch charge {k['Q']} C, |B| {k['B0']} T, f_rev "
                     f"{k['f_rev']} Hz")
    if notes:
        res['notes'] = notes

    # --- user metrics
    for name, m in METRICS.items():
        if m.fn is None:
            continue
        if name in res:
            raise ValueError(f"user metric {name!r} would overwrite a built-in score key")
        try:
            res[name] = m.fn(run, res, cfg)
        except Exception as ex:                                     # one bad user metric must not lose the score
            res[name] = float('nan')
            res.setdefault('metric_errors', {})[name] = f"{type(ex).__name__}: {ex}"
            warnings.warn(f"{run.tag}: metric {name!r} failed: {type(ex).__name__}: {ex}", stacklevel=2)
    res['seconds'] = round(time.time() - t_start, 1)
    return res
