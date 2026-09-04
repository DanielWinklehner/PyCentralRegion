"""
closed_orbit.py - Gordon closed-orbit solver for midplane field maps.

Static equilibrium orbits (SEOs) of a cyclotron median-plane field by the
method of M. M. Gordon [Part. Accel. 16 (1984) 39]: the equations of motion
are integrated with the azimuth theta as the independent variable,

    dr/dtheta   = r p_r / p_theta
    dp_r/dtheta = p_theta - r B(r, theta)
    ds/dtheta   = r (B rho) / p_theta                 (path length)

with momenta measured in units of magnetic rigidity (T m), p_theta =
sqrt((B rho)^2 - p_r^2), and the closed orbit is the full-turn fixed point
(r, p_r)(0) = (r, p_r)(2 pi), found by Newton iteration.  No symmetry is
assumed, so 2-fold or asymmetric maps work as well as 8-fold ones.  The
revolution period is T = L / v with L the closed-orbit length; the radial
tune comes from the finite-difference one-turn matrix and the vertical tune
from the linearised vertical motion along the converged orbit,

    dz/dtheta   = r p_z / p_theta
    dp_z/dtheta = z [ r dB/dr - (p_r/p_theta) dB/dtheta ]

(median-plane symmetry: B_r = z dB_z/dr, B_theta = (z/r) dB_z/dtheta).  The
second term is the Thomas (flutter) focusing; the radial-gradient term gives
nu_z^2 = -k in a smooth field.  Together these reproduce nu_z^2 = -k +
F N^2/(N^2 - 1) in the small-flutter limit and stay exact at large flutter.

Everything is vectorised over a batch of energies (fixed-step RK4 in theta,
one field lookup per stage for the whole batch), and the field lookup is a
cubic B-spline (scipy.ndimage.spline_filter once, map_coordinates with
prefilter=False per call), so a radial scan of ~200 orbits takes seconds.
Validated against PyCentralRegion's Cartesian time-domain SEO tracker to
< 1e-6 in revolution frequency on the IsoDAR HCHC-60 maps (relative 4th
harmonic ~0.9), where the first-order analytic scalloping formulas are
~0.4 % off.

Sign convention: the field objects return B > 0 in the hills (the sign of
the map is flipped automatically if needed); the ion then circulates in +theta.
Physics is orientation-invariant, so this is purely a bookkeeping choice.

Usage
-----
    from PyCentralRegion.closed_orbit import CartesianMidplane, orbits_at_radii
    fld = CartesianMidplane.from_field(design.bfield)     # PyPATools 2D Field
    eo  = orbits_at_radii(fld, radii_m, species)          # dict of arrays
    eo['f_rev'], eo['E_MeV'], eo['nu_r'], eo['nu_z'], eo['converged'], ...

Species objects need ``.mass_mev`` and ``.q`` (PyPATools / cyclotron_optimizer
IonSpecies both qualify).
"""
import warnings
from typing import Optional

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter

CLIGHT = 299792458.0
RIGIDITY_K = 299.792458        # p[MeV/c] = K * |q| * (B rho)[T m]

__all__ = ['CartesianMidplane', 'PolarMidplane', 'closed_orbits',
           'orbits_at_radii', 'azimuthal_stats', 'kinematics']


# ---------------------------------------------------------------------------
# kinematics helpers
# ---------------------------------------------------------------------------
def _species_mq(species):
    m0 = float(getattr(species, 'mass_mev'))
    q = getattr(species, 'q', None)
    if q is None:
        q = getattr(species, 'charge_state', 1.0)
    return m0, abs(float(q))


def kinematics(energies_mev, species):
    """(p [MeV/c], gamma, beta, B rho [T m]) for kinetic energies [MeV]."""
    m0, q = _species_mq(species)
    E = np.asarray(energies_mev, dtype=float)
    p = np.sqrt(E ** 2 + 2.0 * E * m0)
    gamma = 1.0 + E / m0
    beta = p / (gamma * m0)
    return p, gamma, beta, p / (RIGIDITY_K * q)


def energy_from_brho(brho, species):
    """Kinetic energy [MeV] from magnetic rigidity [T m]."""
    m0, q = _species_mq(species)
    p = RIGIDITY_K * q * np.asarray(brho, dtype=float)
    return np.sqrt(p ** 2 + m0 ** 2) - m0


# ---------------------------------------------------------------------------
# field lookups
# ---------------------------------------------------------------------------
class _Midplane:
    """Common interface: B(r, theta) [T], dB(r, theta) -> (dB/dr, dB/dtheta)."""
    r_min = 0.0
    r_max = np.inf
    n_rot = 1          # rotational symmetry order (for seeding only)
    sign = 1.0

    def B(self, r, theta):  # pragma: no cover - abstract
        raise NotImplementedError

    def dB(self, r, theta, h_r=1e-4, h_t=1e-4):
        """Central-difference gradients of the spline (smooth in both)."""
        r, theta = np.broadcast_arrays(np.asarray(r, float), np.asarray(theta, float))
        dBdr = (self.B(r + h_r, theta) - self.B(r - h_r, theta)) / (2.0 * h_r)
        dBdt = (self.B(r, theta + h_t) - self.B(r, theta - h_t)) / (2.0 * h_t)
        return dBdr, dBdt


class CartesianMidplane(_Midplane):
    """Cubic-spline Bz(x, y) lookup on a regular Cartesian grid, bz[iy, ix] (m, T)."""

    def __init__(self, x, y, bz_yx, sign=None, n_rot=1):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        bz_yx = np.asarray(bz_yx, dtype=np.float64)
        if bz_yx.shape != (len(y), len(x)):
            raise ValueError(f'bz_yx must have shape (len(y), len(x)) = '
                             f'{(len(y), len(x))}, got {bz_yx.shape}')
        self.x0, self.dx = float(x[0]), float(x[1] - x[0])
        self.y0, self.dy = float(y[0]), float(y[1] - y[0])
        self.nx, self.ny = len(x), len(y)
        if sign is None:
            iy, ix = int(np.argmin(np.abs(y))), int(np.argmin(np.abs(x)))
            w = max(1, min(50, self.nx // 10, self.ny // 10))
            sign = np.sign(bz_yx[iy - w:iy + w + 1, ix - w:ix + w + 1].mean()) or 1.0
        self.sign = float(sign)
        self.coef = spline_filter(bz_yx, order=3)
        self.r_max = float(min(abs(x[0]), abs(x[-1]), abs(y[0]), abs(y[-1])))
        self.n_rot = int(n_rot)

    @classmethod
    def from_field(cls, field, **kw):
        """From a PyPATools 2D ``Field`` built on a regular x/y grid (from_file /
        from_arrays); ``field.scaling`` is applied."""
        grid, values = field.grid, field.grid_values
        if grid is None or values is None or 'z' not in values:
            raise ValueError('Field carries no regular-grid Bz array')
        x = np.asarray(grid['x'], dtype=float)
        y = np.asarray(grid['y'], dtype=float)
        bz = np.asarray(values['z'], dtype=float)
        if bz.ndim == 3:
            if bz.shape[2] != 1:
                raise ValueError('3D field map: closed_orbit needs a 2D midplane map')
            bz = bz[:, :, 0]
        if bz.shape == (len(x), len(y)):
            bz = bz.T
        elif bz.shape != (len(y), len(x)):
            raise ValueError(f'unexpected Bz array shape {bz.shape} for grid '
                             f'{len(x)} x {len(y)}')
        return cls(x, y, bz * float(getattr(field, 'scaling', 1.0)), **kw)

    def B(self, r, theta):
        r, theta = np.broadcast_arrays(np.asarray(r, float), np.asarray(theta, float))
        X = r * np.cos(theta)
        Y = r * np.sin(theta)
        return self.sign * map_coordinates(
            self.coef, [(Y - self.y0) / self.dy, (X - self.x0) / self.dx],
            order=3, prefilter=False, mode='nearest')


class PolarMidplane(_Midplane):
    """Cubic-spline Bz(r, theta) lookup on a regular polar grid bz[ir, ith].

    The angular samples may cover only the fundamental sector of the field's
    symmetry: ``period`` is the rotational period (2 pi / n_rot) and
    ``mirror`` says the sector is the half-period [0, period/2] with mirror
    planes at theta = 0 and theta = period/2.  A mirror-folded sector MUST
    include the theta = period/2 sample (endpoint=True); an unfolded period
    is sampled with endpoint=False (a trailing duplicate of theta = period is
    dropped).  Full-circle data: period = 2 pi, mirror = False.
    """

    PAD = 4

    def __init__(self, r, theta, bz_rt, period=2.0 * np.pi, mirror=False, sign=None):
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float)
        bz = np.asarray(bz_rt, dtype=np.float64)
        if bz.shape != (len(r), len(theta)):
            raise ValueError(f'bz_rt must have shape (len(r), len(theta)) = '
                             f'{(len(r), len(theta))}, got {bz.shape}')
        if len(theta) < 2:
            raise ValueError('need at least two azimuthal samples')
        dt = float(theta[1] - theta[0])
        if not np.allclose(np.diff(theta), dt, rtol=0, atol=1e-9 * abs(dt)):
            raise ValueError('theta samples must be uniform')
        if abs(theta[0]) > 1e-12:
            raise ValueError('theta samples must start at 0')
        period = float(period)
        span = float(theta[-1])
        if mirror:
            half = 0.5 * period
            if abs(span - half) > 1e-6 * period:
                raise ValueError(
                    'mirror-folded sector must include the theta = period/2 sample '
                    f'(got theta[-1] = {np.degrees(span):.4f} deg, expected '
                    f'{np.degrees(half):.4f} deg); sample the sector with endpoint=True')
            n = len(theta) - 1
            full = np.concatenate([bz[:, :n + 1], bz[:, n - 1:0:-1]], axis=1)
        else:
            if abs(span + dt - period) < 1e-6 * period:
                full = bz
            elif abs(span - period) < 1e-6 * period:
                full = bz[:, :-1]
            else:
                raise ValueError(
                    f'theta samples must cover one period ({np.degrees(period):.4f} deg) '
                    f'with endpoint=False; got theta[-1] = {np.degrees(span):.4f} deg')
        self.nt = full.shape[1]
        self.dt = period / self.nt
        self.period = period
        self.n_rot = max(1, int(round(2.0 * np.pi / period)))
        self.mirror = bool(mirror)
        self.r0, self.dr = float(r[0]), float(r[1] - r[0])
        self.r_min, self.r_max = float(r[0]), float(r[-1])
        if sign is None:
            k = max(2, min(len(r), int(round(0.5 / self.dr))))
            sign = np.sign(full[:k].mean()) or 1.0
        self.sign = float(sign)
        p = self.PAD
        self.coef = spline_filter(np.concatenate([full[:, -p:], full, full[:, :p]], axis=1),
                                  order=3)

    def B(self, r, theta):
        r, theta = np.broadcast_arrays(np.asarray(r, float), np.asarray(theta, float))
        it = np.mod(theta / self.dt, self.nt) + self.PAD
        return self.sign * map_coordinates(
            self.coef, [(r - self.r0) / self.dr, it],
            order=3, prefilter=False, mode='nearest')


# ---------------------------------------------------------------------------
# azimuthal statistics (seeds + reporting)
# ---------------------------------------------------------------------------
def azimuthal_stats(field, radii_m, n_theta=720, n_harm=12):
    """Circle statistics of the map at the given radii.

    :returns: dict with B0 (azimuthal mean), F (flutter <B^2>/<B>^2 - 1),
        Cn ((Nr, n_harm) relative harmonic amplitudes, n = 1..n_harm),
        k (field index r dB0/dr / B0), a_dom (relative amplitude of the
        dominant sector harmonic n = field.n_rot, 0 if n_rot == 1).
    """
    radii_m = np.asarray(radii_m, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    bz = field.B(radii_m[:, None], theta[None, :])
    B0 = bz.mean(axis=1)
    F = (bz ** 2).mean(axis=1) / B0 ** 2 - 1.0
    spec = np.fft.rfft(bz, axis=1) / n_theta
    Cn = 2.0 * np.abs(spec[:, 1:n_harm + 1]) / B0[:, None]
    if len(radii_m) >= 4:
        from scipy.interpolate import CubicSpline
        order = np.argsort(radii_m)
        dB0 = CubicSpline(radii_m[order], B0[order], bc_type='natural')(radii_m, nu=1)
        k = radii_m * dB0 / B0
    else:
        k = np.zeros_like(B0)
    n = int(getattr(field, 'n_rot', 1))
    if n < 2:
        # unknown symmetry (Cartesian map): take the strongest harmonic n >= 2
        # of the outermost radius as the sector number
        n = int(np.argmax(Cn[-1, 1:])) + 2
    a_dom = Cn[:, n - 1] if 2 <= n <= n_harm else np.zeros_like(B0)
    return dict(B0=B0, F=F, Cn=Cn, k=k, a_dom=a_dom, n_dom=n)


# ---------------------------------------------------------------------------
# integrator
# ---------------------------------------------------------------------------
def _integrate(field, r, pr, brho, n_steps, vertical=False):
    """Fixed-step RK4 over one turn for a batch.

    Returns (r, pr, s, r_mean, r_min, r_max, bad, Z) with Z the (N, 2, 2)
    vertical fundamental matrix when ``vertical`` else None.
    """
    h = 2.0 * np.pi / n_steps
    r, pr = np.array(r, dtype=float), np.array(pr, dtype=float)
    N = len(r)
    s = np.zeros(N)
    rsum = np.zeros(N)
    rcos = np.zeros(N)
    rsin = np.zeros(N)
    rmin = np.full(N, np.inf)
    rmax = np.full(N, -np.inf)
    bad = np.zeros(N, dtype=bool)
    Z = np.tile(np.eye(2), (N, 1, 1)) if vertical else None
    r_lo = getattr(field, 'r_min', 0.0)
    r_hi = getattr(field, 'r_max', np.inf)

    def rhs(th, r_, pr_, Z_):
        pth2 = brho ** 2 - pr_ ** 2
        ok = pth2 > 0
        pth = np.sqrt(np.where(ok, pth2, 1.0))
        B = field.B(r_, th)
        dr = np.where(ok, r_ * pr_ / pth, 0.0)
        dp = np.where(ok, pth - r_ * B, 0.0)
        ds = np.where(ok, r_ * brho / pth, 0.0)
        dZ = None
        if Z_ is not None:
            dBdr, dBdt = field.dB(r_, th)
            a12 = np.where(ok, r_ / pth, 0.0)
            a21 = np.where(ok, r_ * dBdr - (pr_ / pth) * dBdt, 0.0)
            dZ = np.empty_like(Z_)
            dZ[:, 0, :] = a12[:, None] * Z_[:, 1, :]
            dZ[:, 1, :] = a21[:, None] * Z_[:, 0, :]
        return dr, dp, ds, dZ, ~ok

    for i in range(n_steps):
        th = i * h
        k1 = rhs(th, r, pr, Z)
        k2 = rhs(th + 0.5 * h, r + 0.5 * h * k1[0], pr + 0.5 * h * k1[1],
                 None if Z is None else Z + 0.5 * h * k1[3])
        k3 = rhs(th + 0.5 * h, r + 0.5 * h * k2[0], pr + 0.5 * h * k2[1],
                 None if Z is None else Z + 0.5 * h * k2[3])
        k4 = rhs(th + h, r + h * k3[0], pr + h * k3[1],
                 None if Z is None else Z + h * k3[3])
        bad |= k1[4] | k2[4] | k3[4] | k4[4]
        rsum += r
        rcos += r * np.cos(th)
        rsin += r * np.sin(th)
        np.minimum(rmin, r, out=rmin)
        np.maximum(rmax, r, out=rmax)
        r = r + (h / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        pr = pr + (h / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        s = s + (h / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        if Z is not None:
            Z = Z + (h / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        bad |= ~np.isfinite(r) | (r <= r_lo) | (r >= r_hi)
    r_h1 = 2.0 * np.hypot(rcos, rsin) / n_steps      # first-harmonic amplitude of r(theta)
    return r, pr, s, rsum / n_steps, rmin, rmax, bad, Z, r_h1


def _tune_from_matrix(M, center):
    """(nu, nu^2 signed) from a 2x2 one-turn matrix batch (N, 2, 2).

    cos(2 pi nu) = tr/2 fixes nu up to sign and integer; the sign of M12
    (= beta sin 2 pi nu, beta > 0) fixes the sign, and the integer is chosen
    so that nu lies nearest ``center`` (1 for the radial tune of an
    isochronous cyclotron, nu_r ~ gamma; 0 for the vertical tune):
    sin > 0 -> nu = center + frac, sin < 0 -> nu = max(center, 1) - frac.
    |tr/2| > 1 is unstable -> nu NaN and nu^2 continued signed as
    center^2 - (arccosh(tr/2) / 2 pi)^2 (tr/2 > 1, integer stopband) or
    (center + 1/2)^2 + (...)^2 (tr/2 < -1, half-integer stopband), so it
    stays finite through the instability.
    """
    tr2 = 0.5 * (M[:, 0, 0] + M[:, 1, 1])
    stable = np.abs(tr2) <= 1.0
    frac = np.arccos(np.clip(tr2, -1.0, 1.0)) / (2.0 * np.pi)
    nu = np.where(M[:, 0, 1] > 0, center + frac, max(center, 1.0) - frac)
    nu = np.where(stable, nu, np.nan)
    mu_im = np.arccosh(np.maximum(np.abs(tr2), 1.0)) / (2.0 * np.pi)
    nu_sq_unstable = np.where(tr2 > 1.0,
                              center ** 2 - mu_im ** 2,           # integer stopband
                              (center + 0.5) ** 2 + mu_im ** 2)   # half-integer stopband
    nu_sq = np.where(stable, nu ** 2, nu_sq_unstable)
    return nu, nu_sq


def closed_orbits(field, energies_mev, species, r_seed_m, pr_seed=None, *,
                  n_steps=3600, tol_m=1e-10, max_iter=30, max_step_m=2e-2,
                  vertical=True, displaced_tol=0.01, verbose=False):
    """Closed orbits at the given kinetic energies (batch).

    :param field: CartesianMidplane | PolarMidplane (or any object with
        ``B(r, theta)``/``dB(r, theta)``/``r_min``/``r_max``)
    :param energies_mev: (N,) kinetic energies [MeV]
    :param species: object with ``.mass_mev`` and ``.q``
    :param r_seed_m: (N,) starting radius at theta = 0 [m]
    :param pr_seed: (N,) starting radial momentum in rigidity units [T m]
    :param n_steps: RK4 steps per turn (3600 = 0.1 deg; converged to ~1e-9)
    :param vertical: also integrate the linearised vertical motion (nu_z)
    :param displaced_tol: orbits whose first radial harmonic exceeds this
        fraction of the mean radius are flagged ``displaced`` (off-centre
        closed orbits of the nu_r = 1 family near the pole edge)
    :returns: dict of (N,) arrays: E_MeV, brho, gamma, beta, r0, pr0, r_mean,
        r_min, r_max, L, T_rev, f_rev, B_avg_orbit, nu_r, nu_r_sq, nu_z,
        nu_z_sq, r_h1, displaced, converged, residual_m (NaN where not
        converged; displaced orbits are returned but flagged)
    """
    E = np.atleast_1d(np.asarray(energies_mev, dtype=float))
    p, gamma, beta, brho = kinematics(E, species)
    N = len(E)

    r0 = np.atleast_1d(np.asarray(r_seed_m, dtype=float)).copy()
    pr0 = np.zeros(N) if pr_seed is None else np.atleast_1d(np.asarray(pr_seed, dtype=float)).copy()
    conv = np.zeros(N, dtype=bool)
    dead = np.zeros(N, dtype=bool)
    resid = np.full(N, np.inf)
    M = np.zeros((N, 2, 2))
    stats = None
    eps_r = 1e-6
    eps_p = 1e-6 * brho

    for it in range(max_iter):
        R = np.concatenate([r0, r0 + eps_r, r0])
        P = np.concatenate([pr0, pr0, pr0 + eps_p])
        BR = np.concatenate([brho, brho, brho])
        rE, pE, sE, rmean, rmin, rmax, bad, _, rh1 = _integrate(field, R, P, BR, n_steps)
        rE = rE.reshape(3, N)
        pE = pE.reshape(3, N)
        bad = bad.reshape(3, N).any(axis=0)
        Fr = rE[0] - r0
        Fp = pE[0] - pr0
        M[:, 0, 0] = (rE[1] - rE[0]) / eps_r
        M[:, 0, 1] = (rE[2] - rE[0]) / eps_p
        M[:, 1, 0] = (pE[1] - pE[0]) / eps_r
        M[:, 1, 1] = (pE[2] - pE[0]) / eps_p
        stats = (sE[:N], rmean[:N], rmin[:N], rmax[:N], rh1[:N])
        resid = np.hypot(Fr, Fp / brho)
        conv = (np.abs(Fr) < tol_m) & (np.abs(Fp) < tol_m * brho) & ~bad
        dead |= bad
        if verbose:
            print(f'  closed_orbits: iter {it}: converged {conv.sum()}/{N}, '
                  f'max residual {np.nanmax(np.where(dead, 0, resid)):.2e} m, '
                  f'failed {dead.sum()}', flush=True)
        if (conv | dead).all():
            break
        J = M.copy()
        J[:, 0, 0] -= 1.0
        J[:, 1, 1] -= 1.0
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        dr = -(J[:, 1, 1] * Fr - J[:, 0, 1] * Fp) / det
        dp = -(-J[:, 1, 0] * Fr + J[:, 0, 0] * Fp) / det
        scale = np.maximum(1.0, np.maximum(np.abs(dr) / max_step_m,
                                           np.abs(dp) / (0.05 * brho)))
        upd = ~conv & ~dead
        r0 = np.where(upd, r0 + dr / scale, r0)
        pr0 = np.where(upd, pr0 + dp / scale, pr0)

    L, rmean, rmin, rmax, r_h1 = stats
    ok = conv & ~dead
    # A converged fixed point can still be one of the DISPLACED closed orbits
    # that exist near the nu_r = 1 crossing at the pole edge (orbit centre off
    # by several cm, ~30 kHz off in frequency). Flag them: a centred orbit's
    # first radial harmonic is at most the map's own first-harmonic offset
    # (mm level), a displaced one is a few % of r.
    displaced = ok & (r_h1 > displaced_tol * rmean)
    T = L / (beta * CLIGHT)
    nu_r, nu_r_sq = _tune_from_matrix(M, 1.0)

    nu_z = np.full(N, np.nan)
    nu_z_sq = np.full(N, np.nan)
    if vertical and ok.any():
        idx = np.where(ok)[0]
        *_, Z, _ = _integrate(field, r0[idx], pr0[idx], brho[idx], n_steps, vertical=True)
        nz, nz_sq = _tune_from_matrix(Z, 0.0)
        nu_z[idx], nu_z_sq[idx] = nz, nz_sq

    def _nan(a):
        return np.where(ok, a, np.nan)

    return dict(E_MeV=E, brho=brho, gamma=gamma, beta=beta,
                r0=_nan(r0), pr0=_nan(pr0), r_mean=_nan(rmean), r_min=_nan(rmin),
                r_max=_nan(rmax), L=_nan(L), T_rev=_nan(T), f_rev=_nan(1.0 / T),
                B_avg_orbit=_nan(2.0 * np.pi * brho / L),
                nu_r=_nan(nu_r), nu_r_sq=_nan(nu_r_sq), nu_z=nu_z, nu_z_sq=nu_z_sq,
                r_h1=_nan(r_h1), displaced=displaced,
                converged=ok, residual_m=resid)


def orbits_at_radii(field, radii_m, species, energy_seed_mev=None, x_seed=None, *,
                    r0_seed_m=None, pr0_seed=None, radius_kind='mean', n_refine=4,
                    tol_r_m=1e-6, **kw):
    """Closed orbits whose radius matches the requested radii.

    Secant iteration on the energy so that the orbit's mean radius (over
    theta; ``radius_kind='mean'``) or its theta = 0 radius (``'r0'``) equals
    ``radii_m``.  Seeds default to the circle-approximation energy from the
    azimuthal-average field and a scallop estimate r(0) = r (1 + a_N/(N^2-1))
    from the dominant sector harmonic.  Orbits that fail to close from their
    seed are retried with energy/radius seeds extrapolated from converged
    neighbours (needed near the pole edge, where the seeds are far off).

    Warm start: pass ``energy_seed_mev``, ``r0_seed_m`` and ``pr0_seed`` from a
    previous solve at the same radii (e.g. the last optimizer iterate); the
    first Newton solve then converges in 2-3 iterations and the secant needs
    one correction.  NaN entries in the seeds fall back to the cold seeds.

    :returns: the ``closed_orbits`` dict plus ``r_target``; NaN where no
        closed orbit exists.
    """
    radii_m = np.atleast_1d(np.asarray(radii_m, dtype=float))
    key = 'r_mean' if radius_kind == 'mean' else 'r0'
    m0, q = _species_mq(species)

    warm_r0 = None
    if r0_seed_m is not None:
        warm_r0 = np.broadcast_to(np.asarray(r0_seed_m, dtype=float), radii_m.shape).copy()
        if not np.isfinite(warm_r0).all():
            warm_r0 = None if not np.isfinite(warm_r0).any() else warm_r0
    energy_seed_given = energy_seed_mev is not None
    if energy_seed_mev is None or (x_seed is None and warm_r0 is None):
        st = azimuthal_stats(field, radii_m)
        if energy_seed_mev is None:
            n = st['n_dom']
            a = st['a_dom']
            # <B>_path ~ B0 (1 + a x/2 + n^2 x^2/4), x = a/(n^2-1)   (first order)
            x = a / max(n ** 2 - 1.0, 1.0)
            energy_seed_mev = energy_from_brho(np.abs(st['B0']) * radii_m *
                                               (1.0 + 0.5 * a * x + 0.25 * n ** 2 * x ** 2),
                                               species)
        if x_seed is None:
            x_seed = st['a_dom'] / max(st['n_dom'] ** 2 - 1.0, 1.0)
    E = np.atleast_1d(np.asarray(energy_seed_mev, dtype=float)).copy()
    if x_seed is None:
        x_seed = 0.0
    xs = np.broadcast_to(np.asarray(x_seed, dtype=float), radii_m.shape)
    r_seed = radii_m * (1.0 + xs) if key == 'r_mean' else radii_m.copy()
    p_seed = None
    if warm_r0 is not None:
        r_seed = np.where(np.isfinite(warm_r0), warm_r0, r_seed)
        if pr0_seed is not None:
            p_seed = np.broadcast_to(np.asarray(pr0_seed, dtype=float), radii_m.shape).copy()
            p_seed = np.where(np.isfinite(p_seed) & np.isfinite(warm_r0), p_seed, 0.0)

    def _drop_displaced(o):
        for name, val in o.items():
            if isinstance(val, np.ndarray) and val.dtype.kind == 'f' and val.shape == o['displaced'].shape \
                    and name not in ('E_MeV', 'brho', 'gamma', 'beta', 'residual_m'):
                val[o['displaced']] = np.nan
        o['converged'] = o['converged'] & ~o['displaced']
        return o

    # dE/dr along the closed-orbit family: from the seed energies across the
    # radii when they were given (warm start / previous solve), else the
    # fixed-field rigidity slope (exact for a flat field, ~20 % off in an
    # isochronous one -- costs one extra secant step)
    dEdr_seed = None
    if energy_seed_given and len(radii_m) >= 3 and np.all(np.diff(radii_m) > 0):
        g = np.gradient(E, radii_m)
        if np.all(np.isfinite(g)) and np.all(g > 0):
            dEdr_seed = g

    eo = _drop_displaced(closed_orbits(field, E, species, r_seed, pr_seed=p_seed, **kw))
    E_prev, r_prev = None, None
    for _ in range(n_refine):
        ok = np.isfinite(eo[key])
        if not ok.any():
            break
        miss = radii_m - eo[key]
        if np.nanmax(np.abs(miss[ok])) < tol_r_m:
            break
        dEdr = (eo['gamma'] * eo['beta'] ** 2 * (E + m0)) / np.where(ok, eo[key], 1.0)
        if dEdr_seed is not None:
            dEdr = dEdr_seed
        if E_prev is not None:
            sec = (E - E_prev) / (eo[key] - r_prev)
            good = ok & np.isfinite(sec) & (sec > 0)
            dEdr = np.where(good, sec, dEdr)
        E_prev, r_prev = E.copy(), eo[key].copy()
        E_new = np.where(ok, E + dEdr * miss, E)
        seed_r = np.where(ok, eo['r0'] + miss, r_seed)
        seed_p = np.where(ok, eo['pr0'], 0.0)
        eo = _drop_displaced(closed_orbits(field, E_new, species, seed_r, pr_seed=seed_p, **kw))
        E = E_new

    # rescue orbits that never closed: re-seed from the converged neighbours
    for _ in range(3):
        bad = ~np.isfinite(eo[key])
        good = ~bad
        if not bad.any() or good.sum() < 4:
            break
        order = np.argsort(eo[key][good])
        rg, Eg, r0g = eo[key][good][order], E[good][order], eo['r0'][good][order]
        kk = min(12, len(rg))
        cE = np.polyfit(rg[-kk:], Eg[-kk:], 2)
        cR = np.polyfit(rg[-kk:], r0g[-kk:] / rg[-kk:], 1)
        E_b = np.where(radii_m[bad] > rg[-1], np.polyval(cE, radii_m[bad]),
                       np.interp(radii_m[bad], rg, Eg))
        r0_b = radii_m[bad] * np.polyval(cR, radii_m[bad])
        sub = _drop_displaced(closed_orbits(field, E_b, species, r0_b, **kw))
        idx = np.where(bad)[0]
        for name in eo:
            if name in sub and np.ndim(eo[name]) == 1 and len(eo[name]) == len(radii_m):
                eo[name][idx] = sub[name]
        E[idx] = E_b
        okb = np.isfinite(sub[key])
        if okb.any():          # one secant correction for the rescued ones
            j = idx[okb]
            dEdr = (eo['gamma'][j] * eo['beta'][j] ** 2 * (E[j] + m0)) / eo[key][j]
            E2 = E[j] + dEdr * (radii_m[j] - eo[key][j])
            sub2 = _drop_displaced(closed_orbits(field, E2, species,
                                                 eo['r0'][j] + (radii_m[j] - eo[key][j]),
                                                 pr_seed=eo['pr0'][j], **kw))
            for name in eo:
                if name in sub2 and np.ndim(eo[name]) == 1 and len(eo[name]) == len(radii_m):
                    eo[name][j] = sub2[name]
            E[j] = E2
    eo['E_MeV'] = E
    eo['r_target'] = radii_m
    if not np.isfinite(eo[key]).all():
        n_bad = int((~np.isfinite(eo[key])).sum())
        warnings.warn(f'closed_orbit.orbits_at_radii: no closed orbit at {n_bad} of '
                      f'{len(radii_m)} radii (r = '
                      f'{1e3 * radii_m[~np.isfinite(eo[key])].min():.0f}..'
                      f'{1e3 * radii_m[~np.isfinite(eo[key])].max():.0f} mm)',
                      RuntimeWarning, stacklevel=2)
    return eo
