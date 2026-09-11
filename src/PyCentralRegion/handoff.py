"""
handoff.py - Spiral-inflector hand-off bunches as initial beams.

Reads the openPMD plane-crossing files written by the HCHC-60 spiral-inflector
deck (spec version 1, 2026-09-07: ``BunchTrack.py --save-openpmd``, see
``Spiral_inflector/Docs/openpmd_plane_crossing_handoff.md``) and turns them
into a ``ParticleDistribution`` for ``AcceleratedOrbitFinder``.

Two launch modes
----------------
timed (default)
    Every particle starts ON the plane where it was recorded and is released
    at its own crossing time ``t_rel`` relative to the bunch centre
    (``tracking.TimedRelease``, driven by ``ParticleDistribution.birth_time``).
    Exact: no assumption about the field between the plane and anywhere else.
snapshot
    Common-time bunch, the old way: every particle is moved along its own
    orbit circle by ``-t_rel`` (uniform central field, cyclotron frequency
    ``omega_c = 2 pi f_rf / harmonic`` unless given), all born at t = 0.

The reference particle - prepended by the finder as the virtual particle 0,
born at t = 0, i.e. at the bunch centre's crossing time - is the DESIGN
particle at the plane origin, travelling along the plane normal at the
design speed (``reference='design'``), or the bunch centroid (mean position,
mean direction, mean speed; ``'centroid'``). The finder's ``bunch_phase_deg``
is therefore the RF phase at which the bunch CENTRE crosses the plane, the
``phi_0`` of the spec, whichever particle is released first.

Frame: the deck frame is the machine frame up to a rotation about z
(``rotation_deg`` - where the exit sits relative to the dees is a design
choice of the central region) and, if the deck's z axis points the other
way, ``flip_z``. Units SI throughout; momenta in the file are eV/c.
"""
import warnings
from typing import Optional, Union
import numpy as np

from PyPATools.particles import ParticleDistribution
from PyPATools.species import IonSpecies

CLIGHT = 2.99792458e8
_PREFIX = "PyPATools:"


def _attr(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return v.astype(float) if v.dtype.kind in 'fiu' else v
    if isinstance(v, np.generic):
        return v.item()
    return v


def load_handoff(path) -> dict:
    """Raw content of a hand-off file (plane-crossing or lab6d snapshot).

    Returns dict(r (N,3) m, p_ev_c (N,3), time (N,) s or None, weight (N,) C
    or None, meta {PyPATools attributes without the prefix}, and any of
    phase / t_rel / u / v that the file carries)."""
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        pp = f.attrs.get("particlesPath", "particles")
        pp = pp.decode() if isinstance(pp, bytes) else str(pp)
        sp = f[pp]
        g = sp[list(sp.keys())[0]]
        meta = {k[len(_PREFIX):]: _attr(v) for k, v in g.attrs.items() if k.startswith(_PREFIX)}
        meta['species_group'] = list(sp.keys())[0]
        meta['numParticles'] = int(g.attrs.get('numParticles', 0))

        def rec(name):
            obj = g[name]
            if isinstance(obj, h5py.Dataset):
                return obj[...].astype(float)
            n = int(np.asarray(obj.attrs["shape"]).ravel()[0])
            return np.full(n, float(obj.attrs["value"]))

        out['r'] = np.column_stack([rec("position/" + c) for c in "xyz"])
        out['p_ev_c'] = np.column_stack([rec("momentum/" + c) for c in "xyz"])
        out['time'] = rec("time") if "time" in g else None
        out['weight'] = rec("weight") if "weight" in g else None
        for k in ("phase", "t_rel", "u", "v"):
            if k in g:
                out[k] = rec(k)
    out['meta'] = meta
    out['path'] = str(path)
    return out


def _rot(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_beam_from_handoff(source: Union[str, dict], species: Optional[IonSpecies] = None,
                           rotation_deg: float = 0.0, launch: str = 'timed',
                           reference: str = 'design', n_particles: Optional[int] = None,
                           seed: int = 0, flip_z: bool = False, harmonic: int = 4,
                           omega_c: Optional[float] = None, verbose: bool = True) -> ParticleDistribution:
    """Initial beam from a hand-off file (path) or ``load_handoff`` dict.

    The returned distribution carries, beyond positions and momenta:
      birth_time      (N,) s, relative to the bunch centre (zeros for 'snapshot')
      reference_state (x (3,) m, v (3,) m/s) of the reference particle at t = 0
      macro_charge_c  charge per macro-particle after sub-sampling
      handoff_meta    provenance (file, mode, rotation, counts, plane, t_ref)
    ``AcceleratedOrbitFinder`` honours all of them.
    """
    if launch not in ('timed', 'snapshot'):
        raise ValueError("launch must be 'timed' or 'snapshot'")
    if reference not in ('design', 'centroid'):
        raise ValueError("reference must be 'design' or 'centroid'")
    d = source if isinstance(source, dict) else load_handoff(source)
    meta = d['meta']
    mode = str(meta.get('mode', 'plane_crossing'))
    if species is None:
        species = IonSpecies(str(meta.get('pypatools_species_name', 'H2_1+')))
    m_ev = meta.get('species_mass_mev')
    m_ev = float(m_ev) * 1e6 if m_ev is not None else getattr(species, 'mass_mev', None)
    if m_ev is None:
        raise ValueError("species mass unknown: the file has no species_mass_mev and the species no mass_mev")

    r = np.array(d['r'], dtype=float)
    p = np.array(d['p_ev_c'], dtype=float)
    n_total = len(r)
    e_tot = np.sqrt((p ** 2).sum(1) + m_ev ** 2)
    v = p / e_tot[:, None] * CLIGHT
    if d.get('t_rel') is not None:
        t_rel = np.array(d['t_rel'], dtype=float)
    elif d.get('time') is not None and np.ptp(d['time']) > 0:
        t_ref = float(meta.get('phase_reference_time_s', np.mean(d['time'])))
        t_rel = np.asarray(d['time'], dtype=float) - t_ref
    else:
        t_rel = np.zeros(n_total)
        if mode != 'plane_crossing':
            warnings.warn(f"{mode}: common-time snapshot, every particle is born at t = 0", stacklevel=2)
    weight = d.get('weight')
    w_mean = float(np.mean(weight)) if weight is not None else float(meta.get('bunch_charge_c', 0.0)) / max(n_total, 1)

    idx = np.arange(n_total)
    if n_particles is not None and n_particles < n_total:
        idx = np.sort(np.random.default_rng(seed).choice(n_total, int(n_particles), replace=False))
    r, v, t_rel = r[idx], v[idx], t_rel[idx]
    n = len(idx)

    # ---- reference particle (before the frame change, transformed with the rest)
    plane_o = meta.get('plane_origin_m')
    plane_n = meta.get('plane_normal')
    v_design = meta.get('design_exit_velocity_mps')
    if reference == 'design' and (plane_o is None or plane_n is None or v_design is None):
        warnings.warn("no plane / design attributes in the file: using the bunch centroid as reference", stacklevel=2)
        reference = 'centroid'
    if reference == 'design':
        x_ref = np.asarray(plane_o, dtype=float)
        nrm = np.asarray(plane_n, dtype=float)
        v_ref = nrm / np.linalg.norm(nrm) * float(np.linalg.norm(v_design))
    else:
        x_ref = r.mean(axis=0)
        speed = np.linalg.norm(v, axis=1)
        dir_mean = (v / speed[:, None]).mean(axis=0)
        v_ref = dir_mean / np.linalg.norm(dir_mean) * float(speed.mean())

    # ---- frame: rotation about z, optional z flip
    R = _rot(rotation_deg)
    r, v = r @ R.T, v @ R.T
    x_ref, v_ref = R @ x_ref, R @ v_ref
    if flip_z:
        r[:, 2] *= -1.0
        v[:, 2] *= -1.0
        x_ref[2] *= -1.0
        v_ref[2] *= -1.0

    # ---- snapshot: advance every particle along its own orbit circle by -t_rel
    birth = t_rel.copy()
    if launch == 'snapshot':
        f_rf = float(meta.get('rf_frequency_hz', 0.0))
        if omega_c is None:
            if f_rf <= 0:
                raise ValueError("snapshot launch needs omega_c (no rf_frequency_hz in the file)")
            omega_c = 2.0 * np.pi * f_rf / float(harmonic)
        sense = np.sign(np.mean(r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0])) or 1.0     # +1: counter-clockwise
        speed = np.hypot(v[:, 0], v[:, 1])
        rho = speed / omega_c
        n_hat = np.column_stack([v[:, 1], -v[:, 0]]) / speed[:, None]              # right of travel
        centre = r[:, :2] - sense * rho[:, None] * n_hat                           # centre is left of travel for CCW
        ang = sense * omega_c * (-t_rel)
        c, s = np.cos(ang), np.sin(ang)
        rel = r[:, :2] - centre
        r[:, 0] = centre[:, 0] + c * rel[:, 0] - s * rel[:, 1]
        r[:, 1] = centre[:, 1] + s * rel[:, 0] + c * rel[:, 1]
        vx, vy = v[:, 0].copy(), v[:, 1].copy()
        v[:, 0] = c * vx - s * vy
        v[:, 1] = s * vx + c * vy
        r[:, 2] += v[:, 2] * (-t_rel)
        birth = np.zeros(n)

    pd = ParticleDistribution(species=species, x_vec=r.copy(), p_vec=np.zeros_like(r))
    pd.set_p_from_v_vec(v.copy())
    pd.birth_time = birth
    pd.reference_state = (x_ref, v_ref)
    pd.macro_charge_c = w_mean * n_total / n
    pd.handoff_meta = {
        'file': d.get('path'), 'mode': mode, 'launch': launch, 'reference': reference,
        'rotation_deg': float(rotation_deg), 'flip_z': bool(flip_z),
        'n_particles': int(n), 'n_in_file': int(n_total), 'seed': int(seed),
        'rf_frequency_hz': meta.get('rf_frequency_hz'), 't_ref_s': meta.get('phase_reference_time_s'),
        'plane_origin_m': x_ref.tolist() if reference == 'design' else None,
        'birth_time_rms_s': float(np.std(birth)), 'birth_time_range_s': [float(birth.min()), float(birth.max())],
    }
    if verbose:
        ekin = (e_tot[idx] - m_ev) * 1e-3
        print(f"[handoff] {n} of {n_total} particles ({mode}, launch {launch}, reference {reference}, "
              f"rotation {rotation_deg:+.1f} deg): E = {ekin.mean():.2f} keV (rms {100 * ekin.std() / ekin.mean():.2f}%), "
              f"birth times {birth.min() * 1e9:+.1f}..{birth.max() * 1e9:+.1f} ns (rms {np.std(birth) * 1e9:.2f}), "
              f"reference at r = {np.hypot(x_ref[0], x_ref[1]) * 1e3:.1f} mm, az = "
              f"{np.degrees(np.arctan2(x_ref[1], x_ref[0])):.1f} deg, |v| = {np.linalg.norm(v_ref):.4e} m/s")
    return pd
