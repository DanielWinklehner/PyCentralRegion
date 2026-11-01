"""Loading 3D bunch-tracking runs for scoring, and saving scores.

``RunData`` holds what ``metrics.score_run`` scores a run on: the recorded tracks of a subset of its bunch, the ghost
record of a ghost run, the run's json summary, the reference particle's orbit, and from its hand-off file the release
phase and the vertical coordinates of every hand-off particle with the hand-off particle of every track.

Two layouts with the same json and pickle contents are read:

track stage
    ``<run_dir>/track.json``, ``<run_dir>/track_data.pkl`` and ``<run_dir>/config.json``, written by ``pcr track``.
    The hand-off file is ``json['inputs']['handoff']``, the loss limits ``json['limits']``, the bunch's sampling seed
    ``config['track']['seed_final']``; the probe files are ``<run_dir>/probe_<a>deg.npz``.
legacy
    ``<results_dir>/bunch_simulation3d<tag>.json`` and ``bunch_simulation3d<tag>_data.pkl`` of the original design
    deck. The hand-off file is the one its log ``<results_dir>/logs/s4_run<tag>.log`` names ("[inflector] hand-off
    <name>.h5"), looked up in results_dir and in the folders given; the probe files are
    ``<results_dir>/probe<tag>_<a>deg.npz``.

The pickle's ``['final']['tracks']``: ids (N,) indices into the tracked bunch; steps (S,) the recorded step numbers;
xyz (S, N, 3) [m] (a slot before birth or after a loss holds whatever the tracker left there); stop_step (N,) the
metal-hit step, n_steps + 1 when there was none (losses at the radius and height limits are not recorded); birth_step
(N,); n_steps; steps_per_turn; E_mev (N,) the final kinetic energy; handoff_ids (N,), when recorded, the hand-off
particle of each track. ``['final']['n']`` is the bunch size and ``['final']['ghosts']`` (ghost runs) the ids that
became ghosts and when. ``['ref'][phase]['traj']`` is the reference particle's orbit at that phase, every step.

The bunch is the whole hand-off file or a sample of it (``handoff.make_beam_from_handoff``: n particles drawn with
the run's seed), so a track id is a hand-off index only for an unsampled bunch. ``handoff_track_ids`` finds the
mapping and checks it: a timed release gives every particle the birth step round((t_rel - t_clock0) / dt), so the
birth steps of the tracks must be their hand-off release times t_rel on one step grid.
"""
import functools
import json
import pickle
import re
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .metrics import alive_mask, ghost_first_step, probe_crossings, rotation_sense, run_constants, survivors, wrap_deg

TRACK_JSON, TRACK_PKL, TRACK_CONFIG = 'track.json', 'track_data.pkl', 'config.json'
LEGACY_JSON, LEGACY_PKL, LEGACY_LOG = 'bunch_simulation3d{}.json', 'bunch_simulation3d{}_data.pkl', 's4_run{}.log'
TRACK_PROBE, LEGACY_PROBE = 'probe_', 'probe{}_'          # probe files: <prefix><azimuth>deg.npz
_HANDOFF_IN_LOG = re.compile(r"\[inflector\] hand-off (\S+\.h5)")
_PROBE_ANGLE = r"(-?\d+(?:\.\d+)?)deg\.npz"
MIN_BIRTH_SPREAD = 8         # birth steps spanning fewer steps cannot tell a right hand-off mapping from a wrong one


@dataclass
class RunData:
    tag: str
    ids: np.ndarray                      # (N,) indices into the tracked bunch
    steps: np.ndarray                    # (S,) recorded step numbers
    xyz: np.ndarray                      # (S, N, 3) [m]
    stop_step: np.ndarray                # (N,) metal-hit step; n_steps + 1 when none
    birth_step: np.ndarray               # (N,)
    n_steps: int
    steps_per_turn: int
    E_mev: np.ndarray                    # (N,) final kinetic energy
    ghost_ids: Optional[np.ndarray] = None    # ghost runs: ids that became ghosts, and when
    ghost_step: Optional[np.ndarray] = None
    info: dict = field(default_factory=dict)  # the run json
    sc_recorded: bool = False            # the pickle's final record carries a space-charge log
    pkl_final: dict = field(default_factory=dict)   # scalars of the pickle's final record (phase, n, transmission)
    handoff: str = ''                    # hand-off file
    beam: str = ''                       # label of the injected beam (beam_label; default: the hand-off file's stem)
    handoff_ids: Optional[np.ndarray] = None        # (N,) the hand-off particle of each track; None when unknown
    release_phase_deg: Optional[np.ndarray] = None  # per hand-off particle, wrapped to +-180
    handoff_z_m: Optional[np.ndarray] = None        # per hand-off particle: z and z' = p_z / p_perp at the plane
    handoff_zp_rad: Optional[np.ndarray] = None
    handoff_meta: dict = field(default_factory=dict)
    ref_xyz: Optional[np.ndarray] = None            # (S_ref, 3) [m] the reference particle's orbit, every step
    probe_files: dict = field(default_factory=dict)  # {azimuth [deg]: the tracker's probe file}
    notes: list = field(default_factory=list)       # what could not be loaded or checked, and why
    source: str = ''                     # the data file

    @property
    def n_tracks(self):
        return len(self.ids)

    @property
    def ghost_run(self):
        return self.ghost_ids is not None

    def of_tracks(self, per_particle):
        """A per-hand-off-particle array at the tracks' hand-off particles; None when either is missing."""
        if per_particle is None or self.handoff_ids is None:
            return None
        return per_particle[self.handoff_ids]


# ---------------------------------------------------------------------------------------------------------------------
Handoff = namedtuple('Handoff', 'phase_deg t_rel_s z_m zp_rad meta')


@functools.lru_cache(maxsize=8)
def handoff_particles(path):
    """``Handoff`` (release phase [deg] wrapped to +-180, release time t_rel [s], z [m], z' = p_z / p_perp, meta) of
    every particle of a hand-off file. t_rel as ``make_beam_from_handoff`` takes it (the file's t_rel, else time minus
    the phase reference time, else none); the phase is the file's per-particle 'phase' [rad] (2 pi f_rf t_rel), else
    360 f_rf t_rel. Cached: a campaign scores many runs of one hand-off; the arrays are read-only."""
    from ..handoff import load_handoff
    d = load_handoff(path)
    meta = dict(d['meta'])
    t_rel = d.get('t_rel')
    if t_rel is None and d.get('time') is not None and np.ptp(d['time']) > 0:
        t_rel = np.asarray(d['time']) - float(meta.get('phase_reference_time_s', np.mean(d['time'])))
    if d.get('phase') is not None:
        ph = np.degrees(d['phase'])
    elif t_rel is not None and meta.get('rf_frequency_hz'):
        ph = 360.0 * float(meta['rf_frequency_hz']) * np.asarray(t_rel)
    else:
        raise ValueError(f"{path}: no per-particle phase or release time")
    p = np.asarray(d['p_ev_c'])
    arrays = [wrap_deg(ph), None if t_rel is None else np.asarray(t_rel, dtype=float).copy(),
              np.asarray(d['r'])[:, 2].copy(), p[:, 2] / np.hypot(p[:, 0], p[:, 1])]
    for a in arrays:
        if a is not None:
            a.flags.writeable = False
    return Handoff(*arrays, meta)


def _release_matches(birth_step, t_rel):
    """Whether birth steps are the release times t_rel on one step grid, birth = round((t_rel - t0) / dt): every step
    within one of a straight line in t_rel (rounding leaves half a step; a wrong mapping scatters by the bunch
    length)."""
    b = np.asarray(birth_step, dtype=float)
    k = np.polyfit(t_rel, b, 1)
    return bool(k[0] > 0 and np.max(np.abs(b - np.polyval(k, t_rel))) <= 1.0)


def handoff_track_ids(ids, birth_step, n_bunch, n_handoff, t_rel=None, seed=None, recorded=None):
    """(hand-off index of every track or None, note). Candidates in turn: the recorded mapping; the track ids
    themselves when the bunch is the whole hand-off file; the bunch drawn as ``make_beam_from_handoff`` draws it,
    np.sort(default_rng(seed).choice(n_handoff, n_bunch, replace=False)), indexed by the track ids. The first whose
    hand-off release times match the tracks' birth steps (``_release_matches``) is taken; when the birth steps cannot
    tell, the first candidate is taken with a note."""
    ids = np.asarray(ids, dtype=np.int64)
    cands = []
    if recorded is not None:
        cands.append(('recorded', np.asarray(recorded, dtype=np.int64)))
    if not n_bunch or n_bunch >= n_handoff:
        cands.append(('track ids', ids))
    elif seed is not None:
        idx = np.sort(np.random.default_rng(int(seed)).choice(n_handoff, int(n_bunch), replace=False))
        cands.append((f"bunch of {n_bunch} drawn with seed {seed}", idx[ids] if ids.max() < len(idx) else None))
    cands = [(name, hid) for name, hid in cands if hid is not None and len(hid) == len(ids) and
             (not len(hid) or hid.max() < n_handoff)]
    if not cands:
        return None, (f"no mapping of the {len(ids)} tracks (bunch of {n_bunch}) onto the {n_handoff} hand-off "
                      f"particles: release-phase metrics skipped")
    if t_rel is None:
        return cands[0][1], (f"hand-off index from the {cands[0][0]}, not checked (the hand-off file has no release "
                             f"times)")
    if len(ids) < 2 or np.ptp(birth_step) < MIN_BIRTH_SPREAD:
        return cands[0][1], (f"hand-off index from the {cands[0][0]}, not checked (birth steps span fewer than "
                             f"{MIN_BIRTH_SPREAD} steps)")
    for name, hid in cands:
        if _release_matches(birth_step, t_rel[hid]):
            return hid, None
    return None, ("the tracks' birth steps do not match the hand-off release times for any mapping ("
                  + ', '.join(n for n, _ in cands) + "): release-phase metrics skipped")


def _reference_orbit(data, phase):
    """The reference particle's orbit (S, 3) [m] at the final phase from the pickle's 'ref', None when missing."""
    ref = data.get('ref') if isinstance(data, dict) else None
    if not ref:
        return None
    r = ref.get(phase)
    if r is None:
        r = next((v for k, v in ref.items() if phase is not None and np.isclose(float(k), float(phase))), None)
    if r is None and len(ref) == 1:
        r = next(iter(ref.values()))
    tr = None if r is None else r.get('traj')
    return None if tr is None or len(tr) == 0 else np.asarray(tr, dtype=float)[:, :3]


def _probe_files(folder, prefix):
    """{azimuth: file} of the probe files <prefix><azimuth>deg.npz in folder (exactly: the files of a run whose tag
    extends this one's are not taken)."""
    name = re.compile(re.escape(prefix) + _PROBE_ANGLE)
    out = {}
    for f in sorted(Path(folder).glob(prefix + '*deg.npz')):
        m = name.fullmatch(f.name)
        if m:
            out[float(m.group(1))] = str(f)
    return out


def _run_from_pickle(pkl, info, tag, handoff, beam, seed, probe_files):
    with open(pkl, 'rb') as fh:
        data = pickle.load(fh)
    fin = data.get('final', data) if isinstance(data, dict) else {}
    tr = fin.get('tracks') or {}
    arr = (lambda key, dtype, empty: np.asarray(tr.get(key, empty), dtype=dtype))
    gh = fin.get('ghosts')
    run = RunData(tag=tag, ids=arr('ids', np.int64, []), steps=arr('steps', np.int64, []),
                  xyz=arr('xyz', float, np.zeros((0, 0, 3))), stop_step=arr('stop_step', np.int64, []),
                  birth_step=arr('birth_step', np.int64, []), n_steps=int(tr.get('n_steps', 0)),
                  steps_per_turn=int(tr.get('steps_per_turn', 1)), E_mev=arr('E_mev', float, []),
                  ghost_ids=None if gh is None else np.asarray(gh['ids'], dtype=np.int64),
                  ghost_step=None if gh is None else np.asarray(gh['step'], dtype=np.int64), info=info,
                  sc_recorded=fin.get('space_charge') is not None,
                  pkl_final={k: fin[k] for k in ('phase', 'n', 'transmission', 'ghost_transmission') if k in fin},
                  ref_xyz=_reference_orbit(data, fin.get('phase')), probe_files=probe_files, source=str(pkl))
    del data, fin
    if run.ref_xyz is None:
        run.notes.append("no reference orbit in the pickle: best_slit gives absolute slit centres")
    if handoff and Path(handoff).exists():
        run.handoff = str(handoff)
        h = handoff_particles(str(Path(handoff).resolve()))
        run.release_phase_deg, run.handoff_z_m, run.handoff_zp_rad, run.handoff_meta = h.phase_deg, h.z_m, h.zp_rad, \
            h.meta
        run.handoff_ids, note = handoff_track_ids(run.ids, run.birth_step, int(run.pkl_final.get('n') or 0),
                                                  len(h.phase_deg), h.t_rel_s, seed, tr.get('handoff_ids'))
        if note:
            run.notes.append(note)
    else:
        run.notes.append(f"hand-off file {'not found: ' + str(handoff) if handoff else 'unknown'}: release-phase "
                         f"metrics skipped")
    run.beam = beam_label(beam, handoff)
    return run


def beam_label(beam, handoff):
    """The label of a run's injected beam: beam when it is a string; when it is a mapping, its entry for the hand-off
    file's name or stem (e.g. {'handoff_sp1Lo_R10_centroid.h5': 'sp1Lo_c1'}, for runs whose hand-off a log names);
    else the hand-off file's stem."""
    if isinstance(beam, dict):
        beam = (beam.get(Path(handoff).name) or beam.get(Path(handoff).stem) or '') if handoff else ''
    return beam or (Path(handoff).stem if handoff else '')


def _sampling_seed(config):
    """The final run's sampling seed from a resolved run configuration, else the tracker's default."""
    from ..config import TrackConfig
    return int((config.get('track') or {}).get('seed_final', TrackConfig().seed_final))


def load_track_run(run_dir, handoff=None, beam='', tag=None):
    """A ``pcr track`` run folder. The hand-off file: the argument, else json['inputs']['handoff'] (relative paths
    against the folder), else none (the release-phase metrics are then skipped). beam: a label or a mapping
    (``beam_label``). The tag defaults to the folder's name: the runs of a campaign share their configured name."""
    run_dir = Path(run_dir)
    info = json.loads((run_dir / TRACK_JSON).read_text()) if (run_dir / TRACK_JSON).exists() else {}
    config = json.loads((run_dir / TRACK_CONFIG).read_text()) if (run_dir / TRACK_CONFIG).exists() else {}
    handoff = handoff or (info.get('inputs') or {}).get('handoff') or None
    if handoff and not Path(handoff).is_absolute() and (run_dir / handoff).exists():
        handoff = run_dir / handoff
    return _run_from_pickle(run_dir / TRACK_PKL, info, tag or run_dir.resolve().name, handoff, beam,
                            _sampling_seed(config), _probe_files(run_dir, TRACK_PROBE))


def legacy_handoff(results_dir, tag, handoff_dirs=()):
    """The hand-off file a legacy run's log names, found in results_dir or one of handoff_dirs; None otherwise."""
    log = Path(results_dir) / 'logs' / LEGACY_LOG.format(tag)
    if not log.exists():
        return None
    with open(log, 'r', errors='ignore') as fh:
        m = _HANDOFF_IN_LOG.search(fh.read(400_000))
    if m is None:
        return None
    for folder in (results_dir, *handoff_dirs):
        f = Path(folder) / m.group(1)
        if f.exists():
            return f
    return None


def load_legacy_run(results_dir, tag, handoff=None, handoff_dirs=(), beam='', seed=None):
    """A run of the original design deck in results_dir. The hand-off file: the argument, else the one its log names
    (``legacy_handoff``). beam: a label or a mapping from hand-off file to label (``beam_label``). seed: the bunch's
    sampling seed (the deck records none; default the tracker's)."""
    results_dir = Path(results_dir)
    js = results_dir / LEGACY_JSON.format(tag)
    info = json.loads(js.read_text()) if js.exists() else {}
    handoff = handoff or legacy_handoff(results_dir, tag, handoff_dirs)
    return _run_from_pickle(results_dir / LEGACY_PKL.format(tag), info, tag, handoff, beam,
                            _sampling_seed({}) if seed is None else int(seed),
                            _probe_files(results_dir, LEGACY_PROBE.format(tag)))


def load_run(path, tag=None, **kw):
    """A run in either layout: a ``pcr track`` folder (track.json / track_data.pkl), or a legacy results folder and
    the run's tag. Keyword arguments go to the loader (handoff, beam; handoff_dirs and seed for the legacy layout)."""
    path = Path(path)
    if (path / TRACK_PKL).exists():
        return load_track_run(path, tag=tag, **kw)
    if tag is not None and (path / LEGACY_PKL.format(tag)).exists():
        return load_legacy_run(path, tag, **kw)
    raise FileNotFoundError(f"{path}: neither {TRACK_PKL} nor {LEGACY_PKL.format(tag or '<tag>')}")


def legacy_tags(results_dir):
    """Tags of every legacy run in results_dir that has a data pickle."""
    a, b = LEGACY_PKL.split('{}')
    return sorted(p.name[len(a):-len(b)] for p in Path(results_dir).glob(LEGACY_PKL.format('*')))


# ---------------------------------------------------------------------------------------------------------------------
# probes: the tracker's files, and the same built from recorded tracks
def load_probe(path):
    """A tracker probe file (every particle of the bunch) as forward crossings in the format of
    ``metrics.probe_crossings`` (j = bunch particle, pass, R and z [mm], t [s]) plus survived (per bunch particle), the
    final energy E_mev at the crossing and the azimuth: for ``metrics.recorded_probe_turns`` and
    ``metrics.compare_probes``."""
    with np.load(path, allow_pickle=False) as d:
        fw = np.asarray(d['direction']) > 0
        xyz = np.asarray(d['xyz'])[fw]
        return {'j': np.asarray(d['id'])[fw].astype(np.int64), 'pass': np.asarray(d['turn'])[fw],
                'R': 1e3 * np.hypot(xyz[:, 0], xyz[:, 1]), 'z': 1e3 * xyz[:, 2], 't': np.asarray(d['t'])[fw],
                'E_mev': np.asarray(d['E_mev'])[fw], 'survived': np.asarray(d['survived'], dtype=bool),
                'angle_deg': float(d['angle_deg'])}


def virtual_probe(run, angle_deg, cfg=None):
    """A probe at any azimuth built after the run from its recorded tracks, in the layout of the tracker's probe files
    (``probe_<a>deg.npz``: xyz, vxyz, t, id, turn, direction, step, E_mev, survived, angle_deg, sense, n,
    steps_per_turn, and particle_ids, the bunch particle of each track), every crossing forward and back; id is the
    track index. The crossing rules are the tracker's (``metrics.probe_crossings``); the velocity is the chord over
    the record interval, so E_mev is that of the mean velocity between two records (np.savez it to write the file)."""
    from ..simulate3d import kinetic_energy_mev
    from .objectives import ScoreConfig
    cfg = cfg or ScoreConfig()
    k = run_constants(run, cfg)
    steps = np.asarray(run.steps)
    gstep = ghost_first_step(run.ids, run.ghost_ids, run.ghost_step)
    alive = alive_mask(steps, run.xyz, run.birth_step, run.stop_step, gstep)
    sense = rotation_sense(run.xyz, alive)
    pr = probe_crossings(run.xyz, alive, steps, 1.0 / k['f_rev'] / run.steps_per_turn, sense, angle_deg,
                         forward_only=False)
    mass = run.handoff_meta.get('species_mass_mev')
    E = kinetic_energy_mev(pr['v'], float(mass)) if mass is not None and len(pr['j']) else np.full(len(pr['j']),
                                                                                                    np.nan)
    surv = survivors(run.xyz[-1], run.stop_step, run.n_steps, gstep, k['r_max'], k['z_max'])
    return {'xyz': pr['xyz'], 'vxyz': pr['v'], 't': pr['t'], 'id': pr['j'].astype(np.int32),
            'turn': pr['pass'].astype(np.int32), 'direction': pr['direction'], 'step': pr['step'].astype(np.int32),
            'E_mev': E, 'survived': surv, 'angle_deg': np.float64(angle_deg), 'sense': np.float64(sense),
            'n': np.int64(run.n_tracks), 'steps_per_turn': np.int64(run.steps_per_turn), 'particle_ids': run.ids}


# ---------------------------------------------------------------------------------------------------------------------
def clean(o):
    """A score as plain JSON data: numpy scalars and arrays to Python, NaN and inf to None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def save_score(score, path):
    """Write a score as JSON (``clean``), atomically: a reader never sees a half-written file."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(clean(score), indent=1))
    tmp.replace(path)


def load_score(path):
    with open(path) as fh:
        return json.load(fh)


def save_table(scores, path, columns):
    """A CSV of scores, one row per run: columns are dotted score paths (``config.get_dotted``), empty where missing."""
    import csv
    from ..config import get_dotted
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for s in scores:
            row = []
            for c in columns:
                v = get_dotted(s, c)
                row.append('' if v is None or isinstance(v, (dict, list)) else v)
            w.writerow(row)
