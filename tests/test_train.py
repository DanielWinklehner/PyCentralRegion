"""Bunch trains (track.bunches_ahead / bunches_behind): the train a bunch is put in (handoff.make_train: release
times, labels, charge weights), the run's clock and records with companions (AcceleratedOrbitFinder.track_with_rf on
a stub engine: the own bunch's turns, statistics and beam record are those of the bunch alone), the space charge of a
train on one mesh (tracking.SpaceChargeKick with the FFT solver on the CPU: charge per bunch, two displaced bunches
against the closed form, the weightless control, momentum balance) and the train's mesh and table in simulate3d."""
import dataclasses
import math
import types

import numpy as np
import pytest

pytest.importorskip('PyPATools')

from PyPATools.particles import ParticleDistribution                                          # noqa: E402
from PyPATools.species import IonSpecies                                                      # noqa: E402

from PyCentralRegion import config as C                                                       # noqa: E402
from PyCentralRegion import simulate3d as S                                                   # noqa: E402
from PyCentralRegion.accelerated_orbit_finder import AcceleratedOrbitFinder                   # noqa: E402
from PyCentralRegion.handoff import TRAIN_EXTRAS, make_train, train_period_s                  # noqa: E402
from PyCentralRegion.tracking import SpaceChargeKick, TrackingResult                          # noqa: E402

K_E = 8.9875517923e9
SPECIES = IonSpecies('H2_1+')


def _bunch(n=5, seed=0):
    """A small bunch with what a hand-off beam carries: birth times, reference, macro-charge, hand-off indices."""
    rng = np.random.default_rng(seed)
    pd = ParticleDistribution(species=SPECIES, x_vec=rng.normal(size=(n, 3)) * 1e-3,
                              p_vec=rng.normal(size=(n, 3)) * 1e-3)
    pd.birth_time = rng.uniform(-8e-9, 12e-9, n)
    pd.reference_state = (np.array([0.06, 0.0, 0.0]), np.array([0.0, 2.5e6, 0.0]))
    pd.macro_charge_c = 2.5e-13
    pd.handoff_meta = {'rf_frequency_hz': 32.8e6}
    pd.handoff_index = np.arange(100, 100 + n)
    return pd


# ---------------------------------------------------------------------------------------------------------------------
# the train
def test_train_period_is_harmonic_rf_periods():
    """One bunch every `harmonic` RF periods: one revolution, whatever the harmonic."""
    f_rev = 8.191375e6
    for h in (1, 4, 6):
        assert train_period_s(h, h * f_rev) == pytest.approx(1.0 / f_rev, rel=1e-15)
        assert train_period_s(h, 32.8e6) == pytest.approx(h / 32.8e6, rel=1e-15)
    for bad in ((0, 32.8e6), (4, 0.0), (4, -1.0)):
        with pytest.raises(ValueError):
            train_period_s(*bad)


def test_make_train_release_times_labels_and_weights():
    """Bunch k is the bunch released k periods later: rows of bunch 0 first and untouched, then -1 .. -n_ahead, then
    1 .. n_behind; the reference, macro-charge and hand-off record stay one bunch's."""
    b = _bunch(5)
    period = train_period_s(4, 4 * 8.191375e6)
    t = make_train(b, 2, 1, period, charge=0.5)
    labels = [0, -1, -2, 1]
    assert t.numpart == 20 and t.n_scored == 5 and t.train['labels'] == labels
    assert np.array_equal(t.bunch_of, np.repeat(labels, 5))
    assert np.array_equal(t.x_vec[:5], b.x_vec) and np.array_equal(t.p_vec[:5], b.p_vec)       # bunch 0 bit for bit
    assert np.array_equal(t.birth_time[:5], b.birth_time)
    for j, k in enumerate(labels):
        rows = slice(5 * j, 5 * j + 5)
        assert np.array_equal(t.x_vec[rows], b.x_vec) and np.array_equal(t.p_vec[rows], b.p_vec)
        assert np.allclose(t.birth_time[rows] - b.birth_time, k * period, rtol=0.0, atol=1e-21)
        assert np.array_equal(t.handoff_index[rows], b.handoff_index)
        assert np.all(t.charge_weight[rows] == (1.0 if k == 0 else 0.5))
    assert t.reference_state is b.reference_state and t.macro_charge_c == b.macro_charge_c
    assert all(hasattr(t, name) for name in TRAIN_EXTRAS)
    # on the tracker's step grid a period is a whole number of steps: every bunch is released on the same grid
    spt = 1000
    dt = period / spt
    step = np.rint((t.birth_time - t.birth_time.min()) / dt).astype(int)
    own = step[:5]
    for j, k in enumerate(labels):
        assert np.array_equal(step[5 * j:5 * j + 5], own + k * spt)
    alone = make_train(b, 0, 0, period)
    assert alone.numpart == 5 and np.all(alone.charge_weight == 1.0) and alone.train['labels'] == [0]
    for bad in ((-1, 0, period, 1.0), (1, 0, 0.0, 1.0), (1, 0, period, -0.1)):
        with pytest.raises(ValueError):
            make_train(b, *bad)


def test_train_of_follows_the_config():
    """BunchTracker.train_of: the bunch itself by default; with bunches ahead or behind, a train at the period of
    machine.harmonic tracking-RF periods (one revolution) and the charge of track.train_charge."""
    b = _bunch(4)
    f_mhz = 8.191375

    def tracker(harmonic=4, **track):
        return types.SimpleNamespace(tc=dataclasses.replace(C.TrackConfig(), **track),
                                     mc=types.SimpleNamespace(harmonic=harmonic), f_mhz=f_mhz)
    assert S.BunchTracker.train_of(tracker(), b) is b
    for h in (4, 6):
        t = S.BunchTracker.train_of(tracker(harmonic=h, bunches_ahead=1, bunches_behind=2, train_charge=0.0), b)
        assert t.train['labels'] == [0, -1, 1, 2] and t.train['period_s'] == pytest.approx(1e-6 / f_mhz, rel=1e-14)
        assert np.all(t.charge_weight[4:] == 0.0) and np.all(t.charge_weight[:4] == 1.0)
    with pytest.raises(C.ConfigError):
        S.BunchTracker.train_of(tracker(bunches_ahead=-1), b)


# ---------------------------------------------------------------------------------------------------------------------
# the run's clock and records (the real callback on a stub engine, as in test_reference_clock)
PERIOD, SPT, DT, R, PHI0 = 103, 100, 1e-9, 0.1, 0.3


class StubEngine:
    """Every particle on the circle of radius R, PERIOD steps per revolution from its own birth step on (frozen at
    its launch azimuth before); ``kill``: {row: first step at which it is inactive}."""
    algorithm = 'rk4'

    def __init__(self, kill=None):
        self.kill = dict(kill or {})
        self.extra_interactions = []
        self.steps_run = None
        self.birth = None

    def track_multiparticle(self, pd_init, dt, n_steps, callback, callback_frequency=1, show_progress=False, t0=0.0):
        n = int(pd_init.numpart)
        phi0 = np.arctan2(pd_init.x_vec[:, 1], pd_init.x_vec[:, 0])
        rel = [e for e in self.extra_interactions if hasattr(e, 'birth_step')]
        birth = self.birth = np.asarray(rel[0].birth_step) if rel else np.zeros(n, dtype=int)
        active = np.asarray(pd_init.alive, dtype=bool).copy()
        omega = 2 * np.pi / (PERIOD * dt)
        for step in range(n_steps):
            active |= birth <= step
            for row, s in self.kill.items():
                if step >= s:
                    active[row] = False
            phi = phi0 + 2 * np.pi * np.maximum(step + 1 - birth, 0) / PERIOD
            r = R * np.column_stack([np.cos(phi), np.sin(phi), np.zeros(n)])
            v = R * omega * np.column_stack([-np.sin(phi), np.cos(phi), np.zeros(n)])
            t = t0 + (step + 1) * dt
            if not active.any():
                self.steps_run = step + 1
                return TrackingResult(False, step + 1, r, v, active, {'termination': 'all_lost', 'time': t})
            if callback(step, r, v, active.copy(), t):
                self.steps_run = step + 1
                return TrackingResult(True, step + 1, r, v, active, {'termination': 'turns_or_energy_reached',
                                                                     'time': t})
        self.steps_run = n_steps
        return TrackingResult(True, n_steps, r, v, active, {'time': t0 + n_steps * dt})


OWN_BIRTH = np.array([-10 * DT, -5 * DT])           # the own bunch's two particles; the reference is born at 0


def _run(labels=(0,), kill=None, max_turns=4, launch='timed'):
    """A train of two-particle bunches (``labels``: 0 first, as make_train orders them) one revolution apart, behind
    the reference (row 0)."""
    n_b = len(labels)
    phi = PHI0 + np.concatenate([[0.0], np.tile([0.02, -0.02], n_b)])
    birth = np.concatenate([[0.0]] + [OWN_BIRTH + k * PERIOD * DT for k in labels])
    beam = types.SimpleNamespace(x_vec=R * np.column_stack([np.cos(phi), np.sin(phi), np.zeros(len(phi))]),
                                 numpart=len(phi), alive=np.ones(len(phi), dtype=bool), birth_time=birth,
                                 bunch_of=np.concatenate([[0], np.repeat(labels, 2)]))
    eng = StubEngine(kill)
    cav = types.SimpleNamespace(omega=2 * np.pi * 4 / (PERIOD * DT), get_total_phase_rad=lambda: 0.0)
    design = types.SimpleNamespace(species=types.SimpleNamespace(mass_mev=938.272), rf_cavities=[cav])
    f = types.SimpleNamespace(_n_ref=1, design=design, record_centroid=False, center_source='reference', launch=launch,
                              engine=eng, space_charge=None, is_multiparticle=True, steps_per_turn=SPT,
                              target_energy_mev=np.inf, verbose=False, fast_path=False, fast3d=False, n_particles=2,
                              _n_companions=2 * (n_b - 1))
    out = AcceleratedOrbitFinder.track_with_rf(f, beam, DT, max_turns, section_angle=PHI0 - 1e-3, save_full_beam=True)
    return f, eng, out


def test_train_records_are_the_own_bunchs_in_its_own_clock():
    """With a bunch ahead and one behind, the turns, the statistics and the beam record of the run are those of the
    bunch alone, step for step; the train's clock starts one period earlier."""
    f1, e1, (ok1, ts1, _, traj1, _, std1, ids1, fb1) = _run()
    f3, e3, (ok3, ts3, _, traj3, _, std3, ids3, fb3) = _run(labels=(0, -1, 1))
    assert ids1 == [9 + k * PERIOD for k in range(1, 5)] and ids3 == ids1          # the reference is born at step 10
    assert [t.n_active for t in ts3] == [t.n_active for t in ts1] == [2] * 4
    assert fb3.shape == fb1.shape and np.array_equal(fb3, fb1, equal_nan=True)     # the own bunch only, same steps
    assert np.array_equal(traj3, traj1) and np.array_equal(std3, std1)
    assert f1.last_record_offset == 0 and f3.last_record_offset == PERIOD
    assert e3.steps_run == e1.steps_run + PERIOD
    # release: every bunch on the same step grid, one period apart (rows: reference, bunch 0, -1, +1)
    assert list(e1.birth) == [10, 0, 5] and list(e3.birth) == [10 + PERIOD, PERIOD, PERIOD + 5, 0, 5,
                                                              2 * PERIOD, 2 * PERIOD + 5]
    la1, la3 = f1.last_launch, f3.last_launch
    assert 'train' not in la1 and la3['train'] == {'t0_s': -10 * DT - PERIOD * DT, 'record_offset_steps': PERIOD,
                                                   'n_companions': 4}
    assert {k: v for k, v in la3.items() if k != 'train'} == la1                   # t0, reference birth step, ...
    assert la1['t0_s'] == -10 * DT and la1['reference_birth_step'] == 10
    assert f1.last_train is None
    assert f3.last_train == {'labels': [-1, 0, 1], 'alive': [[2, 2, 2]] * 4, 'record_offset_steps': PERIOD}
    assert f3.last_stop == 'turns' and len(f3.last_final_state[0]) == 7


def test_train_companions_do_not_count_and_do_not_keep_a_run_alive():
    """Lost companions change the table per bunch, not the own bunch's statistics; an own bunch that is gone ends
    the run as 'beam_lost' although companions fly on; a train needs the timed launch."""
    # turns are logged at steps 215, 318, 421 and 524 of the train's clock; rows 3 (bunch -1) and 5 (bunch +1) die
    f, eng, (_, ts, *_rest) = _run(labels=(0, -1, 1), kill={3: PERIOD + 150, 5: 2 * PERIOD + 300})
    assert [t.n_active for t in ts] == [2] * 4
    assert f.last_train['alive'] == [[2, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 1]]
    # the own bunch (rows 1, 2) lost at step 300 of its own clock; the reference flies on, so do the companions
    f, eng, (_, ts, *_rest) = _run(labels=(0, -1, 1), kill={1: PERIOD + 300, 2: PERIOD + 300})
    assert f.last_stop == 'beam_lost' and eng.steps_run == PERIOD + 301 and len(ts) == 2
    # and with its reference gone too
    f, eng, (_, ts, *_rest) = _run(labels=(0, -1, 1), kill={0: PERIOD + 250, 1: PERIOD + 300, 2: PERIOD + 300})
    assert f.last_stop == 'beam_lost' and eng.steps_run == PERIOD + 301
    with pytest.raises(ValueError, match='timed launch'):
        _run(labels=(0, -1), launch='snapshot')


# ---------------------------------------------------------------------------------------------------------------------
# space charge of a train on one mesh
def _sphere(n, radius, centre, rng):
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    return np.asarray(centre) + radius * rng.uniform(size=(n, 1)) ** (1.0 / 3.0) * u


def _hook(n_cells, current_a=8e-3, f_hz=32.8e6, coupled=True):
    from PyPATools.poisson_fft import FFTPoissonSolver
    solver = FFTPoissonSolver(pad_cells=3, window_quantile=0.0, use_gpu=False, n_cells=n_cells, frame='bunch')
    return SpaceChargeKick(solver, resolve_every=1, bunch_current_a=current_a, bunch_frequency_hz=f_hz, kick_z=True,
                           train_coupled=coupled)


def _prepared(n, charge, n_cells=(32, 96, 32), coupled=True):
    """A hook prepared for a train of two n-particle bunches (bunch 0 and one ahead) behind a reference row."""
    b = ParticleDistribution(species=SPECIES, x_vec=np.zeros((n, 3)), p_vec=np.zeros((n, 3)))
    train = make_train(b, 1, 0, 1.2e-7, charge=charge)
    beam = types.SimpleNamespace(numpart=2 * n + 1, n_scored=n, bunch_of=np.concatenate([[0], train.bunch_of]),
                                 charge_weight=np.concatenate([[1.0], train.charge_weight]))
    hook = _hook(n_cells, coupled=coupled)
    hook.prepare(beam, n_ref=1, species=SPECIES)
    return hook


def test_train_charge_is_one_bunch_charge_per_bunch():
    """Each bunch carries I / f, shared by its own particles (not by all the particles of the beam); the other
    bunches' charge scales with their weight; weightless ones deposit nothing but go into the solve."""
    n, q_bunch = 400, 8e-3 / 32.8e6
    hook = _prepared(n, 1.0)
    assert hook.charges[0] == 0.0 and np.all(hook.charges[1:] == q_bunch / n)
    assert hook.bunch_charge_c == pytest.approx(q_bunch, rel=1e-12) and hook.n_charged == 2 * n
    assert hook.train['labels'] == [-1, 0] and hook.train['bunch_charge_c'] == pytest.approx([q_bunch] * 2, rel=1e-12)
    assert hook.train['total_charge_c'] == pytest.approx(2 * q_bunch, rel=1e-12) and not hook.window_only.any()
    assert hook.train['coupled'] is True and _prepared(n, 1.0, coupled=False).train['coupled'] is False
    hook = _prepared(n, 0.25)
    assert np.all(hook.charges[1:n + 1] == q_bunch / n) and np.all(hook.charges[n + 1:] == 0.25 * q_bunch / n)
    assert hook.bunch_charge_c == pytest.approx(q_bunch, rel=1e-12)
    hook = _prepared(n, 0.0)
    assert np.all(hook.charges[n + 1:] == 0.0) and hook.n_charged == n and hook.bunch_charge_c > 0
    assert np.array_equal(np.flatnonzero(hook.window_only), np.arange(n + 1, 2 * n + 1)) and not hook.noop
    assert hook.report()['train']['n_weightless'] == n
    # a bunch alone: every real particle shares the bunch charge, no train record, nothing weightless
    plain = _hook((32, 32, 32))
    plain.prepare(types.SimpleNamespace(numpart=n + 1), n_ref=1, species=SPECIES)
    assert np.all(plain.charges[1:] == q_bunch / n) and plain.train is None and not plain.window_only.any()
    assert plain.bunch_charge_c == float(plain.charges.sum()) and 'train' not in plain.report()


def test_two_displaced_bunches_against_the_closed_form():
    """Two uniformly charged spheres side by side on one mesh. On bunch 0 the field is its own (Q r / R^3 inside)
    plus the other's (a point charge outside a sphere); a weightless neighbour leaves the own field alone on the same
    mesh, so the difference is the neighbour's field; the forces of the train sum to zero. The control
    (train_coupled False) gives bunch 0 that own field and the neighbour the field of the coupled train."""
    n, radius, d = 60000, 5e-3, 30e-3
    rng = np.random.default_rng(3)
    a, b = _sphere(n, radius, (0.0, 0.0, 0.0), rng), _sphere(n, radius, (0.0, d, 0.0), rng)
    r = np.vstack([[[0.0, 0.0, 0.0]], a, b])
    v = np.tile([2.5e6, 0.0, 0.0], (len(r), 1))                  # the bunch frame's first axis: x; across the train: y
    active = np.ones(len(r), dtype=bool)
    q_bunch = 8e-3 / 32.8e6
    fields = {}
    for charge in (1.0, 0.0):
        hook = _prepared(n, charge)
        hook._solve(0, r, v, active, 0.0)
        assert hook.n_solves == 1 and tuple(hook.solver.shape) == (32, 96, 32)
        assert hook.has_field[1:].all() and not hook.has_field[0]           # the reference is exempt
        fields[charge] = hook.E.copy()
        if charge:
            force = (hook.charges[:, None] * hook.E).sum(axis=0)
            assert np.linalg.norm(force) < 1e-12 * np.abs(hook.charges[:, None] * hook.E).sum()
    rows_a, rows_b = slice(1, n + 1), slice(n + 1, 2 * n + 1)
    rms = (lambda e: float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))))
    own_exact = K_E * q_bunch * a / radius ** 3
    sep = a - np.array([0.0, d, 0.0])
    other_exact = K_E * q_bunch * sep / np.linalg.norm(sep, axis=1)[:, None] ** 3
    # the neighbour's field on bunch 0: train minus control, against the point-charge law
    other = fields[1.0][rows_a] - fields[0.0][rows_a]
    assert rms(other - other_exact) < 0.005 * rms(other_exact)
    # the own field (the control), away from the sharp edge of the sphere
    core = np.linalg.norm(a, axis=1) < 0.8 * radius
    assert rms(fields[0.0][rows_a][core] - own_exact[core]) < 0.05 * rms(own_exact[core])
    # the whole field of the train on bunch 0
    assert rms(fields[1.0][rows_a][core] - (own_exact + other_exact)[core]) < 0.05 * rms((own_exact + other_exact)[core])
    # the weightless neighbour feels bunch 0 (a point charge at its distance) and nothing of itself
    felt = K_E * q_bunch * b / np.linalg.norm(b, axis=1)[:, None] ** 3
    assert rms(fields[0.0][rows_b] - felt) < 0.005 * rms(felt)
    # the control: two solves on the same mesh; bunch 0 feels its own charge alone, the neighbour every charge
    hook = _prepared(n, 1.0, coupled=False)
    hook._solve(0, r, v, active, 0.0)
    assert hook.n_solves == 1 and hook.solver.n_solves == 2 and hook.n_charged == 2 * n
    assert np.array_equal(hook.E[rows_a], fields[0.0][rows_a])          # as beside a weightless neighbour, bit for bit
    assert np.array_equal(hook.E[rows_b], fields[1.0][rows_b])          # as in the coupled train
    # before the own bunch is released (or after it is lost) the control is one solve of what flies
    hook = _prepared(n, 1.0, coupled=False)
    only_b = active.copy()
    only_b[rows_a] = False
    hook._solve(0, r, v, only_b, 0.0)
    assert hook.solver.n_solves == 1 and hook.has_field[rows_b].all() and not hook.has_field[rows_a].any()


# ---------------------------------------------------------------------------------------------------------------------
# simulate3d: the train's mesh and its table
def test_train_mesh_and_solver():
    tc = dataclasses.replace(C.TrackConfig(), sc_gpu=False)
    assert S.train_mesh(tc, 64) == ((64, 512, 64), 4) and S.train_mesh(tc, 128) == ((128, 512, 128), 1)
    assert S.train_mesh(dataclasses.replace(tc, sc_train_cells_radial=256), 32)[0] == (32, 256, 32)
    with pytest.raises(C.ConfigError, match='bunch frame'):
        S.train_mesh(dataclasses.replace(tc, sc_frame='lab'), 64)
    with pytest.raises(C.ConfigError, match='at least 9'):
        S.train_mesh(dataclasses.replace(tc, sc_train_cells_radial=8), 64)
    # the arrays of a solve: today's 128^3 with four cached Green spectra stays under the solver's 2 GiB pool, a
    # train's 128 x 512 x 128 does not
    assert 1.0 < S.sc_mesh_memory_gb((128, 128, 128), 4) < 2.0 < S.sc_mesh_memory_gb((128, 512, 128), 1)
    assert S.sc_mesh_memory_gb((64, 512, 64), 4) == pytest.approx(S.sc_mesh_memory_gb((128, 128, 128), 4))
    alone = S.space_charge_kick(tc, 64, 5, 8.0, 32.8e6)
    train = S.space_charge_kick(tc, 64, 5, 8.0, 32.8e6, train=True)
    assert alone.solver.n_cells == (64, 64, 64) and alone.solver.cache_size == 4       # as before
    assert train.solver.n_cells == (64, 512, 64) and train.every == alone.every == 5
    assert S.space_charge_kick(tc, 128, 1, 8.0, 32.8e6, train=True).solver.cache_size == 1
    assert S.space_charge_kick(tc, 0, 5, 8.0, 32.8e6, train=True).solver.n_cells is None    # the cell-size mode


def test_train_record_table():
    """The table per bunch: live particles per logged turn and at the end, metal losses, end energy and radius."""
    n, n_ref = 3, 1
    labels = np.repeat([0, -1, 1], n)
    r = np.zeros((1 + 3 * n, 3))
    r[1:, 0] = np.repeat([0.20, 0.25, 0.15], n)
    v = np.zeros_like(r)
    v[:, 1] = 1.0e7
    act = np.ones(len(r), dtype=bool)
    act[[2, 5, 6]] = False                                       # one of bunch 0, two of bunch -1
    finder = types.SimpleNamespace(_n_ref=n_ref, last_final_state=(r, v, act), last_record_offset=1000)
    res = types.SimpleNamespace(metadata={'train': {'labels': [-1, 0, 1], 'alive': [[3, 3, 0], [2, 2, 3]],
                                                    'record_offset_steps': 1000},
                                          'space_charge': {'train': {'labels': [-1, 0, 1],
                                                                     'bunch_charge_c': [2e-10, 2e-10, 2e-10]}}})
    hits = np.array([[1.0, 1200.0, 0.2, 0.0], [4.0, 300.0, 0.25, 0.0], [5.0, 900.0, 0.25, 0.0]])
    me = types.SimpleNamespace(finder=finder,
                               tc=types.SimpleNamespace(steps_per_turn=1000, train_charge=1.0, train_coupled=True),
                               mass_mev=float(SPECIES.mass_mev))
    beam = types.SimpleNamespace(bunch_of=labels, n_scored=n)
    rec = S.BunchTracker._train_record(me, beam, res, hits)
    assert rec['labels'] == [-1, 0, 1] and rec['period_steps'] == 1000 and rec['record_offset_steps'] == 1000
    assert rec['alive'] == {'-1': [3, 2], '0': [3, 2], '1': [0, 3]}
    assert rec['alive_end'] == [1, 2, 3] and rec['lost_metal'] == [2, 1, 0]
    assert rec['r_mean_end_mm'] == pytest.approx([250.0, 200.0, 150.0])
    assert rec['charge_c'] == [2e-10] * 3 and rec['n_per_bunch'] == 3 and rec['coupled'] is True
    assert all(e > 0 and math.isfinite(e) for e in rec['e_mean_end_mev'])
