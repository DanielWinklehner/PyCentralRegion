"""MOBO driver: unit cube, store and resume, the campaign lock, constraints and failures, geometry batches and cache,
the samplers' pinning, process trees (time limit, Ctrl+C, time budget, a killed driver), verify / status / dry run,
and a toy campaign through real subprocesses that ends near the toy's front.

Tests that need optuna skip without it (the physics environment has none); the GP tests also need torch and a
working bounded L-BFGS-B."""
import dataclasses
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import PyCentralRegion
from PyCentralRegion import config as C
from PyCentralRegion.mobo import driver, evaluate, space, store, toy

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / 'configs' / 'examples' / 'toy' / 'campaign.toml'
BASE = TOY.with_name('base.toml')
PILOT = ROOT / 'configs' / 'examples' / 'hchc1p5' / 'mobo_pilot.toml'
CAMPAIGN = ROOT / 'configs' / 'examples' / 'hchc1p5' / 'mobo_campaign.toml'
SRC = Path(PyCentralRegion.__file__).resolve().parents[1]          # subprocesses import what the tests import
HV_FRONT = (2.0 / 3.0) * (1.0 - (1.0 - toy.I_MAX) ** 1.5)
# a smaller toy problem for the convergence test: the RF phase, a log-scaled trim and a continuous geometry variable
SMALL = ['mobo.variables=[{name="phase", target="toy.phases_deg", low=-40.0, high=0.0, as_list=true}, '
         '{name="trim", target="toy.trim_t", low=1e-5, high=1e-2, log=true}, '
         '{name="width", target="toy.gap_width_mm", low=2.0, high=8.0, kind="geometry"}]',
         'mobo.warm_start=[]', 'mobo.fixed=["toy.volt_scale=1.02", "toy.n_posts=2"]']
SLOW = 'mobo.fixed=["toy.sleep_s=60.0", "toy.child_sleep_s=60.0"]'   # evaluations that run until they are stopped


def campaign(folder, *sets):
    return driver.load(TOY, [f'mobo.campaign_dir={(Path(folder) / "camp").as_posix()}', *sets])


def inprocess(cmd, eval_dir, timeout_s=0.0, env=None, on_start=None, keep_fds=()):
    """The toy evaluator called in the test process (same arguments, same result.json), for speed."""
    t0 = time.monotonic()
    rc = toy.main(cmd[cmd.index('--config'):])
    return evaluate.Outcome(returncode=rc, wall_s=time.monotonic() - t0, result=evaluate.read_result(eval_dir))


def exact(cfg, params):
    """Noise-free toy scores of a design."""
    run = C.load(cfg.base_config_file, list(cfg.fixed) + space.overrides(cfg, params))
    return toy.scores({**toy.DEFAULTS, **run['toy'], 'noise': 0.0})


def wait_for(condition, seconds=60.0, what=''):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.05)


def child_pid(eval_dir):
    path = Path(eval_dir) / 'child.pid'
    return json.loads(path.read_text()) if path.exists() else None


@pytest.fixture
def subprocess_env(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(SRC), os.environ.get('PYTHONPATH', '')]))


def _require_gp():
    if importlib.util.find_spec('optuna') is None or importlib.util.find_spec('torch') is None:
        pytest.skip('GP sampler needs optuna and torch')
    probe = ('import numpy as np, scipy.optimize as so; '
             'so.fmin_l_bfgs_b(lambda x: (((x - .3) ** 2).sum(), 2 * (x - .3)), np.array([.9, .1]), bounds=[(0, 1)] * 2)')
    if subprocess.run([sys.executable, '-c', probe], capture_output=True, timeout=300).returncode != 0:
        pytest.skip("scipy's bounded L-BFGS-B does not work in this environment")


# ---------------------------------------------------------------------------------------------------------------------
def test_unit_cube_mapping():
    lin = space.Variable('volt', 'track.volt_scale', 0.95, 1.10)
    assert (lin.physical(0.0), lin.physical(1.0), lin.physical(-3.0)) == (0.95, 1.10, 0.95)
    assert lin.physical(lin.unit(1.02)) == pytest.approx(1.02, abs=1e-15)
    log = space.Variable('trim', 'btrim.amplitude_t', 1e-5, 1e-2, log=True)
    assert log.physical(0.5) == pytest.approx(10 ** -3.5) and log.unit(1e-3) == pytest.approx(2.0 / 3.0)
    assert log.physical(1.0) == 1e-2                           # exp(log(high)) clipped to high
    posts = space.Variable('posts', 'geometry.n_posts', 0, 4, integer=True)
    values = [posts.physical(u) for u in (np.arange(1000) + 0.5) / 1000]
    assert all(isinstance(k, int) for k in values) and np.bincount(values).tolist() == [200] * 5   # equal shares
    for u in (0.0, 0.33, 0.5, 0.99):
        assert posts.physical(posts.snap(u)) == posts.physical(u) and posts.snap(posts.snap(u)) == posts.snap(u)
    assert posts.unit(2) == 0.5                                # the centre of its cell
    n = space.Variable('n', 'geometry.n', 1, 64, log=True, integer=True)
    assert [n.physical(n.unit(k)) for k in range(1, 65)] == list(range(1, 65))
    phase = space.Variable('phase', 'track.phases_deg', -40.0, 0.0, as_list=True)
    item = phase.override(phase.physical(0.625))
    assert item == 'track.phases_deg=[-15.0]' and C.parse_override(item) == (['track', 'phases_deg'], [-15.0])
    assert C.parse_override(posts.override(3))[1] == 3
    assert C.parse_override(f"paths.fields={space.toml_value('D:/a b (c)/f.pkl')}")[1] == 'D:/a b (c)/f.pkl'


def test_campaign_config_and_validation(tmp_path):
    cfg = campaign(tmp_path)
    assert [v.kind for v in cfg.variables] == ['cheap', 'cheap', 'cheap', 'geometry', 'geometry']
    assert cfg.fidelity('search').overrides == ('toy.noise=0.01',) and Path(cfg.base_config_file).is_file()
    assert isinstance(cfg.fidelity('verify'), space.Fidelity) and Path(cfg.campaign_dir).is_absolute()
    bad = {'mobo.sampler="bo"': 'sampler',
           'mobo.variables=[{name="a", target="run.out_dir"}]': 'target',
           'mobo.variables=[{name="a", target="x.y", low=1.0, high=1.0}]': 'low < high',
           'mobo.variables=[{name="a", target="x.y", low=0.0, high=1.0, log=true}]': 'log',
           'mobo.variables=[{name="a", target="x.y"}, {name="a", target="x.z"}]': 'unique',
           'mobo.objectives=[{metric="I_final", direction="up"}]': 'direction',
           'mobo.constraints=[{metric="E_med", op=">"}]': 'op',
           'mobo.constraints=[{metric="E_med", value=2.9}, {metric="E_med", value=2.9}]': 'repeat',
           'mobo.search_fidelity="fast"': 'fast',
           'mobo.warm_start=[{phse=1.0}]': 'phse'}
    for item, match in bad.items():
        with pytest.raises(C.ConfigError, match=match):
            campaign(tmp_path, item).validate()
    with pytest.raises(C.ConfigError, match='nope'):
        campaign(tmp_path, 'mobo.fidelities.search.nope=1')
    # overrides are checked against their [section] before anything runs: a misspelt target, an absent array item
    for item, match in {'mobo.variables=[{name="v", target="track.volt_scal"}]': 'volt_scal',
                        'mobo.variables=[{name="w", target="geometry.windows.0.z_hi_mm", kind="geometry"}]': 'no array',
                        'mobo.fidelities.verify.overrides=["track.n_finale=0"]': 'n_finale'}.items():
        with pytest.raises(C.ConfigError, match=match):
            driver.Campaign(campaign(tmp_path, item, *(['mobo.warm_start=[]'] * item.startswith('mobo.var'))))


def test_objectives_constraints_and_usable_values(tmp_path):
    cfg = campaign(tmp_path)
    s = {'I_final': 0.5, 'valley_all': 0.3, 'ratio': 2.0, 'E_med': 3.0}
    assert space.objective_values(cfg, s) == [0.5, 0.3, 2.0]
    assert space.constraint_values(cfg, s) == [pytest.approx(-0.1)]        # E_med >= 2.9 holds: <= 0
    assert space.constraint_values(cfg, {**s, 'E_med': 2.5}) == [pytest.approx(0.4)]
    assert space.Constraint('x', '<=', 1.0).violation(1.5) == 0.5
    assert math.isnan(space.metric({'a': {'b': None}}, 'a.b')) and space.metric({'a': {'b': 2}}, 'a.b') == 2.0
    with pytest.raises(KeyError):
        space.metric(s, 'nope')
    camp = driver.Campaign(cfg, runner=inprocess)
    assert camp.values({'state': 'done', 'scores': s}) == ([0.5, 0.3, 2.0], [pytest.approx(-0.1)])
    assert camp.values({'state': 'done', 'scores': {**s, 'ratio': float('nan')}}) is None
    assert camp.values({'state': 'done', 'scores': {k: v for k, v in s.items() if k != 'ratio'}}) is None
    assert camp.values({'state': 'failed'}) is None
    with pytest.raises(C.ConfigError, match='ratio'):                     # a metric no evaluation has is misnamed
        camp.check_metrics({'0000': {'state': 'done', 'scores': {k: v for k, v in s.items() if k != 'ratio'}}})
    camp.check_metrics({'0000': {'state': 'done', 'scores': {'skip': '12 tracks'}}})   # a skipped run: no metrics


def test_store_round_trip(tmp_path):
    st = store.Store(tmp_path / 'c')
    st.append('0000', 'proposed', unit={'a': 0.5}, params={'a': 1.0})
    st.append('0000', 'running', pid=12)
    st.append('0000', 'done', scores={'x': 1.0})
    st.append('0001', 'proposed', unit={'a': 0.1})
    with open(st.log, 'a') as fh:
        fh.write('{"id": "0001", "sta')                       # a line cut short by a killed writer
    st.append('0001', 'failed', error='interrupted')
    with pytest.warns(UserWarning, match='incomplete'):
        evals = st.evaluations()
    assert list(evals) == ['0000', '0001'] and st.next_id(evals) == '0002'
    assert evals['0000']['state'] == 'done' and evals['0000']['pid'] == 12 and evals['0000']['unit'] == {'a': 0.5}
    assert {'t_proposed', 't_running', 't_done'} <= set(evals['0000']) and evals['0001']['state'] == 'failed'
    st.append('0000.verify', 'proposed', parent='0000')
    st.append('0000.verify', 'failed', error='timeout')
    st.append('0000.verify', 'proposed', parent='0000')         # a second attempt restarts the record
    with pytest.warns(UserWarning):
        evals = st.evaluations()
    assert evals['0000.verify']['attempt'] == 2 and 'error' not in evals['0000.verify']
    assert st.next_id(evals) == '0002'


def test_campaign_lock_is_exclusive(tmp_path):
    path = tmp_path / 'c' / 'driver.lock'
    with store.CampaignLock(path) as lock:
        assert json.loads(path.read_text())['pid'] == os.getpid()
        with pytest.raises(store.CampaignBusy, match=f"in use by driver pid {os.getpid()}"):
            store.CampaignLock(path).__enter__()
        assert lock.inherit == (() if os.name == 'nt' else (lock.fd,))
    assert path.read_text() == ''                              # free: an empty note
    with store.CampaignLock(path):
        pass


def test_resume_continues_with_the_same_proposals(tmp_path):
    pytest.importorskip('optuna')
    a = driver.Campaign(campaign(tmp_path / 'a', 'mobo.max_evals=14'), runner=inprocess)
    assert a.optimize() == 0
    b = driver.Campaign(campaign(tmp_path / 'b'), runner=inprocess)
    b.optimize(max_evals=6)
    b.optimize(max_evals=14)                                    # stopped by its budget, continued
    ua = [ev['unit'] for ev in a.store.evaluations().values()]
    assert len(ua) == 14 and ua == [ev['unit'] for ev in b.store.evaluations().values()]
    assert len({json_key(u) for u in ua}) == 14

    # a driver that died during an evaluation leaves it 'running': the next start records it interrupted (left out
    # of the study and of the budget, not run again) and continues; one proposed but never started runs as proposed
    st = b.store
    st.append('0014', 'proposed', **b.propose(st.evaluations()))
    st.append('0014', 'running', pid=2 ** 22 + 7)
    b.optimize(max_evals=15)
    evals = st.evaluations()
    assert evals['0014']['state'] == 'failed' and evals['0014']['interrupted']
    assert evals['0014']['error'] == 'interrupted: the driver stopped while it ran'
    assert evals['0015']['state'] == 'done' and b.spent(evals) == 15 and st.lock_path.read_text() == ''
    study, _ = b.study(evals, None)
    assert len(study.trials) == 15                              # 0014 is left out
    st.append('0016', 'proposed', **b.propose(evals))
    b.optimize(max_evals=16)
    assert st.evaluations()['0016']['state'] == 'done' and len(st.evaluations()) == 17


def json_key(unit):
    return tuple(sorted(unit.items()))


def test_failed_evaluations(tmp_path):
    optuna = pytest.importorskip('optuna')
    sets = ['mobo.fixed=["toy.fail_above_mm=4.0"]', 'mobo.max_evals=12']
    camp = driver.Campaign(campaign(tmp_path, *sets), runner=inprocess)
    camp.optimize()
    evals = camp.store.evaluations()
    search = camp.search(evals)
    failed = [ev for ev in search if ev['state'] == 'failed']
    assert failed and all('toy build failed' in ev['error'] and ev['params']['width'] > 4.0 for ev in failed)
    assert all(ev['state'] == 'done' for ev in search if ev['params']['width'] <= 4.0)
    assert camp.spent(evals) == 12                              # failures of the design count
    # a geometry whose build failed has no fields: no cheap evaluations are pinned to it
    assert not {ev.get('pinned_to') for ev in search} & {ev['id'] for ev in failed}
    # the sampler sees a failed evaluation as a pending trial ...
    study, _ = camp.study(evals, optuna.samplers.RandomSampler(seed=0))
    assert [t.state == optuna.trial.TrialState.RUNNING for t in study.trials] == [ev['state'] == 'failed' for ev in search]
    assert all(list(t.constraints) == ['E_med>=2.9'] for t in study.trials if t.state.is_finished())
    # ... or, with failed_as_infeasible, as a complete trial with the worst values and an extra violated constraint
    camp2 = driver.Campaign(campaign(tmp_path, *sets, 'mobo.failed_as_infeasible=true'), runner=inprocess)
    study2, _ = camp2.study(evals, optuna.samplers.RandomSampler(seed=0))
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study2.trials)
    assert [t.constraints['evaluated'] for t in study2.trials] == [float(ev['state'] == 'failed') for ev in search]
    ok = [t.values for t, ev in zip(study2.trials, search) if ev['state'] == 'done']
    worst = [t.values for t, ev in zip(study2.trials, search) if ev['state'] == 'failed'][0]
    assert worst == [min(v[0] for v in ok), max(v[1] for v in ok), max(v[2] for v in ok)]


def test_skipped_runs_enter_the_study_as_bad_designs(tmp_path):
    """A run the scorer skipped (too few survivors) is a design that lost its beam: a complete trial with the metrics
    its scores carry and the worst values seen for the rest; never in the Pareto set. With nothing seen yet it is
    pending."""
    optuna = pytest.importorskip('optuna')
    camp = driver.Campaign(campaign(tmp_path, 'mobo.max_evals=6'), runner=inprocess)
    skip = {'tag': 'camp', 'skip': '12 tracked survivors (min_survivors 200)'}
    first = camp.add('0000', camp.propose({}))
    camp.store.append(first['id'], 'done', scores=skip)
    study, _ = camp.study(camp.store.evaluations(), optuna.samplers.RandomSampler(seed=0))
    assert [t.state for t in study.trials] == [optuna.trial.TrialState.RUNNING]
    camp.optimize()
    evals = camp.store.evaluations()
    for eval_id, scores in (('0006', skip), ('0007', {**skip, 'I_final': 0.01})):
        ev = camp.add(eval_id, camp.propose(evals))
        camp.store.append(eval_id, 'done', scores=scores)
        evals = camp.store.evaluations()
    study, _ = camp.study(evals, optuna.samplers.RandomSampler(seed=0))
    ok = [v for ev in camp.search(evals) if (v := camp.values(ev)) is not None]
    worst = [min(v[0][0] for v in ok), max(v[0][1] for v in ok), max(v[0][2] for v in ok)]
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE and study.trials[0].values == worst
    assert study.trials[-1].values == [0.01] + worst[1:]
    assert study.trials[-1].constraints == {'E_med>=2.9': max(v[1][0] for v in ok)}
    assert not {'0000', '0006', '0007'} & set(camp.summary(evals)['pareto']) and camp.spent(evals) == 8


def test_time_limit_stops_the_whole_tree(tmp_path, subprocess_env):
    """An evaluation past its time limit is stopped with the child it started; a finished one's straggler too."""
    cmd = evaluate.command('', 'PyCentralRegion.mobo.toy', BASE, ['toy.sleep_s=60', 'toy.child_sleep_s=60'],
                           tmp_path / 'e')
    pids = []
    out = evaluate.run(cmd, tmp_path / 'e', timeout_s=3.0, env=evaluate.child_env(''), on_start=pids.append)
    assert out.timed_out and not out.interrupted and out.result is None and out.wall_s < 30
    assert child_pid(tmp_path / 'e') and not store.pid_alive(child_pid(tmp_path / 'e')) and not store.pid_alive(pids[0])
    assert (tmp_path / 'e' / 'stderr.log').exists()
    cmd = evaluate.command('', 'PyCentralRegion.mobo.toy', BASE, ['toy.child_sleep_s=60'], tmp_path / 'f')
    out = evaluate.run(cmd, tmp_path / 'f', env=evaluate.child_env(''))
    assert out.returncode == 0 and out.result['status'] == 'ok' and not store.pid_alive(child_pid(tmp_path / 'f'))


def test_ctrl_c_stops_the_whole_tree(tmp_path, subprocess_env):
    """Ctrl+C in the driver (here a SIGINT 2 s in) stops the evaluation and its child at once."""
    cmd = evaluate.command('', 'PyCentralRegion.mobo.toy', BASE, ['toy.sleep_s=60', 'toy.child_sleep_s=60'],
                           tmp_path / 'e')
    pids = []
    timer = threading.Timer(2.0, signal.raise_signal, (signal.SIGINT,))
    timer.start()
    try:
        out = evaluate.run(cmd, tmp_path / 'e', env=evaluate.child_env(''), on_start=pids.append)
    finally:
        timer.cancel()
    assert out.interrupted and not out.timed_out and out.wall_s < 30
    assert not store.pid_alive(pids[0]) and not store.pid_alive(child_pid(tmp_path / 'e'))


def test_time_budget_stops_the_running_evaluation(tmp_path, subprocess_env):
    """At the end of max_hours the running evaluation is stopped with its tree and recorded interrupted; it does not
    count against max_evals and is not run again. The geometry files it had made (the toy writes them before it
    sleeps) are complete: they stay, and the geometry's cheap evaluations use them."""
    pytest.importorskip('optuna')
    camp = driver.Campaign(campaign(tmp_path, SLOW, 'mobo.max_evals=3'))
    t0 = time.monotonic()
    assert camp.optimize(max_hours=6.0 / 3600.0) == 0
    assert time.monotonic() - t0 < 30
    evals = camp.store.evaluations()
    ev = evals['0000']
    assert ev['state'] == 'failed' and ev['interrupted'] and ev['error'] == 'interrupted: the time budget ran out'
    assert not store.pid_alive(ev['pid']) and not store.pid_alive(child_pid(camp.store.eval_dir('0000')))
    assert list(evals) == ['0000'] and camp.spent(evals) == 0
    gdir = camp.store.geometry_dir(ev['geometry'])
    assert ev['builds'] and (gdir / driver.FIELDS_FILE).exists() and (gdir / driver.MODEL_FILE).exists()
    again = driver.Campaign(campaign(tmp_path, 'mobo.max_evals=3'), runner=inprocess)
    assert again.optimize() == 0
    evals = again.store.evaluations()
    assert list(evals) == ['0000', '0001', '0002', '0003'] and again.spent(evals) == 3
    assert evals['0000']['state'] == 'failed' and evals['0001']['pinned_to'] == '0000'


def test_a_stopped_build_leaves_its_geometry_buildable(tmp_path):
    """A build commits its outputs one by one, the model last, and never overwrites one: what a build stopped before
    its model left is moved aside, so the next build of that geometry starts clean. Complete files stay."""
    camp = driver.Campaign(campaign(tmp_path), runner=inprocess)
    ev = camp.add('0000', camp.propose({}))                     # the warm start: a new geometry
    gdir = camp.store.geometry_dir(ev['geometry'])

    def stopped_build(cmd, eval_dir, timeout_s=0.0, env=None, on_start=None, keep_fds=()):
        on_start(12345)
        for name in ('model3d.msh', 'model3d.part.step'):       # committed, and not yet committed
            (gdir / name).write_text('a build that did not finish')
        return evaluate.Outcome(interrupted=True)

    camp.runner = stopped_build
    lock = store.CampaignLock(camp.store.lock_path)
    assert camp.execute(ev, lock) == 'stopped by the user'
    assert sorted(p.name for p in gdir.iterdir()) == ['geometry.json', 'stopped-0000']
    assert sorted(p.name for p in (gdir / 'stopped-0000').iterdir()) == ['model3d.msh', 'model3d.part.step']
    camp.runner = inprocess                                      # the same geometry again: it builds
    again = camp.add('0001', camp.proposal(camp.cfg.search_fidelity, 'warm', ev['unit'], ev['params']))
    assert again['geometry'] == ev['geometry'] and camp.execute(again, lock) == ''
    assert camp.store.evaluations()['0001']['state'] == 'done' and (gdir / driver.FIELDS_FILE).exists()
    camp.runner = stopped_build                                  # a model made: nothing is moved
    third = camp.add('0002', camp.proposal(camp.cfg.search_fidelity, 'warm', ev['unit'], ev['params']))
    assert camp.execute(third, lock) == 'stopped by the user' and (gdir / 'model3d.msh').exists()


def test_interrupted_followers_leave_the_geometry_its_cheap_evaluations(tmp_path):
    """A cheap evaluation pinned to a geometry and then interrupted does not use up one of its cheap_per_geometry; an
    evaluation that only read the geometry's files (they existed) leaves them in place."""
    pytest.importorskip('optuna')
    camp = driver.Campaign(campaign(tmp_path, 'mobo.max_evals=1'), runner=inprocess)
    assert camp.optimize() == 0
    evals = camp.store.evaluations()
    ev = camp.add(camp.store.next_id(evals), camp.propose(evals))
    assert ev['kind'] == 'pinned' and ev['pinned_to'] == '0000' and ev['geometry'] == evals['0000']['geometry']

    def stopped(cmd, eval_dir, timeout_s=0.0, env=None, on_start=None, keep_fds=()):
        on_start(12345)
        return evaluate.Outcome(interrupted=True)

    camp.runner = stopped
    assert camp.execute(ev, store.CampaignLock(camp.store.lock_path)) == 'stopped by the user'
    evals = camp.store.evaluations()
    assert evals['0001']['interrupted'] and 'builds' not in evals['0001']
    assert (camp.store.geometry_dir(ev['geometry']) / driver.FIELDS_FILE).exists()
    kind, head = camp.plan(evals)
    assert kind == 'pinned' and head['id'] == '0000'


def test_evaluations_run_the_drivers_code(tmp_path):
    """The evaluations import the PyCentralRegion the driver runs (its source folder first on PYTHONPATH); a physics
    interpreter that cannot import the evaluator stops the campaign before an evaluation is spent on it."""
    env = evaluate.child_env('')
    assert env['PYTHONPATH'].split(os.pathsep)[0] == str(SRC) and env['MPLBACKEND'] == 'Agg'
    camp = driver.Campaign(campaign(tmp_path, 'mobo.evaluator="PyCentralRegion.mobo.nope"'))
    with pytest.raises(C.ConfigError, match='cannot import the evaluator PyCentralRegion.mobo.nope'):
        camp.optimize()
    assert not camp.store.log.exists()


def test_toy_build_refuses_fields_of_another_geometry(tmp_path):
    fields = (tmp_path / 'g' / 'fields.pkl').as_posix()
    sets = [f'paths.model3d="{(tmp_path / "g" / "model3d.pkl").as_posix()}"', f'paths.fields="{fields}"']
    assert toy.main(['--config', str(BASE), '--out', str(tmp_path / 'a'), *sum([['--set', s] for s in sets], [])]) == 0
    other = sets + ['toy.gap_width_mm=5.0']
    assert toy.main(['--config', str(BASE), '--out', str(tmp_path / 'b'), *sum([['--set', s] for s in other], [])]) == 1
    assert 'another geometry' in evaluate.read_result(tmp_path / 'b')['error']


def test_toy_campaign_through_subprocesses_ends_near_the_front(tmp_path, subprocess_env):
    """30 evaluations with TPE, every one a real `python -m PyCentralRegion.mobo.toy` subprocess. On this toy
    problem (calibrated over 8 seeds and 12 random-sampling repetitions) the median distance g - 1 of the last 10
    evaluations from the front is <= 0.14 with TPE and >= 0.22 with random sampling."""
    pytest.importorskip('optuna')
    cfg = campaign(tmp_path, *SMALL, 'mobo.max_evals=30')
    camp = driver.Campaign(cfg)
    assert camp.optimize() == 0
    evals = list(camp.store.evaluations().values())
    assert len(evals) == 30 and all(ev['state'] == 'done' for ev in evals)
    assert all('toy.noise=0.01' in ev['overrides'] for ev in evals)
    # cheap evaluations follow every new geometry with its geometry coordinates exactly
    by_id = {ev['id']: ev for ev in evals}
    pinned = [ev for ev in evals if ev['kind'] == 'pinned']
    assert pinned and all(ev['unit']['width'] == by_id[ev['pinned_to']]['unit']['width'] and
                          ev['geometry'] == by_id[ev['pinned_to']]['geometry'] for ev in pinned)
    # one build per geometry key, shared by its evaluations (the toy fails on fields of another geometry)
    keys = {ev['geometry'] for ev in evals}
    built = {p.parent.name for p in (Path(cfg.campaign_dir) / 'geometries').glob('*/model3d.pkl')}
    assert built == keys and len(keys) == sum(ev['kind'] != 'pinned' for ev in evals)
    scores = [exact(cfg, ev['params']) for ev in evals]
    distance = [s['distance'] for s in scores]
    front = [(s['I_final'], s['valley_all']) for s in scores if s['E_med'] >= 2.9]
    assert np.median(distance[-10:]) < 0.18
    assert toy.hypervolume(front) > 0.8 * HV_FRONT


def test_killed_driver_stops_its_evaluation(tmp_path, subprocess_env):
    """A driver killed hard takes its evaluation and everything the evaluation started with it (Windows: the job
    object; POSIX: the evaluation's group survives, holding the campaign lock, until it is stopped). The next start
    records the evaluation interrupted and continues, deterministically."""
    pytest.importorskip('optuna')
    st = store.Store(tmp_path / 'camp')
    sets = [f'mobo.campaign_dir={st.dir.as_posix()}', 'mobo.max_evals=5', SLOW]
    proc = subprocess.Popen([sys.executable, '-m', 'PyCentralRegion.mobo.driver', 'optimize', str(TOY)]
                            + sum([['--set', s] for s in sets], []), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        wait_for(lambda: proc.poll() is not None or child_pid(st.eval_dir('0000')), what='the evaluation to start')
        assert proc.poll() is None, proc.stderr.read().decode()
    finally:
        proc.kill()
        proc.wait()
        proc.stderr.close()
    eval_pid = st.evaluations()['0000']['pid']
    child = child_pid(st.eval_dir('0000'))
    if os.name == 'nt':
        wait_for(lambda: not store.pid_alive(eval_pid) and not store.pid_alive(child), 10.0, 'the tree to end')
    else:
        with pytest.raises(store.CampaignBusy, match='left running'):
            driver.Campaign(campaign(tmp_path), runner=inprocess).optimize()
        os.killpg(eval_pid, signal.SIGKILL)
        wait_for(lambda: not store.pid_alive(child), 10.0, 'the tree to end')
    camp = driver.Campaign(campaign(tmp_path, 'mobo.max_evals=5'), runner=inprocess)
    shutil.copytree(st.dir, tmp_path / 'twin' / 'camp')
    assert camp.optimize() == 0
    evals = camp.store.evaluations()
    assert evals['0000']['error'] == 'interrupted: the driver stopped while it ran' and camp.spent(evals) == 5
    twin = driver.Campaign(campaign(tmp_path / 'twin', 'mobo.max_evals=5'), runner=inprocess)
    twin.optimize()
    assert [ev['unit'] for ev in twin.store.evaluations().values()] == [ev['unit'] for ev in evals.values()]


def test_tpe_pins_with_partial_fixed_sampler(tmp_path):
    optuna = pytest.importorskip('optuna')
    camp = driver.Campaign(campaign(tmp_path, 'mobo.max_evals=12'), runner=inprocess)
    camp.optimize()
    for n_startup in (100, 4):                                  # random sampling, then the multivariate TPE
        cfg = campaign(tmp_path, f'mobo.n_startup={n_startup}')
        sampler = driver.make_sampler(cfg, 3, {'u_width': 0.25, 'u_posts': 0.7})
        assert isinstance(sampler, optuna.samplers.PartialFixedSampler)
        study, dists = camp.study(camp.store.evaluations(), sampler)
        trial = study.ask(dists)
        assert trial.params['u_width'] == 0.25 and trial.params['u_posts'] == 0.7
        assert all(0.0 <= trial.params[k] <= 1.0 for k in dists)


def test_gp_acquisition_is_optimised_over_the_free_coordinates():
    _require_gp()
    from optuna._gp.acqf import BaseAcquisitionFunc
    from optuna._gp.search_space import SearchSpace
    from optuna.distributions import FloatDistribution

    class Bowl(BaseAcquisitionFunc):
        def eval_acqf(self, x):
            return -((x - 0.8) ** 2).sum(-1)

    names = ['u_a', 'u_b', 'u_c']
    sampler = driver.pinned_gp_sampler()(fixed={'u_b': 0.2}, seed=0)
    sampler._names = names
    x = sampler._optimize_acqf(Bowl(np.ones(3), SearchSpace({n: FloatDistribution(0.0, 1.0) for n in names})), None)
    assert x[1] == 0.2 and x[0] == pytest.approx(0.8, abs=1e-3) and x[2] == pytest.approx(0.8, abs=1e-3)


def test_gp_campaign_pins_the_geometry(tmp_path):
    _require_gp()
    cfg = campaign(tmp_path, 'mobo.sampler="gp"', 'mobo.n_startup=4', 'mobo.max_evals=8')
    camp = driver.Campaign(cfg, runner=inprocess)
    assert camp.optimize() == 0
    evals = camp.store.evaluations()
    assert len(evals) == 8 and all(ev['state'] == 'done' for ev in evals.values())
    for ev in evals.values():
        if ev['kind'] == 'pinned':
            head = evals[ev['pinned_to']]
            assert (ev['unit']['width'], ev['unit']['posts']) == (head['unit']['width'], head['unit']['posts'])
    sampler = driver.make_sampler(cfg, 0, {'u_width': 0.25, 'u_posts': 0.7})
    study, dists = camp.study(evals, sampler)
    trial = study.ask(dists)                                    # past the startup: the pinned acquisition
    assert trial.params['u_width'] == 0.25 and trial.params['u_posts'] == 0.7
    assert all(0.0 <= trial.params[k] <= 1.0 for k in dists)


def test_verify_status_and_dry_run(tmp_path, subprocess_env, capsys):
    pytest.importorskip('optuna')
    camp_dir = tmp_path / 'camp'
    sets = ['--set', f'mobo.campaign_dir={camp_dir.as_posix()}', '--set', 'mobo.max_evals=8']
    assert driver.main(['optimize', str(TOY), *sets, '--dry-run']) == 0
    assert 'next: 0000 warm' in capsys.readouterr().out and not camp_dir.exists()   # the dry run writes nothing
    assert driver.main(['optimize', str(TOY), *sets]) == 0
    assert driver.main(['optimize-verify', str(TOY), *sets, '--pareto']) == 0
    assert driver.main(['optimize-verify', str(TOY), *sets, '--ids', '0', '3']) == 0
    cfg = campaign(tmp_path)
    camp = driver.Campaign(cfg)
    evals = camp.store.evaluations()
    pareto = camp.summary(evals)['pareto']
    assert pareto and all(f'{i}.verify' in evals for i in pareto + ['0000', '0003'])
    for i in pareto + ['0000']:
        v = evals[f'{i}.verify']
        assert v['state'] == 'done' and v['fidelity'] == 'verify' and v['params'] == evals[i]['params']
        assert 'toy.noise=0.01' not in v['overrides']           # the verify fidelity is the exact toy
        assert v['scores']['I_final'] == pytest.approx(exact(cfg, v['params'])['I_final'], abs=1e-12)
    # a failed re-evaluation runs again on request; the folder of the failed attempt is kept
    broken = sets + ['--set', 'mobo.fidelities.broken.overrides=["toy.fail_above_mm=0.0"]']
    assert driver.main(['optimize-verify', str(TOY), *broken, '--ids', '1', '--fidelity', 'broken']) == 0
    assert driver.main(['optimize-verify', str(TOY), *broken, '--ids', '1', '--fidelity', 'broken']) == 0
    assert camp.store.evaluations()['0001.broken']['attempt'] == 2
    assert (camp_dir / 'evals' / '0001.broken.attempt1' / evaluate.RESULT).exists()
    capsys.readouterr()
    assert driver.main(['optimize-status', str(TOY), *sets]) == 0
    out = capsys.readouterr().out
    assert 'Pareto set' in out and f'{pareto[0]}.verify' in out and 'best feasible' in out
    assert driver.main(['optimize-verify', str(TOY), *sets, '--ids', 'nope']) == 1


def test_pilot_config_dry_run(capsys):
    """The pilot's first evaluation (its warm start, the design as tracked) needs neither optuna nor the GPU."""
    cfg = driver.load(PILOT)
    if not Path(cfg.physics_python).is_file():
        pytest.skip('the pilot names a physics interpreter of the machine it runs on')
    assert [v.kind for v in cfg.variables] == ['cheap'] * 4 and cfg.sampler == 'gp'
    assert [o.metric for o in cfg.objectives] == ['I_final', 'valley_all', 'ratio']
    camp = driver.Campaign(cfg)
    if camp.store.log.exists():
        pytest.skip('the pilot campaign has started')
    assert camp.dry_run() == 0
    out = capsys.readouterr().out
    for item in ('track.phases_deg=[-15.0]', 'track.volt_scale=1.0', 'btrim.amplitude_t=0.0', 'btrim.phase_deg=0.0',
                 'btrim.r_min_m=0.05', 'track.n_final=10000', 'track.n_tracks=10000', 'PyCentralRegion.cli evaluate'):
        assert item in out
    assert 'paths.model3d' not in out                          # the base config's own model and fields


def test_campaign_config_dry_run(capsys):
    """The full campaign: five geometry variables (the channel aperture, the inflector rotation) and eight cheap ones
    (RF, trim, slit); its first evaluation is the base design, built into the campaign's geometry cache."""
    from PyCentralRegion.geometry.options import GeometryConfig
    if 'aperture_inner_mm' not in {f.name for f in dataclasses.fields(GeometryConfig)}:
        pytest.skip('[geometry] has no aperture_inner_mm .. aperture_taper_length_mm yet')
    cfg = driver.load(CAMPAIGN)
    if not Path(cfg.physics_python).is_file():
        pytest.skip('the campaign names a physics interpreter of the machine it runs on')
    assert [v.kind for v in cfg.variables] == ['geometry'] * 5 + ['cheap'] * 8
    camp = driver.Campaign(cfg)
    if camp.store.log.exists():
        pytest.skip('the campaign has started')
    assert camp.dry_run() == 0
    out = capsys.readouterr().out
    params = camp.warm_point(0)
    key = space.geometry_key(camp.base(cfg.search_fidelity), cfg, params)
    assert params['si_rot'] == pytest.approx(4.726034523988023) and params['slit_hw'] == 0.0
    assert f"geometries/{key}/fields.pkl" in out and 'track.n_tracks=10000' in out
