"""Multi-objective Bayesian optimisation of central-region designs, with the full 3D model in the loop.

    pcr optimize        <mobo.toml> [--max-evals N] [--max-hours H] [--dry-run] [--set key=value ...]
    pcr optimize-verify <mobo.toml> (--ids ID ... | --pareto) [--fidelity NAME]
    pcr optimize-status <mobo.toml>

One evaluation of a design is the whole chain build -> 3D BEM -> bunch tracking with space charge -> scores, run as
a subprocess of the physics environment (``evaluate``). It costs minutes and one GPU, so the driver runs one
evaluation at a time and chooses each with a sample-efficient multi-objective sampler (optuna's TPE or Gaussian
process, both constrained) on the unit cube of the design variables (``space``).

Cost structure: a new electrode geometry needs a build and a 3D BEM (about 10 min); the RF phase, dee voltage, a
magnetic trim or a slit reuse the fields and only re-track (1-2 min). After every new geometry the driver therefore
runs ``cheap_per_geometry`` evaluations with the geometry coordinates held at that geometry's values and only the
cheap ones proposed (TPE through optuna's PartialFixedSampler; the GP through a subclass that maximises its
acquisition over the free coordinates while its Gaussian processes keep seeing every coordinate). Evaluations of one
geometry share one model and one field map under ``<campaign_dir>/geometries/<key>/``.

Fidelities: the sampler learns from the search fidelity (e.g. 10k particles, SC 64^3 every 5 steps, 7 turns);
``optimize-verify`` re-evaluates chosen designs (e.g. the Pareto set) at the verify fidelity (every particle, 128^3
every step) and stores them next to the originals, so the search-to-verify shift of every score can be read off.

State: the store (``store``) is the only state. Before every proposal the optuna study is rebuilt in memory from it
(``optuna.trial.create_trial`` with the constraint values set on the trial), with a sampler seeded from the
campaign seed and the evaluation's index: a campaign stopped and continued proposes exactly what an uninterrupted
one would. Failed evaluations go to the sampler as pending (RUNNING) trials, which keeps TPE (constant liar) and
the GP (q-batch acquisition) from proposing the same design again without inventing objective values for it; with
``failed_as_infeasible`` they enter as complete trials with the worst values seen and an extra violated constraint.
A run the scorer skipped (too few survivors to score) is not a failure but a design known to lose its beam: it
enters as a complete trial with the metrics its scores carry and the worst values seen for the others, so the
sampler learns to leave that region (a slit closed too far, an inflector turned too far).
An evaluation stopped from outside (Ctrl+C, the time budget, a driver that died) says nothing about its design: it
is recorded failed ('interrupted'), left out of the study and of the budget, and not run again. The model and the
field cache of a geometry are written whole, so a stop never leaves them cut short: once made they serve the
geometry's later evaluations. A build stopped before its model leaves outputs that the next build of that geometry
would refuse to overwrite: those are moved to ``geometries/<key>/stopped-<id>/``.

The driver never schedules or restarts itself: it runs until its budget is spent or it is stopped, and running it
again continues the campaign. When it stops - Ctrl+C, SIGTERM / Ctrl+Break, the end of max_hours, or killed - the
running evaluation stops with it, with every process it started (``evaluate.run``).

The driver needs only the standard library, numpy and optuna >= 5 (torch and scipy for the GP sampler); it never
imports the physics modules (the [section] dataclasses it checks the overrides against are plain dataclasses).
"""
import argparse
import copy
import functools
import importlib.util
import json
import math
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from .. import config
from . import evaluate, space
from .space import MoboConfig
from .store import CampaignBusy, CampaignLock, Store

# the files of one geometry, in <campaign_dir>/geometries/<key>/ (the evaluations' paths.model3d and paths.fields)
MODEL_FILE, FIELDS_FILE = 'model3d.pkl', 'fields.pkl'


def log(msg):
    print(f"[mobo] {msg}", flush=True)


def load(path, overrides=()):
    """The [mobo] section of a campaign file (includes and key=value overrides applied)."""
    return config.section(config.load(path, overrides), 'mobo', MoboConfig)


def _fmt(x):
    if isinstance(x, float):
        return 'nan' if math.isnan(x) else f'{x:.4g}'
    return '-' if x is None else str(x)


def pareto_mask(Y):
    """Non-dominated rows of Y, every column to be minimised."""
    Y = np.asarray(Y, dtype=float)
    return np.array([not np.any(np.all(Y <= y, axis=1) & np.any(Y < y, axis=1)) for y in Y], dtype=bool)


# ---------------------------------------------------------------------------------------------------------------------
# samplers
def import_optuna():
    """optuna, version checked: constraints are set on the trials (optuna 5's API, ``create_trial(constraints=)``)."""
    import optuna
    if int(optuna.__version__.split('.')[0]) < 5:
        raise config.ConfigError(f"the MOBO driver needs optuna >= 5.0, the environment has {optuna.__version__}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def step_seed(seed, step):
    """Seed of the sampler proposing evaluation number `step`. The study is rebuilt for every proposal, so one seed
    for all of them would hand each fresh sampler the same random draws."""
    return int(np.random.SeedSequence([int(seed), int(step)]).generate_state(1)[0] >> 1)


def make_sampler(cfg, seed, fixed=None):
    """The campaign's optuna sampler; `fixed` = {parameter: unit value} holds those coordinates."""
    optuna = import_optuna()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', optuna.exceptions.ExperimentalWarning)
        if cfg.sampler == 'tpe':
            tpe = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=seed,
                                             n_startup_trials=cfg.n_startup)
            return optuna.samplers.PartialFixedSampler(fixed, tpe) if fixed else tpe
        if importlib.util.find_spec('torch') is None:
            raise config.ConfigError("sampler 'gp' needs torch (the CPU build is enough) in the driver's environment")
        return pinned_gp_sampler()(fixed=fixed, seed=seed, n_startup_trials=cfg.n_startup)


@functools.lru_cache(maxsize=None)
def pinned_gp_sampler():
    """optuna's GPSampler with some coordinates held at given values.

    The Gaussian processes are fit on every coordinate, so the evaluations of all geometries inform the model; only
    the acquisition is maximised over the free coordinates. It is wrapped so that the pinned coordinates are
    replaced before the wrapped acquisition sees them: their gradient is zero, the optimiser (optuna's own,
    unchanged) never moves them, and the pinned values are exactly the unit values because a [0, 1] float
    parameter's normalised value is the value itself. Uses optuna's GPSampler._optimize_acqf(acqf, best_params)
    hook and BaseAcquisitionFunc (both private; checked against optuna 5.0)."""
    import torch
    from optuna._gp.acqf import BaseAcquisitionFunc
    from optuna.samplers import GPSampler

    class PinnedAcqf(BaseAcquisitionFunc):
        def __init__(self, acqf, index, values):
            self._acqf = acqf
            self._mask = torch.zeros(acqf.search_space.dim, dtype=torch.bool)
            self._mask[index] = True
            self._values = torch.zeros(acqf.search_space.dim, dtype=torch.float64)
            self._values[index] = torch.as_tensor(values, dtype=torch.float64)
            super().__init__(acqf.length_scales, acqf.search_space)

        def eval_acqf(self, x):
            return self._acqf.eval_acqf(torch.where(self._mask, self._values.to(x.dtype), x))

    class PinnedGPSampler(GPSampler):
        def __init__(self, *, fixed=None, **kwargs):
            super().__init__(**kwargs)
            self._fixed = dict(fixed or {})
            self._names = []

        def sample_relative(self, study, trial, search_space):
            self._names = list(search_space)          # the order of the GP's internal coordinates
            params = super().sample_relative(study, trial, search_space)
            return {**params, **{k: v for k, v in self._fixed.items() if k in params}}

        def sample_independent(self, study, trial, param_name, param_distribution):
            if param_name in self._fixed:
                return self._fixed[param_name]
            return super().sample_independent(study, trial, param_name, param_distribution)

        def _optimize_acqf(self, acqf, best_params):
            index = [i for i, name in enumerate(self._names) if name in self._fixed]
            if not index:
                return super()._optimize_acqf(acqf, best_params)
            values = [self._fixed[self._names[i]] for i in index]
            if best_params is not None:
                best_params = best_params.copy()
                best_params[:, index] = values
            x = np.array(super()._optimize_acqf(PinnedAcqf(acqf, index, values), best_params), dtype=float)
            x[index] = values
            return x

    return PinnedGPSampler


# ---------------------------------------------------------------------------------------------------------------------
class Campaign:
    """A campaign: its configuration, its store, and the way evaluations are run (runner = evaluate.run, or an
    in-process stand-in with the same signature in tests)."""

    def __init__(self, cfg, runner=evaluate.run):
        self.cfg = cfg.validate()
        for key in ('base_config_file', 'physics_python'):
            if getattr(cfg, key) and not Path(getattr(cfg, key)).is_file():
                raise config.ConfigError(f"[mobo] {key} {getattr(cfg, key)} does not exist")
        self.store = Store(cfg.campaign_dir)
        self.runner = runner
        self._base = {}
        self.check_overrides()

    # ---- configuration of one evaluation
    def base(self, fidelity):
        """The base run config with the fixed and the fidelity's overrides (what the geometry key hashes)."""
        if fidelity not in self._base:
            sets = list(self.cfg.fixed) + list(self.cfg.fidelity(fidelity).overrides)
            self._base[fidelity] = config.load(self.cfg.base_config_file, overrides=sets)
        return self._base[fidelity]

    def check_overrides(self):
        """Every override must name a key that its [section] knows, with a value it accepts: a misspelt target would
        leave every evaluation of the campaign the same design. Each fidelity's base is checked, and each variable at
        both ends of its range. Sections without a registered dataclass (the toy's) are not checked."""
        cfg = self.cfg
        items = list(cfg.fixed) + [s for f in cfg.fidelities.values() for s in f.overrides] + [v.target + '='
                                                                                              for v in cfg.variables]
        names = sorted({s.split('.')[0] for s in items} & set(config.SECTIONS))
        cases = [(name, []) for name in cfg.fidelities]
        cases += [(cfg.search_fidelity, [v.override(v.physical(u))]) for v in cfg.variables for u in (0.0, 1.0)]
        for fidelity, sets in cases:
            run = copy.deepcopy(self.base(fidelity))
            try:
                for item in sets:
                    config.set_dotted(run, *config.parse_override(item))
                for name in names:
                    config.section(run, name)
            except ImportError as ex:
                warnings.warn(f"[mobo] overrides not checked: {ex}", stacklevel=2)
                return
            except (config.ConfigError, TypeError, ValueError) as ex:
                raise config.ConfigError(f"[mobo] {' '.join(sets) or f'fidelity {fidelity!r}'}: {ex}") from None

    def current(self, v):
        """The value a variable's target has in the base config: set there, else its section's default."""
        base = self.base(self.cfg.search_fidelity)
        x = config.get_dotted(base, v.target, space.MISSING)
        name, _, rest = v.target.partition('.')
        if x is space.MISSING and name in config.SECTIONS:
            try:
                x = config.get_dotted(config.to_dict(config.section(base, name)), rest)
            except ImportError:
                x = None
        if v.as_list and isinstance(x, (list, tuple)):
            x = x[0] if x else None
        return None if x is space.MISSING else x

    def proposal(self, fidelity, kind, unit, params=None, **extra):
        """Everything an evaluation needs, from its unit-cube coordinates (or the physical values themselves, for
        a warm start or a verify): physical values, geometry key, the full override list (fixed, fidelity,
        variables, then the geometry's model and fields paths)."""
        cfg = self.cfg
        params = space.physical(cfg, unit) if params is None else params
        key = space.geometry_key(self.base(fidelity), cfg, params)
        sets = list(cfg.fixed) + list(cfg.fidelity(fidelity).overrides) + space.overrides(cfg, params)
        if key:
            folder = self.store.geometry_dir(key)
            sets += [f"paths.model3d={space.toml_value((folder / MODEL_FILE).as_posix())}",
                     f"paths.fields={space.toml_value((folder / FIELDS_FILE).as_posix())}"]
        return dict(fidelity=fidelity, kind=kind, unit=unit, params=params, geometry=key, overrides=sets, **extra)

    def search(self, evals):
        """The search-fidelity evaluations (not the re-evaluations of `verify`), in order."""
        return [ev for ev in evals.values() if 'parent' not in ev]

    def values(self, ev):
        """(objectives, constraints) of a done evaluation, None when it has none or a value is not finite."""
        if ev.get('state') != 'done':
            return None
        try:
            obj = space.objective_values(self.cfg, ev.get('scores', {}))
            con = space.constraint_values(self.cfg, ev.get('scores', {}))
        except KeyError:
            return None
        return (obj, con) if all(math.isfinite(v) for v in obj + con) else None

    def check_metrics(self, evals):
        """A metric that no scored evaluation has is a misnamed one: stop instead of feeding the sampler nothing.
        (Runs the scorer skipped - too few survivors - carry no metrics and do not count.)"""
        done = [ev.get('scores', {}) for ev in evals.values() if ev.get('state') == 'done']
        done = [s for s in done if 'skip' not in s]
        if not done:
            return
        for m in [o.metric for o in self.cfg.objectives] + [c.metric for c in self.cfg.constraints]:
            if all(config.get_dotted(s, m, space.MISSING) is space.MISSING for s in done):
                raise config.ConfigError(f"[mobo] metric {m!r} is in no evaluation's scores; they have: "
                                         f"{', '.join(sorted(done[-1])) or 'nothing'}")

    # ---- the optuna study, rebuilt from the store
    def study(self, evals, sampler):
        """The study of the search evaluations so far. A done evaluation enters with its objective and constraint
        values. A run the scorer skipped (too few survivors to score: a design that lost its beam) is known to be
        bad: it enters with the values its scores have and the worst ones seen for the rest. A failed evaluation is
        a pending trial, or with failed_as_infeasible a complete one with the worst values seen and the extra
        constraint 'evaluated' violated. Stand-in values need values seen: until then those are pending too."""
        optuna = import_optuna()
        cfg = self.cfg
        dists = {v.param: optuna.distributions.FloatDistribution(0.0, 1.0) for v in cfg.variables}
        study = optuna.create_study(directions=[o.direction for o in cfg.objectives], sampler=sampler)
        rows = [ev for ev in self.search(evals) if ev['state'] in ('done', 'failed') and not ev.get('interrupted')]
        vals = [self.values(ev) for ev in rows]
        good = [v for v in vals if v is not None]
        worst_obj = [(max if o.direction == 'minimize' else min)(v[0][i] for v in good) if good else math.nan
                     for i, o in enumerate(cfg.objectives)]
        worst_con = [max(v[1][i] for v in good) if good else math.nan for i in range(len(cfg.constraints))]
        labels = [c.label() for c in cfg.constraints] + ['evaluated'] * cfg.failed_as_infeasible
        for ev, v in zip(rows, vals):
            params = {var.param: ev['unit'][var.name] for var in cfg.variables}
            skipped = v is None and ev['state'] == 'done' and 'skip' in (ev.get('scores') or {})
            if v is None and not (good and (skipped or cfg.failed_as_infeasible)):
                study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.RUNNING, params=params,
                                                          distributions=dists))
                continue
            if v is not None:
                obj, con = v
            elif skipped:
                obj, con = self.stand_in(ev.get('scores') or {}, worst_obj, worst_con)
            else:
                obj, con = worst_obj, worst_con
            con = con + [0.0 if v is not None or skipped else 1.0] * cfg.failed_as_infeasible
            study.add_trial(optuna.trial.create_trial(params=params, distributions=dists, values=obj,
                                                      constraints=dict(zip(labels, con)) or None))
        return study, dists

    def stand_in(self, scores, worst_obj, worst_con):
        """(objectives, constraints) of a run the scorer skipped: the metrics its scores have, the worst seen for
        the others."""
        def value(path, fallback, f):
            try:
                x = space.metric(scores, path)
            except KeyError:
                return fallback
            return f(x) if math.isfinite(x) else fallback

        cfg = self.cfg
        obj = [value(o.metric, w, lambda x, o=o: o.scale * x) for o, w in zip(cfg.objectives, worst_obj)]
        con = [value(c.metric, w, c.violation) for c, w in zip(cfg.constraints, worst_con)]
        return obj, con

    # ---- what to evaluate next
    def plan(self, evals):
        """('pinned', head evaluation) while the newest geometry has fewer than cheap_per_geometry cheap followers
        (interrupted ones do not count) and its fields were made, else ('warm', index) for the next warm-start point,
        else ('free', None)."""
        cfg = self.cfg
        search = self.search(evals)
        if cfg.geometry_variables and cfg.cheap_variables and cfg.cheap_per_geometry > 0:
            seen, head = set(), None
            for i, ev in enumerate(search):
                if ev['kind'] != 'pinned' and ev['geometry'] not in seen:
                    head = i
                seen.add(ev['geometry'])
            if head is not None:
                n_pinned = sum(ev['kind'] == 'pinned' and not ev.get('interrupted') for ev in search[head + 1:])
                built = (self.store.geometry_dir(search[head]['geometry']) / FIELDS_FILE).exists()
                if n_pinned < cfg.cheap_per_geometry and built:
                    return 'pinned', search[head]
        n_warm = sum(ev['kind'] == 'warm' for ev in search)
        if n_warm < len(cfg.warm_start):
            return 'warm', n_warm
        return 'free', None

    def warm_point(self, i):
        """Physical values of warm-start point i; a variable it omits takes the base config's value."""
        point = self.cfg.warm_start[i]
        params = {}
        for v in self.cfg.variables:
            x = point.get(v.name, self.current(v))
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not v.low <= x <= v.high or (
                    v.integer and x != int(x)):
                raise config.ConfigError(f"[mobo] warm_start[{i}]: {v.name} = {x!r} is not a value in "
                                         f"[{v.low:g}, {v.high:g}]" + ('' if v.name in point else
                                                                       f" (the base config's {v.target})"))
            params[v.name] = int(x) if v.integer else float(x)
        return params

    def propose(self, evals):
        cfg = self.cfg
        step = len(self.search(evals))
        kind, arg = self.plan(evals)
        if kind == 'warm':
            params = self.warm_point(arg)
            unit = {v.name: v.unit(params[v.name]) for v in cfg.variables}
            return self.proposal(cfg.search_fidelity, 'warm', unit, params, step=step, warm=arg)
        seed = step_seed(cfg.seed, step)
        pinned = {v.name: arg['unit'][v.name] for v in cfg.geometry_variables} if kind == 'pinned' else {}
        sampler = make_sampler(cfg, seed, {f'u_{k}': u for k, u in pinned.items()})
        study, dists = self.study(evals, sampler)
        trial = study.ask(dists)
        unit = {v.name: v.snap(trial.params[v.param]) for v in cfg.variables}
        unit.update(pinned)
        extra = {'pinned_to': arg['id']} if pinned else {}
        return self.proposal(cfg.search_fidelity, kind, unit, step=step, sampler=cfg.sampler, seed=seed, **extra)

    # ---- running evaluations
    def add(self, eval_id, proposal):
        self.store.append(eval_id, 'proposed', **proposal)
        return self.store.evaluations()[eval_id]

    def execute(self, ev, lock, deadline=None):
        """Run one evaluation and record its outcome; returns why it was stopped from outside ('' = it was not):
        'stopped by the user' or 'the time budget ran out' (deadline: time.monotonic() of the end of max_hours)."""
        cfg = self.cfg
        folder = self.store.eval_dir(ev['id'])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / evaluate.RESULT).unlink(missing_ok=True)
        builds = False                    # whether this evaluation makes its geometry's model and fields
        if ev['geometry']:
            gdir = self.store.geometry_dir(ev['geometry'])
            builds = not (gdir / FIELDS_FILE).exists()
            gfile = gdir / 'geometry.json'
            if not gfile.exists():
                gfile.parent.mkdir(parents=True, exist_ok=True)
                gfile.write_text(json.dumps(space.geometry_parts(self.base(ev['fidelity']), cfg, ev['params']),
                                            indent=1, sort_keys=True, default=str))
        cmd = evaluate.command(cfg.physics_python, cfg.evaluator, cfg.base_config_file, ev['overrides'], folder)
        (folder / 'command.txt').write_text(subprocess.list2cmdline(cmd) + '\n')
        geo = f" (geometry {ev['geometry']}{', new' if builds else ''})" if ev['geometry'] else ''
        log(f"{ev['id']} {ev['kind']}{geo}: {self.design(ev)}")

        def started(pid):
            self.store.append(ev['id'], 'running', pid=pid, **({'builds': True} if builds else {}))

        limit = 60.0 * cfg.fidelity(ev['fidelity']).timeout_min
        left = deadline - time.monotonic() if deadline else 0.0
        timeout = min([t for t in (limit, max(left, 1e-3) if deadline else 0.0) if t > 0], default=0.0)
        outcome = self.runner(cmd, folder, timeout_s=timeout, env=evaluate.child_env(cfg.physics_python),
                              on_start=started, keep_fds=lock.inherit)
        stop = 'stopped by the user' if outcome.interrupted else ''
        if outcome.timed_out and deadline and (not limit or left < limit):
            stop = 'the time budget ran out'
        self.finish(ev, outcome, stop=stop, builds=builds)
        return stop

    def set_aside(self, ev):
        """The evaluation was to make its geometry's model and it ended without one. A build commits its outputs one
        by one, the model last, and never overwrites an output, so what it left would stop every later build of that
        geometry: move it to <geometry>/stopped-<id>/. (The model and the field cache are written whole: once they
        exist they are complete, and they stay.)"""
        folder = self.store.geometry_dir(ev['geometry'])
        if (folder / MODEL_FILE).exists() or not folder.is_dir():
            return
        left = [f for f in folder.iterdir() if f.is_file() and f.name != 'geometry.json']
        if not left:
            return
        dest = folder / f"stopped-{ev['id']}"
        dest.mkdir(exist_ok=True)
        for f in left:
            for attempt in range(20):             # a process of the stopped tree may still be closing the file
                try:
                    f.replace(dest / f.name)
                    break
                except PermissionError:
                    time.sleep(0.25)
            else:
                log(f"{ev['id']}: WARNING could not move {f} aside: remove it before geometry {ev['geometry']} is "
                    f"built again")
        log(f"{ev['id']}: what the build of geometry {ev['geometry']} left ({', '.join(f.name for f in left)}) is in "
            f"{dest}")

    def finish(self, ev, outcome, stop='', how='', builds=False):
        """Record the outcome of an evaluation: done with its scores, failed with the reason, or failed and
        'interrupted' when it was stopped from outside (stop = the reason). builds: it was to make its geometry's
        model and fields; when it failed, what an unfinished build left is moved aside (``set_aside``)."""
        cfg, r = self.cfg, outcome.result
        common = dict(wall_s=round(float(outcome.wall_s), 2), returncode=outcome.returncode, **({'how': how} if how else {}))
        error = ''
        if stop:
            error = f'interrupted: {stop}'
        elif outcome.timed_out:
            error = f"timeout after {cfg.fidelity(ev['fidelity']).timeout_min:g} min"
        elif r is None:
            error = (f"no {evaluate.RESULT} (exit code {outcome.returncode}): "
                     f"{evaluate.tail(self.store.eval_dir(ev['id']) / 'stderr.log')}")
        elif r['status'] != 'ok':
            error = r.get('error') or f"failed (exit code {outcome.returncode})"
        if error:
            self.store.append(ev['id'], 'failed', error=error, interrupted=bool(stop), **common)
            log(f"{ev['id']} failed after {outcome.wall_s:.0f} s: {error}")
            if builds:
                self.set_aside(ev)
            return
        scores = r.get('scores') or {}
        done = {'state': 'done', 'scores': scores}
        v = self.values(done)
        self.store.append(ev['id'], 'done', scores=scores, objectives=v[0] if v else None,
                          constraints=v[1] if v else None, **common)
        log(f"{ev['id']} done in {outcome.wall_s:.0f} s: " + self.describe(done))
        self.check_metrics(self.store.evaluations())

    def design(self, ev):
        return ' '.join(f"{v.name}={_fmt(ev['params'][v.name])}" for v in self.cfg.variables)

    def describe(self, ev):
        """One line of an evaluation's objectives and constraints."""
        scores = ev.get('scores', {})
        if 'skip' in scores:
            return f"not scored ({scores['skip']})"
        text = ' '.join(f"{o.metric}={_fmt(config.get_dotted(scores, o.metric))}" for o in self.cfg.objectives)
        for c in self.cfg.constraints:
            x = config.get_dotted(scores, c.metric)
            text += f" | {c.label()}: {_fmt(x)}" + (' (violated)' if x is not None and not c.violation(x) <= 0 else '')
        return text

    def recover(self):
        """Settle what a stopped driver left: an evaluation still marked running ended with it (its process tree
        is gone: see evaluate.run). Adopt the result it wrote, else record it interrupted."""
        for ev in self.store.evaluations().values():
            if ev['state'] == 'running':
                r = evaluate.read_result(self.store.eval_dir(ev['id']))
                outcome = evaluate.Outcome(wall_s=r['wall_s'] if r else 0.0, result=r)
                self.finish(ev, outcome, stop='' if r else 'the driver stopped while it ran',
                            how='adopted after the driver stopped' if r else '', builds=bool(ev.get('builds')))

    def preflight(self):
        """The physics interpreter must find the evaluator: a broken environment would otherwise fail every
        evaluation of the budget, one after the other. Run like an evaluation (its tree stopped with the driver),
        logs in <campaign_dir>/preflight/; only with the real runner, a few seconds."""
        if self.runner is not evaluate.run:
            return
        cfg = self.cfg
        python, module = cfg.physics_python or sys.executable, cfg.evaluator.split()[0]
        cmd = [python, '-c', f"import importlib.util, sys; sys.exit(importlib.util.find_spec({module!r}) is None)"]
        folder = self.store.dir / 'preflight'
        try:
            out = evaluate.run(cmd, folder, timeout_s=300.0, env=evaluate.child_env(cfg.physics_python))
        except OSError as ex:
            raise config.ConfigError(f"[mobo] the physics interpreter {python} does not run: {ex}") from None
        if out.interrupted:
            raise KeyboardInterrupt
        if out.returncode != 0:
            why = evaluate.tail(folder / 'stderr.log', 1) or ('no answer in 300 s' if out.timed_out else 'not found')
            raise config.ConfigError(f"[mobo] the physics interpreter {python} cannot import the evaluator "
                                     f"{module}: {why}")

    # ---- commands
    def optimize(self, max_evals=None, max_hours=None):
        """Run evaluations until max_evals search evaluations are spent (interrupted ones do not count) or
        max_hours have passed (the evaluation still running then is stopped); 130 when stopped by the user."""
        cfg = self.cfg
        max_evals = cfg.max_evals if max_evals is None else max_evals
        max_hours = cfg.max_hours if max_hours is None else max_hours
        deadline = time.monotonic() + 3600.0 * max_hours if max_hours else None
        with CampaignLock(self.store.lock_path) as lock:
            self.recover()
            self.preflight()
            self.check_metrics(self.store.evaluations())
            log(f"campaign {cfg.campaign_dir}: {self.spent(self.store.evaluations())} search evaluations so far, "
                f"budget {max_evals}" + (f" or {max_hours:g} h" if max_hours else ''))
            while True:
                evals = self.store.evaluations()
                search = self.search(evals)
                waiting = [ev for ev in search if ev['state'] == 'proposed']    # proposed, never started
                if deadline and time.monotonic() >= deadline:
                    log(f"budget spent: {max_hours:g} h")
                    break
                if waiting:
                    ev = waiting[0]
                elif self.spent(evals) >= max_evals:
                    log(f"budget spent: {self.spent(evals)} of max_evals = {max_evals} evaluations")
                    break
                else:
                    ev = self.add(self.store.next_id(evals), self.propose(evals))
                stop = self.execute(ev, lock, deadline)
                if stop == 'stopped by the user':
                    log("stopped by the user; `pcr optimize` again continues the campaign")
                    return 130
                if stop:
                    log(f"budget spent: {max_hours:g} h ({ev['id']} was stopped)")
                    break
        return 0

    def spent(self, evals):
        """Search evaluations that count against max_evals: every one with an outcome except the interrupted."""
        return sum(ev['state'] in ('done', 'failed') and not ev.get('interrupted') for ev in self.search(evals))

    def dry_run(self):
        """Print the campaign and the next evaluation's command; run and write nothing."""
        cfg = self.cfg
        evals = self.store.evaluations()
        log(f"campaign {cfg.campaign_dir}: {self.spent(evals)} of {cfg.max_evals} search evaluations in the store")
        log(f"base {cfg.base_config_file}; sampler {cfg.sampler}, seed {cfg.seed}, n_startup {cfg.n_startup}")
        for v in cfg.variables:
            log(f"  variable {v.name:12s} {v.kind:8s} {v.target} in [{v.low:g}, {v.high:g}]"
                + ' log' * v.log + ' integer' * v.integer + ' list' * v.as_list + f" (base: {_fmt(self.current(v))})")
        for o in cfg.objectives:
            log(f"  objective {o.direction} {o.metric}" + (f" x {o.scale:g}" if o.scale != 1 else ''))
        for c in cfg.constraints:
            log(f"  constraint {c.label()}")
        for name, f in cfg.fidelities.items():
            log(f"  fidelity {name}: {' '.join(f.overrides) or '(base config)'}"
                + (f", timeout {f.timeout_min:g} min" if f.timeout_min else ''))
        waiting = [ev for ev in self.search(evals) if ev['state'] == 'proposed']
        ev = waiting[0] if waiting else {'id': self.store.next_id(evals), **self.propose(evals)}
        cmd = evaluate.command(cfg.physics_python, cfg.evaluator, cfg.base_config_file, ev['overrides'],
                               self.store.eval_dir(ev['id']))
        log(f"next: {ev['id']} {ev['kind']} {self.design(ev)}")
        print(subprocess.list2cmdline(cmd))
        return 0

    def verify(self, ids=(), pareto=False, fidelity=None):
        """Re-evaluate search evaluations at the verify fidelity, as <id>.<fidelity> next to them, with exactly the
        physical values of the original. A failed or interrupted re-evaluation is run again (its folder is kept as
        <id>.<fidelity>.attempt<k>)."""
        fid = fidelity or self.cfg.verify_fidelity
        self.cfg.fidelity(fid)
        with CampaignLock(self.store.lock_path) as lock:
            self.recover()
            evals = self.store.evaluations()
            targets = [f"{int(i):04d}" if str(i).isdigit() else str(i) for i in ids]
            targets = list(dict.fromkeys(targets + (self.summary(evals)['pareto'] if pareto else [])))
            for pid in targets:
                if pid not in evals or 'parent' in evals[pid] or 'unit' not in evals[pid]:
                    raise config.ConfigError(f"no search evaluation {pid!r} in {self.store.log}")
            todo = [pid for pid in targets if evals.get(f"{pid}.{fid}", {}).get('state') != 'done']
            for pid in targets:
                if pid not in todo:
                    log(f"{pid}.{fid} is done already")
            if todo:
                self.preflight()
            for pid in todo:
                vid = f"{pid}.{fid}"
                folder = self.store.eval_dir(vid)
                if vid in evals and folder.exists():
                    folder.rename(folder.with_name(f"{vid}.attempt{evals[vid]['attempt']}"))
                ev = self.add(vid, self.proposal(fid, 'verify', evals[pid]['unit'], evals[pid]['params'], parent=pid))
                if self.execute(ev, lock):
                    return 130
        return 0

    def summary(self, evals=None):
        """Pareto set (search fidelity; feasible evaluations, or all usable ones while none is feasible) and the
        best feasible value of every objective."""
        cfg = self.cfg
        evals = self.store.evaluations() if evals is None else evals
        usable = [(ev['id'], v) for ev in self.search(evals) if (v := self.values(ev)) is not None]
        feasible = [(i, v) for i, v in usable if all(c <= 0 for c in v[1])]
        pool = feasible or usable
        sign = np.array([1.0 if o.direction == 'minimize' else -1.0 for o in cfg.objectives])
        pareto = [pool[k][0] for k in np.flatnonzero(pareto_mask([sign * v[0] for _, v in pool]))] if pool else []
        best = {}
        for j, o in enumerate(cfg.objectives):
            if feasible:
                i, v = (min if o.direction == 'minimize' else max)(feasible, key=lambda iv: iv[1][0][j])
                best[o.metric] = (v[0][j] / o.scale, i)
        return {'pareto': pareto, 'feasible': bool(feasible), 'best': best, 'n_usable': len(usable)}

    def status(self):
        cfg = self.cfg
        evals = self.store.evaluations()
        summ = self.summary(evals)
        head = (['id', 'kind', 'state', 'wall_s'] + [v.name for v in cfg.variables] + [o.metric for o in cfg.objectives]
                + [c.label() for c in cfg.constraints] + ['P'] + (['geometry'] if cfg.geometry_variables else []))
        rows, notes = [], []
        for ev in sorted(evals.values(), key=lambda e: e['id']):           # <id>.verify right after <id>
            scores = ev.get('scores', {})
            row = [ev['id'], ev.get('kind', '?'), ev['state'], _fmt(ev.get('wall_s'))]
            row += [_fmt(ev.get('params', {}).get(v.name)) for v in cfg.variables]
            row += [_fmt(config.get_dotted(scores, o.metric)) for o in cfg.objectives]
            row += [_fmt(config.get_dotted(scores, c.metric)) for c in cfg.constraints]
            row += ['*' if ev['id'] in summ['pareto'] else ''] + ([ev.get('geometry') or ''] if cfg.geometry_variables else [])
            rows.append(row)
            if ev['state'] == 'failed' or 'skip' in scores:
                notes.append(f"  {ev['id']}: {ev.get('error') or 'not scored: ' + str(scores.get('skip'))}")
        widths = [max(len(str(r[k])) for r in [head] + rows) for k in range(len(head))]
        print(f"campaign {cfg.campaign_dir}: {len(self.search(evals))} search evaluations ({self.spent(evals)} count "
              f"against max_evals = {cfg.max_evals}), {summ['n_usable']} usable")
        for r in [head] + rows:
            print('  '.join(str(x).rjust(w) for x, w in zip(r, widths)))
        print('\n'.join(notes))
        which = 'feasible' if summ['feasible'] else 'usable (none is feasible yet)'
        print(f"Pareto set at fidelity '{cfg.search_fidelity}', {which}: {', '.join(summ['pareto']) or '-'}")
        if summ['best']:
            print('best feasible: ' + ', '.join(f"{m} {_fmt(x)} ({i})" for m, (x, i) in summ['best'].items()))
        return 0


# ---------------------------------------------------------------------------------------------------------------------
def _stop_signal(signum, frame):
    raise KeyboardInterrupt(signal.Signals(signum).name)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='pcr', description='multi-objective optimisation of a central region')
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_opt = sub.add_parser('optimize', help='run (or continue) a campaign until its budget is spent')
    p_opt.add_argument('--max-evals', type=int, help='campaign budget of search evaluations (overrides max_evals)')
    p_opt.add_argument('--max-hours', type=float, help='hours of this run (overrides max_hours)')
    p_opt.add_argument('--dry-run', action='store_true', help='print the next evaluation; run and write nothing')
    p_ver = sub.add_parser('optimize-verify', help='re-evaluate evaluations at the verify fidelity')
    which = p_ver.add_mutually_exclusive_group(required=True)
    which.add_argument('--ids', nargs='+', help='evaluation ids, e.g. 3 7 0012')
    which.add_argument('--pareto', action='store_true', help="the search fidelity's Pareto set")
    p_ver.add_argument('--fidelity', help='fidelity name (default: verify_fidelity)')
    sub.add_parser('optimize-status', help='table of evaluations, Pareto set, best values')
    for p in sub.choices.values():
        p.add_argument('config', help='campaign TOML with a [mobo] section')
        p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='override of the campaign TOML')
    args = parser.parse_args(argv)
    # SIGTERM (and Ctrl+Break on Windows, SIGHUP on POSIX) stop the driver like Ctrl+C: the evaluation is stopped and
    # recorded before the driver exits
    stops = [s for s in (getattr(signal, n, None) for n in ('SIGTERM', 'SIGBREAK', 'SIGHUP')) if s is not None]
    previous = {s: signal.signal(s, _stop_signal) for s in stops}
    try:
        campaign = Campaign(load(args.config, args.set))
        if args.cmd == 'optimize':
            return campaign.dry_run() if args.dry_run else campaign.optimize(args.max_evals, args.max_hours)
        if args.cmd == 'optimize-verify':
            return campaign.verify(args.ids or (), args.pareto, args.fidelity)
        return campaign.status()
    except (config.ConfigError, CampaignBusy) as ex:
        print(f"pcr {args.cmd}: {ex}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"pcr {args.cmd}: stopped", file=sys.stderr)
        return 130
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


if __name__ == '__main__':
    sys.exit(main())
