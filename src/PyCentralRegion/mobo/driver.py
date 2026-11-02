"""Multi-objective Bayesian optimisation of central-region designs, with the full 3D model in the loop.

    pcr optimize        <mobo.toml> [--max-evals N] [--max-hours H] [--dry-run] [--set key=value ...]
    pcr optimize-verify <mobo.toml> (--ids ID ... | --pareto) [--fidelity NAME] [--set key=value ...]
    pcr optimize-status <mobo.toml> [--set key=value ...]

One evaluation of a design is the whole chain build -> 3D BEM -> bunch tracking with space charge -> scores, run as
a subprocess of the physics environment (``evaluate``). It costs minutes and one GPU, so the driver runs one
evaluation at a time and chooses each with a sample-efficient multi-objective sampler (optuna's TPE or Gaussian
process, both constrained) on the unit cube of the design variables (``space``).

Cost structure: a new electrode geometry needs a build and its 3D BEM fields (on the worked example the build takes
~20 s, the fields ~8 min, that evaluation ~10 min in all); the RF phase, dee voltage, a magnetic trim or a slit reuse
the fields and only re-track (1.5-2 min, most of it the tracking kernel's compile in every new process). After every
new geometry the driver therefore runs ``cheap_per_geometry`` evaluations with the geometry coordinates held at that
geometry's values and only the cheap ones proposed (TPE through optuna's PartialFixedSampler; the GP through a
subclass that maximises its acquisition over the free coordinates while its Gaussian processes keep seeing every
coordinate). Evaluations of one geometry share one model and one field map under ``<campaign_dir>/geometries/<key>/``.

Fidelities: the sampler learns from the search fidelity (e.g. 10k particles, SC 64^3 every 5 steps);
``optimize-verify`` re-evaluates chosen designs (e.g. the Pareto set) at the verify fidelity (every particle, 128^3
every step) and stores them next to the originals, so the search-to-verify shift of every score can be read off.
That shift measures the resolution only when the fidelities differ in nothing else (the same turns, the same
stopping energy).

State: the store (``store``) is the only state. Before every proposal the optuna study is rebuilt in memory from it
(``optuna.trial.create_trial`` with the constraint values set on the trial), with a sampler seeded from the
campaign seed and the evaluation's index: every proposal follows from the store alone, so a campaign stopped and
continued proposes what one uninterrupted run would have proposed from the same evaluations - an uninterrupted run's
designs up to the stop, and after it the proposals of a study without the evaluation that was running at the stop
(it is lost, see below). A proposal not yet started when the driver stopped is run first, and then nothing is lost.
The study places every design at its stored unit coordinates, so the variables are the campaign's for good: a
variable added, removed, renamed, re-ranged or re-targeted once evaluations exist would move every stored design
under the sampler, and the driver refuses to continue (``check_store``; a new search space is a new campaign_dir,
whose warm_start can repeat earlier designs). Objectives and constraints are read from the stored scores at every
proposal, so those may change: a constraint added later judges the earlier evaluations too.

What enters the study:
* a scored evaluation: a complete trial with its objective and constraint values;
* an evaluation that finished but whose scores cannot be used - the scorer skipped the run (too few tracks or
  survivors), or an objective or constraint metric is missing (no probe has enough complete passes for valley_all),
  null or not finite: the design lost its beam too early to be measured. That is a bad design, not an unknown one,
  and the sampler must learn to leave its region (a slit closed too far, an inflector turned too far): a complete
  trial with the worst values seen of every objective and constraint, made infeasible by one extra constraint
  'scored' (1 for it, 0 for the scored ones; present only while such a trial exists). The store records why
  (``unusable``) and optimize-status shows it;
* a failed evaluation (the build, the BEM or the tracking failed, or a timeout): a pending (RUNNING) trial, which
  keeps TPE (constant liar) and the GP (q-batch acquisition) from proposing that design again without inventing
  values for it; with ``failed_as_infeasible`` the same infeasible stand-in as an unusable one. Stand-ins need
  values seen: until every objective and constraint has one, they are pending too;
* an evaluation stopped from outside (Ctrl+C, the time budget, a driver that died) says nothing about its design:
  it is recorded failed ('interrupted'), left out of the study and of the budget, and not run again - unless it is
  a warm-start point, which was asked for by name: that is evaluated again until it has an outcome.

The model and the field cache of a geometry are written whole, so a stop never leaves them cut short: once made they
serve the geometry's later evaluations. A build stopped before its model leaves outputs that the next build of that
geometry would refuse to overwrite: those are moved to ``geometries/<key>/stopped-<id>/``.

A metric no scored evaluation has (no run the scorer skipped counts) is a misnamed one and stops the campaign; a
metric that ``pcr evaluate``'s scorer documents (``analysis.objectives.METRICS``) is never taken for misnamed, since a
lossy run lacks some of those legitimately.

The driver never schedules or restarts itself: it runs until its budget is spent or it is stopped, and running it
again continues the campaign. Before the first evaluation it checks that the physics interpreter finds the evaluator
(``preflight``), and a run stops after ``max_failed_in_a_row`` failed evaluations in a row: a physics environment
that cannot run an evaluation would otherwise spend the whole budget on failures. When it stops - Ctrl+C, SIGTERM /
Ctrl+Break, the end of max_hours, or killed - the running evaluation stops with it, with every process it started
(``evaluate.run``).

The driver needs only the standard library, numpy and optuna >= 5 (torch and scipy for the GP sampler); it never
imports the physics modules (the [section] dataclasses it checks the overrides against are plain dataclasses).
"""
import argparse
import copy
import functools
import importlib.util
import itertools
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


def value(scores, path):
    """A score as a float, nan when it is missing, null or not a finite number."""
    try:
        return space.metric(scores, path)
    except KeyError:
        return math.nan


def runs_pcr_evaluate(evaluator):
    """Whether the evaluator is `pcr evaluate`: its run config is read through the [section]s of config.SECTIONS."""
    return evaluator.split()[:1] == ['PyCentralRegion.cli']


@functools.lru_cache(maxsize=None)
def scorer_metrics(evaluator):
    """The metrics `pcr evaluate`'s scorer documents (analysis.objectives.METRICS; importing it needs numpy only), when
    the evaluator is `pcr evaluate`; None for another evaluator or when the table does not import here."""
    if not runs_pcr_evaluate(evaluator):
        return None
    try:
        from ..analysis.objectives import METRICS
    except ImportError:
        return None
    return frozenset(METRICS)


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
        leave every evaluation of the campaign the same design. Each fidelity's base is checked, and the variables of
        each [section] at every corner of their box (the campaign's other variables at the base's values): a
        section's own checks can relate its keys (a channel taper must start at r > 0), and a corner they refuse
        would fail every evaluation proposed near it. With `pcr evaluate` as the evaluator an override must also name
        one of the sections it reads (config.SECTIONS): one of another section would change nothing. Another
        evaluator's own sections (the toy's) are its business and are not checked."""
        cfg = self.cfg
        items = list(cfg.fixed) + [s for f in cfg.fidelities.values() for s in f.overrides]
        roots = {config.parse_override(s)[0][0] for s in items} | {v.target.split('.')[0] for v in cfg.variables}
        other = sorted(roots - set(config.SECTIONS))
        if other and runs_pcr_evaluate(cfg.evaluator):
            raise config.ConfigError(f"[mobo] override(s) of [{'], ['.join(other)}]: pcr evaluate reads only the "
                                     f"sections [{'], ['.join(config.SECTIONS)}]")
        names = sorted(roots & set(config.SECTIONS))
        boxes = {name: [v for v in cfg.variables if v.target.split('.')[0] == name] for name in names}
        cases = [(fidelity, [], names) for fidelity in cfg.fidelities]
        cases += [(cfg.search_fidelity, [v.override(v.physical(u)) for v, u in zip(box, corner)], [name])
                  for name, box in boxes.items() if box for corner in itertools.product((0.0, 1.0), repeat=len(box))]
        for fidelity, sets, check in cases:
            base = self.base(fidelity)
            run = copy.deepcopy({name: base.get(name, {}) for name in check})     # a section reads only its table
            try:
                for item in sets:
                    config.set_dotted(run, *config.parse_override(item))
                for name in check:
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
            sets += [f"paths.model3d={config.toml_value((folder / MODEL_FILE).as_posix())}",
                     f"paths.fields={config.toml_value((folder / FIELDS_FILE).as_posix())}"]
        return dict(fidelity=fidelity, kind=kind, unit=unit, params=params, geometry=key, overrides=sets, **extra)

    def search(self, evals):
        """The search-fidelity evaluations (not the re-evaluations of `verify`), in order."""
        return [ev for ev in evals.values() if 'parent' not in ev]

    def check_store(self, evals):
        """The study places every stored design at its stored unit coordinates: the variables must still map them
        onto the values the design was evaluated with, set on the same targets. A variable added, removed, renamed,
        re-ranged, re-targeted or mapped otherwise once evaluations exist would move every stored design under the
        sampler (or leave it without a coordinate): that is another search space, so another campaign. (A pinned
        follower carries its head's exact geometry values, which the unit cube gives back to within rounding.)"""
        cfg = self.cfg
        names = sorted(v.name for v in cfg.variables)

        def moved(ev):
            if sorted(ev.get('unit', {})) != names:
                return f"it has the variables {sorted(ev.get('unit', {}))}, the campaign {names}"
            for v in cfg.variables:
                x, y = ev.get('params', {}).get(v.name), v.physical(ev['unit'][v.name])
                if v.integer or isinstance(x, bool) or not isinstance(x, (int, float)):
                    same = x == y and type(x) is type(y)
                else:
                    same = math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12 * (v.high - v.low))
                if not same:
                    return f"its {v.name} was {x!r}, which the variable now maps to {y!r}"
                if v.override(x) not in ev.get('overrides', ()):
                    return f"its {v.name} was not set as {v.override(x)}"
            return ''

        for ev in self.search(evals):
            if why := moved(ev):
                raise config.ConfigError(
                    f"[mobo] the variables changed since evaluation {ev['id']} ({self.store.log}): {why}. A campaign "
                    f"keeps its search space: run the changed one in another campaign_dir (its warm_start can repeat "
                    f"designs of this one)")

    @property
    def metrics(self):
        """The metric of every objective, then of every constraint."""
        return [o.metric for o in self.cfg.objectives] + [c.metric for c in self.cfg.constraints]

    def unusable(self, scores):
        """Why the scores of an evaluation that finished cannot be used as they are ('' = they can): the scorer
        skipped the run, or a metric of an objective or a constraint is missing, null or not finite."""
        if 'skip' in scores:
            return f"not scored: {scores['skip']}"
        bad = []
        for m in dict.fromkeys(self.metrics):
            x = config.get_dotted(scores, m, space.MISSING)
            if x is space.MISSING:
                bad.append(f"no {m}")
            elif not math.isfinite(value(scores, m)):
                bad.append(f"{m} = {'null' if x is None else x}")
        return ', '.join(bad)

    def values(self, ev):
        """(objectives, constraints) of a done evaluation whose scores are usable, else None."""
        scores = ev.get('scores') or {}
        if ev.get('state') != 'done' or self.unusable(scores):
            return None
        return space.objective_values(self.cfg, scores), space.constraint_values(self.cfg, scores)

    def check_metrics(self, evals):
        """A metric that no scored evaluation has (no run the scorer skipped counts) is a misnamed one: stop instead
        of feeding the sampler nothing. A metric the scorer of `pcr evaluate` documents is never misnamed: a run that
        lost its beam early lacks some of them (valley_all needs probes with complete passes)."""
        done = [ev.get('scores') or {} for ev in evals.values() if ev.get('state') == 'done']
        done = [s for s in done if 'skip' not in s]
        if not done:
            return
        known = scorer_metrics(self.cfg.evaluator) or ()
        for m in dict.fromkeys(self.metrics):
            if m not in known and all(config.get_dotted(s, m, space.MISSING) is space.MISSING for s in done):
                raise config.ConfigError(f"[mobo] metric {m!r} is in no evaluation's scores; they have: "
                                         f"{', '.join(sorted(done[-1])) or 'nothing'}")

    def worst(self, evals):
        """The worst value seen of every objective and every constraint, over the finite values in the scores of the
        search evaluations (an unusable run's own metrics count too); None while one of them has none."""
        cfg = self.cfg
        scores = [ev.get('scores') or {} for ev in self.search(evals) if ev.get('state') == 'done']
        obj, con = [], []
        for o in cfg.objectives:
            xs = [o.scale * x for s in scores if math.isfinite(x := value(s, o.metric))]
            obj.append((max if o.direction == 'minimize' else min)(xs, default=math.nan))
        for c in cfg.constraints:
            con.append(max((c.violation(x) for s in scores if math.isfinite(x := value(s, c.metric))),
                           default=math.nan))
        return None if any(math.isnan(x) for x in obj + con) else (obj, con)

    # ---- the optuna study, rebuilt from the store
    def study(self, evals, sampler):
        """The study of the search evaluations so far (see the module docstring for what enters it how): a scored
        evaluation with its values; an unusable one, and with failed_as_infeasible a failed one, as a stand-in with
        the worst values seen and the extra constraint 'scored' violated; a failed one otherwise, and a stand-in
        while no values have been seen, as a pending trial; an interrupted one not at all."""
        optuna = import_optuna()
        cfg = self.cfg
        dists = {v.param: optuna.distributions.FloatDistribution(0.0, 1.0) for v in cfg.variables}
        study = optuna.create_study(directions=[o.direction for o in cfg.objectives], sampler=sampler)
        worst = self.worst(evals)
        trials = []                                    # (params, values or None = pending, constraints, stand-in)
        for ev in self.search(evals):
            if ev['state'] not in ('done', 'failed') or ev.get('interrupted'):
                continue
            params = {var.param: ev['unit'][var.name] for var in cfg.variables}
            v = self.values(ev)
            if v is not None:
                trials.append((params, *v, 0.0))
            elif worst is not None and (ev['state'] == 'done' or cfg.failed_as_infeasible):
                trials.append((params, *worst, 1.0))
            else:
                trials.append((params, None, None, None))
        extra = any(t[3] for t in trials)              # the constraint 'scored' while a stand-in is in the study
        labels = [c.label() for c in cfg.constraints] + ['scored'] * extra
        for params, obj, con, bad in trials:
            if obj is None:
                study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.RUNNING, params=params,
                                                          distributions=dists))
            else:
                study.add_trial(optuna.trial.create_trial(params=params, distributions=dists, values=obj,
                                                          constraints=dict(zip(labels, con + [bad] * extra)) or None))
        return study, dists

    # ---- what to evaluate next
    def plan(self, evals):
        """('pinned', head evaluation) while the newest geometry has fewer than cheap_per_geometry cheap followers
        (interrupted ones do not count) and its fields were made, else ('warm', index) for the first warm-start point
        without an outcome (an interrupted one is evaluated again: unlike a proposal, it was asked for), else
        ('free', None)."""
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
        done = {ev.get('warm') for ev in search if ev['kind'] == 'warm' and not ev.get('interrupted')}
        todo = [i for i in range(len(cfg.warm_start)) if i not in done]
        return ('warm', todo[0]) if todo else ('free', None)

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
        # a pinned proposal evaluates its head's geometry with the head's own physical values: a warm start's need not
        # survive the round trip through the unit cube (4.726034523988023 deg comes back ...024), and one ulp would be
        # another geometry key, so another build and BEM
        params = {**space.physical(cfg, unit), **{k: arg['params'][k] for k in pinned}}
        extra = {'pinned_to': arg['id']} if pinned else {}
        return self.proposal(cfg.search_fidelity, kind, unit, params, step=step, sampler=cfg.sampler, seed=seed,
                             **extra)

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
                                            indent=1, sort_keys=True, default=config.json_default))
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
        common = dict(wall_s=round(float(outcome.wall_s), 2), returncode=outcome.returncode,
                      **({'how': how} if how else {}))
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
        v, why = self.values(done), self.unusable(scores)
        self.store.append(ev['id'], 'done', scores=scores, objectives=v[0] if v else None,
                          constraints=v[1] if v else None, **({'unusable': why} if why else {}), **common)
        log(f"{ev['id']} done in {outcome.wall_s:.0f} s: " + self.describe(done)
            + (f" -> unusable ({why}): a bad design to the sampler" if why else ''))
        self.check_metrics(self.store.evaluations())

    def design(self, ev):
        return ' '.join(f"{v.name}={_fmt(ev['params'][v.name])}" for v in self.cfg.variables)

    def describe(self, ev):
        """One line of an evaluation's objectives and constraints ('-' = missing or null)."""
        scores = ev.get('scores') or {}
        if 'skip' in scores:
            return f"not scored ({scores['skip']})"
        text = ' '.join(f"{o.metric}={_fmt(config.get_dotted(scores, o.metric))}" for o in self.cfg.objectives)
        for c in self.cfg.constraints:
            x = value(scores, c.metric)
            text += (f" | {c.label()}: {_fmt(config.get_dotted(scores, c.metric))}"
                     + (' (violated)' if math.isfinite(x) and c.violation(x) > 0 else ''))
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
        max_hours have passed (the evaluation still running then is stopped); 130 when stopped by the user, 1 after
        max_failed_in_a_row failed evaluations in a row (a broken environment would otherwise spend the budget)."""
        cfg = self.cfg
        max_evals = cfg.max_evals if max_evals is None else max_evals
        max_hours = cfg.max_hours if max_hours is None else max_hours
        deadline = time.monotonic() + 3600.0 * max_hours if max_hours else None
        with CampaignLock(self.store.lock_path) as lock:
            self.recover()
            self.check_store(self.store.evaluations())
            self.preflight()
            self.check_metrics(self.store.evaluations())
            log(f"campaign {cfg.campaign_dir}: {self.spent(self.store.evaluations())} search evaluations so far, "
                f"budget {max_evals}" + (f" or {max_hours:g} h" if max_hours else ''))
            failed = 0                                  # failed evaluations in a row, in this run
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
                rec = self.store.evaluations()[ev['id']]
                failed = failed + 1 if rec['state'] == 'failed' else 0
                if cfg.max_failed_in_a_row and failed >= cfg.max_failed_in_a_row:
                    log(f"stopped: {failed} evaluations in a row failed (max_failed_in_a_row), the last ({ev['id']}) "
                        f"with: {rec.get('error')}. That points at the environment or the campaign, not at the "
                        f"designs: see {self.store.eval_dir(ev['id']) / 'stderr.log'}. `pcr optimize` again "
                        f"continues the campaign")
                    return 1
        return 0

    def spent(self, evals):
        """Search evaluations that count against max_evals: every one with an outcome except the interrupted."""
        return sum(ev['state'] in ('done', 'failed') and not ev.get('interrupted') for ev in self.search(evals))

    def dry_run(self):
        """Print the campaign and the next evaluation's command; run and write nothing."""
        cfg = self.cfg
        evals = self.store.evaluations()
        self.check_store(evals)
        log(f"campaign {cfg.campaign_dir}: {self.spent(evals)} of {cfg.max_evals} search evaluations in the store")
        log(f"base {cfg.base_config_file}; sampler {cfg.sampler}, seed {cfg.seed}, n_startup {cfg.n_startup}")
        for v in cfg.variables:
            log(f"  variable {v.name:12s} {v.kind:8s} {v.target} in [{v.low:g}, {v.high:g}]"
                + ' log' * v.log + ' integer' * v.integer + ' list' * v.as_list + f" (base: {_fmt(self.current(v))})")
        known = scorer_metrics(cfg.evaluator)
        note = " (not in the scorer's table analysis.objectives.METRICS: checked against the first scored evaluation)"
        unlisted = {m: '' if known is None or m in known else note for m in self.metrics}
        for o in cfg.objectives:
            log(f"  objective {o.direction} {o.metric}" + (f" x {o.scale:g}" if o.scale != 1 else '')
                + unlisted[o.metric])
        for c in cfg.constraints:
            log(f"  constraint {c.label()}" + unlisted[c.metric])
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
            self.check_store(evals)
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
            scores = ev.get('scores') or {}
            why = self.unusable(scores) if ev['state'] == 'done' else ''
            row = [ev['id'], ev.get('kind', '?'), ev['state'] + '!' * bool(why), _fmt(ev.get('wall_s'))]
            row += [_fmt(ev.get('params', {}).get(v.name)) for v in cfg.variables]
            row += [_fmt(config.get_dotted(scores, o.metric)) for o in cfg.objectives]
            row += [_fmt(config.get_dotted(scores, c.metric)) for c in cfg.constraints]
            row += ['*' if ev['id'] in summ['pareto'] else '']
            row += [ev.get('geometry') or ''] if cfg.geometry_variables else []
            rows.append(row)
            if ev['state'] == 'failed':
                notes.append(f"  {ev['id']}: {ev.get('error')}")
            elif why:
                notes.append(f"  {ev['id']}: unusable scores ({why}): a bad design to the sampler")
        widths = [max(len(str(r[k])) for r in [head] + rows) for k in range(len(head))]
        print(f"campaign {cfg.campaign_dir}: {len(self.search(evals))} search evaluations ({self.spent(evals)} count "
              f"against max_evals = {cfg.max_evals}), {summ['n_usable']} usable")
        for r in [head] + rows:
            print('  '.join(str(x).rjust(w) for x, w in zip(r, widths)))
        if notes:
            print("failed and unusable (done!: finished, but the scores lack a metric; such a design enters the study "
                  "as infeasible, with the worst values seen):")
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
    previous = {s: signal.signal(s, _stop_signal) for s in evaluate.STOP_SIGNALS if s != signal.SIGINT}
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
