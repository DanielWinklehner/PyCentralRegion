"""The search space of a MOBO campaign: the [mobo] section, the unit cube, objectives, constraints, geometry keys.

The optimiser never sees physical units. Every design variable is a parameter ``u_<name>`` in [0, 1] that maps
linearly (or logarithmically) onto [low, high]. Two reasons: the GP's length scales and TPE's Parzen windows then
mean the same on every axis, and a proposal can hold some coordinates fixed (the geometry of an electrode model
that is already built) by passing their unit values unchanged, because for a [0, 1] float parameter optuna's
normalised value is the unit value itself. Integer variables (integer low and high) map [0, 1] onto
[low - 0.5, high + 0.5] and round, so every integer gets the same share of the cube; the unit value is then snapped
to the unit value of that integer (``Variable.snap``), so that two evaluations of the same design carry the same
coordinates.

The stored unit coordinates are what the sampler sees of every evaluation, so the variables are fixed for a campaign:
the driver refuses to continue a store that they no longer map onto its evaluated values (``Campaign.check_store``).

A variable is one of two kinds. 'geometry' variables change the electrodes, so a new value needs a new build and
3D BEM (about 10 min); 'cheap' variables (RF phase, dee voltage, a magnetic trim, a slit) reuse the fields and only
re-track (1-2 min). The driver uses the kind to batch cheap evaluations on each new geometry, and the geometry key (a
hash of everything the build and the BEM read) to share one model and one field map between the evaluations of a
geometry.

Objectives and constraints are read from an evaluation's ``scores`` by dotted path. Constraints follow optuna's
convention: a value <= 0 is feasible, so ``metric >= value`` is stored as ``value - metric``.
"""
import dataclasses
import hashlib
import json
import math
import warnings

from ..config import ConfigError, from_dict, get_dotted, json_default, toml_value

KINDS = ('cheap', 'geometry')
DIRECTIONS = ('maximize', 'minimize')
OPS = ('>=', '<=')
SAMPLERS = ('tpe', 'gp')
# the parts of a run configuration the electrode build and the BEM read (the inflector's rotation places the housing
# and the beam channel; the hand-off file gives the emitting plane): the geometry key hashes them
GEOMETRY_SECTIONS = ('geometry', 'machine', 'inflector', 'fields')
GEOMETRY_PATHS = ('seed', 'handoff', 'housing_outline', 'housing_step')
# config keys the driver and the evaluator own: no variable may set them
RESERVED_TARGETS = ('paths.model3d', 'paths.fields', 'run.out_dir')


@dataclasses.dataclass
class Variable:
    name: str = ''                   # short name; the optimiser's parameter is u_<name>
    target: str = ''                 # dotted config key the value overrides, e.g. track.volt_scale
    low: float = 0.0
    high: float = 1.0
    log: bool = False                # logarithmic mapping (low > 0)
    integer: bool = False            # integer values
    as_list: bool = False            # the target is a list: it is set to [value] (e.g. track.phases_deg)
    kind: str = 'cheap'              # 'geometry': needs a new build + BEM; 'cheap': same fields, only tracking

    @property
    def param(self):
        return f'u_{self.name}'

    def _range(self):
        lo, hi = (self.low - 0.5, self.high + 0.5) if self.integer else (self.low, self.high)
        return (math.log(lo), math.log(hi)) if self.log else (lo, hi)

    def physical(self, u):
        """Unit value -> physical value (clipped to [low, high]; an int for integer variables)."""
        lo, hi = self._range()
        x = lo + min(max(float(u), 0.0), 1.0) * (hi - lo)
        x = math.exp(x) if self.log else x
        if self.integer:
            return int(min(max(round(x), math.ceil(self.low)), math.floor(self.high)))
        return min(max(x, self.low), self.high)

    def unit(self, x):
        """Physical value -> unit value (an integer maps inside its cell: the centre, for a linear variable)."""
        lo, hi = self._range()
        x = math.log(x) if self.log else float(x)
        return min(max((x - lo) / (hi - lo), 0.0), 1.0)

    def snap(self, u):
        """The unit value of what is actually evaluated."""
        return self.unit(self.physical(u)) if self.integer else min(max(float(u), 0.0), 1.0)

    def override(self, x):
        return f"{self.target}={toml_value([x] if self.as_list else x)}"


@dataclasses.dataclass
class Objective:
    metric: str = ''                 # dotted path into the evaluation's scores, e.g. I_final
    direction: str = 'maximize'      # maximize | minimize
    scale: float = 1.0               # the sampler sees scale * metric


@dataclasses.dataclass
class Constraint:
    metric: str = ''                 # dotted path into the evaluation's scores
    op: str = '>='                   # '>=' | '<='
    value: float = 0.0

    def violation(self, x):
        """optuna's convention: <= 0 is feasible."""
        return self.value - x if self.op == '>=' else x - self.value

    def label(self):
        return f"{self.metric}{self.op}{self.value:g}"


@dataclasses.dataclass
class Fidelity:
    overrides: tuple = ()            # config overrides of this fidelity, e.g. ["track.n_final=10000"]
    timeout_min: float = 0.0         # an evaluation running longer is stopped and recorded failed; 0 = no limit


@dataclasses.dataclass
class MoboConfig:
    campaign_dir: str = ''           # the campaign's folder: evals.jsonl, evals/<id>/, geometries/<key>/
    base_config_file: str = ''       # run TOML every evaluation starts from
    physics_python: str = ''         # absolute path of the evaluations' interpreter; '' = the driver's own
    evaluator: str = 'PyCentralRegion.cli evaluate'   # module (and arguments) run with -m for one evaluation
    fixed: tuple = ()                # overrides applied to every evaluation, before the fidelity's
    variables: tuple[Variable, ...] = ()
    objectives: tuple[Objective, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    fidelities: dict = dataclasses.field(default_factory=lambda: {'search': Fidelity(), 'verify': Fidelity()})
    search_fidelity: str = 'search'  # the fidelity the sampler learns from
    verify_fidelity: str = 'verify'  # the fidelity of `pcr optimize-verify`
    sampler: str = 'tpe'             # tpe | gp
    seed: int = 0
    n_startup: int = 10              # evaluations before the sampler's model takes over from random sampling
    max_evals: int = 20              # budget: search-fidelity evaluations of the whole campaign (interrupted ones
    #                                  do not count)
    max_hours: float = 0.0           # budget of one `pcr optimize` run: then the running evaluation stops; 0 = none
    cheap_per_geometry: int = 3      # cheap evaluations, geometry held fixed, after every new geometry
    warm_start: tuple = ()           # points evaluated first: tables {name = physical value}; {} = the base design
    failed_as_infeasible: bool = False   # give failed evaluations to the sampler as infeasible trials
    max_failed_in_a_row: int = 5     # a `pcr optimize` run stops after this many failed evaluations in a row; 0 = never

    def __post_init__(self):
        self.fidelities = {k: v if isinstance(v, Fidelity) else from_dict(Fidelity, v, f"[mobo.fidelities.{k}]")
                           for k, v in dict(self.fidelities).items()}

    @property
    def geometry_variables(self):
        return [v for v in self.variables if v.kind == 'geometry']

    @property
    def cheap_variables(self):
        return [v for v in self.variables if v.kind == 'cheap']

    def fidelity(self, name):
        if name not in self.fidelities:
            raise ConfigError(f"[mobo] no fidelity {name!r}; defined: {', '.join(self.fidelities) or 'none'}")
        return self.fidelities[name]

    def validate(self):
        """Check what a campaign needs beyond well-formed tables; raises ConfigError naming the problem."""
        for key in ('campaign_dir', 'base_config_file'):
            if not getattr(self, key):
                raise ConfigError(f"[mobo] {key} is not set")
        if not self.variables or not self.objectives:
            raise ConfigError("[mobo] needs at least one [[mobo.variables]] and one [[mobo.objectives]]")
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names) or not all(names):
            raise ConfigError(f"[mobo] variable names must be unique and not empty: {names}")
        for v in self.variables:
            where = f"[mobo] variable {v.name!r}"
            if v.kind not in KINDS:
                raise ConfigError(f"{where}: kind {v.kind!r} is not one of {KINDS}")
            if '.' not in v.target or v.target in RESERVED_TARGETS or v.target.startswith('mobo.'):
                raise ConfigError(f"{where}: target {v.target!r} must be a dotted run-config key other than "
                                  f"{', '.join(RESERVED_TARGETS)} (the driver sets those)")
            if not v.low < v.high:
                raise ConfigError(f"{where}: needs low < high (got {v.low}, {v.high})")
            if v.integer and not (float(v.low).is_integer() and float(v.high).is_integer()):
                raise ConfigError(f"{where}: an integer variable needs integer low and high (got {v.low}, {v.high}): "
                                  f"else the integers at the ends get other shares of the unit cube than the rest")
            if v.log and (v.low - 0.5 * v.integer) <= 0:
                raise ConfigError(f"{where}: a log variable needs low > 0 (integer: low >= 1)")
            if v.kind == 'cheap' and v.target.split('.')[0] in GEOMETRY_SECTIONS:
                warnings.warn(f"{where} is 'cheap' but sets {v.target}, which the build or the BEM may read: "
                              f"its evaluations reuse one geometry's fields", stacklevel=2)
        for o in self.objectives:
            if not o.metric or o.direction not in DIRECTIONS or not o.scale > 0:
                raise ConfigError(f"[mobo] objective {o.metric!r}: needs a metric, a direction in {DIRECTIONS} "
                                  f"and a scale > 0")
        for c in self.constraints:
            if not c.metric or c.op not in OPS:
                raise ConfigError(f"[mobo] constraint {c.metric!r}: needs a metric and op in {OPS}")
        labels = [c.label() for c in self.constraints]
        if len(set(labels)) != len(labels):
            raise ConfigError(f"[mobo] constraints repeat: {labels}")
        if self.sampler not in SAMPLERS:
            raise ConfigError(f"[mobo] sampler {self.sampler!r} is not one of {SAMPLERS}")
        if self.max_failed_in_a_row < 0:
            raise ConfigError(f"[mobo] max_failed_in_a_row must be >= 0 (0 = never stop), got "
                              f"{self.max_failed_in_a_row}")
        self.fidelity(self.search_fidelity)
        for i, point in enumerate(self.warm_start):
            unknown = set(point) - set(names)
            if unknown:
                raise ConfigError(f"[mobo] warm_start[{i}]: unknown variable(s) {sorted(unknown)}")
        return self


# ---------------------------------------------------------------------------------------------------------------------
# values
def physical(cfg, unit):
    """{name: unit value} -> {name: physical value}."""
    return {v.name: v.physical(unit[v.name]) for v in cfg.variables}


def overrides(cfg, params):
    """The --set overrides of the design variables."""
    return [v.override(params[v.name]) for v in cfg.variables]


MISSING = object()


def metric(scores, path):
    """A score as a float; nan when it is not a finite number. KeyError when the scores do not have it at all."""
    value = get_dotted(scores, path, default=MISSING)
    if value is MISSING:
        raise KeyError(path)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def objective_values(cfg, scores):
    return [o.scale * metric(scores, o.metric) for o in cfg.objectives]


def constraint_values(cfg, scores):
    return [c.violation(metric(scores, c.metric)) for c in cfg.constraints]


# ---------------------------------------------------------------------------------------------------------------------
# geometry
def geometry_parts(base, cfg, params):
    """What one geometry's build and BEM depend on: the geometry-relevant parts of the base config (fidelity and
    fixed overrides applied) and the values of the geometry variables."""
    paths = base.get('paths', {})
    return {'sections': {s: base.get(s, {}) for s in GEOMETRY_SECTIONS},
            'paths': {p: paths.get(p, '') for p in GEOMETRY_PATHS},
            'variables': {v.target: params[v.name] for v in cfg.geometry_variables}}


def geometry_key(base, cfg, params):
    """Stable 12-hex-digit key of a geometry; None when the campaign has no geometry variables (the evaluations then
    use the base config's own model and fields)."""
    if not cfg.geometry_variables:
        return None
    blob = json.dumps(geometry_parts(base, cfg, params), sort_keys=True, separators=(',', ':'), default=json_default)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
