"""Run configuration: TOML files -> one nested dict -> typed sections.

A run is described by one TOML file. It may name other files in a top-level ``include = [...]`` (paths relative to
the including file); the including file wins over what it includes, later includes win over earlier ones, and
command-line overrides (``section.key=value``, the value parsed as TOML, else taken as a string) win over all files.

Each stage reads its own section into the dataclass its module defines::

    cfg = load('run.toml', overrides=['track.turns=4'])
    track = section(cfg, 'track', TrackConfig)

``section`` rejects keys the dataclass does not define, so a misspelt key fails instead of being ignored. The
documented defaults of every section are in ``configs/defaults/<section>.toml``; ``SECTIONS`` maps a section name to
its dataclass (imported only when asked for, so this module needs nothing beyond the standard library).

Paths: in the ``[paths]`` section, and for any key ending in ``_path``, ``_dir`` or ``_file`` anywhere, a relative
path is taken relative to the file that sets it. An empty string means "not given".

Roots: a string of a run file (an include, a path, an override value) may name a disk root, ``${name}/...``, so that a
run file names no drive and runs unchanged on another computer. The roots are this computer's, in its paths file, the
format every package of the chain reads (``configs/examples/paths.example.toml``)::

    [roots]                       # absolute paths; TOML literal strings keep Windows backslashes as they are
    inputs = 'E:\\data'
    work = '/mnt/results'

The paths file is the first of: ``load(paths_file=)`` (``pcr --paths FILE``), the environment variables
``MACHINE_PATHS`` and ``PYCENTRALREGION_PATHS``, ``~/.config/pycentralregion/paths.toml``. It is read only when a run
file names a root; an unknown name is an error that says which file was read (or where the search looked). ``$$`` is a
literal ``$``; a string with a root is normalised for the platform after the substitution. Tables other than
``[roots]`` belong to people (``[meta]``) or to other packages and are ignored. One root is built in: ``${pcr}``, the
PyCentralRegion source tree this module is part of, so that run files kept elsewhere include its defaults as
``include = ["${pcr}/configs/defaults/core.toml"]``. A path without a root works as before (absolute, or relative to
its file), so run files written with drive letters keep loading.

What the stages share to write values and files out lives here too: ``toml_value`` (the TOML literal that
``parse_override`` reads back), ``write_atomic`` (a file written whole under its temporary name ``part_path``, then
renamed by ``commit_part``) and the JSON conversions of numpy values ``json_default`` and ``json_clean``.
"""
import copy
import dataclasses
import importlib
import json
import math
import os
import re
import sys
import tomllib
import typing
from pathlib import Path

PATH_SUFFIXES = ('_path', '_dir', '_file')
PATHS_ENV = ('MACHINE_PATHS', 'PYCENTRALREGION_PATHS')            # this computer's paths file, in this order ...
PATHS_FILE = Path.home() / '.config' / 'pycentralregion' / 'paths.toml'   # ... else this one
ROOT_REF = re.compile(r'\$(?:(\$)|\{([A-Za-z_][A-Za-z0-9_]*)\}|(\{[^}]*\}?))')   # $$ | ${name} | a malformed ${...}
BUILTIN_ROOTS = ('pcr',)

# section name -> 'module:Class' of its dataclass
SECTIONS = {
    'run': 'PyCentralRegion.config:RunConfig',
    'paths': 'PyCentralRegion.config:PathsConfig',
    'machine': 'PyCentralRegion.config:MachineConfig',
    'magnet': 'PyCentralRegion.config:MagnetConfig',
    'inflector': 'PyCentralRegion.config:InflectorConfig',
    'geometry': 'PyCentralRegion.geometry.options:GeometryConfig',
    'post_rule': 'PyCentralRegion.geometry.options:PostRuleConfig',
    'fields': 'PyCentralRegion.config:FieldsConfig',
    'track': 'PyCentralRegion.config:TrackConfig',
    'slit': 'PyCentralRegion.config:SlitConfig',
    'btrim': 'PyCentralRegion.btrim:BTrimConfig',
    'score': 'PyCentralRegion.analysis.objectives:ScoreConfig',
    'mobo': 'PyCentralRegion.mobo.space:MoboConfig',
}


class ConfigError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------------------
# loading
def load(path, overrides=(), paths_file=None):
    """Read a TOML run file (with its includes) and apply ``key=value`` overrides. Returns a plain nested dict.
    ``${name}`` roots come from ``paths_file``, else from this computer's paths file (see the module docstring)."""
    path = Path(path).resolve()
    roots = Roots(paths_file)
    cfg = _load_file(path, (), roots)
    for item in overrides:
        keys, value = parse_override(item)
        set_dotted(cfg, keys, roots.expand(value, f"override {item!r}"))
    cfg['_source'] = str(path)
    return cfg


def use_paths_file(path):
    """Make ``path`` this process's paths file (``pcr --paths``), and that of the processes it starts: it becomes
    ``MACHINE_PATHS``, the first place the search looks."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"--paths {path}: no such file")
    os.environ[PATHS_ENV[0]] = str(path)
    return path


def find_paths_file(paths_file=None):
    """This computer's paths file: (path or None, the places looked at, in order)."""
    if paths_file:
        return Path(paths_file).expanduser(), [f"paths_file={paths_file}"]
    looked = []
    for var in PATHS_ENV:
        value = os.environ.get(var, '')
        looked.append(f"${var}" + (f" = {value}" if value else " (not set)"))
        if value:
            return Path(value).expanduser(), looked
    looked.append(str(PATHS_FILE))
    return (PATHS_FILE, looked) if PATHS_FILE.is_file() else (None, looked)


def _load_file(path, seen, roots=None):
    if path in seen:
        raise ConfigError(f"include loop: {' -> '.join(map(str, seen + (path,)))}")
    roots = roots or Roots()
    with open(path, 'rb') as fh:
        try:
            data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as ex:
            raise ConfigError(f"{path}: {ex}") from None
    data = roots.expand(data, str(path))
    _resolve_paths(data, path.parent, in_paths=False)
    includes = data.pop('include', [])
    if isinstance(includes, str):
        includes = [includes]
    base = {}
    for inc in includes:
        base = merge(base, _load_file((path.parent / inc).resolve(), seen + (path,), roots))
    return merge(base, data)


def source_tree():
    """The PyCentralRegion source tree this module belongs to (the ``${pcr}`` root): the folder that holds ``src/``
    and ``configs/``. None for an installed copy, which carries no ``configs/``."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / 'configs' / 'defaults').is_dir() else None


class Roots:
    """The ``${name}`` roots of run files: the built-in ``pcr`` (``source_tree``) and the ``[roots]`` table of this
    computer's paths file (``find_paths_file``), read when a run file first names a root it must hold. Roots are
    absolute paths (after ``~``); they are not checked for existence here."""

    def __init__(self, paths_file=None):
        self.paths_file = paths_file
        self.file = None
        self._table = None

    def table(self, where=''):
        if self._table is None:
            self.file, looked = find_paths_file(self.paths_file)
            if self.file is None or not self.file.is_file():
                raise ConfigError(f"{where}: names a root, but " + (
                    f"the paths file {self.file} does not exist" if self.file is not None else
                    "no paths file was found (looked at: " + '; '.join(looked) + ")") +
                    ". A paths file maps root names to this computer's folders, [roots] name = 'absolute path' "
                    "(configs/examples/paths.example.toml): pass it with pcr --paths FILE or set MACHINE_PATHS")
            try:
                data = tomllib.loads(self.file.read_text(encoding='utf-8'))
            except tomllib.TOMLDecodeError as ex:
                raise ConfigError(f"{self.file}: {ex}") from None
            loose = sorted(k for k, v in data.items() if not isinstance(v, dict))
            if loose:
                raise ConfigError(f"{self.file}: {', '.join(loose)} at the top level; a paths file holds [roots] "
                                  f"(and tables named after a package, for that package)")
            table = data.get('roots', {})
            bad = sorted(k for k, v in table.items() if not isinstance(v, str) or not v or '${' in v
                         or not Path(os.path.expanduser(v)).is_absolute())
            if bad:
                raise ConfigError(f"{self.file}: [roots] {', '.join(bad)}: a root is an absolute path (after ~), "
                                  f"not empty and naming no other root")
            clash = sorted(set(table) & set(BUILTIN_ROOTS))
            if clash:
                raise ConfigError(f"{self.file}: [roots] {', '.join(clash)} is built in and cannot be redefined")
            self._table = {k: os.path.expanduser(v) for k, v in table.items()}
        return self._table

    def get(self, name, where=''):
        if name == 'pcr':
            root = source_tree()
            if root is None:
                raise ConfigError(f"{where}: ${{pcr}} is the PyCentralRegion source tree, but this copy of the package "
                                  f"({Path(__file__).resolve().parent}) has no configs/ beside it: use a checkout "
                                  f"(PYTHONPATH=<checkout>/src or pip install -e)")
            return str(root)
        table = self.table(where)
        if name not in table:
            raise ConfigError(f"{where}: no root {name!r} in the paths file {self.file} "
                              f"(it has {', '.join(sorted(table)) or 'none'})")
        return table[name]

    def expand(self, value, where=''):
        """``value`` (a string, or a dict / list of them, as TOML reads it) with every ``${name}`` replaced by its root
        and ``$$`` by ``$``; a string that named a root is then normalised for the platform (``os.path.normpath``)."""
        if isinstance(value, str):
            if '$' not in value:
                return value
            named = []

            def sub(m):
                if m.group(1):
                    return '$'
                if m.group(3) is not None:
                    raise ConfigError(f"{where}: {value!r}: '${m.group(3)}' is not a root (${{name}} with an "
                                      f"identifier for a name; $$ for a literal $)")
                named.append(m.group(2))
                return self.get(m.group(2), where)

            out = ROOT_REF.sub(sub, value)
            return os.path.normpath(out) if named else out
        if isinstance(value, dict):
            return {k: self.expand(v, where) for k, v in value.items()}
        if isinstance(value, list):
            return [self.expand(v, where) for v in value]
        return value


def _resolve_paths(node, folder, in_paths):
    for key, value in node.items():
        if isinstance(value, dict):
            _resolve_paths(value, folder, in_paths=in_paths or key == 'paths')
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):     # array of tables
            for v in value:
                _resolve_paths(v, folder, in_paths=in_paths)
        elif isinstance(value, str) and value and (in_paths or key.endswith(PATH_SUFFIXES)):
            p = Path(value)
            node[key] = str(p if p.is_absolute() else (folder / p).resolve())


def merge(base, top):
    """Deep merge of two config dicts: tables merge key by key, everything else (arrays included) is replaced."""
    out = copy.deepcopy(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(item):
    """'track.turns=4' -> (['track', 'turns'], 4). The value is parsed as a TOML value, else kept as a string."""
    if '=' not in item:
        raise ConfigError(f"override {item!r} is not key=value")
    key, raw = item.split('=', 1)
    keys = [k.strip() for k in key.strip().split('.')]
    if not all(keys):
        raise ConfigError(f"override {item!r}: empty key")
    try:
        value = tomllib.loads(f"v = {raw.strip()}")['v']
    except tomllib.TOMLDecodeError:
        value = raw.strip()
    return keys, value


def toml_value(x):
    """A Python value as the TOML literal ``parse_override`` reads back: a bool, an int, a float (its repr, the shortest
    text that reads back to the same float, nan and inf included), a str (json.dumps: JSON's escapes are TOML's, but
    for characters beyond U+FFFF) or a list or tuple of them."""
    if isinstance(x, bool):
        return 'true' if x else 'false'
    if isinstance(x, int):
        return repr(int(x))
    if isinstance(x, float):
        return repr(float(x))
    if isinstance(x, str):
        return json.dumps(x)
    if isinstance(x, (list, tuple)):
        return '[' + ', '.join(toml_value(v) for v in x) + ']'
    raise TypeError(f"no TOML literal for {type(x).__name__} {x!r}")


def _split(keys):
    return keys.split('.') if isinstance(keys, str) else [str(k) for k in keys]


def get_dotted(cfg, keys, default=None):
    """``cfg['a']['b']`` for ``'a.b'``; a part that is an integer indexes an array (of tables), so
    ``'geometry.windows.0.half_height_mm'`` is the first window's half height. ``default`` when the path is absent."""
    node = cfg
    for k in _split(keys):
        if isinstance(node, dict) and k in node:
            node = node[k]
        elif isinstance(node, list) and k.isdigit() and int(k) < len(node):
            node = node[int(k)]
        else:
            return default
    return node


def set_dotted(cfg, keys, value):
    """Set ``cfg['a']['b']`` for ``'a.b'``, creating missing tables. An integer part indexes an array element that
    the config already has (an override can change one item of an array of tables, not create one)."""
    keys = _split(keys)
    where = '.'.join(keys)
    node = cfg
    for i, k in enumerate(keys):
        last = i == len(keys) - 1
        if isinstance(node, list):
            if not (k.isdigit() and int(k) < len(node)):
                raise ConfigError(f"cannot set {where}: {k!r} is not an index of an array of {len(node)} items")
            if last:
                node[int(k)] = value
                return
            node = node[int(k)]
        elif isinstance(node, dict):
            if last:
                node[k] = value
                return
            if k not in node and keys[i + 1].isdigit():
                raise ConfigError(f"cannot set {where}: there is no array {k!r} to index (set the whole array)")
            node = node.setdefault(k, {})
        else:
            raise ConfigError(f"cannot set {where}: {keys[i - 1]!r} is not a table or an array")


# ---------------------------------------------------------------------------------------------------------------------
# typed sections
def section_class(name):
    if name not in SECTIONS:
        raise ConfigError(f"unknown section [{name}]; known: {', '.join(SECTIONS)}")
    module, cls = SECTIONS[name].split(':')
    return getattr(importlib.import_module(module), cls)


def section(cfg, name, cls=None):
    """The [name] table of cfg as an instance of its dataclass (defaults for missing keys, error on unknown ones)."""
    cls = cls or section_class(name)
    data = cfg.get(name, {}) if isinstance(cfg, dict) else {}
    return from_dict(cls, data, where=f"[{name}]")


def from_dict(cls, data, where=''):
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a table, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {', '.join(unknown)}; known: {', '.join(fields)}")
    missing = [n for n, f in fields.items() if n not in data and _default_of(f) is dataclasses.MISSING]
    if missing:
        raise ConfigError(f"{where}: {', '.join(missing)} not set; these keys have no default"
                          + (" (the machine's properties: set them in the machine file the run includes, see "
                             "docs/machine.md)" if cls is MachineConfig else ""))
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, value in data.items():
        kwargs[name] = _coerce(hints.get(name), _default_of(fields[name]), value, f"{where} {name}")
    return cls(**kwargs)


def _default_of(field):
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return dataclasses.MISSING


def _coerce(hint, default, value, where):
    """Light coercion of TOML values to the field's type: int -> float, list -> tuple, table -> nested dataclass,
    list of tables -> list of dataclasses (when the annotation says so)."""
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if dataclasses.is_dataclass(hint) and isinstance(value, dict):
        return from_dict(hint, value, where)
    if origin in (list, tuple) and args and dataclasses.is_dataclass(args[0]) and isinstance(value, list):
        items = [from_dict(args[0], v, f"{where}[{i}]") for i, v in enumerate(value)]
        return tuple(items) if origin is tuple else items
    if (hint is float or isinstance(default, float)) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if (origin is tuple or isinstance(default, tuple)) and isinstance(value, list):
        return tuple(value)
    return value


def to_dict(obj):
    """A section dataclass (or any nested mix of dataclasses, tuples and dicts) as plain JSON-able data."""
    if dataclasses.is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def save_resolved(cfg, path):
    """Write the fully merged config as JSON next to a run's outputs (what the run actually used)."""
    with open(path, 'w') as fh:
        json.dump(cfg, fh, indent=1, sort_keys=True)


def require(value, what):
    """Return a given path, or fail with a message naming the missing config key."""
    if not value:
        raise ConfigError(f"{what} is not set")
    return value


# ---------------------------------------------------------------------------------------------------------------------
# outputs: files written whole, JSON
def part_path(path):
    """The temporary name of an output: <stem>.part<suffix> beside it. It keeps the extension, by which gmsh,
    matplotlib and numpy choose the format, and no stage takes a file of that name for a finished output."""
    p = Path(path)
    return p.with_name(p.stem + '.part' + p.suffix)


def commit_part(path, overwrite=True):
    """Rename the temporary file of ``path`` (``part_path``) to ``path``; returns ``path``. ``overwrite`` False refuses
    a ``path`` that exists, one that appeared while the file was written (ConfigError; the temporary file stays)."""
    path, tmp = Path(path), part_path(path)
    if not overwrite and path.exists():
        raise ConfigError(f"{path} appeared while it was written: not overwritten; the new file is {tmp}")
    os.replace(tmp, path)
    return path


def write_atomic(path, write, overwrite=True):
    """Write an output whole or not at all: ``write(tmp)`` writes it under its temporary name ``part_path(path)``, which
    ``commit_part`` then renames to ``path``. A killed run leaves no truncated output that a stage would reuse, and a
    ``write`` that raises leaves no temporary file. Returns ``path``."""
    tmp = part_path(path)
    try:
        write(tmp)
    except BaseException:
        try:                              # the error of the write is the one that propagates
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return commit_part(path, overwrite)


def _numpy():
    """numpy when it is loaded, else None: a numpy value comes with a loaded numpy, so the JSON conversions take numpy
    values without this module needing numpy."""
    return sys.modules.get('numpy')


def json_default(v):
    """``default`` of json.dumps: a numpy scalar or array as Python numbers and lists, anything else (a Path, a date)
    as its str. NaN and inf are left to json (NaN, Infinity), as track.json has them."""
    np = _numpy()
    if np is not None and isinstance(v, np.generic):
        return v.item()
    if np is not None and isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def json_clean(o):
    """Plain JSON data, as score.json and result.json hold scores: numpy scalars and arrays as Python numbers and lists,
    NaN and inf as None (null), mapping keys as str."""
    np = _numpy()
    if isinstance(o, dict):
        return {str(k): json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_clean(v) for v in o]
    if np is None:                                # no numpy loaded, so no numpy value
        return None if isinstance(o, float) and not math.isfinite(o) else o
    if isinstance(o, np.ndarray):
        return json_clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


# ---------------------------------------------------------------------------------------------------------------------
# sections of the core pipeline (the geometry, B-trim, score and MOBO sections live with their modules)
@dataclasses.dataclass
class RunConfig:
    name: str = ''                   # run name; output files carry it
    out_dir: str = ''                # results folder
    cache_dir: str = ''              # field-map crops and other reusable caches
    full_speed: bool = True          # Windows: opt the process out of power throttling, which moves a process without
    #                                  a window (a detached campaign, an evaluation of the driver) to the efficiency
    #                                  cores of a hybrid CPU: half the speed (process.keep_full_speed)


@dataclasses.dataclass
class PathsConfig:
    bfield_3d: str = ''              # 3D magnet map (COMSOL text export, x y z Bx By Bz)
    bfield_2d: str = ''              # midplane magnet map (optional: midplane check of the 3D crop)
    seed: str = ''                   # gap-geometry optimizer result (.pkl): gap lines, reference orbit, phase, f_rev
    handoff: str = ''                # injected bunch: openPMD plane-crossing file of the spiral inflector (.h5)
    inflector_efield: str = ''       # spiral-inflector static E-field (.pickle, PyPATools Field)
    housing_step: str = ''           # spiral-inflector housing solid (.stp)
    housing_outline: str = ''        # housing outline at the midplane (.txt) for the electrode builder
    model3d: str = ''                # 3D electrode model (.pkl, ElectrodeModel); written by `pcr build`
    fields: str = ''                 # BEM field cache (.pkl); written by `pcr fields`


@dataclasses.dataclass
class MachineConfig:
    """The machine's properties. None has a default: a run sets every one, usually in a machine file its designs
    include (docs/machine.md); a missing one is a ConfigError naming it."""
    species: str                     # PyPATools species name
    harmonic: int                    # RF harmonic h
    n_dees: int                      # dees (4-fold symmetric machines are supported)
    dee_shape: str                   # 'radial' (gap lines from straight segments); 'spiral' is not supported yet
    dee_center_angles_deg: tuple     # azimuth of every dee's centre line [deg]
    dee_opening_deg: float           # angular width of every dee
    dee_voltage_kv: float            # dee voltage the electrode model is built (and the BEM normalised) at
    design_voltage_kv: float         # dee voltage of the 1D gap model the seed geometry was optimized with
    gap_width_m: float               # nominal gap width of the dee-system template
    gap_angle_deg: float             # gap width law: max(gap_width_min, r * gap_angle)
    gap_width_min_m: float
    r_min_m: float                   # inner end of the gap lines
    r_max_cavity_m: float            # outer end of the gap lines
    r_max_m: float                   # tracking: particles beyond this radius are lost
    z_max_m: float                   # tracking: particles beyond |z| are lost
    target_energy_mev: float         # tracking: a run ends at the turn its reference reaches it (see [track])
    n_turns_design: int              # turns of the seed optimization (rebuilds its dee system)
    skip_turns_design: int


@dataclasses.dataclass
class MagnetConfig:
    sign: float = 1.0                # multiplies the map (the map's Bz sign vs the tracker's convention)
    scale: float = 1.0               # multiplies the map (field-level tuning)
    crop_r_m: float = 0.40           # radius of the 3D crop kept in memory
    flatten_r_m: float = 0.0         # > 0: flatten the central field inside this radius (Maxwell-consistent); 0 = off


@dataclasses.dataclass
class InflectorConfig:
    rotation_deg: float = 0.0        # rotation of the spiral inflector about the machine axis (field, housing, bunch)
    refine: int = 4                  # E-field grid refinement before a rotation (trilinear interpolation stays exact)
    refine_z_max_m: float = 0.030    # |z| kept when refining
    zshift_m: float = 0.0            # vertical shift of the inflector field
    flip_z: bool = False
    housing_scale: float = 1e-3      # housing STEP units -> m


@dataclasses.dataclass
class FieldsConfig:
    dxy_m: float = 0.0015            # BEM field grid spacing in x and y
    margin_m: float = 0.005          # grid half-width = model r_cut + margin
    z_half_m: float = 0.028          # grid covers |z| <= z_half
    dz_m: float = 0.002
    stack_z_half_m: float = 0.030    # the z slices of track.obstacles = 'raster': |z| range, spacing
    stack_dz_m: float = 0.001
    stack_dxy_m: float = 5e-4        # x-y node spacing of the metal test (columns and slices) and of the BEM mask
    max_elements: int = 80000        # refuse larger meshes
    tol: float = 1e-5                # GMRES tolerance
    solver: str = 'auto'             # BEM solver backend: auto | cupy | scipy | strong | dense
    symmetry: str = 'auto'           # auto: a mesh that is mirror symmetric about z = 0 (geometry.mirror_mesh) is solved
    #                                  for its upper half and its grid evaluated on z >= 0 only | off | require
    pot_device: str = 'gpu'          # potential evaluation: cpu | gpu
    pot_precision: str = 'single'    # single | double
    pot_chunk: int = 250000          # points per potential batch (on the GPU also held to one short launch)
    pot_check: int = 3000            # points re-evaluated in double precision as a check
    gpu_fit_safety: float = 1.05     # GPU memory head-room factor of the potential batches


@dataclasses.dataclass
class TrackConfig:
    pusher: str = 'boris_rel'
    steps_per_turn: int = 1000
    turns: int = 8
    stop_at_target_energy: bool = True   # end a run at the turn its reference reaches machine.target_energy_mev;
    #                                      false: every run tracks `turns` turns (the same turn for every design)
    fast3d: bool = True              # numba 3D fast path
    fast3d_interp: str = 'point'
    volt_scale: float = 1.0          # dee voltage / the model's voltage
    phases_deg: tuple = ()           # RF phases; several = a phase scan; empty = the seed's phase and +-10 deg
    n_scan: int = 4000               # particles per phase-scan run
    n_final: int = 0                 # particles of the final run; 0 = the whole hand-off bunch
    seed_scan: int = 0               # bunch sampling seeds
    seed_final: int = 2
    sc: bool = True                  # space charge
    sc_cells: int = 64               # phase scan: FFT grid nodes per axis (0 = cell size sc_h_m)
    sc_every: int = 5                # phase scan: steps between Poisson solves
    sc_final_cells: int = 128        # the final run's grid and solve interval
    sc_final_every: int = 1
    sc_h_m: float = 5e-4             # FFT cell size when the cells per axis are 0
    sc_max_cells: int = 8000000      # cap on the total FFT cells when the cell size sets the grid
    sc_window_q: float = 0.002       # particles outside the [q, 1 - q] quantiles per axis neither deposit nor feel SC
    sc_frame: str = 'bunch'          # 'bunch' (grid turned onto the mean horizontal velocity) | 'lab'
    sc_gpu: bool = True
    sc_current_ma: float = 0.0       # 0 = the hand-off file's current
    sc_gpu_pool_gb: float = 0.0      # > 0: limit of the GPU memory pool of the space-charge solver [GiB]; 0 = the
    #                                  solver's own (2 GiB). A train's mesh at 128 nodes needs more
    # bunch train (opt-in): the bunch of the run among the bunches of the turns next to it, released one every
    # machine.harmonic RF periods (one revolution), all on one space-charge mesh; 0 and 0 = the bunch alone
    bunches_ahead: int = 0           # bunches released before the run's bunch (older: they fly on the turns outside it)
    bunches_behind: int = 0          # bunches released after it (younger: inside it)
    train_coupled: bool = True       # false = the control of a train run: the run's bunch feels only its own charge,
    #                                  on the train's mesh; the other bunches fly as before (two solves per step)
    train_charge: float = 1.0        # charge of those bunches relative to the run's bunch; 0 = weightless: tracked on
    #                                  the same mesh, but nothing feels them and they feel no charge of their own
    sc_train_cells_radial: int = 512  # train: FFT grid nodes across the train (the radial axis of the bunch frame);
    #                                   along the bunch and vertically the nodes stay sc_cells / sc_final_cells
    n_tracks: int = 20000            # particles whose full tracks are stored (final run)
    track_every: int = 4             # steps between stored track points
    probes_deg: tuple = (25.0, 45.0)  # azimuths of the radial probes (final run)
    snapshot_ns: tuple = (10.0, 20.0, 30.0, 40.0)   # times [ns] of full-beam snapshots (final run)
    obstacles: str = 'columns'       # metal test: 'columns' (exact in z) | 'raster' (the deck's z slices, 1 mm apart)
    #                                  | 'mesh' (the model's triangles, GPU, no fast path)
    extra_posts_mm: tuple = ()       # full-height grounded posts [[x, y, r], ...] mm, obstacle test only
    extra_boxes_mm: tuple = ()       # full-height grounded boxes [[xmin, xmax, ymin, ymax], ...] mm, obstacle test only
    extra_jaws_mm: tuple = ()        # vertical jaw pairs [[x, y, r, z_half], ...] mm: metal at |z| >= z_half over the disc,
    #                                  obstacle test only


@dataclasses.dataclass
class SlitConfig:
    """A two-sided radial slit: two thin full-height jaws on the ray at ``azimuth_deg`` that pass the particles with
    ``|r - r_c| < half_width_mm`` there and stop those that hit a jaw. An obstacle only (the jaws add no field): exact
    for jaws mounted inside a dee, away from its gaps, where the field is zero. The centre ``r_c`` follows the design:
    the reference particle's radius at its crossing number ``pass_index`` of the ray (the probes' pass numbering, from
    the reference run at the bunch's phase), or ``r_center_mm`` with ``pass_index = -1``, plus ``offset_mm``. The jaws
    must stay clear of the neighbouring turns (``jaw_length_mm`` < the turn gap: the tracker warns when the reference
    crosses a jaw on another pass) and be thicker than a tracking step there (checked)."""
    half_width_mm: float = 0.0       # half the opening between the jaws; 0 = no slit
    azimuth_deg: float = 315.0       # the ray the jaws sit on
    pass_index: int = 3              # centred on the reference particle's crossing number pass_index; -1 = r_center_mm
    offset_mm: float = 0.0           # added to the centre
    r_center_mm: float = 0.0         # the slit centre when pass_index = -1
    jaw_length_mm: float = 6.0       # radial length of each jaw
    thickness_mm: float = 2.0        # azimuthal thickness of each jaw
