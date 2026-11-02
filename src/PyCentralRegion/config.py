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
import sys
import tomllib
import typing
from pathlib import Path

PATH_SUFFIXES = ('_path', '_dir', '_file')

# section name -> 'module:Class' of its dataclass
SECTIONS = {
    'run': 'PyCentralRegion.config:RunConfig',
    'paths': 'PyCentralRegion.config:PathsConfig',
    'machine': 'PyCentralRegion.config:MachineConfig',
    'magnet': 'PyCentralRegion.config:MagnetConfig',
    'inflector': 'PyCentralRegion.config:InflectorConfig',
    'geometry': 'PyCentralRegion.geometry.options:GeometryConfig',
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
def load(path, overrides=()):
    """Read a TOML run file (with its includes) and apply ``key=value`` overrides. Returns a plain nested dict."""
    path = Path(path).resolve()
    cfg = _load_file(path, ())
    for item in overrides:
        keys, value = parse_override(item)
        set_dotted(cfg, keys, value)
    cfg['_source'] = str(path)
    return cfg


def _load_file(path, seen):
    if path in seen:
        raise ConfigError(f"include loop: {' -> '.join(map(str, seen + (path,)))}")
    with open(path, 'rb') as fh:
        try:
            data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as ex:
            raise ConfigError(f"{path}: {ex}") from None
    _resolve_paths(data, path.parent, in_paths=False)
    includes = data.pop('include', [])
    if isinstance(includes, str):
        includes = [includes]
    base = {}
    for inc in includes:
        base = merge(base, _load_file((path.parent / inc).resolve(), seen + (path,)))
    return merge(base, data)


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
    species: str = 'H2_1+'           # PyPATools species name
    harmonic: int = 4                # RF harmonic h
    n_dees: int = 4                  # dees (4-fold symmetric machines are supported)
    dee_shape: str = 'radial'        # 'radial' (gap lines from straight segments); 'spiral' is not supported yet
    dee_center_angles_deg: tuple = (45.0, 135.0, 225.0, 315.0)
    dee_opening_deg: float = 42.0    # angular width of every dee
    dee_voltage_kv: float = 65.0     # dee voltage the electrode model is built (and the BEM normalised) at
    design_voltage_kv: float = 60.0  # dee voltage of the 1D gap model the seed geometry was optimized with
    gap_width_m: float = 0.01        # nominal gap width of the dee-system template
    gap_angle_deg: float = 2.5       # gap width law: max(gap_width_min, r * gap_angle)
    gap_width_min_m: float = 6.5e-3
    r_min_m: float = 0.005           # inner end of the gap lines
    r_max_cavity_m: float = 0.37     # outer end of the gap lines
    r_max_m: float = 0.38            # tracking: particles beyond this radius are lost
    z_max_m: float = 0.030           # tracking: particles beyond |z| are lost
    target_energy_mev: float = 3.0
    n_turns_design: int = 6          # turns of the seed optimization (rebuilds its dee system)
    skip_turns_design: int = 1


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
    stack_z_half_m: float = 0.030    # metal raster (obstacle test): |z| range, spacing
    stack_dz_m: float = 0.001
    stack_dxy_m: float = 5e-4
    max_elements: int = 80000        # refuse larger meshes
    tol: float = 1e-5                # GMRES tolerance
    solver: str = 'auto'             # BEM solver backend: auto | cupy | scipy | strong | dense
    pot_device: str = 'gpu'          # potential evaluation: cpu | gpu
    pot_precision: str = 'single'    # single | double
    pot_chunk: int = 250000          # points per potential batch
    pot_check: int = 3000            # points re-evaluated in double precision as a check
    gpu_fit_safety: float = 1.05     # GPU memory head-room factor of the potential batches


@dataclasses.dataclass
class TrackConfig:
    pusher: str = 'boris_rel'
    steps_per_turn: int = 1000
    turns: int = 8
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
    n_tracks: int = 20000            # particles whose full tracks are stored (final run)
    track_every: int = 4             # steps between stored track points
    probes_deg: tuple = (25.0, 45.0)  # azimuths of the radial probes (final run)
    snapshot_ns: tuple = (10.0, 20.0, 30.0, 40.0)   # times [ns] of full-beam snapshots (final run)
    obstacles: str = 'raster'        # metal test: 'raster' (the cached slices) | 'mesh' (the model's triangles)
    extra_posts_mm: tuple = ()       # full-height grounded posts [[x, y, r], ...] mm, obstacle test only
    extra_boxes_mm: tuple = ()       # full-height grounded boxes [[xmin, xmax, ymin, ymax], ...] mm, obstacle test only


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
