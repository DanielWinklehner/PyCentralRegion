"""Config loader: includes, merge order, overrides, typed sections, path resolution, and the documented defaults."""
import dataclasses
import importlib.util
import tomllib
from pathlib import Path

import pytest

from PyCentralRegion import config as C

DEFAULTS = Path(__file__).resolve().parents[1] / 'configs' / 'defaults'


def _write(path, text):
    path.write_text(text)
    return path


def test_include_merge_and_overrides(tmp_path):
    _write(tmp_path / 'base.toml', '[track]\nturns = 8\nphases_deg = [-10.0, 0.0]\n[paths]\nseed = "seed.pkl"\n')
    (tmp_path / 'sub').mkdir()
    run = _write(tmp_path / 'sub' / 'run.toml', 'include = ["../base.toml"]\n[track]\nturns = 4\n')
    cfg = C.load(run, overrides=['track.n_scan=100', 'run.name=abc', 'track.pusher=rk4'])
    assert cfg['track']['turns'] == 4                         # the including file wins
    assert cfg['track']['phases_deg'] == [-10.0, 0.0]         # untouched keys survive the merge
    assert cfg['track']['n_scan'] == 100 and cfg['run']['name'] == 'abc' and cfg['track']['pusher'] == 'rk4'
    assert Path(cfg['paths']['seed']) == (tmp_path / 'seed.pkl').resolve()   # relative to the file that set it
    tr = C.section(cfg, 'track')
    assert tr.turns == 4 and tr.phases_deg == (-10.0, 0.0) and isinstance(tr.phases_deg, tuple)


def test_unknown_key_fails(tmp_path):
    cfg = C.load(_write(tmp_path / 'r.toml', '[track]\nturnz = 4\n'))
    with pytest.raises(C.ConfigError, match='turnz'):
        C.section(cfg, 'track')


def test_int_to_float_and_include_loop(tmp_path):
    cfg = C.load(_write(tmp_path / 'r.toml', '[machine]\ndee_opening_deg = 40\n'))
    assert isinstance(C.section(cfg, 'machine').dee_opening_deg, float)
    _write(tmp_path / 'a.toml', 'include = ["b.toml"]\n')
    _write(tmp_path / 'b.toml', 'include = ["a.toml"]\n')
    with pytest.raises(C.ConfigError, match='include loop'):
        C.load(tmp_path / 'a.toml')


def _importable(name):
    module = C.SECTIONS[name].split(':')[0]
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.parametrize('name', sorted(C.SECTIONS))
def test_documented_defaults_match_dataclasses(name):
    """Every section's dataclass defaults are written out in configs/defaults/*.toml, with the same values."""
    if not _importable(name):
        pytest.skip(f"[{name}] module not present")
    documented = {}
    for f in sorted(DEFAULTS.glob('*.toml')):
        with open(f, 'rb') as fh:
            documented.update(tomllib.load(fh))
    assert name in documented, f"[{name}] is not documented in {DEFAULTS}"
    cls = C.section_class(name)
    defaults = C.to_dict(cls())
    table = documented[name]
    assert set(table) == {f.name for f in dataclasses.fields(cls)}
    for key, value in table.items():
        if isinstance(value, float) or isinstance(defaults[key], float):
            assert value == pytest.approx(defaults[key]), key
        else:
            assert C.to_dict(C._coerce(None, defaults[key], value, key)) == defaults[key], key
