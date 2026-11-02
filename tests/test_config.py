"""Config loader: includes, merge order, overrides, typed sections, path resolution, and the documented defaults; the
shared writers: TOML literals, files written whole, JSON of numpy values."""
import dataclasses
import importlib.util
import json
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


def test_array_index_overrides_and_paths(tmp_path):
    """An integer part of a dotted key indexes an array of tables; *_file keys inside such arrays are resolved."""
    _write(tmp_path / 'r.toml', '[g]\nwindows = [{gap = 1, half_mm = 10.0, spec_file = "w1.json"}, {gap = 3, half_mm = 12.3}]\n')
    cfg = C.load(tmp_path / 'r.toml', overrides=['g.windows.1.half_mm=11.5'])
    assert cfg['g']['windows'][1]['half_mm'] == 11.5 and cfg['g']['windows'][0]['half_mm'] == 10.0
    assert C.get_dotted(cfg, 'g.windows.1.gap') == 3 and C.get_dotted(cfg, 'g.windows.2.gap', 'absent') == 'absent'
    assert Path(cfg['g']['windows'][0]['spec_file']) == (tmp_path / 'w1.json').resolve()
    with pytest.raises(C.ConfigError, match='not an index'):
        C.load(tmp_path / 'r.toml', overrides=['g.windows.2.half_mm=1.0'])
    with pytest.raises(C.ConfigError, match='no array'):
        C.load(tmp_path / 'r.toml', overrides=['g.posts.0.r_mm=1.0'])


def _lists(x):
    return [_lists(v) for v in x] if isinstance(x, (list, tuple)) else x


def test_toml_value_reads_back():
    """Every kind of value toml_value writes is a TOML literal that parse_override reads back as that value (a tuple
    as a list): the same type, the same float to the last bit, the same characters."""
    values = [True, False, 0, -7, 2 ** 62, 0.1, -0.0, 1e-05, 1e16, 5e-324, 1.0 / 3.0, 4.726034523988023, -15.0,
              float('inf'), float('-inf'), float('nan'), '', 'rk4', 'D:/a b (c)/f.pkl', 'C:\\Users\\a b\\f.h5',
              'say "hi"', 'tab\tnew\nline\x01\x7f', '\u00b5m, \u00c5, \u00b15\u00b0', [], [-15.0], (0.0, 2.0),
              [1, 2.5, 'x', True, [False, []]]]
    for x in values:
        text = C.toml_value(x)
        assert repr(tomllib.loads(f"v = {text}")['v']) == repr(_lists(x)), text      # a literal, not a fallback
        assert repr(C.parse_override(f"track.k={text}")[1]) == repr(_lists(x)), text
    assert C.toml_value([0.1, 'a']) == '[0.1, "a"]' and C.toml_value(-0.0) == '-0.0' and C.toml_value(1e16) == '1e+16'
    with pytest.raises(TypeError, match='no TOML literal'):
        C.toml_value(None)


def test_write_atomic(tmp_path):
    """An output is written under its temporary name <stem>.part<suffix> and renamed: a write that raises leaves no
    file behind, and with overwrite False an output that appeared meanwhile is kept (the new file stays temporary)."""
    out = tmp_path / 'score.json'
    assert C.part_path(out) == tmp_path / 'score.part.json'
    assert C.part_path(tmp_path / 'cr_model_a_model3d.pkl').name == 'cr_model_a_model3d.part.pkl'
    seen = []

    def write(p):
        seen.append(p)
        assert not out.exists()
        p.write_text('{}')

    assert C.write_atomic(out, write) == out
    assert seen == [tmp_path / 'score.part.json'] and out.read_text() == '{}'
    C.write_atomic(out, lambda p: p.write_text('[1]'))                         # overwrite True replaces it
    assert out.read_text() == '[1]' and sorted(p.name for p in tmp_path.iterdir()) == ['score.json']

    def killed(p):
        p.write_text('half a file')
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        C.write_atomic(tmp_path / 'track.json', killed)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['score.json']        # no temporary file, no output

    def raced(p):                                                              # another run writes the output too
        p.write_text('ours')
        (tmp_path / 'fields.pkl').write_text('theirs')

    with pytest.raises(C.ConfigError, match='appeared while it was written'):
        C.write_atomic(tmp_path / 'fields.pkl', raced, overwrite=False)
    assert (tmp_path / 'fields.pkl').read_text() == 'theirs' and (tmp_path / 'fields.part.pkl').read_text() == 'ours'


def test_json_of_numpy_values():
    """json_default (track.json: numpy as Python, NaN kept, anything else as its str) and json_clean (score.json and
    result.json: NaN and inf as null, keys as str)."""
    np = pytest.importorskip('numpy')
    data = {'n': np.int64(3), 'x': np.float32(0.5), 'v': np.array([1.0, np.nan]), 'p': Path('a'), 1: np.bool_(True)}
    assert json.dumps(data, default=C.json_default) == '{"n": 3, "x": 0.5, "v": [1.0, NaN], "p": "a", "1": true}'
    clean = C.json_clean({'v': np.array([1.0, np.nan]), 2: (np.float64(np.inf), np.int32(4)), 'b': np.bool_(False)})
    assert clean == {'v': [1.0, None], '2': [None, 4], 'b': False}
    assert [type(v) for v in clean['2'][1:] + [clean['b']]] == [int, bool]


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
