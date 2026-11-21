"""`pcr` command line: `show` prints the resolved configuration with the overrides applied, and every stage command
exists with the documented arguments; `pcr evaluate` with [post_rule] (its stages replaced by stand-ins)."""
import json
from pathlib import Path

import pytest

from PyCentralRegion import cli
from PyCentralRegion.config import ConfigError


def test_show_applies_overrides(tmp_path, capsys):
    run = tmp_path / 'run.toml'
    run.write_text('[track]\nturns = 8\n[paths]\nseed = "seed.pkl"\n')
    assert cli.main(['show', str(run), '--set', 'track.turns=4', '--set', 'track.phases_deg=[-15.0]']) == 0
    cfg = json.loads(capsys.readouterr().out)
    assert cfg['track']['turns'] == 4 and cfg['track']['phases_deg'] == [-15.0]
    assert cfg['paths']['seed'] == str((tmp_path / 'seed.pkl').resolve())


def test_bad_override_fails(tmp_path):
    run = tmp_path / 'run.toml'
    run.write_text('[geometry]\nwindows = [{gap = 1}]\n')
    with pytest.raises(ConfigError, match='not an index'):
        cli.main(['show', str(run), '--set', 'geometry.windows.3.gap=2'])


@pytest.mark.parametrize('command', ['show', 'build', 'fields', 'track', 'score'])
def test_stage_commands_take_a_run_file(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([command, '--help'])
    assert exc.value.code == 0 and '--set' in capsys.readouterr().out


def test_score_writes_plain_json_once(tmp_path, capsys):
    """`pcr score` on a run folder in the tracker's layout: score.json holds plain JSON (NaN as null), and a second
    call reuses it instead of scoring again."""
    pytest.importorskip('scipy')
    import pickle
    from test_analysis import _spiral_run                    # the synthetic accelerated bunch of the analysis tests
    from PyCentralRegion.analysis import runs as R
    run = _spiral_run(n=300, halo=0.1)
    keys = ('ids', 'steps', 'xyz', 'stop_step', 'birth_step', 'n_steps', 'steps_per_turn', 'E_mev')
    final = {'phase': 0.0, 'n': run.n_tracks, 'transmission': run.info['final']['transmission'],
             'tracks': {k: getattr(run, k) for k in keys}}
    out = tmp_path / 'design_a'
    out.mkdir()
    with open(out / R.TRACK_PKL, 'wb') as fh:
        pickle.dump({'final': final, 'ref': {0.0: {'traj': run.ref_xyz}}}, fh)
    (out / R.TRACK_JSON).write_text(json.dumps(run.info))
    toml = tmp_path / 'run.toml'
    toml.write_text(f'[run]\nout_dir = "{out.as_posix()}"\n[score]\nmin_tracks = 50\nmin_survivors = 20\n')
    assert cli.main(['score', str(toml)]) == 0
    scores = json.loads((out / 'score.json').read_text())
    assert scores['tag'] == 'design_a' and 0.0 < scores['valley_all'] < 0.1 and scores['I_final'] > 0
    assert cli.main(['score', str(toml)]) == 0 and 'exists: reused' in capsys.readouterr().out


@pytest.mark.parametrize('command, option', [('optimize', '--max-evals'), ('optimize-verify', '--pareto'),
                                             ('optimize-status', 'campaign TOML')])
def test_campaign_commands_reach_the_driver(command, option, capsys):
    """The optimize commands go to mobo.driver with their name, which parses the rest itself."""
    with pytest.raises(SystemExit) as exc:
        cli.main([command, '--help'])
    assert exc.value.code == 0 and option in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------------------------
# posts in the loop: `pcr evaluate` with [post_rule] enabled
POST = {'gap': 1, 'side': 'dee', 's_mm': 40.0, 'length_mm': 20.0}
SCORES = {'I_final_az': 5.0, 'I_2pct_best_az': 4.0, 'z_p95_az': 3.0, 'E_med_az': 3.1, 'T_az': 0.6}
SCAN = (-45.0, -30.0, -15.0, 0.0, 15.0)          # the rule's default phase scan


class _Stop(BaseException):
    """A kill: what no stage catches."""


class _Stages:
    """Stand-ins for the stages of `pcr evaluate` that write and reuse their outputs as the real ones do: the build
    writes a build's files (the model last), each holding the number of posts it was built with; the field map takes
    its model's, and a track reads its field map's; the tracker's metal test ('columns') reads a model. ``fail``
    {'build' | 'columns' | 'fields' | 'bare track': exception}: raised by the build, the metal test or the field solve
    of a model with posts, or by the bare pass's tracking; ``fail_tol``: the geometry.occ_fuzzy_mm of the model at
    which they fail (None: at every one). ``log``: (stage, its output, what it did)."""

    def __init__(self, monkeypatch, posts=(POST,), fail=None, fail_tol=None):
        simulate3d = pytest.importorskip('PyCentralRegion.simulate3d')
        post_rule = pytest.importorskip('PyCentralRegion.geometry.post_rule')
        from PyCentralRegion.geometry.build import output_files
        self.output_files, self.posts, self.fail, self.log = output_files, list(posts), dict(fail or {}), []
        self.fail_tol, self.tol = fail_tol, None              # the tolerance of the last model built
        monkeypatch.setattr(cli, 'build_stage', self.build)
        monkeypatch.setattr(cli, 'score_stage', self.score)
        monkeypatch.setattr(cli, '_free_gpu', lambda: None)
        monkeypatch.setattr(simulate3d, 'fields_stage', self.fields)
        monkeypatch.setattr(simulate3d, 'track_stage', self.track)
        monkeypatch.setattr(simulate3d, 'load_model3d', lambda path: dict(json.loads(Path(path).read_text()),
                                                                          path=Path(path)))
        monkeypatch.setattr(simulate3d, 'column_obstacles', self.columns)
        monkeypatch.setattr(post_rule, 'posts_from_run', self.place)

    def stages(self):
        return [entry[0] for entry in self.log]

    def _fails(self, stage):
        return stage in self.fail and (self.fail_tol is None or self.tol in self.fail_tol)

    def build(self, cfg):
        out = self.output_files(cfg['paths']['model3d'])
        if out['model3d'].exists():
            self.log.append(('build', out['model3d'], 'reused'))
            return
        if any(f.exists() for f in out.values()):
            raise ConfigError("other outputs of a build exist: never overwritten")
        n = len(cfg.get('geometry', {}).get('posts', []))
        self.tol = cfg.get('geometry', {}).get('occ_fuzzy_mm', 0.05)
        self.log.append(('build', out['model3d'], n))
        for name, f in out.items():
            if name == 'model3d' and n and self._fails('build'):
                raise self.fail['build']
            f.write_text(json.dumps({'posts': n}))

    def columns(self, model3d, fc, verbose=True, odd='raise'):
        assert odd == 'raise'
        self.log.append(('columns', model3d['path'], model3d['posts']))
        if model3d['posts'] and self._fails('columns'):
            raise self.fail['columns']

    def fields(self, cfg, verbose=True):
        out = Path(cfg['paths']['fields'])
        if out.exists():
            self.log.append(('fields', out, 'reused'))
            return out
        model = json.loads(Path(cfg['paths']['model3d']).read_text())
        self.log.append(('fields', out, model['posts']))
        if model['posts'] and self._fails('fields'):
            raise self.fail['fields']
        out.write_text(json.dumps(model))
        return out

    def track(self, cfg, verbose=True):
        out = Path(cfg['run']['out_dir'])
        if out.name == 'track' and 'bare track' in self.fail:
            raise self.fail['bare track']
        self.log.append(('track', out, (tuple(cfg['track']['phases_deg']), cfg['magnet']['scale'],
                                        json.loads(Path(cfg['paths']['fields']).read_text())['posts'])))
        out.mkdir(parents=True, exist_ok=True)
        (out / 'track_data.pkl').write_text('tracks')
        summary = {'best_phase': -30.0}
        (out / 'track.json').write_text(json.dumps(summary))
        return summary

    def score(self, cfg):
        self.log.append(('score', Path(cfg['run']['out_dir']), None))
        return dict(SCORES)

    def place(self, cfg, rule=None, run_dir=None):
        out = Path(cfg['run']['out_dir'])
        assert (out / 'track_data.pkl').exists() and rule.enabled
        self.log.append(('place', out, len(self.posts)))
        return {'posts': [dict(p, between_bands=[1, 2], r_mm=100.0) for p in self.posts], 'edges': [], 'n_tracks': 10,
                'n_survivors': 6, 'rule': {}}


def _evaluation(geo, out):
    """The config `pcr evaluate` gets from the campaign driver: the geometry's folder ``geo``, the evaluation's ``out``,
    the evaluation's phase and level."""
    return {'paths': {'model3d': str(geo / 'model3d.pkl'), 'fields': str(geo / 'fields.pkl')},
            'run': {'out_dir': str(out)}, 'track': {'phases_deg': [-15.0]}, 'magnet': {'scale': 1.002},
            'post_rule': {'enabled': True}}


def test_evaluate_places_posts_from_a_bare_run(tmp_path, monkeypatch):
    """The first evaluation of a geometry tracks its bare electrodes in <geometry>/bare at the rule's phase scan and
    level, places the posts from those tracks, and builds, solves and tracks the geometry with them at its own phase
    and level; a later evaluation of the geometry only tracks, with the same posts."""
    st = _Stages(monkeypatch)
    geo = tmp_path / 'geometries' / 'g'
    bare = geo / 'bare'
    geo.mkdir(parents=True)
    cfg = _evaluation(geo, tmp_path / 'e0')
    scores = cli.evaluate(cfg)
    assert st.log == [('build', bare / 'model3d.pkl', 0), ('fields', bare / 'fields.pkl', 0),
                      ('track', bare / 'track', (SCAN, 1.0, 0)), ('score', bare / 'track', None),
                      ('place', bare / 'track', 1), ('build', geo / 'model3d.pkl', 1),
                      ('columns', geo / 'model3d.pkl', 1), ('fields', geo / 'fields.pkl', 1),
                      ('track', tmp_path / 'e0', ((-15.0,), 1.002, 1)), ('score', tmp_path / 'e0', None)]
    assert cfg['geometry']['posts'] == [POST]
    record = json.loads((bare / cli.POSTS_RECORD).read_text())
    assert record['posts'] == [dict(POST, between_bands=[1, 2], r_mm=100.0)] and record['bare_phase_deg'] == -30.0
    assert record['model'] == 'model3d.pkl'
    assert record['bare_scores'] == SCORES and 'fallback' not in record
    assert scores == {**SCORES, 'post_rule': {'n_posts': 1, 'fallback': '', 'bare_phase_deg': -30.0,
                                              'n_survivors_bare': 6, 'bare': SCORES}}
    # the bare pass keeps its model and scores, not its field map or tracks
    assert (bare / 'model3d.pkl').exists() and (bare / 'track' / 'track.json').exists()
    assert not (bare / 'fields.pkl').exists() and not (bare / 'track' / 'track_data.pkl').exists()
    st.log.clear()
    cfg = _evaluation(geo, tmp_path / 'e1')
    cli.evaluate(cfg)
    assert st.log == [('track', tmp_path / 'e1', ((-15.0,), 1.002, 1)), ('score', tmp_path / 'e1', None)]
    assert cfg['geometry']['posts'] == [POST]


def test_a_geometry_without_posts_keeps_its_bare_model(tmp_path, monkeypatch):
    st = _Stages(monkeypatch, posts=())
    geo = tmp_path / 'g'
    geo.mkdir()
    cfg = _evaluation(geo, tmp_path / 'e0')
    scores = cli.evaluate(cfg)
    assert st.stages() == ['build', 'fields', 'track', 'score', 'place', 'track', 'score']
    assert st.log[-2][2] == ((-15.0,), 1.002, 0) and cfg['geometry']['posts'] == []
    assert scores['post_rule']['n_posts'] == 0 and scores['post_rule']['fallback'] == ''
    for f in [*st.output_files(geo / 'model3d.pkl').values(), geo / 'fields.pkl']:
        assert json.loads(f.read_text()) == {'posts': 0}                # the bare outputs, moved up
    assert not any((geo / 'bare').glob('model3d*')) and not (geo / 'bare' / 'fields.pkl').exists()


@pytest.mark.parametrize('stage, error', [('build', RuntimeError('the fuse dropped a post')),
                                          ('columns', ValueError('column test: the solid is not closed')),
                                          ('fields', ConfigError('3D model too large'))])
def test_a_failed_model_with_posts_falls_back_to_the_bare_one(tmp_path, monkeypatch, capsys, stage, error):
    st = _Stages(monkeypatch, fail={stage: error})
    geo = tmp_path / 'g'
    geo.mkdir()
    cfg = _evaluation(geo, tmp_path / 'e0')
    scores = cli.evaluate(cfg)
    reached = {'build': [], 'columns': ['columns'], 'fields': ['columns', 'fields']}[stage]
    assert st.stages() == (['build', 'fields', 'track', 'score', 'place'] + (['build'] + reached) * 3     # 3 tolerances
                           + ['track', 'score'])
    assert st.log[-2][2] == ((-15.0,), 1.002, 0) and cfg['geometry']['posts'] == []
    assert cfg['geometry']['occ_fuzzy_mm'] == 0.05                        # the bare model's, after the retries
    why = f"{type(error).__name__}: {error}"
    record = json.loads((geo / 'bare' / cli.POSTS_RECORD).read_text())
    assert record['fallback'] == why and 'occ_fuzzy_mm' not in record
    assert record['failed_at'] == [{'occ_fuzzy_mm': mm, 'error': why} for mm in (0.05, 0.02, 0.1)]
    assert scores['post_rule']['fallback'] == why and scores['post_rule']['n_posts'] == 0
    assert 'this geometry stays without posts' in capsys.readouterr().out
    for f in [*st.output_files(geo / 'model3d.pkl').values(), geo / 'fields.pkl']:
        assert json.loads(f.read_text()) == {'posts': 0}                # the bare outputs, over what the failure left
    st.log.clear()
    cfg = _evaluation(geo, tmp_path / 'e1')
    cli.evaluate(cfg)
    assert st.stages() == ['track', 'score'] and cfg['geometry']['posts'] == []


@pytest.mark.parametrize('stage, error', [('build', RuntimeError('The 1D mesh seems not to be forming a closed loop')),
                                          ('columns', ValueError('column test: the solid is not closed')),
                                          ('fields', ConfigError('3D model too large'))])
def test_a_model_with_posts_that_fails_is_built_again_at_another_tolerance(tmp_path, monkeypatch, stage, error):
    """A model with posts that fails at the configured geometry.occ_fuzzy_mm is built again at [post_rule]
    fuzzy_retry_mm's tolerances in turn; the record keeps the one that worked, and the geometry's next runs set it."""
    st = _Stages(monkeypatch, fail={stage: error}, fail_tol=(0.05,))
    geo = tmp_path / 'g'
    geo.mkdir()
    cfg = _evaluation(geo, tmp_path / 'e0')
    scores = cli.evaluate(cfg)
    reached = {'build': [], 'columns': ['columns'], 'fields': ['columns', 'fields']}[stage]
    assert st.stages() == (['build', 'fields', 'track', 'score', 'place', 'build'] + reached
                           + ['build', 'columns', 'fields', 'track', 'score'])
    assert cfg['geometry']['posts'] == [POST] and cfg['geometry']['occ_fuzzy_mm'] == 0.02
    record = json.loads((geo / 'bare' / cli.POSTS_RECORD).read_text())
    assert 'fallback' not in record and record['occ_fuzzy_mm'] == 0.02
    assert record['failed_at'] == [{'occ_fuzzy_mm': 0.05, 'error': f"{type(error).__name__}: {error}"}]
    assert scores['post_rule']['n_posts'] == 1 and scores['post_rule']['fallback'] == ''
    assert json.loads((geo / 'fields.pkl').read_text()) == {'posts': 1}
    st.log.clear()
    cfg = _evaluation(geo, tmp_path / 'e1')
    cli.evaluate(cfg)
    assert st.stages() == ['track', 'score'] and cfg['geometry']['occ_fuzzy_mm'] == 0.02


def test_without_retry_tolerances_a_failed_model_falls_back_at_once(tmp_path, monkeypatch):
    st = _Stages(monkeypatch, fail={'build': RuntimeError('the fuse dropped a post')})
    geo = tmp_path / 'g'
    geo.mkdir()
    cfg = _evaluation(geo, tmp_path / 'e0')
    cfg['post_rule']['fuzzy_retry_mm'] = []
    cli.evaluate(cfg)
    assert st.stages().count('build') == 2                     # the bare pass's, and one with posts
    record = json.loads((geo / 'bare' / cli.POSTS_RECORD).read_text())
    assert record['fallback'] == 'RuntimeError: the fuse dropped a post' and len(record['failed_at']) == 1


@pytest.mark.parametrize('stage, error', [('fields', MemoryError()), ('fields', PermissionError('locked')),
                                          ('columns', type('OutOfMemoryError', (Exception,), {})('cupy'))])
def test_an_error_of_the_machine_fails_the_evaluation_without_a_fallback(tmp_path, monkeypatch, stage, error):
    """Memory, the GPU or the file system: nothing about the model with posts, so no fallback for good; the
    geometry's next evaluation builds on what is there."""
    st = _Stages(monkeypatch, fail={stage: error})
    geo = tmp_path / 'g'
    geo.mkdir()
    with pytest.raises(type(error)):
        cli.evaluate(_evaluation(geo, tmp_path / 'e0'))
    assert 'fallback' not in json.loads((geo / 'bare' / cli.POSTS_RECORD).read_text())
    assert (geo / 'model3d.pkl').exists() and not (geo / 'fields.pkl').exists()
    st.fail.clear()
    st.log.clear()
    cfg = _evaluation(geo, tmp_path / 'e1')
    cli.evaluate(cfg)
    assert st.log[:2] == [('columns', geo / 'model3d.pkl', 1), ('fields', geo / 'fields.pkl', 1)]
    assert cfg['geometry']['posts'] == [POST]


def test_a_stopped_geometry_continues_where_it_was(tmp_path, monkeypatch):
    geo, bare = tmp_path / 'g', tmp_path / 'g' / 'bare'
    geo.mkdir()
    # stopped in the bare pass's tracking: its model and field map are reused
    st = _Stages(monkeypatch, fail={'bare track': _Stop()})
    with pytest.raises(_Stop):
        cli.evaluate(_evaluation(geo, tmp_path / 'e0'))
    st.fail.clear()
    st.log.clear()
    cli.evaluate(_evaluation(geo, tmp_path / 'e1'))
    assert st.log[:3] == [('build', bare / 'model3d.pkl', 'reused'), ('fields', bare / 'fields.pkl', 'reused'),
                          ('track', bare / 'track', (SCAN, 1.0, 0))]
    assert st.stages()[3:] == ['score', 'place', 'build', 'columns', 'fields', 'track', 'score']
    # stopped in the build with posts: what it left goes, the bare pass is not repeated
    geo, bare = tmp_path / 'h', tmp_path / 'h' / 'bare'
    geo.mkdir()
    st = _Stages(monkeypatch, fail={'build': _Stop()})
    with pytest.raises(_Stop):
        cli.evaluate(_evaluation(geo, tmp_path / 'e2'))
    assert (geo / 'model3d.msh').exists() and not (geo / 'model3d.pkl').exists()
    st.fail.clear()
    st.log.clear()
    cfg = _evaluation(geo, tmp_path / 'e3')
    cli.evaluate(cfg)
    assert st.stages() == ['build', 'columns', 'fields', 'track', 'score']
    assert st.log[0] == ('build', geo / 'model3d.pkl', 1) and cfg['geometry']['posts'] == [POST]
    # stopped while the bare outputs move up (no posts): the model went first, the rest follows
    geo = tmp_path / 'k'
    geo.mkdir()
    st = _Stages(monkeypatch, posts=())
    moved, real_move = [], cli._move

    def move(source, target):
        if len(moved) == 2:
            raise _Stop()
        moved.append(Path(target).name)
        real_move(source, target)

    monkeypatch.setattr(cli, '_move', move)
    with pytest.raises(_Stop):
        cli.evaluate(_evaluation(geo, tmp_path / 'e4'))
    assert moved == ['model3d.pkl', 'model3d_dees.png'] and (geo / 'model3d.pkl').exists()
    monkeypatch.setattr(cli, '_move', real_move)
    st.log.clear()
    cli.evaluate(_evaluation(geo, tmp_path / 'e5'))
    assert st.stages() == ['track', 'score']
    for f in [*st.output_files(geo / 'model3d.pkl').values(), geo / 'fields.pkl']:
        assert json.loads(f.read_text()) == {'posts': 0}


def test_post_rule_guards(tmp_path, monkeypatch):
    _Stages(monkeypatch)
    geo = tmp_path / 'g'
    geo.mkdir()
    (geo / 'model3d.pkl').write_text('{}')
    with pytest.raises(ConfigError, match='without the post record'):
        cli.evaluate(_evaluation(geo, tmp_path / 'e0'))
    cfg = _evaluation(tmp_path / 'h', tmp_path / 'e1')
    cfg['geometry'] = {'posts': [POST]}
    with pytest.raises(ConfigError, match='geometry.posts must be empty'):
        cli.evaluate(cfg)
    # a second model in a folder with the first one's post record
    geo = tmp_path / 'k'
    geo.mkdir()
    cli.evaluate(_evaluation(geo, tmp_path / 'e2'))
    cfg = _evaluation(geo, tmp_path / 'e3')
    cfg['paths'] = {'model3d': str(geo / 'other_model3d.pkl'), 'fields': str(geo / 'other_fields.pkl')}
    with pytest.raises(ConfigError, match='post record of the model model3d.pkl, not of other_model3d.pkl'):
        cli.evaluate(cfg)
