"""`pcr` command line: `show` prints the resolved configuration with the overrides applied, and every stage command
exists with the documented arguments."""
import json

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
