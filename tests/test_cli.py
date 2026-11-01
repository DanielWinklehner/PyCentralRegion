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
