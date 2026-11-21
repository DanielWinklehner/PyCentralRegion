"""The process a computation runs in: the opt-out from Windows' power throttling (process.keep_full_speed), and the
two places that apply it - every `pcr` stage command by run.full_speed of its run file, the campaign driver by
mobo.full_speed of its campaign file. Needs neither the physics stack nor optuna."""
import sys
from pathlib import Path

import pytest

from PyCentralRegion import cli, process

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / 'configs' / 'examples' / 'toy' / 'campaign.toml'


def test_keep_full_speed_takes_the_execution_speed_control():
    """On Windows the process takes the execution-speed control and leaves its state off (never throttled); twice is
    once; elsewhere there is nothing to set. It tells what happened and never raises."""
    said = process.keep_full_speed()
    if sys.platform != 'win32':
        assert said == 'not needed' and process.power_throttling() is None
        return
    assert said == 'set', said
    control, state = process.power_throttling()
    assert control & process.EXECUTION_SPEED and not state & process.EXECUTION_SPEED
    assert process.keep_full_speed() == 'set' and process.power_throttling() == (control, state)


def test_a_failed_call_is_told_not_raised(monkeypatch):
    def broken():
        raise OSError('no such call')
    monkeypatch.setattr(process, '_kernel32', broken)
    monkeypatch.setattr(process.sys, 'platform', 'win32')
    assert process.keep_full_speed() == 'not set (OSError: no such call)' and process.power_throttling() is None
    monkeypatch.setattr(process.sys, 'platform', 'linux')
    assert process.keep_full_speed() == 'not needed'


def test_every_stage_command_applies_run_full_speed(tmp_path, monkeypatch, capsys):
    """`pcr <stage> run.toml` opts out once its run file is read; run.full_speed = false leaves the process alone."""
    calls = []
    monkeypatch.setattr(process, 'keep_full_speed', lambda: calls.append(1) or 'set')
    toml = tmp_path / 'run.toml'
    toml.write_text('[run]\nname = "a"\n')
    assert cli.main(['show', str(toml)]) == 0 and len(calls) == 1
    assert cli.main(['show', str(toml), '--set', 'run.full_speed=false']) == 0 and len(calls) == 1
    toml.write_text('[run]\nname = "a"\nfull_speed = false\n')
    assert cli.main(['show', str(toml)]) == 0 and len(calls) == 1
    assert cli.main(['show', str(toml), '--set', 'run.full_speed=true']) == 0 and len(calls) == 2
    capsys.readouterr()


def test_the_campaign_driver_applies_mobo_full_speed(tmp_path, monkeypatch, capsys):
    pytest.importorskip('optuna')
    from PyCentralRegion.mobo import driver
    calls = []
    monkeypatch.setattr(process, 'keep_full_speed', lambda: calls.append(1) or 'set')
    sets = ['--set', f'mobo.campaign_dir={(tmp_path / "camp").as_posix()}']
    assert driver.main(['optimize-status', str(TOY), *sets]) == 0 and len(calls) == 1
    assert driver.main(['optimize-status', str(TOY), *sets, '--set', 'mobo.full_speed=false']) == 0 and len(calls) == 1
    capsys.readouterr()
