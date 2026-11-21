"""The campaign store: the source of truth of a MOBO campaign, resumable and reproducible.

    <campaign_dir>/evals.jsonl          append-only, one JSON line per state change of an evaluation
    <campaign_dir>/evals/<id>/          command.txt, stdout.log, stderr.log, result.json and the stage outputs
    <campaign_dir>/geometries/<key>/    the electrode model and BEM fields shared by the evaluations of one geometry
    <campaign_dir>/driver.lock          held by the driver that runs the campaign (one evaluation at a time: one GPU)

An evaluation moves through the states proposed -> running -> done | failed. Its record is the merge of its lines in
order (``evaluations``); a line carries only what is new (the proposal's unit and physical parameters, overrides,
fidelity and geometry key; the subprocess id; the scores, objectives and constraints; the error and wall time), plus
a timestamp. Nothing else holds state: the driver rebuilds its optuna study from these lines before every proposal,
so a campaign continues exactly where it stopped, and a line is never rewritten.

Search evaluations are numbered 0000, 0001, ...; the re-evaluation of evaluation 0007 at fidelity 'verify' is
0007.verify, stored next to it.
"""
import datetime
import json
import os
import socket
import warnings
from pathlib import Path

STATES = ('proposed', 'running', 'done', 'failed')


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


class Store:
    def __init__(self, campaign_dir):
        self.dir = Path(campaign_dir)
        self.log = self.dir / 'evals.jsonl'
        self.lock_path = self.dir / 'driver.lock'

    def eval_dir(self, eval_id):
        return self.dir / 'evals' / eval_id

    def geometry_dir(self, key):
        return self.dir / 'geometries' / key

    def append(self, eval_id, state, **fields):
        """Add one state line (flushed to disk before returning)."""
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        self.dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps({'id': eval_id, 'state': state, 't': now(), **fields}, sort_keys=True)
        with open(self.log, 'ab') as fh:
            if fh.tell() and not self._ends_with_newline():
                line = '\n' + line                    # after a line cut short by a killed writer
            fh.write((line + '\n').encode('utf-8'))
            fh.flush()
            os.fsync(fh.fileno())

    def _ends_with_newline(self):
        with open(self.log, 'rb') as fh:
            fh.seek(-1, os.SEEK_END)
            return fh.read(1) == b'\n'

    def records(self):
        """Every state line in order. A line cut short by a killed writer is skipped: its state change never
        completed."""
        if not self.log.exists():
            return []
        out, bad = [], []
        for i, line in enumerate(self.log.read_text(encoding='utf-8').splitlines()):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                bad.append(i + 1)
        if bad:
            warnings.warn(f"{self.log}: skipped incomplete line(s) {bad}", stacklevel=2)
        return out

    def evaluations(self):
        """{id: merged record} in the order the evaluations were proposed; t_<state> holds each state's time. A new
        'proposed' line for an id starts a new attempt (a repeated verify): the record restarts from it."""
        evals = {}
        for rec in self.records():
            if rec['state'] == 'proposed':
                attempt = evals.get(rec['id'], {}).get('attempt', 0) + 1
                evals[rec['id']] = {'attempt': attempt}
            ev = evals.setdefault(rec['id'], {'attempt': 1})
            ev.update(rec)
            ev[f"t_{rec['state']}"] = rec['t']
        return evals

    @staticmethod
    def next_id(evals):
        return f"{sum('.' not in i for i in evals):04d}"


# ---------------------------------------------------------------------------------------------------------------------
# one driver per campaign
class CampaignBusy(RuntimeError):
    pass


def pid_alive(pid):
    """Whether a process with this id runs (never signals it: os.kill(pid, 0) terminates a process on Windows)."""
    if not pid:
        return False
    if os.name == 'nt':
        import ctypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, int(pid))       # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5                         # access denied: it exists
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259                         # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


if os.name == 'nt':
    import msvcrt

    _LOCK_AT = 1 << 30                # the locked byte: far past the holder's note, which stays readable

    def _try_lock(fd):
        os.lseek(fd, _LOCK_AT, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock(fd):
        os.lseek(fd, _LOCK_AT, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(fd):
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)


class CampaignLock:
    """Exclusive ownership of a campaign: one driver, and so one evaluation, at a time.

    An operating-system lock on ``driver.lock`` (a byte-range lock on Windows, flock on POSIX), held from the start
    of a driver to its end. The system releases it when the driver ends, however it ends (a hard kill included), so
    the lock never goes stale and nothing has to judge whether a recorded process still runs. The file holds a note
    of its holder (pid, host, start), written once when the lock is taken, for the message a second driver prints;
    an empty file is a free campaign. The file is never deleted: on POSIX a driver that deleted it on its way out
    could leave the next driver holding a lock on a file that a third one no longer sees.

    The evaluation subprocess must not outlive the driver either (a second driver would start a second job beside it
    on the GPU). On Windows it cannot: it runs in a job object that dies with the driver (``evaluate.run``). On POSIX a
    hard-killed driver's process group survives it, so the evaluation inherits the locked descriptor (``inherit``):
    the campaign stays locked until that evaluation ends too."""

    def __init__(self, path):
        self.path = Path(path)
        self.fd = None
        self.info = {'pid': os.getpid(), 'host': socket.gethostname(), 'started': now()}

    @property
    def inherit(self):
        """Descriptors an evaluation subprocess keeps open (POSIX: the lock; Windows: none, the job object rules)."""
        return () if os.name == 'nt' or self.fd is None else (self.fd,)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_BINARY', 0))
        try:
            _try_lock(fd)
        except OSError:
            os.lseek(fd, 0, os.SEEK_SET)
            note = os.read(fd, 4096).decode('utf-8', 'replace')
            os.close(fd)
            raise CampaignBusy(self._busy(note)) from None
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps(self.info).encode('utf-8'))
        self.fd = fd
        return self

    def _busy(self, note):
        try:
            held = json.loads(note)
        except json.JSONDecodeError:
            held = {}
        who = f"driver pid {held.get('pid', '?')} on {held.get('host', '?')} (since {held.get('started', '?')})"
        if held.get('host') == socket.gethostname() and not pid_alive(held.get('pid')):
            return (f"{self.path} is locked by an evaluation that the stopped {who} left running: wait for it to end "
                    f"(or stop it), then start again")
        return f"the campaign is in use by {who}: stop it first"

    def __exit__(self, *exc):
        os.ftruncate(self.fd, 0)                     # an empty note while the lock is still held, then release
        _unlock(self.fd)
        os.close(self.fd)
        self.fd = None
        return False
