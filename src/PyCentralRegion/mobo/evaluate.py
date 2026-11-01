"""One evaluation = one subprocess: the evaluator contract, its command line and environment, logs, time limit.

The contract, followed by ``pcr evaluate`` and by the toy evaluator (``python -m PyCentralRegion.mobo.toy``)::

    <python> -m <evaluator module> --config <base run toml> --set key=value ... --out <eval_dir>

runs build -> fields -> track -> score with the overrides applied (later ones win) and writes
``<eval_dir>/result.json``::

    {"status": "ok" | "failed", "scores": {...}, "error": "...", "wall_s": ..., "config": {...resolved...}}

with exit code 0 when ok. The driver runs evaluations in a separate interpreter because they need the physics
environment (numba, CUDA, bempp, gmsh) while the driver needs only optuna, and because a crash, a hang or a GPU
out-of-memory in one evaluation then costs that evaluation, never the campaign.

An evaluation is stopped as one process tree - the evaluator and everything it starts - when it runs out of time,
when the user stops the driver, and when the driver ends in any other way: on Windows it runs in a job object that
the system empties when the driver's last handle to it closes (a hard kill of the driver included); on POSIX in a
session of its own, signalled as a group (a hard-killed driver cannot signal it: the campaign lock, which the
evaluation inherits, then keeps a new driver from starting beside it). Nothing an evaluation starts outlives it:
stragglers are stopped when it ends.

The subprocess inherits the driver's environment, with the source folder of the driver's own PyCentralRegion in
front of PYTHONPATH (the evaluations run the code the driver runs, whether or not the physics environment has the
package installed; an installed copy in site-packages is never put on another interpreter's path), MPLBACKEND=Agg
and, when the interpreter belongs to a conda environment, CONDA_PREFIX and the DLL folders of that environment in
front of PATH (what activation would do: CUDA libraries fail to load without them on Windows).
"""
import argparse
import dataclasses
import functools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

RESULT = 'result.json'
POLL_S = 0.25                    # the driver checks for Ctrl+C and the time limit this often while it waits
GRACE_S = 5.0                    # POSIX: SIGTERM to SIGKILL


# ---------------------------------------------------------------------------------------------------------------------
# the evaluator side
def add_arguments(parser):
    """The evaluator's command-line arguments (shared by `pcr evaluate` and the toy evaluator)."""
    parser.add_argument('--config', required=True, help='base run TOML')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='config override (repeatable; later ones win)')
    parser.add_argument('--out', required=True, help='evaluation folder: result.json and the stage outputs')
    return parser


def evaluator_parser(prog, description=''):
    return add_arguments(argparse.ArgumentParser(prog=prog, description=description))


def write_result(out_dir, status, scores=None, error='', wall_s=0.0, config=None):
    """Write result.json atomically (a reader never sees half a file)."""
    if status not in ('ok', 'failed'):
        raise ValueError(f"status must be 'ok' or 'failed', not {status!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (RESULT + '.tmp')
    tmp.write_text(json.dumps({'status': status, 'scores': scores or {}, 'error': str(error),
                               'wall_s': float(wall_s), 'config': config or {}}, indent=1, default=str))
    os.replace(tmp, out_dir / RESULT)


def read_result(out_dir):
    """result.json of an evaluation folder as a dict; None when there is none (or it is unreadable)."""
    path = Path(out_dir) / RESULT
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return result if isinstance(result, dict) and result.get('status') in ('ok', 'failed') else None


# ---------------------------------------------------------------------------------------------------------------------
# the driver side
def command(python, evaluator, base_config, overrides, out_dir):
    cmd = [python or sys.executable, '-m', *evaluator.split(), '--config', str(base_config)]
    for item in overrides:
        cmd += ['--set', item]
    return cmd + ['--out', str(out_dir)]


def conda_prefix(python):
    """The conda environment an interpreter belongs to (<prefix>/python.exe or <prefix>/bin/python), or None."""
    exe = Path(python or sys.executable)
    for folder in (exe.parent, exe.parent.parent):
        if (folder / 'conda-meta').is_dir():
            return folder
    return None


def package_root():
    """The folder the driver's PyCentralRegion is imported from when that is a source tree (a checkout, an editable
    install); None for an installed copy (a site-packages folder holds another interpreter's packages)."""
    root = Path(__file__).resolve().parents[2]
    return None if {'site-packages', 'dist-packages'} & set(root.parts) else root


def child_env(python):
    env = dict(os.environ, MPLBACKEND='Agg')
    root = package_root()
    if root is not None:
        rest = [p for p in env.get('PYTHONPATH', '').split(os.pathsep)
                if p and os.path.normcase(os.path.abspath(p)) != os.path.normcase(str(root))]
        env['PYTHONPATH'] = os.pathsep.join([str(root)] + rest)
    prefix = conda_prefix(python)
    if prefix is not None:
        env['CONDA_PREFIX'] = str(prefix)
        dirs = ([prefix, prefix / 'Library' / 'mingw-w64' / 'bin', prefix / 'Library' / 'usr' / 'bin',
                 prefix / 'Library' / 'bin', prefix / 'Scripts'] if os.name == 'nt' else [prefix / 'bin'])
        env['PATH'] = os.pathsep.join([str(d) for d in dirs] + [env.get('PATH', '')])
    return env


@functools.lru_cache(maxsize=None)
def _win32():
    """kernel32 / ntdll functions of the job object, with their signatures (64-bit handles must not be truncated)."""
    import ctypes
    from ctypes import wintypes
    k32, ntdll = ctypes.WinDLL('kernel32', use_last_error=True), ctypes.WinDLL('ntdll')
    H, B, D = wintypes.HANDLE, wintypes.BOOL, wintypes.DWORD
    for fn, res, args in ((k32.CreateJobObjectW, H, (wintypes.LPVOID, wintypes.LPCWSTR)),
                          (k32.SetInformationJobObject, B, (H, ctypes.c_int, wintypes.LPVOID, D)),
                          (k32.AssignProcessToJobObject, B, (H, H)),
                          (k32.TerminateJobObject, B, (H, wintypes.UINT)),
                          (k32.OpenProcess, H, (D, B, D)),
                          (k32.CloseHandle, B, (H,)),
                          (ntdll.NtResumeProcess, ctypes.c_long, (H,))):
        fn.restype, fn.argtypes = res, args

    class BasicLimits(ctypes.Structure):          # JOBOBJECT_BASIC_LIMIT_INFORMATION
        _fields_ = [('user_time_process', ctypes.c_int64), ('user_time_job', ctypes.c_int64), ('flags', D),
                    ('min_working_set', ctypes.c_size_t), ('max_working_set', ctypes.c_size_t),
                    ('active_processes', D), ('affinity', ctypes.c_size_t), ('priority_class', D),
                    ('scheduling_class', D)]

    class ExtendedLimits(ctypes.Structure):       # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        _fields_ = [('basic', BasicLimits), ('io_counters', ctypes.c_uint64 * 6), ('process_memory', ctypes.c_size_t),
                    ('job_memory', ctypes.c_size_t), ('peak_process_memory', ctypes.c_size_t),
                    ('peak_job_memory', ctypes.c_size_t)]
    return ctypes, k32, ntdll, ExtendedLimits


class _JobTree:
    """Windows: the evaluation in a job object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. Every process it starts joins
    the job, and the system ends them all when the job's last handle closes - ours, which closes when we stop the
    tree or when the driver itself ends, however it ends. The evaluation is created suspended and resumed only once
    it is in the job, so not even its first child can escape."""
    CREATE_SUSPENDED = 0x00000004

    def __init__(self):
        ctypes, k32, _, ExtendedLimits = _win32()
        self.job = k32.CreateJobObjectW(None, None)
        if not self.job:
            raise ctypes.WinError(ctypes.get_last_error())
        info = ExtendedLimits()
        info.basic.flags = 0x2000                                        # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(self.job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.WinError(ctypes.get_last_error())
            k32.CloseHandle(self.job)
            raise error

    def popen_kwargs(self, keep_fds):
        # a process group of its own: Ctrl+C in the console reaches the driver only, which then stops the tree
        return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | self.CREATE_SUSPENDED}

    def started(self, proc):
        """Put the suspended process into the job, then let it run (ntdll's NtResumeProcess: subprocess keeps no
        handle of the main thread to resume)."""
        ctypes, k32, ntdll, _ = _win32()
        handle = k32.OpenProcess(0x0001 | 0x0100 | 0x0800, False, proc.pid)   # TERMINATE | SET_QUOTA | SUSPEND_RESUME
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not k32.AssignProcessToJobObject(self.job, handle):
                raise OSError(f"could not put the evaluation into a job object ({ctypes.WinError(ctypes.get_last_error())})"
                              f": the driver runs no evaluation it could not stop with everything it starts")
            status = ntdll.NtResumeProcess(handle)
            if status:
                raise OSError(f"could not resume the evaluation process (NTSTATUS {status & 0xffffffff:#010x})")
        finally:
            k32.CloseHandle(handle)

    def stop(self, proc):
        """End every process of the tree (a finished evaluation's stragglers too) and release the job."""
        if self.job:
            _, k32, _, _ = _win32()
            k32.TerminateJobObject(self.job, 1)
            k32.CloseHandle(self.job)
            self.job = None
        if proc is not None:
            if proc.poll() is None:               # not in the job (started() failed): still suspended
                proc.kill()
            proc.wait()


class _GroupTree:
    """POSIX: the evaluation leads a session of its own; the tree is its process group (SIGTERM, then SIGKILL)."""

    def popen_kwargs(self, keep_fds):
        return {'start_new_session': True, 'pass_fds': tuple(keep_fds)}

    def started(self, proc):
        pass

    def stop(self, proc):
        if proc is None:
            return
        if proc.poll() is None:                   # still running: ask the group to end, then insist
            self._signal(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=GRACE_S)
            except subprocess.TimeoutExpired:
                pass
        self._signal(proc, signal.SIGKILL)        # what is left of the group (a finished evaluation's stragglers)
        proc.wait()

    @staticmethod
    def _signal(proc, sig):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass


@dataclasses.dataclass
class Outcome:
    returncode: int = None
    wall_s: float = 0.0
    timed_out: bool = False          # stopped at the time limit
    interrupted: bool = False        # stopped by Ctrl+C (or SIGTERM / SIGBREAK) in the driver
    result: dict = None              # result.json, None when the evaluator wrote none


def run(cmd, eval_dir, timeout_s=0.0, env=None, on_start=None, keep_fds=()):
    """Run one evaluation command with its output in eval_dir/stdout.log, stderr.log, and stop its whole process
    tree at the time limit (timeout_s > 0), on Ctrl+C, or when the evaluation ends (stragglers). on_start(pid) is
    called once the process runs; keep_fds are descriptors it inherits (POSIX: the campaign lock)."""
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    tree = _JobTree() if os.name == 'nt' else _GroupTree()
    out, proc = Outcome(), None
    t0 = time.monotonic()
    with open(eval_dir / 'stdout.log', 'w') as fo, open(eval_dir / 'stderr.log', 'w') as fe:
        try:
            proc = subprocess.Popen(cmd, stdout=fo, stderr=fe, stdin=subprocess.DEVNULL, env=env,
                                    **tree.popen_kwargs(keep_fds))
            tree.started(proc)
            if on_start is not None:
                on_start(proc.pid)
            while True:                           # short waits: a blocking one would not see Ctrl+C on Windows
                try:
                    proc.wait(timeout=POLL_S)
                    break
                except subprocess.TimeoutExpired:
                    if timeout_s and time.monotonic() - t0 >= timeout_s:
                        out.timed_out = True
                        break
        except KeyboardInterrupt:
            out.interrupted = True
        finally:
            while True:                           # a second Ctrl+C must not cut the stop short
                try:
                    tree.stop(proc)
                    break
                except KeyboardInterrupt:
                    out.interrupted = True
    out.returncode, out.wall_s, out.result = proc and proc.returncode, time.monotonic() - t0, read_result(eval_dir)
    return out


def tail(path, n=5):
    """The last n lines of a log (for error messages)."""
    try:
        lines = Path(path).read_text(errors='replace').strip().splitlines()
    except OSError:
        return ''
    return ' | '.join(lines[-n:])
