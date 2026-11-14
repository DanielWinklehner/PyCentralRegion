"""The process a computation runs in: keeping it at full speed on Windows.

Windows 11 manages the execution speed of every process that does not ask otherwise ("power throttling"). A process
with a visible window in the foreground runs on the machine's fast cores at full clock rate. One without a window - a
campaign started detached, an evaluation the campaign driver starts with a hidden console - starts there too and is
moved within seconds to the efficiency cores of a hybrid CPU, where every thread of it runs at about half the speed
for the rest of its life.

Measured 2026-11-14 on a 8 P + 12 E core desktop, the same search-fidelity tracked run (10,000 particles, 7 turns)
started from a shell with a window and detached through WMI: 25 s against 47 s (the compiled kernel 3.0 against
10.5 s, the per-step records 2.4 against 4.0 s, the space-charge solves 8.2 against 16.6 s); a fixed single-thread
loop at the end of the run 0.51 s on logical CPUs 0-15 against 0.86 s on CPUs 16-27. Every campaign up to that day
ran that way: the same evaluation of a new geometry took 708 s in its campaign and 364 s from a shell, with the same
scores. With ``keep_full_speed`` the detached run takes 22 s.
"""
import sys

PROCESS_POWER_THROTTLING = 4                 # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
EXECUTION_SPEED = 0x1                        # PROCESS_POWER_THROTTLING_EXECUTION_SPEED


def _kernel32():
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):           # PROCESS_POWER_THROTTLING_STATE
        _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG), ('StateMask', wintypes.ULONG)]

    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    for fn in (k32.SetProcessInformation, k32.GetProcessInformation):
        fn.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
        fn.restype = wintypes.BOOL
    return ctypes, k32, State


def keep_full_speed():
    """Tell Windows not to manage this process's execution speed: it then stays on the fast cores at full clock rate
    whether or not it has a window (the documented opt-out, SetProcessInformation(ProcessPowerThrottling) with the
    execution-speed control taken and its state off). The priority class is not touched, so the desktop stays as
    responsive as with any other computation at normal priority. Every thread of the process is covered, the
    compiled kernels' and the libraries' included; a child process sets its own.

    Returns what happened as text: 'set', 'not needed' (not Windows), or why it could not be set (an old Windows, a
    restricted process). Never raises: a computation must not fail over its speed."""
    if sys.platform != 'win32':
        return 'not needed'
    try:
        ctypes, k32, State = _kernel32()
        state = State(1, EXECUTION_SPEED, 0)
        if k32.SetProcessInformation(k32.GetCurrentProcess(), PROCESS_POWER_THROTTLING, ctypes.byref(state),
                                     ctypes.sizeof(state)):
            return 'set'
        return f"not set (Windows error {ctypes.get_last_error()})"
    except Exception as exc:
        return f"not set ({type(exc).__name__}: {exc})"


def power_throttling():
    """(control mask, state mask) of this process's power throttling, None where it cannot be read (not Windows, an
    old Windows). (0, 0) is a process Windows manages as it sees fit; after ``keep_full_speed`` the control mask
    holds EXECUTION_SPEED and the state mask does not."""
    if sys.platform != 'win32':
        return None
    try:
        ctypes, k32, State = _kernel32()
        state = State(1, 0, 0)
        if not k32.GetProcessInformation(k32.GetCurrentProcess(), PROCESS_POWER_THROTTLING, ctypes.byref(state),
                                         ctypes.sizeof(state)):
            return None
        return int(state.ControlMask), int(state.StateMask)
    except Exception:
        return None
