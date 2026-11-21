"""Set-up shared by every test module."""
import signal

SIGINT_HANDLER = signal.getsignal(signal.SIGINT)


def pytest_runtest_teardown(item, nextitem):
    """gmsh.initialize() sets SIGINT to SIG_DFL and gmsh 4.15's finalize() does not put the old handler back (it keeps
    it in a local), so after any build with gmsh in this process (test_build_cuts, the demo's inputs and chain in
    test_demo) the next SIGINT ends pytest from the thread that raised it: the Ctrl+C tests of test_mobo raise one from
    a timer thread, and the whole run hangs in that exit. Every test leaves the handler pytest started with, also one
    whose gmsh ran in a module fixture's set-up."""
    if signal.getsignal(signal.SIGINT) is not SIGINT_HANDLER:
        signal.signal(signal.SIGINT, SIGINT_HANDLER)
