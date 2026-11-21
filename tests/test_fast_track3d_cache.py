"""The compiled 3D kernel's serial and parallel builds each keep an on-disk numba cache of their own.

numba files a cached function by its source file, qualified name and first line, and keys the entry on the argument
types and the bytecode, never on the compile options. Two dispatchers of ONE Python function, one serial and one
parallel=True, would therefore share one cache entry, and each would load the build the other saved last: the parallel
kernel got the serial build (2026-09-16) and was then compiled afresh in every process, with the cache off (about 136 s
of every evaluation's first track in November 2026). The parallel kernel is compiled from a renamed copy of the body
instead (fast_track3d._renamed)."""
import pytest


def _kernels():
    pytest.importorskip('PyPATools')
    from PyCentralRegion import fast_track3d as F
    if not F.HAS_NUMBA:
        pytest.skip('numba not installed')
    return F


def _index_name(dispatcher):
    """The cache index file a dispatcher reads and writes, None with the cache off (numba internals: the test is
    skipped if they moved)."""
    cache = getattr(dispatcher, '_cache', None)
    if cache is None:
        pytest.skip("this numba keeps no Dispatcher._cache")
    if type(cache).__name__ == 'NullCache':                 # cache=False
        return None
    name = getattr(getattr(cache, '_cache_file', None), '_index_name', None)
    if name is None:
        pytest.skip("this numba keeps its cache index elsewhere (Dispatcher._cache._cache_file._index_name)")
    return name


def test_the_parallel_kernel_has_a_cache_file_of_its_own():
    F = _kernels()
    ser, par = F._chunk3d_serial, F._chunk3d_par
    assert par.targetoptions.get('parallel') and not ser.targetoptions.get('parallel')
    names = [_index_name(ser), _index_name(par)]
    assert None not in names, f"a kernel build runs with the cache off: {names}"
    assert names[0] != names[1], names                      # one shared entry would hand one build to both
    assert ser.py_func.__code__.co_code == par.py_func.__code__.co_code       # one body, compiled twice
    assert par.py_func.__code__.co_filename == ser.py_func.__code__.co_filename


def test_the_renamed_copy_is_the_same_function():
    F = _kernels()
    f = F._renamed(F._chunk3d_impl, '_copy_for_the_test')
    assert f is not F._chunk3d_impl and f.__name__ == f.__qualname__ == '_copy_for_the_test'
    assert f.__module__ == F._chunk3d_impl.__module__ and f.__globals__ is F._chunk3d_impl.__globals__
    assert f.__code__.co_code == F._chunk3d_impl.__code__.co_code
    assert f.__code__.co_firstlineno == F._chunk3d_impl.__code__.co_firstlineno
    assert F._chunk3d_impl.__name__ == '_chunk3d_impl'          # the original is left as it was
