"""The space-charge kick of tracking.SpaceChargeKick: the compiled kick against the NumPy one, bit for bit. The NumPy
kick is the reference the compiled 3D kernel was written against; the compiled one replaces it on the steps the hook
kicks itself (every solve step: every step of a run that solves at every step)."""
import numpy as np
import pytest

pytest.importorskip('PyPATools')
pytest.importorskip('numba')

from PyCentralRegion import tracking as T                                                     # noqa: E402

Q_OVER_M = 4.79e7           # about H2+ [C/kg]


class _Solver:
    def solve_eb(self, positions, charges, velocities):
        raise AssertionError("no solve in a kick test")


def _hook(n, rng, kick_z, with_b, zero_rows):
    hook = T.SpaceChargeKick(_Solver(), kick_z=kick_z)
    hook._reset_state(n)
    hook.q_over_m = Q_OVER_M
    hook.E = rng.normal(size=(n, 3)) * 2.0e4
    hook.B = rng.normal(size=(n, 3)) * 1.0e-3 if with_b else np.zeros((n, 3))
    hook.E[zero_rows] = 0.0
    hook.B[zero_rows] = 0.0
    return hook


@pytest.mark.parametrize('kick_z', [True, False])
@pytest.mark.parametrize('with_b', [False, True])
def test_compiled_kick_is_the_numpy_kick_bit_for_bit(kick_z, with_b):
    rng = np.random.default_rng(11 + 2 * kick_z + with_b)
    n = 100_000
    v0 = rng.normal(size=(n, 3)) * 5.0e6 + np.array([1.2e7, -3.0e6, 2.0e5])     # up to a few per cent of c
    mask = rng.uniform(size=n) < 0.8
    zero_rows = rng.uniform(size=n) < 0.1                                       # rows without a field are not touched
    out = {}
    for compiled in (False, True):
        hook = _hook(n, np.random.default_rng(5), kick_z, with_b, zero_rows)
        hook.compiled_kick = compiled
        v = v0.copy()
        for _ in range(3):                                                      # kick upon kick: errors would grow
            hook._kick(v, mask, 1.2e-10)
        out[compiled] = (v, hook.n_kick_steps)
    assert out[True][1] == out[False][1] == 3
    assert np.array_equal(out[True][0], out[False][0])                          # every bit
    moved = np.any(out[True][0] != v0, axis=1)
    assert np.array_equal(moved, mask & ~zero_rows)                             # and only the rows with a field
    if not kick_z and not with_b:
        only_ez = _hook(n, np.random.default_rng(5), False, False, zero_rows)
        only_ez.E[:, :2] = 0.0                                                  # E_z alone is no kick without kick_z
        v = v0.copy()
        only_ez._kick(v, mask, 1.2e-10)
        assert np.array_equal(v, v0) and only_ez.n_kick_steps == 0


def test_compiled_kick_without_a_field_is_no_kick():
    for compiled in (False, True):
        hook = _hook(50, np.random.default_rng(1), True, True, np.ones(50, dtype=bool))
        hook.compiled_kick = compiled
        v0 = np.random.default_rng(2).normal(size=(50, 3)) * 1e6
        v = v0.copy()
        hook._kick(v, np.ones(50, dtype=bool), 1e-10)
        assert np.array_equal(v, v0) and hook.n_kick_steps == 0
