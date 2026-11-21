"""The space-charge kick in the compiled 3D kernel against the Python path: with a timed release, a solve every step
or every few steps, with and without the vertical kick, a magnetic part and an exempt reference particle, the kernel
(which calls ``SpaceChargeKick.refresh`` on the steps that need the solver and applies every kick itself) leaves the
positions, the velocities, the stored fields and the counters of ``TrackingEngine`` with ``SpaceChargeKick.apply``,
to the last bit. And a solve that runs out of memory keeps the previous field. A stand-in solver, no GPU."""
import types

import numpy as np
import pytest

pytest.importorskip('PyPATools')
from PyPATools.field import Field                                            # noqa: E402

from PyCentralRegion import fast_track3d as F                                # noqa: E402
from PyCentralRegion.accelerated_orbit_finder import make_beam_from_state    # noqa: E402
from PyCentralRegion.central_region import CentralRegion                     # noqa: E402
from PyCentralRegion.tracking import SpaceChargeKick, TimedRelease, TrackingEngine    # noqa: E402


class _LinearSolver:
    """Stands in for the FFT solver: a field that grows linearly from the centroid of the depositing particles, with
    a strength that follows their number; ``gather`` gives the last solve's field elsewhere. ``b`` adds a magnetic
    part; ``fail_at`` makes those solves (counted from 1) run out of memory."""

    def __init__(self, b=0.0, fail_at=()):
        self.b, self.fail_at, self.calls = float(b), set(fail_at), 0
        self.centre, self.k = np.zeros(3), 0.0

    def solve_eb(self, positions, charges, velocities=None):
        self.calls += 1
        if self.calls in self.fail_at:
            raise MemoryError('no room for the grid')
        self.centre = positions.mean(axis=0)
        self.k = 4.0e9 * float(np.sum(charges)) / 1e-12
        e = self.gather(positions)
        return e, self.b * np.cross(np.array([0.0, 0.0, 1.0]), positions - self.centre)

    def gather(self, positions):
        return self.k * (np.asarray(positions, dtype=float) - self.centre) * [1.0, 0.7, 0.4]


def _run(fast, every, kick_z, b, n_ref, fail_at=()):
    rng = np.random.default_rng(11)
    n, dt, n_steps = 48, 1e-10, 61
    r0 = rng.normal(size=(n, 3)) * [2e-3, 1e-3, 1e-3]
    u = rng.normal(size=(n, 3)) * [1.0, 1.0, 0.2]
    v0 = 1e6 * u / np.linalg.norm(u, axis=1)[:, None]
    birth = np.zeros(n, dtype=np.int64)
    birth[n // 2:] = rng.integers(1, 23, n - n // 2)                         # half the bunch is released later
    birth[0] = 0
    design = CentralRegion(name='space charge', dimensionality='3D')
    design.set_species('proton')
    design.set_magnetic_field(Field.zero())
    design.set_electric_field(Field.zero())
    engine = TrackingEngine(design, algorithm='boris_rel', dimensionality='3D', use_rf=False, max_radius_m=0.5,
                            verbose=False, z_max=0.1)
    pd = make_beam_from_state(design.species, r0, v0)
    pd.alive = birth == 0
    release = TimedRelease(birth)
    solver = _LinearSolver(b=b, fail_at=fail_at)
    sc = SpaceChargeKick(solver, resolve_every=every, macro_charge_c=2.5e-14, kick_z=kick_z)
    sc.prepare(pd, n_ref=n_ref, species=design.species)
    engine.extra_interactions = [release, sc]
    if not fast:
        res = engine.track_multiparticle(pd, dt=dt, n_steps=n_steps, callback=lambda *a: False, show_progress=False)
    else:
        finder = types.SimpleNamespace(engine=engine, fast3d_parallel=False, fast3d_threads=None)
        args = F.build_kernel_args3d(finder, pd, dt, n_steps, 0.0, birth, release, n_ref, interp='point')
        res = F.run_fast3d(finder, args, lambda *a: False, dt, 0.0)
        assert finder.last_fast_path['used'] and finder.last_fast_path['sc_solves'] == sc.n_solves
    return res, sc, solver


@pytest.mark.parametrize('every, kick_z, b, n_ref', [(1, True, 0.0, 0), (3, True, 0.0, 0), (5, False, 0.0, 1),
                                                     (4, True, 2e-3, 1), (1, False, 2e-3, 0)])
def test_the_kernel_kicks_as_the_python_path_does(every, kick_z, b, n_ref):
    if not F.HAS_NUMBA:
        pytest.skip('the compiled kernel needs numba')
    (py, sc_py, s_py), (fa, sc_fa, s_fa) = _run(False, every, kick_z, b, n_ref), _run(True, every, kick_z, b, n_ref)
    assert sc_py.n_solves == sc_fa.n_solves == s_py.calls == s_fa.calls >= 61 // every
    assert sc_py.n_kick_steps == sc_fa.n_kick_steps > 50 and sc_py.n_skipped == sc_fa.n_skipped
    np.testing.assert_array_equal(fa.active, py.active)
    np.testing.assert_array_equal(fa.r_final, py.r_final)
    np.testing.assert_array_equal(fa.v_final, py.v_final)
    np.testing.assert_array_equal(sc_fa.E, sc_py.E)
    np.testing.assert_array_equal(sc_fa.B, sc_py.B)
    np.testing.assert_array_equal(sc_fa.has_field, sc_py.has_field)
    assert np.all(fa.active) and np.abs(fa.v_final - np.asarray(make_v0())).max() > 1e3      # the kicks did something


def make_v0():
    rng = np.random.default_rng(11)
    rng.normal(size=(48, 3))
    u = rng.normal(size=(48, 3)) * [1.0, 1.0, 0.2]
    return 1e6 * u / np.linalg.norm(u, axis=1)[:, None]


def test_a_solve_out_of_memory_keeps_the_previous_field():
    """The third and fourth solves fail: the field of the second stays on the particles (and is the one kicked with)
    until the fifth; both paths agree, and the failures are counted."""
    if not F.HAS_NUMBA:
        pytest.skip('the compiled kernel needs numba')
    (py, sc_py, s_py), (fa, sc_fa, s_fa) = (_run(fast, 5, True, 0.0, 0, fail_at=(3, 4)) for fast in (False, True))
    assert sc_py.n_memory_errors == sc_fa.n_memory_errors == 2 and sc_py.n_skipped == sc_fa.n_skipped == 2
    assert sc_py.n_solves == sc_fa.n_solves == s_py.calls - 2
    np.testing.assert_array_equal(fa.r_final, py.r_final)
    np.testing.assert_array_equal(fa.v_final, py.v_final)
    ok, _, _ = _run(False, 5, True, 0.0, 0)
    assert np.abs(ok.v_final - py.v_final).max() > 0.0                       # the two lost solves are felt
    # a failed solve leaves the stored field as it was: the state after the failing solve is the state before it
    sc = SpaceChargeKick(_LinearSolver(fail_at=(2,)), resolve_every=1, macro_charge_c=1e-14, kick_z=True)
    pd = types.SimpleNamespace(numpart=4, species=None)
    species = types.SimpleNamespace(charge=1.602e-19, mass_kg=1.673e-27)
    sc.prepare(pd, n_ref=0, species=species)
    r = np.arange(12.0).reshape(4, 3) * 1e-3
    v = np.ones((4, 3))
    active = np.array([True, True, True, False])
    assert sc.refresh(0, r, v, active, 0.0)
    kept_e, kept_has = sc.E.copy(), sc.has_field.copy()
    assert kept_has.tolist() == [True, True, True, False] and np.any(kept_e != 0.0)
    assert sc.refresh(1, r + 1e-3, v, np.ones(4, dtype=bool), 1e-10)
    np.testing.assert_array_equal(sc.E, kept_e)
    np.testing.assert_array_equal(sc.has_field, kept_has)
    assert sc.n_memory_errors == 1 and sc.n_solves == 1
