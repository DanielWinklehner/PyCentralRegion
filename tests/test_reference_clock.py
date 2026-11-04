"""The turn clock of a bunch run (AcceleratedOrbitFinder.track_with_rf) when its virtual reference is lost: the real
callback driven by a stub engine on an analytic circular orbit (no field, no PyPATools tracking). A turn is logged at
each crossing of the reference's launch azimuth; once the reference is gone the fallback clock counts the remaining
turns every mean period after its last logged crossing (the nominal steps per turn when none was logged), so the run
still stops after max_turns instead of spending its step budget with a turn log that ended early. A reference not yet
born (timed release) is not lost; a run whose real particles are all gone stops as 'beam_lost'."""
import types

import numpy as np
import pytest

pytest.importorskip('PyPATools')
from PyCentralRegion.accelerated_orbit_finder import AcceleratedOrbitFinder, TURN_BUDGET_MARGIN   # noqa: E402
from PyCentralRegion.simulate3d import final_transmission                                    # noqa: E402
from PyCentralRegion.tracking import TrackingResult                                          # noqa: E402

PERIOD = 103            # true revolution period [steps]
SPT = 100               # the finder's nominal steps per turn
DT = 1e-9
R = 0.1                 # orbit radius [m]
PHI0 = 0.3              # the reference's launch azimuth [rad]: the section is there (not the +x axis)


class StubEngine:
    """Every particle on the circle of radius R, PERIOD steps per revolution, from its own launch azimuth from its
    birth step on (frozen before); kill_ref / kill_real: the step from which the reference / the real particles are
    inactive. Births follow the finder's TimedRelease hook (its birth_step)."""
    algorithm = 'rk4'

    def __init__(self, kill_ref=None, kill_real=None):
        self.kill_ref, self.kill_real = kill_ref, kill_real
        self.extra_interactions = []
        self.steps_run = None

    def track_multiparticle(self, pd_init, dt, n_steps, callback, callback_frequency=1, show_progress=False,
                            t0=0.0):
        n = int(pd_init.numpart)
        phi0 = np.arctan2(pd_init.x_vec[:, 1], pd_init.x_vec[:, 0])
        rel = [e for e in self.extra_interactions if hasattr(e, 'birth_step')]
        birth = np.asarray(rel[0].birth_step) if rel else np.zeros(n, dtype=int)
        active = np.asarray(pd_init.alive, dtype=bool).copy()
        omega = 2 * np.pi / (PERIOD * dt)
        r = v = None
        for step in range(n_steps):
            active |= birth <= step
            if self.kill_ref is not None and step >= self.kill_ref:
                active[0] = False
            if self.kill_real is not None and step >= self.kill_real:
                active[1:] = False
            phi = phi0 + 2 * np.pi * np.maximum(step + 1 - birth, 0) / PERIOD
            r = R * np.column_stack([np.cos(phi), np.sin(phi), np.zeros(n)])
            v = R * omega * np.column_stack([-np.sin(phi), np.cos(phi), np.zeros(n)])
            t = t0 + (step + 1) * dt
            if not active.any():
                self.steps_run = step + 1
                return TrackingResult(False, step + 1, r, v, active, {'termination': 'all_lost', 'time': t})
            if callback(step, r, v, active.copy(), t):
                self.steps_run = step + 1
                return TrackingResult(True, step + 1, r, v, active, {'termination': 'turns_or_energy_reached',
                                                                     'time': t})
        self.steps_run = n_steps
        return TrackingResult(True, n_steps, r, v, active, {'time': t0 + n_steps * dt})


def _finder(engine, launch='snapshot'):
    cav = types.SimpleNamespace(omega=2 * np.pi * 4 / (PERIOD * DT), get_total_phase_rad=lambda: 0.0)
    design = types.SimpleNamespace(species=types.SimpleNamespace(mass_mev=938.272), rf_cavities=[cav])
    return types.SimpleNamespace(_n_ref=1, design=design, record_centroid=False, center_source='reference',
                                 launch=launch, engine=engine, space_charge=None, is_multiparticle=True,
                                 steps_per_turn=SPT, target_energy_mev=np.inf, verbose=False, fast_path=False,
                                 fast3d=False, n_particles=2)


def _beam(birth_time=None):
    """The reference (row 0) at PHI0, two real particles a little ahead and behind it."""
    phi = PHI0 + np.array([0.0, 0.02, -0.02])
    return types.SimpleNamespace(x_vec=R * np.column_stack([np.cos(phi), np.sin(phi), np.zeros(3)]), numpart=3,
                                 alive=np.ones(3, dtype=bool), birth_time=birth_time)


def _run(kill_ref=None, kill_real=None, birth_time=None, max_turns=5):
    eng = StubEngine(kill_ref, kill_real)
    f = _finder(eng, 'timed' if birth_time is not None else 'snapshot')
    # the section a hair behind the launch: the reference's states land exactly on PHI0 + 2 pi k otherwise
    out = AcceleratedOrbitFinder.track_with_rf(f, _beam(birth_time), DT, max_turns, section_angle=PHI0 - 1e-3)
    success, turn_stats, _, _, _, _, turn_ids, _ = out
    return f, eng, turn_stats, turn_ids


def test_reference_survives_unchanged():
    """No loss: turns at the reference's crossings (step k PERIOD - 1), the stop at the 5th, no fallback."""
    f, eng, ts, ids = _run()
    assert ids == [k * PERIOD - 1 for k in range(1, 6)]
    assert len(ts) == 5 and eng.steps_run == 5 * PERIOD
    assert f.last_reference_lost is None and f.last_stop == 'turns' and not f.last_budget_exhausted


def test_reference_lost_after_two_turns():
    """Lost at step 250 after crossings at 102 and 205: the clock goes on every (205 - (-1)) / 2 = 103 steps, so
    the run ends on its 5th turn at step 514 (the reference's own 5th crossing), not at the step budget."""
    f, eng, ts, ids = _run(kill_ref=250)
    assert ids == [102, 205]
    rl = f.last_reference_lost
    assert rl == {'step': 250, 'launch_step': -1, 'turns_logged': 2, 'period_steps': 103.0,
                  'clock_steps': [308, 411, 514]}
    assert len(ts) == 5 and eng.steps_run == 515 < (5 + TURN_BUDGET_MARGIN) * SPT
    assert f.last_stop == 'turns' and not f.last_budget_exhausted
    assert [t.n_active for t in ts] == [2] * 5


def test_reference_lost_before_its_first_turn():
    """Lost before any crossing: the nominal steps per turn from its launch."""
    f, eng, ts, ids = _run(kill_ref=50)
    assert ids == []
    rl = f.last_reference_lost
    assert rl['turns_logged'] == 0 and rl['period_steps'] == SPT and rl['clock_steps'] == [99, 199, 299, 399, 499]
    assert len(ts) == 5 and f.last_stop == 'turns'


def test_unborn_reference_is_not_lost():
    """Timed release: the reference born 10 steps after the earliest particle is frozen, not lost, until then; its
    turns count from its own launch. Lost later, the clock period is still the true one."""
    birth = np.array([0.0, -10 * DT, -5 * DT])
    f, eng, ts, ids = _run(birth_time=birth)
    assert ids == [9 + k * PERIOD for k in range(1, 6)] and f.last_reference_lost is None and len(ts) == 5
    f, eng, ts, ids = _run(kill_ref=260, birth_time=birth)
    rl = f.last_reference_lost
    assert ids == [112, 215] and rl['launch_step'] == 9 and rl['period_steps'] == PERIOD
    assert rl['clock_steps'] == [318, 421, 524] and len(ts) == 5


def test_beam_lost_stops_the_run():
    """The real particles gone: the run stops at once, as 'beam_lost' while the reference is alive, as 'all_lost'
    (nothing active: the engine ends it) when the reference was already on the clock."""
    f, eng, ts, ids = _run(kill_real=300)
    assert f.last_stop == 'beam_lost' and eng.steps_run == 301 and len(ts) == 2 and f.last_reference_lost is None
    f, eng, ts, ids = _run(kill_ref=150, kill_real=380)
    assert f.last_stop == 'all_lost' and eng.steps_run == 381
    assert f.last_reference_lost['clock_steps'] == [205, 308] and len(ts) == 3


def test_final_transmission_rule():
    assert final_transmission([60, 50, 40], 'turns', 100) == 0.4
    assert final_transmission([60, 50], 'beam_lost', 100) == 0.0
    assert final_transmission([60], 'all_lost', 100) == 0.0
    assert final_transmission([], 'turns', 100) == 0.0
    assert final_transmission([60, 55], 'budget', 100) == 0.55
