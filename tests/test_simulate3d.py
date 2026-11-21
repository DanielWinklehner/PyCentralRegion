"""simulate3d: the radial slit (its jaw raster, its place on the fast path, the placement rules, the pass numbering it
is placed by), the inflector-rotation guard, the hand-off bunch's species mass and snapshot launch, and the survivors
of a full run (the scorer's rule). No field cache needed: synthetic rasters, a stub tracker, a synthetic hand-off and
beam record."""
import types

import numpy as np
import pytest

pytest.importorskip('PyPATools')

from PyCentralRegion import config as C                     # noqa: E402
from PyCentralRegion import simulate3d as S                 # noqa: E402

PHASE = -15.0


def _slit(**kw):
    return C.SlitConfig(**{'half_width_mm': 3.0, 'azimuth_deg': 315.0, 'jaw_length_mm': 6.0, 'thickness_mm': 2.0, **kw})


def _point(u, w=0.0, z=0.0, phi_deg=315.0):
    """A point at u along the ray at phi_deg, w across it (counter-clockwise), height z."""
    phi = np.deg2rad(phi_deg)
    return np.array([[u * np.cos(phi) - w * np.sin(phi), u * np.sin(phi) + w * np.cos(phi), z]])


def _stub(sl, passes, steps_per_turn=1000):
    """What BunchTracker.slit_for reads: [slit], the reference crossings per phase, z_max and steps per turn."""
    return types.SimpleNamespace(slit=sl, slit_passes={} if passes is None else {PHASE: passes},
                                 mc=types.SimpleNamespace(z_max_m=0.03),
                                 tc=types.SimpleNamespace(steps_per_turn=steps_per_turn))


def test_slit_raster_jaws():
    """Jaws [r_c - hw - L, r_c - hw] and [r_c + hw, r_c + hw + L] on the ray, |w| <= t / 2, full height; the edges
    are good to the raster's 0.1 mm (points 0.15 mm off an edge fall on the right side)."""
    test, jaws = S.slit_raster(_slit(), 0.150, 0.03)
    assert np.allclose(jaws, [[0.141, 0.147], [0.153, 0.159]], rtol=0.0, atol=1e-12)

    def inside(u, w=0.0, z=0.0):
        return bool(test(_point(u, w, z))[0])
    for u in (0.150, 0.14715, 0.15285, 0.14085, 0.15915):          # the opening, and beyond the jaws
        assert not inside(u), u
    for u in (0.14115, 0.1445, 0.14685, 0.15315, 0.156, 0.15885):   # the jaws, at every height
        assert inside(u) and inside(u, z=0.029) and inside(u, z=-0.029), u
    assert inside(0.156, w=0.00085) and inside(0.156, w=-0.00085)
    assert not inside(0.156, w=0.00115) and not inside(0.156, w=-0.00115)
    assert not inside(-0.156)                                       # the opposite ray is free


def test_slit_is_the_fast_paths_second_raster():
    """Combined with the model's metal raster, the slit reaches the compiled 3D kernel as its second raster."""
    from PyCentralRegion.electrodes3d import combine_obstacles
    from PyCentralRegion.fast_track3d import _raster_parts
    xs, zs = np.arange(-0.2, 0.2 + 1e-3, 2e-3), np.arange(-0.03, 0.0305, 1e-2)
    metal = S._raster_inside(np.zeros((len(xs), len(xs), len(zs)), dtype=bool), xs, zs)
    slit, _ = S.slit_raster(_slit(), 0.150, 0.03)
    parts = _raster_parts(combine_obstacles(metal, slit))
    assert len(parts) == 2
    _, grid, ox, oy, oz, hx, hy, hz = parts[1]
    assert (hx, hy, hz) == pytest.approx((S.SLIT_H_M, S.SLIT_H_M, 0.03))
    assert int(grid.sum()) == 3 * int(slit.grid[:, :, 0].sum()) > 0
    both = combine_obstacles(metal, slit)
    assert both(_point(0.156))[0] and not both(_point(0.150))[0]


def test_slit_placement():
    """Centre = the reference radius at pass_index plus offset_mm (or r_center_mm with -1); passes that cross a jaw
    are reported; a missing pass, a missing reference run and a jaw thinner than a step are refused."""
    passes = {1: 0.100, 2: 0.126, 3: 0.150, 4: 0.156}
    test, rec = S.BunchTracker.slit_for(_stub(_slit(offset_mm=0.5), passes), PHASE)
    assert rec['r_center_mm'] == pytest.approx(150.5)
    assert np.allclose(rec['jaws_mm'], [[141.5, 147.5], [153.5, 159.5]], rtol=0.0, atol=1e-9)
    assert rec['reference_on_jaw_mm'] == {'4': pytest.approx(156.0)}            # pass 4 runs into the outer jaw
    assert rec['columns'] == int(test.grid[:, :, 0].sum()) and not test(_point(0.1505))[0]
    _, rec = S.BunchTracker.slit_for(_stub(_slit(pass_index=-1, r_center_mm=112.0), passes), PHASE)
    assert rec['r_center_mm'] == pytest.approx(112.0) and rec['reference_on_jaw_mm'] == {}
    assert S.BunchTracker.slit_for(_stub(_slit(half_width_mm=0.0), None), PHASE) == (None, None)
    with pytest.raises(C.ConfigError, match='pass_index = 5'):
        S.BunchTracker.slit_for(_stub(_slit(pass_index=5), passes), PHASE)
    with pytest.raises(RuntimeError, match='reference'):
        S.BunchTracker.slit_for(_stub(_slit(), None), PHASE)
    with pytest.raises(C.ConfigError, match='step over'):                     # 2 pi 0.159 m / 300 = 3.3 mm > 2 mm
        S.BunchTracker.slit_for(_stub(_slit(), passes, steps_per_turn=300), PHASE)


def test_reference_passes_on_the_slit_ray():
    """The slit is placed by the probes' pass numbering: a particle starting just past the ray (in its direction of
    motion) meets it first on pass 1, a revolution later."""
    from PyCentralRegion.accelerated_orbit_finder import ProbeRecorder
    n = 4000
    travel = np.linspace(0.0, 3.5 * 2.0 * np.pi, n)                  # 3.5 turns counter-clockwise from 320 deg
    th, r = np.deg2rad(320.0) + travel, 0.07 + 0.01 * travel / (2.0 * np.pi)
    xyz = np.column_stack([r * np.cos(th), r * np.sin(th), np.zeros(n)])
    v = np.gradient(xyz, axis=0)
    pr = ProbeRecorder([315.0])
    for k in range(n):
        pr.record(k, xyz[k:k + 1], v[k:k + 1], np.array([True]), float(k))
    d = pr.data()[0]
    assert list(d['pass']) == [1, 2, 3] and list(d['direction']) == [1, 1, 1]
    expect = 0.07 + 0.01 * np.array([355.0, 715.0, 1075.0]) / 360.0
    assert np.hypot(d['xyz'][:, 0], d['xyz'][:, 1]) == pytest.approx(expect, abs=2e-6)


def test_full_record_survivors():
    """The survivors of a full run (final energies, the probe files' 'survived') are the scorer's rule
    (``analysis.metrics.survivors``) on the last full states: not a particle lost on metal (its slot still inside), one
    beyond r_max, one at |z| = z_max, one with a NaN velocity or a NaN position. That is the tracker's former rule
    (finite state, inside, not among the metal hits), and the tracks carry the stop steps the scorer reads."""
    from PyCentralRegion.accelerated_orbit_finder import SparseBeamRecord
    from PyCentralRegion.analysis.metrics import survivors
    n, n_steps, dt = 7, 30, 1e-10
    fb = SparseBeamRecord([1e-9], n, 0.0, dt, track_ids=np.arange(n), track_every=1)
    r, v = np.tile([0.2, 0.0, 0.001], (n, 1)), np.tile([0.0, 2e7, 0.0], (n, 1))
    for k in range(n_steps):
        if k == n_steps - 1:
            r[2, 0], r[3, 2], v[4, 1], r[5] = 0.40, 0.03, np.nan, np.nan
        fb.record(k, r, v)
    hits = np.array([[1.0, 12.0, 0.2, 0.0]])                          # particle 1 on metal at step 12
    tracker = types.SimpleNamespace(mc=types.SimpleNamespace(z_max_m=0.03), finder=types.SimpleNamespace(r_max=0.38),
                                    tc=types.SimpleNamespace(snapshot_ns=[1.0], steps_per_turn=100), mass_mev=938.272,
                                    f_mhz=100.0)                       # 100 steps per turn of 10 ns: dt
    res = types.SimpleNamespace(n_turns=1, metadata={'launch': {'t0_s': 0.0}},
                                trajectory_reference=np.zeros((n_steps, 3)))
    bunch = types.SimpleNamespace(birth_time=np.zeros(n), handoff_index=np.arange(n))
    out = {}
    ok = S.BunchTracker._full_record(tracker, out, fb, res, bunch, hits, np.arange(n))
    assert ok.tolist() == [True, False, False, False, False, False, True]
    last = fb[-1]
    former = (np.isfinite(last).all(axis=1) & (np.hypot(last[:, 0], last[:, 1]) < 0.38) & (np.abs(last[:, 2]) < 0.03)
              & ~np.isin(np.arange(n), hits[:, 0].astype(int)))
    assert np.array_equal(ok, former)
    assert len(out['final_energies_mev']) == 2 and out['n_lost_vertical'] == 1
    tr = out['tracks']
    assert tr['stop_step'].tolist() == [n_steps + 1, 12] + [n_steps + 1] * 5 and tr['n_steps'] == n_steps
    assert not out['snapshots'][0]['lost'].any() and out['snapshots'][0]['step'] == 10    # the hit comes later
    # the scorer on the tracks (positions only: it cannot see particle 4's NaN velocity) finds the same survivors
    seen = survivors(tr['xyz'][-1], tr['stop_step'], tr['n_steps'], np.full(n, np.inf), 0.38, 0.03)
    assert np.array_equal(seen, ok | (np.arange(n) == 4))


def test_si_rotation_guard():
    model = types.SimpleNamespace(params={'si_rotation_deg': 4.726})
    S.check_si_rotation(model, 4.726)
    S.check_si_rotation(types.SimpleNamespace(params={}), 7.0)        # a model that does not record it
    with pytest.raises(C.ConfigError, match='turned by 4.726'):
        S.check_si_rotation(model, 0.0, 'fields.pkl')


def test_tracker_follows_the_models_si_rotation():
    """si_rotation = 'corner' turns the inflector with gap 0's ray (so with the dee opening): the run takes the angle
    its model was built around. 'fixed' leaves inflector.rotation_deg to the guard; a model without the record too."""
    model = types.SimpleNamespace(params={'si_rotation_deg': 7.0})
    cfg = {'geometry': {'si_rotation': 'corner'}, 'inflector': {'rotation_deg': 4.726}}
    S.follow_model_si_rotation(cfg, model)
    assert cfg['inflector']['rotation_deg'] == 7.0
    S.check_si_rotation(model, cfg['inflector']['rotation_deg'])
    cfg = {'geometry': {'si_rotation': 'fixed'}, 'inflector': {'rotation_deg': 4.726}}
    S.follow_model_si_rotation(cfg, model)
    assert cfg['inflector']['rotation_deg'] == 4.726
    cfg = {'geometry': {'si_rotation': 'corner'}}
    S.follow_model_si_rotation(cfg, types.SimpleNamespace(params={}))
    assert cfg == {'geometry': {'si_rotation': 'corner'}}


def _handoff(meta):
    """A load_handoff dict: three 62 keV protons (p = 10.787 MeV/c) on a plane, with the given file attributes."""
    p = np.sqrt(0.062 ** 2 + 2 * 0.062 * 938.272) * 1e6
    return {'r': np.zeros((3, 3)), 'p_ev_c': np.array([[p, 0.0, 0.0], [0.0, p, 0.0], [0.0, 0.0, p]]),
            'time': None, 'weight': None, 'meta': meta, 'path': 'synthetic.h5'}


def test_handoff_species_mass():
    """The hand-off's mass: the file's species_mass_mev, else the species' (MeV, like the file's); a species of
    another mass, or no species at all, is refused."""
    from PyCentralRegion.handoff import make_beam_from_handoff
    from PyPATools.species import IonSpecies
    proton = IonSpecies('proton')

    def speed(meta, **kw):                     # the centroid reference: the particles' mean speed
        pd = make_beam_from_handoff(_handoff(meta), reference='centroid', verbose=False, **kw)
        return np.linalg.norm(pd.reference_state[1])

    v_62kev = np.sqrt(1.0 - 1.0 / (1.0 + 0.062 / 938.272) ** 2) * 2.99792458e8
    assert speed({'species_mass_mev': 938.272}, species=proton) == pytest.approx(v_62kev, rel=1e-6)
    assert speed({}, species=proton) == pytest.approx(v_62kev, rel=1e-6)
    with pytest.raises(ValueError, match='another species'):
        speed({'species_mass_mev': 1877.268}, species=proton)
    with pytest.raises(ValueError, match='names no species'):
        speed({})


@pytest.mark.parametrize('sense', [1.0, -1.0])
def test_handoff_snapshot_sense(sense):
    """The snapshot launch moves every particle back along its own orbit circle by its t_rel, in the beam's sense of
    rotation (``probes.beam_sense``): on a circle about the axis, released a quarter period late it starts a quarter
    turn behind, released early a quarter turn ahead, and every particle is born at t = 0."""
    from PyCentralRegion.handoff import make_beam_from_handoff
    from PyPATools.species import IonSpecies
    d = _handoff({'species_mass_mev': 938.272})
    p = np.linalg.norm(d['p_ev_c'][0])
    speed = p / np.hypot(p, 938.272e6) * 2.99792458e8
    omega = speed / 0.02                                         # the orbit of r = 20 mm about the machine axis
    d.update(r=np.tile([0.02, 0.0, 0.001], (3, 1)), p_ev_c=np.tile([0.0, sense * p, 0.0], (3, 1)),
             t_rel=np.array([0.0, 0.5, -0.5]) * np.pi / omega)
    pd = make_beam_from_handoff(d, species=IonSpecies('proton'), launch='snapshot', reference='centroid',
                                omega_c=omega, verbose=False)
    np.testing.assert_allclose(pd.x_vec, [[0.02, 0.0, 0.001], [0.0, -sense * 0.02, 0.001], [0.0, sense * 0.02, 0.001]],
                               rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(pd.v_vec, [[0.0, sense * speed, 0.0], [speed, 0.0, 0.0], [-speed, 0.0, 0.0]],
                               rtol=0.0, atol=1e-6 * speed)
    assert not pd.birth_time.any()
