"""analysis: known answers of the scoring library on synthetic bunches (no machine data): the core-variance factor,
the bunch shape of a Gaussian of known aspect ratio and tilt, per-particle orbit centres of circles with known
centres, probe crossings at a known radius, the turn structure, energy windows, the survivor and hand-off-index rules,
the operating-point rule, a whole synthetic run through score_run, and a synthetic space-charge twin whose
guiding-centre core rotates by a known angle per turn."""
import json
import pickle

import numpy as np
import pytest

from PyCentralRegion.analysis import metrics as M
from PyCentralRegion.analysis import objectives as O
from PyCentralRegion.analysis import runs as R
from PyCentralRegion.analysis import twins as TW
from PyCentralRegion.config import get_dotted

pytest.importorskip('scipy')


def test_core_variance_factor_formula_and_monte_carlo():
    """(1 - (1 - q)(1 - ln(1 - q))) / q is 0.8423 at q = 0.95 and is what trimming a 2D Gaussian to its q core
    leaves of the variance."""
    assert M.core_variance_factor(0.95) == pytest.approx(0.8423, abs=1e-4)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400_000, 2))
    m2 = np.sum(x ** 2, axis=1)
    for q in (0.9, 0.95):
        core = m2 <= np.quantile(m2, q)
        assert np.mean(x[core] ** 2) == pytest.approx(M.core_variance_factor(q), rel=3e-3)


def _frame_bunch(rng, n, sig_major, sig_minor, tilt_deg, R=100.0, theta=0.7, sense=1.0, K=4):
    """K instants of a Gaussian bunch centred at radius R [mm], azimuth theta, its major axis tilt_deg from u_r
    towards u_l (ahead for sense)."""
    a = np.deg2rad(tilt_deg)
    P = []
    for _ in range(K):
        p = rng.normal(size=(n, 2)) * [sig_major, sig_minor]
        u_r, u_l = p[:, 0] * np.cos(a) - p[:, 1] * np.sin(a), p[:, 0] * np.sin(a) + p[:, 1] * np.cos(a)
        er, el = np.array([np.cos(theta), np.sin(theta)]), sense * np.array([-np.sin(theta), np.cos(theta)])
        P.append(R * er + u_r[:, None] * er + u_l[:, None] * el)
    return np.array(P)


@pytest.mark.parametrize('sense', [1.0, -1.0])
def test_shape_stats_of_a_gaussian_bunch(sense):
    rng = np.random.default_rng(1)
    P = _frame_bunch(rng, 40_000, 6.0, 2.0, 20.0, sense=sense)
    u = M.bunch_coordinates(P, sense)
    Cc, d2 = M.core_covariance(u, 0.95)
    st = M.shape_stats(Cc, d2, 3.0)
    f = np.sqrt(M.core_variance_factor(0.95))
    assert st['ratio'] == pytest.approx(3.0, rel=0.02)
    assert st['sig_1'] == pytest.approx(2.0 * f, rel=0.02) and st['sig_2'] == pytest.approx(6.0 * f, rel=0.02)
    assert st['tilt'] == pytest.approx(20.0, abs=1.0)
    assert st['s_round'] == pytest.approx(np.sqrt(12.0) * f, rel=0.02)
    assert st['halo3'] == pytest.approx(np.exp(-4.5), abs=0.002)          # 2D Gaussian beyond 3 sigma
    rd = M.roundness(P, sense, 0.95)                      # the cavity optimizer's roundness: one implementation
    assert np.exp(rd['R']) == pytest.approx(st['ratio'])


def test_gauss_gradients_round_bunch():
    """A round Gaussian of rms s has the central gradient rho0 / (3 eps0), rho0 = 1 / ((2 pi)^1.5 s^3) per coulomb."""
    s = 2e-3
    g = M.gauss_gradients(s, s, s)
    assert g[0] == pytest.approx(1.0 / (3 * M.EPS0 * (2 * np.pi) ** 1.5 * s ** 3), rel=1e-9)
    assert g[1] == pytest.approx(g[0])


def test_isochronous_field():
    assert M.isochronous_field(8.191375e6, 1876.124) == pytest.approx(1.0744, rel=1e-3)   # H2+


# ---------------------------------------------------------------------------------------------------------------------
def _circles(centres_mm, rho_mm, spt=400, turns=4, phi0=0.3, sense=1.0):
    """Particles on circles of radius rho about known centres, one record per step: xyz (S, n, 3) [m]."""
    s = np.arange(spt * turns)
    phi = phi0 + sense * 2 * np.pi * s / spt
    c = np.asarray(centres_mm, dtype=float) * 1e-3
    xy = c[None, :, :] + 1e-3 * np.asarray(rho_mm)[None, :, None] * np.stack([np.cos(phi), np.sin(phi)], 1)[:, None]
    return np.concatenate([xy, np.zeros(xy.shape[:2] + (1,))], axis=2), s


@pytest.mark.parametrize('sense', [1.0, -1.0])
def test_turn_centres_recover_known_centres(sense):
    rng = np.random.default_rng(2)
    cen = rng.normal(0.0, 1.5, size=(50, 2))
    xyz, s = _circles(cen, rng.uniform(80.0, 120.0, 50), sense=sense)
    alive = np.ones(xyz.shape[:2], bool)
    assert M.rotation_sense(xyz, alive) == sense
    c = M.turn_centres(xyz[..., :2], alive, sense)
    full = np.all(np.isfinite(c), axis=(0, 2))
    assert full.sum() >= 3
    # the fit r(theta) = a + b (theta - theta_mid) + cx cos + cy sin is first order in |c| / rho: the rest is
    # below a quarter of |c|^2 / rho (0.04 mm for the largest offset here)
    assert np.abs(c[:, full] - cen[:, None, :]).max() < 0.25 * np.max(np.sum(cen ** 2, axis=1)) / 80.0
    fan = M.centre_fan(c, degree=3)
    row = fan[1]
    lam = np.linalg.eigvalsh(np.cov(cen.T, bias=True))
    assert row['spread'] == pytest.approx(np.sqrt(lam.sum()), rel=0.01)
    assert row['proj'] == pytest.approx(np.sqrt(lam.sum() / 2), rel=0.01)
    assert row['proj_max'] == pytest.approx(np.sqrt(lam[-1]), rel=0.01)
    assert row['coh'] == pytest.approx(np.hypot(*cen.mean(axis=0)), abs=0.02)


def test_probe_crossings_at_known_radius_and_pass_numbering():
    rho = np.array([90.0, 100.0, 110.0])
    xyz, s = _circles(np.zeros((3, 2)), rho, spt=1000, turns=3, phi0=0.0)
    xyz[:, :, 2] = 1e-3 * np.array([0.5, -1.0, 2.0])
    alive = np.ones(xyz.shape[:2], bool)
    alive[:600, 2] = False                               # the third starts past 45 deg: its first crossing is pass 1
    pr = M.probe_crossings(xyz, alive, s, 1e-9, 1.0, 45.0)
    for j in range(3):
        m = pr['j'] == j
        assert np.allclose(pr['R'][m], rho[j], atol=1e-3)
        assert np.allclose(pr['z'][m], 1e3 * xyz[0, j, 2])
        assert list(pr['pass'][m]) == ([0, 1, 2] if j < 2 else [1, 2])
    assert np.allclose(pr['t'][pr['j'] == 0], (np.array([0, 1, 2]) + 0.125) * 1000 * 1e-9, atol=2e-9)
    # a reference spiral: {pass: R} at the crossing
    k = np.arange(3000)
    r = 60.0 + 24.0 * k / 1000
    ref = 1e-3 * np.column_stack([r * np.cos(2 * np.pi * k / 1000), r * np.sin(2 * np.pi * k / 1000), 0 * k])
    rc = M.reference_crossings(ref, 90.0, 1.0)
    assert sorted(rc) == [0, 1, 2]
    assert rc[1] == pytest.approx(60.0 + 24.0 * 1.25, abs=0.01)


def _cfg(**kw):
    return O.ScoreConfig(**{'min_tracks': 50, 'min_survivors': 20, **kw})


def test_turn_structure_three_passes():
    rng = np.random.default_rng(3)
    n = 3000
    j = np.repeat(np.arange(n), 3)
    p = np.tile([4, 5, 6], n)
    Rr = np.array([100.0, 124.0, 148.0])[p - 4] + rng.normal(0.0, 3.0, 3 * n)
    pr = {'j': j, 'pass': p, 'R': Rr, 'z': 0 * Rr, 't': 0 * Rr}
    ts = M.turn_structure(pr, np.ones(n, bool), n, _cfg())
    assert ts['passes'] == [4, 5, 6]
    assert np.allclose(ts['R'], [100.0, 124.0, 148.0], atol=0.3)
    assert np.allclose(ts['sigma'], 3.0, rtol=0.06)
    assert ts['sep'] == pytest.approx(8.0, rel=0.06)
    assert ts['valley'] < 0.01 and ts['halo'] == pytest.approx(2 * 0.0062, abs=0.004)   # Gaussian beyond 2.5 sigma
    assert M.turn_structure(pr, np.ones(n, bool), 2 * n, _cfg()) is None       # no pass is complete


# ---------------------------------------------------------------------------------------------------------------------
def test_energy_windows_exact():
    rng = np.random.default_rng(4)
    e = np.r_[rng.normal(3.0, 0.02, 300), rng.normal(3.2, 0.01, 200)]       # two peaks
    for rel in (0.02, 0.01):
        n, c = M.energy_window_best(e, rel)
        grid = np.linspace(e.min(), e.max(), 20001)
        brute = max(np.sum(np.abs(e - g) <= rel * g) for g in grid[::20])
        assert n >= brute
        assert np.sum((e >= c * (1 - rel)) & (e <= c * (1 + rel))) == n
    med = np.median(e)
    assert M.energy_window_median(e, 0.02) == np.sum(np.abs(e - med) <= 0.02 * med)
    out = M.energy_metrics(e, 1000, 8.0, np.tile([0.3, 0.0, 0.0], (len(e), 1)), (0.02,))
    assert out['sigR_E'] == pytest.approx(0.5 * 300.0 * out['dE_E_robust'])
    assert out['I_2pct_best'] >= out['I_2pct']


def test_survivors_and_losses():
    n_steps = 100
    steps = np.arange(0, n_steps, 10)
    xyz = np.zeros((len(steps), 5, 3))
    xyz[..., 0] = 0.1
    xyz[-1, 1, 0] = 0.5                       # left at the radius limit: no metal hit recorded
    xyz[-1, 2, :] = np.nan                    # not finite
    xyz[:3, 4, 0] = 0.5                       # outside the limit before its birth only
    stop = np.array([n_steps + 1, n_steps + 1, n_steps + 1, 50, n_steps + 1])
    birth = np.array([0, 0, 0, 0, 30])
    g = M.ghost_first_step(np.arange(5), np.array([0, 0]), np.array([70, 40]))
    assert g[0] == 40 and np.isinf(g[1:]).all()
    surv = M.survivors(xyz[-1], stop, n_steps, g, 0.38, 0.03)
    assert list(surv) == [False, False, False, False, True]
    loss = M.loss_steps(steps, xyz, birth, stop, g, n_steps, surv, 0.38, 0.03)
    assert list(loss[:4]) == [40, 90, 90, 50] and np.isinf(loss[4])


def test_handoff_track_ids_rule():
    rng = np.random.default_rng(5)
    n_h, n_b, dt = 2000, 600, 1e-10
    t_rel = rng.normal(0.0, 2e-9, n_h)
    idx = np.sort(np.random.default_rng(2).choice(n_h, n_b, replace=False))
    ids = np.sort(np.random.default_rng(12345).choice(n_b, 200, replace=False))
    birth = np.round((t_rel[idx][ids] + 1e-8) / dt).astype(int)
    hid, note = R.handoff_track_ids(ids, birth, n_b, n_h, t_rel, seed=2)
    assert note is None and np.array_equal(hid, idx[ids])
    hid, note = R.handoff_track_ids(ids, birth, n_b, n_h, t_rel, seed=3)      # wrong seed: refused
    assert hid is None and 'do not match' in note
    hid, note = R.handoff_track_ids(ids, birth, n_b, n_h, t_rel, seed=3, recorded=idx[ids])
    assert note is None and np.array_equal(hid, idx[ids])
    full = np.round((t_rel[ids] + 1e-8) / dt).astype(int)                       # unsampled bunch: the ids themselves
    hid, note = R.handoff_track_ids(ids, full, 0, n_h, t_rel, seed=2)
    assert note is None and np.array_equal(hid, ids)
    hid, note = R.handoff_track_ids(ids, np.zeros(len(ids), int), n_b, n_h, t_rel, seed=2)   # no timed release
    assert np.array_equal(hid, idx[ids]) and 'not checked' in note


def test_pick_operating_point():
    rows = [{'v': 1.00, 'phase': -10, 'transmission': 0.60, 'e_final': 3.0, 'de_e': 0.010},
            {'v': 1.05, 'phase': -15, 'transmission': 0.612, 'e_final': 3.0, 'de_e': 0.020},
            {'v': 1.10, 'phase': -20, 'transmission': 0.609, 'e_final': 3.1, 'de_e': 0.008},
            {'v': 1.20, 'phase': -20, 'transmission': 0.700, 'e_final': 3.2, 'de_e': 0.005},
            {'v': 1.00, 'phase': -30, 'transmission': 0.800, 'e_final': 2.8, 'de_e': 0.001}]
    pk = M.pick_operating_point(rows, _cfg())
    assert pk['best_t'] == 0.612 and pk['winner'] is rows[2] and pk['uncapped_best'] is rows[3] and pk['n_ok'] == 3
    with pytest.raises(ValueError):
        M.pick_operating_point(rows[3:], _cfg())


# ---------------------------------------------------------------------------------------------------------------------
H, SPT, TURNS, EVERY = 4, 400, 6, 4


def _spiral_run(n=600, seed=6, rotate_deg=0.0, offsets=True, phase=True, halo=0.0, tag='syn'):
    """A synthetic accelerated bunch: guiding centres (g_r, g_l) about a spiral R_b = 60 + 24 tau mm (tau in turns),
    optionally rotating by rotate_deg per turn in (r, l); the azimuth lead g_l / 60 mm conserved (phase, an
    isochronous machine) or g_l / R_b (the arc length); per-particle orbit-centre offsets; a share halo of the
    particles radially 4x wider; 5 % lost on metal in turns 1-2, one particle leaving at the radius limit. Release
    phase = -h x the initial azimuth lead."""
    rng = np.random.default_rng(seed)
    n_steps = SPT * TURNS
    steps = np.arange(0, n_steps, EVERY)
    tau = steps / SPT
    g0 = np.column_stack([rng.normal(0.0, 1.5, n), rng.normal(0.0, 4.0, n)])
    g0[rng.random(n) < halo, 0] *= 4.0
    a = np.deg2rad(rotate_deg) * tau[:, None]
    g_r = np.cos(a) * g0[:, 0] - np.sin(a) * g0[:, 1]
    g_l = np.sin(a) * g0[:, 0] + np.cos(a) * g0[:, 1]
    Rb = 60.0 + 24.0 * tau[:, None]
    th = 0.1 + 2 * np.pi * tau[:, None] + g_l / (60.0 if phase else Rb)      # start off the probe planes
    c = rng.normal(0.0, 0.4, (n, 2)) if offsets else np.zeros((n, 2))
    r = Rb + g_r
    xyz = 1e-3 * np.stack([c[:, 0] + r * np.cos(th), c[:, 1] + r * np.sin(th), np.broadcast_to(
        rng.normal(0.0, 1.0, n), r.shape)], axis=2)
    stop = np.full(n, n_steps + 1)
    lost = rng.choice(n, n // 20, replace=False)
    stop[lost] = rng.integers(SPT, 2 * SPT, len(lost))
    for i in lost:
        xyz[steps >= stop[i], i] = np.nan
    wall = np.setdiff1d(np.arange(n), lost)[0]
    xyz[-1, wall, :2] *= 0.40 / np.hypot(*xyz[-1, wall, :2])            # out radially to 0.40 m > r_max
    k = np.arange(n_steps)
    rr, tr = 60.0 + 24.0 * k / SPT, 0.1 + 2 * np.pi * k / SPT
    ref = 1e-3 * np.column_stack([rr * np.cos(tr), rr * np.sin(tr), 0 * k])
    T = float(np.mean((stop > n_steps) & (np.arange(n) != wall)))
    info = {'rf_mhz': 32.0, 'f_rev_mhz': 8.0, 'harmonic': H, 'limits': {'r_max_m': 0.38, 'z_max_m': 0.03},
            'final': {'transmission': T, 'phase': 0.0, 'sc': False}, 'space_charge': {'current_ma': 5.0}}
    return R.RunData(tag=tag, ids=np.arange(n), steps=steps, xyz=xyz, stop_step=stop, birth_step=np.zeros(n, int),
                     n_steps=n_steps, steps_per_turn=SPT, E_mev=3.0 + 0.03 * rng.standard_normal(n), info=info,
                     handoff_ids=np.arange(n), release_phase_deg=-H * np.degrees(g0[:, 1] / 60.0),
                     handoff_z_m=xyz[0, :, 2], handoff_zp_rad=np.zeros(n), ref_xyz=ref)


def test_score_run_on_a_synthetic_bunch():
    run = _spiral_run(halo=0.1)
    sc = M.score_run(run, _cfg())
    assert 'skip' not in sc
    for name, m in O.METRICS.items():                    # every listed objective is a top-level scalar
        if m.fn is None and not name.startswith('rot_'):
            assert np.isfinite(get_dotted(sc, name)), name
    assert sc['T'] == pytest.approx(run.info['final']['transmission']) and sc['I_final'] == pytest.approx(5.0 * sc['T'])
    assert sc['n_surv'] == round(sc['T'] * run.n_tracks)
    assert sc['E_med'] == pytest.approx(3.0, abs=0.01) and sc['dE_E_robust'] == pytest.approx(0.01, rel=0.15)
    assert sc['slope'] == pytest.approx(1.0, abs=0.02)                    # release phase conserved
    assert sc['compress_t3'] == pytest.approx(1.0, abs=0.05)
    assert sc['turn_sep'] > 8 and 0.0 < sc['valley_all'] < 0.1 and sc['valley_rad20'] < sc['valley_all']
    assert sc['fan_proj_last'] == pytest.approx(0.4, rel=0.2)           # the centre offsets, rms 0.4 mm per axis
    assert set(sc['probes']) == {f"{a:g}" for a in O.ScoreConfig().probe_angles_deg}
    bs = get_dotted(sc, 'best_slit.20')                                # a seed for a slit campaign
    assert bs['pass_index'] >= 0 and bs['offset_mm'] == pytest.approx(
        bs['r_center_mm'] - M.reference_crossings(run.ref_xyz, bs['azimuth_deg'], 1.0)[bs['pass_index']])
    assert get_dotted(sc, 'turns.3.sig_r') == sc['sig_r_t3']
    assert R.clean(sc) == R.clean(R.clean(sc))                         # JSON-clean


def test_score_without_handoff_and_saving(tmp_path):
    """No hand-off mapping: the release-phase block is left out, the rest is scored; scores save as plain JSON."""
    run = _spiral_run(n=300, halo=0.1)
    run.handoff_ids = None
    sc = M.score_run(run, _cfg())
    assert 'W_eff' not in sc and 'slope' not in sc and np.isfinite(sc['valley_all']) and np.isfinite(sc['ratio'])
    R.save_score(sc, tmp_path / 'score.json')
    back = R.load_score(tmp_path / 'score.json')
    assert back['valley_all'] == sc['valley_all'] and back['turns']['3']['sig_r'] == sc['sig_r_t3']
    R.save_table([back, {'tag': 'empty'}], tmp_path / 'table.csv', ['tag', 'I_final', 'turns.3.ratio', 'best_slit'])
    rows = (tmp_path / 'table.csv').read_text().splitlines()
    assert rows[0] == 'tag,I_final,turns.3.ratio,best_slit' and rows[2] == 'empty,,,'


def test_track_folder_round_trip(tmp_path):
    """A run folder in the tracker's layout (track.json, track_data.pkl, probe files) loads back into the same run:
    the folder names it, the probe files of this run alone are found, and it scores like the run it was written from."""
    run = _spiral_run(n=300, halo=0.1)
    run.handoff_ids = None
    tracks = {key: getattr(run, key) for key in ('ids', 'steps', 'xyz', 'stop_step', 'birth_step', 'n_steps',
                                                 'steps_per_turn', 'E_mev')}
    final = {'phase': 0.0, 'n': run.n_tracks, 'transmission': run.info['final']['transmission'], 'tracks': tracks}
    d = tmp_path / 'design_a'
    d.mkdir()
    with open(d / R.TRACK_PKL, 'wb') as fh:
        pickle.dump({'final': final, 'ref': {0.0: {'traj': run.ref_xyz}}}, fh)
    (d / R.TRACK_JSON).write_text(json.dumps(run.info))
    for name in ('probe_45deg.npz', 'probe_-90deg.npz', 'probe_45deg_old.npz'):
        (d / name).write_bytes(b'')
    back = R.load_run(d)
    assert back.tag == 'design_a' and back.handoff_ids is None and back.probe_files.keys() == {45.0, -90.0}
    assert any('hand-off file unknown' in n for n in back.notes)
    np.testing.assert_array_equal(back.xyz, run.xyz)
    np.testing.assert_array_equal(back.ref_xyz, run.ref_xyz)
    same = (lambda s: R.clean({k: v for k, v in s.items() if k not in ('tag', 'seconds', 'notes')}))
    assert same(M.score_run(back, _cfg())) == same(M.score_run(run, _cfg()))


def test_virtual_probe_round_trip(tmp_path):
    """A probe file built from the tracks (tracker layout) reads back into the same turn structure as the score's
    virtual probe, and matches itself crossing by crossing."""
    pytest.importorskip('PyPATools')                    # virtual_probe takes the kinetic energy from simulate3d
    run = _spiral_run(n=300, halo=0.1)
    run.handoff_meta = {'species_mass_mev': 1876.124}
    sc = M.score_run(run, _cfg())
    vp = R.virtual_probe(run, 45.0, _cfg())
    assert np.all(vp['E_mev'] > 0) and np.all(vp['direction'] == 1)    # the synthetic bunch never turns back
    rec = dict(vp, id=run.ids[vp['id']])                # the tracker's files index bunch particles
    cmp = M.compare_probes(vp, rec)
    assert cmp['n_matched'] == cmp['n_virtual'] > 0 and cmp['dR_max'] == 0.0
    np.savez(tmp_path / 'probe_45deg.npz', **rec)
    rt = M.recorded_probe_turns(R.load_probe(tmp_path / 'probe_45deg.npz'), _cfg())
    assert rt['R'] == sc['probes']['45']['R'] and rt['valley'] == sc['probes']['45']['valley']
    assert R._probe_files(tmp_path, R.TRACK_PROBE) == {45.0: str(tmp_path / 'probe_45deg.npz')}


def test_register_metric_is_scored():
    O.register_metric('_test_twice_T', lambda run, score, cfg: 2 * score['T'], 'max', 'test only')
    try:
        sc = M.score_run(_spiral_run(n=300), _cfg())
        assert sc['_test_twice_T'] == pytest.approx(2 * sc['T'])
    finally:
        O.METRICS.pop('_test_twice_T')


def test_vortex_twin_recovers_a_known_rotation():
    """Guiding centres rotating by -10 deg per turn in the SC-on twin only: the SC part of every map is -10 deg (the
    radius-time shear that the windows add is common to both twins and cancels)."""
    on = _spiral_run(n=400, rotate_deg=-10.0, offsets=False, phase=False, tag='on')
    off = _spiral_run(n=400, offsets=False, phase=False, tag='off')
    tw = TW.vortex_twin(on, off, cfg=_cfg(twin_boot=5))
    maps = [r for r in tw['turns'] if 'sc' in r]
    assert len(maps) >= 3 and tw['n_both'] == pytest.approx(on.info['final']['transmission'] * 400)
    for r in maps:
        assert r['sc']['rot'] == pytest.approx(-10.0, abs=0.3)
        assert r['off']['rot'] == pytest.approx(0.0, abs=1.0)
    assert tw['rot_sc_total'] == pytest.approx(-10.0 * len(maps), abs=1.0)
    assert np.isfinite(tw['rot_sc_total_err'])


def test_kernel_weights_sum_to_one():
    w = TW.kernel_weights(0.7, 1.7, range(-1, 4))
    assert sum(w.values()) == pytest.approx(1.0)
    assert max(w, key=w.get) == 1
