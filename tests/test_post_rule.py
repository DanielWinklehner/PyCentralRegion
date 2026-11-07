"""The post rule (geometry.post_rule) on synthetic tracks: counterclockwise spirals whose passes cross a straight edge
along the +x axis in bands 50 mm apart, so where the bands, the rows between them and the inner post must come out is
known without the tracker."""
import pickle

import numpy as np
import pytest

from PyCentralRegion.config import ConfigError
from PyCentralRegion.geometry.options import PostRuleConfig

PR = pytest.importorskip('PyCentralRegion.geometry.post_rule')

EDGE = np.array([[20.0, 0.0], [300.0, 0.0]])                       # s = x - 20 mm
POLY = np.array([[20.0, 0.0], [300.0, 0.0], [300.0, -60.0], [20.0, -60.0]])   # its electrode: below the edge
SIZE = 10.0                                                          # post_size_mm


def _spiral(n=400, end_deg=1500.0, dr=0.0, sense=1.0, seed=3):
    """n particles on spirals r = 80 + delta + 50 (phi - 360) / 360 mm (phi the unwrapped azimuth [deg], delta uniform
    in [-5, 5] mm, + dr), one record per 2 deg from a start in [10, 60] deg to end_deg: pass p crosses the +x axis at
    r = 30 + 50 p + delta, i.e. edge arclength s = 10 + 50 p + delta. sense -1 mirrors them (a clockwise beam).
    Returns xyz (S, n, 3) [mm] and live (S, n) (False past end_deg)."""
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-5.0, 5.0, n) + dr
    phi = rng.uniform(10.0, 60.0, n)[None, :] + 2.0 * np.arange(int(end_deg / 2.0))[:, None]
    r = 80.0 + delta[None, :] + 50.0 * (phi - 360.0) / 360.0
    th = np.radians(phi)
    xyz = np.stack([r * np.cos(th), sense * r * np.sin(th), np.zeros_like(r)], axis=2)
    return xyz, phi <= end_deg


def _place(rule=None, edges=None, **kw):
    xyz, live = _spiral(**kw)
    edges = edges or {(1, 'dee'): (EDGE, POLY, 'dee 0'), (2, 'dee'): (EDGE, POLY, 'dee 1'),
                      (6, 'dee'): (EDGE, POLY, 'dee 3')}
    return PR.place_posts(rule or PostRuleConfig(), edges, xyz, live, np.ones(xyz.shape[1], bool), SIZE)


def test_bands_rows_and_the_inner_post():
    posts, records = _place()
    bands = records[0]['bands']
    assert sorted(bands) == [1, 2, 3, 4] and all(b['share'] == 2.0 for b in bands.values())   # the edge and its offset
    for p, b in bands.items():          # the edge's crossings at s 10 + 50 p +- 5, its offset's up to 1.6 mm lower
        assert 3.0 + 50 * p < b['e_lo'] < 5.5 + 50 * p and 14.5 + 50 * p < b['e_hi'] < 15.0 + 50 * p
        assert b['r_e_lo'] == pytest.approx(b['e_lo'] + 20.0)
    assert [(p['gap'], p['between_bands']) for p in posts] == [(1, [0, 1]), (1, [1, 2]), (1, [2, 3]), (2, [1, 2]),
                                                               (2, [2, 3])]                 # gap 6 has none
    inner, row1, row2 = posts[:3]
    hi = bands[1]['e_lo'] - 5.0                                          # inner_clear_mm inside the first band
    assert inner['s_mm'] == pytest.approx(hi - 10.0, abs=1e-3) and inner['length_mm'] == pytest.approx(20.0, abs=1e-3)
    for post, (a, b) in ((row1, (1, 2)), (row2, (2, 3))):                # clear_mm to each band's edge
        lo_s, hi_s = bands[a]['e_hi'] + 5.0, bands[b]['e_lo'] - 5.0
        assert post['s_mm'] == pytest.approx(0.5 * (lo_s + hi_s), abs=1e-3)
        assert post['length_mm'] == pytest.approx(hi_s - lo_s, abs=1e-3) and 28.0 < post['length_mm'] < 31.0
        assert post['r_mm'] == pytest.approx(post['s_mm'] + 20.0, abs=1e-3) and post['side'] == 'dee'
    assert posts[3:] == [dict(p, gap=2) for p in posts[1:3]] and all(not r['skipped'] for r in records)


def test_what_the_rule_leaves_out():
    # tracks that end before pass 3: no band 3, no row 2
    posts, records = _place(PostRuleConfig(gaps=(1,), inner_gaps=()), end_deg=1000.0)
    assert [p['between_bands'] for p in posts] == [[1, 2]]
    assert records[0]['skipped'] == ['row 2: only passes [1, 2] have >= 0.2']
    # an inner post kept above a radius beyond the first band
    posts, records = _place(PostRuleConfig(gaps=(), inner_gaps=(1,), inner_rmin_mm=75.0))
    assert posts == [] and records[0]['skipped'][0].startswith('inner: ') and 'free after the clearances' in \
        records[0]['skipped'][0]
    # an electrode on neither side of its edge, and no survivors
    far = POLY + [0.0, -500.0]
    posts, records = _place(edges={(1, 'dee'): (EDGE, far, 'dee 0'), (2, 'dee'): (EDGE, POLY, 'dee 1')})
    assert 'neither side of the edge is inside dee 0' in records[0]['skipped'][0] and records[0]['bands'] == {}
    assert [p['gap'] for p in posts] == [2, 2]
    xyz, live = _spiral()
    posts, records = PR.place_posts(PostRuleConfig(), {(1, 'dee'): (EDGE, POLY, 'dee 0')}, xyz, live,
                                    np.zeros(xyz.shape[1], bool), SIZE)
    assert posts == [] and records[0]['skipped'] == ['no survivors']


def test_only_survivors_count():
    """Lost particles (here 3 mm further out on every pass) move no band edge."""
    xyz, live = _spiral(n=400)
    lost, lost_live = _spiral(n=200, dr=3.0, seed=4)
    rule, edges = PostRuleConfig(), {(1, 'dee'): (EDGE, POLY, 'dee 0')}
    alone = PR.place_posts(rule, edges, xyz, live, np.ones(400, bool), SIZE)
    mixed = PR.place_posts(rule, edges, np.concatenate([xyz, lost], axis=1), np.concatenate([live, lost_live], axis=1),
                           np.r_[np.ones(400, bool), np.zeros(200, bool)], SIZE)
    assert mixed == alone


def test_a_clockwise_beam_is_refused():
    with pytest.raises(NotImplementedError, match='clockwise'):
        _place(sense=-1.0, edges={(1, 'dee'): (EDGE, POLY * [1.0, -1.0], 'dee 0')})


def test_split_at_bends_and_s_at_radius():
    bent = np.array([[0.0, 0.0], [100.0, 0.0], [100.0 + 100.0 * np.cos(np.radians(30.0)), 50.0]])
    ext = 0.5 * SIZE * (1.0 - np.tan(np.radians(15.0))) - 0.25          # each piece runs on past the bend by this
    assert PR.split_at_bends(bent, 20.0, 60.0, SIZE) == [(20.0, 60.0)]
    assert PR.split_at_bends(bent, 80.0, 130.0, SIZE) == [pytest.approx((80.0, 100.0 + ext)),
                                                           pytest.approx((100.0 - ext, 130.0))]
    assert PR.split_at_bends(bent, 80.0, 105.0, SIZE) == [pytest.approx((80.0, 100.0 + ext))]    # 8.4 mm: dropped
    # square ends stop at the bend: past it a square corner would stick out of a convex corner
    assert PR.split_at_bends(bent, 80.0, 130.0, SIZE, round_ends=False) == [(80.0, 100.0), (100.0, 130.0)]
    line = np.array([[30.0, 0.0], [100.0, 0.0]])
    assert PR.s_at_radius(line, 45.0) == pytest.approx(15.0, abs=0.05)
    assert PR.s_at_radius(line, 20.0) == 0.0 and PR.s_at_radius(line, 120.0) == pytest.approx(70.0)


def test_load_tracks(tmp_path):
    """live from birth to the metal hit, finite, inside the loss limits ever after; survived: live at the end."""
    n_steps, steps = 100, np.arange(0, 100, 10)
    xyz = np.zeros((len(steps), 5, 3))
    xyz[..., 0] = 0.1                                          # 100 mm
    xyz[6:, 2, 0] = 0.5                                        # out beyond r_max at step 60, and stays out
    xyz[4, 3, 1] = np.nan                                      # one bad record
    stop = np.array([n_steps + 1, 50, n_steps + 1, n_steps + 1, n_steps + 1])     # particle 1 hits metal at step 50
    birth = np.array([0, 0, 0, 0, 30])
    tracks = {'steps': steps, 'xyz': xyz, 'stop_step': stop, 'birth_step': birth, 'n_steps': n_steps}
    with open(tmp_path / PR.TRACK_DATA, 'wb') as fh:
        pickle.dump({'final': {'tracks': tracks}}, fh)
    got, live, survived = PR.load_tracks(tmp_path / PR.TRACK_DATA, 380.0, 30.0)
    assert got[0, 0, 0] == pytest.approx(100.0)
    assert list(live.sum(axis=0)) == [10, 5, 6, 9, 7] and list(survived) == [True, False, False, True, True]


def test_rule_options():
    assert PostRuleConfig().gaps == (1, 2, 3, 4, 5) and PostRuleConfig().inner_gaps == (1,)
    for bad in (dict(gaps=(0, 1)), dict(inner_gaps=(0,))):
        with pytest.raises(ConfigError, match='gap 0 gets no posts'):
            PostRuleConfig(**bad)
    with pytest.raises(ConfigError, match='at least one phase'):
        PostRuleConfig(scan_phases_deg=())
    for bad in (dict(lo_q=0.5), dict(min_share=0.0), dict(clear_mm=-1.0), dict(inner_len_mm=0.0),
                dict(scan_magnet_scale=0.0)):
        with pytest.raises(ConfigError, match='0 <= lo_q < 0.5'):
            PostRuleConfig(**bad)
