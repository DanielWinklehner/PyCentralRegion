"""Two things an evaluation with posts by rule repeats, done once: the post rule's crossings of a polyline by the
tracks share the tracks' segment boxes (and test each of its segments against the nearby track segments only), and
the layout's distance from the housing outline is remembered by its arguments. Both must give what the plain
computation gives, to the last bit."""
import numpy as np
import pytest

PR = pytest.importorskip('PyCentralRegion.geometry.post_rule')
B = pytest.importorskip('PyCentralRegion.geometry.builder')


def _reference_crossings(edge, xyz, live):
    """Every live track segment against every segment of the polyline, one pair at a time (the formulas of
    ``post_rule.crossings``), in its order: polyline segment, then record, then particle."""
    seg = np.diff(edge, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    sl = np.concatenate([[0.0], np.cumsum(seg_len)])
    out = []
    for m in range(len(seg)):
        for k in range(len(xyz) - 1):
            for n in range(xyz.shape[1]):
                if not (live[k, n] and live[k + 1, n]):
                    continue
                a, r = xyz[k, n, :2], xyz[k + 1, n, :2] - xyz[k, n, :2]
                den = r[0] * seg[m, 1] - r[1] * seg[m, 0]
                if not abs(den) > 1e-12:
                    continue
                d = edge[m] - a
                t = (d[0] * seg[m, 1] - d[1] * seg[m, 0]) / den
                u = (d[0] * r[1] - d[1] * r[0]) / den
                if 0 <= t <= 1 and 0 <= u <= 1:
                    out.append((n, k, a[0] + t * r[0], a[1] + t * r[1], sl[m] + u * seg_len[m]))
    return np.array(out).reshape(-1, 5)


def _tracks(seed=3, n=40, records=90):
    """Outward spirals of n particles (mm), a few of them not yet born, lost or outside for part of the run."""
    rng = np.random.default_rng(seed)
    th = np.linspace(0.0, 5.0 * np.pi, records)[:, None] + rng.uniform(0.0, 2.0 * np.pi, n)[None, :]
    r = 40.0 + 9.0 * th + rng.normal(0.0, 3.0, n)[None, :]
    xyz = np.stack([r * np.cos(th), r * np.sin(th), rng.normal(0.0, 2.0, (records, n))], axis=2)
    live = np.ones((records, n), dtype=bool)
    live[:rng.integers(1, 20), ::7] = False
    live[rng.integers(40, 80):, 3::9] = False
    return xyz, live


def test_crossings_with_shared_boxes_are_the_plain_ones():
    xyz, live = _tracks()
    boxes = PR.segment_boxes(xyz, live)
    assert boxes[0].shape == (len(xyz) - 1, xyz.shape[1]) and np.array_equal(boxes[0], live[:-1] & live[1:])
    assert np.array_equal(boxes[1], np.minimum(xyz[:-1, :, 0], xyz[1:, :, 0]))
    assert np.array_equal(boxes[4], np.maximum(xyz[:-1, :, 1], xyz[1:, :, 1]))
    edges = [np.array([[10.0, 5.0], [60.0, 20.0], [140.0, 95.0], [230.0, 120.0]]),          # three segments, two bends
             np.array([[-20.0, -30.0], [-150.0, -160.0]]),                                    # one segment
             np.array([[0.0, 50.0], [0.0, 120.0], [-40.0, 200.0], [-40.0, 260.0], [15.0, 300.0]]),   # an axis-parallel one
             np.array([[500.0, 500.0], [600.0, 620.0]])]                                      # never crossed
    total = 0
    for edge in edges:
        ref = _reference_crossings(edge, xyz, live)
        for got in (PR.crossings(edge, xyz, live), PR.crossings(edge, xyz, live, boxes)):
            who, rec, pts, arc = got
            assert len(who) == len(ref)
            np.testing.assert_array_equal(who, ref[:, 0].astype(int))
            np.testing.assert_array_equal(rec, ref[:, 1].astype(int))
            np.testing.assert_array_equal(pts, ref[:, 2:4])
            np.testing.assert_array_equal(arc, ref[:, 4])
        total += len(ref)
    assert total > 100 and len(_reference_crossings(edges[3], xyz, live)) == 0
    # a track segment ending exactly on the polyline, and one along it (parallel: no crossing), are told as before
    xyz2 = np.array([[[0.0, -1.0, 0.0], [5.0, 5.0, 0.0]], [[0.0, 0.0, 0.0], [9.0, 5.0, 0.0]],
                     [[0.0, 1.0, 0.0], [12.0, 5.0, 0.0]]])
    live2 = np.ones((3, 2), dtype=bool)
    flat = np.array([[-10.0, 0.0], [10.0, 0.0], [20.0, 5.0]])
    ref = _reference_crossings(flat, xyz2, live2)
    who, rec, pts, arc = PR.crossings(flat, xyz2, live2, PR.segment_boxes(xyz2, live2))
    assert len(ref) == 2 and np.array_equal(who, ref[:, 0].astype(int)) and np.array_equal(rec, ref[:, 1].astype(int))
    np.testing.assert_array_equal(arc, ref[:, 4])


def test_wall_distance_is_remembered_not_changed():
    """A repeated call gets the stored answer, as a fresh array each time; other points or another outline are
    computed; the values are the plain computation's (signed: inside the outline negative)."""
    ang = np.linspace(0.0, 2.0 * np.pi, 41)[:-1]
    wall = np.column_stack([30.0 * np.cos(ang), 20.0 * np.sin(ang)])
    rng = np.random.default_rng(5)
    pts = rng.uniform(-45.0, 45.0, (60, 2))
    B._WALL_DISTANCE_MEMO.clear()
    first = B._wall_distance(pts, wall)
    assert len(B._WALL_DISTANCE_MEMO) == 1
    seg = np.roll(wall, -1, axis=0) - wall
    plain = np.empty(len(pts))
    for i, p in enumerate(pts):
        t = np.clip(((p - wall) * seg).sum(axis=1) / (seg * seg).sum(axis=1), 0.0, 1.0)
        plain[i] = np.hypot(*(wall + t[:, None] * seg - p).T).min()
    inside = (pts[:, 0] / 30.0) ** 2 + (pts[:, 1] / 20.0) ** 2 < 0.97
    outside = (pts[:, 0] / 30.0) ** 2 + (pts[:, 1] / 20.0) ** 2 > 1.03
    assert np.allclose(np.abs(first), plain, rtol=1e-13, atol=0.0)
    assert np.all(first[inside] < 0.0) and np.all(first[outside] > 0.0) and inside.sum() > 5 and outside.sum() > 5
    again = B._wall_distance(pts, wall)
    assert np.array_equal(again, first) and again is not first and len(B._WALL_DISTANCE_MEMO) == 1
    again[:] = 0.0                                                # a caller's change does not reach the stored answer
    assert np.array_equal(B._wall_distance(pts.tolist(), wall), first)
    assert not np.array_equal(B._wall_distance(pts + 1.0, wall), first) and len(B._WALL_DISTANCE_MEMO) == 2
    assert not np.array_equal(B._wall_distance(pts, 1.1 * wall), first) and len(B._WALL_DISTANCE_MEMO) == 3
    one = B._wall_distance(pts[0], wall)                          # a single point, as most calls are
    assert one.shape == (1,) and one[0] == first[0]
    limit = B._WALL_DISTANCE_MEMO_MAX
    try:                                                          # the store is emptied when it is full, not grown
        B._WALL_DISTANCE_MEMO_MAX = 4
        for k in range(6):
            B._wall_distance(pts + 10.0 + k, wall)
        assert len(B._WALL_DISTANCE_MEMO) <= 4
    finally:
        B._WALL_DISTANCE_MEMO_MAX = limit
        B._WALL_DISTANCE_MEMO.clear()
