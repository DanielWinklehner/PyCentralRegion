"""metal_columns and the plane sections. The column test puts every face at its own height (a sloped one at the node's
x and y), breaks ties on edges and vertices by one rule so that every column of a closed solid has an even number of
crossings, keeps a hollow solid's cavity, merges overlapping solids and refuses a solid that is not closed (the fields
stage's mask leaves it out where it is open); the compiled kernel carries it beside the slit's raster and stops what
the Python path stops, and load_fields builds the test a run asks for. ``loops_from_mesh`` gives closed outlines
whatever lies in or near the plane (the ground's section that a merge grid dropped is reproduced), cuts a face at its
own height alike whatever the rounding of its vertices, and gives none where a solid only touches the plane.
Synthetic solids only."""
import dataclasses
import pickle
import types
import warnings

import numpy as np
import pytest

from PyCentralRegion import metal_columns as M

MM = 1e-3
XS = M.column_grid(10 * MM, 1 * MM)              # nodes every millimetre over |x|, |y| <= 10 mm


def _box(x0, x1, y0, y1, z0, z1, top=None):
    """Vertices and outward triangles of a box [m]; ``top(x, y)`` gives the heights of the top corners instead (a
    sloped top, planar when it is linear). Both horizontal faces are split along the diagonal (x0, y0) - (x1, y1)."""
    v = np.array([[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], dtype=float)
    if top is not None:
        v[4:, 2] = [top(x, y) for x, y in v[4:, :2]]
    quads = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5))
    return v, np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], dtype=np.int64)


def _solids(*shells):
    """One vertex array and one triangle array for several shells, and the triangle range of each."""
    v, f, ranges, n_v, n_f = [], [], [], 0, 0
    for sv, sf in shells:
        v.append(sv)
        f.append(sf + n_v)
        ranges.append((n_f, n_f + len(sf)))
        n_v, n_f = n_v + len(sv), n_f + len(sf)
    return np.vstack(v), np.vstack(f), ranges


def _columns(v, f, ranges, xs=XS):
    starts, lo, hi, info = M.build_columns(v, f, ranges, xs)
    return M.ColumnObstacles(starts, lo, hi, xs, info=info)


def test_faces_act_at_their_own_height():
    """Faces between the levels a slice raster would test act where they are, and on their surface (closed
    intervals); x and y are good to half a node spacing, off the grid is vacuum, (N, 2) points are at z = 0."""
    v, f = _box(-3.5 * MM, 4.5 * MM, -2.5 * MM, 5.5 * MM, -13.0 * MM, 16.3 * MM)
    cols = _columns(v, f, [(0, len(f))])
    np.testing.assert_array_equal(cols.column(0.0, 0.0), [[-13.0 * MM, 16.3 * MM]])
    assert cols.column(5 * MM, 0.0).shape == (0, 2)
    z = [-13.0 * MM, 16.3 * MM, 0.0, 16.3 * MM + 1e-12, -13.0 * MM - 1e-12]
    assert cols(np.array([[0.0, 0.0, zk] for zk in z])).tolist() == [True, True, True, False, False]
    assert cols(np.array([[4.4 * MM, 0.0, 0.0], [4.6 * MM, 0.0, 0.0]])).tolist() == [True, False]
    assert cols(np.array([[0.0, 0.0], [6 * MM, 0.0]])).tolist() == [True, False]
    assert not cols(np.array([[20 * MM, 0.0, 0.0]]))[0]
    assert cols.info['n_intervals'] == int(np.diff(cols.starts).sum()) and cols.info['max_per_column'] == 1
    assert cols.info['odd_columns'] == {}
    v, f = _box(6 * MM, 12 * MM, -12 * MM, -6 * MM, -1 * MM, 1 * MM)      # metal out to the grid's corner (10, -10) mm
    edge = _columns(v, f, [(0, len(f))])
    pts = [[10.4 * MM, -10.4 * MM, 0.0], [10.6 * MM, -9 * MM, 0.0], [9 * MM, -10.6 * MM, 0.0]]
    assert edge(np.array(pts)).tolist() == [True, False, False]          # off the grid, not the edge column's


def test_a_sloped_face_at_the_nodes_x_and_y():
    def top(x, y):
        return 13 * MM + 0.25 * x - 0.1 * y
    v, f = _box(-6 * MM, 6 * MM, -6 * MM, 6 * MM, -2 * MM, 0.0, top=top)
    cols = _columns(v, f, [(0, len(f))])
    for x, y in ((0.0, 0.0), (4 * MM, -3 * MM), (-6 * MM, 5 * MM), (2 * MM, 2 * MM)):
        np.testing.assert_allclose(cols.column(x, y), [[-2 * MM, top(x, y)]], rtol=0.0, atol=1e-15)


def test_ties_on_edges_and_vertices():
    """Box edges, corners and both faces' diagonals on nodes: no column has an odd number of crossings, and a node on
    the outline is inside exactly when the point just beside it, (x + e, y + e^2), is: x0 <= x < x1, y0 <= y < y1."""
    v, f = _box(-3 * MM, 3 * MM, -3 * MM, 3 * MM, -1 * MM, 2 * MM)
    cols = _columns(v, f, [(0, len(f))])
    k = np.rint(XS / MM).astype(int)
    for i, x in enumerate(k):
        for j, y in enumerate(k):
            got = cols.column(XS[i], XS[j]).tolist()
            assert got == ([[-1 * MM, 2 * MM]] if (-3 <= x < 3 and -3 <= y < 3) else []), (x, y)


def test_a_vertex_on_a_node():
    """A pyramid whose apex lies over a node, where the base's diagonal and the edges between its sides meet as well:
    one crossing of the sides and one of the base in every column, up to the apex."""
    h = 7 * MM
    v = np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0], [0, 0, 0]], dtype=float) * MM
    v[4, 2] = h
    f = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    cols = _columns(v, f, [(0, len(f))])
    for (x, y), top in (((0, 0), h), ((1, 0), h / 2), ((1, 1), h / 2), ((-1, -1), h / 2), ((0, -1), h / 2)):
        np.testing.assert_allclose(cols.column(x * MM, y * MM), [[0.0, top]], rtol=0.0, atol=1e-15)


def test_a_hollow_solid_keeps_its_cavity():
    shells = _solids(_box(-6 * MM, 6 * MM, -6 * MM, 6 * MM, -5 * MM, 5 * MM),
                     _box(-2 * MM, 2 * MM, -2 * MM, 2 * MM, -1 * MM, 1 * MM))
    v, f, _ = shells
    cols = _columns(v, f, [(0, len(f))])                       # one solid of two shells
    np.testing.assert_array_equal(cols.column(0.0, 0.0), [[-5 * MM, -1 * MM], [1 * MM, 5 * MM]])
    np.testing.assert_array_equal(cols.column(4 * MM, 0.0), [[-5 * MM, 5 * MM]])
    assert cols(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3 * MM]])).tolist() == [False, True]


def test_solids_are_merged():
    """Separate solids may overlap (an un-fused housing) or touch: their intervals are merged."""
    v, f, ranges = _solids(_box(-4 * MM, 4 * MM, -4 * MM, 4 * MM, 0.0, 3 * MM),
                           _box(-2 * MM, 6 * MM, -4 * MM, 4 * MM, 2 * MM, 5 * MM),
                           _box(-4 * MM, 4 * MM, -4 * MM, 4 * MM, -3 * MM, 0.0))
    cols = _columns(v, f, ranges)
    np.testing.assert_array_equal(cols.column(0.0, 0.0), [[-3 * MM, 5 * MM]])
    np.testing.assert_array_equal(cols.column(-3 * MM, 0.0), [[-3 * MM, 3 * MM]])
    np.testing.assert_array_equal(cols.column(5 * MM, 0.0), [[2 * MM, 5 * MM]])


def test_a_solid_that_is_not_closed_is_refused():
    """A box without the bottom triangle (x0, y0), (x0, y1), (x1, y1) is refused; with odd = 'drop' (the fields
    stage's mask) it is left out of the 15 columns under the hole (y > x) and kept in the others, the diagonal's
    included (its tie goes to the triangle that is there)."""
    v, f = _box(-3 * MM, 3 * MM, -3 * MM, 3 * MM, -1 * MM, 1 * MM)
    with pytest.raises(ValueError, match=r'cross lid an odd number of times in \d+ columns: the solid is not closed'):
        M.build_columns(v, f[1:], [(0, len(f) - 1)], XS, names=['lid'])
    starts, lo, hi, info = M.build_columns(v, f[1:], [(0, len(f) - 1)], XS, names=['lid'], odd='drop')
    cols = M.ColumnObstacles(starts, lo, hi, XS, info=info)
    assert info['odd_columns'] == {'lid': 15}
    k = np.rint(XS / MM).astype(int)
    for i, x in enumerate(k):
        for j, y in enumerate(k):
            want = [[-1 * MM, 1 * MM]] if (-3 <= x < 3 and -3 <= y < 3 and y <= x) else []
            assert cols.column(XS[i], XS[j]).tolist() == want, (x, y)
    with pytest.raises(ValueError, match="odd = 'keep'"):
        M.build_columns(v, f, [(0, len(f))], XS, odd='keep')


def test_full_height_columns():
    """Posts and boxes make their columns metal at every height; every other column keeps its intervals, also behind a
    column that had two (its intervals move up by one)."""
    v, f = _box(-3 * MM, 3 * MM, -3 * MM, 3 * MM, -1 * MM, 1 * MM)
    cols = _columns(v, f, [(0, len(f))])
    mask = np.zeros((len(XS), len(XS)), dtype=bool)
    mask[np.flatnonzero(np.abs(XS - 5 * MM) < 1e-9), :] = True     # the row x = 5 mm, vacuum before
    mask[np.flatnonzero(np.abs(XS) < 1e-9), np.flatnonzero(np.abs(XS) < 1e-9)] = True   # the box's centre column
    full = cols.with_full_height(mask)
    pts = [[5 * MM, -7 * MM, 0.04], [0.0, 0.0, -0.5], [1 * MM, 1 * MM, 0.5 * MM], [1 * MM, 1 * MM, 2.5 * MM]]
    assert full(np.array(pts)).tolist() == [True, True, True, False]
    for x, y in ((1 * MM, 1 * MM), (-3 * MM, 2 * MM), (7 * MM, 0.0)):
        np.testing.assert_array_equal(full.column(x, y), cols.column(x, y))
    with pytest.raises(ValueError, match='mask'):
        cols.with_full_height(mask[:-1])
    v, f, _ = _solids(_box(-6 * MM, 6 * MM, -6 * MM, 6 * MM, -5 * MM, 5 * MM),
                      _box(-2 * MM, 2 * MM, -2 * MM, 2 * MM, -1 * MM, 1 * MM))
    hollow = _columns(v, f, [(0, len(f))])                        # two intervals in the columns over the cavity
    post = np.zeros((len(XS), len(XS)), dtype=bool)
    post[np.flatnonzero(np.abs(XS) < 1e-9), np.flatnonzero(np.abs(XS) < 1e-9)] = True
    full = hollow.with_full_height(post)
    np.testing.assert_array_equal(full.column(0.0, 0.0), [[-np.inf, np.inf]])
    for i, x in enumerate(XS):
        for j, y in enumerate(XS):
            if not post[i, j]:
                np.testing.assert_array_equal(full.column(x, y), hollow.column(x, y))


def test_the_kernel_carries_the_columns_beside_the_slit():
    """With the slit's raster, the column test is split off for the compiled 3D kernel's column slot (the very tuple
    it tests), the slit for its one raster."""
    pytest.importorskip('PyPATools')
    from PyCentralRegion import config as C
    from PyCentralRegion import simulate3d as S
    from PyCentralRegion.electrodes3d import combine_obstacles
    from PyCentralRegion.fast_track3d import _column_parts, _raster_parts
    v, f = _box(-3.5 * MM, 4.5 * MM, -2.5 * MM, 5.5 * MM, -1.3 * MM, 1.6 * MM)
    cols = _columns(v, f, [(0, len(f))])
    slit, _ = S.slit_raster(C.SlitConfig(half_width_mm=3.0, azimuth_deg=315.0, jaw_length_mm=6.0, thickness_mm=2.0),
                            0.150, 0.03)
    both = combine_obstacles(cols, slit)
    parts, rasters = _column_parts(both), _raster_parts(both)
    assert len(parts) == 1 and parts[0] is cols.packed
    assert len(rasters) == 1 and rasters[0][5:] == pytest.approx((S.SLIT_H_M, S.SLIT_H_M, 0.03))
    assert _raster_parts(cols) == [] and _column_parts(slit) == []


def test_the_kernel_stops_what_the_python_path_stops():
    """Straight flights (no fields) through the columns of a box with a sloped top and of a hollow box, and through
    the slit's raster: the compiled 3D kernel stops the same particles at the same steps and points as the Python path
    (``tracking.MetalTerminator`` over the same test), testing every step and every third step; its compiled copy of
    ``in_columns`` is the current one (the selftest)."""
    pytest.importorskip('PyPATools')
    from PyPATools.field import Field
    from PyCentralRegion import config as C
    from PyCentralRegion import fast_track3d as F
    from PyCentralRegion import simulate3d as S
    from PyCentralRegion.accelerated_orbit_finder import make_beam_from_state
    from PyCentralRegion.central_region import CentralRegion
    from PyCentralRegion.electrodes3d import combine_obstacles
    from PyCentralRegion.tracking import MetalTerminator, TrackingEngine
    if not F.HAS_NUMBA:
        pytest.skip('the compiled kernel needs numba')
    assert F._require_selftest()['columns_point_exact']

    def top(x, y):
        return 1 * MM + 0.2 * x - 0.1 * y
    v, f, ranges = _solids(_box(-8 * MM, -2 * MM, -6 * MM, 6 * MM, -2 * MM, 0.0, top=top),
                           _box(2 * MM, 8 * MM, -8 * MM, -1 * MM, -3 * MM, 3 * MM),
                           _box(4 * MM, 6 * MM, -6 * MM, -3 * MM, -1 * MM, 1 * MM))
    cols = _columns(v, f, [ranges[0], (ranges[1][0], ranges[2][1])])           # the second solid is hollow
    slit, _ = S.slit_raster(C.SlitConfig(half_width_mm=1.0, azimuth_deg=90.0, jaw_length_mm=3.0, thickness_mm=2.0),
                            5 * MM, 0.01)
    both = combine_obstacles(cols, slit)
    rng = np.random.default_rng(7)
    r0 = rng.uniform(-1.0, 1.0, (400, 3)) * [9 * MM, 9 * MM, 4 * MM]
    r0 = r0[~both(r0)][:64]
    u = rng.normal(size=(len(r0), 3)) * [1.0, 1.0, 0.3]
    v0 = 1e6 * u / np.linalg.norm(u, axis=1)[:, None]                         # 0.1 mm per step
    dt, n_steps = 1e-10, 150
    design = CentralRegion(name='columns', dimensionality='3D')
    design.set_species('proton')
    design.set_magnetic_field(Field.zero())
    design.set_electric_field(Field.zero())

    def run(fast, every):
        engine = TrackingEngine(design, algorithm='boris_rel', dimensionality='3D', use_rf=False, max_radius_m=0.5,
                                verbose=False, z_max=0.1)
        metal = MetalTerminator(both, every=every)
        engine.extra_terminators = [metal]
        pd = make_beam_from_state(design.species, r0, v0)
        if not fast:
            return engine.track_multiparticle(pd, dt=dt, n_steps=n_steps, callback=lambda *a: False,
                                              show_progress=False), metal
        finder = types.SimpleNamespace(engine=engine, fast3d_parallel=False, fast3d_threads=None)
        args = F.build_kernel_args3d(finder, pd, dt, n_steps, 0.0, np.zeros(len(r0), dtype=np.int64), None, 0,
                                     interp='batch')
        assert args['cols'] is cols.packed and args['ras_a'][0] == 1 and args['ras_b'][0] == 0
        res = F.run_fast3d(finder, args, lambda *a: False, dt, 0.0)
        assert finder.last_fast_path['used'] and not finder.last_fast_path['parallel']
        return res, metal

    for every in (1, 3):
        (py, m_py), (fa, m_fa) = run(False, every), run(True, every)
        assert 10 < len(m_fa.hits) < len(r0) - 10, every
        assert m_fa.hits == m_py.hits and m_fa.hits_z == m_py.hits_z, every
        np.testing.assert_array_equal(fa.active, py.active)
        np.testing.assert_array_equal(fa.r_final, py.r_final)


class _Model:
    """What the metal tests read of an electrode model."""

    def __init__(self, v, f, ranges, names):
        self.vertices, self.triangles = v, f
        self.params = {'dim': 3, 'solids': [{'name': n, 'tri_range': r} for n, r in zip(names, ranges)]}

    def extent(self):
        return 6 * MM


def test_load_fields_builds_the_runs_metal_test(tmp_path):
    """load_fields builds what track.obstacles asks for from the cached model: the columns see faces at +-2.6 mm
    there, the slices 1 mm apart at +-2.5 mm (the half level next to them); extra posts are metal at every height."""
    pytest.importorskip('PyPATools')
    pytest.importorskip('trimesh')
    from PyPATools.field import Field
    from PyCentralRegion import config as C
    from PyCentralRegion import simulate3d as S
    from PyCentralRegion.fields3d import field_to_arrays
    v, f = _box(-3 * MM, 3 * MM, -3 * MM, 3 * MM, -2.6 * MM, 2.6 * MM)
    g = np.linspace(-0.01, 0.01, 3)
    bem = Field.from_arrays(grid={'x': g, 'y': g, 'z': g}, values={k: np.zeros((3, 3, 3)) for k in 'xyz'}, dim=3,
                            units='m', interpolator_backend='numba', label='bem')
    path = tmp_path / 'fields.pkl'
    with open(path, 'wb') as fh:
        pickle.dump({'bem3d': field_to_arrays(bem), 'rf': None, 'stack': {'kind': 'columns'},
                     'model3d': _Model(v, f, [(0, len(f))], ['dee0']), 'info': {}}, fh)
    fc = C.FieldsConfig(stack_z_half_m=5 * MM, stack_dz_m=1 * MM, stack_dxy_m=0.5 * MM, margin_m=1 * MM)
    tc = C.TrackConfig()
    assert tc.obstacles == 'columns'
    pts = np.array([[0.0, 0.0, z * MM] for z in (2.45, 2.55, 2.65, -2.45, -2.55, -2.65)])
    d = S.load_fields(path, tc, fc)
    assert d['info']['metal_test']['kind'] == 'columns'
    assert d['stack'](pts).tolist() == [True, True, False, True, True, False]
    d = S.load_fields(path, dataclasses.replace(tc, obstacles='raster'), fc)
    assert d['info']['metal_test'] == {'kind': 'raster', 'cached': False, 'dxy_m': pytest.approx(0.5 * MM),
                                       'dz_m': pytest.approx(1 * MM)}
    assert d['stack'](pts).tolist() == [True, False, False, True, False, False]
    post = dataclasses.replace(tc, extra_posts_mm=((5.0, 5.0, 0.6),))
    for kind in ('columns', 'raster'):
        d = S.load_fields(path, dataclasses.replace(post, obstacles=kind), fc)
        assert d['stack'](np.array([[5 * MM, 5.5 * MM, 4 * MM], [5 * MM, 6 * MM, 0.0]])).tolist() == [True, False]
        assert d['info']['extra_posts'] == [[5.0, 5.0, 0.6]]
    with pytest.raises(C.ConfigError, match="'columns' or 'raster'"):
        S.load_fields(path, dataclasses.replace(post, obstacles='mesh'), fc)
    with pytest.raises(C.ConfigError, match="'columns', 'raster' or 'mesh'"):
        S.load_fields(path, dataclasses.replace(tc, obstacles='slices'), fc)
    opened = _Model(v, f[1:], [(0, len(f) - 1)], ['dee0'])                   # a hole in its bottom
    path = tmp_path / 'open.pkl'
    with open(path, 'wb') as fh:
        pickle.dump({'bem3d': field_to_arrays(bem), 'rf': None, 'stack': {'kind': 'columns'}, 'model3d': opened,
                     'info': {}}, fh)
    with pytest.raises(ValueError, match=r"cross dee0 an odd number of times in \d+ columns: the solid is not closed "
                                         r"\(use the stacked slices, track\.obstacles = 'raster'\)"):
        S.load_fields(path, tc, fc)
    d = S.load_fields(path, dataclasses.replace(tc, obstacles='raster'), fc)
    assert d['stack'](pts).tolist() == [True, False, False, True, False, False]
    assert S.column_obstacles(opened, fc, verbose=False, odd='drop').info['odd_columns']['dee0'] > 0     # the mask


def test_load_fields_tracks_on_a_caches_own_slices(tmp_path):
    """A cache that carries its slices (the design deck's, or one written before 2026-11-02) is tracked on them with
    track.obstacles = 'raster', not on slices cut again from its model; extra posts go in at every height, and the
    columns are built from the model whatever the cache carries."""
    pytest.importorskip('PyPATools')
    from PyPATools.field import Field
    from PyCentralRegion import config as C
    from PyCentralRegion import simulate3d as S
    from PyCentralRegion.fields3d import field_to_arrays
    v, f = _box(-3 * MM, 3 * MM, -3 * MM, 3 * MM, -2.6 * MM, 2.6 * MM)
    g = np.linspace(-0.01, 0.01, 3)
    bem = Field.from_arrays(grid={'x': g, 'y': g, 'z': g}, values={k: np.zeros((3, 3, 3)) for k in 'xyz'}, dim=3,
                            units='m', interpolator_backend='numba', label='bem')
    xs, zs = M.column_grid(6 * MM, 0.5 * MM), np.linspace(-5 * MM, 5 * MM, 11)
    grid = np.zeros((len(xs), len(xs), len(zs)), dtype=bool)
    i = int(np.flatnonzero(np.abs(xs - 2 * MM) < 1e-9)[0])
    grid[i, i, 5:] = True                                         # the cache's metal: one column at (2, 2) mm, z >= 0
    path = tmp_path / 'deck.pkl'
    with open(path, 'wb') as fh:
        pickle.dump({'bem3d': field_to_arrays(bem), 'rf': None,
                     'stack': {'kind': 'raster', 'grid': grid, 'xs': xs, 'zs': zs},
                     'model3d': _Model(v, f, [(0, len(f))], ['dee0']), 'info': {}}, fh)
    fc = C.FieldsConfig(stack_z_half_m=5 * MM, stack_dz_m=1 * MM, stack_dxy_m=0.5 * MM, margin_m=1 * MM)
    tc = dataclasses.replace(C.TrackConfig(), obstacles='raster')
    pts = np.array([[0.0, 0.0, 0.0], [2 * MM, 2 * MM, 3 * MM], [2 * MM, 2 * MM, -3 * MM]])
    d = S.load_fields(path, tc, fc)
    assert d['info']['metal_test'] == {'kind': 'raster', 'cached': True, 'dxy_m': pytest.approx(0.5 * MM),
                                       'dz_m': pytest.approx(1 * MM)}
    assert d['stack'](pts).tolist() == [False, True, False]
    d = S.load_fields(path, dataclasses.replace(tc, extra_posts_mm=((-2.0, -2.0, 0.3),)), fc)
    assert d['stack'](np.vstack([pts, [[-2 * MM, -2 * MM, -4 * MM]]])).tolist() == [False, True, False, True]
    d = S.load_fields(path, dataclasses.replace(tc, obstacles='columns'), fc)
    assert d['stack'](pts).tolist() == [True, False, False]


def _area(loop):
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_sections_are_closed_whatever_lies_near_the_plane():
    """A box whose side walls carry a ring of vertices 1.7e-8 m above the plane, cut there, on the ring, on its faces
    and outside it: one outline, through a vertex in the plane once (the four corners there, the eight edges of the
    walls elsewhere); a hollow box gives its outer and inner outlines."""
    pytest.importorskip('PyPATools')                       # electrodes3d imports the physics stack
    from PyCentralRegion.electrodes3d import loops_from_mesh
    zr = 15 * MM + 1.7e-8
    corners = np.array([[-5, -4], [6, -4], [6, 7], [-5, 7]], dtype=float) * MM
    v = np.vstack([np.column_stack([corners, np.full(4, z)]) for z in (0.0, zr, 30 * MM)])
    f = [[0, 2, 1], [0, 3, 2], [8, 9, 10], [8, 10, 11]]                     # bottom, top
    for k in range(4):
        n = (k + 1) % 4
        for lo, up in ((0, 4), (4, 8)):                                       # the walls below and above the ring
            f += [[lo + k, lo + n, up + n], [lo + k, up + n, up + k]]
    mesh = types.SimpleNamespace(vertices=v, faces=np.array(f))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for z in (15 * MM, zr, 15 * MM + 3.4e-8, 0.0, 22 * MM):
            loops = loops_from_mesh(mesh, z=z)
            assert len(loops) == 1 and _area(loops[0]) == pytest.approx(11 * 11 * MM * MM, rel=1e-12), z
            assert len(np.unique(loops[0], axis=0)) == len(loops[0]) == (4 if z in (zr, 0.0) else 8), z
        assert loops_from_mesh(mesh, z=30 * MM) == [] and loops_from_mesh(mesh, z=-1 * MM) == []
        v, f, _ = _solids(_box(-6 * MM, 6 * MM, -6 * MM, 6 * MM, -5 * MM, 5 * MM),
                          _box(-2 * MM, 2 * MM, -2 * MM, 2 * MM, -1 * MM, 1 * MM))
        loops = loops_from_mesh(types.SimpleNamespace(vertices=v, faces=f), z=0.0)
        assert sorted(_area(lp) for lp in loops) == pytest.approx([16 * MM * MM, 144 * MM * MM], rel=1e-12)


def test_a_vertex_just_below_the_plane_on_a_nearly_level_face():
    """The ground's section at z = +-15 mm, dropped until 2026-11-02: a tent whose top carries a vertex 17 nm below
    the plane on a gently sloped, slightly folded face, so that three of its edges meet the plane within a micrometre
    of each other, and a merge grid of 1e-6 m fused the outer two but not the middle one (the code of that day drops
    this section: 'open outline ... gap 3.59 mm'). One outline, through all three, with the section's area."""
    pytest.importorskip('PyPATools')                       # electrodes3d imports the physics stack
    from PyCentralRegion.electrodes3d import loops_from_mesh
    z, depth, radius = 15 * MM, 1.7e-8, 5 * MM
    angles = (-135, -45, 0, 45, 135, 180)
    dist = {-45: 0.6e-6, 0: 0.8e-6, 45: 0.6e-6}             # where the vertex's edges to the ring meet the plane [m]
    zv, n = z - depth, len(angles)
    a = np.deg2rad(angles)
    ring = np.column_stack([radius * np.cos(a), radius * np.sin(a),
                            [zv + depth * radius / dist[k] if k in dist else z - 0.2 * MM for k in angles]])
    v = np.vstack([[0.0, 0.0, zv], ring, np.column_stack([ring[:, :2], np.full(n, z - 10 * MM)]),
                   [0.0, 0.0, z - 10 * MM]])
    f = []
    for k in range(n):                                      # the top's fan, the walls, the bottom's fan
        m = (k + 1) % n
        f += [[0, 1 + k, 1 + m], [1 + k, 1 + n + k, 1 + n + m], [1 + k, 1 + n + m, 1 + m],
              [2 * n + 1, 1 + n + m, 1 + n + k]]
    at = {k: 1 + i for i, k in enumerate(angles)}

    def cut(p, q):                                          # where the edge p-q meets the plane
        t = (z - v[p, 2]) / (v[q, 2] - v[p, 2])
        return v[p, :2] + t * (v[q, :2] - v[p, :2])
    outline = np.array([cut(at[-135], at[-45]), cut(0, at[-45]), cut(0, at[0]), cut(0, at[45]), cut(at[135], at[45]),
                        v[at[45], :2], v[at[0], :2], v[at[-45], :2]])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        loops = loops_from_mesh(types.SimpleNamespace(vertices=v, faces=np.array(f)), z=z)
    assert len(loops) == 1 and _area(loops[0]) == pytest.approx(_area(outline), rel=1e-12)
    assert np.sum(np.hypot(*loops[0].T) < 1e-6) == 3      # the three crossings next to the vertex, all kept


def test_a_face_at_its_own_height_whatever_its_rounding():
    """Cut at the nominal height of a face whose corners carry rounding of either sign (a few 1e-18 m, as the worked
    example's faces do): a bottom face is the box's footprint and a top face gives none, as 1e-15 m above it; a plane
    beyond SECTION_Z_TOL_M of the face cuts it as any other."""
    pytest.importorskip('PyPATools')                       # electrodes3d imports the physics stack
    from PyCentralRegion.electrodes3d import SECTION_Z_TOL_M, loops_from_mesh
    z0 = 13 * 1e-3                                          # 0.013000000000000001, one ulp above 0.013
    ulp = np.spacing(z0)
    noise = np.array([1, -2, 0, 3]) * ulp
    v, f = _box(-5 * MM, 5 * MM, -5 * MM, 5 * MM, z0, 20 * MM)
    v[:4, 2] += noise
    w, g = _box(-5 * MM, 5 * MM, -5 * MM, 5 * MM, 0.0, z0)
    w[4:, 2] += noise
    bottom, top = types.SimpleNamespace(vertices=v, faces=f), types.SimpleNamespace(vertices=w, faces=g)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for z in (z0, 0.013, z0 - 3 * ulp, z0 + 4 * ulp, z0 + 1e-15):
            loops = loops_from_mesh(bottom, z=z)
            assert len(loops) == 1 and _area(loops[0]) == pytest.approx(100 * MM * MM, rel=1e-12), z
            assert loops_from_mesh(top, z=z) == [], z
        assert loops_from_mesh(bottom, z=z0 - 2 * SECTION_Z_TOL_M) == []
        assert len(loops_from_mesh(top, z=z0 - 2 * SECTION_Z_TOL_M)) == 1


def test_a_solid_touching_the_plane_has_no_section_there():
    """A pyramid with its apex in the plane, a tetrahedron with an edge in it and a prism lying on an edge of two
    segments lie above the plane: the section just above has no area and no outline; the pyramid's base in the plane
    is its outline, its top gives none."""
    pytest.importorskip('PyPATools')                       # electrodes3d imports the physics stack
    from PyCentralRegion.electrodes3d import loops_from_mesh
    h = 7 * MM
    v = np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0], [0, 0, 0]], dtype=float) * MM
    v[4, 2] = h
    f = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    up = types.SimpleNamespace(vertices=v, faces=f)
    down = types.SimpleNamespace(vertices=v * [1.0, 1.0, -1.0] + [0.0, 0.0, h], faces=f)
    edge = types.SimpleNamespace(vertices=np.array([[0, 0, 0], [4, 0, 0], [2, -2, 3], [2, 2, 3]], dtype=float) * MM,
                                 faces=np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]))
    # the edge y = 0, x = 0..10 mm split at 5 mm; the sides rise to y = -+3 mm at z = 5 mm
    prism = types.SimpleNamespace(
        vertices=np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0], [0, -3, 5], [0, 3, 5], [10, -3, 5], [10, 3, 5]],
                          dtype=float) * MM,
        faces=np.array([[0, 1, 3], [1, 5, 3], [1, 2, 5], [0, 4, 1], [1, 4, 6], [1, 6, 2], [3, 5, 6], [3, 6, 4],
                        [0, 3, 4], [2, 6, 5]]))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert loops_from_mesh(down, z=0.0) == [] and loops_from_mesh(edge, z=0.0) == []
        assert loops_from_mesh(prism, z=0.0) == []
        assert len(loops_from_mesh(down, z=h / 2)) == 1 and len(loops_from_mesh(edge, z=1 * MM)) == 1
        loops = loops_from_mesh(prism, z=1 * MM)
        assert len(loops) == 1 and _area(loops[0]) == pytest.approx(12 * MM * MM, rel=1e-12)
        loops = loops_from_mesh(up, z=0.0)
        assert len(loops) == 1 and _area(loops[0]) == pytest.approx(16 * MM * MM, rel=1e-12)
        assert loops_from_mesh(up, z=h) == []
