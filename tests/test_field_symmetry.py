"""Mirror symmetry in the field build: the test for a mirror-symmetric mesh (gap_fields.mirror_pairs), the half
solve it allows (the upper triangles' rows with the mirror image folded in) against the whole mesh's, the potential
evaluated on the levels z >= 0 only (fields3d.bem_field3d), the GMRES that tests its residual after every product, the
potential kernels built once, and the builder options that make such a mesh (exit_opening, mirror_mesh, lc_far_mm).
Small hand-made meshes on the CPU; the last test builds the worked example when its input files are on this machine
(about 25 s), else it skips."""
import contextlib
import io
import signal
from pathlib import Path

import numpy as np
import pytest

from PyCentralRegion import config as C

pytest.importorskip('PyPATools')
from PyCentralRegion import gap_fields as G                      # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / 'configs' / 'examples' / 'hchc1p5'


@pytest.fixture(autouse=True)
def sigint_handler_kept():
    """gmsh.initialize() leaves SIGINT at SIG_DFL (see test_build_cuts): each test here leaves the handler it found."""
    handler = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGINT, handler)


def _upper_box(x0, x1, y0, y1, h, n=3):
    """The upper half (z = 0 .. h) of a box's surface as (vertices, triangles): the top and the four sides, n cells a
    side, without the cut face at z = 0. Outward normals."""
    verts, index, tris = [], {}, []

    def node(p):
        key = tuple(np.round(p, 12))
        if key not in index:
            index[key] = len(verts)
            verts.append(p)
        return index[key]

    def quad_grid(origin, e1, e2, n1, n2):
        for i in range(n1):
            for j in range(n2):
                a = node(origin + e1 * i / n1 + e2 * j / n2)
                b = node(origin + e1 * (i + 1) / n1 + e2 * j / n2)
                c = node(origin + e1 * (i + 1) / n1 + e2 * (j + 1) / n2)
                d = node(origin + e1 * i / n1 + e2 * (j + 1) / n2)
                tris.extend([(a, b, c), (a, c, d)])

    ex, ey, ez = np.array([x1 - x0, 0.0, 0.0]), np.array([0.0, y1 - y0, 0.0]), np.array([0.0, 0.0, h])
    o = np.array([x0, y0, 0.0])
    quad_grid(o + ez, ex, ey, n, n)                              # top, normal +z
    quad_grid(o, ex, ez, n, n)                                   # y = y0, normal -y
    quad_grid(o + ex, ey, ez, n, n)                              # x = x1, normal +x
    quad_grid(o + ex + ey, -ex, ez, n, n)                        # y = y1, normal +y
    quad_grid(o + ey, -ey, ez, n, n)                             # x = x0, normal -x
    return np.array(verts), np.array(tris, dtype=np.int64)


def _mirrored(parts):
    """A whole model from [(upper vertices, upper triangles, potential), ...]: per solid the upper triangles, then
    their mirror images (orientation turned), the nodes on z = 0 shared - the layout builder.mirror_model makes."""
    verts, tris, pots, solids, n, n_tri = [], [], [], [], 0, 0
    for v, t, pot in parts:
        on = v[:, 2] == 0.0
        image = np.arange(len(v))
        image[~on] = len(v) + np.arange(int((~on).sum()))
        both = np.vstack([v, v[~on] * [1.0, 1.0, -1.0]])
        block = np.vstack([t, image[t][:, [0, 2, 1]]]) + n
        verts.append(both)
        tris.append(block)
        pots.append(np.full(len(block), pot))
        solids.append({'name': f'solid{len(solids)}', 'potential': pot, 'tri_range': (n_tri, n_tri + len(block))})
        n += len(both)
        n_tri += len(block)
    return G.ElectrodeModel(vertices=np.vstack(verts), triangles=np.vstack(tris), potentials=np.concatenate(pots),
                            wedges=[], params={'dim': 3, 'solids': solids})


def _two_boxes(n=3):
    """A 'dee' at -1000 V beside a grounded box, 6 mm apart: mirror symmetric about z = 0, triangle for triangle."""
    return _mirrored([(*_upper_box(0.000, 0.030, 0.0, 0.040, 0.012, n), -1000.0),
                      (*_upper_box(0.036, 0.066, 0.0, 0.040, 0.016, n), 0.0)])


def test_mirror_pairs():
    """Every triangle of a mirrored mesh has its image, one of each pair is 'upper'; a node moved, a potential
    changed, a face triangulated the other way, or a triangle more, and the mesh is no longer one."""
    model = _two_boxes()
    upper, partner = G.mirror_pairs(model)
    n = model.n_elements
    cz = model.vertices[model.triangles][:, :, 2].mean(axis=1)
    assert upper.sum() * 2 == n and np.array_equal(partner[partner], np.arange(n)) and not np.any(partner == np.arange(n))
    assert np.all(cz[upper] > 0.0) and np.all(cz[~upper] < 0.0) and np.all(upper[partner] != upper)
    pa = model.vertices[model.triangles[upper]]                  # the image of a triangle is that triangle mirrored
    pb = model.vertices[model.triangles[partner[upper]]] * [1.0, 1.0, -1.0]
    assert np.allclose(pa.mean(axis=1), pb.mean(axis=1), rtol=0.0, atol=1e-15)
    for k in range(3):                                           # corner for corner, in some order
        assert np.all(np.min(np.abs(pa[:, k, None, :] - pb).sum(axis=2), axis=1) == 0.0)

    moved = _two_boxes()
    moved.vertices[np.argmax(moved.vertices[:, 2]), 2] += 1e-4
    assert G.mirror_pairs(moved) is None
    other = _two_boxes()
    other.potentials[np.flatnonzero(~upper)[0]] = 5.0            # a pair at two potentials
    assert G.mirror_pairs(other) is None
    flipped = _two_boxes()
    k = int(np.flatnonzero(~upper)[0])                           # two lower triangles of one quad, the other diagonal
    t = flipped.triangles
    mate = next(j for j in np.flatnonzero(~upper) if j != k and len(set(t[j]) & set(t[k])) == 2)
    shared = sorted(set(t[k]) & set(t[mate]))
    a = (set(t[k]) - set(shared)).pop()
    b = (set(t[mate]) - set(shared)).pop()
    t[k], t[mate] = (a, shared[0], b), (a, b, shared[1])
    assert G.mirror_pairs(flipped) is None
    odd = _two_boxes()
    odd.triangles = odd.triangles[:-1]
    odd.potentials = odd.potentials[:-1]
    assert G.mirror_pairs(odd) is None


def test_mirror_pairs_needs_no_particular_order():
    """The pairs are found whatever the order of the triangles and of the nodes (a mesh mirrored by other means)."""
    model = _two_boxes()
    rng = np.random.default_rng(4)
    pn, pt = rng.permutation(len(model.vertices)), rng.permutation(model.n_elements)
    inverse = np.empty_like(pn)
    inverse[pn] = np.arange(len(pn))
    shuffled = G.ElectrodeModel(vertices=model.vertices[pn], triangles=inverse[model.triangles][pt],
                                potentials=model.potentials[pt], wedges=[], params={})
    upper, partner = G.mirror_pairs(shuffled)
    cz = shuffled.vertices[shuffled.triangles][:, :, 2].mean(axis=1)
    assert upper.sum() * 2 == shuffled.n_elements and np.all(cz[upper] > 0.0)
    assert np.allclose(cz[partner], -cz) and np.array_equal(shuffled.potentials[partner], shuffled.potentials)


def test_fold_columns():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((7, 40))
    runs_a, runs_b = np.r_[0:10, 20:30], np.r_[10:20, 30:40]     # blocks, as the builder lays a mirrored mesh out
    assert np.array_equal(G._fold_columns(a, runs_a, runs_b), a[:, runs_a] + a[:, runs_b])
    scattered = rng.permutation(40)                              # no runs: the plain gather
    assert np.array_equal(G._fold_columns(a, scattered[:20], scattered[20:]), a[:, scattered[:20]] + a[:, scattered[20:]])
    big = rng.standard_normal((3, 4000))
    idx = rng.permutation(4000)
    assert np.array_equal(G._fold_columns(big, idx[:2000], idx[2000:]), big[:, idx[:2000]] + big[:, idx[2000:]])


def test_gmres_tests_the_residual_after_every_product():
    """The GPU solver's loop, run here on numpy arrays: it stops at the product that reaches the tolerance (not at
    the end of the restart cycle), its residuals are those of the solution it returns, a restart continues, and
    maxiter is reported."""
    rng = np.random.default_rng(1)
    n = 300
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    a = (q * np.geomspace(1.0, 30.0, n)) @ q.T + 0.02 * rng.standard_normal((n, n))
    a = a * rng.uniform(0.5, 50.0, n)[None, :]                   # badly scaled columns: what the Jacobi scaling is for
    b = rng.standard_normal(n)
    d = np.diag(a).copy()
    res = []
    x, info, products = G._gmres_gpu(a, b, d, 1e-9, 1000, 20000, res, xp=np)
    true = np.linalg.norm(a @ x - b) / np.linalg.norm(b)
    assert info == 0 and 0 < products < n and len(res) == products
    assert true <= 1.5e-9 and res[-1] <= 1e-9 < res[-2]          # stopped at the first product under the tolerance
    assert np.all(np.diff(res) <= 1e-12)                         # the residual of GMRES never grows
    assert np.allclose(x, np.linalg.solve(a, b), rtol=1e-6, atol=1e-9)
    res2 = []
    x2, info2, products2 = G._gmres_gpu(a, b, d, 1e-9, 20, 20000, res2, xp=np)    # short cycles: restarts
    assert info2 == 0 and products2 >= products and np.allclose(x2, x, rtol=1e-5, atol=1e-8)
    res3 = []
    _, info3, products3 = G._gmres_gpu(a, b, d, 1e-9, 1000, 7, res3, xp=np)
    assert info3 == 7 and products3 == 7 and res3[-1] > 1e-9
    zero, info0, n0 = G._gmres_gpu(a, np.zeros(n), d, 1e-9, 1000, 100, [], xp=np)
    assert info0 == 0 and n0 == 0 and not zero.any()


def test_half_solve_is_the_whole_solve():
    """On a mirror-symmetric mesh the half system gives the whole system's charge and potential; 'auto' takes it and
    solves it directly, 'off' never does, 'require' refuses any other mesh, and the strong form is the whole mesh's."""
    pytest.importorskip('bempp_cl')
    model = _two_boxes(n=4)
    half = G.solve_gap_field(model, verbose=False)
    whole = G.solve_gap_field(model, solver='dense', symmetry='off', verbose=False)
    assert half.mirror and half.n_unknowns == model.n_elements // 2 and half.solver == 'dense'
    assert half.n_iterations == 0 and half.final_residual < 1e-10
    assert not whole.mirror and whole.n_unknowns == model.n_elements
    xh, xw = np.asarray(half.neumann.coefficients, dtype=float), np.asarray(whole.neumann.coefficients, dtype=float)
    upper, partner = G.mirror_pairs(model)
    assert np.array_equal(xh[upper], xh[partner[upper]])         # the same charge on both triangles of a pair
    assert np.allclose(xh, xw, rtol=0.0, atol=2e-4 * np.abs(xw).max())    # to Bempp's quadrature between a pair's images
    area = 0.5 * np.linalg.norm(np.cross(*(model.vertices[model.triangles[:, k]] - model.vertices[model.triangles[:, 0]]
                                           for k in (1, 2))), axis=1)
    for s in model.params['solids']:                             # each electrode's total charge
        a, b = s['tri_range']
        assert np.sum(xh[a:b] * area[a:b]) == pytest.approx(np.sum(xw[a:b] * area[a:b]), rel=2e-5)
    pts = np.array([[0.033, 0.020, 0.0], [0.033, 0.020, 0.006], [0.033, 0.020, -0.006], [0.015, -0.010, 0.004],
                    [0.050, 0.050, -0.010], [0.033, 0.010, 0.020]])
    ph, pw = half.potential(pts), whole.potential(pts)
    assert np.allclose(ph, pw, rtol=0.0, atol=0.05)              # of 1000 V
    assert ph[1] == pytest.approx(ph[2], abs=1e-9)               # even in z
    inside = half.potential(np.array([[0.015, 0.020, 0.0], [0.051, 0.020, 0.0]]))
    assert inside == pytest.approx([-1000.0, 0.0], abs=60.0)     # each box at its potential, as well as this mesh can
    it = G.solve_gap_field(model, solver='scipy', tol=1e-10, verbose=False)
    assert it.mirror and it.solver == 'scipy' and it.n_iterations > 0
    assert np.allclose(np.asarray(it.neumann.coefficients), xh, rtol=0.0, atol=1e-6 * np.abs(xh).max())
    assert not G.solve_gap_field(model, solver='strong', tol=1e-8, verbose=False).mirror
    lopsided = _mirrored([(*_upper_box(0.0, 0.03, 0.0, 0.04, 0.012, 2), -1000.0)])
    lopsided.vertices[np.argmax(lopsided.vertices[:, 2]), 2] += 1e-3
    assert not G.solve_gap_field(lopsided, verbose=False).mirror
    with pytest.raises(RuntimeError, match='not mirror symmetric'):
        G.solve_gap_field(lopsided, symmetry='require', verbose=False)
    with pytest.raises(ValueError, match="symmetry must be"):
        G.solve_gap_field(model, symmetry='half', verbose=False)


def test_potential_batches_share_one_kernel_build():
    """The potential of a solution is the same whatever the batch size, and the same as one Bempp operator per batch
    gives (the kernels are Bempp's own, built once)."""
    pytest.importorskip('bempp_cl')
    sol = G.solve_gap_field(_two_boxes(n=3), verbose=False)
    rng = np.random.default_rng(2)
    pts = np.column_stack([rng.uniform(-0.01, 0.08, 257), rng.uniform(-0.01, 0.05, 257), rng.uniform(-0.03, 0.03, 257)])
    one = sol.potential(pts, chunk=1000)
    assert np.array_equal(sol.potential(pts, chunk=50), one)
    built = sol._kernels.get(('cpu', 'double'))
    sol._kernels[('cpu', 'double')] = None                       # one Bempp operator per batch
    assert np.array_equal(sol.potential(pts, chunk=100), one) or built is None
    assert np.allclose(sol.potential(pts, chunk=100), one, rtol=0.0, atol=1e-9)
    assert len(sol.potential(np.zeros((0, 3)))) == 0


class _EvenPotential:
    """Stands in for a GapFieldSolution: a potential that is even in z, and a count of the points it was asked for."""
    mirror = True

    def __init__(self):
        self.asked = []

    def potential(self, pts, **kw):
        self.asked.append(np.array(pts))
        return 3.0 * pts[:, 0] - 2.0 * pts[:, 1] ** 2 + 40.0 * pts[:, 2] ** 2 + np.cos(30.0 * pts[:, 2]) * pts[:, 0]


def test_mirror_levels_and_the_half_grid():
    """Levels that mirror onto each other are found; with them the potential is asked for on z >= 0 only and the field
    is the whole grid's (E_x and E_y to the last bit); other levels are evaluated whole."""
    F = pytest.importorskip('PyCentralRegion.fields3d')
    zs = np.arange(-0.028, 0.028 + 1e-9, 0.002)
    assert np.array_equal(F.mirror_levels(zs), len(zs) - 1 - np.arange(len(zs)))
    assert np.array_equal(F.mirror_levels(np.array([-0.003, -0.001, 0.001, 0.003])), [3, 2, 1, 0])     # no level at 0
    assert F.mirror_levels(np.array([-0.002, 0.0, 0.003])) is None
    assert F.mirror_levels(np.array([0.0, 0.002, 0.004])) is None and F.mirror_levels(np.array([0.0])) is None
    xs = np.linspace(-0.02, 0.02, 9)
    whole, half = _EvenPotential(), _EvenPotential()
    f_whole = F.bem_field3d(whole, xs, xs, zs, verbose=False)
    f_half = F.bem_field3d(half, xs, xs, zs, mirror_z=True, verbose=False)
    n_xy, n_up = len(xs) ** 2, int(np.sum(zs > -1e-12))
    assert len(whole.asked[0]) == n_xy * len(zs) and len(half.asked[0]) == n_xy * n_up
    assert np.all(half.asked[0][:, 2] > -1e-12)
    for k in 'xyz':
        assert np.allclose(f_half._values[k], f_whole._values[k], rtol=0.0, atol=1e-9 * np.abs(f_whole._values[k]).max())
    ex, ez = np.asarray(f_half._values['x']), np.asarray(f_half._values['z'])
    assert np.array_equal(ex, ex[:, :, ::-1]) and np.allclose(ez, -ez[:, :, ::-1], rtol=0.0, atol=1e-9 * np.abs(ez).max())
    k0 = len(zs) // 2
    assert np.abs(ez[:, :, k0]).max() <= 1e-9 * np.abs(ez).max()                 # no vertical field on the midplane
    # inside a cylinder, as the fields stage evaluates it; and levels that do not mirror
    ring, ring_half = _EvenPotential(), _EvenPotential()
    a = F.bem_field3d(ring, xs, xs, zs, r_max=0.012, verbose=False)
    b = F.bem_field3d(ring_half, xs, xs, zs, r_max=0.012, mirror_z=True, verbose=False)
    assert len(ring_half.asked[0]) * len(zs) == len(ring.asked[0]) * n_up
    assert np.allclose(b._values['y'], a._values['y'], rtol=0.0, atol=1e-9 * np.abs(a._values['y']).max())
    skew = _EvenPotential()
    F.bem_field3d(skew, xs, xs, zs + 0.0005, mirror_z=True, verbose=False)
    assert len(skew.asked[0]) == n_xy * len(zs)

    # a metal mask that is not mirror symmetric to the last bit (a face on a grid level): with mirror_z the upper
    # levels' mask serves both sides, so the zeroed nodes mirror too
    def lopsided_metal(p):
        return (p[:, 2] > 0.0159999) | (p[:, 2] < -0.0180001)     # the "plates": at +16 mm and at -18 mm

    c = F.bem_field3d(_EvenPotential(), xs, xs, zs, metal=lopsided_metal, erode=1, mirror_z=True, verbose=False)
    cx = np.asarray(c._values['x'])
    assert np.array_equal(cx, cx[:, :, ::-1])
    assert np.all(cx[1:-1, 1:-1, -2] == 0.0) and np.all(cx[1:-1, 1:-1, 1] == 0.0)     # deep in the plates, both sides
    assert np.all(cx[1:-1, 1:-1, 4] == 0.0) and np.any(cx[1:-1, 1:-1, 6] != 0.0)      # -20 mm too; the layer at -16 kept
    d = F.bem_field3d(_EvenPotential(), xs, xs, zs, metal=lopsided_metal, erode=1, verbose=False)
    dx = np.asarray(d._values['x'])
    assert not np.array_equal(dx, dx[:, :, ::-1])                 # without it the mask is the metal test's, level by level


def test_exit_opening_centred():
    """A round exit ring off the median plane is centred on it with the exported ring's exit half height (it then
    contains the exported ring); a polygon is stretched in z; the emitting plane is levelled; the bounds read the
    centred opening."""
    B = pytest.importorskip('PyCentralRegion.geometry.builder')
    ang = np.radians(np.arange(0.0, 360.0, 10.0))
    n_hat, u_hat = np.array([-0.14, 0.99, 0.0]) / np.hypot(0.14, 0.99), np.array([0.99, 0.14, 0.0]) / np.hypot(0.14, 0.99)
    centre = np.array([54.5, 32.1, -0.76])
    ring = centre + 10.0 * (np.cos(ang)[:, None] * u_hat + np.sin(ang)[:, None] * np.array([0.0, 0.0, 1.0]))
    ol = {'outer': np.zeros((4, 2)), 'aperture': ring}
    out = B.centre_exit_aperture(ol)
    c, r, spread = B._circle_fit(out['aperture'])
    assert c == pytest.approx([54.5, 32.1, 0.0], abs=1e-9) and r == pytest.approx(10.76, abs=1e-9) and spread < 1e-9
    assert B.exit_half_height_mm(out) == pytest.approx(B.exit_half_height_mm(ol)) == pytest.approx(10.76)
    assert out['aperture_centred'] == {'shape': 'round', 'radius_mm': [pytest.approx(10.0), pytest.approx(10.76)],
                                       'z_centre_mm': pytest.approx(-0.76)}
    assert np.array_equal(ol['aperture'], ring) and abs(np.ptp((out['aperture'] - c) @ n_hat)) < 1e-9   # same plane
    box = np.array([[30.0, -5.0, -9.0], [30.0, 5.0, -9.0], [30.0, 5.0, 11.0], [30.0, -5.0, 11.0]])
    sq = B.centre_exit_aperture({'aperture': box})
    assert np.array_equal(sq['aperture'][:, :2], box[:, :2]) and sorted(sq['aperture'][:, 2]) == [-11.0, -11.0, 11.0, 11.0]
    plane = {'point': np.array([55.8, 22.3, -0.19]), 'normal': np.array([-0.1394, 0.9901, -0.0144])}
    lev = B.level_plane(plane)
    assert lev['normal'][2] == 0.0 and np.linalg.norm(lev['normal']) == pytest.approx(1.0)
    assert lev['tilt_deg'] == pytest.approx(-0.825, abs=2e-3) and plane['normal'][2] == -0.0144
    brick = B.beam_channel(out['aperture'], lev, extend=200.0, back=1.0)
    assert np.allclose(brick['start'][:, 2], out['aperture'][:, 2]) and brick['sweep'][2] == 0.0   # a level, centred bore
    with pytest.raises(C.ConfigError, match='level'):
        B.level_plane({'normal': np.array([0.0, 0.0, 1.0])})


def test_mirror_mesh_options():
    """mirror_mesh needs the centred opening and centred hardware; lc_far_mm its ramp above the channel and inside the
    electrodes."""
    options = pytest.importorskip('PyCentralRegion.geometry.options')
    GC = options.GeometryConfig
    assert (GC().exit_opening, GC().mirror_mesh, GC().lc_far_mm) == ('export', False, 0.0)
    with pytest.raises(C.ConfigError, match="exit_opening = 'centred'"):
        GC(mirror_mesh=True)
    with pytest.raises(C.ConfigError, match='exit_opening'):
        GC(exit_opening='center')
    ok = GC(mirror_mesh=True, exit_opening='centred', windows=(options.WindowSpec(1, 'dee', 10.0, 30.0, 8.0),))
    assert ok.mirror_mesh
    with pytest.raises(C.ConfigError, match=r"window on gap 1 dee \(z_center_mm = 2\)"):
        GC(mirror_mesh=True, exit_opening='centred', windows=(options.WindowSpec(1, 'dee', 10.0, 30.0, 8.0, 2.0),))
    with pytest.raises(C.ConfigError, match=r"hole on gap 0 dee \(z_mm = -1\)"):
        GC(mirror_mesh=True, exit_opening='centred', holes=(options.HoleSpec(0, 'dee', 5.0, 20.0, -1.0, 10.0, 30.0),))
    far = GC(lc_far_mm=32.0)                                     # defaults: 30 mm channel, ramp 34 .. 48 mm of 50
    assert (far.lc_far_z0_mm, far.lc_far_z1_mm) == (34.0, 48.0)
    with pytest.raises(C.ConfigError, match='must exceed lc_mm'):
        GC(lc_far_mm=8.0)
    with pytest.raises(C.ConfigError, match=r'\|z\| >= 31 mm'):
        GC(lc_far_mm=32.0, aperture_outer_mm=50.0, plate_half_gap_mm=40.0, half_height_mm=60.0, lc_far_z0_mm=30.0)
    with pytest.raises(C.ConfigError, match='half_height_mm = 50'):
        GC(lc_far_mm=32.0, lc_far_z1_mm=55.0)
    with pytest.raises(C.ConfigError, match='lc_far_mm must be >= 0'):
        GC(lc_far_mm=-1.0)


def test_mirror_model_closes_the_upper_half():
    """builder.mirror_model: the triangles on the cut plane go, the rest gets its image, the solids stay contiguous
    blocks, and the result passes mirror_pairs and the tracker's column test (closed solids)."""
    B = pytest.importorskip('PyCentralRegion.geometry.builder')
    from PyCentralRegion.metal_columns import ColumnObstacles
    parts, n_cut = [], 0
    verts, tris, pots, solids, nv, nt = [], [], [], [], 0, 0
    for (x0, x1, h, pot) in ((0.000, 0.030, 0.012, -1000.0), (0.036, 0.066, 0.016, 0.0)):
        v, t = _upper_box(x0, x1, 0.0, 0.040, h, 3)
        on = np.flatnonzero(v[:, 2] == 0.0)                      # close the half at z = 0, as the cut solid's mesh is
        c = len(v)
        v = np.vstack([v, [[0.5 * (x0 + x1), 0.020, 0.0]]])
        ring = on[np.argsort(np.arctan2(v[on, 1] - 0.020, v[on, 0] - 0.5 * (x0 + x1)))]
        cap = np.array([(c, ring[(k + 1) % len(ring)], ring[k]) for k in range(len(ring))])
        n_cut += len(cap)
        t = np.vstack([t, cap])
        verts.append(v)
        tris.append(t + nv)
        pots.append(np.full(len(t), pot))
        solids.append({'name': f'solid{len(solids)}', 'potential': pot, 'tri_range': (nt, nt + len(t))})
        nv += len(v)
        nt += len(t)
        parts.append(len(t) - len(cap))
    half = G.ElectrodeModel(vertices=np.vstack(verts) + [0.0, 0.0, 3e-12], triangles=np.vstack(tris),
                            potentials=np.concatenate(pots), wedges=[], params={'dim': 3, 'solids': solids})
    whole = B.mirror_model(half, verbose=False)
    assert whole.n_elements == 2 * sum(parts) and whole.params['mirror_mesh'] is True
    assert [s['tri_range'] for s in whole.params['solids']] == [(0, 2 * parts[0]), (2 * parts[0], 2 * sum(parts))]
    assert np.array_equal(np.unique(whole.potentials[:2 * parts[0]]), [-1000.0])
    upper, partner = G.mirror_pairs(whole)
    assert np.array_equal(np.flatnonzero(upper)[:parts[0]], np.arange(parts[0]))        # upper block first, then images
    assert np.array_equal(partner[:parts[0]], parts[0] + np.arange(parts[0]))
    cols = ColumnObstacles.from_model(whole, spacing=1e-3, extent=0.07, odd='raise')    # closed: no odd column
    inside = cols(np.array([[0.015, 0.02, 0.0], [0.015, 0.02, -0.011], [0.015, 0.02, 0.013], [0.051, 0.02, -0.015],
                            [0.033, 0.02, 0.0]]))
    assert inside.tolist() == [True, True, False, True, False]
    below = G.ElectrodeModel(vertices=half.vertices - [0.0, 0.0, 1e-3], triangles=half.triangles,
                             potentials=half.potentials, wedges=[], params=half.params)
    with pytest.raises(RuntimeError, match='below the median plane'):
        B.mirror_model(below, verbose=False)


def test_the_worked_example_builds_a_mirrored_mesh(tmp_path):
    """The worked example with its exit opening centred, its mesh mirrored and its far faces coarse: every electrode's
    upper half holds half its volume, the mesh passes mirror_pairs and the column test, it has a third fewer triangles,
    and the CAD file is the whole model's. Without exit_opening = 'centred' the same electrodes are refused."""
    pytest.importorskip('PyCentralRegion.geometry.builder')
    pytest.importorskip('gmsh')
    from PyCentralRegion.geometry import build as GB
    from PyCentralRegion.metal_columns import ColumnObstacles
    cfg_path = EXAMPLES / 'design_v6o_c1_cold.toml'
    if not cfg_path.exists():
        pytest.skip("the worked example is not in this checkout")

    def build(name, extra):
        model = tmp_path / name / 'model3d.pkl'
        cfg = C.load(cfg_path, overrides=extra + ['geometry.plot_dees=false', f'paths.model3d="{model.as_posix()}"'])
        pc = C.section(cfg, 'paths')
        if not all(Path(p).exists() for p in (pc.seed, pc.housing_outline, pc.handoff, pc.housing_step)):
            pytest.skip("the design's input files are not on this machine")
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            GB.build(cfg)
        import pickle
        with open(model, 'rb') as fh:
            return pickle.load(fh), log.getvalue(), model

    plain, _, _ = build('plain', [])
    assert G.mirror_pairs(plain) is None and not plain.params.get('mirror_mesh')
    sym, txt, path = build('sym', ['geometry.exit_opening="centred"', 'geometry.mirror_mesh=true',
                                   'geometry.lc_far_mm=32.0'])
    assert sym.params['mirror_mesh'] is True and G.mirror_pairs(sym) is not None
    assert "exit opening centred" in txt and "mirror mesh: every electrode cut at the median plane" in txt
    assert sym.n_elements < 0.8 * plain.n_elements               # the far faces are coarse
    ColumnObstacles.from_model(sym, spacing=5e-4, extent=sym.extent() + 0.005, odd='raise')
    import json
    rec = json.loads(path.with_name('model3d_build.json').read_text())
    assert rec['mesh']['mirror'] is True and rec['exit_opening']['shape'] == 'round'
    assert all(abs(v - 0.5) < 1e-4 for v in rec['mesh']['upper_volume_share'].values())
    assert path.with_name('model3d.step').stat().st_size > 0.8 * (tmp_path / 'plain' / 'model3d.step').stat().st_size
    cfg = C.load(cfg_path, overrides=['geometry.plot_dees=false', f'paths.model3d="{(tmp_path / "x" / "m.pkl").as_posix()}"'])
    cfg['geometry']['mirror_mesh'] = True
    with pytest.raises(C.ConfigError, match="exit_opening = 'centred'"):
        C.section(cfg, 'geometry')
    # a mirrored mesh that cannot be made costs the speed, not the geometry: the electrodes are built again and meshed whole
    from PyCentralRegion.geometry import builder as B

    def refuse(gmsh, groups, verbose=True):
        raise B.MirrorRefused("not mirror symmetric about the median plane: dee3 (a test)")
    real = B.cut_upper_half
    B.cut_upper_half = refuse
    try:
        whole, txt, path = build('refused', ['geometry.exit_opening="centred"', 'geometry.mirror_mesh=true'])
    finally:
        B.cut_upper_half = real
    assert "mirror mesh NOT made (MirrorRefused: not mirror symmetric about the median plane: dee3 (a test))" in txt
    assert not whole.params.get('mirror_mesh') and G.mirror_pairs(whole) is None
    rec = json.loads(path.with_name('model3d_build.json').read_text())
    assert rec['mesh']['mirror'] is False and 'dee3 (a test)' in rec['mesh']['mirror_refused']
    assert whole.n_elements > 0.9 * plain.n_elements             # the whole electrodes' mesh, the far faces as fine
    ColumnObstacles.from_model(whole, spacing=5e-4, extent=whole.extent() + 0.005, odd='raise')
    mirror = C.section(C.load(EXAMPLES / 'design_v6o_c1_mirror.toml'), 'geometry')          # the example that sets them
    assert (mirror.exit_opening, mirror.mirror_mesh, mirror.lc_far_mm) == ('centred', True, 32.0)
    paths = C.section(C.load(EXAMPLES / 'design_v6o_c1_mirror.toml'), 'paths')
    assert paths.model3d != C.section(C.load(cfg_path), 'paths').model3d
