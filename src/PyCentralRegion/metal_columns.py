"""
metal_columns.py - the metal of a 3D electrode model as vertical columns: the tracker's obstacle test, exact in z.

Every node of a square x-y grid holds the heights at which the vertical line through it enters and leaves metal.
The stacked raster (``electrodes3d.stacked_obstacles``) samples the solids on z levels and tests the nearest level,
so a face acts at the half-level next to it: with 1 mm levels a channel's height is seen in 2 mm steps of its full
height, and a face lying on a level acts half a level off (the MOBO pilot's audit, 2026-11-02). The columns are cut
from the triangles instead, so every face acts at its own height.

The cut: each triangle is tested against the grid nodes inside its x-y bounding box with exact integer predicates
(x and y on a lattice of ``LATTICE_M``), ties broken by one fixed perturbation of the line, (x + e, y + e^2) with
e -> 0+. The line through a node then meets no edge and no vertex, so it crosses a closed solid an even number of
times whatever the orientation of its triangles. A solid that some line crosses an odd number of times is not closed:
it is refused, or (``odd='drop'``, the fields stage's mask) left out of those columns. The crossings of each solid
are paired into metal intervals (even-odd, so a hollow solid's holes come out right, as in the slices) and the
intervals of all solids merged (separate bodies, such as an un-fused housing, may overlap). Faces count as metal
(closed intervals).

A point is tested in the column of its nearest node: exact in z (a sloped face at the node's x and y), good to half a
cell in x-y, vacuum off the grid. ``in_columns`` is that test for one point. ``ColumnObstacles.__call__`` and the
compiled 3D kernel (``fast_track3d``) both run it, so the two paths agree hit for hit; the kernel's selftest checks
that its compiled copy is the current one.
"""
import time

import numpy as np

try:
    from numba import njit
except ImportError:                                   # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]

LATTICE_M = 1e-9              # x-y lattice of the exact predicates [m]; coarser only for a model wider than ~1 m
_MAX_LATTICE = 2 ** 30        # lattice coordinates stay below this, so the predicates' products fit in int64


@njit(cache=True, inline='always')
def _side(ex, ey, px, py, ax, ay):
    """+1 when the lattice point p lies left of the directed edge from a along (ex, ey), -1 when right. A point on
    the edge's line takes the side of p + (e, e^2), e -> 0+; 0 only for an edge of length zero."""
    s = ex * (py - ay) - ey * (px - ax)
    if s > 0:
        return 1
    if s < 0:
        return -1
    if ey != 0:
        return -1 if ey > 0 else 1
    if ex != 0:
        return 1 if ex > 0 else -1
    return 0


@njit(cache=True)
def _crossings(vx, vy, vz, tris, tri_solid, nodes, counts, starts, zs, sid, fill, store):
    """One pass over the triangles: count (``store`` False: counts[c + 1] += 1) or store (z and solid at
    starts[c] + fill[c]) the crossings of the vertical lines through the grid nodes, line (i, j) at lattice point
    (nodes[i], nodes[j]), column c = i * len(nodes) + j."""
    n = len(nodes)
    for t in range(tris.shape[0]):
        a = tris[t, 0]
        b = tris[t, 1]
        c = tris[t, 2]
        ax, ay, bx, by, cx, cy = vx[a], vy[a], vx[b], vy[b], vx[c], vy[c]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if area == 0:
            continue                       # seen edge-on (a vertical face): the perturbed line never meets it
        i0 = np.searchsorted(nodes, min(ax, bx, cx))
        i1 = np.searchsorted(nodes, max(ax, bx, cx), side='right')
        j0 = np.searchsorted(nodes, min(ay, by, cy))
        j1 = np.searchsorted(nodes, max(ay, by, cy), side='right')
        for i in range(i0, i1):
            px = nodes[i]
            for j in range(j0, j1):
                py = nodes[j]
                s = _side(bx - ax, by - ay, px, py, ax, ay)
                if s == 0 or _side(cx - bx, cy - by, px, py, bx, by) != s \
                        or _side(ax - cx, ay - cy, px, py, cx, cy) != s:
                    continue
                col = i * n + j
                if not store:
                    counts[col + 1] += 1
                    continue
                # the height of the triangle's plane at the node (barycentric weights of b and c, exact integers)
                wb = (ax - cx) * (py - cy) - (ay - cy) * (px - cx)
                wc = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
                k = starts[col] + fill[col]
                zs[k] = vz[a] + (wb * (vz[b] - vz[a]) + wc * (vz[c] - vz[a])) / area
                sid[k] = tri_solid[t]
                fill[col] += 1


@njit(cache=True)
def _intervals(starts, zs, sid, n_solids, drop_odd):
    """The metal intervals of every column: its crossings sorted by (solid, z), paired per solid, and the pairs of
    all solids merged. A solid crossed an odd number of times in a column gives no interval there with ``drop_odd``.
    Returns (interval starts per column, lower ends, upper ends, odd columns per solid)."""
    n_cols = len(starts) - 1
    out_starts = np.zeros(n_cols + 1, np.int64)
    lo = np.empty(starts[n_cols] // 2 + 1)
    hi = np.empty(starts[n_cols] // 2 + 1)
    odd = np.zeros(n_solids, np.int64)
    most = 0
    for c in range(n_cols):
        most = max(most, starts[c + 1] - starts[c])
    tlo = np.empty(most // 2 + 1)
    thi = np.empty(most // 2 + 1)
    w = 0
    for c in range(n_cols):
        s, e = starts[c], starts[c + 1]
        for k in range(s + 1, e):                  # insertion sort: a column holds a handful of crossings
            zk, sk = zs[k], sid[k]
            m = k - 1
            while m >= s and (sid[m] > sk or (sid[m] == sk and zs[m] > zk)):
                zs[m + 1] = zs[m]
                sid[m + 1] = sid[m]
                m -= 1
            zs[m + 1] = zk
            sid[m + 1] = sk
        nt = 0
        k = s
        while k < e:
            q = k
            while q < e and sid[q] == sid[k]:
                q += 1
            if (q - k) % 2:
                odd[sid[k]] += 1
                if drop_odd:
                    k = q
                    continue
            for m in range(k, q - 1, 2):
                tlo[nt] = zs[m]
                thi[nt] = zs[m + 1]
                nt += 1
            k = q
        for k in range(1, nt):
            lk, hk = tlo[k], thi[k]
            m = k - 1
            while m >= 0 and tlo[m] > lk:
                tlo[m + 1] = tlo[m]
                thi[m + 1] = thi[m]
                m -= 1
            tlo[m + 1] = lk
            thi[m + 1] = hk
        if nt:
            clo, chi = tlo[0], thi[0]
            for k in range(1, nt):
                if tlo[k] <= chi:
                    chi = max(chi, thi[k])
                else:
                    lo[w] = clo
                    hi[w] = chi
                    w += 1
                    clo, chi = tlo[k], thi[k]
            lo[w] = clo
            hi[w] = chi
            w += 1
        out_starts[c + 1] = w
    return out_starts, lo[:w].copy(), hi[:w].copy(), odd


@njit(cache=True, inline='always')
def in_columns(co, x, y, z):
    """The column test of one point. ``co`` = (use, starts, lo, hi, x0, y0, hx, hy, nx, ny) as packed by
    ``ColumnObstacles``: the nearest node's column, metal on its closed intervals, vacuum off the grid.

    The compiled 3D kernel inlines this function, and numba checks its cached kernel against fast_track3d.py only:
    after an edit here the kernel keeps its old copy until that cache is renewed (``fast_track3d.selftest`` finds a
    stale copy and the run falls back to the Python path)."""
    if co[0] == 0:
        return False
    i = int(np.rint((x - co[4]) / co[6]))
    j = int(np.rint((y - co[5]) / co[7]))
    if i < 0 or i >= co[8] or j < 0 or j >= co[9]:
        return False
    starts, lo, hi = co[1], co[2], co[3]
    c = i * co[9] + j
    for m in range(starts[c], starts[c + 1]):
        if z < lo[m]:
            return False
        if z <= hi[m]:
            return True
    return False


@njit(cache=True)
def _in_columns_batch(co, x, y, z, out):
    for p in range(len(x)):
        out[p] = in_columns(co, x[p], y[p], z[p])


# the kernel's placeholder when a run has no column test (same types as a packed one)
NO_COLUMNS = (0, np.zeros(1, dtype=np.int64), np.zeros(1), np.zeros(1), 0.0, 0.0, 1.0, 1.0, 0, 0)


def column_grid(extent: float, spacing: float) -> np.ndarray:
    """The x (= y) nodes [m] of a column test over |x|, |y| <= ``extent``: those of ``electrodes3d.raster_obstacles``
    and ``stacked_obstacles`` for the same extent and spacing, so the column test and the slices share their nodes."""
    n = int(np.ceil(2.0 * float(extent) / spacing)) + 1
    return np.linspace(-float(extent), float(extent), n)


def build_columns(vertices, triangles, solid_ranges, xs, names=None, odd: str = 'raise'):
    """The column intervals of the closed solids ``triangles[n0:n1]`` for (n0, n1) in ``solid_ranges`` over the
    square grid of nodes ``xs`` [m] (x = y). Returns (starts, lo, hi, info). A solid that some line crosses an odd
    number of times is not closed: ``odd = 'raise'`` refuses it, ``'drop'`` leaves it out of those columns (the
    columns where it is closed keep it) and counts them in info['odd_columns'] {solid name: columns}."""
    if odd not in ('raise', 'drop'):
        raise ValueError(f"build_columns: odd = {odd!r} ('raise' or 'drop')")
    xs = np.asarray(xs, dtype=float)
    x0 = float(xs[0])
    v = np.asarray(vertices, dtype=float)
    span = max(float(xs[-1] - x0), float(np.max(np.abs(v[:, :2] - x0))))
    lattice = max(LATTICE_M, span / (_MAX_LATTICE - 1))
    nodes = np.rint((xs - x0) / lattice).astype(np.int64)
    vx = np.ascontiguousarray(np.rint((v[:, 0] - x0) / lattice).astype(np.int64))
    vy = np.ascontiguousarray(np.rint((v[:, 1] - x0) / lattice).astype(np.int64))
    vz = np.ascontiguousarray(v[:, 2])
    tris, tri_solid = [], []
    for k, (n0, n1) in enumerate(solid_ranges):
        tris.append(np.asarray(triangles[n0:n1], dtype=np.int64))
        tri_solid.append(np.full(n1 - n0, k, dtype=np.int64))
    tris = np.ascontiguousarray(np.vstack(tris))
    tri_solid = np.ascontiguousarray(np.concatenate(tri_solid))
    n_cols = len(xs) * len(xs)
    counts = np.zeros(n_cols + 1, dtype=np.int64)
    empty_f, empty_i = np.zeros(0), np.zeros(0, dtype=np.int64)
    _crossings(vx, vy, vz, tris, tri_solid, nodes, counts, counts, empty_f, empty_i, empty_i, False)
    starts = np.cumsum(counts)
    zs = np.empty(int(starts[-1]))
    sid = np.empty(int(starts[-1]), dtype=np.int64)
    fill = np.zeros(n_cols, dtype=np.int64)
    _crossings(vx, vy, vz, tris, tri_solid, nodes, counts, starts, zs, sid, fill, True)
    out_starts, lo, hi, n_odd = _intervals(starts, zs, sid, len(solid_ranges), odd == 'drop')
    names = list(names) if names is not None else [f'solid {k}' for k in range(len(solid_ranges))]
    bad = {names[k]: int(n_odd[k]) for k in range(len(n_odd)) if n_odd[k]}
    if bad and odd == 'raise':
        raise ValueError("column test: vertical lines cross " + ', '.join(f"{k} an odd number of times in {n} columns"
                                                                         for k, n in bad.items())
                         + ": the solid is not closed (use the stacked slices, track.obstacles = 'raster')")
    info = {'n_crossings': int(starts[-1]), 'n_intervals': int(len(lo)), 'lattice_m': lattice,
            'max_per_column': int(np.max(np.diff(out_starts))) if n_cols else 0, 'odd_columns': bad}
    return out_starts, lo, hi, info


class ColumnObstacles:
    """Metal test ``inside(xyz) -> bool`` exact in z (see the module docstring): the nearest node's column of metal
    intervals. (N, 2) input is taken at z = 0. Carries ``xs`` (nodes in x and y), ``spacing``, ``ndim = 3`` (read by
    ``tracking.MetalTerminator``) and ``packed``, the tuple the compiled 3D kernel tests (``fast_track3d``)."""

    ndim = 3

    def __init__(self, starts, lo, hi, xs, name: str = 'metal columns', info=None):
        self.xs = np.asarray(xs, dtype=float)
        n = len(self.xs)
        self.starts = np.ascontiguousarray(starts, dtype=np.int64)
        self.lo = np.ascontiguousarray(lo, dtype=float)
        self.hi = np.ascontiguousarray(hi, dtype=float)
        if self.starts.shape != (n * n + 1,) or self.lo.shape != self.hi.shape \
                or int(self.starts[-1]) != len(self.lo):
            raise ValueError("ColumnObstacles: starts, lo and hi do not match the grid")
        self.spacing = float(self.xs[1] - self.xs[0])
        x0 = float(self.xs[0])
        self.packed = (1, self.starts, self.lo, self.hi, x0, x0, self.spacing, self.spacing, n, n)
        self.name = name
        self.info = dict(info or {})

    @classmethod
    def from_model(cls, model, spacing: float = 5e-4, extent=None, verbose: bool = False,
                   odd: str = 'raise') -> 'ColumnObstacles':
        """The column test of a 3D electrode model (params['solids'] with each solid's triangle range) over
        |x|, |y| <= ``extent`` (default: the model's own) at node ``spacing`` [m]. ``odd``: what to do with a solid
        that is not closed (``build_columns``)."""
        if not (model.params.get('dim') == 3 and 'solids' in model.params):
            raise ValueError("ColumnObstacles needs a 3D electrode model (params['solids'])")
        t0 = time.time()
        xs = column_grid(extent if extent is not None else model.extent(), spacing)
        solids = model.params['solids']
        starts, lo, hi, info = build_columns(model.vertices, model.triangles, [s['tri_range'] for s in solids], xs,
                                             names=[s['name'] for s in solids], odd=odd)
        info.update(spacing_m=float(xs[1] - xs[0]), extent_m=float(xs[-1]), build_s=time.time() - t0)
        if verbose:
            print(f"[metal_columns] {len(xs)}x{len(xs)} columns at {info['spacing_m'] * 1e3:.2f} mm: "
                  f"{info['n_intervals']:,} metal intervals ({info['n_crossings']:,} crossings) in "
                  f"{info['build_s']:.1f} s")
        return cls(starts, lo, hi, xs, info=info)

    def __call__(self, pts) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        x, y = np.ascontiguousarray(p[:, 0]), np.ascontiguousarray(p[:, 1])
        z = np.ascontiguousarray(p[:, 2]) if p.shape[1] >= 3 else np.zeros(len(p))
        out = np.zeros(len(p), dtype=np.bool_)
        _in_columns_batch(self.packed, x, y, z, out)
        return out

    def column(self, x: float, y: float) -> np.ndarray:
        """The metal intervals (K, 2) [m] of the column nearest to (x, y); empty off the grid."""
        i = int(np.rint((x - self.xs[0]) / self.spacing))
        j = int(np.rint((y - self.xs[0]) / self.spacing))
        n = len(self.xs)
        if not (0 <= i < n and 0 <= j < n):
            return np.zeros((0, 2))
        c = i * n + j
        s, e = self.starts[c], self.starts[c + 1]
        return np.column_stack([self.lo[s:e], self.hi[s:e]])

    def with_full_height(self, mask: np.ndarray, name: str = None) -> 'ColumnObstacles':
        """A copy in which the columns of the (n, n) boolean ``mask`` are metal at every height (full-height posts
        and boxes put into the test)."""
        n = len(self.xs)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (n, n):
            raise ValueError(f"with_full_height: mask {mask.shape} for a {n}x{n} grid")
        full = mask.ravel()
        counts = np.diff(self.starts)
        new_counts = np.where(full, 1, counts)
        starts = np.concatenate([[0], np.cumsum(new_counts)]).astype(np.int64)
        lo = np.full(int(starts[-1]), -np.inf)
        hi = np.full(int(starts[-1]), np.inf)
        col = np.repeat(np.arange(n * n), counts)
        keep = ~full[col]
        dst = starts[col[keep]] + (np.arange(len(self.lo))[keep] - self.starts[col[keep]])
        lo[dst], hi[dst] = self.lo[keep], self.hi[keep]
        return ColumnObstacles(starts, lo, hi, self.xs, name=name or self.name, info=self.info)
