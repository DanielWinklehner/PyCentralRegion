"""
polyprism.py - Planar polygon unions and graded tall-wall prisms via gmsh/OCC.

The grounded hub of ``gap_fields.build_gap_electrodes`` is normally a polar
star-shaped ring (circular post or trajectory-following scroll) whose caps are
meshed by concentric shrinking. The spiral-inflector HOUSING, used as the
scroll since 2026-09-10, is an arbitrary simple polygon in the median plane
(a C-shaped spiral band with the exit plate and its opening) onto which the
dummy-dee spokes merge. This module provides the two operations that need:

``union_polygons``
    Boolean union of closed polygons (OCC fuse), returned as planar faces with
    their outer loop, hole loops and a triangulation of the cap whose boundary
    nodes are exactly the loop nodes. Boundary edges are kept at the caller's
    discretisation (the input polygons are densified first and gmsh meshes
    every edge with one element).
``mesh_prism``
    Extrudes such faces between the caller's z levels: boundary nodes carry a
    full column of levels shared by caps and walls (graded rows as in
    ``gap_fields._mesh_wedge``), interior cap nodes exist on the two caps only.
    The result is a watertight closed surface per face.

No shapely / earcut dependency: gmsh (already required by electrodes3d) does
both the boolean and the cap triangulation. Units: metres in, metres out
(gmsh works in mm internally, see ``MM``).
"""
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MM = 1000.0


# ============================================================================
# Polygon utilities (pure numpy)
# ============================================================================
def clean_polygon(poly: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Open (K, 2) copy of a polygon: closing duplicate and consecutive
    near-duplicates (< ``tol``) removed."""
    p = np.asarray(poly, dtype=float)[:, :2]
    if len(p) > 1 and np.hypot(*(p[0] - p[-1])) < tol:
        p = p[:-1]
    keep = [0]
    for i in range(1, len(p)):
        if np.hypot(*(p[i] - p[keep[-1]])) >= tol:
            keep.append(i)
    p = p[keep]
    if len(p) > 1 and np.hypot(*(p[0] - p[-1])) < tol:
        p = p[:-1]
    if len(p) < 3:
        raise ValueError("polygon has fewer than 3 distinct vertices")
    return p


def densify_polygon(poly: np.ndarray, ds: float) -> np.ndarray:
    """Open (K, 2) polygon with every edge split so that no edge exceeds ``ds``
    (original vertices kept)."""
    p = clean_polygon(poly)
    out = []
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        n = max(int(np.ceil(np.hypot(*(b - a)) / ds)), 1)
        for k in range(n):
            out.append(a + (b - a) * k / n)
    return np.asarray(out)


def simplify_polygon(poly: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker simplification of a closed polygon: drop vertices that
    deviate less than ``tol`` [m] from the chord of their neighbours. The
    two vertices farthest apart are always kept (they split the ring into
    two open chains). Use before ``densify_polygon`` to coarsen a finely
    sampled outline (a STEP section at ~1 mm) to a mesh-friendly spacing."""
    p = clean_polygon(poly)
    n = len(p)
    if n <= 4:
        return p
    i0 = 0
    d = np.hypot(*(p - p[i0]).T)
    i1 = int(np.argmax(d))

    def rdp(a, b):
        """Indices to keep strictly between a and b (open chain a..b, a < b)."""
        if b - a < 2:
            return []
        seg = p[b] - p[a]
        L = np.hypot(*seg)
        q = p[a + 1:b] - p[a]
        dist = np.abs(q[:, 0] * seg[1] - q[:, 1] * seg[0]) / L if L > 0 else np.hypot(*q.T)
        k = int(np.argmax(dist))
        if dist[k] < tol:
            return []
        m = a + 1 + k
        return rdp(a, m) + [m] + rdp(m, b)

    keep = [i0] + rdp(i0, i1) + [i1]
    # second chain: i1 .. n-1, 0 (wrap) -> shift so that it is a plain index range
    shifted = np.vstack([p[i1:], p[:i0 + 1]])
    p_shift = shifted

    def rdp2(a, b):
        if b - a < 2:
            return []
        seg = p_shift[b] - p_shift[a]
        L = np.hypot(*seg)
        q = p_shift[a + 1:b] - p_shift[a]
        dist = np.abs(q[:, 0] * seg[1] - q[:, 1] * seg[0]) / L if L > 0 else np.hypot(*q.T)
        k = int(np.argmax(dist))
        if dist[k] < tol:
            return []
        m = a + 1 + k
        return rdp2(a, m) + [m] + rdp2(m, b)

    keep2 = [(i1 + j) % n for j in rdp2(0, len(p_shift) - 1)]
    out = p[sorted(set(keep + keep2))]
    return out if len(out) >= 3 else p


def signed_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_contains(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path as MplPath
    return MplPath(np.asarray(poly, dtype=float)).contains_points(np.atleast_2d(pts))


def nest_loops(loops: Sequence[np.ndarray]) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
    """Group closed loops into (outer, [holes]) by even-odd nesting depth.

    A loop enclosed by an odd number of other loops is a hole of its
    immediate parent; deeper islands become outers of their own."""
    loops = [clean_polygon(l) for l in loops]
    n = len(loops)
    depth = np.zeros(n, dtype=int)
    parent = -np.ones(n, dtype=int)
    for i in range(n):
        best_area = np.inf
        for j in range(n):
            if i == j:
                continue
            if polygon_contains(loops[j], loops[i][:1])[0]:
                depth[i] += 1
                a = abs(signed_area(loops[j]))
                if a < best_area:
                    best_area, parent[i] = a, j
    out = []
    for i in range(n):
        if depth[i] % 2 == 0:
            holes = [loops[k] for k in range(n) if parent[k] == i and depth[k] == depth[i] + 1]
            out.append((loops[i], holes))
    return out


def _chain_segments(segs: np.ndarray, pts: np.ndarray) -> List[List[int]]:
    """Order (E, 2) node-index segments into closed loops (lists of node ids)."""
    adj: Dict[int, List[int]] = {}
    for a, b in segs:
        if a == b:
            continue
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    seen = set()
    loops = []
    for start in adj:
        if start in seen:
            continue
        chain = [start]
        seen.add(start)
        cur, prev = start, None
        while True:
            nxt = [v for v in adj[cur] if v != prev and v not in seen]
            if not nxt:
                break
            cur, prev = nxt[0], cur
            seen.add(cur)
            chain.append(cur)
        if len(chain) >= 3:
            loops.append(chain)
    return loops


# ============================================================================
# gmsh session guard
# ============================================================================
class _GmshModel:
    """Context: a fresh gmsh model, initialising gmsh if nobody has."""

    def __init__(self, name: str = "polyprism", verbose: bool = False):
        self.name = name
        self.verbose = verbose
        self.owned = False

    def __enter__(self):
        import gmsh
        self.gmsh = gmsh
        if not gmsh.isInitialized():
            gmsh.initialize()
            self.owned = True
        gmsh.option.setNumber("General.Terminal", 1 if self.verbose else 0)
        gmsh.model.add(self.name)
        gmsh.model.setCurrent(self.name)
        return gmsh

    def __exit__(self, *exc):
        try:
            if self.owned:
                self.gmsh.finalize()
            else:
                self.gmsh.model.remove()
        except Exception:
            pass
        return False


# ============================================================================
# Union + cap triangulation
# ============================================================================
def union_polygons(polygons: Sequence, ds: float, cap_size: Optional[float] = None,
                   verbose: bool = False) -> List[dict]:
    """Fuse closed polygons [m] and triangulate the result.

    ``polygons``: sequence of (K, 2) closed outlines, or (outer, [holes])
    tuples. Every input outline is densified to edges <= ``ds``; the boundary
    of the union keeps that discretisation exactly (one mesh element per
    edge). ``cap_size`` is the interior triangle size of the cap
    triangulation (default 4 ds).

    Returns one dict per resulting face:
        nodes   (N, 2) [m]  all nodes of the face (boundary + interior)
        tris    (T, 3)      cap triangles (indices into nodes)
        loops   list of closed node-id lists (first = outer, then holes),
                each walked once around, NOT repeating the start
        outer   (K, 2) the outer loop coordinates
        holes   [(K, 2), ...] hole loop coordinates
    """
    if cap_size is None:
        cap_size = 4.0 * ds
    items = []
    for p in polygons:
        if isinstance(p, tuple):
            outer, holes = p
        else:
            outer, holes = p, []
        items.append((densify_polygon(outer, ds), [densify_polygon(h, ds) for h in holes]))

    with _GmshModel("polyprism_union", verbose) as gmsh:
        occ = gmsh.model.occ

        def add_loop(poly):
            pts = [occ.addPoint(x * MM, y * MM, 0.0) for x, y in poly]
            lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
            return occ.addCurveLoop(lines)

        faces = []
        for outer, holes in items:
            loops = [add_loop(outer)] + [add_loop(h) for h in holes]
            faces.append(occ.addPlaneSurface(loops))
        if len(faces) > 1:
            out, _ = occ.fuse([(2, faces[0])], [(2, f) for f in faces[1:]],
                              removeObject=True, removeTool=True)
            faces = [t for d, t in out if d == 2]
        occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05 * ds * MM)
        gmsh.option.setNumber("Mesh.MeshSizeMax", cap_size * MM)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)

        ntag, ncoord, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(ncoord).reshape(-1, 3)[:, :2] / MM
        remap = {int(t): i for i, t in enumerate(ntag)}
        result = []
        for f in faces:
            etypes, etags, enodes = gmsh.model.mesh.getElements(2, f)
            tris = []
            for ty, nodes in zip(etypes, enodes):
                if ty == 2:
                    tris.append(np.vectorize(remap.get)(np.asarray(nodes).reshape(-1, 3)))
            if not tris:
                continue
            tris = np.vstack(tris)
            segs = []
            for d, c in gmsh.model.getBoundary([(2, f)], oriented=False, recursive=False):
                et1, _, en1 = gmsh.model.mesh.getElements(1, abs(c))
                for ty, nodes in zip(et1, en1):
                    if ty == 1:
                        segs.append(np.vectorize(remap.get)(np.asarray(nodes).reshape(-1, 2)))
            segs = np.vstack(segs)
            loops = _chain_segments(segs, coords)
            if not loops:
                raise RuntimeError("union_polygons: face without a closed boundary")
            # local node numbering per face
            used = np.unique(np.concatenate([tris.ravel(), segs.ravel()]))
            local = -np.ones(len(coords), dtype=int)
            local[used] = np.arange(len(used))
            nodes = coords[used]
            tris_l = local[tris]
            loops_l = [[int(local[i]) for i in lp] for lp in loops]
            areas = [abs(signed_area(nodes[lp])) for lp in loops_l]
            order = np.argsort(areas)[::-1]
            loops_l = [loops_l[i] for i in order]
            result.append({'nodes': nodes, 'tris': tris_l, 'loops': loops_l,
                           'outer': nodes[loops_l[0]],
                           'holes': [nodes[lp] for lp in loops_l[1:]]})
    if verbose:
        print(f"[polyprism] union of {len(items)} polygons -> {len(result)} face(s): "
              + ", ".join(f"{len(r['outer'])} boundary nodes, {len(r['holes'])} hole(s), "
                          f"{len(r['tris'])} cap tris" for r in result))
    return result


# ============================================================================
# Prism mesh
# ============================================================================
def mesh_prism(faces: Sequence[dict], z_levels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Closed surfaces of the faces extruded over ``z_levels`` (ascending, [m]).

    Returns (vertices (N, 3), triangles (M, 3), face_of_triangle (M,)).
    Boundary nodes carry every z level (walls stitched to the caps), interior
    cap nodes only the two cap levels."""
    z_levels = np.asarray(z_levels, dtype=float)
    nz = len(z_levels)
    verts: List[Tuple[float, float, float]] = []
    tris: List[Tuple[int, int, int]] = []
    owner: List[int] = []
    for fi, face in enumerate(faces):
        nodes, cap, loops = face['nodes'], face['tris'], face['loops']
        on_boundary = np.zeros(len(nodes), dtype=bool)
        for lp in loops:
            on_boundary[lp] = True
        base = len(verts)
        col: Dict[int, List[int]] = {}
        top = np.full(len(nodes), -1, dtype=int)
        bot = np.full(len(nodes), -1, dtype=int)
        for i, (x, y) in enumerate(nodes):
            if on_boundary[i]:
                ids = []
                for z in z_levels:
                    verts.append((x, y, z))
                    ids.append(len(verts) - 1)
                col[i] = ids
                bot[i], top[i] = ids[0], ids[-1]
            else:
                verts.append((x, y, z_levels[0]))
                bot[i] = len(verts) - 1
                verts.append((x, y, z_levels[-1]))
                top[i] = len(verts) - 1
        n0 = len(tris)
        for a, b, c in cap:
            tris.append((top[a], top[b], top[c]))
            tris.append((bot[a], bot[c], bot[b]))
        for lp in loops:
            k = len(lp)
            for j in range(k):
                a, b = lp[j], lp[(j + 1) % k]
                ca, cb = col[a], col[b]
                for iz in range(nz - 1):
                    tris.append((ca[iz], cb[iz], cb[iz + 1]))
                    tris.append((ca[iz], cb[iz + 1], ca[iz + 1]))
        owner.extend([fi] * (len(tris) - n0))
    v = np.asarray(verts, dtype=float)
    t = np.asarray(tris, dtype=int)
    areas = 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
    bad = int(np.sum(areas < 1e-14))
    if bad:
        raise RuntimeError(f"mesh_prism: {bad} degenerate triangles")
    return v, t, owner


def watertight(vertices: np.ndarray, triangles: np.ndarray) -> bool:
    """Every edge shared by exactly two triangles (closed surface test)."""
    e = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))
