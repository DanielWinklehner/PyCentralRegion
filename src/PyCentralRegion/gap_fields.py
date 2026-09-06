"""
gap_fields.py - Real 2D electrostatic gap fields via bempp_cl (gap_model='bem2d').

Builds CLOSED metal solids from the RFCavity segment chains - dee wedges (metal
between the entry and exit gap of one dee) and ground / dummy-dee wedges (metal
between adjacent dees) - solves the Laplace Dirichlet problem with the
spyral_inflector solve_bempp pattern (DP0 space, single-layer operator, GMRES),
evaluates the potential on a midplane Cartesian grid, and packages
E = -grad(phi) as a PyPATools ``Field`` wrapped in a ``TimedField``:

    E(x, y, t) = E_static(x, y) * cos(omega * t + bunch_phase)

The per-gap 0/180 phase pattern of a dee system is folded into the STATIC
potential signs (dee metal at -V*cos(entry_phase), ground at 0), which is exact
for phases in {0, 180}: one static solve covers all gaps, and RF frequency /
bunch-phase changes only re-modulate - no re-solve. Only GEOMETRY changes (and
a change of the radial voltage-profile SHAPE, see below) invalidate a solution,
which is why attachment to a design is an explicit call
(``AcceleratedOrbitFinder.attach_bem_field``).

Radial voltage profile (``voltage_profile``): a resonant dee is not an
equipotential at RF - the gap voltage varies with radius (standing wave along
the dee / stem). The quasi-static model absorbs this as a radially varying
Dirichlet value on the dee metal, phi_dee(r) = -V*cos(entry_phase) * scale(r),
with the SHAPE scale(r) taken from the full RF cavity model (``VoltageProfile``:
an (r, V) table normalized to 1 at a reference radius, so the cavity ``voltage``
stays the peak voltage at that radius). Ground and post stay at 0. The
amplitude still scales freely; only the shape needs a re-solve.

2D model ("tall extrusion" limit): electrode footprints are extruded in z far
beyond the gap width and the field is evaluated in the median plane only. The
walls are FIELD SOURCES, not obstacles - the tracked orbit passes through the
wall surface where the real machine has the vertical dee aperture. This keeps
the integrated gap voltage exact (integral of E dl = V) and the in-plane fringe
real; aperture softening in z is 3D (roadmap items 5/6) physics.

Geometry notes:
 * Chains use the ACTUAL (rotated) segment endpoints; rotation-induced
   disconnects are bridged with straight edges - this re-imposes the electrode
   continuity that the thin-gap model ignores.
 * Chains are truncated below the radius where a wedge's two boundary chains
   come closer than ``min_metal_width`` (auto-computed; the converging gap
   lines near the machine center would otherwise self-intersect). First
   crossings happen well outside this radius.
 * Wedge caps are meshed with a polar ruled blend between the two boundary
   chains (radius and CCW azimuth interpolated), which handles any wedge span
   (a single-dee system has a ~315 deg ground wedge) without folding.

Part of: PyCentralRegion module. bempp_cl is imported lazily inside the solve
functions so it stays an optional dependency (first solve pays a one-time
numba JIT of ~13 s per process).
"""

import time
import warnings
from dataclasses import dataclass, field as dataclass_field
from typing import List, Optional, Tuple

import numpy as np

from PyPATools.field import Field, TimedField


# ============================================================================
# Polyline helpers
# ============================================================================
def gap_chain(cavity, merge_tol: float = 5e-4) -> np.ndarray:
    """Connected 2D centerline polyline of a gap from its (rotated) segments.

    Walks ``cavity.segments`` in order using the ACTUAL stored endpoints (which
    include midpoint rotations) and inserts bridge edges across the
    rotation-induced disconnects. Returns (K, 2) [m], inner to outer.

    Disconnects SHORTER than ``merge_tol`` (default 0.5 mm) are absorbed
    instead of bridged: a sub-mm bridge edge would propagate as a sliver
    column through every wall row and cap cell (0.2 mm next to ~12 mm), which
    was measured to stall the BEM GMRES solve. Absorbing it moves the metal
    edge by < merge_tol - far below mesh fidelity.
    """
    pts: List[np.ndarray] = []
    for seg in cavity.segments:
        p1 = np.asarray(seg['p1'][:2], dtype=float)
        p2 = np.asarray(seg['p2'][:2], dtype=float)
        if not pts:
            pts.append(p1)
        elif np.hypot(*(p1 - pts[-1])) > merge_tol:
            pts.append(p1)  # bridge edge across a rotation disconnect
        if np.hypot(*(p2 - pts[-1])) > merge_tol:
            pts.append(p2)
        else:
            pts[-1] = p2    # keep the chain's true endpoint
    if len(pts) < 2:
        raise ValueError("gap chain has fewer than 2 distinct points")
    return np.array(pts)


def offset_polyline(pts: np.ndarray, offset) -> np.ndarray:
    """Offset a polyline perpendicular in-plane with miter joins.

    Positive offset displaces to the LEFT of the direction of travel (for an
    inner-to-outer chain that is the CCW / higher-azimuth side). Miter length
    is capped at 3x|offset| for near-degenerate kinks. ``offset`` may be a
    scalar or a per-vertex array (length len(pts)) for variable-width offsets
    (tapered gaps).
    """
    d = np.diff(pts, axis=0)
    lengths = np.hypot(d[:, 0], d[:, 1])
    if np.any(lengths < 1e-12):
        raise ValueError("offset_polyline: degenerate (zero-length) edge")
    t = d / lengths[:, None]
    normals = np.column_stack([-t[:, 1], t[:, 0]])   # left normals per edge
    offs = np.broadcast_to(np.asarray(offset, dtype=float), (len(pts),))

    out = np.empty_like(pts)
    out[0] = pts[0] + offs[0] * normals[0]
    out[-1] = pts[-1] + offs[-1] * normals[-1]
    for i in range(1, len(pts) - 1):
        m = normals[i - 1] + normals[i]
        mn = np.hypot(*m)
        if mn < 1e-9:  # 180-degree kink; fall back to the next edge normal
            out[i] = pts[i] + offs[i] * normals[i]
            continue
        m /= mn
        denom = max(float(np.dot(m, normals[i])), 1.0 / 3.0)  # miter cap 3x
        out[i] = pts[i] + offs[i] * m / denom
    return out


def _densify_polyline_inner(pts: np.ndarray, r_limit: float,
                            max_ds: float) -> np.ndarray:
    """Subdivide edges of an inner-to-outer chain below radius ``r_limit``.

    Inserts a vertex exactly where an edge crosses ``r_limit`` (the taper
    kink) and resamples the sub-``r_limit`` parts at ``max_ds``. Beyond
    ``r_limit`` the chain is untouched (a linear-in-r offset of a straight
    edge is exact with vertices only at the kinks).
    """
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        ra = float(np.hypot(a[0], a[1]))
        rb = float(np.hypot(b[0], b[1]))
        if min(ra, rb) < r_limit:
            # vertex at the r_limit crossing (|a + t(b-a)| = r_limit)
            d = b - a
            aa = float(d @ d)
            bb = 2.0 * float(a @ d)
            cc = float(a @ a) - r_limit * r_limit
            disc = bb * bb - 4.0 * aa * cc
            t_cross = None
            if disc > 0.0 and aa > 0.0:
                for t in ((-bb - np.sqrt(disc)) / (2 * aa),
                          (-bb + np.sqrt(disc)) / (2 * aa)):
                    if 1e-9 < t < 1.0 - 1e-9:
                        t_cross = t
                        break
            length = float(np.hypot(d[0], d[1]))
            n_sub = max(int(np.ceil(length / max_ds)), 1)
            ts = list(np.linspace(0.0, 1.0, n_sub + 1)[1:])
            # Insert the kink vertex only when clear of existing vertices;
            # drop subdivision points that (nearly) collide with it - exact
            # collisions happen (e.g. kink at 7/29 of an edge) and produce
            # degenerate zero-length edges.
            if t_cross is not None:
                eps = min(0.25 / n_sub, 0.2)
                if eps < t_cross < 1.0 - eps:
                    ts = [t for t in ts
                          if abs(t - t_cross) > eps or t >= 1.0 - 1e-12]
                    ts = sorted(ts + [t_cross])
            for t in ts:
                out.append(a + t * d)
        else:
            out.append(b)
    return np.array(out)


def offset_gap_boundary(cavity, side: float, merge_tol: float = 5e-4) -> np.ndarray:
    """Boundary chain of a gap CHANNEL: centerline offset by ``side * w/2``.

    ``side`` is +1 for the CCW (left-of-travel) boundary, -1 for CW. With a
    tapered gap (``gap_width_inner`` / ``gap_taper_radius``) the offset uses
    the LOCAL width at each vertex radius; the chain is densified through the
    tapered span (with an exact vertex at the taper transition radius) so the
    piecewise-linear boundary follows the taper.
    """
    pts = gap_chain(cavity, merge_tol=merge_tol)
    if getattr(cavity, 'gap_width_inner', None) is None:
        return offset_polyline(pts, side * cavity.gap_width / 2.0)
    pts = _densify_polyline_inner(pts, cavity.gap_taper_radius, max_ds=0.005)
    radii = np.hypot(pts[:, 0], pts[:, 1])
    return offset_polyline(pts, side * 0.5 * cavity.gap_width_at(radii))


def _truncate_chain_inner(pts: np.ndarray, r_inner: float) -> np.ndarray:
    """Drop the part of an inner-to-outer chain below radius ``r_inner``.

    The first surviving point is interpolated to lie exactly at r_inner.
    """
    radii = np.hypot(pts[:, 0], pts[:, 1])
    if radii[0] >= r_inner:
        return pts
    outside = np.where(radii >= r_inner)[0]
    if len(outside) == 0:
        raise ValueError(f"chain lies entirely inside r_inner={r_inner * 1000:.1f} mm")
    k = int(outside[0])
    a, b = pts[k - 1], pts[k]
    # |a + s (b - a)| = r_inner, s in [0, 1]
    ab = b - a
    A = float(np.dot(ab, ab))
    B = 2.0 * float(np.dot(a, ab))
    C = float(np.dot(a, a)) - r_inner ** 2
    disc = max(B * B - 4 * A * C, 0.0)
    roots = [(-B + np.sqrt(disc)) / (2 * A), (-B - np.sqrt(disc)) / (2 * A)]
    s_ok = [s for s in roots if -1e-9 <= s <= 1 + 1e-9]
    s = min(max(min(s_ok) if s_ok else 0.0, 0.0), 1.0)
    p_cross = a + s * ab
    return np.vstack([p_cross, pts[k:]])


def _extend_chain_inner(pts: np.ndarray, r_to: float) -> np.ndarray:
    """Extend an inner-to-outer chain radially inward to radius ``r_to``.

    Prepends one point at ``r_to`` along the chain start's azimuth (a radial
    spoke edge). No-op if the chain already starts at or below ``r_to``.
    """
    p0 = pts[0]
    r0 = float(np.hypot(p0[0], p0[1]))
    if r0 <= r_to + 1e-12:
        return pts
    return np.vstack([p0 * (r_to / r0), pts])


def _fillet_polyline(pts: np.ndarray, radius: float,
                     angle_min_deg: float = 8.0,
                     max_edge_frac: float = 0.4) -> np.ndarray:
    """Replace interior kinks sharper than ``angle_min_deg`` with tangent arcs.

    Classic corner fillet: tangent offsets t = R tan(turn/2) along both edges,
    clamped to ``max_edge_frac`` of each adjacent edge (two fillets can share
    an edge without overlapping); the radius shrinks with the clamp. Arcs are
    sampled at >= ~1 mm chords so the fillet never introduces the sub-mm
    vertex spacings that create sliver mesh columns.
    """
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        p_prev, p, p_next = pts[i - 1], pts[i], pts[i + 1]
        v1 = p - p_prev
        v2 = p_next - p
        l1, l2 = np.hypot(*v1), np.hypot(*v2)
        if l1 < 1e-12 or l2 < 1e-12:
            out.append(p)
            continue
        u1, u2 = v1 / l1, v2 / l2
        turn = float(np.arccos(np.clip(u1 @ u2, -1.0, 1.0)))
        if turn < np.deg2rad(angle_min_deg) or turn > np.pi - 1e-3:
            out.append(p)
            continue
        t = radius * np.tan(turn / 2.0)
        t_max = max_edge_frac * min(l1, l2)
        if t > t_max:
            t = t_max
        r_eff = t / np.tan(turn / 2.0)
        a = p - t * u1                       # tangent point on incoming edge
        b = p + t * u2                       # tangent point on outgoing edge
        # arc center: perpendicular to u1 at a, on the turning side
        sign = np.sign(u1[0] * u2[1] - u1[1] * u2[0])
        n1 = sign * np.array([-u1[1], u1[0]])
        c = a + r_eff * n1
        ang_a = np.arctan2(a[1] - c[1], a[0] - c[0])
        n_seg = int(np.clip(np.ceil(turn / np.deg2rad(30.0)), 1, 8))
        for j in range(n_seg + 1):
            ang = ang_a + sign * turn * j / n_seg
            out.append(c + r_eff * np.array([np.cos(ang), np.sin(ang)]))
    out.append(pts[-1])
    return np.array(out)


def _turn1_profile(trajectory):
    """Turn-1 radius profile of a tracked trajectory.

    Returns (r_at(pts) -> radius of turn 1 at each point's azimuth,
    prog_of(pts) -> azimuthal progress in [0, 2 pi) from the injection
    azimuth in the direction of travel, scroll_xy(prog) factory input
    (phi0, s_dir, u1, r1)).
    """
    xy = np.asarray(trajectory, dtype=float)[:, :2]
    phi = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    if abs(phi[-1] - phi[0]) < 2.0 * np.pi - 1e-6:
        raise ValueError("trim_trajectory must cover at least one full turn")
    s_dir = 1.0 if phi[-1] > phi[0] else -1.0
    u = np.maximum.accumulate(s_dir * (phi - phi[0]))
    r = np.hypot(xy[:, 0], xy[:, 1])
    m = u <= 2.0 * np.pi
    u1, r1 = u[m], r[m]
    if u1[-1] < 2.0 * np.pi:
        u1 = np.append(u1, 2.0 * np.pi)
        r1 = np.append(r1, float(np.interp(2.0 * np.pi, u, r)))
    phi0 = float(phi[0])

    def prog_of(pts):
        ph = np.arctan2(pts[:, 1], pts[:, 0])
        return np.mod(s_dir * (ph - phi0), 2.0 * np.pi)

    def r_at(pts):
        return np.interp(prog_of(pts), u1, r1)

    return r_at, prog_of, phi0, s_dir, u1, r1


def _truncate_chain_at_curve(pts: np.ndarray, cut_at,
                             floor_r: float = 0.0) -> np.ndarray:
    """Trim an inner-to-outer chain where it first crosses a radius curve.

    ``cut_at(points (M,2)) -> (M,) cut radius``; the effective cut is
    max(cut, floor_r). The crossing point is found by bisection on the
    first crossing edge; original vertices are kept beyond it.
    """
    def clearance(p):
        return float(np.hypot(p[0], p[1])
                     - max(float(cut_at(p[None, :])[0]), floor_r))

    if clearance(pts[0]) >= 0.0:
        return pts
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        n = max(int(np.ceil(np.hypot(*(b - a)) / 0.002)), 1)
        ts = np.linspace(0.0, 1.0, n + 1)
        pos = next((i for i, t in enumerate(ts)
                    if clearance(a + t * (b - a)) >= 0.0), None)
        if pos is None:
            continue
        t_lo, t_hi = (ts[pos - 1], ts[pos]) if pos > 0 else (0.0, 0.0)
        for _ in range(30):
            tm = 0.5 * (t_lo + t_hi)
            if clearance(a + tm * (b - a)) >= 0.0:
                t_hi = tm
            else:
                t_lo = tm
        return np.vstack([a + t_hi * (b - a), pts[k + 1:]])
    raise ValueError("chain lies entirely inside the trim curve")


def _first_polyline_intersection(chain: np.ndarray, poly: np.ndarray):
    """First intersection of ``chain`` with ``poly``, walking from chain[0].

    Returns (i_seg, t, J) - chain segment index, fraction along it, and the
    intersection point - or None.
    """
    c0 = poly[:-1]
    d2 = np.diff(poly, axis=0)
    for i in range(len(chain) - 1):
        a = chain[i]
        d1 = chain[i + 1] - a
        denom = d1[0] * d2[:, 1] - d1[1] * d2[:, 0]      # d1 x d2
        ok = np.abs(denom) > 1e-18
        if not ok.any():
            continue
        rel = c0 - a                                      # (N, 2)
        safe = np.where(ok, denom, 1.0)
        t = np.where(ok, (rel[:, 0] * d2[:, 1] - rel[:, 1] * d2[:, 0]) / safe, 2.0)
        u = np.where(ok, (rel[:, 0] * d1[1] - rel[:, 1] * d1[0]) / safe, 2.0)
        hit = ok & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
        if hit.any():
            tt = float(np.min(t[hit]))
            return i, tt, a + tt * d1
    return None


def _clamp_chain_out_of_hub(pts: np.ndarray, prog_of, scroll_r, phi0: float,
                            s_dir: float, step_prog: float,
                            setback_arc: float) -> np.ndarray:
    """Move chain points that fall INSIDE the scroll hub onto the step face.

    A chain sweeping azimuthally past the scroll termination would run
    buried inside the hub's outer branch (and leave an uncovered sliver
    along the termination face). Offending points keep their radius but are
    clamped just DOWNSTREAM of the step face (progress step_prog +
    setback_arc/r) - the wedge boundary then runs parallel to the face with
    a small slot.
    """
    rr = np.hypot(pts[:, 0], pts[:, 1])
    inside = rr < scroll_r(prog_of(pts)) - 1e-9
    if not inside.any():
        return pts
    out = pts.copy()
    for i in np.where(inside)[0]:
        az = phi0 + s_dir * (step_prog + setback_arc / max(rr[i], 1e-3))
        out[i] = rr[i] * np.array([np.cos(az), np.sin(az)])
    return out


def _scroll_ring(arcs: List[np.ndarray], scroll_xy, prog_of, post_ds: float,
                 step_prog: Optional[float], seam: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, List[bool], List[bool]]:
    """Rim of the scroll hub: spoke seam arcs + window fills + closure.

    ``scroll_xy(prog) -> (2,)`` evaluates the scroll boundary; ``arcs`` are
    the merged ground wedges' row-0 arcs (verbatim seam coordinates).

    Closure modes:
    * ``seam`` given: the spiral runs up to the junction (seam[0]) and the
      rim CLOSES along the wrap dummy-dee's sampled chain (seam, junction ->
      tip corner) - interior seam edges shared verbatim with that wedge, so
      hub and dummy-dee form one contiguous area. seam[-1] must equal the
      first point of that wedge's row-0 arc, which becomes the ring start.
    * an arc spans the turn wrap (its end prog < start prog - the wrap
      dummy-dee's seam arc): that arc IS the closure - the spiral runs up to
      its start and the rim continues along it across the wrap.
    * ``step_prog`` given (and no wrap-spanning arc): radial step face there,
      then an inner-branch hold back to the wrap (profile continuous across
      the wrap).
    """
    ring_pts: List[np.ndarray] = []
    is_wall: List[bool] = []
    seam_edge: List[bool] = []

    def add(p, wall):
        ring_pts.append(np.asarray(p, dtype=float))
        is_wall.append(wall)
        seam_edge.append(False)

    def fill_smooth(p_from, p_to):
        if p_to - p_from < 1e-9:
            return
        r_avg = float(np.hypot(*scroll_xy(0.5 * (p_from + p_to))))
        n = max(int(np.ceil((p_to - p_from) * r_avg / post_ds)), 1)
        for i in range(n - 1):
            add(scroll_xy(p_from + (p_to - p_from) * (i + 1) / n), True)

    def fill(p_from, p_to):
        if step_prog is not None and p_from < step_prog < p_to:
            fill_smooth(p_from, step_prog)
            c_hi = scroll_xy(step_prog)          # outer corner of the face
            c_lo = scroll_xy(step_prog + 1e-9)   # hold-side corner
            add(c_hi, True)
            n = max(int(np.ceil(np.hypot(*(c_lo - c_hi)) / post_ds)), 1)
            for i in range(1, n):
                add(c_hi + (c_lo - c_hi) * i / n, True)
            add(c_lo, True)
            fill_smooth(step_prog + 1e-9, p_to)
        else:
            fill_smooth(p_from, p_to)

    def add_arc(arc, first_wall=True):
        for i, p in enumerate(arc):
            ring_pts.append(np.asarray(p, dtype=float))
            is_wall.append((i == 0 and first_wall) or i == len(arc) - 1)
            seam_edge.append(i < len(arc) - 1)

    arcs = sorted(arcs, key=lambda a: float(prog_of(a[:1])[0]))

    if seam is not None:
        # rotate so the wrap dummy-dee's arc (whose first point closes the
        # seam) starts the ring; its tip corner is fully seam (no wall)
        k0 = int(np.argmin([np.hypot(*(a[0] - seam[-1])) for a in arcs]))
        if np.hypot(*(arcs[k0][0] - seam[-1])) > 1e-9:
            raise RuntimeError("scroll seam does not land on a spoke arc")
        arcs = arcs[k0:] + arcs[:k0]
        prog_j = float(prog_of(seam[:1])[0])
        add_arc(arcs[0], first_wall=False)
        prev = float(prog_of(arcs[0][-1:])[0])
        for arc in arcs[1:]:
            fill_smooth(prev, float(prog_of(arc[:1])[0]))
            add_arc(arc)
            prev = float(prog_of(arc[-1:])[0])
        fill_smooth(prev, prog_j)
        # closure along the dummy-dee edge: junction -> tip corner; the last
        # seam vertex IS ring_pts[0], so stop one short and mark the cyclic
        # closing edge as seam
        for i, p in enumerate(seam[:-1]):
            ring_pts.append(np.asarray(p, dtype=float))
            is_wall.append(i == 0)               # junction joins the wall rim
            seam_edge.append(True)
        return _checked_ring(ring_pts, is_wall, seam_edge)

    p0 = [float(prog_of(a[:1])[0]) for a in arcs]
    p1 = [float(prog_of(a[-1:])[0]) for a in arcs]
    wrap = [k for k in range(len(arcs)) if p1[k] < p0[k]]
    if wrap:
        # An arc spans the turn wrap (the wrap dummy-dee's seam arc, trimmed
        # onto the outer spiral on one side and near the injection azimuth on
        # the other): that arc IS the closure. Treating it as a normal spoke
        # would leave prev ~ 0 after it and lay a SECOND full spiral wrap
        # (plus the step face) on top of the whole ring - a self-intersecting
        # rim with doubled wall/cap sheets that stalls the BEM GMRES solve.
        if len(wrap) > 1 or wrap[0] != len(arcs) - 1:
            raise RuntimeError("scroll ring: unexpected wrap-spanning spoke "
                               "arcs (CW beam?) - geometry inspection needed")
        add_arc(arcs[-1])
        prev = p1[-1]
        for k, arc in enumerate(arcs[:-1]):
            fill_smooth(prev, p0[k])
            add_arc(arc)
            prev = p1[k]
        fill_smooth(prev, p0[-1])
        # cyclic closing edge lands back on the wrap arc's start corner
        return _checked_ring(ring_pts, is_wall, seam_edge)

    add(scroll_xy(0.0), True)                    # ring anchor (wrap is smooth)
    prev = 0.0
    for arc in arcs:
        fill(prev, float(prog_of(arc[:1])[0]))
        add_arc(arc)
        prev = float(prog_of(arc[-1:])[0])
    fill(prev, 2.0 * np.pi)
    # cyclic closing edge back to the anchor: profile continuous at the wrap
    return _checked_ring(ring_pts, is_wall, seam_edge)


def _checked_ring(ring_pts: List[np.ndarray], is_wall: List[bool],
                  seam_edge: List[bool]
                  ) -> Tuple[np.ndarray, List[bool], List[bool]]:
    """Assemble the rim polygon and verify it is simple (no self-crossing).

    A self-intersecting rim extrudes into interpenetrating wall/cap sheets,
    which makes the first-kind BEM system near-singular (GMRES stalls at any
    iteration count) - fail loudly at build time instead.
    """
    ring = np.array(ring_pts)
    a0 = ring
    d = np.roll(ring, -1, axis=0) - ring
    for i in range(len(ring)):
        denom = d[i, 0] * d[:, 1] - d[i, 1] * d[:, 0]
        ok = np.abs(denom) > 1e-18
        rel = a0 - a0[i]
        safe = np.where(ok, denom, 1.0)
        t = np.where(ok, (rel[:, 0] * d[:, 1] - rel[:, 1] * d[:, 0]) / safe, -1.0)
        u = np.where(ok, (rel[:, 0] * d[i, 1] - rel[:, 1] * d[i, 0]) / safe, -1.0)
        hit = ok & (t > 1e-9) & (t < 1 - 1e-9) & (u > 1e-9) & (u < 1 - 1e-9)
        hit[i] = False
        if hit.any():
            j = int(np.where(hit)[0][0])
            p = a0[i] + t[j] * d[i]
            raise RuntimeError(
                f"scroll hub rim self-intersects at ({p[0] * 1000:.1f}, "
                f"{p[1] * 1000:.1f}) mm - the hub would interpenetrate the "
                f"electrodes and stall the BEM solve; geometry inspection "
                f"needed")
    return ring, is_wall, seam_edge


def _mesh_scroll_hub(ring: np.ndarray, is_wall: List[bool],
                     seam_edge: List[bool], z_levels: np.ndarray,
                     post_ds: float
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed surface of the scroll hub (rim walls + concentric-shrink caps).

    Same construction as the circular hub, for an arbitrary star-shaped rim.
    Cap bands from radial rim edges (the step) collapse to zero area and are
    dropped - the region they would cover is zero-area by construction.
    """
    n_ring = len(ring)
    nz = len(z_levels)
    z_bot, z_top = z_levels[0], z_levels[-1]
    verts: List[Tuple[float, float, float]] = []

    def add(x, y, z):
        verts.append((x, y, z))
        return len(verts) - 1

    columns = {}
    rim_top = np.empty(n_ring, dtype=int)
    rim_bot = np.empty(n_ring, dtype=int)
    for i, p in enumerate(ring):
        if is_wall[i]:
            ids = [add(p[0], p[1], z) for z in z_levels]
            columns[i] = ids
            rim_bot[i], rim_top[i] = ids[0], ids[-1]
        else:
            rim_bot[i] = add(p[0], p[1], z_bot)
            rim_top[i] = add(p[0], p[1], z_top)

    r_max = float(np.max(np.hypot(ring[:, 0], ring[:, 1])))
    n_r = max(int(np.ceil(r_max / post_ds)), 2)
    fracs = np.linspace(1.0, 0.0, n_r + 1)[1:-1]
    rings_top, rings_bot = [rim_top], [rim_bot]
    for f in fracs:
        rings_top.append(np.array([add(f * p[0], f * p[1], z_top) for p in ring]))
        rings_bot.append(np.array([add(f * p[0], f * p[1], z_bot) for p in ring]))
    c_top = add(0.0, 0.0, z_top)
    c_bot = add(0.0, 0.0, z_bot)

    tris: List[Tuple[int, int, int]] = []
    for lt, lb, nt, nb in zip(rings_top[:-1], rings_bot[:-1],
                              rings_top[1:], rings_bot[1:]):
        for i in range(n_ring):
            j = (i + 1) % n_ring
            tris += [(lt[i], lt[j], nt[i]), (lt[j], nt[j], nt[i]),
                     (lb[i], nb[i], lb[j]), (lb[j], nb[i], nb[j])]
    for i in range(n_ring):
        j = (i + 1) % n_ring
        tris += [(rings_top[-1][i], rings_top[-1][j], c_top),
                 (rings_bot[-1][j], rings_bot[-1][i], c_bot)]
    for i in range(n_ring):
        j = (i + 1) % n_ring
        if seam_edge[i]:
            continue
        ca, cb = columns[i], columns[j]
        for iz in range(nz - 1):
            a, b, c, d = ca[iz], cb[iz], cb[iz + 1], ca[iz + 1]
            tris += [(a, b, c), (a, c, d)]

    V = np.array(verts)
    T = np.array(tris, dtype=int)
    areas = 0.5 * np.linalg.norm(
        np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]), axis=1)
    return V, T[areas > 1e-13], ring


def warn_if_trajectory_hits_post(model: 'ElectrodeModel', trajectory,
                                 label: str = "tracked orbit") -> float:
    """Warn (warnings module + stdout) if the trajectory enters the post.

    Returns the minimum in-plane clearance [m] to the central post footprint
    (negative if any trajectory point lies inside). No-op (inf) without a
    post/scroll.
    """
    posts = [w for w in model.wedges if w.kind == 'post']
    if not posts:
        return float('inf')
    from matplotlib.path import Path as MplPath
    poly = np.asarray(posts[0].polygon, dtype=float)
    xy = np.asarray(trajectory, dtype=float)[:, :2]
    # the beam legitimately GRAZES the scroll step face as it emerges from
    # the inflector exit aperture: exclude the first ~15 deg of travel
    phi = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    xy = xy[np.abs(phi - phi[0]) > np.deg2rad(15.0)]
    if len(xy) == 0:
        return float('inf')
    inside = MplPath(poly).contains_points(xy)
    sub = xy[::max(len(xy) // 4000, 1)]
    sub_inside = MplPath(poly).contains_points(sub)
    a = poly
    b = np.roll(poly, -1, axis=0)
    d = b - a
    lens2 = np.maximum(np.sum(d * d, axis=1), 1e-18)

    def dist(p):
        t = np.clip(np.sum((p - a) * d, axis=1) / lens2, 0.0, 1.0)
        proj = a + t[:, None] * d
        return float(np.min(np.hypot(proj[:, 0] - p[0], proj[:, 1] - p[1])))

    if inside.any():
        depth = max((dist(p) for p in sub[sub_inside]), default=0.0)
        msg = (f"[gap_fields] WARNING: {label} enters the central post "
               f"footprint ({int(inside.sum())} points, depth up to "
               f"~{depth * 1000:.1f} mm) - the beam would hit the plug. "
               f"Regenerate the scroll from this trajectory or increase "
               f"the clearances.")
        warnings.warn(msg)
        print(msg)
        return -depth
    return min(dist(p) for p in sub)


def _mesh_post_hub(arcs: List[np.ndarray], radius: float, z_levels: np.ndarray,
                   post_ds: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grounded central hub: a disk whose rim is STITCHED to the ground spokes.

    ``arcs`` are the ground wedges' inner (row-0) arcs, each lying exactly on
    the circle of ``radius`` - the hub reuses their coordinates verbatim, so
    hub caps and spoke caps share the seam polylines and the union is ONE
    contiguous closed surface. Rim walls are meshed only across the azimuthal
    WINDOWS between consecutive spokes; the spoke spans are interior seams
    (the spokes are meshed with ``open_inner``).

    Meshed finer than the wedge arcs (post_ds): the exterior field at the rim
    is the full dee-tip fringe, and coarse panels leak a percent-level ghost
    of it into the interior that the hub exists to shield.

    Returns (vertices (N,3), triangles (M,3), footprint circle (P,2)).
    """
    arcs = sorted(arcs, key=lambda a: np.mod(np.arctan2(a[0, 1], a[0, 0]),
                                             2.0 * np.pi))
    ring_pts: List[np.ndarray] = []
    is_wall: List[bool] = []      # vertex carries a full z column (rim wall)
    seam_edge: List[bool] = []    # edge i -> i+1 is a spoke seam (no wall)

    def _az(p):
        return float(np.arctan2(p[1], p[0]))

    for j, arc in enumerate(arcs):
        for i, p in enumerate(arc):
            ring_pts.append(np.asarray(p, dtype=float))
            is_wall.append(i == 0 or i == len(arc) - 1)  # spoke corners
            seam_edge.append(i < len(arc) - 1)           # edges inside the arc
        nxt = arcs[(j + 1) % len(arcs)]
        az0, az1 = _az(arc[-1]), _az(nxt[0])
        span = np.mod(az1 - az0, 2.0 * np.pi)
        n_fill = max(int(np.ceil(span * radius / post_ds)) - 1, 0)
        for i in range(n_fill):
            az = az0 + span * (i + 1) / (n_fill + 1)
            ring_pts.append(radius * np.array([np.cos(az), np.sin(az)]))
            is_wall.append(True)
            seam_edge.append(False)
        # seam_edge for the edge leaving the last fill point (or the arc end)
        # toward the next arc start is already False (window edge)
    ring = np.array(ring_pts)
    n_ring = len(ring)
    nz = len(z_levels)
    z_bot, z_top = z_levels[0], z_levels[-1]

    verts: List[Tuple[float, float, float]] = []

    def add(x, y, z):
        verts.append((x, y, z))
        return len(verts) - 1

    # rim vertices: full z columns where a wall exists, top/bottom otherwise
    columns = {}
    rim_top = np.empty(n_ring, dtype=int)
    rim_bot = np.empty(n_ring, dtype=int)
    for i, p in enumerate(ring):
        if is_wall[i]:
            ids = [add(p[0], p[1], z) for z in z_levels]
            columns[i] = ids
            rim_bot[i], rim_top[i] = ids[0], ids[-1]
        else:
            rim_bot[i] = add(p[0], p[1], z_bot)
            rim_top[i] = add(p[0], p[1], z_top)

    # concentric cap rings (same azimuth sampling, scaled) + center point
    n_r = max(int(np.ceil(radius / post_ds)), 2)
    fracs = np.linspace(1.0, 0.0, n_r + 1)[1:-1]
    rings_top, rings_bot = [rim_top], [rim_bot]
    for f in fracs:
        rt = np.array([add(f * p[0], f * p[1], z_top) for p in ring])
        rb = np.array([add(f * p[0], f * p[1], z_bot) for p in ring])
        rings_top.append(rt)
        rings_bot.append(rb)
    c_top = add(0.0, 0.0, z_top)
    c_bot = add(0.0, 0.0, z_bot)

    tris: List[Tuple[int, int, int]] = []
    for lt, lb, nt, nb in zip(rings_top[:-1], rings_bot[:-1],
                              rings_top[1:], rings_bot[1:]):
        for i in range(n_ring):
            j = (i + 1) % n_ring
            tris += [(lt[i], lt[j], nt[i]), (lt[j], nt[j], nt[i]),
                     (lb[i], nb[i], lb[j]), (lb[j], nb[i], nb[j])]
    for i in range(n_ring):   # center fans
        j = (i + 1) % n_ring
        tris += [(rings_top[-1][i], rings_top[-1][j], c_top),
                 (rings_bot[-1][j], rings_bot[-1][i], c_bot)]
    # rim walls across the windows only
    for i in range(n_ring):
        j = (i + 1) % n_ring
        if seam_edge[i]:
            continue
        ca, cb = columns[i], columns[j]
        for iz in range(nz - 1):
            a, b, c, d = ca[iz], cb[iz], cb[iz + 1], ca[iz + 1]
            tris += [(a, b, c), (a, c, d)]

    return np.array(verts), np.array(tris, dtype=int), ring


def _arclength_fractions(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    seg = np.hypot(*(np.diff(pts, axis=0).T))
    cl = np.concatenate([[0.0], np.cumsum(seg)])
    return cl / cl[-1], float(cl[-1])


def _sample_polyline(pts: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Sample a polyline at normalized arclength fractions (exact on vertices)."""
    seg = np.hypot(*(np.diff(pts, axis=0).T))
    cl = np.concatenate([[0.0], np.cumsum(seg)])
    target = fractions * cl[-1]
    x = np.interp(target, cl, pts[:, 0])
    y = np.interp(target, cl, pts[:, 1])
    return np.column_stack([x, y])


def _matched_fractions(chain_a: np.ndarray, chain_b: np.ndarray, ds: float) -> np.ndarray:
    """Shared arclength-fraction breakpoints for two chains.

    Union of both chains' vertex fractions (kinks preserved on both), with any
    interval longer than ``ds`` (on the longer chain) subdivided uniformly.
    """
    fa, la = _arclength_fractions(chain_a)
    fb, lb = _arclength_fractions(chain_b)
    base = np.unique(np.concatenate([fa, fb]))
    # Merge breakpoints closer than ~4% of a mesh cell: kinks from the two
    # chains landing at nearly the same fraction would otherwise create
    # sliver columns (bad BEM conditioning). Shifts a kink by < 0.04 * ds.
    min_sep = 0.04 * ds / max(la, lb)
    keep = [base[0]]
    for f in base[1:]:
        if f - keep[-1] > min_sep:
            keep.append(f)
    if keep[-1] < 1.0:
        keep[-1] = 1.0
    base = np.array(keep)

    l_max = max(la, lb)
    out = [base[0]]
    for f0, f1 in zip(base[:-1], base[1:]):
        n = max(int(np.ceil((f1 - f0) * l_max / ds)), 1)
        for i in range(1, n):
            out.append(f0 + (f1 - f0) * i / n)
        out.append(f1)
    return np.array(out)


def _azimuths_unwrapped(pts: np.ndarray) -> np.ndarray:
    return np.unwrap(np.arctan2(pts[:, 1], pts[:, 0]))


def _ccw_width(chain_lo: np.ndarray, chain_hi: np.ndarray,
               radii: np.ndarray) -> np.ndarray:
    """Arc width [m] of the wedge between two chains at the given radii.

    The CCW span is anchored at the chains' OUTER ends (wrapped into [0, 2 pi))
    and continued inward via each chain's unwrapped azimuth, so it stays
    correct for any wedge span and goes NEGATIVE if the chains cross.

    Chains are densified first: the azimuth of an OFFSET line varies like
    arcsin(offset / r) along a long straight segment, so interpolating between
    the sparse polyline vertices grossly overestimates the inner-radius span
    (measured: r_inner 87 mm instead of 18 mm on radial chains from r=5 mm).
    """
    def az_at(chain, r):
        _, length = _arclength_fractions(chain)
        n = max(int(np.ceil(length / 0.002)) + 1, len(chain))
        dense = _sample_polyline(chain, np.linspace(0.0, 1.0, n))
        rr = np.hypot(dense[:, 0], dense[:, 1])
        az = _azimuths_unwrapped(dense)
        rr_mono = np.maximum.accumulate(rr)
        return np.interp(r, rr_mono, az), az[-1]

    az_lo, az_lo_end = az_at(chain_lo, radii)
    az_hi, az_hi_end = az_at(chain_hi, radii)
    d_ref = np.mod(az_hi_end - az_lo_end, 2.0 * np.pi)
    d = d_ref + (az_hi - az_hi_end) - (az_lo - az_lo_end)
    return d * radii


# ============================================================================
# Wedge construction + meshing
# ============================================================================
@dataclass
class Wedge:
    """One closed metal region (footprint) with a (reference) potential."""
    kind: str                 # 'dee', 'ground' or 'post'
    potential: float          # static Dirichlet value [V]; with a voltage_profile
                              # the dee value AT r_ref (elements scale with r)
    chain_lo: np.ndarray      # (K, 2) boundary on the low-azimuth side
    chain_hi: np.ndarray      # (K, 2) boundary on the high-azimuth side
    polygon: np.ndarray = None            # closed 2D outline (set at mesh time)
    label: str = ""


@dataclass
class ElectrodeModel:
    """Watertight triangle surface mesh of all wedges + per-element potentials."""
    vertices: np.ndarray      # (N, 3) [m]
    triangles: np.ndarray     # (M, 3) int
    potentials: np.ndarray    # (M,) [V]
    wedges: List[Wedge] = dataclass_field(default_factory=list)
    params: dict = dataclass_field(default_factory=dict)

    @property
    def n_elements(self) -> int:
        return len(self.triangles)

    def extent(self) -> float:
        """Max |x|, |y| over all vertices [m]."""
        return float(np.max(np.abs(self.vertices[:, :2])))


def _z_levels(height: float, dz0: float, growth: float) -> np.ndarray:
    """Symmetric z levels: ``dz0`` at the median plane, geometric growth out."""
    half = [0.0]
    dz = dz0
    while half[-1] < height / 2.0 - 1e-12:
        half.append(min(half[-1] + dz, height / 2.0))
        dz *= growth
    # merge a sliver top row left by the clip
    if len(half) >= 3 and (half[-1] - half[-2]) < 0.3 * (half[-2] - half[-3]):
        del half[-2]
    half = np.array(half)
    return np.concatenate([-half[:0:-1], half])


def _cap_grid(lo_pts: np.ndarray, hi_pts: np.ndarray, arc_ds: float) -> np.ndarray:
    """Polar ruled blend between two matched-sample chains -> (K, S, 2) grid.

    Column s=0 is exactly ``lo_pts``, s=S-1 exactly ``hi_pts``; interior columns
    interpolate radius linearly and azimuth along the CCW span. Never folds for
    wedge spans in (0, 2 pi).
    """
    az_lo = np.arctan2(lo_pts[:, 1], lo_pts[:, 0])
    az_hi = np.arctan2(hi_pts[:, 1], hi_pts[:, 0])
    r_lo = np.hypot(lo_pts[:, 0], lo_pts[:, 1])
    r_hi = np.hypot(hi_pts[:, 0], hi_pts[:, 1])
    daz = np.mod(az_hi - az_lo, 2.0 * np.pi)

    span_outer = daz[-1] * 0.5 * (r_lo[-1] + r_hi[-1])
    n_v = max(int(np.ceil(span_outer / arc_ds)), 2)
    S = n_v + 1

    K = len(lo_pts)
    grid = np.empty((K, S, 2))
    grid[:, 0, :] = lo_pts
    grid[:, -1, :] = hi_pts
    for m in range(1, S - 1):
        s = m / (S - 1)
        r = (1.0 - s) * r_lo + s * r_hi
        az = az_lo + s * daz
        grid[:, m, 0] = r * np.cos(az)
        grid[:, m, 1] = r * np.sin(az)
    return grid


def _mesh_wedge(cap: np.ndarray, z_levels: np.ndarray, open_inner: bool = False,
                seam_rows: int = 0, seam_side: str = 'lo'
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Watertight closed surface (top cap, bottom cap, side walls) of one wedge.

    ``cap`` is the (K, S, 2) footprint grid. Boundary grid points carry a full
    column of z levels (shared by the caps and the side walls); interior points
    exist only on the two caps. With ``open_inner`` the inner (row-0) edge gets
    NO wall: that edge is an interior seam shared with the central post hub
    (its caps continue the surface there). ``seam_rows`` additionally opens
    the ``seam_side`` chain edge for rows 0..seam_rows (the scroll-closure
    seam: the hub cap continues the surface along that chain span). Returns
    (vertices (N,3), triangles (M,3), boundary polygon (P,2)).
    """
    K, S = cap.shape[:2]
    nz = len(z_levels)
    z_bot, z_top = z_levels[0], z_levels[-1]

    verts: List[Tuple[float, float, float]] = []

    def add(x, y, z):
        verts.append((x, y, z))
        return len(verts) - 1

    seam_col = 0 if seam_side == 'lo' else S - 1
    on_boundary = np.zeros((K, S), dtype=bool)
    on_boundary[0, :] = on_boundary[-1, :] = True
    on_boundary[:, 0] = on_boundary[:, -1] = True
    if open_inner:
        # inner-edge interior points: cap-only (no wall -> no z column)
        on_boundary[0, 1:-1] = False
    if seam_rows > 0:
        # seam-chain span: cap-only strictly below the junction row; the
        # tip corner (0, seam_col) is fully seam (arc seam + chain seam)
        on_boundary[1:seam_rows, seam_col] = False
        if open_inner:
            on_boundary[0, seam_col] = False

    vid_top = np.full((K, S), -1, dtype=int)
    vid_bot = np.full((K, S), -1, dtype=int)
    columns = {}
    for k in range(K):
        for m in range(S):
            x, y = cap[k, m]
            if on_boundary[k, m]:
                ids = [add(x, y, z) for z in z_levels]
                columns[(k, m)] = ids
                vid_bot[k, m] = ids[0]
                vid_top[k, m] = ids[-1]
            else:
                vid_bot[k, m] = add(x, y, z_bot)
                vid_top[k, m] = add(x, y, z_top)

    tris: List[Tuple[int, int, int]] = []
    # caps
    for k in range(K - 1):
        for m in range(S - 1):
            a, b = vid_top[k, m], vid_top[k + 1, m]
            c, d = vid_top[k + 1, m + 1], vid_top[k, m + 1]
            tris += [(a, b, c), (a, c, d)]
            a, b = vid_bot[k, m], vid_bot[k + 1, m]
            c, d = vid_bot[k + 1, m + 1], vid_bot[k, m + 1]
            tris += [(a, c, b), (a, d, c)]
    # side walls around the boundary ring
    ring = ([(k, 0) for k in range(K)] +
            [(K - 1, m) for m in range(1, S)] +
            [(k, S - 1) for k in range(K - 2, -1, -1)] +
            [(0, m) for m in range(S - 2, 0, -1)])
    for i in range(len(ring)):
        pa, pb = ring[i], ring[(i + 1) % len(ring)]
        if open_inner and pa[0] == 0 and pb[0] == 0:
            continue   # inner-edge segment: seam to the post hub, no wall
        if (seam_rows > 0 and pa[1] == seam_col and pb[1] == seam_col
                and max(pa[0], pb[0]) <= seam_rows):
            continue   # scroll-closure seam segment, no wall
        ca = columns[pa]
        cb = columns[pb]
        for iz in range(nz - 1):
            a, b, c, d = ca[iz], cb[iz], cb[iz + 1], ca[iz + 1]
            tris += [(a, b, c), (a, c, d)]

    polygon = np.array([cap[k, m] for (k, m) in ring])
    return np.array(verts), np.array(tris, dtype=int), polygon


# ============================================================================
# Radial dee-voltage profile (shape of the Dirichlet data)
# ============================================================================
class VoltageProfile:
    """Relative dee voltage vs radius, scale(r) = V(r) / V(r_ref).

    A resonant dee is not an equipotential at RF: the gap voltage varies with
    radius (standing wave along the dee / stem). ``build_gap_electrodes``
    multiplies each dee element's Dirichlet value by ``scale(r)`` at the
    element's centroid radius, so the cavity ``voltage`` remains the peak
    voltage AT ``r_ref`` and the tabulated shape sets the rest. Ground and the
    central post stay at 0. Laplace is linear, so the amplitude still scales
    freely; only a change of the SHAPE needs a re-solve.

    Parameters
    ----------
    r, v : array-like
        Tabulated radius [m] and gap voltage (any unit - only the shape is
        used), e.g. the gap-centerline voltage of the CST / COMSOL cavity
        eigenmode. Unsorted input is sorted; duplicate radii are rejected.
    r_ref : float, optional
        Normalization radius [m], scale(r_ref) = 1. Default: the innermost
        tabulated radius - the central region the design voltage refers to.

    Outside the table the end values are HELD: below ``r_min`` the dee tips
    are a lumped capacitance at the innermost voltage; above ``r_max`` the
    profile is unknown (``build_gap_electrodes`` warns if the electrodes
    extend past the table).
    """

    def __init__(self, r, v, r_ref: Optional[float] = None):
        r = np.asarray(r, dtype=float).ravel()
        v = np.asarray(v, dtype=float).ravel()
        if r.size < 2 or r.shape != v.shape:
            raise ValueError("VoltageProfile needs >= 2 (r, V) samples of equal length")
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
            raise ValueError("VoltageProfile: non-finite entries in the table")
        if np.any(r < 0.0):
            raise ValueError("VoltageProfile: negative radii in the table")
        order = np.argsort(r)
        r, v = r[order], v[order]
        if np.any(np.diff(r) <= 0.0):
            raise ValueError("VoltageProfile: duplicate radii in the table")
        self.r_ref = float(r[0] if r_ref is None else r_ref)
        v_ref = float(np.interp(self.r_ref, r, v))
        if abs(v_ref) <= 1e-12 * float(np.max(np.abs(v))):
            raise ValueError(f"VoltageProfile: V(r_ref = {self.r_ref:.4f} m) is zero")
        scale = v / v_ref
        if np.any(scale <= 0.0):
            raise ValueError("VoltageProfile: the voltage changes sign (or vanishes) "
                             "along the radius - a dee gap does not do that; "
                             "check the table columns / units")
        self.r = r
        self.scale_tab = scale

    @property
    def r_min(self) -> float:
        return float(self.r[0])

    @property
    def r_max(self) -> float:
        return float(self.r[-1])

    def __call__(self, r):
        """scale(r) = V(r) / V(r_ref); end values held outside the table."""
        return np.interp(np.asarray(r, dtype=float), self.r, self.scale_tab)

    @classmethod
    def from_file(cls, path, r_scale: float = 1.0,
                  r_ref: Optional[float] = None, usecols=(0, 1),
                  **loadtxt_kwargs) -> 'VoltageProfile':
        """Two-column text table (radius, voltage).

        ``r_scale`` converts the radius column to meters (1e-3 for a mm
        export). Extra keyword arguments go to ``numpy.loadtxt``
        (``delimiter``, ``skiprows``, ``comments``, ...).
        """
        data = np.atleast_2d(np.loadtxt(path, usecols=usecols, **loadtxt_kwargs))
        return cls(data[:, 0] * r_scale, data[:, 1], r_ref=r_ref)

    def __repr__(self) -> str:
        return (f"VoltageProfile(r = {self.r_min * 1000:.1f}..{self.r_max * 1000:.1f} mm, "
                f"r_ref = {self.r_ref * 1000:.1f} mm, scale = "
                f"{self.scale_tab.min():.3f}..{self.scale_tab.max():.3f})")


def _voltage_scale(voltage_profile):
    """Normalize ``voltage_profile`` to (scale callable, info dict).

    (None, None) if unset. Accepts a ``VoltageProfile``, an (N, 2) array of
    (r [m], V) rows (wrapped with the default r_ref), or any vectorized
    callable scale(r) used as-is (the caller owns its normalization).
    """
    if voltage_profile is None:
        return None, None
    if isinstance(voltage_profile, VoltageProfile):
        prof = voltage_profile
    elif callable(voltage_profile):
        name = getattr(voltage_profile, '__name__', type(voltage_profile).__name__)
        return voltage_profile, {'kind': 'callable', 'name': name,
                                 'r_ref': None, 'r_min': None, 'r_max': None}
    else:
        arr = np.asarray(voltage_profile, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("voltage_profile must be a VoltageProfile, a callable "
                             "scale(r), or an (N, 2) array of (r [m], V) rows")
        prof = VoltageProfile(arr[:, 0], arr[:, 1])
    return prof, {'kind': 'table', 'r_ref': prof.r_ref, 'r_min': prof.r_min,
                  'r_max': prof.r_max,
                  'scale_range': (float(prof.scale_tab.min()),
                                  float(prof.scale_tab.max()))}


# ============================================================================
# Dee-system -> electrode model
# ============================================================================
def _check_phase(deg: float, targets=(0.0, 180.0), tol: float = 1e-3) -> float:
    d = np.mod(deg, 360.0)
    for t in targets:
        if abs(d - t) < tol or abs(d - t - 360.0) < tol:
            return t
    raise ValueError(
        f"gap phase {deg:.4f} deg is not in the 0/180 pattern - a single static "
        f"BEM solve is only exact for phases in {{0, 180}} (mod 360)")


def build_gap_electrodes(design_or_cavities,
                         height: Optional[float] = None,
                         chain_ds: float = 0.012,
                         arc_ds: float = 0.040,
                         wall_dz: float = 0.003,
                         wall_growth: float = 1.5,
                         min_metal_width: float = 0.004,
                         r_inner: Optional[float] = None,
                         max_r_inner: Optional[float] = None,
                         post_tip_gap: Optional[float] = None,
                         post_min_radius: Optional[float] = None,
                         center_post_radius: Optional[float] = None,
                         trim_trajectory: Optional[np.ndarray] = None,
                         traj_tip_clearance: Optional[float] = None,
                         fillet_radius: Optional[float] = None,
                         voltage_profile=None,
                         verbose: bool = True) -> ElectrodeModel:
    """Closed dee/ground solids from a dee system's RF gaps.

    Parameters
    ----------
    design_or_cavities : CentralRegion or list of RFCavity
        Gaps in CREATION order: consecutive pairs (2j, 2j+1) are the
        (entry, exit) gaps of dee j.
    height : float, optional
        Total extrusion height [m]; default 8x the largest gap width
        (the "tall wall" 2D limit - midplane field converged at ~4x).
    chain_ds : float
        Along-chain mesh step on the gap-facing walls [m].
    arc_ds : float
        Azimuthal mesh step on caps and arc walls [m].
    wall_dz : float
        Wall mesh row height at the median plane [m] (rows grow geometrically
        by ``wall_growth`` toward the caps).
    min_metal_width : float
        Minimum wedge arc width [m]; chains are truncated below the radius
        where a wedge would get thinner than this (auto ``r_inner``).
    r_inner : float, optional
        Explicit inner truncation radius [m]; overrides the auto-computation.
    max_r_inner : float, optional
        Buildability guard: raise if the (auto or explicit) truncation radius
        ends up AT or ABOVE this value. Pass the beam injection radius so a
        geometry whose electrodes cannot exist where the beam first crosses
        the gaps fails loudly instead of silently losing the first kick(s).
    post_tip_gap : float, optional
        Opt-in grounded central post, sized RELATIVE to the electrodes: the
        post radius is (truncation radius - post_tip_gap), so the radial gap
        between the dee tips and the post is exactly this value for every
        geometry (voltage holding: the tip-window field is ~ V / post_tip_gap;
        8 mm at 60 kV ~ the 6 MV/m gap field). Every GROUND wedge is extended
        inward as a radial spoke ONTO the post - hub and spokes share their
        seam coordinates and form one contiguous grounded solid, which bounds
        the field at the center. Dee wedges keep the normal truncation.
    post_min_radius : float, optional
        Physical floor for the post (e.g. the inflector housing). If the
        electrodes reach so deep that the post would shrink below this, the
        TRUNCATION is raised to ``post_min_radius + post_tip_gap`` instead
        (electrodes that deep serve no beam anyway - the beam never orbits
        below the injection region).
    center_post_radius : float, optional
        Absolute alternative to ``post_tip_gap``: fixed post radius [m]
        regardless of where the electrodes truncate. Mutually exclusive.
    trim_trajectory : array (N, >=2), optional
        SCROLL mode: a reference design trajectory covering at least one full
        turn (e.g. ``OptimizedOrbit.trajectory_reference``). Its first turn
        defines a radius profile r1(azimuth); every electrode inner edge is
        trimmed to follow it (dee tips at r1 - traj_tip_clearance) and the
        central post becomes a SCROLL polygon at
        r1 - traj_tip_clearance - post_tip_gap. The spiral TERMINATES on the
        wrap dummy-dee: at the junction where that wedge's edge crosses the
        spiral (rim closes along the edge, an interior seam), or - if the
        edge was trimmed exactly ONTO the spiral - along the wedge's inner
        arc across the wrap. Only when neither exists (no ground wedge at
        the injection azimuth) does the scroll end in a radial step face one
        tip gap past the outermost electrode tip, holding the inner-branch
        radius back to the wrap.
        Ground wedges merge contiguously onto the scroll. Requires
        ``post_tip_gap``; exclusive with r_inner / center_post_radius /
        post_min_radius. Gaps are active at every beam crossing by
        construction.
    traj_tip_clearance : float, optional
        SCROLL mode: radial clearance between the beam CENTER (turn-1
        trajectory) and the dee tips [m] - includes the beam half-width and
        a field-quality safety margin. Default: the largest gap width at
        r_min over all gaps (= gap_width_inner for tapered gaps).
    fillet_radius : float, optional
        Opt-in corner fillets: interior kinks of the boundary chains (segment
        node jogs, taper kinks, spoke junctions) are replaced by tangent arcs
        of this radius [m] (auto-reduced where adjacent edges are short).
    voltage_profile : VoltageProfile, (N, 2) array or callable, optional
        Radial dee-voltage SHAPE scale(r) from the RF cavity model (see
        ``VoltageProfile``): every dee element's Dirichlet value is multiplied
        by scale(r) at its centroid radius, so the cavity voltage is the peak
        voltage at the profile's reference radius and the gap voltage follows
        the cavity outward. An (N, 2) table is (r [m], V) rows with the
        default reference (innermost radius); a callable is used as-is
        (vectorized, caller-normalized). Ground and post stay at 0. Recorded
        in ``params['voltage_profile']``; warns if the dee electrodes extend
        beyond a tabulated profile (outermost value held there).
    """
    cavities = getattr(design_or_cavities, 'rf_cavities', design_or_cavities)
    v_scale, v_info = _voltage_scale(voltage_profile)
    r_dee_lo, r_dee_hi = np.inf, 0.0
    n = len(cavities)
    if n < 2 or n % 2 != 0:
        raise ValueError(f"need an even number of gaps (entry/exit per dee), got {n}")

    # --- dee grouping + phase-pattern checks (creation order: pairs) ---------
    dees = []
    for j in range(n // 2):
        entry, exit_ = cavities[2 * j], cavities[2 * j + 1]
        d_phase = np.mod(exit_.phase_deg - entry.phase_deg, 360.0)
        if abs(d_phase - 180.0) > 1e-3:
            raise ValueError(f"dee {j}: exit phase must be entry + 180 deg "
                             f"(got {d_phase:.4f})")
        entry_phase = _check_phase(entry.phase_deg)
        if abs(entry.voltage - exit_.voltage) > 1e-6 * max(entry.voltage, 1.0):
            raise ValueError(f"dee {j}: entry/exit voltages differ")
        span = np.mod(exit_.base_angle - entry.base_angle, 360.0)
        if not 0.0 < span < 180.0:
            raise ValueError(f"dee {j}: CCW span entry->exit is {span:.2f} deg "
                             f"(expected in (0, 180))")
        # Phi_dee(t) = -V cos(entry_phase) * cos(omega t + bunch) reproduces the
        # thin-gap energy gains qV cos(omega t + phase_gap + bunch) exactly for
        # phases in {0, 180} (ground at 0).
        potential = -entry.voltage * np.cos(np.deg2rad(entry_phase))
        dees.append({'entry': entry, 'exit': exit_, 'potential': potential,
                     'index': j})
    dees.sort(key=lambda d: np.mod(d['entry'].base_angle, 360.0))

    # --- offset boundary chains (into each wedge) ---------------------------
    # +g/2 = left of inner->outer travel = CCW side. A wedge's low-azimuth
    # boundary is its gap chain shifted CCW (into the wedge), the high-azimuth
    # boundary shifted CW.
    wedges: List[Wedge] = []
    for jj, dee in enumerate(dees):
        lo = offset_gap_boundary(dee['entry'], +1.0)
        hi = offset_gap_boundary(dee['exit'], -1.0)
        wedges.append(Wedge('dee', dee['potential'], lo, hi,
                            label=f"dee{dee['index']}"))
        nxt = dees[(jj + 1) % len(dees)]
        lo_g = offset_gap_boundary(dee['exit'], +1.0)
        hi_g = offset_gap_boundary(nxt['entry'], -1.0)
        wedges.append(Wedge('ground', 0.0, lo_g, hi_g,
                            label=f"ground{dee['index']}-{nxt['index']}"))

    # --- per-wedge pinch radii (mesh-validity floors) -------------------------
    pinches = []
    for w in wedges:
        r0 = max(np.hypot(*w.chain_lo[0]), np.hypot(*w.chain_hi[0]), 1e-4)
        # only the converging inner region matters for the width check
        r_hi_scan = min(np.hypot(*w.chain_lo[-1]), np.hypot(*w.chain_hi[-1]), 0.15)
        ladder = np.arange(r0, r_hi_scan, 5e-4)
        pinch = 0.0
        if len(ladder):
            width = _ccw_width(w.chain_lo, w.chain_hi, ladder)
            bad = np.where(width < min_metal_width)[0]
            if len(bad) > 0:
                pinch = float(ladder[bad[-1]] + 5e-4)
        pinches.append(pinch)
    r_inner_auto = max(pinches) if pinches else 0.0

    # --- mode setup: scroll (trajectory-following) or circular post -----------
    r_post = None
    scroll_mode = trim_trajectory is not None
    if scroll_mode:
        if post_tip_gap is None:
            raise ValueError("trim_trajectory requires post_tip_gap (radial "
                             "dee-tip-to-scroll gap)")
        if (center_post_radius is not None or post_min_radius is not None
                or r_inner is not None):
            raise ValueError("trim_trajectory is exclusive with r_inner, "
                             "center_post_radius and post_min_radius")
        if post_tip_gap < 0.002:
            raise ValueError("post_tip_gap must be >= 2 mm (dee-to-post "
                             "voltage holding; ~60 kV wants >= ~8 mm)")
        r1_at, prog_of, phi0, s_dir, u1, r1v = _turn1_profile(trim_trajectory)
        d_tip = (traj_tip_clearance if traj_tip_clearance is not None
                 else max(float(c.gap_width_at(c.r_min)) for c in cavities))
        r_scroll_min = float(np.min(r1v)) - d_tip - post_tip_gap
        if r_scroll_min < 0.002:
            raise ValueError(
                f"scroll degenerates: min turn-1 radius {np.min(r1v)*1000:.1f} mm "
                f"minus clearances leaves {r_scroll_min*1000:.1f} mm")

        r_inner = r_inner_auto   # reported floor; trims follow the corridor
        if verbose:
            print(f"[gap_fields] scroll mode: turn-1 r "
                  f"{np.min(r1v)*1000:.1f}-{np.max(r1v)*1000:.1f} mm, tip "
                  f"clearance {d_tip*1000:.1f} mm, scroll gap "
                  f"{post_tip_gap*1000:.1f} mm (pinch floor "
                  f"{r_inner_auto*1000:.1f} mm)")
        # max_r_inner guard: satisfied by construction (tips sit d_tip below
        # every first crossing).
    else:
        if post_tip_gap is not None and center_post_radius is not None:
            raise ValueError("give either post_tip_gap or center_post_radius, "
                             "not both")
        if r_inner is None:
            r_inner = r_inner_auto
        # central post sizing (may RAISE the truncation to fit the post)
        if post_tip_gap is not None:
            if post_tip_gap < 0.002:
                raise ValueError("post_tip_gap must be >= 2 mm (dee-to-post "
                                 "voltage holding; ~60 kV wants >= ~8 mm)")
            if post_min_radius is not None:
                r_inner = max(r_inner, post_min_radius + post_tip_gap)
            r_post = r_inner - post_tip_gap
            if r_post < 0.002:
                raise ValueError(
                    f"post radius r_inner - post_tip_gap = {r_post * 1000:.1f} mm "
                    f"is degenerate; set post_min_radius to give the post a floor")
        elif center_post_radius is not None:
            r_electrode_min = r_inner if r_inner > 0 else min(
                min(np.hypot(*w.chain_lo[0]), np.hypot(*w.chain_hi[0]))
                for w in wedges)
            if center_post_radius <= 0.0:
                raise ValueError("center_post_radius must be > 0")
            if r_electrode_min - center_post_radius < 0.002:
                raise ValueError(
                    f"center_post_radius = {center_post_radius * 1000:.1f} mm must "
                    f"lie >= 2 mm below the electrode inner radius "
                    f"{r_electrode_min * 1000:.1f} mm")
            r_post = center_post_radius

        if verbose:
            print(f"[gap_fields] inner truncation radius: {r_inner * 1000:.1f} mm"
                  + (f", central post r = {r_post * 1000:.1f} mm"
                     if r_post is not None else ""))
        if max_r_inner is not None and r_inner >= max_r_inner:
            raise ValueError(
                f"gap electrodes truncate at r_inner = {r_inner * 1000:.1f} mm, at "
                f"or above the required active radius {max_r_inner * 1000:.1f} mm "
                f"(e.g. the beam injection radius): the beam would cross gaps "
                f"where no electrode exists. Open up the inter-gap clearances "
                f"(see the CavityGeometryOptimizer clearance guard) or taper the "
                f"gap width toward the center (RFCavity "
                f"gap_width_inner/gap_taper_radius).")

    gap_widths = [c.gap_width for c in cavities]
    if height is None:
        height = 8.0 * max(gap_widths)
    z_levels = _z_levels(height, wall_dz, wall_growth)

    # --- scroll closure: the spiral runs onto the wrap dummy-dee ---------------
    # Pass 1: trim all chains at the pure corridor. The spiral then TERMINATES
    # where it intersects the wrap dummy-dee's gap-offset edge (the dummy-dee
    # whose boundary sweeps across the turn wrap), and the closing edge of the
    # hub FOLLOWS that edge back to the wedge tip - an interior seam, so hub
    # and dummy-dee form one contiguous grounded area. If the wrap dummy-dee's
    # edge instead lands exactly ON the spiral (trimmed onto it, no transversal
    # crossing), its inner arc spans the wrap and IS the hub closure (handled
    # in _scroll_ring; no step face). Last-resort fallback (wrap sector not
    # covered by a ground arc, e.g. injection inside a dee sector): radial
    # step one tip gap past the outermost electrode tip + inner-branch hold
    # back to the wrap.
    scroll_trims = None
    if scroll_mode:
        scroll_trims = []
        prog_tips = 0.0
        for w, pinch in zip(wedges, pinches):
            off = d_tip if w.kind == 'dee' else d_tip + post_tip_gap
            floor = pinch + 5e-4 if pinch > 0 else 0.0

            def cut(pts, _o=off):
                return r1_at(pts) - _o

            lo = _truncate_chain_at_curve(w.chain_lo, cut, floor)
            hi = _truncate_chain_at_curve(w.chain_hi, cut, floor)
            scroll_trims.append([lo, hi])
            for ch in (lo, hi):
                p = float(prog_of(ch[:1])[0])
                if p < 2.0 * np.pi - 0.05:   # exclude wrap-adjacent starts
                    prog_tips = max(prog_tips, p)

        # spiral polyline over the end-of-turn sector, at the HUB offset
        ps = np.linspace(np.pi, 2.0 * np.pi, 600)
        rs = np.interp(ps, u1, r1v) - d_tip - post_tip_gap
        azs = phi0 + s_dir * ps
        spiral_poly = np.column_stack([rs * np.cos(azs), rs * np.sin(azs)])

        special = None   # (iw, side_idx, seg_i, J, prog_J)
        wrap_iw = None   # ground wedge whose inner arc spans the turn wrap
        for iw, w in enumerate(wedges):
            if w.kind != 'ground':
                continue
            for side_idx, ch in enumerate(scroll_trims[iw]):
                hit = _first_polyline_intersection(ch, spiral_poly)
                if hit is None:
                    continue
                seg_i, t, J = hit
                arclen = (np.sum(np.hypot(*np.diff(ch[:seg_i + 1], axis=0).T))
                          + t * np.hypot(*(ch[seg_i + 1] - ch[seg_i])))
                if arclen < 1e-3:
                    continue   # trimmed exactly onto the spiral: normal spoke
                prog_j = float(prog_of(J[None, :])[0])
                if special is None or prog_j > special[4]:
                    special = (iw, side_idx, seg_i, J, prog_j)

        if special is not None:
            sp_iw, sp_side, seg_i, sp_J, prog_J = special
            # insert the junction as a chain vertex (shared seam coordinate)
            ch = scroll_trims[sp_iw][sp_side]
            if np.hypot(*(ch[seg_i] - sp_J)) > 1e-9:
                ch = np.vstack([ch[:seg_i + 1], sp_J, ch[seg_i + 1:]])
                scroll_trims[sp_iw][sp_side] = ch
            step_prog = None
            r_seam_tip = float(np.hypot(*scroll_trims[sp_iw][sp_side][0]))

            def scroll_r(prog):
                base = np.interp(prog, u1, r1v) - d_tip - post_tip_gap
                return np.where(np.asarray(prog, dtype=float) <= prog_J,
                                base, r_seam_tip)

            if verbose:
                print(f"[gap_fields] scroll closes onto "
                      f"{wedges[sp_iw].label}'s "
                      f"{'lower' if sp_side == 0 else 'upper'} edge at "
                      f"r = {np.hypot(*sp_J)*1000:.1f} mm "
                      f"(az {np.rad2deg(np.arctan2(sp_J[1], sp_J[0])):.1f} deg)")
        else:
            # no transversal junction: does a ground wedge's inner arc span
            # the turn wrap (chain tips on both sides of the injection
            # azimuth)? Then that arc is the natural hub closure.
            for iw2, w2 in enumerate(wedges):
                if w2.kind != 'ground':
                    continue
                lo2, hi2 = scroll_trims[iw2]
                if (float(prog_of(hi2[:1])[0])
                        < float(prog_of(lo2[:1])[0])):
                    wrap_iw = iw2
                    break

        if special is None and wrap_iw is not None:
            # wrap-arc closure: the spiral terminates at the wrap dummy-dee's
            # low-side tip and the hub rim continues along that wedge's inner
            # (row-0) arc across the wrap - hub and wrap dummy-dee merge
            # contiguously along the shared seam arc; no radial step face.
            lo2, hi2 = scroll_trims[wrap_iw]
            p_lo = float(prog_of(lo2[:1])[0])
            p_hi = float(prog_of(hi2[:1])[0])
            r_lo_tip = float(np.hypot(*lo2[0]))
            r_hi_tip = float(np.hypot(*hi2[0]))
            arc_span = float(np.mod(p_hi - p_lo, 2.0 * np.pi))
            step_prog = None
            prog_J = p_hi    # clamp target: just downstream of the closure

            def scroll_r(prog):
                # hub radius: spiral outside the wrap arc's sector, the
                # arc's linear radius blend (see _cap_grid) across it
                base = np.interp(prog, u1, r1v) - d_tip - post_tip_gap
                dp = np.mod(np.asarray(prog, dtype=float) - p_lo, 2.0 * np.pi)
                s = np.minimum(dp / arc_span, 1.0)
                return np.where(dp <= arc_span,
                                (1.0 - s) * r_lo_tip + s * r_hi_tip, base)

            if verbose:
                print(f"[gap_fields] scroll closes along "
                      f"{wedges[wrap_iw].label}'s inner arc across the wrap "
                      f"(r {r_lo_tip*1000:.1f} -> {r_hi_tip*1000:.1f} mm, "
                      f"no step face)")
        elif special is None:
            # last-resort fallback: radial step past the last tip + hold sector
            r_at_end = float(np.interp(prog_tips, u1, r1v)) - d_tip - post_tip_gap
            step_prog = min(prog_tips + post_tip_gap / max(r_at_end, 1e-3),
                            2.0 * np.pi - 0.01)
            prog_J = step_prog
            r_hold = float(r1v[0]) - d_tip - post_tip_gap

            def scroll_r(prog):
                base = np.interp(prog, u1, r1v) - d_tip - post_tip_gap
                return np.where(np.asarray(prog, dtype=float) <= step_prog,
                                base, r_hold)

            if verbose:
                print(f"[gap_fields] scroll ends at the last electrode edge: "
                      f"step at {np.rad2deg(phi0 + s_dir * step_prog):.1f} deg "
                      f"(r {float(scroll_r(step_prog))*1000:.1f} -> "
                      f"{r_hold*1000:.1f} mm)")

        def scroll_xy(prog):
            rr = float(scroll_r(prog))
            az = phi0 + s_dir * prog
            return np.array([rr * np.cos(az), rr * np.sin(az)])

    # --- mesh each wedge ------------------------------------------------------
    all_verts, all_tris, all_pots = [], [], []
    v_offset = 0
    post_arcs = []
    seam_pts = None
    for iw, (w, pinch) in enumerate(zip(wedges, pinches)):
        is_special = scroll_mode and special is not None and iw == special[0]
        if scroll_mode:
            lo, hi = scroll_trims[iw]
            if not is_special and iw != wrap_iw:
                # keep wedge boundaries out of the hub (a chain sweeping past
                # the closure would run buried under the outer branch); dees
                # get the full voltage-gap setback off the closure. The wrap
                # wedge (arc closure) is exempt: its chains ARE the seam.
                setback = 0.0015 if w.kind == 'ground' else post_tip_gap
                lo = _clamp_chain_out_of_hub(lo, prog_of, scroll_r, phi0,
                                             s_dir, prog_J, setback)
                hi = _clamp_chain_out_of_hub(hi, prog_of, scroll_r, phi0,
                                             s_dir, prog_J, setback)
            as_spoke = w.kind == 'ground'
        else:
            lo = _truncate_chain_inner(w.chain_lo, r_inner) if r_inner > 0 else w.chain_lo
            hi = _truncate_chain_inner(w.chain_hi, r_inner) if r_inner > 0 else w.chain_hi
            as_spoke = r_post is not None and w.kind == 'ground'
            if as_spoke:
                # grounded spoke ONTO the central post (radial side edges; the
                # polar-blend cap turns the inner closure into an arc exactly
                # on the post circle, which the hub mesh reuses as its seam)
                lo = _extend_chain_inner(lo, r_post)
                hi = _extend_chain_inner(hi, r_post)
        if fillet_radius is not None:
            lo = _fillet_polyline(lo, fillet_radius)
            hi = _fillet_polyline(hi, fillet_radius)
        fr = _matched_fractions(lo, hi, chain_ds)
        lo_s = _sample_polyline(lo, fr)
        hi_s = _sample_polyline(hi, fr)
        cap = _cap_grid(lo_s, hi_s, arc_ds)
        seam_rows = 0
        if is_special:
            # the hub's closing edge follows this wedge's chain from the
            # junction back to the tip: the SAMPLED boundary rows are the
            # shared seam (verbatim coordinates), walls open along them
            if special[1] != 0:
                raise NotImplementedError(
                    "scroll closure onto a wedge's high-azimuth edge "
                    "(clockwise beam) is not implemented")
            side_s = lo_s
            k_j = int(np.argmin(np.hypot(side_s[:, 0] - special[3][0],
                                         side_s[:, 1] - special[3][1])))
            if k_j < 1:
                raise RuntimeError("scroll junction degenerates to the wedge "
                                   "tip - geometry inspection needed")
            seam_rows = k_j
            seam_pts = side_s[k_j::-1].copy()   # junction -> tip corner
        verts, tris, polygon = _mesh_wedge(cap, z_levels, open_inner=as_spoke,
                                           seam_rows=seam_rows, seam_side='lo')
        if as_spoke:
            post_arcs.append(cap[0, :, :].copy())
        w.chain_lo, w.chain_hi = lo, hi
        w.polygon = polygon
        all_verts.append(verts)
        all_tris.append(tris + v_offset)
        pots = np.full(len(tris), w.potential)
        pot_txt = f"{w.potential:+9.1f} V"
        if w.kind == 'dee':
            r_v = np.hypot(verts[:, 0], verts[:, 1])
            r_dee_lo, r_dee_hi = min(r_dee_lo, r_v.min()), max(r_dee_hi, r_v.max())
            if v_scale is not None:
                # DP0 Dirichlet data: one value per element, taken at the
                # element centroid radius (caps and walls alike)
                cen = verts[tris].mean(axis=1)
                pots = pots * v_scale(np.hypot(cen[:, 0], cen[:, 1]))
                pot_txt = f"{pots.min():+9.1f}..{pots.max():+9.1f} V"
        all_pots.append(pots)
        v_offset += len(verts)
        if verbose:
            print(f"[gap_fields]   {w.label:>14s} ({w.kind:6s}) @ {pot_txt}: "
                  f"{len(tris):5d} tris, cap grid {cap.shape[0]}x{cap.shape[1]}")

    # --- grounded central post hub (stitched to the spokes) --------------------
    if scroll_mode:
        post_ds = min(arc_ds, 0.004)
        ring, iwl, se = _scroll_ring(post_arcs, scroll_xy, prog_of, post_ds,
                                     step_prog, seam=seam_pts)
        p_verts, p_tris, p_poly = _mesh_scroll_hub(ring, iwl, se, z_levels,
                                                   post_ds)
        post = Wedge('post', 0.0, p_poly, p_poly,
                     polygon=p_poly, label='center-scroll')
        wedges.append(post)
        all_verts.append(p_verts)
        all_tris.append(p_tris + v_offset)
        all_pots.append(np.zeros(len(p_tris)))
        v_offset += len(p_verts)
        if verbose:
            rr = np.hypot(ring[:, 0], ring[:, 1])
            print(f"[gap_fields]   {post.label:>14s} ({post.kind:6s}) @ "
                  f"{post.potential:+9.1f} V: {len(p_tris):5d} tris, "
                  f"r = {rr.min()*1000:.1f}-{rr.max()*1000:.1f} mm, "
                  f"{len(post_arcs)} spokes merged")
    elif r_post is not None:
        p_verts, p_tris, p_circle = _mesh_post_hub(post_arcs, r_post, z_levels,
                                                   post_ds=min(arc_ds, 0.004))
        post = Wedge('post', 0.0, p_circle, p_circle,
                     polygon=p_circle, label='center-post')
        wedges.append(post)
        all_verts.append(p_verts)
        all_tris.append(p_tris + v_offset)
        all_pots.append(np.zeros(len(p_tris)))
        v_offset += len(p_verts)
        if verbose:
            print(f"[gap_fields]   {post.label:>14s} ({post.kind:6s}) @ "
                  f"{post.potential:+9.1f} V: {len(p_tris):5d} tris, "
                  f"r = {r_post * 1000:.1f} mm, {len(post_arcs)} spokes")

    model = ElectrodeModel(
        vertices=np.vstack(all_verts),
        triangles=np.vstack(all_tris),
        potentials=np.concatenate(all_pots),
        wedges=wedges,
        params={'height': height, 'chain_ds': chain_ds, 'arc_ds': arc_ds,
                'wall_dz': wall_dz, 'wall_growth': wall_growth,
                'min_metal_width': min_metal_width, 'r_inner': r_inner,
                'post_mode': ('scroll' if scroll_mode
                              else 'circle' if r_post is not None else None),
                'post_radius': r_post, 'post_tip_gap': post_tip_gap,
                'post_min_radius': post_min_radius,
                'traj_tip_clearance': d_tip if scroll_mode else None,
                'scroll_junction': (tuple(special[3]) if scroll_mode
                                    and special is not None else None),
                'scroll_junction_prog': (special[4] if scroll_mode
                                         and special is not None else None),
                'scroll_closure': (('seam' if special is not None else
                                    'wrap-arc' if wrap_iw is not None else
                                    'step') if scroll_mode else None),
                'fillet_radius': fillet_radius,
                'voltage_profile': v_info},
    )

    # voltage-profile coverage (tabulated profiles only)
    if v_info is not None and v_info['r_max'] is not None:
        if r_dee_hi > v_info['r_max'] + 1e-3:
            warnings.warn(
                f"dee electrodes extend to r = {r_dee_hi * 1000:.1f} mm but the "
                f"voltage profile is tabulated only to {v_info['r_max'] * 1000:.1f} mm; "
                f"the outermost value is held beyond the table", stacklevel=2)
        if verbose and r_dee_lo < v_info['r_min'] - 1e-3:
            print(f"[gap_fields] voltage profile: innermost tabulated value held "
                  f"below r = {v_info['r_min'] * 1000:.1f} mm (dee tips reach "
                  f"{r_dee_lo * 1000:.1f} mm)")
    if verbose and v_info is not None:
        if v_info['kind'] == 'table':
            print(f"[gap_fields] voltage profile: scale "
                  f"{v_info['scale_range'][0]:.3f}..{v_info['scale_range'][1]:.3f} "
                  f"over r = {v_info['r_min'] * 1000:.1f}..{v_info['r_max'] * 1000:.1f} mm, "
                  f"reference (scale = 1) at {v_info['r_ref'] * 1000:.1f} mm")
        else:
            print(f"[gap_fields] voltage profile: callable {v_info['name']}")

    # degenerate-element guard
    v = model.vertices
    t = model.triangles
    areas = 0.5 * np.linalg.norm(
        np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
    n_bad = int(np.sum(areas < 1e-14))
    if n_bad:
        raise RuntimeError(f"electrode mesh has {n_bad} degenerate triangles")
    if verbose:
        print(f"[gap_fields] total: {model.n_elements} elements, "
              f"{len(model.vertices)} vertices, height {height * 1000:.0f} mm")
    return model


# ============================================================================
# BEM solve + field extraction
# ============================================================================
def _bempp():
    import bempp_cl.api as bempp
    try:
        import pyopencl  # noqa: F401
    except ImportError:
        bempp.DEFAULT_DEVICE_INTERFACE = "numba"
    return bempp


@dataclass
class GapFieldSolution:
    """Solved surface charge (Neumann data) + evaluation helpers."""
    model: ElectrodeModel
    space: object
    neumann: object
    gmres_info: int
    solve_time_s: float
    n_iterations: int = 0

    def potential(self, pts: np.ndarray, chunk: int = 50000,
                  verbose: bool = False) -> np.ndarray:
        """Potential [V] at (M, 3) points via the single-layer representation."""
        from bempp_cl.api.operators.potential import laplace as lap_pot
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        out = np.empty(len(pts))
        for i0 in range(0, len(pts), chunk):
            block = pts[i0:i0 + chunk]
            op = lap_pot.single_layer(self.space, block.T.copy())
            out[i0:i0 + len(block)] = np.real(op * self.neumann).ravel()
            if verbose and len(pts) > chunk:
                print(f"[gap_fields]   potential eval {i0 + len(block)}/{len(pts)}")
        return out

    def efield_midplane(self, pts_2d: np.ndarray, h: float = 2e-4) -> np.ndarray:
        """(Ex, Ey) [V/m] at (M, 2) midplane points via central differences."""
        pts_2d = np.atleast_2d(np.asarray(pts_2d, dtype=float))
        m = len(pts_2d)
        probe = np.zeros((4 * m, 3))
        probe[0 * m:1 * m, :2] = pts_2d + [h, 0.0]
        probe[1 * m:2 * m, :2] = pts_2d - [h, 0.0]
        probe[2 * m:3 * m, :2] = pts_2d + [0.0, h]
        probe[3 * m:4 * m, :2] = pts_2d - [0.0, h]
        phi = self.potential(probe)
        ex = -(phi[0 * m:1 * m] - phi[1 * m:2 * m]) / (2 * h)
        ey = -(phi[2 * m:3 * m] - phi[3 * m:4 * m]) / (2 * h)
        return np.column_stack([ex, ey])

    def to_field(self,
                 spacing: float = 0.0015,
                 margin: float = 0.02,
                 extent: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                 chunk: int = 50000,
                 mask_metal: bool = True,
                 interpolator_backend: str = 'auto',
                 verbose: bool = True) -> Field:
        """Evaluate phi on a midplane grid and return E = -grad(phi) as a Field.

        The returned ``Field`` is dim-2 Cartesian (x, y) with Ez = 0 and E = 0
        outside the grid. With ``mask_metal`` the DEEP interior of each metal
        footprint (eroded ~3 cells in from the wall) is set to exactly 0 (the
        physical shielded value; the BEM interior is constant up to noise).
        The ~3-cell layer just inside the wall is deliberately NOT masked: it
        carries the grid-smeared field jump of the wall surface, and zeroing
        it would lose ~h/gap_width of the integrated gap voltage (measured:
        11% at h=1 mm, g=10 mm).
        """
        if extent is None:
            r = self.model.extent() + margin
            extent = ((-r, r), (-r, r))
        (x0, x1), (y0, y1) = extent
        nx = max(int(np.round((x1 - x0) / spacing)) + 1, 2)
        ny = max(int(np.round((y1 - y0) / spacing)) + 1, 2)
        xs = np.linspace(x0, x1, nx)
        ys = np.linspace(y0, y1, ny)
        gx, gy = np.meshgrid(xs, ys, indexing='ij')
        pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])

        t0 = time.time()
        phi = self.potential(pts, chunk=chunk, verbose=verbose).reshape(nx, ny)
        if verbose:
            print(f"[gap_fields] potential grid {nx}x{ny} evaluated "
                  f"in {time.time() - t0:.1f} s")

        ex = -np.gradient(phi, xs, axis=0)
        ey = -np.gradient(phi, ys, axis=1)

        if mask_metal:
            from matplotlib.path import Path as MplPath
            from scipy.ndimage import binary_erosion
            pts_2d = np.column_stack([gx.ravel(), gy.ravel()])
            inside = np.zeros(len(pts_2d), dtype=bool)
            for w in self.model.wedges:
                inside |= MplPath(w.polygon).contains_points(pts_2d)
            # keep the wall-jump layer (see docstring); zero only deep interior
            deep = binary_erosion(inside.reshape(nx, ny), iterations=3)
            ex[deep] = 0.0
            ey[deep] = 0.0

        return Field.from_arrays(
            grid={'x': xs, 'y': ys},
            values={'x': ex, 'y': ey, 'z': np.zeros_like(ex)},
            label="BEM 2D gap field (static pattern)",
            interpolator_backend=interpolator_backend,
        )


def solve_gap_field(model: ElectrodeModel,
                    tol: float = 1e-5,
                    maxiter: int = 20000,
                    restart: int = 1000,
                    verbose: bool = True) -> GapFieldSolution:
    """Laplace Dirichlet solve (DP0 / single-layer / GMRES) on the model.

    Uses the STRONG form (mass-matrix preconditioned) - the plain weak-form
    first-kind system stalls at these element counts / size ratios. bempp
    passes a callback to scipy, which puts scipy's gmres in 'legacy' mode:
    ``maxiter`` counts INNER iterations, not restart cycles.

    The iteration count is set by conditioning, which degrades sharply with
    thin metal features (opposite faces of a thin fin carry near-canceling
    charge - intrinsically hard for the first-kind single-layer operator).
    Measured on the same ~27k-element scroll geometry: 792 iterations at
    min_metal_width = 2 mm vs 9641 at 1 mm. tol 1e-5 is comfortable: field
    accuracy is mesh-limited at ~0.3%, well above the residual.
    """
    bempp = _bempp()
    from bempp_cl.api.operators.boundary import laplace as lap_bnd
    from bempp_cl.api.linalg import gmres

    grid = bempp.Grid(model.vertices.T.copy(),
                      model.triangles.T.astype(np.uint32).copy())
    space = bempp.function_space(grid, "DP", 0)
    dirichlet = bempp.GridFunction(space, coefficients=model.potentials.astype(float))
    slp = lap_bnd.single_layer(space, space, space)

    t0 = time.time()
    neumann, info, residuals, n_iter = gmres(
        slp, dirichlet, tol=tol, maxiter=maxiter, restart=restart,
        use_strong_form=True, return_residuals=True,
        return_iteration_count=True)
    dt = time.time() - t0
    if info != 0:
        last = residuals[-1] if len(residuals) else float('nan')
        raise RuntimeError(
            f"gap-field GMRES did not converge (info={info}, tol={tol}, "
            f"maxiter={maxiter}, residual reached {last:.2e}). Slowly "
            f"grinding convergence usually means thin-feature conditioning: "
            f"raise maxiter or increase min_metal_width (thin fins are "
            f"intrinsically hard for this operator). A hard stall at O(0.1+) "
            f"residual points to intersecting/overlapping electrode "
            f"surfaces instead.")
    if verbose:
        print(f"[gap_fields] GMRES solved {space.global_dof_count} DOFs "
              f"in {dt:.1f} s ({n_iter} iterations)")
    return GapFieldSolution(model=model, space=space, neumann=neumann,
                            gmres_info=info, solve_time_s=dt,
                            n_iterations=int(n_iter))


def make_bem_efield(design,
                    build_kwargs: Optional[dict] = None,
                    solve_kwargs: Optional[dict] = None,
                    field_kwargs: Optional[dict] = None,
                    verbose: bool = True) -> Tuple[TimedField, GapFieldSolution]:
    """One call: electrodes -> solve -> gridded Field -> TimedField.

    The TimedField's omega/phase are synced from ``design.rf_cavities[0]``
    (omega includes the harmonic; phase is the BUNCH offset only - the per-gap
    0/180 phases are baked into the static potential signs). The tracking
    engine re-syncs both before every run, so ``set_bunch_phase`` /
    ``set_rf_frequency`` stay effective without re-solving.
    """
    model = build_gap_electrodes(design, verbose=verbose, **(build_kwargs or {}))
    solution = solve_gap_field(model, verbose=verbose, **(solve_kwargs or {}))
    static = solution.to_field(verbose=verbose, **(field_kwargs or {}))
    cav = design.rf_cavities[0]
    timed = TimedField(static, omega=float(cav.omega),
                       phase=float(cav.bunch_phase_offset))
    return timed, solution


# ============================================================================
# Diagnostics
# ============================================================================
def plot_electrode_footprints(model: ElectrodeModel, ax=None, show_chains: bool = True):
    """Plot the wedge footprints colored by potential (median-plane view)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    pots = [w.potential for w in model.wedges]
    p_max = max(abs(min(pots)), abs(max(pots)), 1.0)
    for w in model.wedges:
        c = plt.cm.coolwarm(0.5 + 0.5 * w.potential / p_max)
        ax.add_patch(MplPolygon(w.polygon, closed=True, facecolor=c,
                                edgecolor='k', linewidth=0.5, alpha=0.8))
        if show_chains:
            ax.plot(w.chain_lo[:, 0], w.chain_lo[:, 1], 'k-', lw=0.5)
            ax.plot(w.chain_hi[:, 0], w.chain_hi[:, 1], 'k-', lw=0.5)
    r = model.extent() * 1.05
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Gap electrodes ({model.n_elements} elements)')
    return ax
