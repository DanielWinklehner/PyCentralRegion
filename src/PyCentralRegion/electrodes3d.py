"""
electrodes3d.py - 3D electrode solids for the central-region BEM (OpenCASCADE
via gmsh).

Turns the 2D electrode FOOTPRINTS of ``gap_fields.build_gap_electrodes`` (dee
and ground wedges, scroll / central post - every in-plane feature the 2D
optimizer produced: trims, scroll, spokes, fillets) into closed 3D metal
solids with the vertical structure of a real cavity center, surface-meshes
them with gmsh, and returns an ``ElectrodeModel`` that
``gap_fields.solve_gap_field`` solves unchanged:

  * DEE: footprint prism, clipped to the outer height H_d(r) and hollowed by
    the aperture h_d(r); both are solids of revolution about the machine axis,
    so a dee that is lower / narrower toward the center is one radial profile
    each (``HeightProfiles``). Inside the crop a dee is normally TWO plates
    (they join only beyond the crop). Optional edge BARS: aperture-closing
    walls along an interval of a gap chain (the IBA-style tip bars that keep
    the fringe field out of the dee between beam crossings).
  * GROUND: hill wedges with the hill gap h_g(r), the scroll / central post as
    a full-height solid, grounded POSTS (vertical cylinders), and the valley
    ROOF (liner / valley floor at +-valley_height/2) over everything that is
    not hill or scroll. The ground is ONE solid by construction: the crop
    cylinder minus the valley prism (a 2D boolean of the footprints,
    extruded) minus the hill-gap solid of revolution; bars are notches in
    the gap / aperture tools. No OCC fuse is involved (fusing ~30 parts with
    coincident faces cost minutes per call); only grounded external STEP
    solids are fused in, optionally (``ExtraSolid.fuse``).
  * CROP: cylinder r <= r_cut, |z| <= z_cut (open truncation - the field is
    trustworthy about one pole gap inside r_cut; keep the seam annulus there).

With every profile set to None the builder reproduces the tall-wall 2D model
(plain prisms of height 2*z_cut), which is the regression test against
``build_gap_electrodes`` + ``solve_gap_field``.

Units: the 2D model and every parameter are SI [m]; OCC modeling is done in
mm for robustness and converted back. gmsh is imported lazily (optional
dependency; the pip wheel bundles OpenCASCADE).
"""

import time
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .gap_fields import ElectrodeModel, Wedge, offset_polyline, _voltage_scale

Profile = Union[None, float, Callable[[np.ndarray], np.ndarray], np.ndarray]
MM = 1000.0


# ============================================================================
# Parameters
# ============================================================================
def _as_profile(p: Profile, name: str) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Normalize a height profile to a vectorized callable h(r [m]) -> [m] (or None)."""
    if p is None:
        return None
    if callable(p):
        return p
    if np.isscalar(p):
        v = float(p)
        return lambda r: np.full(np.shape(r), v, dtype=float)
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name}: expected None, a float, a callable or an (N, 2) "
                         f"table of (r [m], h [m])")
    order = np.argsort(arr[:, 0])
    rr, hh = arr[order, 0], arr[order, 1]
    return lambda r: np.interp(np.asarray(r, dtype=float), rr, hh)


@dataclass
class HeightProfiles:
    """Vertical structure of the cavity center, all FULL heights [m].

    dee_aperture    h_d(r): inner height of the dee (beam channel). None = no
                    aperture (solid dee).
    dee_height      H_d(r): outer height of the dee. None = full crop height.
    ground_aperture h_g(r): hill gap between the pole / dummy-dee faces. None =
                    solid ground wedges (tall-wall limit).
    valley_height   full height between the valley floors above / below the
                    dee (the liner). None = no roof.
    z_cut, r_cut    crop half-height and radius [m].

    Each profile is a float (constant), a vectorized callable of r [m], or an
    (N, 2) table (r, h) in m; tables are interpolated linearly with the end
    values held.
    """
    dee_aperture: Profile = None
    dee_height: Profile = None
    ground_aperture: Profile = None
    valley_height: Optional[float] = None
    z_cut: float = 0.060
    r_cut: float = 0.250

    @classmethod
    def iba(cls, r_cut: float = 0.250, z_cut: float = 0.060) -> 'HeightProfiles':
        """The IBA PDR cavity center as measured on the COMSOL vacuum CAD
        (dee aperture 26 -> 50 mm, dee height 66 -> 90 mm, hill gap 34 -> 48
        mm between r ~ 0.15 and 0.25 m, valley floors at +-55 mm)."""
        return cls(dee_aperture=np.array([[0.00, 0.026], [0.148, 0.026], [0.247, 0.050], [1.0, 0.050]]),
                   dee_height=np.array([[0.00, 0.066], [0.148, 0.066], [0.247, 0.090], [1.0, 0.090]]),
                   ground_aperture=np.array([[0.00, 0.034], [0.125, 0.034], [0.235, 0.048],
                                             [0.30, 0.050], [1.0, 0.050]]),
                   valley_height=0.110, z_cut=z_cut, r_cut=r_cut)

    @classmethod
    def tall_wall(cls, height: float, r_cut: float) -> 'HeightProfiles':
        """Plain prisms of the given full height: the 2D mesher's geometry."""
        return cls(z_cut=0.5 * height, r_cut=r_cut)


@dataclass
class Bar:
    """Aperture-closing wall (full electrode height) along an edge of a wedge.

    wedge   label of the wedge ('dee0', 'ground0-1', ...); side 'lo' or 'hi'
    (the wedge's low- / high-azimuth gap chain) with the radial interval
    [r_from, r_to] [m] along that chain, or side 'tip' for the wedge's whole
    INNER edge (the trimmed tip front; r_from / r_to unused); bar width [m]
    measured into the wedge. Bars are notches in the aperture (dee) / hill
    gap (ground) tools, so they close the aperture over its full height and
    are part of their electrode by construction (no fuse).
    """
    wedge: str
    side: str
    r_from: float = 0.0
    r_to: float = 0.0
    width: float = 0.010


@dataclass
class Post:
    """Grounded vertical cylinder (full crop height) at (x, y) [m]."""
    x: float
    y: float
    radius: float = 0.005


@dataclass
class Patch:
    """Extra metal block: a footprint polygon (K, 2) [m] extruded between z0
    and z1 [m], added as its OWN closed solid (grounded by default, or a
    powered electrode). It may overlap other metal - overlapping bodies are
    fine for the Dirichlet BEM, exactly coincident faces are not - so let a
    patch penetrate its neighbours by ~0.5-1 mm rather than touch them.
    Used for junction fillets / bridges, e.g. between the inflector housing
    plate and the bar of the neighbouring dummy dee."""
    polygon: np.ndarray
    z0: float
    z1: float
    potential: float = 0.0
    name: str = ''


@dataclass
class ExtraSolid:
    """An external CAD solid (STEP) placed into the model, e.g. the spiral
    inflector housing exported by spyral_inflector / py_electrodes.

    path         STEP file; every volume in it is used.
    potential    Dirichlet value [V]; 0 fuses it into the ground solid (so it
                 may overlap the scroll / hills), any other value makes it a
                 separate electrode.
    scale        multiply the file's coordinates by this to get metres
                 (py_electrodes writes metres -> 1.0; a mm file -> 1e-3).
    rotation_deg rotation about the machine axis (z) applied after scaling.
    mirror_z     mirror through the median plane z = 0 (deck-frame inflector
                 files -> machine frame; commutes with the z rotation).
    translation  (dx, dy, dz) [m] applied last.
    name         label (default: the file stem).
    fuse         grounded solids only: True (default) puts the solid into the
                 one OCC fuse of all ground parts; False keeps it a separate
                 grounded body (no boolean with the other parts). Use False
                 for a complex B-rep such as the housing STEP (~1400 faces
                 with splines and tiny edges), where the general fuse takes
                 20+ min. Overlapping metal is allowed: faces inside a
                 grounded region carry no charge in the Dirichlet BEM, they
                 only cost a few elements.
    crop         False skips the crop against the model cylinder (r_cut,
                 z_cut) for this solid, i.e. it keeps whatever sticks out of
                 the model volume; with fuse=False that also avoids the
                 second expensive boolean on the B-rep.
    """
    path: str
    potential: float = 0.0
    scale: float = 1.0
    rotation_deg: float = 0.0
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = ''
    mirror_z: bool = False
    fuse: bool = True
    crop: bool = True


# ============================================================================
# OCC helpers (mm)
# ============================================================================
def _dedupe_ring(poly_m: np.ndarray, tol_m: float = 2e-6) -> np.ndarray:
    pts = np.asarray(poly_m, dtype=float)
    keep = [pts[0]]
    for p in pts[1:]:
        if np.hypot(*(p - keep[-1])) > tol_m:
            keep.append(p)
    if len(keep) > 2 and np.hypot(*(keep[0] - keep[-1])) <= tol_m:
        keep.pop()
    return np.array(keep)


def _prism(occ, poly_m: np.ndarray, z0_mm: float, z1_mm: float) -> int:
    """Extruded polygon footprint (m) between z0 and z1 (mm); returns the volume tag."""
    pts = _dedupe_ring(poly_m) * MM
    if len(pts) < 3:
        raise ValueError("footprint polygon degenerates to fewer than 3 points")
    p_tags = [occ.addPoint(x, y, z0_mm) for x, y in pts]
    l_tags = [occ.addLine(p_tags[i], p_tags[(i + 1) % len(p_tags)]) for i in range(len(p_tags))]
    loop = occ.addCurveLoop(l_tags)
    surf = occ.addPlaneSurface([loop])
    out = occ.extrude([(2, surf)], 0.0, 0.0, z1_mm - z0_mm)
    vols = [t for d, t in out if d == 3]
    if len(vols) != 1:
        raise RuntimeError("extrusion did not produce exactly one volume")
    return vols[0]


def _face(occ, poly_m: np.ndarray, z_mm: float, holes: Sequence[np.ndarray] = ()) -> int:
    """Plane surface at height z from a closed polygon (m) with optional hole
    polygons (m); returns the surface tag."""
    loops = []
    for ring in (poly_m, *holes):
        pts = _dedupe_ring(np.asarray(ring, dtype=float)) * MM
        if len(pts) < 3:
            raise ValueError("face polygon degenerates to fewer than 3 points")
        p_tags = [occ.addPoint(x, y, z_mm) for x, y in pts]
        l_tags = [occ.addLine(p_tags[i], p_tags[(i + 1) % len(p_tags)]) for i in range(len(p_tags))]
        loops.append(occ.addCurveLoop(l_tags))
    return occ.addPlaneSurface(loops)


BAR_OVERSHOOT = 0.0005      # [m] notch tools reach this far outside their wedge edge


def _prism_with_hole(occ, outer_m: np.ndarray, inner_m: np.ndarray, z0_mm: float, z1_mm: float) -> int:
    """Extruded ring: outer footprint minus the inner footprint (both m), as ONE
    plane surface with a hole - no boolean involved."""
    loops = []
    for poly in (outer_m, inner_m):
        pts = _dedupe_ring(poly) * MM
        p_tags = [occ.addPoint(x, y, z0_mm) for x, y in pts]
        l_tags = [occ.addLine(p_tags[i], p_tags[(i + 1) % len(p_tags)]) for i in range(len(p_tags))]
        loops.append(occ.addCurveLoop(l_tags))
    surf = occ.addPlaneSurface(loops)
    out = occ.extrude([(2, surf)], 0.0, 0.0, z1_mm - z0_mm)
    vols = [t for d, t in out if d == 3]
    if len(vols) != 1:
        raise RuntimeError("ring extrusion did not produce exactly one volume")
    return vols[0]


def _revolved(occ, h_of_r: Callable, r_max_m: float, n_min: int = 4) -> int:
    """Solid of revolution |z| <= h(r)/2 for 0 <= r <= r_max (mm tag)."""
    r = np.linspace(0.0, r_max_m, max(n_min, int(np.ceil(r_max_m / 0.0025)) + 1))
    h = np.asarray(h_of_r(r), dtype=float)
    if np.any(h <= 0):
        raise ValueError("height profile must be positive everywhere")
    keep = [0]                                  # drop samples on straight stretches
    for i in range(1, len(r) - 1):
        z_lin = h[keep[-1]] + (h[i + 1] - h[keep[-1]]) * (r[i] - r[keep[-1]]) / (r[i + 1] - r[keep[-1]])
        if abs(h[i] - z_lin) > 1e-7:
            keep.append(i)
    keep.append(len(r) - 1)
    r, h = r[keep] * MM, 0.5 * h[keep] * MM
    lower = [occ.addPoint(ri, 0.0, -hi) for ri, hi in zip(r, h)]
    upper = [occ.addPoint(ri, 0.0, hi) for ri, hi in zip(r[::-1], h[::-1])]
    ring = lower + upper
    lines = [occ.addLine(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
    surf = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    out = occ.revolve([(2, surf)], 0, 0, 0, 0, 0, 1, 2.0 * np.pi)
    vols = [t for d, t in out if d == 3]
    if len(vols) != 1:
        raise RuntimeError("revolve did not produce exactly one volume")
    return vols[0]


def _offset_ring(poly_m: np.ndarray, inward: float) -> np.ndarray:
    """Closed polygon offset INWARD by ``inward`` [m] (miter joins), for the
    hollow scroll: the ring between ``poly`` and this is the rim metal."""
    pts = _dedupe_ring(poly_m)
    # signed area -> orientation; the interior of a CCW ring is LEFT of travel,
    # and offset_polyline's positive offset goes left
    area = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]))
    sign = +1.0 if area > 0 else -1.0
    wrapped = np.vstack([pts[-1], pts, pts[0], pts[1]])
    off = offset_polyline(wrapped, sign * inward)[1:-2]
    return off


def _import_step(occ, gmsh, extra: 'ExtraSolid') -> List[int]:
    """Import a STEP file, scale it to mm, rotate about z and translate; returns volume tags."""
    out = occ.importShapes(extra.path)
    vols = [(d, t) for d, t in out if d == 3]
    if not vols:
        raise RuntimeError(f"{extra.path}: no volumes in the STEP file")
    f = extra.scale * MM
    if abs(f - 1.0) > 1e-12:
        occ.dilate(vols, 0.0, 0.0, 0.0, f, f, f)
    if extra.rotation_deg:
        occ.rotate(vols, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, np.radians(extra.rotation_deg))
    if getattr(extra, 'mirror_z', False):
        occ.mirror(vols, 0.0, 0.0, 1.0, 0.0)          # plane z = 0
    dx, dy, dz = extra.translation
    if dx or dy or dz:
        occ.translate(vols, dx * MM, dy * MM, dz * MM)
    return [t for d, t in vols]


def _inner_edge(wedge: Wedge) -> np.ndarray:
    """The wedge's inner (tip) edge from its polygon ring, chain_hi[0] -> chain_lo[0].

    The ring runs chain_lo inward->outward, outer arc, chain_hi outward->inward,
    then the inner edge back to the start; it is counter-clockwise, so the
    wedge interior is LEFT of travel along this edge."""
    poly = np.asarray(wedge.polygon, dtype=float)
    lo0 = np.asarray(wedge.chain_lo[0], dtype=float)
    hi0 = np.asarray(wedge.chain_hi[0], dtype=float)
    i_lo = int(np.argmin(np.hypot(*(poly - lo0).T)))
    i_hi = int(np.argmin(np.hypot(*(poly - hi0).T)))
    if i_hi <= i_lo:
        seg = poly[i_hi:i_lo + 1]
    else:
        seg = np.vstack([poly[i_hi:], poly[:i_lo + 1]])
    if len(seg) < 2:
        raise ValueError(f"{wedge.label}: could not extract the inner edge")
    return seg


def _bar_footprint(wedge: Wedge, bar: Bar, overshoot: float = 0.0) -> np.ndarray:
    """Closed strip polygon (m) of a bar along a wedge chain interval or its
    inner edge; ``overshoot`` [m] extends the strip OUTWARD past the wedge
    edge (used for the notch tools so no face coincides with a wedge wall)."""
    if bar.side == 'tip':
        seg = _inner_edge(wedge)
        off = offset_polyline(seg, +bar.width)          # left of travel = into the wedge
        outer = offset_polyline(seg, -overshoot) if overshoot else seg
        return np.vstack([outer, off[::-1]])
    if bar.side not in ('lo', 'hi'):
        raise ValueError("Bar.side must be 'lo', 'hi' or 'tip'")
    chain = np.asarray(wedge.chain_lo if bar.side == 'lo' else wedge.chain_hi, dtype=float)
    r = np.hypot(chain[:, 0], chain[:, 1])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(chain, axis=0).T))])
    if bar.r_from < r.min() - 1e-9 or bar.r_to > r.max() + 1e-9 or bar.r_to <= bar.r_from:
        raise ValueError(f"Bar on {bar.wedge}/{bar.side}: interval [{bar.r_from}, {bar.r_to}] m "
                         f"outside the chain radii {r.min():.4f}..{r.max():.4f} m")
    s_from = float(np.interp(bar.r_from, r, s))     # chains are monotonic in r
    s_to = float(np.interp(bar.r_to, r, s))
    inside = (s > s_from) & (s < s_to)
    s_pts = np.concatenate([[s_from], s[inside], [s_to]])
    seg = np.column_stack([np.interp(s_pts, s, chain[:, 0]), np.interp(s_pts, s, chain[:, 1])])
    sign = +1.0 if bar.side == 'lo' else -1.0          # into the wedge (see gap_fields)
    off = offset_polyline(seg, sign * bar.width)
    outer = offset_polyline(seg, -sign * overshoot) if overshoot else seg
    return np.vstack([outer, off[::-1]])


def _wedge(model2d: ElectrodeModel, label: str) -> Wedge:
    for w in model2d.wedges:
        if w.label == label:
            return w
    raise ValueError(f"unknown wedge {label!r} (have {[w.label for w in model2d.wedges]})")


def _densify(polyline: np.ndarray, ds: float = 5e-4) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a polyline to ~``ds`` spacing; returns (points (K, 2), arc length (K,))."""
    pl = np.asarray(polyline, dtype=float)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pl, axis=0).T))])
    n = max(2, int(np.ceil(s[-1] / ds)) + 1)
    ss = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(ss, s, pl[:, 0]), np.interp(ss, s, pl[:, 1])]), ss


def _nearest_trajectory_radius(pts: np.ndarray, trajectory) -> np.ndarray:
    """Radius [m] of the trajectory sample nearest to each point (KD-tree)."""
    from scipy.spatial import cKDTree
    xy = np.asarray(trajectory, dtype=float)[:, :2]
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    _, k = cKDTree(xy).query(pts)
    return np.hypot(xy[k, 0], xy[k, 1])


def _distance_to_trajectory(pts: np.ndarray, trajectory) -> np.ndarray:
    """Euclidean midplane distance [m] from points to a trajectory polyline:
    the nearest sample (KD-tree) refined to the two polyline segments meeting
    there, so coarse sampling of the outer turns does not bias the result."""
    from scipy.spatial import cKDTree
    xy = np.asarray(trajectory, dtype=float)[:, :2]
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    d, k = cKDTree(xy).query(pts)
    if len(xy) < 2:
        return d
    for ka, kb in ((np.maximum(k - 1, 0), k), (k, np.minimum(k + 1, len(xy) - 1))):
        a, b = xy[ka], xy[kb]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-18)
        t = np.clip(((pts - a) * ab).sum(1) / L2, 0.0, 1.0)
        d = np.minimum(d, np.hypot(*(pts - (a + t[:, None] * ab)).T))
    return d


def _runs(mask: np.ndarray):
    """(start, end) index pairs of the True runs of a boolean array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(np.diff(m.astype(int)))
    return list(zip(edges[0::2], edges[1::2] - 1))


def bar_clearances(model2d: ElectrodeModel, bars: Sequence[Bar], trajectory) -> List[float]:
    """Midplane clearance [m] between the trajectory and each bar's footprint:
    the minimum distance to the footprint boundary, NEGATIVE (minus the deepest
    penetration) when trajectory points lie inside the footprint."""
    from matplotlib.path import Path as MplPath
    xy = np.asarray(trajectory, dtype=float)[:, :2]
    out = []
    for b in bars:
        fp = _bar_footprint(_wedge(model2d, b.wedge), b)
        ring, _ = _densify(np.vstack([fp, fp[:1]]))
        inside = MplPath(fp).contains_points(xy)
        if inside.any():
            from scipy.spatial import cKDTree
            depth = cKDTree(ring).query(xy[inside])[0].max()
            out.append(-float(depth))
        else:
            out.append(float(_distance_to_trajectory(ring, xy).min()))
    return out


def auto_bars(model2d: ElectrodeModel, design_or_cavities, trajectory,
              margin: float = 0.008, r_max: float = 0.20, width: float = 0.010,
              tip: bool = True, ground: bool = True, tip_width: float = 0.003,
              beam_halfwidth: float = 0.0, margin_inner: Optional[float] = None,
              min_length: Optional[float] = None) -> List[Bar]:
    """Bars between the beam crossings, IBA style, from a reference trajectory.

    A bar goes on every interval of a gap chain (dee wedges and, with
    ``ground``, hill wedges) that stays farther than ``margin + beam_halfwidth``
    from the trajectory - a EUCLIDEAN midplane clearance to the centroid path
    plus the bunch half-width, so oblique crossings and the beam running
    along a chain near the tips are handled alike - out to ``r_max``;
    intervals shorter than 2 x ``width`` are skipped. With ``tip`` every dee
    also gets a wall of ``tip_width`` along its whole inner edge (a 2-3 mm
    sheet closes the aperture as well as a thick bar). That wall is
    mandatory, so when its outer face comes within the clearance of the
    trajectory a warning reports by how much: the remedy is the 2D tip
    clearance (``traj_tip_clearance`` in build_gap_electrodes) or the
    injection radius, not the bar. ``design_or_cavities`` is accepted for
    API compatibility; the intervals follow from the trajectory distance.
    ``margin_inner`` (default ``margin``) replaces ``margin`` on the INNER
    side of every opening - chain points whose radius is below that of the
    nearest trajectory point - so an opening can be widened toward the axis
    only (particles off the design phase gain less and run further in).
    ``min_length`` (default 2 * width) drops shorter free intervals.
    Use ``bar_clearances`` / ``check_trajectory_clearance`` to verify.
    """
    min_len = 2.0 * width if min_length is None else float(min_length)
    clearance = margin + beam_halfwidth
    clearance_in = (margin if margin_inner is None else margin_inner) + beam_halfwidth
    ds_chain = 5e-4
    n_f = max(3, int(np.ceil(width / 1e-3)) + 1)          # samples across the strip
    # the strip is sampled at ds_chain along the chain and width/(n_f-1) across
    # it; add half of the coarser spacing so the bar footprints (checked with
    # bar_clearances) never fall short of the requested clearance
    pad = 0.5 * max(ds_chain, width / (n_f - 1))
    clearance_eff = clearance + pad
    clearance_in_eff = clearance_in + pad
    bars: List[Bar] = []
    notes = []
    for w in model2d.wedges:
        if w.kind == 'dee' or (w.kind == 'ground' and ground):
            for side in ('lo', 'hi'):
                pts, _ = _densify(w.chain_lo if side == 'lo' else w.chain_hi, ds_chain)
                r = np.hypot(pts[:, 0], pts[:, 1])
                # the whole strip (chain edge, offset edge and the end faces of
                # every candidate interval) must keep the clearance, not just
                # the chain line: a bar's inner corner can sit closer to the
                # beam running inside the wedge than the chain does
                off = offset_polyline(pts, (+1.0 if side == 'lo' else -1.0) * width)
                d_min = np.full(len(pts), np.inf)
                for f in np.linspace(0.0, 1.0, n_f):
                    d_min = np.minimum(d_min, _distance_to_trajectory(pts + f * (off - pts), trajectory))
                inner = r < _nearest_trajectory_radius(pts, trajectory)
                free = (d_min >= np.where(inner, clearance_in_eff, clearance_eff)) & (r <= r_max)
                for a, b in _runs(free):
                    r_from, r_to = float(r[a:b + 1].min()), float(r[a:b + 1].max())
                    if r_to - r_from >= min_len:
                        bars.append(Bar(w.label, side, r_from, r_to, width))
        if w.kind == 'dee' and tip:
            bar = Bar(w.label, 'tip', 0.0, 0.0, tip_width)
            bars.append(bar)
            c = bar_clearances(model2d, [bar], trajectory)[0]
            if c < clearance:
                notes.append(f"{w.label}: tip wall ({tip_width * 1e3:.1f} mm) outer face "
                             f"{c * 1e3:+.1f} mm from the trajectory")
    if notes:
        warnings.warn(
            f"auto_bars: tip walls closer than the {clearance * 1e3:.1f} mm clearance "
            f"(margin {margin * 1e3:.1f} + beam half-width {beam_halfwidth * 1e3:.1f} mm) - "
            + "; ".join(notes) + ". Increase the 2D tip clearance (traj_tip_clearance) "
            "or the injection radius; the wall itself is mandatory.", stacklevel=2)
    return bars


def _boolean(occ, op, objects: Dict[str, List[int]], tool: int) -> Dict[str, List[int]]:
    """Apply an OCC boolean (occ.cut / occ.intersect) of ONE tool to several
    keyed object groups, one call per key with a fresh copy of the tool.

    One call for all groups would let OCC treat overlapping groups (a bar
    inside its dee, a housing inside the scroll) as a single argument and
    scramble the per-input result map. Each copy of the tool is consumed by
    its call, so no result shares sub-shapes with a dangling entity; the
    original tool is removed at the end."""
    result: Dict[str, List[int]] = {}
    for k, vols in objects.items():
        if not vols:
            result[k] = []
            continue
        tool_copy = occ.copy([(3, tool)])[0][1]
        out, _ = op([(3, v) for v in vols], [(3, tool_copy)], removeObject=True, removeTool=True)
        result[k] = [t for d, t in out if d == 3]
    occ.remove([(3, tool)], recursive=True)
    return result


# ============================================================================
# Builder
# ============================================================================
def build_electrodes_3d(model2d: ElectrodeModel,
                        profiles: HeightProfiles,
                        bars: Sequence[Bar] = (),
                        posts: Sequence[Post] = (),
                        extra_solids: Sequence[ExtraSolid] = (),
                        patches: Sequence[Patch] = (),
                        scroll_thickness: Optional[float] = None,
                        size_min: float = 0.002,
                        size_max: float = 0.012,
                        dist_min: float = 0.004,
                        dist_max: float = 0.050,
                        r_fine_ground: Optional[float] = None,
                        edge_fillet: Optional[float] = None,
                        heal_tolerance: Optional[float] = None,
                        voltage_profile=None,
                        geometry_out: Optional[str] = None,
                        verbose: bool = True) -> ElectrodeModel:
    """Closed 3D metal solids from the 2D footprints + vertical profiles -> surface mesh.

    Parameters
    ----------
    model2d : ElectrodeModel
        Output of ``gap_fields.build_gap_electrodes`` (its wedges' polygons and
        chains are the footprints; its potentials are reused).
    profiles : HeightProfiles
        Vertical structure and crop (see the class).
    bars, posts : sequences of Bar / Post
        Aperture-closing walls on dee chains and grounded vertical cylinders.
    extra_solids : sequence of ExtraSolid
        External STEP solids (e.g. the spiral-inflector housing) placed with
        their own scale / rotation / shift; grounded ones are fused into the
        ground solid, others become separate electrodes.
    patches : sequence of Patch
        Extra metal blocks (footprint polygon x z-range), each its own solid;
        grounded ones join the ground group, others are separate electrodes.
    scroll_thickness : float, optional
        Make the scroll / central post a RING of this wall thickness [m]
        instead of a filled prism, so that a housing placed inside it (see
        ``extra_solids``) shapes the interior. Default: filled.
    size_min, size_max, dist_min, dist_max : float [m]
        gmsh size field: ``size_min`` within ``dist_min`` of the vertical
        (gap-facing) dee walls and the central ground features, growing to
        ``size_max`` at ``dist_max``; on top of that the size grows
        quadratically with |z| to ``size_max`` at ``z_cut``, so elements are
        fine where the beam is and coarse on plate flats, roof, caps and the
        far parts of tall walls.
    r_fine_ground : float, optional
        Ground walls inside this radius also seed the fine size (default: the
        scroll / post extent + 20 mm).
    edge_fillet : float, optional
        Fillet radius [m] applied to the dee body edges (after the aperture
        cut, before the bars). Skipped with a warning if OCC cannot do it.
    heal_tolerance : float, optional
        OCC shape-healing tolerance [m] applied to the finished solids
        (removes sliver edges / faces the booleans may leave). Off by default.
    voltage_profile : see ``gap_fields.build_gap_electrodes``.
    geometry_out : str, optional
        Path prefix: write the finished solids (before meshing) to ``<prefix>.brep`` and ``<prefix>.step`` [mm].
    """
    import gmsh
    t_start = time.time()
    v_scale, v_info = _voltage_scale(voltage_profile)
    h_d = _as_profile(profiles.dee_aperture, 'dee_aperture')
    H_d = _as_profile(profiles.dee_height, 'dee_height')
    h_g = _as_profile(profiles.ground_aperture, 'ground_aperture')
    z_cut = float(profiles.z_cut)
    r_cut = float(profiles.r_cut)
    if profiles.valley_height is not None and profiles.valley_height >= 2 * z_cut:
        raise ValueError("valley_height must be smaller than 2 * z_cut for the roof to exist")
    if H_d is not None and float(np.max(H_d(np.linspace(0, r_cut, 50)))) > 2 * z_cut + 1e-9:
        raise ValueError("dee_height exceeds the crop height 2 * z_cut")

    dee_wedges = [w for w in model2d.wedges if w.kind == 'dee']
    hill_wedges = [w for w in model2d.wedges if w.kind == 'ground']
    post_wedges = [w for w in model2d.wedges if w.kind == 'post']

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.option.setNumber("General.Verbosity", 2)
    occ = gmsh.model.occ
    zc = z_cut * MM
    zp = zc + 1.0          # prisms overshoot the crop: coincident cap faces break OCC's intersect
    r_tool = r_cut + 0.010

    # ---- 1. tools --------------------------------------------------------------------
    # Everything is CUT from a few simple solids (crop cylinder, prisms, solids
    # of revolution): no per-part crop, no bar fuses, no N-part ground fuse.
    # OCC's general fuse of many parts sharing coincident faces cost minutes
    # per call; cuts of simple solids cost well under a second. The bars are
    # NOTCHES in the aperture tools: what the aperture cut leaves standing in
    # a notch is the bar, joined to the plates by construction. Bars overshoot
    # their wedge edge outward by BAR_OVERSHOOT so no notch face coincides
    # with a wedge wall.
    known_wedges = {w.label for w in dee_wedges + hill_wedges}
    for b in bars:
        if b.wedge not in known_wedges:
            raise ValueError(f"Bar on unknown wedge {b.wedge!r}; wedges: {sorted(known_wedges)}")
    dee_bars = [(w, b) for w in dee_wedges for b in bars if b.wedge == w.label]
    ground_bars = [(w, b) for w in hill_wedges for b in bars if b.wedge == w.label]
    if dee_bars and h_d is None:
        warnings.warn("dee bars ignored: the dees have no aperture (solid plates)", stacklevel=2)
        dee_bars = []
    if ground_bars and h_g is None:
        warnings.warn("ground bars ignored: the hills have no gap (solid wedges)", stacklevel=2)
        ground_bars = []
    housing_mode = model2d.params.get('post_mode') == 'housing'
    crop = occ.addCylinder(0, 0, -zc, 0, 0, 2 * zc, r_cut * MM)
    t_lap = [time.time()]

    def lap(label):
        if verbose:
            now = time.time()
            print(f"[electrodes3d] {label}: {now - t_lap[0]:.1f} s")
            t_lap[0] = now

    def one_volume(out, what):
        vols = [t for d, t in out if d == 3]
        if len(vols) != 1:
            raise RuntimeError(f"{what}: expected one solid, got {len(vols)}")
        return vols[0]

    def notched(profile, pairs, keep=()):
        """Solid of revolution |z| <= profile(r)/2 minus the bar prisms
        (notches) and minus the ``keep`` prisms (regions that must stay full
        height, e.g. the scroll / posts)."""
        tool = _revolved(occ, profile, r_tool)
        tools = [(3, _prism(occ, _bar_footprint(w, b, overshoot=BAR_OVERSHOOT), -zp, zp)) for w, b in pairs]
        tools += [(3, t) for t in keep]
        if tools:
            out, _ = occ.cut([(3, tool)], tools, removeObject=True, removeTool=True)
            tool = one_volume(out, "aperture tool with notches")
        return tool

    # ---- 2. dees: prism ∩ (crop ∩ outer height) - (aperture - bar notches) ------------
    dee_tool = occ.copy([(3, crop)])[0][1]
    if H_d is not None:
        out, _ = occ.intersect([(3, dee_tool)], [(3, _revolved(occ, H_d, r_tool))],
                               removeObject=True, removeTool=True)
        dee_tool = one_volume(out, "dee crop/height tool")
    aperture_tool = notched(h_d, dee_bars) if h_d is not None else None
    solids: Dict[str, List[int]] = {}
    for w in dee_wedges:
        body = _prism(occ, w.polygon, -zp, zp)
        out, _ = occ.intersect([(3, body)], [(3, occ.copy([(3, dee_tool)])[0][1])],
                               removeObject=True, removeTool=True)
        bodies = [t for d, t in out if d == 3]
        if aperture_tool is not None and bodies:
            out, _ = occ.cut([(3, t) for t in bodies], [(3, occ.copy([(3, aperture_tool)])[0][1])],
                             removeObject=True, removeTool=True)
            bodies = [t for d, t in out if d == 3]
        if not bodies:
            raise RuntimeError(f"{w.label}: no metal left after the crop / height / aperture booleans")
        solids[w.label] = bodies
    occ.remove([(3, dee_tool)], recursive=True)
    if aperture_tool is not None:
        occ.remove([(3, aperture_tool)], recursive=True)
    lap(f"{len(dee_wedges)} dees (crop, outer height, aperture with {len(dee_bars)} bar notches)")
    dee_keys = [w.label for w in dee_wedges]
    if edge_fillet:
        occ.synchronize()
        for k in dee_keys:
            filleted = []
            for body in solids[k]:
                curves = []
                for d, c in gmsh.model.getBoundary([(3, body)], oriented=False, recursive=True):
                    if d == 1:
                        b = gmsh.model.getBoundingBox(1, c)
                        if np.hypot(max(abs(b[0]), abs(b[3])), max(abs(b[1]), abs(b[4]))) < (r_cut - 0.002) * MM:
                            curves.append(c)
                try:
                    out = occ.fillet([body], curves, [edge_fillet * MM], removeVolume=True)
                    filleted.extend(t for d, t in out if d == 3)
                except Exception as exc:                  # OCC fillet failures are common
                    warnings.warn(f"{k}: edge fillet failed ({exc}); continuing without", stacklevel=2)
                    filleted.append(body)
            solids[k] = filleted
        lap('edge fillets')

    # ---- 3. ground: crop cylinder - valley prism - (hill gap - bar notches - hub) ------
    # The valley footprint is a 2D boolean (disk minus the hill / hub / post
    # footprints), extruded to the valley height (or through the whole crop
    # when there is no roof); hills, hub and posts are what the cut leaves.
    hubs: List[Tuple[np.ndarray, List[np.ndarray]]] = []      # (outer polygon, holes) at full height
    if housing_mode:
        # the 2D hub is the housing OUTLINE fused with the dummy-dee spokes: a
        # tall-wall stand-in. In 3D the merged dummy dees are ordinary ground
        # (hill) wedges - their polygons already run into the housing wall and
        # they get the hill profile like every other dummy dee - and the
        # housing itself should come from its STEP file (a grounded
        # ExtraSolid); without one the outline is extruded, with a warning.
        if not any(float(ex.potential) == 0.0 for ex in extra_solids):
            warnings.warn("housing-mode 2D model without a grounded ExtraSolid: the housing "
                          "outline is extruded as a tall-wall prism (pass the housing STEP "
                          "as ExtraSolid(potential=0) for the real shape)", stacklevel=2)
            hubs += [(np.asarray(w.polygon, dtype=float), [np.asarray(h, dtype=float) for h in w.holes])
                     for w in post_wedges]
    else:
        for w in post_wedges:
            poly = np.asarray(w.polygon, dtype=float)
            hubs.append((poly, [_offset_ring(poly, scroll_thickness)] if scroll_thickness else []))
    roof = profiles.valley_height is not None
    zv = 0.5 * profiles.valley_height * MM if roof else zp
    blockers = [(2, _face(occ, w.polygon, -zv)) for w in hill_wedges]
    blockers += [(2, _face(occ, poly, -zv, holes)) for poly, holes in hubs]
    blockers += [(2, occ.addDisk(p.x * MM, p.y * MM, -zv, p.radius * MM, p.radius * MM)) for p in posts]
    disk = occ.addDisk(0, 0, -zv, (r_cut + 0.001) * MM, (r_cut + 0.001) * MM)
    if blockers:
        out, _ = occ.cut([(2, disk)], blockers, removeObject=True, removeTool=True)
        valley_faces = [t for d, t in out if d == 2]
    else:
        valley_faces = [disk]
    valley = [t for d, t in occ.extrude([(2, f) for f in valley_faces], 0.0, 0.0, 2 * zv) if d == 3]
    out, _ = occ.cut([(3, occ.copy([(3, crop)])[0][1])], [(3, v) for v in valley],
                     removeObject=True, removeTool=True)
    ground_body = [t for d, t in out if d == 3]
    if h_g is not None and ground_body:
        keep = [_prism(occ, poly, -zp, zp) for poly, _ in hubs]
        keep += [occ.addCylinder(p.x * MM, p.y * MM, -zp, 0, 0, 2 * zp, p.radius * MM) for p in posts]
        gap_tool = notched(h_g, ground_bars, keep)
        out, _ = occ.cut([(3, t) for t in ground_body], [(3, gap_tool)], removeObject=True, removeTool=True)
        ground_body = [t for d, t in out if d == 3]
    if not ground_body:
        raise RuntimeError("no ground metal left after the valley / hill-gap booleans")
    lap(f"ground block (valley over {len(blockers)} footprints, hill gap with {len(ground_bars)} bar notches)")

    # ---- 4. external solids: crop, fuse grounded ones into the ground (optional) -----
    extra_keys: List[Tuple[str, float]] = []
    separate_extra: List[str] = []              # grounded, kept out of the ground fuse
    to_crop: Dict[str, List[int]] = {}
    for j, ex in enumerate(extra_solids):
        name = ex.name or f"extra#{j}"
        t_imp = time.time()
        bodies = _import_step(occ, gmsh, ex)
        if getattr(ex, 'crop', True):
            to_crop[name] = bodies
        else:
            solids[name] = bodies
        extra_keys.append((name, float(ex.potential)))
        if float(ex.potential) == 0.0 and not getattr(ex, 'fuse', True):
            separate_extra.append(name)
        if verbose:
            occ.synchronize()
            n_f = len({f for d, f in gmsh.model.getBoundary([(3, b) for b in bodies], oriented=False,
                                                            recursive=False)})
            print(f"[electrodes3d]   {name}: {len(bodies)} body(ies), {n_f} faces imported in "
                  f"{time.time() - t_imp:.1f} s ({'cropped' if getattr(ex, 'crop', True) else 'uncropped'}, "
                  f"{'fused' if name not in separate_extra else 'separate'})")
    for j, pa in enumerate(patches):
        name = pa.name or f"patch#{j}"
        if name in solids:
            raise ValueError(f"duplicate solid name {name!r}")
        solids[name] = [_prism(occ, np.asarray(pa.polygon, dtype=float), float(pa.z0) * MM, float(pa.z1) * MM)]
        if float(pa.potential) == 0.0:
            separate_extra.append(name)          # own grounded body, no boolean
        else:
            extra_keys.append((name, float(pa.potential)))
    if to_crop:
        solids.update(_boolean(occ, occ.intersect, to_crop, occ.copy([(3, crop)])[0][1]))
        for k in to_crop:
            if not solids[k]:
                raise RuntimeError(f"{k}: nothing left after the crop (r_cut {r_cut} m, z_cut {z_cut} m)")
        lap(f"crop of {len(to_crop)} external solid(s)")
    occ.remove([(3, crop)], recursive=True)
    fused_extra = [k for k, pot in extra_keys if pot == 0.0 and k not in separate_extra]
    ground_parts = list(ground_body) + [t for k in fused_extra for t in solids[k]]
    if len(ground_parts) > 1:
        # OCC's fuse fails on pieces touching along exactly coincident faces
        # (it returns nothing). Fall back to the un-fused parts: interior
        # faces inside a grounded region carry no charge in the BEM, they
        # only cost elements.
        parts_copy = [occ.copy([(3, t)])[0][1] for t in ground_parts]
        out, _ = occ.fuse([(3, parts_copy[0])], [(3, t) for t in parts_copy[1:]],
                          removeObject=True, removeTool=True)
        fused = [t for d, t in out if d == 3]
        if fused:
            occ.remove([(3, t) for t in ground_parts], recursive=True)
            ground_solids = fused
        else:
            warnings.warn("OCC could not fuse the ground parts; keeping them separate "
                          "(interior faces are harmless for the Dirichlet BEM)", stacklevel=2)
            ground_solids = list(ground_parts)
        lap(f"ground fuse of {len(ground_parts)} parts ({', '.join(fused_extra)})")
    else:
        ground_solids = list(ground_parts)
    ground_solids += [t for k in separate_extra for t in solids[k]]     # un-fused grounded extras
    dee_solids: List[Tuple[str, float, List[int]]] = [
        (w.label, float(w.potential), solids[w.label]) for w in dee_wedges]
    dee_solids += [(k, pot, solids[k]) for k, pot in extra_keys if pot != 0.0]   # powered extras

    if heal_tolerance:
        occ.synchronize()                     # healShapes only sees bound entities
        healed = []
        for name, pot, bodies in dee_solids:
            out = occ.healShapes([(3, b) for b in bodies], heal_tolerance * MM,
                                 fixDegenerated=True, fixSmallEdges=True, fixSmallFaces=True,
                                 sewFaces=False, makeSolids=False)
            healed.append((name, pot, [t for d, t in out if d == 3] or bodies))
        dee_solids = healed
        out = occ.healShapes([(3, t) for t in ground_solids], heal_tolerance * MM,
                             fixDegenerated=True, fixSmallEdges=True, fixSmallFaces=True,
                             sewFaces=False, makeSolids=False)
        ground_solids = [t for d, t in out if d == 3] or ground_solids
        lap('healShapes')
    occ.synchronize()
    if verbose:
        print(f"[electrodes3d] solids: {len(dee_solids)} dee, {len(ground_solids)} ground"
              + (f" ({len(separate_extra)} separate)" if separate_extra else '')
              + f" in {time.time() - t_start:.1f} s")
        for name, pot, bodies in dee_solids:
            vol = sum(occ.getMass(3, t) for t in bodies) / MM**3 * 1e6
            print(f"[electrodes3d]   {name:14s} {vol:9.1f} cm3 in {len(bodies)} piece(s) @ {pot:+9.1f} V")
        for t in ground_solids:
            print(f"[electrodes3d]   ground {t:7d} {occ.getMass(3, t) / MM**3 * 1e6:9.1f} cm3 @ 0 V")

    # ---- 5. mesh sizing: fine near the vertical gap walls and near the midplane -
    def faces_of(vols):
        return sorted({f for d, f in gmsh.model.getBoundary([(3, t) for t in vols],
                                                            oriented=False, recursive=False)})

    def is_vertical(f):
        try:
            bmin, bmax = gmsh.model.getParametrizationBounds(2, f)
            n = gmsh.model.getNormal(f, [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])])
            return abs(float(n[2])) < 0.5
        except Exception:
            return False

    def wall_curves(faces, r_max_m):
        cs = set()
        for f in faces:
            b = gmsh.model.getBoundingBox(2, f)
            if np.hypot(max(abs(b[0]), abs(b[3])), max(abs(b[1]), abs(b[4]))) >= r_max_m * MM:
                continue
            if not is_vertical(f):
                continue
            for d, c in gmsh.model.getBoundary([(2, f)], oriented=False, recursive=False):
                cs.add(abs(c))
        return sorted(cs)

    dee_faces = faces_of([t for _, _, bodies in dee_solids for t in bodies])
    gnd_faces = faces_of(ground_solids)
    if r_fine_ground is None:
        r_fine_ground = 0.02 + max([float(np.hypot(*np.asarray(w.polygon).T).max()) for w in post_wedges]
                                   + [p.radius + np.hypot(p.x, p.y) for p in posts] + [0.0])
    curves = wall_curves(dee_faces, r_cut - 0.002) + wall_curves(gnd_faces, r_fine_ground)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", curves)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 300)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", size_min * MM)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", size_max * MM)
    gmsh.model.mesh.field.setNumber(2, "DistMin", dist_min * MM)
    gmsh.model.mesh.field.setNumber(2, "DistMax", dist_max * MM)
    gmsh.model.mesh.field.add("MathEval", 3)
    gmsh.model.mesh.field.setString(3, "F", f"{size_min * MM} + {(size_max - size_min) * MM} * "
                                            f"(Abs(z) / {max(z_cut, 1e-6) * MM})^2")
    gmsh.model.mesh.field.add("Max", 4)
    gmsh.model.mesh.field.setNumbers(4, "FieldsList", [2, 3])
    gmsh.model.mesh.field.setAsBackgroundMesh(4)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.4 * size_min * MM)
    groups = []
    for i, (name, pot, bodies) in enumerate(dee_solids):
        tag = gmsh.model.addPhysicalGroup(2, faces_of(bodies), tag=10 + i)
        groups.append((name, pot, tag))
    # un-fused grounded extras are their own solids (they may overlap the
    # fused ground; per-solid outlines let the obstacle rasters OR them)
    separate_bodies = {t for k in separate_extra for t in solids[k]}
    tag = gmsh.model.addPhysicalGroup(2, faces_of([t for t in ground_solids if t not in separate_bodies]), tag=1)
    groups.append(('ground', 0.0, tag))
    for j, k in enumerate(separate_extra):
        tag = gmsh.model.addPhysicalGroup(2, faces_of(solids[k]), tag=2 + j)
        groups.append((k, 0.0, tag))

    # ---- 6. geometry export (BEFORE meshing), then mesh -----------------------------
    # The OCC solids exist here as real BREP with the physical groups already assigned, so this
    # writes the geometry that is ACTUALLY meshed rather than a re-derivation of it. Coordinates
    # are in mm (the builder works in mm internally).
    if geometry_out:
        os.makedirs(os.path.dirname(geometry_out) or '.', exist_ok=True)
        for _ext in ('brep', 'step'):
            try:
                gmsh.write(f"{geometry_out}.{_ext}")
                if verbose:
                    print(f"[electrodes3d] wrote {geometry_out}.{_ext} (CAD, pre-mesh, mm)")
            except Exception as _exc:                      # noqa: BLE001
                warnings.warn(f"geometry export to .{_ext} failed: {type(_exc).__name__}: {_exc}",
                              stacklevel=2)

    t0 = time.time()
    gmsh.model.mesh.generate(2)
    ntag, ncoord, _ = gmsh.model.mesh.getNodes()
    verts = ncoord.reshape(-1, 3) / MM
    remap = {int(t): i for i, t in enumerate(ntag)}
    tris, pots, solid_info = [], [], []
    for name, pot, tag in groups:
        n0 = sum(len(t) for t in tris)
        for f in gmsh.model.getEntitiesForPhysicalGroup(2, tag):
            etypes, etags, enodes = gmsh.model.mesh.getElements(2, f)
            for ty, nodes in zip(etypes, enodes):
                if ty == 2:
                    t = np.vectorize(remap.get)(np.asarray(nodes).reshape(-1, 3))
                    tris.append(t)
                    p = np.full(len(t), pot)
                    if v_scale is not None and pot != 0.0:
                        cen = verts[t].mean(axis=1)
                        p = p * v_scale(np.hypot(cen[:, 0], cen[:, 1]))
                    pots.append(p)
        n1 = sum(len(t) for t in tris)
        solid_info.append({'name': name, 'potential': pot, 'tri_range': (n0, n1)})
    gmsh.finalize()
    tris = np.vstack(tris)
    pots = np.concatenate(pots)
    used = np.unique(tris)
    idx = -np.ones(len(verts), dtype=int)
    idx[used] = np.arange(len(used))
    verts, tris = verts[used], idx[tris]
    areas = 0.5 * np.linalg.norm(np.cross(verts[tris[:, 1]] - verts[tris[:, 0]],
                                          verts[tris[:, 2]] - verts[tris[:, 0]]), axis=1)
    n_bad = int(np.sum(areas < 1e-14))
    if n_bad:
        raise RuntimeError(f"3D electrode mesh has {n_bad} degenerate triangles")
    if verbose:
        tiny = int(np.sum(areas < (0.3 * size_min) ** 2))
        print(f"[electrodes3d] mesh: {len(tris)} triangles, {len(verts)} nodes in "
              f"{time.time() - t0:.1f} s; edge ~ {np.sqrt(areas.min() / 0.433) * MM:.2f}.."
              f"{np.sqrt(areas.max() / 0.433) * MM:.1f} mm ({tiny} tiny); "
              + ", ".join(f"{s['name']} {s['tri_range'][1] - s['tri_range'][0]}" for s in solid_info))
    params = dict(model2d.params)
    params.update({'dim': 3, 'z_cut': z_cut, 'r_cut': r_cut,
                   'dee_aperture': profiles.dee_aperture is not None,
                   'dee_height': profiles.dee_height is not None,
                   'ground_aperture': profiles.ground_aperture is not None,
                   'valley_height': profiles.valley_height,
                   'bars': [vars(b) for b in bars], 'posts': [vars(p) for p in posts],
                   'extra_solids': [vars(e) for e in extra_solids],
                   'scroll_thickness': scroll_thickness,
                   'edge_fillet': edge_fillet, 'size_min': size_min, 'size_max': size_max,
                   'solids': solid_info, 'voltage_profile': v_info})
    return ElectrodeModel(vertices=verts, triangles=tris, potentials=pots,
                          wedges=list(model2d.wedges), params=params)


# ============================================================================
# Diagnostics
# ============================================================================
def _mesh_groups(model: ElectrodeModel) -> List[dict]:
    """Triangle groups of a model: its solids (3D builder) or, for any other
    ElectrodeModel, one group per distinct potential."""
    if model.params.get('solids'):
        return [dict(s) for s in model.params['solids']]
    pots = np.asarray(model.potentials, dtype=float)
    groups = []
    for v in np.unique(pots):
        idx = np.flatnonzero(pots == v)
        groups.append({'name': f"{v:+.0f} V", 'potential': float(v), 'indices': idx})
    return groups


def show_model(model: ElectrodeModel, title: str = 'electrodes', write: Optional[str] = None,
               run_gui: bool = True) -> None:
    """Load a finished electrode mesh (2D or 3D ``ElectrodeModel``) into a
    fresh gmsh session as discrete surfaces - one per solid / potential, named
    physical groups, coloured per group - optionally write it (``write`` =
    .msh / .vtk / .stl ... path, any format gmsh writes) and open the gmsh
    GUI (``run_gui``). Coordinates are shown in mm."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(title)
    verts = np.asarray(model.vertices, dtype=float) * MM
    tris = np.asarray(model.triangles, dtype=np.int64)
    node_tags = np.arange(1, len(verts) + 1, dtype=np.uint64)
    next_elem = 1
    first = True
    for g in _mesh_groups(model):
        if 'indices' in g:
            sel = tris[g['indices']]
        else:
            n0, n1 = g['tri_range']
            sel = tris[n0:n1]
        if len(sel) == 0:
            continue
        tag = gmsh.model.addDiscreteEntity(2)
        if first:                                   # nodes are global; attach them once
            gmsh.model.mesh.addNodes(2, tag, node_tags, verts.ravel())
            first = False
        etags = np.arange(next_elem, next_elem + len(sel), dtype=np.uint64)
        next_elem += len(sel)
        gmsh.model.mesh.addElementsByType(tag, 2, etags, (sel + 1).astype(np.uint64).ravel())
        pg = gmsh.model.addPhysicalGroup(2, [tag])
        pot = g.get('potential')
        gmsh.model.setPhysicalName(2, pg, f"{g['name']}" + (f" ({pot:+.0f} V)" if pot is not None else ''))
    gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
    gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
    gmsh.option.setNumber("Mesh.ColorCarousel", 2)      # colour by physical group
    gmsh.option.setNumber("Mesh.Light", 1)
    gmsh.option.setNumber("General.Axes", 1)
    if write:
        gmsh.write(str(write))
    if run_gui:
        gmsh.fltk.run()
    gmsh.finalize()


def solid_meshes(model: ElectrodeModel):
    """{name: trimesh.Trimesh} of the closed solids of a 3D electrode model."""
    import trimesh
    out = {}
    for s in model.params['solids']:
        n0, n1 = s['tri_range']
        out[s['name']] = trimesh.Trimesh(vertices=model.vertices, faces=model.triangles[n0:n1],
                                         process=False)
    return out


def metal_mask(model: ElectrodeModel, pts: np.ndarray) -> np.ndarray:
    """True where (M, 3) points [m] lie inside any metal solid of a 3D model."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    inside = np.zeros(len(pts), dtype=bool)
    for name, m in solid_meshes(model).items():
        try:
            inside |= m.contains(pts)
        except Exception as exc:
            warnings.warn(f"metal_mask: contains() failed for {name} ({exc})", stacklevel=2)
    return inside


SECTION_Z_TOL_M = 1e-12      # a vertex this close to the plane of a section lies in it [m] (see loops_from_mesh)


def _reduced(loop: np.ndarray) -> np.ndarray:
    """The closed outline ``loop`` (K, 2) without a point twice in a row and without runs that go out to a point and
    straight back (what a solid touching the plane along edges leaves), also where the outline closes."""
    out = []
    for p in map(tuple, loop):
        if out and out[-1] == p:
            continue
        if len(out) >= 2 and out[-2] == p:          # ..., a, b, a: b is the tip of a run out and back
            out.pop()
            continue
        out.append(p)
    while len(out) >= 3:
        if out[0] == out[-1]:
            out.pop()
        elif out[-2] == out[0]:                     # the tip is the last point
            del out[-2:]
        elif out[1] == out[-1]:                     # the tip is the first point
            del out[0]
            out.pop()
        else:
            break
    return np.array(out, dtype=float).reshape(-1, 2)


def loops_from_mesh(mesh, z: float = 0.0, name: str = 'solid') -> List[np.ndarray]:
    """Closed outlines (K, 2) [m] of one closed ``trimesh.Trimesh`` in the
    plane at height ``z``.

    The cut is made edge by edge. A vertex within ``SECTION_Z_TOL_M`` of
    ``z`` lies in the plane: a face's vertices carry rounding of either sign
    at its nominal height (-13 * 1e-3 is not -0.013), up to ~1e-17 m on the
    worked example, so an exact test would cut a face at its own height into
    a patchwork. A vertex above the plane and out of it is above; every
    other one is below, so the section is the one just above ``z``: a face
    lying in the plane belongs to the solid on its upper side. Each mesh edge
    with a vertex on either side gives one point (a vertex in the plane is
    the point of all its crossed edges), each triangle it bounds one segment,
    and the segments are chained through the edges they share (vertices at
    one position count as one). A closed mesh whose every edge bounds two
    triangles therefore always gives closed outlines, whatever lies in or
    near the plane. Runs out to a point and straight back are removed, so a
    solid that only touches the plane from above, at a vertex or along
    edges, gives no outline there. Chaining the segment ends on a merge grid
    instead, as until 2026-11-02, fused crossings that were not neighbours
    where a vertex lay 1.7e-8 m off the plane on a nearly level face, and
    dropped the ground's whole section there. Chains left open by a mesh that
    is not closed (or by an edge of more than two triangles) are kept when
    the gap is below 1 mm, otherwise dropped with a warning."""
    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=np.int64)
    loops = []
    if not len(f):
        return loops
    v, inv = np.unique(v, axis=0, return_inverse=True)
    f = np.asarray(inv).reshape(-1)[f]
    in_plane = np.abs(v[:, 2] - float(z)) <= SECTION_Z_TOL_M
    above = (v[:, 2] > float(z)) & ~in_plane
    a = above[f]
    cut = a.any(axis=1) & ~a.all(axis=1)
    f, a = f[cut], a[cut]
    if not len(f):
        return loops
    # the two edges of every cut triangle that join a vertex below to one above, keyed by their vertex pair
    edges = np.stack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1)
    crossed = np.stack([a[:, 0] != a[:, 1], a[:, 1] != a[:, 2], a[:, 2] != a[:, 0]], axis=1)
    pairs = np.sort(edges[crossed].reshape(-1, 2, 2), axis=2)
    keys, ids = np.unique(pairs[:, :, 0] * len(v) + pairs[:, :, 1], return_inverse=True)
    ids = np.asarray(ids).reshape(-1, 2)
    i, j = keys // len(v), keys % len(v)
    lo, up = np.where(above[i], j, i), np.where(above[i], i, j)
    t = (float(z) - v[lo, 2]) / (v[up, 2] - v[lo, 2])
    t[in_plane[lo]] = 0.0
    pts = v[lo, :2] + t[:, None] * (v[up, :2] - v[lo, :2])
    adj: Dict[int, List[int]] = {}
    for a, b in ids:
        if a == b:
            continue
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    seen = set()
    for start in adj:
        if start in seen:
            continue
        # walk to one end of the chain (or around the loop)
        chain = [start]
        seen.add(start)
        for direction in (0, 1):
            cur, prev = start, None
            while True:
                nxt = [k for k in adj[cur] if k != prev and k not in seen]
                if not nxt:
                    break
                cur, prev = nxt[0], cur
                seen.add(cur)
                chain.append(cur) if direction == 0 else chain.insert(0, cur)
        loop = _reduced(pts[chain])
        if len(loop) < 3:
            continue
        # a closed loop is walked entirely in the first direction and ends
        # next to its start; anything else is an open chain
        closed = chain[0] == start and start in adj.get(chain[-1], [])
        gap = float(np.hypot(*(loop[0] - loop[-1])))
        if not closed and gap > 1e-3:
            warnings.warn(f"loops_from_mesh: open outline of {name} at z = {z * 1e3:.1f} mm "
                          f"(gap {gap * 1e3:.2f} mm) dropped", stacklevel=2)
            continue
        loops.append(loop)
    return loops


def section_loops(model: ElectrodeModel, z: float = 0.0) -> List[np.ndarray]:
    """Closed outlines (K, 2) [m] of the metal of a 3D model in the plane at
    height ``z``: plane sections of every solid (see ``loops_from_mesh``)."""
    loops = []
    for name, m in solid_meshes(model).items():
        loops.extend(loops_from_mesh(m, z=z, name=name))
    return loops


def raster_obstacles(outlines: Sequence[np.ndarray], spacing: float = 5e-4,
                     extent: float = 0.3, beam_halfwidth: float = 0.0,
                     verbose: bool = False) -> Callable[[np.ndarray], np.ndarray]:
    """Even-odd raster of closed outlines (K, 2) [m] on a square grid of
    ``spacing`` over |x|, |y| <= ``extent`` -> ``inside(xy) -> bool``.

    Nested outlines are holes (even-odd rule). ``beam_halfwidth`` dilates the
    metal so the test reads 'the bunch envelope touches metal'. Points off the
    grid are vacuum; the raster is exact to half a cell. The callable carries
    ``grid``, ``xs`` and ``spacing`` for plotting. Feed it to
    ``tracking.MetalTerminator`` / ``AcceleratedOrbitFinder.obstacle_mask``;
    ``combine_obstacles`` ORs several."""
    from matplotlib.path import Path as MplPath
    ext = float(extent)
    n = int(np.ceil(2.0 * ext / spacing)) + 1
    xs = np.linspace(-ext, ext, n)
    h = float(xs[1] - xs[0])
    t0 = time.time()
    grid = np.zeros((n, n), dtype=bool)
    for loop in outlines:
        loop = np.asarray(loop, dtype=float)[:, :2]
        # bounding-box window of grid nodes, even-odd accumulation
        i0, i1 = np.searchsorted(xs, loop[:, 0].min() - h), np.searchsorted(xs, loop[:, 0].max() + h)
        j0, j1 = np.searchsorted(xs, loop[:, 1].min() - h), np.searchsorted(xs, loop[:, 1].max() + h)
        i0, j0 = max(i0 - 1, 0), max(j0 - 1, 0)
        if i1 <= i0 or j1 <= j0:
            continue
        gx, gy = np.meshgrid(xs[i0:i1], xs[j0:j1], indexing='ij')
        inside = MplPath(loop).contains_points(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        grid[i0:i1, j0:j1] ^= inside
    if beam_halfwidth > 0:
        from scipy.ndimage import binary_dilation
        k = int(np.ceil(beam_halfwidth / h))
        yy, xx = np.ogrid[-k:k + 1, -k:k + 1]
        grid = binary_dilation(grid, structure=(xx ** 2 + yy ** 2) <= k ** 2)
    if verbose:
        print(f"[electrodes3d] obstacle grid {n}x{n} at {h * 1e3:.2f} mm "
              f"({100 * grid.mean():.1f}% metal, {len(outlines)} outlines) in {time.time() - t0:.1f} s")

    def inside(xy):
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        i = np.rint((xy[:, 0] + ext) / h).astype(int)
        j = np.rint((xy[:, 1] + ext) / h).astype(int)
        ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
        out = np.zeros(len(xy), dtype=bool)
        out[ok] = grid[i[ok], j[ok]]
        return out

    inside.grid, inside.xs, inside.spacing = grid, xs, h
    return inside


def combine_obstacles(*tests: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """OR of several ``inside(xy)`` tests (None entries ignored)."""
    tests = [t for t in tests if t is not None]
    if not tests:
        return None
    if len(tests) == 1:
        return tests[0]

    def inside(xy):
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        out = np.zeros(len(xy), dtype=bool)
        for t in tests:
            out |= np.asarray(t(xy), dtype=bool)
        return out

    inside.parts = tests
    # a 3D part (ndim = 3) needs z: tracking.MetalTerminator reads this
    inside.ndim = max(int(getattr(t, 'ndim', 2)) for t in tests)

    # If any part can answer the stronger question - did this STEP cross a surface - expose it, so
    # MetalTerminator uses it. The parts that cannot (the RF cavity map, which is a raster with no
    # surfaces) still contribute their point test on the step's end, so the combination is never
    # weaker than the point-only version.
    seg_parts = [t for t in tests if getattr(t, 'segment', None) is not None]
    if seg_parts:
        pt_parts = [t for t in tests if getattr(t, 'segment', None) is None]

        def segment(r_prev, r):
            r = np.atleast_2d(np.asarray(r, dtype=float))
            hit = np.zeros(len(r), dtype=bool)
            where = np.array(r[:, :3], dtype=float)
            for t in seg_parts:
                h, w = t.segment(r_prev, r)
                h = np.asarray(h, dtype=bool)
                fresh = h & ~hit
                where[fresh] = np.asarray(w, dtype=float)[fresh]
                hit |= h
            for t in pt_parts:
                cols = 3 if int(getattr(t, 'ndim', 2)) == 3 else 2
                hit |= np.asarray(t(r[:, :cols]), dtype=bool)
            return hit, where

        inside.segment = segment
    return inside


def midplane_obstacles(model: ElectrodeModel, spacing: float = 5e-4, z: float = 0.0,
                       beam_halfwidth: float = 0.0, extent: Optional[float] = None,
                       verbose: bool = False) -> Callable[[np.ndarray], np.ndarray]:
    """Fast midplane metal test ``inside(xy) -> bool`` for tracking.

    3D models (this module): the plane sections of the solids at height ``z``
    (``section_loops``) rasterised with the even-odd rule (holes such as the
    hollow scroll come out right). 2D tall-wall models: only the scroll /
    central post / housing wedges (and their holes) are obstacles - the dee
    and hill footprints are potential regions the beam flies through between
    the plates. See ``raster_obstacles`` for the grid semantics
    (``spacing`` well below the thinnest wall, ``beam_halfwidth`` dilation)."""
    ext = float(extent if extent is not None else model.extent())
    if model.params.get('dim') == 3 and 'solids' in model.params:
        # even-odd per solid (holes), OR across solids: separate grounded
        # bodies (an un-fused housing) may overlap the fused ground
        tests = []
        for name, m in solid_meshes(model).items():
            loops = loops_from_mesh(m, z=z, name=name)
            if loops:
                tests.append(raster_obstacles(loops, spacing=spacing, extent=ext,
                                              beam_halfwidth=beam_halfwidth, verbose=verbose))
        if len(tests) > 1:
            grid = np.logical_or.reduce([t.grid for t in tests])
            xs, h, n = tests[0].xs, tests[0].spacing, len(tests[0].xs)

            def inside(xy):
                xy = np.atleast_2d(np.asarray(xy, dtype=float))
                i = np.rint((xy[:, 0] + ext) / h).astype(int)
                j = np.rint((xy[:, 1] + ext) / h).astype(int)
                ok = (i >= 0) & (i < n) & (j >= 0) & (j < n)
                out = np.zeros(len(xy), dtype=bool)
                out[ok] = grid[i[ok], j[ok]]
                return out

            inside.grid, inside.xs, inside.spacing = grid, xs, h
            return inside
        outlines = [] if not tests else None
        if tests:
            return tests[0]
    elif model.params.get('post_mode') == 'housing':
        # the union hub also holds the hill spokes (flown through between the
        # plates): only the housing outline is an obstacle
        outlines = [np.asarray(l, dtype=float) for l in model.params['housing']['loops']]
    else:
        outlines = []
        for w in model.wedges:
            if w.kind == 'post':
                outlines.append(np.asarray(w.polygon, dtype=float))
                outlines.extend(np.asarray(h, dtype=float) for h in getattr(w, 'holes', ()))
    return raster_obstacles(outlines, spacing=spacing, extent=ext,
                            beam_halfwidth=beam_halfwidth, verbose=verbose)


def stacked_obstacles(model: ElectrodeModel, z_levels, spacing: float = 5e-4, extent: Optional[float] = None,
                      beam_halfwidth: float = 0.0, verbose: bool = False) -> Callable[[np.ndarray], np.ndarray]:
    """3D metal test ``inside(xyz) -> bool`` for tracking: the plane sections
    of every solid at each of ``z_levels`` [m] (sorted, evenly spaced)
    rasterised as in ``midplane_obstacles`` and stacked; a point is tested on
    the nearest plane, so the test is exact to half a cell in x-y and half a
    level spacing in z. Beyond the first / last level (and off the grid) the
    space is vacuum - bound it with the tracker's vertical limit. (N, 2)
    input is taken at z = 0. The dee plates, hill faces, bars, patches and an
    un-fused housing solid all become real 3D apertures. The callable
    carries ``grid`` (nx, ny, nz), ``xs``, ``zs``, ``spacing`` and
    ``ndim = 3`` (read by ``tracking.MetalTerminator``). The tracker's
    default, ``metal_columns.ColumnObstacles``, is the same test exact in z."""
    if not (model.params.get('dim') == 3 and 'solids' in model.params):
        raise ValueError("stacked_obstacles needs a 3D electrode model (build_electrodes_3d)")
    zs = np.sort(np.asarray(z_levels, dtype=float).ravel())
    if len(zs) < 2 or np.ptp(np.diff(zs)) > 1e-9 * max(abs(zs).max(), 1e-3):
        raise ValueError("z_levels must be >= 2 evenly spaced heights")
    ext = float(extent if extent is not None else model.extent())
    meshes = solid_meshes(model)
    t0 = time.time()
    grid = None
    xs = h = None
    for k, z in enumerate(zs):
        plane = None
        for name, m in meshes.items():
            loops = loops_from_mesh(m, z=float(z), name=name)
            if not loops:
                continue
            t = raster_obstacles(loops, spacing=spacing, extent=ext, beam_halfwidth=beam_halfwidth)
            plane = t.grid if plane is None else (plane | t.grid)
            xs, h = t.xs, t.spacing
        if grid is None:
            if xs is None:
                n = int(np.ceil(2.0 * ext / spacing)) + 1
                xs = np.linspace(-ext, ext, n)
                h = float(xs[1] - xs[0])
            grid = np.zeros((len(xs), len(xs), len(zs)), dtype=bool)
        if plane is not None:
            grid[:, :, k] = plane
    n = len(xs)
    dz = float(zs[1] - zs[0])
    z0 = float(zs[0])
    if verbose:
        print(f"[electrodes3d] stacked obstacles {n}x{n}x{len(zs)} at {h * 1e3:.2f} mm x {dz * 1e3:.1f} mm, "
              f"z {zs[0] * 1e3:.0f}..{zs[-1] * 1e3:.0f} mm ({100 * grid.mean():.1f}% metal) in {time.time() - t0:.0f} s")

    def inside(pts):
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        i = np.rint((p[:, 0] + ext) / h).astype(int)
        j = np.rint((p[:, 1] + ext) / h).astype(int)
        if p.shape[1] >= 3:
            k = np.rint((p[:, 2] - z0) / dz).astype(int)
        else:
            k = np.full(len(p), int(round(-z0 / dz)))
        ok = (i >= 0) & (i < n) & (j >= 0) & (j < n) & (k >= 0) & (k < len(zs))
        out = np.zeros(len(p), dtype=bool)
        out[ok] = grid[i[ok], j[ok], k[ok]]
        return out

    inside.grid, inside.xs, inside.zs, inside.spacing, inside.ndim = grid, xs, zs, h, 3
    return inside


_mesh_inside_kernel = None
_mesh_segment_kernel = None


def _define_mesh_kernels():
    """Warp kernels, built once on first use.

    Not inside a method: a kernel created per call is re-registered with Warp every time and
    leaks (py_electrodes measured ~100 KB a call, 2.6 GB over a 1,500-particle run)."""
    global _mesh_inside_kernel, _mesh_segment_kernel
    if _mesh_inside_kernel is not None:
        return
    import warp as wp

    @wp.kernel
    def _inside(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.uint8)):
        i = wp.tid()
        sign = float(0.0)
        q = wp.mesh_query_point_sign_winding_number(mesh, pts[i], 1.0e6, sign)
        if q.result and q.sign < 0.0:
            out[i] = wp.uint8(1)

    @wp.kernel
    def _segment(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3f),
                 directions: wp.array(dtype=wp.vec3f), lengths: wp.array(dtype=wp.float32),
                 hits: wp.array(dtype=wp.uint8), points: wp.array(dtype=wp.vec3f)):
        i = wp.tid()
        max_d = lengths[i]
        if max_d <= 0.0:
            return
        q = wp.mesh_query_ray(mesh, origins[i], directions[i], max_d)
        if q.result:
            hits[i] = wp.uint8(1)
            points[i] = origins[i] + directions[i] * q.t

    _mesh_inside_kernel, _mesh_segment_kernel = _inside, _segment


class MeshObstacles:
    """Exact metal test straight off the triangle mesh, on the GPU, through a Warp BVH.

    A drop-in for ``stacked_obstacles``: called with (N, 3) points it returns the point-in-metal
    mask and carries ``ndim = 3``, so ``tracking.MetalTerminator`` feeds it all three columns.
    What it adds is ``segment(r_prev, r)`` - did this step cross a surface, and where.

    Why, given the raster is already fast:
      - it is EXACT. The raster decides metal by even-odd point-in-polygon on z slices, so it is
        good only to half a cell in x-y and half a level in z, and it goes wrong outright where a
        metal face coincides with a sampled plane - the degeneracy that emptied a 15-turn run.
      - ``segment`` catches a particle that steps THROUGH a thin wall between two samples, which
        testing the positions at either end misses completely.
      - measured on the real problem (40,000 particles, 37,280 triangles, 0.155 mm step):
        0.062 ms/step against 0.70 ms for the raster lookup, and no 39 s raster build. The
        brute-force kernel the spiral inflector uses - every ray against every triangle - takes
        13.4 ms/step, which is where the "won't this be slower?" instinct comes from; the BVH is
        216x faster than that and agrees with it exactly.

    The catch, and why ``segment`` is not the whole story: a crossing test cannot see a particle
    that is ALREADY inside metal and never crosses a surface. Birth positions therefore still go
    through the point test - which is what ``__call__`` is for.

    ``metal_columns.ColumnObstacles`` (the tracker's default) is exact in z too, on the CPU and in
    the compiled kernel, but has no ``segment``.
    """

    ndim = 3

    def __init__(self, model: ElectrodeModel, device: str = 'cuda:0'):
        import warp as wp
        wp.init()
        _define_mesh_kernels()
        self._wp = wp
        self.device = device
        v = np.ascontiguousarray(np.asarray(model.vertices, dtype=np.float32))
        f = np.ascontiguousarray(np.asarray(model.triangles, dtype=np.int32).ravel())
        self._points = wp.from_numpy(v, dtype=wp.vec3f, device=device)
        self._indices = wp.from_numpy(f, dtype=wp.int32, device=device)
        self.mesh = wp.Mesh(points=self._points, indices=self._indices)
        self.n_tri = len(f) // 3
        self.n_inside_calls = 0
        self.n_segment_calls = 0
        self._buf = {}

    def _slot(self, n, name, dtype):
        """Device buffers reused across steps - a fresh allocation every step would dominate."""
        key = (name, n)
        if key not in self._buf:
            self._buf[key] = self._wp.zeros(n, dtype=dtype, device=self.device)
        return self._buf[key]

    def __call__(self, pts) -> np.ndarray:
        """Point-in-metal for (N, 2 or 3) points; (N, 2) is taken at z = 0, as the raster does."""
        p = np.atleast_2d(np.asarray(pts, dtype=np.float32))
        if p.shape[1] == 2:
            p = np.column_stack([p, np.zeros(len(p), dtype=np.float32)])
        n = len(p)
        if n == 0:
            return np.zeros(0, dtype=bool)
        wp = self._wp
        d_p = wp.from_numpy(np.ascontiguousarray(p[:, :3]), dtype=wp.vec3f, device=self.device)
        d_out = self._slot(n, 'inside', wp.uint8)
        d_out.zero_()
        wp.launch(_mesh_inside_kernel, dim=n, device=self.device,
                  inputs=[self.mesh.id, d_p, d_out])
        wp.synchronize()
        self.n_inside_calls += 1
        return d_out.numpy().astype(bool)

    def segment(self, r_prev, r):
        """Did the step from r_prev to r cross a surface? Returns (hit_mask, hit_points [m])."""
        a = np.ascontiguousarray(np.asarray(r_prev, dtype=np.float32)[:, :3])
        b = np.ascontiguousarray(np.asarray(r, dtype=np.float32)[:, :3])
        n = len(a)
        if n == 0:
            return np.zeros(0, dtype=bool), np.zeros((0, 3))
        d = b - a
        length = np.linalg.norm(d, axis=1).astype(np.float32)
        good = length > 0.0
        dirs = np.zeros_like(d)
        dirs[good] = d[good] / length[good, None]
        wp = self._wp
        # Reused device buffers, filled in place. Allocating three arrays per step (30,000 of
        # them a run) costs more than the query itself.
        d_o, d_d, d_l = (self._slot(n, 'org', wp.vec3f), self._slot(n, 'dir', wp.vec3f),
                         self._slot(n, 'len', wp.float32))
        d_o.assign(a)
        d_d.assign(np.ascontiguousarray(dirs))
        d_l.assign(np.ascontiguousarray(length))
        d_hit = self._slot(n, 'hit', wp.uint8)
        d_pt = self._slot(n, 'pt', wp.vec3f)
        d_hit.zero_()
        wp.launch(_mesh_segment_kernel, dim=n, device=self.device,
                  inputs=[self.mesh.id, d_o, d_d, d_l, d_hit, d_pt])
        wp.synchronize()
        self.n_segment_calls += 1
        hit = d_hit.numpy().astype(bool)
        # The hit POINTS are 12 bytes a particle and are wanted only for the few that actually hit
        # - which on most steps is none at all. Fetching them unconditionally moved ~480 kB per
        # step, ~14 GB over a run, and made the exact test slower end to end than the raster it
        # replaces even though its kernel is ten times cheaper.
        if not hit.any():
            return hit, np.zeros((n, 3))
        return hit, d_pt.numpy().astype(float)


def check_trajectory_clearance(model_or_inside, trajectory, beam_halfwidth: float = 0.0,
                               z: float = 0.0, raise_on_contact: bool = True,
                               label: str = 'trajectory', spacing: float = 1e-3) -> dict:
    """Probe a trajectory - and, with ``beam_halfwidth``, its +- envelope normal
    to the path - against the midplane metal of a model (or an ``inside`` test
    from ``midplane_obstacles``).

    Returns dict(n_contact, n_points, first_index, first_point_mm, radius_mm).
    With ``raise_on_contact`` any probe inside metal raises RuntimeError,
    otherwise a warning: a centroid riding inside a wall sees E = 0 there and
    reports a perfect energy gain, which is exactly the silent failure this
    check is for. Run it on the reference trajectory before a 3D solve and on
    the re-tracked trajectory after it.
    """
    inside = (model_or_inside if callable(model_or_inside)
              else midplane_obstacles(model_or_inside, spacing=spacing, z=z))
    xy = np.asarray(trajectory, dtype=float)[:, :2]
    probes = [xy]
    if beam_halfwidth > 0 and len(xy) > 1:
        t = np.gradient(xy, axis=0)
        t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-12)[:, None]
        nrm = np.column_stack([t[:, 1], -t[:, 0]])
        probes += [xy + beam_halfwidth * nrm, xy - beam_halfwidth * nrm]
    hit = np.zeros(len(xy), dtype=bool)
    for p in probes:
        hit |= np.asarray(inside(p), dtype=bool)
    res = {'n_contact': int(hit.sum()), 'n_points': int(len(xy)), 'first_index': None,
           'first_point_mm': None, 'radius_mm': None}
    if hit.any():
        k = int(np.argmax(hit))
        res.update(first_index=k, first_point_mm=[round(float(v) * 1e3, 2) for v in xy[k]],
                   radius_mm=round(float(np.hypot(*xy[k])) * 1e3, 2))
        msg = (f"{label}: {res['n_contact']} of {len(xy)} points (envelope +-{beam_halfwidth * 1e3:.1f} mm) "
               f"inside metal, first at step {k}: ({xy[k, 0] * 1e3:.1f}, {xy[k, 1] * 1e3:.1f}) mm, "
               f"r = {res['radius_mm']} mm")
        if raise_on_contact:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)
    return res


def plot_sections(model: ElectrodeModel, planes, ax=None, lim=None, colors=None):
    """Draw section outlines of the 3D solids [mm]. ``planes`` = list of
    (normal (3,), origin (3,) [m], abscissa index, ordinate index, title)."""
    import matplotlib.pyplot as plt
    import trimesh
    meshes = solid_meshes(model)
    if ax is None:
        _, ax = plt.subplots(1, len(planes), figsize=(6 * len(planes), 6))
    axes = np.atleast_1d(ax)
    palette = colors or {}
    for axi, (normal, origin, ix, iy, title) in zip(axes, planes):
        for name, m in meshes.items():
            seg = trimesh.intersections.mesh_plane(m, np.asarray(normal, float), np.asarray(origin, float))
            c = palette.get(name, 'tab:blue' if name != 'ground' else 'tab:red')
            for s in np.asarray(seg):
                axi.plot(s[:, ix] * MM, s[:, iy] * MM, '-', color=c, lw=0.8)
        if lim is not None:
            axi.set_xlim(lim[0]); axi.set_ylim(lim[1])
        axi.set_aspect('equal'); axi.grid(True, lw=0.3); axi.set_title(title, fontsize=9)
    return axes
