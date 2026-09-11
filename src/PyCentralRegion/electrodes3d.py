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
    not hill or scroll. All ground pieces are fused into one solid so no
    interior faces reach the mesh.
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
    measured into the wedge. Dee bars are clipped to the dee height and
    fused into the dee, ground bars are fused into the ground.
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
    """
    path: str
    potential: float = 0.0
    scale: float = 1.0
    rotation_deg: float = 0.0
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = ''
    mirror_z: bool = False


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


def _bar_footprint(wedge: Wedge, bar: Bar) -> np.ndarray:
    """Closed strip polygon (m) of a bar along a wedge chain interval or its inner edge."""
    if bar.side == 'tip':
        seg = _inner_edge(wedge)
        off = offset_polyline(seg, +bar.width)          # left of travel = into the wedge
        return np.vstack([seg, off[::-1]])
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
    return np.vstack([seg, off[::-1]])


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
              beam_halfwidth: float = 0.0) -> List[Bar]:
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
    Use ``bar_clearances`` / ``check_trajectory_clearance`` to verify.
    """
    clearance = margin + beam_halfwidth
    ds_chain = 5e-4
    n_f = max(3, int(np.ceil(width / 1e-3)) + 1)          # samples across the strip
    # the strip is sampled at ds_chain along the chain and width/(n_f-1) across
    # it; add half of the coarser spacing so the bar footprints (checked with
    # bar_clearances) never fall short of the requested clearance
    clearance_eff = clearance + 0.5 * max(ds_chain, width / (n_f - 1))
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
                free = (d_min >= clearance_eff) & (r <= r_max)
                for a, b in _runs(free):
                    r_from, r_to = float(r[a:b + 1].min()), float(r[a:b + 1].max())
                    if r_to - r_from >= 2.0 * width:
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
                        scroll_thickness: Optional[float] = None,
                        size_min: float = 0.002,
                        size_max: float = 0.012,
                        dist_min: float = 0.004,
                        dist_max: float = 0.050,
                        r_fine_ground: Optional[float] = None,
                        edge_fillet: Optional[float] = None,
                        heal_tolerance: Optional[float] = None,
                        voltage_profile=None,
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

    # ---- 1. full-height prisms of every footprint --------------------------------
    prisms: Dict[str, List[int]] = {}
    known_wedges = {w.label for w in dee_wedges + hill_wedges}
    for b in bars:
        if b.wedge not in known_wedges:
            raise ValueError(f"Bar on unknown wedge {b.wedge!r}; wedges: {sorted(known_wedges)}")
    for w in dee_wedges + hill_wedges:
        prisms[w.label] = [_prism(occ, w.polygon, -zp, zp)]
        for j, bar in enumerate([b for b in bars if b.wedge == w.label]):
            prisms[f"{w.label}#bar{j}"] = [_prism(occ, _bar_footprint(w, bar), -zp, zp)]
    uncropped: Dict[str, List[int]] = {}
    hub_keys: List[str] = []
    housing_mode = model2d.params.get('post_mode') == 'housing'
    if housing_mode:
        # the 2D hub is the housing OUTLINE fused with the dummy-dee spokes: a
        # tall-wall stand-in. In 3D the spokes are extruded and the housing
        # itself should come from its STEP file (a grounded ExtraSolid);
        # without one the outline is extruded, with a warning.
        for j, (lbl, ring) in enumerate(model2d.params.get('spoke_polygons', [])):
            key = f"spoke#{j}:{lbl}"
            prisms[key] = [_prism(occ, np.asarray(ring), -zp, zp)]
            hub_keys.append(key)
        if not any(float(ex.potential) == 0.0 for ex in extra_solids):
            warnings.warn("housing-mode 2D model without a grounded ExtraSolid: the housing "
                          "outline is extruded as a tall-wall prism (pass the housing STEP "
                          "as ExtraSolid(potential=0) for the real shape)", stacklevel=2)
            for w in post_wedges:
                body = _prism(occ, np.asarray(w.polygon), -zp, zp)
                for h in w.holes:
                    out, _ = occ.cut([(3, body)], [(3, _prism(occ, np.asarray(h), -zp - 1.0, zp + 1.0))],
                                     removeObject=True, removeTool=True)
                    body = [t for d, t in out if d == 3][0]
                prisms[w.label] = [body]
                hub_keys.append(w.label)
    for w in ([] if housing_mode else post_wedges):
        hub_keys.append(w.label)
        if scroll_thickness:
            # OCC's intersect drops this toroidal solid against the crop
            # cylinder; it lies inside the crop anyway, so build it at the
            # exact crop height and skip the crop
            uncropped[w.label] = [_prism_with_hole(occ, np.asarray(w.polygon),
                                                   _offset_ring(np.asarray(w.polygon), scroll_thickness),
                                                   -zc, zc)]
        else:
            prisms[w.label] = [_prism(occ, w.polygon, -zp, zp)]
    for j, p in enumerate(posts):
        prisms[f"post#{j}"] = [occ.addCylinder(p.x * MM, p.y * MM, -zp, 0, 0, 2 * zp, p.radius * MM)]
    extra_keys: List[Tuple[str, float]] = []
    for j, ex in enumerate(extra_solids):
        name = ex.name or f"extra#{j}"
        prisms[name] = _import_step(occ, gmsh, ex)
        extra_keys.append((name, float(ex.potential)))

    # ---- 2. crop everything (one boolean) -----------------------------------------
    crop = occ.addCylinder(0, 0, -zc, 0, 0, 2 * zc, r_cut * MM)
    solids = _boolean(occ, occ.intersect, prisms, crop)
    for k, vols in solids.items():
        if not vols:
            raise RuntimeError(f"{k}: nothing left after the crop (r_cut {r_cut} m, z_cut {z_cut} m)")
    solids.update(uncropped)

    # ---- 3. dees: outer height, aperture, fillet, bars ----------------------------
    dee_keys = [w.label for w in dee_wedges]
    bar_keys = [k for k in solids if '#bar' in k and k.split('#')[0] in dee_keys]
    ground_bar_keys = [k for k in solids if '#bar' in k and k.split('#')[0] not in dee_keys]
    if H_d is not None:
        clipped = _boolean(occ, occ.intersect, {k: solids[k] for k in dee_keys + bar_keys}, _revolved(occ, H_d, r_tool))
        solids.update(clipped)
    if h_d is not None:
        hollowed = _boolean(occ, occ.cut, {k: solids[k] for k in dee_keys}, _revolved(occ, h_d, r_tool))
        solids.update(hollowed)
    for k in dee_keys:
        if not solids[k]:
            raise RuntimeError(f"{k}: no metal left after the height / aperture booleans")
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
    for k in bar_keys:
        dee = k.split('#')[0]
        out, _ = occ.fuse([(3, b) for b in solids[dee]], [(3, b) for b in solids[k]],
                          removeObject=True, removeTool=True)
        solids[dee] = [t for d, t in out if d == 3]
        del solids[k]

    # ---- 4. ground: hill gap, roof, fuse ----------------------------------------
    hill_keys = [w.label for w in hill_wedges]
    scroll_keys = list(hub_keys)
    post_keys = [k for k in solids if k.startswith('post#')]
    roof_parts: List[int] = []
    if profiles.valley_height is not None:
        zv = 0.5 * profiles.valley_height * MM
        blockers = [occ.copy([(3, t)])[0][1] for k in hill_keys + scroll_keys for t in solids[k]]
        for z0, z1 in ((zv, zc), (-zc, -zv)):
            slab = occ.addCylinder(0, 0, z0, 0, 0, z1 - z0, r_cut * MM)
            tools = [(3, occ.copy([(3, b)])[0][1]) for b in blockers]
            out, _ = occ.cut([(3, slab)], tools, removeObject=True, removeTool=True)
            roof_parts.extend(t for d, t in out if d == 3)
        occ.remove([(3, b) for b in blockers], recursive=True)
    if h_g is not None:
        gapped = _boolean(occ, occ.cut, {k: solids[k] for k in hill_keys}, _revolved(occ, h_g, r_tool))
        solids.update(gapped)
    grounded_extra = [k for k, pot in extra_keys if pot == 0.0]
    ground_parts = ([t for k in hill_keys + ground_bar_keys + scroll_keys + post_keys + grounded_extra
                     for t in solids[k]] + roof_parts)
    if verbose:
        for k in scroll_keys + post_keys + grounded_extra + ground_bar_keys:
            print(f"[electrodes3d]   {k:14s} {sum(occ.getMass(3, t) for t in solids[k]) / MM**3 * 1e6:9.1f} cm3 "
                  f"in {len(solids[k])} piece(s) @ 0 V (ground part)")
    occ.synchronize()
    known = {t for d, t in gmsh.model.getEntities(3)}
    owners = {t: k for k in hill_keys + scroll_keys + post_keys + grounded_extra for t in solids[k]}
    owners.update({t: 'roof' for t in roof_parts})
    missing = [(t, owners[t]) for t in ground_parts if t not in known]
    if missing:
        raise RuntimeError(f"ground parts vanished before the fuse: {missing}; "
                           f"solids = { {k: v for k, v in solids.items()} }")
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
    else:
        ground_solids = list(ground_parts)
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
    occ.synchronize()
    if verbose:
        print(f"[electrodes3d] solids: {len(dee_solids)} dee, {len(ground_solids)} ground "
              f"(fused from {len(ground_parts)} parts) in {time.time() - t_start:.1f} s")
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
    tag = gmsh.model.addPhysicalGroup(2, gnd_faces, tag=1)
    groups.append(('ground', 0.0, tag))

    # ---- 6. mesh + export -----------------------------------------------------------
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


def loops_from_mesh(mesh, z: float = 0.0, tol: float = 1e-6, name: str = 'solid') -> List[np.ndarray]:
    """Closed outlines (K, 2) [m] of one closed ``trimesh.Trimesh`` in the
    plane at height ``z``: plane-section segments chained by shared
    endpoints. Chains left open by a non-watertight mesh are kept when the
    gap is below 1 mm, otherwise dropped with a warning."""
    import trimesh
    loops = []
    seg = trimesh.intersections.mesh_plane(mesh, np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, float(z)]))
    if len(seg) == 0:
        return loops
    seg = np.asarray(seg)[:, :, :2]
    keys = np.round(seg.reshape(-1, 2) / tol).astype(np.int64)
    _, ids = np.unique(keys, axis=0, return_inverse=True)
    ids = ids.reshape(-1, 2)
    pts = np.zeros((ids.max() + 1, 2))
    pts[ids.ravel()] = seg.reshape(-1, 2)
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
                nxt = [v for v in adj[cur] if v != prev and v not in seen]
                if not nxt:
                    break
                cur, prev = nxt[0], cur
                seen.add(cur)
                chain.append(cur) if direction == 0 else chain.insert(0, cur)
        loop = pts[chain]
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


def section_loops(model: ElectrodeModel, z: float = 0.0, tol: float = 1e-6) -> List[np.ndarray]:
    """Closed outlines (K, 2) [m] of the metal of a 3D model in the plane at
    height ``z``: plane sections of every solid (see ``loops_from_mesh``)."""
    loops = []
    for name, m in solid_meshes(model).items():
        loops.extend(loops_from_mesh(m, z=z, tol=tol, name=name))
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
        outlines = section_loops(model, z=z)
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
