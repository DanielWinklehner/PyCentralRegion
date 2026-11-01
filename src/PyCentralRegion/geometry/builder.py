"""3D electrodes of the central region: dees, dummy dees, housing and hardware as OCC solids, meshed for the BEM.

Inputs: the seed optimization's gap centre lines (a DeeSystem of RFCavity gaps, segment ends in METRES); the housing
outline the spiral-inflector export writes beside its STEP files (the closed polygon its wall is extruded from, the
corners of the exit opening and the curves joining the points, in MILLIMETRES); and the hand-off bunch, whose
plane-crossing cloud IS the plane the central region emits from. Everything here is in mm; the one conversion from
the gap lines' metres is MM.

Construction, in the order ``build.build`` runs it:
  1. gap lines: inert variable segments zeroed, node radii snapped between turns, junctions moved, gap 0 re-fitted
     onto the turned exit plate (``neutralise_inert_segments``, ``move_junctions``, ``refit_gap0_on_plate``);
  2. 2D boundaries: each gap centre line offset by half the LOCAL gap width toward its dee, inner ends held off the
     housing (dee0: the gap-0 / gap-1 vertex, optionally cut back or rounded), closed by an arc at r_max; the dummy
     dees from the other sides of the same lines (``dee_boundaries``, ``dummy_boundaries``);
  3. OCC solids: the housing extruded from its outline, the footprints extruded over +-H and hollowed by a revolved
     aperture profile, the ground either the dummy wedges fused onto the housing or ('star') ONE 2D layout - the
     r_max disk minus every dee's keep-out - extruded, the beam channel cut through it; posts, windows, holes and
     boxes fused into their owners (``build_model``);
  4. a surface mesh (uniform or radially graded, refined on hardware and on a round nose, slivers repaired by edge
     flips) and the ElectrodeModel the BEM solves (``mesh_and_report``, ``to_electrode_model``).

The options come from one ``configure(geometry, machine, inflector)`` call; every function reads them when it runs,
so the tools that rebuild the edges (post placement, the gap-0 hole, edge shifts) see the edges the build sees. The
builder models four radial dees (straight-segment gap lines, an entry / exit gap pair per dee, a dummy dee between
neighbours) and ``configure`` refuses other machines (``machine.check_supported``). No environment variable is read.
"""
import json
import time
from pathlib import Path

import numpy as np

from PyCentralRegion.config import ConfigError, InflectorConfig, MachineConfig
from PyCentralRegion.machine import apply_seed_geometry, check_supported
from PyCentralRegion.rf_cavity import DeeSystem, create_dee_system
from .options import GeometryConfig, channel_height_mm

MM = 1000.0                 # metres -> millimetres

OPT = GeometryConfig()      # the options of the current build: configure() sets them
MACHINE = MachineConfig()
INFLECTOR = InflectorConfig()
EXIT_SPEC_FILE = ''
_EXIT_SPEC = {}

STAR_R_EXT_MM = 20.0        # star keep-outs run this far past r_max, so no edge of theirs lies on the r_max circle
STAR_SEAM_MM = 0.5          # gap / tip bands reach this far INTO the dee: overlaps, never shared edges
GAP0_FACE_OVERLAP_MM = 2.0  # the gap-0 ground face reaches this far into the housing
PLATE_CORNER_SNAP_MM = 0.25
POST_WALL_SINK_MM = 0.5     # a post joined to an end wall sits this deep in it everywhere (>> the fuzzy tolerance)


def configure(geometry: GeometryConfig = None, machine: MachineConfig = None, inflector: InflectorConfig = None,
              exit_plane_spec_file: str = None) -> None:
    """Set the options every function of this module reads: [geometry], [machine] and [inflector] of the run config.

    ``exit_plane_spec_file`` overrides geometry.exit_plane_spec_file (``build`` resolves its default from the housing
    outline's folder). Refused, changing nothing: a machine the builder does not model (``machine.check_supported``)
    and a geometry.r_max_mm beyond the cavity radius the gap lines end at (the dees would stop short of the model's
    edge)."""
    global OPT, MACHINE, INFLECTOR, EXIT_SPEC_FILE
    geometry, machine = geometry or GeometryConfig(), machine or MachineConfig()
    check_supported(machine)
    if geometry.r_max_mm > float(machine.r_max_cavity_m) * MM + 1e-9:
        raise ConfigError(f"geometry.r_max_mm = {geometry.r_max_mm:g} mm lies beyond [machine] r_max_cavity_m = "
                          f"{float(machine.r_max_cavity_m) * MM:g} mm, where the gap lines end: the dees would stop "
                          f"short of the model's edge (0 = the cavity radius)")
    OPT, MACHINE, INFLECTOR = geometry, machine, inflector or InflectorConfig()
    EXIT_SPEC_FILE = str(OPT.exit_plane_spec_file if exit_plane_spec_file is None else exit_plane_spec_file)
    _EXIT_SPEC.clear()


def channel_heights(owner: str = 'dee0') -> tuple:
    """The channel's full heights (inside, outside the taper) [mm] of the electrode ``owner``: a dee's, or for a
    dummy dee and the ground the dummy dees' own (geometry.dummy_aperture_*_mm, the dees' when 0)."""
    return OPT.aperture_heights() if str(owner).startswith('dee') else OPT.dummy_aperture_heights()


def channel_half_height(r_mm, owner: str = 'dee0', heights=None) -> float:
    """Half the channel's full height [mm] at radius r_mm on electrode ``owner`` (or for the given heights)."""
    return 0.5 * channel_height_mm(r_mm, OPT.aperture_radii(), channel_heights(owner) if heights is None else heights)


def r_max_mm() -> float:
    """Where the dees and the model end [mm]: geometry.r_max_mm, else the cavity radius the gap lines run to."""
    return float(OPT.r_max_mm) if OPT.r_max_mm > 0.0 else float(MACHINE.r_max_cavity_m) * MM


def exit_spec() -> dict:
    """The spiral-inflector export's exit_plane_spec.json: the exit plate line, its corner, and the gap-0 azimuth the
    export placed the inflector for."""
    if not EXIT_SPEC_FILE:
        raise ConfigError("geometry.exit_plane_spec_file is not set (si_rotation = 'corner' and gap0_on_plate read it)")
    if EXIT_SPEC_FILE not in _EXIT_SPEC:
        _EXIT_SPEC[EXIT_SPEC_FILE] = json.loads(Path(EXIT_SPEC_FILE).read_text())
    return _EXIT_SPEC[EXIT_SPEC_FILE]


# =====================================================================================================================
# 1. the gap centre lines
# =====================================================================================================================
def _half_width_mm(cav, pts_mm, side: float):
    """Signed half gap width [mm] at every point of a centre line given in mm.

    The width varies with radius under the gap law (a constant-angle wedge floored at a minimum slot), so a boundary
    is offset by the LOCAL width; offsetting by the nominal width builds a constant gap the optimizer never designed.
    """
    pts = np.asarray(pts_mm, dtype=float)
    return side * 0.5 * np.asarray(cav.gap_width_at(np.hypot(pts[:, 0], pts[:, 1]) / MM), dtype=float) * MM


def load_gap_lines(pkl_path):
    """The RF gap centre lines of the seed optimization's geometry: (DeeSystem, geometry dict, result).

    Each gap is an RFCavity whose ``segments`` are the straight pieces of its centre line ('p1' / 'p2' in metres).
    The gap-width LAW is [machine]'s: results carry no record of the law they were optimized under, and a geometry
    optimized under another one rebuilds with different boundaries."""
    import pickle
    with open(pkl_path, 'rb') as fh:
        result = pickle.load(fh)
    geom = result.metadata['optimal_geometry']
    m = MACHINE
    if len(geom['segment_angles_per_gap']) != 2 * m.n_dees:
        raise ConfigError(f"the seed {pkl_path} has {len(geom['segment_angles_per_gap'])} gaps, [machine] "
                          f"{2 * m.n_dees}")
    # RFCavity needs angles and radii at construction when it has variable segments: seeded with gap 0's, set per gap
    # (machine.apply_seed_geometry: the opening first - it re-places every gap's azimuth - then each gap's segments).
    # The dees are built at [machine] dee_voltage_kv: the potential of the electrode model.
    dees = create_dee_system(m.r_min_m, m.r_max_cavity_m, list(m.dee_center_angles_deg), m.dee_opening_deg,
                             voltage=m.dee_voltage_kv * 1e3, harmonic=m.harmonic, gap_width=m.gap_width_m,
                             gap_angle_deg=m.gap_angle_deg, gap_width_min=m.gap_width_min_m,
                             n_variable_segments=len(geom['segment_angles_per_gap'][0]),
                             segment_angles=list(geom['segment_angles_per_gap'][0]),
                             segment_radii=list(geom['segment_radii_per_gap'][0]))
    op = geom.get('opening_angle_deg')
    if op is not None and abs(float(op) - m.dee_opening_deg) > 1e-9:
        print(f"  NOTE: the seed's dee opening {float(op):.6f} deg is used, not [machine] dee_opening_deg "
              f"{m.dee_opening_deg:.6f}")
    apply_seed_geometry(dees, geom)
    print(f"  gap width law: angular {m.gap_angle_deg:g} deg floored at {m.gap_width_min_m * MM:.2f} mm  -> "
          f"{dees.gaps[0].gap_width_at(0.080) * MM:.2f} mm at r = 80 mm, "
          f"{dees.gaps[0].gap_width_at(0.200) * MM:.2f} mm at r = 200 mm")
    print("  NOTE: the law is [machine]'s, not the seed's. A geometry optimized under a different gap width rebuilds "
          "with DIFFERENT boundaries.")
    return dees, geom, result


def opening_deg(dees: DeeSystem) -> float:
    """The dee opening as the gap lines have it (entry -> exit base angle of dee 0)."""
    return float((dees.gaps[1].base_angle - dees.gaps[0].base_angle) % 360.0)


# =====================================================================================================================
# 2. the housing outline
# =====================================================================================================================
def read_housing_outline(path) -> dict:
    """The housing outline text file written by the spiral-inflector export.

    '#' comments, 'key value' lines ('units mm' or 'units m' - metres when the file has no units line -, 'frame'
    a label), and section headers 'name count' followed by that many rows: 'outer' (x y), 'aperture' (x y z) and
    optionally 'curves' (kind i0 i1). Returns {'frame', 'units', 'outer' (N, 2), 'aperture' (M, 3)} in mm and, when
    the file has them, 'curves' as [(kind, i0, i1), ...]: the exporter simplifies the outline itself (one straight
    plate at the exit, square to the design orbit, one spline round the rest). Files without curves are a plain
    polygon for ``simplify_outline``.
    """
    out, rows, name, left = {'frame': 'deck', 'units': 'm'}, [], None, 0
    with open(path, encoding='utf-8') as fh:
        for raw in fh:
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            if left > 0:
                if name == 'curves':
                    kind, i0, i1 = line.split()
                    rows.append((kind, int(i0), int(i1)))
                else:
                    rows.append([float(v) for v in line.split()])
                left -= 1
                if left == 0:
                    out[name] = rows if name == 'curves' else np.array(rows, dtype=float)
                continue
            key, _, rest = line.partition(' ')
            if key in ('outer', 'aperture', 'curves'):
                name, left, rows = key, int(rest.strip()), []
            else:
                out[key] = rest.strip()
    for key in ('outer', 'aperture'):
        if key not in out:
            raise ValueError(f"{path}: no '{key}' section")
    if out['units'] not in ('mm', 'm'):
        raise ValueError(f"{path}: unknown units {out['units']!r}")
    scale = MM if out['units'] == 'm' else 1.0
    out['outer'] = out['outer'] * scale
    out['aperture'] = out['aperture'] * scale
    out['units'] = 'mm'
    return out


def simplify_outline(outer: np.ndarray, cluster_tol: float = None, keep_every: int = None,
                     n_spline: int = 600) -> dict:
    """A convex-hull outline (two long flat faces joined by a finely sampled curve) as two straight faces and one
    spline - for outline files without a curves section.

    1. the two LONG LINES are the two longest edges; they cut the loop into two arcs;
    2. a CLUSTER is the run of vertices at an arc's end within `cluster_tol` of the long line it touches (if the two
       runs of one arc meet, the whole arc is one cluster: the corner where the two faces meet);
    3. each cluster collapses to its mean, every `keep_every`-th remaining vertex survives;
    4. a cubic spline runs cluster mean -> survivors -> cluster mean, and the long lines close the loop.
    Returns the plot data plus `curve` (the closed outline) and `nodes` / `edges` (what the solid is built from).
    """
    from scipy.interpolate import splev, splprep
    cluster_tol = OPT.cluster_tol_mm if cluster_tol is None else cluster_tol
    keep_every = OPT.keep_every if keep_every is None else keep_every

    n = len(outer)
    e = np.roll(outer, -1, axis=0) - outer
    length = np.hypot(e[:, 0], e[:, 1])
    i1, i2 = sorted(int(k) for k in np.argsort(-length)[:2])      # edge i runs vertex i -> i+1

    def off(line_i, idx):
        """perpendicular distance of vertex idx from the infinite line through edge line_i"""
        d = e[line_i] / length[line_i]
        v = outer[idx] - outer[line_i]
        return abs(v[0] * d[1] - v[1] * d[0])

    arcs = [[(i1 + 1 + k) % n for k in range((i2 - i1) % n)],      # head of edge i1 .. tail of i2
            [(i2 + 1 + k) % n for k in range((i1 - i2) % n)]]      # head of edge i2 .. tail of i1
    head_line, tail_line = [i1, i2], [i2, i1]

    clusters, loose = [], []
    for arc, hl, tl in zip(arcs, head_line, tail_line):
        f = 0
        while f < len(arc) and off(hl, arc[f]) <= cluster_tol:
            f += 1
        b = len(arc) - 1
        while b >= 0 and off(tl, arc[b]) <= cluster_tol:
            b -= 1
        if f > b:                                   # the two runs met: the whole arc is one knot
            clusters.append(list(arc))
            loose.append([])
        else:
            clusters.append(list(arc[:f]))
            clusters.append(list(arc[b + 1:]))
            loose.append(list(arc[f:b + 1]))

    pieces, marks = [], {"reps": [], "kept": [], "cluster_idx": [], "loose_idx": []}
    nodes, edges = [], []
    ci = 0
    for k, arc in enumerate(arcs):
        if not loose[k]:
            rep = outer[clusters[ci]].mean(axis=0)
            marks["reps"].append(rep)
            marks["cluster_idx"].append(clusters[ci])
            pieces.append(rep[None, :])
            nodes.append(rep)
            ci += 1
        else:
            r0, r1 = outer[clusters[ci]].mean(axis=0), outer[clusters[ci + 1]].mean(axis=0)
            marks["reps"] += [r0, r1]
            marks["cluster_idx"] += [clusters[ci], clusters[ci + 1]]
            marks["loose_idx"] += loose[k]
            kept = outer[loose[k][::keep_every]]
            marks["kept"].append(kept)
            pts = np.vstack([r0, kept, r1])
            tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0.0, k=min(3, len(pts) - 1), per=False)
            pieces.append(np.column_stack(splev(np.linspace(0.0, 1.0, n_spline), tck)))
            nodes += [r0, r1]
            edges.append(("spline", kept))
            ci += 2
        edges.append(("line", None))                # the long line on to the next arc
    marks["reps"] = np.array(marks["reps"])
    marks["kept"] = np.vstack(marks["kept"]) if marks["kept"] else np.zeros((0, 2))
    return {"long_edges": (i1, i2), "arcs": arcs, "clusters": clusters, "loose": loose,
            "curve": np.vstack(pieces), "cluster_tol": cluster_tol, "keep_every": keep_every,
            "given": False, "nodes": np.array(nodes), "edges": edges, **marks}


def outline_as_given(ol: dict, n_spline: int = 600) -> dict:
    """The outline as the spiral-inflector export built it, unpacked into ``simplify_outline``'s shape.

    Simplifying it again would apply a second, different set of rules to a simplification. The outline is not convex:
    the exit plate is a chord across the electrode hull, with a re-entrant kink of a few degrees at each end.
    """
    from scipy.interpolate import splev, splprep

    outer, curves = np.asarray(ol['outer'], dtype=float), list(ol['curves'])
    n = len(outer)
    nodes = [outer[i0] for _, i0, _ in curves]
    edges, pieces = [], []
    for kind, i0, i1 in curves:
        chain = outer[[(i0 + k) % n for k in range((i1 - i0) % n + 1)]]
        if kind == 'spline':
            edges.append(("spline", chain[1:-1]))
            tck, _ = splprep([chain[:, 0], chain[:, 1]], s=0.0, k=min(3, len(chain) - 1), per=False)
            pieces.append(np.column_stack(splev(np.linspace(0.0, 1.0, n_spline), tck))[:-1])
        else:
            edges.append(("line", None))
            pieces.append(chain[:-1])
    return {"given": True, "curve": np.vstack(pieces), "nodes": np.array(nodes), "edges": edges,
            "keep_every": 1, "cluster_tol": 0.0, "long_edges": (0, 0), "arcs": [], "clusters": [],
            "loose": [], "reps": np.array(nodes), "kept": np.zeros((0, 2)), "cluster_idx": [], "loose_idx": []}


def housing_outline(ol: dict) -> dict:
    """The outline the housing solid and the dee clearances are built from: as given, or simplified."""
    return outline_as_given(ol) if ol.get('curves') else simplify_outline(ol['outer'])


def exit_half_height_mm(ol: dict) -> float:
    """The spiral-inflector housing exit's half height [mm]: the largest |z| of the exit aperture's corners (the
    channel, centred on the median plane, clears the exit when its half height is at least this). Read off the
    outline's 'aperture' section, after any aperture_inset_mm."""
    ap = np.asarray(ol['aperture'], dtype=float)
    if ap.ndim != 2 or ap.shape[1] < 3:
        raise ConfigError("the housing outline's exit aperture has no z coordinates: the channel bound cannot be "
                          "checked")
    return float(np.abs(ap[:, 2]).max())


def exit_radius_mm(ol: dict) -> float:
    """The spiral-inflector housing exit's outermost radius [mm]: the largest distance of its exit aperture's points
    from the machine axis. The channel heights there are held to the exit's height too (``options.check_bounds``)."""
    ap = np.asarray(ol['aperture'], dtype=float)
    return float(np.hypot(ap[:, 0], ap[:, 1]).max())


# =====================================================================================================================
# 3. the emitting plane and the beam channel
# =====================================================================================================================
def emitting_plane(handoff_path) -> dict:
    """The plane the central region's particles are emitted from, measured from the hand-off file.

    Every particle is released ON the plane where the spiral inflector recorded it, so the hand-off cloud IS the
    emitting plane: its normal is the cloud's degenerate singular direction, turned to point downstream. (Not the exit
    plate's plane of exit_plane_spec.json, which sits upstream of it and is tilted to it.)
    """
    from PyCentralRegion.handoff import load_handoff
    h = load_handoff(str(handoff_path))
    r = np.asarray(h['r'], dtype=float) * MM
    c = r.mean(axis=0)
    _, s, vt = np.linalg.svd(r - c, full_matrices=False)
    n = vt[-1]
    beam = np.asarray(h['p_ev_c'], dtype=float).mean(axis=0)
    if n.dot(beam) < 0.0:
        n = -n
    return {'point': c, 'normal': n, 'n_particles': len(r), 'beam': beam / np.linalg.norm(beam),
            'cloud': r, 'flatness': float(np.abs((r - c).dot(n)).max()),
            'extent': (float(np.ptp((r - c).dot(vt[0]))), float(np.ptp((r - c).dot(vt[1]))))}


def inset_aperture(aperture: np.ndarray, long_side: float, short_side: float) -> np.ndarray:
    """A rectangular exit aperture shrunk symmetrically: `long_side` mm off each end of its longer edge, `short_side`
    off each end of the shorter; centre, plane and orientation kept.

    Gap 0's counter-post IS the housing, so the aperture's long edges play the part the facing post pairs play at the
    other gaps: narrowing them is the gap-0 equivalent of tightening those jaws.
    """
    ap = np.asarray(aperture, dtype=float)
    if len(ap) != 4:
        raise ConfigError(f"aperture_inset_mm needs a 4-corner exit aperture, the outline has {len(ap)} points")
    c = ap.mean(axis=0)
    e1, e2 = ap[1] - ap[0], ap[2] - ap[1]
    l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    if min(l1, l2) < 1e-9:
        raise ValueError("degenerate aperture: cannot inset")
    u1, u2 = e1 / l1, e2 / l2
    d1, d2 = (long_side, short_side) if l1 >= l2 else (short_side, long_side)
    if l1 - 2.0 * d1 <= 0.0 or l2 - 2.0 * d2 <= 0.0:
        raise ConfigError(f"aperture_inset_mm closes the aperture: {l1:.3f} x {l2:.3f} mm minus 2x({d1:.3f}, {d2:.3f})")
    out = []
    for q in ap:
        v = q - c
        s1 = np.sign(float(v.dot(u1))) or 1.0
        s2 = np.sign(float(v.dot(u2))) or 1.0
        out.append(c + s1 * (0.5 * l1 - d1) * u1 + s2 * (0.5 * l2 - d2) * u2)
    print(f"  aperture inset: {l1:.3f} x {l2:.3f} -> {l1 - 2.0 * d1:.3f} x {l2 - 2.0 * d2:.3f} mm "
          f"({d1:.3f} / {d2:.3f} mm off each side)")
    return np.asarray(out, dtype=float)


def beam_channel(aperture: np.ndarray, plane: dict, extend: float = None, back: float = None) -> dict:
    """The beam channel: the exit aperture's footprint on the emitting plane, swept along the plane normal.

    Each aperture corner (any polygon: a round exit opening is a ring of points) is projected normally onto the
    emitting plane, and the polygon is swept `extend` mm downstream and `back` mm upstream: a right prism, both end
    faces parallel to the emitting plane.

    `back` is not zero because the particles are released ON the emitting plane: if it were the channel's closed end,
    every particle would be born on a metal face, and the obstacle raster (even-odd point-in-polygon on z slices)
    splits a coincident face about evenly between metal and void - half the bunch reads as buried at birth. The fault
    is coincidence, not proximity, so a clearance of a couple of raster cells suffices; the beam is not moved.
    """
    extend = OPT.channel_extend_mm if extend is None else extend
    back = OPT.channel_back_mm if back is None else back
    p0, n = plane['point'], plane['normal']
    d = (aperture - p0).dot(n)                       # signed distance of each corner
    near = aperture - d[:, None] * n                 # the polygon, on the emitting plane
    e = np.roll(near, -1, axis=0) - near
    lens = np.linalg.norm(e, axis=1)
    u = e / lens[:, None]
    ang = np.degrees(np.arccos(np.clip([-u[k - 1].dot(u[k]) for k in range(len(near))], -1.0, 1.0)))
    cen = near.mean(axis=0)
    area = 0.5 * float(np.linalg.norm(sum(np.cross(near[k] - cen, near[(k + 1) % len(near)] - cen)
                                          for k in range(len(near)))))
    return {'aperture': aperture, 'near': near, 'far': near + float(extend) * n, 'start': near - float(back) * n,
            'dist': d, 'normal': n, 'sweep': n, 'extend': float(extend), 'back': float(back),
            'edges': lens, 'angles': ang, 'face_area': area,
            'ap_edges': np.linalg.norm(np.roll(aperture, -1, axis=0) - aperture, axis=1)}


# =====================================================================================================================
# 4. gap centre lines as chains: junctions, inert segments
# =====================================================================================================================
def _on_circle(a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    """Where the segment a -> b, running from inside the circle |p| = r about the machine axis to outside it, crosses
    that circle (the outward root of |a + s (b - a)| = r, held to the segment)."""
    ab = b - a
    A, B, C = float(ab.dot(ab)), 2.0 * float(a.dot(ab)), float(a.dot(a)) - r ** 2
    s = (-B + np.sqrt(max(B * B - 4.0 * A * C, 0.0))) / (2.0 * A)
    return a + min(max(s, 0.0), 1.0) * ab


def _clip_outer(chain: np.ndarray, r_max: float) -> np.ndarray:
    """Cut an inner-to-outer chain at r_max, landing the last point exactly on that circle."""
    p = np.asarray(chain, dtype=float)
    r = np.hypot(p[:, 0], p[:, 1])
    if r[-1] <= r_max:
        return p
    k = int(np.argmax(r > r_max))
    if k == 0:
        raise ValueError(f"chain starts beyond r_max = {r_max:.1f} mm")
    return np.vstack([p[:k], _on_circle(p[k - 1], p[k], r_max)])


def junction(cav):
    """The variable -> fixed junction of one gap: the step, and where the lines would meet [mm].

    RFCavity ends the variable chain at base_angle + sum(excursions), rotates the last segment about its midpoint, and
    starts the FIXED segment as a radial ray at the nominal base angle at the last variable radius. The junction so
    carries a step, partly angular (the leftover excursion) and partly radial (the rotation moves the end off its
    circle) - the thin-gap idiom of the 1D model, but a visible jog in the edge of a dee built from these lines.
    Returns the two junction points, the net angular step, the jog, and where the last variable segment's LINE
    crosses the fixed ray.
    """
    var = [s for s in cav.segments if s['type'] == 'variable']
    fix = [s for s in cav.segments if s['type'] == 'fixed'][0]
    a = np.asarray(var[-1]['p1'][:2], dtype=float) * MM
    b = np.asarray(var[-1]['p2'][:2], dtype=float) * MM
    f1 = np.asarray(fix['p1'][:2], dtype=float) * MM
    f2 = np.asarray(fix['p2'][:2], dtype=float) * MM
    u = (b - a) / np.linalg.norm(b - a)
    v = (f2 - f1) / np.linalg.norm(f2 - f1)
    meet, r_meet = None, np.nan
    det = u[0] * (-v[1]) - u[1] * (-v[0])
    if abs(det) > 1e-12:
        t = ((f1 - a)[0] * (-v[1]) - (f1 - a)[1] * (-v[0])) / det
        meet = a + t * u
        r_meet = float(np.hypot(*meet))
    d_az = float((np.degrees(np.arctan2(b[1], b[0])) - np.degrees(np.arctan2(f1[1], f1[0])) + 180.0) % 360.0 - 180.0)
    return {'var_end': b, 'fix_start': f1, 'fix_end': f2, 'r_trans': float(np.hypot(*f1)),
            'jog': float(np.hypot(*(b - f1))), 'step_deg': d_az, 'meet': meet, 'r_meet': r_meet,
            'dr': r_meet - float(np.hypot(*f1)),
            'seg_deg': float(np.degrees(np.arccos(np.clip(abs(u.dot(v)), -1.0, 1.0))))}


def gap_centre_line(cav, junction_tol: float = None, r_max: float = None):
    """The gap centre line in mm, with the variable -> fixed junction closed where it can be: (chain, junction, closed).

    When the net angular step is under `junction_tol`, the last variable segment runs on (or back) to where its line
    crosses the fixed ray, and the ray starts from that point: the jog disappears at the price of moving the
    transition radius by `dr`. Above the tolerance the lines would only meet far inside the geometry.
    """
    junction_tol = OPT.junction_tol_deg if junction_tol is None else junction_tol
    j = junction(cav)
    var = [s for s in cav.segments if s['type'] == 'variable']
    pts = [np.asarray(var[0]['p1'][:2], dtype=float) * MM]
    for s in var[:-1]:
        pts += [np.asarray(s['p2'][:2], dtype=float) * MM, np.asarray(s['p1'][:2], dtype=float) * MM]
    pts.append(np.asarray(var[-1]['p1'][:2], dtype=float) * MM)
    close = abs(j['step_deg']) <= junction_tol and j['meet'] is not None and np.isfinite(j['r_meet'])
    chain = np.array(pts + ([j['meet'], j['fix_end']] if close else [j['var_end'], j['fix_start'], j['fix_end']]))
    chain = _dedupe(chain)                           # the splice can produce a duplicate
    return (_clip_outer(chain, r_max) if r_max else chain), j, close


def match_gaps_to_orbit(dees: DeeSystem, trj: np.ndarray, verbose: bool = True):
    """The seed's gap lines matched to its reference orbit exactly as ``machine.seed_dee_system`` matches the
    tracker's: every variable segment the orbit never crosses made nominal (``rf_cavity.neutralise_inert_segments``:
    it shapes no crossing but would put a kink and a jog in the dee edge), then every node radius put midway between
    the two turns that bracket it (``rf_cavity.snap_nodes_between_turns``: a segment joint never coincides with a beam
    crossing). In place; geometry.snap_nodes = false skips both."""
    from PyCentralRegion.rf_cavity import neutralise_inert_segments, snap_nodes_between_turns
    trj = np.asarray(trj, dtype=float)
    changed = neutralise_inert_segments(dees.gaps, trj, verbose=verbose)
    if verbose:
        print()
    report = snap_nodes_between_turns(dees.gaps, trj, verbose=verbose)
    return changed, report


def move_junctions(dees: DeeSystem, moves, verbose: bool = True):
    """Move gaps' variable -> fixed junctions to new radii ALONG the last variable segment's own line.

    The node snap places each junction midway between two turns of the THIN design orbit, but the 3D beam crosses at
    other radii and can run straight through the jog. ``moves`` are JunctionMove items (or (gap, r_mm) pairs): radii
    from the 3D beam's free bands between passes. The last variable segment keeps its LINE - its outer radius becomes
    r, its excursion and midpoint rotation are re-solved so both stored ends stay on the old line - so every crossing
    of the variable part sees the same gap line; only the hand-over to the fixed ray (and the jog) moves. In place.
    """
    from scipy.optimize import least_squares
    out = []
    for mv in moves:
        g, r_new = (int(mv.gap), float(mv.r_mm) * 1e-3) if hasattr(mv, 'gap') else (int(mv[0]), float(mv[1]) * 1e-3)
        cav = dees.gaps[g]
        n = cav.n_variable_segments
        if n == 0:
            raise ConfigError(f"junction move: gap {g} has no variable segment")
        radii, angles = list(cav.segment_radii), list(cav.segment_angles)
        rots = list(cav.segment_rotations) if cav.segment_rotations else [0.0] * n
        r_prev = cav.segment_radii[-2] if n > 1 else cav.r_min
        if not r_prev < r_new < cav.r_max:
            raise ConfigError(f"junction move: gap {g} radius {r_new * MM:.1f} mm outside "
                             f"({r_prev * MM:.1f}, {cav.r_max * MM:.1f})")
        j_old = junction(cav)
        seg = [s for s in cav.segments if s['type'] == 'variable'][-1]
        a0 = np.asarray(seg['p1'][:2], dtype=float)
        u = np.asarray(seg['p2'][:2], dtype=float) - a0
        u /= np.linalg.norm(u)
        nrm = np.array([-u[1], u[0]])

        def resid(x):
            cav.update_geometry(segment_radii=radii[:-1] + [r_new], segment_angles=angles[:-1] + [float(x[0])],
                                segment_rotations=rots[:-1] + [float(x[1])])
            s = [q for q in cav.segments if q['type'] == 'variable'][-1]
            return MM * np.array([(np.asarray(s['p1'][:2]) - a0) @ nrm, (np.asarray(s['p2'][:2]) - a0) @ nrm])

        sol = least_squares(resid, x0=[angles[-1], rots[-1]], xtol=1e-14, ftol=1e-14)
        err = float(np.max(np.abs(resid(sol.x))))
        if err > 1e-3:
            raise ValueError(f"junction move: gap {g} could not keep the segment line (off by {err:.4f} mm)")
        j_new = junction(cav)
        out.append((g, j_old, j_new))
        if verbose:
            print(f"[junction] gap{g} ({cav.base_angle:.1f} deg): junction r {j_old['r_trans']:.1f} -> "
                  f"{j_new['r_trans']:.1f} mm along the same line (excursion {angles[-1]:+.3f} -> {sol.x[0]:+.3f} deg, "
                  f"rotation {rots[-1]:+.3f} -> {sol.x[1]:+.3f} deg, line kept to {err * 1e3:.2f} um); jog "
                  f"{j_old['jog']:.1f} -> {j_new['jog']:.1f} mm, net step {j_old['step_deg']:+.2f} -> "
                  f"{j_new['step_deg']:+.2f} deg")
    return out


# =====================================================================================================================
# 5. the spiral inflector turned about the machine axis
# =====================================================================================================================
def si_rotation_deg(dees: DeeSystem) -> float:
    """The spiral inflector's rotation about the machine axis [deg, +ccw] the electrodes are built with.

    'fixed': [inflector] rotation_deg, the angle the tracker turns the inflector field, housing and bunch by (a
    campaign that varies the rotation uses this mode). 'corner': gap 0's base angle minus the exit spec's gap azimuth.
    The export placed the inflector for its own dee opening by a rule (the exit plate's outer corner a fixed distance
    off gap 0's ray); another opening moves gap 0's ray, and turning the whole inflector by this angle keeps the rule
    exactly, since it is rotation-equivariant. Either way the model records the angle (params['si_rotation_deg']) and
    the tracker refuses an inflector.rotation_deg that differs from it.
    """
    if OPT.si_rotation == 'corner':
        return float(dees.gaps[0].base_angle) - float(exit_spec()['gap_azimuth_deg'])
    return float(INFLECTOR.rotation_deg)


def _rot2(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s], [s, c]])


def rotate_points(xyz: np.ndarray, deg: float) -> np.ndarray:
    """(..., 2 or 3) points or vectors turned by deg about the machine (z) axis."""
    out = np.array(xyz, dtype=float, copy=True)
    out[..., :2] = out[..., :2] @ _rot2(deg).T
    return out


def rotate_outline(ol: dict, deg: float) -> dict:
    """The housing outline (outer polygon, exit aperture) turned by deg; curves are index ranges and stay."""
    if deg:
        ol = dict(ol)
        ol['outer'] = rotate_points(ol['outer'], deg)
        ol['aperture'] = rotate_points(ol['aperture'], deg)
    return ol


def rotate_plane(plane: dict, deg: float) -> dict:
    """emitting_plane's dict (point, normal, beam direction, particle cloud) turned by deg about the machine axis."""
    if deg:
        plane = dict(plane)
        for k in ('point', 'normal', 'beam', 'cloud'):
            plane[k] = rotate_points(plane[k], deg)
    return plane


def plate_line(spec: dict, half_gap_m: float = None):
    """Gap 0's centre line on the exit plate: the plate's outer face shifted by half a gap along its downstream
    normal. Returns (point (2,) [m], unit direction (2,) pointing outward, normal (2,)), unrotated.

    Pass `half_gap_m` from the gap law: the spec's own half_gap_mm is the export's assumption of a flat gap, and
    taking it would seat the first gap in the wrong place.
    """
    b = spec['baseline']
    p = np.asarray(b['point_mm'][:2], dtype=float) * 1e-3
    n = np.asarray(b['normal'][:2], dtype=float)
    n /= np.linalg.norm(n)
    d = np.array([n[1], -n[0]])          # along the face
    if np.dot(d, p) < 0:                 # outward
        d = -d
    half_gap = float(spec.get('half_gap_mm', 5.0)) * 1e-3 if half_gap_m is None else float(half_gap_m)
    return p + half_gap * n, d, n


def fit_gap_to_line(cav, point, direction, s_range=(-0.06, 0.12), n: int = 4001) -> dict:
    """Lay the cavity's single variable segment ON the line point + s * direction [m]: angle, outer radius and
    midpoint rotation, with its midpoint placed along the line to minimise the jog to the fixed radial segment.

    One variable segment only: two collinear segments on a straight line add no shape freedom, only reach."""
    if cav.n_variable_segments != 1:
        raise ValueError(f"fit_gap_to_line needs exactly ONE variable segment, the gap has {cav.n_variable_segments}")
    b = np.radians(float(cav.base_angle))
    e_b = np.array([np.cos(b), np.sin(b)])
    p1 = cav.r_min * e_b
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    best = None
    for s_m in np.linspace(s_range[0], s_range[1], n):
        m = np.asarray(point) + s_m * d
        p2 = 2.0 * m - p1
        r_outer = float(np.hypot(*p2))
        if not (cav.r_min * 1.5 < r_outer < cav.r_max * 0.9):
            continue
        L = float(np.hypot(*(p2 - p1)))
        jog = float(np.hypot(*(m + 0.5 * L * d - r_outer * e_b)))
        if best is None or jog < best[0]:
            best = (jog, p2, r_outer)
    jog, p2, r_outer = best
    alpha = (np.degrees(np.arctan2(p2[1], p2[0]) - b) + 180.0) % 360.0 - 180.0
    rho = (np.degrees(np.arctan2(d[1], d[0]) - np.arctan2(*(p2 - p1)[::-1])) + 180.0) % 360.0 - 180.0
    cav.update_geometry(segment_angles=[float(alpha)], segment_radii=[float(r_outer)], segment_rotations=[float(rho)])
    seg = cav.segments[0]
    q1, q2 = np.asarray(seg['p1'][:2]), np.asarray(seg['p2'][:2])
    off = [abs(float(d[0] * (q - point)[1] - d[1] * (q - point)[0])) for q in (q1, q2)]
    return {'angle_deg': float(alpha), 'radius_m': float(r_outer), 'rotation_deg': float(rho),
            'inner_end_mm': (q1 * MM).tolist(), 'outer_end_mm': (q2 * MM).tolist(),
            'inner_r_mm': float(np.hypot(*q1) * MM), 'outer_r_mm': float(np.hypot(*q2) * MM),
            'jog_to_fixed_segment_mm': float(jog * MM), 'max_offset_from_line_mm': float(max(off) * MM)}


def refit_gap0_on_plate(dees: DeeSystem, verbose: bool = True) -> dict:
    """Gap 0's single variable segment back onto the turned exit-plate line at half the local gap width.

    Run AFTER the node snap: gap 0's junction is the plate corner by construction, and the snap, which moves the node
    radius without keeping the line, slides the plate segment off the plate - the gap-0 slot then reads up to twice
    its design width in 3D while the thin model, which never snaps, keeps it."""
    sp = exit_spec()
    cav = dees.gaps[0]
    rot = si_rotation_deg(dees)
    half = 0.5 * float(cav.gap_width_at(float(sp['baseline']['r_mm']) * 1e-3))
    point, d, _ = plate_line(sp, half_gap_m=half)
    R = _rot2(rot)
    fit = fit_gap_to_line(cav, R @ point, R @ d)
    if verbose:
        C = R @ np.asarray(sp['corner']['rotated_mm'][:2], dtype=float)
        eb = np.array([np.cos(np.radians(cav.base_angle)), np.sin(np.radians(cav.base_angle))])
        print(f"[SI rotation] {rot:+.3f} deg (si_rotation = {OPT.si_rotation}; gap-0 ray {float(cav.base_angle):.3f} "
              f"deg, the export's {float(sp['gap_azimuth_deg']):.1f}): plate corner at ({C[0]:.2f}, {C[1]:.2f}) mm, "
              f"{eb[0] * C[1] - eb[1] * C[0]:+.3f} mm off the ray; gap 0 re-fitted on the plate line: r "
              f"{fit['inner_r_mm']:.1f} .. {fit['outer_r_mm']:.1f} mm, off the line "
              f"{fit['max_offset_from_line_mm']:.1e} mm, jog to the radial part {fit['jog_to_fixed_segment_mm']:.2f} mm")
    return fit


# =====================================================================================================================
# 6. chain helpers: distance to the housing, crossings, cuts
# =====================================================================================================================
def _wall_distance(pts: np.ndarray, wall: np.ndarray) -> np.ndarray:
    """Signed distance of each point from the housing outline, + outside, - inside."""
    from matplotlib.path import Path as MplPath
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    seg = np.roll(wall, -1, axis=0) - wall
    ll = np.maximum(np.einsum('ij,ij->i', seg, seg), 1e-30)
    d = np.empty(len(pts))
    for i, p in enumerate(pts):
        t = np.clip(np.einsum('ij,ij->i', p - wall, seg) / ll, 0.0, 1.0)
        d[i] = np.min(np.hypot(*(wall + t[:, None] * seg - p).T))
    return np.where(MplPath(wall).contains_points(pts), -d, d)


def _densify(chain: np.ndarray, ds: float = 0.05):
    """(points every ds mm along the chain, their arclengths, the chain's own vertex arclengths)."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(chain, axis=0).T))])
    ss = np.linspace(0.0, s[-1], max(2, int(s[-1] / ds) + 1))
    return np.column_stack([np.interp(ss, s, chain[:, 0]), np.interp(ss, s, chain[:, 1])]), ss, s


def _dedupe(p: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    keep = [p[0]]
    for q in p[1:]:
        if np.hypot(*(q - keep[-1])) > tol:
            keep.append(q)
    return np.array(keep)


def _reach_inner(chain: np.ndarray, wall: np.ndarray, target: float, step: float = 0.25, max_march: float = 300.0):
    """Take a chain's inner end to where its housing distance equals `target`, either way: cut back when it starts
    too far in, EXTENDED when it starts short (a strongly rotated segment can leave the inner end outside the
    housing). The extension runs along the chain's own first segment, else - when that runs nearly parallel to a flat
    housing face and never crosses in - straight for the housing centroid; it is buried in the housing either way."""
    p = np.asarray(chain, dtype=float)
    if _wall_distance(p[:1], wall)[0] > target:
        t = np.arange(step, max_march, step)
        for u in (p[0] - p[1], wall.mean(axis=0) - p[0]):
            n = np.linalg.norm(u)
            if n < 1e-12:
                continue
            d = _wall_distance(p[0] + t[:, None] * (u / n), wall)
            ok = np.where(d <= target)[0]
            if len(ok):
                return np.vstack([p[0] + t[ok[0]] * (u / n), p]), float(d[ok[0]]), True
        return p, float(_wall_distance(p[:1], wall)[0]), False
    return _cut_inner_at_clearance(p, wall, target)


def _cut_inner_at_clearance(chain: np.ndarray, wall: np.ndarray, clearance: float):
    """Drop the inner part of a chain that comes closer than `clearance` to the housing: walking outward from inside
    the housing the distance rises through zero and past the clearance; the chain starts at the first point clear."""
    dn, ss, s = _densify(chain)
    d = _wall_distance(dn, wall)
    ok = np.where(d >= clearance)[0]
    if not len(ok):
        raise ValueError("the whole chain is within the clearance of the housing")
    if ok[0] == 0:
        return chain, float(d[0]), False                     # already clear at its inner end
    keep = chain[np.searchsorted(s, ss[ok[0]]):]
    return np.vstack([dn[ok[0]], keep]), float(d[ok[0]]), True


def _crossings(a: np.ndarray, b: np.ndarray):
    """The points where polyline a crosses polyline b, walking a from its inner end, as (point, i_a, i_b) with the
    segment indices (parallel segments never cross; the ends count, within 1e-9 of a segment's length)."""
    for i in range(len(a) - 1):
        p, r = a[i], a[i + 1] - a[i]
        for j in range(len(b) - 1):
            q, t = b[j], b[j + 1] - b[j]
            den = r[0] * t[1] - r[1] * t[0]
            if abs(den) < 1e-12:
                continue
            ta = ((q - p)[0] * t[1] - (q - p)[1] * t[0]) / den
            tb = ((q - p)[0] * r[1] - (q - p)[1] * r[0]) / den
            if -1e-9 <= ta <= 1 + 1e-9 and -1e-9 <= tb <= 1 + 1e-9:
                yield p + ta * r, i, j


def _all_crossings(a: np.ndarray, b: np.ndarray) -> list:
    """Every point where polyline a crosses polyline b: [(point, i_a, i_b), ...]."""
    return list(_crossings(a, b))


def _first_crossing(a: np.ndarray, b: np.ndarray):
    """First point (walking a from its inner end) where polyline a crosses polyline b: (point, i_a) or (None, None)."""
    return next(((q, i) for q, i, _ in _crossings(a, b)), (None, None))


def _clip_inner(chain: np.ndarray, r_min: float) -> np.ndarray:
    """Cut an inner-to-outer chain back to r_min, landing its first point exactly on that circle."""
    p = np.asarray(chain, dtype=float)
    r = np.hypot(p[:, 0], p[:, 1])
    if r[0] >= r_min - 1e-12:
        return p
    k = int(np.argmax(r >= r_min))
    if k == 0:
        raise ValueError(f"chain never reaches r_min = {r_min:.3f} mm")
    return np.vstack([_on_circle(p[k - 1], p[k], r_min), p[k:]])


def _meet_chains(a: np.ndarray, b: np.ndarray):
    """Where two inner-to-outer chains meet: their first real crossing, else the intersection of their innermost
    segments EXTENDED inward. Returns (point, ia, ib) with the segment indices, -1 for the inward extension."""
    q, ia = _first_crossing(a, b)
    if q is not None:
        _, ib = _first_crossing(b, a)
        return q, ia, (ib if ib is not None else 0)
    ua, ub = a[0] - a[1], b[0] - b[1]           # inward directions
    den = ua[0] * ub[1] - ua[1] * ub[0]
    if abs(den) < 1e-12:
        return None, None, None
    d = b[0] - a[0]
    t = (d[0] * ub[1] - d[1] * ub[0]) / den
    return a[0] + t * ua, -1, -1


def _start_at(chain: np.ndarray, q: np.ndarray, i: int) -> np.ndarray:
    """Re-start a chain at q: on segment i, or prepended when i == -1 (inward extension)."""
    return np.vstack([q, chain]) if i < 0 else np.vstack([q, chain[i + 1:]])


def _advance_chain(chain: np.ndarray, d: float) -> np.ndarray:
    """Drop the first `d` mm of arclength, keeping the cut point as the new first vertex."""
    if d <= 0.0:
        return chain
    seg = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    sl = np.concatenate([[0.0], np.cumsum(seg)])
    if d >= sl[-1]:
        raise ValueError(f"cut of {d:.2f} mm is longer than the boundary itself ({sl[-1]:.2f} mm)")
    k = int(np.clip(np.searchsorted(sl, d) - 1, 0, len(seg) - 1))
    t = (d - sl[k]) / seg[k]
    return np.vstack([chain[k] + t * (chain[k + 1] - chain[k]), chain[k + 1:]])


def _nearest_on_chain(chain: np.ndarray, p: np.ndarray):
    """(distance, arclength) of the point of a polyline nearest to p."""
    seg = np.diff(chain, axis=0)
    ll = np.maximum(np.einsum('ij,ij->i', seg, seg), 1e-30)
    t = np.clip(np.einsum('ij,ij->i', p - chain[:-1], seg) / ll, 0.0, 1.0)
    dist = np.hypot(*(chain[:-1] + t[:, None] * seg - p).T)
    k = int(np.argmin(dist))
    sl = np.concatenate([[0.0], np.cumsum(np.sqrt(ll))])
    return float(dist[k]), float(sl[k] + t[k] * np.sqrt(ll[k]))


def _arc(r: float, a0: float, a1: float, ccw: bool = True, ds: float = 2.0) -> np.ndarray:
    """Arc of radius r [mm] from azimuth a0 to a1 [rad], the given way round; sampled for plots and measurements
    only - the solid gets a true circle arc."""
    d = (a1 - a0) % (2.0 * np.pi) if ccw else -((a0 - a1) % (2.0 * np.pi))
    n = max(2, int(np.ceil(abs(d) * r / ds)) + 1)
    t = a0 + d * np.linspace(0.0, 1.0, n)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


# =====================================================================================================================
# 7. the dees' inner closures
# =====================================================================================================================
def origin_arc_closure(lo: np.ndarray, hi: np.ndarray, wall: np.ndarray, ds: float = 1.0):
    """Both inner ends on the SAME radius - the larger of the two - closed by an arc about the machine axis, so the
    tapered tip is trimmed by a plain cylinder (which an arc following the housing could not be)."""
    r_lo, r_hi = float(np.hypot(*lo[0])), float(np.hypot(*hi[0]))
    r_in = max(r_lo, r_hi)
    lo2, hi2 = _clip_inner(lo, r_in), _clip_inner(hi, r_in)
    a_hi = float(np.arctan2(hi2[0][1], hi2[0][0]))
    a_lo = float(np.arctan2(lo2[0][1], lo2[0][0]))
    pts = _arc(r_in, a_hi, a_lo, ccw=False, ds=ds)     # back the way the loop came: hi -> lo
    return lo2, hi2, {'pts': pts, 'kind': 'arc at origin', 'radius': r_in, 'centre': np.zeros(2),
                      'span': float(np.hypot(*(lo2[0] - hi2[0]))),
                      'sweep_deg': float(np.degrees((a_hi - a_lo) % (2.0 * np.pi))),
                      'r_lo': r_lo, 'r_hi': r_hi, 'trimmed': abs(r_lo - r_hi), 'from': 'lo' if r_lo > r_hi else 'hi',
                      'min_d': float(_wall_distance(pts, wall).min())}


def _on_clearance_locus(direction: np.ndarray, centre: np.ndarray, wall: np.ndarray, clearance: float,
                        r_hi: float = 400.0) -> np.ndarray:
    """The point `clearance` off the housing on the ray from `centre` along `direction`: the housing distance rises
    monotonically along an outward ray from inside it, so a bisection lands on the locus exactly."""
    u = np.asarray(direction, dtype=float)
    u = u / np.linalg.norm(u)
    lo, hi = 0.0, r_hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _wall_distance([centre + mid * u], wall)[0] < clearance:
            lo = mid
        else:
            hi = mid
    return centre + 0.5 * (lo + hi) * u


def housing_locus_closure(lo: np.ndarray, hi: np.ndarray, wall: np.ndarray, clearance: float,
                          ds: float = 1.0) -> dict:
    """Close a dee's inner end ALONG the housing's clearance locus, from hi's inner end to lo's ('hug').

    Both ends already sit on the locus; the closure is sampled on rays from the housing centroid between their two
    directions (the shorter way round), each ray's point bisected onto the locus, so the metal keeps exactly
    `clearance` off the wall everywhere."""
    centre = wall.mean(axis=0)
    p_from, p_to = np.asarray(hi[0], dtype=float), np.asarray(lo[0], dtype=float)
    a0 = float(np.arctan2(*(p_from - centre)[::-1]))
    a1 = float(np.arctan2(*(p_to - centre)[::-1]))
    da = (a1 - a0 + np.pi) % (2.0 * np.pi) - np.pi              # shorter way round, signed
    r_mean = 0.5 * (np.hypot(*(p_from - centre)) + np.hypot(*(p_to - centre)))
    n = max(3, int(np.ceil(abs(da) * r_mean / ds)) + 1)
    mid = [_on_clearance_locus(np.array([np.cos(a), np.sin(a)]), centre, wall, clearance)
           for a in a0 + da * np.linspace(0.0, 1.0, n)[1:-1]]
    pts = np.vstack([p_from] + mid + [p_to])
    return {'pts': pts, 'kind': 'housing locus', 'radius': np.nan, 'centre': centre,
            'span': float(np.hypot(*(p_to - p_from))), 'sweep_deg': float(np.degrees(abs(da))),
            'min_d': float(_wall_distance(pts, wall).min())}


def housing_offset_polygon(wall: np.ndarray, offset: float, n: int = 720) -> np.ndarray:
    """The closed locus `offset` off the housing, sampled on n rays from its centroid [mm]."""
    centre = wall.mean(axis=0)
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.array([_on_clearance_locus(np.array([np.cos(t), np.sin(t)]), centre, wall, offset) for t in a])


def _blunt_tip(lo: np.ndarray, hi: np.ndarray, width: float):
    """Advance two chains that start at their crossing by the same arclength d until their first points are `width`
    apart: a needle tip becomes a straight face. Returns (lo, hi, d)."""
    def sep(d):
        return float(np.hypot(*(_advance_chain(lo, d)[0] - _advance_chain(hi, d)[0])))
    d_hi = 1.0
    while sep(d_hi) < width:
        d_hi *= 2.0
        if d_hi > 0.5 * min(np.sum(np.linalg.norm(np.diff(lo, axis=0), axis=1)),
                            np.sum(np.linalg.norm(np.diff(hi, axis=0), axis=1))):
            raise ValueError(f"the boundaries never open to {width} mm")
    d_lo = 0.0
    for _ in range(60):
        mid = 0.5 * (d_lo + d_hi)
        if sep(mid) < width:
            d_lo = mid
        else:
            d_hi = mid
    return _advance_chain(lo, d_hi), _advance_chain(hi, d_hi), d_hi


def dee0_round_nose(lo: np.ndarray, hi: np.ndarray, R: float, wall: np.ndarray) -> dict:
    """dee0's vertex tip (lo and hi start at it) as a round of radius R tangent to both edges.

    The vertex is a knife edge the full height of the model at the dee voltage, with ground one gap away: the one
    acute convex edge on any dee and the strongest field singularity in the BEM. The centre is where the two edges,
    offset R into the dee, cross (lo's dee side is its left, hi's its right), so an edge kinked near the tip is handled
    too; the tangent points are the centre's feet on the edges, 'skip' their arclengths from the vertex."""
    from PyCentralRegion.gap_fields import offset_polyline
    V = np.asarray(lo[0], dtype=float)
    if np.hypot(*(np.asarray(hi[0]) - V)) > 1e-6:
        raise ValueError("dee0 round nose: lo and hi do not start at one vertex (a trimmed / cut tip)")
    C, _ = _first_crossing(offset_polyline(lo, +R), offset_polyline(hi, -R))
    if C is None:
        raise ValueError(f"dee0 round nose: the edges offset {R:g} mm into the dee never cross")
    (d_lo, s_lo), (d_hi, s_hi) = _nearest_on_chain(lo, C), _nearest_on_chain(hi, C)
    if max(abs(d_lo - R), abs(d_hi - R)) > 1e-6:
        raise ValueError(f"dee0 round nose: the round is not tangent to both edges ({d_lo:.4f} / {d_hi:.4f} mm from "
                         f"its centre for R {R:g})")
    T_lo, T_hi = _advance_chain(lo, s_lo)[0], _advance_chain(hi, s_hi)[0]
    a_hi, a_lo = np.arctan2(*(T_hi - C)[::-1]), np.arctan2(*(T_lo - C)[::-1])
    da = (a_lo - a_hi + np.pi) % (2.0 * np.pi) - np.pi                # the short way: the side facing the vertex
    a = a_hi + da * np.linspace(0.0, 1.0, max(3, int(np.ceil(abs(da) * R / 0.25)) + 1))
    pts = C + R * np.column_stack([np.cos(a), np.sin(a)])
    pts[0], pts[-1] = T_hi, T_lo
    u_lo, u_hi = lo[1] - lo[0], hi[1] - hi[0]
    ang = float(np.degrees(np.arccos(np.clip(u_lo @ u_hi / np.hypot(*u_lo) / np.hypot(*u_hi), -1.0, 1.0))))
    return {'pts': pts, 'kind': 'round', 'radius': float(R), 'centre': C, 'vertex': V, 'skip': (s_lo, s_hi),
            'angle_deg': ang, 'sweep_deg': float(np.degrees(abs(da))), 'span': float(np.hypot(*(T_lo - T_hi))),
            'min_d': float(_wall_distance(pts, wall).min())}


def metal_edges(d: dict):
    """(lo, hi) as the metal has them. A round dee0 nose starts them at its tangent points; d['lo'] / d['hi'] keep the
    vertex as their arclength origin, so hardware specs do not move with the round."""
    tip = d['tip']
    if tip.get('kind') != 'round':
        return d['lo'], d['hi']
    return (_dedupe(_advance_chain(d['lo'], tip['skip'][0]), 1e-6),
            _dedupe(_advance_chain(d['hi'], tip['skip'][1]), 1e-6))


# =====================================================================================================================
# 8. the dees and the dummy dees in 2D
# =====================================================================================================================
def dee_boundaries(dees: DeeSystem, wall: np.ndarray, r_max: float = None, junction_tol: float = None,
                   clearance: float = None, dee0_trim: float = None, dee0_clear: float = None):
    """The two boundary chains of each dee [mm] out to r_max, with the inner and outer closures: one dict per dee.

    A dee sits between its ENTRY gap (gap 2j) and its EXIT gap (2j+1). Its boundaries are those centre lines offset by
    half the local gap width TOWARDS the dee - the entry gap to its CCW side (`lo`), the exit gap to its CW side
    (`hi`) - with gap_fields.offset_polyline, which gap_fields uses for its wedges too.

    Inner ends: the boundaries of gaps 1 .. n stop `clearance` off the housing, measured on the METAL (the offset
    edge), not on the centre line, which would let the edge sit half a gap closer. dee0 ends in a VERTEX where its two
    boundaries meet - its entry gap's last segment can swing across the dee's own exit gap, so a clearance cut would
    leave no tip at all - optionally cut back to a clearance (dee0_clear) or by a fixed arclength (dee0_trim), or
    rounded (geometry.dee0_nose_mm). Dees 1 .. n-1 close on an arc about the axis ('arc') or along the housing's
    clearance locus ('hug'); a PINCHED dee (its boundaries cross outside the housing) starts at its outermost crossing,
    blunted to a straight face; geometry.dee_rmin moves an inner end out by hand. Each of these moves the arclength
    origin of that dee's two gap edges, and says so.
    """
    from PyCentralRegion.gap_fields import offset_polyline
    r_max = r_max_mm() if r_max is None else r_max
    junction_tol = OPT.junction_tol_deg if junction_tol is None else junction_tol
    clearance = OPT.housing_clearance_mm if clearance is None else clearance
    trim = float(OPT.dee0_trim_mm if dee0_trim is None else dee0_trim)
    dee0_clear = float(OPT.dee0_clearance_mm if dee0_clear is None else dee0_clear)
    rmin_of = {int(item.dee): float(item.r_mm) for item in OPT.dee_rmin}

    raw, jun, closed = [], [], []
    for cav in dees.gaps:
        ch, j, cl = gap_centre_line(cav, junction_tol, r_max=None)
        raw.append(ch)
        jun.append(j)
        closed.append(cl)
    # one boundary per gap: gap 2j is dee j's lo edge (offset +, into the dee), gap 2j+1 its hi edge (offset -).
    # The FULL centre line is offset and the RESULT clipped at r_max: a point on the r_max circle offset sideways lands
    # outside it, so clipping first would leave the boundary ends off the arc they have to meet.
    bnd = []
    for g, cav in enumerate(dees.gaps):
        s = +1.0 if g % 2 == 0 else -1.0
        bnd.append(_clip_outer(offset_polyline(raw[g], _half_width_mm(cav, raw[g], s)), r_max))

    inner = {}
    full = [b.copy() for b in bnd]
    for g in range(1, len(bnd)):
        bnd[g], d0, moved = _cut_inner_at_clearance(bnd[g], wall, clearance)
        inner[g] = {'rule': 'clearance', 'd': d0, 'moved': moved, 'start': bnd[g][0].copy()}
    # dee0's vertex, taken on the FULL boundaries (a property of the two lines, not of where gap 1 was cut for
    # clearance) and on their inward extensions when they no longer reach each other
    meet, i0, i1 = _meet_chains(full[0], full[1])
    if meet is None:
        inner[0] = {'rule': 'gap 0 || gap 1 - left as it is', 'moved': False,
                    'd': float(_wall_distance(bnd[0][:1], wall)[0]), 'start': bnd[0][0].copy()}
    else:
        bnd[0] = _start_at(full[0], meet, i0)
        bnd[1] = _start_at(full[1], meet, i1)
        how = 'meets gap 1' if i0 >= 0 else 'meets gap 1 (extended)'
        cut = ''                                   # what was cut off the vertex, if anything
        if dee0_clear > 0.0:
            bnd[0], _, cut_lo = _cut_inner_at_clearance(bnd[0], wall, dee0_clear)
            bnd[1], _, cut_hi = _cut_inner_at_clearance(bnd[1], wall, dee0_clear)
            cut = f"cut to {dee0_clear:.2f} mm housing clearance" if cut_lo or cut_hi else ''
            how += f', then cut to {dee0_clear:.2f} mm clearance' if cut else f', {dee0_clear:.2f} mm clear: not cut'
        elif trim > 0.0:
            bnd[0] = _advance_chain(bnd[0], trim)
            bnd[1] = _advance_chain(bnd[1], trim)
            cut = f"cut back {trim:.2f} mm"
            how += f', then cut back {trim:.2f} mm'
        d_meet = float(_wall_distance([meet], wall)[0])
        inner[0] = {'rule': how, 'moved': True, 'meet': meet, 'trim': trim,
                    'd': float(_wall_distance(bnd[0][:1], wall)[0]), 'start': bnd[0][0].copy()}
        inner[1] = {'rule': how.replace('gap 1', 'gap 0'), 'moved': True, 'meet': meet, 'trim': trim,
                    'd': float(_wall_distance(bnd[1][:1], wall)[0]), 'start': bnd[1][0].copy()}
        if cut:
            print(f"[dee0] tip {cut}: the gap0/gap1 vertex at r = {np.hypot(*meet):.2f} mm "
                  f"(housing clearance {d_meet:+.2f} mm) becomes a {np.hypot(*(bnd[0][0] - bnd[1][0])):.2f} mm face "
                  f"from r = {np.hypot(*bnd[1][0]):.2f} to {np.hypot(*bnd[0][0]):.2f} mm, clearing the housing by "
                  f"{min(inner[0]['d'], inner[1]['d']):+.2f} mm")
            print("[dee0] ARCLENGTH ORIGIN MOVED on the gap-0-dee and gap-1-dee edges - any post spec written "
                  "against a different dee0 tip is stale and must be RE-DERIVED")
        elif dee0_clear > 0.0:
            print(f"[dee0] tip not cut: the gap0/gap1 vertex at r = {np.hypot(*meet):.2f} mm already clears the housing "
                  f"by {d_meet:+.2f} mm (dee0_clearance_mm {dee0_clear:.2f}); the arclength origin stays at the vertex")
    for g in range(len(bnd)):                      # what the metal really clears, end to end
        inner[g]['min_d'] = float(_wall_distance(_densify(bnd[g], 0.05)[0], wall).min())

    out = []
    for j in range(len(dees.gaps) // 2):
        entry, exit_ = dees.gaps[2 * j], dees.gaps[2 * j + 1]
        span = np.mod(exit_.base_angle - entry.base_angle, 360.0)
        if not 0.0 < span < 180.0:
            raise ValueError(f"dee {j}: entry -> exit spans {span:.2f} deg CCW, expected (0, 180); the gap pairing is "
                             f"not entry / exit")
        c_lo, lo, j_lo, cl_lo = _clip_outer(raw[2 * j], r_max), bnd[2 * j], jun[2 * j], closed[2 * j]
        c_hi, hi, j_hi, cl_hi = _clip_outer(raw[2 * j + 1], r_max), bnd[2 * j + 1], jun[2 * j + 1], closed[2 * j + 1]
        pinched = None
        if j > 0 and OPT.dee_tip_width_mm > 0.0:
            xs = _all_crossings(lo, hi)
            if xs:
                meet, i_lo, i_hi = max(xs, key=lambda c: float(np.hypot(*c[0])))
                lo, hi = _start_at(lo, meet, i_lo), _start_at(hi, meet, i_hi)
                lo, hi, d_cut = _blunt_tip(lo, hi, OPT.dee_tip_width_mm)
                d_meet = float(_wall_distance([meet], wall)[0])
                pinched = {'meet': meet, 'cut': d_cut, 'n_cross': len(xs)}
                for g, b in ((2 * j, lo), (2 * j + 1, hi)):
                    inner[g].update(rule=inner[g]['rule'] + f', PINCHED: starts at the crossing r {np.hypot(*meet):.2f}',
                                    moved=True, start=b[0].copy(), d=float(_wall_distance(b[:1], wall)[0]),
                                    min_d=float(_wall_distance(_densify(b, 0.05)[0], wall).min()))
                print(f"[dee{j}] PINCHED: its boundaries cross {len(xs)} time(s), the outermost at r = "
                      f"{np.hypot(*meet):.2f} mm, az {np.degrees(np.arctan2(meet[1], meet[0])):.1f} deg "
                      f"({d_meet:+.2f} mm off the housing); tip cut back {d_cut:.2f} mm to a "
                      f"{OPT.dee_tip_width_mm:.1f} mm straight face at r {np.hypot(*lo[0]):.2f} / "
                      f"{np.hypot(*hi[0]):.2f} mm - no housing end wall, slotted from its corner-post radius")
                print(f"[dee{j}] ARCLENGTH ORIGIN MOVED on the gap-{2 * j}-dee and gap-{2 * j + 1}-dee edges - post "
                      f"specs written against an unpinched dee{j} are stale and must be RE-DERIVED")
        r_set = rmin_of.get(j, 0.0) if j > 0 else 0.0
        rmin_cut = r_set > min(float(np.hypot(*lo[0])), float(np.hypot(*hi[0])))
        if rmin_cut:                               # an inner end moved out by hand overrides the pinch / hug tip
            r_was = (float(np.hypot(*lo[0])), float(np.hypot(*hi[0])))
            lo, hi = _clip_inner(lo, r_set), _clip_inner(hi, r_set)
            pinched = None
            for g, b in ((2 * j, lo), (2 * j + 1, hi)):
                inner[g].update(rule=inner[g]['rule'] + f', RMIN: moved out to r {r_set:.2f} (dee_rmin)', moved=True,
                                start=b[0].copy(), d=float(_wall_distance(b[:1], wall)[0]),
                                min_d=float(_wall_distance(_densify(b, 0.05)[0], wall).min()))
            print(f"[dee{j}] RMIN (dee_rmin): inner end moved out from r {r_was[0]:.2f} / {r_was[1]:.2f} to "
                  f"{r_set:.2f} mm, closed by an arc about the axis")
            print(f"[dee{j}] ARCLENGTH ORIGIN MOVED on the gap-{2 * j}-dee and gap-{2 * j + 1}-dee edges - post specs "
                  f"written without dee_rmin are stale and must be RE-DERIVED")
        # the inner closure runs the other way round the loop: hi's inner end back to lo's
        if pinched is not None:
            tip = {'pts': np.vstack([hi[0], lo[0]]), 'kind': 'straight', 'radius': np.nan, 'centre': None,
                   'span': float(np.hypot(*(lo[0] - hi[0]))), 'pinched': True, **pinched,
                   'min_d': float(_wall_distance([hi[0], lo[0]], wall).min())}
        elif j == 0:
            # a vertex when uncut (both boundaries start at the meeting point); cut, the tip is a straight face
            span0 = float(np.hypot(*(lo[0] - hi[0])))
            tip = {'pts': np.vstack([hi[0], lo[0]]), 'kind': 'straight' if span0 > 1e-6 else 'vertex',
                   'radius': np.nan, 'centre': None, 'span': span0,
                   'min_d': float(_wall_distance([hi[0], lo[0]], wall).min())}
            if OPT.dee0_nose_mm > 0.0:
                if tip['kind'] != 'vertex':
                    raise ConfigError(f"dee0_nose_mm rounds a VERTEX tip; dee0's is a {span0:.2f} mm straight face "
                                     f"(dee0_trim_mm / dee0_clearance_mm cut it)")
                tip = dee0_round_nose(lo, hi, OPT.dee0_nose_mm, wall)
                print(f"[dee0] NOSE ROUNDED (dee0_nose_mm): the {tip['angle_deg']:.1f} deg vertex at r "
                      f"{np.hypot(*tip['vertex']):.2f} mm -> R {OPT.dee0_nose_mm:g} mm round, centre "
                      f"({tip['centre'][0]:.3f}, {tip['centre'][1]:.3f}), tip at r "
                      f"{np.hypot(*tip['pts'].T).min():.2f} mm, tangent to the gap-0 / gap-1 edges "
                      f"{tip['skip'][0]:.2f} / {tip['skip'][1]:.2f} mm from the vertex; arclength origin KEPT at the "
                      f"vertex (post specs unchanged); housing clearance {tip['min_d']:.2f} mm")
        elif rmin_cut or OPT.dee_inner != 'hug':
            lo, hi, tip = origin_arc_closure(lo, hi, wall)
        else:
            tip = housing_locus_closure(lo, hi, wall, clearance, ds=OPT.hug_ds_mm)
        # the outer closure: lo runs inner -> outer, so the arc continues CCW from its end round to the end of hi
        a0 = float(np.arctan2(lo[-1][1], lo[-1][0]))
        a1 = float(np.arctan2(hi[-1][1], hi[-1][0]))
        out.append({'label': f'dee{j}', 'span_deg': span, 'r_max': float(r_max),
                    'voltage': float(entry.voltage), 'phase_deg': float(entry.phase_deg),
                    'gaps': (2 * j, 2 * j + 1), 'closed': (cl_lo, cl_hi), 'junction': (j_lo, j_hi),
                    'arc': _arc(r_max, a0, a1, ccw=True), 'arc_deg': float(np.degrees((a1 - a0) % (2.0 * np.pi))),
                    'inner': (inner[2 * j], inner[2 * j + 1]), 'clearance': float(clearance), 'tip': tip,
                    'centre_lo': c_lo, 'centre_hi': c_hi, 'lo': lo, 'hi': hi})
    return out


def dee_width(d: dict, n: int = 600):
    """(radius, width) of a dee across its two boundaries [mm]; width <= 0 where they have crossed over."""
    def track(chain):
        r = np.hypot(chain[:, 0], chain[:, 1])
        a = np.unwrap(np.arctan2(chain[:, 1], chain[:, 0]))
        k = np.argsort(r)
        return r[k], a[k]

    r_lo, a_lo = track(d['lo'])
    r_hi, a_hi = track(d['hi'])
    rr = np.linspace(max(r_lo[0], r_hi[0]), min(r_lo[-1], r_hi[-1]), n)
    da = np.interp(rr, r_hi, a_hi) - np.interp(rr, r_lo, a_lo)
    da = (da + np.pi) % (2.0 * np.pi) - np.pi          # signed angle across the dee
    return rr, da * rr


def dummy_boundaries(dees: DeeSystem, wall: np.ndarray, r_max: float = None, junction_tol: float = None,
                     overlap: float = None):
    """The dummy dees (the grounded wedges between the dees), built like the dees from the OTHER sides of the same
    gap lines: dummy k sits between dee k's exit gap (offset +half a gap) and dee k+1's entry gap (offset -).

    Its inner ends run INTO the housing by `overlap`, so the union with the housing is an overlap rather than two
    faces meeting. Two boundaries that cross before their inner ends (converging gap lines) would make a bow-tie
    footprint OCC cannot hollow: the dummy starts at the OUTERMOST crossing, a vertex when that is inside the housing
    (the union hides it), cut back to a straight face geometry.dummy_tip_width_mm wide when it is outside."""
    from PyCentralRegion.gap_fields import offset_polyline
    r_max = r_max_mm() if r_max is None else r_max
    junction_tol = OPT.junction_tol_deg if junction_tol is None else junction_tol
    overlap = OPT.dummy_overlap_mm if overlap is None else overlap

    n = len(dees.gaps)
    out = []
    for k in range(n // 2):
        g_lo, g_hi = 2 * k + 1, (2 * k + 2) % n
        offs = []
        for g, s in ((g_lo, +1.0), (g_hi, -1.0)):
            cav = dees.gaps[g]
            ch, _, _ = gap_centre_line(cav, junction_tol, r_max=None)
            off = _clip_outer(offset_polyline(ch, _half_width_mm(cav, ch, s)), r_max)
            off, d0, _ = _reach_inner(off, wall, -abs(overlap))
            offs.append((off, d0))
        (lo, d_lo), (hi, d_hi) = offs
        meet, i_lo, i_hi = _meet_chains(lo, hi)
        xs = _all_crossings(lo, hi)
        if len(xs) > 1:      # _meet_chains could pair two DIFFERENT crossings: the footprint would self-intersect
            meet, i_lo, i_hi = max(xs, key=lambda c: float(np.hypot(*c[0])))
            print(f"[dummy{k}] boundaries cross {len(xs)} times (r " + ", ".join(f"{np.hypot(*c[0]):.2f}" for c in xs)
                  + " mm): starting at the outermost")
        tip_kind = 'straight line'
        if meet is not None and i_lo >= 0 and i_hi >= 0:
            lo, hi = _start_at(lo, meet, i_lo), _start_at(hi, meet, i_hi)
            d_meet = float(_wall_distance(np.array([meet]), wall)[0])
            if d_meet > 0.0 and OPT.dummy_tip_width_mm > 0.0:
                lo, hi, d_cut = _blunt_tip(lo, hi, OPT.dummy_tip_width_mm)
                d_lo = float(_wall_distance(lo[:1], wall)[0])
                d_hi = float(_wall_distance(hi[:1], wall)[0])
                print(f"[dummy{k}] boundaries cross at r = {np.hypot(*meet):.2f} mm, az "
                      f"{np.degrees(np.arctan2(meet[1], meet[0])):.1f} deg, {d_meet:+.2f} mm OUTSIDE the housing: tip "
                      f"cut back {d_cut:.2f} mm to a {OPT.dummy_tip_width_mm:.1f} mm straight face at r "
                      f"{np.hypot(*lo[0]):.2f} / {np.hypot(*hi[0]):.2f} mm (housing distance {d_lo:+.2f} / "
                      f"{d_hi:+.2f} mm)")
            else:
                d_lo = d_hi = d_meet
                tip_kind = 'vertex'
                print(f"[dummy{k}] boundaries cross at r = {np.hypot(*meet):.2f} mm, az "
                      f"{np.degrees(np.arctan2(meet[1], meet[0])):.1f} deg (housing distance {d_lo:+.2f} mm): tip made "
                      f"a vertex there")
        a0 = float(np.arctan2(lo[-1][1], lo[-1][0]))
        a1 = float(np.arctan2(hi[-1][1], hi[-1][0]))
        out.append({'label': f'dummy{k}', 'gaps': (g_lo, g_hi), 'r_max': float(r_max), 'lo': lo, 'hi': hi,
                    'arc': _arc(r_max, a0, a1, ccw=True), 'arc_deg': float(np.degrees((a1 - a0) % (2.0 * np.pi))),
                    'tip': {'pts': np.vstack([hi[0], lo[0]]), 'kind': tip_kind, 'radius': np.inf, 'centre': None,
                            'span': float(np.hypot(*(lo[0] - hi[0]))), 'min_d': float(min(d_lo, d_hi))},
                    'depth': (d_lo, d_hi), 'voltage': 0.0})
    return out


def footprint_polygon(d: dict, arc_ds: float = 2.0) -> np.ndarray:
    """One closed footprint [mm]: lo (in -> out), the r_max arc, hi (out -> in), then the tip."""
    tip = d['tip']['pts']
    lo, hi = metal_edges(d)
    parts = [lo, d['arc'][1:-1], hi[::-1]]
    if len(tip) > 2:                       # an arc: it already runs hi -> lo
        parts.append(tip[1:-1])
    ring = _dedupe(np.vstack(parts))
    return ring[:-1] if np.hypot(*(ring[0] - ring[-1])) < 1e-9 else ring


def gap_edge(foot, dummy, gap: int, side: str):
    """(the boundary of one gap on one side, the footprint of the solid it belongs to, that solid's label).

    Gap 2j is dee j's `lo` edge and gap 2j+1 its `hi`; gap 2k+1 is dummy k's `lo` and gap 2k+2 its `hi`. So every
    gap has one dee side and one dummy side, facing each other across the gap."""
    n = 2 * len(foot)
    if side == 'dee':
        j, which = divmod(gap, 2)
        d = foot[j % len(foot)]
        return np.asarray(d['hi' if which else 'lo'], dtype=float), footprint_polygon(d), d['label']
    if side == 'dummy':
        if gap % 2:                       # odd gap: dummy (gap-1)/2, its lo edge
            k, key = (gap - 1) // 2, 'lo'
        else:                             # even gap: dummy gap/2 - 1, its hi edge
            k, key = (gap // 2 - 1) % (n // 2), 'hi'
        d = dummy[k % len(dummy)]
        return np.asarray(d[key], dtype=float), footprint_polygon(d), d['label']
    raise ConfigError(f"side must be 'dee' or 'dummy', got {side!r}")


def corner_post_radius(d: dict, wall_t: float, n: int = 4000) -> float:
    """Where a vertex-tipped dee's aperture tool must start so the corner post is at least `wall_t` thick.

    The metal left inside the tool radius is a wedge that pinches to nothing, so starting the tool at r_inner + wall_t
    would leave a sliver; this also requires the dee to be `wall_t` WIDE there. The width is taken on the uncut edges,
    and dee_width interpolates the angle linearly in r between chain vertices (on long single segments it reads the
    wedge narrower than it is): a longer corner post than the rule asks for, kept so a round nose changes only the nose.
    """
    rr, w = dee_width(d, n)
    r_in = (float(np.hypot(*np.asarray(d['tip']['pts']).T).min()) if d['tip']['kind'] == 'round'
            else float(np.hypot(*d['lo'][0])))
    ok = np.where(w >= wall_t)[0]
    r_wide = float(rr[ok[0]]) if len(ok) else float(rr[-1])
    return max(r_in + wall_t, r_wide)


# =====================================================================================================================
# 9. the star ground: one 2D layout
# =====================================================================================================================
# The grounded metal as ONE layout instead of four dummy wedges, each with tip rules that break whenever two gap lines
# converge: the disk r <= r_max minus every dee's KEEP-OUT - the dee, its two gaps (its edges offset by the local
# gap-law width, so the dummy edges are the old ones) and one gap width round its inner end. Dummy dees, the fill where
# converging ones meet and the ground round every dee tip follow from that; a pinched dummy is simply absent. Two dees
# closer than two gaps + star_min_ground_mm leave no ground between them and are refused, saying where.
def _star_gap_floor_mm() -> float:
    return float(OPT.star_tip_gap_mm) if OPT.star_tip_gap_mm > 0.0 else float(MACHINE.gap_width_min_m) * MM


def _extend_out(chain: np.ndarray, r_ext: float) -> np.ndarray:
    """An inner-to-outer chain carried on along its last segment and cut exactly on the r_ext circle."""
    p = _dedupe(chain)
    u = p[-1] - p[-2]
    u = u / np.linalg.norm(u)
    return _clip_outer(np.vstack([p, p[-1] + u * (r_ext - np.hypot(*p[-1]) + 40.0)]), r_ext)


def star_clash_check(foot, w0: float):
    """Every pair of dees must leave room for ground between their keep-outs: >= 2 gaps + star_min_ground_mm."""
    from scipy.spatial import cKDTree

    def dense(poly, ds=0.25):
        p = np.vstack([poly, poly[:1]])
        return np.vstack([a + np.linspace(0, 1, max(1, int(np.ceil(np.hypot(*(b - a)) / ds))), endpoint=False)[:, None]
                          * (b - a) for a, b in zip(p[:-1], p[1:])])
    pts = {d['label']: dense(footprint_polygon(d)) for d in foot}
    need = 2.0 * w0 + OPT.star_min_ground_mm
    bad = []
    names = sorted(pts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            dd, k = cKDTree(pts[b]).query(pts[a])
            m = int(np.argmin(dd))
            if dd[m] < need:
                p, q = pts[a][m], pts[b][k[m]]
                bad.append(f"{a} - {b}: {dd[m]:.2f} mm at r {np.hypot(*p):.1f} / {np.hypot(*q):.1f} mm, az "
                           f"{np.degrees(np.arctan2(p[1], p[0])):.1f} deg")
    if bad:
        raise ConfigError(f"star ground: dees closer than 2 gaps + {OPT.star_min_ground_mm:g} mm = {need:.1f} mm leave "
                         f"no ground between them - move an inner end out (geometry.dee_rmin):\n    "
                         + "\n    ".join(bad))


def snap_to_plate_corner(edge: np.ndarray, dees: DeeSystem, tol: float = PLATE_CORNER_SNAP_MM) -> np.ndarray:
    """Gap 0's dummy edge with its vertices near the turned exit-plate corner replaced by the corner itself.

    With gap 0 on the plate the edge runs ON the plate face and leaves the corner on the radial part's dummy edge, which
    passes through the corner by construction - but as the offset of a centre line with a jog there, the three offset
    lines meet only nearly in one point: two vertices a few hundredths of a mm apart, just above the fuzzy boolean
    tolerance, make a micro-edge and sliver triangles. Every vertex within `tol` of the corner becomes the corner, so
    the layout and the housing share it exactly."""
    C = rotate_points(np.asarray(exit_spec()['corner']['rotated_mm'][:2], dtype=float), si_rotation_deg(dees))
    e = np.asarray(edge, dtype=float)
    dist = np.hypot(*(e - C).T)
    near = np.flatnonzero(dist < tol)
    if not near.size:
        return e
    print(f"[SI rotation] gap-0 dummy edge: {near.size} vertex/vertices within {tol:g} mm of the plate corner "
          f"({', '.join(f'{v:.4f}' for v in dist[near])} mm) -> the corner ({C[0]:.3f}, {C[1]:.3f})")
    return np.vstack([e[:near[0]], C[None, :], e[near[-1] + 1:]])


def star_keepout(occ, d: dict, dees: DeeSystem, z: float, w0: float, r_ext: float):
    """One dee's keep-out as OCC faces at height z: the dee (carried past r_max), its two gap bands, a band one gap
    wide round its inner end and a disk on each inner corner, fused."""
    from matplotlib.path import Path as MPath
    from PyCentralRegion.gap_fields import offset_polyline

    lo, hi = _extend_out(d['lo'], r_ext), _extend_out(d['hi'], r_ext)
    faces = [_occ_loop(occ, footprint_edges(dict(d, lo=lo, hi=hi)), z)]
    lo, hi = metal_edges(dict(d, lo=lo, hi=hi))          # a round dee0 nose: the bands start at its tangent points
    g_lo, g_hi = d['gaps']
    w_lo = 2.0 * np.abs(_half_width_mm(dees.gaps[g_lo], lo, 1.0))
    w_hi = 2.0 * np.abs(_half_width_mm(dees.gaps[g_hi], hi, 1.0))
    # the gap bands: the lo edge's dummy side is to its right (CW), the hi edge's to its left (CCW)
    dummy_lo = offset_polyline(lo, -w_lo)
    if OPT.gap0_on_plate and g_lo == 0:
        dummy_lo = snap_to_plate_corner(dummy_lo, dees)
    faces.append(_occ_polygon(occ, np.vstack([offset_polyline(lo, +STAR_SEAM_MM), dummy_lo[::-1]]), z))
    faces.append(_occ_polygon(occ, np.vstack([offset_polyline(hi, -STAR_SEAM_MM),
                                              offset_polyline(hi, +w_hi)[::-1]]), z))
    tip = d['tip']
    if tip['kind'] == 'round':
        # an annular sector about the nose's centre with true arcs, its ends the radii through the tangent points
        # (where the two gap bands end), so the ground follows the round at w0
        C, R = np.asarray(tip['centre'], dtype=float), float(tip['radius'])
        a_lo = float(np.arctan2(*(lo[0] - C)[::-1]))
        a_hi = float(np.arctan2(*(hi[0] - C)[::-1]))
        c = occ.addPoint(float(C[0]), float(C[1]), z)
        q = [occ.addPoint(float(C[0] + r * np.cos(a)), float(C[1] + r * np.sin(a)), z) for r, a in
             ((R - STAR_SEAM_MM, a_hi), (R + w0, a_hi), (R + w0, a_lo), (R - STAR_SEAM_MM, a_lo))]
        ls = [occ.addLine(q[0], q[1]), occ.addCircleArc(q[1], c, q[2]), occ.addLine(q[2], q[3]),
              occ.addCircleArc(q[3], c, q[0])]
        faces.append(occ.addPlaneSurface([occ.addCurveLoop(ls)]))
    elif tip['kind'] == 'arc at origin':
        r_t = float(tip['radius'])
        a_lo = float(np.arctan2(d['lo'][0][1], d['lo'][0][0]))
        a_hi = float(np.arctan2(d['hi'][0][1], d['hi'][0][0]))
        c = occ.addPoint(0.0, 0.0, z)
        q = [occ.addPoint(r * np.cos(a), r * np.sin(a), z) for r, a in
             ((r_t + STAR_SEAM_MM, a_lo), (r_t + STAR_SEAM_MM, a_hi), (r_t - w0, a_hi), (r_t - w0, a_lo))]
        ls = [occ.addCircleArc(q[0], c, q[1]), occ.addLine(q[1], q[2]), occ.addCircleArc(q[2], c, q[3]),
              occ.addLine(q[3], q[0])]
        faces.append(occ.addPlaneSurface([occ.addCurveLoop(ls)]))
    elif len(tip['pts']) >= 2 and tip['kind'] != 'vertex':
        tp = _dedupe(tip['pts'])
        if len(tp) >= 2:
            inside = MPath(footprint_polygon(d)).contains_point
            mid = len(tp) // 2 if len(tp) > 2 else 0
            probe = offset_polyline(tp, +w0)
            pm = 0.5 * (probe[mid] + probe[min(mid + 1, len(tp) - 1)])
            s = -1.0 if inside(pm) else +1.0             # the side AWAY from the dee
            faces.append(_occ_polygon(occ, np.vstack([offset_polyline(tp, -s * STAR_SEAM_MM),
                                                      offset_polyline(tp, s * w0)[::-1]]), z))
    corners = [np.asarray(lo[0], dtype=float)]
    if np.hypot(*(np.asarray(hi[0]) - corners[0])) > 1e-6:
        corners.append(np.asarray(hi[0], dtype=float))
    for p in corners if tip['kind'] != 'round' else []:      # a round has no corners: its band covers them
        faces.append(occ.addDisk(float(p[0]), float(p[1]), z, w0, w0))
    out, _ = occ.fuse([(2, faces[0])], [(2, f) for f in faces[1:]], removeObject=True, removeTool=True)
    return [e for e in out if e[0] == 2]


def star_ground_layout(occ, foot, dees: DeeSystem, z: float, r_max: float, verbose: bool = True):
    """The ground's 2D layout at height z: the r_max disk minus every dee's keep-out (OCC faces)."""
    w0 = _star_gap_floor_mm()
    star_clash_check(foot, w0)
    keep = []
    for d in foot:
        keep += star_keepout(occ, d, dees, z, w0, r_max + STAR_R_EXT_MM)
    disk = occ.addDisk(0.0, 0.0, z, r_max, r_max)
    out, _ = occ.cut([(2, disk)], keep, removeObject=True, removeTool=True)
    faces = [e for e in out if e[0] == 2]
    occ.synchronize()
    if verbose:
        import gmsh
        areas = sorted((gmsh.model.occ.getMass(2, t) for _, t in faces), reverse=True)
        print(f"[star] ground layout: disk r {r_max:.0f} mm minus {len(foot)} dee keep-outs (gaps by the gap law, "
              f"{w0:.2f} mm round the inner ends) -> {len(faces)} face(s), area " + ", ".join(f"{a:.0f}" for a in areas)
              + " mm^2" + ("" if len(faces) == 1 else "  (more than one: ground islands)"))
    return faces


def gap0_face_polygon(d0: dict, cav0, wall: np.ndarray, g_max: float, overlap: float = GAP0_FACE_OVERLAP_MM,
                      ds: float = 2.0, verbose: bool = True):
    """The gap-0 ground face's outline [mm] (closed polygon), or None when the housing is nowhere within g_max.

    The beam channel is cut through every ground plate from r = 0, so inside it the only ground on gap 0's dummy side
    can be the inflector housing face, some mm behind the design edge: the median plane then sees gap 0 wider than
    designed. The face fills, full height, the strip between gap 0's dummy edge (dee0's gap-0 edge offset by the local
    gap-law width, the star keep-out's own edge) and the housing wherever the housing is within g_max of that edge,
    reaching `overlap` into the housing."""
    from matplotlib.path import Path as MPath
    from PyCentralRegion.gap_fields import offset_polyline
    lo = _densify(_dedupe(np.asarray(d0['lo'], dtype=float)), ds)[0]   # straight runs stay on their line
    w = 2.0 * np.abs(_half_width_mm(cav0, lo, 1.0))
    edge = offset_polyline(lo, -w)                # the dummy side of dee0's gap-0 edge, as star_keepout's band
    nrm = offset_polyline(lo, -(w + 1.0)) - edge  # unit normals away from dee0, towards the housing
    nrm /= np.hypot(*nrm.T)[:, None]
    # distance from the edge to the housing ALONG that normal (the face fills exactly that); inf = the ray misses it
    inside = MPath(np.asarray(wall, dtype=float)).contains_points
    t = np.arange(0.0, g_max + 0.05, 0.05)
    hit = inside((edge[:, None, :] + t[None, :, None] * nrm[:, None, :]).reshape(-1, 2)).reshape(len(edge), len(t))
    dist = np.where(hit.any(axis=1), t[np.argmax(hit, axis=1)], np.inf)
    keep = np.isfinite(dist)
    if not keep.any():
        return None
    runs = np.split(np.flatnonzero(keep), np.flatnonzero(np.diff(np.flatnonzero(keep)) > 1) + 1)
    run = max(runs, key=len)                      # the one stretch along the housing's flat face
    if len(run) < 2:
        return None
    inner = edge[run] + (dist[run] + overlap)[:, None] * nrm[run]
    if not inside(inner).all():
        raise RuntimeError("gap-0 face: an inner point is not inside the housing")
    if verbose:
        r = np.hypot(*edge[run].T)
        print(f"[gap0 face] ground fills gap 0's dummy side down to the housing (gap0_face_mm {g_max:g}): r "
              f"{r.min():.1f} .. {r.max():.1f} mm along the edge ({len(run)} points every {ds:g} mm), housing "
              f"{dist[run].min():.2f} .. {dist[run].max():.2f} mm behind the design edge (mean "
              f"{dist[run].mean():.2f}), design gap {w[run].min():.2f} .. {w[run].max():.2f} mm")
    return np.vstack([edge[run], inner[::-1]])


# =====================================================================================================================
# 10. OCC helpers
# =====================================================================================================================
def _aperture_ring(r0: float, r1: float, step=None, h=None):
    """(r, z) ring of the aperture profile between r0 and r1, a closed polygon in the x-z plane: the beam channel's
    HALF height, h[0]/2 inside step[0], h[1]/2 beyond step[1], linear between. Revolved about z it is the tool that
    hollows an electrode out."""
    step = OPT.aperture_radii() if step is None else step
    h = OPT.aperture_heights() if h is None else h
    rr = sorted(set([r0] + [v for v in step if r0 < v < r1] + [r1]))
    zz = [0.5 * channel_height_mm(v, step, h) for v in rr]
    return list(zip(rr, [-v for v in zz])) + list(zip(rr[::-1], zz[::-1]))


def _occ_polygon(occ, pts, z):
    """Closed polygon at height z -> plane surface tag."""
    p = [occ.addPoint(float(x), float(y), float(z)) for x, y in pts]
    ls = [occ.addLine(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
    return occ.addPlaneSurface([occ.addCurveLoop(ls)])


def _occ_profile_revolve(occ, ring):
    """Revolve an (r, z) ring a full turn about z; returns the volume tag."""
    p = [occ.addPoint(float(r), 0.0, float(z)) for r, z in ring]
    ls = [occ.addLine(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
    s = occ.addPlaneSurface([occ.addCurveLoop(ls)])
    v = [t for d, t in occ.revolve([(2, s)], 0, 0, 0, 0, 0, 1, 2.0 * np.pi) if d == 3]
    if len(v) != 1:
        raise RuntimeError(f"aperture tool revolve gave {len(v)} volumes")
    return v[0]


def _strip_orphans(gmsh, occ) -> int:
    """Delete every entity that is not part of some solid.

    ``revolve`` builds a tool from its sketch without consuming it and ``cut(..., removeTool=True)`` removes only the
    tool VOLUME, so every sketch (and the channel's construction face) would survive into the .step. Rather than track
    entities the booleans renumber, this asks which are reachable from a volume; the rest goes, high dimension first
    and non-recursively, so an edge still needed by a kept surface is never removed under it."""
    occ.synchronize()
    keep = set()
    for v in gmsh.model.getEntities(3):
        keep.add(v)
        for e in gmsh.model.getBoundary([v], combined=False, oriented=False, recursive=True):
            keep.add((e[0], abs(e[1])))
    dropped = 0
    for dim in (2, 1, 0):
        for ent in gmsh.model.getEntities(dim):
            if (ent[0], abs(ent[1])) in keep:
                continue
            try:
                occ.remove([ent], recursive=False)
                dropped += 1
            except Exception:                # still referenced by something that is kept
                pass
    occ.synchronize()
    return dropped


def footprint_edges(d: dict):
    """The footprint as REAL edges - ('poly', pts) / ('arc', p0, p1, centre) / ('spline', pts) - not a sampled
    polygon: every sampled vertex is a point gmsh has to honour, so sampled arcs mesh at the sampling, not at lc."""
    lo, hi = metal_edges(d)
    e = [('poly', lo), ('arc', lo[-1], hi[-1], np.zeros(2)), ('poly', hi[::-1])]   # the r_max arc about the origin
    tip = d['tip']
    if tip['kind'] == 'round':                         # dee0's nose: one arc, < 180 deg, facing the old vertex
        e.append(('arc', hi[0], lo[0], np.asarray(tip['centre'], dtype=float)))
    elif tip['kind'] == 'arc at origin':
        e.append(('arc', d['hi'][0], d['lo'][0], np.zeros(2)))
    elif tip['kind'] in ('straight line', 'arc', 'straight'):
        e.append(('poly', np.vstack([d['hi'][0], d['lo'][0]])))
    elif tip['kind'] == 'housing locus':
        e.append(('poly', np.asarray(tip['pts'], dtype=float)))       # hi[0] -> along the locus -> lo[0]
    # 'vertex': the two boundaries already meet, nothing to add
    return e


def _occ_loop(occ, edges, z: float):
    """A closed curve loop from footprint_edges at height z; returns the surface tag."""
    pt = {}

    def P(q):
        k = (round(float(q[0]), 9), round(float(q[1]), 9))
        if k not in pt:
            pt[k] = occ.addPoint(k[0], k[1], z)
        return pt[k]

    curves = []
    for e in edges:
        if e[0] == 'poly':
            q = np.asarray(e[1], dtype=float)
            for a, b in zip(q[:-1], q[1:]):
                if np.hypot(*(b - a)) > 1e-9:
                    curves.append(occ.addLine(P(a), P(b)))
        elif e[0] == 'arc':
            curves.append(occ.addCircleArc(P(e[1]), P(e[3]), P(e[2])))
        elif e[0] == 'spline':
            curves.append(occ.addSpline([P(v) for v in np.asarray(e[1], dtype=float)]))
    return occ.addPlaneSurface([occ.addCurveLoop(curves)])


def housing_edges(sim: dict):
    """The housing outline as its straight faces and splines, not hundreds of sampled lines."""
    nodes, out = sim['nodes'], []
    for ni, (kind, knots) in enumerate(sim['edges']):
        a, b = nodes[ni % len(nodes)], nodes[(ni + 1) % len(nodes)]
        out.append(('spline', np.vstack([a, knots, b])) if kind == 'spline' else ('poly', np.vstack([a, b])))
    return out


def _circle_fit(pts: np.ndarray):
    """(centre, radius, relative spread) of a ring of 3D points, or None for fewer than 12 (a polygon, not a circle)."""
    p = np.asarray(pts, dtype=float)
    if len(p) < 12:
        return None
    c = p.mean(axis=0)
    r = np.linalg.norm(p - c, axis=1)
    return c, float(r.mean()), float(r.std() / r.mean())


# =====================================================================================================================
# 11. hardware on the gap edges: posts, windows, round holes, collimator boxes
# =====================================================================================================================
def _post_corners(c, tangent, normal, size: float, length: float = None):
    """The post's plan outline: a square (geometry.post_shape 'square'), or a regular post_facets-gon of the same width.

    ``length`` (default ``size``) stretches the post ALONG the edge while its depth stays ``size``: a rectangle, or for
    the faceted shape a stadium (two half-polygons joined by straight sides).

    A square prism has four 90 deg edges running its full height; at a conducting wedge of exterior angle 3 pi / 2 the
    charge density diverges as r^(-1/3), which BEM cannot resolve, and those edges sit in the gap, where the field is
    strongest. A 16-gon's vertices are 202.5 deg exterior, r^(-0.11): the same blocking width, the singularity
    essentially gone - and the shape one would build at the dee voltage, since a sharp edge is where it sparks. A
    polygon rather than an OCC cylinder, so the build, the orbit-clearance test and the hit counting read one array.
    """
    c = np.asarray(c, dtype=float)
    length = float(size if length is None else length)
    if OPT.post_shape == 'square':
        return np.array([c + 0.5 * (a * length * tangent + b * size * normal)
                         for a, b in ((-1, -1), (+1, -1), (+1, +1), (-1, +1))])
    if abs(length - size) < 1e-9:
        a = np.linspace(0.0, 2.0 * np.pi, int(OPT.post_facets), endpoint=False)
        return np.array([c + 0.5 * size * (np.cos(k) * tangent + np.sin(k) * normal) for k in a])
    half = 0.5 * (length - size)
    m = int(OPT.post_facets) // 2
    end = []
    for sgn, k0 in ((+1.0, -0.5 * np.pi), (-1.0, 0.5 * np.pi)):       # +t end, then -t end
        for k in np.linspace(k0, k0 + np.pi, m + 1):
            end.append(c + sgn * half * tangent + 0.5 * size * (np.cos(k) * tangent + np.sin(k) * normal))
    return np.array(end)


def _edge_point(edge: np.ndarray, s: float):
    """(point, unit tangent, arclengths of the vertices) at arclength s along an edge polyline."""
    seg = np.diff(edge, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    sl = np.concatenate([[0.0], np.cumsum(seg_len)])
    k = int(np.clip(np.searchsorted(sl, s) - 1, 0, len(seg) - 1))
    t = seg[k] / seg_len[k]
    return edge[k] + (s - sl[k]) * t, t, sl


def _into(poly, p, t, what: str, owner: str):
    """The edge normal at p that points INTO the solid whose footprint is poly."""
    from matplotlib.path import Path as MplPath
    inside = MplPath(poly)
    n = np.array([-t[1], t[0]])
    if not inside.contains_point(p + 0.25 * n):
        n = -n
    if not inside.contains_point(p + 0.25 * n):
        raise RuntimeError(f"{what}: neither side of the edge is inside {owner}")
    return n


def _channel_top(corners, owner: str, aperture) -> float:
    """The highest half height [mm] of ``owner``'s channel over a piece of hardware's plan-view footprint (its corners
    [mm]). Posts, window blocks and hole plates run ``post_overhang_mm`` past this: inside the taper the channel
    changes across a footprint, and a block as high as the channel at one point of it would leave the channel open
    over its higher end. The profile is monotonic in r, so the highest is at the footprint's nearest or farthest
    radius (the machine axis never lies inside hardware)."""
    p = np.asarray(corners, dtype=float)
    ab = np.roll(p, -1, axis=0) - p
    t = np.clip(-np.einsum('ij,ij->i', p, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-30), 0.0, 1.0)
    r_near, r_far = float(np.hypot(*(p + t[:, None] * ab).T).min()), float(np.hypot(p[:, 0], p[:, 1]).max())
    return max(channel_half_height(r_near, owner, aperture), channel_half_height(r_far, owner, aperture))


def _post_on_edge(edge, poly, s: float, aperture, size: float, offset: float, overhang: float, name: str,
                  owner: str, length: float = None) -> dict:
    """One post on `edge` at arclength `s`, its centre `offset` into the solid whose footprint is `poly` - the same
    routine serves a dee edge and the dummy edge facing it, and the two come out pointing at each other across the
    gap. ``length`` (default ``size``) is its extent along the edge, centred on `s`; ``aperture`` the channel heights
    of the owner (None: ``channel_heights(owner)``). It runs `overhang` past the channel (``_channel_top``)."""
    p, t, sl = _edge_point(edge, s)
    if not (0.0 <= s <= sl[-1]):
        raise ConfigError(f"post {name}: arclength {s:.2f} mm is off the edge (0 .. {sl[-1]:.2f})")
    n = _into(poly, p, t, f"post {name}", owner)
    c = p + offset * n
    length = float(size if length is None else length)
    corners = _post_corners(c, t, n, size, length)
    return {'name': name, 'owner': owner, 's': float(s), 'point': p, 'tangent': t, 'normal': n, 'centre': c,
            'corners': corners, 'half_height': _channel_top(corners, owner, aperture) + overhang,
            'length': length, 'r': float(np.hypot(*p)), 'r_centre': float(np.hypot(*c))}


def edge_crossings(edge: np.ndarray, path: np.ndarray, first_only: bool = True):
    """Where a particle path crosses an edge polyline: (path index, arclength [mm], crossing point, z or None).

    EXACT segment-segment intersection. The shortcut - the sign of the cross product against the NEAREST edge segment
    - reports crossings that never happened: far from the edge the nearest segment changes as the particle orbits and
    the sign flips with it, which misplaces hardware by many mm. `path` is (N, 2) or (N, 3); x-y is used and z
    interpolated. Returns the crossings in path order, or just the first."""
    e = np.asarray(edge, dtype=float)[:, :2]
    p = np.asarray(path, dtype=float)[:, :2]
    seg = np.diff(e, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    sl = np.concatenate([[0.0], np.cumsum(seg_len)])
    a, r = p[:-1], np.diff(p, axis=0)
    out = []
    for m in range(len(seg)):
        c, sv = e[m], seg[m]
        den = r[:, 0] * sv[1] - r[:, 1] * sv[0]
        ok = np.abs(den) > 1e-12
        if not ok.any():
            continue
        d = c - a
        t = np.full(len(a), np.nan)
        u = np.full(len(a), np.nan)
        t[ok] = (d[ok, 0] * sv[1] - d[ok, 1] * sv[0]) / den[ok]
        u[ok] = (d[ok, 0] * r[ok, 1] - d[ok, 1] * r[ok, 0]) / den[ok]
        hit = ok & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
        for k in np.nonzero(hit)[0]:
            z = None
            if np.asarray(path).shape[1] > 2:
                z = float(path[k][2] + t[k] * (path[k + 1][2] - path[k][2]))
            out.append((int(k), float(sl[m] + u[k] * seg_len[m]), a[k] + t[k] * r[k], z))
    out.sort(key=lambda v: v[0])
    if first_only:
        return out[0] if out else None
    return out


def gap_window(foot, dummy, gap: int, side: str, s_lo: float, s_hi: float, z_lo: float, z_hi: float,
               aperture=None, depth: float = None, overhang: float = None, name: str = None,
               where: str = None) -> list:
    """A vertical window on one gap edge: two blocks leaving an opening from z_lo to z_hi (the OPENING: a 26 mm
    window symmetric about the median plane is (-13, +13), WindowSpec's half_height_mm 13).

    Built like a post - near face on the edge, `depth` into the electrode, fused into the solid that owns the edge -
    but spanning an arclength RANGE and split into an upper and a lower block, each running from the opening to
    `overhang` past the channel's highest point over the window (``_channel_top``; solid metal already, so the fuse has
    real overlap instead of a coincident face). The opening must lie inside the channel at the window's middle
    (|z_center| + half height <= the channel's half height there): a ConfigError names ``where`` (the config item)
    otherwise."""
    depth = OPT.post_size_mm if depth is None else depth
    overhang = OPT.post_overhang_mm if overhang is None else overhang
    where = where or f"the gap-{gap} {side} window"
    edge, poly, owner = gap_edge(foot, dummy, int(gap), side)
    aperture = channel_heights(owner) if aperture is None else aperture
    p_lo, t_lo, sl = _edge_point(edge, s_lo)
    p_hi, t_hi, _ = _edge_point(edge, s_hi)
    if not (0.0 <= s_lo < s_hi <= sl[-1]):
        raise ConfigError(f"{where}: arclengths {s_lo:.2f}..{s_hi:.2f} mm are off the gap-{gap} {side} edge "
                          f"(0 .. {sl[-1]:.2f})")
    n = _into(poly, p_lo, t_lo, f"gap-{gap} window", owner)
    # the plan footprint FOLLOWS the edge (its vertices in between included, each offset along its own bisector
    # normal): a chord between the ends leaves the block's face off the edge wherever the edge kinks inside the window
    # (the gap law's floor, a node), beyond the fuzzy boolean tolerance, and the union can then silently drop the block
    seg = np.diff(edge, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    pts, nrm = [p_lo], [n]
    for k in range(1, len(sl) - 1):
        if s_lo + 1e-6 < sl[k] < s_hi - 1e-6:
            tb = seg[k - 1] / seg_len[k - 1] + seg[k] / seg_len[k]          # bisector of the two segments
            nb = np.array([-tb[1], tb[0]]) / np.linalg.norm(tb)
            pts.append(edge[k][:2])
            nrm.append(nb if nb @ n > 0 else -nb)
    pts.append(p_hi)
    n_hi = np.array([-t_hi[1], t_hi[0]])
    nrm.append(n_hi if n_hi @ n > 0 else -n_hi)
    corners = np.array(pts + [p + depth * nn for p, nn in zip(pts[::-1], nrm[::-1])])
    mid = 0.5 * (p_lo + p_hi)
    r_mid = float(np.hypot(*mid))
    ap_half = channel_half_height(r_mid, owner, aperture)
    top = _channel_top(corners, owner, aperture) + overhang
    if not (float(z_hi) <= ap_half + 1e-9 and float(z_lo) >= -ap_half - 1e-9 and float(z_hi) > float(z_lo)):
        raise ConfigError(f"{where}: the opening {float(z_lo):+.3f} .. {float(z_hi):+.3f} mm must lie inside the channel "
                          f"at the window, |z_center_mm| + half_height_mm <= {ap_half:.3f} mm (half the channel height "
                          f"on {owner} at r = {r_mid:.1f} mm)")
    out = []
    base = name or f"g{int(gap)}_{side}_win"
    for tag, lo, hi in (('up', float(z_hi), top), ('dn', -top, float(z_lo))):
        out.append({'name': f"{base}_{tag}", 'owner': owner, 'kind': 'window', 'corners': corners,
                    'z_lo': lo, 'z_hi': hi, 's_lo': float(s_lo), 's_hi': float(s_hi), 'r_mid': r_mid,
                    'opening': (float(z_lo), float(z_hi)), 'ap_half': ap_half, 'centre': mid + 0.5 * depth * n,
                    'half_height': 0.5 * (hi - lo), 'z_mid': 0.5 * (hi + lo), 'point': mid, 'r': r_mid,
                    'r_centre': float(np.hypot(*(mid + 0.5 * depth * n))), 's': 0.5 * (float(s_lo) + float(s_hi))})
    return out


def gap_aperture(foot, dummy, gap: int, side: str, r_hole: float, s_c: float, z_c: float, s_lo: float, s_hi: float,
                 aperture=None, depth: float = None, overhang: float = None, name: str = None) -> dict:
    """A CIRCULAR aperture on one gap edge: a plate the beam passes through by a round hole.

    The plate's near face is the chord from s_lo to s_hi on the edge, `depth` into the electrode that owns the edge,
    the whole channel height plus `overhang` each way (``_channel_top``); a cylinder of radius `r_hole` is cut through
    it along the edge normal, centred on the face at arclength s_c and height z_c [mm]."""
    depth = OPT.post_size_mm if depth is None else depth
    overhang = OPT.post_overhang_mm if overhang is None else overhang
    edge, poly, owner = gap_edge(foot, dummy, int(gap), side)
    aperture = channel_heights(owner) if aperture is None else aperture
    p_lo, _, sl = _edge_point(edge, s_lo)
    p_hi, _, _ = _edge_point(edge, s_hi)
    p_c, t_c, _ = _edge_point(edge, s_c)
    if not (0.0 <= s_lo < s_c < s_hi <= sl[-1]):
        raise ConfigError(f"gap-{gap} hole: need 0 <= s_lo < s_c < s_hi <= {sl[-1]:.2f} mm, got "
                          f"{s_lo:.2f} / {s_c:.2f} / {s_hi:.2f}")
    if not (s_lo <= s_c - r_hole and s_c + r_hole <= s_hi):
        raise ConfigError(f"gap-{gap} hole: the R {r_hole:g} mm hole runs past the plate "
                          f"({s_c - r_hole:.2f}..{s_c + r_hole:.2f} vs {s_lo:.2f}..{s_hi:.2f})")
    n = _into(poly, p_c, t_c, f"gap-{gap} aperture", owner)
    tt = (p_hi - p_lo) / np.linalg.norm(p_hi - p_lo)
    p_face = p_lo + float(np.dot(p_c - p_lo, tt)) * tt     # the hole centre ON the plate face (the chord)
    r_mid = float(np.hypot(*p_c))
    ap_half = channel_half_height(r_mid, owner, aperture)
    corners = np.array([p_lo, p_hi, p_hi + depth * n, p_lo + depth * n])
    h = _channel_top(corners, owner, aperture) + overhang
    if abs(z_c) + r_hole >= h:
        raise ConfigError(f"gap-{gap} hole: the hole (z {z_c:+.1f} +- {r_hole:g}) reaches past the plate "
                          f"(+-{h:.1f} mm)")
    return {'name': name or f"g{int(gap)}_{side}_ap", 'owner': owner, 'kind': 'aperture',
            'corners': corners, 'z_lo': -h, 'z_hi': h,
            'half_height': h, 'hole_centre': np.array([p_face[0], p_face[1], float(z_c)]), 'hole_axis': n,
            'hole_r': float(r_hole), 'depth': float(depth), 's_lo': float(s_lo), 's_hi': float(s_hi), 's': float(s_c),
            'r_mid': r_mid, 'r': r_mid, 'ap_half': ap_half, 'point': p_c, 'centre': p_c + 0.5 * depth * n,
            'r_centre': float(np.hypot(*(p_c + 0.5 * depth * n)))}


def gap_posts(foot, dummy, spec, aperture=None, size: float = None, offset: float = None,
              overhang: float = None) -> list:
    """Posts on gap edges. `spec`: PostSpec items or (gap, side, s[, length]) tuples. Both sides of one gap at the
    same arclength make a facing pair - a jaw defining the outer edge of that gap's opening. ``aperture`` None: each
    post's own electrode's channel heights."""
    from collections import Counter
    size = OPT.post_size_mm if size is None else size
    offset = OPT.post_offset_mm if offset is None else offset
    overhang = OPT.post_overhang_mm if overhang is None else overhang
    out = []
    for item in spec:
        gap, side, s, length = ((item.gap, item.side, item.s_mm, item.length_mm or None) if hasattr(item, 'gap')
                                else (*item[:3], item[3] if len(item) > 3 else None))
        edge, poly, owner = gap_edge(foot, dummy, int(gap), side)
        out.append(_post_on_edge(edge, poly, float(s), aperture, size, offset, overhang, f"g{int(gap)}_{side}", owner,
                                 length=None if length is None else float(length)))
    # an index only where names collide, numbered by arclength: 1 is always the innermost of a group
    for base, k in Counter(q['name'] for q in out).items():
        if k > 1:
            for i, q in enumerate(sorted((q for q in out if q['name'] == base), key=lambda q: q['s']), start=1):
                q['name'] = f"{base}_{i}"
    return out


def gap0_posts(foot, s_inner: float, s_outer: float, aperture=None, size: float = None, offset: float = None,
               overhang: float = None) -> list:
    """Two posts on dee0's gap-0 edge at arclengths s_inner / s_outer, one either side of the beam's first crossing:
    the slot between them is the gap's defined opening, the rest of the aperture plugged. Arclength, not radius,
    because the edge is not radial."""
    aperture = channel_heights('dee0') if aperture is None else aperture
    size = OPT.post_size_mm if size is None else size
    offset = OPT.post_offset_mm if offset is None else offset
    overhang = OPT.post_overhang_mm if overhang is None else overhang
    d0 = foot[0]
    edge = np.asarray(d0['lo'], dtype=float)
    poly = footprint_polygon(d0)
    posts = []
    for name, s in (('inner', float(s_inner)), ('outer', float(s_outer))):
        q = _post_on_edge(edge, poly, s, aperture, size, offset, overhang, f'post_{name}', d0['label'])
        q.pop('length')
        posts.append(q)
    return posts


def extra_boxes(spec, aperture=None, overhang: float = None):
    """BoxSpec items -> full-height grounded box items for build_model (owner 'ground'), running `overhang` into the
    plates over the box's whole radius range (of the dees' or the dummy dees' channel, whichever is taller there)."""
    overhang = OPT.post_overhang_mm if overhang is None else overhang
    heights = [aperture] if aperture is not None else [channel_heights('dee0'), channel_heights('ground')]
    out = []
    for k, b in enumerate(spec):
        xa, xb, ya, yb = float(b.x_min_mm), float(b.x_max_mm), float(b.y_min_mm), float(b.y_max_mm)
        r_lo = float(np.hypot(max(xa, 0.0, -xb), max(ya, 0.0, -yb)))
        r_hi = float(max(np.hypot(x, y) for x in (xa, xb) for y in (ya, yb)))
        h = max(channel_half_height(r, heights=hh) for r in np.linspace(r_lo, r_hi, 64) for hh in heights) + overhang
        c = np.array([0.5 * (xa + xb), 0.5 * (ya + yb)])
        out.append({'name': f'colbox{k}', 'owner': 'ground', 'kind': 'box', 's': 0.0, 'point': c,
                    'corners': np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]]), 'half_height': h,
                    'centre': c, 'r': r_lo, 'r_centre': float(np.hypot(*c)), 'length': xb - xa})
    return out


def hardware(foot, dummy, verbose: bool = True) -> list:
    """Every post, window block, hole plate and box the [geometry] options ask for, in build order.

    A window ALWAYS spans post centre to post centre when its edge carries two or more posts: a cylindrical post's
    footprint tapers to nothing at its edge, so a window that stops there leaves an uncovered wedge. Snapping here,
    rather than trusting the spec, means bounds written under an edge-to-edge convention cannot survive silently."""
    posts = gap_posts(foot, dummy, OPT.posts) if OPT.posts else []
    centres = {}
    for p in OPT.posts:
        centres.setdefault((int(p.gap), p.side), []).append(float(p.s_mm))
    if OPT.gap0_posts_mm:
        posts += gap0_posts(foot, *OPT.gap0_posts_mm)
        centres.setdefault((0, 'dee'), []).extend(float(s) for s in OPT.gap0_posts_mm)
    snapped = []
    for i, w in enumerate(OPT.windows):
        s_lo, s_hi = float(w.s_lo_mm), float(w.s_hi_mm)
        c = centres.get((int(w.gap), w.side))
        if c and len(c) >= 2:
            lo, hi = min(c), max(c)
            if abs(lo - s_lo) > 1e-6 or abs(hi - s_hi) > 1e-6:
                snapped.append(f"    gap {w.gap} {w.side:<5} {s_lo:7.2f}..{s_hi:7.2f} -> {lo:7.2f}..{hi:7.2f} mm "
                               f"({s_lo - lo:+.2f} / {s_hi - hi:+.2f})")
            s_lo, s_hi = lo, hi
        elif c and verbose:
            print(f"  [window] gap {w.gap} {w.side}: only one post on this edge, bounds left as given "
                  f"({s_lo:.2f}..{s_hi:.2f} mm)")
        posts += gap_window(foot, dummy, int(w.gap), w.side, s_lo, s_hi, w.z_lo_mm, w.z_hi_mm,
                            where=f"geometry.windows.{i} (gap {w.gap} {w.side}, half_height_mm {w.half_height_mm:g})")
    if snapped and verbose:
        print(f"\n  [window] snapped {len(snapped)} window(s) to their post centres:")
        print("\n".join(snapped))
    for h in OPT.holes:
        posts.append(gap_aperture(foot, dummy, int(h.gap), h.side, float(h.r_mm), float(h.s_mm), float(h.z_mm),
                                  float(h.s_lo_mm), float(h.s_hi_mm)))
    return posts + extra_boxes(OPT.boxes)


def join_posts_to_end_walls(posts, rings: dict, polys: dict, verbose: bool = True) -> list:
    """Carry every plain post end that stops short of, or straddles, a hug dee's end wall into that wall.

    A post end landing a fraction of a mm off the end wall, at an angle to it, leaves a wedge of vacuum between two
    pieces of the SAME electrode, meshed with needle triangles. A post end within geometry.post_wall_join_mm of the
    wall, or only partly inside it, grows along its edge (its outer end stays put) until it sits POST_WALL_SINK_MM deep
    everywhere. The crevice it fills has one potential on both sides, so it is field-free: the field does not change.
    ``rings``: per dee label, the end wall's open-side boundary (the housing offset by clearance + wall); ``polys``:
    the dees' footprints - a grown post must stay inside its dee. Updates the posts in place; returns the joined names.
    """
    from matplotlib.path import Path as MplPath
    joined = []
    if OPT.post_wall_join_mm <= 0.0:
        return joined

    def end_face(c, t, n, size, length, sgn):
        """(the post's end on the sgn*t side, its corners): a square post's straight end, a faceted one's cap tip."""
        corners = _post_corners(c, t, n, size, length)
        ol = _densify(np.vstack([corners, corners[:1]]), 0.1)[0]
        u = sgn * (ol - c) @ t
        tip = 1e-6 if OPT.post_shape == 'square' else 0.067 * size      # faceted: the cap within +-30 deg of its axis
        return ol[u >= 0.5 * length - tip], corners

    for q in posts:
        ring = rings.get(q.get('owner'))
        if ring is None or q.get('kind') is not None or 'z_lo' in q:
            continue                                   # plain posts on hug dees only
        t, n = np.asarray(q['tangent'], dtype=float), np.asarray(q['normal'], dtype=float)
        c0, L0 = np.asarray(q['centre'], dtype=float), float(q['length'])
        size = float(np.ptp((np.asarray(q['corners'], dtype=float) - c0) @ n))
        # the end facing the wall: the one reaching deeper into it (depth = -signed distance from the ring, + inside)
        depth = {s: -_wall_distance(end_face(c0, t, n, size, L0, s)[0], ring) for s in (+1.0, -1.0)}
        sgn = max(depth, key=lambda s: depth[s].max())
        d = depth[sgn]
        if d.min() >= POST_WALL_SINK_MM or d.max() < -OPT.post_wall_join_mm:
            continue                                   # already well inside the wall, or well clear of it
        for e in np.arange(0.05, OPT.post_wall_join_mm + size + 1e-9, 0.05):
            c, L = c0 + sgn * 0.5 * e * t, L0 + e
            face, corners = end_face(c, t, n, size, L, sgn)
            if (-_wall_distance(face, ring)).min() >= POST_WALL_SINK_MM:
                break
        else:
            raise RuntimeError(f"{q['name']}: its end is {-d.max():.2f} .. {-d.min():.2f} mm off {q['owner']}'s end "
                               f"wall and growing it {OPT.post_wall_join_mm + size:.1f} mm does not sink it in")
        if not MplPath(polys[q['owner']]).contains_points(c + 0.98 * (corners - c)).all():
            raise RuntimeError(f"{q['name']}: carried {e:.2f} mm into {q['owner']}'s end wall it would leave the dee "
                               f"(into the clearance to the housing) - move the post instead")
        q.update(corners=corners, centre=c, length=float(L), r_centre=float(np.hypot(*c)), wall_join_mm=float(e))
        joined.append(q['name'])
        if verbose:
            print(f"[posts] {q['name']} on {q['owner']}: its {'outer' if sgn > 0 else 'inner'} end was "
                  f"{max(-d.max(), 0.0):.2f} .. {-d.min():.2f} mm off the end wall -> carried {e:.2f} mm along the "
                  f"edge into it ({POST_WALL_SINK_MM:g} mm deep everywhere); length {L0:g} -> {L:.2f} mm")
    return joined


# =====================================================================================================================
# 12. the model in gmsh: footprints -> prisms -> hollowed -> the ground with the housing -> mesh sizes
# =====================================================================================================================
def build_model(foot, dummy, sim, brick, dees: DeeSystem, posts=None, verbose: bool = True, half: float = None,
                wall_t: float = None, r_cut: float = None, lc: float = None, aperture=None, dummy_aperture=None,
                lc_posts: float = None, lc_outer: float = None, r_fine: float = None, r_coarse: float = None):
    """The whole central region as OCC solids, hollowed and fused, with the mesh size fields set. Leaves gmsh
    initialised; returns (gmsh, groups {label: volume tags}, aperture-tool radii, footprints).

    Ground 'star': built from its 2D layout (star_ground_layout) - extruded, hollowed by the channel outside the
    housing, the inflector housing merged in, its exit opening cut (a cylinder when it is round); the dummy footprints
    are not built and posts on dummy edges go into the ground. Ground 'dummies': the dummy wedges fused onto the
    housing, the beam channel cut out of the housing first.

    The channel heights are geometry's (``aperture`` / ``dummy_aperture`` override them as (inner, outer) pairs, on
    geometry's taper radii). The dummy dees' own heights make an asymmetric gap, a vertical lens (the wider side has
    the longer fringe field), and change how well each electrode shields its own midplane, so the accelerating voltage
    the beam sees. A graded mesh (lc_outer > lc) keeps lc inside r_fine and lc_outer beyond r_coarse: the dense BEM of
    a whole machine is otherwise too large for the GPU.
    """
    import gmsh
    half = OPT.half_height_mm if half is None else half
    wall_t = OPT.dee_wall_mm if wall_t is None else wall_t
    r_cut = r_max_mm() if r_cut is None else r_cut
    lc = OPT.lc_mm if lc is None else lc
    dummy_ap = tuple(dummy_aperture or (OPT.dummy_aperture_heights() if aperture is None else aperture))
    aperture = tuple(OPT.aperture_heights() if aperture is None else aperture)
    lc_posts = OPT.lc_posts_mm if lc_posts is None else lc_posts
    lc_outer = OPT.lc_outer_mm if lc_outer is None else lc_outer
    r_fine = OPT.r_fine_mm if r_fine is None else r_fine
    r_coarse = OPT.r_coarse_mm if r_coarse is None else r_coarse

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("central_region")
    occ = gmsh.model.occ
    t0 = time.time()
    # near-coincident features (a post end or a dee end within a fraction of a mm of a taper kink circle) leave
    # micro-faces in the booleans; the fuzzy tolerance merges edges closer than this instead. 0.05 mm removes every
    # edge under 0.1 mm and moves no metal; much larger values fail to mesh or leave a dummy unhollowed.
    if OPT.occ_fuzzy_mm > 0.0:
        gmsh.option.setNumber("Geometry.ToleranceBoolean", float(OPT.occ_fuzzy_mm))
        if verbose:
            print(f"[build] OCC fuzzy boolean tolerance {OPT.occ_fuzzy_mm:g} mm (occ_fuzzy_mm)")

    def say(m):
        if verbose:
            print(f"[build] {m}  ({time.time() - t0:.1f} s)")

    star = OPT.ground == 'star'
    # ---- 1. the housing: its outline extruded, the beam channel cut out ---------------------------------------------
    hs = _occ_loop(occ, housing_edges(sim), -half)
    house = [t for d, t in occ.extrude([(2, hs)], 0, 0, 2.0 * half) if d == 3][0]
    if brick is not None and star:
        say("housing solid (star: merged into the ground after the channel cut, exit opening cut after that)")
    elif brick is not None:
        ch_s = _occ_polygon(occ, brick['near'][:, :2], 0.0)
        occ.remove([(2, ch_s)], recursive=True)
        # the prism starts `back` mm upstream of the emitting plane (see beam_channel); any polygon (a round exit
        # opening is a ring of points)
        p = [occ.addPoint(*q) for q in brick['start']]
        ls = [occ.addLine(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
        face = occ.addPlaneSurface([occ.addCurveLoop(ls)])
        d = (brick['extend'] + brick['back']) * brick['sweep']
        chan = [t for dd, t in occ.extrude([(2, face)], *d) if dd == 3][0]
        out, _ = occ.cut([(3, house)], [(3, chan)], removeObject=True, removeTool=True)
        house = [t for dd, t in out if dd == 3][0]
        say("housing solid with the beam recess")

    # ---- 2. the footprints, extruded ----------------------------------------------------------------------------------
    solids, polys = {}, {}
    for d in list(foot) + ([] if star else list(dummy)):
        polys[d['label']] = footprint_polygon(d)
        s = _occ_loop(occ, footprint_edges(d), -half)
        solids[d['label']] = [t for dd, t in occ.extrude([(2, s)], 0, 0, 2.0 * half) if dd == 3][0]
    say(f"{len(solids)} prisms extruded over z = {-half:+.0f} .. {half:+.0f} mm")

    # ---- 3. hollow each one with its own aperture tool ----------------------------------------------------------------
    r_tool = {}
    hug = {d['label'] for d in foot if d['tip'].get('kind') == 'housing locus'}
    for d in foot:
        r_tool[d['label']] = (corner_post_radius(d, wall_t) if d['label'] == 'dee0' or d['tip'].get('pinched')
                              else 0.0 if d['label'] in hug     # the slot all the way in; end wall below
                              else float(np.hypot(*d['lo'][0])) + wall_t)
    for d in dummy:
        r_tool[d['label']] = 0.0          # all the way to the axis: splits it into two plates
    # a hug dee's end wall is the part of the dee within clearance + wall_t of the housing, full height, cut from a
    # copy of the prism BEFORE the slot and fused back after it - so it follows the housing, not a circle
    end_wall, rings = {}, {}
    if hug:
        wall_xy = np.asarray(sim['curve'], dtype=float)
        for d in foot:
            if d['label'] not in hug:
                continue
            ring = rings[d['label']] = housing_offset_polygon(wall_xy, float(d['clearance']) + float(wall_t),
                                                              n=OPT.hug_ring_n)
            off_s = _occ_polygon(occ, ring, -half)
            off = [t for dd, t in occ.extrude([(2, off_s)], 0, 0, 2.0 * half) if dd == 3][0]
            band, _ = occ.intersect(occ.copy([(3, solids[d['label']])]), [(3, off)], removeObject=True,
                                    removeTool=True)
            end_wall[d['label']] = [t for dd, t in band if dd == 3]
        say("dee end walls along the housing (" + ", ".join(f"{k}: {len(v)}" for k, v in sorted(end_wall.items()))
            + ")")
    pieces = {}
    dummy_labels = {d['label'] for d in dummy}
    if dummy_ap != aperture:
        say(f"asymmetric gap: dees {aperture} mm, dummy dees {dummy_ap} mm")
    for label, vol in solids.items():
        tool = _occ_profile_revolve(occ, _aperture_ring(r_tool[label], r_cut + 20.0,
                                                        h=dummy_ap if label in dummy_labels else aperture))
        out, _ = occ.cut([(3, vol)], [(3, tool)], removeObject=True, removeTool=True)
        pieces[label] = [t for dd, t in out if dd == 3]
        if end_wall.get(label):
            out, _ = occ.fuse([(3, t) for t in pieces[label]], [(3, t) for t in end_wall[label]],
                              removeObject=True, removeTool=True)
            pieces[label] = [t for dd, t in out if dd == 3]
    say("hollowed: " + ", ".join(f"{k} -> {len(v)}" for k, v in sorted(pieces.items()))
        + (f"  (end walls along the housing: {', '.join(sorted(hug))})" if hug else ""))

    # ---- 3a. star: the ground from its 2D layout ----------------------------------------------------------------------
    # layout -> extrusion -> minus the channel (the dummies' full-radius aperture profile; the housing keeps its full
    # height because it is merged in AFTER) -> the inflector housing merged in -> its exit opening cut
    if star:
        gl = star_ground_layout(occ, foot, dees, -half, float(foot[0]['r_max']), verbose=verbose)
        gp = [e for e in occ.extrude(gl, 0, 0, 2.0 * half) if e[0] == 3]
        tool = _occ_profile_revolve(occ, _aperture_ring(0.0, r_cut + 20.0, h=dummy_ap))
        out, _ = occ.cut(gp, [(3, tool)], removeObject=True, removeTool=True)
        gnd = [e for e in out if e[0] == 3]
        n_plates = len(gnd)
        out, _ = occ.fuse(gnd, [(3, house)], removeObject=True, removeTool=True)
        gnd = [e for e in out if e[0] == 3]
        if OPT.gap0_face_mm > 0.0:                # the gap-0 ground face, before the exit opening is cut through it
            d0 = next(d for d in foot if d['label'] == 'dee0')
            poly = gap0_face_polygon(d0, dees.gaps[d0['gaps'][0]], np.asarray(sim['curve'], dtype=float),
                                     OPT.gap0_face_mm, verbose=verbose)
            if poly is None:
                say(f"gap-0 face: the housing is nowhere within {OPT.gap0_face_mm:g} mm of gap 0's dummy edge - "
                    f"none built")
            else:
                from scipy.spatial import cKDTree
                w0 = _star_gap_floor_mm()
                pq = _densify(np.vstack([poly, poly[:1]]), 0.25)[0]
                for d in foot:                    # it must not reach into any dee's keep-out
                    fp = footprint_polygon(d)
                    dd, _ = cKDTree(_densify(np.vstack([fp, fp[:1]]), 0.25)[0]).query(pq)
                    if dd.min() < w0 - 0.3:       # dee0's own gap edge sits exactly one gap width away
                        raise RuntimeError(f"gap-0 face comes within {dd.min():.2f} mm of {d['label']} (< {w0:.2f})")
                    if verbose and d['label'] == 'dee0':
                        print(f"[gap0 face] nearest dee0 metal {dd.min():.2f} mm (gap floor {w0:.2f})")
                fv = [e for e in occ.extrude([(2, _occ_polygon(occ, poly, -half))], 0, 0, 2.0 * half) if e[0] == 3]
                out, _ = occ.fuse(gnd, fv, removeObject=True, removeTool=True)
                gnd = [e for e in out if e[0] == 3]
                say(f"gap-0 face fused into the ground (full height) -> {len(gnd)} solid(s)")
        how = "no exit opening"
        if brick is not None:
            d = (brick['extend'] + brick['back']) * brick['sweep']
            fit = _circle_fit(brick['start'])
            if fit is not None and fit[2] < 0.01:
                c, r_ap, _ = fit
                cut_tool = occ.addCylinder(float(c[0]), float(c[1]), float(c[2]), float(d[0]), float(d[1]),
                                           float(d[2]), r_ap)
                how = f"exit opening cut as a cylinder R {r_ap:.2f} mm"
            else:
                p = [occ.addPoint(*q) for q in brick['start']]
                ls = [occ.addLine(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
                face = occ.addPlaneSurface([occ.addCurveLoop(ls)])
                cut_tool = [t for dd, t in occ.extrude([(2, face)], *d) if dd == 3][0]
                how = f"exit opening cut as its {len(p)}-point polygon"
            out, _ = occ.cut(gnd, [(3, cut_tool)], removeObject=True, removeTool=True)
            gnd = [e for e in out if e[0] == 3]
        pieces['ground'] = [t for _, t in gnd]
        say(f"star ground: layout extruded, channel cut -> {n_plates} solid(s), SI housing merged in, {how} -> "
            f"{len(gnd)} solid(s)")

    # ---- 3b. the hardware, fused into its owners ------------------------------------------------------------------
    # AFTER hollowing: the aperture tool is a full revolution, so hardware fused before it would be cut out again over
    # exactly the height it is meant to block. Posts on a dummy edge go into that dummy (star: the ground) BEFORE the
    # dummies are merged into the housing.
    if posts:
        if rings:
            join_posts_to_end_walls(posts, rings, polys, verbose=verbose)
        by_owner = {}
        for q in posts:
            own = q.get('owner', 'dee0')
            by_owner.setdefault('ground' if star and own.startswith('dummy') else own, []).append(q)
        for owner, qs in sorted(by_owner.items()):
            if owner not in pieces:
                raise RuntimeError(f"posts named owner {owner!r}, which is not a built solid ({sorted(pieces)})")
            tags = []
            for q in qs:
                # a post straddles the median plane; a window block sits above or below the opening (its own z range)
                z0 = float(q.get('z_lo', -q['half_height']))
                z1 = float(q.get('z_hi', q['half_height']))
                v = [t for d, t in occ.extrude([(2, _occ_polygon(occ, q['corners'], z0))], 0, 0, z1 - z0) if d == 3]
                if len(v) != 1:
                    raise RuntimeError(f"{q['name']}: extrude gave {len(v)} volumes")
                if q.get('kind') == 'aperture':
                    # the round hole: a cylinder along the edge normal, 1 mm proud of both plate faces
                    hc, nn, dep = q['hole_centre'], q['hole_axis'], q['depth']
                    base = hc - np.array([nn[0], nn[1], 0.0]) * 1.0
                    cyl = occ.addCylinder(float(base[0]), float(base[1]), float(base[2]), float(nn[0] * (dep + 2.0)),
                                          float(nn[1] * (dep + 2.0)), 0.0, float(q['hole_r']))
                    cut, _ = occ.cut([(3, v[0])], [(3, cyl)], removeObject=True, removeTool=True)
                    v = [t for d, t in cut if d == 3]
                    if len(v) != 1:
                        raise RuntimeError(f"{q['name']}: the hole cut gave {len(v)} volumes")
                tags.append(v[0])
            out, _ = occ.fuse([(3, t) for t in pieces[owner]], [(3, t) for t in tags], removeObject=True,
                              removeTool=True)
            pieces[owner] = [t for dd, t in out if dd == 3]
            say(f"fused {len(tags)} post(s) into {owner} -> {len(pieces[owner])} solid(s): "
                + ", ".join(q['name'] for q in qs))
            # a union that meets near-coincident faces can drop a tool WITHOUT an error (only BOP alerts in the log):
            # every block's footprint centre at its mid-height must be inside the result
            occ.synchronize()
            lost = []
            for q in qs:
                if q.get('kind') == 'aperture':     # (its centre is the hole)
                    continue
                c = np.asarray(q['corners'], dtype=float)
                xyz = [float(c[:, 0].mean()), float(c[:, 1].mean()),
                       0.5 * (float(q.get('z_lo', -q['half_height'])) + float(q.get('z_hi', q['half_height'])))]
                if not any(gmsh.model.isInside(3, t, xyz) for t in pieces[owner]):
                    lost.append(f"{q['name']} ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f})")
            if lost:
                raise RuntimeError(f"the fuse into {owner} LOST metal - outside every resulting solid: "
                                   + ", ".join(lost))
            say(f"  checked: every post / window block of {owner} is inside the result")
            for q in qs:                            # a grounded box must not reach into a dee: that is a short
                if q.get('kind') != 'box':
                    continue
                c = np.asarray(q['corners'], dtype=float)
                (xa, ya), (xb, yb) = c.min(axis=0), c.max(axis=0)
                for u in np.linspace(0.02, 0.98, 25):
                    for v in np.linspace(0.02, 0.98, 5):
                        xyz = [float(xa + u * (xb - xa)), float(ya + v * (yb - ya)), 0.0]
                        for other, tl in pieces.items():
                            if other != owner and any(gmsh.model.isInside(3, t, xyz) for t in tl):
                                raise RuntimeError(f"{q['name']} reaches into {other} at ({xyz[0]:.2f}, {xyz[1]:.2f})")
                say(f"  {q['name']}: x {xa:.2f}..{xb:.2f}, y {ya:.2f}..{yb:.2f} mm, z +-{q['half_height']:.1f} mm, "
                    f"clear of every other electrode")

    # ---- 4. the ground ------------------------------------------------------------------------------------------------
    if star:
        ground = pieces.pop('ground')
        occ.synchronize()
        say(f"ground (star) -> {len(ground)} solid(s)")
    else:
        ground = [(3, house)]
        for d in dummy:
            ground += [(3, t) for t in pieces.pop(d['label'])]
        out, _ = occ.fuse([ground[0]], ground[1:], removeObject=True, removeTool=True)
        ground = [t for dd, t in out if dd == 3]
        occ.synchronize()
        say(f"ground = housing + {len(dummy)} dummy dees -> {len(ground)} solid(s)")
    if not ground:
        raise RuntimeError("the ground boolean left NO solid - the build is broken (see the BOPAlgo alerts above)")
    say(f"stripped {_strip_orphans(gmsh, occ)} construction entities that belong to no solid (the revolve sketches "
        f"and their edges)")

    groups = {'ground': ground}
    for d in foot:
        groups[d['label']] = pieces[d['label']]
    # physical groups on the VOLUMES for bookkeeping and on their SURFACES because only 2D is meshed: gmsh.write saves
    # the elements of physical groups, so volume-only groups leave an empty .msh after generate(2)
    for name, vols in groups.items():
        gmsh.model.addPhysicalGroup(3, vols, name=name)
        faces = sorted({abs(s) for v in vols for _, s in gmsh.model.getBoundary([(3, v)], oriented=False)})
        gmsh.model.addPhysicalGroup(2, faces, name=f"{name}_surface")

    # ---- 5. mesh sizes ------------------------------------------------------------------------------------------------
    # a uniform size: unless the point / curvature / boundary drivers are off, gmsh refines on every small feature and
    # lc is an upper bound it never approaches. The BEM is DENSE (8 N^2 bytes): the element count is what binds.
    graded = bool(lc_outer) and lc_outer > lc
    mesh_max = max(lc, lc_outer) if graded else lc
    mesh_min = float(OPT.mesh_size_min_mm) if OPT.mesh_size_min_mm > 0.0 else 0.5 * lc
    for opt, val in (("Mesh.MeshSizeMax", mesh_max), ("Mesh.MeshSizeMin", mesh_min),
                     ("Mesh.MeshSizeFromPoints", 0), ("Mesh.MeshSizeFromCurvature", 0),
                     ("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.Algorithm", int(OPT.mesh_algorithm))):
        gmsh.option.setNumber(opt, val)
    if int(OPT.mesh_algorithm) != 6 or abs(mesh_min - 0.5 * lc) > 1e-12:
        say(f"mesh: algorithm {int(OPT.mesh_algorithm)}, MeshSizeMin {mesh_min:g} mm (mesh_algorithm / "
            f"mesh_size_min_mm)")
    fld = gmsh.model.mesh.field
    ids = []
    if graded:
        # lc inside r_fine, lc_outer beyond r_coarse, linear between; the clamp written with Sqrt only (|t| =
        # Sqrt(t*t)), independent of which functions gmsh's MathEval knows
        t_ = f"((Sqrt(x*x + y*y) - {r_fine:g}) / {r_coarse - r_fine:g})"
        t = fld.add("MathEval")
        fld.setString(t, "F", f"{lc:g} + {lc_outer - lc:g} * 0.5 * (Sqrt({t_}*{t_}) - Sqrt(({t_} - 1)*({t_} - 1)) + 1)")
        ids.append(t)
        say(f"graded mesh: {lc:g} mm inside r = {r_fine:g} mm, {lc_outer:g} mm beyond {r_coarse:g} mm")
    # local refinement on the hardware: the posts are fused into their parents by now and have no surfaces of their
    # own to refine by, but a Cylinder field is pure geometry - one per post on its axis, lc_posts inside, combined by
    # Min. Every refined post adds elements to the dense BEM, so the element count is printed.
    if posts and lc_posts and lc_posts < lc:
        for q in posts:
            t = fld.add("Cylinder")
            if q.get('kind') == 'aperture':        # the hole's rim: a cylinder field on its own (horizontal) axis
                hc, nn, dep = q['hole_centre'], q['hole_axis'], q['depth']
                mid = hc + np.array([nn[0], nn[1], 0.0]) * 0.5 * dep
                vals = (("XCenter", float(mid[0])), ("YCenter", float(mid[1])), ("ZCenter", float(mid[2])),
                        ("XAxis", float(nn[0] * (0.5 * dep + lc_posts))), ("YAxis", float(nn[1] * (0.5 * dep + lc_posts))),
                        ("ZAxis", 0.0), ("Radius", float(q['hole_r']) + 2.0 * lc_posts))
            else:
                z0, z1 = float(q.get('z_lo', -q['half_height'])), float(q.get('z_hi', q['half_height']))
                c = np.asarray(q['corners'], dtype=float)
                cx, cy = float(c[:, 0].mean()), float(c[:, 1].mean())
                vals = (("XCenter", cx), ("YCenter", cy), ("ZCenter", 0.5 * (z0 + z1)), ("XAxis", 0.0), ("YAxis", 0.0),
                        ("ZAxis", 0.5 * abs(z1 - z0) + lc_posts),
                        ("Radius", float(np.max(np.hypot(c[:, 0] - cx, c[:, 1] - cy))) + lc_posts))
            for k, v in vals + (("VIn", lc_posts), ("VOut", mesh_max)):
                fld.setNumber(t, k, v)
            ids.append(t)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(0.5 * lc, lc_posts))   # the field is clamped by MeshSizeMin
        say(f"local mesh refinement: {len(posts)} cylinder field(s) at {lc_posts:g} mm (bulk {lc:g} mm)")
    # a round dee0 nose: nose_lc on the round, twice that out to the ground's gap round it, full height - the bulk
    # mesh's chords would otherwise cut the gap round the nose noticeably
    nose = next((d['tip'] for d in foot if d['tip'].get('kind') == 'round'), None)
    if nose is not None and 0.0 < OPT.nose_lc_mm < lc:
        C, R, nl = nose['centre'], float(nose['radius']), float(OPT.nose_lc_mm)
        zones = ((R + 1.5, nl), (R + _star_gap_floor_mm() + 2.0 * nl, min(2.0 * nl, lc)))
        for rad, size in zones:
            t = fld.add("Cylinder")
            for k, v in (("XCenter", float(C[0])), ("YCenter", float(C[1])), ("ZCenter", 0.0), ("XAxis", 0.0),
                         ("YAxis", 0.0), ("ZAxis", float(half) + size), ("Radius", rad), ("VIn", size),
                         ("VOut", mesh_max)):
                fld.setNumber(t, k, v)
            ids.append(t)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(gmsh.option.getNumber("Mesh.MeshSizeMin"), nl))
        say("dee0 nose: " + ", ".join(f"{s:g} mm within {r_:.1f} mm" for r_, s in zones)
            + " of the round's centre, full height (nose_lc_mm)")
    if ids:
        m = fld.add("Min")
        fld.setNumbers(m, "FieldsList", ids)
        fld.setAsBackgroundMesh(m)
    return gmsh, groups, r_tool, polys


# =====================================================================================================================
# 13. the surface mesh and the ElectrodeModel
# =====================================================================================================================
def _tri_angles(P: np.ndarray) -> np.ndarray:
    """(m, 3, 3) triangle corners -> (m, 3) interior angles [deg]."""
    out = []
    for i in range(3):
        u, v = P[:, (i + 1) % 3] - P[:, i], P[:, (i + 2) % 3] - P[:, i]
        c = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-300)
        out.append(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return np.column_stack(out)


def flip_bad_triangles(gmsh, min_deg: float = None, passes: int = 4, verbose: bool = True) -> int:
    """Edge flips that lift triangles with an angle below ``min_deg``; returns the flip count.

    Frontal-Delaunay can close its front straight across a boundary curve that bows out of its chord by microns (a
    junction jog running tangentially across a taper cone), leaving a flat triangle. Repaired per surface in gmsh's own
    mesh (so the .msh, the pickle and the BEM all see it), no node moved and no boundary curve touched: a bad triangle
    (p, q, r) and the neighbour (q, p, s) across an interior edge become (p, s, r), (s, q, r) when both keep the
    surface's orientation, s-r is not an edge already, and the pair's smallest angle goes up."""
    min_deg = OPT.mesh_flip_min_deg if min_deg is None else min_deg
    if min_deg <= 0.0:
        return 0
    nt, nc, _ = gmsh.model.mesh.getNodes()
    X = np.zeros((int(np.max(nt)) + 1, 3))
    X[np.asarray(nt, dtype=np.int64)] = np.asarray(nc, dtype=float).reshape(-1, 3)
    n_flip, n_surf, worst = 0, 0, [180.0, 180.0]

    def normal(t):
        return np.cross(X[t[1]] - X[t[0]], X[t[2]] - X[t[0]])

    for _, s in gmsh.model.getEntities(2):
        et, etg, en = gmsh.model.mesh.getElements(2, s)
        if len(et) != 1 or int(et[0]) != 2:
            continue                                   # pure 3-node-triangle surfaces only
        tags = np.asarray(etg[0], dtype=np.int64)
        T = np.asarray(en[0], dtype=np.int64).reshape(-1, 3).copy()
        a0 = _tri_angles(X[T]).min()
        if a0 >= min_deg:
            continue
        flips = 0
        for _ in range(passes):
            edges = {}
            for i, tri in enumerate(T):
                for k in range(3):
                    edges.setdefault(frozenset((int(tri[k]), int(tri[(k + 1) % 3]))), []).append(i)
            ang = _tri_angles(X[T])
            bad = np.flatnonzero(ang.min(1) < min_deg)
            done = set()
            for i in bad[np.argsort(ang[bad].min(1))]:
                if i in done:
                    continue
                best = None
                for k in range(3):
                    p, q, r = (int(T[i][(k + m) % 3]) for m in range(3))
                    nb = [j for j in edges[frozenset((p, q))] if j != i]
                    if len(nb) != 1 or nb[0] in done:
                        continue                       # a boundary curve, or a neighbour flipped this pass
                    j = nb[0]
                    s_ = [int(v) for v in T[j] if v != p and v != q][0]
                    if frozenset((r, s_)) in edges:
                        continue
                    new = np.array([[p, s_, r], [s_, q, r]])
                    n_old = normal(T[i]) + normal(T[j])
                    if min(normal(new[0]) @ n_old, normal(new[1]) @ n_old) <= 0.0:
                        continue                       # a non-convex pair: the flip would fold it
                    m_new = float(_tri_angles(X[new]).min())
                    if m_new > min(ang[i].min(), ang[j].min()) + 1e-9 and (best is None or m_new > best[0]):
                        best = (m_new, j, new)
                if best is not None:
                    _, j, new = best
                    T[i], T[j] = new
                    done |= {int(i), int(j)}
            if not done:
                break
            flips += len(done) // 2
        if flips:
            gmsh.model.mesh.removeElements(2, s)
            gmsh.model.mesh.addElementsByType(s, 2, tags.tolist(), T.ravel().tolist())
            n_flip, n_surf = n_flip + flips, n_surf + 1
        worst[0] = min(worst[0], float(a0))
        worst[1] = min(worst[1], float(_tri_angles(X[T]).min()))
    if verbose and n_flip:
        print(f"[mesh] {n_flip} edge flip(s) in {n_surf} surface(s) for triangles with an angle < {min_deg:g} deg "
              f"(mesh_flip_min_deg): smallest angle {worst[0]:.3f} -> {worst[1]:.3f} deg (nodes unchanged)")
    return n_flip


def mesh_and_report(gmsh, groups, prefix: str = None, verbose: bool = True):
    """Surface-mesh the model, repair slivers, report per-electrode element counts and quality, and write
    <prefix>.msh / <prefix>.step. Returns (vertices [m], triangles, owner per triangle, triangles per electrode)."""
    t0 = time.time()
    gmsh.model.mesh.generate(2)
    flip_bad_triangles(gmsh, verbose=verbose)
    say = (lambda m: print(f"[mesh] {m}")) if verbose else (lambda m: None)
    say(f"generated in {time.time() - t0:.1f} s")
    surf_of = {}
    for name, vols in groups.items():
        for v in vols:
            for _, s in gmsh.model.getBoundary([(3, v)], oriented=False):
                surf_of[abs(s)] = name
    nt, nc, _ = gmsh.model.mesh.getNodes()          # (tags, COORDINATES, parametric coordinates)
    nodes = {int(t): i for i, t in enumerate(nt)}
    verts = np.asarray(nc, dtype=float).reshape(-1, 3) / MM
    tris, owner, per = [], [], {}
    for s, name in surf_of.items():
        et, _, ec = gmsh.model.mesh.getElements(2, s)
        for t, conn in zip(et, ec):
            if int(t) != 2:                                          # 3-node triangles only
                continue
            idx = np.vectorize(nodes.__getitem__)(np.asarray(conn, dtype=np.int64).reshape(-1, 3))
            tris.append(idx)
            owner += [name] * len(idx)
            per[name] = per.get(name, 0) + len(idx)
    tris = np.vstack(tris)
    say(f"{len(verts):,} nodes, {len(tris):,} triangles over {len(surf_of)} surfaces")
    for k in sorted(per):
        say(f"  {k:<8s} {per[k]:7,d} triangles ({100.0 * per[k] / len(tris):5.1f} %)")
    a = verts[tris]
    area = 0.5 * np.linalg.norm(np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]), axis=1) * MM ** 2
    edge = np.stack([np.linalg.norm(a[:, (i + 1) % 3] - a[:, i], axis=1) for i in range(3)]).T * MM
    say(f"  element area {area.min():.4f} .. {area.max():.2f} mm^2 (median {np.median(area):.2f}); "
        f"shortest edge {edge.min():.4f} mm")
    if prefix:
        gmsh.write(f"{prefix}.msh")
        gmsh.write(f"{prefix}.step")
        say(f"wrote {prefix}.msh and {prefix}.step")
    return verts, tris, np.array(owner), per


def dee_potential(dees: DeeSystem) -> float:
    """The potential the dees are built at [V]: dee 0's entry gap voltage at its phase, with the sign of the
    deceleration-free convention of the gap model (-V cos(phase))."""
    return -float(dees.gaps[0].voltage) * np.cos(np.deg2rad(dees.gaps[0].phase_deg))


def to_electrode_model(gmsh, groups, dee_voltage: float, half_height: float = None, r_cut: float = None,
                       aperture=None, si_rotation_deg: float = 0.0, verbose: bool = True):
    """The meshed model as a PyCentralRegion ElectrodeModel, for solve_gap_field and every consumer of 3D models.

    Those consumers (solid_meshes, metal_mask, stacked_obstacles, the drawing) rely on the triangles being ordered in
    CONTIGUOUS blocks per solid with params['solids'] giving each block's [n0, n1) range, so they are gathered group by
    group (dees first), not surface by surface. params['si_rotation_deg'] is the spiral-inflector rotation the
    electrodes were built with: the tracker turns the inflector field, housing and bunch by the same angle."""
    from PyCentralRegion.gap_fields import ElectrodeModel
    half_height = OPT.half_height_mm if half_height is None else half_height
    r_cut = r_max_mm() if r_cut is None else r_cut
    aperture = OPT.aperture_heights() if aperture is None else aperture
    dummy_ap = OPT.dummy_aperture_heights()

    nt, nc, _ = gmsh.model.mesh.getNodes()
    idx = {int(t): i for i, t in enumerate(nt)}
    verts = np.asarray(nc, dtype=float).reshape(-1, 3) / MM
    blocks, pots, solids, n = [], [], [], 0
    for name in sorted(groups, key=lambda k: (not k.startswith('dee'), k)):
        got = []
        for v in groups[name]:
            for _, s in gmsh.model.getBoundary([(3, v)], oriented=False):
                et, _, ec = gmsh.model.mesh.getElements(2, abs(s))
                for t, conn in zip(et, ec):
                    if int(t) == 2:
                        got.append(np.vectorize(idx.__getitem__)(np.asarray(conn, dtype=np.int64).reshape(-1, 3)))
        if not got:
            continue
        b = np.vstack(got)
        pot = dee_voltage if name.startswith('dee') else 0.0
        blocks.append(b)
        pots.append(np.full(len(b), pot))
        solids.append({'name': name, 'potential': pot, 'tri_range': (n, n + len(b))})
        n += len(b)
    tris = np.vstack(blocks)
    pots = np.concatenate(pots)
    used = np.unique(tris)                                  # drop nodes no triangle references
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    verts, tris = verts[used], remap[tris]
    area = 0.5 * np.linalg.norm(np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]]),
                                axis=1)
    bad = int(np.sum(area < 1e-14))
    if bad:
        raise RuntimeError(f"{bad} degenerate triangles in the 3D electrode mesh")
    z_cut, r_c = half_height / MM, r_cut / MM
    step = OPT.aperture_radii()
    params = {'dim': 3, 'z_cut': z_cut, 'r_cut': r_c, 'solids': solids, 'source': 'PyCentralRegion.geometry',
              'dee_aperture': True, 'ground_aperture': True, 'dee_height': False, 'valley_height': None,
              'bars': [], 'posts': [], 'extra_solids': [], 'voltage_profile': None,
              'profiles': {'dee_aperture': [[0.0, aperture[0] / MM], [step[0] / MM, aperture[0] / MM],
                                            [step[1] / MM, aperture[1] / MM], [1.0, aperture[1] / MM]],
                           'dummy_aperture': [[0.0, dummy_ap[0] / MM], [step[0] / MM, dummy_ap[0] / MM],
                                              [step[1] / MM, dummy_ap[1] / MM], [1.0, dummy_ap[1] / MM]]},
              # every consumer reads the radial layout from here. The BEM field is the whole accelerating field
              # (bem_only, always: the tracker blends no RF-cavity map on); 'seam' is where such a blend would start,
              # inside the dees' end r_cut, kept for the consumers of the original design deck's layout
              'layout': {'r_features': r_c, 'seam': (r_c - 0.055, r_c - 0.030), 'r_cut': r_c, 'z_cut': z_cut,
                         'bem_only': True},
              'si_rotation_deg': float(si_rotation_deg)}
    model = ElectrodeModel(vertices=verts, triangles=tris, potentials=pots, wedges=[], params=params)
    if verbose:
        print(f"[model] {len(tris):,} triangles, {len(verts):,} nodes; "
              + ", ".join(f"{s['name']} {s['tri_range'][1] - s['tri_range'][0]}" for s in solids))
        print(f"[model] z_cut {z_cut * MM:.1f} mm, r_cut {r_c * MM:.1f} mm, "
              f"seam {params['layout']['seam'][0] * MM:.0f}..{params['layout']['seam'][1] * MM:.0f} mm, "
              f"spiral inflector turned {si_rotation_deg:+.6f} deg")
    return model
