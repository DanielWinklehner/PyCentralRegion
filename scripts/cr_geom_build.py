"""
cr_geom_build.py - central-region geometry builder, rebuilt from scratch, step by step.

WHY THIS EXISTS. The existing builder (electrodes3d.build_electrodes_3d, driven by
HCHC-60_bem3d_confirmation.build_model3d) produces a model with problems that show up as solver
trouble rather than as obvious breakage (Daniel, 2026-09-16):
  1. the housing carries top and bottom plates it should not have - its radial ends are open too;
  2. the posts complicate things before the basics are settled;
  3. there are leftovers and artifacts;
  4. merging the dummy dees with the housing yields PARALLEL BUT DISTINCT triangles, which can
     trip any solver.
Symptomatically: 74,530 elements for a central region, with the housing meshed at 2.6 mm against a
4 mm minimum and holding 57 % of all sub-minimum triangles, and a STEP export carrying 215,909
entities for what is conceptually eight solids.

So this is a clean rebuild, one piece at a time, with each step runnable and inspectable.

=============================================================================================
STEP 1 (this file): the two inputs and nothing else.
  * read the RF gap CENTRE LINES from the optimizer result and print them;
  * read the HOUSING OUTLINE - the polygon the housing wall is extruded from and the four
    corners of its exit aperture - and print it.
  NO solid, no gmsh, no dees, no ground, no posts, no bars, no crop, no meshing, no BEM.
=============================================================================================

THE HOUSING IS NOT A STEP FILE ANY MORE (Daniel, 2026-09-17: "let's simplify the housing").
The spiral inflector writes housing_outline.txt next to its STEP files now, in the same frame
and units: the convex polygon its wall is extruded from, taken BEFORE any extrusion, plus the
four corners of the exit opening. The central region builds its own housing from those two
things, so nothing here has to import, crop or repair 002_Housing.step.

UNITS - the one thing worth being pedantic about up front, because the old builder mixes them:
  * RFCavity segment endpoints are in METRES (the tracking convention).
  * The outline file carries its own units; the baseline-frame one is in MILLIMETRES.
  * So everything printed below is in mm, and the conversion happens exactly once, at MM.

Usage
    python cr_geom_build.py --outline <...>\\steps_baseline_mm\\housing_outline.txt
    python cr_geom_build.py --outline <...> --plot            # ... and show it
    python cr_geom_build.py --outline <...> --plot outline.png

Env
    HCHC60_RESULT_PKL    optimizer result carrying the gap geometry
                         (default: output_inflector/optimized_cavity_geometry_pickb_nocut.pkl)
    HCHC60_HOUSING_OUTLINE  housing outline text file
    HCHC60_EXPORT        an export_baseline_levelled folder; the outline is taken from its
                         steps_baseline_mm/ unless --outline says otherwise
    PYCENTRALREGION_SRC  PyCentralRegion source dir, if not installed
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_SRC = os.environ.get('PYCENTRALREGION_SRC')
if _SRC:
    sys.path.insert(0, _SRC)

from PyCentralRegion.rf_cavity import (DeeSystem, create_dee_system,      # noqa: E402
                                       snap_nodes_between_turns)

MM = 1000.0          # metres -> millimetres. the baseline outline and gmsh are both in mm.

# --- the dee system, as the deck defines it (HCHC-60_optimize_cavity_geometry.py) -----------
R_MIN = 0.005                                   # m, inner cavity radius
R_MAX = float(os.environ.get('HCHC60_R_MAX_CAV', 0.60))   # m, outer cavity radius
DEE_CENTER_ANGLES = [45.0, 135.0, 225.0, 315.0]
HARMONIC = 4
DEE_OPENING_ANGLE = 180.0 / HARMONIC            # 45 deg, the classical synchronous dee angle
CAVITY_VOLTAGE = 65000.0
# The accelerating gap follows the CAD: a constant-angle wedge floored at a minimum slot
# (Daniel, 2026-09-20), width(r) = max(GAP_WIDTH_MIN, r * GAP_ANGLE_DEG). GAP_WIDTH stays as
# the nominal the DeeSystem is seeded with; the LOCAL width is what the boundaries are offset
# by - see _half_width_mm. Keep these in step with HCHC-60_optimize_cavity_geometry.py, or the
# geometry built here is not the geometry the optimizer designed.
GAP_WIDTH = 0.010                               # m
GAP_ANGLE_DEG = float(os.environ.get('HCHC60_GAP_ANGLE_DEG', 2.5))
GAP_WIDTH_MIN = float(os.environ.get('HCHC60_GAP_WIDTH_MIN_MM', 6.5)) * 1e-3

# --- the vertical structure, as the old builder ended up using it ---------------------------
# electrodes3d.HeightProfiles.iba plus the decisions of 2026-09-13/16 that
# HCHC-60_bem3d_confirmation.build_model3d bakes in: the beam channel is 34 mm full height out to
# r = 148 mm and 50 mm from r = 247 mm (the IBA CAD's 26 mm first-turn value is a mistake of the
# RF design - no reason for the dee to be narrower than the dummy dee), and dee and dummy dee
# share the same two breakpoints so the channel is vertically symmetric at every radius.
DEE_APERTURE_MM = (34.0, 50.0)     # full INNER height of the beam channel: first turn, outer
APERTURE_R_MM = (148.0, 247.0)     # the shared radial step
R_CUT_MM = 305.0                   # the old builder's crop (250 seam + 25 blend + 30 margin)


def _height(r_mm, step, h):
    """Full height [mm] at radius r [mm]: h[0] inside step[0], h[1] beyond step[1], linear between."""
    return np.interp(np.asarray(r_mm, dtype=float),
                     [0.0, step[0], step[1], 1.0e4], [h[0], h[0], h[1], h[1]])


_DEF_PKL = HERE / 'output_inflector' / 'optimized_cavity_geometry_pickb_nocut.pkl'
_DEF_OUTLINE = (Path(os.environ.get('HCHC60_EXPORT', '')) / 'steps_baseline_mm' / 'housing_outline.txt'
                if os.environ.get('HCHC60_EXPORT') else None)


# ============================================================================================
# 1a. the gap centre lines
# ============================================================================================
def _half_width_mm(cav, pts_mm, side: float):
    """Signed half gap width [mm] at every point of a centre line given in MILLIMETRES.

    The gap width varies with radius under either RFCavity law - the angular wedge
    (gap_angle_deg/gap_width_min) or the linear taper (gap_width_inner/gap_taper_radius) -
    so the boundary has to be offset by the LOCAL width. Offsetting by the nominal
    cav.gap_width built a constant-width gap while the optimizer had designed a varying
    one, which is the geometry the BEM would then have solved.
    """
    pts = np.asarray(pts_mm, dtype=float)
    fn = getattr(cav, 'gap_width_at', None)
    if fn is None:                                   # pre-law cavity objects
        return side * 0.5 * float(cav.gap_width) * MM
    r_m = np.hypot(pts[:, 0], pts[:, 1]) / MM
    return side * 0.5 * np.asarray(fn(r_m), dtype=float) * MM


def load_gap_lines(pkl_path: Path):
    """The RF gap centre lines of the optimizer's winning geometry.

    Returns (DeeSystem, geometry dict). Each gap is an RFCavity whose ``segments`` are the
    straight pieces of its centre line, with 'p1' / 'p2' in METRES.
    """
    with open(pkl_path, 'rb') as fh:
        result = pickle.load(fh)
    geom = result.metadata['optimal_geometry']

    n_seg = len(geom['segment_angles_per_gap'][0])
    # RFCavity REQUIRES angles and radii at construction when n_variable_segments > 0 (they
    # default to [] and then fail the length check), so seed with gap 0's and override per gap
    # below. The radii must also be strictly increasing between r_min and r_max.
    dees = create_dee_system(R_MIN, R_MAX, DEE_CENTER_ANGLES, DEE_OPENING_ANGLE,
                             voltage=CAVITY_VOLTAGE, harmonic=HARMONIC, gap_width=GAP_WIDTH,
                             gap_angle_deg=GAP_ANGLE_DEG, gap_width_min=GAP_WIDTH_MIN,
                             n_variable_segments=n_seg,
                             segment_angles=list(geom['segment_angles_per_gap'][0]),
                             segment_radii=list(geom['segment_radii_per_gap'][0]))
    # the opening angle is a system-level parameter; apply it BEFORE the per-gap segments, as
    # it recomputes every gap's azimuth from the stored dee centres
    if geom.get('opening_angle_deg') is not None:
        dees.apply_opening_angle(float(geom['opening_angle_deg']))
    rots = geom.get('segment_rotations_per_gap')
    for g, cav in enumerate(dees.gaps):
        cav.update_geometry(segment_angles=list(geom['segment_angles_per_gap'][g]),
                            segment_radii=list(geom['segment_radii_per_gap'][g]),
                            segment_rotations=(list(rots[g]) if rots else None))
    # The gap LAW comes from this module, not from the pickle - old results carry no record of
    # the width they were optimized with. A geometry optimized under the flat 10 mm gap and
    # rebuilt here gets the angular law instead, so its boundaries move. Say so loudly: it
    # changes the geometry, and a silent change is how a rebuilt config stops matching its own
    # tracking results.
    _law = (f"angular {GAP_ANGLE_DEG:g} deg floored at {GAP_WIDTH_MIN * MM:.2f} mm"
            f"  -> {dees.gaps[0].gap_width_at(0.080) * MM:.2f} mm at r = 80 mm, "
            f"{dees.gaps[0].gap_width_at(0.200) * MM:.2f} mm at r = 200 mm")
    print(f"  gap width law: {_law}")
    print(f"  NOTE: the law is this module's, not the pickle's. A geometry optimized under a "
          f"different gap width rebuilds with DIFFERENT boundaries.")
    return dees, geom, result


def print_gap_lines(dees: DeeSystem, geom: dict, result) -> None:
    print("=" * 96)
    print("RF GAP CENTRE LINES")
    print("=" * 96)
    print(f"  {len(dees.gaps)} gaps on {dees.n_dees} dees, opening angle "
          f"{dees.opening_angle:.3f} deg, harmonic {HARMONIC}")
    print(f"  r_min {R_MIN * MM:.1f} mm, r_max {R_MAX * MM:.1f} mm, "
          f"{CAVITY_VOLTAGE / 1e3:.1f} kV/gap, nominal gap width {GAP_WIDTH * MM:.1f} mm")
    print(f"  source: bunch phase {float(result.bunch_phase_deg):+.4f} deg, "
          f"f_rev {float(result.rf_frequency_mhz):.6f} MHz")
    print()
    print(f"  {'gap':>4} {'seg':>4} {'x1':>9} {'y1':>9} {'x2':>9} {'y2':>9} "
          f"{'len':>8} {'r1':>8} {'r2':>8} {'az1':>8} {'az2':>8}")
    print(f"  {'':>4} {'':>4} {'[mm]':>9} {'[mm]':>9} {'[mm]':>9} {'[mm]':>9} "
          f"{'[mm]':>8} {'[mm]':>8} {'[mm]':>8} {'[deg]':>8} {'[deg]':>8}")
    print("  " + "-" * 92)
    for g, cav in enumerate(dees.gaps):
        for si, seg in enumerate(cav.segments):
            p1 = np.asarray(seg['p1'], dtype=float)[:2] * MM
            p2 = np.asarray(seg['p2'], dtype=float)[:2] * MM
            r1, r2 = np.hypot(*p1), np.hypot(*p2)
            a1 = np.degrees(np.arctan2(p1[1], p1[0])) % 360.0
            a2 = np.degrees(np.arctan2(p2[1], p2[0])) % 360.0
            print(f"  {g:>4} {si:>4} {p1[0]:9.3f} {p1[1]:9.3f} {p2[0]:9.3f} {p2[1]:9.3f} "
                  f"{np.linalg.norm(p2 - p1):8.3f} {r1:8.3f} {r2:8.3f} {a1:8.3f} {a2:8.3f}")
        print("  " + "-" * 92)
    # continuity: a gap's segments should form ONE polyline, end to end
    print("  continuity check (segment k end -> segment k+1 start, should be 0):")
    worst = 0.0
    for g, cav in enumerate(dees.gaps):
        for si in range(len(cav.segments) - 1):
            d = float(np.linalg.norm(np.asarray(cav.segments[si]['p2'], dtype=float)[:2]
                                     - np.asarray(cav.segments[si + 1]['p1'], dtype=float)[:2]))
            worst = max(worst, d)
    print(f"    worst gap between consecutive segments: {worst * MM:.6f} mm")


# ============================================================================================
# 1b. the housing outline
# ============================================================================================
def read_housing_outline(path: Path) -> dict:
    """The housing outline text file written by the spiral inflector.

    '#' comments, 'key value' lines, and a section header 'name count' followed by that many
    rows. Returns {'frame', 'units', 'outer' (N,2), 'aperture' (4,3)} and, when the file has
    one, 'curves' as [(kind, i0, i1), ...] - deliberately a plain reader so this script stays
    standalone.

    Since 2026-09-19 the spiral inflector simplifies the outline itself (one straight plate at
    the exit, square to the design orbit, and one spline round the rest) and writes the curves
    beside the points; `curves` is that. Older files have none, and their points are a plain
    closed polygon for `simplify_outline` to work on.
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
    out['outer'] = out['outer'] * scale        # everything below is in mm
    out['aperture'] = out['aperture'] * scale
    out['units'] = 'mm'
    return out


def print_housing_outline(ol: dict, path: Path) -> None:
    outer, ap = ol['outer'], ol['aperture']
    n = len(outer)
    edges = np.roll(outer, -1, axis=0) - outer
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    # turn direction at each vertex (np.cross on 2-vectors is deprecated in numpy 2)
    prev = outer - np.roll(outer, 1, axis=0)
    cross = prev[:, 0] * edges[:, 1] - prev[:, 1] * edges[:, 0]
    area = 0.5 * float(np.sum(outer[:, 0] * np.roll(outer[:, 1], -1) - np.roll(outer[:, 0], -1) * outer[:, 1]))
    r = np.hypot(outer[:, 0], outer[:, 1])
    az = np.degrees(np.arctan2(outer[:, 1], outer[:, 0])) % 360.0

    print()
    print("=" * 96)
    print("HOUSING OUTLINE")
    print("=" * 96)
    print(f"  file   : {path}")
    print(f"  frame  : {ol['frame']}   units: {ol['units']} (converted to mm if they were metres)")
    print(f"  outer polygon: {n} vertices, perimeter {lengths.sum():.3f} mm, "
          f"area {abs(area):.1f} mm^2, {'counter-clockwise' if area > 0 else 'clockwise'}")
    print(f"    radius from the machine axis {r.min():.3f} .. {r.max():.3f} mm, "
          f"edge length {lengths.min():.3f} .. {lengths.max():.3f} mm")
    print(f"    convex: {'yes' if np.all(cross > 0) or np.all(cross < 0) else 'NO'}")
    print()
    print(f"  {'i':>4} {'x':>10} {'y':>10} {'r':>10} {'az':>9} {'edge':>9}")
    print(f"  {'':>4} {'[mm]':>10} {'[mm]':>10} {'[mm]':>10} {'[deg]':>9} {'[mm]':>9}")
    print("  " + "-" * 56)
    for i in range(n):
        print(f"  {i:>4} {outer[i, 0]:10.3f} {outer[i, 1]:10.3f} {r[i]:10.3f} {az[i]:9.3f} {lengths[i]:9.3f}")

    # the aperture: four corners, so the centre, the two edge directions and the normal follow
    c = ap.mean(axis=0)
    u, w = ap[1] - ap[0], ap[3] - ap[0]
    a_len, b_len = float(np.linalg.norm(u)), float(np.linalg.norm(w))
    u, w = u / a_len, w / b_len
    nvec = np.cross(u, w)
    print()
    print(f"  aperture: {len(ap)} corners, {a_len:.3f} x {b_len:.3f} mm "
          f"(edges perpendicular to {np.degrees(np.arcsin(abs(float(u.dot(w))))):.4f} deg)")
    print(f"  {'i':>4} {'x':>10} {'y':>10} {'z':>10} {'r':>10} {'az':>9}")
    print(f"  {'':>4} {'[mm]':>10} {'[mm]':>10} {'[mm]':>10} {'[mm]':>10} {'[deg]':>9}")
    print("  " + "-" * 56)
    for i, p in enumerate(ap):
        print(f"  {i:>4} {p[0]:10.3f} {p[1]:10.3f} {p[2]:10.3f} {np.hypot(p[0], p[1]):10.3f} "
              f"{np.degrees(np.arctan2(p[1], p[0])) % 360.0:9.3f}")
    print(f"    centre ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}) mm, "
          f"r {np.hypot(c[0], c[1]):.3f} mm, azimuth {np.degrees(np.arctan2(c[1], c[0])) % 360.0:.3f} deg")
    print(f"    normal ({nvec[0]:+.5f}, {nvec[1]:+.5f}, {nvec[2]:+.5f}) -> "
          f"azimuth {np.degrees(np.arctan2(nvec[1], nvec[0])) % 360.0:.3f} deg, "
          f"elevation {np.degrees(np.arcsin(np.clip(nvec[2], -1, 1))):+.3f} deg")

    # where the opening sits on the polygon: the nearest outer edge to the aperture centre
    d = np.array([_seg_distance(c[:2], outer[i], outer[(i + 1) % n]) for i in range(n)])
    i = int(np.argmin(d))
    print(f"    nearest outer edge: {i} -> {(i + 1) % n}, centre {d[i]:.3f} mm from it "
          f"(the wall runs between the outer polygon and its inward offset)")


def plot_outline(ol: dict, dees: DeeSystem, out: str = None) -> None:
    """Top view of the housing outline over the RF gap centre lines, plus the aperture face on.

    Left: the outer polygon with its vertices, the exit opening projected into the median plane,
    and the gap centre lines for scale. Right: the opening in its own plane, so its tilt out of
    the median plane is visible (the corner labels are z).
    """
    import matplotlib
    if out:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    outer, ap = ol['outer'], ol['aperture']
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 7.0),
                                 gridspec_kw={'width_ratios': [1.65, 1.0]})

    # --- left: top view -----------------------------------------------------------------
    for g, cav in enumerate(dees.gaps):
        for si, seg in enumerate(cav.segments):
            p1 = np.asarray(seg['p1'], dtype=float)[:2] * MM
            p2 = np.asarray(seg['p2'], dtype=float)[:2] * MM
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color='0.75', lw=1.0, zorder=1,
                    label='RF gap centre lines' if (g == 0 and si == 0) else None)
    ring = np.vstack([outer, outer[0]])
    ax.plot(ring[:, 0], ring[:, 1], '-', color='C3', lw=1.8, zorder=3,
            label='housing outer polygon ({} vertices)'.format(len(outer)))
    ax.plot(outer[:, 0], outer[:, 1], '.', color='C3', ms=4, zorder=4)
    rect = np.vstack([ap[:, :2], ap[0, :2]])
    ax.plot(rect[:, 0], rect[:, 1], '-', color='C2', lw=2.4, zorder=5,
            label='exit aperture (projected)')
    ax.plot([0.0], [0.0], '+', color='k', ms=10, mew=1.5, zorder=6, label='machine axis')
    r = np.hypot(outer[:, 0], outer[:, 1]).max()
    for rr in (100.0, 200.0, 300.0):
        ax.add_patch(plt.Circle((0, 0), rr, fill=False, color='0.88', lw=0.8, zorder=0))
    ax.set_xlim(-1.35 * r, 1.35 * r)
    ax.set_ylim(-1.35 * r, 1.35 * r)
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('top view, {} frame (grey circles r = 100, 200, 300 mm)'.format(ol['frame']))
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.25)

    # --- right: the opening in its own plane --------------------------------------------
    c = ap.mean(axis=0)
    u, w = ap[1] - ap[0], ap[3] - ap[0]
    a_len, b_len = float(np.linalg.norm(u)), float(np.linalg.norm(w))
    u, w = u / a_len, w / b_len
    uu, ww = (ap - c).dot(u), (ap - c).dot(w)
    bx.plot(np.append(uu, uu[0]), np.append(ww, ww[0]), '-', color='C2', lw=2.4)
    bx.plot(uu, ww, 'o', color='C2', ms=6)
    for i in range(len(ap)):
        bx.annotate('{}  z = {:+.1f}'.format(i, ap[i, 2]), (uu[i], ww[i]), fontsize=8,
                    textcoords='offset points', xytext=(6, 6 if ww[i] > 0 else -12))
    bx.axhline(0.0, color='0.85', lw=0.8)
    bx.axvline(0.0, color='0.85', lw=0.8)
    bx.set_aspect('equal')
    bx.set_xlabel('along the opening [mm]')
    bx.set_ylabel('across the opening [mm]')
    bx.set_title('exit aperture, face on: {:.1f} x {:.1f} mm\ncentre ({:+.1f}, {:+.1f}, {:+.1f}) mm, '
                 'z {:+.1f} .. {:+.1f} mm'.format(a_len, b_len, c[0], c[1], c[2],
                                                  ap[:, 2].min(), ap[:, 2].max()), fontsize=10)
    bx.grid(alpha=0.25)

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140)
        print(f"  wrote {out}")
    else:
        plt.show()


def simplify_outline(outer: np.ndarray, cluster_tol: float = 0.5, keep_every: int = 5,
                     n_spline: int = 600) -> dict:
    """Turn the 94-gon into two straight faces and one spline.

    The polygon comes out of a convex hull of circle-offset points, so it is two long flat faces
    joined by a finely sampled curve, with a tight knot of nearly collinear points wherever the
    curve meets a face. That structure is what this collapses:

    1. the two LONG LINES are the two longest edges;
    2. they cut the loop into two arcs; a CLUSTER is the run of vertices at an arc's end that
       stays within `cluster_tol` of the long line it touches (if the two runs of one arc meet,
       the whole arc is one cluster - that happens at the corner where the two faces meet);
    3. each cluster collapses to its mean;
    4. of what is left, every `keep_every`-th vertex survives;
    5. a periodic-free cubic spline runs through cluster mean -> survivors -> cluster mean, and
       the long lines close the loop between the cluster means.

    Returns everything the plot needs plus `curve`, the resulting closed outline.
    """
    from scipy.interpolate import splev, splprep

    n = len(outer)
    e = np.roll(outer, -1, axis=0) - outer
    length = np.hypot(e[:, 0], e[:, 1])
    i1, i2 = sorted(int(k) for k in np.argsort(-length)[:2])      # edge i runs vertex i -> i+1

    def off(line_i, idx):
        """perpendicular distance of vertex idx from the infinite line through edge line_i"""
        d = e[line_i] / length[line_i]
        v = outer[idx] - outer[line_i]
        return abs(v[0] * d[1] - v[1] * d[0])

    # the two long edges split the cyclic vertex list into two arcs
    arcs = [[(i1 + 1 + k) % n for k in range((i2 - i1) % n)],      # head of edge i1 .. tail of i2
            [(i2 + 1 + k) % n for k in range((i1 - i2) % n)]]      # head of edge i2 .. tail of i1
    head_line = [i1, i2]           # the long edge that ends at each arc's first vertex
    tail_line = [i2, i1]           # the long edge that starts at each arc's last vertex

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

    # rebuild the loop: per arc, either one cluster mean, or mean -> spline -> mean
    pieces, marks = [], {"reps": [], "kept": [], "cluster_idx": [], "loose_idx": []}
    ci = 0
    for k, arc in enumerate(arcs):
        if not loose[k]:
            rep = outer[clusters[ci]].mean(axis=0)
            marks["reps"].append(rep)
            marks["cluster_idx"].append(clusters[ci])
            pieces.append(rep[None, :])
            ci += 1
            continue
        r0 = outer[clusters[ci]].mean(axis=0)
        r1 = outer[clusters[ci + 1]].mean(axis=0)
        marks["reps"] += [r0, r1]
        marks["cluster_idx"] += [clusters[ci], clusters[ci + 1]]
        marks["loose_idx"] += loose[k]
        kept_idx = loose[k][::keep_every]
        kept = outer[kept_idx]
        marks["kept"].append(kept)
        pts = np.vstack([r0, kept, r1])
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0.0, k=min(3, len(pts) - 1), per=False)
        u = np.linspace(0.0, 1.0, n_spline)
        pieces.append(np.column_stack(splev(u, tck)))
        ci += 2
    curve = np.vstack(pieces)                       # closed by the straight long lines between pieces

    # the same thing as a closed loop of corner NODES joined by edges, which is what the .geo
    # needs: every arc contributes one node (a knot) or two with a spline between them, and the
    # long line to the next arc follows
    nodes, edges = [], []
    ci = 0
    for k, arc in enumerate(arcs):
        if not loose[k]:
            nodes.append(outer[clusters[ci]].mean(axis=0))
            ci += 1
        else:
            nodes.append(outer[clusters[ci]].mean(axis=0))
            edges.append(("spline", outer[loose[k][::keep_every]]))
            nodes.append(outer[clusters[ci + 1]].mean(axis=0))
            ci += 2
        edges.append(("line", None))                # the long line on to the next arc

    marks["reps"] = np.array(marks["reps"])
    marks["kept"] = np.vstack(marks["kept"]) if marks["kept"] else np.zeros((0, 2))
    return {"long_edges": (i1, i2), "arcs": arcs, "clusters": clusters, "loose": loose,
            "curve": curve, "cluster_tol": cluster_tol, "keep_every": keep_every,
            "given": False, "nodes": np.array(nodes), "edges": edges, **marks}


def outline_as_given(ol: dict, n_spline: int = 600) -> dict:
    """The outline as the spiral inflector already built it - nothing to simplify.

    It now writes a `curves` section saying how to join its points: one straight plate at the
    exit, square to the design orbit, and one spline round the rest. Simplifying that again
    here would be simplifying a simplification, with two different sets of rules; this just
    unpacks it into the same shape `simplify_outline` returns, so everything downstream is
    unchanged.

    Note the outline is NOT convex any more: the exit plate is a chord across the electrode
    hull, so there is a re-entrant kink of a few degrees at each of its ends.
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
            "loose": [], "reps": np.array(nodes), "kept": np.zeros((0, 2)),
            "cluster_idx": [], "loose_idx": []}


def print_given(ol: dict, sim: dict) -> None:
    """What came out of the spiral inflector, for an outline it simplified itself."""
    outer, curve = np.asarray(ol['outer'], dtype=float), sim['curve']
    print()
    print("=" * 96)
    print("HOUSING OUTLINE, AS THE SPIRAL INFLECTOR BUILT IT")
    print("=" * 96)
    print(f"  the file carries its own curves, so nothing is simplified here.")
    for k, (kind, i0, i1) in enumerate(ol['curves']):
        n = len(outer)
        m = (i1 - i0) % n + 1
        a, b = outer[i0], outer[i1]
        print(f"    curve {k}: {kind:6s} points {i0:3d} .. {i1:3d} ({m:3d} of them), "
              f"({a[0]:+8.3f},{a[1]:+8.3f}) -> ({b[0]:+8.3f},{b[1]:+8.3f}), "
              f"{np.hypot(*(b - a)):7.3f} mm end to end")
        if kind == 'line':
            d = (b - a) / np.linalg.norm(b - a)
            print(f"             direction {np.degrees(np.arctan2(d[1], d[0])) % 180.0:7.3f} deg, "
                  f"outward normal {np.degrees(np.arctan2(-d[0], d[1])) % 360.0:7.3f} deg")
    e = np.roll(curve, -1, axis=0) - curve
    r = np.hypot(curve[:, 0], curve[:, 1])
    print(f"  sampled at {len(curve)} points: perimeter {np.hypot(e[:, 0], e[:, 1]).sum():.3f} mm, "
          f"radius {r.min():.3f} .. {r.max():.3f} mm")
    dev = outline_deviation(outer, curve)
    print(f"  it bulges up to {np.abs(dev).max():.3f} mm past the straight lines between its own "
          f"points - that is the spline's own curvature between them, not an error")


CHANNEL_BACK = 1.0      # [mm] how far UPSTREAM of the emitting plane the beam channel starts.
#                         The particles are born on that plane, so it must not also be the metal
#                         face that closes the channel - see beam_channel's docstring. A clearance
#                         is all that is needed: the fault is coincidence, not proximity, and 1 mm
#                         is two cells of the 0.5 mm obstacle raster. The beam is not moved.


def emitting_plane(handoff_path: Path) -> dict:
    """The plane the central region's particles are emitted from, measured from the hand-off file.

    handoff.py releases every particle ON the plane where the spiral inflector recorded it, so the
    hand-off cloud IS the emitting plane - no need to take anyone's word for where it is. The
    normal is the cloud's degenerate singular direction, turned to point downstream.

    NOTE this is NOT exit_plane_spec.json's plane: that one (the exit plate's outer face pushed by
    half a gap) sits 8.9 mm upstream and is tilted ~10 deg to this one.
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
    """Shrink the housing's exit aperture, symmetrically, by a margin taken off EACH side.

    The four corners come from the spiral-inflector export and are a rectangle in space. This
    keeps the rectangle's centre, plane and orientation and only pulls the edges in: `long_side`
    mm off each end of the longer edge, `short_side` mm off each end of the shorter one. So
    `long_side=10` turns the 40 mm dimension into 20 mm.

    Why this is a jaw and not just a collimator: gap 0's counter-post IS the housing (Daniel,
    2026-09-17), so the aperture's long edges play the part the facing post pairs play at gaps
    1 and up, and narrowing them is the gap-0 equivalent of tightening those jaws.
    """
    ap = np.asarray(aperture, dtype=float)
    c = ap.mean(axis=0)
    e1, e2 = ap[1] - ap[0], ap[2] - ap[1]
    l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    if min(l1, l2) < 1e-9:
        raise SystemExit("degenerate aperture: cannot inset")
    u1, u2 = e1 / l1, e2 / l2
    # whichever edge is longer takes `long_side`
    d1, d2 = (long_side, short_side) if l1 >= l2 else (short_side, long_side)
    if l1 - 2.0 * d1 <= 0.0 or l2 - 2.0 * d2 <= 0.0:
        raise SystemExit(f"--aperture-inset closes the aperture: {l1:.3f} x {l2:.3f} mm "
                         f"minus 2x({d1:.3f}, {d2:.3f})")
    out = []
    for q in ap:
        v = q - c
        s1 = np.sign(float(v.dot(u1))) or 1.0
        s2 = np.sign(float(v.dot(u2))) or 1.0
        out.append(c + s1 * (0.5 * l1 - d1) * u1 + s2 * (0.5 * l2 - d2) * u2)
    out = np.asarray(out, dtype=float)
    print(f"  aperture inset: {l1:.3f} x {l2:.3f} -> "
          f"{l1 - 2.0 * d1:.3f} x {l2 - 2.0 * d2:.3f} mm "
          f"({d1:.3f} / {d2:.3f} mm off each side)")
    return out


def beam_channel(aperture: np.ndarray, plane: dict, extend: float = 200.0,
                 back: float = CHANNEL_BACK) -> dict:
    """The beam channel: the aperture's footprint on the emitting plane, swept downstream.

    1. each aperture corner is projected NORMALLY onto the emitting plane - straight down onto
       it, along the plane's own normal;
    2. the four points are joined into a rectangle. They are exactly coplanar by construction,
       and near-exactly rectangular: the aperture plane is only ~1.2 deg off the emitting plane,
       so the projection shortens an edge by under 10 um;
    3. that rectangle is swept `extend` mm along the SAME normal, in the direction the beam
       travels (emitting_plane already turned the normal to agree with the mean momentum), and
       `back` mm the other way.

    So this is a right prism: both end faces parallel to the emitting plane, length exactly
    `back + extend`. That is the difference from the earlier version, whose far face was the
    tilted aperture pushed along the normal.

    Why `back` is not zero
    ----------------------
    The particles are released ON the emitting plane. With back = 0 that plane IS the prism's
    closed end, so every particle is born exactly on a metal boundary - and the obstacle
    rasteriser, which decides metal by even-odd point-in-polygon on z slices, splits a
    coincident face about evenly between metal and void. Half the bunch then reads as buried
    and the rest is killed within its first fraction of a millimetre, which is what emptied the
    first 15-turn run (51.8 % of the channel face metal at the plane, 3.2 % five mm downstream,
    0.0 % at twenty).

    So the base rectangle is set back into the housing by a clearance. The fault is coincidence,
    not proximity, so the clearance only has to be larger than the raster cell; the beam itself
    is not moved. The volume this frees was open anyway - the real aperture plane is 9.3 .. 9.9
    mm further back still, and the throat between the two is space the beam flies through.
    """
    p0, n = plane['point'], plane['normal']
    d = (aperture - p0).dot(n)                       # signed distance of each corner
    near = aperture - d[:, None] * n                 # the rectangle, on the emitting plane
    far = near + float(extend) * n
    start = near - float(back) * n                   # the prism's actual upstream face

    e = np.roll(near, -1, axis=0) - near
    lens = np.linalg.norm(e, axis=1)
    u = e / lens[:, None]
    ang = np.degrees(np.arccos(np.clip([-u[k - 1].dot(u[k]) for k in range(4)], -1.0, 1.0)))
    area = 0.5 * float(np.linalg.norm(np.cross(near[2] - near[0], near[3] - near[1])))
    return {'aperture': aperture, 'near': near, 'far': far, 'start': start, 'dist': d,
            'normal': n, 'sweep': n, 'extend': float(extend), 'back': float(back),
            'edges': lens, 'angles': ang, 'face_area': area,
            'ap_edges': np.linalg.norm(np.roll(aperture, -1, axis=0) - aperture, axis=1)}


def geo_string(sim: dict, half_height: float = 50.0, brick: dict = None, lc: float = 3.0) -> str:
    """The simplified outline as a gmsh .geo curve loop: one Spline, two Lines.

    gmsh's OpenCASCADE `Spline` interpolates the points it is given, like the scipy spline the
    knots were chosen with - but it is not the SAME interpolant, so the curve gmsh builds has to
    be measured on its own (open_in_gmsh does that) rather than assumed to match.
    """
    nodes, edges = sim["nodes"], sim["edges"]
    z0 = -float(half_height)
    how = (['//   the outline exactly as the spiral inflector wrote it: {} curve(s), {} point(s),'
            .format(len(sim["edges"]), len(sim["nodes"]) + sum(
                0 if k != "spline" else len(p) for k, p in sim["edges"])),
            '//   no second simplification here.']
           if sim.get("given") else
           ['//   the two long straight faces stay straight, everything between them is one spline',
            '//   through every {}th hull vertex, and each knot of nearly collinear hull points'
            .format(sim["keep_every"]),
            '//   collapsed to its mean.'])
    out = ['// HCHC-60 central region: the housing wall, and the beam channel through its aperture.',
           '// Built by cr_geom_build.py from the spiral inflector\'s housing_outline.txt:',
           *how,
           '//   The loop is then extruded over z = {:+.1f} .. {:+.1f} mm as'.format(z0, -z0),
           '//   SURFACES only: no top or bottom plate, radial ends open, exactly as the real housing.',
           '// Coordinates are MILLIMETRES in the baseline (machine) frame.',
           'SetFactory("OpenCASCADE");',
           'lc = {:g};   // placeholder characteristic length - nothing is meshed here yet'.format(lc),
           '']
    # every node is a Point; a spline edge also needs its interior knots as Points
    tag, pts, node_tag, edge_pts = 0, [], [], []
    for i, nd in enumerate(nodes):
        tag += 1
        node_tag.append(tag)
        pts.append((tag, nd, 'loop node {}'.format(i)))
    for k, (kind, knots) in enumerate(edges):
        if kind != 'spline':
            edge_pts.append(None)
            continue
        tags = []
        for knot in knots:
            tag += 1
            tags.append(tag)
            pts.append((tag, knot, 'spline knot'))
        edge_pts.append(tags)

    out.append('// ---- 1. the housing outline, at the bottom of the extrusion --------------------')
    out.append('// {} loop nodes (cluster means) + {} spline knots'.format(len(nodes), tag - len(nodes)))
    for t, p, what in pts:
        out.append('Point({}) = {{ {:.6f}, {:.6f}, {:.6f}, lc }};   // {}'.format(t, p[0], p[1], z0, what))

    # the brick's 8 corners, defined BEFORE the extrusion so its automatic tags cannot collide
    b_pt = {}
    if brick is not None:
        out.append('')
        out.append('// ---- 2. the beam channel ------------------------------------------------------')
        out.append('//   the four aperture corners projected NORMALLY onto the plane the central region')
        out.append('//   emits its particles from, joined into a rectangle, then swept {:.1f} mm along'
                   .format(brick['extend']))
        out.append('//   that same normal ({:+.6f}, {:+.6f}, {:+.6f}), downstream.'.format(*brick['sweep']))
        out.append('//   Nine decimal places because the four points must stay coplanar for the')
        out.append('//   Plane Surface: rounding to um would put them ~5e-7 mm out, at OCC tolerance.')
        for j, p in enumerate(brick['near']):
            tag += 1
            b_pt[j] = tag
            out.append('Point({}) = {{ {:.9f}, {:.9f}, {:.9f}, lc }};   // corner {} on the emitting '
                       'plane ({:.3f} mm before the aperture)'.format(tag, p[0], p[1], p[2], j,
                                                                      brick['dist'][j]))

    out.append('')
    out.append('// ---- curves ------------------------------------------------------------------')
    ctag, loop = 0, []
    n_nodes = len(nodes)
    for ni, (kind, _) in enumerate(edges):
        a, b = node_tag[ni % n_nodes], node_tag[(ni + 1) % n_nodes]
        ctag += 1
        loop.append(ctag)
        if kind == 'spline':
            chain = ', '.join(str(t) for t in ([a] + edge_pts[ni] + [b]))
            out.append('Spline({}) = {{ {} }};'.format(ctag, chain))
        else:
            out.append('Line({}) = {{ {}, {} }};'.format(ctag, a, b))
    wall_curves = list(loop)
    out.append('Curve Loop(1) = {{ {} }};'.format(', '.join(str(c) for c in loop)))
    out.append('Plane Surface(1) = { 1 };   // the housing cross-section, extruded to a solid below')

    if brick is not None:
        rect = []
        for j in range(4):
            ctag += 1
            rect.append(ctag)
            out.append('Line({}) = {{ {}, {} }};   // rectangle edge {}, {:.4f} mm'
                       .format(ctag, b_pt[j], b_pt[(j + 1) % 4], j, brick['edges'][j]))
        out.append('Curve Loop(11) = {{ {} }};'.format(', '.join(str(c) for c in rect)))
        out.append('Plane Surface(101) = { 11 };   // the rectangle, on the emitting plane')

    out.append('')
    out.append('// ---- the extrusions, LAST so their automatic tags collide with nothing above --')
    out.append('// A surface Extrude returns [0] the far surface, [1] the VOLUME, [2..] the sides.')
    out.append('house[] = Extrude {{ 0, 0, {:g} }} {{ Surface{{ 1 }}; }};   '
               '// the housing prism, z {:+.1f} .. {:+.1f}'.format(-2.0 * z0, z0, -z0))
    if brick is not None:
        d = brick['extend'] * brick['sweep']
        out.append('chan[]  = Extrude {{ {:.9f}, {:.9f}, {:.9f} }} {{ Surface{{ 101 }}; }};   '
                   '// the beam channel, {:.1f} mm downstream'.format(d[0], d[1], d[2], brick['extend']))
        out.append('')
        out.append('// ---- cut the channel out of the housing --------------------------------------')
        out.append('// The channel starts on the emitting plane, INSIDE the prism, and runs out through')
        out.append('// the wall, so what it leaves behind is a rectangular recess: blind at the inner')
        out.append('// end (that end face IS the emitting-plane rectangle), open where it breaks out.')
        out.append('// Drop the second Delete to keep the channel as its own volume as well.')
        out.append('housing[] = BooleanDifference{ Volume{ house[1] }; Delete; }'
                   '{ Volume{ chan[1] }; Delete; };')
    out.append('')
    return '\n'.join(out)


def open_in_gmsh(geo_path: Path, outer: np.ndarray, gui: bool = True, n_sample: int = 400,
                 half_height: float = None) -> None:
    """Load the .geo, report what OCC actually built, and open the GUI."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.open(str(geo_path))
    gmsh.model.occ.synchronize()

    curves, surfaces, volumes = (gmsh.model.getEntities(d) for d in (1, 2, 3))
    print()
    print("=" * 96)
    print("GMSH")
    print("=" * 96)
    print(f"  {geo_path}")
    print(f"  {len(gmsh.model.getEntities(0))} points, {len(curves)} curves, "
          f"{len(surfaces)} surfaces, {len(volumes)} volumes")
    zb = -float(half_height) if half_height is not None else None

    def flat_at(dim, tag, z):
        bb = gmsh.model.getBoundingBox(dim, tag)
        return abs(bb[2] - z) < 1e-6 and abs(bb[5] - z) < 1e-6

    for d, t in volumes:
        vol = gmsh.model.occ.getMass(d, t)
        faces = [abs(f) for _, f in gmsh.model.getBoundary([(d, t)], oriented=False)]
        com = gmsh.model.occ.getCenterOfMass(d, t)
        bb = gmsh.model.getBoundingBox(d, t)
        print(f"  volume {t}: {vol:.3f} mm^3, {len(faces)} faces, "
              f"centre of mass ({com[0]:+.3f}, {com[1]:+.3f}, {com[2]:+.3f}) mm")
        print(f"    bbox x {bb[0]:+8.3f}..{bb[3]:+8.3f}  y {bb[1]:+8.3f}..{bb[4]:+8.3f}  "
              f"z {bb[2]:+8.3f}..{bb[5]:+8.3f} mm")
        # the bottom cap is the full cross-section: the recess spans z -20..+18 and never reaches
        # it, so area x height is the UNCUT prism exactly, and the shortfall is the recess
        caps = [f for f in faces if zb is not None and flat_at(2, f, zb)]
        if caps and half_height is not None:
            cross = sum(gmsh.model.occ.getMass(2, f) for f in caps)
            uncut = cross * 2.0 * half_height
            print(f"    cross-section {cross:.3f} mm^2 x {2.0 * half_height:.1f} mm = {uncut:.3f} mm^3 "
                  f"uncut, so the recess took {uncut - vol:.3f} mm^3 "
                  f"({100.0 * (uncut - vol) / uncut:.3f} %)")
            print( "      ^ only good to ~0.02 % of the prism (~150 mm^3): OCC integrates the")
            print( "        spline-bounded solid numerically. Take the recess from the exact")
            print( "        planar figure above, not from this difference.")
            # the outline: the boundary curves of that bottom cap
            sampled = []
            for _, c in gmsh.model.getBoundary([(2, caps[0])], oriented=False):
                lo, hi = gmsh.model.getParametrizationBounds(1, abs(c))
                ts = np.linspace(float(lo[0]), float(hi[0]), n_sample)
                p = np.asarray(gmsh.model.getValue(1, abs(c), ts)).reshape(-1, 3)
                print(f"    outline curve {abs(c)}: {gmsh.model.getType(1, abs(c)):<12s} "
                      f"length {np.hypot(*np.diff(p[:, :2], axis=0).T).sum():8.3f} mm")
                sampled.append(p[:, :2])
            if sampled:
                dev = outline_deviation(outer, np.vstack(sampled))
                print(f"    against the original 94-gon: {dev.min():+.4f} .. {dev.max():+.4f} mm; "
                      f"worst inward {(-dev.min()):.4f} mm -> 3.000 mm clearance becomes "
                      f"{3.0 + dev.min():.3f} mm")
        for f in faces:
            nb = len(gmsh.model.getBoundary([(2, f)], oriented=False))
            fb = gmsh.model.getBoundingBox(2, f)
            note = ''
            if zb is not None and flat_at(2, f, zb):
                note = '   bottom cap'
            elif zb is not None and flat_at(2, f, -zb):
                note = '   top cap'
            elif nb > 4:
                note = '   <- broken through by the recess'
            print(f"      face {f:>3}: {gmsh.model.occ.getMass(2, f):10.3f} mm^2, {nb} curves, "
                  f"z {fb[2]:+7.2f}..{fb[5]:+7.2f}{note}")
    # anything not attached to a volume (there should be nothing once the boolean has run)
    owned = {t for d, t in volumes for _, t in gmsh.model.getBoundary([(d, t)], oriented=False)}
    loose = [t for _, t in surfaces if abs(t) not in {abs(v) for v in owned}]
    if loose:
        print(f"  {len(loose)} surface(s) not attached to any volume: {loose}")
    if gui:
        print("  opening the gmsh GUI (close the window to continue) ...")
        gmsh.fltk.run()
    gmsh.finalize()


def outline_deviation(outer: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """Signed distance of every point of the simplified outline from the original polygon.

    Measured this way round on purpose: the curve is sampled far more finely than the 94-gon, so
    a bulge BETWEEN two original vertices is caught, which comparing the vertices to the curve
    would miss. + = the new outline is outside the old one (the housing grew, it reaches further
    into the central region), - = inside it (the housing shrank, and so did the 3 mm clearance to
    the electrodes).
    """
    from matplotlib.path import Path
    seg = np.roll(outer, -1, axis=0) - outer
    ll = np.maximum(np.einsum('ij,ij->i', seg, seg), 1e-30)
    d = np.empty(len(curve))
    for i, p in enumerate(curve):
        t = np.clip(np.einsum('ij,ij->i', p - outer, seg) / ll, 0.0, 1.0)
        d[i] = np.min(np.hypot(*(outer + t[:, None] * seg - p).T))
    return np.where(Path(outer).contains_points(curve), -d, d)


def plot_simplified(ol: dict, sim: dict, out: str = None) -> None:
    """Left: the classified vertices. Right: the housing that comes out of them, same axes."""
    import matplotlib
    if out:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    outer, curve = ol['outer'], sim['curve']
    i1, i2 = sim['long_edges']
    n = len(outer)
    in_cluster = np.zeros(n, dtype=bool)
    for c in sim['clusters']:
        in_cluster[c] = True

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.0, 7.2), sharex=True, sharey=True)
    ring = np.vstack([outer, outer[0]])

    # ---- left: the original vertices, classified ---------------------------------------
    ax.plot(ring[:, 0], ring[:, 1], '-', color='0.85', lw=1.0, zorder=1)
    for k, li in enumerate((i1, i2)):
        a, b = outer[li], outer[(li + 1) % n]
        ax.plot([a[0], b[0]], [a[1], b[1]], '-', color='0.35', lw=2.6, zorder=2,
                label='the two long lines' if k == 0 else None)
    ax.plot(outer[~in_cluster, 0], outer[~in_cluster, 1], 'o', color='tab:orange', ms=5, zorder=4,
            label='not in a cluster ({})'.format(int((~in_cluster).sum())))
    ax.plot(outer[in_cluster, 0], outer[in_cluster, 1], 'o', color='tab:blue', ms=6, zorder=5,
            label='in a cluster ({} in {})'.format(int(in_cluster.sum()), len(sim['clusters'])))
    ax.set_title('the outline as exported: {} vertices\ncluster = within {:.2f} mm of a long line'
                 .format(n, sim['cluster_tol']), fontsize=11)
    ax.legend(loc='lower left', fontsize=8)

    # ---- right: the housing that comes out of it ---------------------------------------
    bx.plot(ring[:, 0], ring[:, 1], '-', color='0.85', lw=1.0, zorder=1, label='original 94-gon')
    closed = np.vstack([curve, curve[0]])
    bx.plot(closed[:, 0], closed[:, 1], '-', color='tab:red', lw=2.0, zorder=3,
            label='simplified housing')
    bx.plot(sim['kept'][:, 0], sim['kept'][:, 1], 'o', color='tab:orange', ms=5, zorder=4,
            label='every {}th loose vertex ({})'.format(sim['keep_every'], len(sim['kept'])))
    bx.plot(sim['reps'][:, 0], sim['reps'][:, 1], 'o', color='tab:blue', ms=7, zorder=5,
            label='cluster means ({})'.format(len(sim['reps'])))
    dev = outline_deviation(outer, curve)
    bx.set_title('spline through the survivors + the two straight faces\n'
                 'new outline vs the old: {:+.3f} .. {:+.3f} mm (- = shrank, clearance lost)'
                 .format(dev.min(), dev.max()), fontsize=11)
    bx.legend(loc='lower left', fontsize=8)

    r = np.hypot(outer[:, 0], outer[:, 1]).max()
    for axis in (ax, bx):
        axis.set_aspect('equal')
        axis.set_xlabel('x [mm]')
        axis.grid(alpha=0.25)
    ax.set_ylabel('y [mm]')
    ax.set_xlim(-0.35 * r, 1.25 * r)
    ax.set_ylim(-0.95 * r, 0.75 * r)

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140)
        print(f"  wrote {out}")
    else:
        plt.show()


def print_simplified(ol: dict, sim: dict) -> None:
    outer = ol['outer']
    i1, i2 = sim['long_edges']
    n = len(outer)
    e = np.roll(outer, -1, axis=0) - outer
    print()
    print("=" * 96)
    print("SIMPLIFIED HOUSING")
    print("=" * 96)
    for li in (i1, i2):
        print(f"  long line {li:>3} -> {(li + 1) % n:<3} : {np.hypot(*e[li]):8.3f} mm  "
              f"({outer[li, 0]:+8.3f}, {outer[li, 1]:+8.3f}) -> ({outer[(li + 1) % n, 0]:+8.3f}, "
              f"{outer[(li + 1) % n, 1]:+8.3f})")
    print(f"  cluster tolerance {sim['cluster_tol']:.2f} mm -> {len(sim['clusters'])} clusters, "
          f"{sum(len(c) for c in sim['clusters'])} of {n} vertices")
    for c, rep in zip(sim['cluster_idx'], sim['reps']):
        span = float(np.max([np.hypot(*(outer[a] - outer[b])) for a in c for b in c])) if len(c) > 1 else 0.0
        print(f"    vertices {c[0]:>3}..{c[-1]:<3} ({len(c):>2} pts, spanning {span:6.3f} mm) "
              f"-> ({rep[0]:+8.3f}, {rep[1]:+8.3f})")
    print(f"  loose vertices {sum(len(x) for x in sim['loose'])}, every {sim['keep_every']}th kept "
          f"-> {len(sim['kept'])} spline knots (+ the 2 cluster means at the ends)")
    dev = outline_deviation(outer, sim['curve'])
    print(f"  the new outline against the old, over {len(dev)} samples: {dev.min():+.4f} .. {dev.max():+.4f} mm "
          f"(+ = grew outwards, - = shrank)")
    print(f"    worst inward move {(-dev.min()):.4f} mm -> the 3.000 mm clearance to the electrodes "
          f"becomes {3.0 + dev.min():.3f} mm at that point")
    print(f"  perimeter {np.hypot(*(np.roll(sim['curve'], -1, axis=0) - sim['curve']).T).sum():.3f} mm "
          f"(was {np.hypot(*e.T).sum():.3f} mm)")


def recess_on_wall(brick: dict, sim: dict) -> dict:
    """Where the channel breaks out of the housing, and how big the recess actually is.

    The channel is swept along the emitting-plane normal, but the wall it leaves through is a
    different plane - so its opening is the aperture's SHADOW along that normal, not the aperture
    itself, and the four corners break out at quite different depths. Everything here is planar,
    so these are exact; the recess volume from gmsh's (uncut - cut) is not, because OCC integrates
    the spline-bounded solid numerically and is ~0.02 % out, which is over 1 % of the recess.
    """
    n, nodes = brick['sweep'], sim['nodes']
    c = brick['aperture'].mean(axis=0)[:2]
    best = None
    for i in range(len(nodes)):
        a, b = nodes[i], nodes[(i + 1) % len(nodes)]
        t = np.clip(np.dot(c - a, b - a) / max(np.dot(b - a, b - a), 1e-30), 0.0, 1.0)
        d = float(np.linalg.norm(c - (a + t * (b - a))))
        if best is None or d < best[0]:
            best = (d, a, b)
    _, a, b = best
    e = b - a
    wn = np.array([e[1], -e[0], 0.0])
    wn /= np.linalg.norm(wn)
    if wn.dot(n) < 0.0:
        wn = -wn                                          # outward, in the sweep direction
    w0 = np.array([a[0], a[1], 0.0])
    depth = np.array([float((w0 - p).dot(wn) / n.dot(wn)) for p in brick['near']])
    return {'wall_a': a, 'wall_b': b, 'wall_normal': wn, 'depth': depth,
            'volume': float(depth.mean() * brick['face_area']),
            'opening': float(brick['face_area'] / abs(n.dot(wn))),
            'tilt': float(np.degrees(np.arccos(np.clip(abs(n.dot(wn)), -1.0, 1.0))))}


# =============================================================================================
# 1c. the dees - STEP A: the boundary chains only
# =============================================================================================
R_MAX_DEE_MM = 305.0    # where the dees stop. The gap centre lines run to 600 mm (the cavity
                        # r_max the DeeSystem is built with), far beyond anything the central
                        # region models; the old builder cropped at 305 mm (the 250 mm seam,
                        # plus a 25 mm blend and a 30 mm margin). --r-max overrides.


def _clip_outer(chain: np.ndarray, r_max: float) -> np.ndarray:
    """Cut an inner-to-outer chain at r_max, landing the last point exactly on that circle."""
    p = np.asarray(chain, dtype=float)
    r = np.hypot(p[:, 0], p[:, 1])
    if r[-1] <= r_max:
        return p
    k = int(np.argmax(r > r_max))
    if k == 0:
        raise ValueError(f"chain starts beyond r_max = {r_max:.1f} mm")
    a, b = p[k - 1], p[k]
    ab = b - a
    A = float(ab.dot(ab))
    B = 2.0 * float(a.dot(ab))
    C = float(a.dot(a)) - r_max ** 2
    s = (-B + np.sqrt(max(B * B - 4.0 * A * C, 0.0))) / (2.0 * A)
    return np.vstack([p[:k], a + min(max(s, 0.0), 1.0) * ab])


JUNCTION_TOL_DEG = 1.0   # close the variable->fixed junction when the net angular step is
                         # under this. Above it the two lines only meet far inside the geometry.


def junction(cav):
    """The variable -> fixed junction of one gap: the step, and where the lines would meet.

    RFCavity._build_geometry ends the variable chain at base_angle + sum(excursions), rotates the
    last segment about its midpoint, and then starts the FIXED segment as a RADIAL ray at the
    NOMINAL base_angle at the last variable radius. The junction therefore carries a step that is
    partly angular (the leftover excursion) and partly radial (the rotation moves the endpoint off
    its nominal circle). It is left there deliberately - the comment in _build_geometry calls it
    the same thin-gap idiom as the rotations - but a dee built from these lines inherits it as a
    visible jog in its edge.

    Returns the two junction points, the net angular step, the jog, and the point where the last
    variable segment's LINE crosses the fixed ray. All in mm.
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
    d_az = float((np.degrees(np.arctan2(b[1], b[0])) - np.degrees(np.arctan2(f1[1], f1[0]))
                  + 180.0) % 360.0 - 180.0)
    return {'var_end': b, 'fix_start': f1, 'fix_end': f2, 'r_trans': float(np.hypot(*f1)),
            'jog': float(np.hypot(*(b - f1))), 'step_deg': d_az, 'meet': meet, 'r_meet': r_meet,
            'dr': r_meet - float(np.hypot(*f1)),
            'seg_deg': float(np.degrees(np.arccos(np.clip(abs(u.dot(v)), -1.0, 1.0))))}


def gap_centre_line(cav, junction_tol: float = JUNCTION_TOL_DEG, r_max: float = None):
    """The gap centre line in mm, with the variable->fixed junction closed where it can be.

    When the net angular step is under `junction_tol`, the transition radius is RELAXED: the last
    variable segment is run on (or back) to where its line actually crosses the fixed ray, and the
    fixed ray starts from that same point. The jog disappears and the chain is continuous, at the
    price of moving the transition radius by `dr`. Above the tolerance nothing is touched - the
    lines would only meet far inside the geometry.

    Returns (chain, junction dict, closed?).
    """
    j = junction(cav)
    var = [s for s in cav.segments if s['type'] == 'variable']
    pts = [np.asarray(var[0]['p1'][:2], dtype=float) * MM]
    for s in var[:-1]:
        pts += [np.asarray(s['p2'][:2], dtype=float) * MM,
                np.asarray(s['p1'][:2], dtype=float) * MM]
    pts.append(np.asarray(var[-1]['p1'][:2], dtype=float) * MM)

    close = (abs(j['step_deg']) <= junction_tol and j['meet'] is not None
             and np.isfinite(j['r_meet']))
    if close:
        chain = np.array(pts + [j['meet'], j['fix_end']])
    else:
        chain = np.array(pts + [j['var_end'], j['fix_start'], j['fix_end']])
    # drop any duplicate the splice may have produced
    keep = [chain[0]]
    for p in chain[1:]:
        if np.hypot(*(p - keep[-1])) > 1e-9:
            keep.append(p)
    chain = np.array(keep)
    return (_clip_outer(chain, r_max) if r_max else chain), j, close


def print_junctions(dees: DeeSystem, junction_tol: float = JUNCTION_TOL_DEG) -> None:
    print()
    print("=" * 96)
    print("GAP CENTRE LINES - the variable -> fixed junction")
    print("=" * 96)
    print("  The fixed segment is a RADIAL ray at the nominal base angle; the variable chain ends")
    print("  at base angle + excursions, rotated about its midpoint. The leftover is a jog.")
    print(f"  Closing it where the net angular step is <= {junction_tol:.2f} deg.")
    print()
    print(f"  {'gap':>4} {'excursion':>10} {'rotation':>9} {'net step':>9} {'jog':>8} "
          f"{'r_trans':>9} {'-> r_meet':>10} {'dr':>9} {'seg/ray':>8}  action")
    print(f"  {'':>4} {'[deg]':>10} {'[deg]':>9} {'[deg]':>9} {'[mm]':>8} "
          f"{'[mm]':>9} {'[mm]':>10} {'[mm]':>9} {'[deg]':>8}")
    print("  " + "-" * 94)
    for g, cav in enumerate(dees.gaps):
        j = junction(cav)
        nominal = j['jog'] < 1e-6
        close = abs(j['step_deg']) <= junction_tol
        r_meet = '     -    ' if nominal or not np.isfinite(j['r_meet']) else f"{j['r_meet']:10.3f}"
        dr = '    -    ' if nominal or not np.isfinite(j['dr']) else f"{j['dr']:+9.3f}"
        print(f"  {g:>4} {float(np.sum(cav.segment_angles)):10.4f} "
              f"{(float(cav.segment_rotations[-1]) if cav.segment_rotations else 0.0):9.4f} "
              f"{j['step_deg']:9.4f} {j['jog']:8.3f} {j['r_trans']:9.3f} {r_meet} "
              f"{dr} {j['seg_deg']:8.3f}  "
              + ("nominal - collinear, no junction" if nominal
                 else "CLOSED" if close else "left as it is"))


CAVITY_CLEARANCE_INNER = 10.0   # [mm] how close the dee METAL may come to the housing


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
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(chain, axis=0).T))])
    ss = np.linspace(0.0, s[-1], max(2, int(s[-1] / ds) + 1))
    return np.column_stack([np.interp(ss, s, chain[:, 0]), np.interp(ss, s, chain[:, 1])]), ss, s


def _reach_inner(chain: np.ndarray, wall: np.ndarray, target: float, step: float = 0.25,
                 max_march: float = 300.0):
    """Take a chain's inner end to where its housing distance equals `target`, either way.

    Cuts it back when it starts too far in, and EXTENDS it along its own first segment when it
    starts short. Gap 0 needs the extension: the -40 deg rotation of its last variable segment
    leaves its inner end already outside the housing, so there is nothing to cut.
    """
    p = np.asarray(chain, dtype=float)
    if _wall_distance(p[:1], wall)[0] > target:
        t = np.arange(step, max_march, step)
        # Along its own first segment first. That fails for gap 0: its line runs nearly PARALLEL
        # to the housing's long flat face, so marching along it never crosses in. Fall back to
        # heading straight for the housing centroid, which always does - the extension is buried
        # inside the housing and disappears in the fuse.
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
    """Drop the inner part of a chain that comes closer than `clearance` to the housing.

    Every gap centre line starts inside the housing (the origin is inside its outline), so walking
    outward the distance rises through zero and on past the clearance; the chain is cut at the
    first point that clears it.
    """
    dn, ss, s = _densify(chain)
    d = _wall_distance(dn, wall)
    ok = np.where(d >= clearance)[0]
    if not len(ok):
        raise ValueError("the whole chain is within the clearance of the housing")
    if ok[0] == 0:
        return chain, float(d[0]), False                     # already clear at its inner end
    s_cut = ss[ok[0]]
    keep = chain[np.searchsorted(s, s_cut):]
    return np.vstack([dn[ok[0]], keep]), float(d[ok[0]]), True


def _first_crossing(a: np.ndarray, b: np.ndarray):
    """First point (walking a from its inner end) where polyline a crosses polyline b."""
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
                return p + ta * r, i
    return None, None


def _clip_inner(chain: np.ndarray, r_min: float) -> np.ndarray:
    """Cut an inner-to-outer chain back to r_min, landing its first point exactly on that circle."""
    p = np.asarray(chain, dtype=float)
    r = np.hypot(p[:, 0], p[:, 1])
    if r[0] >= r_min - 1e-12:
        return p
    k = int(np.argmax(r >= r_min))
    if k == 0:
        raise ValueError(f"chain never reaches r_min = {r_min:.3f} mm")
    a, b = p[k - 1], p[k]
    ab = b - a
    A = float(ab.dot(ab))
    B = 2.0 * float(a.dot(ab))
    C = float(a.dot(a)) - r_min ** 2
    s = (-B + np.sqrt(max(B * B - 4.0 * A * C, 0.0))) / (2.0 * A)
    return np.vstack([a + min(max(s, 0.0), 1.0) * ab, p[k:]])


def origin_arc_closure(lo: np.ndarray, hi: np.ndarray, wall: np.ndarray, ds: float = 1.0):
    """Put both inner ends on the SAME radius - the larger of the two - and close with an arc
    centred on the machine axis.

    The point is the tapered inner part that has to be cut out later: a tip bounded by an arc
    about the origin can be trimmed by a plain cylinder, which a housing-following arc cannot.
    The boundary that already sits further out sets the radius; the other is cut back to it.
    """
    r_lo, r_hi = float(np.hypot(*lo[0])), float(np.hypot(*hi[0]))
    r_in = max(r_lo, r_hi)
    lo2, hi2 = _clip_inner(lo, r_in), _clip_inner(hi, r_in)
    a_hi = float(np.arctan2(hi2[0][1], hi2[0][0]))
    a_lo = float(np.arctan2(lo2[0][1], lo2[0][0]))
    pts = _arc(r_in, a_hi, a_lo, ccw=False, ds=ds)     # back the way the loop came: hi -> lo
    return lo2, hi2, {'pts': pts, 'kind': 'arc at origin', 'radius': r_in,
                      'centre': np.zeros(2), 'span': float(np.hypot(*(lo2[0] - hi2[0]))),
                      'sweep_deg': float(np.degrees((a_hi - a_lo) % (2.0 * np.pi))),
                      'r_lo': r_lo, 'r_hi': r_hi, 'trimmed': abs(r_lo - r_hi),
                      'from': 'lo' if r_lo > r_hi else 'hi',
                      'min_d': float(_wall_distance(pts, wall).min())}


def _on_clearance_locus(direction: np.ndarray, centre: np.ndarray, wall: np.ndarray,
                        clearance: float, r_hi: float = 400.0) -> np.ndarray:
    """March out from `centre` along `direction` to the point that is `clearance` off the housing.

    The housing distance rises monotonically along an outward ray from inside it, so a bisection
    lands on the locus exactly - the same locus both boundary ends were truncated onto.
    """
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


def _circle_through(p0, p1, p2):
    """Centre and radius of the circle through three points; (None, inf) if they are collinear."""
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None, np.inf
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
          + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
          + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    c = np.array([ux, uy])
    return c, float(np.hypot(*(p0 - c)))


def inner_closure(p_from: np.ndarray, p_to: np.ndarray, wall: np.ndarray, clearance: float,
                  ds: float = 1.0) -> dict:
    """Close a dee's inner end with an arc that follows the housing.

    The two boundary ends already sit on the clearance locus. A third point is placed on that same
    locus half way between them (bisector of the two, seen from the housing centroid), and the
    circle through the three is the arc: three points at the clearance means the arc tracks the
    wall instead of cutting a chord across it.

    Degenerate cases fall back to a straight segment: a dee whose boundaries were cut where they
    meet (dee0) has its two ends within a fraction of a millimetre of each other.
    """
    centre = wall.mean(axis=0)
    span = float(np.hypot(*(p_to - p_from)))
    if span < 1.0:
        return {'pts': np.vstack([p_from, p_to]), 'kind': 'point', 'radius': np.nan,
                'centre': None, 'span': span, 'min_d': float(_wall_distance(
                    [p_from, p_to], wall).min())}
    a0 = np.arctan2(*(p_from - centre)[::-1])
    a1 = np.arctan2(*(p_to - centre)[::-1])
    half = a0 + 0.5 * ((a1 - a0 + np.pi) % (2.0 * np.pi) - np.pi)
    mid = _on_clearance_locus(np.array([np.cos(half), np.sin(half)]), centre, wall, clearance)
    c, r = _circle_through(p_from, mid, p_to)
    if c is None or not np.isfinite(r) or r > 5.0e3:
        return {'pts': np.vstack([p_from, p_to]), 'kind': 'straight', 'radius': np.inf,
                'centre': None, 'span': span,
                'min_d': float(_wall_distance(_densify(np.vstack([p_from, p_to]), 0.05)[0],
                                              wall).min())}
    t0, tm, t1 = (np.arctan2(*(p - c)[::-1]) for p in (p_from, mid, p_to))
    ccw = ((tm - t0) % (2.0 * np.pi)) < ((t1 - t0) % (2.0 * np.pi))
    d = (t1 - t0) % (2.0 * np.pi) if ccw else -((t0 - t1) % (2.0 * np.pi))
    n = max(3, int(np.ceil(abs(d) * r / ds)) + 1)
    t = t0 + d * np.linspace(0.0, 1.0, n)
    pts = c + r * np.column_stack([np.cos(t), np.sin(t)])
    return {'pts': pts, 'kind': 'arc', 'radius': r, 'centre': c, 'span': span,
            'sweep_deg': float(np.degrees(abs(d))), 'ccw': bool(ccw), 'mid': mid,
            'min_d': float(_wall_distance(pts, wall).min())}


def _meet_chains(a: np.ndarray, b: np.ndarray):
    """Where two inner-to-outer chains meet.

    First a real crossing between segments; failing that, the intersection of their two innermost
    segments EXTENDED inward. Returns (point, ia, ib) where ia / ib index the segment the point
    lies on, or -1 when it sits on the inward extension.
    """
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


def _arc(r: float, a0: float, a1: float, ccw: bool = True, ds: float = 2.0) -> np.ndarray:
    """Arc of radius r [mm] from azimuth a0 to a1 [rad], going the given way round.

    Sampled for plotting and for measuring only - in the .geo this becomes a single Circle, so
    the sampling never reaches the geometry.
    """
    d = (a1 - a0) % (2.0 * np.pi) if ccw else -((a0 - a1) % (2.0 * np.pi))
    n = max(2, int(np.ceil(abs(d) * r / ds)) + 1)
    t = a0 + d * np.linspace(0.0, 1.0, n)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def dee_boundaries(dees: DeeSystem, wall: np.ndarray, r_max: float = R_MAX_DEE_MM,
                   junction_tol: float = JUNCTION_TOL_DEG,
                   clearance: float = CAVITY_CLEARANCE_INNER):
    """The two boundary chains of each dee, in mm, out to r_max. Nothing closed, nothing extruded.

    A dee sits between its ENTRY gap and its EXIT gap (the DeeSystem stores them as consecutive
    pairs). Its boundaries are those two centre lines offset by half a gap width TOWARDS the dee:
    the entry gap to its CCW side, the exit gap to its CW side. That is what gap_fields does to
    build its wedges, so offset_gap_boundary is reused rather than re-derived - it also handles a
    tapered gap, where the offset follows the local width.
    """
    from PyCentralRegion.gap_fields import offset_polyline

    # ---- the inner ends -------------------------------------------------------------------
    # Gaps 1..7 stop `clearance` away from the housing. Gap 0 is different: its last variable
    # segment carries a -40.3 deg rotation, which swings its inner end round to azimuth 106 deg -
    # past gap 1's 66 deg base angle - so it reaches across its own dee. It is cut where it meets
    # gap 1 instead, which is what closes dee0's crossed-over tip.
    raw, jun, closed = [], [], []
    for cav in dees.gaps:
        ch, j, cl = gap_centre_line(cav, junction_tol, r_max=None)
        raw.append(ch)
        jun.append(j)
        closed.append(cl)

    # One boundary per gap: gap 2j is dee j's low-azimuth edge (offset +, into the dee), gap 2j+1
    # its high-azimuth edge (offset -). The clearance is measured on THIS - the metal - and not on
    # the centre line: the offset is half a gap width, so a centre line held 5 mm off the housing
    # can put its own edge 5 mm closer, which is what gap 1 was doing (1.25 mm of real clearance
    # against 5.00 claimed).
    # Offset the FULL centre line and clip the RESULT at r_max, not the other way round: a point
    # sitting on the r_max circle, offset half a gap width sideways, lands 0.04 mm outside it, so
    # clipping first leaves the boundary ends off the arc they have to meet.
    bnd = []
    for g, cav in enumerate(dees.gaps):
        s = +1.0 if g % 2 == 0 else -1.0
        bnd.append(_clip_outer(offset_polyline(raw[g], _half_width_mm(cav, raw[g], s)), r_max))

    inner = {}
    full = [b.copy() for b in bnd]
    for g in range(1, len(bnd)):
        bnd[g], d0, moved = _cut_inner_at_clearance(bnd[g], wall, clearance)
        inner[g] = {'rule': 'clearance', 'd': d0, 'moved': moved, 'start': bnd[g][0].copy()}
    # dee0 gets a VERTEX, not an arc: both of its boundaries start where gap 0 and gap 1 meet.
    # Taken on the FULL boundaries so the vertex is a real property of the two lines rather than
    # an artefact of where gap 1 happened to be cut for clearance - and on their inward extensions
    # when they no longer reach each other (snapping gap 1's node to 117.4 mm did exactly that).
    # TODO more checks here: nothing yet guarantees the vertex clears the housing.
    meet, i0, i1 = _meet_chains(full[0], full[1])
    if meet is None:
        inner[0] = {'rule': 'gap 0 || gap 1 - left as it is', 'moved': False,
                    'd': float(_wall_distance(bnd[0][:1], wall)[0]), 'start': bnd[0][0].copy()}
    else:
        bnd[0] = _start_at(full[0], meet, i0)
        bnd[1] = _start_at(full[1], meet, i1)
        how = 'meets gap 1' if i0 >= 0 else 'meets gap 1 (extended)'
        d_meet = float(_wall_distance([meet], wall)[0])
        inner[0] = {'rule': how, 'moved': True, 'meet': meet, 'd': d_meet, 'start': meet.copy()}
        inner[1] = {'rule': how.replace('gap 1', 'gap 0'), 'moved': True, 'meet': meet,
                    'd': d_meet, 'start': meet.copy()}
    for g in range(len(bnd)):                      # what the metal really clears, end to end
        inner[g]['min_d'] = float(_wall_distance(_densify(bnd[g], 0.05)[0], wall).min())

    def boundary(g, cav, side):
        return _clip_outer(raw[g], r_max), bnd[g], jun[g], closed[g]

    out = []
    for j in range(len(dees.gaps) // 2):
        entry, exit_ = dees.gaps[2 * j], dees.gaps[2 * j + 1]
        span = np.mod(exit_.base_angle - entry.base_angle, 360.0)
        if not 0.0 < span < 180.0:
            raise ValueError(f"dee {j}: entry->exit spans {span:.2f} deg CCW, expected (0, 180); "
                             f"the gap pairing is not entry/exit")
        c_lo, lo, j_lo, cl_lo = boundary(2 * j, entry, +1.0)
        c_hi, hi, j_hi, cl_hi = boundary(2 * j + 1, exit_, -1.0)
        # the inner closure runs the other way round the loop: hi's inner end back to lo's.
        # dee0 keeps the gap0-meets-gap1 rule (its two boundaries are cut where they cross, so
        # there is nothing to bridge); the rest get both ends on one radius and an arc about the
        # origin, so the tapered tip can later be trimmed with a plain cylinder.
        if j == 0:
            # a plain vertex: both boundaries already start at the gap0 / gap1 meeting point
            tip = {'pts': np.vstack([hi[0], lo[0]]), 'kind': 'vertex', 'radius': np.nan,
                   'centre': None, 'span': float(np.hypot(*(lo[0] - hi[0]))),
                   'min_d': float(_wall_distance([hi[0], lo[0]], wall).min())}
        else:
            lo, hi, tip = origin_arc_closure(lo, hi, wall)
        # the outer closure: lo runs inner -> outer, so the arc continues CCW from its end round
        # to the end of hi, which is the dee's high-azimuth boundary
        a0 = float(np.arctan2(lo[-1][1], lo[-1][0]))
        a1 = float(np.arctan2(hi[-1][1], hi[-1][0]))
        arc = _arc(r_max, a0, a1, ccw=True)
        arc_deg = float(np.degrees((a1 - a0) % (2.0 * np.pi)))
        out.append({'label': f'dee{j}', 'span_deg': span, 'r_max': float(r_max),
                    'voltage': float(entry.voltage), 'phase_deg': float(entry.phase_deg),
                    'gaps': (2 * j, 2 * j + 1), 'closed': (cl_lo, cl_hi),
                    'junction': (j_lo, j_hi), 'arc': arc, 'arc_deg': arc_deg,
                    'inner': (inner[2 * j], inner[2 * j + 1]), 'clearance': float(clearance),
                    'tip': tip,
                    'centre_lo': c_lo, 'centre_hi': c_hi, 'lo': lo, 'hi': hi})
    return out


def dee_width(d: dict, n: int = 600):
    """(radius, width) of a dee across its two boundaries [mm].

    The centre lines start at r = 5 mm and each boundary is offset 5 mm INTO the dee, so close to
    the axis the two boundaries meet and cross over: the dee has no tip there at all. This is the
    curve that says where it actually begins.
    """
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


def print_dee_boundaries(foot) -> None:
    print()
    print("=" * 96)
    print("DEES - step A: the boundary chains")
    print("=" * 96)
    print("  each dee runs from its entry gap to its exit gap, both centre lines offset half a")
    print(f"  gap width into the dee, clipped at r_max = {foot[0]['r_max']:.1f} mm "
          f"(the centre lines themselves run to {R_MAX * MM:.0f} mm)")
    print()
    print(f"  {'dee':>6} {'span':>8} {'V':>9} {'phase':>8} {'pts':>9} {'r range':>18} "
          f"{'width at':>10} {'widest':>16}")
    print(f"  {'':>6} {'[deg]':>8} {'[V]':>9} {'[deg]':>8} {'lo/hi':>9} {'[mm]':>18} "
          f"{'r=100 mm':>10} {'[mm]':>16}")
    print("  " + "-" * 92)
    for d in foot:
        rr, w = dee_width(d)
        r_all = np.hypot(np.vstack([d['lo'], d['hi']])[:, 0], np.vstack([d['lo'], d['hi']])[:, 1])
        print(f"  {d['label']:>6} {d['span_deg']:8.3f} {d['voltage']:9.0f} {d['phase_deg']:8.2f} "
              f"{len(d['lo']):4d}/{len(d['hi']):<4d} {r_all.min():7.2f}..{r_all.max():7.2f} "
              f"{np.interp(100.0, rr, w):10.2f} {w.max():8.2f} at r={rr[np.argmax(w)]:5.0f}")
    print()
    print(f"  the inner ends, with the METAL held {foot[0]['clearance']:.2f} mm clear of the housing")
    print("  (measured on the offset boundary, not on the gap centre line - the offset is half a")
    print("   gap width, so measuring on the centre line lets the edge sit that much closer):")
    print(f"    {'gap':>5} {'rule':>24} {'inner end [mm]':>22} {'r':>9} {'at the end':>11} "
          f"{'closest':>9}")
    for d in foot:
        for gi, rec in zip(d['gaps'], d['inner']):
            p = rec['start']
            flag = '  <- under' if rec['min_d'] < foot[0]['clearance'] - 1e-3 else ''
            print(f"    {gi:>5} {rec['rule']:>24} ({p[0]:+8.3f}, {p[1]:+8.3f}) "
                  f"{np.hypot(*p):9.3f} {rec['d']:11.3f} {rec['min_d']:9.3f}{flag}")
    print()
    print("  the inner closures - both ends put on the LARGER of the two radii, closed by an arc")
    print("  about the origin so the tapered tip can be trimmed later with a plain cylinder:")
    print(f"    {'dee':>6} {'kind':>14} {'r(lo)':>9} {'r(hi)':>9} {'r used':>9} {'trimmed':>9} "
          f"{'sweep':>9} {'length':>9} {'to housing':>11}")
    for d in foot:
        t = d['tip']
        ln = float(np.hypot(*np.diff(t['pts'], axis=0).T).sum()) if len(t['pts']) > 1 else 0.0
        r_lo, r_hi = t.get('r_lo', np.nan), t.get('r_hi', np.nan)
        print(f"    {d['label']:>6} {t['kind']:>14} {r_lo:9.3f} {r_hi:9.3f} "
              f"{(t['radius'] if np.isfinite(t['radius']) else float('nan')):9.3f} "
              f"{t.get('trimmed', 0.0):9.3f} {t.get('sweep_deg', 0.0):8.3f}d {ln:9.3f} "
              f"{t['min_d']:11.3f}")
    print()
    print(f"  the outer arcs, at r_max = {foot[0]['r_max']:.1f} mm:")
    print(f"    {'dee':>6} {'lo end az':>11} {'hi end az':>11} {'arc':>10} {'length':>10} "
          f"{'lo end r':>11} {'hi end r':>11}")
    for d in foot:
        a0 = np.degrees(np.arctan2(d['lo'][-1][1], d['lo'][-1][0])) % 360.0
        a1 = np.degrees(np.arctan2(d['hi'][-1][1], d['hi'][-1][0])) % 360.0
        print(f"    {d['label']:>6} {a0:10.4f}d {a1:10.4f}d {d['arc_deg']:9.4f}d "
              f"{np.radians(d['arc_deg']) * d['r_max']:10.3f} "
              f"{np.hypot(*d['lo'][-1]):11.5f} {np.hypot(*d['hi'][-1]):11.5f}")
    print()
    print("  where each dee opens up: the radius beyond which it never again narrows below the")
    print("  given width. Below the 0 mm column the two boundaries have crossed over entirely.")
    print(f"    {'dee':>6} " + " ".join("{:>13}".format("w >= {:.0f} mm".format(t))
                                        for t in (0.0, 2.0, 5.0, 10.0)))
    for d in foot:
        rr, w = dee_width(d)
        cells = []
        for t in (0.0, 2.0, 5.0, 10.0):
            bad = np.where(w <= t)[0]
            if not len(bad):
                cells.append('{:8.2f} mm'.format(rr[0]))          # open from the very start
            elif bad[-1] + 1 < len(rr):
                cells.append('{:8.2f} mm'.format(rr[bad[-1] + 1]))
            else:
                cells.append('never')
        print(f"    {d['label']:>6} " + " ".join("{:>13}".format(c) for c in cells))


def gap_crossings(cav, xy: np.ndarray, merge: float = 1.0) -> np.ndarray:
    """Radii [mm] at which the orbit crosses this gap, near-duplicate grazes merged."""
    crossed, t, _ = cav.check_crossings_batch(xy[:-1], xy[1:])
    idx = np.where(crossed)[0]
    if not len(idx):
        return np.zeros(0)
    pts = xy[idx] + t[idx, None] * (xy[idx + 1] - xy[idx])
    r = np.sort(np.hypot(pts[:, 0], pts[:, 1])) * MM
    keep = [r[0]]
    for v in r[1:]:
        if v - keep[-1] > merge:
            keep.append(v)
    return np.asarray(keep)


def neutralise_inert_segments(dees: DeeSystem, trj: np.ndarray, verbose: bool = True):
    """A variable segment the beam never crosses is doing nothing: make it nominal.

    Gap 5 here is the case: its variable segment spans r = 5..95 mm but the orbit does not reach
    that gap until 120.4 mm, so the segment's angle and rotation have no effect on any crossing.
    Leaving it in is not harmless - it puts a kink and a 21 mm jog in the dee edge for nothing,
    and snapping its node "between turns" can only move it OUTWARD past a real crossing, which
    would change the beam dynamics. Setting the excursion and the rotation to zero makes the gap
    a plain radial line at its nominal base angle, and the junction disappears with it.

    Run BEFORE the snap, on the radii the optimizer produced.
    """
    xy = np.asarray(trj, dtype=float)[:, :2]
    changed = []
    for gi, cav in enumerate(dees.gaps):
        if cav.n_variable_segments == 0:
            continue
        cr = gap_crossings(cav, xy)
        edges = [cav.r_min * MM] + [float(v) * MM for v in cav.segment_radii]
        ang, rot = list(cav.segment_angles), list(cav.segment_rotations or
                                                  [0.0] * cav.n_variable_segments)
        touched = False
        for i in range(cav.n_variable_segments):
            lo, hi = edges[i], edges[i + 1]
            if np.any((cr >= lo) & (cr <= hi)):
                continue
            if ang[i] == 0.0 and rot[i] == 0.0:
                continue
            changed.append((gi, i, lo, hi, ang[i], rot[i],
                            float(cr[0]) if len(cr) else float('nan')))
            ang[i], rot[i], touched = 0.0, 0.0, True
        if touched:
            cav.update_geometry(segment_angles=ang, segment_rotations=rot)
    if verbose:
        if not changed:
            print("  every variable segment has at least one beam crossing; none neutralised")
        for gi, i, lo, hi, a0, r0, first in changed:
            print(f"  gap{gi} segment {i} spans r {lo:.2f}..{hi:.2f} mm but the orbit first "
                  f"reaches this gap at {first:.2f} mm")
            print(f"    -> no crossing inside it: angle {a0:+.4f} -> 0, rotation {r0:+.4f} -> 0 "
                  f"(nominal radial line at {dees.gaps[gi].base_angle:.1f} deg)")
    return changed


DUMMY_OVERLAP = 6.0     # [mm] how far the dummy dees reach INSIDE the housing surface, so the
                        # union with the housing is a real overlap and not a face-to-face touch
DEE_WALL = 5.0          # [mm] radial thickness of the dee's solid inner wall: the aperture tool
                        # starts this far outside the dee's own inner radius


def dummy_boundaries(dees: DeeSystem, wall: np.ndarray, r_max: float = R_MAX_DEE_MM,
                     junction_tol: float = JUNCTION_TOL_DEG, overlap: float = DUMMY_OVERLAP):
    """The dummy dees (the grounded wedges between the dees), built like the dees.

    Each sits between one dee's EXIT gap and the next dee's ENTRY gap, so it takes the OTHER side
    of those two gap lines: the exit gap offset +half a gap width, the next entry gap offset -.
    Two differences from a dee:
      * the inner ends run INWARD past the housing surface by `overlap`, so the union with the
        housing solid is a genuine overlap rather than two faces meeting exactly;
      * the inner end is closed with a straight line, not an arc.
    """
    from PyCentralRegion.gap_fields import offset_polyline

    n = len(dees.gaps)
    out = []
    for k in range(n // 2):
        g_lo, g_hi = 2 * k + 1, (2 * k + 2) % n
        chains, offs = [], []
        for g, s in ((g_lo, +1.0), (g_hi, -1.0)):
            cav = dees.gaps[g]
            ch, _, _ = gap_centre_line(cav, junction_tol, r_max=None)
            off = _clip_outer(offset_polyline(ch, _half_width_mm(cav, ch, s)), r_max)
            off, d0, _ = _reach_inner(off, wall, -abs(overlap))
            chains.append(ch)
            offs.append((off, d0))
        (lo, d_lo), (hi, d_hi) = offs
        a0 = float(np.arctan2(lo[-1][1], lo[-1][0]))
        a1 = float(np.arctan2(hi[-1][1], hi[-1][0]))
        out.append({'label': f'dummy{k}', 'gaps': (g_lo, g_hi), 'r_max': float(r_max),
                    'lo': lo, 'hi': hi, 'arc': _arc(r_max, a0, a1, ccw=True),
                    'arc_deg': float(np.degrees((a1 - a0) % (2.0 * np.pi))),
                    'tip': {'pts': np.vstack([hi[0], lo[0]]), 'kind': 'straight line',
                            'radius': np.inf, 'centre': None,
                            'span': float(np.hypot(*(lo[0] - hi[0]))),
                            'min_d': float(min(d_lo, d_hi))},
                    'depth': (d_lo, d_hi), 'voltage': 0.0})
    return out


def print_dummies(dummy) -> None:
    print()
    print("=" * 96)
    print("DUMMY DEES")
    print("=" * 96)
    print("  the grounded wedges between the dees: the other side of each dee's two gap lines,")
    print("  run inward PAST the housing surface so the union with it is a real overlap")
    print(f"    {'':>7} {'gaps':>8} {'inner ends [mm]':>34} {'into the housing':>18} {'tip':>9}")
    for d in dummy:
        print(f"    {d['label']:>7} {str(d['gaps']):>8} ({d['lo'][0][0]:+8.3f}, {d['lo'][0][1]:+8.3f}) "
              f"({d['hi'][0][0]:+8.3f}, {d['hi'][0][1]:+8.3f}) "
              f"{d['depth'][0]:8.3f} {d['depth'][1]:8.3f} {d['tip']['span']:9.3f}")


def reference_orbit(result, r_max: float = None) -> np.ndarray:
    """The optimizer's single-particle reference orbit, in mm, x-y only."""
    trj = np.asarray(getattr(result, 'trajectory_reference', None), dtype=float)
    if trj is None or trj.ndim != 2 or len(trj) < 2:
        return None
    xy = trj[:, :2] * MM
    if r_max is not None:
        keep = np.hypot(xy[:, 0], xy[:, 1]) <= r_max
        if keep.any():
            xy = xy[:np.argmax(~keep) if (~keep).any() else len(xy)]
    return xy


def print_orbit(foot, orbit: np.ndarray, result) -> None:
    if orbit is None:
        print("\n  (no reference orbit in the result)")
        return
    r = np.hypot(orbit[:, 0], orbit[:, 1])
    print()
    print(f"  the reference orbit: {len(orbit):,} points, r {r.min():.2f} .. {r.max():.2f} mm, "
          f"{int(getattr(result, 'n_turns', 0))} turns, "
          f"final energy {float(getattr(result, 'final_energy_mev', np.nan)):.3f} MeV")
    print("  The beam runs INSIDE the dee aperture and crosses the gaps, so it crosses the lo/hi")
    print("  edge lines by design - a distance to those is not a clearance. What matters in plan")
    print("  view is the inner arc (the tip must not sit where the beam is) and the reach of r_max.")
    print(f"    {'dee':>6} {'crossings of':>14} {'lo / hi':>10} {'to the inner arc':>18}")
    for d in foot:
        n_x = []
        for key in ('lo', 'hi'):
            c = d[key]
            n = 0
            for i in range(len(orbit) - 1):
                p, u = orbit[i], orbit[i + 1] - orbit[i]
                for k in range(len(c) - 1):
                    q, v = c[k], c[k + 1] - c[k]
                    den = u[0] * v[1] - u[1] * v[0]
                    if abs(den) < 1e-12:
                        continue
                    ta = ((q - p)[0] * v[1] - (q - p)[1] * v[0]) / den
                    tb = ((q - p)[0] * u[1] - (q - p)[1] * u[0]) / den
                    if 0.0 <= ta <= 1.0 and 0.0 <= tb <= 1.0:
                        n += 1
            n_x.append(n)
        tip = d['tip']['pts']
        d_tip = float(np.min(np.hypot(*(tip[:, None, :] - orbit[None, :, :]).T)))
        print(f"    {d['label']:>6} {'the edges:':>14} {n_x[0]:4d} /{n_x[1]:4d} {d_tip:18.3f}")
    print(f"    the orbit reaches {r.max():.3f} mm against r_max = {foot[0]['r_max']:.1f} mm: "
          f"{foot[0]['r_max'] - r.max():+.3f} mm of margin")


def plot_dee_boundaries(foot, ol: dict, out: str = None, orbit: np.ndarray = None) -> None:
    """Left: the boundary chains over the gap centre lines and the housing. Right: dee width."""
    import matplotlib
    if out:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14.5, 7.2),
                                 gridspec_kw={'width_ratios': [1.25, 1.0]})
    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    ring = np.vstack([ol['outer'], ol['outer'][0]])
    ax.plot(ring[:, 0], ring[:, 1], '-', color='0.6', lw=1.2, zorder=2, label='housing outline')
    if orbit is not None:
        ax.plot(orbit[:, 0], orbit[:, 1], '-', color='k', lw=0.7, alpha=0.75, zorder=7,
                label='reference orbit')
    for i, d in enumerate(foot):
        c = colours[i % len(colours)]
        for key in ('centre_lo', 'centre_hi'):
            ch = d[key]
            ax.plot(ch[:, 0], ch[:, 1], '--', color='0.8', lw=0.9, zorder=1,
                    label='gap centre lines' if (i == 0 and key == 'centre_lo') else None)
        ax.plot(d['lo'][:, 0], d['lo'][:, 1], '-', color=c, lw=1.6, zorder=3,
                label='{} boundaries'.format(d['label']))
        ax.plot(d['hi'][:, 0], d['hi'][:, 1], '-', color=c, lw=1.6, zorder=3)
        ax.plot(d['arc'][:, 0], d['arc'][:, 1], '-', color=c, lw=2.8, zorder=4,
                label='arc at r_max' if i == 0 else None)
        ax.plot(d['tip']['pts'][:, 0], d['tip']['pts'][:, 1], '-', color=c, lw=2.8, zorder=4,
                label='inner arc' if i == 0 else None)
    for i, d in enumerate(foot):
        for j, closed in zip(d['junction'], d['closed']):
            if closed:
                ax.plot(*j['meet'], 'o', mfc='none', mec='k', ms=9, mew=1.4, zorder=6,
                        label='junction closed' if i == 0 and j is d['junction'][0] else None)
            else:
                ax.plot([j['var_end'][0], j['fix_start'][0]],
                        [j['var_end'][1], j['fix_start'][1]], '-', color='k', lw=2.2, zorder=6,
                        label='junction jog (left)' if i == 0 else None)
    ax.plot([0], [0], '+', color='k', ms=10, mew=1.5, zorder=5)
    r_max = foot[0]['r_max']
    for rr in np.arange(50.0, r_max + 1e-9, 50.0):
        ax.add_patch(plt.Circle((0, 0), rr, fill=False, color='0.9', lw=0.8, zorder=0))
    ax.add_patch(plt.Circle((0, 0), r_max, fill=False, color='0.7', lw=1.0, ls=':', zorder=0))
    ax.set_xlim(-1.08 * r_max, 1.08 * r_max)
    ax.set_ylim(-1.08 * r_max, 1.08 * r_max)
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('dee boundaries: each gap centre line offset half a gap width into the dee\n'
                 'grey circles every 50 mm, dotted = r_max = {:.0f} mm'.format(r_max), fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.2)

    for i, d in enumerate(foot):
        rr, w = dee_width(d)
        bx.plot(rr, w, '-', color=colours[i % len(colours)], lw=1.6, label=d['label'])
    bx.axhline(0.0, color='k', lw=1.0)
    bx.set_xscale('log')
    bx.set_xlabel('radius [mm]  (log)')
    bx.set_ylabel('width across the dee [mm]')
    bx.set_title('where each dee actually begins:\nwidth <= 0 means the two boundaries have crossed',
                 fontsize=10)
    bx.legend(fontsize=8)
    bx.grid(alpha=0.25, which='both')

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140)
        print(f"  wrote {out}")
    else:
        plt.show()



# =============================================================================================
# 2. the model in gmsh: footprints -> prisms -> hollowed -> the ground fused with the housing
# =============================================================================================
def _aperture_ring(r0: float, r1: float, step=APERTURE_R_MM, h=DEE_APERTURE_MM):
    """(r, z) ring of the aperture profile between r0 and r1, as a closed polygon in the x-z plane.

    The profile is the beam channel's HALF height: h[0]/2 inside step[0], h[1]/2 beyond step[1],
    linear between. Revolved about z it is the tool that hollows an electrode out.
    """
    rr = [r0] + [v for v in step if r0 < v < r1] + [r1]
    rr = sorted(set(rr))
    zz = [0.5 * float(_height(v, step, h)) for v in rr]
    lower = list(zip(rr, [-v for v in zz]))
    upper = list(zip(rr[::-1], zz[::-1]))
    return lower + upper


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
    out = occ.revolve([(2, s)], 0, 0, 0, 0, 0, 1, 2.0 * np.pi)
    v = [t for d, t in out if d == 3]
    if len(v) != 1:
        raise RuntimeError(f"aperture tool revolve gave {len(v)} volumes")
    return v[0]


def _strip_orphans(gmsh, occ) -> int:
    """Delete every entity that is not part of some solid.

    The aperture tools are made by revolving a sketch: a ring of points, the lines through them,
    a curve loop and a plane surface. `revolve` builds the tool from that sketch but does not
    consume it, and `cut(..., removeTool=True)` removes only the tool VOLUME - so the sketch
    survives the boolean and is written into the .step and shown in the GUI, one per electrode.
    The same is true of the beam channel's own construction face.

    Rather than track each tool's entities (which the booleans renumber anyway), this asks the
    model the direct question: which entities are reachable from a volume? Everything else is
    construction and goes. Sweeping high dimension first, non-recursively, means a surface's
    edges are not deleted out from under a surface that is still needed - they are simply
    considered on their own in the next pass.
    """
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
            except Exception:                # still referenced by something we are keeping
                pass
    occ.synchronize()
    return dropped


def footprint_edges(d: dict):
    """The footprint as REAL edges rather than a sampled polygon.

    Sampling the arcs and the housing spline into polylines put ~8,100 short Line entities in the
    model, every one of them a vertex gmsh has to honour: the curved parts came out meshed at
    ~2 mm however large lc was, while the genuine Circles and Hyperbolas sat at lc. So the loop is
    described here as ('poly', pts) / ('arc', p0, p1, centre) / ('spline', pts) and built with
    addCircleArc / addSpline, exactly as the .geo path does.
    """
    e = [('poly', d['lo']),
         ('arc', d['lo'][-1], d['hi'][-1], np.zeros(2)),        # the r_max arc, about the origin
         ('poly', d['hi'][::-1])]
    tip = d['tip']
    if tip['kind'] == 'arc at origin':
        e.append(('arc', d['hi'][0], d['lo'][0], np.zeros(2)))
    elif tip['kind'] in ('straight line', 'arc', 'straight'):
        e.append(('poly', np.vstack([d['hi'][0], d['lo'][0]])))
    # 'vertex' (dee0): the two boundaries already meet, nothing to add
    return e


def _occ_loop(occ, edges, z: float):
    """Build a closed curve loop from footprint_edges at height z; returns the surface tag."""
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
            q = np.asarray(e[1], dtype=float)
            curves.append(occ.addSpline([P(v) for v in q]))
    return occ.addPlaneSurface([occ.addCurveLoop(curves)])


def housing_edges(sim: dict):
    """The housing outline as its two straight faces and one spline, not 600 sampled lines."""
    nodes, edges, out = sim['nodes'], sim['edges'], []
    ni = 0
    for kind, knots in edges:
        a, b = nodes[ni % len(nodes)], nodes[(ni + 1) % len(nodes)]
        if kind == 'spline':
            out.append(('spline', np.vstack([a, knots, b])))
        else:
            out.append(('poly', np.vstack([a, b])))
        ni += 1
    return out


def footprint_polygon(d: dict, arc_ds: float = 2.0) -> np.ndarray:
    """One closed footprint: lo (in->out), the r_max arc, hi (out->in), then the tip."""
    tip = d['tip']['pts']
    parts = [d['lo'], d['arc'][1:-1], d['hi'][::-1]]
    if len(tip) > 2:                       # an arc: it already runs hi -> lo
        parts.append(tip[1:-1])
    ring = np.vstack(parts)
    keep = [ring[0]]
    for p in ring[1:]:
        if np.hypot(*(p - keep[-1])) > 1e-9:
            keep.append(p)
    if np.hypot(*(keep[0] - keep[-1])) < 1e-9:
        keep.pop()
    return np.array(keep)


def corner_post_radius(d: dict, wall_t: float, n: int = 4000) -> float:
    """Where dee0's aperture tool must start so the corner post is at least `wall_t` thick.

    dee0 ends in a VERTEX, so the metal left inside the tool radius is a wedge that pinches to
    nothing. Starting the tool at r_inner + wall_t would leave a sliver; this also requires the
    dee to be at least wall_t WIDE there, so the post is wall_t in both directions.
    """
    rr, w = dee_width(d, n)
    r_in = float(np.hypot(*d['lo'][0]))
    ok = np.where(w >= wall_t)[0]
    r_wide = float(rr[ok[0]]) if len(ok) else float(rr[-1])
    return max(r_in + wall_t, r_wide)


POST_SIZE = 10.0        # [mm] square cross-section of a gap-0 post
POST_OFFSET = 5.0       # [mm] post CENTRE from the dee edge; at half the size the near face sits
#                         flush with the edge - "flat side aligned with the dee edge"
POST_OVERHANG = 5.0     # [mm] how far the post runs above and below the dee's vertical aperture.
#                         The dee is already solid outside the aperture, so this adds no metal; it
#                         gives the fuse real overlap instead of a coincident face, which OCC
#                         would otherwise leave as a zero-thickness sliver.


def gap_edge(foot, dummy, gap: int, side: str):
    """The boundary of one gap on one side, and the polygon of the solid it belongs to.

    dee_boundaries stores gap 2j as dee j's `lo` edge and gap 2j+1 as its `hi`; dummy_boundaries
    stores gap 2k+1 as dummy k's `lo` and gap 2k+2 as its `hi`. So every gap has exactly one dee
    side and one dummy-dee side, and they face each other across the gap.
    """
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
    raise ValueError(f"side must be 'dee' or 'dummy', got {side!r}")


POST_SHAPE = 'square'    # 'square' or 'cylinder'; set from --post-shape
POST_FACETS = 16         # facets of a 'cylinder' post


def _post_corners(c, tangent, normal, size: float):
    """The post's plan outline: a square, or a regular POST_FACETS-gon of the same width.

    A square prism has four 90 deg edges running its full height, and at a conducting wedge of
    exterior angle 3*pi/2 the charge density diverges as r^(-1/3) - which BEM cannot resolve, so
    the near field there never converges under refinement and the matrix conditions poorly. Those
    edges sit exactly in the gap, where the field is strongest and where the post is supposed to
    be doing its shaping. A 16-gon's vertices are 202.5 deg exterior, giving r^(-0.11): the same
    blocking width and jaw position, with the singularity essentially gone. It is also the shape
    an engineer would build at 65 kV, since a sharp edge is where it sparks.

    Returned as a polygon rather than an OCC cylinder deliberately: the build extrudes 'corners',
    and the orbit-clearance test, the placement collision test and the hit counting all read the
    same array, so one representation keeps every one of them correct without a second code path.
    """
    c = np.asarray(c, dtype=float)
    if POST_SHAPE == 'square':
        return np.array([c + 0.5 * size * (a * tangent + b * normal)
                         for a, b in ((-1, -1), (+1, -1), (+1, +1), (-1, +1))])
    a = np.linspace(0.0, 2.0 * np.pi, int(POST_FACETS), endpoint=False)
    return np.array([c + 0.5 * size * (np.cos(k) * tangent + np.sin(k) * normal) for k in a])


def _post_on_edge(edge, poly, s: float, aperture, size: float, offset: float,
                  overhang: float, name: str, owner: str) -> dict:
    """One square post on `edge` at arclength `s`, near face flush with the edge.

    The post is placed on whichever side of the edge lies inside `poly` - the solid it will be
    fused into - so the same routine serves a dee edge and a dummy-dee edge facing it, and the two
    come out pointing at each other across the gap without any sign bookkeeping here.
    """
    from matplotlib.path import Path as MplPath
    seg = np.diff(edge, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    sl = np.concatenate([[0.0], np.cumsum(seg_len)])
    if not (0.0 <= s <= sl[-1]):
        raise ValueError(f"post {name}: arclength {s:.2f} mm is off the edge (0 .. {sl[-1]:.2f})")
    inside = MplPath(poly)
    k = int(np.clip(np.searchsorted(sl, s) - 1, 0, len(seg) - 1))
    t = seg[k] / seg_len[k]
    p = edge[k] + (s - sl[k]) * t
    n = np.array([-t[1], t[0]])
    if not inside.contains_point(p + 0.25 * n):
        n = -n
    if not inside.contains_point(p + 0.25 * n):
        raise RuntimeError(f"post {name}: neither side of the edge is inside {owner}")
    c = p + offset * n
    h = 0.5 * float(_height(np.hypot(*p), APERTURE_R_MM, aperture)) + overhang
    corners = _post_corners(c, t, n, size)
    return {'name': name, 'owner': owner, 's': float(s), 'point': p, 'tangent': t, 'normal': n,
            'centre': c, 'corners': corners, 'half_height': h,
            'r': float(np.hypot(*p)), 'r_centre': float(np.hypot(*c))}


def edge_crossings(edge: np.ndarray, path: np.ndarray, first_only: bool = True):
    """Where a particle path crosses an edge polyline: (arclength [mm], crossing point).

    EXACT segment-segment intersection. The obvious shortcut - take the sign of the cross product
    against the NEAREST edge segment and look for sign changes - is wrong, and wrong in a way that
    silently misplaces hardware: far from the edge the nearest segment changes as the particle
    orbits, and the sign flips with it, so the test reports crossings that never happened. Measured
    on gap 2 (2026-09-17) it put the 99.95 percentile 13.2 mm too far in, which is a post standing
    inside the beam. It looks fine on gap 0 and gap 1 only because the beam reaches those edges
    almost immediately, before it can wander.

    `path` is (N, 2) or (N, 3); only x-y is used. Returns a list in path order, or just the first.
    """
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
            q = a[k] + t[k] * r[k]
            z = None
            if np.asarray(path).shape[1] > 2:
                z = float(path[k][2] + t[k] * (path[k + 1][2] - path[k][2]))
            out.append((int(k), float(sl[m] + u[k] * seg_len[m]), q, z))
    out.sort(key=lambda v: v[0])
    if first_only:
        return out[0] if out else None
    return out


def gap_window(foot, dummy, gap: int, side: str, s_lo: float, s_hi: float,
               z_lo: float, z_hi: float, aperture=DEE_APERTURE_MM, depth: float = POST_SIZE,
               overhang: float = POST_OVERHANG, name: str = None) -> list:
    """A vertical aperture on one gap edge: two blocks leaving an opening from z_lo to z_hi.

    The same construction as a post - a rectangular prism on the edge, near face flush with it,
    `depth` into the electrode, fused into whatever solid owns that edge - but spanning an
    arclength RANGE rather than sitting at a point, and split into an upper and a lower block so
    the beam passes between them. Each block runs from the opening to `overhang` past the
    aperture, which is already solid metal there, so the fuse has real overlap instead of a
    coincident face.

    z_lo / z_hi are the OPENING, so a 26 mm window symmetric about the median plane is
    (-13, +13).
    """
    edge, poly, owner = gap_edge(foot, dummy, int(gap), side)
    from matplotlib.path import Path as MplPath
    seg = np.diff(edge, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    sl = np.concatenate([[0.0], np.cumsum(seg_len)])
    if not (0.0 <= s_lo < s_hi <= sl[-1]):
        raise ValueError(f"gap-{gap} window: arclengths {s_lo:.2f}..{s_hi:.2f} mm are off the "
                         f"edge (0 .. {sl[-1]:.2f})")
    inside = MplPath(poly)

    def _pt(s):
        k = int(np.clip(np.searchsorted(sl, s) - 1, 0, len(seg) - 1))
        t = seg[k] / seg_len[k]
        return edge[k] + (s - sl[k]) * t, t

    p_lo, t_lo = _pt(s_lo)
    p_hi, t_hi = _pt(s_hi)
    n = np.array([-t_lo[1], t_lo[0]])
    if not inside.contains_point(p_lo + 0.25 * n):
        n = -n
    if not inside.contains_point(p_lo + 0.25 * n):
        raise RuntimeError(f"gap-{gap} window: neither side of the edge is inside {owner}")
    # plan footprint: the strip of edge from s_lo to s_hi, `depth` into the electrode
    corners = np.array([p_lo, p_hi, p_hi + depth * n, p_lo + depth * n])
    r_mid = float(np.hypot(*(0.5 * (p_lo + p_hi))))
    ap_half = 0.5 * float(_height(r_mid, APERTURE_R_MM, aperture))
    top = ap_half + overhang
    out = []
    base = name or f"g{int(gap)}_{side}_win"
    for tag, lo, hi in (('up', float(z_hi), top), ('dn', -top, float(z_lo))):
        if hi <= lo:
            raise ValueError(f"{base}_{tag}: opening {z_lo:+.1f}..{z_hi:+.1f} mm leaves no metal "
                             f"against an aperture half-height of {ap_half:.1f} mm")
        out.append({'name': f"{base}_{tag}", 'owner': owner, 'kind': 'window',
                    'corners': corners, 'z_lo': lo, 'z_hi': hi,
                    's_lo': float(s_lo), 's_hi': float(s_hi), 'r_mid': r_mid,
                    'opening': (float(z_lo), float(z_hi)), 'ap_half': ap_half,
                    'centre': 0.5 * (p_lo + p_hi) + 0.5 * depth * n,
                    'half_height': 0.5 * (hi - lo), 'z_mid': 0.5 * (hi + lo),
                    'point': 0.5 * (p_lo + p_hi), 'r': r_mid,
                    'r_centre': float(np.hypot(*(0.5 * (p_lo + p_hi) + 0.5 * depth * n))),
                    's': 0.5 * (float(s_lo) + float(s_hi))})
    return out


def gap_posts(foot, dummy, spec, aperture=DEE_APERTURE_MM, size: float = POST_SIZE,
              offset: float = POST_OFFSET, overhang: float = POST_OVERHANG) -> list:
    """Posts on arbitrary gap edges.

    `spec` is a list of (gap, side, arclength) with side in {'dee', 'dummy'}. Giving both sides of
    one gap the same arclength puts a facing pair across it - a jaw that defines the outer edge of
    that gap's opening.
    """
    out = []
    seen = {}
    for gap, side, s in spec:
        edge, poly, owner = gap_edge(foot, dummy, int(gap), side)
        # A gap+side can carry more than one post (an inner/outer pair on the same edge), so the
        # name needs an index or the build log reads "g1_dee, g1_dee" and a misplaced post cannot
        # be told from its partner.
        key = (int(gap), side)
        seen[key] = seen.get(key, 0) + 1
        nm = f"g{int(gap)}_{side}"
        out.append(_post_on_edge(edge, poly, float(s), aperture, size, offset, overhang,
                                 nm, owner))
    # Index only where there is an actual collision, so a lone post keeps its plain name; and
    # number by arclength, so 1 is always the innermost of a group rather than whatever order the
    # spec happened to list.
    from collections import Counter
    cnt = Counter(q['name'] for q in out)
    for base, k in cnt.items():
        if k < 2:
            continue
        group = sorted((q for q in out if q['name'] == base), key=lambda q: q['s'])
        for i, q in enumerate(group, start=1):
            q['name'] = f"{base}_{i}"
    return out


def gap0_posts(foot, s_inner: float, s_outer: float, aperture=DEE_APERTURE_MM,
               size: float = POST_SIZE, offset: float = POST_OFFSET,
               overhang: float = POST_OVERHANG) -> list:
    """Two square posts on dee0's gap-0 edge, flanking the beam.

    The beam crosses that edge once per turn, and on turn 1 over a narrow band of radius. A post
    either side of that band leaves a slot the beam passes through and plugs the rest of the
    aperture - which is what gives the accelerating gap a defined opening, the job the old model's
    `bars` did and the thing this posts-free rebuild has been missing.

    ``s_inner`` / ``s_outer`` are ARCLENGTHS along the edge [mm] from its inner end, one either
    side of the beam. Arclength rather than radius because the edge is not radial, so equal steps
    in r are not equal steps along the metal.

    Each post is a `size` square in plan, its near face on the edge, its sides along the edge's
    local tangent and normal, extruded over the aperture height at its own radius plus `overhang`
    each way.
    """
    from matplotlib.path import Path as MplPath
    d0 = foot[0]
    edge = np.asarray(d0['lo'], dtype=float)
    seg = np.diff(edge, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    sl = np.concatenate([[0.0], np.cumsum(seg_len)])
    inside = MplPath(footprint_polygon(d0))

    posts = []
    for name, s in (('inner', float(s_inner)), ('outer', float(s_outer))):
        if not (0.0 <= s <= sl[-1]):
            raise ValueError(f"gap-0 post {name}: arclength {s:.2f} mm is off the edge "
                             f"(0 .. {sl[-1]:.2f} mm)")
        k = int(np.clip(np.searchsorted(sl, s) - 1, 0, len(seg) - 1))
        t = seg[k] / seg_len[k]
        p = edge[k] + (s - sl[k]) * t
        n = np.array([-t[1], t[0]])
        if not inside.contains_point(p + 0.25 * n):      # the normal that points INTO the dee
            n = -n
        if not inside.contains_point(p + 0.25 * n):
            raise RuntimeError(f"gap-0 post {name}: neither side of the edge is inside dee0")
        c = p + offset * n
        h = 0.5 * float(_height(np.hypot(*p), APERTURE_R_MM, aperture)) + overhang
        corners = _post_corners(c, t, n, size)
        posts.append({'name': f'post_{name}', 'owner': d0['label'], 's': s, 'point': p,
                      'tangent': t, 'normal': n, 'centre': c, 'corners': corners,
                      'half_height': h,
                      'r': float(np.hypot(*p)), 'r_centre': float(np.hypot(*c))})
    return posts


def print_posts(posts) -> None:
    print()
    print("=" * 96)
    print("GAP-0 POSTS")
    print("=" * 96)
    for q in posts:
        if q.get('kind') == 'window':
            print(f"  {q['name']:<14} on {q['owner']:<8} arclength {q['s_lo']:7.2f} .. "
                  f"{q['s_hi']:7.2f} mm at r ~ {q['r_mid']:7.3f} mm")
            print(f"                 block z = {q['z_lo']:+.1f} .. {q['z_hi']:+.1f} mm "
                  f"(opening {q['opening'][0]:+.1f} .. {q['opening'][1]:+.1f} = "
                  f"{q['opening'][1] - q['opening'][0]:.1f} mm; aperture half-height "
                  f"{q['ap_half']:.1f} mm)")
            continue
        print(f"  {q['name']:<14} on {q.get('owner', '?'):<8} arclength {q['s']:7.2f} mm, edge "
              f"point ({q['point'][0]:+8.3f}, {q['point'][1]:+8.3f}) at r = {q['r']:7.3f} mm")
        print(f"                 centre ({q['centre'][0]:+8.3f}, {q['centre'][1]:+8.3f}) at r = "
              f"{q['r_centre']:7.3f} mm, z = {-q['half_height']:+.1f} .. {q['half_height']:+.1f} mm")
    if len(posts) == 2 and posts[0].get('owner') == posts[1].get('owner'):
        gap = abs(posts[1]['s'] - posts[0]['s']) - POST_SIZE
        print(f"  slot between them: {gap:.2f} mm along the edge "
              f"(centres {abs(posts[1]['s'] - posts[0]['s']):.2f} mm apart, posts {POST_SIZE:.0f} mm wide)")


def build_model(foot, dummy, sim, brick, half: float = 50.0, wall_t: float = DEE_WALL,
                r_cut: float = R_CUT_MM, lc: float = 8.0, verbose: bool = True,
                aperture=DEE_APERTURE_MM, posts=None, dummy_aperture=None):
    """The whole central region as OCC solids, hollowed and fused. Leaves gmsh initialised.

    ``dummy_aperture`` (inner, outer) [mm] gives the DUMMY DEES a different vertical opening from
    the dees. None = the same as ``aperture``. An asymmetric gap is a vertical lens: the wider side
    has the longer fringe field, so a crossing that enters wide and leaves narrow is focused more
    than one that does the reverse. It also changes how well each electrode shields its own
    midplane, and so the accelerating voltage the beam actually sees there.
    """
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("central_region")
    occ = gmsh.model.occ
    t0 = time.time()

    def say(m):
        if verbose:
            print(f"[build] {m}  ({time.time() - t0:.1f} s)")

    # ---- 1. the housing: its outline extruded, with the beam channel cut out ---------------
    hs = _occ_loop(occ, housing_edges(sim), -half)
    house = [t for d, t in occ.extrude([(2, hs)], 0, 0, 2.0 * half) if d == 3][0]
    if brick is not None:
        ch_s = _occ_polygon(occ, brick['near'][:, :2], 0.0)
        occ.remove([(2, ch_s)], recursive=True)
        # the prism starts `back` mm UPSTREAM of the emitting plane, so the face that closes it
        # is not the plane the particles are born on (see beam_channel)
        p = [occ.addPoint(*q) for q in brick['start']]
        ls = [occ.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
        face = occ.addPlaneSurface([occ.addCurveLoop(ls)])
        d = (brick['extend'] + brick['back']) * brick['sweep']
        chan = [t for dd, t in occ.extrude([(2, face)], *d) if dd == 3][0]
        out, _ = occ.cut([(3, house)], [(3, chan)], removeObject=True, removeTool=True)
        house = [t for dd, t in out if dd == 3][0]
    say("housing solid with the beam recess")

    # ---- 2. the eight footprints, extruded ------------------------------------------------
    solids, polys = {}, {}
    for d in list(foot) + list(dummy):
        polys[d['label']] = footprint_polygon(d)          # kept for the plots and the area checks
        s = _occ_loop(occ, footprint_edges(d), -half)
        solids[d['label']] = [t for dd, t in occ.extrude([(2, s)], 0, 0, 2.0 * half) if dd == 3][0]
    say(f"{len(solids)} prisms extruded over z = {-half:+.0f} .. {half:+.0f} mm")

    # ---- 3. hollow each one with its own aperture tool -------------------------------------
    r_tool = {}
    for d in foot:
        r_in = float(np.hypot(*d['lo'][0]))
        r_tool[d['label']] = (corner_post_radius(d, wall_t) if d['label'] == 'dee0'
                              else r_in + wall_t)
    for d in dummy:
        r_tool[d['label']] = 0.0          # all the way to the axis: splits it into two plates
    pieces = {}
    _dummy_ap = tuple(aperture) if dummy_aperture is None else tuple(dummy_aperture)
    _dummy_labels = {d['label'] for d in dummy}
    if _dummy_ap != tuple(aperture):
        say(f"asymmetric gap: dees {tuple(aperture)} mm, dummy dees {_dummy_ap} mm")
    for label, vol in solids.items():
        _ap = _dummy_ap if label in _dummy_labels else tuple(aperture)
        tool = _occ_profile_revolve(occ, _aperture_ring(r_tool[label], r_cut + 20.0, h=_ap))
        out, _ = occ.cut([(3, vol)], [(3, tool)], removeObject=True, removeTool=True)
        pieces[label] = [t for dd, t in out if dd == 3]
    say("hollowed: " + ", ".join(f"{k} -> {len(v)}" for k, v in sorted(pieces.items())))

    # ---- 3b. the gap-0 posts, fused into dee0 ---------------------------------------------
    # AFTER hollowing, deliberately: the aperture tool is a full revolution, so a post fused
    # before it would simply be cut out again over exactly the height it is meant to block.
    if posts:
        # Group by owner: a post on a dummy-dee edge belongs to that dummy dee, not to a dee, and
        # it has to be fused BEFORE the dummies are merged into the housing (step 4 below) or it
        # would be fused into a solid that no longer exists under that name.
        by_owner = {}
        for q in posts:
            by_owner.setdefault(q.get('owner', 'dee0'), []).append(q)
        for owner, qs in sorted(by_owner.items()):
            if owner not in pieces:
                raise RuntimeError(f"posts named owner {owner!r}, which is not a built solid "
                                   f"({sorted(pieces)})")
            tags = []
            for q in qs:
                # a post straddles the median plane; a window block sits entirely above or below
                # the opening, so it carries its own z_lo / z_hi
                z0 = float(q.get('z_lo', -q['half_height']))
                z1 = float(q.get('z_hi', q['half_height']))
                f = _occ_polygon(occ, q['corners'], z0)
                v = [t for d, t in occ.extrude([(2, f)], 0, 0, z1 - z0) if d == 3]
                if len(v) != 1:
                    raise RuntimeError(f"{q['name']}: extrude gave {len(v)} volumes")
                tags.append(v[0])
            out, _ = occ.fuse([(3, t) for t in pieces[owner]], [(3, t) for t in tags],
                              removeObject=True, removeTool=True)
            pieces[owner] = [t for dd, t in out if dd == 3]
            say(f"fused {len(tags)} post(s) into {owner} -> {len(pieces[owner])} solid(s): "
                + ", ".join(q['name'] for q in qs))

    # ---- 4. fuse the dummy dees onto the housing ------------------------------------------
    ground = [(3, house)]
    for d in dummy:
        ground += [(3, t) for t in pieces.pop(d['label'])]
    out, _ = occ.fuse([ground[0]], ground[1:], removeObject=True, removeTool=True)
    ground = [t for dd, t in out if dd == 3]
    occ.synchronize()
    say(f"ground = housing + 4 dummy dees -> {len(ground)} solid(s)")

    n_orphan = _strip_orphans(gmsh, occ)
    say(f"stripped {n_orphan} construction entities that belong to no solid "
        f"(the revolve sketches and their edges)")

    groups = {'ground': ground}
    for d in foot:
        groups[d['label']] = pieces[d['label']]
    # Physical groups on the VOLUMES for bookkeeping, and on their SURFACES because only 2D gets
    # meshed: gmsh.write saves the elements that belong to a physical group, so volume-only
    # groups leave an EMPTY .msh once generate(2) is all that has run.
    for name, vols in groups.items():
        gmsh.model.addPhysicalGroup(3, vols, name=name)
        faces = sorted({abs(s) for v in vols
                        for _, s in gmsh.model.getBoundary([(3, v)], oriented=False)})
        gmsh.model.addPhysicalGroup(2, faces, name=f"{name}_surface")
    # A uniform size: without switching the point / curvature / boundary drivers off, gmsh
    # refines on every small feature and lc is only an upper bound it never approaches (lc = 10
    # gave 123k triangles at an effective ~2 mm). The BEM here is DENSE - 8 N^2 bytes - so the
    # element count is the binding constraint, not the geometry.
    for opt, val in (("Mesh.MeshSizeMax", lc), ("Mesh.MeshSizeMin", 0.5 * lc),
                     ("Mesh.MeshSizeFromPoints", 0), ("Mesh.MeshSizeFromCurvature", 0),
                     ("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.Algorithm", 6)):
        gmsh.option.setNumber(opt, val)
    return gmsh, groups, r_tool, polys


def mesh_and_report(gmsh, groups, prefix: str = None, verbose: bool = True):
    """Surface-mesh the model and report per-electrode element counts and quality."""
    t0 = time.time()
    gmsh.model.mesh.generate(2)
    say = (lambda m: print(f"[mesh] {m}")) if verbose else (lambda m: None)
    say(f"generated in {time.time() - t0:.1f} s")

    surf_of = {}
    for name, vols in groups.items():
        for v in vols:
            for _, s in gmsh.model.getBoundary([(3, v)], oriented=False):
                surf_of[abs(s)] = name
    # getNodes returns (tags, COORDINATES, parametric coords); the third is empty unless
    # asked for, and taking it as the coordinates is what made this report 0 nodes
    nt, nc, _ = gmsh.model.mesh.getNodes()
    nodes = {int(t): i for i, t in enumerate(nt)}
    verts = np.asarray(nc, dtype=float).reshape(-1, 3) / MM          # mm -> m
    tris, pots, owner = [], [], []
    per = {}
    for s, name in surf_of.items():
        et, _, ec = gmsh.model.mesh.getElements(2, s)
        for t, conn in zip(et, ec):
            if int(t) != 2:                                          # 3-node triangles only
                continue
            c = np.asarray(conn, dtype=np.int64).reshape(-1, 3)
            idx = np.vectorize(nodes.__getitem__)(c)
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




def to_electrode_model(gmsh, groups, dee_voltage: float, half_height: float, r_cut: float,
                       aperture=DEE_APERTURE_MM, verbose: bool = True):
    """The meshed model as a PyCentralRegion ElectrodeModel, ready for solve_gap_field and for
    every consumer of build_electrodes_3d's output.

    The contract those consumers rely on (solid_meshes, metal_mask, stacked_obstacles,
    _draw_model3d) is that the triangles are ordered in CONTIGUOUS blocks per solid and that
    params['solids'] gives each block's [n0, n1) range - so the triangles are gathered group by
    group here rather than surface by surface.
    """
    from PyCentralRegion.gap_fields import ElectrodeModel

    nt, nc, _ = gmsh.model.mesh.getNodes()
    idx = {int(t): i for i, t in enumerate(nt)}
    verts = np.asarray(nc, dtype=float).reshape(-1, 3) / MM            # mm -> m
    blocks, pots, solids, n = [], [], [], 0
    for name in sorted(groups, key=lambda k: (not k.startswith('dee'), k)):
        got = []
        for v in groups[name]:
            for _, s in gmsh.model.getBoundary([(3, v)], oriented=False):
                et, _, ec = gmsh.model.mesh.getElements(2, abs(s))
                for t, conn in zip(et, ec):
                    if int(t) == 2:
                        got.append(np.vectorize(idx.__getitem__)(
                            np.asarray(conn, dtype=np.int64).reshape(-1, 3)))
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
    area = 0.5 * np.linalg.norm(np.cross(verts[tris[:, 1]] - verts[tris[:, 0]],
                                         verts[tris[:, 2]] - verts[tris[:, 0]]), axis=1)
    bad = int(np.sum(area < 1e-14))
    if bad:
        raise RuntimeError(f"{bad} degenerate triangles in the 3D electrode mesh")

    z_cut, r_c = half_height / MM, r_cut / MM
    params = {'dim': 3, 'z_cut': z_cut, 'r_cut': r_c, 'solids': solids,
              'source': 'cr_geom_build', 'dee_aperture': True, 'ground_aperture': True,
              'dee_height': False, 'valley_height': None,
              'bars': [], 'posts': [], 'extra_solids': [], 'voltage_profile': None,
              'profiles': {'dee_aperture': [[0.0, aperture[0] / MM],
                                            [APERTURE_R_MM[0] / MM, aperture[0] / MM],
                                            [APERTURE_R_MM[1] / MM, aperture[1] / MM],
                                            [1.0, aperture[1] / MM]]},
              # every consumer reads the radial layout from here; the dees stop at r_cut, so the
              # blend onto the COMSOL map has to start inside it
              'layout': {'r_features': r_c, 'seam': (r_c - 0.055, r_c - 0.030), 'r_cut': r_c,
                         'z_cut': z_cut}}
    model = ElectrodeModel(vertices=verts, triangles=tris, potentials=pots, wedges=[],
                           params=params)
    if verbose:
        print(f"[model] {len(tris):,} triangles, {len(verts):,} nodes; "
              + ", ".join(f"{s['name']} {s['tri_range'][1] - s['tri_range'][0]}" for s in solids))
        print(f"[model] z_cut {z_cut * MM:.1f} mm, r_cut {r_c * MM:.1f} mm, "
              f"seam {params['layout']['seam'][0] * MM:.0f}..{params['layout']['seam'][1] * MM:.0f} mm")
    return model


def bem_solve(verts, tris, owner, dee_voltage: float, solver: str = 'auto',
              tol: float = 1e-5, verbose: bool = True, restart: int = None):
    """Laplace Dirichlet solve on the meshed model through PyCentralRegion's own solver."""
    from PyCentralRegion.gap_fields import ElectrodeModel, solve_gap_field
    pots = np.where(np.char.startswith(owner.astype(str), 'dee'), dee_voltage, 0.0)
    model = ElectrodeModel(vertices=verts, triangles=tris, potentials=pots,
                           params={'source': 'cr_geom_build'})
    n = len(tris)
    gb = 8.0 * n * n / 1024 ** 3
    if verbose:
        print(f"[bem] {n:,} elements -> dense matrix {gb:.2f} GB; "
              f"{int((pots != 0).sum()):,} at {dee_voltage:+.0f} V, "
              f"{int((pots == 0).sum()):,} grounded")
    t0 = time.time()
    sol = solve_gap_field(model, tol=tol, solver=solver, verbose=verbose, restart=restart)
    if verbose:
        print(f"[bem] solved in {time.time() - t0:.1f} s  [solver {sol.solver}, backend "
              f"{sol.device_interface}, assembly {sol.assembly_time_s:.1f} s, solve "
              f"{sol.solve_time_s:.1f} s, {sol.n_iterations} it, residual "
              f"{sol.final_residual:.3e}]")

    # Does the solution reproduce its own boundary condition? Evaluate the potential just off the
    # centroid of a sample of elements, along the outward normal: it must come back to that
    # element's Dirichlet value. This is the check that says the solve is meaningful rather than
    # merely finished - it is not implied by the residual.
    c = verts[tris].mean(axis=1)
    nrm = np.cross(verts[tris][:, 1] - verts[tris][:, 0], verts[tris][:, 2] - verts[tris][:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-30)[:, None]
    rng = np.random.default_rng(0)
    err = {}
    for name in sorted(set(owner.tolist())):
        sel = np.where(owner == name)[0]
        k = sel[rng.choice(len(sel), size=min(400, len(sel)), replace=False)]
        want = pots[k]
        for sign in (+1.0, -1.0):                       # normals may point either way
            got = sol.potential(c[k] + sign * 2.0e-4 * nrm[k])
            e = np.abs(got - want)
            err[name] = min(err.get(name, np.inf), float(np.median(e)))
    if verbose:
        print("[bem] boundary condition recovered 0.2 mm off the surface (median |error|):")
        for name in sorted(err):
            v = abs(dee_voltage) if name.startswith('dee') else abs(dee_voltage)
            print(f"[bem]   {name:<8s} {err[name]:9.1f} V   ({100.0 * err[name] / v:.2f} % of "
                  f"the dee voltage)")
    return model, sol


# ============================================================================
# Looking at a tracking run in 3D, against the geometry it was tracked through
# ============================================================================

def load_tracks(data_pkl: Path, which: str = 'final_sc') -> dict:
    """The tracked-particle paths out of a run's ``bunch_simulation3d<TAG>_data.pkl``.

    Returns the record the deck saved (ids, steps, xyz [m], stop_step, birth_step, ...) with the
    loss information already turned into a per-path live-length, so callers do not have to know
    how the tracker marks a dead slot: after a particle is lost its row keeps whatever the engine
    left there, which draws as a spike to nowhere if taken at face value.
    """
    import pickle
    with open(data_pkl, 'rb') as fh:
        data = pickle.load(fh)
    run = data.get(which) or data.get('final_sc') or data.get('final') or {}
    tr = run.get('tracks')
    if tr is None:
        raise SystemExit(
            f"{data_pkl} has no 'tracks' (runs before HCHC60_N_TRACKS existed, or N_TRACKS=0). "
            f"Available: {sorted(run)}")
    steps = np.asarray(tr['steps'], dtype=int)
    xyz = np.asarray(tr['xyz'], dtype=float)                     # (n_saved, n_ids, 3) [m]
    stop = np.asarray(tr['stop_step'], dtype=int)
    birth = np.asarray(tr.get('birth_step', np.zeros(len(stop))), dtype=int)
    n_valid, first = [], []
    for j in range(xyz.shape[1]):
        ok = (steps >= birth[j]) & (steps <= stop[j]) & np.isfinite(xyz[:, j, :]).all(axis=1)
        idx = np.nonzero(ok)[0]
        first.append(int(idx[0]) if len(idx) else 0)
        n_valid.append(int(idx[-1] - idx[0] + 1) if len(idx) else 0)
    return {**tr, 'steps': steps, 'xyz': xyz, 'stop_step': stop, 'birth_step': birth,
            'first': np.array(first), 'n_valid': np.array(n_valid),
            'lost': stop <= int(tr.get('n_steps', steps[-1] + 1)),
            'hits': np.asarray(run.get('hits', np.zeros((0, 4)))),
            'hits_z': np.asarray(run.get('hits_z', np.zeros(0)))}


def tracks_to_pos(tracks: dict, pos_path: Path, colour_by: str = 'turn',
                  n: int = None, seed: int = 0) -> dict:
    """Write the paths as a gmsh parsed post-processing file (one scalar-line view).

    gmsh draws a ``.pos`` view on top of whatever CAD is loaded, with its own colour scale and
    its own visibility toggle, so the trajectories can be turned on and off and the geometry
    rotated around them. Positions are converted to MILLIMETRES to match the model.

    ``colour_by``: 'turn' (how far round the particle got), 'z' (its height, for spotting where
    it goes vertical), or 'lost' (0 alive / 1 lost, to see only where the losses are).
    """
    steps, xyz = tracks['steps'], tracks['xyz']
    per_turn = float(tracks.get('steps_per_turn', 2000))
    ids = np.arange(xyz.shape[1])
    if n is not None and n < len(ids):
        ids = np.sort(np.random.default_rng(seed).choice(ids, size=n, replace=False))
    lines = ['View "trajectories ({})" {{'.format(colour_by)]
    n_drawn = 0
    for j in ids:
        i0, nv = int(tracks['first'][j]), int(tracks['n_valid'][j])
        if nv < 2:
            continue
        p = xyz[i0:i0 + nv, j, :] * 1000.0                        # m -> mm
        if colour_by == 'turn':
            v = steps[i0:i0 + nv] / per_turn
        elif colour_by == 'z':
            v = p[:, 2]
        elif colour_by == 'lost':
            v = np.full(nv, 1.0 if tracks['lost'][j] else 0.0)
        else:
            raise ValueError(f"colour_by must be 'turn', 'z' or 'lost', got {colour_by!r}")
        for k in range(nv - 1):
            a, b = p[k], p[k + 1]
            lines.append("SL({:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}){{{:.5g},{:.5g}}};".format(
                a[0], a[1], a[2], b[0], b[1], b[2], v[k], v[k + 1]))
        n_drawn += 1
    lines.append('};')

    # Every particle that ended in or on metal, as its own view - the WHOLE run, not just the
    # drawn subset, since that is the loss map worth looking at. MetalTerminator records
    # (id, step, x, y) in metres and the matching z in hits_z, appended in lockstep. Particles
    # lost vertically are NOT here: they left through |z| > Z_MAX, they did not hit anything.
    loss_view = None
    hits = tracks['hits']
    if len(hits):
        loss_view = 1
        lines.append('View "lost to metal" {')
        hz = tracks['hits_z']
        for i, h in enumerate(hits):
            z = float(hz[i]) if i < len(hz) else 0.0
            lines.append("SP({:.4f},{:.4f},{:.4f}){{{:.5g}}};".format(
                h[2] * 1000.0, h[3] * 1000.0, z * 1000.0, int(h[1]) / per_turn))
        lines.append('};')
    Path(pos_path).write_text("\n".join(lines) + "\n")
    return {'n_drawn': n_drawn, 'loss_view': loss_view, 'n_loss': int(len(hits))}


_SI_DEFAULT = ('000_SI_Anode', '001_SI_Cathode', '004_Entrance_Aperture',
               '005_ent0', '006_ext0', '011_ext4')


def _si_steps(which: str, si_dir: Path = None, outline: Path = None):
    """The spiral inflector's exported STEPs, in the BASELINE frame in millimetres.

    Same frame and units as the central-region geometry - `ExportLevelled` writes them that way -
    so they can be merged straight in with no transformation. That is the point: if the two do not
    line up in the view, the disagreement is real.

    `which` is 'default' (the electrodes and apertures), 'all' (adds the 12 triplet rods),
    '+housing' (adds the 5.9 MB housing B-rep, slow to load), or an explicit comma-separated list
    of file stems.
    """
    d = Path(si_dir) if si_dir else (_default_export_dir(outline) / 'steps_baseline_mm')
    if not d.is_dir():
        raise SystemExit(f"spiral inflector STEPs not found: {d}")
    # the inflector export is .stp since 2026-10-16 (.step before); _negative solids are not electrodes
    avail = {f.stem: f for f in sorted(list(d.glob('*.step')) + list(d.glob('*.stp')))
             if not f.stem.lower().endswith('_negative')}
    w = (which or 'default').strip().lower()
    if w in ('default', '1', 'yes', 'true', ''):
        names = list(_SI_DEFAULT)
    elif w == 'all':
        names = list(_SI_DEFAULT) + [k for k in avail if k.startswith(('007_', '008_', '009_',
                                                                      '010_', '012_', '013_',
                                                                      '014_', '015_', '016_',
                                                                      '017_', '018_', '019_'))]
    elif w in ('+housing', 'housing'):
        names = list(_SI_DEFAULT) + ['002_Housing', '003_Housing_Cylinder']
    else:
        names = [v.strip() for v in which.split(',')]
    out = []
    for nm in names:
        if nm in avail:
            out.append(avail[nm])
        else:
            hit = [f for k, f in avail.items() if nm.lower() in k.lower()]
            if hit:
                out.extend(hit)
            else:
                print(f"[view] no spiral-inflector STEP matching {nm!r} in {d}")
    return out


def _default_export_dir(outline: Path = None) -> Path:
    """Where the spiral inflector's exported STEPs live.

    Derived, never hardcoded: the outline the housing is rebuilt from already sits in that very
    folder (<export>/steps_baseline_mm/housing_outline.txt), so its parent IS the STEP directory.
    Falling back to a literal absolute path was worse than useless - one got mangled on the way
    into the file and pointed at a directory that cannot exist.
    """
    if outline:
        p = Path(outline)
        if p.name.lower().endswith('.txt'):
            p = p.parent
        if (p / '000_SI_Anode.step').exists() or p.name == 'steps_baseline_mm':
            return p.parent
        return p
    env = os.environ.get('HCHC60_EXPORT', '').strip()
    if env:
        return Path(env)
    h = os.environ.get('HCHC60_HANDOFF', '').strip()
    if h:
        return Path(h).parent
    raise SystemExit("cannot locate the spiral inflector STEPs: pass --si-dir, or set "
                     "HCHC60_EXPORT / HCHC60_HANDOFF, or give --outline")


def design_orbit_pos(result, pos_path: Path, r_max: float = None) -> int:
    """The design trajectory the cavities were built from, as its own gmsh view.

    This is `trajectory_reference` off the optimized-geometry pickle - the single orbit every gap
    line was laid out around (see gap_centre_line / snap_nodes_between_turns). Drawn beside the
    tracked particles it shows at a glance whether the bunch is following the orbit the geometry
    was designed for, or has been pushed off it.
    """
    trj = np.asarray(getattr(result, 'trajectory_reference', np.zeros((0, 3))), dtype=float)
    if len(trj) < 2:
        return 0
    # to MILLIMETRES first, and clip in millimetres. The pickle keeps the orbit in metres while
    # everything in this file (and the .step) is mm, so clipping before the conversion compares
    # r [m] against a limit in mm and silently never clips.
    p = trj[:, :3] * (1000.0 if np.abs(trj[:, :2]).max() < 10.0 else 1.0)
    if r_max is not None:
        p = p[np.hypot(p[:, 0], p[:, 1]) <= float(r_max)]
        if len(p) < 2:
            return 0
    turn = np.arange(len(p), dtype=float) / max(len(p) - 1, 1)
    lines = ['View "design orbit" {']
    for k in range(len(p) - 1):
        a, b = p[k], p[k + 1]
        lines.append("SL({:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}){{{:.5g},{:.5g}}};".format(
            a[0], a[1], a[2], b[0], b[1], b[2], turn[k], turn[k + 1]))
    lines.append('};')
    Path(pos_path).write_text("\n".join(lines) + "\n")
    return len(p)


def view_tracks(step_path: Path, data_pkl: Path, n: int = 60, colour_by: str = 'turn',
                which: str = 'final_sc', gui: bool = True, seed: int = 0,
                pos_path: Path = None, result=None, r_max: float = None,
                png: Path = None,
                si: str = None, si_dir: Path = None, si_outline: Path = None) -> None:
    """Open the central-region geometry in gmsh with a random selection of trajectories.

    ``step_path`` is the model's .step (``--build`` writes it next to the .msh), ``data_pkl`` a
    run's ``bunch_simulation3d<TAG>_data.pkl``. Rotate freely; each of the geometry, the tracked
    paths, the loss points and the design orbit is its own view, so any of them can be hidden.

    ``result`` is the optimized-geometry pickle's result object; pass it to overlay the DESIGN
    trajectory the cavities were laid out around.
    """
    import gmsh
    tracks = load_tracks(Path(data_pkl), which=which)
    si_files = _si_steps(si, si_dir, si_outline) if si else []
    pos = Path(pos_path) if pos_path else Path(data_pkl).with_suffix('').with_name(
        Path(data_pkl).stem + f"_tracks_{colour_by}.pos")
    info = tracks_to_pos(tracks, pos, colour_by=colour_by, n=n, seed=seed)
    lost = int(np.count_nonzero(tracks['lost']))
    print(f"[view] {info['n_drawn']} of {tracks['xyz'].shape[1]} tracked paths -> {pos}")
    print(f"[view] {lost} of {len(tracks['lost'])} tracked particles were lost to metal; "
          f"{info['n_loss']} particles ended in or on metal in the whole run (the red dots)")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.merge(str(step_path))
    gmsh.model.occ.synchronize()
    cr_vols = {t for _, t in gmsh.model.getEntities(3)}
    cr_surfs = {t for _, t in gmsh.model.getEntities(2)}
    for f in si_files:
        gmsh.merge(str(f))
    if si_files:
        gmsh.model.occ.synchronize()
        new_s = [t for _, t in gmsh.model.getEntities(2) if t not in cr_surfs]
        # the inflector in its own colour, so the two models are told apart at a glance. Both are
        # already in the baseline frame in mm, so nothing is transformed here - if they do not
        # line up, that is a real disagreement and not a units or frame artefact.
        for t in new_s:
            gmsh.model.setColor([(2, t)], 30, 160, 90, 110)
        print(f"[view] spiral inflector: {len(si_files)} STEP file(s), "
              f"{len(new_s)} surfaces -> " + ", ".join(f.name for f in si_files[:6])
              + (" ..." if len(si_files) > 6 else ""))
    gmsh.merge(str(pos))
    # the CAD as a translucent shell, so the trajectories INSIDE it stay visible: transparency is
    # the alpha of the surface colour, not a surface type
    gmsh.option.setNumber("Geometry.Surfaces", 1)
    gmsh.option.setNumber("Geometry.Points", 0)
    gmsh.option.setNumber("Geometry.Curves", 1)
    gmsh.option.setNumber("Geometry.Volumes", 0)
    gmsh.option.setColor("Geometry.Surfaces", 160, 160, 170, 40)
    gmsh.option.setColor("Geometry.Curves", 90, 90, 100, 120)
    gmsh.option.setNumber("General.Trackball", 1)
    gmsh.option.setNumber("General.RotationCenterGravity", 1)

    # the design orbit the cavities were laid out around, merged last so its view index is known
    orbit_view = None
    if result is not None:
        # No clip by default: the tracked particles run well past the 305 mm model too (the RF
        # map takes over outside the seam), so cropping only the reference would make the bunch
        # look as though it had left the orbit. --orbit-r-max crops both eye and orbit when the
        # central region alone is wanted.
        opos = Path(pos).with_name(Path(pos).stem + "_orbit.pos")
        n_orb = design_orbit_pos(result, opos, r_max=r_max)
        if n_orb:
            orbit_view = int(gmsh.option.getNumber("PostProcessing.NbViews"))
            gmsh.merge(str(opos))
            print(f"[view] design orbit: {n_orb:,} points -> {opos}")

    for v in range(int(gmsh.option.getNumber("PostProcessing.NbViews"))):
        gmsh.option.setNumber(f"View[{v}].LineWidth", 2.0)
        gmsh.option.setNumber(f"View[{v}].ShowScale", 1)
    if orbit_view is not None:
        # one solid colour and a fat line: this is a reference to read the particles against, not
        # another scalar field competing for the eye
        gmsh.option.setNumber(f"View[{orbit_view}].LineWidth", 4.0)
        gmsh.option.setNumber(f"View[{orbit_view}].ShowScale", 0)
        oopt = Path(pos).with_name(Path(pos).stem + "_orbit.opt")
        oopt.write_text("View[{}].ColorTable = {{Black}};\n".format(orbit_view))
        gmsh.merge(str(oopt))
    lv = info['loss_view']
    if lv is not None:
        # Small flat red dots. A scalar view is coloured by its COLORMAP: setting
        # View[].Color.Points is accepted and then ignored, which draws the losses as a rainbow
        # by turn number (checked, not assumed). A one-entry colour table is what actually pins
        # them to one colour, and it has to go through the parser - there is no setColorTable in
        # the API. PointType 0 (flat squares), not 1 (shaded spheres): there can be tens of
        # thousands of these.
        gmsh.option.setNumber(f"View[{lv}].PointSize", 4.0)
        gmsh.option.setNumber(f"View[{lv}].PointType", 0)
        gmsh.option.setNumber(f"View[{lv}].ShowScale", 0)
        opt = Path(pos).with_suffix('.opt')
        opt.write_text("View[{}].ColorTable = {{Red}};\n".format(lv))
        gmsh.merge(str(opt))
    if png is not None:
        gmsh.option.setNumber("General.GraphicsWidth", 1100)
        gmsh.option.setNumber("General.GraphicsHeight", 820)
        gmsh.option.setNumber("General.RotationX", 290)
        gmsh.option.setNumber("General.RotationY", 0)
        gmsh.option.setNumber("General.RotationZ", 25)
        gmsh.fltk.initialize()
        gmsh.fltk.update()
        gmsh.write(str(png))
        print(f"[view] snapshot -> {png}")
    if gui:
        print("  opening the gmsh GUI (close the window to continue) ...")
        gmsh.fltk.run()
    gmsh.finalize()


def beam_frames(data_pkl: Path, out: Path = None, which: str = 'final_sc',
                n_max: int = 20000, colour: str = 'E') -> Path:
    """Radial-longitudinal snapshots: the beam from above, turned to put its motion along +y.

    One panel per recorded snapshot time. Each panel takes the particles alive and born at that
    instant, rotates the x-y plane so the bunch's MEAN momentum points along +y, and centres on
    the bunch centroid. So +y' is the longitudinal direction (bunch length, and the head-tail
    order) and x' the radial one, both in the machine midplane - the top-down view.

    The rotation is recomputed per snapshot from that snapshot's own mean momentum, so the frames
    follow the bunch round the machine instead of being fixed to the lab axes.
    """
    import pickle
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(data_pkl, 'rb') as fh:
        data = pickle.load(fh)
    run = data.get(which) or data.get('final_sc') or data.get('final') or {}
    snaps = run.get('snapshots') or []
    if not snaps:
        raise SystemExit(f"{data_pkl} has no snapshots for {which!r}")

    cols = min(4, len(snaps))
    rows = int(np.ceil(len(snaps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    rng = np.random.default_rng(0)
    lims = []
    frames = []
    for s in snaps:
        xyz, v = np.asarray(s['xyz']), np.asarray(s['vxyz'])
        keep = np.asarray(s['born'], dtype=bool) & ~np.asarray(s['lost'], dtype=bool)
        keep &= np.isfinite(xyz).all(axis=1) & np.isfinite(v).all(axis=1)
        if keep.sum() > n_max:
            idx = rng.choice(np.nonzero(keep)[0], size=n_max, replace=False)
            keep = np.zeros(len(keep), bool)
            keep[idx] = True
        p, vv = xyz[keep], v[keep]
        if not len(p):
            frames.append(None)
            continue
        pm = vv[:, :2].mean(axis=0)
        pm = pm / np.linalg.norm(pm)                  # mean momentum direction, in x-y
        # rotate so pm -> +y:  y' along pm, x' 90 deg clockwise of it (outward-ish, right-handed)
        ey = pm
        ex = np.array([ey[1], -ey[0]])
        c = p[:, :2].mean(axis=0)
        q = p[:, :2] - c
        xp, yp = q.dot(ex) * 1000.0, q.dot(ey) * 1000.0
        cval = (np.asarray(s['E_mev'])[keep] if colour == 'E' else p[:, 2] * 1000.0)
        frames.append((s, xp, yp, cval, np.degrees(np.arctan2(ey[1], ey[0])), int(keep.sum())))
        lims.append(max(np.abs(xp).max(), np.abs(yp).max()))

    lim = 1.05 * max(lims) if lims else 1.0
    for ax, fr in zip(axes.ravel(), frames):
        if fr is None:
            ax.set_axis_off()
            continue
        s, xp, yp, cval, ang, nk = fr
        sc = ax.scatter(xp, yp, c=cval, s=1.2, cmap='viridis', linewidths=0, rasterized=True)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.axhline(0.0, color='0.75', lw=0.5, zorder=0)
        ax.axvline(0.0, color='0.75', lw=0.5, zorder=0)
        r_c = np.hypot(*np.asarray(s['ref_xyz'])[:2]) * 1000.0
        ax.set_title("t = {:.0f} ns   N = {:,}\nref r = {:.0f} mm, heading {:+.0f} deg".format(
            s['t_ns'], nk, r_c, ang), fontsize=8)
        ax.set_xlabel("x' radial [mm]", fontsize=8)
        ax.set_ylabel("y' along motion [mm]", fontsize=8)
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("E [MeV]" if colour == 'E' else "z [mm]", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(frames):]:
        ax.set_axis_off()
    fig.suptitle("HCHC-60 central region: beam in the midplane, turned to put mean momentum "
                 "along +y'", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(out) if out else Path(data_pkl).with_name(Path(data_pkl).stem + "_rz_frames.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[frames] {len([f for f in frames if f])} snapshots -> {out}")
    return out


def plot_sections(model_pkl: Path, z_list=None, out: Path = None, r_max: float = 320.0,
                  orbit=None, posts=None) -> Path:
    """x-y slices of a built model at a set of heights.

    Every solid's plane section at each z (electrodes3d.section_loops), which is the SAME
    even-odd outline the obstacle raster uses - so this shows the metal as the tracker sees it,
    not an idealised drawing of it. Nested loops are holes and are drawn unfilled, so a hollowed
    dee reads as an annulus rather than a blob.

    The heights are chosen to straddle the dee aperture: inside it the electrodes are hollow and
    the posts stand alone in the opening; outside it they are solid and the posts merge into the
    plate, which is exactly the contrast worth seeing.
    """
    import pickle
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    sys.path.insert(0, os.environ.get('PYCENTRALREGION_SRC',
                                      r"D:\Dropbox (Personal)\Code\Python\PyCentralRegion\src"))
    from PyCentralRegion.electrodes3d import solid_meshes, loops_from_mesh

    with open(model_pkl, 'rb') as fh:
        model = pickle.load(fh)
    if z_list is None:
        z_list = [0.0, 10.0, 16.0, 18.0, 24.0, 40.0]
    meshes = solid_meshes(model)
    colour = {'dee0': '#d62728', 'dee1': '#ff7f0e', 'dee2': '#2ca02c', 'dee3': '#9467bd',
              'ground': '#1f77b4'}

    cols = min(3, len(z_list))
    rows = int(np.ceil(len(z_list) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.6 * rows), squeeze=False)
    for ax, z in zip(axes.ravel(), z_list):
        n_loops = 0
        for name, m in meshes.items():
            for lp in loops_from_mesh(m, z=z * 1e-3, name=name):
                p = np.asarray(lp) * 1000.0
                ax.add_patch(MplPolygon(p, closed=True, facecolor=colour.get(name, '0.6'),
                                        edgecolor='k', lw=0.35, alpha=0.75, zorder=2))
                n_loops += 1
        if orbit is not None and len(orbit):
            o = np.asarray(orbit)
            ax.plot(o[:, 0], o[:, 1], 'k-', lw=0.8, zorder=4, label='design orbit')
        if posts:
            for q in posts:
                if abs(z) <= q['half_height']:
                    c = q['centre']
                    ax.plot(c[0], c[1], 'w*', ms=9, mec='k', mew=0.6, zorder=5)
        ax.set_aspect('equal')
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        ax.set_title(f"z = {z:+.0f} mm   ({n_loops} loops)", fontsize=10)
        ax.set_xlabel("x [mm]", fontsize=8)
        ax.set_ylabel("y [mm]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.15, lw=0.4)
    for ax in axes.ravel()[len(z_list):]:
        ax.set_axis_off()
    fig.suptitle(f"{Path(model_pkl).name}: metal in the x-y plane, as the obstacle test sees it",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(out) if out else Path(model_pkl).with_name(Path(model_pkl).stem + "_sections.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[sections] {len(z_list)} slices -> {out}")
    return out


def _find_handoff(outline_path: Path):
    """The export folder's hand-off file: the outline lives in <export>/steps_baseline_mm/."""
    for d in (outline_path.parent.parent, outline_path.parent):
        for fn in ('handoff_sc_all_level.h5', 'handoff_vac_all_level.h5'):
            if (d / fn).exists():
                return d / fn
    return None


def inside_housing(points: np.ndarray, wall: np.ndarray, half_height: float) -> np.ndarray:
    """Which points are inside the extruded wall: inside its outline in x-y AND within +-H in z."""
    from matplotlib.path import Path as MplPath
    p = np.atleast_2d(np.asarray(points, dtype=float))
    return MplPath(wall).contains_points(p[:, :2]) & (np.abs(p[:, 2]) <= half_height)


def print_brick(handoff: Path, plane: dict, brick: dict, wall: np.ndarray, half_height: float,
                cloud: np.ndarray = None, recess: dict = None) -> None:
    n, p0 = plane['normal'], plane['point']
    d = brick['dist']
    print()
    print("=" * 96)
    print("EMITTING PLANE AND BEAM CHANNEL")
    print("=" * 96)
    print(f"  from {handoff}")
    print(f"    {plane['n_particles']:,} particles, flat to {plane['flatness']:.2e} mm, "
          f"spread {plane['extent'][0]:.2f} x {plane['extent'][1]:.2f} mm")
    print(f"    point  ({p0[0]:+9.4f}, {p0[1]:+9.4f}, {p0[2]:+9.4f}) mm")
    print(f"    normal ({n[0]:+9.5f}, {n[1]:+9.5f}, {n[2]:+9.5f})   "
          f"mean beam direction is {np.degrees(np.arccos(np.clip(plane['beam'].dot(n), -1, 1))):.3f} deg off it")
    a_u = brick['aperture'][1] - brick['aperture'][0]
    a_w = brick['aperture'][3] - brick['aperture'][0]
    a_n = np.cross(a_u / np.linalg.norm(a_u), a_w / np.linalg.norm(a_w))
    print(f"    the aperture plane is {np.degrees(np.arccos(np.clip(abs(a_n.dot(n)), -1, 1))):.3f} deg off it")
    print()
    print("  THE RECTANGLE (aperture corners projected normally onto the plane)")
    print(f"    edges  {brick['edges'][0]:.4f} x {brick['edges'][1]:.4f} x {brick['edges'][2]:.4f} "
          f"x {brick['edges'][3]:.4f} mm   (aperture {brick['ap_edges'][0]:.4f} x {brick['ap_edges'][1]:.4f})")
    print(f"    corners {', '.join('{:.4f}'.format(v) for v in brick['angles'])} deg "
          f"-> out of square by at most {np.abs(brick['angles'] - 90.0).max():.2e} deg")
    print(f"    area {brick['face_area']:.4f} mm^2, shortened from the aperture's "
          f"{brick['ap_edges'][0] * brick['ap_edges'][1]:.4f} mm^2 by "
          f"{brick['ap_edges'][0] * brick['ap_edges'][1] - brick['face_area']:.4f} mm^2")
    print(f"    swept {brick['extend']:.1f} mm along ({brick['sweep'][0]:+.5f}, {brick['sweep'][1]:+.5f}, "
          f"{brick['sweep'][2]:+.5f}) - the plane normal, downstream: a right prism")
    print(f"    base face set back {brick['back']:.1f} mm UPSTREAM of the emitting plane, so the "
          f"metal that closes the channel is clear of the plane the particles are born on")

    # does the emitting plane really sit inside the wall?
    print()
    print("  IS THE EMITTING PLANE INSIDE THE HOUSING?")
    ins = inside_housing(brick['near'], wall, half_height)
    for j, (p, i) in enumerate(zip(brick['near'], ins)):
        print(f"    near-face corner {j}: ({p[0]:+9.3f}, {p[1]:+9.3f}, {p[2]:+9.3f}) mm   "
              f"{'INSIDE' if i else 'OUTSIDE'} the wall")
    if cloud is not None:
        ic = inside_housing(cloud, wall, half_height)
        print(f"    all {len(cloud):,} hand-off particles: {int(ic.sum()):,} inside, "
              f"{int((~ic).sum()):,} outside; z {cloud[:, 2].min():+.3f} .. {cloud[:, 2].max():+.3f} mm "
              f"(wall {-half_height:+.1f} .. {half_height:+.1f})")
    far_in = inside_housing(brick['far'], wall, half_height)
    print(f"    far-face corners: {int(far_in.sum())} of 4 inside -> the prism "
          f"{'does NOT clear' if far_in.any() else 'clears'} the wall")

    print()
    print(f"  {'i':>4} {'rectangle (emitting plane)':>34}   "
          f"{'far face (+{:.0f} mm)'.format(brick['extend']):>34}   {'to aperture':>11}")
    print(f"  {'':>4} {'[mm]':>34}   {'[mm]':>34}   {'[mm]':>11}")
    print("  " + "-" * 91)
    for j in range(4):
        q, p = brick['near'][j], brick['far'][j]
        print(f"  {j:>4} ({q[0]:+9.3f}, {q[1]:+9.3f}, {q[2]:+9.3f})   "
              f"({p[0]:+9.3f}, {p[1]:+9.3f}, {p[2]:+9.3f})   {d[j]:11.3f}")
    print(f"    the aperture sits {d.min():.3f} .. {d.max():.3f} mm along the sweep (mean {d.mean():.3f}), "
          f"so the prism passes through it and runs on to {brick['extend']:.1f} mm")
    if d.min() <= 0.0:
        print("    WARNING: the aperture crosses the emitting plane - the prism starts past it")

    if recess is not None:
        print()
        print("  THE RECESS THIS CUTS IN THE HOUSING")
        print(f"    it breaks out through the flat face ({recess['wall_a'][0]:+.3f}, "
              f"{recess['wall_a'][1]:+.3f}) -> ({recess['wall_b'][0]:+.3f}, {recess['wall_b'][1]:+.3f}), "
              f"whose normal is {recess['tilt']:.3f} deg off the sweep")
        print(f"    {'i':>4} {'to the wall':>13} {'to the aperture':>17} {'difference':>13}")
        for j in range(4):
            print(f"    {j:>4} {recess['depth'][j]:13.3f} {d[j]:17.3f} "
                  f"{recess['depth'][j] - d[j]:13.3f}")
        print(f"    {'mean':>4} {recess['depth'].mean():13.3f} {d.mean():17.3f} "
              f"{recess['depth'].mean() - d.mean():13.3f}")
        print(f"    recess volume {recess['volume']:.3f} mm^3 = {brick['face_area']:.3f} mm^2 x "
              f"{recess['depth'].mean():.3f} mm mean depth")
        print(f"    the opening it leaves in the wall is {recess['opening']:.3f} mm^2, "
              f"{recess['opening'] - brick['face_area']:.3f} mm^2 more than the channel's own "
              f"cross-section, because the wall is tilted to it")
        print( "    NOTE the opening is the aperture's SHADOW along the sweep, not the aperture "
               "itself: the two")
        print(f"    planes are {recess['tilt']:.1f} deg apart, so the corners break out "
              f"{(recess['depth'] - d).min():+.3f} .. {(recess['depth'] - d).max():+.3f} mm off it.")


def _seg_distance(p, a, b):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-30), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def main():
    ap_ = argparse.ArgumentParser(description="central-region geometry builder, step 1")
    ap_.add_argument('--pkl', default=os.environ.get('HCHC60_RESULT_PKL', str(_DEF_PKL)),
                     help="optimizer result with the gap geometry")
    ap_.add_argument('--outline', default=os.environ.get('HCHC60_HOUSING_OUTLINE',
                                                         str(_DEF_OUTLINE) if _DEF_OUTLINE else ''),
                     help="housing outline text file from the spiral-inflector export")
    ap_.add_argument('--plot', nargs='?', const='', default=None, metavar='PNG',
                     help="plot the outline over the gap lines; a file name saves it, bare --plot shows it")
    ap_.add_argument('--dees', nargs='?', const='', default=None, metavar='PNG',
                     help="step A of the dees: the boundary chains, plotted over the gap centre "
                          "lines; a file name saves the figure, bare --dees shows it")
    ap_.add_argument('--r-max', type=float, default=R_MAX_DEE_MM,
                     help="the dees stop at this radius [mm] (default %(default)s)")
    ap_.add_argument('--build', nargs='?', const='', default=None, metavar='PREFIX',
                     help="build the whole model in gmsh (dees + dummy dees + housing), mesh it, "
                          "and write PREFIX.msh / PREFIX.step")
    ap_.add_argument('--solve', action='store_true', help="after --build, run the BEM solve")
    ap_.add_argument('--pickle', nargs='?', const='', default=None, metavar='PKL',
                     help="write the model as a PyCentralRegion ElectrodeModel pickle, for "
                          "HCHC60_MODEL3D_PKL (default: <PREFIX>_model3d.pkl)")
    ap_.add_argument('--bem-solver', default='auto',
                     help="auto / cupy / scipy / strong / dense")
    ap_.add_argument('--lc', type=float, default=8.0, help="mesh size [mm] (default %(default)s)")
    ap_.add_argument('--restart', type=int, default=None,
                     help="GMRES restart depth; the default is 1000, which only checks "
                          "convergence every 1000 iterations")
    ap_.add_argument('--aperture', default="{:g},{:g}".format(*DEE_APERTURE_MM),
                     help="beam channel full height 'inner,outer' [mm] (default %(default)s)")
    ap_.add_argument('--dee-wall', type=float, default=DEE_WALL,
                     help="radial thickness [mm] of the dee inner wall (default %(default)s)")
    ap_.add_argument('--dummy-overlap', type=float, default=DUMMY_OVERLAP,
                     help="how far the dummy dees reach inside the housing [mm] (default %(default)s)")
    ap_.add_argument('--no-snap', action='store_true',
                     help="do NOT move the segment node radii midway between the turn crossings")
    ap_.add_argument('--cavity-clearance-inner', type=float, default=CAVITY_CLEARANCE_INNER,
                     help="gaps 1..7 stop this far [mm] from the housing; gap 0 is cut where it "
                          "meets gap 1 instead (default %(default)s)")
    ap_.add_argument('--junction-tol', type=float, default=JUNCTION_TOL_DEG,
                     help="close a gap's variable->fixed junction when its net angular step is "
                          "under this [deg] (default %(default)s); 0 disables it")
    ap_.add_argument('--simplify', nargs='?', const='', default=None, metavar='PNG',
                     help="two long lines + clusters -> spline; a file name saves the figure, bare --simplify shows it")
    ap_.add_argument('--cluster-tol', type=float, default=0.5,
                     help="a vertex within this distance [mm] of a long line is in that line's cluster")
    ap_.add_argument('--keep-every', type=int, default=5, help="keep every Nth non-cluster vertex")
    ap_.add_argument('--geo', nargs='?', const='', default=None, metavar='GEO',
                     help="write the simplified outline as a gmsh .geo curve loop "
                          "(default: output_inflector/cr_housing_outline.geo)")
    ap_.add_argument('--gui', action='store_true', help="open that .geo in the gmsh GUI")
    ap_.add_argument('--half-height', type=float, default=50.0,
                     help="the wall is extruded over -H .. +H [mm] (default 50)")
    ap_.add_argument('--brick-extend', type=float, default=200.0,
                     help="sweep the beam channel this far [mm] downstream along the emitting "
                          "plane's normal, so it runs through the aperture and clear of the wall")
    ap_.add_argument('--view-tracks', metavar='DATA_PKL',
                     help="open the geometry in gmsh with trajectories from a run's "
                          "bunch_simulation3d<TAG>_data.pkl (needs --step, or --build's prefix)")
    ap_.add_argument('--step', metavar='STEP',
                     help="the model .step to view (default: <--build prefix>.step)")
    ap_.add_argument('--n-tracks', type=int, default=200,
                     help="how many trajectories to draw (default 200 = every path a normal run "
                          "saves, since HCHC60_N_TRACKS is 200; a dense pass holds far more and "
                          "would need this raised deliberately)")
    ap_.add_argument('--colour-by', default='turn', choices=('turn', 'z', 'lost'),
                     help="what to colour the trajectories by (default turn)")
    ap_.add_argument('--which-run', default='final_sc',
                     help="which run in the data pkl to read (default final_sc)")
    ap_.add_argument('--gap0-posts', metavar='S_IN,S_OUT',
                     help="two square posts on dee0's gap-0 edge at these arclengths [mm] from "
                          "its inner end, flanking the beam (omit for no posts)")
    ap_.add_argument('--sections', metavar='MODEL_PKL',
                     help="x-y slices of a built model's metal, at --section-z heights")
    ap_.add_argument('--section-z', metavar='Z,...',
                     help="heights [mm] for --sections (default 0,10,16,18,24,40)")
    ap_.add_argument('--gap-posts', metavar='G:SIDE:S,...',
                     help="extra posts as 'gap:side:arclength' triples, side dee|dummy, e.g. "
                          "'1:dee:120,1:dummy:118'; both sides of a gap at the same arclength "
                          "make a facing pair")
    ap_.add_argument('--gap-windows', metavar='G:SIDE:SLO:SHI:ZLO:ZHI,...',
                     help="vertical apertures: blocks above and below an opening ZLO..ZHI [mm] "
                          "over arclength SLO..SHI of that gap edge, e.g. '0:dee:31.6:53.2:-13:13'")
    ap_.add_argument('--post-size', type=float, default=POST_SIZE,
                     help=f"post square size [mm] (default {POST_SIZE})")
    ap_.add_argument('--post-offset', type=float, default=POST_OFFSET,
                     help=f"post centre from the dee edge [mm] (default {POST_OFFSET}: face flush)")
    ap_.add_argument('--post-overhang', type=float, default=POST_OVERHANG,
                     help=f"post reach above/below the aperture [mm] (default {POST_OVERHANG})")
    ap_.add_argument('--orbit-r-max', type=float, default=None, metavar='MM',
                     help="crop the design-orbit overlay at this radius [mm] (default: no crop)")
    ap_.add_argument('--si', nargs='?', const='default', metavar='WHICH',
                     help="overlay the spiral inflector's exported STEPs (baseline frame, mm - no "
                          "transformation applied): 'default' = electrodes + apertures, 'all' adds "
                          "the 12 triplet rods, '+housing' adds the 5.9 MB housing B-rep, or a "
                          "comma-separated list of file stems")
    ap_.add_argument('--si-dir', metavar='DIR',
                     help="where those STEPs live (default: <export>/steps_baseline_mm)")
    ap_.add_argument('--png', metavar='PNG',
                     help="write a snapshot of the view to this PNG instead of opening the GUI")
    ap_.add_argument('--no-orbit', action='store_true',
                     help="do not overlay the design trajectory on --view-tracks")
    ap_.add_argument('--frames', metavar='DATA_PKL',
                     help="radial-longitudinal snapshot series (top-down, turned so mean "
                          "momentum is +y) from a run's data pkl")
    ap_.add_argument('--frames-colour', default='E', choices=('E', 'z'),
                     help="colour the snapshot particles by energy or height (default E)")
    ap_.add_argument('--post-shape', choices=('square', 'cylinder'), default='square',
                     help="post cross-section: 'square' (the original 10x10 prism) or "
                          "'cylinder' (a 16-gon of the same width, no sharp edges in the gap)")
    ap_.add_argument('--dummy-aperture', metavar='INNER,OUTER', default='',
                     help="vertical opening of the DUMMY DEES in mm, if different from "
                          "--aperture: e.g. --aperture '26,50' --dummy-aperture '34,50' "
                          "gives a 26 mm dee against a 34 mm dummy dee in the centre, "
                          "both 50 mm outboard. Empty = same as --aperture.")
    ap_.add_argument('--aperture-inset', metavar='LONG,SHORT', default='0,0',
                     help="shrink the housing exit aperture by this many mm off EACH "
                          "side: LONG on the longer edge (the 40 mm, radial one), "
                          "SHORT on the shorter (the 23 mm, vertical one). "
                          "'10,0' makes it 20 x 23 mm. Default '0,0' = as exported.")
    ap_.add_argument('--channel-back', type=float, default=CHANNEL_BACK,
                     help="start the beam channel this far [mm] UPSTREAM of the emitting plane, "
                          f"so its closed face is not the plane the particles are born on "
                          f"(default {CHANNEL_BACK})")
    ap_.add_argument('--handoff', default=os.environ.get('HCHC60_HANDOFF', ''),
                     help="hand-off h5 whose particles define the emitting plane "
                          "(default: handoff_sc_all_level.h5 beside the outline's export folder)")
    a = ap_.parse_args()

    # Looking at a finished run needs the run and the .step, not the gap pickle or the outline,
    # so these are answered before anything is rebuilt.
    if a.sections:
        zs = ([float(v) for v in a.section_z.split(',')] if a.section_z else None)
        plot_sections(Path(a.sections), z_list=zs)
    if a.frames:
        beam_frames(Path(a.frames), which=a.which_run, colour=a.frames_colour)
    if a.view_tracks:
        step = Path(a.step) if a.step else (Path(a.build).with_suffix('.step') if a.build else None)
        if step is None or not Path(step).exists():
            sys.exit(f"model .step not found: pass --step (tried {step})")
        # the design orbit lives on the gap pickle; without it the view simply has no overlay
        res = None
        if not a.no_orbit and a.pkl and Path(a.pkl).exists():
            try:
                _, _, res = load_gap_lines(Path(a.pkl))
            except Exception as exc:                                        # noqa: BLE001
                print(f"[view] no design orbit ({type(exc).__name__}: {exc})")
        view_tracks(Path(step), Path(a.view_tracks), n=a.n_tracks, colour_by=a.colour_by,
                    which=a.which_run, gui=not a.png, result=res, r_max=a.orbit_r_max,
                    png=Path(a.png) if a.png else None, si=a.si, si_dir=a.si_dir,
                    si_outline=Path(a.outline) if a.outline else None)
    if a.frames or a.view_tracks or a.sections:
        return

    pkl = Path(a.pkl)
    if not pkl.exists():
        sys.exit(f"gap geometry not found: {pkl}")
    if not a.outline:
        sys.exit("no housing outline given: pass --outline or set HCHC60_HOUSING_OUTLINE / HCHC60_EXPORT")
    outline_path = Path(a.outline)
    if not outline_path.exists():
        sys.exit(f"housing outline not found: {outline_path}")

    dees, geom, result = load_gap_lines(pkl)

    # Same move as the 2D BEM: put every variable-segment node radius midway between the two turns
    # that bracket it, so a segment joint - a kick-direction change, or the bridge jog at the
    # variable->fixed junction - never coincides with a beam crossing. This REBUILDS the gap
    # geometry, so it has to happen before anything is measured off it.
    trj = np.asarray(getattr(result, 'trajectory_reference', np.zeros((0, 3))), dtype=float)
    if not a.no_snap and len(trj) > 2:
        print()
        print("=" * 96)
        print("SEGMENT NODES: DROP THE INERT ONES, THEN SNAP MIDWAY BETWEEN TURNS")
        print("=" * 96)
        neutralise_inert_segments(dees, trj, verbose=True)
        print()
        snap_nodes_between_turns(dees.gaps, trj, verbose=True)
    elif a.no_snap:
        print("\n  (--no-snap: segment node radii left where the optimizer put them)")

    print_gap_lines(dees, geom, result)

    ol = read_housing_outline(outline_path)
    print_housing_outline(ol, outline_path)

    _ins = tuple(float(v) for v in a.aperture_inset.split(','))
    if len(_ins) != 2:
        raise SystemExit("--aperture-inset wants 'long,short' in mm")
    if any(v != 0.0 for v in _ins):
        ol['aperture'] = inset_aperture(ol['aperture'], _ins[0], _ins[1])

    if a.plot is not None:
        plot_outline(ol, dees, out=(a.plot or None))

    # the simplified outline is what the dees keep their clearance from, so it is needed here too
    sim = None
    if (a.dees is not None or a.simplify is not None or a.geo is not None or a.gui
            or a.build is not None):
        # A file with a `curves` section was simplified by the spiral inflector already, and
        # simplifying it again here would apply a second, different set of rules to it.
        sim = (outline_as_given(ol) if ol.get('curves')
               else simplify_outline(ol['outer'], cluster_tol=a.cluster_tol,
                                     keep_every=a.keep_every))

    if a.dees is not None:
        print_junctions(dees, a.junction_tol)
        foot = dee_boundaries(dees, sim['curve'], r_max=a.r_max, junction_tol=a.junction_tol,
                              clearance=a.cavity_clearance_inner)
        print_dee_boundaries(foot)
        orbit = reference_orbit(result, r_max=a.r_max)
        print_orbit(foot, orbit, result)
        plot_dee_boundaries(foot, ol, out=(a.dees or None), orbit=orbit)

    if a.build is not None:
        foot = dee_boundaries(dees, sim['curve'], r_max=a.r_max, junction_tol=a.junction_tol,
                              clearance=a.cavity_clearance_inner)
        dummy = dummy_boundaries(dees, sim['curve'], r_max=a.r_max,
                                 junction_tol=a.junction_tol, overlap=a.dummy_overlap)
        print_dummies(dummy)
        ho = Path(a.handoff) if a.handoff else _find_handoff(outline_path)
        brick = None
        if ho is not None:
            brick = beam_channel(ol['aperture'], emitting_plane(ho), extend=a.brick_extend,
                                 back=a.channel_back)
        global POST_SHAPE
        POST_SHAPE = a.post_shape
        if POST_SHAPE != 'square':
            print(f"  posts: {POST_SHAPE} ({POST_FACETS} facets), width {POST_SIZE:g} mm")
        ap_mm = tuple(float(v) for v in a.aperture.split(','))
        dummy_ap_mm = (tuple(float(v) for v in a.dummy_aperture.split(','))
                       if a.dummy_aperture else None)
        if dummy_ap_mm is not None and len(dummy_ap_mm) != 2:
            raise SystemExit("--dummy-aperture wants 'inner,outer' in mm")
        if len(ap_mm) != 2:
            raise SystemExit("--aperture wants 'inner,outer' in mm")
        posts = []
        if a.gap_posts:
            # "gap:side:arclength" triples, e.g. "1:dee:120,1:dummy:118,2:dee:..."
            spec = []
            for item in a.gap_posts.split(','):
                try:
                    g, sd, sv = item.split(':')
                    spec.append((int(g), sd.strip(), float(sv)))
                except ValueError:
                    raise SystemExit(f"--gap-posts wants 'gap:side:arclength' triples, got {item!r}")
            posts += gap_posts(foot, dummy, spec, aperture=ap_mm, size=a.post_size,
                               offset=a.post_offset, overhang=a.post_overhang)
        win_spec = []
        if a.gap_windows:
            # "gap:side:s_lo:s_hi:z_lo:z_hi" - the vertical aperture, z_lo..z_hi being the OPENING.
            # Parsed here but BUILT after every post exists, so the bounds can be snapped to the
            # post centres (see below).
            for item in a.gap_windows.split(','):
                f = item.split(':')
                if len(f) != 6:
                    raise SystemExit(f"--gap-windows wants 'gap:side:s_lo:s_hi:z_lo:z_hi', got {item!r}")
                win_spec.append((int(f[0]), f[1].strip(), float(f[2]), float(f[3]),
                                 float(f[4]), float(f[5])))
        if a.gap0_posts:
            try:
                s_in, s_out = (float(v) for v in a.gap0_posts.split(','))
            except ValueError:
                raise SystemExit("--gap0-posts wants 's_inner,s_outer' in mm of arclength")
            posts += gap0_posts(foot, s_in, s_out, aperture=ap_mm, size=a.post_size,
                                offset=a.post_offset, overhang=a.post_overhang)

        # ---- windows, snapped to the post centres ------------------------------------------
        # A window ALWAYS spans post centre to post centre. A cylindrical post's footprint tapers
        # to nothing at its edge, so a window that stops at the edge leaves an uncovered wedge -
        # which is the whole reason the rule exists. Snapping here rather than trusting the string
        # means a --gap-windows spec written under the old edge-to-edge convention cannot survive a
        # rebuild silently: cf4 carried stale edge-to-edge bounds on gaps 0, 1 and 2 (exactly
        # centre +- half a post) while gaps 3 and 4, placed fresh, were correct.
        import re as _re
        _centres = {}
        for q in (posts or []):
            m = _re.match(r'^g(\d+)_(dee|dummy)(?:_\d+)?$', str(q.get('name', '')))
            if m:
                _centres.setdefault((int(m.group(1)), m.group(2)), []).append(float(q['s']))
            elif str(q.get('name', '')).startswith('post_'):
                _centres.setdefault((0, 'dee'), []).append(float(q['s']))   # gap0_posts naming
        _snapped = []
        for g, side, s_lo, s_hi, z_lo, z_hi in win_spec:
            c = _centres.get((g, side))
            if c and len(c) >= 2:
                lo, hi = min(c), max(c)
                if abs(lo - s_lo) > 1e-6 or abs(hi - s_hi) > 1e-6:
                    _snapped.append(f"    gap {g} {side:<5} {s_lo:7.2f}..{s_hi:7.2f} -> "
                                    f"{lo:7.2f}..{hi:7.2f} mm "
                                    f"({s_lo - lo:+.2f} / {s_hi - hi:+.2f})")
                s_lo, s_hi = lo, hi
            elif c:
                print(f"  [window] gap {g} {side}: only one post on this edge, "
                      f"bounds left as given ({s_lo:.2f}..{s_hi:.2f} mm)")
            posts += gap_window(foot, dummy, g, side, s_lo, s_hi, z_lo, z_hi,
                                aperture=ap_mm, depth=a.post_size, overhang=a.post_overhang)
        if _snapped:
            print(f"\n  [window] snapped {len(_snapped)} window(s) to their post centres:")
            print("\n".join(_snapped))

        posts = posts or None
        if posts:
            print_posts(posts)
        gmsh_, groups, r_tool, polys = build_model(
            foot, dummy, sim, brick, half=a.half_height, wall_t=a.dee_wall,
            r_cut=a.r_max, lc=a.lc, aperture=ap_mm, posts=posts,
            dummy_aperture=dummy_ap_mm)
        print()
        print(f"  beam channel {ap_mm[0]:g} mm to r = {APERTURE_R_MM[0]:g} mm, "
              f"{ap_mm[1]:g} mm from r = {APERTURE_R_MM[1]:g} mm")
        print("  aperture tool inner radii [mm]: "
              + ", ".join(f"{k} {v:.3f}" for k, v in sorted(r_tool.items())))
        try:
            pre = a.build or str(HERE / 'output_inflector' / 'cr_model')
            verts, tris, owner, per = mesh_and_report(gmsh_, groups, prefix=pre)
            v_dee0 = -float(dees.gaps[0].voltage) * np.cos(np.deg2rad(dees.gaps[0].phase_deg))
            if a.pickle is not None:
                import pickle as _pickle
                model = to_electrode_model(gmsh_, groups, v_dee0, a.half_height, a.r_max,
                                           aperture=ap_mm)
                out_p = Path(a.pickle) if a.pickle else Path(pre + '_model3d.pkl')
                out_p.parent.mkdir(parents=True, exist_ok=True)
                with open(out_p, 'wb') as fh:
                    _pickle.dump(model, fh, protocol=4)
                print(f"[model] pickled -> {out_p}  "
                      f"(point HCHC60_MODEL3D_PKL at it)")
            if a.solve:
                v_dee = -float(dees.gaps[0].voltage) * np.cos(np.deg2rad(dees.gaps[0].phase_deg))
                bem_solve(verts, tris, owner, v_dee, solver=a.bem_solver, restart=a.restart)
        finally:
            gmsh_.finalize()

    if a.simplify is not None or a.geo is not None or a.gui:
        if sim['given']:
            print_given(ol, sim)
        else:
            print_simplified(ol, sim)
            if a.simplify is not None:
                plot_simplified(ol, sim, out=(a.simplify or None))
        if a.geo is not None or a.gui:
            ho = Path(a.handoff) if a.handoff else _find_handoff(outline_path)
            brick = None
            if ho is None:
                print("\n  NOTE: no hand-off file found, so no emitting plane and no beam channel; "
                      "pass --handoff to get one")
            else:
                plane = emitting_plane(ho)
                brick = beam_channel(ol['aperture'], plane, extend=a.brick_extend,
                                     back=a.channel_back)
                rec = recess_on_wall(brick, sim)
                print_brick(ho, plane, brick, sim['curve'], a.half_height,
                            cloud=plane['cloud'], recess=rec)
            geo = Path(a.geo) if a.geo else (HERE / 'output_inflector' / 'cr_housing_outline.geo')
            geo.parent.mkdir(parents=True, exist_ok=True)
            geo.write_text(geo_string(sim, half_height=a.half_height, brick=brick), encoding='utf-8')
            print(f"\n  wrote {geo}")
            open_in_gmsh(geo, ol['outer'], gui=a.gui, half_height=a.half_height)

    print()
    print("step 1 done: gap centre lines and housing outline read and printed.")
    print("next: your call - the outline is a polygon and a rectangle, so the housing can now be "
          "whatever we decide it should be.")


if __name__ == '__main__':
    main()
