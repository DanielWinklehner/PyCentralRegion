"""Printed reports and plots of the electrode layout, and views of built models and tracked runs.

Reports (what ``build.build`` prints as it goes): the gap centre lines, the housing outline, the variable -> fixed
junctions, the dee and dummy-dee boundaries, the reference orbit against them, and the hardware on the gap edges.
Plots: ``plot_dee_boundaries`` (the 2D layout, written as <model>_dees.png) and ``plot_sections`` (x-y slices of a
built model's metal, as the tracker's obstacle test sees it). Views: ``view_tracks`` opens a model's .step in gmsh
with a run's tracked particle paths, the loss points and the design orbit (``tracks_to_pos`` / ``design_orbit_pos``
write them as gmsh views); ``beam_frames`` plots a run's snapshots from above, turned so the bunch moves along +y'.
The runs are the tracker's ``track_data.pkl`` (or the design deck's ``bunch_simulation3d<TAG>_data.pkl``).

Lengths in mm. The builder's options are read through ``builder`` (configure() first).
"""
from pathlib import Path

import numpy as np

from . import builder as B
from .builder import MM


def _banner(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def _seg_distance(p, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-30), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _subplots(nrows: int, ncols: int, show: bool = False, **kw):
    """(figure, axes): a bare matplotlib Figure for a plot written to a file - no pyplot, so the process's backend and
    its open figures are left alone (a library call must not switch them) - or, ``show``, a pyplot figure for the
    screen."""
    if show:
        import matplotlib.pyplot as plt
        return plt.subplots(nrows, ncols, **kw)
    from matplotlib.figure import Figure
    fig = Figure(figsize=kw.pop('figsize', None))
    return fig, fig.subplots(nrows, ncols, **kw)


# ---------------------------------------------------------------------------------------------------------------------
# reports
def print_gap_lines(dees, result) -> None:
    m = B.MACHINE
    _banner("RF GAP CENTRE LINES")
    print(f"  {len(dees.gaps)} gaps on {dees.n_dees} dees, opening angle {B.opening_deg(dees):.3f} deg, "
          f"harmonic {m.harmonic}")
    print(f"  r_min {m.r_min_m * MM:.1f} mm, r_max {m.r_max_cavity_m * MM:.1f} mm, {m.dee_voltage_kv:.1f} kV/gap, "
          f"nominal gap width {m.gap_width_m * MM:.1f} mm")
    print(f"  source: bunch phase {float(getattr(result, 'bunch_phase_deg', np.nan)):+.4f} deg, "
          f"f_rev {float(getattr(result, 'rf_frequency_mhz', np.nan)):.6f} MHz")
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
            a1 = np.degrees(np.arctan2(p1[1], p1[0])) % 360.0
            a2 = np.degrees(np.arctan2(p2[1], p2[0])) % 360.0
            print(f"  {g:>4} {si:>4} {p1[0]:9.3f} {p1[1]:9.3f} {p2[0]:9.3f} {p2[1]:9.3f} "
                  f"{np.linalg.norm(p2 - p1):8.3f} {np.hypot(*p1):8.3f} {np.hypot(*p2):8.3f} {a1:8.3f} {a2:8.3f}")
        print("  " + "-" * 92)
    worst = 0.0                                   # a gap's segments should form ONE polyline, end to end
    for cav in dees.gaps:
        for si in range(len(cav.segments) - 1):
            worst = max(worst, float(np.linalg.norm(np.asarray(cav.segments[si]['p2'], dtype=float)[:2]
                                                    - np.asarray(cav.segments[si + 1]['p1'], dtype=float)[:2])))
    print("  continuity check (segment k end -> segment k+1 start, should be 0):")
    print(f"    worst gap between consecutive segments: {worst * MM:.6f} mm")


def print_housing_outline(ol: dict, path) -> None:
    outer, ap = ol['outer'], ol['aperture']
    n = len(outer)
    edges = np.roll(outer, -1, axis=0) - outer
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    prev = outer - np.roll(outer, 1, axis=0)
    cross = prev[:, 0] * edges[:, 1] - prev[:, 1] * edges[:, 0]
    area = 0.5 * float(np.sum(outer[:, 0] * np.roll(outer[:, 1], -1) - np.roll(outer[:, 0], -1) * outer[:, 1]))
    r = np.hypot(outer[:, 0], outer[:, 1])
    az = np.degrees(np.arctan2(outer[:, 1], outer[:, 0])) % 360.0
    _banner("HOUSING OUTLINE")
    print(f"  file   : {path}")
    print(f"  frame  : {ol['frame']}   units: {ol['units']} (converted to mm if they were metres)")
    print(f"  outer polygon: {n} vertices, perimeter {lengths.sum():.3f} mm, area {abs(area):.1f} mm^2, "
          f"{'counter-clockwise' if area > 0 else 'clockwise'}")
    print(f"    radius from the machine axis {r.min():.3f} .. {r.max():.3f} mm, edge length {lengths.min():.3f} .. "
          f"{lengths.max():.3f} mm")
    print(f"    convex: {'yes' if np.all(cross > 0) or np.all(cross < 0) else 'NO'}")
    print()
    print(f"  {'i':>4} {'x':>10} {'y':>10} {'r':>10} {'az':>9} {'edge':>9}")
    print(f"  {'':>4} {'[mm]':>10} {'[mm]':>10} {'[mm]':>10} {'[deg]':>9} {'[mm]':>9}")
    print("  " + "-" * 56)
    for i in range(n):
        print(f"  {i:>4} {outer[i, 0]:10.3f} {outer[i, 1]:10.3f} {r[i]:10.3f} {az[i]:9.3f} {lengths[i]:9.3f}")
    c = ap.mean(axis=0)
    fit = B._circle_fit(ap)
    nvec = np.linalg.svd(ap - c)[2][-1]                  # the opening's plane: its least-spread direction
    if nvec.dot(c[:2].tolist() + [0.0]) < 0.0:
        nvec = -nvec
    print()
    if fit is not None and fit[2] < 0.01:
        print(f"  aperture: {len(ap)} corners, a round opening R {fit[1]:.3f} mm")
    else:
        u, w = ap[1] - ap[0], ap[-1] - ap[0]
        cos_uw = abs(float(u.dot(w))) / max(float(np.linalg.norm(u) * np.linalg.norm(w)), 1e-30)
        print(f"  aperture: {len(ap)} corners, {np.linalg.norm(u):.3f} x {np.linalg.norm(w):.3f} mm "
              f"(edges perpendicular to {np.degrees(np.arcsin(min(cos_uw, 1.0))):.4f} deg)")
    print(f"  {'i':>4} {'x':>10} {'y':>10} {'z':>10} {'r':>10} {'az':>9}")
    print(f"  {'':>4} {'[mm]':>10} {'[mm]':>10} {'[mm]':>10} {'[mm]':>10} {'[deg]':>9}")
    print("  " + "-" * 56)
    for i, p in enumerate(ap):
        print(f"  {i:>4} {p[0]:10.3f} {p[1]:10.3f} {p[2]:10.3f} {np.hypot(p[0], p[1]):10.3f} "
              f"{np.degrees(np.arctan2(p[1], p[0])) % 360.0:9.3f}")
    print(f"    centre ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}) mm, r {np.hypot(c[0], c[1]):.3f} mm, "
          f"azimuth {np.degrees(np.arctan2(c[1], c[0])) % 360.0:.3f} deg; z {ap[:, 2].min():+.3f} .. "
          f"{ap[:, 2].max():+.3f} mm (exit half height {np.abs(ap[:, 2]).max():.3f} mm)")
    print(f"    normal ({nvec[0]:+.5f}, {nvec[1]:+.5f}, {nvec[2]:+.5f}) -> azimuth "
          f"{np.degrees(np.arctan2(nvec[1], nvec[0])) % 360.0:.3f} deg, elevation "
          f"{np.degrees(np.arcsin(np.clip(nvec[2], -1, 1))):+.3f} deg")
    d = np.array([_seg_distance(c[:2], outer[i], outer[(i + 1) % n]) for i in range(n)])
    i = int(np.argmin(d))
    print(f"    nearest outer edge: {i} -> {(i + 1) % n}, centre {d[i]:.3f} mm from it")


def print_junctions(dees, junction_tol: float = None) -> None:
    junction_tol = B.OPT.junction_tol_deg if junction_tol is None else junction_tol
    _banner("GAP CENTRE LINES - the variable -> fixed junction")
    print("  The fixed segment is a RADIAL ray at the nominal base angle; the variable chain ends")
    print("  at base angle + excursions, rotated about its midpoint. The leftover is a jog.")
    print(f"  Closing it where the net angular step is <= {junction_tol:.2f} deg.")
    print()
    print(f"  {'gap':>4} {'excursion':>10} {'rotation':>9} {'net step':>9} {'jog':>8} "
          f"{'r_trans':>9} {'-> r_meet':>10} {'dr':>9} {'seg/ray':>8}  action")
    print(f"  {'':>4} {'[deg]':>10} {'[deg]':>9} {'[deg]':>9} {'[mm]':>8} {'[mm]':>9} {'[mm]':>10} {'[mm]':>9} "
          f"{'[deg]':>8}")
    print("  " + "-" * 94)
    for g, cav in enumerate(dees.gaps):
        j = B.junction(cav)
        nominal = j['jog'] < 1e-6
        close = abs(j['step_deg']) <= junction_tol
        r_meet = '     -    ' if nominal or not np.isfinite(j['r_meet']) else f"{j['r_meet']:10.3f}"
        dr = '    -    ' if nominal or not np.isfinite(j['dr']) else f"{j['dr']:+9.3f}"
        print(f"  {g:>4} {float(np.sum(cav.segment_angles)):10.4f} "
              f"{(float(cav.segment_rotations[-1]) if cav.segment_rotations else 0.0):9.4f} "
              f"{j['step_deg']:9.4f} {j['jog']:8.3f} {j['r_trans']:9.3f} {r_meet} {dr} {j['seg_deg']:8.3f}  "
              + ("nominal - collinear, no junction" if nominal else "CLOSED" if close else "left as it is"))


def print_dee_boundaries(foot) -> None:
    _banner("DEES - the boundary chains")
    print("  each dee runs from its entry gap to its exit gap, both centre lines offset half a")
    print(f"  gap width into the dee, clipped at r_max = {foot[0]['r_max']:.1f} mm")
    print()
    print(f"  {'dee':>6} {'span':>8} {'V':>9} {'phase':>8} {'pts':>9} {'r range':>18} {'width at':>10} {'widest':>16}")
    print(f"  {'':>6} {'[deg]':>8} {'[V]':>9} {'[deg]':>8} {'lo/hi':>9} {'[mm]':>18} {'r=100 mm':>10} {'[mm]':>16}")
    print("  " + "-" * 92)
    for d in foot:
        rr, w = B.dee_width(d)
        r_all = np.hypot(*np.vstack([d['lo'], d['hi']]).T)
        print(f"  {d['label']:>6} {d['span_deg']:8.3f} {d['voltage']:9.0f} {d['phase_deg']:8.2f} "
              f"{len(d['lo']):4d}/{len(d['hi']):<4d} {r_all.min():7.2f}..{r_all.max():7.2f} "
              f"{np.interp(100.0, rr, w):10.2f} {w.max():8.2f} at r={rr[np.argmax(w)]:5.0f}")
    print()
    print(f"  the inner ends, with the METAL held {foot[0]['clearance']:.2f} mm clear of the housing")
    print("  (measured on the offset boundary, not on the gap centre line):")
    print(f"    {'gap':>5} {'rule':>24} {'inner end [mm]':>22} {'r':>9} {'at the end':>11} {'closest':>9}")
    for d in foot:
        for gi, rec in zip(d['gaps'], d['inner']):
            p = rec['start']
            flag = '  <- under' if rec['min_d'] < foot[0]['clearance'] - 1e-3 else ''
            print(f"    {gi:>5} {rec['rule']:>24} ({p[0]:+8.3f}, {p[1]:+8.3f}) {np.hypot(*p):9.3f} {rec['d']:11.3f} "
                  f"{rec['min_d']:9.3f}{flag}")
    print()
    print("  the inner closures:")
    print(f"    {'dee':>6} {'kind':>14} {'r(lo)':>9} {'r(hi)':>9} {'r used':>9} {'trimmed':>9} {'sweep':>9} "
          f"{'length':>9} {'to housing':>11}")
    for d in foot:
        t = d['tip']
        ln = float(np.hypot(*np.diff(t['pts'], axis=0).T).sum()) if len(t['pts']) > 1 else 0.0
        rad = t.get('radius', np.nan)
        print(f"    {d['label']:>6} {t['kind']:>14} {t.get('r_lo', np.nan):9.3f} {t.get('r_hi', np.nan):9.3f} "
              f"{(rad if np.isfinite(rad) else float('nan')):9.3f} {t.get('trimmed', 0.0):9.3f} "
              f"{t.get('sweep_deg', 0.0):8.3f}d {ln:9.3f} {t['min_d']:11.3f}")
    print()
    print(f"  the outer arcs, at r_max = {foot[0]['r_max']:.1f} mm:")
    print(f"    {'dee':>6} {'lo end az':>11} {'hi end az':>11} {'arc':>10} {'length':>10} {'lo end r':>11} "
          f"{'hi end r':>11}")
    for d in foot:
        a0 = np.degrees(np.arctan2(d['lo'][-1][1], d['lo'][-1][0])) % 360.0
        a1 = np.degrees(np.arctan2(d['hi'][-1][1], d['hi'][-1][0])) % 360.0
        print(f"    {d['label']:>6} {a0:10.4f}d {a1:10.4f}d {d['arc_deg']:9.4f}d "
              f"{np.radians(d['arc_deg']) * d['r_max']:10.3f} {np.hypot(*d['lo'][-1]):11.5f} "
              f"{np.hypot(*d['hi'][-1]):11.5f}")
    print()
    print("  where each dee opens up: the radius beyond which it never again narrows below the given width")
    print(f"    {'dee':>6} " + " ".join("{:>13}".format(f"w >= {t:.0f} mm") for t in (0.0, 2.0, 5.0, 10.0)))
    for d in foot:
        rr, w = B.dee_width(d)
        cells = []
        for t in (0.0, 2.0, 5.0, 10.0):
            bad = np.where(w <= t)[0]
            cells.append(f"{rr[0]:8.2f} mm" if not len(bad) else f"{rr[bad[-1] + 1]:8.2f} mm"
                         if bad[-1] + 1 < len(rr) else 'never')
        print(f"    {d['label']:>6} " + " ".join("{:>13}".format(c) for c in cells))


def print_dummies(dummy) -> None:
    _banner("DUMMY DEES")
    print("  the grounded wedges between the dees: the other side of each dee's two gap lines,")
    print("  run inward PAST the housing surface so the union with it is a real overlap")
    print(f"    {'':>7} {'gaps':>8} {'inner ends [mm]':>34} {'into the housing':>18} {'tip':>9}")
    for d in dummy:
        print(f"    {d['label']:>7} {str(d['gaps']):>8} ({d['lo'][0][0]:+8.3f}, {d['lo'][0][1]:+8.3f}) "
              f"({d['hi'][0][0]:+8.3f}, {d['hi'][0][1]:+8.3f}) {d['depth'][0]:8.3f} {d['depth'][1]:8.3f} "
              f"{d['tip']['span']:9.3f}")
    if B.OPT.ground == 'star':
        print("  (ground = 'star': the dummy edges above place posts / windows only - the ground is built from the "
              "2D layout, see [star])")


def reference_orbit(result, r_max: float = None):
    """The seed's single-particle reference orbit in mm, x-y only, up to its first exit past r_max (or None)."""
    trj = getattr(result, 'trajectory_reference', None)
    if trj is None:
        return None
    trj = np.asarray(trj, dtype=float)
    if trj.ndim != 2 or len(trj) < 2:
        return None
    xy = trj[:, :2] * MM
    if r_max is not None:
        out = np.hypot(xy[:, 0], xy[:, 1]) > r_max
        if out.any() and not out.all():
            xy = xy[:int(np.argmax(out))]
    return xy


def print_orbit(foot, orbit, result) -> None:
    if orbit is None:
        print("\n  (no reference orbit in the seed)")
        return
    r = np.hypot(orbit[:, 0], orbit[:, 1])
    print()
    print(f"  the reference orbit: {len(orbit):,} points, r {r.min():.2f} .. {r.max():.2f} mm, "
          f"{int(getattr(result, 'n_turns', 0))} turns, final energy "
          f"{float(getattr(result, 'final_energy_mev', np.nan)):.3f} MeV")
    print("  The beam crosses the lo/hi edge lines by design; what matters in plan view is the inner")
    print("  closure (the tip must not sit where the beam is) and the reach of r_max.")
    print(f"    {'dee':>6} {'crossings of':>14} {'lo / hi':>10} {'to the inner arc':>18}")
    for d in foot:
        n_x = [len(B.edge_crossings(d[key], orbit, first_only=False)) for key in ('lo', 'hi')]
        tip = np.asarray(d['tip']['pts'], dtype=float)
        d_tip = float(np.min(np.hypot(*(tip[:, None, :] - orbit[None, :, :]).T))) if len(tip) else float('nan')
        print(f"    {d['label']:>6} {'the edges:':>14} {n_x[0]:4d} /{n_x[1]:4d} {d_tip:18.3f}")
    print(f"    the orbit reaches {r.max():.3f} mm against r_max = {foot[0]['r_max']:.1f} mm: "
          f"{foot[0]['r_max'] - r.max():+.3f} mm of margin")


def print_posts(posts) -> None:
    size = float(B.OPT.post_size_mm)
    _banner("HARDWARE ON THE GAP EDGES")
    for q in posts:
        if q.get('kind') == 'aperture':
            hc = q['hole_centre']
            print(f"  {q['name']:<14} on {q['owner']:<8} plate arclength {q['s_lo']:7.2f} .. {q['s_hi']:7.2f} mm, "
                  f"{q['depth']:.0f} mm deep, z {q['z_lo']:+.1f} .. {q['z_hi']:+.1f} mm")
            print(f"                 round hole R {q['hole_r']:.2f} mm centred at s {q['s']:.2f} mm (face point "
                  f"({hc[0]:+.3f}, {hc[1]:+.3f}) at r = {q['r_mid']:.3f} mm, z {hc[2]:+.2f} mm), axis along the edge "
                  f"normal ({q['hole_axis'][0]:+.3f}, {q['hole_axis'][1]:+.3f})")
        elif q.get('kind') == 'window':
            print(f"  {q['name']:<14} on {q['owner']:<8} arclength {q['s_lo']:7.2f} .. {q['s_hi']:7.2f} mm at r ~ "
                  f"{q['r_mid']:7.3f} mm")
            print(f"                 block z = {q['z_lo']:+.1f} .. {q['z_hi']:+.1f} mm (opening {q['opening'][0]:+.2f} "
                  f".. {q['opening'][1]:+.2f} = {q['opening'][1] - q['opening'][0]:.2f} mm; channel half height "
                  f"{q['ap_half']:.2f} mm)")
        elif q.get('kind') == 'box':
            c = np.asarray(q['corners'], dtype=float)
            print(f"  {q['name']:<14} on ground   x {c[:, 0].min():.2f} .. {c[:, 0].max():.2f}, y {c[:, 1].min():.2f} .. "
                  f"{c[:, 1].max():.2f} mm, z +-{q['half_height']:.1f} mm")
        else:
            print(f"  {q['name']:<14} on {q.get('owner', '?'):<8} arclength {q['s']:7.2f} mm, edge point "
                  f"({q['point'][0]:+8.3f}, {q['point'][1]:+8.3f}) at r = {q['r']:7.3f} mm")
            print(f"                 centre ({q['centre'][0]:+8.3f}, {q['centre'][1]:+8.3f}) at r = "
                  f"{q['r_centre']:7.3f} mm, z = {-q['half_height']:+.1f} .. {q['half_height']:+.1f} mm"
                  + (f", {q['length']:g} mm along the edge" if abs(q.get('length', size) - size) > 1e-9 else ''))
    plain = [q for q in posts if q.get('kind') is None]
    if len(plain) == 2 and plain[0].get('owner') == plain[1].get('owner'):
        gap = abs(plain[1]['s'] - plain[0]['s']) - size
        print(f"  slot between them: {gap:.2f} mm along the edge (centres {abs(plain[1]['s'] - plain[0]['s']):.2f} mm "
              f"apart, posts {size:g} mm wide)")


# ---------------------------------------------------------------------------------------------------------------------
# plots
def plot_dee_boundaries(foot, ol: dict, out=None, orbit=None) -> None:
    """Left: the boundary chains over the gap centre lines and the housing. Right: dee width against radius. Written
    to ``out``, else shown on the screen."""
    from matplotlib.patches import Circle

    fig, (ax, bx) = _subplots(1, 2, show=not out, figsize=(14.5, 7.2), gridspec_kw={'width_ratios': [1.25, 1.0]})
    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    ring = np.vstack([ol['outer'], ol['outer'][0]])
    ax.plot(ring[:, 0], ring[:, 1], '-', color='0.6', lw=1.2, zorder=2, label='housing outline')
    if orbit is not None:
        ax.plot(orbit[:, 0], orbit[:, 1], '-', color='k', lw=0.7, alpha=0.75, zorder=7, label='reference orbit')
    for i, d in enumerate(foot):
        c = colours[i % len(colours)]
        for key in ('centre_lo', 'centre_hi'):
            ch = d[key]
            ax.plot(ch[:, 0], ch[:, 1], '--', color='0.8', lw=0.9, zorder=1,
                    label='gap centre lines' if (i == 0 and key == 'centre_lo') else None)
        m_lo, m_hi = B.metal_edges(d)
        ax.plot(m_lo[:, 0], m_lo[:, 1], '-', color=c, lw=1.6, zorder=3, label=f"{d['label']} boundaries")
        ax.plot(m_hi[:, 0], m_hi[:, 1], '-', color=c, lw=1.6, zorder=3)
        ax.plot(d['arc'][:, 0], d['arc'][:, 1], '-', color=c, lw=2.8, zorder=4, label='arc at r_max' if i == 0 else None)
        ax.plot(d['tip']['pts'][:, 0], d['tip']['pts'][:, 1], '-', color=c, lw=2.8, zorder=4,
                label='inner closure' if i == 0 else None)
    for i, d in enumerate(foot):
        for j, closed in zip(d['junction'], d['closed']):
            if closed:
                ax.plot(*j['meet'], 'o', mfc='none', mec='k', ms=9, mew=1.4, zorder=6,
                        label='junction closed' if i == 0 and j is d['junction'][0] else None)
            else:
                ax.plot([j['var_end'][0], j['fix_start'][0]], [j['var_end'][1], j['fix_start'][1]], '-', color='k',
                        lw=2.2, zorder=6, label='junction jog (left)' if i == 0 else None)
    ax.plot([0], [0], '+', color='k', ms=10, mew=1.5, zorder=5)
    r_max = foot[0]['r_max']
    for rr in np.arange(50.0, r_max + 1e-9, 50.0):
        ax.add_patch(Circle((0, 0), rr, fill=False, color='0.9', lw=0.8, zorder=0))
    ax.add_patch(Circle((0, 0), r_max, fill=False, color='0.7', lw=1.0, ls=':', zorder=0))
    ax.set_xlim(-1.08 * r_max, 1.08 * r_max)
    ax.set_ylim(-1.08 * r_max, 1.08 * r_max)
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('dee boundaries: each gap centre line offset half a gap width into the dee\n'
                 f'grey circles every 50 mm, dotted = r_max = {r_max:.0f} mm', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.2)
    for i, d in enumerate(foot):
        rr, w = B.dee_width(d)
        bx.plot(rr, w, '-', color=colours[i % len(colours)], lw=1.6, label=d['label'])
    bx.axhline(0.0, color='k', lw=1.0)
    bx.set_xscale('log')
    bx.set_xlabel('radius [mm]  (log)')
    bx.set_ylabel('width across the dee [mm]')
    bx.set_title('where each dee actually begins:\nwidth <= 0 means the two boundaries have crossed', fontsize=10)
    bx.legend(fontsize=8)
    bx.grid(alpha=0.25, which='both')
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=140)
        print(f"  wrote {out}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


def plot_sections(model_pkl, z_list=None, out=None, r_max: float = None, orbit=None, posts=None) -> Path:
    """x-y slices of a built model at a set of heights [mm]: every solid's plane section (electrodes3d.section_loops,
    the even-odd outline the obstacle raster uses), so this shows the metal as the tracker sees it. The heights
    straddle the channel: inside it the electrodes are hollow and the posts stand alone in the opening. Each panel
    spans +-r_max [mm], by default 5 % beyond the model's edge (params['r_cut'])."""
    import pickle
    from matplotlib.patches import Polygon as MplPolygon
    from PyCentralRegion.electrodes3d import loops_from_mesh, solid_meshes

    with open(model_pkl, 'rb') as fh:
        model = pickle.load(fh)
    z_list = [0.0, 10.0, 16.0, 18.0, 24.0, 40.0] if z_list is None else list(z_list)
    r_max = 1.05 * float(model.params['r_cut']) * MM if r_max is None else float(r_max)
    meshes = solid_meshes(model)
    colour = {'dee0': '#d62728', 'dee1': '#ff7f0e', 'dee2': '#2ca02c', 'dee3': '#9467bd', 'ground': '#1f77b4'}
    cols = min(3, len(z_list))
    rows = int(np.ceil(len(z_list) / cols))
    fig, axes = _subplots(rows, cols, figsize=(4.6 * cols, 4.6 * rows), squeeze=False)
    for ax, z in zip(axes.ravel(), z_list):
        n_loops = 0
        for name, m in meshes.items():
            for lp in loops_from_mesh(m, z=z * 1e-3, name=name):
                ax.add_patch(MplPolygon(np.asarray(lp) * MM, closed=True, facecolor=colour.get(name, '0.6'),
                                        edgecolor='k', lw=0.35, alpha=0.75, zorder=2))
                n_loops += 1
        if orbit is not None and len(orbit):
            o = np.asarray(orbit)
            ax.plot(o[:, 0], o[:, 1], 'k-', lw=0.8, zorder=4, label='design orbit')
        for q in posts or []:
            if abs(z) <= q['half_height']:
                ax.plot(q['centre'][0], q['centre'][1], 'w*', ms=9, mec='k', mew=0.6, zorder=5)
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
    fig.suptitle(f"{Path(model_pkl).name}: metal in the x-y plane, as the obstacle test sees it", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(out) if out else Path(model_pkl).with_name(Path(model_pkl).stem + "_sections.png")
    fig.savefig(out, dpi=150)
    print(f"[sections] {len(z_list)} slices -> {out}")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# runs: tracked paths in gmsh, snapshots from above
def _run_of(data: dict, which: str) -> dict:
    return data.get(which) or data.get('final_sc') or data.get('final') or {}


def load_tracks(data_pkl, which: str = 'final') -> dict:
    """The tracked-particle paths of a run's data pickle (ids, steps, xyz [m], stop_step, birth_step, ...), with the
    loss information turned into a per-path live length: after a particle is lost its row keeps whatever the engine
    left there, which draws as a spike to nowhere if taken at face value."""
    import pickle
    with open(data_pkl, 'rb') as fh:
        data = pickle.load(fh)
    run = _run_of(data, which)
    tr = run.get('tracks')
    if tr is None:
        raise ValueError(f"{data_pkl} has no 'tracks' (track.n_tracks = 0?); the run has: {sorted(run)}")
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
    return {**tr, 'steps': steps, 'xyz': xyz, 'stop_step': stop, 'birth_step': birth, 'first': np.array(first),
            'n_valid': np.array(n_valid), 'lost': stop <= int(tr.get('n_steps', steps[-1] + 1)),
            'hits': np.asarray(run.get('hits', np.zeros((0, 4)))), 'hits_z': np.asarray(run.get('hits_z', np.zeros(0)))}


def tracks_to_pos(tracks: dict, pos_path, colour_by: str = 'turn', n: int = None, seed: int = 0) -> dict:
    """The paths as a gmsh post-processing file (one scalar-line view, in mm), plus every particle that ended on metal
    as a second view. ``colour_by``: 'turn', 'z' or 'lost' (0 alive / 1 lost)."""
    steps, xyz = tracks['steps'], tracks['xyz']
    per_turn = float(tracks.get('steps_per_turn', 1000))
    ids = np.arange(xyz.shape[1])
    if n is not None and n < len(ids):
        ids = np.sort(np.random.default_rng(seed).choice(ids, size=n, replace=False))
    lines = [f'View "trajectories ({colour_by})" {{']
    n_drawn = 0
    for j in ids:
        i0, nv = int(tracks['first'][j]), int(tracks['n_valid'][j])
        if nv < 2:
            continue
        p = xyz[i0:i0 + nv, j, :] * MM
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
            lines.append(f"SL({a[0]:.4f},{a[1]:.4f},{a[2]:.4f},{b[0]:.4f},{b[1]:.4f},{b[2]:.4f}){{{v[k]:.5g},"
                         f"{v[k + 1]:.5g}}};")
        n_drawn += 1
    lines.append('};')
    loss_view, hits = None, tracks['hits']
    if len(hits):                                 # (id, step, x, y) [m], z in hits_z
        loss_view = 1
        lines.append('View "lost to metal" {')
        hz = tracks['hits_z']
        for i, h in enumerate(hits):
            z = float(hz[i]) if i < len(hz) else 0.0
            lines.append(f"SP({h[2] * MM:.4f},{h[3] * MM:.4f},{z * MM:.4f}){{{int(h[1]) / per_turn:.5g}}};")
        lines.append('};')
    Path(pos_path).write_text("\n".join(lines) + "\n")
    return {'n_drawn': n_drawn, 'loss_view': loss_view, 'n_loss': int(len(hits))}


def design_orbit_pos(result, pos_path, r_max: float = None) -> int:
    """The seed's reference trajectory (the orbit every gap line was laid out around) as its own gmsh view [mm]."""
    trj = np.asarray(getattr(result, 'trajectory_reference', np.zeros((0, 3))), dtype=float)
    if len(trj) < 2:
        return 0
    p = trj[:, :3] * MM
    if r_max is not None:
        p = p[np.hypot(p[:, 0], p[:, 1]) <= float(r_max)]
        if len(p) < 2:
            return 0
    turn = np.arange(len(p), dtype=float) / max(len(p) - 1, 1)
    lines = ['View "design orbit" {']
    for k in range(len(p) - 1):
        a, b = p[k], p[k + 1]
        lines.append(f"SL({a[0]:.4f},{a[1]:.4f},{a[2]:.4f},{b[0]:.4f},{b[1]:.4f},{b[2]:.4f}){{{turn[k]:.5g},"
                     f"{turn[k + 1]:.5g}}};")
    lines.append('};')
    Path(pos_path).write_text("\n".join(lines) + "\n")
    return len(p)


def view_tracks(step_path, data_pkl, n: int = 60, colour_by: str = 'turn', which: str = 'final', gui: bool = True,
                seed: int = 0, pos_path=None, result=None, r_max: float = None, png=None, si_steps=()) -> None:
    """Open a model's .step in gmsh with a random selection of a run's tracked paths, the loss points and (``result``:
    the seed) the design orbit, each its own view; ``si_steps``: the spiral inflector's STEP files to merge in (same
    frame and units, so a disagreement in the view is real). ``png`` writes a snapshot instead of the GUI."""
    import gmsh
    tracks = load_tracks(Path(data_pkl), which=which)
    pos = Path(pos_path) if pos_path else Path(data_pkl).with_name(Path(data_pkl).stem + f"_tracks_{colour_by}.pos")
    info = tracks_to_pos(tracks, pos, colour_by=colour_by, n=n, seed=seed)
    print(f"[view] {info['n_drawn']} of {tracks['xyz'].shape[1]} tracked paths -> {pos}")
    print(f"[view] {int(np.count_nonzero(tracks['lost']))} of {len(tracks['lost'])} tracked particles were lost; "
          f"{info['n_loss']} particles ended on metal in the whole run (the red dots)")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.merge(str(step_path))
    gmsh.model.occ.synchronize()
    cr_surfs = {t for _, t in gmsh.model.getEntities(2)}
    for f in si_steps:
        gmsh.merge(str(f))
    if si_steps:
        gmsh.model.occ.synchronize()
        new_s = [t for _, t in gmsh.model.getEntities(2) if t not in cr_surfs]
        for t in new_s:
            gmsh.model.setColor([(2, t)], 30, 160, 90, 110)
        print(f"[view] spiral inflector: {len(si_steps)} STEP file(s), {len(new_s)} surfaces")
    gmsh.merge(str(pos))
    gmsh.option.setNumber("Geometry.Surfaces", 1)
    gmsh.option.setNumber("Geometry.Points", 0)
    gmsh.option.setNumber("Geometry.Curves", 1)
    gmsh.option.setNumber("Geometry.Volumes", 0)
    gmsh.option.setColor("Geometry.Surfaces", 160, 160, 170, 40)       # a translucent shell: the paths stay visible
    gmsh.option.setColor("Geometry.Curves", 90, 90, 100, 120)
    gmsh.option.setNumber("General.Trackball", 1)
    gmsh.option.setNumber("General.RotationCenterGravity", 1)
    orbit_view = None
    if result is not None:
        opos = pos.with_name(pos.stem + "_orbit.pos")
        n_orb = design_orbit_pos(result, opos, r_max=r_max)
        if n_orb:
            orbit_view = int(gmsh.option.getNumber("PostProcessing.NbViews"))
            gmsh.merge(str(opos))
            print(f"[view] design orbit: {n_orb:,} points -> {opos}")
    for v in range(int(gmsh.option.getNumber("PostProcessing.NbViews"))):
        gmsh.option.setNumber(f"View[{v}].LineWidth", 2.0)
        gmsh.option.setNumber(f"View[{v}].ShowScale", 1)
    if orbit_view is not None:
        gmsh.option.setNumber(f"View[{orbit_view}].LineWidth", 4.0)
        gmsh.option.setNumber(f"View[{orbit_view}].ShowScale", 0)
        oopt = pos.with_name(pos.stem + "_orbit.opt")
        oopt.write_text(f"View[{orbit_view}].ColorTable = {{Black}};\n")
        gmsh.merge(str(oopt))
    lv = info['loss_view']
    if lv is not None:
        # a one-entry colour table is what pins a scalar view to one colour (View[].Color.Points is ignored)
        gmsh.option.setNumber(f"View[{lv}].PointSize", 4.0)
        gmsh.option.setNumber(f"View[{lv}].PointType", 0)
        gmsh.option.setNumber(f"View[{lv}].ShowScale", 0)
        opt = pos.with_suffix('.opt')
        opt.write_text(f"View[{lv}].ColorTable = {{Red}};\n")
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


def beam_frames(data_pkl, out=None, which: str = 'final', n_max: int = 20000, colour: str = 'E') -> Path:
    """The run's snapshots from above, one panel each: the x-y plane turned so the bunch's MEAN momentum points along
    +y' and centred on the centroid, so y' is the longitudinal direction and x' the radial one."""
    import pickle
    with open(data_pkl, 'rb') as fh:
        data = pickle.load(fh)
    snaps = _run_of(data, which).get('snapshots') or []
    if not snaps:
        raise ValueError(f"{data_pkl} has no snapshots for {which!r}")
    cols = min(4, len(snaps))
    rows = int(np.ceil(len(snaps) / cols))
    fig, axes = _subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    rng = np.random.default_rng(0)
    lims, frames = [], []
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
        ey = vv[:, :2].mean(axis=0)
        ey = ey / np.linalg.norm(ey)
        ex = np.array([ey[1], -ey[0]])
        q = p[:, :2] - p[:, :2].mean(axis=0)
        xp, yp = q.dot(ex) * MM, q.dot(ey) * MM
        cval = np.asarray(s['E_mev'])[keep] if colour == 'E' else p[:, 2] * MM
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
        ax.set_title(f"t = {s['t_ns']:.0f} ns   N = {nk:,}\nref r = {np.hypot(*np.asarray(s['ref_xyz'])[:2]) * MM:.0f} "
                     f"mm, heading {ang:+.0f} deg", fontsize=8)
        ax.set_xlabel("x' radial [mm]", fontsize=8)
        ax.set_ylabel("y' along motion [mm]", fontsize=8)
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("E [MeV]" if colour == 'E' else "z [mm]", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(frames):]:
        ax.set_axis_off()
    fig.suptitle("central region: beam in the midplane, turned to put mean momentum along +y'", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(out) if out else Path(data_pkl).with_name(Path(data_pkl).stem + "_rz_frames.png")
    fig.savefig(out, dpi=160)
    print(f"[frames] {len([f for f in frames if f])} snapshots -> {out}")
    return out
