"""`pcr build`: the central region's 3D electrode model from a run config -> paths.model3d.

``build(cfg)`` runs the electrode builder in the design deck's order (``builder`` has the construction, ``views`` the
reports it prints on the way):

  1. inputs: the seed (paths.seed), the housing outline (paths.housing_outline), the hand-off file whose particle cloud
     is the emitting plane (geometry.emitting_plane_file; else the export's own hand-off beside the outline's folder;
     else paths.handoff) and, for si_rotation = 'corner' or gap0_on_plate, the exit plane spec
     (geometry.exit_plane_spec_file; else exit_plane_spec.json of the export folder above the outline's);
  2. the gap lines: matched to the seed's reference orbit (geometry.snap_nodes), junctions moved
     (geometry.junctions), the spiral-inflector rotation (geometry.si_rotation) and gap 0 on the turned exit plate
     (geometry.gap0_on_plate);
  3. the housing outline turned with the inflector, its exit aperture inset (aperture_inset_mm), the channel heights
     held to their bounds (``options.check_bounds``);
  4. the 2D dees and dummy dees, the seed's reference orbit against them, <prefix>_dees.png (geometry.plot_dees);
  5. the emitting plane (turned) and the beam channel, the hardware, the OCC model, the surface mesh;
  6. the ElectrodeModel -> paths.model3d, recording the inflector rotation the electrodes were built around
     (params['si_rotation_deg']: the tracker refuses an inflector.rotation_deg that differs from it), with
     <prefix>.msh (the surface mesh; physical groups dee0..dee3, ground), <prefix>.step (the CAD: the dees and the
     ground as unnamed OCC solids, in mm, machine frame - z = 0 the median plane, the housing and its exit turned by
     the inflector rotation; the ground holds the spiral-inflector housing, its outline extruded over
     +-half_height_mm, with the exit opening cut through it) and <prefix>_build.json (what the build used).

<prefix> is paths.model3d without '_model3d.pkl' (or without '.pkl'). Every output is written under its temporary
name <stem>.part<suffix> (``config.part_path``) and renamed once all of them exist, the model last
(``config.commit_part``): an existing paths.model3d is reused and nothing is rebuilt or overwritten, a build that
finds some of its other outputs but no model refuses to run, and a build that fails (a ConfigError from a bound that
needs the layout, such as a window's) removes its temporary files. No BEM solve here (``pcr fields`` does that).

Tools that work on the same edges the build lays out (each from a run config):
  ``layout(cfg)``      the 2D electrodes exactly as the build has them, with every report;
  ``bounds(cfg)``      the channel-height bounds, for a campaign's variable ranges;
  ``gap0_hole(cfg)``   a round hole on dee0's gap-0 edge around where the hand-off bunch meets it, as a
                       [[geometry.holes]] item (part 2 of the design deck's hand-off aperture tool);
  ``cut_handoff(...)`` a hand-off file cut to a circle on its own plane (that tool's part 1);
  ``edge_shift(cfg, moves)``  a radius change of a point on a gap edge as an arclength (the deck's edge-shift tool).

    python -m PyCentralRegion.geometry.build build run.toml [--set k=v ...]
    python -m PyCentralRegion.geometry.build bounds run.toml
    python -m PyCentralRegion.geometry.build hole run.toml --r 10 [--rim 8] [--dr 0] [--handoff FILE]
    python -m PyCentralRegion.geometry.build shift run.toml 1:dee:39.29:-2 1:dummy:72.43:-2
    python -m PyCentralRegion.geometry.build cut SRC.h5 DST.h5 --r 10 [--centre centroid]
    python -m PyCentralRegion.geometry.build sections MODEL.pkl [--z 0,10,16,18,24,40]
"""
import contextlib
import dataclasses
import io
import json
import pickle
import time
from pathlib import Path

import numpy as np

from PyCentralRegion import config as C
from PyCentralRegion.config import ConfigError
from . import builder as B
from . import views as V
from .options import HoleSpec, channel_height_max_mm, check_bounds

EXPORT_HANDOFFS = ('handoff_sc_all_level.h5', 'handoff_vac_all_level.h5')
EXIT_SPEC_NAME = 'exit_plane_spec.json'
SI_ROTATION_TOL_DEG = 1e-6          # what simulate3d.check_si_rotation allows between the model and the run
MM = B.MM


# ---------------------------------------------------------------------------------------------------------------------
# inputs and outputs
def _sections(cfg) -> dict:
    return {name: C.section(cfg, name) for name in ('geometry', 'machine', 'inflector', 'paths')}


def _existing(path, what: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"{what} = {p} does not exist")
    return p


def _export_dirs(outline) -> tuple:
    """Where the spiral-inflector export keeps its files: the outline lives in <export>/steps_baseline_mm/."""
    p = Path(outline)
    return p.parent.parent, p.parent


def inputs(cfg, plane: bool = True) -> dict:
    """The files a build reads, resolved and checked: {'seed', 'housing_outline', 'emitting_plane',
    'emitting_plane_from', 'exit_plane_spec'} (Paths; the spec None when not needed and not found). ``plane`` False
    skips the emitting plane (the 2D tools do not need it)."""
    s = _sections(cfg)
    g, pc = s['geometry'], s['paths']
    out = {'seed': _existing(C.require(pc.seed, 'paths.seed'), 'paths.seed'),
           'housing_outline': _existing(C.require(pc.housing_outline, 'paths.housing_outline'),
                                        'paths.housing_outline'),
           'emitting_plane': None, 'emitting_plane_from': '', 'exit_plane_spec': None}
    dirs = _export_dirs(out['housing_outline'])
    if plane:
        if g.emitting_plane_file:
            out['emitting_plane'] = _existing(g.emitting_plane_file, 'geometry.emitting_plane_file')
            out['emitting_plane_from'] = 'geometry.emitting_plane_file'
        else:
            found = [d / n for d in dirs for n in EXPORT_HANDOFFS if (d / n).exists()]
            if found:
                out['emitting_plane'], out['emitting_plane_from'] = found[0], "the export's own, beside the outline"
            elif pc.handoff:
                out['emitting_plane'] = _existing(pc.handoff, 'paths.handoff')
                out['emitting_plane_from'] = 'paths.handoff'
            else:
                raise ConfigError(f"no emitting plane for the beam channel: geometry.emitting_plane_file is not set, "
                                  f"none of {', '.join(EXPORT_HANDOFFS)} is in {dirs[0]} or {dirs[1]}, and "
                                  f"paths.handoff is not set")
    needed = g.si_rotation == 'corner' or g.gap0_on_plate
    if g.exit_plane_spec_file:
        out['exit_plane_spec'] = _existing(g.exit_plane_spec_file, 'geometry.exit_plane_spec_file')
    else:
        found = [d / EXIT_SPEC_NAME for d in dirs if (d / EXIT_SPEC_NAME).exists()]
        if found:
            out['exit_plane_spec'] = found[0]
        elif needed:
            raise ConfigError(f"geometry.si_rotation = {g.si_rotation!r} / gap0_on_plate = {g.gap0_on_plate} read the "
                              f"exit plane spec, but geometry.exit_plane_spec_file is not set and there is no "
                              f"{EXIT_SPEC_NAME} in {dirs[0]} or {dirs[1]}")
    return out


def output_files(model3d) -> dict:
    """The files a build writes, in the order they are committed (the model last): {'dees_png', 'msh', 'step',
    'build_json', 'model3d'}; <prefix> is the model's name without '_model3d.pkl' (or '.pkl')."""
    p = Path(model3d)
    stem = p.name[:-len('_model3d.pkl')] if p.name.endswith('_model3d.pkl') else p.stem
    return {'dees_png': p.with_name(stem + '_dees.png'), 'msh': p.with_name(stem + '.msh'),
            'step': p.with_name(stem + '.step'), 'build_json': p.with_name(stem + '_build.json'), 'model3d': p}


# ---------------------------------------------------------------------------------------------------------------------
# the 2D layout the build and the tools share
def _channel_report(g, bounds: dict, exit_half: float) -> None:
    r0, r1 = g.aperture_radii()
    d_in, d_out = g.dummy_aperture_heights()
    same = (d_in, d_out) == g.aperture_heights()
    V._banner("BEAM CHANNEL - full heights and their bounds")
    print(f"  dees: {g.aperture_inner_mm:g} mm inside r = {r0:g} mm, {g.aperture_outer_mm:g} mm beyond r = {r1:g} mm "
          f"(taper centre {g.aperture_taper_center_mm:g}, length {g.aperture_taper_length_mm:g} mm)")
    print("  dummy dees and ground: " + ("the dees' heights" if same else f"{d_in:g} / {d_out:g} mm (an asymmetric gap)"))
    print(f"  bounds: every height <= 2 x (plate_half_gap_mm {g.plate_half_gap_mm:g} - dee_wall_mm {g.dee_wall_mm:g}) "
          f"= {bounds['height_max_mm']:g} mm; the inner heights and the heights at the housing exit's outermost "
          f"radius {bounds['exit_r_mm']:.2f} mm >= 2 x its half height {exit_half:.3f} = {bounds['inner_min_mm']:.3f} mm"
          f"  -> within bounds")


def layout(cfg, verbose: bool = True, inp: dict = None, dees_png=None) -> dict:
    """Configure the builder from cfg and lay the electrodes out in 2D exactly as ``build`` does, printing its reports
    (``verbose``). The tools that place hardware call this, so they see the build's edges: the same node snap,
    junction moves, inflector rotation, dee0 tip and inner closures.

    Returns {'sections', 'inputs', 'dees', 'geom', 'result', 'trajectory' (the seed's reference orbit, m),
    'si_rotation_deg', 'outline' (turned, inset), 'sim' (the outline the housing is built from), 'exit_half_height_mm',
    'bounds', 'foot' (the dees), 'dummy' (the dummy dees), 'orbit' (mm, x-y, cut at r_max)}."""
    if not verbose:
        with contextlib.redirect_stdout(io.StringIO()):
            return layout(cfg, True, inp=inp, dees_png=dees_png)
    s = _sections(cfg)
    g = s['geometry']
    inp = inp or inputs(cfg, plane=False)
    B.configure(g, s['machine'], s['inflector'], exit_plane_spec_file=str(inp['exit_plane_spec'] or ''))
    dees, geom, result = B.load_gap_lines(inp['seed'], note_opening=g.gap_lines != 'connected')
    trj = getattr(result, 'trajectory_reference', None)
    trj = np.zeros((0, 3)) if trj is None else np.asarray(trj, dtype=float)
    if g.gap_lines == 'connected':
        dees = B.connected_dee_system()
        si_rot = B.si_rotation_deg(dees)
        V._banner(f"GAP LINES CONNECTED (geometry.gap_lines): P0 -> P1 -> P2 -> the ray; dee opening "
                  f"{B.opening_deg(dees):.4f} deg, the spiral inflector turned {si_rot:+.4f} deg")
        print("  (the seed gives only its reference orbit: its gap lines and opening are not used)")
        lines = B.lay_connected_gap_lines(dees, verbose=True)
        V.print_gap_lines(dees, result)
        return _layout_rest(cfg, s, g, inp, dees, geom, result, trj, si_rot, dees_png, lines)
    if g.snap_nodes and len(trj) > 2:
        V._banner("SEGMENT NODES: DROP THE INERT ONES, THEN SNAP MIDWAY BETWEEN TURNS")
        B.match_gaps_to_orbit(dees, trj, verbose=True)
    elif g.snap_nodes:
        print("\n  (the seed has no reference orbit: segment node radii left where the optimizer put them)")
    else:
        print("\n  (geometry.snap_nodes = false: segment node radii left where the optimizer put them)")
    if g.junctions:
        V._banner("JUNCTIONS: MOVED BETWEEN THE 3D BEAM'S PASSES (geometry.junctions)")
        B.move_junctions(dees, g.junctions, verbose=True)
    si_rot = B.si_rotation_deg(dees)
    if g.gap0_on_plate:
        V._banner("SI ROTATED ABOUT THE MACHINE AXIS - outline, exit aperture, hand-off plane; gap 0 re-fitted")
        B.refit_gap0_on_plate(dees, verbose=True)
    elif si_rot:
        V._banner(f"SI ROTATED ABOUT THE MACHINE AXIS by {si_rot:+.6f} deg - outline, exit aperture, hand-off plane")
    V.print_gap_lines(dees, result)
    return _layout_rest(cfg, s, g, inp, dees, geom, result, trj, si_rot, dees_png, None)


def _layout_rest(cfg, s, g, inp, dees, geom, result, trj, si_rot, dees_png, lines) -> dict:
    """``layout`` from the gap lines on: the housing, the channel bounds, the dees and the dummy dees."""
    ol = B.rotate_outline(B.read_housing_outline(inp['housing_outline']), si_rot)
    V.print_housing_outline(ol, inp['housing_outline'])
    if any(float(v) != 0.0 for v in g.aperture_inset_mm):
        ol['aperture'] = B.inset_aperture(ol['aperture'], *map(float, g.aperture_inset_mm))
    sim = B.housing_outline(ol)
    exit_half = B.exit_half_height_mm(ol)
    bounds = check_bounds(g, exit_half, B.exit_radius_mm(ol))
    _channel_report(g, bounds, exit_half)

    V.print_junctions(dees)
    foot = B.dee_boundaries(dees, sim['curve'])
    V.print_dee_boundaries(foot)
    orbit = V.reference_orbit(result, r_max=B.r_max_mm())
    V.print_orbit(foot, orbit, result)
    if dees_png:
        V.plot_dee_boundaries(foot, ol, out=str(dees_png), orbit=orbit)
    dummy = B.dummy_boundaries(dees, sim['curve'])
    V.print_dummies(dummy)
    return {'sections': s, 'inputs': inp, 'dees': dees, 'geom': geom, 'result': result, 'trajectory': trj,
            'si_rotation_deg': float(si_rot), 'outline': ol, 'sim': sim, 'exit_half_height_mm': exit_half,
            'bounds': bounds, 'foot': foot, 'dummy': dummy, 'orbit': orbit, 'gap_lines': lines}


def bounds(cfg) -> dict:
    """The channel-height bounds of this config's build [mm] without building: {'height_max_mm' (every height),
    'inner_min_mm' (aperture_inner_mm and dummy_aperture_inner_mm, and the heights at exit_r_mm), 'exit_r_mm' (the
    housing exit's outermost radius), 'exit_half_height_mm', 'plate_half_gap_mm', 'dee_wall_mm'}: what a campaign sets
    its geometry.aperture_*_mm ranges from. Reads the housing outline only and does not check the current heights
    (``build`` does)."""
    g = C.section(cfg, 'geometry')
    pc = C.section(cfg, 'paths')
    ol = B.read_housing_outline(_existing(C.require(pc.housing_outline, 'paths.housing_outline'),
                                          'paths.housing_outline'))
    if any(float(v) != 0.0 for v in g.aperture_inset_mm):
        with contextlib.redirect_stdout(io.StringIO()):
            ol['aperture'] = B.inset_aperture(ol['aperture'], *map(float, g.aperture_inset_mm))
    e = B.exit_half_height_mm(ol)
    return {'height_max_mm': channel_height_max_mm(g), 'inner_min_mm': 2.0 * e, 'exit_r_mm': B.exit_radius_mm(ol),
            'exit_half_height_mm': e, 'plate_half_gap_mm': float(g.plate_half_gap_mm),
            'dee_wall_mm': float(g.dee_wall_mm)}


# ---------------------------------------------------------------------------------------------------------------------
# the build
def build(cfg) -> dict:
    """Build the 3D electrode model of cfg into paths.model3d (see the module docstring). Returns the model's files
    {'dees_png', 'msh', 'step', 'build_json', 'model3d'} (those that exist) as strings; when paths.model3d exists it is
    reused - nothing is built or written, and that is said."""
    s = _sections(cfg)
    g, inf = s['geometry'], s['inflector']
    out = output_files(C.require(s['paths'].model3d, 'paths.model3d'))
    if out['model3d'].exists():
        print(f"[build] {out['model3d']} exists: reused, nothing rebuilt (point paths.model3d elsewhere to rebuild)",
              flush=True)
        return {k: str(v) for k, v in out.items() if v.exists()}
    stale = [str(v) for k, v in out.items() if k != 'model3d' and v.exists()]
    if stale:
        raise ConfigError(f"paths.model3d = {out['model3d']} does not exist but other outputs of a build there do "
                          f"({', '.join(stale)}); they are never overwritten: remove them or point paths.model3d "
                          f"elsewhere")
    t0 = time.time()
    inp = inputs(cfg)                                         # every input checked before anything is built
    out['model3d'].parent.mkdir(parents=True, exist_ok=True)
    part = {k: C.part_path(v) for k, v in out.items()}
    for p in part.values():                                   # leftovers of a build that died: temporary, ours
        p.unlink(missing_ok=True)
    print(f"[build] {out['model3d']}\n[build] seed {inp['seed']}\n[build] housing outline {inp['housing_outline']}",
          flush=True)
    try:
        si_rot = _build_parts(cfg, s, inp, out['model3d'], part, t0)
    except BaseException:
        for p in part.values():           # a build that fails (a window out of its bound, ...) leaves no temporary
            try:                          # file behind; the error it raised is the one that propagates
                p.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    written = {}
    for k, final in out.items():                               # commit: the model last, nothing overwritten
        if part[k].exists():
            written[k] = str(C.commit_part(final, overwrite=False))
    print(f"\n[build] wrote " + ", ".join(Path(v).name for v in written.values()) + f" in {out['model3d'].parent} "
          f"({time.time() - t0:.0f} s)")
    if abs(si_rot - float(inf.rotation_deg)) > SI_ROTATION_TOL_DEG:
        if g.si_rotation == 'corner':
            print(f"[build] the electrodes were built around the inflector turned by {si_rot:+.12g} deg (si_rotation "
                  f"= 'corner'; inflector.rotation_deg = {inf.rotation_deg:+.12g} is not used): the tracker turns it "
                  f"by the model's angle too")
        else:
            print(f"[build] WARNING: the electrodes were built around the inflector turned by {si_rot:+.12g} deg "
                  f"(si_rotation = {g.si_rotation!r}) but inflector.rotation_deg = {inf.rotation_deg:+.12g}: the "
                  f"tracker refuses this model until the run sets inflector.rotation_deg = {si_rot!r}")
    return written


def _build_parts(cfg, s: dict, inp: dict, model3d: Path, part: dict, t0: float) -> float:
    """The build proper: every output of ``build`` under its temporary name ``part[k]``. Returns the inflector
    rotation [deg] the electrodes were built around."""
    g, inf = s['geometry'], s['inflector']
    lay = layout(cfg, verbose=True, inp=inp, dees_png=part['dees_png'] if g.plot_dees else None)
    dees, ol, si_rot = lay['dees'], lay['outline'], lay['si_rotation_deg']
    hole = None
    if g.gap0_hole_r_mm > 0.0:                                # placed for this build's edges and inflector turn
        V._banner("GAP-0 HOLE BY RULE (geometry.gap0_hole_r_mm): where the hand-off bunch meets dee0's gap-0 edge")
        hole = gap0_hole(cfg, r_mm=g.gap0_hole_r_mm, rim_mm=g.gap0_hole_rim_mm, dr_mm=g.gap0_hole_dr_mm,
                         lay=lay)['hole']
        B.OPT = dataclasses.replace(B.OPT, holes=tuple(B.OPT.holes) + (hole,), gap0_hole_r_mm=0.0)
    plane = B.rotate_plane(B.emitting_plane(inp['emitting_plane']), si_rot)
    brick = B.beam_channel(ol['aperture'], plane)
    print(f"\n[channel] emitting plane from {inp['emitting_plane']} ({inp['emitting_plane_from']}): "
          f"{plane['n_particles']:,} particles within {plane['flatness']:.2e} mm of it, point "
          f"({plane['point'][0]:+.3f}, {plane['point'][1]:+.3f}, {plane['point'][2]:+.3f}) mm, normal "
          f"({plane['normal'][0]:+.4f}, {plane['normal'][1]:+.4f}, {plane['normal'][2]:+.4f})")
    print(f"[channel] the {len(ol['aperture'])}-corner exit aperture projected on it, swept "
          f"{-brick['back']:+g} .. {brick['extend']:+g} mm along the normal")
    if g.post_shape != 'square':
        print(f"  posts: {g.post_shape} ({g.post_facets} facets), width {g.post_size_mm:g} mm")
    posts = B.hardware(lay['foot'], lay['dummy'])
    if posts:
        V.print_posts(posts)

    import gmsh
    model = None
    try:
        gm, groups, r_tool, _ = B.build_model(lay['foot'], lay['dummy'], lay['sim'], brick, dees, posts=posts or None)
        r0, r1 = g.aperture_radii()
        print()
        print(f"  beam channel {g.aperture_inner_mm:g} mm to r = {r0:g} mm, {g.aperture_outer_mm:g} mm from "
              f"r = {r1:g} mm" + ("" if g.dummy_aperture_heights() == g.aperture_heights() else
                                  " (dummy dees {:g} / {:g} mm)".format(*g.dummy_aperture_heights())))
        print("  aperture tool inner radii [mm]: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(r_tool.items())))
        B.mesh_and_report(gm, groups, prefix=str(part['msh'].with_suffix('')))
        model = B.to_electrode_model(gm, groups, B.dee_potential(dees), si_rotation_deg=si_rot)
    finally:
        if getattr(gmsh, 'isInitialized', lambda: True)():
            gmsh.finalize()
    with open(part['model3d'], 'wb') as fh:
        pickle.dump(model, fh, protocol=4)

    solids = {sd['name']: int(sd['tri_range'][1] - sd['tri_range'][0]) for sd in model.params['solids']}
    record = {'model3d': str(model3d), 'built': time.strftime('%Y-%m-%d %H:%M:%S'),
              'wall_s': round(time.time() - t0, 1), 'config': cfg.get('_source', ''),
              'inputs': {k: str(v) for k, v in inp.items() if v},
              'si_rotation': g.si_rotation, 'si_rotation_deg': si_rot, 'inflector_rotation_deg': inf.rotation_deg,
              'dee_potential_v': float(B.dee_potential(dees)), 'exit_half_height_mm': lay['exit_half_height_mm'],
              'bounds': lay['bounds'], 'hardware': [q['name'] for q in posts],
              'gap0_hole': dataclasses.asdict(hole) if hole else None,
              'gap_lines': _lines_record(lay['gap_lines']),
              'mesh': {'nodes': int(len(model.vertices)), 'triangles': int(len(model.triangles)), 'solids': solids},
              'gmsh': getattr(gmsh, '__version__', ''),
              'geometry': C.to_dict(g), 'machine': C.to_dict(s['machine']), 'inflector': C.to_dict(inf)}
    part['build_json'].write_text(json.dumps(record, indent=1, default=C.json_default))
    return si_rot


def _lines_record(lines) -> list:
    """The connected gap lines as built, for <prefix>_build.json (None for the seed's lines): per gap its ray, P0, P1,
    P2 in the gap's own frame [mm], P0's and P1's azimuths off the ray and the bends [deg]."""
    if lines is None:
        return None
    keep = ('gap', 'ray_deg', 'P0', 'P1', 'P2', 'd0_deg', 'd1_deg', 'bends_deg', 'y1_max', 'y0_range',
            'corner_bend_deg')
    return [{k: (list(map(float, ln[k])) if isinstance(ln[k], tuple) else ln[k]) for k in keep if k in ln}
            for ln in lines]


# ---------------------------------------------------------------------------------------------------------------------
# tools on the gap edges
def _edge_grid(edge: np.ndarray, n: int):
    """An edge polyline sampled at n arclengths: (s, points (n, 2), r)."""
    seg = np.diff(edge, axis=0)
    L = np.linalg.norm(seg, axis=1)
    sl = np.concatenate([[0.0], np.cumsum(L)])
    s = np.linspace(0.0, sl[-1], n)
    k = np.clip(np.searchsorted(sl, s) - 1, 0, len(seg) - 1)
    pts = edge[k] + (s - sl[k])[:, None] * (seg[k] / L[k, None])
    return s, pts, np.hypot(pts[:, 0], pts[:, 1])


def _shift_on_edge(edge: np.ndarray, s0: float, dr: float, what: str, n: int = 40001) -> dict:
    """The point on an edge whose radius is r(s0) + dr, the crossing nearest s0; the edge has to be monotonic in radius
    only between the two points (the hug dee ends and blunt dummy tips curve the edges near their inner ends)."""
    s_grid, pts, r_grid = _edge_grid(np.asarray(edge, dtype=float), n)
    if not 0.0 <= s0 <= s_grid[-1]:
        raise ConfigError(f"{what}: arclength {s0:g} mm is off the edge (0 .. {s_grid[-1]:.2f} mm)")
    r0 = float(np.interp(s0, s_grid, r_grid))
    f = r_grid - (r0 + dr)
    ks = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) <= 0)[0]
    if not len(ks):
        raise ConfigError(f"{what}: the edge never reaches r = {r0 + dr:.3f} mm (r {r_grid.min():.2f} .. "
                          f"{r_grid.max():.2f} mm)")
    sx = np.array([s_grid[k] + (s_grid[k + 1] - s_grid[k]) * f[k] / (f[k] - f[k + 1]) if f[k] != f[k + 1]
                   else s_grid[k] for k in ks])
    s1 = float(sx[np.argmin(np.abs(sx - s0))])
    lo, hi = sorted((s0, s1))
    rr = r_grid[(s_grid >= lo) & (s_grid <= hi)]
    if len(rr) > 2 and not (np.all(np.diff(rr) > 0) or np.all(np.diff(rr) < 0)):
        raise ConfigError(f"{what}: the edge radius is not monotonic between s {lo:.3f} and {hi:.3f} mm")
    p0 = np.array([np.interp(s0, s_grid, pts[:, 0]), np.interp(s0, s_grid, pts[:, 1])])
    p1 = np.array([np.interp(s1, s_grid, pts[:, 0]), np.interp(s1, s_grid, pts[:, 1])])
    return {'s0': float(s0), 's1': s1, 'r0': r0, 'r1': r0 + dr, 'p0': p0, 'p1': p1,
            'monotonic': bool(np.all(np.diff(r_grid) > 0)), 'r_range': (float(r_grid.min()), float(r_grid.max()))}


def edge_shift(cfg, moves, lay: dict = None, verbose: bool = True) -> list:
    """Points on gap edges moved by a change in RADIUS, as new arclengths: hardware is placed by arclength, moves are
    asked for in radius ("the gap-1 inner posts 2 mm radially inward"). ``moves``: 'gap:side:s:dr' strings or
    (gap, side, s_mm, dr_mm) tuples; the edges are the build's (``layout``). Returns one dict per move (gap, side,
    s0, s1, r0, r1, points) and prints the new 'gap:side:s' list."""
    lay = lay or layout(cfg, verbose=False)
    out = []
    for mv in moves:
        if isinstance(mv, str):
            f = [v.strip() for v in mv.split(':')]
            if len(f) != 4:
                raise ConfigError(f"edge shift {mv!r}: wants 'gap:side:s:dr'")
            mv = (int(f[0]), f[1], float(f[2]), float(f[3]))
        g, side, s0, dr = int(mv[0]), str(mv[1]), float(mv[2]), float(mv[3])
        edge, _, owner = B.gap_edge(lay['foot'], lay['dummy'], g, side)
        r = _shift_on_edge(edge, s0, dr, f"gap {g} {side}")
        r.update(gap=g, side=side, owner=owner)
        out.append(r)
        if verbose:
            if not r['monotonic']:
                print(f"  note: gap {g} {side} edge radius is not monotonic over the whole edge (r "
                      f"{r['r_range'][0]:.2f}..{r['r_range'][1]:.2f} mm); used the crossing nearest s {s0:.3f}")
            print(f"gap {g} {side:5s} ({owner}): s {s0:.4f} -> {r['s1']:.4f} mm (ds {r['s1'] - s0:+.3f}), r "
                  f"{r['r0']:.3f} -> {r['r1']:.3f} mm; edge point ({r['p0'][0]:+.2f}, {r['p0'][1]:+.2f}) -> "
                  f"({r['p1'][0]:+.2f}, {r['p1'][1]:+.2f})")
    if verbose:
        print("NEW=" + ",".join(f"{r['gap']}:{r['side']}:{r['s1']:.4f}" for r in out))
    return out


def _plane_uv(h: dict):
    """Every particle's (u, v) [mm] on its hand-off plane (the export's plane origin and axes)."""
    meta = h['meta']
    missing = [k for k in ('plane_origin_m', 'plane_axis_u', 'plane_axis_v') if k not in meta]
    if missing:
        raise ConfigError(f"{h['path']}: no {', '.join('PyPATools:' + k for k in missing)} attribute(s) - not a "
                          f"plane-crossing hand-off file")
    o = np.asarray(meta['plane_origin_m'], dtype=float) * MM
    eu, ev = np.asarray(meta['plane_axis_u'], dtype=float), np.asarray(meta['plane_axis_v'], dtype=float)
    x = np.asarray(h['r'], dtype=float) * MM - o
    return x @ eu, x @ ev


def _circle_on_plane(h: dict, r_mm: float, centre: str = 'centroid'):
    """(mask of the particles within r_mm of the centre on their hand-off plane, (u, v) of the centre [mm]); centre
    'centroid' (the bunch's mean) or 'origin' (the plane origin, the design exit point)."""
    if centre not in ('centroid', 'origin'):
        raise ConfigError(f"the circle's centre is 'centroid' or 'origin', got {centre!r}")
    u, v = _plane_uv(h)
    cu, cv = (float(u.mean()), float(v.mean())) if centre == 'centroid' else (0.0, 0.0)
    return np.hypot(u - cu, v - cv) <= float(r_mm), (cu, cv)


def cut_handoff(src, dst, r_mm: float, centre: str = 'centroid', verbose: bool = True) -> dict:
    """A copy of the hand-off file ``src`` holding only the particles within r_mm of the centre on its own plane:
    every per-particle record sliced, the charge / current / count attributes scaled, the cut recorded in
    'PyCentralRegion:cut*' attributes of the species group. ``dst`` is never overwritten. Returns {'file', 'n_before',
    'n_kept', 'kept_fraction', 'centre_uv_mm'}."""
    import h5py
    from PyCentralRegion.handoff import load_handoff
    dst = Path(dst)
    if dst.exists():
        raise ConfigError(f"{dst} exists: not overwritten")
    h = load_handoff(str(src))
    mask, (cu, cv) = _circle_on_plane(h, r_mm, centre)
    n0, n1 = len(mask), int(mask.sum())
    if n1 == 0:
        raise ConfigError(f"a circle of R {r_mm:g} mm around the {centre} keeps no particle of {src}")
    ratio = n1 / n0

    def fix_attrs(obj_src, obj_dst):
        for k, val in obj_src.attrs.items():
            if k == 'shape' and np.size(val) == 1 and int(np.ravel(val)[0]) == n0:
                val = np.array([n1], dtype=np.asarray(val).dtype)
            elif k in ('numParticles', 'PyPATools:n_particles'):
                val = np.asarray(n1, dtype=val.dtype) if isinstance(val, np.ndarray) else type(val)(n1)
            elif k in ('totalCharge', 'PyPATools:bunch_charge_c', 'PyPATools:transmitted_current_ma',
                       'PyPATools:transmission_through_housing'):
                val = val * ratio
            obj_dst.attrs[k] = val

    def walk(gs, gd):
        fix_attrs(gs, gd)
        for name, item in gs.items():
            if isinstance(item, h5py.Group):
                walk(item, gd.create_group(name))
            else:
                data = item[()]
                if np.ndim(data) >= 1 and np.shape(data)[0] == n0:
                    data = data[mask]
                fix_attrs(item, gd.create_dataset(name, data=data))

    def write(tmp):
        with h5py.File(str(src), 'r') as fs, h5py.File(str(tmp), 'w') as fd:
            walk(fs, fd)
            pp = fd.attrs.get('particlesPath', 'particles')
            grp = fd[pp.decode() if isinstance(pp, bytes) else str(pp)][h['meta']['species_group']]
            grp.attrs['PyCentralRegion:cut'] = (f"circle R {r_mm:g} mm on the hand-off plane around the {centre} "
                                                f"(u {cu:+.4f}, v {cv:+.4f} mm); kept {n1} of {n0}")
            grp.attrs['PyCentralRegion:cut_r_mm'] = float(r_mm)
            grp.attrs['PyCentralRegion:cut_centre_uv_mm'] = np.array([cu, cv])
            grp.attrs['PyCentralRegion:n_before_cut'] = int(n0)
            grp.attrs['PyCentralRegion:cut_source'] = str(src)
    C.write_atomic(dst, write, overwrite=False)
    if verbose:
        print(f"[cut] {src}: {n0} particles; a circle R {r_mm:g} mm around the {centre} (u {cu:+.3f}, v {cv:+.3f} mm) "
              f"keeps {n1} = {100.0 * ratio:.2f} % -> {dst}")
    return {'file': str(dst), 'n_before': n0, 'n_kept': n1, 'kept_fraction': ratio, 'centre_uv_mm': (cu, cv)}


def _fly_to_edge(edge: np.ndarray, X: np.ndarray, P: np.ndarray):
    """Straight flight along each particle's own momentum (x-y direction) to an edge polyline: the arclength and height
    [mm] of its first crossing (NaN when it never meets it)."""
    seg = np.diff(edge, axis=0)
    L = np.linalg.norm(seg, axis=1)
    sl = np.concatenate([[0.0], np.cumsum(L)])
    pxy = np.linalg.norm(P[:, :2], axis=1)
    D = P[:, :2] / pxy[:, None]
    best = np.full(len(X), np.inf)
    s_hit = np.full(len(X), np.nan)
    for m in range(len(seg)):
        sv, c = seg[m], edge[m]
        den = D[:, 0] * sv[1] - D[:, 1] * sv[0]
        ok = np.abs(den) > 1e-12
        dd = c - X[:, :2]
        safe = np.where(ok, den, 1.0)
        t = np.where(ok, (dd[:, 0] * sv[1] - dd[:, 1] * sv[0]) / safe, np.nan)
        uu = np.where(ok, (dd[:, 0] * D[:, 1] - dd[:, 1] * D[:, 0]) / safe, np.nan)
        hit = ok & (t >= 0.0) & (uu >= 0.0) & (uu <= 1.0) & (t < best)
        best[hit] = t[hit]
        s_hit[hit] = sl[m] + uu[hit] * L[m]
    z_hit = X[:, 2] + best * P[:, 2] / pxy
    return s_hit, np.where(np.isfinite(s_hit), z_hit, np.nan)


def gap0_hole(cfg, r_mm: float = 10.0, rim_mm: float = 8.0, dr_mm: float = 0.0, z_mm: float = 0.0, handoff=None,
              cut_r_mm: float = 0.0, centre: str = 'centroid', lay: dict = None, verbose: bool = True) -> dict:
    """A round hole on dee0's gap-0 edge where the hand-off bunch meets it, as a geometry.holes item.

    The bunch (``handoff``, default paths.handoff: the bunch the tracker launches; ``cut_r_mm`` > 0 keeps only the
    particles within that circle on the hand-off plane, as ``cut_handoff`` would) is turned with the inflector by the
    build's rotation and flown straight along each particle's own momentum to the gap-0 edge of dee0 as ``layout``
    builds it. The hole (radius r_mm, on the plane z_mm) is centred on the crossings' mean arclength, moved by dr_mm in
    RADIUS along the edge when asked; its plate reaches rim_mm beyond the hole on each side. Returns {'hole' (HoleSpec),
    'toml' (the [[geometry.holes]] item), 's_beam_mm', 'z_beam_mm', 's_orbit_mm', 'inside_fraction', 'n_hit', ...}."""
    from PyCentralRegion.handoff import load_handoff
    pc = C.section(cfg, 'paths')
    src = _existing(handoff or C.require(pc.handoff, 'paths.handoff'), 'the hand-off file')
    lay = lay or layout(cfg, verbose=False)
    h = load_handoff(str(src))
    keep = np.ones(len(h['r']), dtype=bool)
    if cut_r_mm > 0.0:
        keep, _ = _circle_on_plane(h, cut_r_mm, centre)
    rot = lay['si_rotation_deg']
    X = B.rotate_points(np.asarray(h['r'], dtype=float) * MM, rot)
    P = B.rotate_points(np.asarray(h['p_ev_c'], dtype=float), rot)
    edge, _, owner = B.gap_edge(lay['foot'], lay['dummy'], 0, 'dee')
    edge = np.asarray(edge, dtype=float)
    s_hit, z_hit = _fly_to_edge(edge, X, P)
    k = keep & np.isfinite(s_hit)
    if not k.any():
        raise ConfigError(f"no particle of {src} meets dee0's gap-0 edge by straight flight")
    s_beam, z_beam = float(s_hit[k].mean()), float(z_hit[k].mean())
    trj = lay['trajectory']
    hit_o = B.edge_crossings(edge, trj[:, :2] * MM, first_only=True) if len(trj) > 2 else None
    s_orb = float(hit_o[1]) if hit_o is not None else float('nan')
    s_c, moved = s_beam, None
    if dr_mm:
        moved = _shift_on_edge(edge, s_beam, float(dr_mm), "the gap-0 hole")
        s_c = moved['s1']
    # the spec to 0.1 um, as it is written into a design file: a hole re-derived from the same beam is then EQUAL to
    # the one in the file (46.12 - 18 is 28.119999999999997 in floating point, not the 28.12 the file holds)
    s_c = round(s_c, 4)
    rr = np.hypot(s_hit[k] - s_c, z_hit[k] - float(z_mm))
    hole = HoleSpec(0, 'dee', float(r_mm), s_c, float(z_mm), round(s_c - float(r_mm) - float(rim_mm), 4),
                    round(s_c + float(r_mm) + float(rim_mm), 4))
    toml = "\n".join(["[[geometry.holes]]"] + [f"{f} = {C.toml_value(round(v, 6) if isinstance(v, float) else v)}"
                                                for f, v in dataclasses.asdict(hole).items()])
    res = {'hole': hole, 'toml': toml, 's_beam_mm': s_beam, 'z_beam_mm': z_beam, 's_orbit_mm': s_orb,
           'edge_length_mm': float(np.linalg.norm(np.diff(edge, axis=0), axis=1).sum()), 'owner': owner,
           'n_particles': int(len(keep)), 'n_kept': int(keep.sum()), 'n_hit': int(k.sum()),
           'inside_fraction': float(np.mean(rr <= r_mm)), 'r_quantiles_mm': {q: float(np.quantile(rr, q))
                                                                             for q in (0.99, 0.999, 1.0)},
           'moved': moved, 'handoff': str(src), 'si_rotation_deg': rot}
    if verbose:
        print(f"[gap0 hole] {src}: {res['n_kept']} of {res['n_particles']} particles"
              + (f" (circle R {cut_r_mm:g} mm around the {centre})" if cut_r_mm > 0.0 else "")
              + f"; {res['n_hit']} meet dee0's gap-0 edge ({owner}, {res['edge_length_mm']:.1f} mm) by straight "
              f"flight, the inflector turned {rot:+.4f} deg")
        print(f"  the beam at the edge: s {s_beam:.2f} mm (rms {s_hit[k].std():.2f}), z {z_beam:+.2f} mm (rms "
              f"{z_hit[k].std():.2f}); the design orbit at s {s_orb:.2f}, z 0")
        if moved:
            print(f"  moved: hole centre s {s_beam:.2f} -> {s_c:.2f} mm along the edge (r {moved['r0']:.2f} -> "
                  f"{moved['r1']:.2f} mm, dr {dr_mm:+g} mm)")
        q = res['r_quantiles_mm']
        print(f"  a hole R {r_mm:g} mm at (s {s_c:.2f}, z {z_mm:+.2f}) holds {100.0 * res['inside_fraction']:.3f} % of "
              f"them; 99 / 99.9 / 100 % need R {q[0.99]:.2f} / {q[0.999]:.2f} / {q[1.0]:.2f} mm; the design orbit "
              f"{s_orb - s_c:+.2f} mm from its centre")
        print(toml)
    return res


# ---------------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog='python -m PyCentralRegion.geometry.build',
                                description=__doc__.split('\n\n')[0])
    sub = p.add_subparsers(dest='command', required=True)

    def with_config(name, text):
        q = sub.add_parser(name, help=text)
        q.add_argument('config', help='run TOML')
        q.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='override a config value')
        return q

    with_config('build', 'build the 3D electrode model -> paths.model3d')
    with_config('bounds', 'print the channel-height bounds')
    q = with_config('hole', 'a round hole on dee0 gap-0 edge where the hand-off bunch meets it')
    q.add_argument('--r', type=float, default=10.0, help='hole radius [mm]')
    q.add_argument('--rim', type=float, default=8.0, help='plate beyond the hole along the edge, each side [mm]')
    q.add_argument('--dr', type=float, default=0.0, help='move the hole by this change in radius [mm]')
    q.add_argument('--z', type=float, default=0.0, help='hole centre height [mm]')
    q.add_argument('--handoff', default=None, help='hand-off file (default paths.handoff)')
    q.add_argument('--cut-r', type=float, default=0.0, help='only the particles within this circle on the plane [mm]')
    q = with_config('shift', "move points on gap edges by a radius change: 'gap:side:s:dr' ...")
    q.add_argument('moves', nargs='+')
    q = sub.add_parser('cut', help='cut a hand-off file to a circle on its plane')
    q.add_argument('src')
    q.add_argument('dst')
    q.add_argument('--r', type=float, required=True, help='circle radius [mm]')
    q.add_argument('--centre', choices=('centroid', 'origin'), default='centroid')
    q = sub.add_parser('sections', help="x-y slices of a built model's metal")
    q.add_argument('model')
    q.add_argument('--z', default=None, help='heights [mm], comma-separated')
    a = p.parse_args(argv)
    if a.command == 'cut':
        cut_handoff(a.src, a.dst, a.r, a.centre)
        return 0
    if a.command == 'sections':
        V.plot_sections(a.model, z_list=[float(v) for v in a.z.split(',')] if a.z else None)
        return 0
    cfg = C.load(a.config, overrides=a.set)
    if a.command == 'build':
        build(cfg)
    elif a.command == 'bounds':
        print(json.dumps(bounds(cfg), indent=1))
    elif a.command == 'hole':
        gap0_hole(cfg, r_mm=a.r, rim_mm=a.rim, dr_mm=a.dr, z_mm=a.z, handoff=a.handoff, cut_r_mm=a.cut_r)
    elif a.command == 'shift':
        edge_shift(cfg, a.moves)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
