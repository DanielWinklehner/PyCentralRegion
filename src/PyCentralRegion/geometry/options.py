"""Options of the 3D electrode builder: the [geometry] section of a run file.

The builder (``PyCentralRegion.geometry.builder``) turns the seed's gap lines, the spiral-inflector housing outline
and the hand-off bunch's emitting plane into the central-region electrodes - the dees, the grounded dummy dees and
housing, posts, windows, round holes and collimator boxes - meshed for the BEM. Everything it can be told is a field
of ``GeometryConfig``; the machine (dee angles, harmonic, gap-width law, cavity radius, dee voltage) comes from
[machine], the spiral inflector's rotation from [inflector] (``si_rotation = "fixed"``) and the input files from
[paths] (seed, housing_outline, handoff; the output is paths.model3d). An input file the builder needs beyond those is
a field here whose name ends in ``_file``.

Lengths are millimetres (the units of the housing outline and of gmsh), angles degrees.

The beam channel: every dee is hollowed by a channel of full height ``aperture_inner_mm`` inside the radial taper and
``aperture_outer_mm`` beyond it, linear in between, from r0 = aperture_taper_center_mm - aperture_taper_length_mm / 2
to r1 = aperture_taper_center_mm + aperture_taper_length_mm / 2 (``channel_height_mm``). The dummy dees (and the star
ground) share the taper radii and take their own heights when ``dummy_aperture_inner_mm`` / ``dummy_aperture_outer_mm``
are non-zero. The four are independent (an optimiser needs no ordering constraint: inner > outer is an inverted
taper). ``check_bounds`` holds them to what can be built, at build time:

* every height <= 2 x (plate_half_gap_mm - dee_wall_mm): the dee plates must fit inside the plate / pole gap;
* the channel clears the spiral-inflector housing's exit, the opening the beam leaves the housing by: the inner
  heights, and the heights at the exit's outermost radius, >= 2 x the exit's half height (the largest |z| of the exit
  aperture the housing outline carries, after aperture_inset_mm). The second binds only for an inverted taper that
  starts inside the exit.

Hardware on a gap edge is placed by ARCLENGTH along that edge [mm] from the edge's inner end. Every gap has one dee
side (gap 2j is dee j's low-azimuth edge, gap 2j+1 its high-azimuth edge) and one dummy side facing it across the
gap. A rule that moves an edge's inner end (dee0's tip cut, a pinched dee, ``dee_rmin``) moves that origin, so specs
written against one build are re-derived for another, not translated (``build.edge_shift`` turns a radius change
into an arclength; a round dee0 nose keeps the origin at the old vertex).

Older run scripts gave the channel and the hardware as command-line strings and JSON files: ``aperture_from_spec``,
``posts_from_spec`` and their siblings and ``from_posts_json`` convert those into this section's fields, and
``to_toml`` writes a section as a run file's [geometry] table.

This module needs only the standard library, so a run file can be checked without the physics stack.
"""
import dataclasses
import json
from pathlib import Path

from PyCentralRegion.config import ConfigError, toml_value

SIDES = ('dee', 'dummy')


def _side(side, what):
    if side not in SIDES:
        raise ConfigError(f"{what}: side must be 'dee' or 'dummy', got {side!r}")


# ---------------------------------------------------------------------------------------------------------------------
# hardware items (arrays of tables in the run file)
@dataclasses.dataclass
class PostSpec:
    """A post on a gap edge: near face flush with the edge (post_offset_mm = half its size), the full channel height
    plus post_overhang_mm. The same arclength on both sides of one gap makes a facing pair: a jaw across the gap."""
    gap: int
    side: str                  # 'dee' | 'dummy'
    s_mm: float                # centre, arclength along the edge
    length_mm: float = 0.0     # extent along the edge, centred on s_mm; 0 = post_size_mm

    def __post_init__(self):
        _side(self.side, f"post on gap {self.gap}")


@dataclasses.dataclass
class WindowSpec:
    """A vertical window on a gap edge: metal above and below an OPENING z_center_mm +- half_height_mm, over the
    arclength s_lo_mm .. s_hi_mm (with two or more posts on that edge the range snaps to their outermost centres).

    The window narrows the gap's field vertically, the same physics as the channel height: half_height_mm is an
    optimisation variable (``geometry.windows.<i>.half_height_mm``), bounded at build time by 0 < half_height_mm and
    |z_center_mm| + half_height_mm <= the channel's half height at the window."""
    gap: int
    side: str
    s_lo_mm: float
    s_hi_mm: float
    half_height_mm: float
    z_center_mm: float = 0.0

    def __post_init__(self):
        _side(self.side, f"window on gap {self.gap}")
        if not self.s_hi_mm > self.s_lo_mm:
            raise ConfigError(f"window on gap {self.gap} {self.side}: wants s_lo_mm < s_hi_mm, got {self.s_lo_mm:g} / "
                              f"{self.s_hi_mm:g}")
        if not self.half_height_mm > 0.0:
            raise ConfigError(f"window on gap {self.gap} {self.side}: half_height_mm must be > 0, got "
                              f"{self.half_height_mm:g}")

    @property
    def z_lo_mm(self) -> float:
        return float(self.z_center_mm) - float(self.half_height_mm)

    @property
    def z_hi_mm(self) -> float:
        return float(self.z_center_mm) + float(self.half_height_mm)


@dataclasses.dataclass
class HoleSpec:
    """A plate on a gap edge (s_lo_mm .. s_hi_mm, post_size_mm deep, the full channel height plus the overhang) with
    a round hole of radius r_mm along the edge normal, centred at arclength s_mm and height z_mm."""
    gap: int
    side: str
    r_mm: float
    s_mm: float
    z_mm: float
    s_lo_mm: float
    s_hi_mm: float

    def __post_init__(self):
        _side(self.side, f"hole on gap {self.gap}")
        if not self.s_lo_mm <= self.s_mm - self.r_mm < self.s_mm + self.r_mm <= self.s_hi_mm:
            raise ConfigError(f"hole on gap {self.gap} {self.side}: the R {self.r_mm:g} mm hole at s {self.s_mm:g} "
                              f"runs past its plate {self.s_lo_mm:g} .. {self.s_hi_mm:g} mm")


@dataclasses.dataclass
class BoxSpec:
    """A grounded full-height rectangular collimator [mm, machine frame], fused into the ground."""
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float

    def __post_init__(self):
        if not (self.x_max_mm > self.x_min_mm and self.y_max_mm > self.y_min_mm):
            raise ConfigError(f"box {self}: wants x_min < x_max and y_min < y_max")


@dataclasses.dataclass
class JunctionMove:
    """Move a gap's variable -> fixed junction to radius r_mm along its last variable segment's own line."""
    gap: int
    r_mm: float


@dataclasses.dataclass
class DeeRmin:
    """Start dee `dee` (1 .. n-1) at r_mm: both edges cut there and closed by an arc about the machine axis."""
    dee: int
    r_mm: float


# ---------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class GeometryConfig:
    """The [geometry] section: every option of the electrode builder (configs/defaults/geometry.toml documents them
    with their defaults)."""
    # --- gap lines -----------------------------------------------------------------------------------------------
    snap_nodes: bool = True              # zero inert variable segments, then put the node radii midway between turns
    junctions: tuple[JunctionMove, ...] = ()   # after the snap: junctions moved to radii between the 3D beam's passes
    junction_tol_deg: float = 1.0        # close a variable -> fixed junction whose net angular step is below this
    # --- spiral inflector ----------------------------------------------------------------------------------------
    si_rotation: str = 'fixed'           # 'fixed': [inflector] rotation_deg; 'corner': gap-0 ray - the spec's azimuth
    gap0_on_plate: bool = False          # re-fit gap 0's variable segment onto the turned exit-plate line
    exit_plane_spec_file: str = ''       # exit_plane_spec.json; '' = the export folder above the housing outline's
    emitting_plane_file: str = ''        # hand-off file whose particle cloud is the emitting plane; '' = the export's
                                         # own (handoff_sc_all_level.h5 beside the outline's folder), else paths.handoff
    channel_back_mm: float = 1.0         # beam channel starts this far upstream of the emitting plane
    channel_extend_mm: float = 200.0     # ... and runs this far downstream along its normal
    aperture_inset_mm: tuple = (0.0, 0.0)   # narrow the exit aperture: off each end of its (longer, shorter) edge
    cluster_tol_mm: float = 0.5          # outline files without curves: vertices this close to a long face cluster
    keep_every: int = 5                  # ... and every Nth loose vertex is a spline knot
    # --- vertical structure --------------------------------------------------------------------------------------
    half_height_mm: float = 50.0         # electrodes and housing span z = -H .. +H
    aperture_inner_mm: float = 26.0      # dee channel full height inside the taper
    aperture_outer_mm: float = 30.0      # ... and outside it
    aperture_taper_center_mm: float = 120.0   # the taper runs linearly over centre -+ length / 2
    aperture_taper_length_mm: float = 40.0    # (> 0)
    dummy_aperture_inner_mm: float = 0.0      # dummy dees' (and star ground's) full heights; 0 = the dees' value
    dummy_aperture_outer_mm: float = 0.0
    plate_half_gap_mm: float = 20.0      # half the vertical gap of the plates / poles the dees sit between: every
                                         # channel height <= 2 x (plate_half_gap_mm - dee_wall_mm)
    # --- dees ----------------------------------------------------------------------------------------------------
    r_max_mm: float = 0.0                # dees and model end here (<= [machine] r_max_cavity_m); 0 = r_max_cavity_m
    housing_clearance_mm: float = 10.0   # the dee metal of gaps 1 .. n stays this far off the housing
    dee_wall_mm: float = 5.0             # thickness of a dee's walls: its solid inner end wall (radially) and its
                                         # plates (the plate bound above)
    dee_inner: str = 'arc'               # dees 1 .. n-1 close on 'arc' (about the axis) or 'hug' (the housing)
    hug_ds_mm: float = 4.0               # 'hug': sampling along the closure
    hug_ring_n: int = 180                # 'hug': rays of the end-wall ring
    dee_rmin: tuple[DeeRmin, ...] = ()   # inner ends moved out by hand (override pinch / hug)
    dee_tip_width_mm: float = 4.0        # a pinched dee's tip is cut back to a face this wide; 0 = off
    dummy_tip_width_mm: float = 4.0      # a dummy tip crossing outside the housing is cut back to this; 0 = vertex
    dummy_overlap_mm: float = 6.0        # dummy dees reach this far into the housing
    dee0_clearance_mm: float = 0.0       # cut dee0's tip back until it clears the housing by this; 0 = off
    dee0_trim_mm: float = 0.0            # or cut it back this far along both edges; 0 = off
    dee0_nose_mm: float = 0.0            # round dee0's vertex with this radius in plan view; 0 = vertex
    # --- ground --------------------------------------------------------------------------------------------------
    ground: str = 'dummies'              # 'dummies' (wedges fused onto the housing) | 'star' (one 2D layout)
    star_tip_gap_mm: float = 0.0         # star: ground gap round the dee tips; 0 = [machine] gap_width_min_m
    star_min_ground_mm: float = 2.0      # star: thinnest ground allowed between two dees' keep-outs
    gap0_face_mm: float = 0.0            # star: fill gap 0's dummy side down to the housing where it is this close
    # --- hardware ------------------------------------------------------------------------------------------------
    posts: tuple[PostSpec, ...] = ()
    gap0_posts_mm: tuple = ()            # (s_inner, s_outer): a post pair on dee0's gap-0 edge
    windows: tuple[WindowSpec, ...] = ()
    holes: tuple[HoleSpec, ...] = ()
    boxes: tuple[BoxSpec, ...] = ()
    post_size_mm: float = 10.0           # post width and depth; window / hole plate depth
    post_offset_mm: float = 5.0          # post centre from its edge (half the size: face flush)
    post_overhang_mm: float = 5.0        # posts and plates reach this far past the channel (its highest over them)
    post_shape: str = 'square'           # 'square' | 'cylinder' (a post_facets-gon, no 90 deg edges in the gap)
    post_facets: int = 16
    post_wall_join_mm: float = 2.0       # carry a post end this close to a hug dee's end wall into it; 0 = never
    # --- mesh ----------------------------------------------------------------------------------------------------
    lc_mm: float = 8.0                   # surface mesh size
    lc_outer_mm: float = 0.0             # > lc_mm: graded, lc_mm inside r_fine_mm, this beyond r_coarse_mm
    r_fine_mm: float = 150.0
    r_coarse_mm: float = 220.0
    lc_posts_mm: float = 0.0             # < lc_mm: refine round every post, window block and hole
    nose_lc_mm: float = 2.0              # a round dee0 nose: this size on it, twice it out to the ground; 0 = off
    mesh_algorithm: int = 6              # gmsh Mesh.Algorithm (6 = Frontal-Delaunay)
    mesh_size_min_mm: float = 0.0        # gmsh Mesh.MeshSizeMin; 0 = lc_mm / 2
    mesh_flip_min_deg: float = 5.0       # repair triangles with an angle below this by edge flips; 0 = off
    occ_fuzzy_mm: float = 0.05           # OCC fuzzy boolean tolerance; 0 = exact booleans
    # --- output --------------------------------------------------------------------------------------------------
    plot_dees: bool = True               # write <model>_dees.png (the 2D layout) next to the model

    def __post_init__(self):
        for name, value, allowed in (('si_rotation', self.si_rotation, ('fixed', 'corner')),
                                     ('dee_inner', self.dee_inner, ('arc', 'hug')),
                                     ('ground', self.ground, ('dummies', 'star')),
                                     ('post_shape', self.post_shape, ('square', 'cylinder'))):
            if value not in allowed:
                raise ConfigError(f"geometry.{name} = {value!r}; allowed: {', '.join(allowed)}")
        if len(self.aperture_inset_mm) != 2:
            raise ConfigError(f"geometry.aperture_inset_mm wants two values, got {self.aperture_inset_mm!r}")
        if len(self.gap0_posts_mm) not in (0, 2):
            raise ConfigError(f"geometry.gap0_posts_mm wants two values or none, got {self.gap0_posts_mm!r}")
        for name in ('aperture_inner_mm', 'aperture_outer_mm', 'aperture_taper_length_mm', 'half_height_mm',
                     'plate_half_gap_mm', 'lc_mm'):
            if not getattr(self, name) > 0.0:
                raise ConfigError(f"geometry.{name} must be > 0, got {getattr(self, name)!r}")
        for name in ('dummy_aperture_inner_mm', 'dummy_aperture_outer_mm', 'dee_wall_mm'):
            if getattr(self, name) < 0.0:
                raise ConfigError(f"geometry.{name} must be >= 0, got {getattr(self, name)!r}")
        r0, r1 = self.aperture_radii()
        if not 0.0 < r0 < r1 < 1000.0:
            raise ConfigError(f"geometry.aperture_taper_center_mm / aperture_taper_length_mm: the taper must lie inside "
                              f"0 < r < 1000 mm, got {r0:g} .. {r1:g} mm (centre {self.aperture_taper_center_mm:g}, "
                              f"length {self.aperture_taper_length_mm:g})")
        if max(self.aperture_heights() + self.dummy_aperture_heights()) >= 2.0 * self.half_height_mm:
            raise ConfigError(f"geometry: a channel height reaches the electrodes' full height 2 x half_height_mm = "
                              f"{2.0 * self.half_height_mm:g} mm")
        if self.lc_outer_mm > self.lc_mm and not self.r_coarse_mm > self.r_fine_mm:
            raise ConfigError(f"geometry: a graded mesh needs r_coarse_mm > r_fine_mm, got {self.r_fine_mm} / "
                              f"{self.r_coarse_mm}")
        # options that would otherwise be ignored without a word
        if self.gap0_face_mm > 0.0 and self.ground != 'star':
            raise ConfigError(f"geometry.gap0_face_mm = {self.gap0_face_mm:g} fills gap 0's dummy side of the star "
                              f"ground; ground = {self.ground!r} builds no such face (set ground = 'star', or "
                              f"gap0_face_mm = 0)")
        if self.dee0_clearance_mm > 0.0 and self.dee0_trim_mm > 0.0:
            raise ConfigError("geometry.dee0_clearance_mm and dee0_trim_mm both cut dee0's tip back: set one of them "
                              "(with both, the clearance rule applies and the trim is unused)")

    # the channel, as the builder reads it
    def aperture_heights(self) -> tuple:
        """The dees' channel full heights (inside, outside) the taper [mm]."""
        return float(self.aperture_inner_mm), float(self.aperture_outer_mm)

    def dummy_aperture_heights(self) -> tuple:
        """The dummy dees' (and the star ground's) channel full heights [mm]; 0 fields take the dees' values."""
        inner, outer = self.aperture_heights()
        return (float(self.dummy_aperture_inner_mm) or inner, float(self.dummy_aperture_outer_mm) or outer)

    def aperture_radii(self) -> tuple:
        """(r0, r1) [mm]: the full heights change linearly from the inner to the outer value between these radii."""
        c, h = float(self.aperture_taper_center_mm), 0.5 * float(self.aperture_taper_length_mm)
        return c - h, c + h


def channel_height_mm(r_mm, radii, heights) -> float:
    """The channel's full height [mm] at radius r_mm [mm]: heights[0] inside radii[0], heights[1] beyond radii[1],
    linear between. The one profile of the build (the aperture tool the electrodes are hollowed with, the hardware
    heights) and of the bounds; the arithmetic is numpy.interp's, bit for bit, so models built before it are unchanged."""
    r, (r0, r1), (h0, h1) = float(r_mm), map(float, radii), map(float, heights)
    if r <= r0:
        return h0
    if r >= r1:
        return h1
    return (h1 - h0) / (r1 - r0) * (r - r0) + h0


# ---------------------------------------------------------------------------------------------------------------------
# bounds of the channel heights (checked at build time)
def channel_height_max_mm(geometry: GeometryConfig) -> float:
    """The largest channel full height the plates leave room for: 2 x (plate_half_gap_mm - dee_wall_mm)."""
    return 2.0 * (float(geometry.plate_half_gap_mm) - float(geometry.dee_wall_mm))


def check_bounds(geometry: GeometryConfig, exit_half_height_mm: float = None, exit_r_mm: float = None) -> dict:
    """Hold the channel heights to what can be built; a ConfigError names the first bound violated.

    * every height (dees and, when set, dummy dees) <= 2 x (plate_half_gap_mm - dee_wall_mm);
    * the channel clears the spiral-inflector housing's exit, whose half height ``exit_half_height_mm`` is the largest
      |z| of its exit aperture (``builder.exit_half_height_mm`` reads it off the housing outline): the inner heights
      >= 2 x exit_half_height_mm, and so are the heights of the dees' and of the dummy dees' channel at the exit's
      outermost radius ``exit_r_mm`` (``builder.exit_radius_mm``). A height profile is monotonic in r, so the two
      together hold the channel over the opening's whole radial span; the second binds only for an inverted taper
      that starts inside the exit. Neither is checked without the exit's half height (no outline at hand).
    Returns the bounds {'height_max_mm', 'inner_min_mm', 'exit_r_mm'} for a campaign to set its variable ranges from."""
    g = geometry
    top = channel_height_max_mm(g)
    heights = [('aperture_inner_mm', g.aperture_inner_mm, True), ('aperture_outer_mm', g.aperture_outer_mm, False)]
    if g.dummy_aperture_inner_mm > 0.0:
        heights.append(('dummy_aperture_inner_mm', g.dummy_aperture_inner_mm, True))
    if g.dummy_aperture_outer_mm > 0.0:
        heights.append(('dummy_aperture_outer_mm', g.dummy_aperture_outer_mm, False))
    if top <= 0.0:
        raise ConfigError(f"geometry.plate_half_gap_mm = {g.plate_half_gap_mm:g} mm leaves no room for the "
                          f"geometry.dee_wall_mm = {g.dee_wall_mm:g} mm dee plates")
    for name, v, _ in heights:
        if float(v) > top + 1e-9:
            raise ConfigError(f"geometry.{name} = {float(v):g} mm exceeds its bound 2 x (geometry.plate_half_gap_mm - "
                              f"geometry.dee_wall_mm) = 2 x ({g.plate_half_gap_mm:g} - {g.dee_wall_mm:g}) = {top:g} mm: "
                              f"the dee plates must fit inside the plate / pole gap")
    lo = None
    if exit_half_height_mm is not None:
        lo = 2.0 * float(exit_half_height_mm)
        for name, v, inner in heights:
            if inner and float(v) < lo - 1e-9:
                raise ConfigError(f"geometry.{name} = {float(v):g} mm is below its bound 2 x the spiral-inflector "
                                  f"housing exit's half height = 2 x {float(exit_half_height_mm):.3f} = {lo:.3f} mm: "
                                  f"the channel must clear the exit the beam leaves the housing by")
        if exit_r_mm is not None:
            profiles = [('aperture_inner_mm / aperture_outer_mm', g.aperture_heights())]
            if g.dummy_aperture_heights() != g.aperture_heights():
                profiles.append(('dummy_aperture_inner_mm / dummy_aperture_outer_mm', g.dummy_aperture_heights()))
            for names, hh in profiles:
                h = channel_height_mm(exit_r_mm, g.aperture_radii(), hh)
                if h < lo - 1e-9:
                    raise ConfigError(f"geometry.{names}: the channel is {h:.3f} mm high at the spiral-inflector housing "
                                      f"exit's outermost radius r = {float(exit_r_mm):.2f} mm, below its bound 2 x the "
                                      f"exit's half height = {lo:.3f} mm: the channel must clear the exit over its "
                                      f"whole radial span (start the taper, aperture_taper_center_mm - "
                                      f"aperture_taper_length_mm / 2, outside that radius, or keep the height there)")
    return {'height_max_mm': top, 'inner_min_mm': lo, 'exit_r_mm': None if exit_r_mm is None else float(exit_r_mm)}


# ---------------------------------------------------------------------------------------------------------------------
# converters from the older command-line strings and JSON files
def _items(spec, sep=','):
    return [s.strip() for s in str(spec or '').split(sep) if s.strip()]


def _fields(item, n, form):
    f = [v.strip() for v in item.split(':')]
    if len(f) not in n:
        raise ConfigError(f"{item!r}: wants {form!r}")
    return f


def _pair(spec, what):
    v = [float(x) for x in _items(spec)]
    if len(v) != 2:
        raise ConfigError(f"{what} wants two comma-separated values, got {spec!r}")
    return v


def aperture_from_spec(heights, radii=None, dummy=None) -> dict:
    """The old "inner,outer" channel heights, "r0,r1" taper radii and "inner,outer" dummy heights -> the
    GeometryConfig channel fields. ``radii`` None keeps the default taper."""
    inner, outer = _pair(heights, 'the channel heights')
    out = {'aperture_inner_mm': inner, 'aperture_outer_mm': outer}
    if radii:
        r0, r1 = _pair(radii, 'the taper radii')
        if not r1 > r0:
            raise ConfigError(f"the taper radii want r0 < r1, got {radii!r}")
        out.update(aperture_taper_center_mm=0.5 * (r0 + r1), aperture_taper_length_mm=r1 - r0)
    if dummy:
        d_in, d_out = _pair(dummy, 'the dummy channel heights')
        out.update(dummy_aperture_inner_mm=d_in, dummy_aperture_outer_mm=d_out)
    return out


def posts_from_spec(spec):
    """'gap:side:s[:length],...' -> PostSpec tuple."""
    out = []
    for item in _items(spec):
        f = _fields(item, (3, 4), 'gap:side:s[:length]')
        out.append(PostSpec(int(f[0]), f[1], float(f[2]), float(f[3]) if len(f) == 4 else 0.0))
    return tuple(out)


def window_from_opening(gap, side, s_lo, s_hi, z_lo, z_hi) -> WindowSpec:
    """A window given by its opening z_lo .. z_hi (the old convention) -> WindowSpec (centre +- half height)."""
    z_lo, z_hi = float(z_lo), float(z_hi)
    if not z_hi > z_lo:
        raise ConfigError(f"window on gap {gap} {side}: wants z_lo < z_hi, got {z_lo:g} / {z_hi:g}")
    return WindowSpec(int(gap), side, float(s_lo), float(s_hi), 0.5 * (z_hi - z_lo), 0.5 * (z_hi + z_lo))


def windows_from_spec(spec):
    """'gap:side:s_lo:s_hi:z_lo:z_hi,...' (z_lo .. z_hi the OPENING) -> WindowSpec tuple."""
    out = []
    for item in _items(spec):
        f = _fields(item, (6,), 'gap:side:s_lo:s_hi:z_lo:z_hi')
        out.append(window_from_opening(int(f[0]), f[1], *f[2:]))
    return tuple(out)


def holes_from_spec(spec):
    """'gap:side:r:s_c:z_c:s_lo:s_hi,...' -> HoleSpec tuple."""
    out = []
    for item in _items(spec):
        f = _fields(item, (7,), 'gap:side:r:s_c:z_c:s_lo:s_hi')
        out.append(HoleSpec(int(f[0]), f[1], *map(float, f[2:])))
    return tuple(out)


def boxes_from_spec(spec):
    """'x_min,x_max,y_min,y_max|...' [mm] -> BoxSpec tuple."""
    out = []
    for item in _items(spec, '|'):
        v = [float(x) for x in item.split(',')]
        if len(v) != 4:
            raise ConfigError(f"{item!r}: wants 'x_min,x_max,y_min,y_max'")
        out.append(BoxSpec(*v))
    return tuple(out)


def junctions_from_spec(spec):
    """'gap:r_mm,...' -> JunctionMove tuple."""
    return tuple(JunctionMove(int(f[0]), float(f[1])) for f in (_fields(i, (2,), 'gap:r_mm') for i in _items(spec)))


def dee_rmin_from_spec(spec):
    """'dee:r_mm,...' -> DeeRmin tuple."""
    return tuple(DeeRmin(int(f[0]), float(f[1])) for f in (_fields(i, (2,), 'dee:r_mm') for i in _items(spec)))


def from_posts_json(path):
    """A post-placement JSON file ({"gap0_posts": [s_in, s_out] | null, "gap_posts": {"g:side": [s, ...]},
    "gap_windows": {"g:side": [s_lo, s_hi, z_lo, z_hi]}}) -> dict(posts, windows, gap0_posts_mm) for GeometryConfig."""
    d = json.loads(Path(path).read_text())
    posts, windows = [], []
    for key, ss in (d.get('gap_posts') or {}).items():
        g, side = key.split(':')
        posts += [PostSpec(int(g), side, float(s)) for s in ss]
    for key, w in (d.get('gap_windows') or {}).items():
        g, side = key.split(':')
        windows.append(window_from_opening(int(g), side, *w))
    g0 = d.get('gap0_posts')
    return {'posts': tuple(posts), 'windows': tuple(windows), 'gap0_posts_mm': tuple(map(float, g0)) if g0 else ()}


def junctions_from_json(path):
    """A junction-state JSON file ({"spec": "gap:r_mm,..."}) -> JunctionMove tuple."""
    return junctions_from_spec(json.loads(Path(path).read_text()).get('spec', ''))


# ---------------------------------------------------------------------------------------------------------------------
def to_toml(geometry, only_changed=True):
    """The [geometry] table of a run file for these options (a GeometryConfig, or a dict of its keyword arguments);
    only what differs from the defaults unless told not."""
    if isinstance(geometry, dict):
        geometry = GeometryConfig(**geometry)
    base = GeometryConfig()
    lines, tables = ['[geometry]'], []
    for f in dataclasses.fields(geometry):
        v = getattr(geometry, f.name)
        if only_changed and v == getattr(base, f.name):
            continue
        if isinstance(v, (tuple, list)) and v and dataclasses.is_dataclass(v[0]):
            for item in v:
                tables.append(f"\n[[geometry.{f.name}]]")
                tables += [f"{k} = {toml_value(x)}" for k, x in dataclasses.asdict(item).items()]
        else:
            lines.append(f"{f.name} = {toml_value(v)}")
    return '\n'.join(lines + tables) + '\n'
