"""
inflector.py - The spiral inflector inside the central-region model.

Since 2026-09-10 the spiral-inflector deck (spyral_inflector) hands over three
things in ONE frame, the machine frame (see
``Spiral_inflector/Docs/coordinate_convention.md``; the deck applies its
z-mirror and the positioning rotation before writing):

  * the bunch at the plane through the ends of the spiral electrodes
    (openPMD plane-crossing file, ``handoff.py``; hand-off distance 0),
  * the STATIC electric field of the inflector electrodes and housing
    (a PyPATools ``Field`` pickle / h5, 3D grid, V/m, metres), and
  * the HOUSING as a STEP solid (mm, grounded).

Everything downstream of that plane happens here: the static field is
superposed on the central region's RF field (``superpose_efield`` /
``StaticPlusRFField``, so the tracker's ``set_time`` modulation keeps working),
the housing becomes a particle-terminating obstacle (2D: its midplane outline
rasterised, ``obstacle``; 3D: the solid itself via ``metal_mask``) and, for
the BEM gap field, the grounded scroll of ``gap_fields.build_gap_electrodes``
(``housing=model.housing_polygon()``) or an ``ExtraSolid`` of the 3D builder.

``InflectorModel`` bundles the three files with one common frame transform
(``rotation_deg`` about z, optional ``flip_z`` mirror) so that bunch, field
and housing always move together; ``AcceleratedOrbitFinder.attach_inflector``
installs field and obstacle for tracking and optimisation.
"""
import os
import time
import warnings
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from PyPATools.field import Field, FieldBase, TimedField, CompositeField

from .handoff import load_handoff, make_beam_from_handoff, _rot


# ============================================================================
# Field superposition
# ============================================================================
def _is_zero_field(f) -> bool:
    return f is None or getattr(f, 'label', '') == 'Zero Field' or getattr(f, '_label', '') == 'Zero Field'


class StaticPlusRFField(TimedField):
    """RF gap field pattern x cos(omega t + phase) + static fields.

    Subclass of ``TimedField`` so the tracker's ``set_time`` and the engine's
    omega / phase re-sync (``TrackingEngine._sync_bem_field``) act on the RF
    part only; the static fields (inflector, stray fields) are added as they
    are. ``rf`` may be None (thin-gap kicks, no RF field to integrate)."""

    def __init__(self, rf: Optional[TimedField], statics: Sequence[FieldBase]):
        self.rf = rf
        self.statics = list(statics)
        if rf is not None:
            super().__init__(rf.static, omega=rf.omega, phase=rf.phase, t=rf.t)
        else:
            super().__init__(Field.zero(dim=0), omega=0.0, phase=0.0, t=0.0)

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        if self.rf is not None:
            out = self.static(pts) * np.cos(self.omega * self.t + self.phase)
        else:
            out = np.zeros((len(pts), 3))
        for s in self.statics:
            out = out + s(pts)
        return out

    def __str__(self):
        return (f"StaticPlusRFField({len(self.statics)} static field(s) + "
                f"{'RF ' + str(self.rf) if self.rf is not None else 'no RF field'})")


def superpose_efield(base, statics: Sequence[FieldBase]) -> FieldBase:
    """``base`` (the design's current E-field: a TimedField, a plain Field,
    a zero field or None) plus the static fields, as one FieldBase.

    A time-modulated base keeps its modulation (``StaticPlusRFField``); a
    static base becomes part of a ``CompositeField``; a zero base drops out."""
    statics = [s for s in statics if s is not None]
    if isinstance(base, StaticPlusRFField):
        base = base.rf
    if not statics:
        return base
    if base is not None and hasattr(base, 'set_time'):
        return StaticPlusRFField(base, statics)
    parts = ([] if _is_zero_field(base) else [base]) + statics
    if len(parts) == 1:
        return parts[0]
    return CompositeField(parts, [1.0] * len(parts))


# ============================================================================
# Static field of the inflector
# ============================================================================
def load_inflector_field(source: Union[str, os.PathLike, FieldBase], rotation_deg: float = 0.0,
                         flip_z: bool = False, backend: str = 'numba',
                         label: Optional[str] = None, verbose: bool = True) -> Field:
    """PyPATools ``Field`` of the inflector's static E-field in the frame of
    the central region.

    ``source``: a Field file (``.pickle`` as written by spyral_inflector's
    ``tracking.export`` / ``bem_reload``, or ``.h5``) or a Field. Grid in
    metres, values in V/m. ``flip_z`` mirrors a DECK-frame map through the
    median plane (z -> -z, E_z -> -E_z; the deck's own machine-frame export
    has this applied already). ``rotation_deg`` rotates the map about z
    (re-sampled on the same grid, so it blurs by up to one cell: keep the
    map on a fine grid or rotate the bunch and housing instead). The result
    is re-wrapped on the fast ``backend`` interpolator (the pickles carry
    scipy's, an order of magnitude slower for tracking)."""
    t0 = time.time()
    if isinstance(source, FieldBase):
        f = source
    else:
        f = Field.from_file(str(source))
    grid, values = f.grid, f.grid_values
    if grid is None or values is None or int(getattr(f, 'dim', 3)) != 3:
        raise ValueError("load_inflector_field needs a gridded 3D Field (grid + grid_values)")
    gx, gy, gz = (np.asarray(grid[k], dtype=float) for k in 'xyz')
    vx, vy, vz = (np.asarray(values[k], dtype=float) for k in 'xyz')
    if flip_z:
        order = np.argsort(-gz)
        gz = -gz[order]
        vx, vy, vz = vx[:, :, order], vy[:, :, order], -vz[:, :, order]
    if rotation_deg:
        R = _rot(rotation_deg)
        X, Y, Z = np.meshgrid(gx, gy, gz, indexing='ij')
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        src = Field.from_arrays(grid={'x': gx, 'y': gy, 'z': gz},
                                values={'x': vx, 'y': vy, 'z': vz}, dim=3, units='m',
                                interpolator_backend=backend)
        e = src(pts @ R) @ R.T            # E'(p) = R E(R^-1 p); pts @ R == (R^-1 p)^T rows
        vx, vy, vz = (e[:, i].reshape(X.shape) for i in range(3))
    out = Field.from_arrays(grid={'x': gx, 'y': gy, 'z': gz}, values={'x': vx, 'y': vy, 'z': vz},
                            dim=3, units='m', interpolator_backend=backend,
                            label=label or f"inflector E-field ({getattr(f, 'label', 'static')})")
    out._metadata = dict(getattr(f, 'metadata', {}) or {})
    out._metadata.update({'source': str(source) if not isinstance(source, FieldBase) else 'Field',
                          'rotation_deg': float(rotation_deg), 'flip_z': bool(flip_z)})
    if verbose:
        emax = float(np.sqrt(vx ** 2 + vy ** 2 + vz ** 2).max())
        print(f"[inflector] E-field grid {len(gx)}x{len(gy)}x{len(gz)}, x {gx[0] * 1e3:.0f}..{gx[-1] * 1e3:.0f}, "
              f"y {gy[0] * 1e3:.0f}..{gy[-1] * 1e3:.0f}, z {gz[0] * 1e3:.0f}..{gz[-1] * 1e3:.0f} mm, "
              f"|E| max {emax / 1e5:.2f} kV/cm"
              + (f", rotated {rotation_deg:+.1f} deg" if rotation_deg else "")
              + (", z-mirrored" if flip_z else "") + f" ({time.time() - t0:.1f} s)")
    return out


def midplane_field(field3d: FieldBase, z: float = 0.0, ez: str = 'zero',
                   backend: str = 'numba', label: Optional[str] = None) -> Field:
    """2D (x, y) slice of a gridded 3D field at height ``z`` for the 2D
    tracking model. ``ez='zero'`` drops E_z (particles stay in the plane;
    the inflector's exit field is horizontal on the design orbit anyway),
    ``'keep'`` interpolates it."""
    grid = field3d.grid
    gx, gy = np.asarray(grid['x'], dtype=float), np.asarray(grid['y'], dtype=float)
    X, Y = np.meshgrid(gx, gy, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, float(z))])
    e = field3d(pts)
    ex, ey = e[:, 0].reshape(X.shape), e[:, 1].reshape(X.shape)
    ezv = e[:, 2].reshape(X.shape) if ez == 'keep' else np.zeros_like(ex)
    out = Field.from_arrays(grid={'x': gx, 'y': gy}, values={'x': ex, 'y': ey, 'z': ezv},
                            dim=2, units='m', interpolator_backend=backend,
                            label=label or f"{getattr(field3d, 'label', 'E')} @ z = {z * 1e3:.1f} mm")
    return out


# ============================================================================
# Housing: STEP -> surface mesh -> midplane outline
# ============================================================================
def housing_mesh(step_path: Union[str, os.PathLike], scale: float = 1e-3, rotation_deg: float = 0.0,
                 flip_z: bool = False, mesh_size: float = 2e-3, verbose: bool = False):
    """Closed ``trimesh.Trimesh`` [m] of every solid in a STEP file, placed
    like ``electrodes3d.ExtraSolid`` (scale to metres, rotate about z,
    mirror through z = 0). ``mesh_size`` [m] bounds the surface triangles
    (2 mm resolves a 4 mm wall; the plane sections are exact on the facets)."""
    import trimesh
    from .polyprism import _GmshModel, MM
    t0 = time.time()
    with _GmshModel("inflector_housing", verbose) as gmsh:
        occ = gmsh.model.occ
        out = occ.importShapes(str(step_path))
        vols = [(d, t) for d, t in out if d == 3]
        if not vols:
            raise RuntimeError(f"{step_path}: no volumes in the STEP file")
        f = float(scale) * MM
        if abs(f - 1.0) > 1e-12:
            occ.dilate(vols, 0.0, 0.0, 0.0, f, f, f)
        if rotation_deg:
            occ.rotate(vols, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, np.radians(rotation_deg))
        if flip_z:
            occ.mirror(vols, 0.0, 0.0, 1.0, 0.0)
        occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.25 * mesh_size * MM)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size * MM)
        gmsh.model.mesh.generate(2)
        ntag, ncoord, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(ncoord).reshape(-1, 3) / MM
        remap = {int(t): i for i, t in enumerate(ntag)}
        tris = []
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        for ty, nodes in zip(etypes, enodes):
            if ty == 2:
                tris.append(np.vectorize(remap.get)(np.asarray(nodes).reshape(-1, 3)))
        tris = np.vstack(tris)
    m = trimesh.Trimesh(vertices=xyz, faces=tris, process=False)
    if verbose:
        print(f"[inflector] housing {os.path.basename(str(step_path))}: {len(vols)} solid(s), "
              f"{len(tris)} triangles, watertight {m.is_watertight}, "
              f"bbox x {xyz[:, 0].min() * 1e3:.0f}..{xyz[:, 0].max() * 1e3:.0f}, "
              f"y {xyz[:, 1].min() * 1e3:.0f}..{xyz[:, 1].max() * 1e3:.0f}, "
              f"z {xyz[:, 2].min() * 1e3:.0f}..{xyz[:, 2].max() * 1e3:.0f} mm ({time.time() - t0:.1f} s)")
    return m


def housing_section(step_or_mesh, z: float = 0.0, **mesh_kwargs) -> List[np.ndarray]:
    """Closed midplane outline(s) (K, 2) [m] of the housing at height ``z``
    (STEP path -> ``housing_mesh(**mesh_kwargs)``, or a ready mesh)."""
    from .electrodes3d import loops_from_mesh
    m = step_or_mesh if not isinstance(step_or_mesh, (str, os.PathLike)) else housing_mesh(step_or_mesh, **mesh_kwargs)
    return loops_from_mesh(m, z=z, name='housing')


# ============================================================================
# The bundle
# ============================================================================
class InflectorModel:
    """Bunch, static E-field and housing of the spiral inflector in one frame.

    Parameters
    ----------
    bunch : path, optional
        openPMD plane-crossing hand-off file (``handoff.py``). Its attributes
        ``PyPATools:efield_file`` / ``PyPATools:housing_file`` (relative to
        the file's folder) locate the other two when not given explicitly.
    efield : path or Field, optional
        Static E-field map (see ``load_inflector_field``).
    housing : path, optional
        STEP file of the housing (``housing_scale`` to metres: 1e-3 for the
        deck's mm exports, 1.0 for py_electrodes' metre files).
    rotation_deg, flip_z :
        Common frame transform applied to all three (rotation about z after
        the files' own frame; ``flip_z`` mirrors deck-frame files through the
        median plane). Files already in the machine frame need neither.
    field_flip_z, housing_flip_z, bunch_flip_z : bool, optional
        Per-file override of ``flip_z`` for mixed inputs (e.g. a deck-frame
        field pickle next to a STEP that the deck already mirrored).
    """

    def __init__(self, bunch=None, efield=None, housing=None, rotation_deg: float = 0.0,
                 flip_z: bool = False, housing_scale: float = 1e-3, housing_mesh_size: float = 2e-3,
                 field_backend: str = 'numba', name: Optional[str] = None, verbose: bool = True,
                 field_flip_z: Optional[bool] = None, housing_flip_z: Optional[bool] = None,
                 bunch_flip_z: Optional[bool] = None):
        self.bunch = str(bunch) if bunch is not None else None
        self.rotation_deg = float(rotation_deg)
        self.flip_z = bool(flip_z)
        self.field_flip_z = bool(flip_z if field_flip_z is None else field_flip_z)
        self.housing_flip_z = bool(flip_z if housing_flip_z is None else housing_flip_z)
        self.bunch_flip_z = bool(flip_z if bunch_flip_z is None else bunch_flip_z)
        self.housing_scale = float(housing_scale)
        self.housing_mesh_size = float(housing_mesh_size)
        self.field_backend = field_backend
        self.verbose = verbose
        self.bunch_meta = None
        if self.bunch is not None:
            self.bunch_meta = load_handoff(self.bunch)['meta']
            folder = os.path.dirname(self.bunch)
            if efield is None and self.bunch_meta.get('efield_file'):
                efield = os.path.join(folder, str(self.bunch_meta['efield_file']))
            if housing is None and self.bunch_meta.get('housing_file'):
                housing = os.path.join(folder, str(self.bunch_meta['housing_file']))
        self._efield_src = efield
        self.housing_path = str(housing) if housing is not None else None
        self.name = name or (os.path.splitext(os.path.basename(self.bunch))[0] if self.bunch
                             else os.path.splitext(os.path.basename(self.housing_path))[0] if self.housing_path
                             else 'inflector')
        self._field = None
        self._field2d = {}
        self._mesh = None
        self._loops = {}

    # ---------------------------------------------------------------- field
    @property
    def has_field(self) -> bool:
        return self._efield_src is not None

    @property
    def field(self) -> Field:
        """3D static E-field in the model frame (lazy)."""
        if self._field is None:
            if self._efield_src is None:
                raise ValueError(f"{self.name}: no E-field given")
            self._field = load_inflector_field(self._efield_src, rotation_deg=self.rotation_deg,
                                               flip_z=self.field_flip_z, backend=self.field_backend,
                                               verbose=self.verbose)
        return self._field

    def midplane_field(self, z: float = 0.0, ez: str = 'zero') -> Field:
        key = (float(z), ez)
        if key not in self._field2d:
            self._field2d[key] = midplane_field(self.field, z=z, ez=ez, backend=self.field_backend,
                                                label=f"{self.name} E-field @ z = {z * 1e3:.1f} mm")
        return self._field2d[key]

    # -------------------------------------------------------------- housing
    @property
    def has_housing(self) -> bool:
        return self.housing_path is not None

    def housing_mesh(self):
        if self._mesh is None:
            if self.housing_path is None:
                raise ValueError(f"{self.name}: no housing given")
            self._mesh = housing_mesh(self.housing_path, scale=self.housing_scale,
                                      rotation_deg=self.rotation_deg, flip_z=self.housing_flip_z,
                                      mesh_size=self.housing_mesh_size, verbose=self.verbose)
        return self._mesh

    def housing_loops(self, z: float = 0.0) -> List[np.ndarray]:
        """Closed outline(s) [m] of the housing metal at height ``z``."""
        key = float(z)
        if key not in self._loops:
            self._loops[key] = housing_section(self.housing_mesh(), z=z)
        return self._loops[key]

    def housing_polygon(self, z: float = 0.0):
        """Outline(s) for ``gap_fields.build_gap_electrodes(housing=...)``."""
        loops = self.housing_loops(z)
        if not loops:
            raise RuntimeError(f"{self.name}: the housing has no section at z = {z * 1e3:.1f} mm")
        return loops

    def obstacle(self, spacing: float = 5e-4, beam_halfwidth: float = 0.0,
                 extent: Optional[float] = None, z: float = 0.0) -> Callable:
        """Midplane metal test of the housing (``electrodes3d.raster_obstacles``)."""
        from .electrodes3d import raster_obstacles
        loops = self.housing_loops(z)
        if extent is None:
            extent = max(float(np.abs(l).max()) for l in loops) + 0.01
        return raster_obstacles(loops, spacing=spacing, extent=extent, beam_halfwidth=beam_halfwidth,
                                verbose=self.verbose)

    def extra_solid(self, potential: float = 0.0, name: Optional[str] = None,
                    fuse: bool = True, crop: bool = True):
        """The housing as an ``electrodes3d.ExtraSolid`` (3D builder).

        ``fuse=False`` keeps the grounded housing a separate body instead of
        putting its B-rep through the OCC fuse of all ground parts (minutes
        instead of tens of minutes); ``crop=False`` also skips the crop
        against the model cylinder. See ``electrodes3d.ExtraSolid``."""
        from .electrodes3d import ExtraSolid
        if self.housing_path is None:
            raise ValueError(f"{self.name}: no housing given")
        return ExtraSolid(self.housing_path, potential=potential, scale=self.housing_scale,
                          rotation_deg=self.rotation_deg, mirror_z=self.housing_flip_z,
                          name=name or f"{self.name}-housing", fuse=fuse, crop=crop)

    def clearance(self, trajectory, z: float = 0.0, skip_deg: float = 0.0) -> dict:
        """Minimum in-plane distance [m] from a trajectory (N, >= 2) to the
        housing metal (negative: inside). ``skip_deg`` ignores the first
        degrees of azimuthal travel (the path inside the housing)."""
        from .gap_fields import _HousingGeom
        xy = np.asarray(trajectory, dtype=float)[:, :2]
        if skip_deg > 0:
            phi = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
            xy = xy[np.abs(phi - phi[0]) > np.radians(skip_deg)]
        geom = _HousingGeom(self.housing_loops(z))
        d = geom.dist(xy)
        i = int(np.argmin(d))
        return {'min_distance_m': float(d[i]), 'at_mm': (xy[i] * 1e3).tolist(),
                'n_inside': int(np.sum(d < 0)), 'n_points': int(len(xy))}

    # ----------------------------------------------------------------- bunch
    def beam(self, **kwargs):
        """``handoff.make_beam_from_handoff`` of the bunch in this frame."""
        if self.bunch is None:
            raise ValueError(f"{self.name}: no bunch file given")
        kwargs.setdefault('verbose', self.verbose)
        return make_beam_from_handoff(self.bunch, rotation_deg=self.rotation_deg, flip_z=self.bunch_flip_z, **kwargs)

    # --------------------------------------------------------------- summary
    def summary(self) -> dict:
        out = {'name': self.name, 'bunch': self.bunch, 'efield': (str(self._efield_src) if not isinstance(self._efield_src, FieldBase) else 'Field'),
               'housing': self.housing_path, 'rotation_deg': self.rotation_deg, 'flip_z': self.flip_z,
               'field_flip_z': self.field_flip_z, 'housing_flip_z': self.housing_flip_z,
               'bunch_flip_z': self.bunch_flip_z, 'housing_scale': self.housing_scale}
        if self.bunch_meta is not None:
            for k in ('mode', 'frame_name', 'handoff_distance_m', 'n_particles', 'rf_frequency_hz'):
                if k in self.bunch_meta:
                    v = self.bunch_meta[k]
                    out['bunch_' + k] = v.tolist() if isinstance(v, np.ndarray) else v
        if self._loops:
            z0 = sorted(self._loops)[0]
            loops = self._loops[z0]
            rr = np.hypot(*np.vstack(loops).T) if loops else np.array([np.nan])
            out['housing_section'] = {'z': z0, 'n_loops': len(loops), 'r_min_mm': float(rr.min() * 1e3),
                                      'r_max_mm': float(rr.max() * 1e3)}
        if self._field is not None:
            g = self._field.grid
            out['field_grid_mm'] = {k: [float(g[k][0] * 1e3), float(g[k][-1] * 1e3), len(g[k])] for k in 'xyz'}
        return out

    def __repr__(self):
        return (f"InflectorModel({self.name}: bunch={'yes' if self.bunch else 'no'}, "
                f"field={'yes' if self.has_field else 'no'}, housing={'yes' if self.has_housing else 'no'}, "
                f"rotation {self.rotation_deg:+.1f} deg, z-mirror field/housing/bunch = "
                f"{self.field_flip_z}/{self.housing_flip_z}/{self.bunch_flip_z})")
