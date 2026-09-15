"""
fields3d.py - 3D field maps for full-3D central-region tracking.

  * ``crop_comsol_3d``      chunked crop of a big COMSOL 3D export (with
                            coordinate columns) to |x|, |y| <= r_max, |z| <=
                            z_max; stamped .npz cache; -> gridded 3D Field.
  * ``load_rf_quarter_map`` the COMSOL RF cavity eigenmode export (regular
                            grid WITHOUT coordinate columns, one quadrant,
                            NaN in metal), cropped and cached.
  * ``fourfold_field``      replicate a first-quadrant map by 90 deg
                            rotations (four identical, in-phase dees) on a
                            full grid; NaN -> 0 with the metal mask kept.
  * ``arc_voltage``         -int E.dl along an arc through a 3D field: the
                            gap voltage, used to scale the RF map to the
                            design's dee voltage.
  * ``RadialBlendField``    inner field inside r0, outer field outside r1,
                            smoothstep between (the BEM / cavity-map seam).
  * ``bem_field3d``         the 3D BEM solution on a grid as a Field, deep
                            metal zeroed by a 3D metal test.
  * ``Raster3D``            bool volume -> ``inside(xyz)`` test (the RF map's
                            metal outside the BEM crop).

Part of: PyCentralRegion module (2026-09-11)
"""
import os
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from PyPATools.field import Field, FieldBase


# ============================================================================
# COMSOL 3D export with coordinate columns (magnet maps): chunked crop + cache
# ============================================================================
def _comsol_header(path, n_lines: int = 9) -> Dict[str, str]:
    out = {}
    with open(path, 'r') as f:
        for _ in range(n_lines):
            line = f.readline()
            if not line.startswith('%'):
                break
            body = line[1:].strip()
            if ':' in body:
                k, v = body.split(':', 1)
                out[k.strip().lower()] = v.strip()
            else:
                out['columns'] = body
    return out


def crop_comsol_3d(path, r_max: float, z_max: Optional[float] = None, cache: Optional[os.PathLike] = None,
                   chunksize: int = 4_000_000, cols: Sequence[int] = (0, 1, 2, 3, 4, 5),
                   verbose: bool = True) -> dict:
    """Crop a COMSOL 3D export (columns x y z Fx Fy Fz in SI, regular grid) to
    |x|, |y| <= r_max, |z| <= z_max [m] without loading the whole file.

    Returns dict(grid={'x','y','z'}, values={'x','y','z'} as (nx, ny, nz)
    arrays, source, n_rows). With ``cache`` (an .npz path) the crop is stored
    stamped with the source file name and reused when the stamp matches.
    """
    path = Path(path)
    if cache is not None:
        cache = Path(cache)
        if cache.exists():
            d = np.load(cache, allow_pickle=False)
            if str(d['source']) == path.name and float(d['r_max']) >= r_max - 1e-9 \
                    and (z_max is None or float(d['z_max']) >= z_max - 1e-9):
                out = {'grid': {k: d['g' + k] for k in 'xyz'}, 'values': {k: d['v' + k] for k in 'xyz'},
                       'source': path.name, 'n_rows': int(d['n_rows']), 'cache': str(cache)}
                if verbose:
                    print(f"[fields3d] {path.name}: crop from cache {cache.name} "
                          f"({out['values']['z'].shape})")
                return out
    import pandas as pd
    t0 = time.time()
    hdr = _comsol_header(path)
    if verbose:
        print(f"[fields3d] cropping {path.name} ({hdr.get('nodes', '?')} nodes) to r <= {r_max * 1e3:.0f} mm"
              + (f", |z| <= {z_max * 1e3:.0f} mm" if z_max is not None else "") + " ...", flush=True)
    keep = []
    n_rows = 0
    reader = pd.read_csv(path, sep=r'\s+', comment='%', header=None, usecols=list(cols), dtype=np.float64,
                         chunksize=chunksize, engine='c')
    for chunk in reader:
        a = chunk.to_numpy()
        n_rows += len(a)
        m = (np.abs(a[:, 0]) <= r_max + 1e-9) & (np.abs(a[:, 1]) <= r_max + 1e-9)
        if z_max is not None:
            m &= np.abs(a[:, 2]) <= z_max + 1e-9
        if np.any(m):
            keep.append(a[m])
        if verbose and n_rows % (10 * chunksize) < chunksize:
            print(f"    {n_rows / 1e6:.0f} M rows read, {sum(len(k) for k in keep) / 1e6:.2f} M kept "
                  f"({time.time() - t0:.0f} s)", flush=True)
    if not keep:
        raise ValueError(f"{path.name}: nothing inside the crop")
    a = np.vstack(keep)
    xs, ys, zs = (np.unique(np.round(a[:, i], 9)) for i in range(3))
    ix = np.searchsorted(xs, np.round(a[:, 0], 9))
    iy = np.searchsorted(ys, np.round(a[:, 1], 9))
    iz = np.searchsorted(zs, np.round(a[:, 2], 9))
    shape = (len(xs), len(ys), len(zs))
    if len(a) != np.prod(shape):
        raise ValueError(f"{path.name}: crop is not a full regular grid ({len(a)} rows for {shape})")
    values = {}
    for k, c in zip('xyz', (3, 4, 5)):
        v = np.full(shape, np.nan)
        v[ix, iy, iz] = a[:, c]
        values[k] = v
    if verbose:
        print(f"[fields3d] {path.name}: {n_rows / 1e6:.1f} M rows -> grid {shape}, "
              f"x {xs[0] * 1e3:.0f}..{xs[-1] * 1e3:.0f}, z {zs[0] * 1e3:.0f}..{zs[-1] * 1e3:.0f} mm "
              f"in {time.time() - t0:.0f} s")
    out = {'grid': {'x': xs, 'y': ys, 'z': zs}, 'values': values, 'source': path.name, 'n_rows': n_rows}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, source=path.name, r_max=r_max, z_max=(np.inf if z_max is None else z_max), n_rows=n_rows,
                 gx=xs, gy=ys, gz=zs, vx=values['x'], vy=values['y'], vz=values['z'])
        out['cache'] = str(cache)
    return out


def field_from_crop(crop: dict, scaling: float = 1.0, label: str = '3D map', backend: str = 'numba') -> Field:
    """Gridded 3D Field (NaN -> 0) from a ``crop_comsol_3d`` result."""
    vals = {k: np.nan_to_num(np.asarray(v, dtype=float)) for k, v in crop['values'].items()}
    f = Field.from_arrays(grid={k: np.asarray(crop['grid'][k], dtype=float) for k in 'xyz'}, values=vals,
                          dim=3, units='m', interpolator_backend=backend, label=label, scaling=scaling)
    return f


# ============================================================================
# The COMSOL RF map (no coordinate columns, one quadrant, NaN in metal)
# ============================================================================
def load_rf_quarter_map(path, r_max: float, cache: Optional[os.PathLike] = None, n_xy: int = 1264, n_z: int = 31,
                        spacing: float = 0.002, z0: float = -0.030, verbose: bool = True) -> dict:
    """The IBA cavity eigenmode export: 1264 x 1264 x 31 nodes, x fastest then
    y then z, x = y = 0..2526 mm, z = -30..30 mm at 2 mm, columns Ex Ey Ez
    (arbitrary amplitude), NaN in metal. Cropped to x, y <= r_max [m]
    (arrays (nx, ny, nz)) and cached as npz when ``cache`` is given."""
    path = Path(path)
    nx = int(np.floor(r_max / spacing + 1e-9)) + 1
    if cache is not None:
        cache = Path(cache)
        if cache.exists():
            d = np.load(cache, allow_pickle=False)
            if str(d['source']) == path.name and d['gx'].shape[0] >= nx:
                out = {'grid': {k: d['g' + k] for k in 'xyz'}, 'values': {k: d['v' + k] for k in 'xyz'},
                       'source': path.name, 'cache': str(cache)}
                if verbose:
                    print(f"[fields3d] RF map from cache {cache.name} ({out['values']['z'].shape})")
                return out
    import pandas as pd
    t0 = time.time()
    if verbose:
        print(f"[fields3d] reading {path.name} ({path.stat().st_size / 1e9:.1f} GB) ...", flush=True)
    a = pd.read_csv(path, sep=r'\s+', comment='%', header=None, dtype=np.float64, engine='c',
                    na_values=['NaN']).to_numpy()
    if a.shape != (n_xy * n_xy * n_z, 3):
        raise ValueError(f"{path.name}: {a.shape} rows/cols, expected {n_xy * n_xy * n_z} x 3")
    a = a.reshape(n_z, n_xy, n_xy, 3)              # [iz, iy, ix, comp]
    sub = a[:, :nx, :nx, :]
    xs = spacing * np.arange(nx)
    zs = z0 + spacing * np.arange(n_z)
    values = {k: np.ascontiguousarray(np.transpose(sub[:, :, :, i], (2, 1, 0))) for i, k in enumerate('xyz')}
    del a
    if verbose:
        n_nan = int(np.isnan(values['x']).sum())
        print(f"[fields3d] RF map cropped to {values['x'].shape} (x, y <= {xs[-1] * 1e3:.0f} mm; "
              f"{100 * n_nan / values['x'].size:.1f} % NaN = metal) in {time.time() - t0:.0f} s")
    out = {'grid': {'x': xs, 'y': xs.copy(), 'z': zs}, 'values': values, 'source': path.name}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, source=path.name, gx=xs, gy=xs, gz=zs, vx=values['x'], vy=values['y'], vz=values['z'])
        out['cache'] = str(cache)
    return out


def fourfold_field(quarter: dict, scaling: float = 1.0, label: str = 'RF cavity map (4-fold)',
                   backend: str = 'numba', r_max: Optional[float] = None) -> Tuple[Field, 'Raster3D']:
    """Replicate a first-quadrant map (x, y >= 0, x[0] = y[0] = 0, same
    spacing on both axes) by 90 deg rotations onto the full square: four
    identical dees in phase. Returns (Field with NaN -> 0, metal Raster3D of
    the NaN nodes)."""
    gx, gy, gz = (np.asarray(quarter['grid'][k], dtype=float) for k in 'xyz')
    if abs(gx[0]) > 1e-12 or abs(gy[0]) > 1e-12 or len(gx) != len(gy):
        raise ValueError("fourfold_field needs a first-quadrant map starting at x = y = 0 with equal axes")
    if r_max is not None:
        n = int(np.searchsorted(gx, r_max + 1e-9))
        gx, gy = gx[:n], gy[:n]
    n = len(gx)
    ex, ey, ez = (np.asarray(quarter['values'][k], dtype=float)[:n, :n, :] for k in 'xyz')
    nz = len(gz)
    full = np.full((3, 2 * n - 1, 2 * n - 1, nz), np.nan)
    c = n - 1                                    # index of x = 0 on the full axis
    # quadrant 0: (x, y) as is
    full[0, c:, c:, :] = ex
    full[1, c:, c:, :] = ey
    full[2, c:, c:, :] = ez
    # rotation by +90 deg: (x, y) -> (-y, x); E -> (-Ey, Ex, Ez). Node (i, j) of the
    # quarter lands at full index (c - j, c + i).
    e90x, e90y = -ey, ex
    full[0, c::-1, c:, :] = np.transpose(e90x, (1, 0, 2))
    full[1, c::-1, c:, :] = np.transpose(e90y, (1, 0, 2))
    full[2, c::-1, c:, :] = np.transpose(ez, (1, 0, 2))
    # 180 deg: (x, y) -> (-x, -y); E -> (-Ex, -Ey, Ez)
    full[0, c::-1, c::-1, :] = -ex
    full[1, c::-1, c::-1, :] = -ey
    full[2, c::-1, c::-1, :] = ez
    # 270 deg: (x, y) -> (y, -x); E -> (Ey, -Ex, Ez); node (i, j) -> (c + j, c - i)
    full[0, c:, c::-1, :] = np.transpose(ey, (1, 0, 2))
    full[1, c:, c::-1, :] = np.transpose(-ex, (1, 0, 2))
    full[2, c:, c::-1, :] = np.transpose(ez, (1, 0, 2))
    axis = np.concatenate([-gx[:0:-1], gx])
    metal = np.isnan(full[0])
    vals = {k: np.nan_to_num(full[i]) for i, k in enumerate('xyz')}
    f = Field.from_arrays(grid={'x': axis, 'y': axis.copy(), 'z': gz}, values=vals, dim=3, units='m',
                          interpolator_backend=backend, label=label, scaling=scaling)
    f._metadata = {'source': quarter.get('source'), 'fourfold': True, 'scaling': float(scaling)}
    return f, Raster3D(metal, (axis, axis.copy(), gz), name='RF map metal (NaN nodes)')


def arc_voltage(field: Callable, r: float, az0_deg: float, az1_deg: float, z: float = 0.0, n: int = 3000) -> float:
    """-int E.dl [V] along the arc of radius r [m] from az0 to az1 [deg] at
    height z (positive = the potential RISES from az0 to az1)."""
    th = np.radians(np.linspace(az0_deg, az1_deg, n))
    pts = np.column_stack([r * np.cos(th), r * np.sin(th), np.full(n, float(z))])
    e = np.asarray(field(pts), dtype=float)
    et = -e[:, 0] * np.sin(th) + e[:, 1] * np.cos(th)
    et = np.where(np.isfinite(et), et, 0.0)
    return -float(np.sum(0.5 * (et[1:] + et[:-1]) * np.diff(th))) * r


# ============================================================================
# 3D metal raster and the radial blend
# ============================================================================
class Raster3D:
    """``inside(xyz) -> bool`` from a boolean volume on a regular grid
    (nearest node; off-grid = vacuum). (N, 2) input is taken at z = 0.
    ``ndim = 3`` tells ``tracking.MetalTerminator`` to pass z."""

    ndim = 3

    def __init__(self, grid: np.ndarray, axes: Sequence[np.ndarray], name: str = 'raster3d'):
        self.grid = np.asarray(grid, dtype=bool)
        self.axes = tuple(np.asarray(a, dtype=float) for a in axes)
        if self.grid.shape != tuple(len(a) for a in self.axes):
            raise ValueError("Raster3D: grid shape does not match the axes")
        self.origin = np.array([a[0] for a in self.axes])
        self.h = np.array([(a[-1] - a[0]) / max(len(a) - 1, 1) for a in self.axes])
        self.name = name

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        if p.shape[1] == 2:
            p = np.column_stack([p, np.zeros(len(p))])
        idx = np.rint((p[:, :3] - self.origin) / self.h).astype(int)
        ok = np.all((idx >= 0) & (idx < np.asarray(self.grid.shape)), axis=1)
        out = np.zeros(len(p), dtype=bool)
        out[ok] = self.grid[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        return out

    def fraction(self) -> float:
        return float(self.grid.mean())


class RadialBlendField(FieldBase):
    """``inner`` inside r0, ``outer`` outside r1, smoothstep-blended between
    (radius in the x-y plane). Each part is evaluated only where its weight
    is non-zero, so the cost is one interpolation per point plus the seam."""

    def __init__(self, inner: FieldBase, outer: FieldBase, r0: float, r1: float, label: str = 'radial blend'):
        if not r1 > r0 >= 0.0:
            raise ValueError("need 0 <= r0 < r1")
        self.inner, self.outer = inner, outer
        self.r0, self.r1 = float(r0), float(r1)
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    def weights(self, pts: np.ndarray) -> np.ndarray:
        rr = np.hypot(pts[:, 0], pts[:, 1])
        t = np.clip((rr - self.r0) / (self.r1 - self.r0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)               # 0 inside r0, 1 outside r1

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        w = self.weights(pts)
        out = np.zeros((len(pts), 3))
        mi = w < 1.0
        mo = w > 0.0
        if np.any(mi):
            out[mi] += (1.0 - w[mi])[:, None] * np.asarray(self.inner(pts[mi]), dtype=float)
        if np.any(mo):
            out[mo] += w[mo][:, None] * np.asarray(self.outer(pts[mo]), dtype=float)
        return out

    def __str__(self):
        return (f"RadialBlendField(inner < {self.r0 * 1e3:.0f} mm: {getattr(self.inner, 'label', self.inner)}; "
                f"outer > {self.r1 * 1e3:.0f} mm: {getattr(self.outer, 'label', self.outer)})")


# ============================================================================
# 3D BEM solution -> gridded Field
# ============================================================================
def bem_field3d(sol, xs, ys, zs, metal: Optional[Callable] = None, chunk: int = 4000, erode: int = 2,
                backend: str = 'numba', label: str = 'BEM 3D gap field', r_max: Optional[float] = None,
                verbose: bool = True) -> Field:
    """E = -grad(phi) of a ``GapFieldSolution`` on the (xs, ys, zs) grid [m]
    as a dim-3 Field. ``metal(xyz) -> bool`` (e.g. ``electrodes3d.stacked_obstacles``)
    zeroes the field ``erode`` cells deep inside metal (the wall jump layer is
    kept so particles feel the field up to the surface); trimesh ``contains``
    on millions of points is avoided.

    ``r_max`` [m] evaluates the potential only inside that cylinder and leaves the rest
    of the box at zero: the corners of a square grid are 21 % of its points and a radial
    blend (``RadialBlendField``) never reads them. Two extra cells are evaluated beyond
    ``r_max`` so the central-difference gradient inside it is still exact."""
    xs, ys, zs = (np.asarray(a, dtype=float) for a in (xs, ys, zs))
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    t0 = time.time()
    if r_max is None:
        phi = sol.potential(pts, chunk=chunk, verbose=verbose).reshape(gx.shape)
        n_eval = len(pts)
    else:
        keep = np.hypot(pts[:, 0], pts[:, 1]) <= float(r_max) + 2.0 * float(np.max(np.abs(np.diff(xs))))
        flat = np.zeros(len(pts))
        flat[keep] = sol.potential(pts[keep], chunk=chunk, verbose=verbose)
        phi = flat.reshape(gx.shape)
        n_eval = int(keep.sum())
    if verbose:
        print(f"[fields3d] BEM potential on {gx.shape} ({n_eval / 1e6:.2f} of "
              f"{pts.shape[0] / 1e6:.2f} M points) in {time.time() - t0:.0f} s")
    ex = -np.gradient(phi, xs, axis=0)
    ey = -np.gradient(phi, ys, axis=1)
    ez = -np.gradient(phi, zs, axis=2) if len(zs) > 1 else np.zeros_like(phi)
    if metal is not None:
        from scipy.ndimage import binary_erosion
        inside = np.asarray(metal(pts), dtype=bool).reshape(gx.shape)
        deep = binary_erosion(inside, iterations=erode) if erode > 0 else inside
        ex[deep] = 0.0
        ey[deep] = 0.0
        ez[deep] = 0.0
    f = Field.from_arrays(grid={'x': xs, 'y': ys, 'z': zs}, values={'x': ex, 'y': ey, 'z': ez}, dim=3, units='m',
                          interpolator_backend=backend, label=label)
    f._metadata = {'phi_min': float(phi.min()), 'phi_max': float(phi.max()), 'n_points': int(pts.shape[0])}
    return f


def field_to_arrays(f: Field) -> dict:
    """Pickle-friendly (grid, values, scaling, label) of a gridded Field."""
    return {'grid': {k: np.asarray(v) for k, v in f._grid.items()},
            'values': {k: np.asarray(v) for k, v in f._values.items()},
            'scaling': float(getattr(f, '_scaling', 1.0)), 'label': f.label, 'dim': int(f.dim),
            'metadata': dict(getattr(f, '_metadata', {}) or {})}


def field_from_arrays(d: dict, backend: str = 'numba') -> Field:
    f = Field.from_arrays(grid=d['grid'], values=d['values'], dim=d.get('dim', 3), units='m',
                          interpolator_backend=backend, label=d.get('label', 'field'), scaling=d.get('scaling', 1.0))
    f._metadata = dict(d.get('metadata', {}))
    return f
