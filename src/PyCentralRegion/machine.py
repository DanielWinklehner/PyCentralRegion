"""The machine a run tracks in, built from a run configuration.

Everything here is an input of the stages, nothing is computed from the design's fields:

* the seed: the gap-geometry optimizer's result (``AcceleratedOrbitFinder.OptimizedOrbit`` pickle) with the gap
  lines, the reference orbit, the bunch phase and the revolution frequency;
* the design: species, a zero static E-field and the dee system of [machine] with the seed's gap geometry, its gap
  lines matched to the seed orbit (inert segments straightened, segment nodes snapped between turns) exactly as the
  electrode builder matches them, so the tracked gaps and the metal describe the same cavities;
* the spiral inflector of [inflector] / [paths] (hand-off bunch, static E-field, housing), turned about the machine
  axis by ``inflector.rotation_deg``;
* the 3D magnet map of [magnet] / [paths], cropped (and cached under ``run.cache_dir``), with Bx = By = 0 on the
  median plane;
* the reference particle on the hand-off plane.

No environment variable is read. Scope: four dees with radial gap lines (``machine.n_dees == 4`` and
``machine.dee_shape == 'radial'``), the layout the electrode builder and the 3D tracker have been validated on;
``check_supported`` raises NotImplementedError for any other layout.
"""
import pickle
import time
import warnings
from pathlib import Path

import numpy as np

from .config import ConfigError, require, section


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_supported(mc):
    """Raise NotImplementedError unless the [machine] layout is one the package implements (4 radial dees)."""
    if mc.n_dees != 4 or mc.dee_shape != 'radial':
        raise NotImplementedError(
            f"machine.n_dees = {mc.n_dees}, machine.dee_shape = {mc.dee_shape!r}: only four dees with radial gap "
            f"lines are implemented")
    if len(mc.dee_center_angles_deg) != mc.n_dees:
        raise ConfigError(f"machine.dee_center_angles_deg has {len(mc.dee_center_angles_deg)} angles for "
                          f"{mc.n_dees} dees")


def load_seed(path):
    """The seed optimization's result (OptimizedOrbit pickle)."""
    with open(require(path, 'paths.seed'), 'rb') as fh:
        return pickle.load(fh)


def seed_orbit(seed):
    """The seed's reference orbit (N, 3) [m]; an error if the pickle carries none."""
    traj = np.asarray(getattr(seed, 'trajectory_reference', np.zeros((0, 3))), dtype=float)
    if len(traj) < 3:
        raise ConfigError("the seed pickle carries no reference orbit: its gap lines cannot be matched to it")
    return traj


# ---------------------------------------------------------------------------------------------------------------------
# dee system
def make_dee_system(mc, f_rev_hz, n_segments=1, segment_radii=(0.15,)):
    """The dee system of [machine] at revolution frequency ``f_rev_hz``, every gap with ``n_segments`` straight
    variable segments (outer radii ``segment_radii``) - the template a seed geometry is applied to."""
    from .rf_cavity import create_dee_system
    check_supported(mc)
    return create_dee_system(
        r_min=mc.r_min_m, r_max=mc.r_max_cavity_m, center_angles=list(mc.dee_center_angles_deg),
        opening_angle=mc.dee_opening_deg, voltage=mc.design_voltage_kv * 1e3, frequency=f_rev_hz,
        phases=[0.0] * mc.n_dees, harmonic=mc.harmonic, gap_width=mc.gap_width_m,
        gap_angle_deg=mc.gap_angle_deg, gap_width_min=mc.gap_width_min_m,
        n_variable_segments=int(n_segments), segment_angles=[0.0] * int(n_segments),
        segment_radii=list(segment_radii))


def apply_seed_geometry(dee_system, geom):
    """Put a seed's ``optimal_geometry`` (opening angle, per-gap segment angles / radii / rotations) on the gaps."""
    if geom.get('opening_angle_deg') is not None:
        dee_system.apply_opening_angle(float(geom['opening_angle_deg']))
    rots = geom.get('segment_rotations_per_gap')
    for g, cav in enumerate(dee_system.gaps):
        cav.update_geometry(segment_angles=[float(a) for a in geom['segment_angles_per_gap'][g]],
                            segment_radii=[float(r) for r in geom['segment_radii_per_gap'][g]],
                            segment_rotations=([float(r) for r in rots[g]] if rots else None))
    return dee_system


def seed_dee_system(mc, seed):
    """The dee system of [machine] with the seed's gap geometry, matched to the seed orbit (``match_gaps_to_orbit``).
    The seed's opening angle, when it has one, wins over ``machine.dee_opening_deg`` (a warning says so)."""
    from .rf_cavity import neutralise_inert_segments, snap_nodes_between_turns
    geom = seed.metadata['optimal_geometry']
    if len(geom['segment_angles_per_gap']) != 2 * mc.n_dees:
        raise ConfigError(f"the seed has {len(geom['segment_angles_per_gap'])} gaps, [machine] {2 * mc.n_dees}")
    op = geom.get('opening_angle_deg')
    if op is not None and abs(float(op) - mc.dee_opening_deg) > 1e-9:
        warnings.warn(f"machine.dee_opening_deg = {mc.dee_opening_deg} but the seed's opening is {float(op)} deg: "
                      f"using the seed's", stacklevel=2)
    radii0 = [float(r) for r in geom['segment_radii_per_gap'][0]]
    ds = make_dee_system(mc, float(seed.rf_frequency_mhz) * 1e6, n_segments=len(radii0), segment_radii=radii0)
    apply_seed_geometry(ds, geom)
    traj = seed_orbit(seed)
    neutralise_inert_segments(ds.gaps, traj, verbose=False)
    snap_nodes_between_turns(ds.gaps, traj, verbose=False)
    return ds


# ---------------------------------------------------------------------------------------------------------------------
# spiral inflector and reference particle
def refined_efield(src, factor, z_max, verbose=False):
    """The inflector E-map on a grid ``factor`` x finer, cropped to |z| <= ``z_max`` [m] (to the enclosing nodes): the
    same trilinear field, sampled. A rotated map is resampled on its own grid, which blurs the steep exit fringe by up
    to a cell; on the refined grid that resampling error shrinks as the cell squared (x4 on a 2.5 mm map: ~2 eV rms of
    the fringe kick)."""
    from PyPATools.field import Field
    from .inflector import load_inflector_field
    f = load_inflector_field(src, rotation_deg=0.0, verbose=False)
    g = {k: np.asarray(f.grid[k], dtype=float) for k in 'xyz'}
    k0 = max(int(np.searchsorted(g['z'], -z_max, side='left')) - 1, 0)
    k1 = int(np.searchsorted(g['z'], z_max, side='right'))
    g['z'] = g['z'][k0:max(k1 + 1, k0 + 2)]
    fine = {k: np.linspace(v[0], v[-1], (len(v) - 1) * max(int(factor), 1) + 1) for k, v in g.items()}
    X, Y, Z = np.meshgrid(fine['x'], fine['y'], fine['z'], indexing='ij')
    e = f(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
    out = Field.from_arrays(grid=fine, values={c: e[:, i].reshape(X.shape) for i, c in enumerate('xyz')}, dim=3,
                            units='m', interpolator_backend='numba',
                            label=f"{getattr(f, 'label', 'inflector E')} (x{factor} refined, z <= {z_max * 1e3:.0f} mm)")
    if verbose:
        say(f"[inflector] E-field refined x{factor} for the rotation: {X.shape[0]}x{X.shape[1]}x{X.shape[2]} nodes, "
            f"z {fine['z'][0] * 1e3:.0f}..{fine['z'][-1] * 1e3:.0f} mm")
    return out


def shifted_efield(path, dz):
    """The inflector E-map translated by ``dz`` [m] along z (a rigid shift of the whole assembly); the path itself
    when there is no shift."""
    from PyPATools.field import Field
    if abs(dz) < 1e-12:
        return str(path)
    f = Field.from_file(str(path))
    grid = {k: np.asarray(v, dtype=float) for k, v in f.grid.items()}
    grid['z'] = grid['z'] + dz
    return Field.from_arrays(grid=grid, values={k: np.asarray(v) for k, v in f.grid_values.items()}, dim=3,
                             units='m', interpolator_backend='numba',
                             label=f"{getattr(f, 'label', 'inflector E')} (z {dz * 1e3:+.3f} mm)")


def make_inflector(cfg, verbose=False):
    """The spiral inflector (``inflector.InflectorModel``) of [inflector] / [paths]: hand-off bunch, static E-field
    (refined before a rotation), housing solid."""
    from .inflector import InflectorModel
    ic, pc = section(cfg, 'inflector'), section(cfg, 'paths')
    handoff = require(pc.handoff, 'paths.handoff')
    efield = shifted_efield(require(pc.inflector_efield, 'paths.inflector_efield'), ic.zshift_m)
    if ic.rotation_deg and ic.refine > 1:
        efield = refined_efield(efield, ic.refine, ic.refine_z_max_m, verbose=verbose)
    return InflectorModel(bunch=handoff, efield=efield, housing=require(pc.housing_step, 'paths.housing_step'),
                          rotation_deg=ic.rotation_deg, flip_z=ic.flip_z, housing_scale=ic.housing_scale,
                          name=Path(handoff).stem, verbose=verbose)


def rotate_z(v, angle_deg):
    """``v`` (x, y[, z]) turned counter-clockwise about the machine axis."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    v = np.asarray(v, dtype=float).copy()
    v[0], v[1] = c * v[0] - s * v[1], s * v[0] + c * v[1]
    return v


def reference_beam(species, handoff_path, rotation_deg):
    """The design particle at the hand-off plane: on the plane origin, moving along the plane normal at the design
    exit speed (v_z included), turned with the inflector. Returns (beam, hand-off meta)."""
    from .accelerated_orbit_finder import make_beam_from_state
    from .handoff import load_handoff
    meta = load_handoff(handoff_path)['meta']
    x0 = rotate_z(meta['plane_origin_m'], rotation_deg)
    n = rotate_z(meta['plane_normal'], rotation_deg)
    n /= np.linalg.norm(n)
    speed = float(np.linalg.norm(np.asarray(meta['design_exit_velocity_mps'], dtype=float)))
    return make_beam_from_state(species, x0, n * speed), meta


# ---------------------------------------------------------------------------------------------------------------------
# magnet
def flatten_centre(crop, r_flat, verbose=True):
    """What-if magnet: flatten the azimuthal mean of the midplane B_z inside ``r_flat`` [m] to its value there, with a
    Maxwell-consistent axisymmetric correction: dB_z(r, z) = f - z^2/2 (f'' + f'/r), dB_r = z f' (median-plane
    expansion of a curl- and divergence-free field), f(r) = <B>(r_flat) - <B>(r) tapered smoothly to 0 at r_flat.
    Not a magnet design: the upper bound of what a central-field fix could buy."""
    from scipy.interpolate import RegularGridInterpolator
    gx, gy, gz = (np.asarray(crop['grid'][k], dtype=float) for k in 'xyz')
    vx, vy, vz = (np.array(crop['values'][k], dtype=float, copy=True) for k in 'xyz')
    k0 = int(np.argmin(np.abs(gz)))
    th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    rs = np.arange(0.0, r_flat + 0.0201, 0.002)

    def mean_b(arr2d):
        g = RegularGridInterpolator((gx, gy), arr2d)
        return np.array([g(np.column_stack([r * np.cos(th), r * np.sin(th)])).mean() if r > 0
                         else float(g([[0.0, 0.0]])[0]) for r in rs])
    bmean = mean_b(np.nan_to_num(vz[:, :, k0]))
    b_ref = float(np.interp(r_flat, rs, bmean))
    f = (b_ref - bmean) * np.clip(1.0 - (rs / r_flat) ** 6, 0.0, 1.0)
    f[rs > r_flat] = 0.0
    fp = np.gradient(f, rs)
    fpp = np.gradient(fp, rs)
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing='ij')
    R = np.hypot(X, Y)
    Rs = np.maximum(R, 1e-9)
    fr = np.interp(R, rs, f, right=0.0)
    fpr = np.interp(R, rs, fp, right=0.0)
    fppr = np.interp(R, rs, fpp, right=0.0)
    vz += fr - 0.5 * Z ** 2 * (fppr + fpr / Rs)
    dbr = Z * fpr
    vx += dbr * X / Rs
    vy += dbr * Y / Rs
    if verbose:
        bm2 = mean_b(vz[:, :, k0])
        say("[B] flat-centre what-if: <B_z> before -> after [T] at r = "
            + ", ".join(f"{r * 1e3:.0f}: {bb:.3f}->{aa:.3f}" for r, bb, aa in zip(rs[::10], bmean[::10], bm2[::10])))
    return {'grid': {'x': gx, 'y': gy, 'z': gz}, 'values': {'x': vx, 'y': vy, 'z': vz},
            'source': crop.get('source'), 'n_rows': crop.get('n_rows')}


def magnet_field_3d(cfg, verbose=True):
    """The 3D magnet map of [paths] / [magnet] as a Field: cropped to ``magnet.crop_r_m`` (the crop is cached under
    ``run.cache_dir/<map stem>/``), Bx = By = 0 on the z = 0 plane (a half model's boundary residual otherwise),
    optionally flattened in the centre, plus the first-harmonic trim of [btrim], scaled by
    ``magnet.sign * magnet.scale``. With ``paths.bfield_2d`` the crop's midplane is compared with that 2D map
    (warning above 2 %). Returns (field, check)."""
    from PyPATools.field import Field
    from .btrim import add_trim, resolved
    from .fields3d import crop_comsol_3d, field_from_crop
    mg, pc, rc = section(cfg, 'magnet'), section(cfg, 'paths'), section(cfg, 'run')
    bmap = Path(require(pc.bfield_3d, 'paths.bfield_3d'))
    cache = (Path(rc.cache_dir) / bmap.stem / f'crop3d_r{round(mg.crop_r_m * 1e3)}.npz') if rc.cache_dir else None
    crop = crop_comsol_3d(bmap, r_max=mg.crop_r_m, cache=cache, verbose=verbose)
    iz0 = int(np.argmin(np.abs(np.asarray(crop['grid']['z']))))
    if abs(float(np.asarray(crop['grid']['z'])[iz0])) > 1e-9:
        raise ConfigError(f"{bmap.name}: the 3D map has no z = 0 plane")
    bt = float(np.max(np.hypot(crop['values']['x'][:, :, iz0], crop['values']['y'][:, :, iz0])))
    crop['values']['x'][:, :, iz0] = 0.0
    crop['values']['y'][:, :, iz0] = 0.0
    if verbose:
        say(f"[B] z = 0 plane: Bx, By zeroed (max in-plane residual was {1e3 * bt:.2f} mT)")
    if mg.flatten_r_m > 0:
        crop = flatten_centre(crop, mg.flatten_r_m, verbose=verbose)
    scale = float(mg.sign) * float(mg.scale)
    bt = section(cfg, 'btrim')
    add_trim(crop, bt, scaling=scale, verbose=verbose)
    trim_t = resolved(bt).amplitude_t
    b3 = field_from_crop(crop, scaling=scale, label=f'3D magnet map ({bmap.stem}, Bx,By = 0 at z = 0)'
                         + (f' + flat centre to {mg.flatten_r_m * 1e3:.0f} mm' if mg.flatten_r_m > 0 else '')
                         + (f' + first-harmonic trim {trim_t * 1e4:.2f} G' if trim_t else ''))
    check = {}
    if pc.bfield_2d:
        b2 = Field.from_file(str(pc.bfield_2d), interpolator_backend='numba', scaling=scale)
        th = np.linspace(0, 2 * np.pi, 73)
        for r in (frac * mg.crop_r_m for frac in (0.125, 0.25, 0.5, 0.75, 0.875)):
            pts = np.column_stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)])
            bz2, bz3 = b2(pts)[:, 2], b3(pts)[:, 2]
            check[f'{r * 1e3:.0f}mm'] = {'bz2_mean': float(bz2.mean()), 'bz3_mean': float(bz3.mean()),
                                        'max_rel_diff': float(np.max(np.abs(bz3 - bz2) / np.abs(bz2).max()))}
        worst = max(v['max_rel_diff'] for v in check.values())
        if verbose:
            say(f"[B] 3D map {crop['values']['z'].shape}; midplane vs {Path(pc.bfield_2d).name}: worst rel diff "
                f"{worst:.2e}; B(0,0,0) 3D {b3(np.zeros((1, 3)))[0, 2]:+.5f} T, 2D {b2(np.zeros((1, 3)))[0, 2]:+.5f} T")
        if worst > 0.02:
            warnings.warn(f"the 3D magnet map differs from the 2D map at the midplane by up to {100 * worst:.1f} %")
    return b3, check


# ---------------------------------------------------------------------------------------------------------------------
# the design
def make_design(cfg, seed, bfield=None):
    """``CentralRegion`` with the species, a zero static E-field, the magnet field ``bfield`` (if given) and the seed's
    dee system; returns (design, dee_system)."""
    from PyPATools.field import Field
    from .central_region import CentralRegion
    mc = section(cfg, 'machine')
    check_supported(mc)
    design = CentralRegion(name=section(cfg, 'run').name or 'central_region', dimensionality='3D')
    design.set_species(mc.species)
    design.set_electric_field(Field.zero())
    if bfield is not None:
        design.set_magnetic_field(bfield)
    ds = seed_dee_system(mc, seed)
    design.clear_rf_cavities()
    for cav in ds.gaps:
        design.add_rf_cavity(cav)
    return design, ds
