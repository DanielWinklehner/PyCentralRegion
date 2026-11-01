"""The `fields` and `track` stages: the 3D BEM field of an electrode model, and 3D bunch tracking through it.

fields   paths.model3d (gap_fields.ElectrodeModel) -> 3D BEM solve -> E-field grid + metal raster -> paths.fields
track    paths.fields + the 3D magnet map + the spiral-inflector bunch -> reference particle per RF phase, phase scan,
         final run with space charge, probes and particle tracks -> run.out_dir: track.json (summary),
         track_data.pkl (arrays), probe_<angle>deg.npz (one per probe), config.json (the resolved configuration)

The BEM field is the whole accelerating field: every dee at the model's potential times ``track.volt_scale``,
modulated in time with the RF of the dee system and the bunch phase of the run; no RF-cavity map is blended in. The
metal test is the model's stacked-slice raster (or its triangle mesh with ``track.obstacles = 'mesh'``), plus the
optional full-height posts and boxes of [track] and the radial slit of [slit] (obstacle tests only: the cached field is
unchanged). The slit is a raster of its own, finer than the model's, so the fast path carries it as its second one.

The field cache and the outputs keep the layout of the design deck this package grew out of (its
``bunch_fields3d<tag>.pkl`` and ``bunch_simulation3d<tag>.json`` / ``_data.pkl``), so a deck cache can be tracked here
and ``analysis`` reads both. Stages reuse an existing output and never overwrite one.
"""
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from .config import ConfigError, require, save_resolved, section, to_dict
from .machine import (check_supported, load_seed, magnet_field_3d, make_design, make_inflector, reference_beam, say,
                      seed_orbit)

TRACK_IDS_SEED = 12345        # sampling seed of the particles whose paths are kept
SLIT_H_M = 1e-4               # node spacing of the slit raster [m]


def kinetic_energy_mev(v, mass_mev):
    beta2 = np.sum(v * v, axis=1) / 299792458.0 ** 2
    return (1.0 / np.sqrt(1.0 - beta2) - 1.0) * mass_mev


def _json_default(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def _write_atomic(path, write):
    """Write ``path`` through a temporary file, so a killed run never leaves a truncated output to be reused."""
    tmp = Path(str(path) + '.part')
    write(tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------------------------
# fields
def load_model3d(path):
    """The 3D electrode model (pickled ``ElectrodeModel`` with params['solids'] and params['layout'])."""
    with open(require(path, 'paths.model3d'), 'rb') as fh:
        model3d = pickle.load(fh)
    for k in ('solids', 'layout'):
        if k not in model3d.params:
            raise ConfigError(f"{path}: the model has no params[{k!r}]")
    return model3d


def metal_raster(model3d, fc, verbose=True):
    """The model's metal on z slices (``electrodes3d.stacked_obstacles``): the tracker's obstacle test."""
    from .electrodes3d import stacked_obstacles
    zs = np.arange(-fc.stack_z_half_m, fc.stack_z_half_m + 0.5 * fc.stack_dz_m, fc.stack_dz_m)
    return stacked_obstacles(model3d, zs, spacing=fc.stack_dxy_m, extent=model3d.extent() + fc.margin_m,
                             verbose=verbose)


def fields_stage(cfg, verbose=True):
    """`pcr fields`: solve the 3D BEM of paths.model3d, sample its E-field on the grid of [fields] (out to the model's
    r_cut + margin), rasterise its metal, and write the cache paths.fields. An existing cache is reused, never
    overwritten. Returns the cache path."""
    from .electrodes3d import check_trajectory_clearance, midplane_obstacles
    from .fields3d import bem_field3d, field_to_arrays
    from .gap_fields import solve_gap_field
    pc, fc, mc = section(cfg, 'paths'), section(cfg, 'fields'), section(cfg, 'machine')
    check_supported(mc)
    out = Path(require(pc.fields, 'paths.fields'))
    if out.exists():
        say(f"[fields] {out} exists: reused (point paths.fields elsewhere to recompute)")
        return out
    t0 = time.time()
    model3d = load_model3d(pc.model3d)
    say(f"[fields] {model3d.n_elements} elements from {Path(pc.model3d).name} ({model3d.params.get('source', '?')})")
    if model3d.n_elements > fc.max_elements:
        raise ConfigError(f"3D model too large ({model3d.n_elements} > fields.max_elements = {fc.max_elements})")
    traj = seed_orbit(load_seed(pc.seed))
    stack = metal_raster(model3d, fc, verbose=verbose)
    say(f"[fields] stacked metal in {time.time() - t0:.0f} s")
    mid = midplane_obstacles(model3d, extent=mc.r_max_m)
    clr = check_trajectory_clearance(mid, traj, raise_on_contact=False, label='seed orbit vs 3D model (z = 0)')
    say(f"[fields] seed orbit vs metal at z = 0: {clr['n_contact']} of {clr['n_points']} points")

    t1 = time.time()
    sol3d = solve_gap_field(model3d, tol=fc.tol, solver=fc.solver, gpu_fit_safety=fc.gpu_fit_safety,
                            verbose=verbose)
    say(f"[fields] solve {time.time() - t1:.0f} s ({sol3d.n_iterations} it, {sol3d.solver}/{sol3d.device_interface})")
    info = {'n_elements_3d': model3d.n_elements, 'solve_s_3d': sol3d.solve_time_s, 'iterations_3d': sol3d.n_iterations,
            'solver_3d': f'{sol3d.solver}/{sol3d.device_interface}',
            'phi_at_handoff_3d_V': float(sol3d.potential(traj[:1], verbose=False)[0]),
            'orbit_contacts': clr['n_contact'], 'bars': 0, 'patches': 0}

    t1 = time.time()
    lay = model3d.params['layout']
    bem_xy = lay['r_cut'] + fc.margin_m
    xs = np.arange(-bem_xy, bem_xy + 0.5 * fc.dxy_m, fc.dxy_m)
    zb = np.arange(-fc.z_half_m, fc.z_half_m + 0.5 * fc.dz_m, fc.dz_m)
    bem3d = bem_field3d(sol3d, xs, xs, zb, metal=stack, chunk=fc.pot_chunk, r_max=bem_xy, pot_device=fc.pot_device,
                        pot_precision=fc.pot_precision, pot_check=fc.pot_check, verbose=verbose)
    info['bem_pot'] = {'device': fc.pot_device, 'precision': fc.pot_precision, 'chunk': fc.pot_chunk,
                       'seconds': time.time() - t1}
    say(f"[fields] BEM field grid {len(xs)}x{len(xs)}x{len(zb)} to r = {bem_xy * 1e3:.0f} mm in {time.time() - t1:.0f} s "
        f"[{fc.pot_device}/{fc.pot_precision}, chunk {fc.pot_chunk}]")
    del sol3d
    info['rf_map'] = {'bem_only': True, 'v_dee_signed': float(np.min(np.asarray(model3d.potentials, dtype=float)))}
    info['layout'] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in lay.items()}
    cache = {'static2d': None, 'bem3d': field_to_arrays(bem3d), 'rf': None, 'rf_metal': None,
             'stack': {'kind': 'raster', 'grid': stack.grid, 'xs': stack.xs, 'zs': stack.zs},
             'model3d': model3d, 'bars': [], 'patches': [], 'info': info, 'seed': str(pc.seed)}
    out.parent.mkdir(parents=True, exist_ok=True)

    def write(p):
        with open(p, 'wb') as fh:
            pickle.dump(cache, fh, protocol=4)
    _write_atomic(out, write)
    say(f"[fields] cache {out} written ({time.time() - t0:.0f} s in all)")
    return out


def _raster_inside(grid, xs, zs):
    """Nearest-node metal test on a (x, y, z) raster with x = y nodes ``xs``."""
    ext, h, n = float(xs[-1]), float(xs[1] - xs[0]), len(xs)
    dz, z0 = float(zs[1] - zs[0]), float(zs[0])

    def inside(pts):
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        i = np.rint((p[:, 0] + ext) / h).astype(int)
        j = np.rint((p[:, 1] + ext) / h).astype(int)
        k = np.rint((p[:, 2] - z0) / dz).astype(int) if p.shape[1] >= 3 else np.full(len(p), int(round(-z0 / dz)))
        ok = (i >= 0) & (i < n) & (j >= 0) & (j < n) & (k >= 0) & (k < len(zs))
        out = np.zeros(len(p), dtype=bool)
        out[ok] = grid[i[ok], j[ok], k[ok]]
        return out
    inside.grid, inside.xs, inside.zs, inside.spacing, inside.ndim = grid, xs, zs, h, 3
    return inside


def add_columns(grid, xs, tc, info):
    """The full-height posts (x, y, r) and boxes (xmin, xmax, ymin, ymax) [mm] of [track] put into a metal raster: a
    raster column is metal when its node lies inside (box edges on half-nodes are exact)."""
    if not (tc.extra_posts_mm or tc.extra_boxes_mm):
        return grid
    grid = grid.copy()
    for (px, py, pr) in ([v * 1e-3 for v in q] for q in tc.extra_posts_mm):
        disc = np.hypot(xs[:, None] - px, xs[None, :] - py) <= pr
        grid[disc, :] = True
        say(f"[metal] extra post at x {px * 1e3:.1f} y {py * 1e3:.1f} mm, radius {pr * 1e3:.1f} mm: "
            f"{int(disc.sum())} raster columns")
    eps = 1e-9
    for (xa, xb, ya, yb) in ([v * 1e-3 for v in q] for q in tc.extra_boxes_mm):
        col = (((xs >= xa - eps) & (xs <= xb + eps))[:, None]) & (((xs >= ya - eps) & (xs <= yb + eps))[None, :])
        grid[col, :] = True
        say(f"[metal] extra box x {xa * 1e3:.2f}..{xb * 1e3:.2f}, y {ya * 1e3:.2f}..{yb * 1e3:.2f} mm: "
            f"{int(col.sum())} raster columns")
    info['extra_posts'] = [list(q) for q in tc.extra_posts_mm]
    info['extra_boxes'] = [list(q) for q in tc.extra_boxes_mm]
    return grid


def slit_raster(sl, r_center_m, z_half_m, h=SLIT_H_M):
    """The jaws of the [slit] ``sl`` centred at radius ``r_center_m``: a nearest-node raster of spacing ``h`` over their
    bounding box (``fields3d.Raster3D``), full height (z nodes -z_half_m, 0, +z_half_m). Along the ray (u) the jaws span
    [r_c - hw - L, r_c - hw] and [r_c + hw, r_c + hw + L], across it (w) |w| <= thickness / 2. Returns (test, jaws [m])."""
    from .fields3d import Raster3D
    rc, hw, length, t = (float(r_center_m), sl.half_width_mm * 1e-3, sl.jaw_length_mm * 1e-3, sl.thickness_mm * 1e-3)
    jaws = ((rc - hw - length, rc - hw), (rc + hw, rc + hw + length))
    phi = np.deg2rad(sl.azimuth_deg)
    eu, ew = np.array([np.cos(phi), np.sin(phi)]), np.array([-np.sin(phi), np.cos(phi)])
    corners = np.array([u * eu + w * ew for u in (jaws[0][0], jaws[1][1]) for w in (-0.5 * t, 0.5 * t)])
    lo, hi = corners.min(axis=0) - 2 * h, corners.max(axis=0) + 2 * h
    xs, ys = (np.arange(lo[k], hi[k] + 0.5 * h, h) for k in (0, 1))
    u = xs[:, None] * eu[0] + ys[None, :] * eu[1]
    w = xs[:, None] * ew[0] + ys[None, :] * ew[1]
    cols = (np.abs(w) <= 0.5 * t) & (((u >= jaws[0][0]) & (u <= jaws[0][1])) | ((u >= jaws[1][0]) & (u <= jaws[1][1])))
    zs = np.array([-z_half_m, 0.0, z_half_m])
    return Raster3D(np.repeat(cols[:, :, None], len(zs), axis=2), (xs, ys, zs), name='slit'), jaws


def load_fields(path, tc, fc):
    """The field cache of ``fields_stage`` (or of the design deck): {'bem3d': Field, 'stack': metal test,
    'model3d', 'info'}. The metal test is the run's choice (``track.obstacles``), not the cache's."""
    from .fields3d import field_from_arrays
    with open(require(path, 'paths.fields'), 'rb') as fh:
        d = pickle.load(fh)
    if d.get('rf') is not None:
        raise NotImplementedError(f"{path}: a cache with an RF-cavity map (BEM + map blend) is not supported")
    info = dict(d['info'])
    if tc.obstacles == 'mesh':
        if tc.extra_posts_mm or tc.extra_boxes_mm:
            raise ConfigError("track.extra_posts_mm / extra_boxes_mm need track.obstacles = 'raster'")
        from .electrodes3d import MeshObstacles
        stack = MeshObstacles(d['model3d'])
        say(f"[metal] mesh obstacles: BVH over {stack.n_tri:,} triangles")
    elif tc.obstacles == 'raster':
        st = d['stack']
        if st.get('kind') != 'raster' or 'grid' not in st:
            st_ = metal_raster(d['model3d'], fc, verbose=False)
            st = {'grid': st_.grid, 'xs': st_.xs, 'zs': st_.zs}
        stack = _raster_inside(add_columns(st['grid'], st['xs'], tc, info), st['xs'], st['zs'])
    else:
        raise ConfigError(f"track.obstacles = {tc.obstacles!r}: 'raster' or 'mesh'")
    return {'bem3d': field_from_arrays(d['bem3d']), 'stack': stack, 'model3d': d['model3d'], 'info': info}


def check_si_rotation(model3d, rotation_deg, where=''):
    """The inflector of a run must sit where its electrode model was built around it: a model that records the turn
    (params['si_rotation_deg'], written by geometry.build) is checked against inflector.rotation_deg; models without
    it (the design deck's) are not."""
    built = model3d.params.get('si_rotation_deg')
    if built is not None and abs(float(built) - float(rotation_deg)) > 1e-6:
        raise ConfigError(f"inflector.rotation_deg = {rotation_deg:g}, but the electrode model {where} was built around "
                          f"the inflector turned by {float(built):g} deg: track with that value or rebuild")


def install(finder, fields, cav0, volt_scale):
    """Put the BEM field (x volt_scale, idempotent: always from the cached value) and the metal test on the finder."""
    from PyPATools.field import TimedField
    bem3d = fields['bem3d']
    if not hasattr(bem3d, '_base_scaling'):
        bem3d._base_scaling = float(getattr(bem3d, 'scaling', 1.0))
    bem3d.scaling = bem3d._base_scaling * float(volt_scale)
    ri = fields['info']['rf_map']
    ri['v_dee_signed'] = float(np.min(np.asarray(fields['model3d'].potentials, dtype=float)))
    ri['volt_scale'] = float(volt_scale)
    say(f"[install] every dee at {volt_scale * abs(ri['v_dee_signed']) / 1e3:.2f} kV (model "
        f"{abs(ri['v_dee_signed']) / 1e3:.1f} kV x volt_scale {volt_scale:.4f})")
    finder._install_efield(TimedField(bem3d, omega=float(cav0.omega), phase=float(cav0.bunch_phase_offset)))
    finder.obstacle_mask = fields['stack']
    finder.housing_obstacle = None


# ---------------------------------------------------------------------------------------------------------------------
# tracking
def space_charge_kick(tc, cells, every, current_ma, f_rf):
    """FFT space-charge hook (``tracking.SpaceChargeKick``) at bunch current ``current_ma`` [mA] and bunch frequency
    ``f_rf`` [Hz]; ``cells`` nodes per axis, or the cell size ``track.sc_h_m`` when 0."""
    from PyPATools.poisson_fft import FFTPoissonSolver
    from .tracking import SpaceChargeKick
    solver = FFTPoissonSolver(h=tc.sc_h_m, pad_cells=3, window_quantile=tc.sc_window_q, use_gpu=tc.sc_gpu,
                              max_cells=tc.sc_max_cells, n_cells=(cells or None), frame=tc.sc_frame)
    return SpaceChargeKick(solver, resolve_every=every, bunch_current_a=current_ma * 1e-3, bunch_frequency_hz=f_rf,
                           kick_z=True, verbose=True)


def _summary_of(run):
    if run is None:
        return None
    strip = ('hits', 'hits_z', 'final_energies_mev', 'final_xyz', 'final_vz', 'initial_xyz', 'initial_energies_mev',
             'hit_turn', 'snapshots', 'tracks')
    d = {k: v for k, v in run.items() if k not in strip}
    if 'final_energies_mev' in run and len(run['final_energies_mev']):
        e = run['final_energies_mev']
        d['energy_spectrum'] = {'mean': float(e.mean()), 'std': float(e.std()), 'p05': float(np.percentile(e, 5)),
                                'p50': float(np.percentile(e, 50)), 'p95': float(np.percentile(e, 95)),
                                'n': int(len(e))}
        z = run['final_xyz'][:, 2]
        d['final_z'] = {'mean_mm': float(z.mean() * 1e3), 'rms_mm': float(z.std() * 1e3),
                        'max_abs_mm': float(np.abs(z).max() * 1e3)}
    return d


class BunchTracker:
    """The tracking model of a run: design, inflector, finder with the cached BEM field installed, reference beam.

    ``reference(phases)`` tracks the design particle, ``run_bunch(...)`` the hand-off bunch; ``track_stage`` runs the
    stage with them. Built from the [run], [paths], [machine], [magnet], [btrim], [inflector], [fields] and [track]
    sections of ``cfg``."""

    def __init__(self, cfg, verbose=True):
        from .accelerated_orbit_finder import AcceleratedOrbitFinder
        self.cfg = cfg
        self.rc, self.pc, self.mc, self.tc = (section(cfg, s) for s in ('run', 'paths', 'machine', 'track'))
        check_supported(self.mc)
        t0 = time.time()
        self.seed = load_seed(self.pc.seed)
        self.traj_seed = seed_orbit(self.seed)
        self.phase0, self.f_mhz = float(self.seed.bunch_phase_deg), float(self.seed.rf_frequency_mhz)
        self.bfield, bcheck = magnet_field_3d(cfg, verbose=verbose)
        self.design, _ = make_design(cfg, self.seed, bfield=self.bfield)
        self.inflector = make_inflector(cfg, verbose=verbose)
        tc, mc = self.tc, self.mc
        finder = AcceleratedOrbitFinder(self.design, target_energy_mev=mc.target_energy_mev, max_radius_m=mc.r_max_m,
                                        algorithm=tc.pusher, steps_per_turn=tc.steps_per_turn, verbose=False,
                                        gap_model='bem2d', dimensionality='3D', z_max=mc.z_max_m)
        finder.attach_inflector(self.inflector, obstacle=False, field_dim=3)
        finder.fast3d = bool(tc.fast3d)
        finder.fast3d_interp = tc.fast3d_interp
        self.finder = finder
        self.mass_mev = float(self.design.species.mass_mev)
        rot = section(cfg, 'inflector').rotation_deg
        self.ref_beam, self.handoff_meta = reference_beam(self.design.species, self.pc.handoff, rot)
        self.fields = load_fields(self.pc.fields, tc, section(cfg, 'fields'))
        check_si_rotation(self.fields['model3d'], rot, self.pc.fields)
        self.slit = section(cfg, 'slit')
        sl = self.slit
        if sl.half_width_mm < 0 or (sl.half_width_mm > 0 and not (sl.jaw_length_mm > 0 and sl.thickness_mm > 0)):
            raise ConfigError("[slit] needs half_width_mm >= 0 (0 = no slit), jaw_length_mm > 0 and thickness_mm > 0")
        if sl.half_width_mm > 0 and not (sl.pass_index >= 0 or (sl.pass_index == -1 and sl.r_center_mm > 0)):
            raise ConfigError("[slit] needs pass_index >= 0, or pass_index = -1 with r_center_mm > 0")
        self.slit_passes = {}        # phase -> {pass: radius [m]} of the reference particle on the slit ray
        install(finder, self.fields, self.design.rf_cavities[0], tc.volt_scale)
        self.info = {'bfield_check': bcheck}
        self.info.update(self.fields['info'])
        # bunch repetition frequency for the space-charge current: the hand-off file's RF, else h x f_rev
        self.f_rf = float(self.handoff_meta.get('rf_frequency_hz') or mc.harmonic * self.f_mhz * 1e6)
        x, v = self.ref_beam.x_vec[0], self.ref_beam.v_vec[0]
        say(f"[track] system built in {time.time() - t0:.0f} s; f_rev {self.f_mhz:.6f} MHz; reference at "
            f"r = {np.hypot(x[0], x[1]) * 1e3:.1f} mm, z = {x[2] * 1e3:+.2f} mm, v_z = {v[2] / 1e3:+.1f} km/s")

    def phases(self):
        """``track.phases_deg``, or the seed's phase and +-10 deg."""
        return ([float(p) for p in self.tc.phases_deg]
                or [round(self.phase0 - 10.0), round(self.phase0), round(self.phase0 + 10.0)])

    def reference(self, phases):
        """The design particle tracked at each phase: {phase: energies per turn, n_turns, orbit, max |z|, ...}. With a
        slit, also its crossings of the slit ray (``slit_passes``), which place the slit of the runs at that phase."""
        from .accelerated_orbit_finder import ProbeRecorder
        sl = self.slit
        ref = {}
        for ph in phases:
            t1 = time.time()
            probe = ProbeRecorder([sl.azimuth_deg]) if sl.half_width_mm > 0 else None
            self.finder.probes = probe
            try:
                r = self.finder.track_once(self.ref_beam, bunch_phase_deg=ph, rf_freq_mhz=self.f_mhz,
                                           max_turns=self.tc.turns)
            finally:
                self.finder.probes = None
            tr = np.asarray(r.trajectory_reference)
            ref[ph] = {'energies': [float(t.mean_energy_mev) for t in r.turn_statistics], 'n_turns': int(r.n_turns),
                       'traj': tr, 'max_abs_z_mm': float(np.abs(tr[:, 2]).max() * 1e3) if len(tr) else 0.0,
                       'obstacles': r.metadata.get('obstacles'), 'time_s': time.time() - t1}
            if probe is not None:
                d = probe.data()[0]
                fwd = d['direction'] > 0
                radii = np.hypot(d['xyz'][fwd, 0], d['xyz'][fwd, 1])
                self.slit_passes[ph] = {int(p): float(v) for p, v in zip(d['pass'][fwd], radii)}
                ref[ph]['slit_passes_mm'] = {str(p): v * 1e3 for p, v in self.slit_passes[ph].items()}
            say(f"[ref] phase {ph:+.1f}: {r.n_turns} turns, {r.final_energy_mev:.3f} MeV, "
                f"|z| max {ref[ph]['max_abs_z_mm']:.2f} mm ({ref[ph]['time_s']:.0f} s)")
        return ref

    def slit_for(self, ph):
        """The [slit] of a run at phase ``ph`` as (obstacle test, record), or (None, None) without one. Its centre is
        the reference particle's radius at crossing number slit.pass_index of the slit ray (``reference`` records them)
        or slit.r_center_mm, plus slit.offset_mm. A jaw thinner than a tracking step plus the raster diagonal could be
        stepped over, so that is refused."""
        sl = self.slit
        if sl.half_width_mm <= 0:
            return None, None
        passes = self.slit_passes.get(ph)
        if passes is None and sl.pass_index >= 0:
            raise RuntimeError(f"slit: no reference run at phase {ph:+.1f} deg (call reference() first)")
        passes = passes or {}
        if sl.pass_index >= 0 and sl.pass_index not in passes:
            raise ConfigError(f"slit.pass_index = {sl.pass_index}: the reference particle at phase {ph:+.1f} deg crosses "
                              f"azimuth {sl.azimuth_deg:g} deg on passes {sorted(passes)} only")
        rc = (passes[sl.pass_index] if sl.pass_index >= 0 else sl.r_center_mm * 1e-3) + sl.offset_mm * 1e-3
        test, jaws = slit_raster(sl, rc, self.mc.z_max_m)
        if jaws[0][0] <= 0.0:
            raise ConfigError(f"[slit]: the inner jaw reaches the machine axis (centre {rc * 1e3:.2f} mm)")
        step = 2.0 * np.pi * jaws[1][1] / self.tc.steps_per_turn
        if sl.thickness_mm * 1e-3 < step + np.sqrt(2.0) * SLIT_H_M:
            raise ConfigError(f"slit.thickness_mm = {sl.thickness_mm:g} is less than a tracking step at the outer jaw "
                              f"({step * 1e3:.2f} mm = 2 pi r / track.steps_per_turn) plus the raster diagonal "
                              f"({np.sqrt(2.0) * SLIT_H_M * 1e3:.2f} mm): particles could step over a jaw")
        on_jaw = {p: r * 1e3 for p, r in sorted(passes.items()) if any(a <= r <= b for a, b in jaws)}
        rec = {'azimuth_deg': sl.azimuth_deg, 'pass_index': sl.pass_index, 'offset_mm': sl.offset_mm,
               'r_center_mm': rc * 1e3, 'half_width_mm': sl.half_width_mm, 'jaw_length_mm': sl.jaw_length_mm,
               'thickness_mm': sl.thickness_mm, 'jaws_mm': [[a * 1e3, b * 1e3] for a, b in jaws],
               'raster_h_mm': SLIT_H_M * 1e3, 'columns': int(test.grid[:, :, 0].sum()),
               'reference_on_jaw_mm': {str(p): v for p, v in on_jaw.items()}}
        say(f"[slit] azimuth {sl.azimuth_deg:g} deg, centre {rc * 1e3:.2f} mm, opening {2 * sl.half_width_mm:g} mm, jaws "
            + ', '.join(f"{a * 1e3:.2f}..{b * 1e3:.2f}" for a, b in jaws) + f" mm, {rec['columns']} raster columns")
        if on_jaw:
            say(f"[slit] WARNING: the reference particle crosses a jaw on pass(es) "
                + ', '.join(f"{p} (r {v:.2f} mm)" for p, v in on_jaw.items()))
        return test, rec

    def run_bunch(self, ph, n, full, seed_rng, cells, every, out_dir=None):
        """Track ``n`` hand-off particles (0 = all; sampled with ``seed_rng``) at phase ``ph`` with space charge on a
        ``cells``^3 grid solved every ``every`` steps. ``full``: also the final / initial states, snapshots, tracks
        and (into ``out_dir``) the probe files. Returns the run record of the deck's layout."""
        from .accelerated_orbit_finder import ProbeRecorder
        tc, finder = self.tc, self.finder
        t1 = time.time()
        bunch = self.inflector.beam(n_particles=(int(n) or None), seed=seed_rng, species=self.design.species)
        r0 = np.column_stack([np.asarray(bunch.x, dtype=float), np.asarray(bunch.y, dtype=float),
                              np.asarray(bunch.z, dtype=float)])
        slit, slit_rec = self.slit_for(ph)
        metal = finder.obstacle_mask
        if slit is not None:
            from .electrodes3d import combine_obstacles
            finder.obstacle_mask = combine_obstacles(metal, slit)
        try:
            n_start_metal = int(np.count_nonzero(np.asarray(finder.obstacle_mask(r0), dtype=bool)))
            if n_start_metal:
                say(f"[metal] {n_start_metal} of {bunch.numpart} particles start INSIDE metal (lost at turn 0)")
            sc_kick, i_ma = None, 0.0
            if tc.sc:
                i_ma = (tc.sc_current_ma if tc.sc_current_ma > 0
                        else 1e3 * float(bunch.macro_charge_c) * bunch.numpart * self.f_rf)
                sc_kick = space_charge_kick(tc, cells, every, i_ma, self.f_rf)
                grid = f"{cells}^3 nodes" if cells else f"h = {tc.sc_h_m * 1e3:.2f} mm"
                say(f"[sc] FFT solver ({'GPU' if tc.sc_gpu else 'CPU'}) {grid} ({tc.sc_frame} frame), every {every} "
                    f"steps, {i_ma:.3f} mA")
            finder.space_charge = sc_kick
            track_ids = None
            if full and tc.n_tracks > 0:
                track_ids = np.sort(np.random.default_rng(TRACK_IDS_SEED).choice(
                    bunch.numpart, size=min(tc.n_tracks, bunch.numpart), replace=False))
            probe = ProbeRecorder(list(tc.probes_deg)) if (full and tc.probes_deg) else None
            finder.probes = probe
            res = finder.track_once(bunch, bunch_phase_deg=ph, rf_freq_mhz=self.f_mhz, max_turns=tc.turns,
                                    save_full_beam=False,
                                    save_times_s=[t * 1e-9 for t in tc.snapshot_ns] if full else None,
                                    track_ids=track_ids, track_every=tc.track_every)
        finally:
            finder.probes = None
            finder.space_charge = None
            finder.obstacle_mask = metal
        term = finder._obstacle_terminator
        hits = np.array(term.hits, dtype=float).reshape(-1, 4) if term is not None and term.hits else np.zeros((0, 4))
        hits_z = np.array(getattr(term, 'hits_z', []), dtype=float) if term is not None else np.zeros(0)
        if slit is not None:
            on_slit = slit(np.column_stack([hits[:, 2], hits[:, 3], hits_z])) if len(hits) else np.zeros(0, bool)
            slit_rec['n_lost'] = int(np.count_nonzero(on_slit))
            slit_rec['n_start_inside'] = int(np.count_nonzero(slit(r0)))
        ts = res.turn_statistics
        alive = [int(t.n_active) for t in ts]
        out = {'phase': ph, 'n': int(bunch.numpart), 'n_turns': int(res.n_turns), 'sc': sc_kick is not None,
               'current_ma': i_ma, 'alive': alive, 'e_mean': [float(t.mean_energy_mev) for t in ts],
               'e_std': [float(t.std_energy_mev) for t in ts], 'std_r': [float(t.std_r) for t in ts],
               'mean_r': [float(t.mean_r) for t in ts], 'mean_x': [float(t.mean_x) for t in ts],
               'mean_y': [float(t.mean_y) for t in ts],
               'std_z': [float(t.std_z) for t in ts], 'max_abs_z': [float(t.max_abs_z) for t in ts],
               'mean_z': [float(t.mean_z) for t in ts],
               'n_lost_metal': int(len(hits)), 'n_start_in_metal': n_start_metal,
               'hits': hits, 'hits_z': hits_z, 'time_s': time.time() - t1,
               'final_energy_mev': float(res.final_energy_mev),
               'transmission': alive[-1] / bunch.numpart if alive else 0.0,
               'space_charge': res.metadata.get('space_charge'),
               'fast_path': res.metadata.get('fast_path')}          # whether the compiled kernel ran, else why not
        if out['space_charge'] is not None:
            out['space_charge'] = {k: v for k, v in out['space_charge'].items() if k != 'log'}
        if slit_rec is not None:
            out['slit'] = slit_rec
        survived = None
        fb = res.metadata.get('full_beam') if full else None
        if fb is not None:
            survived = self._full_record(out, fb, res, bunch, hits, track_ids)
            res.metadata['full_beam'] = None
        if probe is not None and out_dir is not None:
            self._write_probes(probe, out_dir, ph, bunch.numpart, survived, sc_kick is not None)
        say(f"[bunch] phase {ph:+.1f}: {out['n']} particles, alive per turn {alive}, T {100 * out['transmission']:.2f} %, "
            f"E {(out['e_mean'] or [0])[-1]:.3f} +- {(out['e_std'] or [0])[-1]:.3f} MeV, z rms "
            f"{(out['std_z'] or [0])[-1] * 1e3:.2f} mm, lost on metal {out['n_lost_metal']}"
            + (f" (on the slit {slit_rec['n_lost']})" if slit_rec is not None else '') + f" ({out['time_s']:.0f} s)")
        return out

    def _full_record(self, out, fb, res, bunch, hits, track_ids):
        """Final / initial states, snapshots and tracks of a full run into ``out``; returns the survivor mask (finite
        last record, r < r_max, |z| < z_max, not lost on metal)."""
        mc, tc, mass = self.mc, self.tc, self.mass_mev
        last = fb[-1]
        lost_ids = set(int(h[0]) for h in hits)
        ids = np.arange(len(last))
        r_last = np.hypot(last[:, 0], last[:, 1])
        ok = np.isfinite(last).all(axis=1) & (r_last < self.finder.r_max) & (np.abs(last[:, 2]) < mc.z_max_m)
        ok &= ~np.isin(ids, list(lost_ids))
        out['final_energies_mev'] = kinetic_energy_mev(last[ok, 3:6], mass)
        out['final_xyz'] = last[ok, :3]
        out['final_vz'] = last[ok, 5]
        out['initial_xyz'] = fb[0][:, :3]
        out['initial_energies_mev'] = kinetic_energy_mev(fb[0][:, 3:6], mass)
        out['n_lost_vertical'] = int(np.sum(np.isfinite(last).all(axis=1) & (np.abs(last[:, 2]) >= mc.z_max_m)
                                            & ~np.isin(ids, list(lost_ids))))
        n_steps = len(fb)
        out['hit_turn'] = hits[:, 1] / max(n_steps, 1) * max(res.n_turns, 1)
        t_clock0 = float((res.metadata.get('launch') or {}).get('t0_s', 0.0))
        dt = 1.0 / (self.f_mhz * 1e6) / tc.steps_per_turn
        birth = np.asarray(getattr(bunch, 'birth_time', np.zeros(len(last))), dtype=float)
        hit_step = {int(h[0]): int(h[1]) for h in hits}
        snaps = []
        tref = np.asarray(res.trajectory_reference)
        for t_ns in tc.snapshot_ns:
            try:
                k, st = fb.at_time(t_ns * 1e-9)
            except IndexError:
                continue
            if k >= n_steps:
                continue
            snaps.append({'t_ns': t_ns, 'step': k, 'xyz': st[:, :3].copy(), 'vxyz': st[:, 3:6].copy(),
                          'born': birth <= t_ns * 1e-9,
                          'lost': np.array([hit_step.get(i, n_steps + 1) <= k for i in range(len(st))]),
                          'E_mev': kinetic_energy_mev(st[:, 3:6], mass), 'ref_xyz': tref[min(k, len(tref) - 1), :3]})
        out['snapshots'] = snaps
        tr = getattr(fb, 'tracks', None)
        if tr is not None:
            out['tracks'] = {'ids': np.asarray(track_ids), 'steps': np.asarray(fb.track_steps), 'xyz': tr,
                             'stop_step': np.array([hit_step.get(int(i), n_steps + 1) for i in track_ids]),
                             'n_steps': int(n_steps), 'steps_per_turn': int(tc.steps_per_turn),
                             'birth_step': np.asarray(np.round((birth - t_clock0) / dt), dtype=int)[track_ids],
                             'handoff_ids': np.asarray(getattr(bunch, 'handoff_index', ids))[track_ids],
                             'E_mev': kinetic_energy_mev(fb.last[track_ids, 3:6], mass)}
        return ok

    def _write_probes(self, probe, out_dir, ph, n, survived, sc):
        """One file per probe, every crossing of every particle (interpolated in the step): probe_<angle>deg.npz.
        'survived' (indexed by 'id') = alive at the end of the run."""
        for d in probe.data():
            fn = Path(out_dir) / f"probe_{d['angle_deg']:g}deg.npz"
            np.savez_compressed(fn, xyz=d['xyz'], vxyz=d['vxyz'], t=d['t'], id=d['id'], turn=d['pass'],
                                direction=d['direction'], step=d['step'],
                                E_mev=kinetic_energy_mev(d['vxyz'], self.mass_mev) if len(d['id']) else np.zeros(0),
                                survived=(survived if survived is not None else np.zeros(0, dtype=bool)),
                                angle_deg=d['angle_deg'], sense=d['sense'], phase_deg=float(ph), n=int(n),
                                tag=self.rc.name, sc=bool(sc), f_rf_mhz=self.mc.harmonic * self.f_mhz,
                                steps_per_turn=int(self.tc.steps_per_turn))
            say(f"[probe {d['angle_deg']:g} deg] {len(d['id'])} crossings of {len(np.unique(d['id']))} particles "
                f"-> {fn.name}")


def track_stage(cfg, verbose=True):
    """`pcr track`: track the hand-off bunch through the cached BEM field and the 3D magnet map (outputs: see the
    module docstring). Phases: ``track.phases_deg``, or the seed's phase and +-10 deg when empty; with more than one
    phase, ``track.n_scan`` particles scan them and the final run takes the best (transmission x final energy). An
    existing ``track.json`` in run.out_dir is reused, never overwritten. Returns the summary dict."""
    from .electrodes3d import section_loops
    rc, pc, tc = section(cfg, 'run'), section(cfg, 'paths'), section(cfg, 'track')
    out_dir = Path(require(rc.out_dir, 'run.out_dir'))
    out_json = out_dir / 'track.json'
    if out_json.exists():
        say(f"[track] {out_json} exists: reused (point run.out_dir elsewhere to re-track)")
        return json.loads(out_json.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    save_resolved(cfg, out_dir / 'config.json')
    t0 = time.time()
    bt = BunchTracker(cfg, verbose=verbose)
    phases = bt.phases()
    say(f"[track] {rc.name}: pusher {tc.pusher}, {tc.steps_per_turn} steps per turn, {tc.turns} turns; seed "
        f"{Path(pc.seed).name} (phase {bt.phase0:+.1f} deg); phases {phases}; space charge {'on' if tc.sc else 'off'}")
    ref = bt.reference(phases)
    scan = {}
    if len(phases) > 1:
        for ph in phases:
            scan[ph] = bt.run_bunch(ph, tc.n_scan, False, tc.seed_scan, tc.sc_cells, tc.sc_every)
        best_ph = max(phases, key=lambda p: scan[p]['transmission'] * (scan[p]['e_mean'][-1] if scan[p]['e_mean'] else 0))
        say(f"[track] best phase by transmission x energy: {best_ph:+.1f} deg")
    else:
        best_ph = phases[0]
    final = bt.run_bunch(best_ph, tc.n_final, True, tc.seed_final, tc.sc_final_cells, tc.sc_final_every, out_dir)
    say(f"[track] FINAL phase {best_ph:+.1f}: T {100 * final['transmission']:.2f} % = "
        f"{final['transmission'] * final['current_ma']:.3f} mA of {final['current_ma']:.3f} mA, "
        f"E {(final['e_mean'] or [0])[-1]:.3f} +- {(final['e_std'] or [0])[-1]:.3f} MeV")

    mc = bt.mc
    summary = {'seed': str(pc.seed), 'tag': rc.name, 'turns': tc.turns, 'phases': phases,
               'n_scan': int(tc.n_scan) if scan else 0, 'n_final': int(final['n']),
               'f_rev_mhz': bt.f_mhz, 'rf_mhz': mc.harmonic * bt.f_mhz, 'harmonic': mc.harmonic,
               'bmap': Path(pc.bfield_3d).name, 'volt_scale': float(tc.volt_scale),
               'btrim': to_dict(section(cfg, 'btrim')), 'fields': bt.info,
               'space_charge': {'on': bool(tc.sc), 'current_ma': final['current_ma'], 'h_mm': tc.sc_h_m * 1e3,
                                'every': tc.sc_final_every, 'cells': tc.sc_final_cells, 'max_cells': tc.sc_max_cells,
                                'scan_every': tc.sc_every, 'scan_cells': tc.sc_cells, 'window_q': tc.sc_window_q,
                                'frame': tc.sc_frame, 'gpu': bool(tc.sc_gpu)},
               'limits': {'r_max_m': mc.r_max_m, 'z_max_m': mc.z_max_m},
               'reference': {str(k): {kk: vv for kk, vv in v.items() if kk != 'traj'} for k, v in ref.items()},
               'scan': {str(k): _summary_of(v) for k, v in scan.items()}, 'best_phase': best_ph,
               'final': _summary_of(final), 'final_sc': None, 'reference_2d_composite': None,
               'inputs': {'seed': pc.seed, 'handoff': pc.handoff, 'fields': pc.fields, 'model3d': pc.model3d,
                          'bfield_3d': pc.bfield_3d, 'inflector_efield': pc.inflector_efield,
                          'housing_step': pc.housing_step},
               'wall_s': time.time() - t0}
    model3d = bt.fields['model3d']
    data = {'summary': summary, 'scan': scan, 'final': final, 'final_sc': None, 'ref': ref,
            'sections3d': {f'z{int(round(z * 1e3))}': section_loops(model3d, z=z) for z in (0.0, 0.015)},
            'housing_loops': bt.inflector.housing_loops(0.0), 'traj_seed': bt.traj_seed}

    def write_pkl(p):
        with open(p, 'wb') as fh:
            pickle.dump(data, fh, protocol=4)
    _write_atomic(out_dir / 'track_data.pkl', write_pkl)
    _write_atomic(out_json, lambda p: p.write_text(json.dumps(summary, indent=1, default=_json_default)))
    say(f"[track] {out_dir}: track.json, track_data.pkl ({time.time() - t0:.0f} s in all)")
    return summary
