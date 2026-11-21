"""The demo machine's inputs, generated: a worked example of the whole chain that needs nothing proprietary.

    python -m PyCentralRegion.demo [--inputs DIR] [--force]
    pcr evaluate --config configs/examples/demo/design.toml --out output/demo/eval

The machine is ``configs/examples/demo/machine.toml``: four 40 deg dees at 45 + 90 k deg, h = 4, protons, 20 kV, a
1.2 T field, injection at 60 keV, about five turns to 0.8 MeV. Not a real cyclotron; small and round numbers, so that
the chain runs in minutes on a CPU. ``generate`` writes, into the ``inputs/`` folder beside that file (or
``--inputs``), the six files of its ``[paths]``:

``bfield_3d.txt``
    the magnet: an isochronous field B(r) = B0 gamma(r) with a weak-focusing term, -B0 (gamma(r) - eps (r/a)^2) on
    the median plane (negative: a positive ion turns counter-clockwise, the sense the dee system's gaps are numbered
    in), off the plane by the median-plane expansion (B_r = z dB/dr, B_z - z^2/2 (B'' + B'/r)); COMSOL text layout
    (x y z Bx By Bz, a full regular grid with a z = 0 plane), as the pipeline reads a real map;
``housing_outline.txt``, ``housing.stp``
    the spiral-inflector housing, here a plain polygonal block round the centre: its exit plate is gap 0's ground edge
    (gap 0's counter-electrode is the housing), with a rectangular exit aperture; the outline (mm, with 'line' curves
    and the aperture's corners) for the electrode builder, the solid (STEP, mm, the outline extruded with the exit
    recess cut) for the tracker's metal test;
``handoff.h5``
    the injected bunch, openPMD plane-crossing layout (the spiral-inflector hand-off): Gaussian in position, angle,
    energy and crossing time on the emitting plane 3 mm inside the exit plate, 1 mA;
``inflector_efield.pickle``
    a zero static E-field (PyPATools Field): the demo has no inflector electrodes;
``seed.pkl``
    the seed (``AcceleratedOrbitFinder.OptimizedOrbit``): straight radial gaps (one variable segment per gap, at 0
    deg) and the reference orbit of the design particle tracked with thin-gap kicks at the RF phase that gains the
    most energy over the design turns.

Everything is deterministic (fixed random seed). Generating takes about a minute (most of it the seed's phase scan).
"""
import argparse
import dataclasses
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from . import config as C

DEMO_DIR = Path(__file__).resolve().parents[2] / 'configs' / 'examples' / 'demo'
MACHINE_FILE = DEMO_DIR / 'machine.toml'
FILES = {'bfield_3d': 'bfield_3d.txt', 'housing_outline': 'housing_outline.txt', 'housing_step': 'housing.stp',
         'handoff': 'handoff.h5', 'inflector_efield': 'inflector_efield.pickle', 'seed': 'seed.pkl'}
CLIGHT = 299792458.0
QE = 1.602176634e-19


@dataclasses.dataclass
class DemoSpec:
    """What the demo's inputs are made of besides its [machine] table."""
    b0_t: float = 1.2                # central field
    weak_eps: float = 0.002          # weak-focusing term eps (r / weak_a_m)^2 of B / B0
    weak_a_m: float = 0.1
    map_half_m: float = 0.15         # the map spans |x|, |y| <= this ...
    map_dxy_m: float = 0.004         # ... at this spacing
    map_z_m: float = 0.02            # and |z| <= this
    map_dz_m: float = 0.004
    injection_kev: float = 60.0      # kinetic energy of the design particle on the emitting plane (orbit radius
                                     # 29.5 mm: beyond dee 0's solid corner post, which ends where the dee is
                                     # geometry.dee_wall_mm wide, about r 25 mm)
    recess_mm: float = 3.0           # the emitting plane lies this far inside the exit plate
    aperture_mm: tuple = (8.0, 10.0)  # exit aperture: width (in the median plane) x height
    housing_half_height_mm: float = 30.0
    n_particles: int = 20000
    current_ma: float = 1.0
    sigma_mm: float = 1.0            # rms position spread across the plane (radial and vertical) ...
    clip_mm: float = 3.0             # ... cut at +-this
    sigma_mrad: float = 10.0         # rms angle (radial and vertical), cut at 3 sigma
    sigma_de: float = 0.002          # rms relative energy spread
    sigma_phase_deg: float = 10.0    # rms crossing time [deg RF], cut at 3 sigma
    rng_seed: int = 1
    phases_deg: tuple = tuple(range(-60, 61, 5))   # the seed's phase scan
    segment_r_m: float = 0.06        # outer radius of each gap's (straight, 0 deg) variable segment


def machine(path=MACHINE_FILE):
    """The demo's [machine] table (a MachineConfig)."""
    return C.section(C.load(path), 'machine')


def species_constants(name):
    from PyPATools.species import IonSpecies
    sp = IonSpecies(name)
    return float(sp.mass_mev), float(sp.q)


def bz_midplane(r, spec, mass_mev, q):
    """B_z(r) [T] on the median plane and its first two r-derivatives (the field is negative: counter-clockwise ions)."""
    omega = q * QE * spec.b0_t / (mass_mev * 1e6 * QE / CLIGHT ** 2)
    k = omega / CLIGHT
    u = 1.0 - (k * r) ** 2
    g = u ** -0.5
    g1 = k * k * r * u ** -1.5
    g2 = k * k * u ** -1.5 + 3.0 * k ** 4 * r * r * u ** -2.5
    a2 = spec.weak_eps / spec.weak_a_m ** 2
    b = -spec.b0_t * (g - a2 * r * r)
    b1 = -spec.b0_t * (g1 - 2.0 * a2 * r)
    b2 = -spec.b0_t * (g2 - 2.0 * a2)
    return b, b1, b2


def field_at(x, y, z, spec, mass_mev, q):
    """(Bx, By, Bz) [T] off the median plane: B_r = z B', B_z = B - z^2 / 2 (B'' + B' / r) (curl- and divergence-free
    to the order kept)."""
    r = np.hypot(x, y)
    b, b1, b2 = bz_midplane(r, spec, mass_mev, q)
    lap = np.where(r > 1e-9, b2 + b1 / np.maximum(r, 1e-9), 2.0 * b2)
    bz = b - 0.5 * z * z * lap
    br = z * b1
    rs = np.maximum(r, 1e-12)
    return br * x / rs, br * y / rs, bz


def write_bfield(path, spec, mass_mev, q):
    n = int(round(2 * spec.map_half_m / spec.map_dxy_m)) + 1
    nz = int(round(2 * spec.map_z_m / spec.map_dz_m)) + 1
    xs = np.linspace(-spec.map_half_m, spec.map_half_m, n)
    zs = np.linspace(-spec.map_z_m, spec.map_z_m, nz)
    zs[np.argmin(np.abs(zs))] = 0.0
    Z, Y, X = np.meshgrid(zs, xs, xs, indexing='ij')         # x fastest, then y, then z (COMSOL's order)
    bx, by, bz = field_at(X, Y, Z, spec, mass_mev, q)
    rows = np.column_stack([X.ravel(), Y.ravel(), Z.ravel(), bx.ravel(), by.ravel(), bz.ravel()])
    header = (f"% Model: PyCentralRegion demo magnet (analytic: B0 {spec.b0_t} T, isochronous for the demo species, "
              f"weak-focusing eps {spec.weak_eps} at {spec.weak_a_m} m)\n"
              f"% Dimension: 3\n% Nodes: {len(rows)}\n% Expressions: 3\n% Description: Magnetic flux density\n"
              f"% Length unit: m\n% x y z mf.Bx (T) mf.By (T) mf.Bz (T)\n")
    with open(path, 'w', encoding='ascii', newline='\n') as fh:
        fh.write(header)
        np.savetxt(fh, rows, fmt='%.9g')
    return {'nodes': len(rows), 'grid': (n, n, nz), 'b_centre_t': float(bz_midplane(0.0, spec, mass_mev, q)[0])}


# ---------------------------------------------------------------------------------------------------------------------
# the exit: gap 0's ground edge, in the frame of gap 0's line (x along it, y across it, along the beam)
def gap0_frame(mc):
    """(angle of gap 0's line [deg], half its width at the emission radius [m], emission radius [m], unit vectors ex,
    ey) - gap 0 is dee 0's entry gap, at its centre angle minus half the opening."""
    theta = float(mc.dee_center_angles_deg[0]) - 0.5 * float(mc.dee_opening_deg)
    t = math.radians(theta)
    ex, ey = np.array([math.cos(t), math.sin(t), 0.0]), np.array([-math.sin(t), math.cos(t), 0.0])
    return theta, ex, ey


def f_rev_hz(mc, spec):
    """The revolution frequency the field is isochronous for: q B0 / (2 pi m)."""
    mass_mev, q = species_constants(mc.species)
    return q * spec.b0_t * CLIGHT ** 2 / (2.0 * math.pi * mass_mev * 1e6)


def injection(mc, spec):
    """Design particle: kinetic energy [eV], speed [m/s], momentum [eV/c], orbit radius [m] in the central field."""
    mass_mev, q = species_constants(mc.species)
    m_ev = mass_mev * 1e6
    t_ev = spec.injection_kev * 1e3
    p_ev = math.sqrt(t_ev * (t_ev + 2.0 * m_ev))
    speed = CLIGHT * p_ev / (t_ev + m_ev)
    rho = p_ev / (q * CLIGHT * spec.b0_t)
    return t_ev, speed, p_ev, rho


def layout(mc, spec):
    """The housing, its exit and the emitting plane [mm, machine frame]: {'theta', 'ex', 'ey', 'r0', 'y_plate', 'y_emit',
    'outer' (N, 2), 'aperture' (4, 3), 'recess' (x0, x1, y0, y1, z0, z1) in gap-0 frame, 'origin' (3,) the plane origin}."""
    theta, ex, ey = gap0_frame(mc)
    _, _, _, rho = injection(mc, spec)
    r0 = rho * 1e3
    half_gap = 0.5 * max(mc.gap_width_min_m, rho * math.radians(mc.gap_angle_deg)) * 1e3
    yp = -half_gap                                   # the exit plate: gap 0's ground edge
    ye = yp - spec.recess_mm                         # the emitting plane, inside the housing's exit recess
    local = np.array([[r0 + 11.0, yp], [r0 - 13.0, yp], [r0 - 19.0, yp - 3.5], [r0 - 25.0, yp - 10.5],
                      [r0 - 17.0, yp - 16.5], [r0 - 1.0, yp - 15.5], [r0 + 11.0, yp - 7.5]])
    w, h = spec.aperture_mm
    ap_local = np.array([[r0 - w / 2, yp, -h / 2], [r0 + w / 2, yp, -h / 2], [r0 + w / 2, yp, h / 2],
                         [r0 - w / 2, yp, h / 2]])
    R = np.column_stack([ex, ey, [0.0, 0.0, 1.0]])   # gap-0 frame -> machine frame
    outer = np.column_stack([local, np.zeros(len(local))]) @ R.T
    return {'theta': theta, 'ex': ex, 'ey': ey, 'r0': r0, 'y_plate': yp, 'y_emit': ye, 'R': R,
            'outer': outer[:, :2], 'outer_local': local, 'aperture': ap_local @ R.T,
            'recess': (r0 - w / 2, r0 + w / 2, ye - 1.5, yp + 1.0, -h / 2, h / 2),
            'origin': (np.array([r0, ye, 0.0]) @ R.T) * 1e-3}


def write_outline(path, lay):
    n = len(lay['outer'])
    lines = ["# PyCentralRegion demo: the housing outline at the median plane and its exit aperture (generated)",
             "frame machine", "units mm", f"outer {n}"]
    lines += [f"{x:.6f} {y:.6f}" for x, y in lay['outer']]
    lines += [f"curves {n}"] + [f"line {i} {(i + 1) % n}" for i in range(n)]
    lines += [f"aperture {len(lay['aperture'])}"] + [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in lay['aperture']]
    Path(path).write_text("\n".join(lines) + "\n", encoding='utf-8')


def write_housing_step(path, lay, spec):
    """The outline extruded over +-housing_half_height_mm, minus the exit recess (mm), turned into the machine frame."""
    import gmsh
    import signal
    handler = signal.getsignal(signal.SIGINT)
    gmsh.initialize(readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("demo_housing")
        occ = gmsh.model.occ
        H = spec.housing_half_height_mm
        pts = [occ.addPoint(x, y, -H) for x, y in lay['outer_local']]
        ls = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        face = occ.addPlaneSurface([occ.addCurveLoop(ls)])
        vol = [t for d, t in occ.extrude([(2, face)], 0, 0, 2 * H) if d == 3][0]
        x0, x1, y0, y1, z0, z1 = lay['recess']
        box = occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)
        out, _ = occ.cut([(3, vol)], [(3, box)])
        occ.rotate(out, 0, 0, 0, 0, 0, 1, math.radians(lay['theta']))
        occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
        signal.signal(signal.SIGINT, handler)            # gmsh leaves SIGINT at SIG_DFL


# ---------------------------------------------------------------------------------------------------------------------
# the bunch
def write_handoff(path, mc, spec, lay):
    import h5py
    mass_mev, q = species_constants(mc.species)
    m_ev = mass_mev * 1e6
    t_ev, speed, p_ev, _ = injection(mc, spec)
    rng = np.random.default_rng(spec.rng_seed)
    n = int(spec.n_particles)

    def gauss(sigma, cut):
        v = rng.normal(0.0, sigma, n)
        bad = np.abs(v) > cut
        while bad.any():
            v[bad] = rng.normal(0.0, sigma, bad.sum())
            bad = np.abs(v) > cut
        return v

    ex, ey, ez = lay['ex'], lay['ey'], np.array([0.0, 0.0, 1.0])
    du = gauss(spec.sigma_mm, spec.clip_mm) * 1e-3
    dz = gauss(spec.sigma_mm, spec.clip_mm) * 1e-3
    au = gauss(spec.sigma_mrad * 1e-3, 3 * spec.sigma_mrad * 1e-3)
    az = gauss(spec.sigma_mrad * 1e-3, 3 * spec.sigma_mrad * 1e-3)
    ekin = t_ev * (1.0 + gauss(spec.sigma_de, 3 * spec.sigma_de))
    f_rev = f_rev_hz(mc, spec)
    f_rf = mc.harmonic * f_rev
    t_rel = gauss(spec.sigma_phase_deg, 3 * spec.sigma_phase_deg) / 360.0 / f_rf
    origin = lay['origin']
    r = origin[None, :] + du[:, None] * ex[None, :] + dz[:, None] * ez[None, :]
    d = ey[None, :] + au[:, None] * ex[None, :] + az[:, None] * ez[None, :]
    d /= np.linalg.norm(d, axis=1)[:, None]
    p = np.sqrt(ekin * (ekin + 2.0 * m_ev))[:, None] * d
    q_bunch = spec.current_ma * 1e-3 / f_rf
    w = np.full(n, q_bunch / n)
    with h5py.File(path, 'w') as f:
        f.attrs['openPMD'] = '1.1.0'
        f.attrs['openPMDextension'] = 0
        f.attrs['basePath'] = '/'
        f.attrs['particlesPath'] = 'particles'
        f.attrs['software'] = 'PyCentralRegion.demo'
        g = f.create_group('particles').create_group(mc.species)
        g.attrs['numParticles'] = n
        for k, c in zip('xyz', range(3)):
            g.create_dataset(f'position/{k}', data=r[:, c])
            g.create_dataset(f'momentum/{k}', data=p[:, c])
        g.create_dataset('t_rel', data=t_rel)
        g.create_dataset('weight', data=w)
        meta = {'mode': 'plane_crossing', 'pypatools_species_name': mc.species, 'species_mass_mev': mass_mev,
                'species_charge_state': q, 'plane_origin_m': origin, 'plane_normal': ey, 'plane_axis_u': ex,
                'plane_axis_v': ez, 'design_exit_velocity_mps': ey * speed, 'rf_frequency_hz': f_rf,
                'phase_reference_time_s': 0.0, 'bunch_charge_c': q_bunch, 'transmitted_current_ma': spec.current_ma,
                'design_energy_ev': t_ev}
        for k, v in meta.items():
            g.attrs['PyPATools:' + k] = np.asarray(v, dtype=float) if isinstance(v, np.ndarray) else v
    return {'n': n, 'f_rf_hz': f_rf, 'f_rev_hz': f_rev, 'bunch_charge_c': q_bunch}


def write_efield(path):
    from PyPATools.field import Field
    g = {'x': np.linspace(-0.04, 0.04, 9), 'y': np.linspace(-0.04, 0.04, 9), 'z': np.linspace(-0.02, 0.02, 5)}
    zero = np.zeros((9, 9, 5))
    f = Field.from_arrays(grid=g, values={'x': zero, 'y': zero.copy(), 'z': zero.copy()}, dim=3, units='m',
                          interpolator_backend='scipy', label='demo: no inflector field')
    f.save_pickle(str(path))


# ---------------------------------------------------------------------------------------------------------------------
# the seed
def make_seed(mc, spec, lay, f_rev, verbose=True):
    """The seed: the dee system of [machine] with straight radial gaps and the design particle's thin-gap orbit at the
    phase (of spec.phases_deg) that gains the most energy over n_turns_design - 2 turns, tracked on to the target
    energy. Refused when that orbit comes closer than 1 mm to the housing after its first eighth of a turn."""
    from PyPATools.field import Field
    from .accelerated_orbit_finder import AcceleratedOrbitFinder, make_beam_from_state
    from .central_region import CentralRegion
    from .machine import apply_seed_geometry, make_dee_system
    mass_mev, q = species_constants(mc.species)
    xs = np.arange(-spec.map_half_m, spec.map_half_m + 1e-12, 0.001)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    bz = bz_midplane(np.hypot(X, Y), spec, mass_mev, q)[0]
    zero = np.zeros_like(bz)
    b2 = Field.from_arrays(grid={'x': xs, 'y': xs}, values={'x': zero, 'y': zero.copy(), 'z': bz}, dim=2, units='m',
                           interpolator_backend='numba', label='demo magnet, median plane')
    geom = {'n_gaps': 2 * mc.n_dees, 'n_segments': 1, 'opening_angle_deg': float(mc.dee_opening_deg),
            'segment_angles_per_gap': [[0.0] for _ in range(2 * mc.n_dees)],
            'segment_radii_per_gap': [[spec.segment_r_m] for _ in range(2 * mc.n_dees)],
            'segment_rotations_per_gap': None}
    design = CentralRegion(name='demo_seed', dimensionality='2D')
    design.set_species(mc.species)
    design.set_electric_field(Field.zero())
    design.set_magnetic_field(b2)
    ds = make_dee_system(mc, f_rev, n_segments=1, segment_radii=(spec.segment_r_m,))
    apply_seed_geometry(ds, geom)
    for cav in ds.gaps:
        design.add_rf_cavity(cav)
    _, speed, _, _ = injection(mc, spec)
    x0, v0 = lay['origin'], lay['ey'] * speed
    beam = make_beam_from_state(design.species, x0, v0)
    scan_turns = max(int(mc.n_turns_design) - 2, 1)
    finder = AcceleratedOrbitFinder(design, target_energy_mev=math.inf, max_radius_m=mc.r_max_m,
                                    algorithm='rk4_rel', steps_per_turn=500, verbose=False)
    scan, refused = {}, {}
    for ph in spec.phases_deg:
        res = finder.track_once(beam, float(ph), f_rev * 1e-6, max_turns=scan_turns)
        why = _orbit_refused(res, lay)
        if why:
            refused[float(ph)] = why
        else:
            scan[float(ph)] = float(res.final_energy_mev)
    if not scan:
        raise RuntimeError(f"no phase of the demo's scan gives a usable orbit: {refused}")
    best = max(scan, key=scan.get)
    finder = AcceleratedOrbitFinder(design, target_energy_mev=mc.target_energy_mev, max_radius_m=mc.r_max_m,
                                    algorithm='rk4_rel', steps_per_turn=500, verbose=False)
    res = finder.track_once(beam, best, f_rev * 1e-6, max_turns=4 * int(mc.n_turns_design))
    res.metadata['optimal_geometry'] = geom
    res.metadata['demo'] = {'phase_deg': float(best), 'phase_scan_mev': scan,
                            'note': 'PyCentralRegion.demo: straight radial gaps, thin-gap reference orbit'}
    res.metadata.pop('full_beam', None)
    traj = np.asarray(res.trajectory_reference)
    ang = np.unwrap(np.arctan2(traj[:, 1], traj[:, 0]))
    clear = housing_clearance_mm(traj[ang - ang[0] > np.pi / 4] * 1e3, lay['outer'])
    res.metadata['demo']['phases_refused'] = refused
    if verbose:
        print(f"[demo] seed: phase {best:+.0f} deg ({', '.join(f'{p:+.0f}: {e:.3f}' for p, e in scan.items())} MeV "
              f"after {scan_turns} turns); {res.n_turns} turns to {res.final_energy_mev:.3f} MeV, "
              f"r {np.hypot(traj[:, 0], traj[:, 1]).min() * 1e3:.1f}..{np.hypot(traj[:, 0], traj[:, 1]).max() * 1e3:.1f}"
              f" mm, {'counter-clockwise' if ang[-1] > ang[0] else 'CLOCKWISE'}, {clear:.1f} mm off the housing",
              flush=True)
    why = _orbit_refused(res, lay)
    if why:
        raise RuntimeError(f"the demo's reference orbit at {best:+.0f} deg: {why}")
    return res


def _orbit_refused(res, lay):
    """Why an orbit cannot be the seed's ('' when it can): it turns clockwise, comes within 1 mm of the housing or
    within half the injection radius of the axis after its first eighth of a turn."""
    traj = np.asarray(res.trajectory_reference)
    if len(traj) < 3:
        return 'no orbit'
    ang = np.unwrap(np.arctan2(traj[:, 1], traj[:, 0]))
    if not ang[-1] > ang[0]:
        return 'clockwise (the dee system expects counter-clockwise)'
    later = traj[ang - ang[0] > np.pi / 4] * 1e3
    clear = housing_clearance_mm(later, lay['outer'])
    if not clear > 1.0:
        return f'{clear:.1f} mm off the housing'
    r_min = float(np.hypot(later[:, 0], later[:, 1]).min()) if len(later) else math.inf
    if r_min < 0.5 * lay['r0']:
        return f'passes {r_min:.1f} mm from the axis'
    return ''


def housing_clearance_mm(pts, outer):
    """Smallest distance [mm] of the points (N, 2+) [mm] from the closed polygon ``outer`` (negative: inside it)."""
    pts = np.asarray(pts, dtype=float)[:, :2]
    if not len(pts):
        return math.inf
    a, b = np.asarray(outer, dtype=float), np.roll(np.asarray(outer, dtype=float), -1, axis=0)
    ab = b - a
    t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(-1) / (ab * ab).sum(-1)[None], 0.0, 1.0)
    d = np.linalg.norm(pts[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=-1).min(axis=1)
    from matplotlib.path import Path as MPath
    inside = MPath(np.asarray(outer)).contains_points(pts)
    return float(np.where(inside, -d, d).min())


# ---------------------------------------------------------------------------------------------------------------------
def generate(inputs=None, force=False, spec=None, machine_file=MACHINE_FILE, verbose=True):
    """Write the demo's inputs into ``inputs`` (default: the inputs/ folder beside ``machine_file``); files that exist
    are kept unless ``force``. Returns {name: Path}."""
    spec = spec or DemoSpec()
    mc = machine(machine_file)
    folder = Path(inputs) if inputs else Path(machine_file).parent / 'inputs'
    folder.mkdir(parents=True, exist_ok=True)
    paths = {k: folder / v for k, v in FILES.items()}
    t0 = time.time()
    mass_mev, q = species_constants(mc.species)
    lay = layout(mc, spec)

    def todo(key):
        return force or not paths[key].exists()

    def say(msg):
        if verbose:
            print(f"[demo] {msg}", flush=True)

    if todo('bfield_3d'):
        info = write_bfield(paths['bfield_3d'], spec, mass_mev, q)
        say(f"magnet map {info['grid']} nodes, B(0) = {info['b_centre_t']:+.4f} T -> {paths['bfield_3d'].name}")
    if todo('housing_outline'):
        write_outline(paths['housing_outline'], lay)
        say(f"housing outline ({len(lay['outer'])} corners; exit plate on gap 0's ground edge at "
            f"{lay['theta']:.1f} deg) -> {paths['housing_outline'].name}")
    if todo('housing_step'):
        write_housing_step(paths['housing_step'], lay, spec)
        say(f"housing solid -> {paths['housing_step'].name}")
    if todo('handoff'):
        info = write_handoff(paths['handoff'], mc, spec, lay)
        say(f"hand-off bunch: {info['n']} {mc.species} at {spec.injection_kev:g} keV, {spec.current_ma:g} mA, "
            f"f_rf {info['f_rf_hz'] * 1e-6:.4f} MHz -> {paths['handoff'].name}")
    if todo('inflector_efield'):
        write_efield(paths['inflector_efield'])
    if todo('seed'):
        res = make_seed(mc, spec, lay, f_rev_hz(mc, spec), verbose=verbose)
        with open(paths['seed'], 'wb') as fh:
            pickle.dump(res, fh)
        say(f"seed -> {paths['seed'].name}")
    say(f"inputs in {folder} ({time.time() - t0:.0f} s)")
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--inputs', default=None, help="folder for the inputs (default: inputs/ beside the machine file)")
    ap.add_argument('--machine', default=str(MACHINE_FILE), help="the demo machine file")
    ap.add_argument('--force', action='store_true', help="write every input again")
    a = ap.parse_args(argv)
    generate(a.inputs, force=a.force, machine_file=a.machine)
    return 0


if __name__ == '__main__':
    sys.exit(main())
