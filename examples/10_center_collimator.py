"""Central-region phase-slit collimator: azimuth and aperture of a first-turn radial slit.

Shows: a two-piece grounded block in the first turn (tracking.RadialSlitCollimator via the finder's
``collimator`` setting: particles crossing the chosen azimuth outside a radial aperture are intercepted,
first crossing only, since later turns pass tens of mm further out than any physical block). In the
central region the turn-1 radius correlates strongly with RF phase, so the radial slit removes
out-of-phase particles early. Example 09's multi-particle winner is rebuilt (example 06's system with the
optimal geometry) and the slit's azimuth (0-360 deg) and full aperture (centred on the reference
particle's first-turn crossing radius - a design rule, not a free parameter) are scanned against two
end-of-acceleration figures of merit: the number of surviving particles and the turn-matched radial beam
width at THETA_REF_DEG (the ladder rms on the last common turn, the width an extraction septum sees):

    cost = (width / WIDTH_SCALE_MM)^2 + (LOSS_WEIGHT * lost_fraction)^2

Flow: coarse azimuth x aperture scan at SCAN_SPT -> local refine -> full-resolution verification and
the aperture Pareto front at the best azimuth (the width-vs-survivors trade made explicit).
Inputs: <resources>/uCyclo_v3_midplane_field.comsol, <resources>/MuonBunchOutOfInflector.csv and an
optimizer result pickle (default <out>/optimized_cavity_geometry_multi.pkl from example 09).
Runtime: about 55 bunch tracks of 15 turns: about 6 minutes for --particles 8, proportionally longer for
more particles.
Outputs: <out>/collimator_optimization.png, collimator_optimum.json.

Usage:
    python examples/10_center_collimator.py [RESULT_PKL] [--resources DIR] [--out DIR] [--particles N]
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import _common
from PyCentralRegion.accelerated_orbit_finder import AcceleratedOrbitFinder
from PyCentralRegion.machine import apply_seed_geometry
from PyCentralRegion.rf_cavity import DeeSystem

ex06 = _common.load_example('06_optimize_cavity_geometry.py')
ex09 = _common.load_example('09_optimize_cavity_geometry_multiparticle.py')

THETA_REF_DEG = 80.0      # width measured here (septum-entry azimuth)
WIDTH_SCALE_MM = 2.0
LOSS_WEIGHT = 4.0         # same weight as the optimizer's survival residual

MAX_TURNS = 15
SCAN_SPT = 300            # scan resolution (example 09's search resolution)

AZ_COARSE = list(np.arange(0.0, 360.0, 45.0))
AZ_REFINE_STEP = 15.0     # +- around the coarse winner
AP_SCAN_MM = [4.0, 6.0, 8.0, 12.0]
AP_FULL_MM = [3.0, 4.0, 6.0, 8.0, 12.0, 16.0]


def ladder(xy, az_rad):
    """Outbound crossing radii of the half-plane at az_rad [m]."""
    x, y = xy[:, 0], xy[:, 1]
    ca, sa = np.cos(az_rad), np.sin(az_rad)
    u = -x * sa + y * ca
    v = x * ca + y * sa
    idx = np.flatnonzero((u[:-1] < 0.0) & (u[1:] >= 0.0) & (v[:-1] > 0.0))
    if len(idx) == 0:
        return np.empty(0)
    f = -u[idx] / (u[idx + 1] - u[idx])
    xc = x[idx] + f * (x[idx + 1] - x[idx])
    yc = y[idx] + f * (y[idx + 1] - y[idx])
    return np.hypot(xc, yc)


def build_system(field_path, winner, bunch_path, n_particles):
    """Example 06's system with the winner's geometry, a full-resolution finder (500 steps/turn, from
    build_system) and a scan finder, and the bunch."""
    design, finder_final, _, _ = ex06.build_system(
        field_path, rf_frequency=float(winner.rf_frequency_mhz) * 1e6, quiet=True)
    apply_seed_geometry(DeeSystem(list(ex06.DEE_CENTER_ANGLES), ex06.DEE_OPENING_ANGLE, design.rf_cavities),
                        winner.metadata['optimal_geometry'])
    finder_scan = AcceleratedOrbitFinder(
        design, target_energy_mev=ex06.TARGET_ENERGY_MEV,
        max_radius_m=ex06.MAX_RADIUS_M, algorithm='rk4_rel',
        steps_per_turn=SCAN_SPT, verbose=False)
    beam = ex09.load_bunch(bunch_path, design.species, n_particles)
    print(f"winner: {winner.rf_frequency_mhz:.4f} MHz, phase {winner.bunch_phase_deg:.2f} deg, "
          f"{winner.final_energy_mev:.3f} MeV in {winner.n_turns} turns; "
          f"bunch {beam.numpart} particles (centroid prepended)", flush=True)
    return design, finder_final, finder_scan, beam


def run_once(finder, winner, beam, az_deg=None, ap_mm=None):
    """Track the bunch at the winner's RF, with the slit at (az_deg, ap_mm) if given. The finder builds
    the slit itself and centres it on the reference particle's first crossing of az_deg."""
    finder.collimator = None if az_deg is None else {'azimuth_deg': az_deg, 'aperture_mm': ap_mm}
    try:
        return finder.track_once(beam, bunch_phase_deg=float(winner.bunch_phase_deg),
                                 rf_freq_mhz=float(winner.rf_frequency_mhz),
                                 max_turns=MAX_TURNS, save_full_beam=True)
    finally:
        finder.collimator = None


def width_at_ref(res, az_deg=THETA_REF_DEG):
    """Turn-matched ladder rms [mm] on the last COMMON turn at az_deg over the particles still tracked
    at the end (the tracker leaves an intercepted particle where it was stopped); also (turn, mean_r_mm)."""
    fb = res.metadata['full_beam']
    alive = np.any(fb[-1] != fb[-2], axis=1)
    ladders = [radii for radii in (ladder(fb[:, p, :2], np.radians(az_deg)) for p in np.flatnonzero(alive))
               if len(radii)]
    if not ladders:
        return 1e3, 0, 0.0
    k = min(len(radii) for radii in ladders)
    r_k = np.array([radii[k - 1] for radii in ladders]) * 1e3
    return float(r_k.std()), k, float(r_k.mean())


def main():
    ap = _common.arguments(__doc__)
    ap.add_argument('result', nargs='?', type=Path,
                    help='optimizer result pickle (default: <out>/optimized_cavity_geometry_multi.pkl)')
    ap.add_argument('--particles', type=int, default=None,
                    help='deterministic subsample of the bunch, as in example 09 (default: all)')
    args = _common.parse(ap)
    field_path = _common.require(args.resources / ex06.FIELD_FILE)
    bunch_path = _common.require(args.resources / ex09.BUNCH_FILE)
    winner_pkl = _common.require(args.result or args.out / 'optimized_cavity_geometry_multi.pkl',
                                 'run example 09 first, or name a result pickle')
    t0 = time.time()
    print("=" * 70)
    print("CENTER-REGION PHASE-SLIT COLLIMATOR (aperture + azimuth)")
    print("=" * 70)
    with open(winner_pkl, "rb") as f:
        winner = pickle.load(f)
    design, finder_final, finder_scan, beam = build_system(field_path, winner, bunch_path, args.particles)
    n = beam.numpart

    print("\nbaseline (no collimator)...", flush=True)
    base_s = run_once(finder_scan, winner, beam)
    w0_s, k0, rmean0 = width_at_ref(base_s)
    print(f"  scan res: width {w0_s:.2f} mm rms on turn {k0} at "
          f"{THETA_REF_DEG:.0f} deg (mean r {rmean0:.1f} mm)", flush=True)

    def forward(az_deg, ap_mm, finder, tag):
        res = run_once(finder, winner, beam, az_deg, ap_mm)
        coll = res.metadata['collimator']
        lost = coll['n_collimated']
        r1 = coll['r_center_mm'] if coll['r_center_mm'] is not None else float('nan')
        w, k, _ = width_at_ref(res)
        cost = (w / WIDTH_SCALE_MM) ** 2 + (LOSS_WEIGHT * lost / n) ** 2
        print(f"    [{tag}] az {az_deg:6.1f}  ap {ap_mm:5.1f} mm (r1 {r1:5.1f}) -> lost {lost:3d}  "
              f"width {w:5.2f} mm (turn {k})  cost {cost:7.3f}", flush=True)
        return dict(az=az_deg, ap=ap_mm, r1_mm=r1, lost=lost, survivors=n - lost, width=w, turn=k,
                    cost=cost)

    # ---- coarse scan -------------------------------------------------------
    print("\ncoarse scan (azimuth x aperture, scan resolution):", flush=True)
    history = [forward(az, ap, finder_scan, 'scan') for az in AZ_COARSE for ap in AP_SCAN_MM]
    best = min(history, key=lambda h: h['cost'])

    # ---- local refine ------------------------------------------------------
    print(f"\nrefine around az {best['az']:.0f} deg:", flush=True)
    for az in (best['az'] - AZ_REFINE_STEP, best['az'] + AZ_REFINE_STEP):
        for ap in AP_FULL_MM:
            history.append(forward(az % 360.0, ap, finder_scan, 'refine'))
    for ap in AP_FULL_MM:
        if ap not in AP_SCAN_MM:
            history.append(forward(best['az'], ap, finder_scan, 'refine'))
    best = min(history, key=lambda h: h['cost'])
    az_b = best['az']

    # ---- verification + aperture Pareto at full resolution ----------------
    print(f"\nverification + Pareto at az {az_b:.1f} deg, {finder_final.steps_per_turn} steps/turn:",
          flush=True)
    base_f = run_once(finder_final, winner, beam)
    w0_f, k0f, _ = width_at_ref(base_f)
    print(f"    baseline: width {w0_f:.2f} mm (turn {k0f}), {n} particles", flush=True)
    pareto = [forward(az_b, ap, finder_final, 'pareto') for ap in AP_FULL_MM]
    verify = min(pareto, key=lambda h: h['cost'])

    print("\n" + "=" * 70)
    print(f"COLLIMATOR OPTIMUM: azimuth {verify['az']:.1f} deg, aperture {verify['ap']:.1f} mm "
          f"(centered on r1 = {verify['r1_mm']:.1f} mm)")
    print(f"  survivors {verify['survivors']}/{n} ({100 * verify['survivors'] / n:.1f}%), width at "
          f"{THETA_REF_DEG:.0f} deg: {verify['width']:.2f} mm rms (baseline {w0_f:.2f} mm)")
    print("=" * 70, flush=True)

    # phase-slit diagnostic: turn-1 radius vs initial longitudinal offset
    fb = base_f.metadata['full_beam']
    x0 = beam.x_vec[:, :2]
    v0 = beam.v_vec[:, :2]
    t_hat = v0.mean(axis=0) / np.linalg.norm(v0.mean(axis=0))
    s0 = (x0 - x0.mean(axis=0)) @ t_hat * 1e3
    r_t1 = np.full(n, np.nan)
    for p in range(n):
        radii = ladder(fb[:, p, :2], np.radians(az_b))
        if len(radii):
            r_t1[p] = radii[0] * 1e3
    m = ~np.isnan(r_t1)
    corr = float(np.corrcoef(s0[m], r_t1[m])[0, 1])
    print(f"phase-slit diagnostic: corr(initial s0, turn-1 radius at {az_b:.0f} deg) = {corr:+.3f}",
          flush=True)

    plot(history, pareto, verify, w0_f, s0, r_t1, m, n, args.out / 'collimator_optimization.png')

    out = dict(result_pkl=str(winner_pkl), azimuth_deg=verify['az'], aperture_mm=verify['ap'],
               r1_center_mm=verify['r1_mm'], survivors=verify['survivors'], n_particles=n,
               width_mm=verify['width'], baseline_width_mm=w0_f, theta_ref_deg=THETA_REF_DEG,
               width_scale_mm=WIDTH_SCALE_MM, loss_weight=LOSS_WEIGHT, phase_radius_corr=corr,
               pareto=[dict(ap=h['ap'], survivors=h['survivors'], width=h['width']) for h in pareto],
               runtime_s=time.time() - t0)
    with open(args.out / "collimator_optimum.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved summary to {args.out / 'collimator_optimum.json'}", flush=True)


def plot(history, pareto, verify, w0_f, s0, r_t1, m, n, png):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]
    for ap in sorted({h['ap'] for h in history}):
        pts = sorted([(h['az'], h['cost']) for h in history if h['ap'] == ap])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', ms=4, label=f"{ap:.0f} mm")
    ax.set(xlabel='collimator azimuth (deg)', ylabel='cost', yscale='log',
           title='Scan: cost vs azimuth per aperture')
    ax.legend(fontsize=7, title='aperture')
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot([h['survivors'] for h in pareto], [h['width'] for h in pareto], 'o-', color='tab:blue')
    for h in pareto:
        ax.annotate(f"{h['ap']:.0f}", (h['survivors'], h['width']), textcoords='offset points',
                    xytext=(4, 4), fontsize=8)
    ax.axhline(w0_f, color='gray', ls='--', lw=1, label=f'no collimator ({w0_f:.2f} mm)')
    ax.plot(verify['survivors'], verify['width'], '*', ms=16, color='crimson', zorder=5,
            label='chosen optimum')
    ax.set(xlabel='surviving particles', ylabel=f'width at {THETA_REF_DEG:.0f} deg (mm rms)',
           title='Aperture Pareto at the best azimuth (labels = aperture mm)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(s0[m], r_t1[m], s=10, alpha=0.7)
    r1c, ap = verify['r1_mm'], verify['ap']
    ax.axhline(r1c - ap / 2, color='red', ls='--', lw=1.2)
    ax.axhline(r1c + ap / 2, color='red', ls='--', lw=1.2, label=f'aperture {ap:.0f} mm')
    ax.set(xlabel='initial longitudinal offset s0 (mm, ~RF phase)',
           ylabel=f"turn-1 radius at {verify['az']:.0f} deg (mm)",
           title='The slit IS a phase slit: turn-1 radius vs initial phase')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axis('off')
    lines = ["COLLIMATOR OPTIMUM",
             f"  azimuth   {verify['az']:6.1f} deg",
             f"  aperture  {verify['ap']:6.1f} mm",
             f"  center r1 {verify['r1_mm']:6.1f} mm", "",
             f"  survivors {verify['survivors']}/{n} ({100 * verify['survivors'] / n:.1f}%)",
             f"  width     {verify['width']:.2f} mm rms (baseline {w0_f:.2f})", "",
             f"cost = (w/{WIDTH_SCALE_MM:.0f}mm)^2 + ({LOSS_WEIGHT:.0f} * lost_frac)^2"]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=12, family='monospace',
            va='top')

    fig.suptitle('Central-region phase-slit collimator (two-piece grounded block, first turn)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches='tight')
    print(f"saved plot to {png}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
