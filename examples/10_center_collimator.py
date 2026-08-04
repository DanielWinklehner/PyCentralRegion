"""10_center_collimator.py - Central-region phase-slit collimator.

A simple two-piece grounded block during the FIRST TURN: particles
crossing a chosen azimuth outside a radial aperture are intercepted
(PyCentralRegion.tracking.RadialSlitCollimator - first-crossing-only,
since later turns pass tens of mm further out than any physical block).
In the center region the turn-1 radius correlates strongly with RF
phase, so the radial slit removes out-of-phase particles early.

Loads the example-09 multiparticle winner, rebuilds the system (ex06 /
ex07 pattern), and optimizes the collimator's AZIMUTH (0-360 deg) and
radial APERTURE (full width, centered on the bunch-mean first-turn
crossing radius - a design rule, not a free parameter) against the two
end-of-acceleration figures of merit:

    - number of surviving particles, and
    - turn-matched radial beam width at THETA_REF_DEG (the ladder rms
      on the last common turn - the width the extraction septum sees).

cost = (width / WIDTH_SCALE_MM)^2 + (LOSS_WEIGHT * lost_fraction)^2

Flow: coarse azimuth x aperture scan at SCAN_SPT -> local refine ->
full-resolution verification + aperture Pareto at the best azimuth
(the width-vs-survivors trade made explicit).

Usage: python 10_center_collimator.py   (after example 09)
"""
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'src'))

from PyCentralRegion.accelerated_orbit_finder import AcceleratedOrbitFinder
from PyCentralRegion.tracking import RadialSlitCollimator

WINNER_PKL = ROOT / 'output' / 'optimized_cavity_geometry_multi.pkl'
OUTPUT_DIR = ROOT / 'output'

THETA_REF_DEG = 80.0      # width measured here (septum-entry azimuth)
WIDTH_SCALE_MM = 2.0
LOSS_WEIGHT = 4.0         # suite survival-weight convention

MAX_TURNS = 15
SCAN_SPT = 300            # scan resolution (ex09 search resolution)
FINAL_SPT = 500           # verification resolution

AZ_COARSE = list(np.arange(0.0, 360.0, 45.0))
AZ_REFINE_STEP = 15.0     # +- around the coarse winner
AP_SCAN_MM = [4.0, 6.0, 8.0, 12.0]
AP_FULL_MM = [3.0, 4.0, 6.0, 8.0, 12.0, 16.0]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def build_system():
    ex06 = _load(HERE / "06_optimize_cavity_geometry.py", "e6")
    ex07 = _load(HERE / "07_bem_gap_verification.py", "e7")
    ex09 = _load(HERE / "09_optimize_cavity_geometry_multiparticle.py",
                 "e9")
    with open(WINNER_PKL, "rb") as f:
        winner = pickle.load(f)
    design, finder_final, _, _ = ex06.build_system(
        rf_frequency=float(winner.rf_frequency_mhz) * 1e6, quiet=True)
    from PyCentralRegion.rf_cavity import DeeSystem
    dee = DeeSystem(list(ex06.DEE_CENTER_ANGLES), ex06.DEE_OPENING_ANGLE,
                    design.rf_cavities)
    ex07._apply_optimal_geometry(design, dee,
                                 winner.metadata['optimal_geometry'])
    finder_scan = AcceleratedOrbitFinder(
        design, target_energy_mev=ex06.TARGET_ENERGY_MEV,
        max_radius_m=ex06.MAX_RADIUS_M, algorithm='rk4_rel',
        steps_per_turn=SCAN_SPT, verbose=False)
    beam = ex09.load_bunch(design.species)
    print(f"winner: {winner.rf_frequency_mhz:.4f} MHz, phase "
          f"{winner.bunch_phase_deg:.2f} deg, "
          f"{winner.final_energy_mev:.3f} MeV in {winner.n_turns} turns; "
          f"bunch {beam.numpart} particles (centroid prepended)",
          flush=True)
    return winner, design, finder_final, finder_scan, beam


def run_once(finder, winner, beam, coll=None):
    finder.engine.extra_terminators = [coll] if coll is not None else []
    try:
        res = finder.track_once(
            beam, bunch_phase_deg=float(winner.bunch_phase_deg),
            rf_freq_mhz=float(winner.rf_frequency_mhz),
            max_turns=MAX_TURNS, save_full_beam=True)
    finally:
        finder.engine.extra_terminators = []
    return res


def width_at_ref(res, dead, az_deg=THETA_REF_DEG):
    """Turn-matched ladder rms [mm] on the last COMMON turn at az_deg,
    over surviving particles; also (ordinal, mean_r_mm)."""
    fb = res.metadata['full_beam']
    az = np.radians(az_deg)
    ladders = []
    for p in range(fb.shape[1]):
        if p in dead:
            continue
        xy = fb[:, p, :2]
        xy = xy[~np.isnan(xy[:, 0])]
        radii = ladder(xy, az)
        if len(radii):
            ladders.append(radii)
    if not ladders:
        return 1e3, 0, 0.0
    k = min(len(l) for l in ladders)
    r_k = np.array([l[k - 1] for l in ladders]) * 1e3
    return float(r_k.std()), k, float(r_k.mean())


def first_turn_radius(traj_mean, az_deg):
    """Bunch-mean first-crossing radius at az_deg [m] (aperture
    center - design rule)."""
    radii = ladder(np.asarray(traj_mean)[:, :2], np.radians(az_deg))
    if len(radii) == 0:
        raise ValueError(f"reference path never crosses {az_deg} deg")
    return float(radii[0])


def main():
    t0 = time.time()
    print("=" * 70)
    print("CENTER-REGION PHASE-SLIT COLLIMATOR (aperture + azimuth)")
    print("=" * 70)
    if not WINNER_PKL.exists():
        print(f"ERROR: {WINNER_PKL} not found - run example 09 first.")
        sys.exit(1)
    winner, design, finder_final, finder_scan, beam = build_system()
    n = beam.numpart

    print("\nbaseline (no collimator)...", flush=True)
    base_s = run_once(finder_scan, winner, beam)
    w0_s, k0, rmean0 = width_at_ref(base_s, dead=set())
    print(f"  scan res: width {w0_s:.2f} mm rms on turn {k0} at "
          f"{THETA_REF_DEG:.0f} deg (mean r {rmean0:.1f} mm)", flush=True)
    traj_mean = base_s.trajectory_reference

    def forward(az_deg, ap_mm, finder, tag):
        r1 = first_turn_radius(traj_mean, az_deg)
        coll = RadialSlitCollimator(np.radians(az_deg),
                                    r1 - 0.5 * ap_mm * 1e-3,
                                    r1 + 0.5 * ap_mm * 1e-3)
        res = run_once(finder, winner, beam, coll)
        dead = {h[0] for h in coll.hits}
        lost = len(dead)
        w, k, _ = width_at_ref(res, dead)
        cost = (w / WIDTH_SCALE_MM) ** 2 \
            + (LOSS_WEIGHT * lost / n) ** 2
        print(f"    [{tag}] az {az_deg:6.1f}  ap {ap_mm:5.1f} mm "
              f"(r1 {r1 * 1e3:5.1f}) -> lost {lost:3d}  width "
              f"{w:5.2f} mm (turn {k})  cost {cost:7.3f}", flush=True)
        return dict(az=az_deg, ap=ap_mm, r1_mm=r1 * 1e3, lost=lost,
                    survivors=n - lost, width=w, turn=k, cost=cost)

    # ---- coarse scan -------------------------------------------------------
    print("\ncoarse scan (azimuth x aperture, scan resolution):",
          flush=True)
    history = []
    for az in AZ_COARSE:
        for ap in AP_SCAN_MM:
            history.append(forward(az, ap, finder_scan, 'scan'))
    best = min(history, key=lambda h: h['cost'])

    # ---- local refine ------------------------------------------------------
    print(f"\nrefine around az {best['az']:.0f} deg:", flush=True)
    for az in (best['az'] - AZ_REFINE_STEP, best['az'] + AZ_REFINE_STEP):
        for ap in AP_FULL_MM:
            history.append(forward(az % 360.0, ap, finder_scan,
                                   'refine'))
    for ap in AP_FULL_MM:
        if ap not in AP_SCAN_MM:
            history.append(forward(best['az'], ap, finder_scan,
                                   'refine'))
    best = min(history, key=lambda h: h['cost'])
    az_b, ap_b = best['az'], best['ap']

    # ---- verification + aperture Pareto at full resolution ----------------
    print(f"\nverification + Pareto at az {az_b:.1f} deg, "
          f"{FINAL_SPT} steps/turn:", flush=True)
    base_f = run_once(finder_final, winner, beam)
    w0_f, k0f, _ = width_at_ref(base_f, dead=set())
    print(f"    baseline: width {w0_f:.2f} mm (turn {k0f}), "
          f"{n} particles", flush=True)
    pareto = [forward(az_b, ap, finder_final, 'pareto')
              for ap in AP_FULL_MM]
    verify = min(pareto, key=lambda h: h['cost'])

    print("\n" + "=" * 70)
    print(f"COLLIMATOR OPTIMUM: azimuth {verify['az']:.1f} deg, "
          f"aperture {verify['ap']:.1f} mm (centered on r1 = "
          f"{verify['r1_mm']:.1f} mm)")
    print(f"  survivors {verify['survivors']}/{n} "
          f"({100 * verify['survivors'] / n:.1f}%), width at "
          f"{THETA_REF_DEG:.0f} deg: {verify['width']:.2f} mm rms "
          f"(baseline {w0_f:.2f} mm)")
    print("=" * 70, flush=True)

    # phase-slit diagnostic: turn-1 radius vs initial longitudinal offset
    fb = base_f.metadata['full_beam']
    x0 = beam.x_vec[:, :2]
    v0 = beam.v_vec[:, :2]
    t_hat = v0.mean(axis=0) / np.linalg.norm(v0.mean(axis=0))
    s0 = (x0 - x0.mean(axis=0)) @ t_hat * 1e3
    r_t1 = np.full(n, np.nan)
    for p in range(n):
        xy = fb[:, p, :2]
        xy = xy[~np.isnan(xy[:, 0])]
        radii = ladder(xy, np.radians(az_b))
        if len(radii):
            r_t1[p] = radii[0] * 1e3
    m = ~np.isnan(r_t1)
    corr = float(np.corrcoef(s0[m], r_t1[m])[0, 1])
    print(f"phase-slit diagnostic: corr(initial s0, turn-1 radius at "
          f"{az_b:.0f} deg) = {corr:+.3f}", flush=True)

    plot(history, pareto, verify, w0_f, base_f, s0, r_t1, m, n)

    out = dict(azimuth_deg=verify['az'], aperture_mm=verify['ap'],
               r1_center_mm=verify['r1_mm'],
               survivors=verify['survivors'], n_particles=n,
               width_mm=verify['width'], baseline_width_mm=w0_f,
               theta_ref_deg=THETA_REF_DEG,
               width_scale_mm=WIDTH_SCALE_MM, loss_weight=LOSS_WEIGHT,
               phase_radius_corr=corr,
               pareto=[dict(ap=h['ap'], survivors=h['survivors'],
                            width=h['width']) for h in pareto],
               runtime_s=time.time() - t0)
    with open(OUTPUT_DIR / "collimator_optimum.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved summary to {OUTPUT_DIR / 'collimator_optimum.json'}",
          flush=True)


def plot(history, pareto, verify, w0_f, base_f, s0, r_t1, m, n):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]
    aps = sorted({h['ap'] for h in history})
    for ap in aps:
        pts = sorted([(h['az'], h['cost']) for h in history
                      if h['ap'] == ap])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'o-', ms=4,
                label=f"{ap:.0f} mm")
    ax.set_xlabel('collimator azimuth (deg)')
    ax.set_ylabel('cost')
    ax.set_yscale('log')
    ax.legend(fontsize=7, title='aperture')
    ax.grid(alpha=0.3)
    ax.set_title('Scan: cost vs azimuth per aperture')

    ax = axes[0, 1]
    ax.plot([h['survivors'] for h in pareto],
            [h['width'] for h in pareto], 'o-', color='tab:blue')
    for h in pareto:
        ax.annotate(f"{h['ap']:.0f}", (h['survivors'], h['width']),
                    textcoords='offset points', xytext=(4, 4),
                    fontsize=8)
    ax.axhline(w0_f, color='gray', ls='--', lw=1,
               label=f'no collimator ({w0_f:.2f} mm)')
    ax.plot(verify['survivors'], verify['width'], '*', ms=16,
            color='crimson', zorder=5, label='chosen optimum')
    ax.set_xlabel('surviving particles')
    ax.set_ylabel(f'width at {THETA_REF_DEG:.0f} deg (mm rms)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title('Aperture Pareto at the best azimuth '
                 '(labels = aperture mm)')

    ax = axes[1, 0]
    ax.scatter(s0[m], r_t1[m], s=10, alpha=0.7)
    r1c = verify['r1_mm']
    ap = verify['ap']
    ax.axhline(r1c - ap / 2, color='red', ls='--', lw=1.2)
    ax.axhline(r1c + ap / 2, color='red', ls='--', lw=1.2,
               label=f'aperture {ap:.0f} mm')
    ax.set_xlabel('initial longitudinal offset s0 (mm, ~RF phase)')
    ax.set_ylabel(f"turn-1 radius at {verify['az']:.0f} deg (mm)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title('The slit IS a phase slit: turn-1 radius vs '
                 'initial phase')

    ax = axes[1, 1]
    ax.axis('off')
    lines = [f"COLLIMATOR OPTIMUM",
             f"  azimuth   {verify['az']:6.1f} deg",
             f"  aperture  {verify['ap']:6.1f} mm",
             f"  center r1 {verify['r1_mm']:6.1f} mm", "",
             f"  survivors {verify['survivors']}/{n} "
             f"({100 * verify['survivors'] / n:.1f}%)",
             f"  width     {verify['width']:.2f} mm rms "
             f"(baseline {w0_f:.2f})", "",
             f"cost = (w/{WIDTH_SCALE_MM:.0f}mm)^2 + "
             f"({LOSS_WEIGHT:.0f} * lost_frac)^2"]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=12, family='monospace', va='top')

    fig.suptitle('Central-region phase-slit collimator '
                 '(two-piece grounded block, first turn)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    png = OUTPUT_DIR / 'collimator_optimization.png'
    fig.savefig(png, dpi=150, bbox_inches='tight')
    print(f"saved plot to {png}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
