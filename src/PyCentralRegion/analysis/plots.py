"""One-page summary figure of a scored run, drawn from the score dict alone (``metrics.score_run`` output or a saved
score json): acceptance in release phase, the bunch shape per turn, the turn structure at the virtual probes and the
collimation trade-off. matplotlib is imported only when a figure is drawn.
"""
import numpy as np


def summary_figure(score, path=None, dpi=120):
    """Four panels from a score: (a) survival against release phase with the acceptance windows, (b) core aspect
    ratio and sigma_r / sigma_l per turn, (c) turn separation and valley share per probe azimuth, (d) final current
    against valley share for the best radial and vertical slit of each budget. Saved to path when given; returns the
    figure."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 8.0))
    g = (lambda k: np.nan if score.get(k) is None else score[k])

    a = ax[0, 0]
    sc = score.get('survival_curve') or {}
    if sc:
        x = np.array([float(k) for k in sc])
        o = np.argsort(x)
        a.plot(x[o], 100 * np.array(list(sc.values()))[o], 'o-', ms=3)
        for key, c in (('acc90', 'C2'), ('acc80', 'C1')):
            if score.get(f'{key}_lo') is not None:
                a.axvspan(score[f'{key}_lo'], score[f'{key}_hi'], color=c, alpha=0.12, lw=0, label=key)
        a.legend(frameon=False, fontsize=8)
    a.set(xlabel='release phase [RF deg]', ylabel='survival [%]', title=f"(a) acceptance: W_eff {g('W_eff'):.0f} deg")

    a = ax[0, 1]
    tt = sorted((int(k), v) for k, v in (score.get('turns') or {}).items() if k != 'last')
    if tt:
        t = [k for k, _ in tt]
        a.plot(t, [v['ratio'] for _, v in tt], 'o-', color='k', label='aspect ratio')
        a.set_ylabel('r-l aspect ratio (core)')
        b = a.twinx()
        b.plot(t, [v['sig_r'] for _, v in tt], 's--', color='C0', label='sigma_r')
        b.plot(t, [v['sig_l'] for _, v in tt], '^--', color='C3', label='sigma_l')
        b.set_ylabel('core sigma [mm]')
        (h1, l1), (h2, l2) = a.get_legend_handles_labels(), b.get_legend_handles_labels()
        b.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc='upper left')
    a.set(xlabel='turn', title=f"(b) bunch shape: last-turn ratio {g('ratio'):.2f}, halo {100 * g('halo3'):.1f} %")

    a = ax[1, 0]
    pr = score.get('probes') or {}
    if pr:
        az = sorted(float(k) for k in pr)
        a.plot(az, [pr[f"{x:g}"]['sep'] for x in az], 'o-', color='k')
        a.set_ylabel('turn separation [sigma_R]')
        b = a.twinx()
        b.plot(az, [100 * pr[f"{x:g}"]['valley'] for x in az], 's--', color='C3')
        b.set_ylabel('valley share [%]', color='C3')
        a.set_xticks(az)
    a.set(xlabel='probe azimuth [deg]', title=f"(c) turns: valley_all {100 * g('valley_all'):.1f} %, "
                                              f"sigma_R {g('turn_sigR'):.2f} mm")

    a = ax[1, 1]
    sl = score.get('slits') or {}
    i0, v0 = g('I_final'), 100 * g('valley_all')
    for kind, mk in (('radial', 'o-'), ('vertical', 's--')):
        xs, ys = [i0], [v0]
        for b in sorted(sl, key=float):
            xs.append(i0 * (1.0 - sl[b][kind].get('taken', 0.0)))
            ys.append(100 * sl[b][kind]['valley'])
        a.plot(xs, ys, mk, label=f'one {kind} slit')
    a.legend(frameon=False, fontsize=8)
    a.set(xlabel='final current [mA]', ylabel='valley share [%]', title='(d) collimation trade-off')

    fig.suptitle(f"{score.get('tag', '')}: I_final {i0:.2f} mA (T {100 * g('T'):.1f} %), E {g('E_med'):.3f} MeV, "
                 f"dE/E {100 * g('dE_E_robust'):.2f} % (robust)", x=0.01, ha='left')
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=dpi)
    return fig
