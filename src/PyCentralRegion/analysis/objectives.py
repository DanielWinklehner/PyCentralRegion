"""Scoring settings and the table of metrics a design campaign optimises.

``ScoreConfig`` is the ``[score]`` section of a run file: every threshold and choice of ``metrics.score_run`` and
``twins.vortex_twin`` is one of its fields, documented with its default in ``configs/defaults/score.toml``. The defaults
reproduce the 3D campaign's run scoring of 2026-11-01, so new scores stay comparable with its tables. What describes
the machine (harmonic, loss limits, species) is not a scoring choice: it comes from the run itself, its json and
hand-off file and the ``[machine]`` table it was tracked with (``runs.RunData.machine``; ``metrics.run_constants``).

``METRICS`` lists the scalar score keys an optimizer may use as an objective or a constraint: name -> direction hint,
unit and meaning (with what the campaign learned about it). All of them are top-level keys of the score dict, so the
optimisation layer reads them with ``config.get_dotted(score, name)``; nested values have dotted paths too (e.g.
``turns.3.sig_r``, ``best_slit.20.offset_mm``). ``register_metric`` adds a user metric; ``score_run`` evaluates it on
every run after the built-in metrics (it sees the run and the score so far) and stores it under its name.

The campaign's choice (53 runs of one injected beam): maximise I_final (or W_eff), minimise valley_all and the final
ratio; constrain E_med to the design energy and the turn width turn_sigR to <= 3 mm (8 sigma at a 24 mm turn
separation), which needs dE_E_robust <= ~1.3 % at R = 330 mm (sigR_E = R / 2 x dE/E) and a centre fan held below
~2 mm in projection (fan_proj_last). There is no vortex objective: at that bunch density the vortex cannot shape the
beam in 7 turns. The low-fidelity proxies (turns 0-3) are evaluated on the final survivors, which for a turns 0-3 run
are the particles alive at turn 3: the correlations quoted were measured on full runs, re-check them there.
"""
import dataclasses
from typing import Callable, Optional

DIRECTIONS = ('max', 'min', 'target', 'info')


@dataclasses.dataclass
class ScoreConfig:
    b0_t: float = 0.0                    # |B| at the bunch for the vortex prediction when the json has no field check;
    #                                      0 = the isochronous central field 2 pi f_rev m / q of the tracked species
    beam: str = ''                       # label of the injected beam in the score; '' = the hand-off file's stem
    # which runs are scored
    min_tracks: int = 5000               # fewer recorded tracks: the run is skipped
    min_survivors: int = 200             # fewer tracked survivors: the run is skipped
    # energy
    energy_windows: tuple = (0.05, 0.02, 0.01)     # +-rel windows: I_<p>pct (median-centred), I_<p>pct_best (best)
    # fixed-azimuth readout (metrics.readout_metrics: the *_az keys), whatever step the run stopped at
    readout_azimuth_deg: float = 0.0     # the virtual probe [deg]: free of RF field, after the last gap of the final
    #                                      turn and behind the bunch's start by more than its phase spread (0: between
    #                                      the dees of a 4-dee machine with dees at 45 + 90 k deg)
    readout_pass: int = 0                # its pass (the probes' numbering); 0 = the last one before the run's turns
    #                                      end: turns, or turns - 1 when the bunch starts behind the probe
    readout_window: float = 0.02         # the best-placed +-window whose particles also get the vertical keys
    #                                      (*_<p>pct_az): the extraction energy acceptance
    # acceptance in hand-off release phase
    phase_bin_deg: float = 5.0           # release-phase bins of the survival curve
    phase_range_deg: float = 120.0       # bins span +-range (beyond: the survival curve's end bins, not release_bins)
    min_bin_count: int = 40              # bins with fewer tracked particles do not count
    min_bin_survivors: int = 11          # a release bin with fewer survivors reports no final energy and lag
    acceptance_levels: tuple = (0.9, 0.8)          # acc<level>: widest run of bins with survival >= level
    survival_bands_deg: tuple = (30.0, 40.0)       # S30 = survival for |phase| <= 30, S30_40 for 30 < |phase| <= 40
    compression_clip_sigma: float = 2.5  # compression slope fitted to survivors within this many sigma of the median
    loss_turn_edges: tuple = (0.5, 1.0, 2.0)       # classes of the turn of loss (turns after release)
    phase_band_q: float = 0.95           # band<q>: the central release-phase band holding this share of the survivors
    handoff_quantiles: int = 5           # survival per quantile of the hand-off |z| and |z'|
    # bunch shape in the bunch frame, per turn
    n_snapshots: int = 16                # instants per turn
    core_q: float = 0.95                 # core = this share of the particles (smallest Mahalanobis distance)
    halo_sigma: float = 3.0              # halo<s>: share beyond s sigma of the Gaussian-equivalent core
    halo_born_turns: tuple = (1, 2)      # halo_born_t<t>: share of the last-turn halo already halo at turn t
    z_limit_mm: float = 10.0             # z_out<limit>_last: share of the survivors beyond |z| = this at any instant
    #                                      of the last turn (a vertical aperture of +-limit)
    map_core_sigma: float = 2.0          # particles within this many core sigma fit a one-turn map (also the twin's)
    proxy_turns: tuple = (2, 3)          # flat ratio / sig_r / sig_l / s_round / compress / fan keys of these turns
    # per-particle orbit centres (the centre fan)
    fan_degree: int = 3                  # polynomial in release phase that 'explains' the centres
    turn_fill: float = 0.8               # a revolution needs this share of the fullest one's records (among the
    #                                      particles that keep pace: metrics.turn_centres), a fan turn this share of
    #                                      the survivors with a centre
    # virtual probes (radial half-planes) and turn structure
    probe_angles_deg: tuple = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
    probe_passes: int = 3                # the last this many complete passes
    pass_complete: float = 0.95          # a pass is complete when this share of the survivors crossed it
    probe_halo_sigma: float = 2.5        # probe halo: beyond this many sigma from the own turn centre
    hist_bin_mm: float = 1.0             # radial histogram of the valley / peak ratio
    hist_smooth_bins: int = 3            # moving average over this many bins
    # collimation trade-off (one slit at an early pass, turn centres held, no space-charge feedback)
    slit_budgets: tuple = (0.05, 0.10, 0.20, 0.30)  # share of the survivors the slit may take
    slit_passes: tuple = (1, 2, 3)       # passes a slit may sit at
    slit_min_share: float = 0.9          # a pass needs this share of the survivors to carry a slit
    slit_splits: int = 11                # tried splits of a radial budget between the inner and the outer jaw
    # space-charge twin in guiding-centre coordinates (twins.vortex_twin)
    twin_azimuths: int = 32              # azimuths per turn window over which radius and time are averaged
    twin_psi0_deg: float = 45.0          # azimuth where the turn windows start (along the motion)
    twin_min_share: float = 0.98         # a turn window counts when this share of the common survivors fills it
    twin_boot: int = 200                 # paired bootstrap resamples of the errors (0: none)
    twin_boot_seed: int = 1              # their random seed
    # operating point of a voltage x phase bunch grid (pick_operating_point)
    pick_e_min_mev: float = 2.9          # the bunch must reach this final energy
    pick_t_tol: float = 0.005            # within this (absolute) transmission of the best, the smallest dE/E wins
    pick_v_max: float = 1.10             # cap on the dee-voltage scale (wall power grows as V^2)


@dataclasses.dataclass(frozen=True)
class Metric:
    direction: str                       # 'max' | 'min' | 'target' (a constraint) | 'info' (a diagnostic)
    unit: str
    meaning: str
    fn: Optional[Callable] = None        # user metric: fn(run, score, cfg) -> float; None = computed by score_run


# Scalar score keys a campaign may optimise. Names with a number carry a ScoreConfig default (halo3: halo_sigma 3,
# I_2pct: energy window 0.02, acc90: level 0.9, valley_rad20: slit budget 0.2, *_t3: proxy turn 3, the time-turn
# [3, 4) revolutions); other settings rename the key (halo_sigma = 2.5 writes halo2.5), so a campaign names its metrics
# after its own [score]. Spearman rank correlations are over the campaign's 53 runs of one injected beam (2026-11-01).
METRICS = {
    # objectives
    'I_final': Metric('max', 'mA', "final current: transmission T x the injected current I0 (T: the run's full-beam "
                                   "transmission, the tracked real survivors in a ghost run; I0: the run's "
                                   "space-charge current, else the hand-off file's)"),
    'valley_all': Metric('min', '1', "share of the survivors' probe crossings in the middle third between turn "
                                     "centres, all probes pooled (8 every 45 deg), last 3 complete passes: particles "
                                     "between turns; never one probe alone (the 45-deg probe flatters)"),
    'ratio': Metric('min', '1', "r-l aspect ratio of the bunch core at the last turn (95 % Mahalanobis core, bunch "
                                "frame, mean over the instants of the turn)"),
    # constraints
    'E_med': Metric('target', 'MeV', "median final energy of the survivors: reach the design energy"),
    'dE_E_robust': Metric('min', '1', "robust (MAD) relative energy spread of the survivors, end of run (Spearman 0.71 "
                                      "with turn_sigR); turn_sigR <= 3 mm needs <= ~1.3 % at R = 330 mm"),
    'turn_sigR': Metric('min', 'mm', "robust radial width of the last complete pass, mean over the probes: <= 3 mm for "
                                     "8 sigma at a 24 mm turn separation"),
    'sigR_E': Metric('min', 'mm', "sigma_R_E = R_last / 2 x dE_E_robust: the turn width the energy spread alone makes, "
                                  "the floor of turn_sigR (R ~ p ~ sqrt(E))"),
    'fan_proj_last': Metric('min', 'mm', "spread of the survivors' orbit centres projected on the radial direction, "
                                         "rms over the azimuths, last full turn: what the centre fan adds to the turn "
                                         "width; hold below ~2 mm"),
    # further current and acceptance measures
    'T': Metric('max', '1', "transmission to the end of the run"),
    'W_eff': Metric('max', 'deg', "survival curve integrated over the hand-off release phase: the width of an "
                                  "equivalent 100 % flat top; ranks I_final (Spearman 1.00: the same final survival)"),
    'acc90_width': Metric('max', 'deg', "widest release-phase window with survival >= 90 %"),
    'S30': Metric('max', '1', "survival of the particles released within +-30 RF deg of the bunch centre"),
    'I_2pct': Metric('max', 'mA', "current within +-2 % of the median final energy"),
    'I_2pct_best': Metric('max', 'mA', "current in the best-placed +-2 % energy window (robust to a split spectrum)"),
    # the fixed-azimuth readout (metrics.readout_metrics, ScoreConfig.readout_*): the beam at one virtual probe on
    # the last pass of the run's turns. The end-of-run keys above take the last record, which depends on where the
    # run stopped: there (2026-11-04) inside the next gap's field, so its energies are a mid-gap snapshot
    'I_2pct_best_az': Metric('max', 'mA', "current in the best-placed +-2 % energy window at the readout (the "
                                          "extraction energy acceptance)"),
    'I_final_az': Metric('max', 'mA', "current crossing the readout: T_az x I0"),
    'T_az': Metric('max', '1', "share of the tracked bunch crossing the readout"),
    'I_2pct_az': Metric('max', 'mA', "current within +-2 % of the median energy at the readout"),
    'I_1pct_best_az': Metric('max', 'mA', "current in the best-placed +-1 % energy window at the readout"),
    'I_5pct_best_az': Metric('max', 'mA', "current in the best-placed +-5 % energy window at the readout (about the "
                                          "extraction channel's -6/+4 % acceptance)"),
    'E_med_az': Metric('target', 'MeV', "median energy at the readout: reach the design energy"),
    'E_2pct_best_az': Metric('target', 'MeV', "centre of the best-placed +-2 % window at the readout"),
    'dE_E_robust_az': Metric('min', '1', "robust (MAD) relative energy spread at the readout"),
    'eps_z_n_az': Metric('min', 'mm mrad', "normalised rms vertical emittance of the beam at the readout"),
    'eps_z_n_2pct_az': Metric('min', 'mm mrad', "normalised rms vertical emittance of the particles in the "
                                                "best-placed +-2 % window at the readout: the beam to extract"),
    'sig_z_2pct_az': Metric('min', 'mm', "rms height of the particles in that window"),
    'z_p95_az': Metric('min', 'mm', "95 % of |z| of the beam at the readout: what a vertical aperture sees"),
    # the vertical extent over the whole last turn (metrics.vertical_extent, its n_snapshots instants): free of the
    # readout's waist (a breathing envelope read at one azimuth; 0107 of campaign_v6r60o_c1: z_p95_az 7.99 mm, 12.2 mm
    # as the turn mean)
    'z_out10_last': Metric('min', '1', "share of the survivors beyond |z| = 10 mm (score.z_limit_mm) at any instant "
                                       "of the last turn: the part of the beam a +-10 mm aperture would meet"),
    'z_p95_last': Metric('min', 'mm', "95 % of |z| of the survivors, mean over the instants of the last turn"),
    'z_p95_max_last': Metric('min', 'mm', "95 % of |z| of the survivors at the last turn's widest instant"),
    'az_cover': Metric('info', '1', "share of the final survivors that crossed the readout (1: it sees them all)"),
    # beam quality diagnostics
    'turn_valley_max': Metric('min', '1', "valley share at the worst probe"),
    'turn_sep': Metric('max', 'sigma', "turn separation over the mean turn width, mean over the probes"),
    'halo3': Metric('min', '1', "share of the last-turn survivors beyond 3 sigma of the Gaussian-equivalent core"),
    'lag_rms': Metric('min', 'RF deg', "rms final phase lag of the survivors"),
    'fan_proj_max_last': Metric('min', 'mm', "the centre fan's radial projection at the worst azimuth, last full turn"),
    'fan_spread_last': Metric('min', 'mm', "rms spread of the survivors' orbit centres (2D), last full turn"),
    'coh_last': Metric('min', 'mm', "coherent orbit-centre offset (mean of the particles' centres), last full turn"),
    'valley_no_halo': Metric('info', '1', "valley share without the last-turn halo (ideal halo collimation)"),
    'valley_rad20': Metric('info', '1', "valley share after the best single radial slit taking 20 % of the survivors "
                                        "(its settings: best_slit.20.*)"),
    'I_rad20': Metric('info', 'mA', "final current after that slit"),
    'rot_pred_sum': Metric('info', 'deg', "core rotation in (r, l) the bunch's own central field predicts, summed over "
                                          "the turns (vortex < 0): ranks designs by vortex strength"),
    'rot_meas_sum': Metric('info', 'deg', "measured core rotation in the instantaneous bunch frame, summed over the "
                                          "turns: blind to the vortex (orbit-centre offsets), see twins.vortex_twin"),
    'slope': Metric('min', '1', "compression d(final phase lag) / d(release phase) of the survivors (1 = phase "
                                "conserved, < 1 = compressed); from the final lag, so not an early proxy (Spearman "
                                "0.88 with the final ratio)"),
    # low-fidelity proxies (turns 0-3 only)
    'sig_r_t3': Metric('min', 'mm', "core sigma_r at turn 3: proxy for valley_all (Spearman 0.94) and turn_sigR "
                                    "(0.72)"),
    'ratio_t3': Metric('min', '1', "core aspect ratio at turn 3: proxy for the final ratio (Spearman 0.77)"),
    'compress_t3': Metric('min', '1', "compression slope from the turn-3 turn-mean phase lag: candidate proxy for the "
                                      "final ratio (the final-lag slope has 0.88); correlation not yet tested"),
}


def register_metric(name, fn, direction='info', meaning='', unit='', replace=False):
    """Add a user metric: ``fn(run, score, cfg) -> float`` (``run`` a ``runs.RunData``, ``score`` the dict of the
    built-in metrics, ``cfg`` the ``ScoreConfig``). ``score_run`` stores its value as ``score[name]``; a name that a
    built-in metric already produces is refused there, a name already in ``METRICS`` here (unless ``replace``)."""
    if not callable(fn):
        raise TypeError(f"metric {name!r}: fn is not callable")
    if direction not in DIRECTIONS:
        raise ValueError(f"metric {name!r}: direction {direction!r} is not one of {DIRECTIONS}")
    if name in METRICS and not replace:
        raise ValueError(f"metric {name!r} is already defined")
    METRICS[name] = Metric(direction, unit, meaning, fn)
    return METRICS[name]
