"""Scoring of 3D bunch-tracking runs.

    from PyCentralRegion.analysis import load_run, score_run, save_score, ScoreConfig
    run = load_run('results/my_design')                         # a `pcr track` output folder (run.out_dir)
    score = score_run(run, ScoreConfig())                        # or section(cfg, 'score')
    score['I_final'], score['valley_all'], score['ratio']        # objectives.METRICS lists the scalars
    save_score(score, 'score.json')

    run = load_run('results', tag='_v1', handoff_dirs=['beams'], machine=section(cfg, 'machine'))   # a deck run
    twin = vortex_twin(run_sc_on, run_sc_off, score_on)          # the space-charge vortex of an SC-on / SC-off pair

``runs`` loads a run (both the ``pcr track`` layout and the original deck's) with the hand-off particle of every
track, and builds probe files from recorded tracks; ``metrics`` holds the metric functions and ``score_run``;
``objectives`` the ``[score]`` settings (``ScoreConfig``), the table of metrics a campaign may optimise (``METRICS``)
and ``register_metric`` for user metrics; ``twins`` measures the space-charge vortex from an SC-on / SC-off pair of
runs; ``plots`` draws a one-page summary figure. The machine (harmonic, loss limits, species) is the run's own: its
json, hand-off file and the [machine] table it was tracked with (``RunData.machine``), else the ``MachineConfig``
defaults (``metrics.run_constants``).
Importing the package needs only numpy; scipy is used by a few metrics, h5py and PyPATools to read hand-off files,
matplotlib by the figure.
"""
from .objectives import DIRECTIONS, METRICS, Metric, ScoreConfig, register_metric
from .metrics import (score_run, run_state, RunState, run_constants, machine_value, tracked_species, survivors,
                      alive_mask, ghost_first_step, loss_steps, bunch_coordinates, core_covariance,
                      core_variance_factor, shape_stats, roundness, ratio_without_fan, gauss_gradients,
                      vortex_prediction, one_turn_map, map_rotation, isochronous_field, turn_centres, centre_fan,
                      probe_crossings, reference_crossings, turn_structure, recorded_probe_turns, compare_probes,
                      slit_tradeoff, slit_settings, acceptance, compression_slope, final_phase_lag, turn_lag,
                      release_budget, handoff_survival, energy_metrics, energy_window_best, energy_window_median,
                      radial_width_split, pick_operating_point, spearman, score_column, proxy_correlations,
                      campaign_summary)
from .runs import (RunData, Handoff, load_run, load_track_run, load_legacy_run, legacy_handoff, legacy_tags,
                   handoff_particles, handoff_track_ids, load_probe, virtual_probe, beam_label, clean, save_score,
                   load_score, save_table)
from .twins import vortex_twin
from .plots import summary_figure
