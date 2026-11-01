# Porting: from the design deck to the package

The 3D central-region work of October 2026 lived in a design deck: scripts configured by about 90 environment
variables, driven by PowerShell runners (`D:\MIT Dropbox\...\RFQ Direct Injection\CentralRegion`). The package takes
over what is still in use, configured by run files (TOML + dataclasses), and reads no environment variable. This page
says where each piece went, what was retired, and what changes for the deck scripts once this branch is merged.

## Where things went

| deck / old repo | package | command |
|---|---|---|
| `MiniDAR_bunch_simulation_3d.py` (BEM-only path) | `simulate3d.py`: `fields_stage`, `BunchTracker`, `track_stage` | `pcr fields`, `pcr track` |
| deck `cr_geom_build.py` (2026-10-31) | `geometry/` | `pcr build` |
| `strategy_tools/score_runs.py` (run scores) | `analysis/` | `pcr score` |
| MOBO (new) | `mobo/` | `pcr evaluate`, `pcr optimize` |
| repo `scripts/cr_geom_build.py` (an older, diverged copy) | removed: `geometry/` replaces it | |
| repo `examples/12_electrodes3d_hchc60.py`, `13_inflector_in_central_region.py` (machine-bound: deck files and env vars) | removed: `examples/12_electrodes3d.py` (general), `pcr track` with `[inflector]` | |
| the env-var switches inside package modules | explicit arguments, set from the run config | |

## Tracking: deck variables -> run-file keys

`MiniDAR_bunch_simulation_3d.py` and the modules it imports, against what `pcr track` reads:

| deck variable | run-file key |
|---|---|
| `HCHC60_RESULT_PKL` | `paths.seed` |
| `HCHC60_VERIFY_TAG` | `run.name`, `run.out_dir` |
| `HCHC60_MODEL3D_PKL` | `paths.model3d` |
| `HCHC60_REUSE_FIELDS`, `HCHC60_FIELDS_TAG` | `paths.fields` (an existing cache is always reused) |
| `HCHC60_BMAP`, `HCHC60_CACHE_DIR` | `paths.bfield_3d`, `run.cache_dir` |
| `HCHC60_HANDOFF` | `paths.handoff` |
| `HCHC60_EFIELD`, `HCHC60_EFIELD_ZSHIFT_MM` | `paths.inflector_efield`, `inflector.zshift_m` |
| `HCHC60_HOUSING` | `paths.housing_step` |
| `MINIDAR_SI_ROT_DEG`, `_REFINE`, `_ZMAX_MM` | `inflector.rotation_deg`, `inflector.refine`, `inflector.refine_z_max_m` |
| `HCHC60_TARGET_MEV`, `HCHC60_MAX_RADIUS_M` | `machine.target_energy_mev`, `machine.r_max_m` |
| `HCHC60_DEE_KV`, `N_TURNS`, `SKIP_TURNS`, `R_MAX_CAV` | `machine.design_voltage_kv`, `n_turns_design`, `skip_turns_design`, `r_max_cavity_m` |
| `HCHC60_GAP_ANGLE_DEG`, `HCHC60_GAP_WIDTH_MIN_MM` | `machine.gap_angle_deg`, `machine.gap_width_min_m` |
| `MINIDAR_OPENING_DEG` | `machine.dee_opening_deg` |
| `HCHC60_BUNCH_PHASES` | `track.phases_deg` |
| `HCHC60_N_SCAN`, `HCHC60_N_FINAL` | `track.n_scan`, `track.n_final` (0 = the whole bunch) |
| `HCHC60_BUNCH_TURNS` | `track.turns` |
| `HCHC60_PUSHER`, `HCHC60_STEPS_PER_TURN` | `track.pusher`, `track.steps_per_turn` |
| `HCHC60_FAST3D`, `HCHC60_FAST3D_INTERP` | `track.fast3d`, `track.fast3d_interp` |
| `HCHC60_VOLT_SCALE` | `track.volt_scale` |
| `HCHC60_SC`, `HCHC60_SC_CURRENT_MA` | `track.sc`, `track.sc_current_ma` |
| `HCHC60_SC_NCELLS`, `HCHC60_SC_EVERY` | `track.sc_cells`, `track.sc_every` (scan); `track.sc_final_cells`, `track.sc_final_every` (final) |
| `HCHC60_SC_H_MM`, `HCHC60_SC_MAX_CELLS`, `HCHC60_SC_WINDOW_Q` | `track.sc_h_m`, `track.sc_max_cells`, `track.sc_window_q` |
| `HCHC60_SC_GPU`, `HCHC60_SC_FRAME` | `track.sc_gpu`, `track.sc_frame` |
| `HCHC60_SNAPSHOT_NS`, `HCHC60_N_TRACKS`, `HCHC60_TRACK_EVERY`, `HCHC60_PROBES` | `track.snapshot_ns`, `track.n_tracks`, `track.track_every`, `track.probes_deg` |
| `HCHC60_OBSTACLES` | `track.obstacles` |
| `MINIDAR_EXTRA_POSTS`, `MINIDAR_EXTRA_BOXES` | `track.extra_posts_mm`, `track.extra_boxes_mm` (arrays, mm) |
| `HCHC60_BFLAT_R_MM` | `magnet.flatten_r_m` |
| `HCHC60_BEM3D_SOLVER`, `PCR_GPU_FIT_SAFETY` | `fields.solver`, `fields.gpu_fit_safety` |
| `HCHC60_BEM_POT_DEVICE`, `_PRECISION`, `_CHUNK`, `_CHECK` | `fields.pot_device`, `pot_precision`, `pot_chunk`, `pot_check` |
| `MINIDAR_BEM_ONLY` | always: the BEM field is the whole accelerating field |
| `HCHC60_SPARSE_BEAM` | always: the final run records sparsely (snapshots + tracks) |

New, with no deck variable: `[slit]` (a radial slit as a second obstacle raster), `[btrim]` (a first-harmonic magnet
trim), and the inflector-rotation guard (a model records the turn it was built for).

## Building: deck variables -> run-file keys

The deck's `cr_geom_build.py` (its `CR_*` environment and command line) against what `pcr build` reads: `[geometry]`
keys unless another section is named.

| deck variable or flag | run-file key |
|---|---|
| `--pkl`, `--outline` | `paths.seed`, `paths.housing_outline` |
| `--pickle` (or `--build PREFIX`: `PREFIX_model3d.pkl`) | `paths.model3d` |
| `--handoff` | `emitting_plane_file` (empty: the export's own hand-off beside the outline, else `paths.handoff`) |
| `HCHC60_EXPORT` | `exit_plane_spec_file` (empty: the export's `exit_plane_spec.json`) |
| `CR_SI_ROT` unset | `inflector.rotation_deg = 0` |
| `CR_SI_ROT=corner` | `si_rotation = "corner"`, `gap0_on_plate = true`; `inflector.rotation_deg` = the tracker's `MINIDAR_SI_ROT_DEG` |
| `CR_SI_ROT=<deg>` | `si_rotation = "fixed"`, `gap0_on_plate = true`, `inflector.rotation_deg = <deg>` |
| `CR_APERTURE_MM`, `CR_APERTURE_R_MM` (deck defaults `40,40`, `148,247`), `--aperture` | `aperture_inner_mm`, `aperture_outer_mm`, `aperture_taper_center_mm`, `aperture_taper_length_mm` (`options.aperture_from_spec` converts the strings) |
| `--dummy-aperture` | `dummy_aperture_inner_mm`, `dummy_aperture_outer_mm` |
| `CR_GROUND`, `CR_STAR_TIP_GAP_MM`, `CR_GAP0_FACE_MM` | `ground`, `star_tip_gap_mm`, `gap0_face_mm` |
| `CR_DEE_INNER`, `CR_HUG_DS_MM`, `CR_HUG_RING_N`, `CR_DEE_RMIN` | `dee_inner`, `hug_ds_mm`, `hug_ring_n`, `[[geometry.dee_rmin]]` |
| `CR_DEE0_NOSE_MM`, `CR_NOSE_LC_MM`, `--dee0-clearance`, `--dee0-trim` | `dee0_nose_mm`, `nose_lc_mm`, `dee0_clearance_mm`, `dee0_trim_mm` |
| `CR_JUNCTION_R` / `--junction-r`, `--junction-tol`, `--no-snap` | `[[geometry.junctions]]`, `junction_tol_deg`, `snap_nodes = false` |
| `--gap-posts`, `--gap-windows`, `--gap-apertures`, `--gap0-posts`, `CR_EXTRA_BOXES` | `[[geometry.posts]]`, `[[geometry.windows]]`, `[[geometry.holes]]`, `gap0_posts_mm`, `[[geometry.boxes]]` (`options.*_from_spec`, `options.from_posts_json`) |
| `--post-size`, `--post-offset`, `--post-overhang`, `--post-shape`, `CR_POST_WALL_JOIN_MM` | `post_size_mm`, `post_offset_mm`, `post_overhang_mm`, `post_shape`, `post_wall_join_mm` |
| `--r-max`, `--dee-wall`, `--dummy-overlap`, `--cavity-clearance-inner` | `r_max_mm`, `dee_wall_mm`, `dummy_overlap_mm`, `housing_clearance_mm` |
| `--half-height`, `--brick-extend`, `--channel-back`, `--aperture-inset` | `half_height_mm`, `channel_extend_mm`, `channel_back_mm`, `aperture_inset_mm` |
| `--cluster-tol`, `--keep-every` | `cluster_tol_mm`, `keep_every` |
| `--lc`, `--lc-outer`, `--r-fine`, `--r-coarse`, `--lc-posts` | `lc_mm`, `lc_outer_mm`, `r_fine_mm`, `r_coarse_mm`, `lc_posts_mm` |
| `MINIDAR_MESH_ALGO`, `MINIDAR_MESH_MIN`, `CR_MESH_FLIP_MIN_DEG`, `CR_OCC_FUZZY_MM` | `mesh_algorithm`, `mesh_size_min_mm`, `mesh_flip_min_deg`, `occ_fuzzy_mm` |
| `--dees` | `plot_dees` (on by default) |
| `HCHC60_R_MAX_CAV`, `MINIDAR_BEM_DEE_KV`, the gap-width law | `[machine]`: `r_max_cavity_m`, `dee_voltage_kv`, `gap_angle_deg`, `gap_width_min_m` |

The deck computed the gap-width floor as 6.5 * 1e-3 = 0.006500000000000001 m. With that value (the worked example's
`machine.toml`) `pcr build` reproduces the deck's models bit for bit; with 0.0065 the metal is the same but 8 ground
faces mesh differently.

The deck's helper tools are subcommands of `python -m PyCentralRegion.geometry.build`: `MiniDAR_handoff_aperture.py`
-> `hole` (the gap-0 hole as a `[[geometry.holes]]` item) and `cut` (the hand-off cut to a circle on its plane),
`MiniDAR_edge_shift.py` -> `shift`, `cr_geom_build.py --sections` -> `sections`; `bounds` is new (the channel-height
bounds for a campaign's ranges). Not ported: `MiniDAR_place_posts.py` and `MiniDAR_junction_relocate.py` (convert
their spec and state JSON with `options.from_posts_json` and `options.junctions_from_json`; `options.to_toml` writes
the `[geometry]` table), `MiniDAR_slim_pkl.py`, and the viewers (`--view-tracks`, `--frames`, `--plot`, `--geo`).

New, with no deck variable: the channel bound at the housing exit's outermost radius, and the refusals of
`r_max_mm` beyond the cavity, `gap0_face_mm` without the star ground, and `dee0_trim_mm` together with
`dee0_clearance_mm`. Hardware blocks now run `post_overhang_mm` past the channel's highest point over their
footprint; the deck used the channel at one point, which leaves an opening in a steep taper.

## Scoring: strategy tools -> `analysis`

The run scoring of 2026-10-31 / 11-01 (`strategy_tools/` beside that session's notes, run on the deck's
`bunch_simulation3d<tag>_data.pkl`) against `PyCentralRegion.analysis`. Every function reads both layouts: a
`pcr track` folder (`runs.load_track_run`), and a deck results folder with a tag (`runs.load_legacy_run`, which finds
the run's hand-off file from its log); `runs.load_run` picks. The thresholds are `[score]` keys
(`configs/defaults/score.toml`), whose defaults reproduce the 2026-11-01 scores.

| strategy tool | analysis | score keys |
|---|---|---|
| `score_runs.py` | `metrics.score_run`, `runs.save_score`; `pcr score` for a `pcr track` folder | all (`objectives.METRICS` lists the scalars) |
| `aggregate_scores.py` | `runs.save_table` (csv), `metrics.proxy_correlations`, `metrics.spearman`, `metrics.campaign_summary`, `metrics.score_column`; `plots.summary_figure` draws one run (the 6-panel campaign figure is not ported) | |
| `track_probe.py`, `check_probe_vs_real.py` | `runs.virtual_probe` (the tracker's probe rules on recorded tracks, in the probe-file layout), `runs.load_probe`, `metrics.compare_probes`, `metrics.recorded_probe_turns` | `probes`, `turn_*`, `valley_all` |
| `particle_centres.py`, `centre_fan.py` | `metrics.turn_centres`, `metrics.centre_fan`, `metrics.ratio_without_fan` | `coh_*`, `fan_*`, `ratio_no_fan` |
| `phase_accept.py` | `metrics.acceptance`, `metrics.handoff_survival` | `acc<level>_*`, `S30`, `S30_40`, `band<q>_*` |
| `release_phase_budget.py` | `metrics.release_budget`, `metrics.compression_slope`, `metrics.final_phase_lag`, `metrics.radial_width_split` | `slope`, `compress_t<t>`, `lag_rms`, `r_rms_E`, `r_rms_rest`, `dR_dE` |
| `run_roundness.py` | `metrics.roundness` | `ratio` (the same core statistic over the last turn's 16 instants: `[score] n_snapshots`, `core_q`) |
| `compare_runs.py` | `metrics.score_run` per run, then `runs.save_table` | `I_final`, `T`, `I_<p>pct`, `I_<p>pct_best`, `ratio` |
| `vortex/` (the SC-off twin) | `twins.vortex_twin` (`[score] twin_*`) | its own result |
| `split_diag.py` | not ported (a one-off diagnosis of one design's two-peak energy) | |

Differences, all deliberate:
- The core-variance factor is exact (0.842330 for `core_q` = 0.95; the campaign used 0.8423): the halo,
  vortex-prediction and one-turn-map keys differ from the campaign tables by up to 4e-4 relative.
- The strategy tools counted wall losses (r/z limits) as survivors in `release_phase_budget`, `centre_fan`,
  `compare_runs` and `phase_accept`; the port uses the score's survivors everywhere. They also mapped sampled bunches
  to the hand-off file wrongly (`PH[ids]`); the port takes the mapping the tracker records (`handoff_ids`), else
  the seeded draw.
- `pcr score` labels the beam with `[score] beam` (the worked example: `sp1Lo_c1`), else the hand-off file's stem.

From the deck:
- `MiniDAR_vp_bunch_pick.py` -> `metrics.pick_operating_point(rows, cfg)`, the same rule: reach `pick_e_min_mev` with
  a voltage scale <= `pick_v_max`, highest transmission, the smallest dE/E within `pick_t_tol` of it. `--e-min`,
  `--t-tol`, `--v-max` -> `[score] pick_e_min_mev` (2.9), `pick_t_tol` (0.005, what the runners passed; the script's
  own default was 0.01), `pick_v_max` (1.10). The rows are those of the deck's `vp_bunch<tag>.json`
  ({v, phase, transmission, e_final, de_e}). From `pcr track` runs, use one row per entry of track.json's `scan`:
  v = `volt_scale`, phase, transmission, e_final = `e_mean[-1]`, de_e = `e_std[-1] / e_mean[-1]`. The package has
  no voltage x phase grid stage (Retired).
- `MiniDAR_post_optimizer.py`'s metrics -> the roundness R = ln ratio of the 95 % core over 16 instants of the last
  turn in the bunch frame is `metrics.roundness`, the same implementation as the scores' `ratio`. Its ghost runs
  (`MiniDAR_ghost_posts.py`) are read from the deck's pickle (`ghosts` -> `RunData.ghost_ids` / `ghost_step`): a
  ghosted particle counts as lost from its ghost step (`metrics.ghost_first_step`; score keys `ghost_run`, `T` = the
  tracked real survivors in a ghost run, `T_trk`). Not ported: its selection model (`prisms`, `precompute`,
  `optimize`: the predicted T and ratio of a post set from one ghost run).
- `MiniDAR_probe_plot.py` -> not ported: histogram `runs.load_probe(file)['R']` or `runs.virtual_probe(run, a)['R']`;
  `plots.summary_figure` panel (c) shows turn separation and valley share per probe azimuth.
- `MiniDAR_loss_map.py`, `MiniDAR_halo_*.py`, `MiniDAR_collimator_search.py`, `MiniDAR_compare_col.py` -> not
  ported. One slit's trade-off is `metrics.slit_tradeoff` / `slit_settings` (score `I_rad<b>`, `valley_rad<b>`,
  `valley_ver<b>`, `best_slit.<b>.*`), and `[slit]` then tracks it.

## Orchestration: deck runners -> `pcr evaluate`, `pcr optimize`

The deck chained its stages in PowerShell runners that passed results by scraping printed `KEY=` lines and json
files. In the package one process runs one design's chain and writes one result file, and the MOBO driver chains
designs.

| deck | package |
|---|---|
| `RunMiniDAR_S4.ps1` (`-Stage geom` / `run`) | `pcr build`, `pcr fields`, `pcr track` |
| `RunMiniDAR_S4_gapstep.ps1` (fields -> voltage x phase grid -> pick -> every-particle run) | `pcr evaluate --config run.toml --out DIR`: build when `paths.model3d` is missing, fields when `paths.fields` is missing, then track and score into `DIR/result.json` (status, scores, error, wall_s, config). The grid and the pick: `track.phases_deg` lists or a campaign over `track.volt_scale` and `track.phases_deg`, then `analysis.metrics.pick_operating_point` (Scoring) |
| `RunMiniDAR_S4_variant.ps1` (a geometry rebuilt with one change) | `pcr evaluate` with `--set geometry.<key>=...` and new `paths.model3d`, `paths.fields`, `run.out_dir`; in a campaign, a `kind = "geometry"` variable, built once per geometry key under `<campaign_dir>/geometries/<key>/` |
| `RunMiniDAR_post_opt.ps1`, `MiniDAR_post_optimizer.py`, `MiniDAR_ghost_posts.py` (ghost-run post optimizer) | not ported (the tracker has no ghost mode); `pcr optimize` searches geometry variables with a full build and BEM per design, array items included (`geometry.posts.0.s_mm`, `geometry.windows.1.half_height_mm`) |
| `RunMiniDAR_S4_chain2.ps1` (gap by gap: gap-0 hole, then posts from the previous step's tracks) | the gap-0 hole and the hand-off cut: `python -m PyCentralRegion.geometry.build hole` / `cut` (Building); post and junction placement from tracks is not ported |
| `RunMiniDAR_S4_freqscan.ps1`, `MiniDAR_freqscan_pick.py` (f_rev x phase) | not ported: the seed's f_rev is the clock (`HCHC60_BUNCH_FREQ_MHZ`, Retired) |
| `RunMiniDAR_S1.ps1`, `RunMiniDAR_bunchopt.ps1`, `MiniDAR_optimize_*.py` (thin-gap optimisers) | not ported: they make the seed (`paths.seed`) and stay in the deck |
| `MiniDAR_vp_objective.py` | superseded in the deck; not ported |
| the runners' `KEY=` scraping, `-WaitFor`, `conda-hook.ps1` | `result.json` (`mobo.evaluate.read_result`); the driver runs one evaluation at a time with `[mobo] physics_python`, in a job object (Windows) or a process session (POSIX); its store `evals.jsonl` is the campaign's state |

## Retired

Not ported, each with the reason or what replaces it:
- The 2D model build and solve, and the 100-particle timing probe that sized N from `HCHC60_MAX_RUN_S`. Neither
  changes a result; N is set directly.
- The flat z = 0 reference (`HCHC60_REF_FLAT`): the reference particle starts on the hand-off plane.
- `HCHC60_VP_GRID`, `HCHC60_VP_BUNCH` (voltage x phase grids): run several phases or a campaign.
- `HCHC60_ZCENTER` (re-centring the bunch in z and v_z): a fudge.
- Ghost runs (`HCHC60_GHOST_PRISMS`, `HCHC60_METAL_MODEL3D_PKL`), bar iterations (`HCHC60_BAR_*`), the RF-cavity-map
  blend (`HCHC60_VOLT_TAPER`, the BEM + map composite): the BEM-only path replaced them. A cache that carries an RF
  map is refused. The scorer still reads the deck's ghost runs (Scoring).
- `HCHC60_BEM_SCALE` (use `track.volt_scale`), `HCHC60_BUNCH_FREQ_MHZ` (the seed's f_rev is the clock),
  `HCHC60_SC_OPTIONAL` (space charge never fails silently), `HCHC60_FINAL_NOSC` (track an SC-off twin as its own run).
- The deck's figures (`make_figure`, `make_snapshot_figure`, the model drawings) and `bem3d_orbit*.npy`. The
  reference trajectories are in `track_data.pkl`.

## What changes for the deck after the merge

Package modules used to read four settings from the environment. They are explicit arguments now, with the same
library defaults, and the run config sets them:

| was | now | library default |
|---|---|---|
| `HCHC60_BEM_POT_DEVICE`, `HCHC60_BEM_POT_PRECISION` | `GapFieldSolution.potential(device=, precision=)` | cpu, double |
| `PCR_GPU_FIT_SAFETY` | `solve_gap_field(gpu_fit_safety=)` | 1.25 |
| `HCHC60_SC_MOVE` | `SpaceChargeKick(move=)` | off |
| `HCHC60_BEM3D_GEOM_OUT` | `build_electrodes_3d(geometry_out=)` | none |

A deck script that sets one of these variables and calls the package without the argument gets the library default
after the merge. For the potential device that means the CPU in double precision: the same field, more slowly.

## Validation

`pcr build` was checked against the deck's builder (the same model bit for bit), `pcr track` against
`MiniDAR_bunch_simulation_3d.py` on the same cached fields (particle for particle), and `pcr score` against
`score_runs.py` (`docs/pipeline.md`, "Validation of the port"). The numbers are the regression baseline for later
changes.
