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

## Retired

Not ported, each with the reason or what replaces it:
- The 2D model build and solve, and the 100-particle timing probe that sized N from `HCHC60_MAX_RUN_S`. Neither
  changes a result; N is set directly.
- The flat z = 0 reference (`HCHC60_REF_FLAT`): the reference particle starts on the hand-off plane.
- `HCHC60_VP_GRID`, `HCHC60_VP_BUNCH` (voltage x phase grids): run several phases or a campaign.
- `HCHC60_ZCENTER` (re-centring the bunch in z and v_z): a fudge.
- Ghost runs (`HCHC60_GHOST_PRISMS`, `HCHC60_METAL_MODEL3D_PKL`), bar iterations (`HCHC60_BAR_*`), the RF-cavity-map
  blend (`HCHC60_VOLT_TAPER`, the BEM + map composite): the BEM-only path replaced them. A cache that carries an RF
  map is refused.
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

`pcr track` was checked against `MiniDAR_bunch_simulation_3d.py` on the same cached fields, particle for particle
(`docs/pipeline.md`, "Validation of the port"). The numbers are the regression baseline for later changes.
