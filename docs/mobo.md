# Multi-objective optimisation (`pcr optimize`)

`pcr optimize` searches for central-region designs that trade final current against turn separation and bunch
shape. Every design it tries goes through the whole 3D chain: electrode build, 3D BEM, bunch tracking with space
charge, scores. An evaluation costs minutes and the one GPU, so the driver runs one at a time. It picks each with a
sample-efficient multi-objective sampler, optuna's Gaussian process (`gp`) or TPE (`tpe`), both constrained.

Step-by-step commands for a real campaign: [HOWTO_mobo_campaign.md](HOWTO_mobo_campaign.md).

## Two environments

| process | environment | needs |
|---|---|---|
| the driver (`pcr optimize ...`) | small: `pip install -e .[mobo]` | numpy, scipy, optuna >= 5, torch (CPU is enough; `gp` only) |
| each evaluation | the physics environment, `[mobo] physics_python` | numba, PyPATools, bempp, gmsh, cupy |

The driver never imports the physics modules. Each evaluation is a subprocess:

```
<physics_python> -m PyCentralRegion.cli evaluate --config <base run toml> --set key=value ... --out <campaign>/evals/<id>
```

It writes `result.json`: `{"status": "ok" | "failed", "scores", "error", "wall_s", "config"}`.

Running evaluations as subprocesses means a crash, a hang or a GPU out-of-memory costs one evaluation, never the
campaign. The subprocess inherits the driver's environment, `PYTHONPATH` included, so a source checkout runs as it
is. The driver sets `MPLBACKEND=Agg` and, for a conda interpreter, `CONDA_PREFIX` and that environment's DLL folders
(the CUDA libraries do not load without them on Windows). On Windows the evaluation runs in a job object: stopping the
driver in any way, a hard kill included, stops the evaluation and every process it started.

## The campaign file

A campaign file holds `[mobo]` only. It names the run file every evaluation starts from; every key is documented in
`configs/defaults/mobo.toml`.

```toml
[mobo]
campaign_dir = "E:/results/mobo/my_campaign"   # the whole state of the campaign
base_config_file = "my_design.toml"             # the run file every evaluation starts from
physics_python = "C:/.../envs/physics/python.exe"
sampler = "gp"
n_startup = 10            # evaluations (warm start included) before the model replaces random sampling
max_evals = 60            # search evaluations of the whole campaign; a rerun continues to it
cheap_per_geometry = 3    # after every new geometry: this many evaluations on it with only the cheap variables free
fixed = ["btrim.r_min_m=0.05"]                  # overrides of every evaluation
warm_start = [{}]         # {} = the base design itself; other points name variables in physical units

[[mobo.variables]]
name = "phase"
target = "track.phases_deg"
low = -40.0
high = 0.0
as_list = true            # the target is a list: it is set to [value]

[[mobo.objectives]]
metric = "I_final"
direction = "maximize"

[[mobo.constraints]]
metric = "E_med"
op = ">="
value = 2.9

[mobo.fidelities.search]
overrides = ["track.n_final=10000", "track.sc_final_cells=64", "track.sc_final_every=5"]
timeout_min = 30.0
```

Worked examples:
- `configs/examples/toy/campaign.toml`: the analytic toy, which runs in seconds;
- `configs/examples/hchc1p5/mobo_pilot.toml`: cheap variables only, on the worked example's existing fields.

## Variables

The sampler works on `u_<name>` in [0, 1], mapped linearly onto [`low`, `high`] (`log = true`: logarithmically).
With `integer = true` every integer gets the same share of the cube.

The `target` is a dotted run-file key, the same as a `--set` override. An integer part indexes an array of tables:
`geometry.windows.1.half_height_mm`. The driver owns `paths.model3d`, `paths.fields` and `run.out_dir`; a variable
cannot set them.

Before anything runs, every target is applied at both ends of its range and checked against its section. A misspelt
key fails then, instead of leaving every evaluation the same design.

A variable has one of two kinds:

| kind | changes | cost of a new value |
|---|---|---|
| `cheap` | RF phase, dee voltage, magnetic trim, slit, extra posts | re-tracking only, on the same fields |
| `geometry` | the electrodes or anything the build or the BEM reads | a new build and 3D BEM |

The geometry key is a hash of everything the build and the BEM read: the sections `[geometry]`, `[machine]`,
`[inflector]` and `[fields]` and the files `paths.seed`, `paths.handoff`, `paths.housing_outline` and
`paths.housing_step`, after all overrides. Evaluations with the same key share one model and one field map in
`<campaign_dir>/geometries/<key>/`. A `cheap` variable that targets one of these sections gets a warning, because its
evaluations would share fields that do not belong to them.

After every new geometry the driver runs `cheap_per_geometry` evaluations on it, with the geometry coordinates held
and only the cheap ones proposed. Its fields are then used several times before the next build.

A campaign without geometry variables uses the base run file's own `paths.model3d` and `paths.fields`.

Useful targets:

| knob | target | kind |
|---|---|---|
| RF phase | `track.phases_deg` (`as_list = true`) | cheap |
| dee voltage | `track.volt_scale` | cheap |
| first-harmonic trim | `btrim.amplitude_t`, `btrim.phase_deg` | cheap |
| collimating slit | `slit.half_width_mm`, `slit.offset_mm`, `slit.pass_index` (`integer = true`) | cheap |
| spiral-inflector rotation | `inflector.rotation_deg` (with `geometry.si_rotation = "fixed"`) | geometry |
| dee vertical aperture | `geometry.aperture_inner_mm`, `aperture_outer_mm`, `aperture_taper_center_mm`, `aperture_taper_length_mm` | geometry |
| a gap window | `geometry.windows.<i>.half_height_mm` | geometry |

## Objectives and constraints

Both are read from the scores in `result.json` by dotted path. `pcr score` writes every metric; the ones an optimiser
may use are listed with their meaning in `PyCentralRegion.analysis.objectives.METRICS`.

A metric that no scored evaluation has stops the campaign: a typo must not feed the sampler nothing.

Constraints follow optuna's convention: a value <= 0 is feasible, so `metric >= value` is stored as `value - metric`.

The scoring of the 108 earlier 3D runs (2026-11-01) suggests:

| | metric | why |
|---|---|---|
| maximise | `I_final` | final current, T × injected current [mA] |
| minimise | `valley_all` | particles between the turns, mean of 8 probes. Never one probe alone: the 45° probe flatters. |
| minimise | `ratio` | aspect ratio of the bunch core at the last turn |
| constraint | `E_med >= 2.9` | reach the design energy [MeV] |
| constraint | `turn_sigR <= 3.0` | clean turns, 8 σ at a 24 mm turn separation [mm] |
| constraint | `dE_E_robust <= 0.013` | the energy-spread floor of `turn_sigR` at R = 330 mm (or `sigR_E <= 3.0`) |
| constraint | `fan_proj_last <= 2.0` | the centre fan's share of the turn width [mm] |

There is no vortex objective: at this bunch density the vortex cannot shape the beam in 7 turns.

## Fidelities

A fidelity is a named list of overrides, applied after `fixed` and before the variables:
- the sampler learns from `search_fidelity`;
- `pcr optimize-verify` re-evaluates chosen designs at `verify_fidelity`. It stores each next to its original (id
  `0007.verify`), so the search-to-verify shift of every score can be read off.

`timeout_min` stops an evaluation that runs longer, which is then recorded failed.

| fidelity | particles | SC | one evaluation on known fields |
|---|---|---|---|
| search | 10 000 | 64³ every 5 steps | about 2 min (see Costs) |
| verify | all (~98 000) | 128³ every step | about 8 min |

On the worked example (-15°) the search fidelity gave T = 50.45 % = 4.053 mA, the verify fidelity 50.09 % =
4.024 mA.

A cheaper search over the first turns would need early stand-ins for the final objectives. The candidates are:
- `sig_r_t3` for `valley_all` (Spearman 0.94 over the earlier runs);
- `ratio_t3` for `ratio` (0.77);
- `compress_t3`, not yet tested.

All three were correlated on the final survivors, while a turns-0-3 run knows only the particles alive at turn 3.
Re-check the correlations on that set before using them. The compression slope (`slope`), `dE_E_robust` and `W_eff`
come from the end of the run, so they are not early stand-ins.

## Running

```
pcr optimize        campaign.toml --dry-run          # check everything, print the next evaluation's command
pcr optimize        campaign.toml                    # run until max_evals
pcr optimize        campaign.toml --max-hours 10     # ... or for at most 10 h in this run
pcr optimize-status campaign.toml                    # evaluations, Pareto set, best values
pcr optimize-verify campaign.toml --pareto           # re-evaluate the Pareto set at the verify fidelity
pcr optimize-verify campaign.toml --ids 3 7          # ... or chosen evaluations
```

The campaign folder is the whole state. Running `pcr optimize` again continues where the last run stopped, and
proposes exactly what an uninterrupted run would have: the sampler of every proposal is seeded from `seed` and the
evaluation's index, and the study is rebuilt from the store each time.

The driver never schedules or restarts itself. Stopping it ends the running evaluation with every process it
started. The stop can be Ctrl+C, closing its window, a kill, or the end of `--max-hours`. That evaluation is
recorded failed ("interrupted"), does not count against `max_evals` and is not run again; give its design as a
`warm_start` point to evaluate it anyway. One lock file keeps a second driver off the same campaign.

A failed evaluation enters the study as a pending trial: the sampler does not propose it again, and no objective
values are invented for it. With `failed_as_infeasible = true` it enters instead as a complete trial with the worst
values seen and a violated constraint.

## The campaign folder

```
<campaign_dir>/evals.jsonl          append-only: one line per state change of an evaluation (the store)
<campaign_dir>/evals/<id>/          command.txt, stdout.log, stderr.log, result.json and the stage outputs
<campaign_dir>/geometries/<key>/    model3d.pkl, fields.pkl and geometry.json of one geometry
<campaign_dir>/driver.lock          held by the running driver
```

An evaluation goes proposed → running → done | failed. Search evaluations are numbered 0000, 0001, ...

An evaluation stopped while it made its geometry's model or fields has its partial files renamed
(`*.stopped-<id>`), so the next evaluation of that geometry makes them afresh.

## Costs

- **Every evaluation process** compiles the parallel 3D tracking kernel first, about 65 s on 28 threads. numba 0.64
  reloads a cached parallel build single-threaded, so the cache is off on purpose.
- **Search tracking** of 10 000 particles on known fields takes 20-25 s after that. A cheap evaluation therefore takes
  about 1.5-2 min, most of it the compile.
- **A new geometry** adds the build (about 1 min) and the 3D BEM solve.
- **Verify tracking** of every particle with SC 128³ every step takes about 7 min (429 s on the worked example).

## The toy campaign

`configs/examples/toy/campaign.toml` runs the whole loop in seconds without the physics stack or a GPU: the analytic
evaluator `python -m PyCentralRegion.mobo.toy` follows the same contract. The toy covers subprocesses, the store, the
geometry cache, both fidelities and failures. It is ZDT1 dressed up as a central region, so its Pareto front is known
(`PyCentralRegion.mobo.toy`). `tests/test_mobo.py` runs it.
