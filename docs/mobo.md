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
campaign. The subprocess inherits the driver's environment. When the driver's own PyCentralRegion is a source tree (a
checkout or an editable install), that `src` folder goes first on the evaluation's `PYTHONPATH`, so the evaluation
runs the code the driver runs; a copy installed in site-packages is never put on another interpreter's path. The
driver sets `MPLBACKEND=Agg` and, for a conda interpreter, `CONDA_PREFIX` and that environment's DLL folders (the CUDA
libraries do not load without them on Windows).

On Windows the evaluation runs in a job object, so stopping the driver in any way, a hard kill included, stops the
evaluation and every process it started. It also gets a hidden console of its own: Ctrl+C and Ctrl+Break reach the
driver alone, which stops the tree and records the evaluation interrupted, never failed. On POSIX the evaluation leads
a session of its own, which the driver ends (SIGTERM, then SIGKILL) when it stops. A SIGKILL of the driver itself
leaves that session running; the campaign lock, which the evaluation inherits, keeps a new driver off the campaign
until it ends.

## The campaign file

A campaign file holds `[mobo]` only. It names the run file every evaluation starts from; every key is documented in
`configs/defaults/mobo.toml`.

```toml
[mobo]
campaign_dir = "E:/results/mobo/my_campaign"   # the whole state of the campaign
base_config_file = "my_design.toml"             # the run file every evaluation starts from
physics_python = "C:/.../envs/physics/python.exe"
sampler = "gp"
n_startup = 10            # evaluations with values (warm start included, failed ones not) before the model takes over
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
- `configs/examples/toy/campaign.toml`: the analytic toy, 30 evaluations in about 8 s (17 s with the GP);
- `configs/examples/hchc1p5/mobo_pilot.toml`: cheap variables only (RF phase, dee voltage, the first-harmonic trim's
  two components), on the worked example's existing fields, 20 evaluations;
- `configs/examples/hchc1p5/mobo_campaign.toml`: the full campaign: the beam channel's aperture and the
  spiral-inflector rotation (geometry) with the RF phase, the dee voltage and the trim (cheap); 9 variables, 120
  evaluations, about 4.5-5 h (see Costs). Its header lists what the first pilot changed and what the loop does
  not see;
- `configs/examples/hchc1p5/mobo_slit.toml`: a radial slit for the chosen design, a campaign of its own (cheap: the
  slit's opening, azimuth and offset at one fixed pass), 20 evaluations.

## Variables

The sampler works on `u_<name>` in [0, 1], mapped linearly onto [`low`, `high`] (`log = true`: logarithmically).
With `integer = true` (integer `low` and `high`) every integer gets the same share of the cube.

The `target` is a dotted run-file key, the same as a `--set` override. An integer part indexes an array of tables:
`geometry.windows.1.half_height_mm`. The driver owns `paths.model3d`, `paths.fields` and `run.out_dir`; a variable
cannot set them.

Before anything runs, every override is checked against its section: the variables of each section together at every
corner of their box (a section's checks can relate its keys: a channel taper must start at r > 0), the campaign's
other variables at the base's values. With `pcr evaluate` as the evaluator an override must name one of the sections
it reads.
A misspelt key or an invalid corner fails then, instead of leaving every evaluation the same design or failing those
proposed near it. Bounds that need the electrode layout (the channel heights against the housing exit) are checked
at the build; `python -m PyCentralRegion.geometry.build bounds <base run toml>` prints them.

A started campaign keeps its variables. One added, removed, renamed, re-ranged or re-targeted once evaluations exist
would move every stored design under the sampler, so `pcr optimize`, `--dry-run` and `optimize-verify` refuse it:
run a changed search space in another `campaign_dir`. Objectives and constraints may change; they are re-read from the
stored scores, so a constraint added later judges the earlier evaluations too.

A variable has one of two kinds:

| kind | changes | cost of a new value |
|---|---|---|
| `cheap` | RF phase, dee voltage, magnetic trim, slit, extra posts | re-tracking only, on the same fields |
| `geometry` | the electrodes or anything the build or the BEM reads | a new build and 3D BEM |

The geometry key is a hash of what the build and the BEM read: the sections `[geometry]`, `[machine]`, `[inflector]`
and `[fields]` as the base run file writes them (with `fixed` and the fidelity's overrides), the paths `paths.seed`,
`paths.handoff`, `paths.housing_outline` and `paths.housing_step`, and the geometry variables' values. It holds
neither the code's defaults nor the files' contents: after a change of either, start another `campaign_dir`.
The taper's place (`aperture_taper_center_mm`, `aperture_taper_length_mm`) is left out where the dees' channel and
the dummy dees' each have one height at every radius (`GeometryConfig.idle_keys`): there it changes no electrode, it
only adds points to the aperture tool, so the designs of one flat channel share one key and one build, and the parts
list the keys left out (`idle`). No other key is left out (one that changes no electrode, such as `plot_dees`, still
keys the geometry), and every other design's key is the hash of its parts as it always was.
Evaluations with the same key share one model and one field map in `<campaign_dir>/geometries/<key>/`. A `cheap`
variable that targets one of these sections gets a warning, because its evaluations would share fields that do not
belong to them; while one sets a `[geometry]` key, no key is left out (a cheap channel height could taper a channel
the parts call flat).

After every new geometry whose fields were made, the driver runs `cheap_per_geometry` evaluations on it. They take
that evaluation's geometry values exactly (a warm start's included), so they share its key and its fields, and only
the cheap variables are proposed; an interrupted one does not count. A geometry whose build or BEM failed gets no
cheap evaluations: the next evaluation is a new geometry.

A campaign without geometry variables uses the base run file's own `paths.model3d` and `paths.fields`.

Useful targets:

| knob | target | kind |
|---|---|---|
| RF phase | `track.phases_deg` (`as_list = true`) | cheap |
| dee voltage | `track.volt_scale` | cheap |
| first-harmonic trim | `btrim.amplitude_cos_t`, `btrim.amplitude_sin_t` (its Cartesian components) | cheap |
| collimating slit | `slit.half_width_mm`, `slit.azimuth_deg`, `slit.offset_mm`, `slit.pass_index` (`integer = true`) | cheap |
| spiral-inflector rotation | `inflector.rotation_deg` (with `geometry.si_rotation = "fixed"`) | geometry |
| dee vertical aperture | `geometry.aperture_inner_mm`, `aperture_outer_mm`, `aperture_taper_center_mm`, `aperture_taper_length_mm` | geometry |
| a gap window | `geometry.windows.<i>.half_height_mm` | geometry |

Two knobs are awkward for any sampler in their plain form:
- The trim by amplitude and azimuth (`btrim.amplitude_t`, `btrim.phase_deg`): the azimuth is periodic, so -180° and
  +180° are one trim on opposite faces of the box, and it does nothing at zero amplitude. Its Cartesian components
  above, which add to the polar part, have neither a seam nor more than one zero.
- The slit: `slit.half_width_mm = 0` means no slit, which leaves its other coordinates without effect, next to small
  widths that pass almost no beam; and `slit.pass_index` is an unordered choice, not a scale. Give it a campaign of
  its own on the chosen design, with the pass fixed and an open slit in every design (`mobo_slit.toml`).

## Objectives and constraints

Both are read from the scores in `result.json` by dotted path. `pcr score` writes every metric; the ones an optimiser
may use are listed with their meaning in `PyCentralRegion.analysis.objectives.METRICS`.

A metric that no scored evaluation has (runs the scorer skipped do not count) stops the campaign: a typo must not feed
the sampler nothing. With `pcr evaluate` as the evaluator, a metric listed in `METRICS` is never taken for a typo,
since a run that lost its beam early lacks some of those legitimately (`valley_all` and the `turn_*` metrics need
probes with `score.probe_passes` complete passes). The dry run marks every objective or constraint metric that
`METRICS` does not list.

Constraints follow optuna's convention: a value <= 0 is feasible, so `metric >= value` is stored as `value - metric`.

An evaluation that finishes with unusable scores is a bad design, not a failure. That is the case when the scorer
skipped the run (fewer than `score.min_tracks` = 5000 tracks or `score.min_survivors` = 200 survivors), or when a
metric of an objective or a constraint is missing, null or not finite. It enters the study as an infeasible stand-in:
a complete trial with the worst value seen of every objective and constraint and one extra constraint, `scored`,
violated (1 for a stand-in, 0 for the scored trials; present only while a stand-in exists). Until every objective and
constraint has a value seen, stand-ins are pending trials. The store records why (`unusable`); the log and
`pcr optimize-status` show it (state `done!`, the reason under the table). Stand-ins count against `max_evals` and are
never in the Pareto set. The GP learns from their values; TPE ranks infeasible trials by their violation, so for TPE
the `scored` violation is what steers it away.

The scoring of the 109 earlier 3D runs (2026-11-01) suggests:

| | metric | why |
|---|---|---|
| maximise | `I_final` | final current, T × injected current [mA] |
| minimise | `valley_all` | share of the probe crossings between the turns, the 8 probes (every 45°) pooled. Never one probe alone: the 45° probe flatters. |
| minimise | `ratio` | aspect ratio of the bunch core at the last turn |
| constraint | `E_med >= 2.9` | reach the design energy [MeV] |
| constraint | `turn_sigR <= 3.0` | clean turns, 8 σ at a 24 mm turn separation [mm] |
| constraint | `dE_E_robust <= 0.013` | the energy-spread floor of `turn_sigR` at R = 330 mm (or `sigR_E <= 3.0`) |
| constraint | `fan_proj_last <= 2.0` | the centre fan's share of the turn width [mm] |

There is no vortex objective: at this bunch density the vortex cannot shape the beam in 7 turns.

## Fidelities

A fidelity is a named list of overrides, applied after `fixed` and before the variables:
- the sampler learns from `search_fidelity`;
- `pcr optimize-verify` re-evaluates chosen designs at `verify_fidelity` (or `--fidelity NAME`). It stores each next to
  its original (id `0007.verify`), so the search-to-verify shift of every score can be read off. A re-evaluation that
  failed or was interrupted runs again on the next `optimize-verify`; its earlier folder is kept as
  `0007.verify.attempt<k>`.

`timeout_min` stops an evaluation that runs longer, which is then recorded failed.

Let the fidelities differ in resolution only (particles, space-charge grid and interval), not in what is simulated
(turns, energies): then the search-to-verify shift of a score is its resolution error.

Score every design at the same turn too. By default a run ends at the turn its reference reaches
`machine.target_energy_mev`, or after `track.turns`. A design that accelerates more slowly then tracks an extra
turn, and its end-of-run scores jump with it: in the first pilot, two designs below a dee-voltage scale of about 0.97
needed an 8th turn. The worked example's campaigns therefore set `track.turns = 7` and
`track.stop_at_target_energy = false` in `fixed`, so every run tracks exactly 7 turns. The base design reaches 3 MeV
at its 7th turn, so its run should be the one it had with the stop (array for array at 200 particles without space
charge, `pcr_validation\fixes_2026-11-02\track`; with GPU space charge expect the same fates). A slow design ends
below 3 MeV, where the `E_med` constraint judges it: with `E_med >= 2.9` one that ends at 2.9-3.0 MeV stays feasible
(the pilot's design 0015, dee voltage × 0.95 at -29°, ends its 7th turn near 2.93 MeV: a mean of 2.92 in the pilot's
run, a median of 2.94 in a 200-particle rerun without space charge); `>= 3.0` asks for the design energy at the 7th
turn.

| fidelity | particles | SC | one evaluation on known fields |
|---|---|---|---|
| search | 10 000 | 64³ every 5 steps | about 1.6-1.9 min (see Costs) |
| verify | all (~98 000) | 128³ every step | about 7 min |

On the worked example (-15°) the search fidelity gives T = 50.79 % = 4.080 mA (the warm start), the verify fidelity
50.57 % = 4.063 mA, with the column test, the default. The deck's slice test (`track.obstacles = "raster"`) gave
50.40 % = 4.049 mA (the first pilot's warm start) and 50.09 % = 4.024 mA: it puts the channel's faces up to half a
millimetre off (the level ones half a millimetre low), but for slivers of metal or vacuum thinner than a level (146 at
77 nodes), and so stops, net, 39 and 471 particles more. The deck's slices also lack the ground at z = ±15 mm, which
puts the ground's faces near those heights off, those on the channel's taper at r = 113-127 mm (|z| = 14-15 mm) at
±15.5 mm, up to 1.5 mm into the metal, and six at r = 34 and 55 mm; the particles the columns stop on the taper's
faces (144 and 1511), the slices stop too, a median 19 steps later (`docs/pipeline.md`). Every search evaluation
tracks the same
10 000 particles of the hand-off bunch (drawn with `track.seed_final`), so the designs are compared on one sample. A
sample of 10 000 is worth about ±0.04 mA (±0.5 % in T) by itself, mostly the same for every design.

A cheaper search over the first turns would need early stand-ins for the final objectives. The candidates are:
- `sig_r_t3` for `valley_all` (Spearman 0.94 over the earlier runs);
- `ratio_t3` for `ratio` (0.77);
- `compress_t3`, not yet tested.

All three were correlated on the final survivors, while a turns-0-3 run knows only the particles alive at turn 3.
Re-check the correlations on that set before using them. The compression slope (`slope`), `dE_E_robust` and `W_eff`
come from the end of the run, so they are not early stand-ins.

## Samplers

Both samplers are constrained and multi-objective, and both read the same store, so a campaign can switch between
them. Prefer `gp` (it needs torch): on the toy, 40 evaluations reach 95 % of the true front's hypervolume with the GP
and 70 % with TPE, whose pinned cheap followers model only the free coordinates. `tpe` is the default because it needs
no torch. Without the `greenlet` package optuna's GP runs its acquisition restarts one after another, which costs
about 0.5 s per proposal and prints a warning.

## Running

```
pcr optimize        campaign.toml --dry-run          # check everything, print the next evaluation's command
pcr optimize        campaign.toml                    # run until max_evals
pcr optimize        campaign.toml --max-hours 10     # ... or for at most 10 h in this run
pcr optimize        campaign.toml --max-evals 40     # ... or to another campaign budget than max_evals
pcr optimize-status campaign.toml                    # evaluations, Pareto set, best values
pcr optimize-verify campaign.toml --pareto           # re-evaluate the Pareto set at the verify fidelity
pcr optimize-verify campaign.toml --ids 3 7          # ... or chosen evaluations (--fidelity NAME: another fidelity)
```

Every command takes `--set key=value` (repeatable), an override of the campaign file, e.g. `--set mobo.seed=3`. The
command line that `--dry-run` prints (and `command.txt` records) is quoted for cmd.exe; to paste it into Windows
PowerShell 5.1, put `--%` after the interpreter: `& <python> --% -m PyCentralRegion.cli evaluate ...`.

The campaign folder is the whole state. Running `pcr optimize` again continues where the last run stopped. Every
proposal follows from the store alone: the study is rebuilt from the store each time and the sampler seeded from
`seed` and the evaluation's index. A run stopped between evaluations, or after proposing one and before starting it
(that one then runs first), continues exactly as an uninterrupted run would have. A run stopped during an evaluation
loses that evaluation: the evaluations before it are the uninterrupted run's, the ones after it what the study
proposes without it.

The driver never schedules or restarts itself. Before its first evaluation, `pcr optimize` (and `optimize-verify`,
when it has work) checks that `physics_python` runs and finds the package and the evaluator module: the preflight,
a few seconds, logs in `<campaign_dir>/preflight/`. A wrong interpreter, or a `PYTHONPATH` without the package, then
stops the driver at once instead of failing every evaluation. The preflight does not import the physics stack (numba,
bempp, cupy), so a run also stops, with exit code 1, after `max_failed_in_a_row` (default 5) failed evaluations in a
row: a broken environment, not bad designs. Running it again continues. One lock file keeps a second driver off the
same campaign.

Stopping the driver ends the running evaluation with every process it started. The stop can be Ctrl+C, Ctrl+Break or
SIGTERM, closing its window, a kill, or the end of `--max-hours`. That evaluation is recorded failed ("interrupted"),
does not count against `max_evals` and is not run again; give its design as a `warm_start` point to evaluate it
anyway (an interrupted warm-start point is evaluated again).

A failed evaluation (the build, the BEM or the tracking failed, or a timeout) enters the study as a pending trial: the
sampler does not propose it again, and no objective values are invented for it. With `failed_as_infeasible = true` it
enters instead as an infeasible stand-in (see Objectives and constraints).

## The campaign folder

```
<campaign_dir>/evals.jsonl          append-only: one line per state change of an evaluation (the store)
<campaign_dir>/evals/<id>/          command.txt, stdout.log, stderr.log, result.json and the stage outputs
<campaign_dir>/evals/<id>.<fid>/    a re-evaluation at another fidelity (0007.verify); earlier tries <id>.<fid>.attempt<k>
<campaign_dir>/geometries/<key>/    geometry.json (what the key hashes), model3d.pkl with the build's side files
                                    (model3d.msh, model3d.step, model3d_build.json, model3d_dees.png) and fields.pkl
<campaign_dir>/preflight/           stdout.log, stderr.log of the check of the physics interpreter
<campaign_dir>/driver.lock          held by the running driver
```

An evaluation goes proposed → running → done | failed; `optimize-status` marks a done one with unusable scores
`done!`. Search evaluations are numbered 0000, 0001, ...

A geometry's model and field cache are each written under a temporary name (`model3d.part.pkl`, `fields.part.pkl`)
and renamed once complete, so a stop never leaves them cut short: once made, they serve every later evaluation of that
geometry (a stop during the BEM keeps the model, and a later evaluation makes only the fields). A build that failed or
was stopped before its model leaves side files that the next build of that geometry would refuse to overwrite (a
stopped one also its temporary files); the driver moves them all into `geometries/<key>/stopped-<id>/`, so the next
evaluation of that geometry builds afresh.

## Costs

Measured on the worked example (28 threads, one GPU):
- **A cheap evaluation** (the search fidelity on known fields) takes 95-108 s, median 97 s: the pilot's 20 evaluations
  (`mobo_pilot.toml`, 2026-11-02) took 33 min. Of that:
  - every evaluation process compiles the parallel 3D tracking kernel first, about 65 s (numba 0.64 reloads a cached
    parallel build single-threaded, so the cache is off on purpose);
  - tracking the 10 000 particles takes 17-22 s after that, 11-16 s of it the space-charge solves;
  - start-up, the system and the reference particle, the record files and the scoring take the other 13-14 s.
- **A new geometry** adds the build, about 20 s, and its fields: 233 s on the full-chain check of 2026-11-02 (the
  3D BEM solve 52 s, the field grid 59 s, and 122 s for the slices of the metal test, which the fields stage no
  longer cuts: the metal columns that replace them take under a second), so about 2 min now (derived, not yet timed
  as a whole), 485 s in the design deck's run of the same geometry (314 s of it the slices). A new geometry's search
  evaluation therefore takes about 4 min.
- **Verify tracking** of every particle with SC 128³ every step takes 6.5-7 min (397 s on the full-chain check, 429 s
  in the port validation), a verify evaluation on known fields about 7 min in all.
- **A campaign** of 120 evaluations with `cheap_per_geometry = 3` (30 geometries): 30 × 4 + 90 × 1.6 min, about
  4.5 h.
- **The column test**, the default since 2026-11-02 (the pilot ran the deck's slices): one search evaluation has been
  timed with it, the warm start outside a campaign (`pcr_validation\metal_columns_2026-11-02\gpu\warm_search_columns`):
  114 s, its kernel phase (the compile included) 80 s against the pilot's 66.5-70.6 s (67.4 s for the same warm
  start). Its verify twin's kernel phase was 12 s longer than the full-chain check's as well (98.5 s against 86.5 s),
  though the verify runs step about ten times the particles, so the extra is a fixed cost per process, the compile,
  not the test's stepping. The compile time varies from run to run: the CPU runs of that evening
  (`metal_columns_2026-11-02\track`) spent 71-96 s in that phase with either test, the code from before the column
  test 83 s, and repeats differ by up to 9 s. If it stays near 80 s, a cheap evaluation takes about 1.9 min and the
  campaign about 5 h. Its verify tracking took 396 s, as long as the full-chain check's 397 s.

## The toy campaign

`configs/examples/toy/campaign.toml` runs the whole loop (30 evaluations) in about 8 s, 17 s with the GP, without the
physics stack or a GPU: the analytic evaluator `python -m PyCentralRegion.mobo.toy` follows the same contract. The toy
covers subprocesses, the store, the geometry cache, both fidelities, failures (`toy.fail_above_mm`), unusable scores
(`toy.short_above_deg`: no `valley_all`; `toy.skip_above_deg`: the run is skipped) and process trees
(`toy.child_sleep_s`). It is ZDT1 dressed up as a central region, so its Pareto front is known
(`PyCentralRegion.mobo.toy`). `tests/test_mobo.py` runs it.
