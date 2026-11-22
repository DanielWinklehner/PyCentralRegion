# Multi-objective optimisation (`pcr optimize`)

`pcr optimize` searches for central-region designs that trade final current against turn separation and bunch
shape. Every design it tries goes through the whole 3D chain: electrode build, 3D BEM, bunch tracking with space
charge, scores. An evaluation costs minutes and the one GPU, so the driver runs one at a time. It picks each with a
sample-efficient multi-objective sampler, optuna's Gaussian process (`gp`) or TPE (`tpe`), both constrained.

Step-by-step commands for a real campaign: [HOWTO_mobo_campaign.md](HOWTO_mobo_campaign.md).

## Two environments

| process | environment | needs |
|---|---|---|
| the driver (`pcr optimize ...`) | small: `pip install -e .[mobo]` | numpy, scipy, optuna >= 5, torch (CPU is enough; `gp` only), greenlet (`gp`, faster proposals: "Samplers") |
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
campaign_dir = "${work}/mobo/my_campaign"      # the whole state of the campaign (${work}: a root, docs/pipeline.md)
base_config_file = "my_design.toml"             # the run file every evaluation starts from
physics_python = "/path/to/envs/physics/bin/python"   # "" = the driver's own interpreter
sampler = "gp"
n_startup = 10            # evaluations with values (warm start included, failed ones not) before the model takes over
max_evals = 60            # search evaluations of the whole campaign; a rerun continues to it
cheap_per_geometry = 3    # after every new geometry: this many evaluations on it with only the cheap variables free
fixed = ["track.stop_at_target_energy=false"]   # overrides of every evaluation
warm_start = [{}]         # {} = the base design itself; other points name variables in physical units

[[mobo.variables]]
name = "phase"
target = "track.phases_deg"
low = 10.0
high = 60.0
as_list = true            # the target is a list: it is set to [value]

[[mobo.objectives]]
metric = "I_final"
direction = "maximize"

[[mobo.constraints]]
metric = "E_med"
op = ">="
value = 0.75

[mobo.fidelities.search]
overrides = ["track.n_final=10000", "track.sc_final_cells=64", "track.sc_final_every=5"]
timeout_min = 30.0
```

Worked examples:
- `configs/examples/toy/campaign.toml`: the analytic toy, 30 evaluations in about 8 s (17 s with the GP);
- `configs/examples/demo/campaign.toml`: the demo machine (`python -m PyCentralRegion.demo` makes its inputs): the
  dee channel's height inside the taper (geometry) and the RF phase (cheap), two objectives and an energy
  constraint, 12 evaluations of the whole chain on a CPU.

A first-harmonic magnetic trim (`btrim.amplitude_cos_t`, `btrim.amplitude_sin_t`) is a cheap variable a campaign
can add. Add it only for a machine with trim coils, or label its results "with trim" and run it in a
`campaign_dir` of its own.

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
`paths.handoff`, `paths.housing_outline` and `paths.housing_step`, and the geometry variables' values. With posts
placed by rule (`[post_rule] enabled`, "Posts in the loop") it also holds `[post_rule]` and the maps the bare run
reads, `paths.bfield_3d` and `paths.inflector_efield`. It holds
neither the code's defaults nor the files' contents: after a change of either, start another `campaign_dir`.
The taper's place (`aperture_taper_center_mm`, `aperture_taper_length_mm`, `aperture_taper_start_mm`) is left out
where the dees' channel and the dummy dees' each have one height at every radius (`GeometryConfig.idle_keys`): there
it changes no electrode, it
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
| magnet level | `magnet.scale` | cheap |
| spiral-inflector rotation | `inflector.rotation_deg` (with `geometry.si_rotation = "fixed"`; gap 0 then leaves the plate where the turned plate meets its ray, which runs along the plate as it turns) | geometry |
| spiral-inflector rotation, gap 0 held on its plate point | `geometry.si_rotation_offset_deg` (with `geometry.si_rotation = "corner"`: off the corner rule; `docs/si_rotation_gap0.md`) | geometry |
| gap 0's leave point and tilt | `geometry.gap0_leave_mm`, `geometry.gap0_tilt_deg` (connected gap lines; the tilt with the star ground, which gets a nose) | geometry |
| dee opening | `machine.dee_opening_deg` (with `geometry.si_rotation = "corner"` the inflector turns with it) | geometry |
| dee vertical aperture | `geometry.aperture_inner_mm`, `aperture_outer_mm`, `aperture_taper_center_mm` or `aperture_taper_start_mm` (a box that keeps the taper outside a radius), `aperture_taper_length_mm` | geometry |
| a connected gap line | `geometry.gap_line.<i>.r1_mm`, `dr_mm` (P2 at r1 + dr: no ordering constraint), `f0`, `f1` | geometry |
| a gap window | `geometry.windows.<i>.half_height_mm` | geometry |

Two knobs are awkward for any sampler in their plain form:
- The trim by amplitude and azimuth (`btrim.amplitude_t`, `btrim.phase_deg`): the azimuth is periodic, so -180° and
  +180° are one trim on opposite faces of the box, and it does nothing at zero amplitude. Its Cartesian components
  above, which add to the polar part, have neither a seam nor more than one zero.
- The slit: `slit.half_width_mm = 0` means no slit, which leaves its other coordinates without effect, next to small
  widths that pass almost no beam; and `slit.pass_index` is an unordered choice, not a scale. Give it a campaign of
  its own on the chosen design, with the pass fixed and an open slit in every design.

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

A useful set (the thresholds are the machine's):

| | metric | why |
|---|---|---|
| maximise | `I_final` | final current, T × injected current [mA] |
| minimise | `valley_all` | share of the probe crossings between the turns, the 8 probes (every 45°) pooled. Never one probe alone: a single azimuth can flatter. |
| minimise | `ratio` | aspect ratio of the bunch core at the last turn |
| constraint | `E_med >= E` | reach the design energy [MeV]: E a little below `machine.target_energy_mev` |
| constraint | `turn_sigR <= s` | clean turns: s about an eighth of the turn separation [mm] |
| constraint | `dE_E_robust <= d` | the energy spread that keeps `turn_sigR` below s at the extraction radius (or `sigR_E <= s`) |
| constraint | `fan_proj_last <= f` | the centre fan's share of the turn width [mm] |

There is no vortex objective; `analysis.twins.vortex_twin` measures a design's space-charge vortex against a twin
run without space charge.

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
turn, and its end-of-run scores jump with it. A campaign therefore sets `track.stop_at_target_energy = false` in
`fixed` (the demo campaign does), so every run tracks exactly `track.turns` turns. A slow design then ends below
the target energy, where the `E_med` constraint judges it: set the constraint a little below the energy the base
design reaches, so that a design that ends just short of it stays feasible.

| fidelity | particles | SC | one evaluation on known fields |
|---|---|---|---|
| search | 10 000 | 64³ every 5 steps | about 30 s on a full-size model (see Costs) |
| verify | the whole hand-off bunch | 128³ every step | about 5 min |

The column metal test, the default, and the original design scripts' slice test (`track.obstacles = "raster"`)
track the same particles until the two tests first stop one differently; the slices put level faces half a
millimetre off, and on the design compared they passed 0.4-0.5 % less of the beam at both fidelities
(`docs/pipeline.md`, "Validation of the port"). Every search evaluation tracks the same 10 000 particles of the
hand-off bunch (drawn with `track.seed_final`), so the designs are compared on one sample. A sample of 10 000 is
worth about ±0.5 % in T by itself, mostly the same for every design.

A cheaper search over the first turns would need early stand-ins for the final objectives. The candidates are:
- `sig_r_t3` for `valley_all` (Spearman 0.94 over the earlier runs);
- `ratio_t3` for `ratio` (0.77);
- `compress_t3`, not yet tested.

All three were correlated on the final survivors, while a turns-0-3 run knows only the particles alive at turn 3.
Re-check the correlations on that set before using them. The compression slope (`slope`), `dE_E_robust` and `W_eff`
come from the end of the run, so they are not early stand-ins.

## Posts in the loop (`[post_rule]`)

Posts on the gap edges between the turns can be worth a quarter more current (on one design +25 %), but not on
every geometry (on another, with radial gap lines, the rule's 16 posts cost 18 %, each design at its best phase,
mostly as vertical losses in passes 1-2), and where they fit depends on where the beam crosses each gap, which every
new geometry moves. With `[post_rule] enabled = true` in the base run file, `pcr evaluate` places them by rule for
every new geometry, in two passes:

1. **the bare pass**: the geometry without posts is built, solved and tracked in `<geometry>/bare/`, with the rule's
   phase scan (`scan_phases_deg`, -45 ... +15° in 15° steps: a scan of `track.n_scan` particles, then the final run at
   the best phase by transmission × energy) at the magnet level `scan_magnet_scale` (1.000), and scored. The rule
   (`PyCentralRegion.geometry.post_rule`) then places the posts from its survivors' crossings of each gap edge, banded
   by pass: on the gaps of `gaps` (1-5) one post row between the bands of passes k and k + 1 for every k in `rows`
   (1, 2), `clear_mm` (5 mm) clear of each band's 0.5 % / 99.5 % quantiles and split at the edge's bends; on the gaps
   of `inner_gaps` (1) a post inside the first band. `bare/posts.json` records the posts, every edge's bands and what
   was left out, and the bare run's phase and scores;
2. **the geometry with its posts** is built in `<geometry>/`, checked with the tracker's metal test (with
   `track.obstacles = "columns"`, which refuses a model that is not closed) and solved; the evaluation tracks it at
   its own phase and level.

The bare pass runs once per geometry, in its first evaluation. The cheap followers and the verify re-evaluations reuse
the record and the fields with posts, so the posts depend on the geometry and on the settings of its first evaluation
other than its phase and level (the fidelity: the search fidelity), not on an evaluation's phase or level. So the
campaign's cheap variables should be the RF phase and the magnet level only (the driver warns of any other: the bare
pass would take its value from whichever evaluation came first), and its `fixed` overrides should stay as they are
for the whole campaign. A geometry the rule gives no posts keeps its bare model and fields, and so does one whose build,
metal test or field solve with posts fails (the record's `fallback` says why): it is a design without posts, not a
failed evaluation. An error of memory, the GPU or the file system is no fallback: the evaluation fails and the
geometry's next evaluation tries the model with posts again. A stopped evaluation's geometry continues where it was:
every step reuses its output. The record belongs to the model it was made for (`model`): a model with posts by rule
needs a folder of its own, as the campaign's geometries have. The scores in `result.json` hold `post_rule`: the number
of posts, the fallback, the bare run's best phase, survivors and scores (`I_final_az`, `I_2pct_best_az`, `z_p95_az`,
`E_med_az`, `T_az`), so what the posts gave can be read off every geometry.

A new geometry costs about three times what it costs without the rule: on one full-size design about 15 min (bare
build 24 s, bare BEM 3 min, the bare scan and final run 4.8 min, the rule 26 s, the build with posts 45 s, its BEM
4.6 min, the final track 39 s) against about 5 min for a build, its BEM and one track. Cheap followers cost what they
always did.

The rule needs `geometry.posts` empty and a counterclockwise beam, and gives gap 0 no posts (its hardware is the
housing exit and the hole plate). It was tested with round posts (`geometry.post_shape = "cylinder"`); square posts
stop at the edge's bends. It reproduces the placement scripts it was ported from post for post.

## Samplers

Both samplers are constrained and multi-objective, and both read the same store, so a campaign can switch between
them. Prefer `gp` (it needs torch): on the toy, 40 evaluations reach about 95 % of the true front's hypervolume with
the GP and 70-79 % with TPE, whose pinned cheap followers model only the free coordinates (seeds 0-3). `tpe` is the
default because it needs no torch. With the `greenlet` package optuna's GP runs the restarts of its acquisition's
optimiser side by side, evaluating their points together; without it, one after another, with a warning: on the toy a
proposal then takes about 0.4 s instead of 0.2 s (16-40 evaluations). The two make slightly different proposals from
the same store (on the toy within 1e-9 in 242 of 264, by up to 3e-4 in the unit cube), and each later proposal follows
from the scores of the designs before it, so a campaign repeats exactly only in an environment that has greenlet, or
lacks it, as the original did.

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
(that one then runs first), continues exactly as an uninterrupted run would have in the same environment (with the GP,
greenlet installed or not changes the proposals slightly, and the later ones with them: "Samplers"). A run stopped
during an evaluation loses that evaluation: the evaluations before it are the uninterrupted run's, the ones after it
what the study proposes without it.

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
<campaign_dir>/geometries/<key>/bare/  [post_rule]: posts.json, the bare model with its side files (moved up when
                                    the geometry gets no posts; its field map goes once fields.pkl exists) and track/
                                    (the bare run's record files and scores)
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

**The demo** (`configs/examples/demo/`, a desktop CPU, no GPU): a new geometry's evaluation takes about 2 min
(118 s: the build 20 s, the fields 32 s, the tracking 39 s with a three-phase scan of 1000 particles and a final
run of 4000, start-up and scores the rest).

**A full-size four-dee model** (measured 2026-11-14 on one design with 20 posts placed in the loop, search
fidelity, one GPU, the kernel cached):

| one evaluation | detached, before the full-speed opt-out | from a shell | new field code, the same model | ... with the three mirror options |
|---|---|---|---|---|
| a new geometry (bare model, its phase scan, the post rule, the model with posts, the run, the scores) | 708 s | 364 s | 249 s | 180 s; with the tracking and layout changes of that day 152 to 167 s (six runs, mean 160 s) |
| ... of it the two builds / the two field stages / the tracking / the rest | 63 / 410 / 175 / 60 | 29 / 223 / 81 / 31 | 28 / 105 / 85 / 31 | 31 / 36 / 83 / 30 |
| a cheap follower | 60 s | 28-31 s | | 27 s |
| a verify run's tracking (100,625 particles, 128³ every step) | 648 s | 388 s | | 277 s |

- The first two columns are the same code and the same evaluation with the same scores. Windows moves a process
  without a window to the efficiency cores of a hybrid CPU within seconds, where every stage took about twice as
  long. Each `pcr` process now opts out when it starts (`run.full_speed`, `mobo.full_speed`; `docs/pipeline.md`,
  "Run files"), so the way of starting no longer matters: the final code's evaluation, detached and from a shell in
  turn, took 165.7 and 158.3 s detached and 166.9 and 157.3 s from a shell.
- Runs of the same code differ by a few per cent (six runs: 152 to 167 s, standard deviation 5.5 s).
- The field stages: the GPU solver tests its residual after every product, the potential kernels are built once,
  and with `geometry.exit_opening = "centred"`, `mirror_mesh` and `lc_far_mm` the mesh is solved as half a model
  (`docs/pipeline.md`, "fields").
- Tracking is then the largest part: 83 of 180 s, of which the bare model's phase scan (5 runs of 4000 particles)
  takes 45 s. More than half of every tracked run is the space-charge solver (`docs/pipeline.md`, "track").
- A campaign of 46 new geometries, 135 cheap followers and 5 verify runs takes about 3.5 h at these times
  (46 x 160 s + 135 x 27 s + 5 x about 5.5 min).
- The layout and the post rule: an evaluation lays the same electrodes out three times (the bare build, the rule,
  the build with posts); the distance from the housing outline is remembered within the process
  (`builder._wall_distance`), and the rule's crossing tests share the tracks' segment boxes
  (`post_rule.segment_boxes`).
- The tracking kernel's two builds are cached on disk: only the first process after an edit of `fast_track3d.py`
  compiles them (about 3.5 min); every later one loads them in under a second.

## The toy campaign

`configs/examples/toy/campaign.toml` runs the whole loop (30 evaluations) in about 8 s, 17 s with the GP, without the
physics stack or a GPU: the analytic evaluator `python -m PyCentralRegion.mobo.toy` follows the same contract. The toy
covers subprocesses, the store, the geometry cache, both fidelities, failures (`toy.fail_above_mm`), unusable scores
(`toy.short_above_deg`: no `valley_all`; `toy.skip_above_deg`: the run is skipped) and process trees
(`toy.child_sleep_s`). It is ZDT1 dressed up as a central region, so its Pareto front is known
(`PyCentralRegion.mobo.toy`). `tests/test_mobo.py` runs it.
