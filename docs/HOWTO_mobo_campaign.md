# HOWTO: run a MOBO campaign on the 3D model

How to run a campaign, from a dry run to the verified Pareto set and the RF check of the chosen design. The driver
itself is described in [mobo.md](mobo.md). The commands are Windows PowerShell; the worked example is the demo
machine's campaign (`configs/examples/demo/campaign.toml`), and a campaign of your own machine runs the same way
([machine.md](machine.md)).

## 0. Before you start

- One GPU, one campaign: nothing else should use the GPU while a campaign runs. (The demo needs no GPU.)
- Two Python environments:
  - the driver's: numpy, scipy, optuna >= 5, torch (CPU) and greenlet ([mobo.md](mobo.md), "Samplers");
  - the physics one: numba, PyPATools, bempp, gmsh, cupy.
  The driver starts every evaluation with the physics interpreter named in `[mobo] physics_python` (empty: its own
  interpreter, which then needs both sets, as the demo does).
- Free disk. On a full-size four-dee model:

  | item | size |
  |---|---|
  | one geometry's model and fields (`<campaign_dir>\geometries\`) | about 0.3 GB |
  | one search evaluation's output (10 000 stored tracks) | about 0.4 GB |
  | one verify evaluation's output | about 0.9 GB |

  100 evaluations with 25 geometries take about 50 GB. `track.n_tracks` sets most of an evaluation's size.

Every session starts with:

```powershell
$repo = "$HOME\code\PyCentralRegion"
$env:PYTHONPATH = "$repo\src"           # the driver's source; the driver also puts it first on the evaluations' PYTHONPATH
$env:CONDA_PREFIX = "$HOME\miniforge3\envs\pcr-mobo"
$py = "$env:CONDA_PREFIX\python.exe"
$env:MACHINE_PATHS = "$HOME\machines\paths.toml"   # this computer's disk roots, for run files that use ${name}
```

`& $py -m PyCentralRegion.cli` is the `pcr` command. The driver's environment needs no install of the package:
`PYTHONPATH` points at the source tree, and the evaluations run the same code. The demo's files use no root, so it
needs no paths file; a machine whose run files name `${inputs}` or `${work}` does
(`configs/examples/paths.example.toml`).

## 1. The toy campaign: the driver end to end (seconds, no GPU)

The toy campaign answers with analytic functions, so it runs the driver, the store and the subprocesses for real,
without the physics stack or a GPU (30 evaluations, about 8 s):

```powershell
& $py -m PyCentralRegion.cli optimize "$repo\configs\examples\toy\campaign.toml"
```
```powershell
& $py -m PyCentralRegion.cli optimize-status "$repo\configs\examples\toy\campaign.toml"
```

It writes `$repo\output\mobo_toy` (git ignores `output\`). Delete that folder to run the toy again from scratch.

## 2. Check the real campaign without running it

The demo's inputs first (once, about a minute):

```powershell
& $py -m PyCentralRegion.demo
```

```powershell
$campaign = "$repo\configs\examples\demo\campaign.toml"
& $py -m PyCentralRegion.cli optimize $campaign --dry-run
```

The dry run loads the campaign and the base run file. It checks every override against its section (each section's
variables at every corner of their box) and, for a started campaign, that the variables still map its stored designs.
It then prints the next evaluation's command line and writes nothing. It does not start the physics interpreter:
`pcr optimize` checks that first (the preflight: a few seconds, output in `<campaign_dir>\preflight\`) and stops before
the first evaluation when the interpreter cannot find the package. The dry run also marks an objective or constraint
metric that the scorer's table (`PyCentralRegion.analysis.objectives.METRICS`) does not list.

To run one evaluation by hand with the printed command, give it an `--out` folder of its own. The command is quoted
for cmd.exe; in PowerShell put `--%` after the interpreter: `& <python> --% -m PyCentralRegion.cli evaluate ...`.

Before the first real run, check the campaign file:
- `campaign_dir`: a new folder. An existing one is continued, not overwritten.
- `physics_python`: the physics interpreter.
- The fidelities: `[mobo.fidelities.search]`, e.g. 10 000 particles and SC 64³ every 5 steps on a full-size model.
  Let the fidelities differ in resolution only (particles, SC grid and interval), not in `track.turns`.
- The turn every design is scored at: `track.turns` with `track.stop_at_target_energy = false` in `fixed`, as the
  demo campaign does. Without it a run ends at the turn its reference reaches `machine.target_energy_mev`, so a slower
  design tracks an extra turn and its end-of-run scores jump.
- `timeout_min`: about 3× an evaluation, so that a hung evaluation cannot stall the campaign.
- The variables' ranges. For the dee aperture (`python -m PyCentralRegion.geometry.build bounds <base run toml>`
  prints the bounds; `geometry.options.check_bounds` holds the first three at build time):
  - every height <= 2 × (`geometry.plate_half_gap_mm` - `geometry.dee_wall_mm`), so the dee plates fit the gap
    (16 mm on the demo); where `geometry.plate_profile_mm` narrows the gap (a pole nose and its own taper), at each
    radius with the half gap there. To keep every point of the box buildable, give the taper by its start
    (`aperture_taper_start_mm`) with a lower bound no further in than where the nose's taper begins, the inner height
    below the nose's bound and the outer one below the outer gap's: a taper at least as long as the nose's rise to
    the outer bound then never cuts into it;
  - the inner heights >= 2 × the spiral-inflector housing exit's half height, so the channel clears the exit
    (10 mm on the demo);
  - the channel heights at the housing exit's outermost radius (dees and dummy dees) >= the same bound, so the channel
    clears the exit over its whole span; this binds only for an inverted taper that starts inside the exit;
  - every height > 0 and < 2 × `geometry.half_height_mm`, and the taper inside 0 < r0 < r1 < 1000 mm
    (r0, r1 = centre ∓ length / 2).

  The driver refuses before anything runs a range whose corners break the last bullet. A range beyond the first three
  makes evaluations fail at the build instead of exploring designs.

## 3. A pilot: cheap variables on existing fields

A first campaign with cheap variables only (the RF phase `track.phases_deg`, the dee voltage `track.volt_scale`) on
the base design's existing model and fields needs no build: every evaluation is tracking and scoring. It shows how the
scores and the constraints behave before any geometry is varied, at a fraction of the cost. Leave the geometry
variables out and point the base run file's `paths.model3d` and `paths.fields` at the design's model and fields
(`pcr build` and `pcr fields` make them once).

```powershell
& $py -m PyCentralRegion.cli optimize $campaign
```

The driver logs every evaluation as it starts and ends. The first evaluation is the warm start, the design as
tracked: at the same fidelity it gives the design's own run. A first-harmonic trim (`btrim.amplitude_cos_t`,
`btrim.amplitude_sin_t`) is a cheap variable too; add it only for a machine with trim coils, or label the results
"with trim".

## 4. The campaign with geometry variables

The demo campaign varies the dee channel's height inside the taper (`kind = "geometry"`: every new value is an
electrode build and its BEM fields) and the RF phase (cheap: a re-track on that geometry's fields), with two objectives
(`I_final`, `dE_E_robust`) and an energy constraint, 12 evaluations. For a machine of your own, start from it and add
the geometry variables your design needs ([mobo.md](mobo.md), "Useful targets").

Every new geometry is built and solved under `<campaign_dir>\geometries\<key>\`. The first evaluation, the warm start,
rebuilds the base design there.

```powershell
& $py -m PyCentralRegion.cli optimize $campaign --dry-run
```
```powershell
& $py -m PyCentralRegion.cli optimize $campaign --max-hours 10
```

Planning the budget:
- a new geometry costs a build and a BEM solve on top of the tracking: on the demo about 2 min on a desktop CPU
  (build 20 s, fields 32 s, tracking 39 s with a three-phase scan); on a full-size model with posts placed in the
  loop about 2.7 min, with the mirror options ([mobo.md](mobo.md), "Costs");
- the driver follows it with `cheap_per_geometry` cheap evaluations: a re-track on the same fields (about 30 s at the
  search fidelity on a full-size model);
- only the first evaluation process after an edit of `fast_track3d.py` compiles the tracking kernel (about 3.5 min);
  every later one loads it from the cache;
- with `cheap_per_geometry = 3`, a quarter of the evaluations build a geometry;
- `n_startup` evaluations come before the model takes over: at least 2 × the number of variables + 1;
- designs of one flat channel (`ap_inner` = `ap_outer`) share one geometry whatever the taper's place, so they build
  once;
- the campaign then costs about (geometries × new-geometry time) + (followers × cheap time), plus the verify runs.

A radial slit comes after, in a campaign of its own on the design this one chooses: vary the slit's opening, azimuth
and offset at one fixed pass (one campaign per pass), with an open slit in every design ([mobo.md](mobo.md),
"Variables").

With `[post_rule] enabled = true` in the base run file, posts are placed by rule in every new geometry's first
evaluation ([mobo.md](mobo.md), "Posts in the loop"). That first evaluation builds, solves and tracks the geometry
twice, bare and with its posts, so it takes several times a plain one. Run such a campaign detached for a set number
of hours (`--max-hours`) and stop it from the outside when needed: running the same command again continues it.

`--max-hours` bounds one run of the driver. `max_evals` bounds the whole campaign (`--max-evals N` overrides it);
running the same command again continues to it. `--set key=value` overrides any key of the campaign file for one run,
e.g. `--set mobo.seed=3`.

## 5. While it runs

```powershell
& $py -m PyCentralRegion.cli optimize-status $campaign
```

- Each evaluation has its own folder, `<campaign_dir>\evals\<id>\`, with `command.txt`, `stdout.log`, `stderr.log`
  and `result.json`.
- To stop: Ctrl+C (or Ctrl+Break) in the driver's window, or close it. The running evaluation stops with every
  process it started. It is recorded as "interrupted", does not count against the budget and is not run again (give
  its design as a `warm_start` point to evaluate it anyway; an interrupted warm-start point is run again).
- To continue: the same `optimize` command. Every finished evaluation is kept, and the proposals follow from them and
  the seed: a driver stopped between two evaluations loses nothing, one stopped during an evaluation loses that one.
- Do not change the variables of a started campaign: the driver refuses. Run a changed search space in another
  `campaign_dir`; objectives and constraints may change.
- A failed evaluation's error is listed under the status table; its `stderr.log` has the traceback. The sampler does
  not propose that design again.
- `optimize` stops by itself (exit code 1) after 5 failed evaluations in a row (`max_failed_in_a_row`): read the last
  one's `stderr.log`, fix the environment, and run the same command again.
- `done!` in the state column: the run finished, but its scores lack a metric of an objective or a constraint (the
  reason is listed under the table): the scorer skipped the run, or the beam was lost too early for e.g.
  `valley_all`. Such a design enters the study as infeasible, with the worst values seen, so the sampler leaves its
  region; it counts against the budget and is never in the Pareto set.

## 6. Verify the Pareto set

The search fidelity ranks designs; the verify fidelity tracks every particle of the hand-off bunch with a finer
space-charge solve (on a full-size model 128³ every step, about 5 min of tracking an evaluation):

```powershell
& $py -m PyCentralRegion.cli optimize-verify $campaign --pareto
```
```powershell
& $py -m PyCentralRegion.cli optimize-verify $campaign --ids 12 31
```

Each re-evaluation is stored next to its original (`0012.verify`), so `optimize-status` shows both. A re-evaluation
that failed or was interrupted runs again with the same command; its earlier folder is kept as
`0012.verify.attempt1`. Judge the designs by the verify scores, and read the shift between the fidelities: a design
that only looks good at the search fidelity is noise (the search noise can even put a design that misses a
constraint into the search Pareto set).

## 7. Check the chosen design

1. **Run it on its own** from its run file: `evals\<id>\command.txt` lists every override. Write them into a design
   file next to the base design and track it with `pcr track` at the verify fidelity. Look at:
   - the final current [mA];
   - the energy spectrum (a split peak);
   - the turn structure at the eight virtual probes (`turns`, `turn_*` and `probes` in `score.json`; panel (c) of
     `analysis.plots.summary_figure`). The package draws no probe histograms: histogram
     `analysis.runs.load_probe(<run.out_dir>\probe_<a>deg.npz)['R']` for a recorded probe, or
     `analysis.runs.virtual_probe(run, a)['R']` for any azimuth.
2. **RF frequency.** The central-region electrodes are part of the dee capacitance, so a new aperture taper, windows
   or posts shift the cavity's resonance. f_RF must stay h × f_rev (`rf_mhz`, `f_rev_mhz` in track.json), inside the
   tuner's range.
   1. Take the chosen geometry's CAD: `pcr build` writes `<prefix>.step` beside the model
      (`<campaign_dir>\geometries\<key>\model3d.step`): five solids, dee0..dee3 and the ground with the
      spiral-inflector housing fused in, in mm, in the machine frame (origin on the axis, z = 0 the median plane).
   2. Put them into the full cavity model (COMSOL) in place of the old central region.
   3. Solve its eigenfrequency, and compare it with the design that was tuned.
   Field maps and tracking assume the design frequency: a design the tuner cannot bring back to it is not a design.
3. **Energy spread against turn separation.** The energy spread widens the turns at the extraction radius, whatever
   the current says: compare `dE_E_robust` (and `sigR_E`, `turn_sigR` in score.json) with what the extraction
   accepts.
