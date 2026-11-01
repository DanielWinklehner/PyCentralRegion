# HOWTO: run a MOBO campaign on the 3D model

How to run a campaign, from a dry run to the verified Pareto set and the RF check of the chosen design. The driver
itself is described in [mobo.md](mobo.md). The commands are Windows PowerShell; the paths are those of the worked
example (HCHC-1.5), so change them for another machine.

## 0. Before you start

- One GPU, one campaign: nothing else should use the GPU while a campaign runs.
- Two Python environments:
  - the driver's: numpy, scipy, optuna >= 5, torch (CPU), here `pcr-mobo`;
  - the physics one: numba, PyPATools, bempp, gmsh, cupy, here `accel-dev-env`.
  The driver starts every evaluation with the physics interpreter named in `[mobo] physics_python`.
- Free disk. On the worked example:

  | item | size |
  |---|---|
  | one geometry's model and fields (`<campaign_dir>\geometries\`) | about 0.3 GB |
  | one search evaluation's output (10 000 stored tracks) | about 0.4 GB |
  | one verify evaluation's output | about 0.9 GB |

  100 evaluations with 25 geometries take about 50 GB. `track.n_tracks` sets most of an evaluation's size.

Every session starts with:

```powershell
$repo = "D:\Dropbox (Personal)\Code\Python\PyCentralRegion"
$env:PYTHONPATH = "$repo\src"           # the driver's source; the driver also puts it first on the evaluations' PYTHONPATH
$env:CONDA_PREFIX = "C:\Users\Daniel\anaconda3\envs\pcr-mobo"
$py = "$env:CONDA_PREFIX\python.exe"
```

`& $py -m PyCentralRegion.cli` is the `pcr` command. The driver's environment needs no install of the package:
`PYTHONPATH` points at the source tree, and the evaluations run the same code.

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

```powershell
$campaign = "$repo\configs\examples\hchc1p5\mobo_pilot.toml"
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
- The fidelities: `[mobo.fidelities.search]`, e.g. 10 000 particles and SC 64³ every 5 steps. Let the fidelities
  differ in resolution only (particles, SC grid and interval), not in `track.turns`.
- `timeout_min`: about 3× an evaluation, so that a hung evaluation cannot stall the campaign.
- The variables' ranges. For the dee aperture (`python -m PyCentralRegion.geometry.build bounds <base run toml>`
  prints the bounds; `geometry.options.check_bounds` holds the first three at build time):
  - every height <= 2 × (`geometry.plate_half_gap_mm` - `geometry.dee_wall_mm`), so the dee plates fit the gap
    (40 mm on the worked example);
  - the inner heights >= 2 × the spiral-inflector housing exit's half height, so the channel clears the exit
    (21.48 mm on the worked example);
  - the channel heights at the housing exit's outermost radius (dees and dummy dees) >= the same bound, so the channel
    clears the exit over its whole span; this binds only for an inverted taper that starts inside the exit;
  - every height > 0 and < 2 × `geometry.half_height_mm`, and the taper inside 0 < r0 < r1 < 1000 mm
    (r0, r1 = centre ∓ length / 2).

  The driver refuses before anything runs a range whose corners break the last bullet. A range beyond the first three
  makes evaluations fail at the build instead of exploring designs.

## 3. Pilot: cheap variables on existing fields (about 40 min)

`mobo_pilot.toml` optimises RF phase, dee voltage and a first-harmonic trim on the fields of
`design_wnop3_rotn3_g0ap`. No geometry is built, so every evaluation is about 2 min of tracking and scoring:

```powershell
& $py -m PyCentralRegion.cli optimize $campaign
```

The driver logs every evaluation as it starts and ends. The first evaluation is the warm start, the design as
tracked. Its I_final should come out near 4.05 mA (T = 50.45 % of 8.03 mA at the search fidelity).

## 4. The campaign with geometry variables

`mobo_campaign.toml` adds to the pilot's variables the beam channel's aperture (inner and outer height, taper centre
and length) and the spiral-inflector rotation (`kind = "geometry"`), and a radial slit (cheap): 13 variables, 120
evaluations. For another design, start from it or from the pilot's file. Read its header first: it lists what the
loop does not see, among them that the gap-0 hole does not follow the inflector rotation.

Every new geometry is built and solved under `<campaign_dir>\geometries\<key>\`. The first evaluation, the warm start,
rebuilds the base design there (about 10 min; the build equals the design deck's model bit for bit).

```powershell
$campaign = "$repo\configs\examples\hchc1p5\mobo_campaign.toml"
& $py -m PyCentralRegion.cli optimize $campaign --dry-run
```
```powershell
& $py -m PyCentralRegion.cli optimize $campaign --max-hours 10
```

Planning the budget:
- a new geometry's evaluation takes about 10 min on the worked example (build about 20 s, fields about 8 min, then
  the tracking), and the driver follows it with `cheap_per_geometry` cheap evaluations of 1.5-2 min each;
- every evaluation process first compiles the tracking kernel, about 65 s of those 1.5-2 min;
- with `cheap_per_geometry = 3`, a quarter of the evaluations build a geometry;
- `n_startup` evaluations are random (`mobo_campaign.toml`: 16, the warm start and 15 random designs). With 6-13
  variables, 10-16 random points and 60-120 evaluations in all is a reasonable start;
- `mobo_campaign.toml`: 30 × 10 min + 90 × 2 min, about 8 h for its 120 evaluations (`--max-hours 10` runs it to the
  end).

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

The search fidelity ranks designs; the verify fidelity tracks every particle with SC 128³ every step, about 8 min an
evaluation:

```powershell
& $py -m PyCentralRegion.cli optimize-verify $campaign --pareto
```
```powershell
& $py -m PyCentralRegion.cli optimize-verify $campaign --ids 12 31
```

Each re-evaluation is stored next to its original (`0012.verify`), so `optimize-status` shows both. A re-evaluation
that failed or was interrupted runs again with the same command; its earlier folder is kept as
`0012.verify.attempt1`. Judge the designs by the verify scores, and read the shift between the fidelities: a design
that only looks good at 10 000 particles is noise (the search noise can even put a design that misses a constraint
into the search Pareto set).

## 7. Check the chosen design

1. **Run it on its own** from its run file: `evals\<id>\command.txt` lists every override. Write them into a design
   file next to the worked example and track it with `pcr track` at the verify fidelity. Look at:
   - the final current [mA];
   - the energy spectrum (a split peak);
   - the turn structure at the eight virtual probes (`turns`, `turn_*` and `probes` in `score.json`; panel (c) of
     `analysis.plots.summary_figure`). The package draws no probe histograms: histogram
     `analysis.runs.load_probe(<run.out_dir>\probe_<a>deg.npz)['R']` for a recorded probe, or
     `analysis.runs.virtual_probe(run, a)['R']` for any azimuth.
2. **RF frequency.** The central-region electrodes are part of the dee capacitance, so a new aperture taper, windows
   or posts shift the cavity's resonance. f_RF must stay h × f_rev = 4 × 8.1914 MHz = 32.77 MHz (f_rev 8.191375 MHz
   in track.json), inside the tuner's range.
   1. Take the chosen geometry's CAD: `pcr build` writes `<prefix>.step` beside the model
      (`<campaign_dir>\geometries\<key>\model3d.step`): five solids, dee0..dee3 and the ground with the
      spiral-inflector housing fused in, in mm, in the machine frame (origin on the axis, z = 0 the median plane).
   2. Put them into the full cavity model (COMSOL) in place of the old central region.
   3. Solve its eigenfrequency, and compare it with the design that was tuned (the worked example's).
   Field maps and tracking assume the design frequency: a design the tuner cannot bring back to it is not a design.
3. **Energy spread against turn separation.** A spread above about 1.3 % (dE/E robust) makes turns wider than 3 mm
   at the extraction radius, whatever the current says.
