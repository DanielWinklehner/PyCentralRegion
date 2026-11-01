# HOWTO: run a MOBO campaign on the 3D model

How to run a campaign, from a dry run to the verified Pareto set and the RF check of the chosen design. The driver
itself is described in [mobo.md](mobo.md). The commands are Windows PowerShell; the paths are those of the HCHC-1.5
machine, so change them for another one.

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
$env:PYTHONPATH = "$repo\src"                                  # the evaluations inherit it
$env:CONDA_PREFIX = "C:\Users\Daniel\anaconda3\envs\pcr-mobo"
$py = "$env:CONDA_PREFIX\python.exe"
```

`& $py -m PyCentralRegion.cli` is the `pcr` command. The driver's environment needs no install of the package:
`PYTHONPATH` points at the source tree.

## 1. Dry run of the driver (seconds, no GPU)

The toy campaign answers with analytic functions, so it tests the driver, the store and the subprocesses alone:

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

The dry run loads the campaign and the base run file and checks every variable at both ends of its range. It then
prints the next evaluation's command line and writes nothing. To run one evaluation by hand with that command, give
it an `--out` folder of its own.

Before the first real run, check the campaign file:
- `campaign_dir`: a new folder. An existing one is continued, not overwritten.
- `physics_python`: the physics interpreter.
- `[mobo.fidelities.search]`: the search fidelity, e.g. 10 000 particles and SC 64³ every 5 steps.
- `timeout_min`: about 3× an evaluation, so that a hung evaluation cannot stall the campaign.
- The variables' ranges. For the dee aperture (`geometry.options.check_bounds` holds them at build time):
  - every height <= 2 × (`geometry.plate_half_gap_mm` - `geometry.dee_wall_mm`), so the dee plates fit the gap;
  - the inner heights >= 2 × the spiral-inflector housing exit's half height, so the channel clears the exit;
  - the taper length must be > 0.
  A range beyond these bounds makes evaluations fail at the build instead of exploring designs.

## 3. Pilot: cheap variables on existing fields (about 40 min)

`mobo_pilot.toml` optimises RF phase, dee voltage and a first-harmonic trim on the fields of
`design_wnop3_rotn3_g0ap`. No geometry is built, so every evaluation is about 2 min of tracking and scoring:

```powershell
& $py -m PyCentralRegion.cli optimize $campaign
```

The driver logs every evaluation as it starts and ends. The first evaluation is the warm start, the design as
tracked. Its I_final should come out near 4.05 mA (T = 50.45 % of 8.03 mA at the search fidelity).

## 4. The campaign with geometry variables

Start from the pilot's file. Add the geometry variables (`kind = "geometry"`), e.g. the spiral-inflector rotation
and the dee aperture taper. Add the cheap ones, e.g. the phase, the voltage and the slit. Give it a budget:

```powershell
$campaign = "$repo\configs\examples\hchc1p5\my_campaign.toml"
& $py -m PyCentralRegion.cli optimize $campaign --dry-run
```
```powershell
& $py -m PyCentralRegion.cli optimize $campaign --max-hours 10
```

Planning the budget:
- a new geometry costs a build (about 1 min) plus the 3D BEM solve, and the driver follows it with
  `cheap_per_geometry` cheap evaluations of about 2 min each;
- every evaluation process first compiles the tracking kernel, about 1 min of the 2;
- with `cheap_per_geometry = 3`, a quarter of the evaluations build a geometry;
- `n_startup` evaluations are random. With 6-10 variables, 10-15 random points and 60-100 evaluations in all is a
  reasonable start.

`--max-hours` bounds one run of the driver. `max_evals` bounds the whole campaign; running the same command again
continues to it.

## 5. While it runs

```powershell
& $py -m PyCentralRegion.cli optimize-status $campaign
```

- Each evaluation has its own folder, `<campaign_dir>\evals\<id>\`, with `command.txt`, `stdout.log`, `stderr.log`
  and `result.json`.
- To stop: Ctrl+C in the driver's window (or close it). The running evaluation stops with every process it started.
  It is recorded as "interrupted" and does not count against the budget.
- To continue: the same `optimize` command. A stopped campaign continues with exactly the proposals an
  uninterrupted one would have made.
- A failed evaluation shows its error in the status table. Its `stderr.log` has the traceback. The sampler does not
  propose that design again.

## 6. Verify the Pareto set

The search fidelity ranks designs; the verify fidelity tracks every particle with SC 128³ every step, about 8 min an
evaluation:

```powershell
& $py -m PyCentralRegion.cli optimize-verify $campaign --pareto
```
```powershell
& $py -m PyCentralRegion.cli optimize-verify $campaign --ids 12 31
```

Each re-evaluation is stored next to its original (`0012.verify`), so `optimize-status` shows both. Judge the
designs by the verify scores, and read the shift between the fidelities: a design that only looks good at 10 000
particles is noise.

## 7. Check the chosen design

1. **Run it on its own** from its run file: `evals\<id>\command.txt` lists every override. Write them into a design
   file next to the worked example and track it with `pcr track` at the verify fidelity. Look at:
   - the final current [mA];
   - the energy spectrum (a split peak);
   - the eight probes' radial histograms.
2. **RF frequency.** The central-region electrodes are part of the dee capacitance, so a new aperture taper, windows
   or posts shift the cavity's resonance. f_RF must stay h × f_rev = 4 × 8.191 MHz = 32.77 MHz, inside the tuner's
   range.
   1. Export the chosen model's solids.
   2. Put them into the full cavity model (COMSOL) in place of the old central region.
   3. Solve its eigenfrequency, and compare it with the design that was tuned (the worked example's).
   Field maps and tracking assume the design frequency: a design the tuner cannot bring back to it is not a design.
3. **Energy spread against turn separation.** A spread above about 1.3 % (dE/E robust) makes turns wider than 3 mm
   at the extraction radius, whatever the current says.
