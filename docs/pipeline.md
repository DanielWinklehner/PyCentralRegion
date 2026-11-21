# The 3D central-region pipeline

From a gap geometry to a scored 3D bunch simulation, one stage per command. Every stage reads one TOML run file
(plus `--set` overrides), writes its output to a path the run file names, and reuses an output that already exists.

```
seed (2D gap geometry, .pkl) ─┐
machine.toml / design.toml ───┤
                              ├─ pcr build ──> 3D electrode model    paths.model3d
                              ├─ pcr fields ─> 3D BEM field cache    paths.fields
3D magnet map ────────────────┤
spiral-inflector hand-off ────┼─ pcr track ──> track.json, track_data.pkl, probe_<a>deg.npz   run.out_dir
                              └─ pcr score ──> score.json            run.out_dir
pcr evaluate = build -> fields -> track -> score into one folder (result.json): the unit a MOBO campaign calls
pcr optimize = the campaign (docs/mobo.md)
```

Scope: four dees with radial gap lines (`machine.n_dees = 4`, `machine.dee_shape = "radial"`). Anything else raises
`NotImplementedError` up front instead of running a model that was never validated.

## Run files

A run file sets only what differs from the defaults and includes what it builds on:

```toml
include = ["machine.toml"]          # relative to this file; the including file wins

[run]
name = "my_design"
out_dir = "${work}/my_design"          # ${work}: a root of this computer's paths file

[paths]
seed = "optimized_cavity_geometry.pkl"   # relative paths: relative to the file that sets them
fields = "bunch_fields3d_my_design.pkl"

[track]
phases_deg = [-15.0]
```

- Defaults: `configs/defaults/*.toml` write out every key of every section, with a comment. The tests check that
  they equal the dataclass defaults in the code, so the files are the reference.
- Sections: `run`, `paths`, `machine`, `magnet`, `inflector`, `geometry`, `fields`, `track`, `slit`, `btrim`,
  `score`, `mobo`. A key a section does not define is an error: a typo fails, it is not ignored.
- Overrides: `--set section.key=value`, the value read as TOML (`--set track.phases_deg=[-15.0]`,
  `--set track.sc=false`), else as a string. An integer part indexes an array of tables:
  `--set geometry.windows.1.half_height_mm=11.5` changes the second window. An override cannot create an array item.
- `pcr show run.toml [--set ...]` prints the configuration exactly as the stages see it. `pcr track` writes it next
  to its output (`run.out_dir/config.json`), `pcr build` the sections it used into `<prefix>_build.json`, and
  `pcr evaluate` into `result.json`.
- Paths: everything in `[paths]`, and any key ending in `_path`, `_dir` or `_file` (also inside arrays of tables), is
  resolved relative to the file that sets it. `""` means "not given".
- Roots: `${name}` in any string is a disk root of this computer, read from its paths file (`[roots]`, absolute
  paths; `configs/examples/paths.example.toml`). The file is `--paths FILE`, else `$MACHINE_PATHS`, else
  `$PYCENTRALREGION_PATHS`, else `~/.config/pycentralregion/paths.toml`; `${pcr}` is the package's source tree
  and `$$` a literal `$`. Run files that use no root need no paths file, and absolute paths work as before.
  Keeping the drive letters in one file per computer lets the same run files run on every computer.
- No stage reads an environment variable.
- Full speed on Windows (`run.full_speed = true`, the default; `mobo.full_speed` for the campaign driver). Windows 11
  manages the speed of every process that does not ask otherwise. A process without a visible window (a campaign
  started detached, an evaluation the driver starts with a hidden console, a job under the task scheduler) starts on
  the fast cores of a hybrid CPU and is moved to the efficiency cores within seconds, where every thread of it runs
  at about half the speed. Each `pcr` command therefore opts its process out once its run file is read
  (`process.keep_full_speed`: `SetProcessInformation(ProcessPowerThrottling)`, the priority class untouched).
  Measured 2026-11-14 on an 8 P + 12 E core desktop, the same search-fidelity tracked run: 25 s from a shell with a
  window, 47 s detached, 22 s detached with the opt-out; a whole evaluation of a new geometry took 708 s detached
  without it and 364 s from a shell, with the same scores. A script of your own that runs detached calls
  `PyCentralRegion.process.keep_full_speed()` first.

Worked example: `configs/examples/demo/` (`machine.toml` is the machine, `design.toml` one design, `campaign.toml` a
campaign over it). `python -m PyCentralRegion.demo` generates its inputs; setting up a machine of your own:
[machine.md](machine.md).

## Stages

| command | reads | writes | notes |
|---|---|---|---|
| `pcr build` | `[geometry]`, `[machine]`, `[inflector]`, `paths.seed`, `paths.housing_outline`, the emitting-plane hand-off (`geometry.emitting_plane_file`, else the export's own beside the outline, else `paths.handoff`), `exit_plane_spec.json` (for `si_rotation = "corner"` or `gap0_on_plate`) | `paths.model3d`, and beside it `<prefix>.msh`, `<prefix>.step`, `<prefix>_build.json`, `<prefix>_dees.png` | OCC solids, mesh, electrode potentials (`geometry.build`); the `.step` is the CAD: 5 solids (dee0..dee3, the ground with the spiral-inflector housing fused in), mm, machine frame |
| `pcr fields` | `paths.model3d`, `paths.seed`, `[fields]`, `[machine]` | `paths.fields` | 3D BEM solve, E-field grid (zeroed deep inside the metal) |
| `pcr track` | `paths.fields`, `paths.seed`, `paths.bfield_3d`, `paths.handoff`, `paths.inflector_efield`, `paths.housing_step`, `[run]`, `[machine]`, `[magnet]`, `[btrim]`, `[inflector]`, `[fields]`, `[track]`, `[slit]` | `run.out_dir`: `track.json`, `track_data.pkl`, `config.json`, `probe_<a>deg.npz` | reference, phase scan, final run |
| `pcr score` | `run.out_dir` (its `config.json` gives the `[machine]` the run was tracked with), `[score]` | `run.out_dir/score.json` | metrics and objectives (`analysis`); a run with fewer than `score.min_tracks` tracks or `score.min_survivors` survivors scores `{"tag", "skip"}` |
| `pcr evaluate --config run.toml --out DIR` | all of the above | `DIR/result.json` | the chain; a failure is written to result.json, exit code 1 |

Reuse rule: a stage whose output exists says so and returns it. To recompute, point the path elsewhere. The outputs
a stage is judged done by (the build's, the field cache, track.json and track_data.pkl, score.json, result.json) are
written under the temporary name `<stem>.part<suffix>` (`track.part.json` for track.json) and renamed
(`config.write_atomic`; the build's by `config.commit_part` once all of them exist, the model last). A killed run never
leaves a truncated file to be reused, and no stage takes a temporary file for an output. A `pcr build` that fails
removes its temporary files; the bounds it can check before the layout (the channel heights) are checked before
anything is written.

### fields

- The BEM solution of the model (every dee at its potential, ground at 0) is sampled on a grid out to the model's
  `r_cut + fields.margin_m`: `fields.dxy_m` across, `fields.dz_m` in height, `|z| <= fields.z_half_m`.
- The field is zeroed two grid cells inside the metal, as the metal columns below see it; the wall layer is kept, so
  particles feel the field up to the surface. A solid that is not closed is left out of this mask where a vertical
  line crosses it an odd number of times, with a warning and the count in the cache (`stack['odd_columns']`): those
  nodes keep their BEM values, which no point in vacuum reads.
- The cache keeps the layout of the original design scripts' `bunch_fields3d<tag>.pkl`, so their caches track here
  unchanged. It
  holds the model, not a metal test: the tracker builds its own (`track.obstacles`).
- Solver choices: `fields.solver` (auto | cupy | scipy | strong | dense), potentials on the GPU in single precision
  with `fields.pot_check` points re-evaluated in double precision.
- The solve tests the mesh for mirror symmetry about z = 0 (`fields.symmetry = "auto"`,
  `gap_fields.mirror_pairs`: every node and every triangle has its mirror image, a pair at one potential). A mesh
  that passes is solved as half a model: only the rows of the upper triangles are assembled, the mirror image is
  folded into them (A_h[k, l] = A[u_k, u_l] + A[u_k, mirror(u_l)]) and the half system is factored directly on the
  CPU (LAPACK); the charge of an upper triangle is its image's too. The potential is then evaluated on the grid's
  levels z >= 0 only and copied to their mirror levels, so the field is mirror symmetric (no E_z on the midplane).
  `geometry.mirror_mesh` builds such meshes (below); any other mesh takes the whole-mesh path, as before.
  `"off"` never uses the half model, `"require"` fails on a mesh that is not symmetric.
- The whole-mesh path solves with a Jacobi-preconditioned GMRES, on the GPU when the 8 N² bytes of the matrix fit
  its free memory (up to about 42,000 triangles on a 16 GB card), else on the CPU. Since 2026-11-14 the GPU solver
  tests its residual after every matrix-vector product (CuPy's own tests it once per restart cycle of 1000), and it
  lifts the cap PyPATools' space-charge solver puts on CuPy's memory pool for the time of the solve.
- The potential kernels are built once per solve and applied to every batch of points (`fields.pot_chunk`, which on
  the GPU is also held to a number of point-triangle pairs so that one launch stays far below the two seconds after
  which Windows resets a display GPU's driver). Bempp's own potential operator builds its OpenCL programs for every
  batch: 2.1-2.4 s each on the GPU, 54 of the 63 s the 24 batches of the grid took. The values are the same to the
  last bit.

#### Mirror symmetry and the three `[geometry]` options that use it

The electrodes are mirror symmetric about the median plane but for one feature, the spiral-inflector housing's exit
opening, which the export puts where the beam leaves (in one export a ring of R 10 mm centred at z = -0.76 mm, swept
0.82 deg downward). Three options, all off by default, make the field build cheap; a design file sets them in its
`[geometry]`:

| `[geometry]` | what it does | what it changes |
|---|---|---|
| `exit_opening = "centred"` | the opening on the median plane, with the exported one's exit half height (R 10.76 mm in that export, so it contains the exported opening and the channel bounds stay), swept level | the model: the larger opening no longer clips the top of the bunch. On the design measured below, at the search fidelity: 136 more of 10,000 particles leave the housing, 53 more reach the end |
| `mirror_mesh = true` | the build cuts every electrode at z = 0 (after checking that its upper half holds half its volume and that its centre of mass lies on the plane), meshes the upper half and mirrors it; `<prefix>.step` stays the whole CAD, `<prefix>.msh` is the upper half | the triangulation only (the lower half's is the mirror image of the upper's instead of gmsh's own); needs `exit_opening = "centred"` and windows / holes centred on the plane |
| `lc_far_mm = 32.0` | the faces far above and below the channel are meshed up to that size, growing over \|z\| = `lc_far_z0_mm` .. `lc_far_z1_mm` (34 .. 48 mm) | about 30 % fewer triangles; the field in the channel changes by 2e-4 (midplane) to 1.6e-3 (2-6 mm from a plate), median; the tracked result by 0 to 1 particle in 10,000 |

A campaign switches them on in its design file from its first evaluation on: they change the model, and a model
built without `mirror_mesh` is not mirror symmetric as a mesh, so it keeps the whole-mesh path.

Measured 2026-11-14 on one design's geometry with 20 posts, every stage a process of its
own started from one shell, the old code being the commit before these changes (seconds; the fields stage also
builds the metal columns, checks the seed orbit and writes the cache):

| `pcr fields` | triangles / unknowns | assembly | solve | potential grid | stage |
|---|---|---|---|---|---|
| old code | 44,920 / 44,920 | 11.6 | 44.5 (GMRES, CPU, 134 products) | 60 | 124 |
| new code, the same model | 44,920 / 44,920 | 11.6 | 45.2 (the same solve) | 12 | 78 |
| `exit_opening = "centred"`, `mirror_mesh` | 45,938 / 22,969 | 8.3 | 16.4 (LU) | 8 | 40 |
| ... and `lc_far_mm = 32` | 31,858 / 15,929 | 4.5 | 6.1 (LU) | 6 | 24 |
| bare model (40,322 triangles, fits the GPU), old code | 40,322 / 40,322 | 9.5 | 30.2 (GPU, 1000 products) | 60 | 107 |
| bare model, new code | 40,322 / 40,322 | 9.5 | 7.7 (GPU, 56 products) | 11 | 37 |
| bare model, the three options | 26,416 / 13,208 | 3.4 | 3.7 (LU) | 5 | 19 |

The same run tracked through each field (transmission to the end):

| model | search fidelity (10,000 particles) | verification fidelity (100,625 particles) |
|---|---|---|
| the campaign's own record | 80.160 % | 79.025 % |
| new code, the same model | 80.160 % (the field cache is the campaign's bit for bit) | 79.025 % |
| centred opening, mirrored mesh | 80.690 % | 79.707 % |
| ... and coarse far faces | 80.690 % (1 particle differs in turns 3-6) | 79.719 % (12 particles more) |

The gain of 0.5-0.7 % of the beam is the opening's: centred with the exported one's exit half height, it no longer clips the
top of the bunch (1.4 % more particles leave the housing). A whole evaluation of a new geometry with posts in the loop
(bare model, post rule, model with posts, search run, scores) took 364 s with the old code, 249 s with the new code on
the same model and 180 s with the three options (`docs/mobo.md`, "Costs").

### The metal test

`track.obstacles` chooses how the tracker decides that a particle is in metal:
- `"columns"` (default): at every node of an x-y grid (`fields.stack_dxy_m`, 0.5 mm) the heights at which the vertical
  line through it enters and leaves metal, cut from the model's triangles (`metal_columns`; 0.25 s for a model of
  40,000 triangles). A point is tested in the column of its nearest node: exact in z, good to half a node spacing in
  x and y. A model with a solid that is not closed is refused (a line crossing it an odd number of times); `"raster"`
  tracks it. No model built by `pcr build` has been refused so far.
- `"raster"`: the original design scripts' test, the solids' plane sections on z levels (`fields.stack_z_half_m`,
  `fields.stack_dz_m`: 1 mm apart) tested at the nearest level, so a face acts at the half millimetre next to it: a
  channel's full height is seen in 2 mm steps, and a face on a level acts 0.5 mm off. A cache of those scripts (or
  one written before 2026-11-02) carries its slices; for any other cache they are cut at the start of the run (2-5 min
  for a full-size model). Use it to compare with runs of those scripts.
- `"mesh"`: the triangles themselves on the GPU (Warp BVH), with a crossing test per step. The compiled kernel
  cannot carry it, so the run takes the Python path.

The plane sections (`electrodes3d.loops_from_mesh`) are cut edge by edge since 2026-11-02 and come out closed whatever
lies in or near the plane. Before, the segment ends were chained on a 1e-6 m merge grid: where a vertex lay a few
1e-8 m off a level on a nearly level face, crossings that were not neighbours fused and a solid's whole section on
that level was dropped. Slices cut before then (and the caches written until then) can lack a solid on such a level,
which puts its faces there up to a level spacing into the metal.
A vertex within 1e-12 m of the plane lies in it (`SECTION_Z_TOL_M`), and a face in the plane belongs to the solid
above it: the model's faces carry rounding of either sign at their nominal heights (up to 1e-17 m), which would
otherwise decide a cut at a face's own height (the `sections` views at 16 mm, a stack whose levels fall on faces).

### track

`simulate3d.BunchTracker` builds the machine once:
- the design's dee system with the seed's gap lines;
- the cached BEM field, every dee at the model voltage times `track.volt_scale`, modulated with the RF;
- the 3D magnet map (cropped and cached under `run.cache_dir`; optional `[btrim]` first harmonic);
- the spiral inflector of `[inflector]`: its static E-field, the hand-off bunch and the reference particle, all
  turned by `inflector.rotation_deg`.

A model built around a turned inflector records the turn (`params['si_rotation_deg']`). Tracking it with a different
`inflector.rotation_deg` is refused.

Then `track_stage`:
1. tracks the reference particle at every phase of `track.phases_deg` (empty: the seed's phase and ±10 deg);
2. with several phases, tracks `track.n_scan` particles at each and keeps the best phase (transmission × energy);
3. tracks `track.n_final` particles (0 = the whole hand-off bunch) at that phase. This final run also records the
   full final and initial states, snapshots at `track.snapshot_ns`, `track.n_tracks` particle tracks every
   `track.track_every` steps, and the probes of `track.probes_deg`.

Space charge is an FFT Poisson solve (PyPATools), every `track.sc_every` / `sc_final_every` steps, on
`sc_cells` / `sc_final_cells` nodes per axis in the bunch frame, on the GPU with `track.sc_gpu`. The bunch current is
the hand-off file's unless `track.sc_current_ma` is set.

| fidelity | particles | SC grid / interval | use |
|---|---|---|---|
| search | 10 000 | 64³ every 5 steps | optimisation; the particle number dominates the error |
| verification | the whole hand-off bunch | 128³ every step | a design worth building; only z rms is grid-sensitive |

### Bunch train: the neighbouring turns (opt-in)

By default a run tracks one bunch alone. `track.bunches_ahead` and `track.bunches_behind` put it in a train: copies of
the bunch released one every `machine.harmonic` periods of the tracking RF before and after it. That period is one
revolution, exactly `track.steps_per_turn` steps, so at any moment the bunches sit side by side on neighbouring turns:
those released before fly ahead, outside the run's bunch; those released after fly inside it. The bunches in the
other RF buckets, at other azimuths, are not there.

- **Space charge:** every bunch carries the bunch charge I / f_RF, and all of them deposit on one mesh, laid across
  the train: `sc_cells` / `sc_final_cells` nodes along the bunch and vertically, `track.sc_train_cells_radial`
  (512) across the train. Each bunch feels its own field and the others' from the same solve. Measured on one design
  (the tracks of a single-bunch run, CPU): 64 × 512 × 64 nodes over a train of five resolve a bunch's own field as
  its own 64³ window does, 128 × 512 × 128 as 128³; the neighbours change the slope of the bunch's own field by a
  few per cent, and the tails of neighbouring turns interleave at some azimuths, which is why the neighbours are on
  the same mesh and not on a coarser one.
- **The control:** `track.train_coupled = false`. The run's bunch then feels only its own charge, on the train's
  mesh, while the other bunches feel every charge and fly as in the coupled run (two solves per step on the same
  positions, so on the same mesh). A train run and its control differ by the neighbours' forces on the run's bunch;
  the control and a run of the bunch alone differ by the mesh. Compare a train with its control, not with the bunch
  alone: a change of mesh moves the particles too (at 1000 particles on a 32-node mesh more than the neighbours
  do). Measured on one design: at search fidelity the mesh alone changes the turn width by -3 to -4 %, at
  verification fidelity by -0.8 % and every other score by less than 1 %.
- **What to expect** (one design, a train of five against its control, the same at search and at verification
  fidelity): the final current does not change (by 0.00 ± 0.03 % of the beam with every particle), 0.1 % of the
  bunch is lost in turn 2 instead of turn 7, the bunch is 6 % longer, turn separation over turn width drops by
  1.2 %, and the vertical size grows by about 1 % averaged over a turn at unchanged emittance (3 % at one azimuth:
  the breathing of the envelope shifts, so compare turn means, not one readout).
- **`track.train_charge`** scales the other bunches' charge. At 0 they are weightless: tracked on the same mesh, but
  nothing feels them and they feel no charge of their own. That is a cheaper control (one solve per step) and a
  rougher one: without their own space charge the other bunches have other sizes, so the mesh is not the same.
- **What the run records:** the run's own bunch only, in its own clock (step 0 is its first release), in the layout
  of a run of the bunch alone: the per-turn statistics, hits, final states, snapshots, tracks and probes, and so the
  scores. The reference particle, t = 0 and the bunch phase are those of the own bunch. `final.train` in track.json
  adds a table per bunch: live particles at every logged turn and at the end, metal losses, end energy and radius.
  Bunch k has as many live particles as bunch 0 had k turns earlier once the train is long enough: that is the check
  of its length.
- **How long a train:** the run starts `bunches_ahead` turns early, so the own bunch is born with its outer
  neighbours in place and gets its inner ones as they are released. On the design measured, the nearest turn on each
  side gives 83 % of the neighbours' effect on the field slopes, the second 14 %. The bunches ahead fly
  `bunches_ahead` turns further than the own bunch: they must still be inside the field maps and the loss limits
  there (`machine.r_max_m`), or the last turns of the own bunch lose their outer neighbour. On the design measured
  the model ends after turn 7: a bunch ahead leaves at the radial loss limit during its eighth turn (60 % of it
  between turns 7.5 and 7.75), so the run's bunch has its full outer neighbour until its turn 6.5 and about a tenth
  of it at the end. A second bunch ahead adds a neighbour for turns 1 to 5.5 only.
- **Cost:** a train of N bunches tracks N times the particles for `turns + bunches_ahead` turns and solves on a mesh
  `sc_train_cells_radial / sc_cells` times as large; a control solves twice per step. Measured on the GPU
  (2026-11-13, RTX 4070 Ti SUPER): one solve of a five-bunch train takes 18 ms on 64 × 512 × 64 nodes (50,000
  particles) and 127 ms on 128 × 512 × 128 (503,125), against 3 ms and 25 ms for the bunch alone on 64³ and 128³;
  a search-fidelity run of 7 turns takes 14 s for the bunch alone, 51 s in a train of three (its control 88 s) and
  105 s in a train of five (124 s); a verification-fidelity run (100,625 particles per bunch, a solve every step)
  648 s for the bunch alone, 2033 s in a train of five (153 ms per step in the solve) and 3079 s for its control
  (268 ms). These times are of the tracking code of 2026-11-13; the speed-up of 2026-11-14 brought the bunch alone
  to 277 s at verification fidelity, and the train has not been timed again since.
- **GPU memory:** 64 × 512 × 64 nodes hold 1.24 GiB, as the bunch alone on 128³ does; 128 × 512 × 128 hold
  3.41 GiB, more than the solver's default memory pool of 2 GiB, in which every solve of that mesh fails. Set
  `track.sc_gpu_pool_gb` (4 or more); the run refuses a train mesh that cannot fit its pool
  (`simulate3d.sc_mesh_memory_gb`) instead of skipping solves.

The compiled 3D kernel (`track.fast3d`, numba) runs the whole bunch: release, space-charge kicks, the metal test and
the probes. It needs:
- the `boris_rel` pusher;
- plain gridded fields;
- a metal test of at most one column test and two nearest-node rasters: the model's (columns or slices, with any
  extra posts and boxes put in) and the slit's.

Every run record of track.json carries `fast_path`: whether the compiled kernel ran and, if not, why. A run that
cannot use it goes the Python path: same answer, far slower.

The kernel runs in chunks that end where Python has work to do: a space-charge solve, or the last solve's field for
particles just released (`SpaceChargeKick.refresh`). The kernel applies every space-charge kick itself, also the one
of a solve step (until 2026-11-14 that one was applied in NumPy, a dozen passes over the particle arrays per solve:
with a solve every step, a fifth of a verification run). Two runs of the same input agree in every count (alive per
turn, losses) and to about 1e-13 m in the positions, not to the last bit: the GPU sums the deposited charge in no
fixed order.

Where the time of a tracked run goes, measured 2026-11-14 on the design above (7 turns; from a shell;
`fast_path` and `space_charge` of track.json hold the same split for every run):

| | search (10,000, 64³ every 5 steps) | verification (100,625, 128³ every step) |
|---|---|---|
| the bunch run | 14.6 s | 277 s |
| ... compiled kernel (push, fields, metal test, kicks) | 3.1 s | 30 s |
| ... space-charge solves (PyPATools' FFT solver on the GPU, 5.2 / 28 ms each) | 8.4 s | 196 s |
| ... per-step records (turn statistics, probes, tracks) | 2.4 s | 31 s |
| ... the rest | 0.7 s | 20 s |
| before the kernel took the solve steps' kicks | 15.1-16.0 s | 388 s |

The space-charge solver is now more than half of every run. Of its 5.2 ms per solve at the search fidelity, 1.9 ms
is the Green's function of a new window (238 of 1410 solves; nearly every one is a new combination of the three cell
sizes, so a larger cache does not help: `FFTPoissonSolver(cache_size)` 4 or 128 recompute 204 or 182 of them), 1.4 ms
waiting for the GPU at the read-back, 0.8 ms the window's quantiles on the CPU. That is inside PyPATools.

A whole cheap evaluation (`pcr evaluate` on known fields) takes 27 s from a shell: start-up and imports 3 s, the
system 2.6 s, the bunch 14.6 s, the record files and sections 2 s, the scoring 5 s (`turn_centres` 2.3 s, the virtual
probes 1.9 s).

The kernel's two builds, serial below 256 particles and parallel from 256 on, are cached on disk
(`src/PyCentralRegion/__pycache__/fast_track3d._chunk3d_impl-*` and `._chunk3d_par_impl-*`): the first process after an
edit of `fast_track3d.py` compiles them, about 3.5 min, and every later one loads them in under a second. numba checks
that cache against `fast_track3d.py` only, so after an edit of a module the kernel calls into (`metal_columns.py`,
PyPATools' interpolators or pusher) touch `fast_track3d.py`: the kernel's selftest finds a stale column test or
interpolation and sends the run down the Python path, with the reason in `fast_path`.

Outputs in `run.out_dir`:
- `track.json`: summary, per-turn statistics of every run, the reference particle, the inputs;
- `track_data.pkl`: arrays (states, hits, tracks, snapshots, the model's sections);
- `probe_<angle>deg.npz`: every crossing of every particle: position, velocity, time, pass, direction;
- `config.json`.

The layout is the original design scripts' `bunch_simulation3d<tag>.json` / `_data.pkl`, so the analysis reads both.

### Obstacles that cost no BEM

Field-free additions (inside a dee, away from its gaps) change only the metal test. They can be optimised without a
new field:
- `track.extra_posts_mm = [[x, y, r], ...]` and `track.extra_boxes_mm = [[xmin, xmax, ymin, ymax], ...]`: full-height
  grounded posts and boxes: the x-y nodes of the model's metal test inside them are metal at every height.
- `[slit]`: a two-sided radial slit. Two full-height jaws on the ray at `slit.azimuth_deg`: the inner one over
  `[r_c - hw - L, r_c - hw]`, the outer one over `[r_c + hw, r_c + hw + L]`, each `slit.thickness_mm` thick
  (hw = `half_width_mm`, L = `jaw_length_mm`).
  - The centre r_c follows the design: the reference particle's radius at its crossing number `slit.pass_index` of
    the ray (the probes' pass numbering, from the reference run at the bunch's phase), plus `slit.offset_mm`. With
    `pass_index = -1` the centre is `slit.r_center_mm`.
  - The jaws are a 0.1 mm raster of their own, so the edges are good to 0.1 mm, and the compiled kernel carries it
    in a raster slot of its own, beside the model's metal test (the columns, or the slices in the other slot).
  - Refused: a jaw thinner than one tracking step at the outer jaw's outer edge (2π r / `track.steps_per_turn`,
    r = r_c + hw + L) plus the raster diagonal (√2 × 0.1 mm), because particles could step over it; and an inner jaw
    that reaches the machine axis.
  - A warning when the reference particle crosses a jaw on any pass.
  - Every run record gets `slit`: placement, jaws, raster columns, `n_lost` on the jaws, and the reference passes on
    a jaw.

### Validation of the port

The package replaces a set of machine-specific design scripts (the design deck). On one of that machine's designs:
- `pcr build` builds the deck's model bit for bit: the same vertices, triangles and potentials, a byte-identical
  `.msh`, and a `.step` that differs only in its header's time stamp (this needs the deck's gap-width floor exactly as
  it computed it, `6.5 * 1e-3`; a rounded value gives the same metal with a few ground triangles meshed differently).
- `pcr track` reproduces the deck's tracking particle for particle on the same cached fields: bit-identical without
  space charge and with it on the CPU (4000 particles, 64³ every 5 steps); on the GPU the same fates, tracks and probe
  crossings, positions within 1e-13 m (the GPU sums the deposited charge in no fixed order).
- `pcr score` gives the deck's scores exactly, but for the halo and vortex-prediction keys (at most 1.5e-4
  relative): the port uses the exact core-variance factor, 0.842330 for `score.core_q` = 0.95, where the deck used
  0.8423.
- The column metal test (the default since 2026-11-02) against the deck's slices, on the same fields: the runs are the
  same until the two tests first stop a particle differently; at the campaign's two fidelities the columns then let
  0.4-0.5 % more of the beam through, because the slices put level faces half a millimetre off (most of the
  difference at the channel's upper outer face).

The detailed records, with that machine's numbers, are kept with its configurations. `tests/test_demo.py` runs the
whole chain (build, fields, track with space charge, score) on the demo machine; a change that moves a physics result
is checked by evaluating a design before and after it at a fixed fidelity on the CPU (`track.sc_gpu = false`, whose
runs repeat bit for bit) and comparing the two `result.json`.
