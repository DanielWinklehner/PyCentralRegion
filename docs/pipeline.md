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
out_dir = "E:/results/my_design"

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
- No stage reads an environment variable.

Worked example: `configs/examples/hchc1p5/` (`machine.toml` is the machine, `design_wnop3_rotn3_g0ap.toml` one design
whose cached model and fields are on disk).

## Stages

| command | reads | writes | notes |
|---|---|---|---|
| `pcr build` | `[geometry]`, `[machine]`, `[inflector]`, `paths.seed`, `paths.housing_outline`, the emitting-plane hand-off (`geometry.emitting_plane_file`, else the export's own beside the outline, else `paths.handoff`), `exit_plane_spec.json` (for `si_rotation = "corner"` or `gap0_on_plate`) | `paths.model3d`, and beside it `<prefix>.msh`, `<prefix>.step`, `<prefix>_build.json`, `<prefix>_dees.png` | OCC solids, mesh, electrode potentials (`geometry.build`); the `.step` is the CAD: 5 solids (dee0..dee3, the ground with the spiral-inflector housing fused in), mm, machine frame |
| `pcr fields` | `paths.model3d`, `paths.seed`, `[fields]`, `[machine]` | `paths.fields` | 3D BEM solve, E-field grid, metal raster |
| `pcr track` | `paths.fields`, `paths.seed`, `paths.bfield_3d`, `paths.handoff`, `paths.inflector_efield`, `paths.housing_step`, `[run]`, `[machine]`, `[magnet]`, `[btrim]`, `[inflector]`, `[fields]`, `[track]`, `[slit]` | `run.out_dir`: `track.json`, `track_data.pkl`, `config.json`, `probe_<a>deg.npz` | reference, phase scan, final run |
| `pcr score` | `run.out_dir` (its `config.json` gives the `[machine]` the run was tracked with), `[score]` | `run.out_dir/score.json` | metrics and objectives (`analysis`); a run with fewer than `score.min_tracks` tracks or `score.min_survivors` survivors scores `{"tag", "skip"}` |
| `pcr evaluate --config run.toml --out DIR` | all of the above | `DIR/result.json` | the chain; a failure is written to result.json, exit code 1 |

Reuse rule: a stage whose output exists says so and returns it. To recompute, point the path elsewhere. The outputs
a stage is judged done by are written under a temporary name and renamed: `<file>.part` for the field cache,
track.json and track_data.pkl, `<stem>.part<suffix>` for the build's (the model last), `<file>.tmp` for score.json and
result.json. A killed run never leaves a truncated file to be reused. A `pcr build` that fails removes its temporary
files; the bounds it can check before the layout (the channel heights) are checked before anything is written.

### fields

- The BEM solution of the model (every dee at its potential, ground at 0) is sampled on a grid out to the model's
  `r_cut + fields.margin_m`: `fields.dxy_m` across, `fields.dz_m` in height, `|z| <= fields.z_half_m`.
- The metal is rasterised on z slices (`fields.stack_*`, 0.5 mm x 1 mm by default). That raster is the tracker's
  obstacle test.
- The cache keeps the layout of the design deck's `bunch_fields3d<tag>.pkl`, so deck caches track here unchanged.
- Solver choices: `fields.solver` (auto | cupy | scipy | strong | dense), potentials on the GPU in single precision
  with `fields.pot_check` points re-evaluated in double precision.

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
| verification | all (~98 000) | 128³ every step | a design worth building; only z rms is grid-sensitive |

The compiled 3D kernel (`track.fast3d`, numba) runs the whole bunch: release, space-charge kicks, the metal test and
the probes. It needs:
- the `boris_rel` pusher;
- plain gridded fields;
- at most two nearest-node obstacle rasters: the model's (with any extra posts and boxes painted in) and the slit's.

Every run record of track.json carries `fast_path`: whether the compiled kernel ran and, if not, why. A run that
cannot use it goes the Python path: same answer, far slower.

Outputs in `run.out_dir`:
- `track.json`: summary, per-turn statistics of every run, the reference particle, the inputs;
- `track_data.pkl`: arrays (states, hits, tracks, snapshots, the model's sections);
- `probe_<angle>deg.npz`: every crossing of every particle: position, velocity, time, pass, direction;
- `config.json`.

The layout is the deck's `bunch_simulation3d<tag>.json` / `_data.pkl`, so the analysis reads both.

### Obstacles that cost no BEM

Field-free additions (inside a dee, away from its gaps) change only the metal test. They can be optimised without a
new field:
- `track.extra_posts_mm = [[x, y, r], ...]` and `track.extra_boxes_mm = [[xmin, xmax, ymin, ymax], ...]`: full-height
  grounded posts and boxes, painted into the model raster.
- `[slit]`: a two-sided radial slit. Two full-height jaws on the ray at `slit.azimuth_deg`: the inner one over
  `[r_c - hw - L, r_c - hw]`, the outer one over `[r_c + hw, r_c + hw + L]`, each `slit.thickness_mm` thick
  (hw = `half_width_mm`, L = `jaw_length_mm`).
  - The centre r_c follows the design: the reference particle's radius at its crossing number `slit.pass_index` of
    the ray (the probes' pass numbering, from the reference run at the bunch's phase), plus `slit.offset_mm`. With
    `pass_index = -1` the centre is `slit.r_center_mm`.
  - The jaws are a 0.1 mm raster of their own, so the edges are good to 0.1 mm, and the compiled kernel carries it as
    its second raster.
  - Refused: a jaw thinner than one tracking step at the outer jaw's outer edge (2π r / `track.steps_per_turn`,
    r = r_c + hw + L) plus the raster diagonal (√2 × 0.1 mm), because particles could step over it; and an inner jaw
    that reaches the machine axis.
  - A warning when the reference particle crosses a jaw on any pass.
  - Every run record gets `slit`: placement, jaws, raster columns, `n_lost` on the jaws, and the reference passes on
    a jaw.

### Validation of the port

Every record is in `E:\HCHC1.5\Results\CR\pcr_validation\`.

`pcr build` replaces the design deck's electrode builder. The worked example's `design_wnop3_rotn3_g0ap.toml` builds
the deck's 2026-10-31 model bit for bit (`geometry\`): the same vertices, triangles and potentials (20,974 nodes,
41,936 triangles), a byte-identical `.msh`, and a `.step` that differs only in its header's time stamp. The build takes
17-18 s. This needs the deck's gap-width floor 6.5 * 1e-3 = 0.006500000000000001 m, which the example's
`machine.toml` sets; 0.0065 gives the same metal with 8 more ground triangles.

`pcr score` replaces the run scoring of 2026-11-01 (`analysis\`): on the every-particle run of the design (`p128`),
T, I_final (4.0239 mA), valley_all, ratio, E_med, dE_E_robust, turn_sigR, W_eff and ratio_t3 equal the original
scores exactly; the halo and vortex-prediction keys differ by at most 1.5e-4 relative, because the port uses the
exact core-variance factor (0.842330 for `score.core_q` = 0.95) where the campaign used 0.8423.

`pcr track` replaces the design deck's 3D tracking script. On the deck's cached `_wnop3_rotn3_g0ap` fields at -15 deg
it reproduces the deck particle for particle (records and scripts in `track\`):

| run | agreement | T, final current |
|---|---|---|
| no SC, every particle | bit-identical | 55.22 % = 4.436 mA |
| 4000 particles, SC 64³ / 5, CPU | bit-identical | 49.43 % = 3.970 mA |
| the same on the GPU | same fates, positions within 9e-14 m | 49.43 % = 3.970 mA |
| every particle, SC 128³ / 1, GPU | same fates, tracks and probe crossings; within 5e-14 m | 50.09 % = 4.024 mA |

The port's verification run took 429 s, the deck's 510 s.
