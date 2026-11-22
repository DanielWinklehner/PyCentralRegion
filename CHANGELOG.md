# Changelog

## 2026-11-22: spiral-inflector rotation, gap-0 leave point and tilt as variables of their own

Three `[geometry]` keys, each 0 by default, which builds what the code built before (`docs/si_rotation_gap0.md`):

- `si_rotation_offset_deg` (with `si_rotation = "corner"`): the spiral-inflector assembly (housing, exit aperture,
  beam channel, inflector field, hand-off bunch) turned off the corner rule's angle. Gap 0's home point on the exit
  plate turns with it, so gap 0 leaves the plate at the same place on the plate for every offset.
- `gap0_leave_mm` (connected gap lines): where gap 0 leaves the exit plate, along the plate from its home point.
- `gap0_tilt_deg` (connected gap lines, star ground): gap 0's first segment turned clockwise off the plate about that
  point; a grounded nose fills the wedge between the plate and gap 0's dummy edge, the exit opening cut through it.

Unbuildable values are refused with a `ConfigError` (a failed evaluation in a campaign), the config-only ones at load
and at a campaign's preflight. `si_rotation = "fixed"` is unchanged. A MOBO geometry key leaves the three keys out
while they are 0, so existing stores keep their keys. `<prefix>_build.json` records gap 0's leave point (`gap_lines[0]
.leave`) and, with a tilt, the nose (`gap0_nose`). `connected_line` takes the fixed P0 off the ray (`y0`).

## 2026-11-22: main becomes the current code, machine-neutral

### Merged

- The package rework: the `pcr` command and its stages (`build`, `fields`, `track`, `score`, `evaluate`), the 3D
  pipeline, the analysis port, the MOBO driver (`pcr optimize`), posts in the loop, connected gap lines, the exact
  metal test, the dee-field and tracking speed-up of 2026-11-14 and the scorer's last-turn metrics (up to 6581c7c).
- main's commit d357f81 (probes, multiparticle objective options, frozen-gap refit, thin-gap vertical lens): the
  rework already held every piece in refactored form, so the merged tree is the rework's.
- Bunch trains (`track.bunches_ahead`, `track.bunches_behind`, opt-in, off by default). One conflict, in
  `SpaceChargeKick._solve`; a run without a train takes the speed-up's single solve unchanged.

A design evaluated with the merged code and with the code before the merge gives the same outputs bit for bit (one
design, search fidelity, space charge on the CPU; the PR has the numbers).

### No machine in the code

- `[machine]` has no defaults. Every key is the machine's; a run whose files leave one out stops with a
  `ConfigError` listing the missing keys (`docs/machine.md`). The former defaults described one particular machine.
- `[score] pick_e_min_mev` (was 2.9 MeV) and `[btrim] r_min_m`, `r_max_m` (were 0.05 and 0.135 m) default to 0,
  "not set": `metrics.pick_operating_point` and a trim with an amplitude refuse to run without them.
- `geometry.builder.configure()` without arguments resets the builder to its defaults; any other call needs a
  `MachineConfig`.
- `electrodes3d.HeightProfiles.iba` is now `HeightProfiles.reference_cavity` (the same profile). Scripts that call
  the old name need the new one.
- Comments, docstrings and docs no longer name machines, campaigns or designs; measured numbers that were in mA of
  one machine are given as shares of the beam.

### Disk roots: one paths file per computer

Run files can name a disk root, `${inputs}/maps/bfield.txt`, instead of a drive. Each computer's paths file says where
the roots are:

```toml
[meta]                    # optional, for people
name = "desktop"

[roots]                   # absolute paths only (after ~); a root does not name another root
inputs = 'E:\cyclotron'
work = 'F:\cyclotron\results'

[spyral_inflector]        # a table named after a package: only that package reads it
```

- Lookup: `pcr --paths FILE` (any command, also the MOBO commands; the evaluations inherit it), else
  `$MACHINE_PATHS`, else `$PYCENTRALREGION_PATHS`, else `~/.config/pycentralregion/paths.toml`.
- `${name}` is replaced in every string of a run file, then the path is normalised; `$$` is a literal `$`; `${pcr}`
  is built in (the package's source tree, for files kept outside it: `include = ["${pcr}/configs/defaults/core.toml"]`).
- An unknown root, a relative root, a root that names a root, a scalar at the top level or a missing file is an
  error naming the file read (or where the search looked).
- A run file without a root needs no paths file, and absolute paths keep working.
- The format is shared with the other packages of the chain (spyral_inflector reads the same file). The package ships
  only `configs/examples/paths.example.toml`.

### A worked example that needs no data: the demo machine

- `configs/examples/demo/`: `machine.toml` (a small four-dee proton cyclotron, h = 4, 1.2 T, 20 kV, 60 keV injection,
  about five turns to 0.8 MeV), `design.toml` (one design), `campaign.toml` (a 12-evaluation MOBO campaign).
- `python -m PyCentralRegion.demo` generates its six inputs (an analytic magnet map in the COMSOL text layout, the
  seed orbit, an openPMD hand-off bunch, a zero inflector field, the housing's STEP solid and outline) in about a
  minute. The whole chain then runs in about 2 min on a desktop CPU: 91.6 % of the injected 1 mA reaches
  0.813 ± 0.023 MeV.
- `tests/test_demo.py` generates the inputs into a temporary folder, checks their layouts and runs build, fields,
  track (space charge on) and score on a reduced bunch.
- The tests use the demo's machine, or the four-dee machine of `tests/machine_fixtures.py`, a test fixture.

### Moved out of the repository

To the machine's own (private) configuration repository:

| was here | now there | what |
|---|---|---|
| `configs/examples/hchc1p5/` (10 files) | `central_region/configs/hchc1p5/` | that machine's machine file, designs and campaigns, with `${inputs}` / `${work}` roots instead of drives and the `[machine]`, `[score]` and `[btrim]` values that had come from the defaults written out |
| 13 tests of those configs in `tests/test_geometry.py`, `test_mobo.py`, `test_build_cuts.py`, `test_field_symmetry.py` | `central_region/tests/test_hchc1p5_configs.py` | dry runs of its campaigns, its reference layouts, real builds of its designs |
| `docs/porting.md`, and the measured records in `docs/pipeline.md`, `docs/mobo.md`, `docs/HOWTO_mobo_campaign.md` | `central_region/docs/pcr_2026-11-21/` (the four files as they were) | the port from that machine's design scripts, its validation records, its measured costs |

The configs load and evaluate there exactly as they did here.

### Docs

- `README.md`: the generic workflow, starting from the demo.
- `docs/machine.md` (new): setting up a machine: its files, the `[machine]` keys, the inputs and their layouts, the
  paths file, a first run.
- `docs/pipeline.md`, `docs/mobo.md`, `docs/HOWTO_mobo_campaign.md`: the same contents without one machine's names
  and numbers; the HOWTO walks through the demo campaign.
