# Setting up a machine

The package knows no machine. A machine is a few TOML files and its input files, kept wherever you keep them (a
private repository is a good place); the demo machine in `configs/examples/demo/` is the template. Scope: four dees
with radial gap lines; any other layout raises `NotImplementedError` up front.

## 1. The files

| file | holds | demo |
|---|---|---|
| `machine.toml` | what every design of the machine shares: `[machine]`, `[magnet]`, `[paths]` of its inputs, the machine's `[score]` and `[btrim]` values | `configs/examples/demo/machine.toml` |
| `design.toml` | one design: `include = ["machine.toml"]`, its electrode options (`[geometry]`), field grid (`[fields]`), run settings (`[track]`), output paths | `configs/examples/demo/design.toml` |
| `campaign.toml` | a MOBO campaign over a design (`[mobo]` only, [mobo.md](mobo.md)) | `configs/examples/demo/campaign.toml` |
| a paths file | this computer's disk roots, for run files that name `${inputs}`, `${work}`, ... | none: the demo uses relative paths |

A file sets only what differs from the defaults (`configs/defaults/*.toml`, every key with its meaning) and includes
what it builds on; the including file wins. `pcr show design.toml` prints the configuration as the stages see it.

## 2. `[machine]`: no defaults

Every key of `[machine]` is the machine's, so none has a default: a run whose files leave one out stops with a
`ConfigError` that lists the missing keys. The keys (`configs/defaults/core.toml`):

| key | meaning |
|---|---|
| `species` | PyPATools species name (`"proton"`, `"H2_1+"`, `"4He_2+"`, ...) |
| `harmonic` | RF harmonic h |
| `n_dees`, `dee_shape`, `dee_center_angles_deg`, `dee_opening_deg` | the dee layout (4, `"radial"`, one centre azimuth per dee, the angular width) |
| `dee_voltage_kv` | the voltage the electrode model is built and its BEM field normalised at |
| `design_voltage_kv` | the voltage of the 1D gap model the seed geometry was optimised with |
| `gap_width_m`, `gap_angle_deg`, `gap_width_min_m` | the gap-width law: max(`gap_width_min_m`, r × `gap_angle_deg`), and the template's nominal width |
| `r_min_m`, `r_max_cavity_m` | inner and outer end of the gap lines |
| `r_max_m`, `z_max_m` | tracking: particles beyond these are lost |
| `target_energy_mev` | tracking: a run ends at the turn its reference reaches it (`track.stop_at_target_energy`) |
| `n_turns_design`, `skip_turns_design` | the turns of the seed optimisation (to rebuild its dee system) |

Two settings outside `[machine]` are the machine's too and default to 0, "not set":
- `[score] pick_e_min_mev`: the final energy an operating point's bunch must reach. `analysis.metrics.pick_operating_point`
  refuses to pick without it.
- `[btrim] r_min_m`, `r_max_m`: the radii over which a first-harmonic trim has its full amplitude. A trim with an
  amplitude refuses to run without them. Only set a trim for a machine that has trim coils.

## 3. The inputs

| `[paths]` key | what | layout |
|---|---|---|
| `bfield_3d` | the 3D magnet map | COMSOL text export (`x y z Bx By Bz` in SI, a full regular grid with a z = 0 plane; `fields3d.crop_comsol_3d`) |
| `seed` | the 2D gap geometry and its reference orbit | an `AcceleratedOrbitFinder.OptimizedOrbit` pickle from the 2D optimisers (`examples/`) |
| `handoff` | the injected bunch where the spiral inflector hands it over | openPMD plane crossing, with the PyPATools attributes (species, RF frequency, current) |
| `inflector_efield` | the inflector's static E-field | a PyPATools Field pickle |
| `housing_step` | the inflector housing, the tracker's obstacle | STEP, mm, machine frame |
| `housing_outline` | its outline at the median plane and its exit aperture, for the electrode builder | text, mm |

`python -m PyCentralRegion.demo` writes each of them in the layout the pipeline reads (`PyCentralRegion.demo`
documents every one): when an input of your own does not load, compare it with the demo's.

## 4. The paths file

Drive letters and mount points belong to a computer, not to a run file. A run file names a root instead,
`bfield_3d = "${inputs}/maps/bfield.txt"`, and each computer's paths file says where that root is:

```toml
[roots]
inputs = 'E:\cyclotron'          # absolute paths only (a leading ~ is your home); roots do not name other roots
work = '/mnt/results/cyclotron'
```

- The file read is `--paths FILE` (any `pcr` command), else `$MACHINE_PATHS`, else `$PYCENTRALREGION_PATHS`, else
  `~/.config/pycentralregion/paths.toml`. The MOBO driver's evaluations inherit it.
- `${pcr}` is the package's source tree (built in, not a root of the file); `$$` is a literal `$`.
- A run file that names no root needs no paths file. An unknown root, or a root without a paths file, is an error
  that names the file read or where the search looked.
- `[meta]` and tables named after a package (`[PyCentralRegion]`, `[spyral_inflector]`, ...) may hold anything; the
  other packages ignore them. No other keys at the top level.
- Absolute paths in run files keep working.

`configs/examples/paths.example.toml` is a template. The package ships no paths file: keep yours with the machine's
files, one per computer.

## 5. A first run

1. Copy `configs/examples/demo/` to the machine's own folder. Fill `machine.toml`: `[machine]`, `[magnet]`
   (`sign`, `scale`, `crop_r_m`), `[paths]` (with roots), `[score] pick_e_min_mev`.
2. In `design.toml`, size the electrode options for the machine: `python -m PyCentralRegion.geometry.build bounds
   design.toml` prints the channel-height bounds; the build refuses heights outside them.
3. `pcr show design.toml`: every key as the stages will see it.
4. `pcr build design.toml`, then look at `<prefix>_dees.png` beside the model (the gap lines, the dees, the housing).
5. `pcr fields`, `pcr track`, `pcr score` (or `pcr evaluate --config design.toml --out DIR` for all four). Every
   stage reuses an output that exists ([pipeline.md](pipeline.md)).
6. A campaign: start from `configs/examples/demo/campaign.toml` ([HOWTO_mobo_campaign.md](HOWTO_mobo_campaign.md)).

The demo runs the whole chain in about 2 min on a desktop CPU and is what `tests/test_demo.py` checks; a full-size
machine's model takes minutes per stage and a GPU for the space-charge solves ([mobo.md](mobo.md), "Costs").
