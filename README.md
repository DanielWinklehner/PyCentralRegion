# PyCentralRegion

Central-region design for compact cyclotrons, from the gap geometry to a scored 3D bunch simulation:

- **2D design** of the accelerating gaps: static and accelerated equilibrium orbits, gap geometry optimisers (thin-gap
  and 2D BEM gap models), multi-particle tracking in the median plane;
- **3D model**: OCC electrodes built around the designed orbit (dees, dummy dees, posts, the spiral-inflector
  housing), a 3D BEM solve of their field;
- **3D bunch tracking**: the spiral-inflector hand-off bunch through the BEM field and a 3D magnet map, with space
  charge (FFT Poisson, CPU or GPU), probes, tracks, metal hits, optional slit and first-harmonic trim;
- **scoring**: current, turn separation (valleys between turns at eight azimuths), bunch shape, energy spread;
- **multi-objective Bayesian optimisation** of the geometry and the cheap knobs with the whole 3D chain in the loop.

Scope today: four dees with radial gap lines. Other layouts raise `NotImplementedError` until they are implemented.

## Install

PyPATools (fields, particles, the FFT Poisson solver) is a sibling package; install it from its repository first.

```
pip install -e path/to/PyPATools
pip install -e .[track,bem]           # + gpu (cupy), mobo (optuna, torch, greenlet), cavity (dfols), test (pytest)
```

## Quick start: the demo machine

`configs/examples/demo/` is a small four-dee proton cyclotron (h = 4, 1.2 T, 20 kV dees, about five turns to
0.8 MeV) whose inputs the package generates, so the whole chain runs without any data of your own, on a CPU:

```
python -m PyCentralRegion.demo                                          # its inputs (about 1 min)
pcr evaluate --config configs/examples/demo/design.toml --out output/demo/eval   # build, fields, track, score (about 2 min)
pcr optimize configs/examples/demo/campaign.toml                        # a 12-evaluation MOBO campaign
```

`output/demo/eval/result.json` then holds the scores: about 92 % of the injected 1 mA reaches 0.81 MeV.

## Use

Every stage reads one TOML run file; `--set section.key=value` overrides any key.

```
pcr show   design.toml                  # the configuration as the stages see it
pcr build  design.toml                  # 3D electrode model
pcr fields design.toml                  # 3D BEM field cache
pcr track  design.toml                  # bunch tracking with space charge
pcr score  design.toml                  # metrics and objectives
pcr evaluate --config design.toml --out DIR    # all four into DIR (result.json)
pcr optimize campaign.toml              # a MOBO campaign
```

A stage reuses an output that exists and never overwrites one. `python -m PyCentralRegion.cli` is the same command.

A machine is a machine file, its designs and its input files, kept outside this repository. Run files can name disk
roots (`${inputs}/maps/bfield.txt`) that each computer's paths file sets (`--paths FILE`, `$MACHINE_PATHS`, or
`~/.config/pycentralregion/paths.toml`; `configs/examples/paths.example.toml`), so the same files run on every
computer. [docs/machine.md](docs/machine.md) sets up a machine from the demo.

## Documentation

- [docs/machine.md](docs/machine.md): setting up a machine: its files, the `[machine]` keys, the inputs, the paths file.
- [docs/pipeline.md](docs/pipeline.md): run files, the stages and their outputs, tracking fidelity, the obstacles
  that cost no BEM (posts, boxes, the slit), and the validation of the build, the tracking and the scoring.
- [docs/mobo.md](docs/mobo.md): the optimisation campaign: variables, objectives, constraints, fidelities, resuming.
- [docs/HOWTO_mobo_campaign.md](docs/HOWTO_mobo_campaign.md): running a campaign step by step.
- [CHANGELOG.md](CHANGELOG.md): what changed, and what moved out of the repository.
- `configs/defaults/*.toml`: every configuration key with its default and meaning.
- `examples/`: the 2D design workflow, one script per step, on a small four-dee cyclotron: static orbits, 2D
  acceleration and bunch tracking, the per-gap cavity optimisation, solved 2D gap fields, a cavity centre's field from
  its CAD, and 3D electrodes ([examples/README.md](examples/README.md)).
- `python -m PyCentralRegion.geometry.build --help`: the electrode builder's own tools besides `build`: `bounds` (the
  channel-height bounds for a campaign's ranges), `hole` (a gap-0 hole from the hand-off bunch), `shift`, `cut` and
  `sections`.

## Tests

```
pytest
```

The tests need no data files. `tests/test_demo.py` generates the demo's inputs into a temporary folder and runs the
whole chain on a reduced bunch (about a minute); tests that need optional packages (scipy, matplotlib, PyPATools,
gmsh, bempp, optuna, torch) skip without them.
