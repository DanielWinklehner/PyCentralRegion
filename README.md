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
pip install -e .[track,bem]           # + gpu (cupy), mobo (optuna, torch), cavity (dfols), test (pytest)
```

## Use

Every stage reads one TOML run file; `--set section.key=value` overrides any key.

```
pcr show   configs/examples/hchc1p5/design_wnop3_rotn3_g0ap.toml      # the configuration as the stages see it
pcr build  configs/examples/hchc1p5/design_wnop3_rotn3_g0ap.toml      # 3D electrode model
pcr fields configs/examples/hchc1p5/design_wnop3_rotn3_g0ap.toml      # 3D BEM field cache
pcr track  configs/examples/hchc1p5/design_wnop3_rotn3_g0ap.toml      # bunch tracking with space charge
pcr score  configs/examples/hchc1p5/design_wnop3_rotn3_g0ap.toml      # metrics and objectives
pcr evaluate --config run.toml --out DIR                               # all four into DIR (result.json)
pcr optimize campaign.toml                                             # a MOBO campaign
```

A stage reuses an output that exists and never overwrites one. `python -m PyCentralRegion.cli` is the same command.

## Documentation

- [docs/pipeline.md](docs/pipeline.md): run files, the stages and their outputs, tracking fidelity, the obstacles
  that cost no BEM (posts, boxes, the slit), and the validation of the 3D tracking.
- [docs/mobo.md](docs/mobo.md): the optimisation campaign: variables, objectives, constraints, fidelities, resuming.
- [docs/HOWTO_mobo_campaign.md](docs/HOWTO_mobo_campaign.md): running a real campaign step by step.
- [docs/porting.md](docs/porting.md): where the design deck's scripts and environment variables went.
- `configs/defaults/*.toml`: every configuration key with its default and meaning.
- `examples/`: the 2D design workflow, one script per step.

## Tests

```
pytest
```

The fast tests need no data files. Those that need optional packages (PyPATools, optuna, torch) skip without them.
