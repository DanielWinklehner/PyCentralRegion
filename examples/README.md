# Examples

One script per step of the central-region design workflow, on a small four-dee muon cyclotron (h = 4): static
orbits, single-particle and bunch acceleration with thin-gap kicks, the per-gap cavity geometry optimization,
solved 2D gap fields, then the tools that turn a design into metal (a cavity centre from its CAD, 3D electrodes).
Later examples build on earlier results where the table says so. Every script explains in its docstring what it
shows, what it reads and writes, and how long it takes (`python examples/NN_name.py --help`).

## Running

```
python examples/01_find_static_orbits.py --resources D:/path/to/inputs --out D:/path/to/results
```

* `--resources DIR`: the folder with the input files below (default `<repo>/resources`). The input files are
  not part of the repository (`resources/` is git-ignored), so point this at your copy.
* `--out DIR`: where the results go (default `<repo>/output`). A script that builds on another one's result reads
  it from here, so use one `--out` for a chain (06 -> 07 -> 12, 09 -> 10).
* `_common.py` puts this checkout's `src/` first on `sys.path`, so the scripts run against the code next to them
  whether or not the package is installed.
* No GPU is needed. A BEM solve uses the GPU only when cupy finds one and the dense matrix fits its memory
  (`--solver scipy` in 11 and 12, or `CUDA_VISIBLE_DEVICES=-1`, keeps it on the CPU). The dense single-layer
  matrix of an N-element model takes 8 N^2 bytes of RAM (25 GB at 56k elements).

## Inputs

| file | what | used by |
| --- | --- | --- |
| `midplane_field_0.5mm.comsol` | COMSOL export of the median-plane Bz of the muon cyclotron, 0.5 mm grid over +-400 mm (150 MB) | 01, 02, 04, 05 |
| `uCyclo_v3_midplane_field.comsol` | the same export of a later iteration of that magnet (140 MB) | 06-10, 12 |
| `MuonBunchOutOfInflector.csv` | 250 muons at the spiral-inflector exit: x y z [m], vx vy vz [m/s] | 09, 10 |
| `isodar_cavity_efield_center_2mm.npz` | COMSOL E-field of the centre of a 4-dee RF cavity, one quarter, 2 mm grid | 11 |
| a STEP file (`--step`) | vacuum CAD of that cavity quarter [mm] | 11 |

## The examples

Runtimes measured on the CPU (28 logical cores, 6-8 numba / BLAS threads per process) with the options given.

| script | shows | reads | runtime |
| --- | --- | --- | --- |
| `01_find_static_orbits.py` | static equilibrium orbits 50-300 mm (SEOFinder), closure, isochronism | field map | 1.5 min |
| `02_optimize_acceleration.py` | bunch phase, RF frequency and injection point of one particle to 5 MeV (thin-gap kicks, Nelder-Mead) | field map | 30 s |
| `03_analyze_optimization.py` | post-processing of 02's per-evaluation checkpoint CSV | 02's CSV | seconds |
| `04_multiparticle_tracking_2D.py` | a 100-particle Gaussian bunch through 02's RF system, per-turn statistics | field map | 15 s |
| `05_optimize_multiparticle_2D.py` | bunch phase and injection offset optimized for a bunch (envelope term) | field map | 1 min (`--maxiter 3`), 5 min default |
| `06_optimize_cavity_geometry.py` | staged per-gap geometry + RF optimization of a dee system (RF scan, multi-start DFO-LS, verify) | field map | 1 min |
| `07_bem_gap_verification.py` | 06's winner through the solved 2D gap field vs thin-gap kicks | 06's pickle | 1-2 min |
| `08_optimize_cavity_geometry_bem.py` | 06's optimization with the BEM gap field re-solved for every geometry | field map | 1.5 min (`--workers 2 --maxfun 5`), ~20 min per start default |
| `09_optimize_cavity_geometry_multiparticle.py` | 06's optimization for the inflector-exit bunch, with a phase-slit collimator | field map, bunch | 4 min (`--particles 8 --workers 2 --maxfun 10`), overnight for the full bunch |
| `10_center_collimator.py` | azimuth and aperture scan of a first-turn phase slit on 09's winner | 09's pickle, bunch | 6 min (`--particles 8`) |
| `11_cad_bem_vs_comsol.py` | the BEM field of a cavity centre built from its CAD against a COMSOL map | CAD, E-field map | 1.5 min (56k elements, 25 GB RAM) |
| `12_electrodes3d.py` | 06's 2D electrodes as 3D solids (dee aperture, hill gap, roof, bars), solved and re-tracked | 06's pickle | 10 min (`--size-min 0.004 --size-max 0.016 --skip-regression`, 25 GB RAM) |

Options that shorten the long ones: `--rf-frequency 42.2234e6` (06-09) skips the isochronous-frequency scan,
`--particles N` (09, 10) tracks a deterministic subsample of the bunch, `--maxfun N` (06, 08, 09) caps the DFO-LS
evaluations per start, and `--size-min / --size-max` (11, 12) coarsen the BEM mesh.

Scope: the examples, like the package, cover four-dee machines with radial gap lines; the 3D bunch tracking with
space charge, the spiral inflector and the configuration-driven runs are the `pcr` command line (see the
repository README).
