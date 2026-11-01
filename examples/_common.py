"""Shared plumbing of the examples: which package they import, where their inputs are, where their results go.

Every example takes ``--resources DIR`` (its input files; default ``<repo>/resources``) and ``--out DIR`` (its
results; default ``<repo>/output``). An example that builds on another one's result reads that result from
``--out`` and imports the other example as a module (``load_example``) to rebuild the identical system.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# this checkout's package, whether or not (and wherever) PyCentralRegion is installed
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))


def arguments(doc):
    """Argument parser with the common ``--resources`` / ``--out`` options, described by the first line of
    ``doc`` (the example's docstring); add the example's own options to it."""
    ap = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    ap.add_argument('--resources', type=Path, default=ROOT / 'resources',
                    help='folder with the input files (default: <repo>/resources)')
    ap.add_argument('--out', type=Path, default=ROOT / 'output',
                    help='folder for the results (default: <repo>/output)')
    return ap


def parse(ap):
    """Parsed arguments, with the output folder created."""
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def require(path, hint=''):
    """``path`` if it exists, else exit with a message naming it (and how to make it)."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"input not found: {path}" + (f" - {hint}" if hint else ''))
    return path


def bunch_snapshot(full_beam, step, species):
    """Radius [mm], radial velocity [m/s] and kinetic energy [MeV] of the particles still tracked at
    ``step`` of a tracking result's ``metadata['full_beam']`` ((steps, particles, 6): x y z vx vy vz).
    The tracker leaves a particle where it was lost or left the tracking radius, so a particle counts as
    tracked when its row changes over the neighbouring step (rows never recorded are NaN). The result's
    Poincare data hold the reference particle only, so bunch phase space comes from here."""
    import numpy as np
    from PyPATools.global_variables import CLIGHT
    step %= len(full_beam)
    s = full_beam[step]
    other = full_beam[step + 1 if step + 1 < len(full_beam) else step - 1]
    s = s[~np.isnan(s[:, 0]) & np.any(s != other, axis=1)]
    r = np.hypot(s[:, 0], s[:, 1])
    vr = (s[:, 0] * s[:, 3] + s[:, 1] * s[:, 4]) / r
    beta2 = (s[:, 3:6] ** 2).sum(axis=1) / CLIGHT ** 2
    return r * 1e3, vr, species.mass_mev * (1.0 / np.sqrt(1.0 - beta2) - 1.0)


def load_example(filename):
    """Another example (by file name) as a module, e.g. to reuse its system builder; its main() is not run."""
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(f"example_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
