"""Toy evaluator: the evaluator contract of `pcr evaluate`, answered in milliseconds by analytic functions.

    python -m PyCentralRegion.mobo.toy --config <toml> --set key=value ... --out <eval_dir>

reads the [toy] table of the run config and writes result.json with scores named like the real ones, so a toy
campaign has the shape of a real campaign: tests and dry runs exercise the whole driver (subprocesses, store,
geometry cache, fidelities, failures) without the physics stack or a GPU. Nothing here is physics.

The problem is ZDT1 (Zitzler, Deb, Thiele, Evol. Comput. 8, 173 (2000)) dressed up. The RF phase moves along the
front, p = (phase + 40) / 40 in [0, 1]; the other variables only add a distance g - 1 >= 0 from it:

    g          = 1 + 9/4 (dv^2 + dt^2 + dw^2 + dk^2)      dv = (volt_scale - 1.02) / 0.15
                                                          dt = (log10 trim_t + 3.5) / 3
                                                          dw = (gap_width_mm - 4.4) / 6
                                                          dk = (n_posts - 2) / 4
    I_final    = 1 - p                                    (maximise)
    valley_all = g (1 - sqrt(p / g))                      (minimise)
    ratio      = g (1.5 + 0.5 p^2)                        (minimise)
    E_med      = 3.1 - 0.3 I_final^2                      (constraint E_med >= 2.9  <=>  I_final <= sqrt(2/3))

so the Pareto front of (I_final, valley_all) is valley_all = 1 - sqrt(1 - I_final) at g = 1, i.e. volt_scale = 1.02,
trim_t = 10^-3.5, gap_width_mm = 4.4 and n_posts = 2, and the constraint cuts it at I_final = sqrt(2/3).

Geometry: when paths.model3d / paths.fields are set the toy "builds" them (small JSON files holding the geometry
values) unless they exist, and fails when existing fields were made for another geometry, which is how the tests
check that the driver shares a geometry's files only between evaluations of that geometry. gap_width_mm above
fail_above_mm makes the build fail (failure handling); noise adds a deterministic pseudo-random error of that
amplitude to every score (a low-fidelity search); sleep_s makes an evaluation take time after its build (one stopped
then has made its geometry's files, as a real one stopped during its tracking has: interruption tests);
child_sleep_s > 0 starts a child process that sleeps that long and is never waited for, its pid in
<eval_dir>/child.pid (the driver must stop it with the evaluation: process-tree tests).

A late RF phase loses the beam (I_final -> 0 at 0 deg), like a real design: above short_above_deg the run is too
short of survivors for the probes' turn structure, so its scores have no valley_all (as the scorer leaves out
valley_all and turn_* without probes with enough complete passes); above skip_above_deg the scorer skips the run
({'tag', 'skip': reason}, no metrics, as with too few survivors). Both finish ok: they are bad designs, not
failures (the driver's unusable scores).
"""
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from .. import config
from .evaluate import evaluator_parser, write_result

DEFAULTS = {'phases_deg': [-20.0], 'volt_scale': 1.0, 'trim_t': 1e-3, 'gap_width_mm': 3.0, 'n_posts': 2,
            'fail_above_mm': 1e9, 'short_above_deg': 1e9, 'skip_above_deg': 1e9, 'noise': 0.0, 'sleep_s': 0.0,
            'child_sleep_s': 0.0}
GEOMETRY = ('gap_width_mm', 'n_posts')
I_MAX = math.sqrt(2.0 / 3.0)         # the constraint's end of the front


def distance(t):
    """g - 1: how far the non-phase variables put a design from the front."""
    dv = (t['volt_scale'] - 1.02) / 0.15
    dt = (math.log10(t['trim_t']) + 3.5) / 3.0
    dw = (t['gap_width_mm'] - 4.4) / 6.0
    dk = (t['n_posts'] - 2) / 4.0
    return 2.25 * (dv * dv + dt * dt + dw * dw + dk * dk)


def scores(t):
    phase = float(t['phases_deg'][0])
    if phase > t['skip_above_deg']:
        return {'tag': 'toy', 'skip': f"12 tracked survivors (RF phase {phase:g} deg > skip_above_deg)"}
    p = min(max((phase + 40.0) / 40.0, 0.0), 1.0)
    g = 1.0 + distance(t)
    current = 1.0 - p
    out = {'I_final': current, 'valley_all': g * (1.0 - math.sqrt(p / g)), 'ratio': g * (1.5 + 0.5 * p * p),
           'E_med': 3.1 - 0.3 * current * current, 'distance': g - 1.0}
    if t['noise']:
        design = [t[k] for k in ('phases_deg', 'volt_scale', 'trim_t', 'gap_width_mm', 'n_posts')]
        seed = hashlib.sha256(json.dumps(design).encode()).digest()
        for i, key in enumerate(('I_final', 'valley_all', 'ratio', 'E_med')):
            out[key] += t['noise'] * (seed[i] / 127.5 - 1.0)
    if phase > t['short_above_deg']:
        del out['valley_all']
    return out


def front(current):
    """valley_all on the Pareto front."""
    return 1.0 - math.sqrt(1.0 - current)


def hypervolume(points):
    """Area dominated by (I_final, valley_all) points (max, min) inside the box I_final >= 0, valley_all <= 1;
    the feasible front's is (2/3) (1 - (1 - I_MAX)^1.5) = 0.614."""
    area, v_best = 0.0, 1.0
    pts = sorted((max(c, 0.0), min(v, 1.0)) for c, v in points)
    for k in range(len(pts) - 1, -1, -1):          # sweep from the highest current down
        c, v = pts[k]
        v_best = min(v_best, v)
        c_next = pts[k - 1][0] if k else 0.0
        area += (c - c_next) * (1.0 - v_best)
    return area


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config.write_atomic(path, lambda p: p.write_text(json.dumps(data)))


def build(cfg, t):
    """Emulate `pcr build` / `pcr fields`: write the model and fields files unless they exist, then check that
    the fields belong to this geometry."""
    paths = cfg.get('paths', {})
    model, fields = paths.get('model3d', ''), paths.get('fields', '')
    geometry = {k: t[k] for k in GEOMETRY}
    if t['gap_width_mm'] > t['fail_above_mm']:
        raise RuntimeError(f"toy build failed: gap_width_mm {t['gap_width_mm']} > {t['fail_above_mm']}")
    if model and not Path(model).exists():
        _write_json(model, geometry)
    if fields and not Path(fields).exists():
        _write_json(fields, {'geometry': geometry, 'model3d': model})
    if fields and json.loads(Path(fields).read_text())['geometry'] != geometry:
        raise RuntimeError(f"{fields} was made for another geometry than {geometry}")


def main(argv=None):
    args = evaluator_parser('python -m PyCentralRegion.mobo.toy', 'toy evaluator').parse_args(argv)
    t0 = time.monotonic()
    cfg = config.load(args.config, overrides=args.set)
    t = {**DEFAULTS, **cfg.get('toy', {})}
    try:
        build(cfg, t)
        if t['child_sleep_s'] > 0:
            child = subprocess.Popen([sys.executable, '-c', f"import time; time.sleep({float(t['child_sleep_s'])})"])
            _write_json(Path(args.out) / 'child.pid', child.pid)
        time.sleep(float(t['sleep_s']))
        result = scores(t)
    except Exception as ex:                       # the contract: report the failure in result.json, exit 1
        write_result(args.out, 'failed', error=f"{type(ex).__name__}: {ex}", wall_s=time.monotonic() - t0,
                     config=cfg)
        return 1
    write_result(args.out, 'ok', scores=result, wall_s=time.monotonic() - t0, config=cfg)
    return 0


if __name__ == '__main__':
    sys.exit(main())
