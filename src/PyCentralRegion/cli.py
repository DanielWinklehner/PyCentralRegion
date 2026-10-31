"""`pcr`, the command line of the pipeline: one subcommand per stage, each reading a TOML run file.

    pcr show     run.toml [--set k=v ...]   the resolved configuration, as a stage sees it
    pcr build    run.toml [--set ...]       3D electrode model -> paths.model3d              (geometry.build)
    pcr fields   run.toml [--set ...]       3D BEM field cache -> paths.fields               (simulate3d.fields_stage)
    pcr track    run.toml [--set ...]       bunch tracking -> run.out_dir                    (simulate3d.track_stage)
    pcr score    run.toml [--set ...]       metrics of run.out_dir -> run.out_dir/score.json (analysis)
    pcr evaluate --config run.toml [--set ...] --out DIR
                                            build -> fields -> track -> score into DIR; DIR/result.json
    pcr optimize mobo.toml [...]            multi-objective Bayesian optimisation campaign   (mobo.driver)

A stage reuses an output that exists and never overwrites one: point the path elsewhere to recompute. ``--set
section.key=value`` overrides the files; the value is read as TOML (``--set track.phases_deg=[-15.0]``), else as a
string. ``python -m PyCentralRegion.cli`` is the same command.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from . import config


def _load(path, overrides):
    return config.load(path, overrides=overrides or [])


def _free_gpu():
    """Return cupy's cached GPU blocks between stages (the BEM solve leaves gigabytes in the pool)."""
    try:
        import cupy
    except ImportError:
        return
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()


def build_stage(cfg):
    from .geometry.build import build
    return build(cfg)


def score_stage(cfg):
    """Score the tracked run in run.out_dir with [score]; writes run.out_dir/score.json (reused when it exists)."""
    from .analysis.metrics import score_run
    from .analysis.runs import load_run_dir
    out_dir = Path(config.require(config.section(cfg, 'run').out_dir, 'run.out_dir'))
    path = out_dir / 'score.json'
    if path.exists():
        print(f"[score] {path} exists: reused", flush=True)
        return json.loads(path.read_text())
    scores = score_run(load_run_dir(out_dir), config.section(cfg, 'score'))
    tmp = path.with_suffix('.json.part')
    tmp.write_text(json.dumps(scores, indent=1, default=str))
    tmp.replace(path)
    return scores


def evaluate(cfg):
    """build (when paths.model3d is missing) -> fields (when paths.fields is missing) -> track -> score."""
    from .simulate3d import fields_stage, track_stage
    pc = config.section(cfg, 'paths')
    if not (pc.model3d and Path(pc.model3d).exists()):
        build_stage(cfg)
    if not (pc.fields and Path(pc.fields).exists()):
        fields_stage(cfg)
        _free_gpu()
    track_stage(cfg)
    _free_gpu()
    return score_stage(cfg)


def _cmd_evaluate(args):
    from .mobo.evaluate import write_result
    t0 = time.time()
    out = Path(args.out).resolve()
    cfg = {}
    try:
        cfg = _load(args.config, list(args.set) + [f'run.out_dir={json.dumps(str(out))}'])
        scores = evaluate(cfg)
    except Exception as exc:                       # the evaluation failed: say why in result.json, exit 1
        traceback.print_exc()
        write_result(out, 'failed', error=f"{type(exc).__name__}: {exc}", wall_s=time.time() - t0, config=cfg)
        return 1
    write_result(out, 'ok', scores=scores, wall_s=time.time() - t0, config=cfg)
    return 0


def _stage_command(fn):
    def run(args):
        fn(_load(args.config, args.set))
        return 0
    return run


def _cmd_show(args):
    print(json.dumps(_load(args.config, args.set), indent=1, sort_keys=True))
    return 0


def _cmd_track(args):
    from .simulate3d import track_stage
    track_stage(_load(args.config, args.set))
    return 0


def _cmd_fields(args):
    from .simulate3d import fields_stage
    fields_stage(_load(args.config, args.set))
    return 0


def _cmd_optimize(args):
    from .mobo.driver import main as driver_main
    return driver_main(args.rest)


def main(argv=None):
    p = argparse.ArgumentParser(prog='pcr', description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    for name, fn, text in (('show', _cmd_show, 'print the resolved configuration'),
                           ('build', _stage_command(build_stage), 'build the 3D electrode model'),
                           ('fields', _cmd_fields, 'solve the 3D BEM field of the model'),
                           ('track', _cmd_track, 'track the bunch through the 3D fields'),
                           ('score', _stage_command(score_stage), 'score the tracked run')):
        s = sub.add_parser(name, help=text)
        s.add_argument('config', help='run TOML')
        s.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                       help='config override (repeatable; later ones win)')
        s.set_defaults(func=fn)
    from .mobo.evaluate import add_arguments
    s = add_arguments(sub.add_parser('evaluate', help='build -> fields -> track -> score into --out (result.json)'))
    s.set_defaults(func=_cmd_evaluate)
    s = sub.add_parser('optimize', help='run or continue a MOBO campaign (pcr optimize --help)')
    s.add_argument('rest', nargs=argparse.REMAINDER)
    s.set_defaults(func=_cmd_optimize)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
