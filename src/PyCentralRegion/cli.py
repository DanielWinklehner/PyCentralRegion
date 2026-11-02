"""`pcr`, the command line of the pipeline: one subcommand per stage, each reading a TOML run file.

    pcr show     run.toml [--set k=v ...]   the resolved configuration, as a stage sees it
    pcr build    run.toml [--set ...]       3D electrode model -> paths.model3d              (geometry.build)
    pcr fields   run.toml [--set ...]       3D BEM field cache -> paths.fields               (simulate3d.fields_stage)
    pcr track    run.toml [--set ...]       bunch tracking -> run.out_dir                    (simulate3d.track_stage)
    pcr score    run.toml [--set ...]       metrics of run.out_dir -> run.out_dir/score.json (analysis)
    pcr evaluate --config run.toml [--set ...] --out DIR
                                            build -> fields -> track -> score into DIR; DIR/result.json
    pcr optimize        campaign.toml [--max-evals N] [--max-hours H] [--dry-run] [--set k=v ...]
                                            run or continue a multi-objective Bayesian optimisation (mobo.driver)
    pcr optimize-verify campaign.toml (--ids ID ... | --pareto) [--fidelity NAME] [--set k=v ...]
                                            re-evaluate evaluations at the verify fidelity
    pcr optimize-status campaign.toml [--set k=v ...]
                                            table of the evaluations, Pareto set, best values

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
    from .analysis.runs import load_track_run, save_score
    out_dir = Path(config.require(config.section(cfg, 'run').out_dir, 'run.out_dir'))
    path = out_dir / 'score.json'
    if path.exists():
        print(f"[score] {path} exists: reused", flush=True)
        return json.loads(path.read_text())
    sc = config.section(cfg, 'score')
    scores = config.json_clean(score_run(load_track_run(out_dir, beam=sc.beam), sc))      # plain JSON: NaN -> null
    save_score(scores, path)
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
        cfg = _load(args.config, list(args.set) + [f'run.out_dir={config.toml_value(str(out))}'])
        scores = evaluate(cfg)
    except (Exception, SystemExit) as exc:         # the evaluation failed (a library's sys.exit too): say why, exit 1
        traceback.print_exc()
        why = (f"{type(exc).__name__}: {exc}" if isinstance(exc, Exception)
               else f"a library called sys.exit({exc.code}): its message is in stderr.log")
        write_result(out, 'failed', error=why, wall_s=time.time() - t0, config=cfg)
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


OPTIMIZE_COMMANDS = {'optimize': 'run or continue a MOBO campaign',
                     'optimize-verify': 're-evaluate campaign evaluations at the verify fidelity',
                     'optimize-status': 'table of a campaign: evaluations, Pareto set, best values'}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in OPTIMIZE_COMMANDS:
        # the campaign driver parses its own arguments (`pcr optimize --help`); it runs in a small environment of its
        # own, so nothing of the physics stack is imported on this path
        from .mobo.driver import main as driver_main
        return driver_main(argv)
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
    for name, text in OPTIMIZE_COMMANDS.items():          # listed for `pcr --help`; main hands them to the driver
        sub.add_parser(name, help=f"{text} (pcr {name} --help)")
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
