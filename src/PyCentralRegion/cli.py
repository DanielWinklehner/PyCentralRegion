"""`pcr`, the command line of the pipeline: one subcommand per stage, each reading a TOML run file.

    pcr show     run.toml [--set k=v ...]   the resolved configuration, as a stage sees it
    pcr build    run.toml [--set ...]       3D electrode model -> paths.model3d              (geometry.build)
    pcr fields   run.toml [--set ...]       3D BEM field cache -> paths.fields               (simulate3d.fields_stage)
    pcr track    run.toml [--set ...]       bunch tracking -> run.out_dir                    (simulate3d.track_stage)
    pcr score    run.toml [--set ...]       metrics of run.out_dir -> run.out_dir/score.json (analysis)
    pcr evaluate --config run.toml [--set ...] --out DIR
                                            build -> fields -> track -> score into DIR; DIR/result.json
                                            (with [post_rule] enabled the geometry gets its posts by rule first)
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
import copy
import gc
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

from . import config


def _load(path, overrides):
    return config.load(path, overrides=overrides or [])


def _free_gpu():
    """Return cupy's cached GPU blocks between stages (the BEM solve leaves gigabytes in the pool)."""
    gc.collect()                                 # what a finished stage holds in reference cycles (a tracker, a solver)
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
    """build (when paths.model3d is missing) -> fields (when paths.fields is missing) -> track -> score. With [post_rule]
    enabled the model and fields come from ``post_rule_stage`` and the scores tell of it under 'post_rule'."""
    from .simulate3d import fields_stage, track_stage
    pc = config.section(cfg, 'paths')
    record = None
    if config.section(cfg, 'post_rule').enabled:
        record = post_rule_stage(cfg)
    else:
        if not (pc.model3d and Path(pc.model3d).exists()):
            build_stage(cfg)
        if not (pc.fields and Path(pc.fields).exists()):
            fields_stage(cfg)
            _free_gpu()
    track_stage(cfg)
    _free_gpu()
    scores = score_stage(cfg)
    if record is None:
        return scores
    return {**scores, 'post_rule': {'n_posts': len(applied_posts(record)), 'fallback': record.get('fallback', ''),
                                    'bare_phase_deg': record['bare_phase_deg'], 'n_survivors_bare': record['n_survivors'],
                                    'bare': record['bare_scores']}}


# ---------------------------------------------------------------------------------------------------------------------
# posts in the loop
BARE_DIR, POSTS_RECORD = 'bare', 'posts.json'
BARE_SCORES = ('I_final_az', 'I_2pct_best_az', 'z_p95_az', 'E_med_az', 'T_az')


def applied_posts(record):
    """The posts a geometry's record says it was built with: none when the build with them failed."""
    return [] if record.get('fallback') else record['posts']


def _transient(exc):
    """An error that tells nothing about the model with posts (memory, the GPU, the file system): the evaluation fails
    and the geometry's next evaluation tries that model again, instead of keeping the bare one for good."""
    return isinstance(exc, (MemoryError, OSError)) or any('OutOfMemory' in c.__name__ or 'CUDA' in c.__name__
                                                          for c in type(exc).__mro__)


def _move(source, target):
    """source -> target, over an existing target: a rename on one drive, else a copy (paths.fields may be on another)."""
    try:
        source.replace(target)
    except OSError:
        if not source.exists():
            raise
        shutil.copy2(source, target)
        source.unlink()


def post_rule_stage(cfg):
    """Make paths.model3d and paths.fields with the posts [post_rule] places, and set geometry.posts in cfg (in place:
    the run's config.json records them). Returns the geometry's record, <model's folder>/bare/posts.json.

    One geometry, two passes, whichever evaluation of it comes first. The bare pass (``_bare_pass``) tracks the electrodes
    without posts in <model's folder>/bare/ and writes the record; it never runs again for that geometry, so the posts
    depend on the geometry (and the settings of its first evaluation but its phase and level), not on an evaluation's
    phase or level. The record belongs to one model: a model with posts needs a folder of its own. The second pass
    builds the model with the posts, checks it with the tracker's metal test, and solves it; when any of that fails, it
    builds the model again at each geometry.occ_fuzzy_mm of [post_rule] fuzzy_retry_mm in turn (OCC's boolean
    tolerance: two faces that nearly coincide fail at one tolerance and not at another). The record keeps the
    tolerance that worked ('occ_fuzzy_mm', which the geometry's later runs set too) and every failed attempt
    ('failed_at'). A geometry that gets no posts, or whose model with them fails at every tolerance (recorded:
    'fallback', the first attempt's error), keeps its bare model and fields: it is a design without posts, not a
    failed one; an error of memory, the GPU or the file system fails the evaluation instead (``_transient``). Every
    step is reused when its output exists, so a stopped evaluation's geometry continues where it was. The bare field
    map goes once the final one exists."""
    from .geometry.build import output_files
    from .geometry.post_rule import TRACK_DATA
    from .simulate3d import column_obstacles, fields_stage, load_model3d
    pc, rule = config.section(cfg, 'paths'), config.section(cfg, 'post_rule')
    if config.section(cfg, 'geometry').posts:
        raise config.ConfigError("[post_rule] places the posts: geometry.posts must be empty")
    model, fields = Path(config.require(pc.model3d, 'paths.model3d')), Path(config.require(pc.fields, 'paths.fields'))
    bare = model.parent / BARE_DIR
    path = bare / POSTS_RECORD
    if not path.exists():
        if model.exists() or fields.exists():
            raise config.ConfigError(f"{model.parent}: a model or field map exists without the post record {path}: it "
                                     f"was not made by the post rule; point paths.model3d and paths.fields elsewhere")
        _bare_pass(cfg, rule, bare, path)
    record = json.loads(path.read_text())
    if record.get('model', model.name) != model.name:
        raise config.ConfigError(f"{path} is the post record of the model {record['model']}, not of {model.name}: give "
                                 f"each model with posts by rule a folder of its own")
    (bare / 'track' / TRACK_DATA).unlink(missing_ok=True)        # the bare run's tracks served their purpose
    bare_model, bare_fields = bare / model.name, bare / fields.name

    def use(posts):
        cfg.setdefault('geometry', {})['posts'] = [{k: p[k] for k in ('gap', 'side', 's_mm', 'length_mm')} for p in posts]

    def tolerance(mm):                       # this run's geometry.occ_fuzzy_mm: its config.json records it
        config.set_dotted(cfg, 'geometry.occ_fuzzy_mm', float(mm))

    def write():
        config.write_atomic(path, lambda p: p.write_text(json.dumps(record, indent=1)))

    own = float(config.section(cfg, 'geometry').occ_fuzzy_mm)
    use(applied_posts(record))
    if applied_posts(record) and record.get('occ_fuzzy_mm') is not None:
        tolerance(record['occ_fuzzy_mm'])    # the model with posts was built at a retry's tolerance
    if not fields.exists() and applied_posts(record):
        first = float(config.section(cfg, 'geometry').occ_fuzzy_mm)
        failed = []
        for k, mm in enumerate([first] + [x for x in map(float, rule.fuzzy_retry_mm) if x != first]):
            try:
                if k:                        # the failed attempt's model goes: a build reuses a model that exists
                    for f in output_files(model).values():
                        f.unlink(missing_ok=True)
                    tolerance(mm)
                    print(f"[post rule] building the model with {len(record['posts'])} posts again at "
                          f"geometry.occ_fuzzy_mm = {mm:g} mm", flush=True)
                if not model.exists():
                    for f in output_files(model).values():    # what a stopped build left: never overwritten
                        f.unlink(missing_ok=True)
                    build_stage(cfg)
                    if k:                    # the geometry's later runs, and a stopped run's redo, set it too
                        record.update(occ_fuzzy_mm=mm, failed_at=list(failed))
                        write()
                if config.section(cfg, 'track').obstacles == 'columns':     # the tracker refuses a model that is not
                    column_obstacles(load_model3d(model), config.section(cfg, 'fields'), verbose=False, odd='raise')
                fields_stage(cfg)
                break
            except (Exception, SystemExit) as exc:           # a library's sys.exit too, as in _cmd_evaluate
                if isinstance(exc, Exception) and _transient(exc):
                    raise
                failed.append({'occ_fuzzy_mm': mm, 'error': f"{type(exc).__name__}: {exc}"})
                print(f"[post rule] the model with {len(record['posts'])} posts failed at geometry.occ_fuzzy_mm = "
                      f"{mm:g} mm ({failed[-1]['error']})", flush=True)
                _free_gpu()
        else:
            record.pop('occ_fuzzy_mm', None)
            record.update(fallback=failed[0]['error'], failed_at=failed)
            write()
            if float(config.section(cfg, 'geometry').occ_fuzzy_mm) != own:
                tolerance(own)               # the bare model's
            use([])
            tried = ', '.join(f"{a['occ_fuzzy_mm']:g}" for a in failed)
            print(f"[post rule] the model with {len(record['posts'])} posts failed at every tolerance ({tried} mm; "
                  f"first: {record['fallback']}): this geometry stays without posts", flush=True)
        _free_gpu()
    if not fields.exists():          # no posts: the bare outputs become the geometry's, over what a failed model with
        targets = output_files(model)            # posts left: the model first (a geometry folder without its model is
        moves = sorted(output_files(bare_model).items(), key=lambda kv: kv[0] != 'model3d')    # one a stopped
        for name, source in moves:                         # evaluation left: the driver sets its files aside), the
            if source.exists():                            # field map last
                _move(source, targets[name])
        _move(bare_fields, fields)
    bare_fields.unlink(missing_ok=True)
    return record


def _bare_pass(cfg, rule, folder, path):
    """Build, solve and track the electrodes without posts in ``folder``, with the phase scan and the level of ``rule``,
    score the run, place the posts from its tracks and write the record ``path``, last."""
    from .geometry.build import output_files
    from .geometry.post_rule import posts_from_run
    from .simulate3d import fields_stage, track_stage
    folder.mkdir(parents=True, exist_ok=True)
    model = folder / Path(config.section(cfg, 'paths').model3d).name
    run = copy.deepcopy(cfg)
    for key, value in (('paths.model3d', str(model)),
                       ('paths.fields', str(folder / Path(config.section(cfg, 'paths').fields).name)),
                       ('run.out_dir', str(folder / 'track')), ('magnet.scale', float(rule.scan_magnet_scale)),
                       ('track.phases_deg', [float(p) for p in rule.scan_phases_deg])):
        config.set_dotted(run, key, value)
    if not model.exists():
        for f in output_files(model).values():    # what a stopped build left: a build refuses to overwrite it
            f.unlink(missing_ok=True)
    build_stage(run)
    fields_stage(run)
    _free_gpu()
    summary = track_stage(run)
    _free_gpu()
    scores = score_stage(run)
    record = {**posts_from_run(run, rule), 'model': model.name, 'bare_phase_deg': summary['best_phase'],
              'bare_scores': {k: scores.get(k) for k in BARE_SCORES}}
    config.write_atomic(path, lambda p: p.write_text(json.dumps(record, indent=1, default=config.json_default)))
    print(f"[post rule] {len(record['posts'])} posts from the bare run at {record['bare_phase_deg']:+.1f} deg, "
          f"{record['n_survivors']} of {record['n_tracks']} tracks surviving", flush=True)


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
