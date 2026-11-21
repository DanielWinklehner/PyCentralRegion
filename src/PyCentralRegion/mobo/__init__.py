"""Multi-objective Bayesian optimisation (MOBO) of central-region designs with the full 3D model in the loop.

    space     the [mobo] section, the unit cube of the design variables, objectives, constraints, geometry keys
    store     the campaign store (evals.jsonl + one folder per evaluation) and the campaign lock
    evaluate  the evaluator contract and one evaluation as a subprocess of the physics environment
    driver    pcr optimize / optimize-verify / optimize-status
    toy       an analytic evaluator with the same contract, for tests and dry runs

The driver needs the standard library, numpy and optuna only (plus torch for the GP sampler); importing this
package imports none of them, and never the physics modules, so it runs in a small environment of its own.
"""
