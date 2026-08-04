"""
cavity_optimizer.py - RF Cavity Geometry Optimization (user-beam entry point)

Optimizes cavity segment geometry INDEPENDENTLY PER GAP (each gap's segment
angles and node radii are separate optimization parameters) together with RF
parameters (bunch phase, frequency, and optional r0/pr0) to steer a
USER-SUPPLIED initial beam (e.g. a spiral-inflector output) onto a good
accelerated orbit. Cavity base angles stay fixed.

Parameter vector layout:
    [gap0: a_0..a_{n-1}, r_0..r_{n-1}] [gap1: ...] ... [gapG-1: ...] [RF params]
with n = n_segments and G = number of gaps (len(design.rf_cavities)).

Part of: PyCentralRegion module. Dependencies: accelerated_orbit_finder, tracking.

Usage
-----
    from PyCentralRegion import (AcceleratedOrbitFinder, CavityGeometryOptimizer,
                                 make_single_particle_beam)

    beam = make_single_particle_beam(design.species, r=0.03, vr=0.0, v_total=v0)
    finder = AcceleratedOrbitFinder(design, target_energy_mev=5.0)
    geo = CavityGeometryOptimizer(finder, n_segments=2, max_angle_variable=10.0,
                                  max_r_variable=0.20, r_min_cavity=0.005)
    result = geo.optimize(beam, rf_optimize_params=['bunch_phase', 'rf_freq'])
"""

import os
import time
import csv
import numpy as np
from typing import List, Optional, Dict, Tuple
from scipy.optimize import differential_evolution, minimize

from .accelerated_orbit_finder import (AcceleratedOrbitFinder, OptimizedOrbit,
                                       DEFAULT_LS_WEIGHTS, DEFAULT_SKIP_TURNS)


# ============================================================================
# Multiprocessing workers (module-level so they pickle under Windows 'spawn').
#
# Each pool worker builds its OWN full system (field load included) exactly
# once via the user-supplied builder, then evaluates many tasks against it.
# The builder must be a module-level function returning a configured
# CavityGeometryOptimizer; it is pickled by reference, so it must be
# importable from the worker process (define it at the top level of your
# script and keep the `if __name__ == "__main__":` guard).
# ============================================================================
_WORKER_GEO = None


def _pool_init(builder, builder_args, checkpoint_base):
    global _WORKER_GEO
    geo = builder(*builder_args)
    geo.verbose = False
    geo.orbit_finder.verbose = False
    geo.orbit_finder.design.verbose = False   # silence per-eval RF-setting prints
    geo.orbit_finder.checkpoint_file = None
    if checkpoint_base:
        geo.checkpoint_file = f"{checkpoint_base}.worker-{os.getpid()}.csv"
        geo._checkpoint_inited = False
        geo._init_checkpoint_file()
    else:
        geo.checkpoint_file = None
    _WORKER_GEO = geo


def _pool_track_once(task):
    """Stage-A grid point: (phase, freq) with the given frozen geometry."""
    try:
        (phase_deg, freq_hz, x_vec, p_vec, spt, max_turns, r0_mode,
         angles_per_gap, radii_per_gap) = task
        from PyPATools.particles import ParticleDistribution
        geo = _WORKER_GEO
        of = geo.orbit_finder
        of.steps_per_turn = int(spt)
        for g, cavity in enumerate(of.design.rf_cavities):
            cavity.update_geometry(segment_angles=list(angles_per_gap[g]),
                                   segment_radii=list(radii_per_gap[g]))
        # bem2d: stage-A geometry is identical for every task, so the
        # geometry-keyed attach solves once per worker and then no-ops.
        if geo._is_bem and not geo._attach_bem_for_current_geometry():
            return None
        beam = ParticleDistribution(species=of.design.species,
                                    x_vec=np.asarray(x_vec), p_vec=np.asarray(p_vec))
        res = of.track_once(beam, bunch_phase_deg=float(phase_deg),
                            rf_freq_mhz=float(freq_hz) / 1e6,
                            max_turns=int(max_turns), r0_mode=r0_mode)
        return float(res.final_energy_mev), float(phase_deg), float(freq_hz)
    except Exception:
        return None


def _pool_verify(task):
    """Stage-C pre-selection: evaluate ONE solution at final resolution.

    The winner must be chosen by the VERIFIED objective - a start can win at
    search resolution on a marginal synchrotron-capture basin that simply does
    not exist at full resolution (observed: search ||r||^2 best-of-12 gave
    0.028 MeV / 1 turn at 500 spt).
    """
    try:
        (x, x_vec, p_vec, spt, max_turns, ls_weights, rf_optimize_params,
         r0_mode, skip_turns, f0_hz) = task
        from PyPATools.particles import ParticleDistribution
        geo = _WORKER_GEO
        of = geo.orbit_finder
        of.steps_per_turn = int(spt)
        beam = ParticleDistribution(species=of.design.species,
                                    x_vec=np.asarray(x_vec), p_vec=np.asarray(p_vec))
        dt = of._estimate_timestep(float(f0_hz))
        resid = geo.residuals_with_geometry(
            np.asarray(x, dtype=float), beam, dt, int(max_turns),
            dict(ls_weights), list(rf_optimize_params), r0_mode,
            skip_turns=int(skip_turns))
        return {'obj': float(np.sum(resid ** 2)),
                'energy': float(of.last_energy_mev),
                'turns': int(of.last_n_turns)}
    except Exception as e:
        return {'error': repr(e)}


def _pool_dfols(task):
    """Stage-B multi-start: one full DFO-LS run; returns a compact result."""
    try:
        import dfols
        from PyPATools.particles import ParticleDistribution
        (seed_x0, lower, upper, x_vec, p_vec, spt, max_turns, ls_weights,
         rf_optimize_params, r0_mode, skip_turns, maxfun, f0_hz) = task
        geo = _WORKER_GEO
        of = geo.orbit_finder
        of.steps_per_turn = int(spt)
        of.iteration = 0
        of.best_cost = np.inf
        geo.best_cost = np.inf
        beam = ParticleDistribution(species=of.design.species,
                                    x_vec=np.asarray(x_vec), p_vec=np.asarray(p_vec))
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        margin = 1e-6 * (upper - lower)
        x0 = np.clip(np.asarray(seed_x0, dtype=float), lower + margin, upper - margin)
        # dt from the EXPLICIT base frequency, not the worker's mutable cavity
        # state (stage-A tasks leave the last-evaluated frequency behind, and
        # task->worker scheduling varies run to run -> nondeterministic dt).
        dt = of._estimate_timestep(float(f0_hz))

        def objfun(x):
            return geo.residuals_with_geometry(x, beam, dt, int(max_turns),
                                               dict(ls_weights),
                                               list(rf_optimize_params), r0_mode,
                                               skip_turns=int(skip_turns))

        soln = dfols.solve(objfun, x0, bounds=(lower, upper), maxfun=int(maxfun),
                           objfun_has_noise=True, scaling_within_bounds=True,
                           do_logging=False)
        return {'x': np.asarray(soln.x, dtype=float).tolist(),
                'obj': float(soln.obj), 'nf': int(soln.nf), 'msg': str(soln.msg)}
    except Exception as e:
        return {'error': repr(e)}


class CavityGeometryOptimizer:
    """Optimize RF cavity geometry + RF parameters for a supplied initial beam.

    Parameter vector layout:
        [per-gap geometry blocks] [opening_delta (if enabled)] [RF params]

    ``optimize_opening_angle`` adds ONE shared parameter: a delta [deg] on the
    dee opening angle applied to ALL dees (requires ``dee_system``, the
    descriptor returned by ``create_dee_system``). In parallel mode, the worker
    builder must construct the optimizer with the SAME dee_system and flag.

    With a ``gap_model='bem2d'`` finder, the solved BEM gap field is
    re-attached before every tracking run whose geometry changed (skipped for
    RF-only moves - the engine re-syncs omega/phase without a re-solve). This
    is EXPENSIVE (one electrode build + Laplace solve + field gridding per
    geometry change); use coarse ``bem_build_kwargs``/``bem_field_kwargs`` for
    the search and verify winners at full resolution (example 07). Unbuildable
    candidates (mesh/solve failure, ``max_r_inner`` guard) get a graded
    penalty, not a crash.

    ``pinch_target_r_m`` (opt-in) adds a pinch-radius tie-breaker to both
    objective forms (thin-gap and bem2d finders alike): wedges whose metal
    pinches below ``pinch_metal_width_m`` ABOVE the target radius are
    penalized in proportion to the excess. Among near-degenerate
    beam-dynamics optima this steers the search toward geometries whose
    segments run parallel enough near the center that the electrodes can
    follow the beam corridor all the way down (tips buried ~ a gap width
    below their first crossing) instead of ending in a shallow tip just
    under the beam. A scalar applies to every wedge; a sequence of length
    n_gaps gives per-wedge targets (order dee0, ground0, dee1, ground1, ...
    with dees sorted by base angle) - set each to (turn-1 radius at that
    wedge's azimuth - tip clearance) from a reference trajectory so the
    pressure lands only on wedges whose pinch actually blocks the beam
    corridor, not on wedges that are corridor-limited anyway. In parallel
    mode the worker builder must construct the optimizer with the SAME
    pinch settings (as with dee_system and optimize_opening_angle).
    """

    def __init__(self,
                 orbit_finder: AcceleratedOrbitFinder,
                 n_segments: int,
                 max_angle_variable: float,
                 max_r_variable: float,
                 r_min_cavity: float,
                 verbose: bool = True,
                 checkpoint_file: Optional[str] = None,
                 dee_system=None,
                 optimize_opening_angle: bool = False,
                 opening_delta_max: float = 10.0,
                 rotatable_segments: bool = False,
                 rotation_max: float = 15.0,
                 bem_build_kwargs: Optional[Dict] = None,
                 bem_solve_kwargs: Optional[Dict] = None,
                 bem_field_kwargs: Optional[Dict] = None,
                 pinch_target_r_m: Optional[float] = None,
                 pinch_metal_width_m: Optional[float] = None,
                 pinch_weight: float = 50.0):
        if getattr(orbit_finder, 'gap_model', 'thin') not in ('thin', 'bem2d'):
            raise ValueError(f"unsupported gap_model "
                             f"'{getattr(orbit_finder, 'gap_model', 'thin')}'")
        self.orbit_finder = orbit_finder
        self.n_segments = n_segments
        self.n_gaps = len(orbit_finder.design.rf_cavities)
        self.max_angle = max_angle_variable
        self.max_r = max_r_variable
        self.r_min = r_min_cavity
        self.verbose = verbose
        self.checkpoint_file = checkpoint_file

        self.dee_system = dee_system
        self.optimize_opening_angle = bool(optimize_opening_angle)
        self.opening_delta_max = opening_delta_max
        if self.optimize_opening_angle and dee_system is None:
            raise ValueError("optimize_opening_angle=True requires dee_system "
                             "(from create_dee_system)")

        # Rotatable segments: each variable segment gets a per-segment rotation
        # about its midpoint (need not point at the origin) - decouples the
        # crossing azimuth (RF phase) from the kick direction (segment normal).
        self.rotatable_segments = bool(rotatable_segments)
        self.rotation_max = rotation_max

        # Arc clearance between azimuthally adjacent gap CENTERLINES (and
        # minimum radial-band ordering within a gap). Violations are penalized
        # smoothly, like the radii-monotonicity projection. The requirement is
        # gap-width aware: the gaps are channels of finite width, so adjacent
        # centerlines must clear (gap_width_i + gap_width_j)/2 plus
        # min_metal_width_m so a metal dummy-dee sliver always fits between the
        # channels (min_metal_width_m matches gap_fields.build_gap_electrodes;
        # min_clearance_m is an absolute floor for very narrow gaps). The gap
        # adjacency ORDER, nominal spacings, and pair half-width sums are
        # captured here, at construction - candidates must not redefine their
        # own baseline.
        self.min_clearance_m = 0.005
        self.min_metal_width_m = 0.004
        cavs = orbit_finder.design.rf_cavities
        self._clearance_order = sorted(range(len(cavs)),
                                       key=lambda k: cavs[k].base_angle % 360.0)
        _sorted_angles = [cavs[k].base_angle % 360.0 for k in self._clearance_order]
        self._nominal_dphi = [np.deg2rad((_sorted_angles[(i + 1) % len(_sorted_angles)]
                                          - _sorted_angles[i]) % 360.0)
                              for i in range(len(_sorted_angles))]
        # Per-gap width parameters (nominal, taper) in clearance order; the
        # local width at a sample radius comes from _frozen_width_at. Widths
        # and taper are never touched by update_geometry, so freezing them is
        # exact.
        self._gap_width_params = [
            (cavs[k].gap_width, getattr(cavs[k], 'gap_width_inner', None),
             getattr(cavs[k], 'gap_taper_radius', None), cavs[k].r_min)
            for k in self._clearance_order]

        # Opt-in pinch-radius tie-breaker (see class docstring). The pinch
        # radius is where the metal wedge between two neighboring gap chains
        # thins below the electrode build's min_metal_width - i.e. where
        # build_gap_electrodes must truncate the wedge tip. A pinch ABOVE
        # pinch_target_r_m means the tip cannot follow the beam corridor
        # down to the target (the first crossing then sits close over a
        # sharp tip: non-perpendicular gap field AND a thin fin that
        # degrades the BEM solve conditioning). Scalar target: same for all
        # wedges. Sequence (length n_gaps): per-wedge targets in the order
        # dee0, ground0, dee1, ground1, ... (dees sorted by base angle, as
        # in build_gap_electrodes) - set each to (turn-1 radius at that
        # wedge's azimuth - tip clearance) so the pressure lands only on
        # wedges whose pinch actually blocks the corridor. The scan width
        # defaults to the BEM build's min_metal_width, else the clearance
        # guard's.
        if pinch_target_r_m is None:
            self.pinch_target_r_m = None
        else:
            tgt = np.atleast_1d(np.asarray(pinch_target_r_m, dtype=float))
            if len(tgt) == 1:
                tgt = np.full(self.n_gaps, tgt[0])
            elif len(tgt) != self.n_gaps:
                raise ValueError(
                    f"pinch_target_r_m must be a scalar or a sequence of "
                    f"length n_gaps={self.n_gaps} (one per wedge: dee0, "
                    f"ground0, dee1, ground1, ... in dee base-angle order), "
                    f"got length {len(tgt)}")
            self.pinch_target_r_m = tgt
        if pinch_metal_width_m is None:
            pinch_metal_width_m = (bem_build_kwargs or {}).get(
                'min_metal_width', self.min_metal_width_m)
        self.pinch_metal_width_m = float(pinch_metal_width_m)
        self.pinch_weight = float(pinch_weight)

        # BEM-in-the-loop state (gap_model='bem2d' finders only).
        self.bem_build_kwargs = bem_build_kwargs
        self.bem_solve_kwargs = bem_solve_kwargs
        self.bem_field_kwargs = bem_field_kwargs
        self._bem_geom_key = None
        self._bem_last_error = None
        self._n_resid = None

        self.best_cost = np.inf
        self.best_geometry = None
        self._checkpoint_inited = False

    # ------------------------------------------------------------- checkpoint
    def _init_checkpoint_file(self):
        if not self.checkpoint_file or self._checkpoint_inited:
            return
        with open(self.checkpoint_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['iteration']
            for g in range(self.n_gaps):
                header += [f'gap{g}_seg{i}_angle_deg' for i in range(self.n_segments)]
                header += [f'gap{g}_seg{i}_radius_mm' for i in range(self.n_segments)]
                if self.rotatable_segments:
                    header += [f'gap{g}_seg{i}_rotation_deg' for i in range(self.n_segments)]
            if self.optimize_opening_angle:
                header += ['opening_angle_deg']
            header += ['bunch_phase_deg', 'rf_freq_mhz', 'r0', 'vr0',
                       'coll_azimuth_deg', 'coll_aperture_mm',
                       'final_energy_mev', 'n_turns', 'cost', 'success', 'timestamp']
            writer.writerow(header)
        self._checkpoint_inited = True

    def _write_checkpoint(self, angles_per_gap, radii_per_gap, rf_vals, cost,
                          energy, n_turns, success, opening_delta=None,
                          rotations_per_gap=None):
        if not self.checkpoint_file:
            return
        with open(self.checkpoint_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [self.orbit_finder.iteration]
            for g in range(self.n_gaps):
                row += list(angles_per_gap[g])
                row += [r * 1000 for r in radii_per_gap[g]]
                if self.rotatable_segments:
                    row += list(rotations_per_gap[g]) if rotations_per_gap else \
                           [0.0] * self.n_segments
            if self.optimize_opening_angle:
                base = self.dee_system.opening_angle if self.dee_system else 0.0
                row += [base + (opening_delta or 0.0)]
            row += [rf_vals.get('bunch_phase', 0.0), rf_vals.get('rf_freq', 0.0) / 1e6,
                    rf_vals.get('r0', 0.0), rf_vals.get('vr0', 0.0),
                    rf_vals.get('coll_azimuth', 0.0),
                    rf_vals.get('coll_aperture', 0.0),
                    energy, n_turns, cost, success, time.time()]
            writer.writerow(row)

    # --------------------------------------------------------------- geometry
    @property
    def _blk(self) -> int:
        """Per-gap geometry block size: [angles, radii(, rotations)]."""
        return (3 if self.rotatable_segments else 2) * self.n_segments

    @property
    def _n_geo(self) -> int:
        return self.n_gaps * self._blk

    @property
    def _rf_offset(self) -> int:
        """Index where RF params start (after geometry [+ opening delta])."""
        return self._n_geo + (1 if self.optimize_opening_angle else 0)

    def _unpack_params(self, params) -> Tuple[List[List[float]], List[List[float]],
                                              Optional[List[List[float]]],
                                              Optional[float], np.ndarray]:
        """Split the parameter vector into its blocks.

        Returns (angles_per_gap, radii_per_gap, rotations_per_gap,
        opening_delta, rf_params); rotations_per_gap is None unless
        rotatable_segments is enabled, opening_delta is None unless
        optimize_opening_angle is enabled.
        """
        n = self.n_segments
        blk = self._blk
        angles_per_gap, radii_per_gap = [], []
        rotations_per_gap = [] if self.rotatable_segments else None
        for g in range(self.n_gaps):
            base = g * blk
            angles_per_gap.append(list(params[base:base + n]))
            radii_per_gap.append(list(params[base + n:base + 2 * n]))
            if self.rotatable_segments:
                rotations_per_gap.append(list(params[base + 2 * n:base + 3 * n]))
        opening_delta = (float(params[self._n_geo])
                         if self.optimize_opening_angle else None)
        rf_params = np.asarray(params[self._rf_offset:], dtype=float)
        return angles_per_gap, radii_per_gap, rotations_per_gap, opening_delta, rf_params

    def _rotations_for_gap(self, rotations_per_gap, g):
        return rotations_per_gap[g] if rotations_per_gap is not None else None

    def _gap_base_angles(self, opening_delta: Optional[float]) -> List[Optional[float]]:
        """Per-gap base angles for the given opening delta (None -> unchanged)."""
        if opening_delta is None or self.dee_system is None:
            return [None] * self.n_gaps
        return self.dee_system.gap_angles(self.dee_system.opening_angle + opening_delta)

    def _geometry_violation(self, radii_per_gap) -> float:
        """Total monotonicity violation [m] over all gaps (0 if all valid).

        Per gap: r_min < r0 < ... < r_max. Graded (not a flat penalty) so the
        optimizer sees a gradient out of the invalid region instead of a plateau
        that collapses Nelder-Mead.
        """
        violation = 0.0
        for g, radii in enumerate(radii_per_gap):
            r_max = self.orbit_finder.design.rf_cavities[g].r_max
            all_radii = [self.r_min] + list(radii) + [r_max]
            for i in range(len(all_radii) - 1):
                gap = all_radii[i + 1] - all_radii[i]
                if gap <= 0.0:
                    violation += -gap + 1e-6
        return violation

    def _project_radii(self, radii_per_gap, eps: float = 0.002):
        """Project per-gap node radii onto the nearest feasible monotone config.

        Sequential forward clip with ``eps`` spacing inside (r_min, r_max).
        Returns (projected_radii_per_gap, total_projection_distance_m). The
        distance is 0 for already-valid configurations; least-squares paths use
        it as a smooth constraint residual instead of a cost cliff.
        """
        proj, viol = [], 0.0
        for g, radii in enumerate(radii_per_gap):
            r_max = self.orbit_finder.design.rf_cavities[g].r_max
            lo = self.r_min + eps
            out = []
            for k, r in enumerate(radii):
                hi = r_max - eps * (len(radii) - k)
                r_new = min(max(r, lo), hi)
                viol += abs(r_new - r)
                out.append(r_new)
                lo = r_new + eps
            proj.append(out)
        return proj, viol

    @staticmethod
    def _frozen_width_at(params, r: float) -> float:
        """Local gap width [m] at radius r from frozen (nominal, taper) params."""
        w_nom, w_inner, taper_r, r_min = params
        if w_inner is None:
            return w_nom
        frac = min(max((r - r_min) / (taper_r - r_min), 0.0), 1.0)
        return w_inner + frac * (w_nom - w_inner)

    def _clearance_violation(self, n_samples: int = 17) -> float:
        """Total overlap violation [m] of the ACTUAL (possibly rotated) gap lines.

        (a) intra-gap: consecutive segments' radial bands must stay ordered
            (rotation moves endpoint radii; overlapping bands cause spurious
            double kicks);
        (b) inter-gap: the arc clearance between azimuthally adjacent gap
            CENTERLINES, sampled at n_samples radii, must exceed the pair's
            channel half-widths plus min_metal_width_m (worst case: the two
            gaps of the same dee).

        Smooth near the feasible boundary; call AFTER the candidate geometry
        has been applied to the cavities.
        """
        cavs = self.orbit_finder.design.rf_cavities
        viol = 0.0

        # (a) intra-gap radial-band ordering from actual endpoints
        for cav in cavs:
            segs = cav.segments
            for i in range(len(segs) - 1):
                r_hi = max(np.hypot(*segs[i]['p1'][:2]), np.hypot(*segs[i]['p2'][:2]))
                r_lo = min(np.hypot(*segs[i + 1]['p1'][:2]), np.hypot(*segs[i + 1]['p2'][:2]))
                viol += max(0.0, r_hi - r_lo)

        # (b) inter-gap azimuthal arc clearance
        def azimuth_at_radius(segments, r):
            for seg in segments:
                p1 = seg['p1'][:2]
                d = seg['p2'][:2] - p1
                a = d @ d
                if a < 1e-18:
                    continue
                b = 2.0 * (p1 @ d)
                c = (p1 @ p1) - r * r
                disc = b * b - 4.0 * a * c
                if disc < 0.0:
                    continue
                sq = np.sqrt(disc)
                for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)):
                    if 0.0 <= t <= 1.0:
                        p = p1 + t * d
                        return float(np.arctan2(p[1], p[0]))
            return None

        # Required clearance per pair: the full gap-width-aware value wherever
        # the AS-CONSTRUCTED nominal arc at that radius could fit it, so a
        # metal sliver of min_metal_width_m always fits between the two gap
        # channels. Below that radius the nominal (radial) lines legitimately
        # converge and no candidate can satisfy the full requirement either,
        # so fall back to 50% of the nominal arc - a pure crossing guard that
        # never flags the constructed geometry. Requirements depend only on
        # the sample radius and the frozen baseline (__init__), never on the
        # candidate, so the penalty stays smooth in the parameters. Sampling
        # is geometric: the clearance bites at small radii, which a coarse
        # linspace over [r_min, r_max] never visits.
        r_lo = 1.05 * max(c.r_min for c in cavs)
        r_hi = 0.98 * min(c.r_max for c in cavs)
        radii = np.geomspace(r_lo, r_hi, n_samples)
        order = self._clearance_order
        for j, r in enumerate(radii):
            phis = [azimuth_at_radius(cavs[k].segments, r) for k in order]
            for a_idx in range(len(order)):
                p1 = phis[a_idx]
                p2 = phis[(a_idx + 1) % len(order)]
                if p1 is None or p2 is None:
                    continue
                halfw = 0.5 * (
                    self._frozen_width_at(self._gap_width_params[a_idx], r)
                    + self._frozen_width_at(
                        self._gap_width_params[(a_idx + 1) % len(order)], r))
                full_req = max(self.min_clearance_m,
                               halfw + self.min_metal_width_m)
                nominal_arc = self._nominal_dphi[a_idx] * r
                required = full_req if nominal_arc >= full_req else 0.5 * nominal_arc
                dphi = (p2 - p1) % (2.0 * np.pi)
                arc = dphi * r
                if dphi <= np.pi:
                    viol += max(0.0, required - arc)
                else:
                    # neighbor appears on the wrong side: lines crossed
                    viol += required + (dphi - np.pi) * r
        return viol

    def _pinch_excess(self) -> float:
        """Total pinch-radius excess [m] above ``pinch_target_r_m``.

        For every wedge of the CURRENT geometry (dee wedges: the metal
        between one dee's entry/exit chains; ground wedges: between
        azimuthally adjacent dees), find the outermost radius where the
        wedge's arc width falls below ``pinch_metal_width_m`` - the same
        offset-chain scan build_gap_electrodes uses for its truncation
        floor, with the sub-sample crossing interpolated so the term is
        smooth in the segment parameters. Returns the summed excess above
        the target. Call AFTER the candidate geometry has been applied.
        """
        if self.pinch_target_r_m is None:
            return 0.0
        from .gap_fields import offset_gap_boundary, _ccw_width
        cavs = self.orbit_finder.design.rf_cavities
        n_dees = len(cavs) // 2
        order = sorted(range(n_dees),
                       key=lambda k: cavs[2 * k].base_angle % 360.0)
        excess = 0.0
        w_idx = 0
        for pos, j in enumerate(order):
            entry, exit_ = cavs[2 * j], cavs[2 * j + 1]
            nxt_entry = cavs[2 * order[(pos + 1) % n_dees]]
            for lo, hi in ((offset_gap_boundary(entry, +1.0),
                            offset_gap_boundary(exit_, -1.0)),    # dee wedge
                           (offset_gap_boundary(exit_, +1.0),
                            offset_gap_boundary(nxt_entry, -1.0))):  # ground
                target = float(self.pinch_target_r_m[w_idx])
                w_idx += 1
                r0 = max(np.hypot(*lo[0]), np.hypot(*hi[0]), 1e-4)
                r_top = min(np.hypot(*lo[-1]), np.hypot(*hi[-1]), 0.15)
                ladder = np.arange(r0, r_top, 5e-4)
                if not len(ladder):
                    continue
                width = _ccw_width(lo, hi, ladder)
                bad = np.where(width < self.pinch_metal_width_m)[0]
                if not len(bad):
                    continue
                k = int(bad[-1])
                if k + 1 < len(width) and width[k + 1] > width[k]:
                    t = ((self.pinch_metal_width_m - width[k])
                         / (width[k + 1] - width[k]))
                    pinch = float(ladder[k] + np.clip(t, 0.0, 1.0) * 5e-4)
                else:
                    pinch = float(ladder[k] + 5e-4)
                excess += max(0.0, pinch - target)
        return excess

    # ------------------------------------------------------------------- BEM
    @property
    def _is_bem(self) -> bool:
        return getattr(self.orbit_finder, 'gap_model', 'thin') == 'bem2d'

    def _attach_bem_for_current_geometry(self) -> bool:
        """(Re-)solve the BEM gap field for the cavities' CURRENT geometry.

        Skips the solve when the geometry is unchanged since the last attach
        (RF-only moves: the engine re-syncs omega/phase without a re-solve).
        Returns False if the electrode build or solve failed (unbuildable
        candidate); the error is kept on ``self._bem_last_error``.
        """
        of = self.orbit_finder
        key = tuple((tuple(c.segment_angles), tuple(c.segment_radii),
                     tuple(c.segment_rotations), float(c.base_angle))
                    for c in of.design.rf_cavities)
        if key == self._bem_geom_key and of.bem_solution is not None:
            return True
        try:
            of.attach_bem_field(build_kwargs=self.bem_build_kwargs,
                                solve_kwargs=self.bem_solve_kwargs,
                                field_kwargs=self.bem_field_kwargs)
        except Exception as e:
            self._bem_geom_key = None
            self._bem_last_error = repr(e)
            return False
        self._bem_geom_key = key
        return True

    def residuals_with_geometry(self, params, initial_beam, dt, max_turns,
                                ls_weights, rf_param_names, r0_mode,
                                skip_turns: int = DEFAULT_SKIP_TURNS,
                                w_violation: float = 100.0):
        """Residual vector (DFO-LS objective): per-gap geometry + RF params.

        Non-monotone radii are PROJECTED to the nearest feasible configuration
        (tracking still happens) and the projection distance is appended as one
        extra residual, keeping the landscape smooth for the model-based solver.
        ``skip_turns`` exempts the first n turns from centering/smoothness.
        """
        (angles_per_gap, radii_per_gap, rotations_per_gap,
         opening_delta, rf_params) = self._unpack_params(params)
        rf_vals = self.orbit_finder._unpack(rf_params, rf_param_names)
        radii_proj, violation = self._project_radii(radii_per_gap)
        base_angles = self._gap_base_angles(opening_delta)

        for g, cavity in enumerate(self.orbit_finder.design.rf_cavities):
            cavity.update_geometry(segment_angles=angles_per_gap[g],
                                   segment_radii=radii_proj[g],
                                   base_angle=base_angles[g],
                                   segment_rotations=self._rotations_for_gap(
                                       rotations_per_gap, g))
        violation += self._clearance_violation()
        pinch_resid = (self.pinch_weight * self._pinch_excess()
                       if self.pinch_target_r_m is not None else None)

        if self._is_bem and not self._attach_bem_for_current_geometry():
            # Unbuildable candidate (electrode build/solve failure or the
            # max_r_inner guard): graded fallback, worse than any tracked
            # geometry, keeping the clearance (and pinch) terms for a slope
            # back toward feasibility. Sized from the last successful
            # evaluation.
            if self._n_resid is None:
                raise RuntimeError(
                    "BEM attach failed on the FIRST evaluation (cannot size "
                    f"the penalty residual): {self._bem_last_error}")
            self.orbit_finder.iteration += 1
            resid = np.full(self._n_resid, 10.0)
            if pinch_resid is not None:
                resid[-2] = max(w_violation * violation, 10.0)
                resid[-1] = max(pinch_resid, 10.0)
            else:
                resid[-1] = max(w_violation * violation, 10.0)
            self._write_checkpoint(angles_per_gap, radii_proj, rf_vals,
                                   float(np.sum(resid ** 2)), 0.0, 0, False,
                                   opening_delta=opening_delta,
                                   rotations_per_gap=rotations_per_gap)
            return resid

        resid = self.orbit_finder.objective_residuals(
            rf_params, initial_beam, dt, max_turns, ls_weights, rf_param_names,
            r0_mode, skip_turns=skip_turns)
        resid = np.append(resid, w_violation * violation)
        if pinch_resid is not None:
            resid = np.append(resid, pinch_resid)
        self._n_resid = len(resid)

        cost = float(np.sum(resid ** 2))
        self._write_checkpoint(angles_per_gap, radii_proj, rf_vals, cost,
                               self.orbit_finder.last_energy_mev,
                               self.orbit_finder.last_n_turns,
                               self.orbit_finder.last_n_turns > 0,
                               opening_delta=opening_delta,
                               rotations_per_gap=rotations_per_gap)
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_geometry = {
                'segment_angles_per_gap': [list(a) for a in angles_per_gap],
                'segment_radii_per_gap': [list(r) for r in radii_proj],
                'opening_delta': opening_delta,
                'rf_params': rf_params.copy(),
            }
            if self.verbose:
                pinch_note = ("" if pinch_resid is None else
                              f", pinch excess "
                              f"{pinch_resid / self.pinch_weight * 1000:.2f} mm")
                print(f"    eval {self.orbit_finder.iteration}: NEW BEST "
                      f"||r||^2={cost:.3e}, E={self.orbit_finder.last_energy_mev:.3f} MeV, "
                      f"turns={self.orbit_finder.last_n_turns}{pinch_note}")
        elif self.verbose and self.orbit_finder.iteration % 25 == 0:
            print(f"    eval {self.orbit_finder.iteration}: ||r||^2={cost:.3e} "
                  f"(best {self.best_cost:.3e})")
        return resid

    def objective_function_with_geometry(self, params, initial_beam, dt, max_turns,
                                         weights, rf_param_names, r0_mode):
        (angles_per_gap, radii_per_gap, rotations_per_gap,
         opening_delta, rf_params) = self._unpack_params(params)
        rf_vals = self.orbit_finder._unpack(rf_params, rf_param_names)

        violation = self._geometry_violation(radii_per_gap)
        if violation > 0.0:
            cost = 1e10 + 1e6 * violation
            if self.verbose:
                print(f"    Iter {self.orbit_finder.iteration + 1}: invalid geometry "
                      f"(violation={violation * 1000:.2f} mm)")
            # iteration is incremented inside the orbit finder; emulate for the row
            self.orbit_finder.iteration += 1
            self._write_checkpoint(angles_per_gap, radii_per_gap, rf_vals, cost, 0.0, 0, False,
                                   opening_delta=opening_delta,
                               rotations_per_gap=rotations_per_gap)
            return cost

        try:
            base_angles = self._gap_base_angles(opening_delta)
            for g, cavity in enumerate(self.orbit_finder.design.rf_cavities):
                cavity.update_geometry(segment_angles=angles_per_gap[g],
                                       segment_radii=radii_per_gap[g],
                                       base_angle=base_angles[g],
                                       segment_rotations=self._rotations_for_gap(
                                           rotations_per_gap, g))
        except Exception as e:
            if self.verbose:
                print(f"    Iter {self.orbit_finder.iteration + 1}: geometry update failed: {e}")
            self.orbit_finder.iteration += 1
            self._write_checkpoint(angles_per_gap, radii_per_gap, rf_vals, 1e10, 0.0, 0, False,
                                   opening_delta=opening_delta,
                               rotations_per_gap=rotations_per_gap)
            return 1e10

        # Clearance violation (rotated/tilted lines must not overlap): graded.
        clear_viol = self._clearance_violation()
        cost_penalty = 1e6 * clear_viol
        if self.pinch_target_r_m is not None:
            # pinch tie-breaker, squared to match the least-squares form
            cost_penalty += (self.pinch_weight * self._pinch_excess()) ** 2

        if self._is_bem and not self._attach_bem_for_current_geometry():
            if self.verbose:
                print(f"    Iter {self.orbit_finder.iteration + 1}: BEM attach "
                      f"failed: {self._bem_last_error}")
            self.orbit_finder.iteration += 1
            cost = 1e10 + cost_penalty
            self._write_checkpoint(angles_per_gap, radii_per_gap, rf_vals, cost,
                                   0.0, 0, False, opening_delta=opening_delta,
                                   rotations_per_gap=rotations_per_gap)
            return cost

        cost = self.orbit_finder.objective_function(
            rf_params, initial_beam, dt, max_turns, weights, rf_param_names, r0_mode)
        cost += cost_penalty

        # Success-path checkpoint with the REAL energy/turns from the orbit finder.
        self._write_checkpoint(angles_per_gap, radii_per_gap, rf_vals, cost,
                               self.orbit_finder.last_energy_mev,
                               self.orbit_finder.last_n_turns, True,
                               opening_delta=opening_delta,
                               rotations_per_gap=rotations_per_gap)

        if cost < self.best_cost:
            self.best_cost = cost
            self.best_geometry = {
                'segment_angles_per_gap': [list(a) for a in angles_per_gap],
                'segment_radii_per_gap': [list(r) for r in radii_per_gap],
                'opening_delta': opening_delta,
                'rf_params': rf_params.copy(),
            }
            if self.verbose:
                a_all = np.concatenate(angles_per_gap)
                print(f"    Iter {self.orbit_finder.iteration}: NEW BEST cost={cost:.3e}, "
                      f"E={self.orbit_finder.last_energy_mev:.3f} MeV  "
                      f"angle range=[{a_all.min():+.2f},{a_all.max():+.2f}] deg")
        return cost

    # --------------------------------------------------------------- optimize
    def _build_full_param_space(self, initial_beam, rf_optimize_params, rf_bounds, r0_mode):
        """Full parameter space: per-gap geometry blocks (base angles fixed) + RF."""
        of = self.orbit_finder
        r_spacing = (self.max_r - self.r_min) / (self.n_segments + 1)
        geo_bounds, geo_x0, geo_names = [], [], []
        for g in range(self.n_gaps):
            r_max_cav = of.design.rf_cavities[g].r_max
            for i in range(self.n_segments):
                geo_names.append(f'gap{g}_seg{i}_angle')
                geo_bounds.append((-self.max_angle, self.max_angle))
                geo_x0.append(0.0)
            for i in range(self.n_segments):
                geo_names.append(f'gap{g}_seg{i}_radius')
                geo_bounds.append((self.r_min + 0.01, min(self.max_r, r_max_cav - 0.01)))
                geo_x0.append(self.r_min + (i + 1) * r_spacing)
            if self.rotatable_segments:
                for i in range(self.n_segments):
                    geo_names.append(f'gap{g}_seg{i}_rotation')
                    geo_bounds.append((-self.rotation_max, self.rotation_max))
                    geo_x0.append(0.0)

        if self.optimize_opening_angle:
            geo_names.append('opening_delta')
            geo_bounds.append((-self.opening_delta_max, self.opening_delta_max))
            geo_x0.append(0.0)

        rf_bnds, rf_names, rf_x0 = of._build_param_space(
            initial_beam, rf_optimize_params, rf_bounds or {}, r0_mode)

        return geo_bounds + rf_bnds, geo_names + rf_names, geo_x0 + rf_x0

    def optimize(self,
                 initial_beam,
                 max_turns: int = 500,
                 rf_optimize_params: List[str] = ['bunch_phase', 'rf_freq'],
                 rf_bounds: Optional[Dict] = None,
                 method: str = 'differential_evolution',
                 maxiter: int = 100,
                 weights: Optional[Dict] = None,
                 r0_mode: str = 'offset') -> OptimizedOrbit:
        of = self.orbit_finder
        of._set_beam_meta(initial_beam)
        effective_multi = of.is_multiparticle and r0_mode != 'absolute'

        if self.verbose:
            print("\n" + "=" * 70)
            print("CAVITY GEOMETRY + RF PARAMETER OPTIMIZATION (per-gap geometry)")
            print("=" * 70)
            print(f"Target energy: {of.target_energy_mev} MeV, beam numpart={of.n_particles}")
            print(f"Gaps: {self.n_gaps}, variable segments/gap: {self.n_segments} "
                  f"-> {self._n_geo} geometry params"
                  + (" + shared opening angle" if self.optimize_opening_angle else "")
                  + f"; RF params: {rf_optimize_params}")
            print(f"r0_mode={r0_mode}, method={method}")

        if weights is None:
            weights = of._default_weights(effective_multi)
        if rf_bounds is None:
            rf_bounds = {}

        param_bounds, param_names, x0 = self._build_full_param_space(
            initial_beam, rf_optimize_params, rf_bounds, r0_mode)

        of.iteration = 0
        of.best_cost = np.inf
        self.best_cost = np.inf
        self.best_geometry = None
        self._init_checkpoint_file()

        dt = of._estimate_timestep(of._rf_base_frequency())
        args = (initial_beam, dt, max_turns, weights, rf_optimize_params, r0_mode)

        start = time.time()
        if method == 'differential_evolution':
            res = differential_evolution(self.objective_function_with_geometry, param_bounds,
                                         args=args, maxiter=maxiter, workers=1,
                                         updating='deferred', disp=False)
            optimal = res.x
            final_cost = res.fun
        elif method == 'nelder_mead':
            res = minimize(self.objective_function_with_geometry, x0, args=args,
                           method='Nelder-Mead', options={'maxiter': maxiter, 'disp': False})
            optimal = res.x
            final_cost = res.fun
        else:
            raise ValueError(f"Unknown method: {method}")
        elapsed = time.time() - start

        optimized = self._finalize(optimal, rf_optimize_params, initial_beam, max_turns,
                                   r0_mode, param_names, param_bounds, weights,
                                   method, elapsed, final_cost)

        if self.verbose:
            print(f"Final energy: {optimized.final_energy_mev:.3f} MeV, "
                  f"turns: {optimized.n_turns}")
        return optimized

    def _finalize(self, optimal, rf_optimize_params, initial_beam, max_turns, r0_mode,
                  param_names, param_bounds, weights, method, elapsed, final_cost,
                  extra_meta=None) -> OptimizedOrbit:
        """Apply the optimal geometry + RF and do the final tracking run.

        Uses the orbit finder's CURRENT steps_per_turn, so staged optimization can
        raise the resolution before calling this. Radii are projected to the
        nearest feasible (monotone) configuration as a guard.
        """
        of = self.orbit_finder
        (angles_per_gap, radii_per_gap, rotations_per_gap,
         opening_delta, rf_params) = self._unpack_params(optimal)
        radii_per_gap, _ = self._project_radii(radii_per_gap)
        rf_vals = of._unpack(rf_params, rf_optimize_params)
        base_angles = self._gap_base_angles(opening_delta)

        if self.verbose:
            print(f"\nOptimization complete in {elapsed:.1f}s, {of.iteration} evals, "
                  f"final cost {final_cost:.3e}")
            if opening_delta is not None:
                print(f"  dee opening angle: "
                      f"{self.dee_system.opening_angle + opening_delta:.3f} deg "
                      f"(delta {opening_delta:+.3f})")
            for g in range(self.n_gaps):
                rot_str = (f", rot={[f'{r:+.2f}' for r in rotations_per_gap[g]]} deg"
                           if rotations_per_gap is not None else "")
                print(f"  gap {g}: angles={[f'{a:+.2f}' for a in angles_per_gap[g]]} deg, "
                      f"radii={[f'{r*1000:.1f}' for r in radii_per_gap[g]]} mm{rot_str}")

        for g, cavity in enumerate(of.design.rf_cavities):
            cavity.update_geometry(segment_angles=angles_per_gap[g],
                                   segment_radii=radii_per_gap[g],
                                   base_angle=base_angles[g],
                                   segment_rotations=self._rotations_for_gap(
                                       rotations_per_gap, g))
        if self._is_bem and not self._attach_bem_for_current_geometry():
            raise RuntimeError(f"BEM attach failed for the FINAL geometry: "
                               f"{self._bem_last_error}")
        if 'bunch_phase' in rf_vals:
            of.design.set_bunch_phase(rf_vals['bunch_phase'])
        if 'rf_freq' in rf_vals:
            of.design.set_rf_frequency(rf_vals['rf_freq'])

        dt = of._estimate_timestep(of._rf_base_frequency())
        pd_final = of._prepare_beam(initial_beam, rf_vals.get('r0'), rf_vals.get('vr0'), r0_mode)
        of._set_beam_meta(pd_final)
        coll = of._make_collimator(rf_vals)
        of.engine.extra_terminators = [coll] if coll is not None else []
        try:
            result = of.track_with_rf(pd_final, dt, max_turns, save_full_beam=True)
        finally:
            of.engine.extra_terminators = []

        meta = {
            'optimization_time_s': elapsed,
            'total_iterations': of.iteration,
            'optimal_geometry': {
                'segment_angles_per_gap': [list(a) for a in angles_per_gap],
                'segment_radii_per_gap': [list(r) for r in radii_per_gap],
                'segment_rotations_per_gap': ([list(r) for r in rotations_per_gap]
                                              if rotations_per_gap is not None else None),
                'opening_angle_deg': (self.dee_system.opening_angle + opening_delta
                                      if opening_delta is not None else None),
                'n_segments': self.n_segments,
                'n_gaps': self.n_gaps,
            },
        }
        if coll is not None:
            meta['collimator'] = {
                'azimuth_deg': rf_vals.get('coll_azimuth'),
                'aperture_mm': rf_vals.get('coll_aperture'),
                'r_center_mm': (float(coll.r_center_m * 1e3)
                                if coll.r_center_m is not None else None),
                'n_collimated': len(coll.hits)}
            if self.verbose:
                print(f"  collimator: azimuth "
                      f"{rf_vals.get('coll_azimuth', 0.0):.1f} deg, "
                      f"aperture {rf_vals.get('coll_aperture', 0.0):.1f} mm "
                      f"(center r {meta['collimator']['r_center_mm']} mm), "
                      f"{len(coll.hits)} collimated")
        meta.update(extra_meta or {})
        return of._build_result(result, rf_vals, final_cost, param_names, param_bounds,
                                weights, method, elapsed, r0_mode, metadata_extra=meta)

    # ---------------------------------------------------------------- DFO-LS
    def optimize_dfols(self,
                       initial_beam,
                       max_turns: int = 8,
                       rf_optimize_params: List[str] = ['bunch_phase', 'rf_freq'],
                       rf_bounds: Optional[Dict] = None,
                       ls_weights: Optional[Dict] = None,
                       maxfun: Optional[int] = None,
                       seed_x0: Optional[np.ndarray] = None,
                       r0_mode: str = 'offset',
                       skip_turns: int = DEFAULT_SKIP_TURNS,
                       verify_max_turns: Optional[int] = None,
                       verify_steps_per_turn: Optional[int] = None) -> OptimizedOrbit:
        """Optimize with DFO-LS (model-based derivative-free least squares).

        Minimizes ||objective residuals||^2 (see ``residuals_with_geometry``) —
        the recommended optimizer for the high-dimensional per-gap geometry space
        (noise-robust, needs only n+1 evals to build its first model).

        Parameters
        ----------
        maxfun : int, optional
            Evaluation budget (default 30*(n_params+1)).
        seed_x0 : array, optional
            Starting point (e.g. from a stage-A RF scan). Defaults to straight
            segments + nominal radii + RF defaults.
        verify_max_turns / verify_steps_per_turn : int, optional
            If given, the FINAL tracking (and returned result) runs at this
            higher resolution while the search stays at the current settings.
        ls_weights : dict, optional
            Least-squares weights {'energy','center','smooth'}; defaults
            DEFAULT_LS_WEIGHTS (energy-dominant - see its comment for why).
        """
        import dfols

        of = self.orbit_finder
        of._set_beam_meta(initial_beam)
        if ls_weights is None:
            ls_weights = dict(DEFAULT_LS_WEIGHTS)

        param_bounds, param_names, x0_default = self._build_full_param_space(
            initial_beam, rf_optimize_params, rf_bounds, r0_mode)
        lower = np.array([b[0] for b in param_bounds], dtype=float)
        upper = np.array([b[1] for b in param_bounds], dtype=float)
        x0 = np.asarray(seed_x0 if seed_x0 is not None else x0_default, dtype=float)
        # dfols with scaling_within_bounds needs x0 strictly inside the box
        margin = 1e-6 * (upper - lower)
        x0 = np.clip(x0, lower + margin, upper - margin)

        n = len(x0)
        budget = maxfun if maxfun is not None else 30 * (n + 1)

        if self.verbose:
            print("\n" + "=" * 70)
            print("CAVITY GEOMETRY + RF OPTIMIZATION - DFO-LS (per-gap geometry)")
            print("=" * 70)
            print(f"Params: {n} ({self.n_gaps} gaps x {2 * self.n_segments} geometry "
                  f"+ {len(rf_optimize_params)} RF), budget maxfun={budget}")
            print(f"Search: steps_per_turn={of.steps_per_turn}, max_turns={max_turns}")

        of.iteration = 0
        of.best_cost = np.inf
        self.best_cost = np.inf
        self.best_geometry = None
        self._init_checkpoint_file()

        dt = of._estimate_timestep(of._rf_base_frequency())

        def objfun(x):
            return self.residuals_with_geometry(x, initial_beam, dt, max_turns,
                                                ls_weights, rf_optimize_params,
                                                r0_mode, skip_turns=skip_turns)

        start = time.time()
        soln = dfols.solve(objfun, x0, bounds=(lower, upper), maxfun=budget,
                           objfun_has_noise=True, scaling_within_bounds=True,
                           do_logging=False)
        elapsed = time.time() - start

        if self.verbose:
            print(f"DFO-LS: {soln.msg} (nf={soln.nf}, ||r||^2={soln.obj:.3e})")

        # Final tracking (optionally at higher resolution).
        spt_orig = of.steps_per_turn
        if verify_steps_per_turn is not None:
            of.steps_per_turn = verify_steps_per_turn
        try:
            result = self._finalize(
                soln.x, rf_optimize_params, initial_beam,
                verify_max_turns if verify_max_turns is not None else max_turns,
                r0_mode, param_names, param_bounds, ls_weights, 'dfols', elapsed,
                float(soln.obj),
                extra_meta={'dfols_msg': str(soln.msg), 'dfols_nf': int(soln.nf)})
        finally:
            of.steps_per_turn = spt_orig

        if self.verbose:
            print(f"Final energy: {result.final_energy_mev:.3f} MeV, "
                  f"turns: {result.n_turns}")
        return result

    # ---------------------------------------------------------------- staged
    def optimize_staged(self,
                        initial_beam,
                        rf_optimize_params: List[str] = ['bunch_phase', 'rf_freq'],
                        rf_bounds: Optional[Dict] = None,
                        ls_weights: Optional[Dict] = None,
                        search_steps_per_turn: int = 300,
                        search_max_turns: int = 8,
                        final_steps_per_turn: int = 500,
                        final_max_turns: int = 12,
                        phase_grid: Optional[np.ndarray] = None,
                        freq_fracs: Optional[np.ndarray] = None,
                        maxfun: Optional[int] = None,
                        r0_mode: str = 'offset',
                        skip_turns: int = DEFAULT_SKIP_TURNS,
                        workers: int = 1,
                        worker_builder=None,
                        worker_builder_args: tuple = (),
                        n_starts: Optional[int] = None,
                        geometry_jitter_deg: float = 1.0) -> OptimizedOrbit:
        """Three-stage optimization:

        A. Coarse RF scan (geometry frozen straight): grid over bunch phase x
           RF frequency at search resolution - removes the synchrotron
           phase-locking multimodality that a local optimizer can't cross.
        B. DFO-LS on the full per-gap parameter vector from the stage-A seed,
           at search resolution (default 300 steps/turn, 8 turns - the measured
           floor where candidate ranking is still reliable).
        C. Final tracking of the winner at full resolution.

        Parallel mode (``workers > 1``)
        -------------------------------
        Stage A is farmed over a persistent process pool, and stage B becomes a
        MULTI-START: ``n_starts`` (default = workers) independent DFO-LS runs
        from distinct stage-A RF basins (plus small geometry jitter for
        diversity), each with its own ``maxfun`` budget; the best result is
        verified in stage C. Requires ``worker_builder``: a MODULE-LEVEL
        function (picklable by reference) returning a fully configured
        CavityGeometryOptimizer; each worker calls it once (field load
        included) and then evaluates many tasks. Per-eval checkpoints go to
        per-worker files ``<checkpoint_file>.worker-<pid>.csv``.
        """
        if workers > 1:
            if worker_builder is None:
                raise ValueError("workers > 1 requires worker_builder (a module-level "
                                 "function returning a configured CavityGeometryOptimizer)")
            return self._optimize_staged_parallel(
                initial_beam, rf_optimize_params, rf_bounds, ls_weights,
                search_steps_per_turn, search_max_turns,
                final_steps_per_turn, final_max_turns,
                phase_grid, freq_fracs, maxfun, r0_mode, skip_turns,
                workers, worker_builder, worker_builder_args,
                n_starts, geometry_jitter_deg)

        of = self.orbit_finder
        of._set_beam_meta(initial_beam)
        spt_orig = of.steps_per_turn

        if phase_grid is None:
            phase_grid = np.arange(-180.0, 180.0, 30.0)
        if freq_fracs is None:
            freq_fracs = np.linspace(0.98, 1.02, 5)

        try:
            # ---- Stage A: coarse RF scan, geometry frozen at x0 (straight).
            of.steps_per_turn = search_steps_per_turn
            f0 = of._rf_base_frequency()
            _, _, x0 = self._build_full_param_space(
                initial_beam, rf_optimize_params, rf_bounds, r0_mode)
            angles0, radii0, _, _, _ = self._unpack_params(x0)
            for g, cavity in enumerate(of.design.rf_cavities):
                cavity.update_geometry(segment_angles=angles0[g], segment_radii=radii0[g])
            if self._is_bem and not self._attach_bem_for_current_geometry():
                raise RuntimeError(f"BEM attach failed for the stage-A "
                                   f"geometry: {self._bem_last_error}")

            if self.verbose:
                print("\n" + "=" * 70)
                print("STAGE A: RF scan (geometry frozen), "
                      f"{len(phase_grid)}x{len(freq_fracs)} grid @ spt={search_steps_per_turn}")
                print("=" * 70)

            best = (-np.inf, None, None)
            for ph in phase_grid:
                for ff in freq_fracs:
                    res = of.track_once(initial_beam, bunch_phase_deg=float(ph),
                                        rf_freq_mhz=f0 * ff / 1e6,
                                        max_turns=search_max_turns, r0_mode=r0_mode)
                    if res.final_energy_mev > best[0]:
                        best = (res.final_energy_mev, float(ph), f0 * ff)
                        if self.verbose:
                            print(f"  new best: E={best[0]:.3f} MeV @ "
                                  f"phase={best[1]:.0f} deg, f={best[2] / 1e6:.3f} MHz")

            e_seed, ph_seed, f_seed = best
            if self.verbose:
                print(f"Stage A seed: phase={ph_seed:.1f} deg, f={f_seed / 1e6:.4f} MHz "
                      f"(E={e_seed:.3f} MeV in {search_max_turns} turns)")

            # Seed vector: straight geometry + stage-A RF values.
            order = ['bunch_phase', 'rf_freq', 'r0', 'vr0']
            rf_seed_map = {'bunch_phase': ph_seed, 'rf_freq': f_seed}
            seed = np.asarray(x0, dtype=float).copy()
            k = self._rf_offset
            for name in order:
                if name in rf_optimize_params:
                    if name in rf_seed_map:
                        seed[k] = rf_seed_map[name]
                    k += 1

            # ---- Stage B: DFO-LS at search resolution; Stage C: verify at full.
            return self.optimize_dfols(
                initial_beam, max_turns=search_max_turns,
                rf_optimize_params=rf_optimize_params, rf_bounds=rf_bounds,
                ls_weights=ls_weights, maxfun=maxfun, seed_x0=seed, r0_mode=r0_mode,
                skip_turns=skip_turns,
                verify_max_turns=final_max_turns,
                verify_steps_per_turn=final_steps_per_turn)
        finally:
            of.steps_per_turn = spt_orig

    def _optimize_staged_parallel(self, initial_beam, rf_optimize_params, rf_bounds,
                                  ls_weights, search_spt, search_turns, final_spt,
                                  final_turns, phase_grid, freq_fracs, maxfun, r0_mode,
                                  skip_turns, workers, worker_builder,
                                  worker_builder_args, n_starts,
                                  geometry_jitter_deg) -> OptimizedOrbit:
        """Parallel staged optimization (see optimize_staged docstring)."""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        of = self.orbit_finder
        of._set_beam_meta(initial_beam)

        if ls_weights is None:
            ls_weights = dict(DEFAULT_LS_WEIGHTS)
        if phase_grid is None:
            phase_grid = np.arange(-180.0, 180.0, 30.0)
        if freq_fracs is None:
            freq_fracs = np.linspace(0.98, 1.02, 5)
        n_starts = n_starts if n_starts is not None else workers

        param_bounds, param_names, x0 = self._build_full_param_space(
            initial_beam, rf_optimize_params, rf_bounds, r0_mode)
        lower = np.array([b[0] for b in param_bounds], dtype=float)
        upper = np.array([b[1] for b in param_bounds], dtype=float)
        angles0, radii0, _, _, _ = self._unpack_params(x0)
        f0 = of._rf_base_frequency()
        n = len(x0)
        maxfun_per = maxfun if maxfun is not None else 30 * (n + 1)

        x_vec = np.array(initial_beam.x_vec, dtype=float)
        p_vec = np.array(initial_beam.p_vec, dtype=float)

        if self.verbose:
            print("\n" + "=" * 70)
            print(f"STAGED OPTIMIZATION - PARALLEL ({workers} workers)")
            print("=" * 70)
            print(f"Params: {n}; stage A grid {len(phase_grid)}x{len(freq_fracs)}; "
                  f"stage B: {n_starts} DFO-LS starts x maxfun={maxfun_per}")
            print("(each worker builds its own system once - field load takes a moment)")

        # Remove stale per-worker checkpoint files from previous runs (pids can
        # repeat, and the stage-B eval counter globs these files).
        if self.checkpoint_file:
            import glob as _glob
            for stale in _glob.glob(self.checkpoint_file + ".worker-*.csv"):
                try:
                    os.remove(stale)
                except OSError:
                    pass

        t_start = time.time()
        with ProcessPoolExecutor(
                max_workers=workers, initializer=_pool_init,
                initargs=(worker_builder, tuple(worker_builder_args),
                          self.checkpoint_file)) as pool:

            # ---- Stage A: parallel RF grid scan, geometry frozen at x0.
            tasks_a = [(float(ph), float(f0 * ff), x_vec, p_vec, search_spt,
                        search_turns, r0_mode, angles0, radii0)
                       for ph in phase_grid for ff in freq_fracs]
            futs = [pool.submit(_pool_track_once, t) for t in tasks_a]
            bar = (tqdm(total=len(futs), desc="Stage A (RF scan)", ncols=100)
                   if (self.verbose and tqdm) else None)
            scan = []
            best_e = -np.inf
            for fut in as_completed(futs):
                s = fut.result()
                if s is not None:
                    scan.append(s)
                    best_e = max(best_e, s[0])
                if bar:
                    bar.update(1)
                    bar.set_postfix_str(f"best E={best_e:.3f} MeV")
            if bar:
                bar.close()
            if not scan:
                raise RuntimeError("Stage A: all grid evaluations failed")
            scan.sort(key=lambda s: -s[0])

            if self.verbose:
                t_a = time.time() - t_start
                print(f"Stage A done in {t_a:.0f}s; top basins:")
                for E, ph, f in scan[:min(n_starts, 5)]:
                    print(f"  E={E:.3f} MeV @ phase={ph:.0f} deg, f={f / 1e6:.3f} MHz")

            # ---- Seeds: distinct VIABLE phase basins (>= 25% of the best
            # stage-A energy), then jittered clones of the good basins. Seeding
            # non-accelerating phases wastes whole DFO-LS starts.
            e_min = 0.25 * scan[0][0]
            picked = []
            for E, ph, f in scan:
                if len(picked) >= n_starts:
                    break
                if E >= e_min and all(abs(ph - p[1]) > 1e-6 for p in picked):
                    picked.append((E, ph, f))
            if not picked:
                picked.append(scan[0])
            base_seeds = list(picked)
            i_clone = 0
            while len(picked) < n_starts:
                picked.append(base_seeds[i_clone % len(base_seeds)])
                i_clone += 1

            rng = np.random.default_rng(42)
            order = ['bunch_phase', 'rf_freq', 'r0', 'vr0']
            seeds = []
            for i, (E, ph, f) in enumerate(picked):
                s = np.asarray(x0, dtype=float).copy()
                if i > 0 and geometry_jitter_deg > 0:
                    for g in range(self.n_gaps):
                        base = g * self._blk
                        s[base:base + self.n_segments] += rng.uniform(
                            -geometry_jitter_deg, geometry_jitter_deg, self.n_segments)
                k = self._rf_offset
                for name in order:
                    if name in rf_optimize_params:
                        if name == 'bunch_phase':
                            s[k] = ph
                        elif name == 'rf_freq':
                            s[k] = f
                        k += 1
                seeds.append(s)

            # ---- Stage B: multi-start DFO-LS in parallel.
            tasks_b = [(s, lower, upper, x_vec, p_vec, search_spt, search_turns,
                        dict(ls_weights), tuple(rf_optimize_params), r0_mode,
                        int(skip_turns), maxfun_per, float(f0)) for s in seeds]
            futs = [pool.submit(_pool_dfols, t) for t in tasks_b]
            bar = (tqdm(total=len(futs), desc="Stage B (DFO-LS starts)", ncols=100)
                   if (self.verbose and tqdm) else None)

            # Live eval counter: sum rows of the per-worker checkpoint CSVs.
            mon_stop = None
            if bar and self.checkpoint_file:
                import glob as _glob
                import threading

                def _monitor(stop):
                    while not stop.wait(15.0):
                        try:
                            n = 0
                            for fp in _glob.glob(self.checkpoint_file + ".worker-*.csv"):
                                with open(fp, 'rb') as fh:
                                    n += max(sum(1 for _ in fh) - 1, 0)
                            bar.set_postfix_str(f"~{n} evals total")
                        except Exception:
                            pass

                mon_stop = threading.Event()
                threading.Thread(target=_monitor, args=(mon_stop,), daemon=True).start()

            runs = []
            best_obj = np.inf
            for fut in as_completed(futs):
                r = fut.result()
                runs.append(r)
                if bar:
                    if r and 'error' not in r:
                        best_obj = min(best_obj, r['obj'])
                        bar.write(f"  start finished: ||r||^2={r['obj']:.3e} "
                                  f"(nf={r['nf']}, best so far {best_obj:.3e})")
                    bar.update(1)
            if mon_stop is not None:
                mon_stop.set()
            if bar:
                bar.close()

            ok = [r for r in runs if r and 'error' not in r]
            failed = [r for r in runs if r and 'error' in r]
            if self.verbose and failed:
                print(f"WARNING: {len(failed)} DFO-LS start(s) failed: "
                      f"{[r['error'] for r in failed]}")
            if not ok:
                raise RuntimeError(f"All DFO-LS starts failed: "
                                   f"{[r.get('error') for r in runs]}")

            # ---- Stage C part 1: verify EVERY start at FINAL resolution and
            # select by the VERIFIED objective. A start can win at search
            # resolution on a marginal capture basin that does not survive full
            # resolution; the search objective must never pick the winner.
            tasks_v = [(r['x'], x_vec, p_vec, final_spt, final_turns,
                        dict(ls_weights), tuple(rf_optimize_params), r0_mode,
                        int(skip_turns), float(f0)) for r in ok]
            vers = list(pool.map(_pool_verify, tasks_v, chunksize=1))
            for r, v in zip(ok, vers):
                if v and 'error' not in v:
                    r['obj_verified'] = v['obj']
                    r['energy_verified'] = v['energy']
                    r['turns_verified'] = v['turns']
                else:
                    r['obj_verified'] = np.inf
                    r['energy_verified'] = 0.0
                    r['turns_verified'] = 0

        best_run = min(ok, key=lambda r: r['obj_verified'])
        elapsed = time.time() - t_start
        # Total evaluations across the pool (stage A + DFO-LS + verification).
        of.iteration = len(tasks_a) + sum(r['nf'] for r in ok) + len(ok)

        if self.verbose:
            print(f"Stage B+verify done ({elapsed:.0f}s). Starts "
                  f"(search ||r||^2 -> verified ||r||^2 @ {final_spt} spt/"
                  f"{final_turns} turns):")
            for r in sorted(ok, key=lambda r: r['obj_verified']):
                marker = " <-- WINNER" if r is best_run else ""
                fragile = ("  [FRAGILE: does not survive full resolution]"
                           if r['obj_verified'] > 3.0 * r['obj'] else "")
                print(f"  {r['obj']:.3e} -> {r['obj_verified']:.3e}  "
                      f"E={r['energy_verified']:.3f} MeV, "
                      f"turns={r['turns_verified']}{marker}{fragile}")

        # ---- Stage C part 2: final tracking of the verified winner (parent).
        spt_orig = of.steps_per_turn
        of.steps_per_turn = final_spt
        try:
            result = self._finalize(
                np.asarray(best_run['x'], dtype=float), rf_optimize_params,
                initial_beam, final_turns, r0_mode, param_names, param_bounds,
                ls_weights, 'dfols-multistart', elapsed, best_run['obj_verified'],
                extra_meta={'workers': workers,
                            'n_starts': len(ok),
                            'start_objs': [r['obj'] for r in ok],
                            'start_objs_verified': [r['obj_verified'] for r in ok],
                            'dfols_msg': best_run['msg'],
                            'dfols_nf': best_run['nf']})
        finally:
            of.steps_per_turn = spt_orig

        if self.verbose:
            print(f"Final energy: {result.final_energy_mev:.3f} MeV, "
                  f"turns: {result.n_turns}")
        return result


if __name__ == "__main__":
    print("cavity_optimizer.py - RF Cavity Geometry Optimization (user-beam entry point)")
