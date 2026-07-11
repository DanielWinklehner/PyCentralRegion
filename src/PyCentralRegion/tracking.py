"""
tracking.py - Cyclotron tracking adapter on the centralized PyPATools Tracker.

TrackingEngine is now a thin BUILDER: it assembles cyclotron-specific hook
objects (RF-cavity kicks, radial/vertical boundary loss) from a CentralRegion
design and runs them on the generic, geometry-agnostic ``PyPATools.trackers.Tracker``.

All of the actual integration loop, Boris half-step handling, and the canonical
"alive" mask now live in PyPATools.trackers.Tracker. This module only supplies the
cylindrical pieces:

  * RFCavityInteraction      - an Interaction hook wrapping RFCavity crossing+kick
  * RadialBoundaryTerminator - a Terminator hook (2D radial loss)
  * RadialVerticalTerminator - a Terminator hook (3D radial + vertical loss)
  * CallbackRecorder         - a Recorder adapting the legacy callback(step,r,v,active,t)

Part of: PyCentralRegion module
"""
import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

from PyPATools.pusher import Pusher
from PyPATools.particles import ParticleDistribution
from PyPATools.global_variables import CLIGHT
from PyPATools.trackers import Tracker, Interaction, Terminator, Recorder


@dataclass
class TrackingResult:
    """Container for tracking results."""
    success: bool
    n_steps: int
    r_final: np.ndarray  # Final positions (n_particles, 3)
    v_final: np.ndarray  # Final velocities (n_particles, 3)
    active: np.ndarray  # Active particle mask (n_particles,)
    metadata: dict


# ============================================================================
# Cyclotron-specific hooks (all cylindrical assumptions live here)
# ============================================================================
class RFCavityInteraction(Interaction):
    """Apply RF-cavity kicks for every cavity in the design.

    Mirrors the legacy ``_track_step_2d_rf`` RF block exactly: per cavity, detect
    crossings on the active subset, map to a global mask, and apply the kick.
    """

    def __init__(self, design, pusher):
        self.design = design
        self.pusher = pusher

    def apply(self, step, r_prev, v_prev, r, v, active, t, dt):
        n = len(r)
        for cavity in self.design.rf_cavities:
            crossed_a, t_cross_a, seg_ids_a = cavity.check_crossings_batch(
                r_prev[active], r[active]
            )
            if np.any(crossed_a):
                active_indices = np.where(active)[0]
                sel = active_indices[crossed_a]
                # Map active-subset results to full-length, global-aligned arrays so
                # apply_kicks_batch can index them by global particle index safely
                # even when some particles are inactive.
                global_crossed = np.zeros(n, dtype=bool)
                t_cross_full = np.zeros(n)
                seg_ids_full = np.full(n, -1, dtype=int)
                global_crossed[sel] = True
                t_cross_full[sel] = t_cross_a[crossed_a]
                seg_ids_full[sel] = seg_ids_a[crossed_a]
                v, r, _, _ = cavity.apply_kicks_batch(
                    r_prev, r, v, global_crossed, t_cross_full, seg_ids_full,
                    t, dt, self.design, self.pusher
                )
        return r, v, active


class RadialBoundaryTerminator(Terminator):
    """Mark particles lost when their cylindrical radius exceeds r_max (2D)."""

    def __init__(self, r_max):
        self.r_max = r_max

    def update(self, step, r_prev, v_prev, r, v, active, t):
        radii = np.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2)
        active[radii > self.r_max] = False
        return active


class RadialVerticalTerminator(Terminator):
    """Mark particles lost on radial OR vertical excursion (3D)."""

    def __init__(self, r_max, z_max=0.1):
        self.r_max = r_max
        self.z_max = z_max

    def update(self, step, r_prev, v_prev, r, v, active, t):
        radii = np.sqrt(r[:, 0] ** 2 + r[:, 1] ** 2)
        lost = (radii > self.r_max) | (np.abs(r[:, 2]) > self.z_max)
        active[lost] = False
        return active


class CallbackRecorder(Recorder):
    """Adapt a legacy callback(step, r, v, active, t) -> terminate into a Recorder."""

    stop_reason = "turns_or_energy_reached"

    def __init__(self, callback: Callable):
        self.callback = callback

    def record(self, step, r_prev, v_prev, r, v, active, t):
        return self.callback(step, r, v, active, t)


class TrackingEngine:
    """
    Cyclotron tracking engine - builds hooks and runs PyPATools' Tracker.

    Parameters
    ----------
    design : CentralRegion
        Design with fields and RF cavities
    algorithm : str
        Pusher algorithm ('boris', 'rk4', 'rk4_rel', etc.)
    dimensionality : str
        '2D' or '3D'
    use_rf : bool
        Whether to apply RF kicks
    max_radius_m : float
        Maximum allowed radius for particle loss
    verbose : bool
        Print progress
    gap_model : str
        'thin' (default): RF gaps act via the thin-gap kick Interaction hook.
        'bem2d': no kick hook; the RF acceleration comes from continuous
        integration of ``design.efield``, which must be a TimedField wrapping
        the solved BEM gap-field pattern (see PyCentralRegion.gap_fields /
        AcceleratedOrbitFinder.attach_bem_field). The TimedField's omega and
        phase are re-synced from cavity 0 before every run, so bunch-phase and
        RF-frequency changes need no field re-solve.
    """

    def __init__(self,
                 design,
                 algorithm: str = 'rk4_rel',
                 dimensionality: str = '2D',
                 use_rf: bool = False,
                 max_radius_m: float = 0.5,
                 verbose: bool = True,
                 gap_model: str = 'thin'):
        if gap_model not in ('thin', 'bem2d'):
            raise ValueError(f"gap_model must be 'thin' or 'bem2d', got {gap_model!r}")
        self.design = design
        self.algorithm = algorithm
        self.dim = dimensionality
        self.use_rf = use_rf
        self.r_max = max_radius_m
        self.verbose = verbose
        self.gap_model = gap_model

        # Create pusher
        self.pusher = Pusher(design.species, algorithm=algorithm)

        # Temporary ParticleDistribution for calculations
        self.pd_temp = ParticleDistribution(species=design.species)

    def _sync_bem_field(self):
        """bem2d: validate design.efield and re-sync its modulation from cavity 0."""
        ef = self.design.efield
        if not hasattr(ef, 'set_time'):
            raise RuntimeError(
                "gap_model='bem2d' requires design.efield to be a TimedField wrapping "
                "the solved BEM gap-field pattern. Build one with "
                "PyCentralRegion.gap_fields.make_bem_efield() or "
                "AcceleratedOrbitFinder.attach_bem_field().")
        if len(self.design.rf_cavities) == 0:
            raise RuntimeError("gap_model='bem2d' requires RF cavities on the design")
        cav = self.design.rf_cavities[0]
        ef.omega = float(cav.omega)
        ef.phase = float(cav.bunch_phase_offset)

    def _build_hooks(self, callback):
        """Assemble interaction / terminator / recorder hooks for this config."""
        use_kicks = self.use_rf and self.gap_model == 'thin'
        interactions = [RFCavityInteraction(self.design, self.pusher)] if use_kicks else []

        if self.dim == '3D':
            terminators = [RadialVerticalTerminator(self.r_max)]
        else:
            terminators = [RadialBoundaryTerminator(self.r_max)]

        recorders = [CallbackRecorder(callback)] if callback is not None else []
        return interactions, terminators, recorders

    def track_multiparticle(self,
                            pd_init: ParticleDistribution,
                            dt: float,
                            n_steps: int,
                            callback: Optional[Callable] = None,
                            callback_frequency: int = 1,
                            show_progress: bool = True) -> TrackingResult:
        """
        Track multiple particles via the centralized Tracker.

        The callback signature is unchanged: callback(step, r, v, active, t) -> bool
        (return True to terminate). It is invoked every ``callback_frequency`` steps
        with the post-step time, exactly as before.
        """
        interactions, terminators, recorders = self._build_hooks(callback)

        if self.use_rf and self.gap_model == 'bem2d':
            self._sync_bem_field()

        tracker = Tracker(
            self.pusher, self.design.efield, self.design.bfield,
            interactions=interactions, terminators=terminators, recorders=recorders,
        )

        # sync_back=False preserves the legacy contract of not mutating pd_init;
        # the alive mask is still returned in the result.
        res = tracker.run(pd_init, dt, n_steps,
                          record_every=callback_frequency,
                          show_progress=show_progress, sync_back=False)

        if res.stop_reason == "all_lost":
            if self.verbose:
                print(f"\nAll particles lost at step {res.n_steps}")
            return TrackingResult(False, res.n_steps, res.r, res.v, res.active,
                                  {'termination': 'all_lost', 'time': res.t})
        if res.stopped:
            return TrackingResult(True, res.n_steps, res.r, res.v, res.active,
                                  {'termination': 'turns_or_energy_reached', 'time': res.t})
        return TrackingResult(True, res.n_steps, res.r, res.v, res.active, {'time': res.t})


# ========================================================================
# Convenience function for single particle tracking
# ========================================================================
def track_single_particle(design,
                          r0: np.ndarray,
                          v0: np.ndarray,
                          dt: float,
                          n_steps: int,
                          algorithm: str = 'rk4_rel',
                          use_rf: bool = False,
                          callback: Optional[Callable] = None) -> TrackingResult:
    """
    Convenience wrapper for single particle tracking (batch of one).
    """
    pd_init = ParticleDistribution(species=design.species)
    pd_init.x_vec = r0.reshape(1, 3)
    p_vec = v0 / np.sqrt(CLIGHT ** 2 - np.linalg.norm(v0) ** 2)
    pd_init.p_vec = p_vec.reshape(1, 3)

    engine = TrackingEngine(
        design,
        algorithm=algorithm,
        dimensionality='2D',
        use_rf=use_rf,
        verbose=False
    )

    return engine.track_multiparticle(
        pd_init,
        dt=dt,
        n_steps=n_steps,
        callback=callback,
        show_progress=False
    )
