"""
tracking.py - Cyclotron tracking adapter on the centralized PyPATools Tracker.

TrackingEngine is now a thin BUILDER: it assembles cyclotron-specific hook
objects (RF-cavity kicks, radial/vertical boundary loss) from a CentralRegion
design and runs them on the generic, geometry-agnostic ``PyPATools.trackers.Tracker``.

All of the actual integration loop, Boris half-step handling, and the canonical
"alive" mask now live in PyPATools.trackers.Tracker. This module only supplies the
cylindrical pieces:

  * RFCavityInteraction      - an Interaction hook wrapping RFCavity crossing+kick
  * SpaceChargeKick          - an Interaction hook: Poisson solve of the live bunch
                               every n steps, q (E + v x B) dt applied every step
  * RadialBoundaryTerminator - a Terminator hook (2D radial loss)
  * RadialVerticalTerminator - a Terminator hook (3D radial + vertical loss)
  * CallbackRecorder         - a Recorder adapting the legacy callback(step,r,v,active,t)

Part of: PyCentralRegion module
"""
import time
import os
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


class RadialSlitCollimator(Terminator):
    """Central-region phase slit: a simple two-piece grounded block at
    azimuth ``azimuth_rad``. On each particle's FIRST outbound crossing
    of that half-plane (turn 1 - later turns pass tens of mm further
    out, beyond the physical block), particles crossing OUTSIDE the
    radial aperture are intercepted. In the center region the turn-1
    radius correlates strongly with RF phase (energy gained in the
    first gaps), so the radial aperture acts as a phase slit.

    Two aperture modes:
      absolute  - pass ``r_lo_m`` / ``r_hi_m`` explicitly.
      reference - pass ``aperture_m`` (+ ``ref_particle``, default 0):
          the aperture is CENTERED on the reference particle's own
          first-crossing radius (the prepended bunch centroid in the
          multiparticle examples). This makes the slit self-centering
          for ANY candidate RF/geometry inside an optimization loop.
          Particles crossing before the reference are judged
          retroactively a few steps later, once the center is known
          (mm-scale path error on particles that are removed anyway).
          If the reference dies before crossing, no collimation occurs.

    Interceptions are logged in ``hits`` as (particle, step, r_cross_m);
    ``r_center_m`` holds the realized aperture center. ``reset()`` is
    called by TrackingEngine before every run.

    Particle indices in ``hits`` are shifted by ``index_offset`` so they
    address the PHYSICAL bunch: when a virtual reference particle is
    prepended (AcceleratedOrbitFinder.reference_centroid) the tracked
    arrays carry it at index 0, but every consumer - full_beam, the user's
    beam, the plots - is indexed without it.
    """

    def __init__(self, azimuth_rad, r_lo_m=None, r_hi_m=None,
                 aperture_m=None, ref_particle=0, exempt_ref=False,
                 index_offset=0):
        self.azimuth_rad = float(azimuth_rad)
        if aperture_m is None and (r_lo_m is None or r_hi_m is None):
            raise ValueError("give r_lo_m/r_hi_m or aperture_m")
        self.aperture_m = None if aperture_m is None else float(aperture_m)
        self.ref_particle = int(ref_particle)
        # Set when the reference is a VIRTUAL centroid particle rather than
        # beam: removing it would end turn counting. In self-centering mode it
        # sits exactly at the aperture centre and cannot be hit anyway.
        self.exempt_ref = bool(exempt_ref)
        self.index_offset = int(index_offset)
        self._r_lo0 = None if r_lo_m is None else float(r_lo_m)
        self._r_hi0 = None if r_hi_m is None else float(r_hi_m)
        self.reset()

    def reset(self):
        self.seen = None
        self.judged = None
        self.first_r = None
        self.hits = []
        if self.aperture_m is None:
            self.r_center_m = 0.5 * (self._r_lo0 + self._r_hi0)
            self.r_lo_m, self.r_hi_m = self._r_lo0, self._r_hi0
        else:
            self.r_center_m = None          # set by the reference crossing
            self.r_lo_m = self.r_hi_m = None

    def _judge(self, active):
        """Judge all crossed-but-unjudged particles against the (now
        known) aperture."""
        idx = np.flatnonzero(self.seen & ~self.judged)
        for p in idx:
            if self.exempt_ref and p == self.ref_particle:
                self.judged[p] = True
                continue
            rr = self.first_r[p]
            if rr < self.r_lo_m or rr > self.r_hi_m:
                if active[p]:
                    active[p] = False
                    self.hits.append((int(p) - self.index_offset, -1, float(rr)))
            self.judged[p] = True
        return active

    def update(self, step, r_prev, v_prev, r, v, active, t):
        if self.seen is None:
            n = len(r)
            self.seen = np.zeros(n, dtype=bool)
            self.judged = np.zeros(n, dtype=bool)
            self.first_r = np.full(n, np.nan)
        ca, sa = np.cos(self.azimuth_rad), np.sin(self.azimuth_rad)
        u_prev = -r_prev[:, 0] * sa + r_prev[:, 1] * ca
        u = -r[:, 0] * sa + r[:, 1] * ca
        along = r[:, 0] * ca + r[:, 1] * sa
        crossed = active & ~self.seen & (u_prev < 0.0) & (u >= 0.0) \
            & (along > 0.0)
        if np.any(crossed):
            idx = np.flatnonzero(crossed)
            f = -u_prev[idx] / (u[idx] - u_prev[idx])
            rc = r_prev[idx, :2] + f[:, None] * (r[idx, :2] - r_prev[idx, :2])
            self.first_r[idx] = np.hypot(rc[:, 0], rc[:, 1])
            self.seen[idx] = True
            if self.r_center_m is None \
                    and self.seen[self.ref_particle]:
                self.r_center_m = float(self.first_r[self.ref_particle])
                self.r_lo_m = self.r_center_m - 0.5 * self.aperture_m
                self.r_hi_m = self.r_center_m + 0.5 * self.aperture_m
        if self.r_center_m is not None and self.seen is not None \
                and np.any(self.seen & ~self.judged):
            active = self._judge(active)
        return active


class MetalTerminator(Terminator):
    """Lose particles that enter metal.

    ``inside(xy) -> bool array`` is any midplane obstacle test, e.g.
    ``electrodes3d.midplane_obstacles(model)`` (the 3D solids rasterised at
    z = 0, optionally dilated by a beam half-width). The tall-wall 2D
    footprints are NOT obstacles - the beam flies through the dee and hill
    wedges between their plates - only the scroll / central post is.

    Interceptions are logged in ``hits`` as (particle, step, x_m, y_m) with
    ``index_offset`` removed (see RadialSlitCollimator). A virtual reference
    particle (``exempt_ref``) is never removed - losing it would end turn
    counting - but its contacts are logged in ``ref_hits`` as (step, x, y) so
    the caller can fail loudly. ``every`` thins the test to every n-th step.
    ``reset()`` is called by TrackingEngine before every run.
    """

    def __init__(self, inside, index_offset=0, ref_particle=0, exempt_ref=False,
                 every=1):
        self.inside = inside
        self.index_offset = int(index_offset)
        self.ref_particle = int(ref_particle)
        self.exempt_ref = bool(exempt_ref)
        self.every = max(1, int(every))
        self.reset()

    def reset(self):
        self.hits = []
        self.hits_z = []          # z [m] of every entry of hits (3D runs)
        self.ref_hits = []

    def update(self, step, r_prev, v_prev, r, v, active, t):
        if step % self.every:
            return active
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            return active
        # 3D tests (electrodes3d.stacked_obstacles, fields3d.Raster3D) carry ndim = 3
        cols = 3 if getattr(self.inside, 'ndim', 2) == 3 else 2
        hit = np.asarray(self.inside(r[idx, :cols]), dtype=bool)
        for p in idx[hit]:
            p = int(p)
            if self.exempt_ref and p == self.ref_particle:
                self.ref_hits.append((int(step), float(r[p, 0]), float(r[p, 1])))
                continue
            active[p] = False
            self.hits.append((p - self.index_offset, int(step),
                              float(r[p, 0]), float(r[p, 1])))
            self.hits_z.append(float(r[p, 2]))
        return active


class TimedRelease(Interaction):
    """Staggered launch: particle i starts moving at its birth step.

    ``birth_step[i]`` is the first step in which particle i is pushed. The
    particles with birth_step 0 must be alive in the initial distribution;
    the others start with ``alive = False`` - frozen where they are, since the
    tracker pushes, kicks and tests only active particles - and are released
    here at the end of step ``birth_step - 1``. Built by
    ``AcceleratedOrbitFinder.track_with_rf`` from
    ``ParticleDistribution.birth_time`` (see ``handoff.make_beam_from_handoff``);
    the clock starts at the earliest birth, the bunch centre is born at t = 0.
    Not for the Boris pusher (its half-step start would be skipped for late
    particles). ``released`` counts the particles released so far.
    """

    def __init__(self, birth_step):
        self.birth_step = np.asarray(birth_step, dtype=int)
        self.released = int(np.sum(self.birth_step == 0))

    def apply(self, step, r_prev, v_prev, r, v, active, t, dt):
        due = self.birth_step == step + 1
        if np.any(due):
            active = active.copy()
            active[due] = True
            self.released += int(due.sum())
        return r, v, active



def _rotation_from_to(a, b):
    """Rotation matrix taking direction a onto direction b (Rodrigues); identity when they are parallel."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return np.eye(3)
    a = a / na; b = b / nb
    k = np.cross(a, b); s = float(np.linalg.norm(k)); c = float(np.dot(a, b))
    if s < 1e-12:
        return np.eye(3)
    k = k / s
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    th = np.arctan2(s, c)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)

class SpaceChargeKick(Interaction):
    """Space-charge kick of the live bunch from an open-boundary Poisson solve.

    Every ``resolve_every`` steps the ACTIVE particles - born (TimedRelease),
    not lost, and carrying charge (the virtual reference particle carries
    none) - deposit their macro-charges and ``solver.solve_eb(positions,
    charges, velocities)`` returns E (and B) at each of them. The solver is
    typically ``PyPATools.poisson_fft.FFTPoissonSolver``: free space, no
    electrode images, non-relativistic today; when its relativistic version
    lands (rest-frame boost, B = v x E / c^2) the B it returns is applied here
    without any further change. The per-particle field is then applied as the
    impulse q (E + v x B) dt at EVERY step until the next solve, i.e. the
    field travels with the particle instead of staying where the bunch was
    when it was solved. Particles released between two solves receive the
    last solve's field gathered at their own position (``solver.gather``).

    Momentum update (the RF kick's total-speed bookkeeping):
        u = gamma v;   u += (q/m) (E + v x B) dt;   v = u / sqrt(1 + u^2/c^2)
    ``kick_z=False`` (default) drops E_z: the midplane tracker carries z but
    does not push it, so a vertical kick would only distort vz.

    Macro-charge per particle, in order of precedence: ``macro_charge_c``;
    ``bunch_current_a`` with ``bunch_frequency_hz`` (Q = I / f shared by the
    real particles, the handoff.py convention; the frequency defaults to the
    beam's hand-off RF frequency, then to the RF frequency of cavity 0); the
    beam's own ``macro_charge_c`` (set by ``handoff.make_beam_from_handoff``:
    the file's weights, total charge preserved under sub-sampling). All of it
    times ``charge_scale``. With zero total charge the hook is a strict no-op,
    so a 0 mA run reproduces the plain tracking bit for bit.

    ``prepare(beam, n_ref, species, ...)`` is called by
    ``AcceleratedOrbitFinder.track_with_rf`` before every run (sizes the
    per-particle arrays, resets the counters); ``report()`` is what the finder
    stores in the result metadata under 'space_charge': solve count, timing,
    mean and max |E_sc| on the bunch, the bunch rms size at each solve (cells
    per sigma is the resolution check) and the solver's own summary.
    """

    def __init__(self, solver, resolve_every: int = 8, macro_charge_c: Optional[float] = None,
                 bunch_current_a: Optional[float] = None, bunch_frequency_hz: Optional[float] = None,
                 charge_scale: float = 1.0, kick_z: bool = False, min_particles: int = 2,
                 exempt_ref: bool = True, verbose: bool = False, log_limit: int = 20000):
        if not hasattr(solver, 'solve_eb'):
            raise TypeError("solver must provide solve_eb(positions, charges, velocities) -> (E, B)")
        self.solver = solver
        self.move = os.environ.get('HCHC60_SC_MOVE', '0') == '1'   # 2026-09-16: self-field carried with the bunch
        self._c0 = None
        self.every = max(1, int(resolve_every))
        self.macro_charge_c = macro_charge_c
        self.bunch_current_a = bunch_current_a
        self.bunch_frequency_hz = bunch_frequency_hz
        self.charge_scale = float(charge_scale)
        self.kick_z = bool(kick_z)
        self.min_particles = max(1, int(min_particles))
        # the virtual reference particle (finder index 0) never deposits; with
        # exempt_ref it is not kicked either, so the centroid orbit that defines
        # the Poincare section and the turn count is the same at every current
        self.exempt_ref = bool(exempt_ref)
        self.verbose = bool(verbose)
        self.log_limit = int(log_limit)
        self.charges = None
        self.q_over_m = None
        self._reset_state(0)

    def _reset_state(self, n):
        self.E = np.zeros((n, 3))
        self.B = np.zeros((n, 3))
        self.has_field = np.zeros(n, dtype=bool)
        self.kick_mask = np.ones(n, dtype=bool)
        self.deposit_mask = np.zeros(n, dtype=bool)
        self.n_solves = 0
        self.n_skipped = 0
        self.n_kick_steps = 0
        self.solve_time = 0.0
        self.log = []                 # per solve: step, t, n_deposit, mean|E|, max|E|, wall, sx, sy, sz
        self.bunch_charge_c = 0.0
        self.n_charged = 0
        self.noop = True

    def resolve_macro_charge(self, beam, n_real: int, default_frequency_hz: Optional[float] = None) -> float:
        """Charge per real macro-particle [C] (before ``charge_scale``)."""
        if self.macro_charge_c is not None:
            return float(self.macro_charge_c)
        if self.bunch_current_a is not None:
            f = self.bunch_frequency_hz
            if f is None:
                meta = getattr(beam, 'handoff_meta', None) or {}
                f = meta.get('rf_frequency_hz')
            if f is None:
                f = default_frequency_hz
            if f is None or f <= 0:
                raise ValueError("bunch_current_a needs bunch_frequency_hz (bunch repetition frequency)")
            return float(self.bunch_current_a) / float(f) / max(int(n_real), 1)
        q = getattr(beam, 'macro_charge_c', None)
        if q is None:
            raise ValueError("no charge: give macro_charge_c or bunch_current_a, or a beam with macro_charge_c")
        return float(q)

    def prepare(self, beam: ParticleDistribution, n_ref: int = 0, species=None,
                default_frequency_hz: Optional[float] = None):
        """Size the per-particle state for ``beam`` (with ``n_ref`` virtual
        reference particles prepended) and reset the counters."""
        n = int(beam.numpart)
        n_ref = int(n_ref)
        self._reset_state(n)
        if species is None:
            species = getattr(beam, 'species', None)
        if species is None:
            raise ValueError("species needed for q/m")
        self.q_over_m = float(species.charge) / float(species.mass_kg)
        q_macro = self.resolve_macro_charge(beam, n - n_ref, default_frequency_hz) * self.charge_scale
        self.charges = np.full(n, q_macro)
        self.charges[:n_ref] = 0.0
        self.deposit_mask = self.charges != 0.0
        if self.exempt_ref:
            self.kick_mask[:n_ref] = False
        self.n_charged = int(self.deposit_mask.sum())
        self.bunch_charge_c = float(self.charges.sum())
        self.noop = self.bunch_charge_c == 0.0 or self.n_charged == 0
        if self.verbose:
            print(f"    space charge: {self.n_charged} charged particles x {q_macro:.4e} C = "
                  f"{self.bunch_charge_c:.4e} C, solve every {self.every} steps"
                  + (" (zero charge: no-op)" if self.noop else ""))

    # ------------------------------------------------------------------ core
    def _solve(self, step, r, v, active, t):
        dep = active & self.deposit_mask
        n_dep = int(dep.sum())
        self.E[:] = 0.0
        self.B[:] = 0.0
        self.has_field[:] = False
        self._c0 = None
        if n_dep < self.min_particles:
            self.n_skipped += 1
            return
        t0 = time.perf_counter()
        E_dep, B_dep = self.solver.solve_eb(r[dep], self.charges[dep], v[dep])
        self.E[dep] = E_dep
        self.B[dep] = B_dep
        self.has_field[dep] = True
        if self.move:
            # the bunch's mean direction is re-derived at every step from the SAME particles (depositing now,
            # still alive then); the solve's vectors are kept and rotated from there
            self._c0 = True
            self._sol_mask = dep.copy()
            self._v0 = v.copy()
            self._E_solve = self.E.copy()
            self._B_solve = self.B.copy()
        # charged-less particles that are kicked (a non-exempt reference): the field at their position
        others = active & ~dep & self.kick_mask
        if np.any(others) and hasattr(self.solver, 'gather'):
            self.E[others] = self.solver.gather(r[others])
            self.has_field[others] = True
        wall = time.perf_counter() - t0
        self.solve_time += wall
        self.n_solves += 1
        e_abs = np.linalg.norm(E_dep, axis=1)
        if not self.kick_z:
            e_abs = np.hypot(E_dep[:, 0], E_dep[:, 1])
        sig = r[dep].std(axis=0)
        row = [int(step), float(t), n_dep, float(e_abs.mean()), float(e_abs.max()), float(wall),
               float(sig[0]), float(sig[1]), float(sig[2])]
        if len(self.log) < self.log_limit:
            self.log.append(row)
        if self.verbose and (self.n_solves == 1 or self.n_solves % 200 == 0):
            print(f"    SC solve {self.n_solves:5d} @ step {step:6d}: {n_dep:6d} live, |E_sc| mean {row[3]:.3e} / "
                  f"max {row[4]:.3e} V/m, rms size {1e3 * sig[0]:.2f} x {1e3 * sig[1]:.2f} x {1e3 * sig[2]:.2f} mm, "
                  f"{1e3 * wall:.1f} ms", flush=True)

    def _kick(self, v, mask, dt):
        E = self.E[mask]
        B = self.B[mask]
        if not self.kick_z:
            E = E.copy()
            E[:, 2] = 0.0
        # only particles with a non-zero field: keeps E == 0 runs bit-identical
        nz = np.any(E != 0.0, axis=1) | np.any(B != 0.0, axis=1)
        if not np.any(nz):
            return
        idx = np.flatnonzero(mask)[nz]
        E, B = E[nz], B[nz]
        vv = v[idx]
        gamma = 1.0 / np.sqrt(1.0 - np.sum(vv * vv, axis=1) / CLIGHT ** 2)
        u = gamma[:, None] * vv
        force = E
        if np.any(B != 0.0):
            force = E + np.cross(vv, B)
        u = u + (self.q_over_m * dt) * force
        gamma_new = np.sqrt(1.0 + np.sum(u * u, axis=1) / CLIGHT ** 2)
        v[idx] = u / gamma_new[:, None]
        self.n_kick_steps += 1

    def apply(self, step, r_prev, v_prev, r, v, active, t, dt):
        if self.charges is None or len(self.charges) != len(r):
            raise RuntimeError("SpaceChargeKick.prepare(beam, ...) must be called for this beam first")
        if self.noop or not np.any(active):
            return r, v, active
        if step % self.every == 0:
            self._solve(step, r, v, active, t)
        elif self.move and getattr(self, '_c0', None) is not None:
            # 2026-09-16 (HCHC60_SC_MOVE): the stored per-particle vectors of the last solve are ROTATED with the mean
            # direction of the bunch since the solve (the same particles then and now), so the kick turns with the
            # orbit between solves; each particle still carries the field of its own solve position (self-force-free -
            # a re-gather at a mapped-back point picks up the particle's own charge cloud, 1.1 mm off the every-step
            # reference at 120 macro-particles, versus 34 um for the un-rotated per-particle scheme).
            common = self._sol_mask & active
            if np.any(common):
                R = _rotation_from_to(self._v0[common].mean(axis=0), v[common].mean(axis=0))
                who = active & self.kick_mask & self.has_field
                if np.any(who):
                    self.E[who] = self._E_solve[who] @ R.T
                    self.B[who] = self._B_solve[who] @ R.T
            fresh = active & ~self.has_field & self.deposit_mask
            if np.any(fresh) and self.n_solves > 0 and hasattr(self.solver, 'gather'):
                self.E[fresh] = self.solver.gather(r[fresh])
                self.has_field[fresh] = True
        else:
            fresh = active & ~self.has_field & self.deposit_mask
            if np.any(fresh) and self.n_solves > 0 and hasattr(self.solver, 'gather'):
                # released since the last solve: the field of the last solve at their position
                self.E[fresh] = self.solver.gather(r[fresh])
                self.has_field[fresh] = True
        kick = active & self.kick_mask & self.has_field
        if np.any(kick):
            self._kick(v, kick, dt)
        return r, v, active

    # ---------------------------------------------------------------- report
    def report(self) -> dict:
        """Solve count, timings and field statistics of the last run."""
        log = np.asarray(self.log, dtype=float).reshape(-1, 9)
        out = {
            'n_solves': int(self.n_solves), 'n_skipped': int(self.n_skipped),
            'n_kick_steps': int(self.n_kick_steps),
            'resolve_every': int(self.every), 'kick_z': self.kick_z, 'exempt_ref': self.exempt_ref, 'move': bool(self.move),
            'charge_scale': self.charge_scale, 'noop': bool(self.noop),
            'n_charged': int(self.n_charged), 'bunch_charge_c': float(self.bunch_charge_c),
            'macro_charge_c': (float(self.charges[self.deposit_mask][0])
                               if self.charges is not None and self.n_charged else 0.0),
            'solve_time_s': float(self.solve_time),
            'mean_solve_ms': float(1e3 * self.solve_time / self.n_solves) if self.n_solves else 0.0,
            'max_solve_ms': float(1e3 * log[:, 5].max()) if len(log) else 0.0,
            # time-mean over the solves of the mean |E_sc| on the depositing particles
            'mean_abs_e_v_per_m': float(log[:, 3].mean()) if len(log) else 0.0,
            'max_abs_e_v_per_m': float(log[:, 4].max()) if len(log) else 0.0,
            'n_deposit_min_max': [int(log[:, 2].min()), int(log[:, 2].max())] if len(log) else None,
            'rms_size_mm_first_last': ([list(np.round(1e3 * log[0, 6:9], 3)), list(np.round(1e3 * log[-1, 6:9], 3))]
                                       if len(log) else None),
            'log_columns': ['step', 't_s', 'n_deposit', 'mean_abs_e', 'max_abs_e', 'wall_s',
                            'sigma_x_m', 'sigma_y_m', 'sigma_z_m'],
            'log': log,
        }
        if hasattr(self.solver, 'summary'):
            out['solver'] = self.solver.summary()
        return out


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
                 gap_model: str = 'thin',
                 z_max: float = 0.1):
        if gap_model not in ('thin', 'bem2d'):
            raise ValueError(f"gap_model must be 'thin' or 'bem2d', got {gap_model!r}")
        self.design = design
        self.algorithm = algorithm
        self.dim = dimensionality
        self.use_rf = use_rf
        self.r_max = max_radius_m
        # 3D: |z| beyond this is lost (RadialVerticalTerminator); the metal
        # test (MetalTerminator with a 3D inside) handles the real apertures
        self.z_max = float(z_max)
        self.verbose = verbose
        self.gap_model = gap_model

        # Create pusher
        self.pusher = Pusher(design.species, algorithm=algorithm)

        # Extra terminators (e.g. RadialSlitCollimator) appended to the
        # boundary terminator on every run; stateful ones exposing
        # reset() are reset per run. Extra interactions (e.g. TimedRelease)
        # run BEFORE the RF kicks.
        self.extra_terminators = []
        self.extra_interactions = []

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
        interactions = list(self.extra_interactions)
        if use_kicks:
            interactions.append(RFCavityInteraction(self.design, self.pusher))

        if self.dim == '3D':
            terminators = [RadialVerticalTerminator(self.r_max, z_max=self.z_max)]
        else:
            terminators = [RadialBoundaryTerminator(self.r_max)]
        for tm in self.extra_terminators:
            if hasattr(tm, 'reset'):
                tm.reset()
        terminators += list(self.extra_terminators)

        recorders = [CallbackRecorder(callback)] if callback is not None else []
        return interactions, terminators, recorders

    def track_multiparticle(self,
                            pd_init: ParticleDistribution,
                            dt: float,
                            n_steps: int,
                            callback: Optional[Callable] = None,
                            callback_frequency: int = 1,
                            show_progress: bool = True,
                            t0: float = 0.0) -> TrackingResult:
        """
        Track multiple particles via the centralized Tracker.

        The callback signature is unchanged: callback(step, r, v, active, t) -> bool
        (return True to terminate). It is invoked every ``callback_frequency`` steps
        with the post-step time, exactly as before. ``t0`` is the clock at the
        start (negative for a staggered launch whose bunch centre is born at 0).
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
        res = tracker.run(pd_init, dt, n_steps, t0=float(t0),
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
