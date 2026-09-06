"""
accelerated_orbit_finder.py - Accelerated Orbit Optimizer (user-beam entry point)

Tracks and optimizes the acceleration of a USER-SUPPLIED initial beam (e.g. the
output of a spiral-inflector simulation) - not an SEO. The beam is a
``ParticleDistribution`` in the lab frame; single particle (numpart==1) and
multi-particle are handled uniformly (inferred from the beam).

Optimizes RF parameters (bunch phase, RF frequency) and, optionally, the
injection point via r0/pr0 - either as offsets to the supplied beam's centroid
(``r0_mode='offset'``) or as an absolute single-particle launch
(``r0_mode='absolute'``).

Uses the centralized TrackingEngine (-> PyPATools Tracker) and the shared
diagnostics. Part of: PyCentralRegion module.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from scipy.optimize import differential_evolution, minimize
import time
import csv

from .tracking import TrackingEngine
from .diagnostics import (PoincareAnalyzer, calculate_turn_metrics,
                          BeamStatisticsCollector, TurnStatistics)
from PyPATools.particles import ParticleDistribution
from PyPATools.global_variables import CLIGHT


# ============================================================================
# Initial-beam construction helpers (spiral-inflector hand-off)
# ============================================================================
def make_beam_from_state(species, x_vec, v_vec) -> ParticleDistribution:
    """Build a ParticleDistribution from explicit lab-frame positions/velocities.

    x_vec, v_vec : array-like, shape (N, 3) [m] and [m/s] (a single (3,) is ok).
    """
    x_vec = np.atleast_2d(np.asarray(x_vec, dtype=float))
    v_vec = np.atleast_2d(np.asarray(v_vec, dtype=float))
    pd = ParticleDistribution(species=species, x_vec=x_vec.copy(),
                              p_vec=np.zeros_like(x_vec))
    pd.set_p_from_v_vec(v_vec)
    return pd


def make_single_particle_beam(species, r, theta_deg=0.0, vr=0.0,
                              v_total=None, v_az=None) -> ParticleDistribution:
    """Single particle at radius ``r``, azimuth ``theta_deg``, radial velocity ``vr``.

    Provide either ``v_total`` (azimuthal speed solved from v_total^2 - vr^2) or
    ``v_az`` directly. Motion is counter-clockwise (cyclotron convention).
    """
    th = np.deg2rad(theta_deg)
    if v_az is None:
        if v_total is None:
            raise ValueError("provide either v_total or v_az")
        v_az = np.sqrt(max(v_total ** 2 - vr ** 2, 0.0))
    er = np.array([np.cos(th), np.sin(th), 0.0])     # radial unit vector
    et = np.array([-np.sin(th), np.cos(th), 0.0])    # azimuthal (CCW) unit vector
    x = r * er
    v = vr * er + v_az * et
    return make_beam_from_state(species, x.reshape(1, 3), v.reshape(1, 3))


def make_gaussian_beam(species, r_mean, v_tangential, n_particles,
                       r_spread=0.0, vr_spread=0.0, v_perp=0.0) -> ParticleDistribution:
    """Gaussian beam centred at radius ``r_mean`` on the +x axis, tangential speed
    ``v_tangential``, with radial position/velocity spreads. Convenience for tests
    and for synthesising a beam when a real inflector distribution isn't available.
    """
    sigma_px = (vr_spread / np.sqrt(CLIGHT ** 2 - vr_spread ** 2)) if vr_spread > 0 else 1e-20
    pd = ParticleDistribution.generate_distribution(
        species,
        type=['gaussian', 'gaussian', 'gaussian'],
        s_direction='z',
        n_particles=n_particles,
        correlation_matrix=np.eye(6),
        sigma_x=r_spread if r_spread > 0 else 1e-20,
        sigma_px=sigma_px,
        sigma_y=1e-20, sigma_py=1e-20, sigma_z=1e-20, sigma_pz=1e-20,
        cutoff_x=3, cutoff_px=3,
    )
    pd.set_centroid(r_mean, 0.0, 0.0)
    pd.add_mean_momentum(
        (v_perp / np.sqrt(CLIGHT ** 2 - v_perp ** 2)) if v_perp != 0 else 0.0,
        v_tangential / np.sqrt(CLIGHT ** 2 - v_tangential ** 2),
        0.0,
    )
    return pd


def make_beam_from_cylindrical(species, r, theta_deg, z, p_r, p_theta, p_z) -> ParticleDistribution:
    """Single particle from cylindrical lab coordinates.

    Position (r, theta_deg, z) in [m, deg, m]; momentum (p_r, p_theta, p_z) in
    beta*gamma (radial, azimuthal, vertical). Matches a spiral-inflector hand-off
    expressed in (r, theta) components.
    """
    th = np.deg2rad(theta_deg)
    er = np.array([np.cos(th), np.sin(th), 0.0])
    et = np.array([-np.sin(th), np.cos(th), 0.0])
    x = np.array([r * np.cos(th), r * np.sin(th), z])
    p = p_r * er + p_theta * et + np.array([0.0, 0.0, p_z])
    return ParticleDistribution(species=species, x_vec=x.reshape(1, 3), p_vec=p.reshape(1, 3))


# Least-squares weights (DFO-LS residuals). Energy must DOMINATE so the
# optimizer cannot "win" by not accelerating at all (tightly-centered circles
# zero the centering/smoothness residuals more cheaply than real acceleration
# with an imperfect orbit center - observed in multi-start runs). But centering
# and smoothness are NOT mere tie-breakers: central-region matching exists to
# hand the larger machine well-centered orbits and smooth acceleration, so they
# carry real weight - combined with ``skip_turns`` (default 2), which exempts
# the unavoidably lopsided first turns (modified gap geometry + off-orbit
# injection) from the quality residuals.
# Centering uses the SPIRAL-CORRECTED first-harmonic offset (r_center_h1): the
# legacy centroid metric carries an irreducible ~dr/2pi artifact on accelerated
# orbits, so weighting it up just fought the acceleration. With the clean
# metric, center=1.0 exerts real pressure on something the optimizer can zero.
# 'envelope' (per-turn radial beam spread) and 'survival' (per-turn lost
# fraction) apply only to multiparticle beams (numpart > 1); losing beam must
# hurt comparably to not accelerating, hence survival = energy weight.
# 'phase' penalizes per-turn RF-phase excursion from the mean (unwrapped,
# scale 15 deg): without it the optimizer tolerates a large synchronous-phase
# walk (observed: 139 -> 44 deg over 12 turns, with almost no energy gain in
# the final turn). Phase stability at hand-off matters for the main machine.
# 'collimated' prices the central-region phase slit (RadialSlitCollimator,
# optimize via 'coll_azimuth' [deg] / 'coll_aperture' [mm] in the RF param
# list): ONE residual sqrt(w) * n_collimated/n0. Intentional turn-1 removal
# is priced once (it reduces the beam count at the END), unlike dynamic
# 'survival' losses which count per turn - otherwise a cheap 30 keV
# grounded-block interception would cost ~n_turns times a late loss and the
# optimizer would pin the aperture open. Dynamic survival is measured
# against the post-collimator population.
DEFAULT_LS_WEIGHTS = {'energy': 4.0, 'center': 1.0, 'smooth': 0.5,
                      'envelope': 0.5, 'survival': 4.0, 'phase': 0.5,
                      'collimated': 4.0}
DEFAULT_SKIP_TURNS = 2

# Extra revolution periods of tracking budget beyond max_turns. The callback
# aborts the moment the last turn is logged, so this costs nothing on a
# well-behaved orbit; it exists so the final turn can actually complete when
# the real period differs from nominal, or when the first crossing costs a
# full period (launch on the section azimuth). Without it the clock ran out
# ~0.03 periods short of the last turn, silently returning max_turns - 1 turns
# and putting a step discontinuity in the objective.
TURN_BUDGET_MARGIN = 2

# Turn-separation smoothness (see log_turn_curvature). U_SCALE is the second
# difference of log(dr) that counts as one residual unit; DR_EPS keeps the log
# finite for a non-accelerating or shrinking orbit.
U_SCALE = 0.10
DR_EPS = 1e-3        # m

# Saturation limit [residual units] for the smoothness block. An orbit that is
# not accelerating has dr wobbling through zero, and log(max(dr,0) + DR_EPS)
# then swings ~3 log-units per step: measured on such a candidate the block
# reached 814 against an energy block of 15, i.e. entries of magnitude ~57 next
# to entries of ~0.2 on a healthy orbit. That is a conditioning problem for the
# least-squares model, not a modelling one, so the cure is saturation rather
# than a smaller scale. tanh is C-infinity, so no discontinuity is introduced.
#
# Fidelity where it matters: at 1 unit the clip costs 1.3%, at 2 units 5.0%,
# and on the measured 15-turn run (entries ~0.15) under 0.1%. A fully saturated
# block is 0.5 * U_RESID_MAX**2 * n_curvature_entries, ~100 at 15 turns with
# skip 4 - still above the energy block's ~22 ceiling, which is deliberate: a
# garbage orbit should be dominated by "your turn separation is nonsense", and
# that gradient points toward a monotone spiral, i.e. toward acceleration.
# Lower this to ~2.5 to force strict energy dominance everywhere, at the cost
# of compressing genuine kinks.
U_RESID_MAX = 5.0


def _soft_clip(x, limit=U_RESID_MAX):
    """Smoothly saturate |x| at ``limit``; near-identity for small |x|."""
    return limit * np.tanh(np.asarray(x, dtype=float) / limit)


def log_turn_curvature(dr, skip: int = 0) -> np.ndarray:
    """Second difference of log(turn separation) over the kept turns.

    ``dr`` should be a SMOOTH curve; working in log space makes the measure
    scale-free, so multiplying every dr by a constant - what a uniformly larger
    energy gain does - leaves it unchanged. A deviation-from-mean measure does
    not have that property: since r ~ sqrt(E), constant energy gain forces dr to
    shrink as 1/sqrt(E), so the spread grows with dE and riding closer to crest
    is penalized. That irreducible floor is what pinned an earlier 15-turn
    optimization at 86% of crest.

    A monotonic-decrease term is deliberately NOT imposed. The real cavity
    voltage follows a rising V(R) curve, so the energy gain per turn grows with
    radius and dr may legitimately flatten or tick up; a hinge on that would
    fight correct physics the same way the old metric did.

    The floor is not exactly zero: log(dr) is linear against log(E), not
    against turn index, so a constant-gain ramp still shows some curvature.
    Measured over the kept window of the 15-turn run (skip=4, ~1.8 -> 6.3 MeV)
    the kinematic floor is ~0.025 of an observed 0.049 at w=0.5, against a total
    cost of order 3 - and unlike the old metric that floor does NOT grow with
    the energy gain. It does grow if ``skip`` is small enough to keep the very
    low-energy turns (~0.96 for a 0.5 -> 7.9 MeV window). If it ever becomes
    the binding term, the refinement is to fit log(dr) against log(E) - exactly
    linear with slope -1/2 under constant gain - and penalize the residual.

    Returns an empty array when fewer than 3 turn separations survive ``skip``.
    """
    kept = np.asarray(dr, dtype=float)[skip:]
    if len(kept) < 3:
        return np.array([])
    # max(dr, 0) + DR_EPS rather than a clamp: a hard floor is discontinuous at
    # the floor, which is the same class of defect this metric exists to remove.
    # A shrinking orbit is already crushed by the energy block.
    u = np.log(np.maximum(kept, 0.0) + DR_EPS)
    return u[2:] - 2.0 * u[1:-1] + u[:-2]


@dataclass
class RFCrossingData:
    """Single RF crossing data."""
    turn: int
    cavity_id: int
    particle_id: int
    energy_before_kev: float
    energy_after_kev: float
    energy_gain_kev: float
    phase_deg: float
    time: float


@dataclass
class OptimizedOrbit:
    """Result from acceleration optimization (single or multi-particle)."""
    success: bool
    final_energy_mev: float
    n_turns: int
    n_particles: int
    bunch_phase_deg: float
    rf_frequency_mhz: float
    initial_r_mm: float
    initial_vr_m_s: float
    trajectory_reference: np.ndarray
    poincare_points_all: List[List]
    rf_crossings: List[RFCrossingData]
    turn_statistics: List[TurnStatistics]
    turn_metrics: dict
    std_r_per_step: np.ndarray
    cost: float
    metadata: dict = field(default_factory=dict)


class AcceleratedOrbitFinder:
    """
    Optimizer for accelerated orbits of a user-supplied initial beam.

    Parameters
    ----------
    design : CentralRegion
        Design with bfield, species, and RF cavities (frequency already set).
    target_energy_mev : float
        Target final energy [MeV].
    max_radius_m : float
        Maximum radius before a particle is considered lost [m].
    algorithm : str
        Pusher algorithm.
    steps_per_turn : int
        Integration steps per revolution (sets dt from the RF base frequency).
    verbose : bool
    checkpoint_file : str, optional
        CSV checkpoint file.
    gap_model : str
        'thin' (default): thin-gap RF kicks at the segment crossings.
        'bem2d': continuous integration of the real 2D electrostatic gap field
        (bempp solve); call ``attach_bem_field()`` once after the cavity
        geometry is final. RF frequency / bunch phase remain free (they only
        modulate the solved pattern), so ``optimize()`` over RF params and
        ``track_once()`` both work; geometry optimization does not (see
        CavityGeometryOptimizer). Continuous gap integration needs finer
        stepping than kicks - use steps_per_turn >= ~2000.
    reference_centroid : bool
        Multiparticle beams only: prepend the bunch centroid as a virtual
        particle 0 (see ``_with_reference_particle``). It defines the Poincare
        section, the reference trajectory and the per-turn orbit metrics, and
        is excluded from every beam statistic and from the particle count.
    """

    def __init__(self,
                 design,
                 target_energy_mev: float,
                 max_radius_m: float = 0.4,
                 algorithm: str = 'rk4_rel',
                 steps_per_turn: int = 500,
                 verbose: bool = True,
                 checkpoint_file: Optional[str] = None,
                 gap_model: str = 'thin',
                 reference_centroid: bool = True):

        self.design = design
        self.target_energy_mev = target_energy_mev
        self.r_max = max_radius_m
        self.algorithm = algorithm
        self.steps_per_turn = steps_per_turn
        self.verbose = verbose
        self.checkpoint_file = checkpoint_file
        self.gap_model = gap_model
        self.reference_centroid = reference_centroid
        self.bem_solution = None

        if not design.is_valid(verbose=False):
            raise ValueError("Design must have bfield and species")
        if len(design.rf_cavities) == 0:
            raise ValueError("Design must have at least one RF cavity")

        # RF parameters are (re)set on every objective evaluation; gate the
        # design's per-call prints by this finder's verbosity.
        design.verbose = verbose

        # Beam metadata (set per-run from the supplied beam). n_particles is
        # the REAL particle count; _n_ref counts the prepended virtual
        # reference particle, which is not part of the beam.
        self.n_particles = 1
        self.is_multiparticle = False
        self._n_ref = 0

        self.engine = TrackingEngine(
            design, algorithm=algorithm, dimensionality='2D', use_rf=True,
            max_radius_m=max_radius_m, verbose=False, gap_model=gap_model,
        )

        self.iteration = 0
        self.best_cost = np.inf
        self.best_params = None
        self.last_energy_mev = 0.0
        self.last_n_turns = 0
        # Diagnostics from the most recent evaluation: whether tracking ran out
        # of clock before the requested turns were logged (see
        # TURN_BUDGET_MARGIN), and the per-block sums of squares of the
        # least-squares residual, so weight tuning can be observational.
        self.last_budget_exhausted = False
        self.last_resid_blocks = {}
        self.best_resid_blocks = {}
        self._checkpoint_inited = False

    # ------------------------------------------------------------------ utils
    def attach_bem_field(self,
                         build_kwargs: Optional[dict] = None,
                         solve_kwargs: Optional[dict] = None,
                         field_kwargs: Optional[dict] = None,
                         max_r_inner: Optional[float] = None,
                         voltage_profile=None):
        """Solve the BEM gap field for the CURRENT cavity geometry (bem2d).

        Builds the closed dee/ground electrodes from the design's RF gaps,
        solves the Laplace Dirichlet problem, grids the midplane E pattern and
        installs it as ``design.efield`` (a TimedField). Call this once after
        the geometry is final; the snapshot is NOT auto-invalidated by later
        ``update_geometry`` calls - re-attach after any geometry change.
        RF frequency and bunch phase stay free (re-synced before every run).

        ``max_r_inner`` (buildability guard): pass the beam injection radius
        [m]; the electrode build fails loudly if the auto inner truncation
        lands at or above it (the beam would cross gaps that have no
        electrodes there - no kick, fringe only).

        ``voltage_profile`` (radial dee-voltage shape from the RF cavity
        model, see ``gap_fields.VoltageProfile``): forwarded to the electrode
        build; the cavity voltage stays the peak voltage at the profile's
        reference radius. Same as ``build_kwargs['voltage_profile']``.

        Returns the TimedField; the full solution (surface charge, evaluators)
        is kept on ``self.bem_solution`` for diagnostics.
        """
        from .gap_fields import make_bem_efield
        if max_r_inner is not None:
            build_kwargs = {**(build_kwargs or {}), 'max_r_inner': max_r_inner}
        if voltage_profile is not None:
            build_kwargs = {**(build_kwargs or {}),
                            'voltage_profile': voltage_profile}
        timed, solution = make_bem_efield(
            self.design, build_kwargs=build_kwargs, solve_kwargs=solve_kwargs,
            field_kwargs=field_kwargs, verbose=self.verbose)
        self.design.set_electric_field(timed)
        self.bem_solution = solution
        return timed

    def _rf_base_frequency(self) -> float:
        """Base (orbital) frequency stored on the cavities [Hz]."""
        return self.design.rf_cavities[0].frequency

    def _estimate_timestep(self, frequency_hz: float) -> float:
        return (1.0 / frequency_hz) / self.steps_per_turn

    def _set_beam_meta(self, beam: ParticleDistribution, n_ref: int = 0):
        """Record the REAL particle count of a prepared beam.

        ``n_ref`` is the number of prepended virtual reference particles (0 or
        1, as returned by ``_prepare_beam``); they do not count as beam.
        """
        self._n_ref = int(n_ref)
        self.n_particles = int(beam.numpart) - self._n_ref
        self.is_multiparticle = self.n_particles > 1

    def _copy_beam(self, beam: ParticleDistribution) -> ParticleDistribution:
        return ParticleDistribution(species=self.design.species,
                                    x_vec=beam.x_vec.copy(), p_vec=beam.p_vec.copy())

    def _with_reference_particle(self, pd: ParticleDistribution) -> Tuple[ParticleDistribution, int]:
        """Prepend the bunch centroid as a VIRTUAL particle 0.

        The Poincare section, the reference trajectory and every per-turn orbit
        metric (r_avg, dr, r_center_h1, RF phase) are taken from this particle,
        so they describe the centroid orbit. The alternative - the running mean
        over SURVIVORS - steps discontinuously every time a particle is lost,
        putting that step straight into the centering and smoothness residuals.
        Having a real particle at index 0 also makes the section angle exact:
        it is that particle's own launch azimuth, so no centroid/particle-0
        mismatch can shift the first crossing.

        The reference is excluded from every beam statistic (envelope,
        survival, saved bunch) and from ``n_particles``. Single-particle beams
        get none - the one particle already is the reference. Momenta are
        averaged in p (not v); at central-region energies the difference is far
        below the bunch spread.
        """
        if not self.reference_centroid or int(pd.numpart) < 2:
            return pd, 0
        x = np.vstack([pd.x_vec.mean(axis=0), pd.x_vec])
        p = np.vstack([pd.p_vec.mean(axis=0), pd.p_vec])
        return ParticleDistribution(species=self.design.species,
                                    x_vec=x, p_vec=p), 1

    def _prepare_beam(self, initial_beam, r0=None, pr0=None,
                      r0_mode='offset') -> Tuple[ParticleDistribution, int]:
        """Working copy of ``initial_beam`` with optional r0/pr0 applied.

        offset   : r0 shifts the centroid radially [m], pr0 adds radial velocity [m/s].
        absolute : a single reference particle is launched at radius r0 on +x with
                   radial velocity pr0 and the supplied beam's mean speed.

        Returns ``(beam, n_ref)``; the virtual reference particle is prepended
        LAST, so it is the centroid of the beam as actually launched.
        """
        if r0 is None and pr0 is None:
            return self._with_reference_particle(self._copy_beam(initial_beam))

        if r0_mode == 'absolute':
            v_total = float(initial_beam.v_mean_m_per_s)
            cen = initial_beam.centroid
            r = r0 if r0 is not None else float(np.hypot(cen[0], cen[1]))
            vr = pr0 if pr0 is not None else 0.0
            return make_single_particle_beam(self.design.species, r, 0.0, vr,
                                             v_total=v_total), 0

        # offset mode
        pd = self._copy_beam(initial_beam)
        cen = pd.centroid
        rho = float(np.hypot(cen[0], cen[1]))
        r_hat = (np.array([cen[0] / rho, cen[1] / rho, 0.0]) if rho > 0
                 else np.array([1.0, 0.0, 0.0]))
        if r0:
            new_cen = cen + r0 * r_hat
            pd.set_centroid(float(new_cen[0]), float(new_cen[1]), float(cen[2]))
        if pr0:
            bg = pr0 / np.sqrt(CLIGHT ** 2 - pr0 ** 2)
            pd.add_mean_momentum(float(bg * r_hat[0]), float(bg * r_hat[1]), 0.0)
        return self._with_reference_particle(pd)

    @staticmethod
    def _unpack(params, optimize_params) -> Dict[str, float]:
        # canonical order must match _build_param_space's append order
        order = ['bunch_phase', 'rf_freq', 'r0', 'vr0',
                 'coll_azimuth', 'coll_aperture']
        names = [n for n in order if n in optimize_params]
        return dict(zip(names, np.asarray(params, dtype=float)))

    # ------------------------------------------------------------- checkpoint
    def _maybe_init_checkpoint(self):
        if not self.checkpoint_file or self._checkpoint_inited:
            return
        with open(self.checkpoint_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['iteration', 'bunch_phase_deg', 'rf_freq_mhz', 'r0', 'vr0',
                      'final_energy_mev', 'n_turns', 'cost', 'success', 'timestamp']
            if self.is_multiparticle:
                header.extend(['final_std_r_mm', 'envelope_oscillation_mm'])
            writer.writerow(header)
        self._checkpoint_inited = True

    def _write_checkpoint(self, vals, cost, energy, n_turns, success,
                          std_r_mm=None, envelope_osc_mm=None):
        if not self.checkpoint_file:
            return
        with open(self.checkpoint_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [self.iteration, vals.get('bunch_phase', 0.0),
                   vals.get('rf_freq', 0.0) / 1e6, vals.get('r0', 0.0),
                   vals.get('vr0', 0.0), energy, n_turns, cost, success, time.time()]
            if self.is_multiparticle:
                row.extend([std_r_mm or 0.0, envelope_osc_mm or 0.0])
            writer.writerow(row)

    # --------------------------------------------------------------- tracking
    def _make_collimator(self, vals):
        """Central-region phase slit from optimization values
        ('coll_azimuth' [deg], 'coll_aperture' [mm]) with fallback to
        the fixed ``self.collimator`` dict (azimuth_deg / aperture_mm /
        ref_particle). Reference-centered: the aperture centers itself
        on the reference particle's first-crossing radius, so it stays
        valid for any candidate RF/geometry. Returns None if unset."""
        cfg = getattr(self, 'collimator', None) or {}
        az = vals.get('coll_azimuth', cfg.get('azimuth_deg'))
        ap = vals.get('coll_aperture', cfg.get('aperture_mm'))
        if az is None or ap is None:
            return None
        from .tracking import RadialSlitCollimator
        # exempt_ref: with a virtual reference particle at index 0 the slit must
        # never remove it - losing the reference would end turn counting. In
        # self-centering mode it sits exactly at the aperture centre and cannot
        # be hit anyway; the flag also covers a fixed r_lo/r_hi aperture.
        return RadialSlitCollimator(
            np.radians(float(az)), aperture_m=float(ap) * 1e-3,
            ref_particle=int(cfg.get('ref_particle', 0)),
            exempt_ref=bool(self._n_ref), index_offset=self._n_ref)

    def track_with_rf(self,
                      pd_init: ParticleDistribution,
                      dt: float,
                      max_turns: int,
                      save_full_beam: bool = False,
                      section_angle: Optional[float] = None) -> Tuple:
        """Track particle(s) with RF and collect diagnostics (single or multi).

        ``section_angle`` [rad] fixes the Poincare section. The default (None)
        puts it on the reference particle's LAUNCH azimuth, so ``turn N`` means
        N genuine revolutions from injection whatever the injection geometry -
        with a hardcoded section at 0 the meaning of a turn silently depended on
        where injection sat relative to the +x axis, making dr, phase and
        centering non-comparable between runs. Pass 0.0 to force the +x axis
        (SEO comparison).

        ``pd_init`` is expected to come from ``_prepare_beam``; if it carries a
        virtual reference particle (``self._n_ref == 1``) that particle drives
        the Poincare section and the reference trajectory, and is excluded from
        the beam statistics and from ``full_beam``.
        """
        n_ref = self._n_ref
        self.last_budget_exhausted = False

        if section_angle is None:
            x0 = pd_init.x_vec[0]
            section_angle = float(np.arctan2(x0[1], x0[0]))
        # arm_angle=pi: the reference launches ON the section by construction,
        # so the first crossing must not be logged until it is half a
        # revolution away. Without arming, a launch a fraction of a step behind
        # the section registers a crossing within the first few steps and turn 0
        # spans almost no azimuth.
        poincare = PoincareAnalyzer(section_angle=section_angle, arm_angle=np.pi)
        beam_stats_collector = BeamStatisticsCollector(self.design.species, save_frequency=1)

        rf_crossings = []
        trajectory_storage = []
        std_r_storage = []
        turn_ids = []
        turn_counter = [0]
        energy_reached = [False]
        n_recorded = [0]

        # See TURN_BUDGET_MARGIN: the abort below fires the moment the last turn
        # is logged, so the margin is only ever consumed when actually needed.
        n_steps = (max_turns + TURN_BUDGET_MARGIN) * self.steps_per_turn
        full_beam = (np.full((n_steps, self.n_particles, 6), np.nan)
                     if save_full_beam else None)

        def callback(step, r_array, v_array, active, t):
            if not np.any(active):
                return False

            if n_ref:
                real = active.copy()
                real[:n_ref] = False
                if not np.any(real):
                    # Physical beam gone; the virtual reference alone would keep
                    # logging turns over an empty bunch.
                    return True
            else:
                real = active

            # Reference trajectory: the virtual centroid particle when present,
            # else the single particle. The remaining branch (multiparticle with
            # no reference) is the running mean over SURVIVORS, which steps
            # whenever a particle is lost.
            if n_ref or not self.is_multiparticle:
                trajectory_storage.append(r_array[0].copy())
            else:
                trajectory_storage.append(np.mean(r_array[real], axis=0))

            if self.is_multiparticle:
                radii = np.hypot(r_array[real, 0], r_array[real, 1])
                std_r_storage.append(float(np.std(radii)))
            else:
                std_r_storage.append(0.0)

            if full_beam is not None:
                full_beam[step, :, :3] = r_array[n_ref:]
                full_beam[step, :, 3:] = v_array[n_ref:]
            n_recorded[0] = step + 1

            if active[0]:
                # step 0 compares the post-step position with itself, so no
                # crossing can be logged there; that is what keeps a launch
                # exactly on the section from registering immediately.
                r_prev = r_array[0] if step == 0 else callback.r_prev
                crossed, t_frac = poincare.check_crossing(r_prev, r_array[0])

                if crossed:
                    turn_ids.append(step)
                    r_cross = r_prev + t_frac * (r_array[0] - r_prev) if t_frac else r_array[0]
                    v_cross = v_array[0]

                    cav = self.design.rf_cavities[0]
                    phase_rad = np.fmod(cav.omega * t + cav.get_total_phase_rad(), 2.0 * np.pi)
                    phase_deg = np.rad2deg(phase_rad)

                    poincare.record_crossing(
                        turn=turn_counter[0], r=r_cross, v=v_cross, time=t,
                        species=self.design.species, phase_deg=phase_deg,
                    )
                    beam_stats_collector.record(step, r_array[real], v_array[real], t)
                    beam_stats_collector.increment_turn()
                    turn_counter[0] += 1

                    if poincare.crossings[-1].energy_mev >= self.target_energy_mev:
                        energy_reached[0] = True
                        if self.verbose:
                            print(f"    Reached target energy at turn {turn_counter[0]}")
                        return True
                    if turn_counter[0] >= max_turns:
                        if self.verbose:
                            print(f"    Reached max turns: {turn_counter[0]}")
                        return True

                callback.r_prev = r_array[0].copy()
            return False

        try:
            result = self.engine.track_multiparticle(
                pd_init, dt=dt, n_steps=n_steps, callback=callback,
                callback_frequency=1, show_progress=False,
            )
        except Exception as e:
            if self.verbose:
                print(f"    Tracking exception: {e}")
            return (False, [], [], np.array([]), [[]],
                    np.array([]), [], None)

        # Distinguish "aborted on turns/energy" from "ran out of clock": the
        # tracker reports success either way, which is how a silently lost final
        # turn survived a 7272-evaluation optimization.
        self.last_budget_exhausted = (result.metadata.get('termination') is None
                                      and turn_counter[0] < max_turns)
        if self.last_budget_exhausted and self.verbose:
            print(f"    WARNING: step budget exhausted after {turn_counter[0]}/"
                  f"{max_turns} turns ({n_steps} steps) - raise "
                  f"TURN_BUDGET_MARGIN or check isochronism")

        trajectory_ref = np.array(trajectory_storage) if trajectory_storage else np.array([])
        std_r_per_step = np.array(std_r_storage)
        turn_statistics = beam_stats_collector.get_statistics()
        poincare_all = [list(poincare.crossings)]
        success = result.success or energy_reached[0]
        if full_beam is not None:
            full_beam = full_beam[:n_recorded[0]]

        return (success, turn_statistics, rf_crossings, trajectory_ref,
                poincare_all, std_r_per_step, turn_ids, full_beam)

    # -------------------------------------------------------------- objective
    def _default_weights(self, multi):
        # 'smooth' dropped 1000 -> 100 when the metric changed from
        # std(dr)**2 [m^2] to the dimensionless mean squared log-curvature
        # (see log_turn_curvature). The old weight was calibrated against a
        # term of order 1e-5; carrying it over would have made smoothness
        # dominate this cost by two orders of magnitude.
        if multi:
            return {'energy': 5.0, 'spread': 100.0, 'center': 1000.0, 'smooth': 100.0}
        return {'energy': 5.0, 'center': 1000.0, 'smooth': 100.0}

    def objective_function(self, params, initial_beam, dt, max_turns, weights,
                           optimize_params, r0_mode='offset'):
        """Unified objective (minimized). Adapts to single/multi-particle."""
        self.iteration += 1
        self.last_energy_mev = 0.0
        self.last_n_turns = 0

        vals = self._unpack(params, optimize_params)

        if 'bunch_phase' in vals:
            self.design.set_bunch_phase(vals['bunch_phase'])
        if 'rf_freq' in vals:
            self.design.set_rf_frequency(vals['rf_freq'])

        try:
            pd, n_ref = self._prepare_beam(initial_beam, vals.get('r0'),
                                           vals.get('vr0'), r0_mode)
        except Exception as e:
            if self.verbose:
                print(f"    Iter {self.iteration}: beam prep failed: {e}")
            self._write_checkpoint(vals, 1e10, 0.0, 0, False)
            return 1e10

        self._set_beam_meta(pd, n_ref)

        try:
            (success, turn_stats, rf_cross, traj_ref, poincare_all,
             std_r_steps, turn_ids, _) = self.track_with_rf(pd, dt, max_turns)
        except Exception as e:
            if self.verbose:
                print(f"    Iter {self.iteration}: tracking failed: {e}")
            self._write_checkpoint(vals, 1e10, 0.0, 0, False)
            return 1e10

        if not success or len(turn_stats) == 0:
            cost = 1e8
            self._write_checkpoint(vals, cost, 0.0, 0, False)
            return cost

        metrics = calculate_turn_metrics(traj_ref, turn_ids)
        final_energy = turn_stats[-1].mean_energy_mev
        n_turns = len(turn_stats)
        self.last_energy_mev = final_energy
        self.last_n_turns = n_turns

        # Cost (minimized). 5c: bounded target-distance term instead of unbounded -energy.
        w_energy = weights.get('energy', 5.0)
        cost = w_energy * max(0.0, self.target_energy_mev - final_energy)

        w_center = weights.get('center', 1000.0)
        if len(metrics['r_center']) > 0:
            cost += w_center * np.mean(metrics['r_center'])

        # Turn-separation smoothness: mean squared log-curvature (scale-free,
        # so it does not penalize a larger energy gain). ``mean`` rather than
        # ``sum`` keeps the term independent of max_turns, matching the other
        # scalar blocks.
        w_smooth = weights.get('smooth', 100.0)
        curv = log_turn_curvature(metrics['dr'])
        if len(curv) > 0:
            cost += w_smooth * float(np.mean(_soft_clip(curv / U_SCALE) ** 2))

        if self.is_multiparticle:
            envelope_osc = float(np.std(std_r_steps))
            cost += weights.get('spread', 100.0) * envelope_osc
            final_std_r = turn_stats[-1].std_r
        else:
            envelope_osc = 0.0
            final_std_r = 0.0

        self._write_checkpoint(vals, cost, final_energy, n_turns, True,
                               final_std_r * 1000, envelope_osc * 1000)

        if cost < self.best_cost:
            self.best_cost = cost
            self.best_params = np.asarray(params, dtype=float).copy()
            if self.verbose:
                msg = (f"    Iter {self.iteration}: NEW BEST cost={cost:.3e}, "
                       f"E={final_energy:.3f} MeV, turns={n_turns}")
                print(msg)

        return cost

    def objective_residuals(self, params, initial_beam, dt, max_turns, ls_weights,
                            optimize_params, r0_mode='offset',
                            skip_turns: int = DEFAULT_SKIP_TURNS) -> np.ndarray:
        """Residual VECTOR for least-squares optimizers (DFO-LS).

        Fixed length m = max_turns + (max_turns - skip) + (max_turns - 3 - skip),
        independent of how many turns the particle survives (DFO-LS requires
        constant m):
          - energy:  sqrt(w_e) * (E_ramp_i - E_i) / E_target  per turn (ALL
                     turns), where E_ramp is a linear ramp from the beam energy
                     to the target (rewards steady acceleration, not just the
                     endpoint); turns not reached continue with the last
                     achieved energy, so early loss degrades smoothly instead
                     of a cost cliff.
          - center:  sqrt(w_c) * r_center_i / 0.02 m           per turn.
          - smooth:  sqrt(w_s) * clip(d2[log dr]_i / U_SCALE)  per kept turn
                     triple - the second difference of log(turn separation),
                     softly saturated at U_RESID_MAX, see log_turn_curvature
                     and _soft_clip. This replaced a
                     deviation-of-dr-from-its-mean term, which had an
                     irreducible floor (r ~ sqrt(E) forces dr to shrink) that
                     grew with the energy gain, so crest-riding was net
                     penalized.
          - phase:   sqrt(w_p) * (phi_i - mean(phi)) / 15 deg  per kept turn,
                     phi = unwrapped RF phase at the Poincare crossing -
                     penalizes synchronous-phase walk (phase slip).

        ``skip_turns`` exempts the first n turns from the CENTERING, SMOOTHNESS,
        PHASE and ENVELOPE residuals (energy and survival always count): the
        first 1-2 turns are inherently lopsided due to the modified gap geometry
        and the off-orbit injection conditions, and should not be penalized.
        The smoothness/phase means are taken over the kept turns only.

        For multiparticle beams (numpart > 1) two extra blocks are appended:
          - envelope:  sqrt(w_v) * std_r_i / 0.005 m       per kept turn.
          - survival:  sqrt(w_u) * lost_fraction_i         per turn (no skip).
        """
        self.iteration += 1
        self.last_energy_mev = 0.0
        self.last_n_turns = 0

        we = np.sqrt(ls_weights.get('energy', DEFAULT_LS_WEIGHTS['energy']))
        wc = np.sqrt(ls_weights.get('center', DEFAULT_LS_WEIGHTS['center']))
        ws = np.sqrt(ls_weights.get('smooth', DEFAULT_LS_WEIGHTS['smooth']))
        wv = np.sqrt(ls_weights.get('envelope', DEFAULT_LS_WEIGHTS['envelope']))
        wu = np.sqrt(ls_weights.get('survival', DEFAULT_LS_WEIGHTS['survival']))
        wp = np.sqrt(ls_weights.get('phase', DEFAULT_LS_WEIGHTS['phase']))
        C_SCALE, ENV_SCALE, PHASE_SCALE = 0.02, 0.005, 15.0
        skip = max(0, min(int(skip_turns), max_turns))
        n_s = max(max_turns - 1 - skip, 0)       # kept turn separations
        n_c = max(n_s - 2, 0)                    # their second differences
        n_p = max(max_turns - skip, 0)
        is_multi = int(initial_beam.numpart) > 1
        n0 = max(int(initial_beam.numpart), 1)

        e0 = float(initial_beam.mean_energy_mev)
        ramp = e0 + (self.target_energy_mev - e0) * (np.arange(1, max_turns + 1) / max_turns)

        # Defaults = "no acceleration at all", and the graceful failure vector
        # for a candidate whose tracking throws: one residual unit per entry
        # (0.99 for smoothness, which passes through _soft_clip), matching the
        # existing centering convention. Previously smoothness and
        # phase defaulted to zeros, i.e. a candidate that failed to track scored
        # a PERFECT zero on both.
        # These are failure values only - turns that are merely not reached are
        # padded by freezing at the last achieved value further down, because
        # charging for them would reinstate exactly the lost-turn discontinuity
        # TURN_BUDGET_MARGIN exists to remove.
        e_turns = np.full(max_turns, e0)
        c_turns = np.full(max_turns, C_SCALE)
        curv_kept = np.full(n_c, U_SCALE)
        p_kept = np.full(n_p, PHASE_SCALE)
        env_turns = np.full(max_turns, ENV_SCALE)
        surv_turns = np.ones(max_turns)          # default: everything lost

        vals = self._unpack(params, optimize_params)
        if 'bunch_phase' in vals:
            self.design.set_bunch_phase(vals['bunch_phase'])
        if 'rf_freq' in vals:
            self.design.set_rf_frequency(vals['rf_freq'])

        coll = None
        try:
            # The beam is prepared first so _make_collimator knows whether a
            # virtual reference particle is present (it must not be collimated).
            pd, n_ref = self._prepare_beam(initial_beam, vals.get('r0'),
                                           vals.get('vr0'), r0_mode)
            self._set_beam_meta(pd, n_ref)
            coll = self._make_collimator(vals)
            self.engine.extra_terminators = [coll] if coll is not None else []
            (success, turn_stats, _, traj_ref, poincare_all,
             _, turn_ids, _) = self.track_with_rf(pd, dt, max_turns)

            n_turns = len(turn_stats)
            if n_turns > 0:
                energies = np.array([t.mean_energy_mev for t in turn_stats])
                e_turns[:min(n_turns, max_turns)] = energies[:max_turns]
                if n_turns < max_turns:
                    e_turns[n_turns:] = energies[-1]   # freeze at last achieved

                metrics = calculate_turn_metrics(traj_ref, turn_ids)
                # Spiral-corrected first-harmonic center offset: the centroid
                # metric is contaminated by ~dr/2pi per turn for accelerated
                # orbits, which the optimizer cannot drive to zero.
                rc = metrics['r_center_h1']
                if len(rc) > 0:
                    c_turns[:min(len(rc), max_turns)] = rc[:max_turns]
                    if len(rc) < max_turns:
                        c_turns[len(rc):] = rc[-1]
                # Turn separation: penalize CURVATURE of log(dr), not deviation
                # from the mean (see log_turn_curvature). Turns not reached
                # freeze at the last curvature, as everything else does.
                if n_c > 0:
                    curv = log_turn_curvature(metrics['dr'], skip)
                    if len(curv) > 0:
                        m_c = min(len(curv), n_c)
                        curv_kept[:m_c] = curv[:n_c]
                        if m_c < n_c:
                            curv_kept[m_c:] = curv[m_c - 1]

                # Phase stability: unwrapped RF phase at the Poincare crossing
                # per turn; deviation from the mean over kept turns. Turns not
                # reached freeze at the last deviation (smooth degradation).
                phases = np.array([c.phase_deg for c in poincare_all[0]
                                   if c.phase_deg is not None], dtype=float)
                if len(phases) > skip and n_p > 0:
                    ph = np.unwrap(phases, period=360.0)[skip:]
                    dev_p = ph - np.mean(ph)
                    m_p = min(len(dev_p), n_p)
                    p_kept[:m_p] = dev_p[:n_p]
                    if m_p < n_p:
                        p_kept[m_p:] = dev_p[-1]

                if is_multi:
                    stds = np.array([t.std_r for t in turn_stats])
                    env_turns[:min(n_turns, max_turns)] = stds[:max_turns]
                    if n_turns < max_turns:
                        env_turns[n_turns:] = stds[-1]
                    # dynamic survival against the POST-COLLIMATOR
                    # population (intentional slit removal is priced
                    # once, in the 'collimated' block below)
                    n_coll = len(coll.hits) if coll is not None else 0
                    n_eff = max(n0 - n_coll, 1)
                    frac_lost = np.clip(
                        1.0 - np.array([t.n_active for t in turn_stats],
                                       dtype=float) / n_eff, 0.0, 1.0)
                    surv_turns[:min(n_turns, max_turns)] = frac_lost[:max_turns]
                    if n_turns < max_turns:
                        # pad with the last observed loss (early stop on target
                        # energy must not read as "everything lost")
                        surv_turns[n_turns:] = frac_lost[-1]

                self.last_energy_mev = float(energies[-1])
                self.last_n_turns = n_turns
        except Exception as e:
            if self.verbose:
                print(f"    Iter {self.iteration}: residual eval failed: {e}")
        finally:
            self.engine.extra_terminators = []

        # Insertion order IS the residual layout; the dict additionally gives a
        # per-block cost decomposition (see self.last_resid_blocks), which is
        # what the weights should be tuned against.
        blocks = {
            'energy': we * (ramp - e_turns) / self.target_energy_mev,
            'center': wc * c_turns[skip:] / C_SCALE,
            'smooth': ws * _soft_clip(curv_kept / U_SCALE),
            'phase': wp * p_kept / PHASE_SCALE,
        }
        if is_multi:
            blocks['envelope'] = wv * env_turns[skip:] / ENV_SCALE
            blocks['survival'] = wu * surv_turns
            if coll is not None:
                wk = np.sqrt(ls_weights.get(
                    'collimated', DEFAULT_LS_WEIGHTS['collimated']))
                blocks['collimated'] = np.array([wk * len(coll.hits) / n0])
        resid = np.concatenate(list(blocks.values()))
        self.last_resid_blocks = {k: float(np.sum(v ** 2)) for k, v in blocks.items()}

        cost = float(np.sum(resid ** 2))
        self._write_checkpoint(vals, cost, self.last_energy_mev, self.last_n_turns,
                               self.last_n_turns > 0)
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_params = np.asarray(params, dtype=float).copy()
            self.best_resid_blocks = dict(self.last_resid_blocks)
            if self.verbose:
                shares = ", ".join(f"{k} {v:.3f}" for k, v in
                                   self.last_resid_blocks.items())
                print(f"    Iter {self.iteration}: NEW BEST ||r||^2={cost:.3e}, "
                      f"E={self.last_energy_mev:.3f} MeV, turns={self.last_n_turns}"
                      f"\n        blocks: {shares}")
        return resid

    # --------------------------------------------------------------- optimize
    def _build_param_space(self, initial_beam, optimize_params, bounds, r0_mode):
        param_bounds, param_names, x0 = [], [], []
        cen = initial_beam.centroid
        rho = float(np.hypot(cen[0], cen[1]))

        if 'bunch_phase' in optimize_params:
            param_names.append('bunch_phase')
            param_bounds.append(bounds.get('bunch_phase', (-180, 180)))
            x0.append(20.0)
        if 'rf_freq' in optimize_params:
            f0 = self._rf_base_frequency()
            param_names.append('rf_freq')
            param_bounds.append(bounds.get('rf_freq', (f0 * 0.95, f0 * 1.05)))
            x0.append(f0)
        if 'r0' in optimize_params:
            param_names.append('r0')
            if r0_mode == 'absolute':
                param_bounds.append(bounds.get('r0', (rho - 0.010, rho + 0.010)))
                x0.append(rho)
            else:
                param_bounds.append(bounds.get('r0', (-0.010, 0.010)))
                x0.append(0.0)
        if 'vr0' in optimize_params:
            param_names.append('vr0')
            param_bounds.append(bounds.get('vr0', (-5e5, 5e5)))
            x0.append(0.0)
        if 'coll_azimuth' in optimize_params:
            b = bounds.get('coll_azimuth', (0.0, 360.0))
            param_names.append('coll_azimuth')
            param_bounds.append(b)
            x0.append(0.5 * (b[0] + b[1]))
        if 'coll_aperture' in optimize_params:
            b = bounds.get('coll_aperture', (2.0, 20.0))
            param_names.append('coll_aperture')
            param_bounds.append(b)
            x0.append(b[1])          # start wide open (no collimation)
        return param_bounds, param_names, x0

    def optimize(self,
                 initial_beam: ParticleDistribution,
                 max_turns: int = 500,
                 optimize_params: List[str] = ['bunch_phase', 'rf_freq'],
                 method: str = 'differential_evolution',
                 bounds: Optional[dict] = None,
                 weights: Optional[dict] = None,
                 maxiter: int = 100,
                 r0_mode: str = 'offset') -> OptimizedOrbit:
        """Optimize RF (and optional r0/pr0) for acceleration of ``initial_beam``."""
        self._set_beam_meta(initial_beam)
        effective_multi = self.is_multiparticle and r0_mode != 'absolute'

        if self.verbose:
            print("\n" + "=" * 70)
            print(f"ACCELERATED ORBIT OPTIMIZATION "
                  f"({'MULTI' if effective_multi else 'SINGLE'}-PARTICLE)")
            print("=" * 70)
            print(f"Target energy: {self.target_energy_mev} MeV, beam numpart={self.n_particles}")
            print(f"Optimizing: {optimize_params}  (r0_mode={r0_mode}, method={method})")

        if weights is None:
            weights = self._default_weights(effective_multi)
        if bounds is None:
            bounds = {}

        dt = self._estimate_timestep(self._rf_base_frequency())
        param_bounds, param_names, x0 = self._build_param_space(
            initial_beam, optimize_params, bounds, r0_mode)

        self.iteration = 0
        self.best_cost = np.inf
        self.best_params = None
        self._maybe_init_checkpoint()

        args = (initial_beam, dt, max_turns, weights, optimize_params, r0_mode)
        start = time.time()
        if method == 'differential_evolution':
            res = differential_evolution(self.objective_function, param_bounds, args=args,
                                         maxiter=maxiter, workers=1, updating='deferred',
                                         disp=False)
            optimal = res.x
            final_cost = res.fun
        elif method == 'nelder_mead':
            res = minimize(self.objective_function, x0, args=args, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'disp': False})
            optimal = res.x
            final_cost = res.fun
        else:
            raise ValueError(f"Unknown method: {method}")
        elapsed = time.time() - start

        vals = self._unpack(optimal, optimize_params)
        if 'bunch_phase' in vals:
            self.design.set_bunch_phase(vals['bunch_phase'])
        if 'rf_freq' in vals:
            self.design.set_rf_frequency(vals['rf_freq'])

        if self.verbose:
            print(f"\nOptimization complete in {elapsed:.1f}s, "
                  f"{self.iteration} iters, final cost {final_cost:.3e}")

        # Final tracking with optimal parameters.
        pd_final, n_ref = self._prepare_beam(initial_beam, vals.get('r0'),
                                             vals.get('vr0'), r0_mode)
        self._set_beam_meta(pd_final, n_ref)
        result = self.track_with_rf(pd_final, dt, max_turns, save_full_beam=True)
        return self._build_result(result, vals, final_cost, param_names, param_bounds,
                                  weights, method, elapsed, r0_mode,
                                  metadata_extra={'optimization_time_s': elapsed,
                                                  'total_iterations': self.iteration})

    def track_once(self,
                   initial_beam: ParticleDistribution,
                   bunch_phase_deg: float,
                   rf_freq_mhz: float,
                   max_turns: int = 500,
                   r0: Optional[float] = None,
                   pr0: Optional[float] = None,
                   r0_mode: str = 'offset',
                   save_full_beam: bool = False) -> OptimizedOrbit:
        """Single deterministic tracking run (no optimization) of ``initial_beam``."""
        self.design.set_bunch_phase(bunch_phase_deg)
        self.design.set_rf_frequency(rf_freq_mhz * 1e6)

        pd, n_ref = self._prepare_beam(initial_beam, r0, pr0, r0_mode)
        self._set_beam_meta(pd, n_ref)
        dt = self._estimate_timestep(self._rf_base_frequency())

        # fixed-config collimator (self.collimator), if any
        coll = self._make_collimator({})
        self.engine.extra_terminators = [coll] if coll is not None else []
        try:
            result = self.track_with_rf(pd, dt, max_turns,
                                        save_full_beam=save_full_beam)
        finally:
            self.engine.extra_terminators = []
        vals = {'bunch_phase': bunch_phase_deg, 'rf_freq': rf_freq_mhz * 1e6}
        if r0 is not None:
            vals['r0'] = r0
        if pr0 is not None:
            vals['vr0'] = pr0
        meta_extra = {'mode': 'single_run'}
        if coll is not None:
            meta_extra['collimator'] = {
                'azimuth_deg': float(np.degrees(coll.azimuth_rad)),
                'aperture_mm': float(coll.aperture_m * 1e3),
                'r_center_mm': (float(coll.r_center_m * 1e3)
                                if coll.r_center_m is not None else None),
                'n_collimated': len(coll.hits)}
        return self._build_result(result, vals, 0.0, list(vals.keys()), [], {},
                                  'single_run', 0.0, r0_mode,
                                  metadata_extra=meta_extra)

    # ----------------------------------------------------------------- result
    def _build_result(self, result, vals, cost, param_names, param_bounds, weights,
                      method, elapsed, r0_mode, metadata_extra=None) -> OptimizedOrbit:
        (success, turn_stats, rf_cross, traj_ref, poincare_all,
         std_r_steps, turn_ids, full_beam) = result
        metrics = calculate_turn_metrics(traj_ref, turn_ids)
        final_energy = turn_stats[-1].mean_energy_mev if len(turn_stats) > 0 else 0.0

        initial_r_mm = 0.0
        if len(traj_ref) > 0:
            initial_r_mm = float(np.hypot(traj_ref[0][0], traj_ref[0][1]) * 1000)

        meta = {
            'param_names': param_names,
            'param_bounds': param_bounds,
            'weights': weights,
            'optimization_method': method,
            'n_particles': self.n_particles,
            'r0_mode': r0_mode,
            'full_beam': full_beam,
            'envelope_oscillation_mm': float(np.std(std_r_steps) * 1000) if self.is_multiparticle else 0.0,
            # Virtual reference particle (if any) is already stripped from
            # full_beam and n_particles; recorded so downstream plots know the
            # trajectory is the centroid orbit rather than a survivor mean.
            'reference_centroid': bool(self._n_ref),
            'budget_exhausted': bool(self.last_budget_exhausted),
            'resid_blocks_best': dict(self.best_resid_blocks),
        }
        if metadata_extra:
            meta.update(metadata_extra)

        return OptimizedOrbit(
            success=success,
            final_energy_mev=final_energy,
            n_turns=len(turn_stats),
            n_particles=self.n_particles,
            bunch_phase_deg=vals.get('bunch_phase', 0.0),
            rf_frequency_mhz=vals.get('rf_freq', self._rf_base_frequency()) / 1e6,
            initial_r_mm=initial_r_mm,
            initial_vr_m_s=vals.get('vr0', 0.0),
            trajectory_reference=traj_ref,
            poincare_points_all=poincare_all,
            rf_crossings=rf_cross,
            turn_statistics=turn_stats,
            turn_metrics=metrics,
            std_r_per_step=std_r_steps,
            cost=cost,
            metadata=meta,
        )
