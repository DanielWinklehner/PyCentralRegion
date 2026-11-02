"""PyCentralRegion: central-region design for compact cyclotrons (orbits, RF gaps, 3D electrodes, BEM fields,
bunch tracking with space charge, beam-quality scoring and multi-objective optimisation).

Importing the package is cheap. The physics modules (numba, PyPATools, bempp, gmsh) load the first time one of their
names is used, so the configuration, analysis and optimisation layers (``config``, ``analysis``, ``mobo``) also work
in environments without the physics stack. ``from PyCentralRegion import X`` behaves as before.
"""
import importlib

# public name -> submodule that defines it
_EXPORTS = {
    # seo_finder
    'SEOFinder': 'seo_finder', 'StaticOrbit': 'seo_finder', 'PoincarePoint': 'seo_finder',
    'save_seo_database': 'seo_finder', 'load_seo_database': 'seo_finder', 'analyze_isochronism': 'seo_finder',
    # central_region
    'CentralRegion': 'central_region',
    # rf_cavity
    'RFCavity': 'rf_cavity', 'DeeSystem': 'rf_cavity', 'create_dee_system': 'rf_cavity',
    'create_double_gap_cavity': 'rf_cavity', 'create_four_cavity_system': 'rf_cavity',
    'snap_nodes_between_turns': 'rf_cavity', 'check_variable_segments': 'rf_cavity',
    # tracking
    'TrackingEngine': 'tracking', 'TrackingResult': 'tracking', 'track_single_particle': 'tracking',
    'RadialSlitCollimator': 'tracking', 'MetalTerminator': 'tracking', 'TimedRelease': 'tracking',
    # handoff (spiral-inflector hand-off bunches, staggered launch)
    'load_handoff': 'handoff', 'make_beam_from_handoff': 'handoff',
    # accelerated_orbit_finder
    'AcceleratedOrbitFinder': 'accelerated_orbit_finder', 'OptimizedOrbit': 'accelerated_orbit_finder',
    'make_beam_from_state': 'accelerated_orbit_finder', 'make_single_particle_beam': 'accelerated_orbit_finder',
    'make_gaussian_beam': 'accelerated_orbit_finder', 'make_beam_from_cylindrical': 'accelerated_orbit_finder',
    # cavity_optimizer
    'CavityGeometryOptimizer': 'cavity_optimizer',
    # gap_fields (radial dee-voltage shape for the BEM Dirichlet data)
    'VoltageProfile': 'gap_fields',
    # electrodes3d (3D electrode solids, bars, obstacle / clearance checks)
    'HeightProfiles': 'electrodes3d', 'Bar': 'electrodes3d', 'Post': 'electrodes3d', 'Patch': 'electrodes3d',
    'ExtraSolid': 'electrodes3d', 'build_electrodes_3d': 'electrodes3d', 'auto_bars': 'electrodes3d',
    'bar_clearances': 'electrodes3d', 'midplane_obstacles': 'electrodes3d', 'raster_obstacles': 'electrodes3d',
    'combine_obstacles': 'electrodes3d', 'check_trajectory_clearance': 'electrodes3d',
    # inflector (static field + housing + bunch of the spiral inflector in the CR model)
    'InflectorModel': 'inflector', 'load_inflector_field': 'inflector', 'midplane_field': 'inflector',
    'housing_mesh': 'inflector', 'housing_section': 'inflector', 'StaticPlusRFField': 'inflector',
    'superpose_efield': 'inflector',
    # closed_orbit (Gordon polar closed-orbit solver)
    'CartesianMidplane': 'closed_orbit', 'PolarMidplane': 'closed_orbit', 'closed_orbits': 'closed_orbit',
    'orbits_at_radii': 'closed_orbit',
}
_SUBMODULES = ('accelerated_orbit_finder', 'analysis', 'btrim', 'cavity_optimizer', 'central_region', 'cli',
               'closed_orbit', 'config', 'diagnostics', 'electrodes3d', 'fast_track', 'fast_track3d', 'fields3d',
               'frames', 'gap_fields', 'geometry', 'handoff', 'inflector', 'kinematics', 'machine', 'mobo',
               'polyprism', 'probes', 'rf_cavity', 'seo_finder', 'simulate3d', 'tracking')
# the package used to star-import these two modules: their other public names stay reachable
_STAR_MODULES = ('seo_finder', 'central_region')

__all__ = list(_EXPORTS) + ['diagnostics', 'gap_fields', 'closed_orbit', 'electrodes3d', 'polyprism', 'inflector',
                            'handoff']


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(importlib.import_module(f'.{_EXPORTS[name]}', __name__), name)
    elif name in _SUBMODULES:
        value = importlib.import_module(f'.{name}', __name__)
    elif not name.startswith('_'):
        for mod in _STAR_MODULES:
            module = importlib.import_module(f'.{mod}', __name__)
            if hasattr(module, name):
                value = getattr(module, name)
                break
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS) | set(_SUBMODULES))
