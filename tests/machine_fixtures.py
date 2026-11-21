"""Machines for the tests. [machine] has no defaults (every run sets the machine's properties), so a test that needs a
machine takes one of these:

``four_dee_machine(**changes)``
    a synthetic four-dee machine (h = 4, radial dees at 45 + 90 k deg, 42 deg wide, gap lines from 5 to 370 mm): the
    dimensions the builder and analysis unit tests derive their expected numbers from;
``demo_machine()``
    the worked example's machine, configs/examples/demo/machine.toml.
"""
from pathlib import Path

from PyCentralRegion import config as C

DEMO_DIR = Path(__file__).resolve().parents[1] / 'configs' / 'examples' / 'demo'

FOUR_DEE = dict(species='H2_1+', harmonic=4, n_dees=4, dee_shape='radial',
                dee_center_angles_deg=(45.0, 135.0, 225.0, 315.0), dee_opening_deg=42.0, dee_voltage_kv=65.0,
                design_voltage_kv=60.0, gap_width_m=0.01, gap_angle_deg=2.5, gap_width_min_m=6.5e-3, r_min_m=0.005,
                r_max_cavity_m=0.37, r_max_m=0.38, z_max_m=0.030, target_energy_mev=3.0, n_turns_design=6,
                skip_turns_design=1)


def four_dee_machine(**changes):
    return C.MachineConfig(**{**FOUR_DEE, **changes})


def demo_machine():
    return C.section(C.load(DEMO_DIR / 'machine.toml'), 'machine')
