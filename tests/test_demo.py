"""The worked example (configs/examples/demo, PyCentralRegion.demo): its run files resolve without anything of this
computer's, its inputs are generated as the pipeline reads them, and the whole chain (build -> fields -> track -> score)
runs on them: a reduced bunch (one phase, 1000 particles, space charge on), about 45 s with the inputs on a desktop
CPU. The demo needs gmsh, PyPATools and bempp; without them the tests that build skip."""
import dataclasses
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from PyCentralRegion import config as C
from PyCentralRegion.geometry import options as O

from machine_fixtures import DEMO_DIR

DESIGN = DEMO_DIR / 'design.toml'


def test_demo_run_files():
    """The demo's machine, design and campaign files: every [machine] key set, paths under the example's own folders
    (no root, no drive), the electrode options inside their bounds for the generated housing's 5 mm exit half height."""
    cfg = C.load(DESIGN)
    mc = C.section(cfg, 'machine')
    assert (mc.species, mc.harmonic, mc.n_dees, mc.dee_opening_deg) == ('proton', 4, 4, 40.0)
    pc = C.section(cfg, 'paths')
    for key in ('bfield_3d', 'seed', 'handoff', 'inflector_efield', 'housing_step', 'housing_outline'):
        assert Path(getattr(pc, key)).parent == DEMO_DIR / 'inputs', key
    out = DEMO_DIR.parents[2] / 'output' / 'demo'
    assert Path(pc.model3d).parent == out and Path(C.section(cfg, 'run').out_dir) == out
    g = C.section(cfg, 'geometry')
    bounds = O.check_bounds(g, exit_half_height_mm=5.0, exit_r_mm=34.0)
    assert bounds['height_max_mm'] == 16.0 and g.aperture_inner_mm >= bounds['inner_min_mm']
    assert C.section(cfg, 'score').pick_e_min_mev == 0.75 and C.section(cfg, 'track').sc_gpu is False


@pytest.fixture(scope='module')
def demo_inputs(tmp_path_factory):
    """The demo's inputs, generated into a temporary folder with a shorter phase scan for the seed."""
    pytest.importorskip('PyPATools')
    pytest.importorskip('gmsh')
    pytest.importorskip('h5py')
    from PyCentralRegion import demo
    folder = tmp_path_factory.mktemp('demo_inputs')
    spec = dataclasses.replace(demo.DemoSpec(), phases_deg=(20.0, 25.0, 30.0))
    files = demo.generate(folder, spec=spec, verbose=False)
    return folder, files, spec


def test_demo_inputs(demo_inputs):
    """The six inputs in the layouts the pipeline reads: a COMSOL text map (negative B_z, a z = 0 plane), the hand-off
    (openPMD plane crossing, the machine's species and the field's revolution frequency x h), the seed (its optimal
    geometry and a counter-clockwise reference orbit that ends near the target energy), the housing outline with its
    aperture, a STEP solid and a zero E-field."""
    import h5py
    from PyCentralRegion import demo
    folder, files, spec = demo_inputs
    assert sorted(p.name for p in folder.iterdir()) == sorted(demo.FILES.values())
    mc = demo.machine()
    with open(files['bfield_3d']) as fh:
        head = [next(fh) for _ in range(8)]
    assert head[0].startswith('% Model:') and head[6].startswith('% x y z')
    rows = np.loadtxt(files['bfield_3d'], comments='%')
    plane = rows[rows[:, 2] == 0.0]
    assert len(plane) and np.all(plane[:, 3:5] == 0.0)                           # no B_r on the median plane
    near = plane[np.argmin(np.hypot(plane[:, 0], plane[:, 1]))]
    assert near[5] == pytest.approx(-spec.b0_t, rel=1e-4)                       # negative: counter-clockwise ions
    with h5py.File(files['handoff'], 'r') as f:
        g = f['particles/proton']
        assert g.attrs['PyPATools:mode'] == 'plane_crossing' and len(g['position/x']) == spec.n_particles
        f_rf = float(g.attrs['PyPATools:rf_frequency_hz'])
    assert f_rf == pytest.approx(mc.harmonic * demo.f_rev_hz(mc, spec), rel=1e-12)
    with open(files['seed'], 'rb') as fh:
        seed = pickle.load(fh)
    meta = seed.metadata
    assert 'optimal_geometry' in meta and meta['demo']['phase_deg'] in spec.phases_deg
    ref = np.asarray(seed.trajectory_reference)[:, :2]
    turn = np.unwrap(np.arctan2(ref[:, 1], ref[:, 0]))
    assert turn[-1] - turn[0] > 2 * np.pi * (mc.n_turns_design - 1)      # counter-clockwise, every design turn
    text = Path(files['housing_outline']).read_text()
    assert 'units mm' in text and '\naperture 4\n' in text
    assert Path(files['housing_step']).stat().st_size > 0 and Path(files['inflector_efield']).stat().st_size > 0


def test_demo_chain(demo_inputs, tmp_path):
    """build -> fields -> track -> score on the generated inputs, one phase and 1000 particles with space charge: most
    of the bunch reaches the target energy (the whole demo at its own settings: T 91.6 % = 0.916 mA of 1.000 mA,
    E 0.813 +- 0.023 MeV, 2026-11-21)."""
    pytest.importorskip('bempp_cl')
    from PyCentralRegion import cli
    folder, files, spec = demo_inputs
    sets = [f'paths.{k}={C.toml_value(str(v))}' for k, v in files.items()]
    sets += [f'paths.model3d={C.toml_value(str(tmp_path / "model3d.pkl"))}',
             f'paths.fields={C.toml_value(str(tmp_path / "fields.pkl"))}',
             f'run.cache_dir={C.toml_value(str(tmp_path / "cache"))}',
             'track.phases_deg=[35.0]', 'track.n_scan=300', 'track.n_final=1000', 'track.n_tracks=1000',
             'score.min_tracks=500', 'geometry.plot_dees=false']
    out = tmp_path / 'eval'
    args = ['evaluate', '--config', str(DESIGN), '--out', str(out)]
    for s in sets:
        args += ['--set', s]
    assert cli.main(args) == 0
    result = json.loads((out / 'result.json').read_text())
    assert result['status'] == 'ok'
    scores = result['scores']
    assert scores['T'] > 0.7 and scores['I_final'] == pytest.approx(scores['T'] * 1.0, rel=1e-6)
    assert 0.75 < scores['E_med'] < 0.95
    track = json.loads((out / 'track.json').read_text())
    assert track['space_charge']['on'] is True
