"""The electrode builder (PyCentralRegion.geometry): options, the old-spec converters, the channel bounds, the builder's
machine check and channel profile, and the build's file handling. Fast: nothing here meshes. The last tests lay out the
worked example's reference design in 2D when its input files are on this machine (else they skip)."""
import dataclasses
from pathlib import Path

import pytest

from PyCentralRegion import config as C
from PyCentralRegion.config import ConfigError
from PyCentralRegion.geometry import options as O
from PyCentralRegion.geometry.options import GeometryConfig

REPO = Path(__file__).resolve().parents[1]
DESIGN = REPO / 'configs' / 'examples' / 'hchc1p5' / 'design_wnop3_rotn3_g0ap.toml'

# the design deck's build of the reference design wnop3_rotn3_g0ap: its CR_* environment and command line
REF_ENV = {'CR_SI_ROT': 'corner', 'HCHC60_EXPORT': r'E:\X\export', 'MINIDAR_SI_ROT_DEG': '4.726034523988023',
           'CR_GROUND': 'star', 'CR_DEE_INNER': 'hug', 'CR_DEE0_NOSE_MM': '3', 'CR_JUNCTION_R': '2:124.6,3:208.5',
           'CR_APERTURE_MM': '26,32', 'CR_APERTURE_R_MM': '100,140', 'CR_OCC_FUZZY_MM': '0.05'}
REF_ARGS = ['--pkl', 'seed.pkl', '--outline', 'housing_outline.txt', '--handoff', 'handoff_sc_all_level.h5',
            '--build', 'out/cr_model_x', '--pickle', 'out/cr_model_x_model3d.pkl', '--dees', 'out/cr_model_x_dees.png',
            '--lc', '8', '--lc-outer', '16', '--r-fine', '150', '--r-coarse', '220', '--dee0-clearance', '5',
            '--gap-apertures', '0:dee:10:46.12:0:28.12:64.12']


def _load(tmp_path, text, overrides=()):
    p = tmp_path / 'run.toml'
    p.write_text(text)
    return C.load(p, overrides=list(overrides))


# ---------------------------------------------------------------------------------------------------------------------
# options
def test_defaults_and_the_channel():
    g = GeometryConfig()
    assert g.aperture_heights() == (26.0, 30.0) and g.aperture_radii() == (100.0, 140.0)
    assert g.dummy_aperture_heights() == g.aperture_heights()          # 0 = the dees' values
    assert O.check_bounds(g) == {'height_max_mm': 30.0, 'inner_min_mm': None}   # the defaults pass their own bounds
    assert GeometryConfig(dummy_aperture_inner_mm=28.0).dummy_aperture_heights() == (28.0, 30.0)
    assert g.si_rotation == 'fixed'


def test_the_four_channel_fields_are_independent():
    inverted = GeometryConfig(aperture_inner_mm=30.0, aperture_outer_mm=22.0, aperture_taper_center_mm=60.0,
                              aperture_taper_length_mm=100.0)            # no ordering constraint between them
    assert inverted.aperture_radii() == (10.0, 110.0)
    for bad in ({'aperture_taper_length_mm': 0.0}, {'aperture_taper_center_mm': 10.0}, {'aperture_inner_mm': 0.0},
                {'dummy_aperture_outer_mm': -1.0}, {'aperture_outer_mm': 100.0}, {'si_rotation': 'inflector'},
                {'ground': 'plates'}, {'aperture_inset_mm': (1.0,)}, {'lc_outer_mm': 16.0, 'r_coarse_mm': 100.0}):
        with pytest.raises(ConfigError):
            GeometryConfig(**bad)


def test_old_channel_strings():
    assert O.aperture_from_spec("26,32", "100,140") == {
        'aperture_inner_mm': 26.0, 'aperture_outer_mm': 32.0, 'aperture_taper_center_mm': 120.0,
        'aperture_taper_length_mm': 40.0}
    assert O.aperture_from_spec("26, 50", None, "34,50") == {
        'aperture_inner_mm': 26.0, 'aperture_outer_mm': 50.0, 'dummy_aperture_inner_mm': 34.0,
        'dummy_aperture_outer_mm': 50.0}
    for heights, radii in (("26", "100,140"), ("26,32", "140,100"), ("26,32", "100")):
        with pytest.raises(ConfigError):
            O.aperture_from_spec(heights, radii)


def test_bounds_name_the_bound():
    with pytest.raises(ConfigError, match=r"aperture_outer_mm = 32 mm exceeds its bound 2 x \(geometry\.plate_half_gap"):
        O.check_bounds(GeometryConfig(aperture_outer_mm=32.0))
    ok = O.check_bounds(GeometryConfig(aperture_outer_mm=32.0, plate_half_gap_mm=25.0), exit_half_height_mm=10.741)
    assert ok['height_max_mm'] == 40.0 and ok['inner_min_mm'] == pytest.approx(21.482)
    with pytest.raises(ConfigError, match="aperture_inner_mm = 20 mm is below its bound 2 x the spiral-inflector housing"):
        O.check_bounds(GeometryConfig(aperture_inner_mm=20.0), exit_half_height_mm=10.741)
    with pytest.raises(ConfigError, match="dummy_aperture_inner_mm"):
        O.check_bounds(GeometryConfig(dummy_aperture_inner_mm=18.0), exit_half_height_mm=10.0)
    with pytest.raises(ConfigError, match="dummy_aperture_outer_mm"):
        O.check_bounds(GeometryConfig(dummy_aperture_outer_mm=31.0))
    with pytest.raises(ConfigError, match="leaves no room"):
        O.check_bounds(GeometryConfig(plate_half_gap_mm=5.0))


def test_windows_from_toml_and_the_dotted_override(tmp_path):
    cfg = _load(tmp_path, '[[geometry.windows]]\ngap = 0\nside = "dee"\ns_lo_mm = 31.6\ns_hi_mm = 53.2\n'
                'half_height_mm = 13.0\n\n[[geometry.windows]]\ngap = 1\nside = "dummy"\ns_lo_mm = 10\ns_hi_mm = 20\n'
                'half_height_mm = 9\nz_center_mm = 1\n', overrides=['geometry.windows.1.half_height_mm=7.5'])
    w0, w1 = C.section(cfg, 'geometry').windows
    assert isinstance(w0, O.WindowSpec) and (w0.z_lo_mm, w0.z_hi_mm) == (-13.0, 13.0)
    assert w1.half_height_mm == 7.5 and (w1.z_lo_mm, w1.z_hi_mm) == (-6.5, 8.5) and isinstance(w1.s_lo_mm, float)
    assert C.get_dotted(cfg, 'geometry.windows.1.half_height_mm') == 7.5
    assert O.window_from_opening(0, 'dee', 31.6, 53.2, -13, 13) == w0      # the old opening convention
    with pytest.raises(ConfigError, match="half_height_mm must be > 0"):
        O.WindowSpec(0, 'dee', 1.0, 2.0, 0.0)
    with pytest.raises(ConfigError, match="side must be 'dee' or 'dummy'"):
        O.WindowSpec(0, 'ground', 1.0, 2.0, 1.0)
    with pytest.raises(ConfigError, match="unknown key"):
        C.section(_load(tmp_path, '[[geometry.windows]]\ngap = 0\nside = "dee"\ns_lo_mm = 1.0\ns_hi_mm = 2.0\n'
                                  'half_height_mm = 3.0\nz_lo_mm = -3.0\n'), 'geometry')


def test_old_hardware_strings():
    posts = O.posts_from_spec("1:dee:120,1:dummy:118, 2:dee:24.57:20")
    assert posts[2] == O.PostSpec(2, 'dee', 24.57, 20.0) and posts[0].length_mm == 0.0
    (w,) = O.windows_from_spec("0:dee:31.6:53.2:-13:13")
    assert (w.half_height_mm, w.z_center_mm) == (13.0, 0.0)
    (h,) = O.holes_from_spec("0:dee:10:46.12:0:28.12:64.12")
    assert h == O.HoleSpec(0, 'dee', 10.0, 46.12, 0.0, 28.12, 64.12)
    with pytest.raises(ConfigError, match="runs past its plate"):
        O.holes_from_spec("0:dee:10:46.12:0:40:64.12")
    assert O.boxes_from_spec("1,2,3,4|-5,-4,0,1")[1] == O.BoxSpec(-5.0, -4.0, 0.0, 1.0)
    assert O.junctions_from_spec("2:124.6,3:208.5") == (O.JunctionMove(2, 124.6), O.JunctionMove(3, 208.5))
    assert O.dee_rmin_from_spec("1:30") == (O.DeeRmin(1, 30.0),)
    for bad in ("1:dee", "1:roof:3"):
        with pytest.raises(ConfigError):
            O.posts_from_spec(bad)


def test_posts_json(tmp_path):
    p = tmp_path / 'posts.json'
    p.write_text('{"gap0_posts": [30, 60], "gap_posts": {"1:dee": [39.3, 72.4]}, '
                 '"gap_windows": {"1:dee": [39.3, 72.4, -10, 12]}}')
    d = O.from_posts_json(p)
    assert d['gap0_posts_mm'] == (30.0, 60.0) and len(d['posts']) == 2
    assert d['windows'] == (O.WindowSpec(1, 'dee', 39.3, 72.4, 11.0, 1.0),)
    GeometryConfig(**d)


def test_from_deck_reference_build():
    t = O.from_deck(REF_ENV, REF_ARGS)
    g = GeometryConfig(**t['geometry'])
    assert (g.si_rotation, g.gap0_on_plate, g.ground, g.dee_inner, g.plot_dees) == ('corner', True, 'star', 'hug', True)
    assert (g.aperture_inner_mm, g.aperture_outer_mm, g.aperture_taper_center_mm, g.aperture_taper_length_mm) == \
        (26.0, 32.0, 120.0, 40.0)
    assert g.junctions == (O.JunctionMove(2, 124.6), O.JunctionMove(3, 208.5))
    assert g.holes == (O.HoleSpec(0, 'dee', 10.0, 46.12, 0.0, 28.12, 64.12),)
    assert (g.dee0_clearance_mm, g.dee0_nose_mm, g.lc_mm, g.lc_outer_mm) == (5.0, 3.0, 8.0, 16.0)
    assert t['paths'] == {'seed': 'seed.pkl', 'housing_outline': 'housing_outline.txt',
                          'model3d': 'out/cr_model_x_model3d.pkl'}
    assert t['inflector'] == {'rotation_deg': 4.726034523988023}
    assert Path(g.exit_plane_spec_file) == Path(r'E:\X\export') / 'exit_plane_spec.json'
    assert g.emitting_plane_file == 'handoff_sc_all_level.h5'
    bare = O.from_deck({}, '--lc 10')               # the deck's own defaults: 40 mm from 148 to 247 mm, no rotation
    assert (bare['geometry']['aperture_inner_mm'], bare['geometry']['aperture_taper_center_mm'],
            bare['geometry']['lc_mm']) == (40.0, 197.5, 10.0)
    assert bare['inflector'] == {'rotation_deg': 0.0}
    fixed = O.from_deck({'CR_SI_ROT': '-3.5'}, [])
    assert fixed['geometry']['si_rotation'] == 'fixed' and fixed['inflector'] == {'rotation_deg': -3.5}
    with pytest.raises(ConfigError, match="--solve"):
        O.from_deck({}, ['--solve'])


def test_to_toml_round_trip(tmp_path):
    geo = O.from_deck(REF_ENV, REF_ARGS)['geometry']
    back = C.section(_load(tmp_path, O.to_toml(geo)), 'geometry')
    paths = {'emitting_plane_file': '', 'exit_plane_spec_file': ''}       # resolved against the file: compared apart
    assert dataclasses.replace(back, **paths) == dataclasses.replace(GeometryConfig(**geo), **paths)
    assert Path(back.emitting_plane_file).name == 'handoff_sc_all_level.h5'
    assert O.to_toml(GeometryConfig()) == '[geometry]\n'


def test_design_example_geometry_table():
    cfg = C.load(DESIGN)
    g = C.section(cfg, 'geometry')
    assert (g.si_rotation, g.ground, g.dee_inner, g.plate_half_gap_mm) == ('corner', 'star', 'hug', 25.0)
    assert O.check_bounds(g, exit_half_height_mm=10.741)['height_max_mm'] == 40.0
    ref = GeometryConfig(**O.from_deck(REF_ENV, REF_ARGS)['geometry'])
    ignore = {'emitting_plane_file': '', 'exit_plane_spec_file': '', 'plate_half_gap_mm': 0.0}
    assert dataclasses.replace(g, **ignore) == dataclasses.replace(ref, **ignore)


# ---------------------------------------------------------------------------------------------------------------------
# the builder and the build
def test_builder_refuses_other_machines():
    B = pytest.importorskip('PyCentralRegion.geometry.builder')
    with pytest.raises(NotImplementedError):
        B.configure(GeometryConfig(), C.MachineConfig(n_dees=2, dee_center_angles_deg=(90.0, 270.0)))
    with pytest.raises(NotImplementedError):
        B.configure(GeometryConfig(), C.MachineConfig(dee_shape='spiral'))
    with pytest.raises(ConfigError):
        B.configure(GeometryConfig(), C.MachineConfig(dee_center_angles_deg=(45.0, 135.0)))


def test_channel_profile_by_owner():
    B = pytest.importorskip('PyCentralRegion.geometry.builder')
    B.configure(GeometryConfig(aperture_inner_mm=26.0, aperture_outer_mm=32.0, dummy_aperture_inner_mm=34.0,
                               plate_half_gap_mm=25.0), C.MachineConfig())
    assert [B.channel_half_height(r) for r in (50.0, 100.0, 120.0, 140.0, 300.0)] == [13.0, 13.0, 14.5, 16.0, 16.0]
    assert B.channel_half_height(50.0, 'dummy2') == 17.0 and B.channel_half_height(300.0, 'ground') == 16.0
    assert B.r_max_mm() == pytest.approx(370.0)
    with pytest.raises(ConfigError, match="exit_plane_spec_file is not set"):
        B.exit_spec()
    B.configure()


def test_exit_half_height_and_bounds_from_the_outline(tmp_path):
    B = pytest.importorskip('PyCentralRegion.geometry.builder')
    from PyCentralRegion.geometry import build as GB
    outline = tmp_path / 'export' / 'steps_baseline_mm' / 'housing_outline.txt'
    outline.parent.mkdir(parents=True)
    outline.write_text("frame baseline\nunits mm\nouter 4\n0 -10\n30 -10\n30 10\n0 10\n"
                       "aperture 4\n30 -5 -9\n30 5 -9\n30 5 11\n30 -5 11\n")
    assert B.exit_half_height_mm(B.read_housing_outline(outline)) == 11.0
    cfg = _load(tmp_path, f'[paths]\nhousing_outline = "{outline.as_posix()}"\n[geometry]\nplate_half_gap_mm = 25.0\n')
    assert GB.bounds(cfg) == {'height_max_mm': 40.0, 'inner_min_mm': 22.0, 'exit_half_height_mm': 11.0,
                              'plate_half_gap_mm': 25.0, 'dee_wall_mm': 5.0}
    cfg = _load(tmp_path, f'[paths]\nhousing_outline = "{outline.as_posix()}"\n[geometry]\naperture_inset_mm = [0.0, 1.0]\n')
    assert GB.bounds(cfg)['exit_half_height_mm'] == 10.0       # the inset takes 1 mm off each end of the 20 mm side


def test_build_files_reuse_and_refusal(tmp_path, capsys):
    pytest.importorskip('PyCentralRegion.geometry.builder')
    from PyCentralRegion.geometry import build as GB
    out = GB.output_files(tmp_path / 'cr_model_a_model3d.pkl')
    assert [p.name for p in out.values()] == ['cr_model_a_dees.png', 'cr_model_a.msh', 'cr_model_a.step',
                                              'cr_model_a_build.json', 'cr_model_a_model3d.pkl']
    assert GB.output_files(tmp_path / 'model3d.pkl')['msh'].name == 'model3d.msh'
    assert GB._part(out['msh']).name == 'cr_model_a.part.msh'
    model = tmp_path / 'cr_model_a_model3d.pkl'
    cfg = _load(tmp_path, f'[paths]\nmodel3d = "{model.as_posix()}"\n')
    model.write_bytes(b'not rebuilt')
    assert GB.build(cfg) == {'model3d': str(model)}                    # reused: nothing else is needed or written
    assert 'reused' in capsys.readouterr().out and model.read_bytes() == b'not rebuilt'
    model.unlink()
    out['msh'].write_text('left over')
    with pytest.raises(ConfigError, match="never overwritten"):
        GB.build(cfg)
    assert out['msh'].read_text() == 'left over'
    with pytest.raises(ConfigError, match="paths.model3d is not set"):
        GB.build(_load(tmp_path, '[geometry]\n'))


def test_build_inputs_are_checked(tmp_path):
    pytest.importorskip('PyCentralRegion.geometry.builder')
    from PyCentralRegion.geometry import build as GB
    seed = tmp_path / 'seed.pkl'
    seed.write_bytes(b'')
    outline = tmp_path / 'export' / 'steps_baseline_mm' / 'housing_outline.txt'
    outline.parent.mkdir(parents=True)
    outline.write_text('')
    base = f'[paths]\nseed = "{seed.as_posix()}"\nhousing_outline = "{outline.as_posix()}"\n'
    with pytest.raises(ConfigError, match="no emitting plane"):
        GB.inputs(_load(tmp_path, base))
    (tmp_path / 'export' / 'handoff_vac_all_level.h5').write_bytes(b'')
    inp = GB.inputs(_load(tmp_path, base))
    assert inp['emitting_plane'].name == 'handoff_vac_all_level.h5' and inp['exit_plane_spec'] is None
    with pytest.raises(ConfigError, match="exit plane spec"):
        GB.inputs(_load(tmp_path, base + '[geometry]\nsi_rotation = "corner"\n'))
    (tmp_path / 'export' / 'exit_plane_spec.json').write_text('{}')
    assert GB.inputs(_load(tmp_path, base + '[geometry]\ngap0_on_plate = true\n'))['exit_plane_spec'].parent.name == \
        'export'
    with pytest.raises(ConfigError, match="paths.seed = .* does not exist"):
        GB.inputs(_load(tmp_path, base.replace('seed.pkl', 'missing.pkl')))


# ---------------------------------------------------------------------------------------------------------------------
# the reference design in 2D (its inputs live on the design machine)
def _reference_cfg():
    if not DESIGN.exists():
        pytest.skip("the worked example is not in this checkout")
    cfg = C.load(DESIGN)
    pc = C.section(cfg, 'paths')
    if not all(Path(p).exists() for p in (pc.seed, pc.housing_outline, pc.handoff)):
        pytest.skip("the reference design's input files are not on this machine")
    return cfg


@pytest.fixture(scope='module')
def reference_layout():
    pytest.importorskip('PyCentralRegion.geometry.builder')
    from PyCentralRegion.geometry import build as GB
    cfg = _reference_cfg()
    return cfg, GB.layout(cfg, verbose=False)


def test_reference_layout_matches_the_deck(reference_layout):
    """Numbers of the design deck's build log of _wnop3_rotn3_g0ap (2026-10-31)."""
    import numpy as np
    _, lay = reference_layout
    assert lay['si_rotation_deg'] == pytest.approx(4.726034523988023, abs=1e-12)
    assert lay['exit_half_height_mm'] == pytest.approx(10.741, abs=1e-3)
    ends = {d['label']: (np.round(d['lo'][0], 3).tolist(), np.round(d['hi'][0], 3).tolist()) for d in lay['foot']}
    assert ends == {'dee0': ([13.698, 34.273], [13.698, 34.273]), 'dee1': ([-19.402, 16.031], [-24.175, 8.664]),
                    'dee2': ([-31.397, -23.821], [-22.268, -33.641]), 'dee3': ([31.809, -51.275], [69.62, -36.864])}
    assert [d['tip']['kind'] for d in lay['foot']] == ['round', 'housing locus', 'housing locus', 'housing locus']
    assert lay['foot'][0]['tip']['min_d'] == pytest.approx(8.166, abs=1e-3)
    assert [round(d['tip']['span'], 3) for d in lay['dummy']] == [4.333, 4.491, 24.019, 71.133]


def test_reference_gap0_hole_and_edge_shift(reference_layout):
    from PyCentralRegion.geometry import build as GB
    cfg, lay = reference_layout
    res = GB.gap0_hole(cfg, r_mm=10.0, rim_mm=8.0, dr_mm=-2.0, lay=lay, verbose=False)
    assert res['hole'] == C.section(cfg, 'geometry').holes[0]             # the design's hole: 2 mm inside the beam
    assert res['s_beam_mm'] == pytest.approx(48.311, abs=1e-3) and res['n_hit'] == res['n_kept'] == 97958
    (sh,) = GB.edge_shift(cfg, ['0:dee:46.12:0'], lay=lay, verbose=False)
    assert sh['s1'] == pytest.approx(46.12, abs=1e-6) and sh['r0'] == pytest.approx(73.582, abs=1e-3)
    (sh,) = GB.edge_shift(cfg, [(0, 'dee', 48.3109, -2.0)], lay=lay, verbose=False)
    assert sh['s1'] == pytest.approx(46.12, abs=1e-3) and sh['r1'] == pytest.approx(sh['r0'] - 2.0)
