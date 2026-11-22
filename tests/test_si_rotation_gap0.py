"""The spiral-inflector offset and gap 0's leave point (docs/si_rotation_gap0.md), on a synthetic export: an
exit plate 15 deg clockwise of gap 0's ray, its outer corner half a gap off the ray (the corner rule's placement at the
test machine's 42 deg opening, gap 0's ray at 24 deg), a housing round the axis with that plate as its exit face and a
4 x 22 mm exit aperture on it."""
import dataclasses
import json
import pickle
import types

import numpy as np
import pytest

from PyCentralRegion import config as C
from PyCentralRegion.config import ConfigError
from PyCentralRegion.geometry.options import GapLine, GeometryConfig

from machine_fixtures import four_dee_machine

RAY0 = 24.0                  # gap 0's ray at the test machine's opening: dee 0 at 45 deg, 42 deg wide
PLATE_VS_RAY = -15.0         # the exit plate's direction off gap 0's ray
HALF_GAP = 3.25              # the corner's distance off the ray (the export's half gap)
APERTURE_S = (-40.0, -18.0)  # the exit aperture along the plate face from its outer corner [mm]
APERTURE_HALF_Z = 5.0


def _rot(a_deg):
    a = np.radians(a_deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _frame():
    """(machine point of gap-0-frame (x, y), plate face point at arclength s from the corner, d, n) of the export."""
    e, f = _rot(RAY0) @ [1.0, 0.0], _rot(RAY0) @ [0.0, 1.0]
    d = _rot(RAY0 + PLATE_VS_RAY) @ [1.0, 0.0]
    n = np.array([-d[1], d[0]])                   # the plate's downstream normal, towards dee 0

    def at(x, y):
        return x * e + y * f
    corner = at(86.0, -HALF_GAP)

    def face(s):
        return corner + s * d
    return at, face, d, n


def _export(folder):
    """exit_plane_spec.json, housing_outline.txt (in export/steps) and a seed of no use but its shape [paths]."""
    at, face, d, n = _frame()
    exp = folder / 'export'
    (exp / 'steps').mkdir(parents=True, exist_ok=True)
    base = face(-25.0)
    spec = {'gap_azimuth_deg': RAY0, 'half_gap_mm': HALF_GAP,
            'baseline': {'point_mm': [*base, 0.0], 'normal': [*n, 0.0], 'r_mm': float(np.hypot(*base))},
            'corner': {'rotated_mm': [*face(0.0), 0.0]}}
    (exp / 'exit_plane_spec.json').write_text(json.dumps(spec))
    outer = [face(-72.0), face(0.0), face(0.0) - 15.0 * n, at(35.0, -22.0), at(-25.0, -25.0), at(-30.0, 5.0),
             at(-5.0, 10.0)]
    ap = [(*face(APERTURE_S[0]), -APERTURE_HALF_Z), (*face(APERTURE_S[1]), -APERTURE_HALF_Z),
          (*face(APERTURE_S[1]), APERTURE_HALF_Z), (*face(APERTURE_S[0]), APERTURE_HALF_Z)]
    lines = ["frame machine", "units mm", f"outer {len(outer)}"] + [f"{x:.9f} {y:.9f}" for x, y in outer]
    lines += [f"curves {len(outer)}"] + [f"line {i} {(i + 1) % len(outer)}" for i in range(len(outer))]
    lines += [f"aperture {len(ap)}"] + [f"{x:.9f} {y:.9f} {z:.9f}" for x, y, z in ap]
    (exp / 'steps' / 'housing_outline.txt').write_text("\n".join(lines) + "\n")
    seed = types.SimpleNamespace(metadata={'optimal_geometry': {'segment_angles_per_gap': [[0.0]] * 8,
                                                                'segment_radii_per_gap': [[0.2]] * 8}},
                                 trajectory_reference=None)
    with open(folder / 'seed.pkl', 'wb') as fh:
        pickle.dump(seed, fh)
    return exp


def _cfg(folder, **geometry):
    """A connected, star-ground run on the synthetic export, [geometry] keys changed by ``geometry``."""
    _export(folder)
    g = dict(gap_lines='connected', snap_nodes=False, gap0_on_plate=True, si_rotation='corner', ground='star',
             r_max_mm=200.0, plot_dees=False, **geometry)
    lines = [f"{k} = {C.toml_value(v)}" for k, v in g.items()]
    items = [(0, 104.5, 159.5)] + [(k, 110.0, 150.0) for k in range(1, 8)]
    tables = "".join(f"\n[[geometry.gap_line]]\ngap = {k}\nr1_mm = {r1}\nr2_mm = {r2}\n" for k, r1, r2 in items)
    machine = '\n'.join(f"{k} = {C.toml_value(v)}" for k, v in dataclasses.asdict(four_dee_machine()).items())
    p = folder / 'run.toml'
    p.write_text(f"[paths]\nseed = {C.toml_value(str(folder / 'seed.pkl'))}\nhousing_outline = "
                 f"{C.toml_value(str(folder / 'export' / 'steps' / 'housing_outline.txt'))}\n"
                 f"[geometry]\n" + "\n".join(lines) + tables + f"\n[machine]\n{machine}\n")
    return C.load(p)


def _layout(folder, **geometry):
    GB = pytest.importorskip('PyCentralRegion.geometry.build')
    return GB.layout(_cfg(folder, **geometry), verbose=False)


def _gap0(lay):
    """gap 0's polyline [mm] as laid (its vertices)."""
    segs = lay['dees'].gaps[0].segments
    return np.array([np.asarray(segs[0]['p1'][:2]) * 1e3] + [np.asarray(s['p2'][:2]) * 1e3 for s in segs
                                                              if s['type'] == 'variable'])


# ---------------------------------------------------------------------------------------------------------------------
def test_the_two_keys_are_checked_with_the_config():
    g = GeometryConfig()
    assert (g.si_rotation_offset_deg, g.gap0_leave_mm) == (0.0, 0.0) and not g.gap0_moved()
    assert not hasattr(g, 'gap0_tilt_deg')                            # held: docs/si_rotation_gap0.md
    lines = (GapLine(0, 104.5, 159.5), GapLine(1, 110.0, 150.0))
    connected = dict(gap_lines='connected', snap_nodes=False, gap0_on_plate=True, gap_line=lines)
    assert GeometryConfig(si_rotation='corner', si_rotation_offset_deg=-3.0).gap0_moved()
    assert GeometryConfig(gap0_leave_mm=-5.0, **connected).gap0_moved()  # 'fixed' mode: from its own home point
    with pytest.raises(ConfigError, match="si_rotation_offset_deg = 2 turns the spiral inflector off the corner rule"):
        GeometryConfig(si_rotation_offset_deg=2.0)
    with pytest.raises(ConfigError, match="gap0_leave_mm = 5 places gap 0's connected line"):
        GeometryConfig(gap0_leave_mm=5.0)
    for key in ('si_rotation_offset_deg', 'gap0_leave_mm'):
        with pytest.raises(ConfigError, match="must be a finite number"):
            GeometryConfig(si_rotation='corner', **connected, **{key: float('nan')})


def test_at_zero_the_leave_point_is_the_plate_ray_point(tmp_path):
    """Both 0: gap 0 leaves the plate where the corner rule's plate line meets its ray, as before - the same numbers,
    to the bit - and the build record says where the aperture and the corner lie along the plate."""
    B = pytest.importorskip('PyCentralRegion.geometry.builder')
    lay = _layout(tmp_path)
    ln = lay['gap_lines'][0]
    pr = B.plate_ray_point(lay['dees'])
    assert lay['si_rotation_deg'] == 0.0
    assert ln['P0'] == (pr['r'], 0.0) and ln['y0_range'] == (0.0, 0.0)
    pts = _gap0(lay)
    assert np.allclose(pts[1], pr['point'], rtol=0.0, atol=1e-9)
    assert np.allclose(pts[0], pr['point'] - pr['r'] * pr['direction'], rtol=0.0, atol=1e-9)
    rec = ln['leave']
    assert rec['gap0_leave_mm'] == rec['si_rotation_offset_deg'] == 0.0
    assert rec['corner_s_mm'] == pytest.approx(0.0, abs=1.0)          # the corner rule: the corner is next to L
    lo, hi = rec['aperture_s_mm']
    assert lo < hi < 0.0 and hi - lo == pytest.approx(APERTURE_S[1] - APERTURE_S[0])


def test_the_offset_turns_the_assembly_and_keeps_gap_0_on_its_plate_point(tmp_path):
    """si_rotation_offset_deg turns the housing (its outline, the exit aperture), and with it gap 0's leave point and
    first segment, by the offset about the axis: gap 0 leaves the plate at the same place on the plate, while its
    ray, P1 and P2 stay. The model records the turned angle, and the tracker turns field, housing and bunch by it."""
    B = pytest.importorskip('PyCentralRegion.geometry.builder')
    s3d = pytest.importorskip('PyCentralRegion.simulate3d')
    ref = _layout(tmp_path / 'ref')
    L0, d0 = ref['gap_lines'][0]['plate']['point'], ref['gap_lines'][0]['plate']['direction']
    for offset in (-3.0, 2.5):
        lay = _layout(tmp_path / f'o{offset:g}', si_rotation_offset_deg=offset)
        R = _rot(offset)
        pl, rec = lay['gap_lines'][0]['plate'], lay['gap_lines'][0]['leave']
        assert lay['si_rotation_deg'] == pytest.approx(offset, abs=1e-12)
        assert np.allclose(lay['outline']['aperture'][:, :2], ref['outline']['aperture'][:, :2] @ R.T, atol=1e-9)
        assert np.allclose(lay['outline']['outer'], ref['outline']['outer'] @ R.T, atol=1e-9)
        assert np.allclose(pl['point'], R @ L0, atol=1e-9) and np.allclose(pl['direction'], R @ d0, atol=1e-12)
        assert np.allclose(pl['corner'], R @ ref['gap_lines'][0]['plate']['corner'], atol=1e-9)
        assert rec['corner_s_mm'] == pytest.approx(ref['gap_lines'][0]['leave']['corner_s_mm'], abs=1e-9)
        assert rec['aperture_s_mm'] == pytest.approx(ref['gap_lines'][0]['leave']['aperture_s_mm'], abs=1e-9)
        assert rec['across_ray_mm'] == pytest.approx(np.hypot(*L0) * np.sin(np.radians(offset)), rel=1e-9)
        pts, pts0 = _gap0(lay), _gap0(ref)
        assert np.allclose(pts[1], R @ pts0[1], atol=1e-9)                       # the leave point turned
        assert np.allclose(pts[2:], pts0[2:], atol=1e-9)                         # P1, P2 on the ray, unmoved
        model = types.SimpleNamespace(params={'si_rotation_deg': lay['si_rotation_deg']})
        cfg = _cfg(tmp_path / f'o{offset:g}', si_rotation_offset_deg=offset)
        s3d.follow_model_si_rotation(cfg, model)
        assert C.section(cfg, 'inflector').rotation_deg == lay['si_rotation_deg']
        s3d.check_si_rotation(model, C.section(cfg, 'inflector').rotation_deg)
    with pytest.raises(ConfigError, match="no P1 at r1 = 104.500 mm keeps the bend at P1"):
        _layout(tmp_path / 'far', si_rotation_offset_deg=14.0)


def test_the_leave_point_moves_gap_0_along_the_plate_without_turning_anything(tmp_path):
    """gap0_leave_mm slides the point where gap 0 leaves the plate along it; the inflector, the housing and the first
    segment's direction stay. Into the exit aperture it is refused."""
    lay0 = _layout(tmp_path / 'ref')
    pl0 = lay0['gap_lines'][0]['plate']
    for leave in (-10.0, 6.0):
        lay = _layout(tmp_path / f'l{leave:g}', gap0_leave_mm=leave)
        pl, rec = lay['gap_lines'][0]['plate'], lay['gap_lines'][0]['leave']
        assert lay['si_rotation_deg'] == 0.0
        assert np.array_equal(lay['outline']['outer'], lay0['outline']['outer'])
        assert np.allclose(pl['point'], pl0['point'] + leave * pl0['direction'], atol=1e-12)
        assert np.array_equal(pl['home'], pl0['point']) and np.array_equal(pl['direction'], pl0['direction'])
        assert rec['corner_s_mm'] == pytest.approx(lay0['gap_lines'][0]['leave']['corner_s_mm'] - leave, abs=1e-9)
        pts = _gap0(lay)
        assert np.allclose(pts[1], pl['point'], atol=1e-9)
        u = (pts[1] - pts[0]) / np.hypot(*(pts[1] - pts[0]))
        assert np.allclose(u, pl0['direction'], atol=1e-12)                     # still on the plate line
    hi = lay0['gap_lines'][0]['leave']['aperture_s_mm'][1]
    with pytest.raises(ConfigError, match="inward of the exit aperture's outer end"):
        _layout(tmp_path / 'in', gap0_leave_mm=hi - 0.5)


def test_offset_and_leave_point_together(tmp_path):
    """Both at once: the leave point is the home point turned with the assembly, then moved along the turned plate."""
    ref = _layout(tmp_path / 'ref')
    L0, d0 = ref['gap_lines'][0]['plate']['point'], ref['gap_lines'][0]['plate']['direction']
    offset, leave = 2.0, -6.0
    lay = _layout(tmp_path / 'both', si_rotation_offset_deg=offset, gap0_leave_mm=leave)
    pl, rec = lay['gap_lines'][0]['plate'], lay['gap_lines'][0]['leave']
    R = _rot(offset)
    assert np.allclose(pl['home'], R @ L0, atol=1e-9)
    assert np.allclose(pl['point'], R @ L0 + leave * (R @ d0), atol=1e-9)
    assert rec['corner_s_mm'] == pytest.approx(ref['gap_lines'][0]['leave']['corner_s_mm'] - leave, abs=1e-9)
    assert np.allclose(_gap0(lay)[1], pl['point'], atol=1e-9)


def test_the_geometry_key_drops_the_two_keys_at_zero():
    """A MOBO geometry key: the two keys at 0 key like their absence (existing stores keep their keys); not at 0
    they are part of it."""
    from PyCentralRegion.mobo import space
    cfg = space.MoboConfig(campaign_dir='c', base_config_file='b.toml', objectives=(space.Objective('I_final'),),
                           variables=(space.Variable('phi', 'machine.dee_opening_deg', 34.0, 45.0, kind='geometry'),
                                      space.Variable('leave', 'geometry.gap0_leave_mm', -10.0, 10.0, kind='geometry')))
    bare = {'geometry': {'gap_lines': 'connected'}, 'machine': {'dee_opening_deg': 38.0}}
    zero = {'geometry': {'gap_lines': 'connected', 'si_rotation_offset_deg': 0.0, 'gap0_leave_mm': 0},
            'machine': {'dee_opening_deg': 38.0}}
    only_phi = dataclasses.replace(cfg, variables=cfg.variables[:1])
    key = space.geometry_key(bare, only_phi, {'phi': 38.0})
    assert space.geometry_key(zero, only_phi, {'phi': 38.0}) == key
    assert space.geometry_key(bare, cfg, {'phi': 38.0, 'leave': 0.0}) == key
    assert space.geometry_key(bare, cfg, {'phi': 38.0, 'leave': 4.0}) != key
    moved = {'geometry': {'gap_lines': 'connected', 'si_rotation_offset_deg': 1.5}, 'machine': {'dee_opening_deg': 38.0}}
    assert space.geometry_key(moved, only_phi, {'phi': 38.0}) != key
    assert 'idle' not in space.geometry_parts(zero, only_phi, {'phi': 38.0})
