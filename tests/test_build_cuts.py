"""The builder's cuts that must remove metal (geometry.builder._cut_removing). A fuzzy OCC boolean can hand back its
object untouched; on 2026-11-03 that left dees and the star ground whole whenever the channel taper ended beyond
~150 mm, and a retry at a smaller tolerance could leave a sealed void of the tool's shape instead of a channel. The
first test uses boxes; the second builds and meshes two such designs (about 15 s each) when their input files are on
this machine, else it skips."""
import contextlib
import io
import signal
from pathlib import Path

import numpy as np
import pytest

from PyCentralRegion import config as C

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / 'configs' / 'examples' / 'hchc1p5'


@pytest.fixture(autouse=True)
def sigint_handler_kept():
    """gmsh.initialize() sets SIGINT to SIG_DFL and gmsh 4.15's finalize() does not put the old handler back (it keeps
    it in a local), so after a build in this process the next SIGINT ends pytest from the thread that raised it: the
    Ctrl+C tests of test_mobo raise one from a timer thread, and the whole run hung in that exit. Each test here leaves
    the handler it found."""
    handler = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGINT, handler)


def test_a_cut_that_must_remove_metal(monkeypatch):
    """A cut that removes metal is the plain occ.cut; one the fuzzy boolean hands back untouched is made again at the
    next smaller tolerance (and the tolerance is restored); a retry that removes the whole tool (a sealed void) does not
    count; a cut that never counts fails, and so does a first cut that removes the whole tool."""
    B = pytest.importorskip('PyCentralRegion.geometry.builder')
    gmsh = pytest.importorskip('gmsh')
    started = not gmsh.isInitialized()
    if started:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cut_removing")
    tol0 = gmsh.option.getNumber("Geometry.ToleranceBoolean")
    occ = gmsh.model.occ
    try:
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 0.05)
        notes = []
        out = B._cut_removing(gmsh, [(3, occ.addBox(0, 0, 0, 10, 10, 10))],
                              lambda: [(3, occ.addBox(5, 5, 5, 10, 10, 10))], 'box a', notes.append)
        assert B._occ_mass(occ, out) == pytest.approx(875.0) and notes == []
        real = occ.cut

        def untouched_when_fuzzy(obj, tool, **kw):         # what the 0.05 mm boolean did to the channel cuts
            if gmsh.option.getNumber("Geometry.ToleranceBoolean") >= 0.05:
                occ.remove(tool, recursive=True)
                return obj, [obj]
            return real(obj, tool, **kw)

        monkeypatch.setattr(occ, 'cut', untouched_when_fuzzy)
        out = B._cut_removing(gmsh, [(3, occ.addBox(20, 0, 0, 10, 10, 10))],
                              lambda: [(3, occ.addBox(25, 5, 5, 10, 10, 10))], 'box b', notes.append)
        assert B._occ_mass(occ, out) == pytest.approx(875.0)
        assert notes == ["box b: the cut at boolean tolerance 0.05 mm removed no metal; at 0.02 mm it removed 0.1 cm^3 "
                         "of its 1.0 cm^3 tool"]
        assert gmsh.option.getNumber("Geometry.ToleranceBoolean") == 0.05
        with pytest.raises(RuntimeError, match=r"box c: no cut counted \(the 0\.0 cm\^3 tool removed at boolean "
                                               r"tolerance 0\.05 mm: nothing; 0\.02 mm: nothing; 0\.01 mm: nothing; "
                                               r"0\.005 mm: nothing; 0 mm: nothing\) - the solid would be left whole"):
            B._cut_removing(gmsh, [(3, occ.addBox(40, 0, 0, 10, 10, 10))],
                            lambda: [(3, occ.addBox(60, 0, 0, 1, 1, 1))], 'box c')   # a tool that misses
        assert gmsh.option.getNumber("Geometry.ToleranceBoolean") == 0.05

        def void_at_002(obj, tool, **kw):                  # what 0.02 mm did to the jogged design's ground
            tol = gmsh.option.getNumber("Geometry.ToleranceBoolean")
            if tol >= 0.05:
                occ.remove(tool, recursive=True)
                return obj, [obj]
            if tol >= 0.02:
                occ.translate(tool, -10, 0, 0)             # the whole tool inside the box
            return real(obj, tool, **kw)

        monkeypatch.setattr(occ, 'cut', void_at_002)
        notes.clear()
        out = B._cut_removing(gmsh, [(3, occ.addBox(100, 0, 0, 20, 10, 10))],
                              lambda: [(3, occ.addBox(115, 2, 2, 10, 6, 6))], 'box d', notes.append)
        assert B._occ_mass(occ, out) == pytest.approx(2000.0 - 180.0) and len(out) == 1
        assert notes == ["box d: the cut at boolean tolerance 0.05 mm removed no metal; at 0.01 mm it removed 0.2 cm^3 "
                         "of its 0.4 cm^3 tool (not counted: 0.02 mm: 0.4 cm^3)"]
        monkeypatch.undo()
        with pytest.raises(RuntimeError, match=r"box e: the cut at boolean tolerance 0\.05 mm removed 0\.1 cm\^3 of a "
                                               r"0\.1 cm\^3 tool - not a cut through the solid"):
            B._cut_removing(gmsh, [(3, occ.addBox(140, 0, 0, 20, 10, 10))],
                            lambda: [(3, occ.addBox(145, 2, 2, 4, 4, 4))], 'box e')  # a tool inside: a void
        assert gmsh.option.getNumber("Geometry.ToleranceBoolean") == 0.05
    finally:
        gmsh.option.setNumber("Geometry.ToleranceBoolean", tol0)
        gmsh.model.remove()
        if started:
            gmsh.finalize()


@pytest.mark.parametrize('design, overrides', [
    # the connected cold start with the taper over r 100-150 mm: at 0.05 mm dee3 and the ground came out whole
    ('design_connected_cold.toml', ['geometry.aperture_taper_center_mm=125', 'geometry.aperture_taper_length_mm=50']),
    # the jogged design with the taper over r 100-263 mm: the ground's retry at 0.02 mm left a void, 0.01 mm cuts it
    ('design_wnop3_rotn3_g0ap.toml', ['geometry.aperture_taper_center_mm=181.5',
                                      'geometry.aperture_taper_length_mm=163']),
], ids=['connected_t150', 'jogged_t263'])
def test_a_long_channel_taper_is_cut(tmp_path, design, overrides):
    """A design whose channel taper ends beyond 150 mm builds with every solid hollowed: a build that left one whole
    fails (_cut_removing), and the channel splits the star ground into its two plates."""
    pytest.importorskip('PyCentralRegion.geometry.builder')
    from PyCentralRegion.geometry import build as GB
    cfg_path = EXAMPLES / design
    if not cfg_path.exists():
        pytest.skip("the worked example is not in this checkout")
    model = tmp_path / 'model3d.pkl'
    cfg = C.load(cfg_path, overrides=overrides + [f'paths.model3d="{model.as_posix()}"'])
    pc = C.section(cfg, 'paths')
    if not all(Path(p).exists() for p in (pc.seed, pc.housing_outline, pc.handoff, pc.housing_step)):
        pytest.skip("the design's input files are not on this machine")
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        GB.build(cfg)
    txt = log.getvalue()
    notes = [ln for ln in txt.splitlines() if 'removed no metal' in ln]
    assert "channel cut -> 2 solid(s)" in txt and model.exists(), "\n".join(notes)


def test_a_fuse_that_must_keep_its_tools(monkeypatch):
    """A fuse that keeps its tools is the plain occ.fuse on the tools themselves and makes no copy (a copy changes the
    mesh of a model that builds). One that drops a tool, throws or loses metal does not count, nor one that leaves another
    solid unless a tool is meant to stand apart, nor one that adds nothing. The tolerance asked for holds for that fuse
    only."""
    B = pytest.importorskip('PyCentralRegion.geometry.builder')
    gmsh = pytest.importorskip('gmsh')
    started = not gmsh.isInitialized()
    if started:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("fuse_keeping")
    tol0 = gmsh.option.getNumber("Geometry.ToleranceBoolean")
    occ = gmsh.model.occ
    real = occ.fuse
    made = []

    def fuse_boxes(x0, n_apart=0, tool_x=80.0, tol=None):
        """A 100 mm cube at x0 and a 100 x 20 x 20 mm tool at x0 + tool_x (by default 32 of its 40 cm^3 stick out); the
        result's volume [cm^3]."""
        obj = [(3, occ.addBox(x0, 0, 0, 100, 100, 100))]
        tool = [(3, occ.addBox(x0 + tool_x, 40, 40, 100, 20, 20))]
        made[:] = obj + tool
        centre = [x0 + tool_x + 50.0, 50.0, 50.0]

        def lost_of(solids):
            return [] if any(gmsh.model.isInside(3, s, centre) for s in solids) else ['the tool']

        return B._occ_mass(occ, B._fuse_keeping(gmsh, 'dee0', obj, tool, lost_of, n_apart, tol)) / 1e3

    def not_counted(why, **kw):
        with pytest.raises(B._FuseNotCounted) as ex:
            fuse_boxes(**kw)
        assert ex.value.owner == 'dee0' and ex.value.why == why and str(ex.value) == f"the fuse into dee0: {why}"
        assert gmsh.option.getNumber("Geometry.ToleranceBoolean") == 0.05

    try:
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 0.05)
        inputs = []

        def spy(obj, tool, **kw):
            inputs.append((obj + tool, gmsh.option.getNumber("Geometry.ToleranceBoolean")))
            return real(obj, tool, **kw)

        def no_copy(*a, **kw):
            raise AssertionError("a copy ahead of a fuse changes the mesh of a model that builds")

        monkeypatch.setattr(occ, 'fuse', spy)
        monkeypatch.setattr(occ, 'copy', no_copy)
        assert fuse_boxes(0.0) == pytest.approx(1032.0)
        assert inputs == [(made, 0.05)]                    # one fuse, of the object and the tool themselves
        inputs.clear()
        assert fuse_boxes(300.0, tol=0.02) == pytest.approx(1032.0)
        assert inputs == [(made, 0.02)] and gmsh.option.getNumber("Geometry.ToleranceBoolean") == 0.05
        monkeypatch.undo()

        def drops(obj, tool, **kw):                        # what 0.05 mm did to the blocks split at a bend
            occ.remove(tool, recursive=True)
            return obj, [obj]

        monkeypatch.setattr(occ, 'fuse', drops)
        not_counted("LOST metal - outside every resulting solid: the tool", x0=600.0)

        def throws(obj, tool, **kw):                       # eval 0129's dee0
            raise Exception("Union failed - BOPAlgo_AlertBuilderFailed")

        monkeypatch.setattr(occ, 'fuse', throws)
        not_counted("Union failed - BOPAlgo_AlertBuilderFailed", x0=900.0)
        not_counted("Union failed - BOPAlgo_AlertBuilderFailed", x0=900.0, tol=0.02)

        def breaks(obj, tool, **kw):                       # eval 0091's dee0: every centre inside, the dee in pieces
            res, m = real(obj, tool, **kw)
            return occ.cut(res, [(3, occ.addBox(1200.0, 0, 0, 50, 100, 100))], removeObject=True, removeTool=True)

        monkeypatch.setattr(occ, 'fuse', breaks)
        not_counted("1 solid(s) from 1, -468.0 cm^3 added by 40.0 cm^3 of tools", x0=1200.0)
        monkeypatch.undo()

        assert fuse_boxes(1500.0, n_apart=1, tool_x=200.0) == pytest.approx(1040.0)   # a grounded box, meant to stand apart
        not_counted("2 solid(s) from 1, 40.0 cm^3 added by 40.0 cm^3 of tools", x0=1900.0, tool_x=200.0)
        with pytest.raises(B._FuseNotCounted, match=r"1 solid\(s\) from 1, 0\.0 cm\^3 added by 8\.0 cm\^3 of tools"):
            B._fuse_keeping(gmsh, 'dee0', [(3, occ.addBox(2300, 0, 0, 100, 100, 100))],
                            [(3, occ.addBox(2340, 40, 40, 20, 20, 20))], lambda solids: [])              # a tool inside
    finally:
        monkeypatch.undo()
        gmsh.option.setNumber("Geometry.ToleranceBoolean", tol0)
        gmsh.model.remove()
        if started:
            gmsh.finalize()


def test_a_dropped_piece_is_found_with_its_centre_inside_the_other():
    """A post split at a bend is two overlapping pieces. A fuse that drops one leaves its centre in metal when that lies in
    the other piece (gap 2 of evals 0008 and 0083 built that way): the footprint's outline has to be in metal too. A window
    block is tested at its own mid-height, a plate with a round hole only clear of the hole."""
    B = pytest.importorskip('PyCentralRegion.geometry.builder')
    gmsh = pytest.importorskip('gmsh')
    started = not gmsh.isInitialized()
    if started:
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("lost_blocks")
    occ = gmsh.model.occ

    def block(name, x0, x1, **kw):
        return {'name': name, 'corners': [[x0, 0.0], [x1, 0.0], [x1, 10.0], [x0, 10.0]], 'half_height': 5.0, **kw}

    def solid(x0, x1, z0=-5.0, dz=10.0):
        return occ.addBox(x0, 0, z0, x1 - x0, 10, dz)

    try:
        stayed, dropped = block('stayed', 0.0, 20.0), block('dropped', 12.0, 24.0)       # the centre 18 lies in the other
        assert 0.0 < B._block_points(dropped)[0, 0] - 12.0 < 8.0 and len(B._block_points(dropped)) == 5
        assert (B._block_points(dropped)[:, 2] == 0.0).all()

        only_one = solid(0.0, 20.0)
        occ.synchronize()
        assert B._lost_blocks(gmsh, [stayed], [only_one]) == []
        assert B._lost_blocks(gmsh, [stayed, dropped], [only_one]) == ['dropped (23.10, 0.75, 0.00; 2 of 5 points)']   # 18 + 0.85 * 6, 5 - 0.85 * 5
        assert B._lost_blocks(gmsh, [block('all gone', 50.0, 60.0)], [only_one]) == ['all gone (55.00, 5.00, 0.00; 5 of 5 points)']

        both = occ.fuse([(3, solid(100.0, 120.0))], [(3, solid(112.0, 124.0))], removeObject=True, removeTool=True)[0][0][1]
        occ.synchronize()
        assert B._lost_blocks(gmsh, [block('a', 100.0, 120.0), block('b', 112.0, 124.0)], [both]) == []

        up = solid(200.0, 210.0, z0=5.0, dz=10.0)                                        # above the opening, z 5 .. 15
        occ.synchronize()
        win = block('win', 200.0, 210.0, kind='window', z_lo=5.0, z_hi=15.0)
        assert (B._block_points(win)[:, 2] == 10.0).all() and B._lost_blocks(gmsh, [win], [up]) == []
        assert B._lost_blocks(gmsh, [block('win', 200.0, 210.0, kind='window', z_lo=-15.0, z_hi=-5.0)], [up]) == \
            ['win (205.00, 5.00, -10.00; 5 of 5 points)']

        def plate(half_chord):                                                           # centred on the hole, 10 mm deep
            return block('plate', 320.0 - half_chord, 320.0 + half_chord, kind='aperture', z_lo=-15.0, z_hi=15.0,
                         hole_centre=np.array([320.0, 0.0, 0.0]), hole_axis=np.array([0.0, 1.0]), hole_r=10.0)

        holed = occ.cut([(3, occ.addBox(300.0, 0, -15.0, 40, 10, 30))], [(3, occ.addCylinder(320.0, -1.0, 0, 0, 12.0, 0, 10.0))],
                        removeObject=True, removeTool=True)[0][0][1]
        occ.synchronize()
        assert B._block_points(plate(20.0)) == pytest.approx(np.array(
            [[303.0, 0.75, 0.0], [337.0, 0.75, 0.0], [337.0, 9.25, 0.0], [303.0, 9.25, 0.0]]))  # the centre is the hole
        assert B._lost_blocks(gmsh, [plate(20.0)], [holed]) == []
        assert B._lost_blocks(gmsh, [plate(20.0)], [only_one]) == ['plate (303.00, 0.75, 0.00; 4 of 4 points)']
        assert len(B._block_points(plate(11.0))) == 0 and B._lost_blocks(gmsh, [plate(11.0)], [only_one]) == []  # all hole

        notch ={'name': 'L', 'half_height': 5.0, 'corners': [[0, 0], [10, 0], [10, 1], [1, 1], [1, 10], [0, 10]]}
        p = B._block_points(notch)                                                       # not convex: only points on it
        assert len(p) < 7 and all(x <= 10.0 and y <= 10.0 and (x <= 1.0 or y <= 1.0) for x, y, _ in p)
    finally:
        gmsh.model.remove()
        if started:
            gmsh.finalize()


def test_a_fuse_that_does_not_count_starts_the_model_again(monkeypatch):
    """build_model passes its keywords on and starts again from nothing with the failing owner's fuse at the next smaller
    tolerance, the other owners' as they were. A model that counts the first time is one build with no tolerance set;
    a fuse that counts at no tolerance fails, naming each; with exact booleans there is nothing smaller to try."""
    B = pytest.importorskip('PyCentralRegion.geometry.builder')
    dataclasses = pytest.importorskip('dataclasses')
    monkeypatch.setattr(B, 'OPT', dataclasses.replace(B.OPT, occ_fuzzy_mm=0.05))
    calls = []

    def fake(fails):
        def build(foot, dummy, sim, brick, dees, posts=None, verbose=True, fuse_tol=None, **kw):
            calls.append((dict(fuse_tol), kw))
            for owner, smallest_that_counts in fails.items():
                if fuse_tol.get(owner, 0.05) > smallest_that_counts:
                    raise B._FuseNotCounted(owner, 'lost g1')
            return 'the model'
        return build

    monkeypatch.setattr(B, '_build_model', fake({}))
    assert B.build_model(1, 2, 3, 4, 5, verbose=False, lc=8.0) == 'the model'
    assert calls == [({}, {'lc': 8.0})]

    calls.clear()
    monkeypatch.setattr(B, '_build_model', fake({'dee0': 0.01, 'ground': 0.02}))
    assert B.build_model(1, 2, 3, 4, 5, verbose=False) == 'the model'
    assert [c[0] for c in calls] == [{}, {'dee0': 0.02}, {'dee0': 0.01}, {'dee0': 0.01, 'ground': 0.02}]

    monkeypatch.setattr(B, '_build_model', fake({'dee1': -1.0}))
    with pytest.raises(RuntimeError, match=r"the fuse into dee1: none counted at boolean tolerance 0\.05 mm: lost g1; "
                                           r"0\.02 mm: lost g1; 0\.01 mm: lost g1; 0\.005 mm: lost g1; 0 mm: lost g1"):
        B.build_model(1, 2, 3, 4, 5, verbose=False)

    calls.clear()
    monkeypatch.setattr(B, 'OPT', dataclasses.replace(B.OPT, occ_fuzzy_mm=0.0))
    with pytest.raises(RuntimeError, match=r"the fuse into dee1: none counted at boolean tolerance 0 mm: lost g1"):
        B.build_model(1, 2, 3, 4, 5, verbose=False)
    assert len(calls) == 1


def test_a_post_block_split_at_a_bend_is_fused(tmp_path):
    """Eval 0003 of a connected campaign with posts by the between-band rule: at 0.05 mm the fuse into dee0 dropped both
    pieces of gap 1's block, split at a bend of the gap line, which the builder's own check caught and stopped the
    build. It builds now (about 30 s, when the design's input files are on this machine, else it skips)."""
    pytest.importorskip('PyCentralRegion.geometry.builder')
    from PyCentralRegion.geometry import build as GB
    cfg_path = EXAMPLES / 'design_connected_cold.toml'
    if not cfg_path.exists():
        pytest.skip("the worked example is not in this checkout")
    r1 = (104.5, 108.5, 110.5, 112.0, 120.5, 130.5, 148.5, 157.0)
    r2 = (159.5, 156.0, 153.5, 156.0, 168.0, 179.5, 196.0, 202.0)
    posts = [(1, 'dee', 58.584, 22.457), (1, 'dee', 67.009, 13.361), (1, 'dummy', 84.694, 19.095),
             (1, 'dummy', 92.316, 15.117), (2, 'dee', 80.691, 11.988), (2, 'dummy', 96.915, 13.157),
             (3, 'dee', 78.366, 10.163), (3, 'dummy', 92.84, 10.579), (4, 'dummy', 86.118, 11.091)]
    model = tmp_path / 'model3d.pkl'
    overrides = ['machine.dee_opening_deg=32.547930952023954', 'geometry.aperture_inner_mm=26.0',
                 'geometry.aperture_outer_mm=32.0', 'geometry.aperture_taper_center_mm=120.0',
                 'geometry.aperture_taper_length_mm=40.0', 'geometry.post_shape="cylinder"',
                 'geometry.posts=[' + ', '.join(f'{{gap={g}, side="{s}", s_mm={m}, length_mm={n}}}'
                                                for g, s, m, n in posts) + ']', f'paths.model3d="{model.as_posix()}"']
    for g, (a, b) in enumerate(zip(r1, r2)):
        overrides += [f'geometry.gap_line.{g}.r1_mm={a}', f'geometry.gap_line.{g}.f1=0.0',
                      f'geometry.gap_line.{g}.r2_mm={b}'] + ([f'geometry.gap_line.{g}.f0=0.0'] if g else [])
    cfg = C.load(cfg_path, overrides=overrides)
    pc = C.section(cfg, 'paths')
    if not all(Path(p).exists() for p in (pc.seed, pc.housing_outline, pc.handoff, pc.housing_step)):
        pytest.skip("the design's input files are not on this machine")
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        GB.build(cfg)
    txt = log.getvalue()
    notes = [ln for ln in txt.splitlines() if 'the fuse into' in ln]
    assert model.exists() and "into dee0 -> 1 solid(s)" in txt, "\n".join(notes)
