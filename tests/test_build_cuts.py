"""The builder's cuts that must remove metal (geometry.builder._cut_removing). A fuzzy OCC boolean can hand back its
object untouched; on 2026-11-03 that left dees and the star ground whole whenever the channel taper ended beyond
~150 mm, and a retry at a smaller tolerance could leave a sealed void of the tool's shape instead of a channel. The
first test uses boxes; the second builds and meshes two such designs (about 15 s each) when their input files are on
this machine, else it skips."""
import contextlib
import io
import signal
from pathlib import Path

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
