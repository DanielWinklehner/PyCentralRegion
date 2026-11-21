"""The central region's 3D electrodes: dees, grounded dummy dees and spiral-inflector housing, posts, windows, round
holes and collimator boxes, built in gmsh / OpenCASCADE from the seed's gap lines and meshed for the BEM.

    options   GeometryConfig, the [geometry] section (standard library only)
    builder   the construction: gap lines -> 2D boundaries -> OCC solids -> surface mesh -> ElectrodeModel
    build     build(cfg) for a run config, and the tools that work on the same edges (gap-0 hole, edge shift)
    views     printed reports and plots of the layout, and gmsh / matplotlib views of built models and tracks

Only ``options`` is imported here, so reading a run config needs neither numpy nor gmsh.
"""
from .options import GeometryConfig

__all__ = ['GeometryConfig']
