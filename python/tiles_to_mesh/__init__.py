"""
tiles-to-mesh: Fetch Google 3D Tiles, build meshes, clean up, and preview in Jupyter/Colab.

Usage::

    import tiles_to_mesh as ttm

    # 1. Interactive map selection
    selector = ttm.MapSelector(api_key="YOUR_KEY")
    selector.show()

    # 2. Fetch mesh
    mesh = ttm.fetch_mesh(selector.region, api_key="YOUR_KEY", lod=3)

    # 3. Clean up
    mesh.decimate(ratio=0.5)
    mesh.fix_normals()

    # 4. Preview
    mesh.preview()

    # 5. Export
    mesh.export("output.glb")
"""

__version__ = "0.1.0"

from tiles_to_mesh.selector import MapSelector, Region
from tiles_to_mesh.fetcher import fetch_mesh, fetch_mesh_async
from tiles_to_mesh.mesh import Mesh
from tiles_to_mesh.viewer import MeshViewer
from tiles_to_mesh.export import export_glb

__all__ = [
    "MapSelector",
    "Region",
    "fetch_mesh",
    "fetch_mesh_async",
    "Mesh",
    "MeshViewer",
    "export_glb",
]
