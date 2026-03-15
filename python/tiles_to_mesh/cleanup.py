"""
Blender-based mesh cleanup operations.

Uses headless Blender (bpy) to perform mesh processing operations.
Blender is lazy-loaded — only imported when a cleanup method is first called.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np

# Lazy-loaded bpy module
_bpy = None


def _get_bpy():
    """Lazy-load the Blender Python module."""
    global _bpy
    if _bpy is None:
        try:
            import bpy
            _bpy = bpy
        except ImportError:
            raise ImportError(
                "Blender Python module (bpy) is required for mesh cleanup operations.\n"
                "Install it with: pip install bpy\n"
                "Or install tiles-to-mesh with Blender support: pip install tiles-to-mesh[blender]"
            )
    return _bpy


def _clear_scene(bpy_module) -> None:
    """Clear all objects from the Blender scene."""
    bpy = bpy_module
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Clean up orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def _import_mesh(mesh_obj, bpy_module) -> Any:
    """Import a Mesh object into Blender as a new mesh object.

    Args:
        mesh_obj: A tiles_to_mesh.Mesh instance.
        bpy_module: The bpy module.

    Returns:
        The Blender object.
    """
    import bmesh

    bpy = bpy_module
    _clear_scene(bpy)

    # Create new mesh and object
    bpy_mesh = bpy.data.meshes.new("TTM_Mesh")
    bpy_obj = bpy.data.objects.new("TTM_Object", bpy_mesh)

    # Link to scene
    bpy.context.collection.objects.link(bpy_obj)
    bpy.context.view_layer.objects.active = bpy_obj
    bpy_obj.select_set(True)

    # Build mesh from numpy arrays
    vertices = mesh_obj.vertices.tolist()
    faces = mesh_obj.faces.tolist()

    bpy_mesh.from_pydata(vertices, [], faces)
    bpy_mesh.update()

    # Add normals if available
    if mesh_obj.normals is not None and len(mesh_obj.normals) == len(mesh_obj.vertices):
        bpy_mesh.normals_split_custom_set_from_vertices(mesh_obj.normals.tolist())

    # Add UV coordinates if available
    if mesh_obj.texcoords is not None and len(mesh_obj.texcoords) > 0:
        uv_layer = bpy_mesh.uv_layers.new(name="UVMap")
        for face in bpy_mesh.polygons:
            for loop_idx in face.loop_indices:
                vert_idx = bpy_mesh.loops[loop_idx].vertex_index
                if vert_idx < len(mesh_obj.texcoords):
                    uv_layer.data[loop_idx].uv = mesh_obj.texcoords[vert_idx].tolist()

    bpy_mesh.update()
    return bpy_obj


def _extract_mesh(bpy_obj, bpy_module) -> Dict[str, np.ndarray]:
    """Extract mesh data from a Blender object back to numpy arrays.

    Args:
        bpy_obj: A Blender mesh object.
        bpy_module: The bpy module.

    Returns:
        Dict with 'vertices', 'faces', and optionally 'normals' and 'texcoords'.
    """
    bpy = bpy_module

    # Apply all modifiers
    bpy.context.view_layer.objects.active = bpy_obj
    for modifier in bpy_obj.modifiers:
        try:
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        except Exception:
            pass  # Some modifiers may fail to apply

    # Get the evaluated mesh
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = bpy_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.data

    eval_mesh.calc_loop_triangles()

    # Extract vertices
    vertex_count = len(eval_mesh.vertices)
    vertices = np.empty(vertex_count * 3, dtype=np.float32)
    eval_mesh.vertices.foreach_get("co", vertices)
    vertices = vertices.reshape(-1, 3)

    # Extract faces (triangulated)
    tri_count = len(eval_mesh.loop_triangles)
    faces = np.empty(tri_count * 3, dtype=np.uint32)
    eval_mesh.loop_triangles.foreach_get("vertices", faces)
    faces = faces.reshape(-1, 3)

    # Extract normals
    normals = None
    try:
        normals = np.empty(vertex_count * 3, dtype=np.float32)
        eval_mesh.vertices.foreach_get("normal", normals)
        normals = normals.reshape(-1, 3)
    except Exception:
        pass

    # Extract UVs
    texcoords = None
    if eval_mesh.uv_layers:
        uv_layer = eval_mesh.uv_layers.active
        if uv_layer:
            # Build per-vertex UVs (average if multiple loops reference same vertex)
            texcoords = np.zeros((vertex_count, 2), dtype=np.float32)
            counts = np.zeros(vertex_count, dtype=np.int32)
            for loop in eval_mesh.loops:
                vi = loop.vertex_index
                uv = uv_layer.data[loop.index].uv
                texcoords[vi] += [uv[0], uv[1]]
                counts[vi] += 1
            mask = counts > 0
            texcoords[mask] /= counts[mask, np.newaxis]

    result = {
        "vertices": vertices,
        "faces": faces,
    }
    if normals is not None:
        result["normals"] = normals
    if texcoords is not None:
        result["texcoords"] = texcoords

    return result


def run_cleanup_operation(mesh_obj, operation: str, **kwargs) -> Dict[str, np.ndarray]:
    """Run a cleanup operation on a mesh using Blender.

    This is the main dispatch function for all Blender-based operations.

    Args:
        mesh_obj: A tiles_to_mesh.Mesh instance.
        operation: Operation name.
        **kwargs: Operation-specific parameters.

    Returns:
        Dict with updated 'vertices', 'faces', and optionally 'normals', 'texcoords'.
    """
    bpy = _get_bpy()
    bpy_obj = _import_mesh(mesh_obj, bpy)

    if operation == "decimate":
        _op_decimate(bpy_obj, bpy, **kwargs)
    elif operation == "fix_normals":
        _op_fix_normals(bpy_obj, bpy, **kwargs)
    elif operation == "remove_duplicates":
        _op_remove_duplicates(bpy_obj, bpy, **kwargs)
    elif operation == "remove_artifacts":
        _op_remove_artifacts(bpy_obj, bpy, **kwargs)
    elif operation == "merge_seams":
        _op_merge_seams(bpy_obj, bpy, **kwargs)
    elif operation == "smooth":
        _op_smooth(bpy_obj, bpy, **kwargs)
    else:
        raise ValueError(f"Unknown cleanup operation: {operation}")

    return _extract_mesh(bpy_obj, bpy)


# ── Individual Operations ────────────────────────────────────────────


def _op_decimate(bpy_obj, bpy, ratio: float = 0.5, **_) -> None:
    """Apply Decimate modifier to reduce polygon count."""
    modifier = bpy_obj.modifiers.new(name="Decimate", type='DECIMATE')
    modifier.ratio = ratio
    modifier.decimate_type = 'COLLAPSE'


def _op_fix_normals(bpy_obj, bpy, **_) -> None:
    """Recalculate normals to point outward consistently."""
    bpy.context.view_layer.objects.active = bpy_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')


def _op_remove_duplicates(bpy_obj, bpy, threshold: float = 0.0001, **_) -> None:
    """Merge vertices that are closer than threshold."""
    bpy.context.view_layer.objects.active = bpy_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=threshold)
    bpy.ops.object.mode_set(mode='OBJECT')


def _op_remove_artifacts(bpy_obj, bpy, threshold: float = 0.1, **_) -> None:
    """Remove small disconnected mesh fragments.

    Selects each disconnected island, and removes any that have fewer
    than threshold * total_faces faces.
    """
    import bmesh

    bpy.context.view_layer.objects.active = bpy_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(bpy_obj.data)
    bm.faces.ensure_lookup_table()
    total_faces = len(bm.faces)
    min_faces = int(total_faces * threshold)

    # Find connected components
    visited = set()
    islands = []

    for face in bm.faces:
        if face.index in visited:
            continue
        island = set()
        stack = [face]
        while stack:
            f = stack.pop()
            if f.index in visited:
                continue
            visited.add(f.index)
            island.add(f.index)
            for edge in f.edges:
                for linked_face in edge.link_faces:
                    if linked_face.index not in visited:
                        stack.append(linked_face)
        islands.append(island)

    # Remove small islands
    faces_to_remove = set()
    for island in islands:
        if len(island) < min_faces:
            faces_to_remove.update(island)

    if faces_to_remove:
        bm.faces.ensure_lookup_table()
        for idx in faces_to_remove:
            bm.faces[idx].select = True

        # Delete selected faces
        selected = [f for f in bm.faces if f.select]
        bmesh.ops.delete(bm, geom=selected, context='FACES')
        bmesh.update_edit_mesh(bpy_obj.data)

    bpy.ops.object.mode_set(mode='OBJECT')


def _op_merge_seams(bpy_obj, bpy, threshold: float = 0.001, **_) -> None:
    """Merge vertices at tile seam boundaries.

    Similar to remove_duplicates but specifically targets border vertices.
    """
    import bmesh

    bpy.context.view_layer.objects.active = bpy_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(bpy_obj.data)

    # Select border vertices (vertices on boundary edges)
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold(
        extend=False,
        use_wire=True,
        use_boundary=True,
        use_multi_face=False,
        use_non_contiguous=False,
        use_verts=True,
    )

    # Merge close border vertices
    bpy.ops.mesh.remove_doubles(threshold=threshold)
    bpy.ops.object.mode_set(mode='OBJECT')


def _op_smooth(bpy_obj, bpy, iterations: int = 2, factor: float = 0.5, **_) -> None:
    """Apply Laplacian smoothing."""
    modifier = bpy_obj.modifiers.new(name="Smooth", type='SMOOTH')
    modifier.iterations = iterations
    modifier.factor = factor
