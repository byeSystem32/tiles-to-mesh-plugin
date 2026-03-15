"""
Core Mesh class.

Wraps geometry data (vertices, faces, normals, UVs, textures) and provides
chainable cleanup/processing methods powered by Blender (bpy) and Rust.
"""

from __future__ import annotations

import io
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tiles_to_mesh.selector import Region


class Mesh:
    """A 3D mesh with geometry, normals, texture coordinates, and optional textures.

    All cleanup methods return ``self`` for chaining::

        mesh.decimate(ratio=0.5).fix_normals().remove_artifacts()

    Args:
        vertices: Nx3 float32 array of vertex positions.
        faces: Mx3 uint32 array of triangle face indices.
        normals: Optional Nx3 float32 array of vertex normals.
        texcoords: Optional Nx2 float32 array of UV coordinates.
        textures: Optional list of texture dicts with 'data' (bytes) and 'mime' (str).
        region: The geographic Region this mesh was fetched from.
        mode: Fetch mode ("photorealistic" or "geometry").
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: Optional[np.ndarray] = None,
        texcoords: Optional[np.ndarray] = None,
        textures: Optional[List[Dict[str, Any]]] = None,
        region: Optional[Region] = None,
        mode: str = "photorealistic",
    ):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.uint32)
        self.normals = np.asarray(normals, dtype=np.float32) if normals is not None else None
        self.texcoords = np.asarray(texcoords, dtype=np.float32) if texcoords is not None else None
        self.textures = textures or []
        self.region = region
        self.mode = mode
        self._history: List[str] = []

    # ── Properties ───────────────────────────────────────────────────

    @property
    def vertex_count(self) -> int:
        """Number of vertices in the mesh."""
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        """Number of triangular faces in the mesh."""
        return len(self.faces)

    @property
    def has_normals(self) -> bool:
        """Whether the mesh has vertex normals."""
        return self.normals is not None and len(self.normals) > 0

    @property
    def has_texcoords(self) -> bool:
        """Whether the mesh has texture coordinates."""
        return self.texcoords is not None and len(self.texcoords) > 0

    @property
    def has_textures(self) -> bool:
        """Whether the mesh has associated texture images."""
        return len(self.textures) > 0

    @property
    def bounds(self) -> Dict[str, np.ndarray]:
        """Axis-aligned bounding box of the mesh geometry."""
        return {
            "min": self.vertices.min(axis=0),
            "max": self.vertices.max(axis=0),
            "size": self.vertices.max(axis=0) - self.vertices.min(axis=0),
            "center": (self.vertices.max(axis=0) + self.vertices.min(axis=0)) / 2,
        }

    # ── Info ─────────────────────────────────────────────────────────

    def info(self) -> str:
        """Print a summary of the mesh.

        Returns:
            A formatted string with mesh statistics.
        """
        b = self.bounds
        size = b["size"]
        info_str = (
            f"+-- Mesh Info -----------------------------\n"
            f"|  Vertices:   {self.vertex_count:>12,}\n"
            f"|  Faces:      {self.face_count:>12,}\n"
            f"|  Normals:    {'Yes' if self.has_normals else 'No':>12}\n"
            f"|  UVs:        {'Yes' if self.has_texcoords else 'No':>12}\n"
            f"|  Textures:   {len(self.textures):>12}\n"
            f"|  Mode:       {self.mode:>12}\n"
            f"|  Bounds:     {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f}\n"
            f"|  Region:     {self.region if self.region else 'N/A'}\n"
            f"|  Operations: {len(self._history)}\n"
            f"+--------------------------------------------"
        )
        print(info_str)
        return info_str

    def __repr__(self) -> str:
        return (
            f"Mesh(vertices={self.vertex_count:,}, faces={self.face_count:,}, "
            f"textured={self.has_textures}, mode='{self.mode}')"
        )

    # ── Cleanup Operations (Blender-powered) ─────────────────────────

    def decimate(self, ratio: float = 0.5) -> "Mesh":
        """Reduce polygon count using Blender's Decimate modifier.

        Args:
            ratio: Target ratio of faces to keep (0.0 to 1.0). Default 0.5 (50%).

        Returns:
            self (for chaining).
        """
        if not 0.0 < ratio <= 1.0:
            raise ValueError("Decimate ratio must be between 0.0 (exclusive) and 1.0 (inclusive).")

        self._run_blender_operation("decimate", ratio=ratio)
        self._history.append(f"decimate(ratio={ratio})")
        return self

    def fix_normals(self) -> "Mesh":
        """Recalculate and unify vertex normals using Blender.

        Returns:
            self (for chaining).
        """
        self._run_blender_operation("fix_normals")
        self._history.append("fix_normals()")
        return self

    def remove_duplicates(self, threshold: float = 0.0001) -> "Mesh":
        """Merge duplicate/close vertices using Blender.

        Args:
            threshold: Distance threshold for merging vertices. Default 0.0001.

        Returns:
            self (for chaining).
        """
        self._run_blender_operation("remove_duplicates", threshold=threshold)
        self._history.append(f"remove_duplicates(threshold={threshold})")
        return self

    def remove_artifacts(self, threshold: float = 0.1) -> "Mesh":
        """Remove small disconnected mesh fragments.

        Args:
            threshold: Size threshold as fraction of total mesh. Fragments
                smaller than this fraction are removed. Default 0.1 (10%).

        Returns:
            self (for chaining).
        """
        self._run_blender_operation("remove_artifacts", threshold=threshold)
        self._history.append(f"remove_artifacts(threshold={threshold})")
        return self

    def merge_seams(self, threshold: float = 0.001) -> "Mesh":
        """Heal seams between adjacent tiles by merging close border vertices.

        Args:
            threshold: Distance threshold for merging seam vertices.

        Returns:
            self (for chaining).
        """
        self._run_blender_operation("merge_seams", threshold=threshold)
        self._history.append(f"merge_seams(threshold={threshold})")
        return self

    def smooth(self, iterations: int = 2, factor: float = 0.5) -> "Mesh":
        """Apply Laplacian smoothing to the mesh.

        Args:
            iterations: Number of smoothing passes. Default 2.
            factor: Smoothing factor (0.0 to 1.0). Default 0.5.

        Returns:
            self (for chaining).
        """
        self._run_blender_operation("smooth", iterations=iterations, factor=factor)
        self._history.append(f"smooth(iterations={iterations}, factor={factor})")
        return self

    def center(self) -> "Mesh":
        """Move the mesh so its center of mass is at the origin.

        Returns:
            self (for chaining).
        """
        centroid = self.vertices.mean(axis=0)
        self.vertices -= centroid
        self._history.append("center()")
        return self

    def scale(self, factor: float) -> "Mesh":
        """Uniformly scale the mesh.

        Args:
            factor: Scale multiplier.

        Returns:
            self (for chaining).
        """
        self.vertices *= factor
        self._history.append(f"scale(factor={factor})")
        return self

    def flip_normals(self) -> "Mesh":
        """Flip all face normals (reverse winding order).

        Returns:
            self (for chaining).
        """
        self.faces = self.faces[:, ::-1]
        if self.normals is not None:
            self.normals = -self.normals
        self._history.append("flip_normals()")
        return self

    # ── Preview ──────────────────────────────────────────────────────

    def preview(self, **kwargs) -> Any:
        """Show an interactive 3D preview in the notebook.

        Args:
            **kwargs: Passed to MeshViewer (e.g., wireframe=True, width=800).

        Returns:
            The viewer widget.
        """
        from tiles_to_mesh.viewer import MeshViewer

        viewer = MeshViewer(self, **kwargs)
        return viewer.show()

    # ── Export ────────────────────────────────────────────────────────

    def export(self, path: str, **kwargs) -> Path:
        """Export the mesh to a GLB file.

        Args:
            path: Output file path (should end in .glb).
            **kwargs: Additional export options.

        Returns:
            Path to the exported file.
        """
        from tiles_to_mesh.export import export_glb

        return export_glb(self, path, **kwargs)

    def to_glb_bytes(self) -> bytes:
        """Serialize the mesh to GLB format in memory.

        Returns:
            Raw GLB bytes.
        """
        from tiles_to_mesh.export import mesh_to_glb_bytes

        return mesh_to_glb_bytes(self)

    # ── Copy ─────────────────────────────────────────────────────────

    def copy(self) -> "Mesh":
        """Create a deep copy of this mesh.

        Returns:
            A new Mesh with copied data.
        """
        return Mesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
            texcoords=self.texcoords.copy() if self.texcoords is not None else None,
            textures=[dict(t) for t in self.textures],
            region=self.region,
            mode=self.mode,
        )

    # ── Internal: Blender Operations ─────────────────────────────────

    def _run_blender_operation(self, operation: str, **kwargs) -> None:
        """Execute a mesh operation via headless Blender (bpy).

        Imports the mesh into Blender, applies the operation, and
        extracts the result back into numpy arrays.
        """
        from tiles_to_mesh.cleanup import run_cleanup_operation

        result = run_cleanup_operation(self, operation, **kwargs)
        self.vertices = result["vertices"]
        self.faces = result["faces"]
        self.normals = result.get("normals", self.normals)
        self.texcoords = result.get("texcoords", self.texcoords)
