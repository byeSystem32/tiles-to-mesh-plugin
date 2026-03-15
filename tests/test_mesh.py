"""Tests for the Mesh class."""

import numpy as np
import pytest

from tiles_to_mesh.mesh import Mesh
from tiles_to_mesh.selector import Region


def _make_triangle_mesh(**overrides):
    """Create a simple triangle mesh for testing."""
    defaults = {
        "vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        "faces": np.array([[0, 1, 2]], dtype=np.uint32),
        "normals": np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32),
        "texcoords": np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
    }
    defaults.update(overrides)
    return Mesh(**defaults)


class TestMeshProperties:
    """Tests for Mesh properties."""

    def test_vertex_count(self):
        mesh = _make_triangle_mesh()
        assert mesh.vertex_count == 3

    def test_face_count(self):
        mesh = _make_triangle_mesh()
        assert mesh.face_count == 1

    def test_has_normals(self):
        mesh = _make_triangle_mesh()
        assert mesh.has_normals is True

        mesh_no_normals = _make_triangle_mesh(normals=None)
        assert mesh_no_normals.has_normals is False

    def test_has_texcoords(self):
        mesh = _make_triangle_mesh()
        assert mesh.has_texcoords is True

        mesh_no_uv = _make_triangle_mesh(texcoords=None)
        assert mesh_no_uv.has_texcoords is False

    def test_bounds(self):
        mesh = _make_triangle_mesh()
        bounds = mesh.bounds
        np.testing.assert_array_almost_equal(bounds["min"], [0, 0, 0])
        np.testing.assert_array_almost_equal(bounds["max"], [1, 1, 0])
        np.testing.assert_array_almost_equal(bounds["size"], [1, 1, 0])

    def test_repr(self):
        mesh = _make_triangle_mesh()
        r = repr(mesh)
        assert "vertices=3" in r
        assert "faces=1" in r


class TestMeshTransforms:
    """Tests for non-Blender mesh operations."""

    def test_center(self):
        mesh = _make_triangle_mesh(
            vertices=np.array([[10, 10, 10], [12, 10, 10], [10, 12, 10]], dtype=np.float32)
        )
        mesh.center()
        center = mesh.vertices.mean(axis=0)
        np.testing.assert_array_almost_equal(center, [0, 0, 0], decimal=5)

    def test_scale(self):
        mesh = _make_triangle_mesh()
        original = mesh.vertices.copy()
        mesh.scale(2.0)
        np.testing.assert_array_almost_equal(mesh.vertices, original * 2.0)

    def test_flip_normals(self):
        mesh = _make_triangle_mesh()
        original_faces = mesh.faces.copy()
        original_normals = mesh.normals.copy()
        mesh.flip_normals()
        # Faces should be reversed
        np.testing.assert_array_equal(mesh.faces, original_faces[:, ::-1])
        # Normals should be negated
        np.testing.assert_array_almost_equal(mesh.normals, -original_normals)

    def test_copy(self):
        mesh = _make_triangle_mesh()
        mesh_copy = mesh.copy()
        assert mesh_copy is not mesh
        assert mesh_copy.vertices is not mesh.vertices
        np.testing.assert_array_equal(mesh_copy.vertices, mesh.vertices)
        np.testing.assert_array_equal(mesh_copy.faces, mesh.faces)

    def test_chaining(self):
        mesh = _make_triangle_mesh()
        result = mesh.center().scale(2.0).flip_normals()
        assert result is mesh  # All operations return self

    def test_history_tracking(self):
        mesh = _make_triangle_mesh()
        mesh.center().scale(2.0)
        assert len(mesh._history) == 2
        assert "center()" in mesh._history
        assert "scale(factor=2.0)" in mesh._history


class TestMeshInfo:
    """Tests for mesh info display."""

    def test_info_returns_string(self):
        mesh = _make_triangle_mesh()
        info = mesh.info()
        assert "Vertices:" in info
        assert "Faces:" in info
        assert "3" in info
        assert "1" in info
