"""Tests for GLB export functionality."""

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tiles_to_mesh.export import export_glb, mesh_to_glb_bytes
from tiles_to_mesh.mesh import Mesh


def _make_simple_mesh():
    """Create a simple quad mesh (2 triangles) for export testing."""
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.uint32)
    normals = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ], dtype=np.float32)
    texcoords = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ], dtype=np.float32)
    return Mesh(vertices=vertices, faces=faces, normals=normals, texcoords=texcoords)


class TestGLBExport:
    """Tests for GLB serialization."""

    def test_glb_magic_number(self):
        mesh = _make_simple_mesh()
        glb = mesh_to_glb_bytes(mesh)
        magic = struct.unpack_from("<I", glb, 0)[0]
        assert magic == 0x46546C67  # 'glTF'

    def test_glb_version(self):
        mesh = _make_simple_mesh()
        glb = mesh_to_glb_bytes(mesh)
        version = struct.unpack_from("<I", glb, 4)[0]
        assert version == 2

    def test_glb_length_matches(self):
        mesh = _make_simple_mesh()
        glb = mesh_to_glb_bytes(mesh)
        declared_length = struct.unpack_from("<I", glb, 8)[0]
        assert declared_length == len(glb)

    def test_glb_has_json_chunk(self):
        mesh = _make_simple_mesh()
        glb = mesh_to_glb_bytes(mesh)
        # First chunk after 12-byte header should be JSON
        chunk_type = struct.unpack_from("<I", glb, 16)[0]
        assert chunk_type == 0x4E4F534A  # 'JSON'

    def test_glb_json_valid(self):
        mesh = _make_simple_mesh()
        glb = mesh_to_glb_bytes(mesh)
        chunk_length = struct.unpack_from("<I", glb, 12)[0]
        json_data = glb[20:20 + chunk_length]
        gltf = json.loads(json_data)
        assert gltf["asset"]["version"] == "2.0"
        assert gltf["asset"]["generator"] == "tiles-to-mesh"
        assert len(gltf["meshes"]) == 1
        assert "POSITION" in gltf["meshes"][0]["primitives"][0]["attributes"]

    def test_export_to_file(self, tmp_path):
        mesh = _make_simple_mesh()
        out_path = tmp_path / "test_output.glb"
        result = export_glb(mesh, str(out_path))
        assert result.exists()
        assert result.stat().st_size > 0

        # Verify it's valid GLB
        with open(result, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            assert magic == 0x46546C67

    def test_export_creates_parent_dirs(self, tmp_path):
        mesh = _make_simple_mesh()
        out_path = tmp_path / "subdir" / "deep" / "test.glb"
        result = export_glb(mesh, str(out_path))
        assert result.exists()

    def test_glb_with_normals_and_uvs(self):
        mesh = _make_simple_mesh()
        glb = mesh_to_glb_bytes(mesh)
        chunk_length = struct.unpack_from("<I", glb, 12)[0]
        json_data = glb[20:20 + chunk_length]
        gltf = json.loads(json_data)
        attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
        assert "NORMAL" in attrs
        assert "TEXCOORD_0" in attrs

    def test_glb_without_normals(self):
        mesh = _make_simple_mesh()
        mesh.normals = None
        glb = mesh_to_glb_bytes(mesh)
        chunk_length = struct.unpack_from("<I", glb, 12)[0]
        json_data = glb[20:20 + chunk_length]
        gltf = json.loads(json_data)
        attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
        assert "NORMAL" not in attrs
