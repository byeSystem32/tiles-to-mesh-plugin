"""
GLB export utilities.

Serializes Mesh objects to the binary glTF (GLB) format for use in
3D software, game engines, and web viewers.
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def export_glb(mesh, path: str, include_textures: bool = True) -> Path:
    """Export a Mesh to a GLB file.

    Args:
        mesh: A tiles_to_mesh.Mesh instance.
        path: Output file path (should end in .glb).
        include_textures: Whether to include texture images. Default True.

    Returns:
        Path to the exported file.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    glb_bytes = mesh_to_glb_bytes(mesh, include_textures=include_textures)

    with open(out_path, "wb") as f:
        f.write(glb_bytes)

    print(f"✓ Exported to {out_path} ({len(glb_bytes) / 1024 / 1024:.1f} MB)")
    return out_path


def mesh_to_glb_bytes(mesh, include_textures: bool = True) -> bytes:
    """Serialize a Mesh to GLB format bytes.

    This creates a valid GLB (binary glTF 2.0) file containing:
    - Vertex positions
    - Vertex normals (if present)
    - Texture coordinates (if present)
    - Triangle indices
    - Base color texture (if present and include_textures is True)

    Args:
        mesh: A tiles_to_mesh.Mesh instance.
        include_textures: Whether to include textures.

    Returns:
        Raw GLB bytes.
    """
    # Build the binary buffer
    bin_parts: List[bytes] = []
    buffer_views: List[Dict] = []
    accessors: List[Dict] = []
    current_offset = 0

    # ── Vertex positions ──
    positions = mesh.vertices.astype(np.float32)
    pos_bytes = positions.tobytes()
    pos_min = positions.min(axis=0).tolist()
    pos_max = positions.max(axis=0).tolist()

    buffer_views.append({
        "buffer": 0,
        "byteOffset": current_offset,
        "byteLength": len(pos_bytes),
        "target": 34962,  # ARRAY_BUFFER
    })
    accessors.append({
        "bufferView": len(buffer_views) - 1,
        "componentType": 5126,  # FLOAT
        "count": len(positions),
        "type": "VEC3",
        "min": pos_min,
        "max": pos_max,
    })
    pos_accessor_idx = len(accessors) - 1
    bin_parts.append(pos_bytes)
    current_offset += len(pos_bytes)
    # Align to 4 bytes
    padding = (4 - (current_offset % 4)) % 4
    bin_parts.append(b"\x00" * padding)
    current_offset += padding

    # ── Vertex normals ──
    norm_accessor_idx = None
    if mesh.normals is not None and len(mesh.normals) == len(mesh.vertices):
        normals = mesh.normals.astype(np.float32)
        norm_bytes = normals.tobytes()

        buffer_views.append({
            "buffer": 0,
            "byteOffset": current_offset,
            "byteLength": len(norm_bytes),
            "target": 34962,
        })
        accessors.append({
            "bufferView": len(buffer_views) - 1,
            "componentType": 5126,
            "count": len(normals),
            "type": "VEC3",
        })
        norm_accessor_idx = len(accessors) - 1
        bin_parts.append(norm_bytes)
        current_offset += len(norm_bytes)
        padding = (4 - (current_offset % 4)) % 4
        bin_parts.append(b"\x00" * padding)
        current_offset += padding

    # ── Texture coordinates ──
    uv_accessor_idx = None
    if mesh.texcoords is not None and len(mesh.texcoords) > 0:
        texcoords = mesh.texcoords.astype(np.float32)
        uv_bytes = texcoords.tobytes()

        buffer_views.append({
            "buffer": 0,
            "byteOffset": current_offset,
            "byteLength": len(uv_bytes),
            "target": 34962,
        })
        accessors.append({
            "bufferView": len(buffer_views) - 1,
            "componentType": 5126,
            "count": len(texcoords),
            "type": "VEC2",
        })
        uv_accessor_idx = len(accessors) - 1
        bin_parts.append(uv_bytes)
        current_offset += len(uv_bytes)
        padding = (4 - (current_offset % 4)) % 4
        bin_parts.append(b"\x00" * padding)
        current_offset += padding

    # ── Indices ──
    indices = mesh.faces.astype(np.uint32).flatten()
    idx_bytes = indices.tobytes()

    buffer_views.append({
        "buffer": 0,
        "byteOffset": current_offset,
        "byteLength": len(idx_bytes),
        "target": 34963,  # ELEMENT_ARRAY_BUFFER
    })
    accessors.append({
        "bufferView": len(buffer_views) - 1,
        "componentType": 5125,  # UNSIGNED_INT
        "count": len(indices),
        "type": "SCALAR",
    })
    idx_accessor_idx = len(accessors) - 1
    bin_parts.append(idx_bytes)
    current_offset += len(idx_bytes)
    padding = (4 - (current_offset % 4)) % 4
    bin_parts.append(b"\x00" * padding)
    current_offset += padding

    # ── Texture image ──
    images = []
    textures_gltf = []
    materials_gltf = []
    tex_bv_idx = None

    if include_textures and mesh.textures:
        tex = mesh.textures[0]
        tex_data = tex["data"]
        mime = tex.get("mime", "image/jpeg")

        buffer_views.append({
            "buffer": 0,
            "byteOffset": current_offset,
            "byteLength": len(tex_data),
        })
        tex_bv_idx = len(buffer_views) - 1

        images.append({
            "bufferView": tex_bv_idx,
            "mimeType": mime,
        })
        textures_gltf.append({
            "source": 0,
        })
        materials_gltf.append({
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
        })

        bin_parts.append(tex_data if isinstance(tex_data, bytes) else bytes(tex_data))
        current_offset += len(tex_data)
        padding = (4 - (current_offset % 4)) % 4
        bin_parts.append(b"\x00" * padding)
        current_offset += padding
    else:
        materials_gltf.append({
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.56, 0.66, 0.78, 1.0],
                "metallicFactor": 0.1,
                "roughnessFactor": 0.7,
            },
        })

    # ── Build primitive attributes ──
    attributes = {"POSITION": pos_accessor_idx}
    if norm_accessor_idx is not None:
        attributes["NORMAL"] = norm_accessor_idx
    if uv_accessor_idx is not None:
        attributes["TEXCOORD_0"] = uv_accessor_idx

    # ── Assemble glTF JSON ──
    gltf_json = {
        "asset": {
            "version": "2.0",
            "generator": "tiles-to-mesh",
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": attributes,
                "indices": idx_accessor_idx,
                "material": 0,
            }],
        }],
        "materials": materials_gltf,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": current_offset}],
    }

    if images:
        gltf_json["images"] = images
    if textures_gltf:
        gltf_json["textures"] = textures_gltf

    # ── Serialize to GLB ──
    json_bytes = json.dumps(gltf_json, separators=(",", ":")).encode("utf-8")
    # Pad JSON to 4-byte alignment
    json_padding = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * json_padding

    bin_data = b"".join(bin_parts)

    # GLB structure:
    # Header: magic(4) + version(4) + length(4) = 12 bytes
    # JSON chunk: length(4) + type(4) + data
    # BIN chunk: length(4) + type(4) + data
    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)

    glb = io.BytesIO()

    # Header
    glb.write(struct.pack("<III", 0x46546C67, 2, total_length))

    # JSON chunk
    glb.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    glb.write(json_bytes)

    # BIN chunk
    glb.write(struct.pack("<II", len(bin_data), 0x004E4942))
    glb.write(bin_data)

    return glb.getvalue()
