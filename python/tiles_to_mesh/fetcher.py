"""
Tile fetching pipeline.

Orchestrates the Rust core to fetch Google 3D Tiles for a selected region,
merge them into a unified mesh, and apply coordinate transforms.
"""

from __future__ import annotations

import io
import json
import struct
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from tiles_to_mesh.selector import Region
from tiles_to_mesh.mesh import Mesh

# Try importing the Rust extension; fall back to pure-Python if not available
try:
    from tiles_to_mesh._tiles_to_mesh_rs import tiles as _rs_tiles
    from tiles_to_mesh._tiles_to_mesh_rs import mesh as _rs_mesh
    from tiles_to_mesh._tiles_to_mesh_rs import geo as _rs_geo

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

import requests


# ── Constants ────────────────────────────────────────────────────────

TILES_API_BASE = "https://tile.googleapis.com/v1/3dtiles"
SESSION_API_BASE = "https://tile.googleapis.com/v1"

# LOD level → geometric error threshold
LOD_THRESHOLDS = {
    1: 500.0,
    2: 100.0,
    3: 30.0,
    4: 10.0,
    5: 2.0,
}


def fetch_mesh(
    region: Union[Region, List[Tuple[float, float]]],
    api_key: str,
    lod: int = 3,
    mode: str = "photorealistic",
    max_concurrent: int = 10,
    progress: bool = True,
) -> Mesh:
    """Fetch 3D tile data from Google and assemble it into a Mesh.

    This is the primary entry point for converting a geographic region
    into a 3D mesh.

    Args:
        region: A Region object or list of (lat, lng) polygon coordinates.
        api_key: Google Maps API key with 3D Tiles API enabled.
        lod: Level of detail, 1 (coarse) to 5 (maximum detail). Default 3.
        mode: Either "photorealistic" (textured) or "geometry" (untextured).
        max_concurrent: Maximum parallel HTTP requests. Default 10.
        progress: Show progress bar. Default True.

    Returns:
        A Mesh object containing the assembled 3D geometry.

    Raises:
        ValueError: If parameters are invalid.
        RuntimeError: If tile fetching fails.
    """
    if isinstance(region, (list, tuple)) and not isinstance(region, Region):
        region = Region.from_coords(list(region))

    if not 1 <= lod <= 5:
        raise ValueError("LOD must be between 1 and 5")
    if mode not in ("photorealistic", "geometry"):
        raise ValueError("Mode must be 'photorealistic' or 'geometry'")

    polygon = region.polygon

    if HAS_RUST:
        return _fetch_mesh_rust(polygon, api_key, lod, mode, max_concurrent, progress)
    else:
        return _fetch_mesh_python(polygon, api_key, lod, mode, max_concurrent, progress)


async def fetch_mesh_async(
    region: Union[Region, List[Tuple[float, float]]],
    api_key: str,
    lod: int = 3,
    mode: str = "photorealistic",
    max_concurrent: int = 10,
) -> Mesh:
    """Async version of fetch_mesh. Useful in notebook environments with event loops.

    Args:
        region: A Region object or list of (lat, lng) polygon coordinates.
        api_key: Google Maps API key.
        lod: Level of detail (1-5).
        mode: "photorealistic" or "geometry".
        max_concurrent: Maximum parallel requests.

    Returns:
        A Mesh object.
    """
    # For now, wrap the sync version. Can be made truly async later.
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: fetch_mesh(region, api_key, lod, mode, max_concurrent, progress=False)
    )


def _fetch_mesh_rust(
    polygon: List[Tuple[float, float]],
    api_key: str,
    lod: int,
    mode: str,
    max_concurrent: int,
    show_progress: bool,
) -> Mesh:
    """Fetch mesh using the Rust extension for maximum performance."""
    config = _rs_tiles.FetchConfig(
        api_key=api_key,
        lod=lod,
        max_concurrent=max_concurrent,
        mode=mode,
    )
    fetcher = _rs_tiles.TileFetcher(config)

    pbar = tqdm(desc="Fetching tiles", unit="tile") if show_progress else None

    def progress_cb(done: int, total: int):
        if pbar is not None:
            pbar.total = total
            pbar.n = done
            pbar.refresh()

    tile_meshes = fetcher.fetch_tiles_in_region(polygon, progress_cb)

    if pbar:
        pbar.close()

    if not tile_meshes:
        raise RuntimeError("No tiles were fetched for the selected region.")

    # Merge tiles into single mesh
    if show_progress:
        print(f"Merging {len(tile_meshes)} tiles...")
    merged = _rs_mesh.merge_tile_meshes(tile_meshes)

    # Build Mesh object
    positions = np.array(merged.positions, dtype=np.float32).reshape(-1, 3)
    normals = np.array(merged.normals, dtype=np.float32).reshape(-1, 3) if merged.normals else None
    texcoords = np.array(merged.texcoords, dtype=np.float32).reshape(-1, 2) if merged.texcoords else None
    indices = np.array(merged.indices, dtype=np.uint32).reshape(-1, 3)

    # Collect textures from individual tiles
    textures = []
    for tile in tile_meshes:
        if tile.has_texture:
            tex_bytes = tile.get_texture_bytes()
            if tex_bytes:
                textures.append({
                    "data": tex_bytes,
                    "mime": tile.texture_mime or "image/jpeg",
                })

    mesh = Mesh(
        vertices=positions,
        faces=indices,
        normals=normals,
        texcoords=texcoords,
        textures=textures if textures else None,
        region=Region(polygon=polygon),
        mode=mode,
    )

    if show_progress:
        print(f"✓ Mesh ready: {mesh.vertex_count:,} vertices, {mesh.face_count:,} faces")

    return mesh


def _fetch_mesh_python(
    polygon: List[Tuple[float, float]],
    api_key: str,
    lod: int,
    mode: str,
    max_concurrent: int,
    show_progress: bool,
) -> Mesh:
    """Pure-Python fallback for tile fetching (slower, no Rust required)."""
    session = requests.Session()

    # Step 1: Create a session token for the 3D Tiles API
    if show_progress:
        print("Creating API session...")

    root_url = f"{TILES_API_BASE}/root.json?key={api_key}"
    resp = session.get(root_url)
    resp.raise_for_status()
    root_tileset = resp.json()

    # Step 2: Traverse tileset tree and collect tile URLs
    target_error = LOD_THRESHOLDS.get(lod, 30.0)
    polygon_aabb = _compute_aabb(polygon)
    tile_urls = _collect_tile_urls(root_tileset.get("root", {}), polygon_aabb, target_error, api_key)

    if not tile_urls:
        raise RuntimeError(
            "No tiles found for the selected region. "
            "Check that the polygon coordinates are correct and the API key has 3D Tiles access."
        )

    if show_progress:
        print(f"Found {len(tile_urls)} tiles to fetch.")

    # Step 3: Fetch tile GLBs
    all_vertices = []
    all_normals = []
    all_texcoords = []
    all_indices = []
    all_textures = []
    vertex_offset = 0

    iterator = tqdm(tile_urls, desc="Fetching tiles", unit="tile") if show_progress else tile_urls

    for url in iterator:
        try:
            tile_resp = session.get(url, timeout=30)
            tile_resp.raise_for_status()
            glb_data = tile_resp.content

            mesh_data = _parse_glb_python(glb_data)
            if mesh_data is None:
                continue

            verts, norms, uvs, idxs, tex = mesh_data

            all_vertices.append(verts)
            if norms is not None:
                all_normals.append(norms)
            if uvs is not None:
                all_texcoords.append(uvs)

            # Offset indices
            all_indices.append(idxs + vertex_offset)
            vertex_offset += len(verts)

            if tex is not None:
                all_textures.append(tex)

        except Exception as e:
            if show_progress:
                print(f"  Warning: Failed to fetch tile: {e}")
            continue

    if not all_vertices:
        raise RuntimeError("Failed to fetch any tile data.")

    # Step 4: Assemble
    vertices = np.vstack(all_vertices)
    indices = np.vstack(all_indices)
    normals = np.vstack(all_normals) if all_normals else None
    texcoords = np.vstack(all_texcoords) if all_texcoords else None

    mesh = Mesh(
        vertices=vertices,
        faces=indices,
        normals=normals,
        texcoords=texcoords,
        textures=all_textures if all_textures else None,
        region=Region(polygon=polygon),
        mode=mode,
    )

    if show_progress:
        print(f"✓ Mesh ready: {mesh.vertex_count:,} vertices, {mesh.face_count:,} faces")

    return mesh


def _compute_aabb(polygon: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Compute axis-aligned bounding box from polygon. Returns (min_lat, max_lat, min_lng, max_lng)."""
    lats = [p[0] for p in polygon]
    lngs = [p[1] for p in polygon]
    return (min(lats), max(lats), min(lngs), max(lngs))


def _collect_tile_urls(
    node: Dict,
    aabb: Tuple[float, float, float, float],
    target_error: float,
    api_key: str,
) -> List[str]:
    """Recursively collect tile content URLs from the tileset tree."""
    urls = []

    # Check bounding volume
    bv = node.get("boundingVolume", {})
    if "region" in bv:
        region = bv["region"]
        if len(region) >= 4:
            import math
            west = math.degrees(region[0])
            south = math.degrees(region[1])
            east = math.degrees(region[2])
            north = math.degrees(region[3])
            min_lat, max_lat, min_lng, max_lng = aabb
            if east < min_lng or west > max_lng or north < min_lat or south > max_lat:
                return urls

    geometric_error = node.get("geometricError", 0.0)
    children = node.get("children", [])

    if geometric_error <= target_error or not children:
        content = node.get("content", {})
        uri = content.get("uri", "")
        if uri:
            if uri.startswith("http"):
                url = f"{uri}&key={api_key}" if "?" in uri else f"{uri}?key={api_key}"
            else:
                url = f"{TILES_API_BASE}/{uri}?key={api_key}"
            urls.append(url)
    else:
        for child in children:
            urls.extend(_collect_tile_urls(child, aabb, target_error, api_key))

    return urls


def _parse_glb_python(data: bytes):
    """Parse a GLB file in pure Python. Returns (vertices, normals, texcoords, indices, texture) or None."""
    if len(data) < 12:
        return None

    # GLB header
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != 0x46546C67:  # 'glTF'
        return None

    _version = struct.unpack_from("<I", data, 4)[0]
    _length = struct.unpack_from("<I", data, 8)[0]

    # Parse chunks
    offset = 12
    json_chunk = None
    bin_chunk = None

    while offset < len(data):
        if offset + 8 > len(data):
            break
        chunk_length = struct.unpack_from("<I", data, offset)[0]
        chunk_type = struct.unpack_from("<I", data, offset + 4)[0]
        chunk_data = data[offset + 8: offset + 8 + chunk_length]

        if chunk_type == 0x4E4F534A:  # JSON
            json_chunk = json.loads(chunk_data)
        elif chunk_type == 0x004E4942:  # BIN
            bin_chunk = chunk_data

        offset += 8 + chunk_length
        # Align to 4-byte boundary
        offset = (offset + 3) & ~3

    if json_chunk is None or bin_chunk is None:
        return None

    # Extract mesh data from glTF JSON + binary
    accessors = json_chunk.get("accessors", [])
    buffer_views = json_chunk.get("bufferViews", [])
    meshes = json_chunk.get("meshes", [])

    if not meshes:
        return None

    all_positions = []
    all_normals = []
    all_texcoords = []
    all_indices = []
    base_vertex = 0

    for mesh in meshes:
        for primitive in mesh.get("primitives", []):
            attrs = primitive.get("attributes", {})

            # Positions
            if "POSITION" in attrs:
                pos_data = _read_accessor(accessors[attrs["POSITION"]], buffer_views, bin_chunk)
                if pos_data is not None:
                    verts = np.frombuffer(pos_data, dtype=np.float32).reshape(-1, 3)
                    all_positions.append(verts)

            # Normals
            if "NORMAL" in attrs:
                norm_data = _read_accessor(accessors[attrs["NORMAL"]], buffer_views, bin_chunk)
                if norm_data is not None:
                    norms = np.frombuffer(norm_data, dtype=np.float32).reshape(-1, 3)
                    all_normals.append(norms)

            # Texcoords
            if "TEXCOORD_0" in attrs:
                uv_data = _read_accessor(accessors[attrs["TEXCOORD_0"]], buffer_views, bin_chunk)
                if uv_data is not None:
                    uvs = np.frombuffer(uv_data, dtype=np.float32).reshape(-1, 2)
                    all_texcoords.append(uvs)

            # Indices
            if "indices" in primitive:
                idx_accessor = accessors[primitive["indices"]]
                idx_data = _read_accessor(idx_accessor, buffer_views, bin_chunk)
                if idx_data is not None:
                    comp_type = idx_accessor.get("componentType", 5123)
                    if comp_type == 5121:  # UNSIGNED_BYTE
                        idxs = np.frombuffer(idx_data, dtype=np.uint8).astype(np.uint32)
                    elif comp_type == 5123:  # UNSIGNED_SHORT
                        idxs = np.frombuffer(idx_data, dtype=np.uint16).astype(np.uint32)
                    else:  # UNSIGNED_INT
                        idxs = np.frombuffer(idx_data, dtype=np.uint32)
                    all_indices.append(idxs.reshape(-1, 3) + base_vertex)

            if all_positions:
                base_vertex += len(all_positions[-1])

    if not all_positions:
        return None

    vertices = np.vstack(all_positions).astype(np.float32)
    normals = np.vstack(all_normals).astype(np.float32) if all_normals else None
    texcoords = np.vstack(all_texcoords).astype(np.float32) if all_texcoords else None
    indices = np.vstack(all_indices).astype(np.uint32) if all_indices else None

    # Extract texture (first image found)
    texture = None
    images = json_chunk.get("images", [])
    if images:
        img = images[0]
        if "bufferView" in img:
            bv = buffer_views[img["bufferView"]]
            bv_offset = bv.get("byteOffset", 0)
            bv_length = bv["byteLength"]
            texture = {
                "data": bin_chunk[bv_offset:bv_offset + bv_length],
                "mime": img.get("mimeType", "image/jpeg"),
            }

    return vertices, normals, texcoords, indices, texture


def _read_accessor(accessor: Dict, buffer_views: List[Dict], bin_data: bytes) -> Optional[bytes]:
    """Read raw bytes for a glTF accessor."""
    bv_index = accessor.get("bufferView")
    if bv_index is None:
        return None

    bv = buffer_views[bv_index]
    bv_offset = bv.get("byteOffset", 0)
    bv_length = bv["byteLength"]
    acc_offset = accessor.get("byteOffset", 0)

    start = bv_offset + acc_offset
    end = bv_offset + bv_length

    if end > len(bin_data):
        return None

    return bytes(bin_data[start:end])
