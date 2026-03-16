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
    http = requests.Session()

    # Step 1: Fetch the root tileset
    if show_progress:
        print("Creating API session...")

    root_url = f"{TILES_API_BASE}/root.json?key={api_key}"
    resp = http.get(root_url)
    resp.raise_for_status()
    root_tileset = resp.json()

    # Extract the session token from the root tileset.  Google embeds it
    # in child URIs (e.g. "...json?session=TOKEN").  ALL subsequent tile
    # requests require this token or they fail with 400 Bad Request.
    session_token = _extract_session_token(root_tileset)
    if show_progress:
        if session_token:
            print(f"  Session token: {session_token[:12]}…")
        else:
            print("  ⚠ No session token found in root tileset")

    # Step 2: Walk the tileset tree — recursively fetching child tileset
    #         JSON files — until we reach leaf nodes with GLB content.
    target_error = LOD_THRESHOLDS.get(lod, 30.0)
    polygon_aabb = _compute_aabb(polygon)

    if show_progress:
        print("Traversing tileset tree…")

    stats = _TraversalStats(verbose=show_progress)
    glb_urls = _collect_glb_urls(
        node=root_tileset.get("root", {}),
        aabb=polygon_aabb,
        target_error=target_error,
        api_key=api_key,
        session_token=session_token,
        http=http,
        stats=stats,
    )

    if show_progress:
        stats.print_summary()

    if not glb_urls:
        raise RuntimeError(
            "No GLB tiles found for the selected region.\n"
            f"Traversal stats: {stats}\n"
            "Check that:\n"
            "  - The polygon coordinates are correct\n"
            "  - The API key has the 'Map Tiles API' enabled\n"
            "  - The region has 3D tile coverage"
        )

    if show_progress:
        print(f"Found {len(glb_urls)} GLB tiles to download.")
        debug_url = glb_urls[0]
        if api_key in debug_url:
            debug_url = debug_url.replace(api_key, "***")
        print(f"  Example URL: {debug_url}")

    # Step 3: Download GLBs and parse them
    all_vertices = []
    all_normals = []
    all_texcoords = []
    all_indices = []
    all_textures = []
    vertex_offset = 0
    n_skipped = 0

    iterator = tqdm(glb_urls, desc="Downloading GLBs", unit="tile") if show_progress else glb_urls

    for url in iterator:
        try:
            tile_resp = http.get(url, timeout=30)
            tile_resp.raise_for_status()
            glb_data = tile_resp.content

            mesh_data = _parse_glb_python(glb_data)
            if mesh_data is None:
                n_skipped += 1
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

    if show_progress and n_skipped:
        print(f"  ({n_skipped} tiles had no parseable mesh data)")

    if not all_vertices:
        raise RuntimeError(
            "Failed to extract any mesh data from the downloaded tiles.\n"
            "All tile responses were received but contained no valid GLB geometry."
        )

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


# ── Tile tree helpers ─────────────────────────────────────────────────

_BASE_HOST = "https://tile.googleapis.com"


def _compute_aabb(polygon: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Compute axis-aligned bounding box from polygon. Returns (min_lat, max_lat, min_lng, max_lng)."""
    lats = [p[0] for p in polygon]
    lngs = [p[1] for p in polygon]
    return (min(lats), max(lats), min(lngs), max(lngs))


def _extract_session_token(root_tileset: Dict) -> Optional[str]:
    """Extract the session token from child URIs in the root tileset.

    Google embeds ``?session=TOKEN`` in the content URIs of the root
    tileset's children.  We need this token for all subsequent requests.
    """
    from urllib.parse import urlparse, parse_qs

    def _search_node(node: Dict) -> Optional[str]:
        # Check this node's content URI
        content = node.get("content", {})
        uri = content.get("uri", content.get("url", ""))
        if uri and "session=" in uri:
            parsed = parse_qs(urlparse(uri).query if "://" in uri else uri.split("?", 1)[-1])
            tokens = parsed.get("session", [])
            if tokens:
                return tokens[0]
        # Check children
        for child in node.get("children", []):
            token = _search_node(child)
            if token:
                return token
        return None

    root_node = root_tileset.get("root", root_tileset)
    return _search_node(root_node)


def _resolve_uri(uri: str, api_key: str, session_token: Optional[str] = None) -> str:
    """Turn a tileset URI into a full URL with key and session params."""
    # Build the base URL
    if uri.startswith("http"):
        url = uri
    elif uri.startswith("/"):
        url = f"{_BASE_HOST}{uri}"
    else:
        url = f"{TILES_API_BASE}/{uri}"

    # Ensure the URL has both key= and session= parameters
    has_key = "key=" in url
    has_session = "session=" in url

    params_to_add = []
    if not has_key:
        params_to_add.append(f"key={api_key}")
    if not has_session and session_token:
        params_to_add.append(f"session={session_token}")

    if params_to_add:
        sep = "&" if "?" in url else "?"
        url = url + sep + "&".join(params_to_add)

    return url


def _node_intersects_aabb(node: Dict, aabb: Tuple[float, float, float, float]) -> bool:
    """Check whether a tile node's bounding volume intersects the polygon AABB.

    Args:
        node: A 3D Tiles node dict.
        aabb: (min_lat, max_lat, min_lng, max_lng) of the target polygon.

    Supports both "region" (radians) and "box" (ECEF OBB) bounding volumes
    used by Google's Map Tiles API.
    """
    import math

    bv = node.get("boundingVolume", {})

    # ── "region" bounding volume: [west, south, east, north, minH, maxH] in radians
    if "region" in bv:
        region = bv["region"]
        if len(region) >= 4:
            west = math.degrees(region[0])
            south = math.degrees(region[1])
            east = math.degrees(region[2])
            north = math.degrees(region[3])
            min_lat, max_lat, min_lng, max_lng = aabb
            if east < min_lng or west > max_lng or north < min_lat or south > max_lat:
                return False
        return True

    # ── "box" bounding volume: 12 floats — center (3) + 3 half-axis vectors (9)
    #    All in ECEF metres.  Google's Photorealistic 3D Tiles use this format.
    if "box" in bv:
        box = bv["box"]
        if len(box) >= 12:
            cx, cy, cz = box[0], box[1], box[2]

            # Convert ECEF center → lat/lng (spherical approximation)
            center_lat, center_lng = _ecef_to_latlng(cx, cy, cz)

            # Compute the maximum extent of the OBB (half-diagonal length)
            hx = math.sqrt(box[3] ** 2 + box[4] ** 2 + box[5] ** 2)
            hy = math.sqrt(box[6] ** 2 + box[7] ** 2 + box[8] ** 2)
            hz = math.sqrt(box[9] ** 2 + box[10] ** 2 + box[11] ** 2)
            half_diag_m = math.sqrt(hx ** 2 + hy ** 2 + hz ** 2)

            # Convert metres to approximate degrees (generous margin)
            m_per_deg = 111_320.0
            margin_deg = half_diag_m / m_per_deg

            min_lat, max_lat, min_lng, max_lng = aabb
            if (center_lat + margin_deg < min_lat or
                center_lat - margin_deg > max_lat or
                center_lng + margin_deg < min_lng or
                center_lng - margin_deg > max_lng):
                return False
        return True

    # ── "sphere" bounding volume: [cx, cy, cz, radius] in ECEF metres
    if "sphere" in bv:
        sphere = bv["sphere"]
        if len(sphere) >= 4:
            cx, cy, cz, radius = sphere[0], sphere[1], sphere[2], sphere[3]
            center_lat, center_lng = _ecef_to_latlng(cx, cy, cz)
            margin_deg = radius / 111_320.0
            min_lat, max_lat, min_lng, max_lng = aabb
            if (center_lat + margin_deg < min_lat or
                center_lat - margin_deg > max_lat or
                center_lng + margin_deg < min_lng or
                center_lng - margin_deg > max_lng):
                return False
        return True

    # No bounding volume → conservatively keep it
    return True


def _ecef_to_latlng(x: float, y: float, z: float) -> Tuple[float, float]:
    """Convert ECEF coordinates to (latitude, longitude) in degrees.

    Uses a simple spherical approximation (good enough for culling).
    """
    import math
    lng = math.degrees(math.atan2(y, x))
    hyp = math.sqrt(x * x + y * y)
    lat = math.degrees(math.atan2(z, hyp))
    return lat, lng


class _TraversalStats:
    """Tracks what happens during tileset tree traversal for debugging."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.json_fetched = 0
        self.json_failed = 0
        self.json_errors: List[str] = []   # first few error messages
        self.glb_found = 0
        self.nodes_visited = 0
        self.nodes_culled = 0
        self.max_depth = 0

    def print_summary(self) -> None:
        if not self.verbose:
            return
        print(
            f"  Tree traversal: {self.nodes_visited} nodes visited, "
            f"{self.json_fetched} child JSONs fetched, "
            f"{self.json_failed} fetch errors, "
            f"{self.glb_found} GLB URLs found, "
            f"max depth {self.max_depth}"
        )
        for err in self.json_errors[:3]:
            print(f"    ⚠ {err}")

    def __str__(self) -> str:
        return (
            f"nodes={self.nodes_visited}, json_fetched={self.json_fetched}, "
            f"json_failed={self.json_failed}, glb_found={self.glb_found}, "
            f"max_depth={self.max_depth}"
        )


_MAX_TRAVERSAL_DEPTH = 30


def _collect_glb_urls(
    node: Dict,
    aabb: Tuple[float, float, float, float],
    target_error: float,
    api_key: str,
    session_token: Optional[str],
    http: requests.Session,
    stats: _TraversalStats,
    _depth: int = 0,
) -> List[str]:
    """Walk the 3D Tiles tree, fetching child tileset JSONs as needed.

    Google's Map Tiles API returns a tree of tileset JSON files that must
    be fetched recursively.  Each JSON may embed children inline or
    reference them via ``content.uri`` that ends in ``.json``.  Only leaf
    nodes (those whose ``content.uri`` is **not** JSON) contain actual
    GLB mesh data — those are the URLs we collect.
    """
    stats.nodes_visited += 1
    stats.max_depth = max(stats.max_depth, _depth)

    if _depth > _MAX_TRAVERSAL_DEPTH:
        return []

    if not _node_intersects_aabb(node, aabb):
        stats.nodes_culled += 1
        return []

    geometric_error = node.get("geometricError", 0.0)
    children = node.get("children", [])
    content = node.get("content", {})
    content_uri = content.get("uri", content.get("url", ""))

    # ── Should we descend into children? ──────────────────────────────
    want_refine = geometric_error > target_error

    # If we want to refine and have inline children, traverse them.
    if want_refine and children:
        urls: List[str] = []
        for child in children:
            urls.extend(
                _collect_glb_urls(child, aabb, target_error, api_key, session_token, http, stats, _depth + 1)
            )
        return urls

    # If we want to refine but have NO inline children, the content URI
    # may point to a child tileset JSON — fetch and traverse it.
    if want_refine and content_uri and _looks_like_json_uri(content_uri):
        child_tileset = _fetch_child_tileset(content_uri, api_key, session_token, http, stats)
        if child_tileset is not None:
            child_root = child_tileset.get("root", child_tileset)
            return _collect_glb_urls(
                child_root, aabb, target_error, api_key, session_token, http, stats, _depth + 1
            )
        return []

    # ── Leaf node — collect the content URI if it looks like GLB ──────
    if content_uri:
        if _looks_like_json_uri(content_uri):
            # It's a JSON tileset — fetch and traverse it even though we've
            # reached the LOD threshold (the actual GLB is one level deeper).
            child_tileset = _fetch_child_tileset(content_uri, api_key, session_token, http, stats)
            if child_tileset is not None:
                child_root = child_tileset.get("root", child_tileset)
                return _collect_glb_urls(
                    child_root, aabb, target_error, api_key, session_token, http, stats, _depth + 1
                )
            return []
        else:
            stats.glb_found += 1
            return [_resolve_uri(content_uri, api_key, session_token)]

    return []


def _looks_like_json_uri(uri: str) -> bool:
    """Heuristic: does this URI point to a tileset JSON rather than a GLB?"""
    path = uri.split("?")[0]
    return path.endswith(".json")


def _fetch_child_tileset(
    uri: str,
    api_key: str,
    session_token: Optional[str],
    http: requests.Session,
    stats: _TraversalStats,
) -> Optional[Dict]:
    """Fetch a child tileset JSON, returning the parsed dict or None."""
    url = _resolve_uri(uri, api_key, session_token)
    stats.json_fetched += 1
    try:
        resp = http.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        stats.json_failed += 1
        if len(stats.json_errors) < 5:
            err_str = str(e)
            if api_key in err_str:
                err_str = err_str.replace(api_key, "***")
            stats.json_errors.append(err_str)
        return None


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
