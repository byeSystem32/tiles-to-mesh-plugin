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

# Optional: trimesh handles GLB loading far more robustly than our custom
# parser (Draco decompression, interleaved buffers, glTF scene graph…).
try:
    import trimesh as _trimesh

    HAS_TRIMESH = True
except ImportError:
    _trimesh = None  # type: ignore[assignment]
    HAS_TRIMESH = False

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
    tile_entries = _collect_glb_urls(
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

    if not tile_entries:
        raise RuntimeError(
            "No GLB tiles found for the selected region.\n"
            f"Traversal stats: {stats}\n"
            "Check that:\n"
            "  - The polygon coordinates are correct\n"
            "  - The API key has the 'Map Tiles API' enabled\n"
            "  - The region has 3D tile coverage"
        )

    if show_progress:
        print(f"Found {len(tile_entries)} GLB tiles to download.")
        debug_url = tile_entries[0][0]
        if api_key in debug_url:
            debug_url = debug_url.replace(api_key, "***")
        print(f"  Example URL: {debug_url}")

    # Step 3: Download GLBs, parse them, and apply tile transforms
    all_vertices: List[np.ndarray] = []
    all_normals: List[np.ndarray] = []
    all_texcoords: List[np.ndarray] = []
    all_indices: List[np.ndarray] = []
    all_textures: list = []
    vertex_offset = 0
    n_skipped = 0
    n_parse_err = 0
    _first_tile_diagnosed = False

    parser = "trimesh" if HAS_TRIMESH else "builtin"
    if show_progress:
        print(f"  GLB parser: {parser}")

    iterator = tqdm(tile_entries, desc="Downloading GLBs", unit="tile") if show_progress else tile_entries

    for url, tile_transform in iterator:
        try:
            tile_resp = http.get(url, timeout=30)
            tile_resp.raise_for_status()
            glb_data = tile_resp.content

            # ── Diagnostic dump for the very first tile ───────────
            if show_progress and not _first_tile_diagnosed:
                _first_tile_diagnosed = True
                _dump_glb_info(glb_data)

            # ── Parse the GLB ─────────────────────────────────────
            if HAS_TRIMESH:
                parse_result = _parse_glb_with_trimesh(glb_data)
            else:
                parse_result = _parse_glb_python(glb_data)

            if parse_result is None:
                n_skipped += 1
                continue

            verts, norms, uvs, idxs, tex = parse_result

            if verts is None or len(verts) == 0:
                n_skipped += 1
                continue
            if idxs is None or len(idxs) == 0:
                n_skipped += 1
                continue

            # ── Apply the tile's accumulated transform ────────────
            # verts is Nx3.  Extend to homogeneous Nx4, multiply, drop w.
            n = len(verts)
            hom = np.ones((n, 4), dtype=np.float64)
            hom[:, :3] = verts
            transformed = (tile_transform @ hom.T).T[:, :3].astype(np.float32)
            all_vertices.append(transformed)

            # Normals are transformed by the upper-left 3×3 (no translation)
            if norms is not None and len(norms) == n:
                rot = tile_transform[:3, :3]
                tn = (rot @ norms.astype(np.float64).T).T.astype(np.float32)
                lens = np.linalg.norm(tn, axis=1, keepdims=True)
                lens[lens == 0] = 1.0
                tn /= lens
                all_normals.append(tn)

            if uvs is not None and len(uvs) == n:
                all_texcoords.append(uvs)

            # Offset indices
            all_indices.append(idxs + vertex_offset)
            vertex_offset += n

            if tex is not None:
                all_textures.append(tex)

        except Exception as e:
            n_parse_err += 1
            if show_progress and n_parse_err <= 5:
                print(f"  Warning: tile error: {e}")
            continue

    if show_progress and n_skipped:
        print(f"  ({n_skipped} tiles had no parseable mesh data)")
    if show_progress and n_parse_err:
        print(f"  ({n_parse_err} tiles failed to download/parse)")

    if not all_vertices:
        raise RuntimeError(
            "Failed to extract any mesh data from the downloaded tiles.\n"
            "All tile responses were received but contained no valid GLB geometry."
        )

    # Step 4: Assemble — vertices are now in ECEF (metres)
    vertices = np.vstack(all_vertices)
    indices = np.vstack(all_indices)
    normals = np.vstack(all_normals) if all_normals else None
    texcoords = np.vstack(all_texcoords) if all_texcoords else None

    # Step 5: Convert ECEF to local ENU (East-North-Up) so the mesh
    #         is centred near the origin and oriented sensibly.
    centroid_lat, centroid_lng = _polygon_centroid(polygon)
    vertices, normals = _ecef_to_enu_mesh(vertices, normals, centroid_lat, centroid_lng)

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


def _polygon_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Return the (lat, lng) centroid of a polygon."""
    lats = [p[0] for p in polygon]
    lngs = [p[1] for p in polygon]
    return float(np.mean(lats)), float(np.mean(lngs))


def _ecef_to_enu_mesh(
    vertices: np.ndarray,
    normals: Optional[np.ndarray],
    ref_lat: float,
    ref_lng: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert ECEF vertices/normals to a local East-North-Up frame.

    The ENU frame is centred on the WGS-84 surface point at
    (ref_lat, ref_lng) so the mesh ends up near the origin with
    X = East, Y = Up, Z = North  (to match typical 3D viewer conventions
    where Y is up).

    Args:
        vertices: Nx3 float32 array of ECEF positions (metres).
        normals: Optional Nx3 float32 array of ECEF normals.
        ref_lat: Reference latitude in degrees.
        ref_lng: Reference longitude in degrees.

    Returns:
        (vertices_enu, normals_enu) — same shapes, float32.
    """
    import math

    lat_r = math.radians(ref_lat)
    lng_r = math.radians(ref_lng)

    sin_lat = math.sin(lat_r)
    cos_lat = math.cos(lat_r)
    sin_lng = math.sin(lng_r)
    cos_lng = math.cos(lng_r)

    # WGS-84 reference point on the ellipsoid surface
    a = 6_378_137.0            # semi-major axis (m)
    e2 = 6.69437999014e-3      # first eccentricity squared
    N = a / math.sqrt(1 - e2 * sin_lat ** 2)

    ref_x = N * cos_lat * cos_lng
    ref_y = N * cos_lat * sin_lng
    ref_z = N * (1 - e2) * sin_lat

    # Rotation matrix: ECEF → ENU
    # ENU basis vectors in ECEF:
    #   e_east  = [-sin_lng,          cos_lng,         0        ]
    #   e_north = [-sin_lat*cos_lng, -sin_lat*sin_lng, cos_lat  ]
    #   e_up    = [ cos_lat*cos_lng,  cos_lat*sin_lng, sin_lat  ]
    R = np.array([
        [-sin_lng,          cos_lng,          0.0     ],
        [-sin_lat * cos_lng, -sin_lat * sin_lng, cos_lat],
        [cos_lat * cos_lng,  cos_lat * sin_lng,  sin_lat],
    ], dtype=np.float64)

    # Translate then rotate
    delta = vertices.astype(np.float64) - np.array([ref_x, ref_y, ref_z], dtype=np.float64)
    enu = (R @ delta.T).T  # Nx3

    # Swap axes so Y is up: ENU → (East, Up, North) = (x_enu, z_enu, y_enu)
    out_vertices = np.empty_like(enu, dtype=np.float32)
    out_vertices[:, 0] = enu[:, 0]   # X = East
    out_vertices[:, 1] = enu[:, 2]   # Y = Up
    out_vertices[:, 2] = enu[:, 1]   # Z = North

    out_normals = None
    if normals is not None:
        n_enu = (R @ normals.astype(np.float64).T).T
        out_normals = np.empty_like(n_enu, dtype=np.float32)
        out_normals[:, 0] = n_enu[:, 0]
        out_normals[:, 1] = n_enu[:, 2]
        out_normals[:, 2] = n_enu[:, 1]
        # Re-normalise
        lens = np.linalg.norm(out_normals, axis=1, keepdims=True)
        lens[lens == 0] = 1.0
        out_normals /= lens

    return out_vertices, out_normals


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


def _get_node_transform(node: Dict) -> np.ndarray:
    """Extract the 4x4 transform matrix from a 3D Tiles node.

    The ``transform`` property is a 16-element array in **column-major** order
    (OpenGL convention).  If absent, the identity matrix is returned.
    """
    t = node.get("transform")
    if t and len(t) == 16:
        # Column-major → row-major (numpy default)
        return np.array(t, dtype=np.float64).reshape(4, 4).T
    return np.eye(4, dtype=np.float64)


# Return type: list of (url, accumulated_4x4_transform)
_TileEntry = Tuple[str, np.ndarray]


def _collect_glb_urls(
    node: Dict,
    aabb: Tuple[float, float, float, float],
    target_error: float,
    api_key: str,
    session_token: Optional[str],
    http: requests.Session,
    stats: _TraversalStats,
    _depth: int = 0,
    _parent_transform: Optional[np.ndarray] = None,
) -> List[_TileEntry]:
    """Walk the 3D Tiles tree, fetching child tileset JSONs as needed.

    Google's Map Tiles API returns a tree of tileset JSON files that must
    be fetched recursively.  Each JSON may embed children inline or
    reference them via ``content.uri`` that ends in ``.json``.  Only leaf
    nodes (those whose ``content.uri`` is **not** JSON) contain actual
    GLB mesh data — those are the URLs we collect.

    Returns a list of ``(url, transform)`` tuples where *transform* is the
    accumulated 4×4 matrix that places the tile's local vertices into ECEF.
    """
    stats.nodes_visited += 1
    stats.max_depth = max(stats.max_depth, _depth)

    if _depth > _MAX_TRAVERSAL_DEPTH:
        return []

    if not _node_intersects_aabb(node, aabb):
        stats.nodes_culled += 1
        return []

    # Accumulate this node's transform with its parent's
    if _parent_transform is None:
        _parent_transform = np.eye(4, dtype=np.float64)
    node_local = _get_node_transform(node)
    accumulated = _parent_transform @ node_local

    geometric_error = node.get("geometricError", 0.0)
    children = node.get("children", [])
    content = node.get("content", {})
    content_uri = content.get("uri", content.get("url", ""))

    # ── Should we descend into children? ──────────────────────────────
    want_refine = geometric_error > target_error

    # If we want to refine and have inline children, traverse them.
    if want_refine and children:
        entries: List[_TileEntry] = []
        for child in children:
            entries.extend(
                _collect_glb_urls(child, aabb, target_error, api_key, session_token,
                                  http, stats, _depth + 1, accumulated)
            )
        return entries

    # If we want to refine but have NO inline children, the content URI
    # may point to a child tileset JSON — fetch and traverse it.
    if want_refine and content_uri and _looks_like_json_uri(content_uri):
        child_tileset = _fetch_child_tileset(content_uri, api_key, session_token, http, stats)
        if child_tileset is not None:
            child_root = child_tileset.get("root", child_tileset)
            return _collect_glb_urls(
                child_root, aabb, target_error, api_key, session_token,
                http, stats, _depth + 1, accumulated
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
                    child_root, aabb, target_error, api_key, session_token,
                    http, stats, _depth + 1, accumulated
                )
            return []
        else:
            stats.glb_found += 1
            return [(_resolve_uri(content_uri, api_key, session_token), accumulated)]

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


# ── trimesh-based GLB parser ──────────────────────────────────────────


def _parse_glb_with_trimesh(data: bytes):
    """Parse a GLB using trimesh.  Handles Draco, interleaved buffers,
    glTF node transforms, and all accessor types correctly.

    Returns the same 5-tuple as ``_parse_glb_python``:
        (vertices, normals, texcoords, indices, texture)
    or None on failure.
    """
    try:
        result = _trimesh.load(
            io.BytesIO(data),
            file_type="glb",
            force="scene",      # always get a Scene so we apply node xforms
            process=False,       # don't merge / heal; we do that later
        )
    except Exception:
        # If trimesh can't load it, fall back to the builtin parser
        return _parse_glb_python(data)

    if isinstance(result, _trimesh.Scene):
        if len(result.geometry) == 0:
            return _parse_glb_python(data)
        try:
            combined = result.dump(concatenate=True)
        except Exception:
            return _parse_glb_python(data)
    elif isinstance(result, _trimesh.Trimesh):
        combined = result
    else:
        return _parse_glb_python(data)

    if combined is None or len(combined.vertices) == 0 or len(combined.faces) == 0:
        return _parse_glb_python(data)

    vertices = np.asarray(combined.vertices, dtype=np.float32)
    faces = np.asarray(combined.faces, dtype=np.uint32)

    normals = None
    try:
        vn = combined.vertex_normals
        if vn is not None and len(vn) == len(vertices):
            normals = np.asarray(vn, dtype=np.float32)
    except Exception:
        pass

    texcoords = None
    try:
        vis = combined.visual
        if hasattr(vis, "uv") and vis.uv is not None and len(vis.uv) == len(vertices):
            texcoords = np.asarray(vis.uv, dtype=np.float32)
    except Exception:
        pass

    # Texture image
    texture = None
    try:
        vis = combined.visual
        if hasattr(vis, "material") and hasattr(vis.material, "image"):
            img = vis.material.image
            if img is not None:
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                texture = {"data": buf.getvalue(), "mime": "image/jpeg"}
    except Exception:
        pass

    return vertices, normals, texcoords, faces, texture


def _dump_glb_info(data: bytes) -> None:
    """Print diagnostic information about a GLB file (first tile only)."""
    try:
        if len(data) < 12:
            print("    [diag] GLB too short")
            return

        magic = struct.unpack_from("<I", data, 0)[0]
        if magic != 0x46546C67:
            print(f"    [diag] Not a GLB (magic=0x{magic:08X})")
            return

        # Parse JSON chunk
        offset = 12
        json_chunk = None
        bin_size = 0
        while offset < len(data):
            if offset + 8 > len(data):
                break
            chunk_length = struct.unpack_from("<I", data, offset)[0]
            chunk_type = struct.unpack_from("<I", data, offset + 4)[0]
            if chunk_type == 0x4E4F534A:
                json_chunk = json.loads(data[offset + 8: offset + 8 + chunk_length])
            elif chunk_type == 0x004E4942:
                bin_size = chunk_length
            offset += 8 + chunk_length
            offset = (offset + 3) & ~3

        if json_chunk is None:
            print("    [diag] No JSON chunk found")
            return

        exts = json_chunk.get("extensionsUsed", [])
        exts_req = json_chunk.get("extensionsRequired", [])
        meshes = json_chunk.get("meshes", [])
        nodes = json_chunk.get("nodes", [])
        scenes = json_chunk.get("scenes", [])
        accessors = json_chunk.get("accessors", [])
        bvs = json_chunk.get("bufferViews", [])

        n_prims = sum(len(m.get("primitives", [])) for m in meshes)
        has_draco = any("KHR_draco_mesh_compression" in
                        p.get("extensions", {})
                        for m in meshes
                        for p in m.get("primitives", []))
        has_stride = any(bv.get("byteStride", 0) > 0 for bv in bvs)
        node_has_xform = sum(1 for n in nodes if "matrix" in n or "rotation" in n or "translation" in n)

        print(f"    [diag] GLB: {len(data)} bytes, BIN chunk: {bin_size} bytes")
        print(f"    [diag] extensionsUsed: {exts}")
        print(f"    [diag] extensionsRequired: {exts_req}")
        print(f"    [diag] scenes: {len(scenes)}, nodes: {len(nodes)} "
              f"({node_has_xform} with transforms), meshes: {len(meshes)}, "
              f"primitives: {n_prims}")
        print(f"    [diag] accessors: {len(accessors)}, bufferViews: {len(bvs)}, "
              f"interleaved (byteStride>0): {has_stride}")
        print(f"    [diag] Draco compressed primitives: {has_draco}")

        # Show first mesh's first primitive details
        if meshes and meshes[0].get("primitives"):
            prim = meshes[0]["primitives"][0]
            attrs = prim.get("attributes", {})
            print(f"    [diag] First primitive attributes: {list(attrs.keys())}")
            print(f"    [diag]   mode: {prim.get('mode', 4)} (4=TRIANGLES)")
            if "POSITION" in attrs:
                pa = accessors[attrs["POSITION"]]
                print(f"    [diag]   POSITION: count={pa.get('count')}, "
                      f"type={pa.get('type')}, componentType={pa.get('componentType')}, "
                      f"bufferView={pa.get('bufferView')}")
            draco = prim.get("extensions", {}).get("KHR_draco_mesh_compression")
            if draco:
                print(f"    [diag]   Draco ext: bufferView={draco.get('bufferView')}, "
                      f"attrs={draco.get('attributes')}")
    except Exception as e:
        print(f"    [diag] Error inspecting GLB: {e}")


_draco_warned = False


def _decode_draco_primitive(
    draco_ext: Dict,
    buffer_views: List[Dict],
    bin_data: bytes,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]]:
    """Attempt to decode a Draco-compressed glTF primitive.

    Returns (positions, normals, texcoords, indices) or None.
    Requires the ``DracoPy`` package (``pip install DracoPy``).
    """
    global _draco_warned

    try:
        import DracoPy  # type: ignore
    except ImportError:
        if not _draco_warned:
            import warnings
            warnings.warn(
                "This tileset uses Draco mesh compression but DracoPy is not "
                "installed.  Run  pip install DracoPy  for proper mesh decoding.",
                stacklevel=4,
            )
            _draco_warned = True
        return None

    bv_index = draco_ext.get("bufferView")
    if bv_index is None:
        return None

    bv = buffer_views[bv_index]
    bv_offset = bv.get("byteOffset", 0)
    bv_length = bv["byteLength"]
    compressed = bytes(bin_data[bv_offset : bv_offset + bv_length])

    try:
        mesh_obj = DracoPy.decode(compressed)
    except Exception:
        return None

    positions = np.array(mesh_obj.points, dtype=np.float32).reshape(-1, 3)

    normals = None
    if hasattr(mesh_obj, "normals") and mesh_obj.normals is not None and len(mesh_obj.normals):
        normals = np.array(mesh_obj.normals, dtype=np.float32).reshape(-1, 3)

    texcoords = None
    if hasattr(mesh_obj, "tex_coord") and mesh_obj.tex_coord is not None and len(mesh_obj.tex_coord):
        texcoords = np.array(mesh_obj.tex_coord, dtype=np.float32).reshape(-1, 2)

    indices = None
    if hasattr(mesh_obj, "faces") and mesh_obj.faces is not None and len(mesh_obj.faces):
        indices = np.array(mesh_obj.faces, dtype=np.uint32).reshape(-1, 3)

    return positions, normals, texcoords, indices


def _compute_gltf_node_transforms(gltf_json: Dict) -> Dict[int, np.ndarray]:
    """Compute world-space transforms for each *mesh index* in a glTF.

    Walks the scene graph (``scenes`` → ``nodes``) and accumulates
    transforms down to nodes that reference a mesh.  Returns a dict
    mapping mesh-index → 4×4 world transform (float64).
    """
    nodes = gltf_json.get("nodes", [])
    scenes = gltf_json.get("scenes", [])

    # Pre-compute each node's local transform
    local_transforms: List[np.ndarray] = []
    for node in nodes:
        local_transforms.append(_gltf_node_local_transform(node))

    mesh_transforms: Dict[int, np.ndarray] = {}

    def walk(node_idx: int, parent: np.ndarray):
        if node_idx < 0 or node_idx >= len(nodes):
            return
        node = nodes[node_idx]
        world = parent @ local_transforms[node_idx]

        mesh_idx = node.get("mesh")
        if mesh_idx is not None:
            mesh_transforms[mesh_idx] = world

        for child_idx in node.get("children", []):
            walk(child_idx, world)

    identity = np.eye(4, dtype=np.float64)

    if scenes:
        for root_idx in scenes[0].get("nodes", []):
            walk(root_idx, identity)
    else:
        # No scene — walk all root-level nodes
        for i in range(len(nodes)):
            if i not in mesh_transforms:
                walk(i, identity)

    return mesh_transforms


def _gltf_node_local_transform(node: Dict) -> np.ndarray:
    """Compute a node's local 4×4 transform from ``matrix`` or TRS."""
    if "matrix" in node and len(node["matrix"]) == 16:
        # Column-major → row-major
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T

    T = np.eye(4, dtype=np.float64)

    if "translation" in node:
        t = node["translation"]
        T[0, 3] = t[0]
        T[1, 3] = t[1]
        T[2, 3] = t[2]

    if "rotation" in node:
        # Quaternion [x, y, z, w]
        qx, qy, qz, qw = node["rotation"]
        R = np.eye(4, dtype=np.float64)
        R[0, 0] = 1 - 2 * (qy * qy + qz * qz)
        R[0, 1] = 2 * (qx * qy - qz * qw)
        R[0, 2] = 2 * (qx * qz + qy * qw)
        R[1, 0] = 2 * (qx * qy + qz * qw)
        R[1, 1] = 1 - 2 * (qx * qx + qz * qz)
        R[1, 2] = 2 * (qy * qz - qx * qw)
        R[2, 0] = 2 * (qx * qz - qy * qw)
        R[2, 1] = 2 * (qy * qz + qx * qw)
        R[2, 2] = 1 - 2 * (qx * qx + qy * qy)
        T = T @ R

    if "scale" in node:
        s = node["scale"]
        S = np.eye(4, dtype=np.float64)
        S[0, 0] = s[0]
        S[1, 1] = s[1]
        S[2, 2] = s[2]
        T = T @ S

    return T


def _apply_mat4(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4×4 transform to an Nx3 array of points."""
    if np.allclose(mat, np.eye(4)):
        return pts
    n = len(pts)
    hom = np.ones((n, 4), dtype=np.float64)
    hom[:, :3] = pts
    return (mat @ hom.T).T[:, :3].astype(np.float32)


def _apply_mat3_normals(mat: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Transform normals by the upper-left 3×3 of a 4×4 matrix, then renormalise."""
    if np.allclose(mat[:3, :3], np.eye(3)):
        return normals
    rot = mat[:3, :3]
    tn = (rot @ normals.astype(np.float64).T).T.astype(np.float32)
    lens = np.linalg.norm(tn, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    tn /= lens
    return tn


def _triangle_strip_to_triangles(indices: np.ndarray) -> np.ndarray:
    """Convert a TRIANGLE_STRIP index array to plain TRIANGLES."""
    if len(indices) < 3:
        return np.empty(0, dtype=np.uint32)
    tris = []
    for i in range(len(indices) - 2):
        if i % 2 == 0:
            tris.extend([indices[i], indices[i + 1], indices[i + 2]])
        else:
            # Reverse winding for odd triangles
            tris.extend([indices[i + 1], indices[i], indices[i + 2]])
    return np.array(tris, dtype=np.uint32)


def _triangle_fan_to_triangles(indices: np.ndarray) -> np.ndarray:
    """Convert a TRIANGLE_FAN index array to plain TRIANGLES."""
    if len(indices) < 3:
        return np.empty(0, dtype=np.uint32)
    tris = []
    for i in range(1, len(indices) - 1):
        tris.extend([indices[0], indices[i], indices[i + 1]])
    return np.array(tris, dtype=np.uint32)


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

    # Check for Draco compression
    extensions_used = json_chunk.get("extensionsUsed", [])
    has_draco = "KHR_draco_mesh_compression" in extensions_used

    # Also apply glTF node transforms from the scene graph
    node_transforms = _compute_gltf_node_transforms(json_chunk)

    for mesh_idx, mesh in enumerate(meshes):
        # Find the world transform for this mesh (from the glTF node tree)
        mesh_world = node_transforms.get(mesh_idx, np.eye(4, dtype=np.float64))

        for primitive in mesh.get("primitives", []):
            attrs = primitive.get("attributes", {})
            prim_verts = None   # Track positions added in THIS iteration

            # ── If this primitive uses Draco compression, try to decode it ──
            draco_ext = primitive.get("extensions", {}).get(
                "KHR_draco_mesh_compression"
            )
            if draco_ext:
                draco_result = _decode_draco_primitive(
                    draco_ext, buffer_views, bin_chunk
                )
                if draco_result is not None:
                    d_pos, d_norm, d_uv, d_idx = draco_result
                    # Apply glTF node transform
                    d_pos = _apply_mat4(mesh_world, d_pos)
                    all_positions.append(d_pos)
                    if d_norm is not None:
                        d_norm = _apply_mat3_normals(mesh_world, d_norm)
                        all_normals.append(d_norm)
                    if d_uv is not None:
                        all_texcoords.append(d_uv)
                    if d_idx is not None:
                        all_indices.append(d_idx + base_vertex)
                    base_vertex += len(d_pos)
                    continue  # Don't also read the fallback accessors
                # If Draco decode failed, fall through to regular accessors

            # Positions
            if "POSITION" in attrs:
                pos_data = _read_accessor(accessors[attrs["POSITION"]], buffer_views, bin_chunk)
                if pos_data is not None:
                    verts = np.frombuffer(pos_data, dtype=np.float32).reshape(-1, 3)
                    # Apply glTF node transform
                    verts = _apply_mat4(mesh_world, verts)
                    all_positions.append(verts)
                    prim_verts = verts

            # Normals
            if "NORMAL" in attrs and prim_verts is not None:
                norm_data = _read_accessor(accessors[attrs["NORMAL"]], buffer_views, bin_chunk)
                if norm_data is not None:
                    norms = np.frombuffer(norm_data, dtype=np.float32).reshape(-1, 3)
                    norms = _apply_mat3_normals(mesh_world, norms)
                    all_normals.append(norms)

            # Texcoords
            if "TEXCOORD_0" in attrs and prim_verts is not None:
                uv_data = _read_accessor(accessors[attrs["TEXCOORD_0"]], buffer_views, bin_chunk)
                if uv_data is not None:
                    uvs = np.frombuffer(uv_data, dtype=np.float32).reshape(-1, 2)
                    all_texcoords.append(uvs)

            # Indices
            if "indices" in primitive and prim_verts is not None:
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

                    prim_mode = primitive.get("mode", 4)  # default = TRIANGLES

                    if prim_mode == 5:
                        idxs = _triangle_strip_to_triangles(idxs)
                    elif prim_mode == 6:
                        idxs = _triangle_fan_to_triangles(idxs)
                    elif prim_mode != 4:
                        continue

                    if len(idxs) >= 3 and len(idxs) % 3 == 0:
                        all_indices.append(idxs.reshape(-1, 3) + base_vertex)
                    elif len(idxs) >= 3:
                        trim = len(idxs) - (len(idxs) % 3)
                        all_indices.append(idxs[:trim].reshape(-1, 3) + base_vertex)

            # Only update base_vertex if we actually added positions this round
            if prim_verts is not None:
                base_vertex += len(prim_verts)

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
    """Read raw bytes for a glTF accessor, respecting byteStride.

    Google's 3D Tiles frequently use **interleaved** buffer views where
    positions, normals, and UVs share a single buffer view with a stride.
    Without handling ``byteStride`` we'd read the wrong bytes for every
    attribute.
    """
    bv_index = accessor.get("bufferView")
    if bv_index is None:
        return None

    bv = buffer_views[bv_index]
    bv_offset = bv.get("byteOffset", 0)
    acc_offset = accessor.get("byteOffset", 0)
    count = accessor.get("count", 0)
    if count == 0:
        return None

    # Component size in bytes
    comp_type = accessor.get("componentType", 5126)
    _COMP_SIZES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
    comp_size = _COMP_SIZES.get(comp_type, 4)

    # Number of components per element
    _TYPE_COUNTS = {
        "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
        "MAT2": 4, "MAT3": 9, "MAT4": 16,
    }
    type_count = _TYPE_COUNTS.get(accessor.get("type", "SCALAR"), 1)

    element_size = comp_size * type_count
    byte_stride = bv.get("byteStride", 0)

    start = bv_offset + acc_offset

    if byte_stride and byte_stride > element_size:
        # Interleaved — extract each element from the strided layout
        out = bytearray(count * element_size)
        for i in range(count):
            src = start + i * byte_stride
            dst = i * element_size
            if src + element_size > len(bin_data):
                break
            out[dst:dst + element_size] = bin_data[src:src + element_size]
        return bytes(out)
    else:
        # Tightly packed — read a contiguous block
        end = start + count * element_size
        if end > len(bin_data):
            return None
        return bytes(bin_data[start:end])
