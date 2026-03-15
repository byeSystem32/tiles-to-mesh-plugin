"""Type stubs for the Rust native extension module (_tiles_to_mesh_rs)."""

from typing import Callable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

# ── geo module ──────────────────────────────────────────────────────

class GeoPoint:
    lat: float
    lng: float
    alt: float
    def __init__(self, lat: float, lng: float, alt: float = 0.0) -> None: ...
    def to_ecef(self) -> Tuple[float, float, float]: ...

class EnuOrigin:
    lat: float
    lng: float
    alt: float
    def __init__(self, lat: float, lng: float, alt: float = 0.0) -> None: ...

def wgs84_to_ecef(lat: float, lng: float, alt: float) -> npt.NDArray[np.float64]: ...
def ecef_to_enu(x: float, y: float, z: float, origin: EnuOrigin) -> npt.NDArray[np.float64]: ...
def wgs84_to_enu(lat: float, lng: float, alt: float, origin: EnuOrigin) -> npt.NDArray[np.float64]: ...
def batch_ecef_to_enu(ecef: npt.NDArray[np.float64], origin: EnuOrigin) -> npt.NDArray[np.float64]: ...

# ── tiles module ────────────────────────────────────────────────────

class FetchConfig:
    api_key: str
    lod: int
    max_concurrent: int
    mode: str
    def __init__(
        self,
        api_key: str,
        lod: int = 3,
        max_concurrent: int = 10,
        mode: str = "photorealistic",
    ) -> None: ...

class TileFetcher:
    def __init__(self, config: FetchConfig) -> None: ...
    def fetch_root_tileset(self) -> str: ...
    def fetch_tiles_in_region(
        self,
        polygon_coords: List[Tuple[float, float]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[TileMeshData]: ...

class TileMeshData:
    positions: List[float]
    normals: List[float]
    texcoords: List[float]
    indices: List[int]
    texture_mime: Optional[str]
    transform: List[float]
    vertex_count: int
    face_count: int
    has_texture: bool
    def get_texture_bytes(self) -> Optional[bytes]: ...

class ParsedTile:
    meshes: List[TileMeshData]
    name: str

class TileNode:
    content_uri: Optional[str]
    geometric_error: float
    bounding_region: Optional[List[float]]
    child_count: int
    depth: int

# ── mesh module ─────────────────────────────────────────────────────

class MergedMesh:
    positions: List[float]
    normals: List[float]
    texcoords: List[float]
    indices: List[int]
    source_tile_count: int
    vertex_count: int
    face_count: int

def merge_tile_meshes(tile_meshes: list) -> MergedMesh: ...
def clip_mesh_to_polygon(mesh: MergedMesh, polygon: List[Tuple[float, float]]) -> MergedMesh: ...
def apply_transform(mesh: MergedMesh, matrix: List[float]) -> None: ...
