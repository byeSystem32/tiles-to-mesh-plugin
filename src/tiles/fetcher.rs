//! Async HTTP tile fetching from Google 3D Tiles API.
//!
//! Uses reqwest + tokio for parallel tile downloads with rate limiting.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::Semaphore;

use super::parser::TileMeshData;

const TILES_API_BASE: &str = "https://tile.googleapis.com/v1/3dtiles";

/// Configuration for tile fetching.
#[pyclass]
#[derive(Clone, Debug)]
pub struct FetchConfig {
    #[pyo3(get, set)]
    pub api_key: String,
    /// LOD level: 1 (coarse) to 5 (max detail).
    #[pyo3(get, set)]
    pub lod: u8,
    /// Maximum concurrent HTTP requests.
    #[pyo3(get, set)]
    pub max_concurrent: usize,
    /// Fetch mode: "photorealistic" or "geometry".
    #[pyo3(get, set)]
    pub mode: String,
}

#[pymethods]
impl FetchConfig {
    #[new]
    #[pyo3(signature = (api_key, lod=3, max_concurrent=10, mode="photorealistic".to_string()))]
    fn new(api_key: String, lod: u8, max_concurrent: usize, mode: String) -> PyResult<Self> {
        if lod < 1 || lod > 5 {
            return Err(PyRuntimeError::new_err("LOD must be between 1 and 5"));
        }
        if mode != "photorealistic" && mode != "geometry" {
            return Err(PyRuntimeError::new_err(
                "Mode must be 'photorealistic' or 'geometry'",
            ));
        }
        Ok(Self {
            api_key,
            lod,
            max_concurrent,
            mode,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "FetchConfig(lod={}, max_concurrent={}, mode='{}')",
            self.lod, self.max_concurrent, self.mode
        )
    }
}

/// Tileset root response from the API.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TilesetRoot {
    root: TilesetNode,
    #[serde(default)]
    geometric_error: f64,
}

/// A node in the tileset tree.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TilesetNode {
    #[serde(default)]
    bounding_volume: Option<BoundingVolume>,
    #[serde(default)]
    geometric_error: f64,
    #[serde(default)]
    content: Option<TileContent>,
    #[serde(default)]
    children: Vec<TilesetNode>,
    #[serde(default)]
    refine: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BoundingVolume {
    #[serde(default)]
    region: Option<Vec<f64>>,
    #[serde(default)]
    r#box: Option<Vec<f64>>,
    #[serde(default)]
    sphere: Option<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TileContent {
    uri: String,
}

/// Main tile fetcher. Handles the full pipeline from API → parsed mesh data.
#[pyclass]
pub struct TileFetcher {
    config: FetchConfig,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl TileFetcher {
    #[new]
    fn new(config: FetchConfig) -> PyResult<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;
        Ok(Self { config, runtime })
    }

    /// Fetch the root tileset JSON from Google's API.
    fn fetch_root_tileset(&self) -> PyResult<String> {
        let config = self.config.clone();
        self.runtime.block_on(async {
            fetch_root_tileset_async(&config).await
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    /// Fetch all tiles that intersect the given polygon at the configured LOD.
    ///
    /// `polygon_coords`: list of (lat, lng) tuples forming the selection polygon.
    ///
    /// Returns a list of `TileMeshData` objects containing raw vertex/index/uv data.
    #[pyo3(signature = (polygon_coords, progress_callback=None))]
    fn fetch_tiles_in_region(
        &self,
        polygon_coords: Vec<(f64, f64)>,
        progress_callback: Option<PyObject>,
    ) -> PyResult<Vec<TileMeshData>> {
        let config = self.config.clone();
        let result = self.runtime.block_on(async {
            fetch_region_tiles_async(&config, &polygon_coords, progress_callback).await
        });
        result.map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }
}

/// Geometric error threshold for each LOD level.
fn lod_to_geometric_error(lod: u8) -> f64 {
    match lod {
        1 => 500.0,
        2 => 100.0,
        3 => 30.0,
        4 => 10.0,
        5 => 2.0,
        _ => 30.0,
    }
}

/// Fetch the root tileset.json from Google's 3D Tiles API.
async fn fetch_root_tileset_async(config: &FetchConfig) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let client = reqwest::Client::new();
    let url = format!(
        "{}/root.json?key={}",
        TILES_API_BASE, config.api_key
    );
    let resp = client.get(&url).send().await?;
    if !resp.status().is_success() {
        return Err(format!(
            "API request failed with status {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        )
        .into());
    }
    Ok(resp.text().await?)
}

/// Fetch all tiles in a region. This is the main workhorse function.
async fn fetch_region_tiles_async(
    config: &FetchConfig,
    polygon: &[(f64, f64)],
    progress_callback: Option<PyObject>,
) -> Result<Vec<TileMeshData>, Box<dyn std::error::Error + Send + Sync>> {
    let client = Arc::new(reqwest::Client::new());
    let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
    let target_error = lod_to_geometric_error(config.lod);

    // Step 1: Fetch root tileset
    let root_url = format!("{}/root.json?key={}", TILES_API_BASE, config.api_key);
    let root_resp = client.get(&root_url).send().await?;
    if !root_resp.status().is_success() {
        return Err(format!("Failed to fetch root tileset: {}", root_resp.status()).into());
    }
    let root: TilesetRoot = root_resp.json().await?;

    // Step 2: Traverse the tileset tree to find tiles that intersect our polygon
    let tile_urls = collect_tile_urls(&root.root, polygon, target_error, &config.api_key);

    let total = tile_urls.len();
    if total == 0 {
        return Err("No tiles found intersecting the selected region".into());
    }

    // Step 3: Fetch all tile GLBs in parallel
    let mut handles = Vec::with_capacity(total);
    for (idx, url) in tile_urls.into_iter().enumerate() {
        let client = Arc::clone(&client);
        let sem = Arc::clone(&semaphore);
        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let resp = client.get(&url).send().await?;
            let bytes = resp.bytes().await?;
            Ok::<(usize, Vec<u8>), reqwest::Error>((idx, bytes.to_vec()))
        }));
    }

    let mut tiles = Vec::with_capacity(total);
    for handle in handles {
        let (_idx, glb_bytes) = handle.await??;

        // Parse GLB into mesh data
        if let Ok(mesh_data) = super::parser::parse_glb_bytes(&glb_bytes) {
            tiles.push(mesh_data);
        }

        // Report progress
        if let Some(ref cb) = progress_callback {
            Python::with_gil(|py| {
                let _ = cb.call1(py, (tiles.len(), total));
            });
        }
    }

    Ok(tiles)
}

/// Recursively collect tile content URLs that intersect the polygon
/// at the desired geometric error threshold.
fn collect_tile_urls(
    node: &TilesetNode,
    polygon: &[(f64, f64)],
    target_error: f64,
    api_key: &str,
) -> Vec<String> {
    let mut urls = Vec::new();

    // Check if this node's bounding volume intersects our polygon
    if let Some(ref bv) = node.bounding_volume {
        if !bounding_volume_intersects_polygon(bv, polygon) {
            return urls;
        }
    }

    // If we've reached sufficient detail or this is a leaf node
    if node.geometric_error <= target_error || node.children.is_empty() {
        if let Some(ref content) = node.content {
            let url = if content.uri.starts_with("http") {
                format!("{}&key={}", content.uri, api_key)
            } else {
                format!("{}/{}?key={}", TILES_API_BASE, content.uri, api_key)
            };
            urls.push(url);
        }
    } else {
        // Recurse into children
        for child in &node.children {
            urls.extend(collect_tile_urls(child, polygon, target_error, api_key));
        }
    }

    urls
}

/// Check if a bounding volume intersects a polygon defined by lat/lng points.
fn bounding_volume_intersects_polygon(bv: &BoundingVolume, polygon: &[(f64, f64)]) -> bool {
    if let Some(ref region) = bv.region {
        // Region format: [west, south, east, north, minHeight, maxHeight] in radians
        if region.len() >= 4 {
            let west = region[0].to_degrees();
            let south = region[1].to_degrees();
            let east = region[2].to_degrees();
            let north = region[3].to_degrees();

            // Simple bounding box intersection test with polygon AABB
            let (min_lat, max_lat, min_lng, max_lng) = polygon_aabb(polygon);
            return !(east < min_lng || west > max_lng || north < min_lat || south > max_lat);
        }
    }

    // For box or sphere bounding volumes, conservatively return true
    // (proper intersection testing requires more geometry)
    true
}

/// Compute axis-aligned bounding box of a polygon.
fn polygon_aabb(polygon: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let mut min_lat = f64::MAX;
    let mut max_lat = f64::MIN;
    let mut min_lng = f64::MAX;
    let mut max_lng = f64::MIN;

    for &(lat, lng) in polygon {
        min_lat = min_lat.min(lat);
        max_lat = max_lat.max(lat);
        min_lng = min_lng.min(lng);
        max_lng = max_lng.max(lng);
    }

    (min_lat, max_lat, min_lng, max_lng)
}
