//! Tileset tree traversal and LOD selection.
//!
//! Provides utilities for walking the 3D Tiles tileset hierarchy
//! and selecting appropriate tiles based on geometric error thresholds.

use pyo3::prelude::*;
use serde::Deserialize;

/// Represents a node in the 3D Tiles tileset hierarchy (Python-visible).
#[pyclass]
#[derive(Clone, Debug)]
pub struct TileNode {
    #[pyo3(get)]
    pub content_uri: Option<String>,
    #[pyo3(get)]
    pub geometric_error: f64,
    #[pyo3(get)]
    pub bounding_region: Option<Vec<f64>>,
    #[pyo3(get)]
    pub child_count: usize,
    #[pyo3(get)]
    pub depth: usize,
}

#[pymethods]
impl TileNode {
    fn __repr__(&self) -> String {
        format!(
            "TileNode(uri={:?}, error={:.1}, children={}, depth={})",
            self.content_uri, self.geometric_error, self.child_count, self.depth
        )
    }
}

/// Internal tileset node for tree traversal.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InternalTileNode {
    #[serde(default)]
    pub bounding_volume: Option<InternalBoundingVolume>,
    #[serde(default)]
    pub geometric_error: f64,
    #[serde(default)]
    pub content: Option<InternalContent>,
    #[serde(default)]
    pub children: Vec<InternalTileNode>,
    #[serde(default)]
    pub refine: Option<String>,
    #[serde(default)]
    pub transform: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InternalBoundingVolume {
    #[serde(default)]
    pub region: Option<Vec<f64>>,
    #[serde(default, rename = "box")]
    pub bbox: Option<Vec<f64>>,
    #[serde(default)]
    pub sphere: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct InternalContent {
    pub uri: String,
}

/// Select tiles from a tileset tree that intersect a polygon and meet the LOD threshold.
///
/// Returns a flat list of (content_uri, transform) pairs for tiles to fetch.
pub fn select_tiles(
    root: &InternalTileNode,
    polygon_aabb: (f64, f64, f64, f64), // (min_lat, max_lat, min_lng, max_lng)
    target_geometric_error: f64,
    parent_transform: &[f64; 16],
    depth: usize,
) -> Vec<(String, [f64; 16])> {
    let mut result = Vec::new();

    // Compute this node's accumulated transform
    let node_transform = if let Some(ref t) = root.transform {
        multiply_4x4(parent_transform, &array_from_vec(t))
    } else {
        *parent_transform
    };

    // Check bounding volume intersection
    if let Some(ref bv) = root.bounding_volume {
        if !intersects_aabb(bv, polygon_aabb) {
            return result;
        }
    }

    // Decide whether to use this node or drill deeper
    let should_use = root.geometric_error <= target_geometric_error || root.children.is_empty();

    if should_use {
        if let Some(ref content) = root.content {
            result.push((content.uri.clone(), node_transform));
        }
    } else {
        for child in &root.children {
            result.extend(select_tiles(
                child,
                polygon_aabb,
                target_geometric_error,
                &node_transform,
                depth + 1,
            ));
        }
    }

    result
}

/// Check if a bounding volume intersects an AABB defined in lat/lng degrees.
fn intersects_aabb(bv: &InternalBoundingVolume, aabb: (f64, f64, f64, f64)) -> bool {
    let (min_lat, max_lat, min_lng, max_lng) = aabb;

    if let Some(ref region) = bv.region {
        if region.len() >= 4 {
            let west = region[0].to_degrees();
            let south = region[1].to_degrees();
            let east = region[2].to_degrees();
            let north = region[3].to_degrees();
            return !(east < min_lng || west > max_lng || north < min_lat || south > max_lat);
        }
    }

    // Conservatively include for box/sphere bounding volumes
    true
}

/// Multiply two 4x4 matrices (column-major order as used in glTF).
fn multiply_4x4(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut result = [0.0f64; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[row + k * 4] * b[k + col * 4];
            }
            result[row + col * 4] = sum;
        }
    }
    result
}

/// Convert a Vec<f64> to a [f64; 16] array, padding with identity if needed.
fn array_from_vec(v: &[f64]) -> [f64; 16] {
    let mut arr = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    for (i, val) in v.iter().enumerate().take(16) {
        arr[i] = *val;
    }
    arr
}
