//! Mesh processing operations implemented in Rust for performance.
//!
//! Handles mesh merging, polygon clipping, and coordinate transforms.

pub mod clip;
pub mod merge;
pub mod transform;

use pyo3::prelude::*;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(merge::merge_tile_meshes_py, m)?)?;
    m.add_function(wrap_pyfunction!(clip::clip_mesh_to_polygon_py, m)?)?;
    m.add_function(wrap_pyfunction!(transform::apply_transform_py, m)?)?;
    m.add_class::<merge::MergedMesh>()?;
    Ok(())
}
