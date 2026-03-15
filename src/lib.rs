//! Rust core for tiles-to-mesh.
//!
//! Provides high-performance tile fetching, GLB parsing, mesh merging,
//! polygon clipping, and coordinate transforms exposed to Python via PyO3.

pub mod geo;
pub mod mesh;
pub mod tiles;

use pyo3::prelude::*;

/// The native Rust module exposed to Python as `_tiles_to_mesh_rs`.
#[pymodule]
fn _tiles_to_mesh_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Geo submodule
    let geo_module = PyModule::new_bound(m.py(), "geo")?;
    geo::register_module(&geo_module)?;
    m.add_submodule(&geo_module)?;

    // Mesh submodule
    let mesh_module = PyModule::new_bound(m.py(), "mesh")?;
    mesh::register_module(&mesh_module)?;
    m.add_submodule(&mesh_module)?;

    // Tiles submodule
    let tiles_module = PyModule::new_bound(m.py(), "tiles")?;
    tiles::register_module(&tiles_module)?;
    m.add_submodule(&tiles_module)?;

    Ok(())
}
