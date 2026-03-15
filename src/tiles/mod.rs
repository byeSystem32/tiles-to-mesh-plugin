//! 3D Tile fetching and parsing.
//!
//! Handles interaction with the Google Map Tiles API (3D Tiles endpoint),
//! tileset tree traversal, LOD selection, and parallel tile downloading.

pub mod fetcher;
pub mod parser;
pub mod traversal;

use pyo3::prelude::*;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<fetcher::TileFetcher>()?;
    m.add_class::<fetcher::FetchConfig>()?;
    m.add_class::<parser::ParsedTile>()?;
    m.add_class::<parser::TileMeshData>()?;
    m.add_class::<traversal::TileNode>()?;
    Ok(())
}
