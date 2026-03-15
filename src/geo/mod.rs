//! Geospatial coordinate conversions.
//!
//! Supports WGS84 (lat/lng) ↔ ECEF ↔ ENU (local tangent plane) transforms.

pub mod coords;

use pyo3::prelude::*;

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(coords::wgs84_to_ecef_py, m)?)?;
    m.add_function(wrap_pyfunction!(coords::ecef_to_enu_py, m)?)?;
    m.add_function(wrap_pyfunction!(coords::wgs84_to_enu_py, m)?)?;
    m.add_class::<coords::GeoPoint>()?;
    m.add_class::<coords::EnuOrigin>()?;
    Ok(())
}
