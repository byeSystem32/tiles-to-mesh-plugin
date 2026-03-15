//! WGS84 ↔ ECEF ↔ ENU coordinate transforms.
//!
//! These are essential for converting Google 3D Tiles (which use ECEF coordinates)
//! into a local coordinate system suitable for mesh processing.

use numpy::{PyArray1, PyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::f64::consts::PI;

// WGS84 ellipsoid constants
const WGS84_A: f64 = 6_378_137.0; // Semi-major axis (meters)
const WGS84_B: f64 = 6_356_752.314_245; // Semi-minor axis (meters)
const WGS84_E2: f64 = 1.0 - (WGS84_B * WGS84_B) / (WGS84_A * WGS84_A); // First eccentricity squared

/// A geographic point in WGS84 coordinates.
#[pyclass]
#[derive(Clone, Debug)]
pub struct GeoPoint {
    #[pyo3(get, set)]
    pub lat: f64,
    #[pyo3(get, set)]
    pub lng: f64,
    #[pyo3(get, set)]
    pub alt: f64,
}

#[pymethods]
impl GeoPoint {
    #[new]
    #[pyo3(signature = (lat, lng, alt=0.0))]
    fn new(lat: f64, lng: f64, alt: f64) -> Self {
        Self { lat, lng, alt }
    }

    fn __repr__(&self) -> String {
        format!("GeoPoint(lat={}, lng={}, alt={})", self.lat, self.lng, self.alt)
    }

    /// Convert to ECEF coordinates, returns (x, y, z) in meters.
    fn to_ecef(&self) -> (f64, f64, f64) {
        wgs84_to_ecef(self.lat, self.lng, self.alt)
    }
}

/// Origin for ENU (East-North-Up) local coordinate system.
#[pyclass]
#[derive(Clone, Debug)]
pub struct EnuOrigin {
    #[pyo3(get)]
    pub lat: f64,
    #[pyo3(get)]
    pub lng: f64,
    #[pyo3(get)]
    pub alt: f64,
    // Precomputed ECEF of origin
    ecef_x: f64,
    ecef_y: f64,
    ecef_z: f64,
    // Precomputed sin/cos for rotation matrix
    sin_lat: f64,
    cos_lat: f64,
    sin_lng: f64,
    cos_lng: f64,
}

#[pymethods]
impl EnuOrigin {
    #[new]
    #[pyo3(signature = (lat, lng, alt=0.0))]
    fn new(lat: f64, lng: f64, alt: f64) -> Self {
        let (ecef_x, ecef_y, ecef_z) = wgs84_to_ecef(lat, lng, alt);
        let lat_rad = lat.to_radians();
        let lng_rad = lng.to_radians();
        Self {
            lat,
            lng,
            alt,
            ecef_x,
            ecef_y,
            ecef_z,
            sin_lat: lat_rad.sin(),
            cos_lat: lat_rad.cos(),
            sin_lng: lng_rad.sin(),
            cos_lng: lng_rad.cos(),
        }
    }

    fn __repr__(&self) -> String {
        format!("EnuOrigin(lat={}, lng={}, alt={})", self.lat, self.lng, self.alt)
    }
}

/// Convert WGS84 (lat, lng, alt) to ECEF (x, y, z) in meters.
pub fn wgs84_to_ecef(lat_deg: f64, lng_deg: f64, alt: f64) -> (f64, f64, f64) {
    let lat = lat_deg * PI / 180.0;
    let lng = lng_deg * PI / 180.0;
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let sin_lng = lng.sin();
    let cos_lng = lng.cos();

    let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();

    let x = (n + alt) * cos_lat * cos_lng;
    let y = (n + alt) * cos_lat * sin_lng;
    let z = (n * (1.0 - WGS84_E2) + alt) * sin_lat;

    (x, y, z)
}

/// Convert ECEF to ENU relative to an origin.
pub fn ecef_to_enu(
    ecef_x: f64,
    ecef_y: f64,
    ecef_z: f64,
    origin: &EnuOrigin,
) -> (f64, f64, f64) {
    let dx = ecef_x - origin.ecef_x;
    let dy = ecef_y - origin.ecef_y;
    let dz = ecef_z - origin.ecef_z;

    let east = -origin.sin_lng * dx + origin.cos_lng * dy;
    let north =
        -origin.sin_lat * origin.cos_lng * dx
        - origin.sin_lat * origin.sin_lng * dy
        + origin.cos_lat * dz;
    let up =
        origin.cos_lat * origin.cos_lng * dx
        + origin.cos_lat * origin.sin_lng * dy
        + origin.sin_lat * dz;

    (east, north, up)
}

/// Convert WGS84 directly to ENU relative to an origin.
pub fn wgs84_to_enu(lat: f64, lng: f64, alt: f64, origin: &EnuOrigin) -> (f64, f64, f64) {
    let (ex, ey, ez) = wgs84_to_ecef(lat, lng, alt);
    ecef_to_enu(ex, ey, ez, origin)
}

// ── Python-exposed functions ──────────────────────────────────────────

#[pyfunction]
#[pyo3(name = "wgs84_to_ecef")]
pub fn wgs84_to_ecef_py(py: Python<'_>, lat: f64, lng: f64, alt: f64) -> Py<PyArray1<f64>> {
    let (x, y, z) = wgs84_to_ecef(lat, lng, alt);
    PyArray1::from_vec_bound(py, vec![x, y, z]).into()
}

#[pyfunction]
#[pyo3(name = "ecef_to_enu")]
pub fn ecef_to_enu_py(
    py: Python<'_>,
    x: f64,
    y: f64,
    z: f64,
    origin: &EnuOrigin,
) -> Py<PyArray1<f64>> {
    let (e, n, u) = ecef_to_enu(x, y, z, origin);
    PyArray1::from_vec_bound(py, vec![e, n, u]).into()
}

#[pyfunction]
#[pyo3(name = "wgs84_to_enu")]
pub fn wgs84_to_enu_py(
    py: Python<'_>,
    lat: f64,
    lng: f64,
    alt: f64,
    origin: &EnuOrigin,
) -> Py<PyArray1<f64>> {
    let (e, n, u) = wgs84_to_enu(lat, lng, alt, origin);
    PyArray1::from_vec_bound(py, vec![e, n, u]).into()
}

/// Batch-transform an Nx3 array of ECEF coords to ENU.
/// Input: numpy array shape (N, 3) of [x, y, z] ECEF.
/// Returns: numpy array shape (N, 3) of [east, north, up] ENU.
#[pyfunction]
#[pyo3(name = "batch_ecef_to_enu")]
pub fn batch_ecef_to_enu_py<'py>(
    py: Python<'py>,
    ecef: &Bound<'py, PyArray2<f64>>,
    origin: &EnuOrigin,
) -> PyResult<Py<PyArray2<f64>>> {
    use numpy::PyArrayMethods;
    let readonly = ecef.readonly();
    let shape = readonly.shape();
    let rows = shape[0];
    let data = readonly.as_slice()?;
    let mut result = Vec::with_capacity(rows * 3);

    for i in 0..rows {
        let x = data[i * 3];
        let y = data[i * 3 + 1];
        let z = data[i * 3 + 2];
        let (e, n, u) = ecef_to_enu(x, y, z, origin);
        result.push(e);
        result.push(n);
        result.push(u);
    }

    Ok(PyArray2::from_vec2_bound(py, &result.chunks(3).map(|c| c.to_vec()).collect::<Vec<_>>())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?
        .into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgs84_to_ecef_origin() {
        // Equator, prime meridian, sea level
        let (x, y, z) = wgs84_to_ecef(0.0, 0.0, 0.0);
        assert!((x - WGS84_A).abs() < 1.0); // Should be ~6378137m on X axis
        assert!(y.abs() < 1.0);
        assert!(z.abs() < 1.0);
    }

    #[test]
    fn test_ecef_to_enu_identity() {
        let origin = EnuOrigin::new(0.0, 0.0, 0.0);
        let (e, n, u) = ecef_to_enu(origin.ecef_x, origin.ecef_y, origin.ecef_z, &origin);
        assert!(e.abs() < 1e-6);
        assert!(n.abs() < 1e-6);
        assert!(u.abs() < 1e-6);
    }

    #[test]
    fn test_roundtrip_consistency() {
        let lat = 37.7749;
        let lng = -122.4194;
        let alt = 10.0;
        let origin = EnuOrigin::new(lat, lng, 0.0);
        let (e, n, u) = wgs84_to_enu(lat, lng, alt, &origin);
        // Should be near origin, only elevated
        assert!(e.abs() < 0.01);
        assert!(n.abs() < 0.01);
        assert!((u - alt).abs() < 0.01);
    }
}
