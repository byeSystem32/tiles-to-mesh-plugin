//! Clip mesh geometry to a polygon boundary.
//!
//! Removes triangles whose centroids fall outside the user-defined selection polygon.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use super::merge::MergedMesh;

/// Clip a merged mesh to only include triangles inside the given polygon.
///
/// Uses a point-in-polygon (ray casting) test on triangle centroids.
/// This is a conservative approach — triangles on the boundary may be
/// included or excluded based on their centroid position.
pub fn clip_to_polygon(mesh: &MergedMesh, polygon: &[(f64, f64)]) -> MergedMesh {
    let mut new_indices = Vec::new();

    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }

        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);

        // Compute centroid (using x/z as lat/lng proxy — actual mapping depends on coord system)
        let cx = (mesh.positions[i0 * 3] + mesh.positions[i1 * 3] + mesh.positions[i2 * 3]) / 3.0;
        let cy =
            (mesh.positions[i0 * 3 + 2] + mesh.positions[i1 * 3 + 2] + mesh.positions[i2 * 3 + 2])
                / 3.0;

        if point_in_polygon(cx as f64, cy as f64, polygon) {
            new_indices.extend_from_slice(tri);
        }
    }

    // Note: we keep all vertices even if some become unreferenced.
    // A subsequent cleanup pass can remove them if needed.
    MergedMesh {
        positions: mesh.positions.clone(),
        normals: mesh.normals.clone(),
        texcoords: mesh.texcoords.clone(),
        indices: new_indices,
        source_tile_count: mesh.source_tile_count,
    }
}

/// Ray-casting point-in-polygon test.
///
/// Returns true if the point (px, py) is inside the polygon defined by the given vertices.
fn point_in_polygon(px: f64, py: f64, polygon: &[(f64, f64)]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = n - 1;

    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];

        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }

        j = i;
    }

    inside
}

/// Python-exposed clipping function.
#[pyfunction]
#[pyo3(name = "clip_mesh_to_polygon")]
pub fn clip_mesh_to_polygon_py(
    mesh: &MergedMesh,
    polygon: Vec<(f64, f64)>,
) -> PyResult<MergedMesh> {
    if polygon.len() < 3 {
        return Err(PyValueError::new_err(
            "Polygon must have at least 3 vertices",
        ));
    }
    Ok(clip_to_polygon(mesh, &polygon))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_in_polygon() {
        let square = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        assert!(point_in_polygon(5.0, 5.0, &square));
        assert!(!point_in_polygon(15.0, 5.0, &square));
        assert!(!point_in_polygon(-1.0, -1.0, &square));
    }

    #[test]
    fn test_point_in_triangle() {
        let triangle = vec![(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
        assert!(point_in_polygon(5.0, 3.0, &triangle));
        assert!(!point_in_polygon(0.0, 10.0, &triangle));
    }
}
