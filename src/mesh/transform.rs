//! Apply 4x4 transform matrices to mesh vertex data.
//!
//! Used to apply tile-level transforms from the 3D Tiles tileset
//! to convert from ECEF to local coordinates.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use super::merge::MergedMesh;

/// Apply a 4x4 transformation matrix (column-major) to all vertex positions in a mesh.
pub fn apply_transform(mesh: &mut MergedMesh, matrix: &[f64; 16]) {
    let vertex_count = mesh.positions.len() / 3;
    let has_normals = mesh.normals.len() == mesh.positions.len();

    // Extract the 3x3 rotation/scale part for transforming normals
    let normal_matrix = if has_normals {
        Some(inverse_transpose_3x3(matrix))
    } else {
        None
    };

    for i in 0..vertex_count {
        let x = mesh.positions[i * 3] as f64;
        let y = mesh.positions[i * 3 + 1] as f64;
        let z = mesh.positions[i * 3 + 2] as f64;

        // column-major: M[row + col*4]
        let nx = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
        let ny = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
        let nz = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];

        mesh.positions[i * 3] = nx as f32;
        mesh.positions[i * 3 + 1] = ny as f32;
        mesh.positions[i * 3 + 2] = nz as f32;

        // Transform normals
        if let Some(ref nm) = normal_matrix {
            if has_normals {
                let onx = mesh.normals[i * 3] as f64;
                let ony = mesh.normals[i * 3 + 1] as f64;
                let onz = mesh.normals[i * 3 + 2] as f64;

                let tnx = nm[0] * onx + nm[3] * ony + nm[6] * onz;
                let tny = nm[1] * onx + nm[4] * ony + nm[7] * onz;
                let tnz = nm[2] * onx + nm[5] * ony + nm[8] * onz;

                // Normalize
                let len = (tnx * tnx + tny * tny + tnz * tnz).sqrt();
                if len > 1e-10 {
                    mesh.normals[i * 3] = (tnx / len) as f32;
                    mesh.normals[i * 3 + 1] = (tny / len) as f32;
                    mesh.normals[i * 3 + 2] = (tnz / len) as f32;
                }
            }
        }
    }
}

/// Compute the inverse-transpose of the upper-left 3x3 part of a 4x4 matrix.
/// This is used for correctly transforming normals.
fn inverse_transpose_3x3(m: &[f64; 16]) -> [f64; 9] {
    // Extract 3x3 (column-major)
    let a = m[0]; let b = m[1]; let c = m[2];
    let d = m[4]; let e = m[5]; let f = m[6];
    let g = m[8]; let h = m[9]; let i = m[10];

    let det = a * (e * i - f * h) - d * (b * i - c * h) + g * (b * f - c * e);

    if det.abs() < 1e-12 {
        // Fallback to identity
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }

    let inv_det = 1.0 / det;

    // Cofactor matrix (already transposed due to how we compute it)
    [
        (e * i - f * h) * inv_det,
        (f * g - d * i) * inv_det,
        (d * h - e * g) * inv_det,
        (c * h - b * i) * inv_det,
        (a * i - c * g) * inv_det,
        (b * g - a * h) * inv_det,
        (b * f - c * e) * inv_det,
        (c * d - a * f) * inv_det,
        (a * e - b * d) * inv_det,
    ]
}

/// Python-exposed transform function.
#[pyfunction]
#[pyo3(name = "apply_transform")]
pub fn apply_transform_py(mesh: &mut MergedMesh, matrix: Vec<f64>) -> PyResult<()> {
    if matrix.len() != 16 {
        return Err(PyValueError::new_err("Transform matrix must have 16 elements (4x4, column-major)"));
    }
    let mut arr = [0.0f64; 16];
    arr.copy_from_slice(&matrix);
    apply_transform(mesh, &arr);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::merge::MergedMesh;

    #[test]
    fn test_identity_transform() {
        let mut mesh = MergedMesh {
            positions: vec![1.0, 2.0, 3.0],
            normals: vec![0.0, 0.0, 1.0],
            texcoords: vec![0.5, 0.5],
            indices: vec![0],
            source_tile_count: 1,
        };

        let identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];

        apply_transform(&mut mesh, &identity);
        assert!((mesh.positions[0] - 1.0).abs() < 1e-6);
        assert!((mesh.positions[1] - 2.0).abs() < 1e-6);
        assert!((mesh.positions[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_translation_transform() {
        let mut mesh = MergedMesh {
            positions: vec![0.0, 0.0, 0.0],
            normals: vec![0.0, 0.0, 1.0],
            texcoords: vec![],
            indices: vec![0],
            source_tile_count: 1,
        };

        // Translate by (10, 20, 30)
        let translate = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            10.0, 20.0, 30.0, 1.0,
        ];

        apply_transform(&mut mesh, &translate);
        assert!((mesh.positions[0] - 10.0).abs() < 1e-6);
        assert!((mesh.positions[1] - 20.0).abs() < 1e-6);
        assert!((mesh.positions[2] - 30.0).abs() < 1e-6);
    }
}
