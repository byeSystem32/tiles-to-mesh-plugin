//! Merge multiple tile meshes into a single unified mesh.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyList;

/// A merged mesh containing combined geometry from multiple tiles.
#[pyclass]
#[derive(Clone, Debug)]
pub struct MergedMesh {
    /// Flat vertex positions [x, y, z, ...]
    #[pyo3(get)]
    pub positions: Vec<f32>,
    /// Flat vertex normals [nx, ny, nz, ...]
    #[pyo3(get)]
    pub normals: Vec<f32>,
    /// Flat texture coordinates [u, v, ...]
    #[pyo3(get)]
    pub texcoords: Vec<f32>,
    /// Triangle indices
    #[pyo3(get)]
    pub indices: Vec<u32>,
    /// Number of source tiles that were merged
    #[pyo3(get)]
    pub source_tile_count: usize,
}

#[pymethods]
impl MergedMesh {
    #[getter]
    fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }

    #[getter]
    fn face_count(&self) -> usize {
        self.indices.len() / 3
    }

    fn __repr__(&self) -> String {
        format!(
            "MergedMesh(vertices={}, faces={}, from_tiles={})",
            self.vertex_count(),
            self.face_count(),
            self.source_tile_count
        )
    }
}

/// Internal struct for mesh data during merging.
pub struct RawMeshChunk {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub texcoords: Vec<f32>,
    pub indices: Vec<u32>,
}

/// Merge multiple mesh chunks into a single mesh.
///
/// Handles index offsetting so that indices from each chunk correctly reference
/// their vertices in the merged buffer.
pub fn merge_meshes(chunks: &[RawMeshChunk]) -> MergedMesh {
    let total_verts: usize = chunks.iter().map(|c| c.positions.len()).sum();
    let total_indices: usize = chunks.iter().map(|c| c.indices.len()).sum();

    let mut positions = Vec::with_capacity(total_verts);
    let mut normals = Vec::with_capacity(total_verts);
    let mut texcoords = Vec::with_capacity(chunks.iter().map(|c| c.texcoords.len()).sum());
    let mut indices = Vec::with_capacity(total_indices);

    let mut vertex_offset: u32 = 0;

    for chunk in chunks {
        positions.extend_from_slice(&chunk.positions);
        normals.extend_from_slice(&chunk.normals);
        texcoords.extend_from_slice(&chunk.texcoords);

        for &idx in &chunk.indices {
            indices.push(idx + vertex_offset);
        }

        vertex_offset += (chunk.positions.len() / 3) as u32;
    }

    MergedMesh {
        positions,
        normals,
        texcoords,
        indices,
        source_tile_count: chunks.len(),
    }
}

/// Python-exposed function to merge tile mesh data objects.
///
/// Takes a list of TileMeshData objects (from the parser) and returns a MergedMesh.
#[pyfunction]
#[pyo3(name = "merge_tile_meshes")]
pub fn merge_tile_meshes_py(tile_meshes: &Bound<'_, PyList>) -> PyResult<MergedMesh> {
    use crate::tiles::parser::TileMeshData;

    let mut chunks = Vec::with_capacity(tile_meshes.len());

    for item in tile_meshes.iter() {
        let tile: PyRef<TileMeshData> = item.extract()?;
        chunks.push(RawMeshChunk {
            positions: tile.positions.clone(),
            normals: tile.normals.clone(),
            texcoords: tile.texcoords.clone(),
            indices: tile.indices.clone(),
        });
    }

    if chunks.is_empty() {
        return Err(PyValueError::new_err("No meshes to merge"));
    }

    Ok(merge_meshes(&chunks))
}

/// Remove duplicate vertices within a distance threshold.
/// Returns a new MergedMesh with deduplicated vertices and remapped indices.
pub fn remove_duplicate_vertices(mesh: &MergedMesh, threshold: f32) -> MergedMesh {
    let vertex_count = mesh.positions.len() / 3;
    let threshold_sq = threshold * threshold;

    // Map old vertex index → new vertex index
    let mut remap = vec![0u32; vertex_count];
    let mut new_positions = Vec::new();
    let mut new_normals = Vec::new();
    let mut new_texcoords = Vec::new();
    let mut new_count: u32 = 0;

    let has_normals = mesh.normals.len() == mesh.positions.len();
    let has_texcoords = mesh.texcoords.len() / 2 == vertex_count;

    for i in 0..vertex_count {
        let px = mesh.positions[i * 3];
        let py = mesh.positions[i * 3 + 1];
        let pz = mesh.positions[i * 3 + 2];

        // Check against existing new vertices
        let mut found = None;
        for j in 0..(new_count as usize) {
            let dx = new_positions[j * 3] - px;
            let dy = new_positions[j * 3 + 1] - py;
            let dz = new_positions[j * 3 + 2] - pz;
            if dx * dx + dy * dy + dz * dz < threshold_sq {
                found = Some(j as u32);
                break;
            }
        }

        match found {
            Some(existing) => {
                remap[i] = existing;
            }
            None => {
                remap[i] = new_count;
                new_positions.extend_from_slice(&[px, py, pz]);
                if has_normals {
                    new_normals.extend_from_slice(&mesh.normals[i * 3..i * 3 + 3]);
                }
                if has_texcoords {
                    new_texcoords.extend_from_slice(&mesh.texcoords[i * 2..i * 2 + 2]);
                }
                new_count += 1;
            }
        }
    }

    // Remap indices
    let new_indices: Vec<u32> = mesh.indices.iter().map(|&idx| remap[idx as usize]).collect();

    MergedMesh {
        positions: new_positions,
        normals: new_normals,
        texcoords: new_texcoords,
        indices: new_indices,
        source_tile_count: mesh.source_tile_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_two_triangles() {
        let a = RawMeshChunk {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            normals: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            texcoords: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            indices: vec![0, 1, 2],
        };
        let b = RawMeshChunk {
            positions: vec![2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 1.0, 0.0],
            normals: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            texcoords: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            indices: vec![0, 1, 2],
        };

        let merged = merge_meshes(&[a, b]);
        assert_eq!(merged.vertex_count(), 6);
        assert_eq!(merged.face_count(), 2);
        assert_eq!(merged.indices, vec![0, 1, 2, 3, 4, 5]);
    }
}
