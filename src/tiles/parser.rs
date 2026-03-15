//! GLB/glTF binary parsing.
//!
//! Extracts vertex positions, normals, texture coordinates, indices,
//! and texture images from GLB tile data.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Raw mesh data extracted from a single GLB tile.
#[pyclass]
#[derive(Clone, Debug)]
pub struct TileMeshData {
    /// Flat array of vertex positions [x, y, z, x, y, z, ...]
    #[pyo3(get)]
    pub positions: Vec<f32>,
    /// Flat array of vertex normals [nx, ny, nz, ...]
    #[pyo3(get)]
    pub normals: Vec<f32>,
    /// Flat array of texture coordinates [u, v, u, v, ...]
    #[pyo3(get)]
    pub texcoords: Vec<f32>,
    /// Triangle indices
    #[pyo3(get)]
    pub indices: Vec<u32>,
    /// Raw texture image bytes (PNG/JPEG), if present
    texture_data: Option<Vec<u8>>,
    /// MIME type of texture ("image/jpeg", "image/png")
    #[pyo3(get)]
    pub texture_mime: Option<String>,
    /// The 4x4 transform matrix for this tile (column-major)
    #[pyo3(get)]
    pub transform: Vec<f64>,
}

#[pymethods]
impl TileMeshData {
    #[getter]
    fn vertex_count(&self) -> usize {
        self.positions.len() / 3
    }

    #[getter]
    fn face_count(&self) -> usize {
        self.indices.len() / 3
    }

    #[getter]
    fn has_texture(&self) -> bool {
        self.texture_data.is_some()
    }

    /// Get raw texture bytes as Python bytes object.
    fn get_texture_bytes<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        match &self.texture_data {
            Some(data) => Ok(Some(PyBytes::new_bound(py, data))),
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TileMeshData(vertices={}, faces={}, textured={})",
            self.vertex_count(),
            self.face_count(),
            self.has_texture()
        )
    }
}

/// High-level wrapper exposing parsed tile info.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ParsedTile {
    #[pyo3(get)]
    pub meshes: Vec<TileMeshData>,
    #[pyo3(get)]
    pub name: String,
}

#[pymethods]
impl ParsedTile {
    fn __repr__(&self) -> String {
        format!(
            "ParsedTile(name='{}', meshes={})",
            self.name,
            self.meshes.len()
        )
    }
}

/// Parse a GLB binary blob into mesh data.
///
/// This uses the `gltf` crate to parse the binary glTF format and extract
/// all mesh primitives with their geometry and texture data.
pub fn parse_glb_bytes(data: &[u8]) -> Result<TileMeshData, Box<dyn std::error::Error + Send + Sync>> {
    let glb = gltf::Glb::from_slice(data)?;
    let gltf = gltf::Gltf::from_slice(&glb.json)?;
    let bin = glb.bin.unwrap_or_default();

    let mut all_positions: Vec<f32> = Vec::new();
    let mut all_normals: Vec<f32> = Vec::new();
    let mut all_texcoords: Vec<f32> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();
    let mut texture_data: Option<Vec<u8>> = None;
    let mut texture_mime: Option<String> = None;

    // Extract the root transform if present
    let mut transform = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    if let Some(node) = gltf.nodes().next() {
        let m = node.transform().matrix();
        transform = vec![
            m[0][0] as f64, m[0][1] as f64, m[0][2] as f64, m[0][3] as f64,
            m[1][0] as f64, m[1][1] as f64, m[1][2] as f64, m[1][3] as f64,
            m[2][0] as f64, m[2][1] as f64, m[2][2] as f64, m[2][3] as f64,
            m[3][0] as f64, m[3][1] as f64, m[3][2] as f64, m[3][3] as f64,
        ];
    }

    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            let base_vertex = (all_positions.len() / 3) as u32;

            // ── Positions ──
            if let Some(accessor) = primitive.get(&gltf::Semantic::Positions) {
                let view = accessor.view().unwrap();
                let offset = view.offset() + accessor.offset();
                let count = accessor.count();
                let stride = view.stride().unwrap_or(12); // 3 * f32

                for i in 0..count {
                    let start = offset + i * stride;
                    let x = f32::from_le_bytes(bin[start..start + 4].try_into().unwrap());
                    let y = f32::from_le_bytes(bin[start + 4..start + 8].try_into().unwrap());
                    let z = f32::from_le_bytes(bin[start + 8..start + 12].try_into().unwrap());
                    all_positions.extend_from_slice(&[x, y, z]);
                }
            }

            // ── Normals ──
            if let Some(accessor) = primitive.get(&gltf::Semantic::Normals) {
                let view = accessor.view().unwrap();
                let offset = view.offset() + accessor.offset();
                let count = accessor.count();
                let stride = view.stride().unwrap_or(12);

                for i in 0..count {
                    let start = offset + i * stride;
                    let nx = f32::from_le_bytes(bin[start..start + 4].try_into().unwrap());
                    let ny = f32::from_le_bytes(bin[start + 4..start + 8].try_into().unwrap());
                    let nz = f32::from_le_bytes(bin[start + 8..start + 12].try_into().unwrap());
                    all_normals.extend_from_slice(&[nx, ny, nz]);
                }
            }

            // ── Texture Coordinates ──
            if let Some(accessor) = primitive.get(&gltf::Semantic::TexCoords(0)) {
                let view = accessor.view().unwrap();
                let offset = view.offset() + accessor.offset();
                let count = accessor.count();
                let stride = view.stride().unwrap_or(8); // 2 * f32

                for i in 0..count {
                    let start = offset + i * stride;
                    let u = f32::from_le_bytes(bin[start..start + 4].try_into().unwrap());
                    let v = f32::from_le_bytes(bin[start + 4..start + 8].try_into().unwrap());
                    all_texcoords.extend_from_slice(&[u, v]);
                }
            }

            // ── Indices ──
            if let Some(accessor) = primitive.indices() {
                let view = accessor.view().unwrap();
                let offset = view.offset() + accessor.offset();
                let count = accessor.count();

                match accessor.data_type() {
                    gltf::accessor::DataType::U16 => {
                        let stride = view.stride().unwrap_or(2);
                        for i in 0..count {
                            let start = offset + i * stride;
                            let idx = u16::from_le_bytes(bin[start..start + 2].try_into().unwrap());
                            all_indices.push(base_vertex + idx as u32);
                        }
                    }
                    gltf::accessor::DataType::U32 => {
                        let stride = view.stride().unwrap_or(4);
                        for i in 0..count {
                            let start = offset + i * stride;
                            let idx = u32::from_le_bytes(bin[start..start + 4].try_into().unwrap());
                            all_indices.push(base_vertex + idx);
                        }
                    }
                    gltf::accessor::DataType::U8 => {
                        let stride = view.stride().unwrap_or(1);
                        for i in 0..count {
                            let start = offset + i * stride;
                            all_indices.push(base_vertex + bin[start] as u32);
                        }
                    }
                    _ => {}
                }
            }

            // ── Texture ──
            if texture_data.is_none() {
                if let Some(material) = Some(primitive.material()) {
                    if let Some(pbr) = Some(material.pbr_metallic_roughness()) {
                        if let Some(tex_info) = pbr.base_color_texture() {
                            let texture = tex_info.texture();
                            let source = texture.source();
                            match source.source() {
                                gltf::image::Source::View { view, mime_type } => {
                                    let img_offset = view.offset();
                                    let img_len = view.length();
                                    texture_data =
                                        Some(bin[img_offset..img_offset + img_len].to_vec());
                                    texture_mime = Some(mime_type.to_string());
                                }
                                gltf::image::Source::Uri { uri, mime_type } => {
                                    // Handle data URIs
                                    if let Some(data_str) = uri.strip_prefix("data:") {
                                        if let Some((_mime, b64)) = data_str.split_once(";base64,") {
                                            if let Ok(decoded) = base64::Engine::decode(
                                                &base64::engine::general_purpose::STANDARD,
                                                b64,
                                            ) {
                                                texture_data = Some(decoded);
                                                texture_mime = mime_type
                                                    .map(|s| s.to_string())
                                                    .or_else(|| Some("image/png".to_string()));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(TileMeshData {
        positions: all_positions,
        normals: all_normals,
        texcoords: all_texcoords,
        indices: all_indices,
        texture_data,
        texture_mime,
        transform,
    })
}
