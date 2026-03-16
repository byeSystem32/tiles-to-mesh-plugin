"""
Interactive Three.js mesh viewer for Jupyter/Colab notebooks.

Uses pythreejs to render a 3D preview of the mesh with orbit controls,
lighting, wireframe toggle, and texture display.

In Google Colab the pythreejs widget-sync layer has trait incompatibilities
(shadow, target, etc.) so we fall back to a pure-HTML Three.js viewer that
avoids the widget layer entirely.
"""

from __future__ import annotations

import sys
from typing import Any, Optional, Tuple

import numpy as np


def _is_colab() -> bool:
    return "google.colab" in sys.modules


class MeshViewer:
    """Interactive 3D mesh viewer for Jupyter notebooks.

    Renders the mesh using Three.js (via pythreejs) with orbit controls,
    customizable lighting, and display options.

    Args:
        mesh: A tiles_to_mesh.Mesh instance.
        width: Viewer width in pixels. Default 800.
        height: Viewer height in pixels. Default 600.
        wireframe: Show wireframe overlay. Default False.
        show_edges: Highlight mesh edges. Default False.
        background: Background color as hex string. Default "#1a1a2e".
        lighting: Lighting preset. One of "default", "studio", "outdoor". Default "default".
    """

    def __init__(
        self,
        mesh,
        width: int = 800,
        height: int = 600,
        wireframe: bool = False,
        show_edges: bool = False,
        background: str = "#1a1a2e",
        lighting: str = "default",
    ):
        self.mesh = mesh
        self.width = width
        self.height = height
        self.wireframe = wireframe
        self.show_edges = show_edges
        self.background = background
        self.lighting = lighting

    def show(self) -> Any:
        """Render the 3D viewer widget.

        Returns:
            The renderer widget (for display in notebooks).
        """
        # In Colab, pythreejs widget-sync triggers TraitErrors on light
        # shadow/target traits.  Use the pure-HTML renderer instead.
        if _is_colab():
            return self._show_html_fallback()
        try:
            return self._show_pythreejs()
        except ImportError:
            return self._show_html_fallback()

    def _show_pythreejs(self) -> Any:
        """Render using pythreejs."""
        import pythreejs as p3
        from IPython.display import display
        import ipywidgets as widgets

        mesh = self.mesh
        vertices = mesh.vertices
        faces = mesh.faces

        # Compute mesh center and scale for camera positioning
        bounds = mesh.bounds
        center = bounds["center"]
        size = bounds["size"]
        max_dim = float(np.max(size))
        camera_distance = max_dim * 2.0

        # ── Build Three.js geometry ──
        # Flatten for BufferGeometry
        face_vertices = vertices[faces.flatten()]
        position_attr = p3.BufferAttribute(
            array=face_vertices.astype(np.float32),
            normalized=False,
        )

        geometry_attrs = {"position": position_attr}

        # Normals
        if mesh.has_normals:
            face_normals = mesh.normals[faces.flatten()]
            geometry_attrs["normal"] = p3.BufferAttribute(
                array=face_normals.astype(np.float32),
                normalized=False,
            )

        # UVs
        if mesh.has_texcoords:
            face_uvs = mesh.texcoords[faces.flatten()]
            geometry_attrs["uv"] = p3.BufferAttribute(
                array=face_uvs.astype(np.float32),
                normalized=False,
            )

        geometry = p3.BufferGeometry(attributes=geometry_attrs)

        # ── Material ──
        if mesh.has_textures and mesh.textures:
            # Load the first texture
            tex_data = mesh.textures[0]
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(tex_data["data"]))
            # Convert to data texture
            img_array = np.array(img)
            texture = p3.DataTexture(
                data=img_array,
                format="RGBFormat" if img_array.shape[2] == 3 else "RGBAFormat",
                type="UnsignedByteType",
            )
            material = p3.MeshStandardMaterial(
                map=texture,
                wireframe=self.wireframe,
                side="DoubleSide",
            )
        else:
            material = p3.MeshStandardMaterial(
                color="#8fa8c8",
                wireframe=self.wireframe,
                side="DoubleSide",
                metalness=0.1,
                roughness=0.7,
            )

        mesh_obj = p3.Mesh(geometry=geometry, material=material)

        # ── Wireframe overlay ──
        children = [mesh_obj]
        if self.show_edges:
            edge_material = p3.MeshBasicMaterial(
                color="#ffffff",
                wireframe=True,
                transparent=True,
                opacity=0.15,
            )
            edge_mesh = p3.Mesh(geometry=geometry, material=edge_material)
            children.append(edge_mesh)

        # ── Lighting ──
        lights = self._create_lights(center, max_dim)
        children.extend(lights)

        # ── Scene ──
        scene = p3.Scene(
            children=children,
            background=self.background,
        )

        # ── Camera ──
        camera = p3.PerspectiveCamera(
            position=[
                float(center[0]) + camera_distance * 0.5,
                float(center[1]) + camera_distance * 0.7,
                float(center[2]) + camera_distance * 0.5,
            ],
            fov=50,
            near=max_dim * 0.01,
            far=max_dim * 100,
        )

        # ── Controls ──
        controls = [
            p3.OrbitControls(
                controlling=camera,
                target=[float(center[0]), float(center[1]), float(center[2])],
            )
        ]

        # ── Renderer ──
        renderer = p3.Renderer(
            camera=camera,
            scene=scene,
            controls=controls,
            width=self.width,
            height=self.height,
            antialias=True,
        )

        # ── Control panel ──
        wireframe_toggle = widgets.ToggleButton(
            value=self.wireframe,
            description="Wireframe",
            icon="cube",
        )

        def on_wireframe_change(change):
            material.wireframe = change["new"]

        wireframe_toggle.observe(on_wireframe_change, names="value")

        reset_btn = widgets.Button(description="Reset View", icon="refresh")

        def on_reset(_):
            camera.position = [
                float(center[0]) + camera_distance * 0.5,
                float(center[1]) + camera_distance * 0.7,
                float(center[2]) + camera_distance * 0.5,
            ]

        reset_btn.on_click(on_reset)

        info_label = widgets.HTML(
            value=(
                f"<b>Vertices:</b> {mesh.vertex_count:,} | "
                f"<b>Faces:</b> {mesh.face_count:,} | "
                f"<b>Textured:</b> {'Yes' if mesh.has_textures else 'No'}"
            )
        )

        controls_panel = widgets.HBox([wireframe_toggle, reset_btn, info_label])
        container = widgets.VBox([renderer, controls_panel])

        display(container)
        return container

    def _create_lights(self, center: np.ndarray, scale: float) -> list:
        """Create lighting setup based on preset.

        Uses PointLight instead of DirectionalLight to avoid a TraitError
        in newer pythreejs versions where DirectionalLight.target expects
        an Object3D instance rather than a model reference string.
        """
        import pythreejs as p3

        lights = []

        if self.lighting == "studio":
            lights.append(p3.AmbientLight(color="#404040", intensity=0.6))
            lights.append(p3.PointLight(
                color="#ffffff",
                position=[scale * 2, scale * 3, scale * 2],
                intensity=1.0,
            ))
            lights.append(p3.PointLight(
                color="#8888ff",
                position=[-scale * 2, scale, -scale * 2],
                intensity=0.4,
            ))
        elif self.lighting == "outdoor":
            lights.append(p3.AmbientLight(color="#87CEEB", intensity=0.5))
            lights.append(p3.PointLight(
                color="#FFF5E1",
                position=[scale, scale * 5, scale * 2],
                intensity=1.2,
            ))
            lights.append(p3.HemisphereLight(
                skyColor="#87CEEB",
                groundColor="#362D1B",
                intensity=0.4,
            ))
        else:  # default
            lights.append(p3.AmbientLight(color="#404040", intensity=0.5))
            lights.append(p3.PointLight(
                color="#ffffff",
                position=[scale * 2, scale * 3, scale],
                intensity=0.8,
            ))
            lights.append(p3.PointLight(
                color="#ffffff",
                position=[-scale, scale * 2, -scale],
                intensity=0.3,
            ))

        return lights

    def _show_html_fallback(self) -> Any:
        """Viewer using raw Three.js via HTML.

        Works everywhere (Colab, Jupyter, JupyterLab) because it doesn't
        rely on the pythreejs widget-sync layer.  Uses an ES-module import
        of Three.js from a CDN so there are no global-scope collisions.
        """
        from IPython.display import display, HTML
        import json
        import base64
        import uuid

        mesh = self.mesh
        bounds = mesh.bounds
        center = bounds["center"].tolist()
        max_dim = float(np.max(bounds["size"]))

        # Serialize mesh data for JavaScript
        vertices_list = mesh.vertices.flatten().tolist()
        faces_list = mesh.faces.flatten().tolist()

        normals_js = "null"
        if mesh.has_normals:
            normals_js = json.dumps(mesh.normals.flatten().tolist())

        viewer_id = f"ttm-viewer-{uuid.uuid4().hex[:8]}"

        html = f"""
        <div id="{viewer_id}" style="width:{self.width}px;height:{self.height}px;position:relative;border:1px solid #333;border-radius:4px;overflow:hidden;"></div>
        <script type="module">
        import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
        import {{ OrbitControls }} from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';

        (function() {{
            const container = document.getElementById('{viewer_id}');
            if (!container) return;

            const scene = new THREE.Scene();
            scene.background = new THREE.Color('{self.background}');

            const W = {self.width}, H = {self.height};
            const camera = new THREE.PerspectiveCamera(50, W / H, 0.1, {max_dim * 100});
            camera.position.set(
                {center[0] + max_dim * 0.5},
                {center[1] + max_dim * 0.7},
                {center[2] + max_dim * 0.5}
            );

            const renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(W, H);
            renderer.setPixelRatio(window.devicePixelRatio || 1);
            container.appendChild(renderer.domElement);

            const controls = new OrbitControls(camera, renderer.domElement);
            controls.target.set({center[0]}, {center[1]}, {center[2]});
            controls.enableDamping = true;
            controls.dampingFactor = 0.12;
            controls.update();

            // ── Geometry ──────────────────────────────────────────────
            const geometry = new THREE.BufferGeometry();
            const vertices = new Float32Array({json.dumps(vertices_list)});
            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

            const indices = new Uint32Array({json.dumps(faces_list)});
            geometry.setIndex(new THREE.BufferAttribute(indices, 1));

            const normData = {normals_js};
            if (normData) {{
                geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(normData), 3));
            }} else {{
                geometry.computeVertexNormals();
            }}

            const material = new THREE.MeshStandardMaterial({{
                color: 0x8fa8c8,
                wireframe: {'true' if self.wireframe else 'false'},
                side: THREE.DoubleSide,
                metalness: 0.1,
                roughness: 0.7
            }});

            const mesh = new THREE.Mesh(geometry, material);
            scene.add(mesh);

            // ── Lighting ──────────────────────────────────────────────
            scene.add(new THREE.AmbientLight(0x404040, 0.6));
            const d1 = new THREE.DirectionalLight(0xffffff, 0.9);
            d1.position.set({max_dim * 2}, {max_dim * 3}, {max_dim});
            scene.add(d1);
            const d2 = new THREE.DirectionalLight(0xffffff, 0.3);
            d2.position.set(-{max_dim}, {max_dim * 2}, -{max_dim});
            scene.add(d2);

            // ── Info bar ──────────────────────────────────────────────
            const info = document.createElement('div');
            info.style.cssText = 'position:absolute;bottom:0;left:0;right:0;padding:4px 8px;'
                + 'background:rgba(0,0,0,0.6);color:#eee;font:12px monospace;';
            info.textContent = 'Vertices: {mesh.vertex_count:,} | Faces: {mesh.face_count:,}'
                + ' | Drag to rotate, scroll to zoom';
            container.appendChild(info);

            // ── Render loop ───────────────────────────────────────────
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        }})();
        </script>
        """

        display(HTML(html))
