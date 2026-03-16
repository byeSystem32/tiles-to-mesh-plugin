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
        rely on the pythreejs widget-sync layer.

        Three.js is loaded via dynamically-created ``<script>`` elements
        (classic, not ES-module) because Colab's output-cell iframe does
        not support ``<script type="module">`` imports.  Mesh data is
        passed as base64-encoded binary for compactness.
        """
        from IPython.display import display, HTML
        import base64
        import uuid

        mesh = self.mesh
        bounds = mesh.bounds
        center = bounds["center"].tolist()
        max_dim = float(np.max(bounds["size"]))
        if max_dim == 0:
            max_dim = 1.0

        # Encode mesh data as base64 binary (much smaller than JSON)
        verts_b64 = base64.b64encode(
            mesh.vertices.astype(np.float32).tobytes()
        ).decode("ascii")
        faces_b64 = base64.b64encode(
            mesh.faces.astype(np.uint32).tobytes()
        ).decode("ascii")
        norms_b64 = ""
        if mesh.has_normals and mesh.normals is not None:
            norms_b64 = base64.b64encode(
                mesh.normals.astype(np.float32).tobytes()
            ).decode("ascii")

        uid = uuid.uuid4().hex[:8]

        html = f"""
<div id="ttm-{uid}"
     style="width:{self.width}px;height:{self.height}px;position:relative;
            border:1px solid #333;border-radius:4px;overflow:hidden;
            background:{self.background};">
  <div id="ttm-msg-{uid}"
       style="color:#aaa;font:14px sans-serif;padding:20px;">
    Loading 3D viewer…</div>
</div>
<script>
(function() {{
  /* ── helpers ─────────────────────────────────────────────── */
  var uid  = '{uid}';
  var box  = document.getElementById('ttm-' + uid);
  var msg  = document.getElementById('ttm-msg-' + uid);

  function b64buf(b64) {{
    var s = atob(b64), n = s.length, u = new Uint8Array(n);
    for (var i = 0; i < n; i++) u[i] = s.charCodeAt(i);
    return u.buffer;
  }}

  /* ── dynamic script loader ───────────────────────────────── */
  function loadJS(url, cb) {{
    var s = document.createElement('script');
    s.src = url;
    s.onload  = cb;
    s.onerror = function() {{
      msg.textContent = 'Failed to load ' + url;
    }};
    (document.head || document.documentElement).appendChild(s);
  }}

  var THREE_CDN    = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
  var CONTROLS_CDN = 'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js';

  /* If THREE is already on the page (multiple calls), skip loading */
  if (typeof THREE !== 'undefined' && THREE.OrbitControls) {{
    init();
  }} else if (typeof THREE !== 'undefined') {{
    loadJS(CONTROLS_CDN, init);
  }} else {{
    loadJS(THREE_CDN, function() {{
      loadJS(CONTROLS_CDN, init);
    }});
  }}

  /* ── main viewer ─────────────────────────────────────────── */
  function init() {{
    if (msg) msg.remove();

    var verts = new Float32Array(b64buf('{verts_b64}'));
    var faces = new Uint32Array(b64buf('{faces_b64}'));
    var nb64  = '{norms_b64}';
    var norms = nb64 ? new Float32Array(b64buf(nb64)) : null;

    var W = {self.width}, H = {self.height};

    var scene = new THREE.Scene();
    scene.background = new THREE.Color('{self.background}');

    var camera = new THREE.PerspectiveCamera(50, W / H,
        {max_dim * 0.001}, {max_dim * 200});
    camera.position.set(
        {center[0] + max_dim * 0.5},
        {center[1] + max_dim * 0.7},
        {center[2] + max_dim * 0.5});

    var renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(W, H);
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    box.appendChild(renderer.domElement);

    var controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set({center[0]}, {center[1]}, {center[2]});
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.update();

    /* geometry */
    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position',
        new THREE.BufferAttribute(verts, 3));
    geo.setIndex(
        new THREE.BufferAttribute(faces, 1));
    if (norms) {{
      geo.setAttribute('normal',
          new THREE.BufferAttribute(norms, 3));
    }} else {{
      geo.computeVertexNormals();
    }}

    var mat = new THREE.MeshStandardMaterial({{
      color: 0x8fa8c8,
      wireframe: {'true' if self.wireframe else 'false'},
      side: THREE.DoubleSide,
      metalness: 0.1,
      roughness: 0.7
    }});
    scene.add(new THREE.Mesh(geo, mat));

    /* lighting */
    scene.add(new THREE.AmbientLight(0x404040, 0.6));
    var d1 = new THREE.DirectionalLight(0xffffff, 0.9);
    d1.position.set({max_dim * 2}, {max_dim * 3}, {max_dim});
    scene.add(d1);
    var d2 = new THREE.DirectionalLight(0xffffff, 0.3);
    d2.position.set(-{max_dim}, {max_dim * 2}, -{max_dim});
    scene.add(d2);

    /* info bar */
    var info = document.createElement('div');
    info.style.cssText =
        'position:absolute;bottom:0;left:0;right:0;padding:4px 8px;'
      + 'background:rgba(0,0,0,0.6);color:#eee;font:12px monospace;';
    info.textContent =
        'Vertices: {mesh.vertex_count:,} | Faces: {mesh.face_count:,}'
      + ' | Drag to rotate, scroll to zoom';
    box.appendChild(info);

    /* render loop */
    (function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }})();
  }}
}})();
</script>
"""

        display(HTML(html))
