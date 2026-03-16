"""
Google Colab compatibility layer.

Provides Colab-specific adaptations for widget rendering and
communication between JavaScript and Python in the Colab environment.
"""

from __future__ import annotations

import html
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

_IS_COLAB = False

try:
    import google.colab
    _IS_COLAB = True
except ImportError:
    pass


def is_colab() -> bool:
    """Check if running in Google Colab."""
    return _IS_COLAB


def colab_output_eval_js(js_code: str) -> Any:
    """Evaluate JavaScript in Colab's output frame."""
    if not _IS_COLAB:
        raise RuntimeError("Not running in Google Colab")
    from google.colab.output import eval_js
    return eval_js(js_code)


def colab_register_callback(name: str, callback: Callable) -> None:
    """Register a Python callback that can be invoked from JavaScript in Colab."""
    if not _IS_COLAB:
        return
    from google.colab import output
    output.register_callback(name, callback)


class ColabMapSelector:
    """Colab-specific map selector that renders Google Maps inside an <iframe>.

    Colab's Content Security Policy blocks external ``<script src="...">``
    tags in output cells.  To work around this the map is rendered inside an
    ``<iframe>`` whose ``srcdoc`` contains the full Google Maps page.  The
    iframe has its own browsing context so the Maps JS API loads without
    CSP issues.

    Communication flow:
        iframe  →  ``parent.postMessage(coords)``  →  outer listener script
                →  ``google.colab.kernel.invokeFunction``  →  Python callback
    """

    def __init__(
        self,
        api_key: str,
        center: Tuple[float, float] = (40.748817, -73.985428),
        zoom: int = 15,
        height: int = 600,
        map_type: str = "satellite",
    ):
        self._api_key = api_key
        self._center = center
        self._zoom = zoom
        self._height = height
        self._map_type = map_type
        self._polygon_coords: List[Tuple[float, float]] = []
        self._region = None

    # ── public API ────────────────────────────────────────────────────

    def show(self) -> None:
        """Display the interactive map selector in Colab."""
        from IPython.display import display, HTML
        from google.colab import output as colab_output

        # Register Python callback so the outer JS listener can forward coords
        colab_output.register_callback("ttm_set_coords", self._receive_coords)

        display(HTML(self._build_html()))

    @property
    def region(self) -> Optional[Any]:
        """The selected Region, or None if no polygon has been drawn yet."""
        return self._region

    def set_region_programmatic(
        self, coords: List[Tuple[float, float]], name: Optional[str] = None
    ) -> None:
        """Set the region without using the map (for scripting / testing)."""
        from tiles_to_mesh.selector import Region

        self._region = Region.from_coords(coords, name=name)
        print(
            f"✓ Region set programmatically: "
            f"{len(self._region.polygon)} points, "
            f"~{self._region.area_approx_km2:.3f} km²"
        )

    # ── internals ─────────────────────────────────────────────────────

    def _receive_coords(self, coords_json: str) -> None:
        """Callback invoked from JavaScript with the drawn polygon coordinates."""
        from tiles_to_mesh.selector import Region

        coords = json.loads(coords_json)
        self._polygon_coords = [(c[0], c[1]) for c in coords]
        self._region = Region.from_coords(self._polygon_coords)
        print(
            f"✓ Region selected: {len(self._polygon_coords)} points, "
            f"~{self._region.area_approx_km2:.3f} km²"
        )

    # ── HTML builders ─────────────────────────────────────────────────

    def _build_iframe_srcdoc(self) -> str:
        """Build the *inner* HTML page that lives inside the iframe.

        This is a self-contained page that loads the Google Maps JS API,
        shows a drawing manager, and posts coordinates to the parent frame
        via ``postMessage``.
        """
        # NOTE: This string must NOT use Python f-string braces for JS objects.
        # We inject Python values via str.replace() at the end to keep it readable.
        page = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin:0; padding:0; height:100%; overflow:hidden; }
  #map { width:100%; height:100%; }
</style>
</head>
<body>
<div id="map"></div>
<script>
function _ttmInit() {
  var map = new google.maps.Map(document.getElementById('map'), {
    center: { lat: __TTM_LAT__, lng: __TTM_LNG__ },
    zoom: __TTM_ZOOM__,
    mapTypeId: '__TTM_MAPTYPE__',
    mapTypeControl: true,
    streetViewControl: false,
    fullscreenControl: false,
  });

  var dm = new google.maps.drawing.DrawingManager({
    drawingMode: google.maps.drawing.OverlayType.POLYGON,
    drawingControl: true,
    drawingControlOptions: {
      position: google.maps.ControlPosition.TOP_CENTER,
      drawingModes: ['polygon'],
    },
    polygonOptions: {
      fillColor: '#2196F3',
      fillOpacity: 0.25,
      strokeWeight: 2,
      strokeColor: '#1565C0',
      clickable: true,
      editable: true,
      draggable: true,
    },
  });
  dm.setMap(map);

  var curPoly = null;

  function send(polygon) {
    var path = polygon.getPath();
    var coords = [];
    for (var i = 0; i < path.getLength(); i++) {
      var p = path.getAt(i);
      coords.push([p.lat(), p.lng()]);
    }
    // Send to the parent (Colab output cell) via postMessage
    parent.postMessage({ type: 'ttm-polygon', coords: coords }, '*');
  }

  google.maps.event.addListener(dm, 'polygoncomplete', function(polygon) {
    if (curPoly) curPoly.setMap(null);
    curPoly = polygon;
    dm.setDrawingMode(null);
    send(polygon);
    google.maps.event.addListener(polygon.getPath(), 'set_at',    function() { send(polygon); });
    google.maps.event.addListener(polygon.getPath(), 'insert_at', function() { send(polygon); });
  });
}
</script>
<script src="https://maps.googleapis.com/maps/api/js?key=__TTM_KEY__&libraries=drawing&callback=_ttmInit"
        async defer
        onerror="document.body.innerHTML='<p style=color:red>Failed to load Google Maps API. Check your API key.</p>'">
</script>
</body>
</html>"""
        page = (
            page
            .replace("__TTM_LAT__", str(self._center[0]))
            .replace("__TTM_LNG__", str(self._center[1]))
            .replace("__TTM_ZOOM__", str(self._zoom))
            .replace("__TTM_MAPTYPE__", self._map_type)
            .replace("__TTM_KEY__", self._api_key)
        )
        return page

    def _build_html(self) -> str:
        """Build the outer HTML rendered by ``display(HTML(...))``.

        Contains:
        * a status bar
        * an ``<iframe>`` whose ``srcdoc`` is the self-contained Maps page
        * a ``<script>`` that listens for ``postMessage`` from the iframe and
          forwards the coordinates to Python via
          ``google.colab.kernel.invokeFunction``.
        """
        srcdoc = self._build_iframe_srcdoc()
        # json.dumps produces a valid JS string literal (handles all escaping)
        srcdoc_js = json.dumps(srcdoc)

        return f"""
<div id="ttm-colab-status"
     style="padding:8px 12px; font-family:monospace; font-size:13px;
            background:#f0f4ff; border-radius:6px; margin-bottom:6px;">
  ⏳ Loading Google Maps…
</div>
<div id="ttm-colab-container"></div>

<script>
(function() {{
  // ── Create the iframe with srcdoc ──────────────────────────────────
  var iframe = document.createElement('iframe');
  iframe.style.cssText = 'width:100%;height:{self._height}px;border:none;border-radius:6px;';
  iframe.setAttribute('sandbox', 'allow-scripts allow-same-origin allow-popups');
  iframe.srcdoc = {srcdoc_js};
  document.getElementById('ttm-colab-container').appendChild(iframe);

  // ── Listen for polygon coordinates from the iframe ─────────────────
  window.addEventListener('message', function(e) {{
    if (!e.data || e.data.type !== 'ttm-polygon') return;

    var n = e.data.coords.length;
    var statusEl = document.getElementById('ttm-colab-status');
    statusEl.innerHTML =
      '<b style="color:#1565C0;">✓ Selected ' + n + ' points.</b> '
      + 'Edit the polygon or re-draw. Coordinates sent to Python.';

    // Forward to Python via Colab kernel
    google.colab.kernel.invokeFunction(
      'ttm_set_coords', [JSON.stringify(e.data.coords)], {{}}
    );
  }});

  // ── Update status when iframe loads ────────────────────────────────
  iframe.addEventListener('load', function() {{
    var statusEl = document.getElementById('ttm-colab-status');
    if (statusEl.textContent.indexOf('Loading') !== -1) {{
      statusEl.innerHTML = '<b>Draw a polygon</b> on the map to select a region.';
    }}
  }});
}})();
</script>
"""
