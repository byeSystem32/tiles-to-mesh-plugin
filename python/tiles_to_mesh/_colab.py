"""
Google Colab compatibility layer.

Provides Colab-specific adaptations for widget rendering and
communication between JavaScript and Python in the Colab environment.
"""

from __future__ import annotations

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
    """Evaluate JavaScript in Colab's output frame.

    Args:
        js_code: JavaScript code to execute.

    Returns:
        Result from the JavaScript evaluation.
    """
    if not _IS_COLAB:
        raise RuntimeError("Not running in Google Colab")

    from google.colab.output import eval_js
    return eval_js(js_code)


def colab_register_callback(name: str, callback: Callable) -> None:
    """Register a Python callback that can be invoked from JavaScript in Colab.

    Args:
        name: Callback name (used in JS as google.colab.kernel.invokeFunction).
        callback: Python function to call.
    """
    if not _IS_COLAB:
        return

    from google.colab import output
    output.register_callback(name, callback)


class ColabMapSelector:
    """Colab-specific map selector using google.colab.output for JS-Python communication.

    This is used internally by MapSelector when running in Colab, where
    ipywidgets communication does not work reliably.

    The map is rendered in a self-contained <iframe> with srcdoc so that the
    Google Maps JavaScript API loads in a clean browsing context (no CSP
    conflicts with Colab's own scripts).  When the user finishes drawing a
    polygon the coordinates are sent back to Python via
    ``google.colab.kernel.invokeFunction``.
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
        self._region = None  # Optional[Region] — resolved at runtime via tiles_to_mesh.selector

    # ── public API ────────────────────────────────────────────────────

    def show(self) -> None:
        """Display the interactive map selector in Colab."""
        from IPython.display import display, HTML
        from google.colab import output as colab_output

        # Register Python callback so JS can send coordinates back
        colab_output.register_callback("ttm_set_coords", self._receive_coords)

        display(HTML(self._build_html()))

    @property
    def region(self) -> Optional[Any]:
        """The selected Region, or None if the user hasn't drawn a polygon yet."""
        return self._region

    def set_region_programmatic(
        self, coords: List[Tuple[float, float]], name: Optional[str] = None
    ) -> None:
        """Set the region programmatically (no map interaction needed)."""
        from tiles_to_mesh.selector import Region

        self._region = Region.from_coords(coords, name=name)
        print(
            f"✓ Region set programmatically: "
            f"{len(self._region.polygon)} points, "
            f"~{self._region.area_approx_km2:.3f} km²"
        )

    # ── internals ─────────────────────────────────────────────────────

    def _receive_coords(self, coords_json: str) -> None:
        """Callback invoked from JavaScript with the polygon coordinates."""
        from tiles_to_mesh.selector import Region

        coords = json.loads(coords_json)
        self._polygon_coords = [(c[0], c[1]) for c in coords]
        self._region = Region.from_coords(self._polygon_coords)
        print(
            f"✓ Region selected: {len(self._polygon_coords)} points, "
            f"~{self._region.area_approx_km2:.3f} km²"
        )

    def _build_html(self) -> str:
        """Return the full HTML string for the map widget.

        The Google Maps JS API is loaded via a ``callback`` parameter so that
        map initialisation only runs once the library is ready.  The drawing
        manager lets the user draw exactly one polygon; re-drawing replaces the
        previous one.  On completion the coordinates are sent to Python through
        ``google.colab.kernel.invokeFunction``.
        """
        return f"""
<div id="ttm-colab-status"
     style="padding:8px 12px; font-family:monospace; font-size:13px;
            background:#f0f4ff; border-radius:6px; margin-bottom:6px;">
  ⏳ Loading Google Maps…
</div>
<div id="ttm-colab-map" style="width:100%; height:{self._height}px; border-radius:6px; overflow:hidden;"></div>

<script>
// ── helpers ──────────────────────────────────────────────────────────
function _ttmInitMap() {{
  var statusEl = document.getElementById('ttm-colab-status');
  statusEl.innerHTML = '<b>Draw a polygon</b> on the map to select a region.';

  var map = new google.maps.Map(document.getElementById('ttm-colab-map'), {{
    center: {{ lat: {self._center[0]}, lng: {self._center[1]} }},
    zoom: {self._zoom},
    mapTypeId: '{self._map_type}',
    mapTypeControl: true,
    streetViewControl: false,
    fullscreenControl: true,
  }});

  var drawingManager = new google.maps.drawing.DrawingManager({{
    drawingMode: google.maps.drawing.OverlayType.POLYGON,
    drawingControl: true,
    drawingControlOptions: {{
      position: google.maps.ControlPosition.TOP_CENTER,
      drawingModes: ['polygon'],
    }},
    polygonOptions: {{
      fillColor: '#2196F3',
      fillOpacity: 0.25,
      strokeWeight: 2,
      strokeColor: '#1565C0',
      clickable: true,
      editable: true,
      draggable: true,
    }},
  }});
  drawingManager.setMap(map);

  var currentPolygon = null;

  function sendCoords(polygon) {{
    var path = polygon.getPath();
    var coords = [];
    for (var i = 0; i < path.getLength(); i++) {{
      var pt = path.getAt(i);
      coords.push([pt.lat(), pt.lng()]);
    }}
    statusEl.innerHTML =
      '<b style="color:#1565C0;">✓ Selected ' + coords.length + ' points.</b> '
      + 'Edit the polygon or re-draw. Coordinates have been sent to Python.';
    google.colab.kernel.invokeFunction('ttm_set_coords', [JSON.stringify(coords)], {{}});
  }}

  google.maps.event.addListener(drawingManager, 'polygoncomplete', function(polygon) {{
    if (currentPolygon) currentPolygon.setMap(null);
    currentPolygon = polygon;
    drawingManager.setDrawingMode(null);
    sendCoords(polygon);

    google.maps.event.addListener(polygon.getPath(), 'set_at', function() {{ sendCoords(polygon); }});
    google.maps.event.addListener(polygon.getPath(), 'insert_at', function() {{ sendCoords(polygon); }});
  }});
}}

// ── load Google Maps API (with callback) ─────────────────────────────
(function() {{
  if (typeof google !== 'undefined' && google.maps) {{
    _ttmInitMap();
  }} else {{
    var s = document.createElement('script');
    s.src = 'https://maps.googleapis.com/maps/api/js?key={self._api_key}&libraries=drawing&callback=_ttmInitMap';
    s.async = true;
    s.defer = true;
    s.onerror = function() {{
      document.getElementById('ttm-colab-status').innerHTML =
        '<b style="color:red;">Failed to load Google Maps API.</b> '
        + 'Check your API key and that the Maps JavaScript API is enabled.';
    }};
    document.head.appendChild(s);
  }}
}})();
</script>
"""
