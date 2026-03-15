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
    """Colab-specific map selector using google.colab.output for JS↔Python communication.

    This is used internally when running in Colab, where ipywidgets
    communication may not work reliably.
    """

    def __init__(self, api_key: str, center: Tuple[float, float] = (40.748817, -73.985428), zoom: int = 15):
        self._api_key = api_key
        self._center = center
        self._zoom = zoom
        self._polygon_coords: List[Tuple[float, float]] = []

    def show(self) -> None:
        """Display the map selector in Colab."""
        from IPython.display import display, HTML
        from google.colab import output

        # Register the callback for receiving coordinates
        output.register_callback("ttm_set_coords", self._receive_coords)

        html = f"""
        <div id="ttm-colab-map" style="width: 100%; height: 600px;"></div>
        <div id="ttm-colab-status" style="padding: 10px; font-family: monospace;"></div>
        <script src="https://maps.googleapis.com/maps/api/js?key={self._api_key}&libraries=drawing"></script>
        <script>
        (function() {{
            var map = new google.maps.Map(document.getElementById('ttm-colab-map'), {{
                center: {{ lat: {self._center[0]}, lng: {self._center[1]} }},
                zoom: {self._zoom},
                mapTypeId: 'satellite',
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
                    editable: true,
                }},
            }});
            drawingManager.setMap(map);

            google.maps.event.addListener(drawingManager, 'polygoncomplete', function(polygon) {{
                var path = polygon.getPath();
                var coords = [];
                for (var i = 0; i < path.getLength(); i++) {{
                    var pt = path.getAt(i);
                    coords.push([pt.lat(), pt.lng()]);
                }}
                document.getElementById('ttm-colab-status').innerText =
                    'Selected ' + coords.length + ' points. Sending to Python...';

                google.colab.kernel.invokeFunction('ttm_set_coords', [JSON.stringify(coords)], {{}});
            }});
        }})();
        </script>
        """
        display(HTML(html))

    def _receive_coords(self, coords_json: str) -> None:
        """Callback to receive coordinates from JavaScript."""
        coords = json.loads(coords_json)
        self._polygon_coords = [(c[0], c[1]) for c in coords]
        print(f"✓ Received {len(self._polygon_coords)} polygon points")

    @property
    def polygon(self) -> List[Tuple[float, float]]:
        """The selected polygon coordinates."""
        return self._polygon_coords
