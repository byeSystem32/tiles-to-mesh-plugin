"""
Interactive Google Maps polygon selector for Jupyter/Colab notebooks.

Embeds a Google Maps view with drawing tools so users can select a region
by drawing a polygon. The coordinates are captured back into Python.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import ipywidgets as widgets
from IPython.display import display, HTML


@dataclass
class Region:
    """A geographic region defined by a polygon of (lat, lng) coordinates.

    Attributes:
        polygon: List of (latitude, longitude) tuples defining the region boundary.
        name: Optional human-readable name for this region.
    """

    polygon: List[Tuple[float, float]]
    name: Optional[str] = None

    @classmethod
    def from_coords(cls, coords: List[Tuple[float, float]], name: Optional[str] = None) -> "Region":
        """Create a Region from a list of (lat, lng) coordinate tuples.

        Args:
            coords: List of (latitude, longitude) tuples. Must have at least 3 points.
            name: Optional name for the region.

        Returns:
            A Region instance.

        Raises:
            ValueError: If fewer than 3 coordinates are provided.
        """
        if len(coords) < 3:
            raise ValueError("A region must have at least 3 coordinate points.")
        return cls(polygon=coords, name=name)

    @classmethod
    def from_bounds(cls, south: float, west: float, north: float, east: float, name: Optional[str] = None) -> "Region":
        """Create a rectangular Region from bounding box coordinates.

        Args:
            south: Southern latitude boundary.
            west: Western longitude boundary.
            north: Northern latitude boundary.
            east: Eastern longitude boundary.
            name: Optional name for the region.

        Returns:
            A Region instance with a rectangular polygon.
        """
        return cls(
            polygon=[
                (south, west),
                (south, east),
                (north, east),
                (north, west),
            ],
            name=name,
        )

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Axis-aligned bounding box as (south, west, north, east)."""
        lats = [p[0] for p in self.polygon]
        lngs = [p[1] for p in self.polygon]
        return (min(lats), min(lngs), max(lats), max(lngs))

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the region as (lat, lng)."""
        lats = [p[0] for p in self.polygon]
        lngs = [p[1] for p in self.polygon]
        return (sum(lats) / len(lats), sum(lngs) / len(lngs))

    @property
    def area_approx_km2(self) -> float:
        """Approximate area of the polygon in square kilometers using the shoelace formula."""
        import math

        n = len(self.polygon)
        if n < 3:
            return 0.0

        # Convert to approximate meters using center latitude
        center_lat = self.center[0]
        m_per_deg_lat = 111_320.0
        m_per_deg_lng = 111_320.0 * math.cos(math.radians(center_lat))

        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            xi = self.polygon[i][1] * m_per_deg_lng
            yi = self.polygon[i][0] * m_per_deg_lat
            xj = self.polygon[j][1] * m_per_deg_lng
            yj = self.polygon[j][0] * m_per_deg_lat
            area += xi * yj - xj * yi

        return abs(area) / 2.0 / 1e6

    def __repr__(self) -> str:
        name_str = f"'{self.name}'" if self.name else "unnamed"
        return (
            f"Region({name_str}, {len(self.polygon)} points, "
            f"center=({self.center[0]:.4f}, {self.center[1]:.4f}), "
            f"~{self.area_approx_km2:.3f} km²)"
        )


class MapSelector:
    """Interactive Google Maps polygon selector widget for Jupyter/Colab.

    Embeds a Google Maps view with polygon drawing tools. Users draw a polygon
    to define their region of interest, and coordinates are captured into Python.

    Args:
        api_key: Google Maps JavaScript API key.
        center: Initial map center as (lat, lng). Defaults to (40.748817, -73.985428) (NYC).
        zoom: Initial zoom level. Defaults to 15.
        height: Widget height in pixels. Defaults to 600.
        width: Widget width as CSS string. Defaults to "100%".
        map_type: Map type ID. One of "satellite", "roadmap", "terrain", "hybrid".
    """

    def __init__(
        self,
        api_key: str,
        center: Tuple[float, float] = (40.748817, -73.985428),
        zoom: int = 15,
        height: int = 600,
        width: str = "100%",
        map_type: str = "satellite",
    ):
        self._api_key = api_key
        self._center = center
        self._zoom = zoom
        self._height = height
        self._width = width
        self._map_type = map_type
        self._region: Optional[Region] = None
        self._widget_id = f"ttm-map-{uuid.uuid4().hex[:8]}"
        self._output = widgets.Output()
        self._coords_text = widgets.Textarea(
            value="",
            placeholder="Polygon coordinates will appear here...",
            description="Coords:",
            layout=widgets.Layout(width="100%", height="80px"),
            disabled=True,
        )
        self._status = widgets.HTML(value="<b>Draw a polygon on the map to select a region.</b>")
        self._confirm_btn = widgets.Button(
            description="Confirm Selection",
            button_style="success",
            icon="check",
            disabled=True,
            layout=widgets.Layout(width="200px"),
        )
        self._clear_btn = widgets.Button(
            description="Clear",
            button_style="warning",
            icon="trash",
            layout=widgets.Layout(width="100px"),
        )

    @property
    def region(self) -> Optional[Region]:
        """The selected region, or None if no selection has been made."""
        return self._region

    def show(self) -> None:
        """Display the interactive map selector widget."""
        self._confirm_btn.on_click(self._on_confirm)
        self._clear_btn.on_click(self._on_clear)

        buttons = widgets.HBox([self._confirm_btn, self._clear_btn])
        container = widgets.VBox([
            self._status,
            self._output,
            self._coords_text,
            buttons,
        ])

        with self._output:
            display(HTML(self._build_map_html()))

        display(container)

    def _on_confirm(self, _btn) -> None:
        """Handle confirm button click."""
        coords_str = self._coords_text.value.strip()
        if coords_str:
            try:
                coords = json.loads(coords_str)
                self._region = Region(polygon=[(c["lat"], c["lng"]) for c in coords])
                self._status.value = (
                    f'<b style="color: green;">✓ Region selected: {len(self._region.polygon)} points, '
                    f"~{self._region.area_approx_km2:.3f} km²</b>"
                )
                self._confirm_btn.disabled = True
                self._confirm_btn.description = "Selection Confirmed"
            except (json.JSONDecodeError, KeyError) as e:
                self._status.value = f'<b style="color: red;">Error parsing coordinates: {e}</b>'

    def _on_clear(self, _btn) -> None:
        """Handle clear button click."""
        self._region = None
        self._coords_text.value = ""
        self._confirm_btn.disabled = True
        self._confirm_btn.description = "Confirm Selection"
        self._status.value = "<b>Draw a polygon on the map to select a region.</b>"
        # Re-render map
        self._output.clear_output()
        with self._output:
            display(HTML(self._build_map_html()))

    def _build_map_html(self) -> str:
        """Build the HTML/JS for the Google Maps widget with drawing tools."""
        widget_id = self._widget_id
        coords_widget_model = self._coords_text.model_id

        return f"""
        <div id="{widget_id}" style="width: {self._width}; height: {self._height}px;"></div>
        <script>
        (function() {{
            // Load Google Maps API if not already loaded
            if (typeof google === 'undefined' || typeof google.maps === 'undefined') {{
                var script = document.createElement('script');
                script.src = 'https://maps.googleapis.com/maps/api/js?key={self._api_key}&libraries=drawing&callback=initTTMMap_{widget_id}';
                script.async = true;
                script.defer = true;
                document.head.appendChild(script);
            }} else {{
                initTTMMap_{widget_id}();
            }}

            window.initTTMMap_{widget_id} = function() {{
                var mapDiv = document.getElementById('{widget_id}');
                if (!mapDiv) return;

                var map = new google.maps.Map(mapDiv, {{
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
                        drawingModes: [
                            google.maps.drawing.OverlayType.POLYGON,
                        ],
                    }},
                    polygonOptions: {{
                        fillColor: '#2196F3',
                        fillOpacity: 0.25,
                        strokeWeight: 2,
                        strokeColor: '#1565C0',
                        clickable: true,
                        editable: true,
                        draggable: true,
                        zIndex: 1,
                    }},
                }});
                drawingManager.setMap(map);

                var currentPolygon = null;

                function updateCoords(polygon) {{
                    var path = polygon.getPath();
                    var coords = [];
                    for (var i = 0; i < path.getLength(); i++) {{
                        var pt = path.getAt(i);
                        coords.push({{ lat: pt.lat(), lng: pt.lng() }});
                    }}

                    // Push coordinates to the ipywidgets textarea
                    var coordsJson = JSON.stringify(coords);

                    // Use Jupyter widget communication
                    var kernel = IPython.notebook ? IPython.notebook.kernel :
                                 (google.colab ? google.colab.kernel : null);

                    if (kernel) {{
                        kernel.execute(
                            "import ipywidgets; " +
                            "w = ipywidgets.Widget.widgets['" + "{coords_widget_model}" + "']; " +
                            "w.value = '" + coordsJson.replace(/'/g, "\\\\'") + "'"
                        );
                    }}

                    // Also try direct Jupyter comms approach
                    try {{
                        var cells = document.querySelectorAll('.cell');
                        var event = new CustomEvent('ttm-coords-update', {{ detail: coordsJson }});
                        document.dispatchEvent(event);
                    }} catch(e) {{}}
                }}

                google.maps.event.addListener(drawingManager, 'polygoncomplete', function(polygon) {{
                    if (currentPolygon) {{
                        currentPolygon.setMap(null);
                    }}
                    currentPolygon = polygon;
                    drawingManager.setDrawingMode(null);

                    updateCoords(polygon);

                    // Listen for edits to the polygon
                    google.maps.event.addListener(polygon.getPath(), 'set_at', function() {{
                        updateCoords(polygon);
                    }});
                    google.maps.event.addListener(polygon.getPath(), 'insert_at', function() {{
                        updateCoords(polygon);
                    }});

                    // Enable confirm button via kernel execute
                    try {{
                        var kernel = IPython.notebook ? IPython.notebook.kernel :
                                     (google.colab ? google.colab.kernel : null);
                        if (kernel) {{
                            kernel.execute(
                                "import ipywidgets; " +
                                "w = [w for w in ipywidgets.Widget.widgets.values() " +
                                "if hasattr(w, 'description') and w.description == 'Confirm Selection']; " +
                                "w[0].disabled = False if w else None"
                            );
                        }}
                    }} catch(e) {{}}
                }});
            }};
        }})();
        </script>
        """

    def set_region_programmatic(self, coords: List[Tuple[float, float]], name: Optional[str] = None) -> None:
        """Set the region programmatically without using the map widget.

        Useful for testing or when coordinates are already known.

        Args:
            coords: List of (lat, lng) tuples defining the polygon.
            name: Optional region name.
        """
        self._region = Region.from_coords(coords, name=name)
        self._status.value = (
            f'<b style="color: green;">✓ Region set programmatically: '
            f"{len(self._region.polygon)} points</b>"
        )
