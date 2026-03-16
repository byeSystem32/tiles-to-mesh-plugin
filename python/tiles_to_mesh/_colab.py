"""
Google Colab compatibility layer.

Provides Colab-specific adaptations for widget rendering and
communication between JavaScript and Python in the Colab environment.

The interactive map uses ``ipyleaflet`` (a Jupyter-native widget) instead of
the Google Maps JavaScript API, because Colab's Content Security Policy
blocks external scripts from ``maps.googleapis.com``.
"""

from __future__ import annotations

import json
from typing import Any, Callable, List, Optional, Tuple

_IS_COLAB = False

try:
    import google.colab  # noqa: F401
    _IS_COLAB = True
except ImportError:
    pass


def is_colab() -> bool:
    """Check if running in Google Colab."""
    return _IS_COLAB


class ColabMapSelector:
    """Colab-specific map selector using ``ipyleaflet``.

    Google Maps JS API cannot be loaded in Colab due to CSP restrictions.
    This selector uses ``ipyleaflet`` with satellite tiles and a polygon
    drawing control instead.  The widget protocol works natively in Colab,
    so no JS-to-Python hacks are needed.
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
        self._region = None
        self._draw_control = None

    # ── public API ────────────────────────────────────────────────────

    def show(self) -> None:
        """Display the interactive map selector in Colab."""
        try:
            self._show_ipyleaflet()
        except ImportError:
            print(
                "⚠️  ipyleaflet is not installed. Install it with:\n"
                "    !pip install ipyleaflet\n"
                "Then restart the runtime (Runtime → Restart runtime) and re-run.\n"
                "\n"
                "Alternatively, set coordinates manually:\n"
                "    selector.set_region_programmatic([\n"
                '        (40.7484, -73.9867),\n'
                '        (40.7492, -73.9867),\n'
                '        (40.7492, -73.9845),\n'
                '        (40.7484, -73.9845),\n'
                '    ], name="My Region")'
            )

    @property
    def region(self) -> Optional[Any]:
        """The selected Region, or None if no polygon has been drawn yet."""
        return self._region

    def set_region_programmatic(
        self, coords: List[Tuple[float, float]], name: Optional[str] = None
    ) -> None:
        """Set the region without the map widget."""
        from tiles_to_mesh.selector import Region

        self._region = Region.from_coords(coords, name=name)
        print(
            f"✓ Region set programmatically: "
            f"{len(self._region.polygon)} points, "
            f"~{self._region.area_approx_km2:.3f} km²"
        )

    # ── ipyleaflet implementation ─────────────────────────────────────

    def _show_ipyleaflet(self) -> None:
        """Render the map using ipyleaflet with satellite tiles."""
        import ipyleaflet as ipl
        import ipywidgets as widgets
        from IPython.display import display

        # ── Satellite tile layer ──────────────────────────────────────
        # Use ESRI World Imagery (free, no API key required for tile display)
        satellite = ipl.TileLayer(
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attribution="Tiles © Esri",
            name="Satellite",
            max_zoom=19,
        )

        # ── Map ──────────────────────────────────────────────────────
        m = ipl.Map(
            center=self._center,
            zoom=self._zoom,
            scroll_wheel_zoom=True,
            layout=widgets.Layout(width="100%", height=f"{self._height}px"),
        )
        # Replace the default basemap with satellite
        m.clear_layers()
        m.add(satellite)

        # ── Draw control (polygon only) ──────────────────────────────
        draw_control = ipl.DrawControl(
            polygon={
                "shapeOptions": {
                    "color": "#2196F3",
                    "fillColor": "#2196F3",
                    "fillOpacity": 0.20,
                    "weight": 2,
                },
            },
            # Disable other shape types
            polyline={},
            circle={},
            circlemarker={},
            rectangle={
                "shapeOptions": {
                    "color": "#2196F3",
                    "fillColor": "#2196F3",
                    "fillOpacity": 0.20,
                    "weight": 2,
                },
            },
            marker={},
        )
        draw_control.on_draw(self._on_draw)
        m.add(draw_control)
        self._draw_control = draw_control

        # ── Status label ─────────────────────────────────────────────
        status = widgets.HTML(
            value=(
                '<b>Draw a polygon or rectangle</b> on the map to select a region. '
                'Use the toolbar on the left.'
            )
        )
        self._status_widget = status

        container = widgets.VBox([status, m])
        display(container)

    def _on_draw(self, control: Any, action: str, geo_json: dict) -> None:
        """Handle draw events from ipyleaflet DrawControl."""
        from tiles_to_mesh.selector import Region

        if action != "created":
            return

        geom = geo_json.get("geometry", {})
        geom_type = geom.get("type", "")

        if geom_type == "Polygon":
            # GeoJSON coordinates are [lng, lat], we need (lat, lng)
            ring = geom["coordinates"][0]
            coords = [(lat, lng) for lng, lat in ring]
            # GeoJSON polygons repeat the first point at the end; drop it
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]
        else:
            return

        if len(coords) < 3:
            return

        self._region = Region.from_coords(coords)
        self._status_widget.value = (
            f'<b style="color:#1565C0;">✓ Region selected:</b> '
            f'{len(coords)} points, '
            f'~{self._region.area_approx_km2:.3f} km²'
        )
        print(
            f"✓ Region selected: {len(coords)} points, "
            f"~{self._region.area_approx_km2:.3f} km²"
        )
