"""Tests for the Region and MapSelector classes."""

import pytest

from tiles_to_mesh.selector import Region, MapSelector


class TestRegion:
    """Tests for the Region dataclass."""

    def test_from_coords(self):
        coords = [(40.0, -74.0), (40.1, -74.0), (40.1, -73.9), (40.0, -73.9)]
        region = Region.from_coords(coords, name="test")
        assert region.name == "test"
        assert len(region.polygon) == 4
        assert region.polygon[0] == (40.0, -74.0)

    def test_from_coords_requires_3_points(self):
        with pytest.raises(ValueError, match="at least 3"):
            Region.from_coords([(0, 0), (1, 1)])

    def test_from_bounds(self):
        region = Region.from_bounds(40.0, -74.0, 40.1, -73.9)
        assert len(region.polygon) == 4
        south, west, north, east = region.bounds
        assert south == pytest.approx(40.0)
        assert west == pytest.approx(-74.0)
        assert north == pytest.approx(40.1)
        assert east == pytest.approx(-73.9)

    def test_center(self):
        region = Region.from_bounds(40.0, -74.0, 40.1, -73.9)
        lat, lng = region.center
        assert lat == pytest.approx(40.05)
        assert lng == pytest.approx(-73.95)

    def test_bounds(self):
        coords = [(40.0, -74.0), (40.2, -73.8), (40.1, -74.1)]
        region = Region.from_coords(coords)
        south, west, north, east = region.bounds
        assert south == pytest.approx(40.0)
        assert north == pytest.approx(40.2)
        assert west == pytest.approx(-74.1)
        assert east == pytest.approx(-73.8)

    def test_area_approx(self):
        # ~1km x 1km rectangle near equator
        region = Region.from_bounds(0.0, 0.0, 0.009, 0.009)
        area = region.area_approx_km2
        assert 0.5 < area < 1.5  # Rough check

    def test_repr(self):
        region = Region.from_bounds(40.0, -74.0, 40.1, -73.9, name="NYC")
        r = repr(region)
        assert "NYC" in r
        assert "4 points" in r


class TestMapSelector:
    """Tests for the MapSelector widget."""

    def test_creation(self):
        selector = MapSelector(api_key="test_key")
        assert selector._api_key == "test_key"
        assert selector.region is None

    def test_set_region_programmatic(self):
        selector = MapSelector(api_key="test_key")
        coords = [(40.0, -74.0), (40.1, -74.0), (40.1, -73.9), (40.0, -73.9)]
        selector.set_region_programmatic(coords, name="Test Area")
        assert selector.region is not None
        assert selector.region.name == "Test Area"
        assert len(selector.region.polygon) == 4

    def test_custom_center_and_zoom(self):
        selector = MapSelector(
            api_key="test_key",
            center=(51.5074, -0.1278),
            zoom=12,
            map_type="hybrid",
        )
        assert selector._center == (51.5074, -0.1278)
        assert selector._zoom == 12
        assert selector._map_type == "hybrid"
