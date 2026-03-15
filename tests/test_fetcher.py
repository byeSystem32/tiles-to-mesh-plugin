"""Tests for the tile fetcher module."""

import pytest

from tiles_to_mesh.fetcher import _compute_aabb, _collect_tile_urls


class TestFetcherHelpers:
    """Tests for fetcher helper functions."""

    def test_compute_aabb(self):
        polygon = [(40.0, -74.0), (40.1, -73.8), (40.05, -74.1)]
        min_lat, max_lat, min_lng, max_lng = _compute_aabb(polygon)
        assert min_lat == pytest.approx(40.0)
        assert max_lat == pytest.approx(40.1)
        assert min_lng == pytest.approx(-74.1)
        assert max_lng == pytest.approx(-73.8)

    def test_collect_tile_urls_empty_node(self):
        node = {}
        urls = _collect_tile_urls(node, (40.0, 40.1, -74.0, -73.9), 30.0, "fake_key")
        assert urls == []

    def test_collect_tile_urls_leaf_with_content(self):
        node = {
            "geometricError": 10.0,
            "content": {"uri": "tiles/test.glb"},
            "children": [],
        }
        urls = _collect_tile_urls(node, (40.0, 40.1, -74.0, -73.9), 30.0, "fake_key")
        assert len(urls) == 1
        assert "fake_key" in urls[0]

    def test_collect_tile_urls_recurses_into_children(self):
        node = {
            "geometricError": 100.0,
            "children": [
                {
                    "geometricError": 10.0,
                    "content": {"uri": "tiles/child1.glb"},
                    "children": [],
                },
                {
                    "geometricError": 10.0,
                    "content": {"uri": "tiles/child2.glb"},
                    "children": [],
                },
            ],
        }
        urls = _collect_tile_urls(node, (40.0, 40.1, -74.0, -73.9), 30.0, "fake_key")
        assert len(urls) == 2
