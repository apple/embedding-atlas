"""Tests for GIS column detection and geoparquet geometry extraction."""

import struct

import pandas as pd
import pytest

from embedding_atlas.cli import (
    extract_coordinates_from_geometry,
    find_geometry_column,
    find_gis_columns,
)


# ---------------------------------------------------------------------------
# find_gis_columns (named column pairs)
# ---------------------------------------------------------------------------


class TestFindGisColumns:
    def test_lon_lat(self):
        assert find_gis_columns(["id", "lon", "lat", "name"]) == ("lon", "lat")

    def test_longitude_latitude(self):
        assert find_gis_columns(["longitude", "latitude", "name"]) == (
            "longitude",
            "latitude",
        )

    def test_lng_lat(self):
        assert find_gis_columns(["lng", "lat"]) == ("lng", "lat")

    def test_case_insensitive(self):
        assert find_gis_columns(["LON", "LAT", "name"]) == ("LON", "LAT")

    def test_x_y(self):
        assert find_gis_columns(["x", "y", "z"]) == ("x", "y")

    def test_no_match(self):
        assert find_gis_columns(["name", "value", "count"]) == (None, None)

    def test_priority_longitude_over_x(self):
        """longitude/latitude should be preferred over x/y."""
        result = find_gis_columns(["x", "y", "longitude", "latitude"])
        assert result == ("longitude", "latitude")


# ---------------------------------------------------------------------------
# find_geometry_column
# ---------------------------------------------------------------------------


def _make_wkb_point(lon: float, lat: float) -> bytes:
    """Create a minimal WKB Point (little-endian)."""
    return struct.pack("<bIdd", 1, 1, lon, lat)


class TestFindGeometryColumn:
    def test_detects_geometry_column_by_name(self):
        df = pd.DataFrame(
            {"geometry": [_make_wkb_point(2.35, 48.86)], "name": ["Paris"]}
        )
        assert find_geometry_column(df) == "geometry"

    def test_detects_geom_column(self):
        df = pd.DataFrame({"geom": [_make_wkb_point(2.35, 48.86)], "name": ["Paris"]})
        assert find_geometry_column(df) == "geom"

    def test_ignores_non_binary_geometry(self):
        df = pd.DataFrame({"geometry": ["not-binary"], "name": ["test"]})
        assert find_geometry_column(df) is None

    def test_no_geometry_column(self):
        df = pd.DataFrame({"lon": [2.35], "lat": [48.86]})
        assert find_geometry_column(df) is None


# ---------------------------------------------------------------------------
# extract_coordinates_from_geometry
# ---------------------------------------------------------------------------


class TestExtractCoordinates:
    def test_extracts_lon_lat_from_wkb_points(self):
        df = pd.DataFrame(
            {
                "geometry": [
                    _make_wkb_point(2.35, 48.86),
                    _make_wkb_point(13.38, 52.52),
                    _make_wkb_point(12.50, 41.90),
                ],
                "name": ["Paris", "Berlin", "Rome"],
            }
        )
        result, x_col, y_col = extract_coordinates_from_geometry(df, "geometry")
        assert x_col == "lon"
        assert y_col == "lat"
        assert result[x_col].tolist() == pytest.approx([2.35, 13.38, 12.50])
        assert result[y_col].tolist() == pytest.approx([48.86, 52.52, 41.90])

    def test_handles_existing_lon_lat_columns(self):
        """If lon/lat already exist, use lon_1/lat_1."""
        df = pd.DataFrame(
            {
                "geometry": [_make_wkb_point(2.35, 48.86)],
                "lon": [0.0],
                "lat": [0.0],
            }
        )
        result, x_col, y_col = extract_coordinates_from_geometry(df, "geometry")
        assert x_col == "lon_1"
        assert y_col == "lat_1"
        assert result[x_col].iloc[0] == pytest.approx(2.35)

    def test_drops_invalid_geometries(self):
        """Non-point geometries should result in dropped rows."""
        df = pd.DataFrame(
            {
                "geometry": [
                    _make_wkb_point(2.35, 48.86),
                    b"\x01\x03\x00\x00\x00" + b"\x00" * 50,  # fake polygon
                    _make_wkb_point(13.38, 52.52),
                ],
                "name": ["Paris", "Polygon", "Berlin"],
            }
        )
        result, x_col, y_col = extract_coordinates_from_geometry(df, "geometry")
        assert len(result) == 2
        assert result["name"].tolist() == ["Paris", "Berlin"]

    def test_big_endian_wkb(self):
        """WKB can be big-endian (byte order = 0)."""
        wkb = struct.pack(">bIdd", 0, 1, 2.35, 48.86)
        df = pd.DataFrame({"geometry": [wkb], "name": ["Paris"]})
        result, x_col, y_col = extract_coordinates_from_geometry(df, "geometry")
        assert result[x_col].iloc[0] == pytest.approx(2.35)
        assert result[y_col].iloc[0] == pytest.approx(48.86)
