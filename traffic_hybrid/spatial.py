"""
spatial.py — Dynamic road attribute extraction for Pan-India generalization.

Given any (lat, lng) coordinate in India, this module resolves the underlying
road attributes (functional class, lane count, speed limit, one-way status)
using OpenStreetMap data via the `osmnx` library.

All OSM queries are cached by bounding box to avoid redundant network calls.
The module degrades gracefully if `osmnx` is not installed — it falls back to
road-class heuristics without raising errors, so the rest of the pipeline
continues to function.

Usage
-----
    from traffic_hybrid.spatial import get_road_attributes, latlon_to_h3

    attrs = get_road_attributes(12.9177, 77.6232)
    # RoadAttributes(road_class='primary', lanes=2, speed_limit_kmh=60.0, ...)

    h3_idx = latlon_to_h3(12.9177, 77.6232, resolution=8)
    # '88283082edfffff'
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from traffic_hybrid.schemas import (
    DEFAULT_FREE_FLOW_SPEED_KMPH,
    DEFAULT_LANE_COUNT,
    ROAD_CLASSES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import osmnx as ox  # type: ignore[import]
    _OSMNX_AVAILABLE = True
except ImportError:  # pragma: no cover
    ox = None
    _OSMNX_AVAILABLE = False

try:
    import h3  # type: ignore[import]
    _H3_AVAILABLE = True
except ImportError:  # pragma: no cover
    h3 = None
    _H3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Road attribute data structure
# ---------------------------------------------------------------------------

@dataclass
class RoadAttributes:
    """
    Resolved road geometry and classification for a given coordinate.

    All fields have safe defaults so downstream code never sees None.
    """
    road_class: str = "unknown"
    lanes: int = 1
    speed_limit_kmh: float = 40.0
    is_one_way: bool = False
    osm_highway_tag: str = "unknown"   # Raw OSM highway= tag value
    source: str = "heuristic"          # 'osm', 'heuristic'

    def __post_init__(self) -> None:
        # Normalise road_class to the canonical list
        if self.road_class not in ROAD_CLASSES:
            self.road_class = _map_osm_tag_to_road_class(self.osm_highway_tag)
        # Apply defaults if not resolved
        if self.lanes <= 0:
            self.lanes = DEFAULT_LANE_COUNT.get(self.road_class, 1)
        if self.speed_limit_kmh <= 0:
            self.speed_limit_kmh = DEFAULT_FREE_FLOW_SPEED_KMPH.get(self.road_class, 40.0)


# ---------------------------------------------------------------------------
# OSM highway tag → canonical road class mapping
# ---------------------------------------------------------------------------

_OSM_TAG_TO_CLASS: dict[str, str] = {
    "motorway": "motorway",
    "motorway_link": "motorway",
    "trunk": "trunk",
    "trunk_link": "trunk",
    "primary": "primary",
    "primary_link": "primary",
    "secondary": "secondary",
    "secondary_link": "secondary",
    "tertiary": "tertiary",
    "tertiary_link": "tertiary",
    "residential": "residential",
    "living_street": "residential",
    "service": "residential",
    "unclassified": "unclassified",
    "road": "unclassified",
}


def _map_osm_tag_to_road_class(osm_tag: str) -> str:
    """Map a raw OSM ``highway=`` tag to the canonical ROAD_CLASSES list."""
    return _OSM_TAG_TO_CLASS.get(osm_tag.lower().strip(), "unknown")


def _parse_osm_lanes(raw: Any) -> int:
    """Parse OSM lane value which may be a string, int, or list."""
    if isinstance(raw, list):
        raw = raw[0]
    try:
        return max(1, int(str(raw).strip()))
    except (ValueError, TypeError):
        return 0


def _parse_osm_speed(raw: Any) -> float:
    """
    Parse OSM ``maxspeed=`` tag.
    Examples: '60', '60 mph', 'IN:urban', None
    Indian-specific default urban speed: 50 km/h.
    """
    if raw is None:
        return 0.0
    tag = str(raw).strip().lower()
    # Indian IST zone defaults
    if tag in ("in:urban", "urban"):
        return 50.0
    if tag in ("in:rural", "rural"):
        return 70.0
    if tag in ("in:motorway", "motorway"):
        return 100.0
    # Remove units and parse numeric portion
    tag = tag.replace("km/h", "").replace("kmh", "").replace("kph", "").strip()
    if "mph" in tag:
        try:
            return round(float(tag.replace("mph", "").strip()) * 1.60934, 1)
        except ValueError:
            return 0.0
    try:
        return float(tag)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# H3 spatial indexing
# ---------------------------------------------------------------------------

def latlon_to_h3(lat: float, lng: float, resolution: int = 8) -> str:
    """
    Convert a (lat, lng) coordinate pair to an Uber H3 hexagonal cell index.

    Resolution 8 gives ~0.74 km² cells (roughly 500m diameter), ideal for
    caching and grouping road segment observations.

    Returns an empty string if the ``h3`` library is not installed.

    Parameters
    ----------
    lat, lng : float
        WGS-84 coordinates.
    resolution : int
        H3 resolution in [0, 15]. Defaults to 8.

    Returns
    -------
    str
        H3 index string (e.g. '88283082edfffff') or '' if unavailable.
    """
    if not _H3_AVAILABLE:
        return ""
    try:
        return h3.geo_to_h3(lat, lng, resolution)
    except Exception as exc:  # pragma: no cover
        logger.debug("H3 indexing failed for (%s, %s): %s", lat, lng, exc)
        return ""


def h3_neighbors(h3_index: str, k_rings: int = 1) -> list[str]:
    """
    Return the H3 neighbor cells within ``k_rings`` of the given index.
    Useful for spatial aggregation and join operations.
    Returns an empty list if H3 is unavailable.
    """
    if not _H3_AVAILABLE or not h3_index:
        return []
    try:
        return list(h3.k_ring(h3_index, k_rings))
    except Exception:  # pragma: no cover
        return []


# ---------------------------------------------------------------------------
# OSM road attribute extraction
# ---------------------------------------------------------------------------

# Internal bounding-box cache: keyed by (rounded lat, rounded lng, radius_m)
# Stores raw OSMnx GeoDataFrames of road edges to avoid repeated API calls.
_osm_edge_cache: dict[tuple[float, float, int], Any] = {}
_OSM_CACHE_RADIUS_M: int = 200


def _get_cached_osm_edges(lat: float, lng: float) -> Any | None:
    """
    Retrieve OSM edges near a point, using an internal in-process cache.
    Returns None if osmnx is unavailable or the query fails.
    """
    if not _OSMNX_AVAILABLE:
        return None

    # Round to 4 decimal places (~11m) for cache key stability
    cache_key = (round(lat, 4), round(lng, 4), _OSM_CACHE_RADIUS_M)
    if cache_key in _osm_edge_cache:
        return _osm_edge_cache[cache_key]

    try:
        ox.settings.log_console = False
        graph = ox.graph_from_point(
            (lat, lng),
            dist=_OSM_CACHE_RADIUS_M,
            network_type="drive",
            retain_all=False,
        )
        edges = ox.graph_to_gdfs(graph, nodes=False)
        _osm_edge_cache[cache_key] = edges
        return edges
    except Exception as exc:
        logger.debug("OSM query failed for (%.4f, %.4f): %s", lat, lng, exc)
        _osm_edge_cache[cache_key] = None
        return None


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Return the great-circle distance in metres between two WGS-84 points."""
    r = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lng2 - lng1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def get_road_attributes(lat: float, lng: float) -> RoadAttributes:
    """
    Resolve road geometry and classification for any coordinate in India.

    Strategy
    --------
    1. Query OpenStreetMap (via osmnx) for road edges within 200m of the point.
    2. Select the nearest edge by midpoint Haversine distance.
    3. Extract highway tag, lane count, and speed limit from the edge attributes.
    4. If osmnx is unavailable or no edges are found, return heuristic defaults.

    Parameters
    ----------
    lat, lng : float
        WGS-84 coordinates of the road point.

    Returns
    -------
    RoadAttributes
        Resolved road class, lanes, speed limit, and one-way status.
        Falls back to safe defaults on any error.
    """
    edges = _get_cached_osm_edges(lat, lng)

    if edges is None or edges.empty:
        logger.debug("No OSM edges found near (%.4f, %.4f) — using heuristics.", lat, lng)
        return RoadAttributes(source="heuristic")

    # Find the nearest edge by computing Haversine from query point to edge midpoint
    try:
        # OSM edge geometries: use midpoint of the LineString
        edges = edges.copy()
        edges["_midlat"] = edges.geometry.apply(
            lambda geom: geom.interpolate(0.5, normalized=True).y
        )
        edges["_midlng"] = edges.geometry.apply(
            lambda geom: geom.interpolate(0.5, normalized=True).x
        )
        edges["_dist_m"] = edges.apply(
            lambda row: _haversine_m(lat, lng, row["_midlat"], row["_midlng"]),
            axis=1,
        )
        nearest = edges.loc[edges["_dist_m"].idxmin()]
    except Exception as exc:
        logger.debug("Edge selection failed: %s", exc)
        return RoadAttributes(source="heuristic")

    # Parse attributes from the nearest edge
    osm_highway = nearest.get("highway", "unknown")
    if isinstance(osm_highway, list):
        osm_highway = osm_highway[0]
    osm_highway = str(osm_highway).lower().strip()

    road_class = _map_osm_tag_to_road_class(osm_highway)
    lanes_raw = nearest.get("lanes", 0)
    speed_raw = nearest.get("maxspeed", None)
    oneway_raw = nearest.get("oneway", False)

    lanes = _parse_osm_lanes(lanes_raw) or DEFAULT_LANE_COUNT.get(road_class, 1)
    speed = _parse_osm_speed(speed_raw) or DEFAULT_FREE_FLOW_SPEED_KMPH.get(road_class, 40.0)
    is_one_way = bool(oneway_raw)

    return RoadAttributes(
        road_class=road_class,
        lanes=lanes,
        speed_limit_kmh=speed,
        is_one_way=is_one_way,
        osm_highway_tag=osm_highway,
        source="osm",
    )


# ---------------------------------------------------------------------------
# Convenience: snap a lat/lng click to its H3 index + road attributes
# ---------------------------------------------------------------------------

@dataclass
class SpatialSnapshot:
    """Combined spatial resolution result for a clicked map point."""
    lat: float
    lng: float
    h3_index: str
    road: RoadAttributes


def resolve_spatial_context(lat: float, lng: float, h3_resolution: int = 8) -> SpatialSnapshot:
    """
    One-stop function: resolve H3 index and road attributes for any coordinate.

    Called by the web UI's click-handler and the telemetry harvester when
    enriching probe observations with static road metadata.

    Parameters
    ----------
    lat, lng : float
        WGS-84 coordinates.
    h3_resolution : int
        H3 resolution for spatial grouping (default 8 = ~0.7 km² cells).

    Returns
    -------
    SpatialSnapshot
        Combined H3 index and resolved road attributes.
    """
    h3_idx = latlon_to_h3(lat, lng, h3_resolution)
    road = get_road_attributes(lat, lng)
    return SpatialSnapshot(lat=lat, lng=lng, h3_index=h3_idx, road=road)
