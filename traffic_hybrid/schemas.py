"""
schemas.py — Universal data contract for Pan-India traffic congestion prediction.

Replaces the hardcoded 4-junction integer ID system with a location-agnostic
road segment schema. Any coordinate in India can produce a valid observation
without requiring a pre-existing historical CSV entry.

Key design decision — Congestion Index (CI):
    CI = 1.0 - clamp(current_speed / free_flow_speed, 0, 1)

    CI == 0.0  →  free-flowing (current speed equals free-flow speed)
    CI == 1.0  →  fully gridlocked (zero current speed)

This normalization makes the target dimensionless and comparable across all
road types (motorway in Delhi vs residential lane in Kochi), which is the
prerequisite for cross-city generalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

# ---------------------------------------------------------------------------
# Road classification constants
# ---------------------------------------------------------------------------

# Ordered from highest capacity to lowest — used for feature encoding and
# fallback speed-limit / lane-count heuristics when OSM data is unavailable.
ROAD_CLASSES: list[str] = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "unclassified",
    "unknown",
]

# Default free-flow speed limits (km/h) per road class for Indian roads.
# Derived from MoRTH (Ministry of Road Transport & Highways) guidelines and
# typical TomTom free-flow observations. Used as fallback when live API data
# is unavailable.
DEFAULT_FREE_FLOW_SPEED_KMPH: dict[str, float] = {
    "motorway": 100.0,
    "trunk": 80.0,
    "primary": 60.0,
    "secondary": 50.0,
    "tertiary": 40.0,
    "residential": 30.0,
    "unclassified": 30.0,
    "unknown": 40.0,
}

# Default lane counts per road class (one direction) for India.
DEFAULT_LANE_COUNT: dict[str, int] = {
    "motorway": 3,
    "trunk": 2,
    "primary": 2,
    "secondary": 2,
    "tertiary": 1,
    "residential": 1,
    "unclassified": 1,
    "unknown": 1,
}


# ---------------------------------------------------------------------------
# Indian city corridor archetypes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CityCorridorAnchor:
    """A named monitoring anchor point for telemetry collection."""
    city: str
    corridor_name: str
    lat: float
    lng: float
    road_class: str = "primary"
    description: str = ""


# Curated representative corridors across Indian city archetypes.
# These are used as default seeds for the telemetry harvester (M3).
INDIA_CORRIDOR_ANCHORS: list[CityCorridorAnchor] = [
    # Bengaluru — IT / Tech Corridors
    CityCorridorAnchor("bengaluru", "Silk Board Junction", 12.9177, 77.6232, "primary", "Chronic bottleneck on ORR"),
    CityCorridorAnchor("bengaluru", "Marathahalli Bridge", 12.9592, 77.6974, "primary", "ORR eastern peak hub"),
    CityCorridorAnchor("bengaluru", "Whitefield Road", 12.9698, 77.7499, "primary", "IT park access road"),
    CityCorridorAnchor("bengaluru", "Hebbal Flyover", 13.0358, 77.5970, "trunk", "North Bengaluru airport arterial"),
    # Mumbai — Coastal / Dense Arterials
    CityCorridorAnchor("mumbai", "Western Express Highway Andheri", 19.1136, 72.8697, "trunk", "High-speed 8-lane arterial"),
    CityCorridorAnchor("mumbai", "Eastern Freeway Sewri", 18.9922, 72.8547, "motorway", "Mumbai's only elevated freeway"),
    CityCorridorAnchor("mumbai", "BKC Bandra Kurla Complex", 19.0596, 72.8656, "primary", "Financial district hub"),
    CityCorridorAnchor("mumbai", "Sion Circle", 19.0378, 72.8624, "primary", "Central island-suburb distributor"),
    # Delhi-NCR — Wide Ring and Radial Roads
    CityCorridorAnchor("delhi", "Ring Road AIIMS", 28.5665, 77.2100, "trunk", "Chronic inbound congestion point"),
    CityCorridorAnchor("delhi", "DND Flyway Noida Entry", 28.5491, 77.3133, "motorway", "Peak-hour bottleneck"),
    CityCorridorAnchor("delhi", "Cyber City Gurgaon", 28.4950, 77.0890, "primary", "Corporate campus access"),
    CityCorridorAnchor("delhi", "ITO Junction", 28.6294, 77.2410, "primary", "Central institutional transit hub"),
    # Pune — Fast-growing Tier-2 IT
    CityCorridorAnchor("pune", "Hinjawadi Phase 1", 18.5912, 73.7389, "primary", "IT SEZ entry junction"),
    CityCorridorAnchor("pune", "FC Road Shivajinagar", 18.5269, 73.8427, "secondary", "Dense commercial arterial"),
    CityCorridorAnchor("pune", "Magarpatta City Hadapsar", 18.5146, 73.9312, "primary", "Eastern IT corridor"),
    # Hyderabad — Deccan Tech Hub
    CityCorridorAnchor("hyderabad", "Hitec City Cyber Towers", 17.4504, 78.3808, "primary", "Premier tech park bottleneck"),
    CityCorridorAnchor("hyderabad", "Gachibowli Junction", 17.4401, 78.3489, "primary", "Financial district and ORR connector"),
    CityCorridorAnchor("hyderabad", "Punjagutta Crossing", 17.4265, 78.4526, "primary", "Central commercial crossing"),
    # Chennai — Coastal Industrial & Auto/IT
    CityCorridorAnchor("chennai", "OMR Tidel Park", 12.9877, 80.2520, "primary", "IT expressway gateway"),
    CityCorridorAnchor("chennai", "Kathipara Junction Guindy", 13.0067, 80.2038, "trunk", "Major arterial cloverleaf"),
    CityCorridorAnchor("chennai", "Anna Nagar Roundtana", 13.0850, 80.2100, "primary", "Dense commercial hub"),
    # Kolkata — Eastern Metropolitan Hub
    CityCorridorAnchor("kolkata", "EM Bypass Science City", 22.5408, 88.3965, "trunk", "Eastern arterial expressway"),
    CityCorridorAnchor("kolkata", "Park Circus 7-Point Crossing", 22.5413, 88.3687, "primary", "Central 7-way crossroads"),
    CityCorridorAnchor("kolkata", "Sector V Salt Lake", 22.5744, 88.4338, "primary", "East India tech hub"),
    # Ahmedabad — Western Industrial Corridor
    CityCorridorAnchor("ahmedabad", "SG Highway ISKCON Cross Road", 23.0284, 72.5068, "trunk", "Sarkhej-Gandhinagar arterial"),
    CityCorridorAnchor("ahmedabad", "Ashram Road Income Tax Circle", 23.0396, 72.5714, "primary", "CBD corridor"),
    # Jaipur — Northern Tourism & Transit
    CityCorridorAnchor("jaipur", "JLN Marg Jawahar Circle", 26.8524, 75.8054, "primary", "Airport-city boulevard"),
    CityCorridorAnchor("jaipur", "Ajmer Road 200 Feet Bypass", 26.8920, 75.7360, "trunk", "Western freight bottleneck"),
    # Kochi — Southern Port & Coastal Arterial
    CityCorridorAnchor("kochi", "Edappally Toll Junction", 10.0261, 76.3125, "trunk", "NH 66 & NH 544 intersection"),
    # Lucknow — Central Hindi Heartland Hub
    CityCorridorAnchor("lucknow", "Hazratganj Crossing", 26.8500, 80.9490, "primary", "State capital commercial center"),
]


# ---------------------------------------------------------------------------
# Universal road segment observation schema
# ---------------------------------------------------------------------------

@dataclass
class RoadSegmentObservation:
    """
    A single observation of traffic state at a road segment, anywhere in India.

    This schema is the universal input contract for the Pan-India model.
    It is location-agnostic — no junction ID required.

    Fields
    ------
    lat, lng : float
        WGS-84 coordinates of the observed road point.
    h3_index : str
        Uber H3 hexagonal cell index at resolution 8 (~0.7 km² cells).
        Used for spatial neighborhood grouping and API-call caching.
        Empty string when H3 library is not installed (graceful degradation).
    road_class : str
        OpenStreetMap functional road class. One of ROAD_CLASSES.
    lanes : int
        Number of lanes in the observed direction (OSM or fallback heuristic).
    speed_limit_kmh : float
        Posted speed limit in km/h (OSM or MoRTH default).
    timestamp : datetime
        Observation timestamp (IST — no timezone info required; treat as local).
    hour : int
        Hour of day [0, 23].
    day_of_week : int
        Day of week [0=Monday, 6=Sunday].
    is_weekend : bool
        True if Saturday or Sunday.
    is_indian_holiday : bool
        True if the date is a gazetted national or state-level Indian holiday.
    rain_mm_per_hr : float
        Precipitation rate in mm/hr at the time of observation.
        0.0 when weather data is unavailable.
    current_speed_kmh : float
        Live speed observed at this road segment (km/h).
    free_flow_speed_kmh : float
        Speed under uncongested conditions (km/h).
    congestion_index : float
        Normalized congestion score in [0.0, 1.0].
        Computed from current_speed and free_flow_speed.
        Value of 0.0 means free-flowing; 1.0 means gridlocked.
    city : str
        City name (e.g. "bengaluru", "mumbai"). Optional but useful for
        grouping in evaluation and visualization.
    source : str
        Data source identifier (e.g. "tomtom", "mappls", "synthetic").
    """

    # --- Spatial identity ---
    lat: float
    lng: float
    h3_index: str = ""
    road_class: str = "unknown"
    lanes: int = 1
    speed_limit_kmh: float = 40.0

    # --- Temporal signals ---
    timestamp: datetime = field(default_factory=datetime.now)
    hour: int = 0
    day_of_week: int = 0
    is_weekend: bool = False
    is_indian_holiday: bool = False

    # --- Environmental context ---
    rain_mm_per_hr: float = 0.0

    # --- Traffic state ---
    current_speed_kmh: float = 0.0
    free_flow_speed_kmh: float = 40.0
    congestion_index: float = 0.0

    # --- Metadata ---
    city: str = "unknown"
    source: str = "unknown"

    # Class-level constants exposed for downstream modules
    ROAD_CLASSES: ClassVar[list[str]] = ROAD_CLASSES
    DEFAULT_FREE_FLOW_SPEED_KMPH: ClassVar[dict[str, float]] = DEFAULT_FREE_FLOW_SPEED_KMPH

    def __post_init__(self) -> None:
        """Auto-derive fields that can be computed from others."""
        # Sync temporal fields from timestamp if they are still at defaults
        if self.timestamp and self.hour == 0 and self.day_of_week == 0:
            self.hour = self.timestamp.hour
            self.day_of_week = self.timestamp.weekday()
            self.is_weekend = self.day_of_week >= 5

        # Compute congestion_index from speeds if not explicitly set
        if self.congestion_index == 0.0 and self.free_flow_speed_kmh > 0:
            self.congestion_index = compute_congestion_index(
                self.current_speed_kmh, self.free_flow_speed_kmh
            )

        # Normalise road_class to a known value
        if self.road_class not in ROAD_CLASSES:
            self.road_class = "unknown"

        # Apply road-class defaults if lanes/speed_limit were not provided
        if self.lanes <= 0:
            self.lanes = DEFAULT_LANE_COUNT.get(self.road_class, 1)
        if self.speed_limit_kmh <= 0:
            self.speed_limit_kmh = DEFAULT_FREE_FLOW_SPEED_KMPH.get(self.road_class, 40.0)

    def to_feature_dict(self) -> dict[str, float]:
        """
        Return a flat dict of numeric features ready for model inference.
        Road class is one-hot encoded over ROAD_CLASSES.
        This is the canonical feature vector for XGBoost and BiLSTM tabular input.
        """
        one_hot = {
            f"road_class_{cls}": float(self.road_class == cls)
            for cls in ROAD_CLASSES
        }
        return {
            "lat": self.lat,
            "lng": self.lng,
            "lanes": float(self.lanes),
            "speed_limit_kmh": self.speed_limit_kmh,
            "hour_sin": math.sin(2.0 * math.pi * self.hour / 24.0),
            "hour_cos": math.cos(2.0 * math.pi * self.hour / 24.0),
            "dow_sin": math.sin(2.0 * math.pi * self.day_of_week / 7.0),
            "dow_cos": math.cos(2.0 * math.pi * self.day_of_week / 7.0),
            "is_weekend": float(self.is_weekend),
            "is_indian_holiday": float(self.is_indian_holiday),
            "rain_mm_per_hr": self.rain_mm_per_hr,
            "current_speed_kmh": self.current_speed_kmh,
            "free_flow_speed_kmh": self.free_flow_speed_kmh,
            "congestion_index": self.congestion_index,
            **one_hot,
        }

    def __repr__(self) -> str:
        return (
            f"RoadSegmentObservation("
            f"city={self.city!r}, "
            f"road_class={self.road_class!r}, "
            f"lat={self.lat:.4f}, lng={self.lng:.4f}, "
            f"CI={self.congestion_index:.3f}, "
            f"speed={self.current_speed_kmh:.1f}/{self.free_flow_speed_kmh:.1f} km/h, "
            f"rain={self.rain_mm_per_hr:.1f} mm/hr)"
        )


# ---------------------------------------------------------------------------
# Core metric: Congestion Index
# ---------------------------------------------------------------------------

def compute_congestion_index(
    current_speed_kmh: float,
    free_flow_speed_kmh: float,
) -> float:
    """
    Compute the normalized Congestion Index (CI) for a road segment.

    CI = 1.0 - clamp(current_speed / free_flow_speed, 0.0, 1.0)

    This is the universal, dimensionless target variable for the Pan-India
    model. It is comparable across all road types and cities.

    Parameters
    ----------
    current_speed_kmh : float
        Observed speed at the road segment in km/h.
    free_flow_speed_kmh : float
        Speed under free-flow (uncongested) conditions in km/h.

    Returns
    -------
    float
        Congestion Index in [0.0, 1.0].
        0.0 = perfectly free-flowing.
        1.0 = fully gridlocked.
    """
    if free_flow_speed_kmh <= 0.0:
        return 0.0
    ratio = current_speed_kmh / free_flow_speed_kmh
    return round(1.0 - max(0.0, min(1.0, ratio)), 6)


def classify_congestion_index(ci: float) -> int:
    """
    Map a Congestion Index value to a 3-tier congestion class.

    Class 0 (Low)    : CI < 0.33  — free-flowing
    Class 1 (Medium) : CI < 0.66  — moderate slowdown
    Class 2 (High)   : CI >= 0.66 — heavy congestion / gridlock

    These thresholds align with the 3-tier output classification used by
    the BiLSTM + XGBoost ensemble in the existing codebase.
    """
    if ci < 0.33:
        return 0
    if ci < 0.66:
        return 1
    return 2
