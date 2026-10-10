"""
context.py — Indian contextual signals for pan-India traffic prediction.

Provides two categories of non-spatial, non-traffic context features that
are critical for accurate traffic forecasting across Indian cities:

1. **Indian Calendar Context**: National and state-specific gazetted holidays,
   long weekends, and major festival periods (Diwali, Dussehra, Eid, etc.)
   that significantly alter commuter patterns.

2. **Weather Context**: Live precipitation intensity and visibility data
   fetched from OpenWeatherMap (free tier). Monsoon rain is the single
   largest external variance driver for Indian urban traffic.

Both modules degrade gracefully — they return safe zero-filled defaults
if the required libraries or API keys are not available.

Usage
-----
    from traffic_hybrid.context import IndianCalendar, WeatherClient

    cal = IndianCalendar(state_code="KA")  # Karnataka
    print(cal.is_holiday(date(2026, 11, 1)))  # Kannada Rajyotsava → True

    wx = WeatherClient(api_key="YOUR_OWM_KEY")
    obs = wx.get_current(lat=12.9177, lng=77.6232)
    print(obs.rain_mm_per_hr)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: `holidays` library for Indian gazetted holidays
# ---------------------------------------------------------------------------

try:
    import holidays as _holidays_lib  # type: ignore[import]
    _HOLIDAYS_AVAILABLE = True
except ImportError:
    _holidays_lib = None
    _HOLIDAYS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Indian Calendar Context
# ---------------------------------------------------------------------------

# State codes supported by the `holidays` library for India
# Reference: https://python-holidays.readthedocs.io/en/latest/countries/india.html
INDIA_STATE_CODES: dict[str, str] = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "DL": "Delhi",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PY": "Puducherry",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "WB": "West Bengal",
}

# City → state code mapping for the telemetry corridor cities
CITY_TO_STATE: dict[str, str] = {
    "bengaluru": "KA",
    "bangalore": "KA",
    "mumbai": "MH",
    "pune": "MH",
    "delhi": "DL",
    "ncr": "DL",
    "hyderabad": "TS",
    "chennai": "TN",
    "kolkata": "WB",
    "ahmedabad": "GJ",
    "jaipur": "RJ",
    "lucknow": "UP",
    "kochi": "KL",
}


class IndianCalendar:
    """
    Calendar utility for checking Indian national and state-specific holidays.

    Wraps the ``holidays`` library. If the library is not installed, all
    ``is_holiday()`` calls return False (safe degradation).

    Parameters
    ----------
    state_code : str, optional
        Two-letter Indian state code (e.g. 'KA' for Karnataka, 'MH' for
        Maharashtra). When provided, state-specific optional holidays are
        included alongside national holidays.
    """

    def __init__(self, state_code: str | None = None) -> None:
        self.state_code = (state_code or "").upper()
        self._cache: dict[int, Any] = {}  # year → holidays dict

    def _get_year_holidays(self, year: int) -> Any:
        if year in self._cache:
            return self._cache[year]
        if not _HOLIDAYS_AVAILABLE:
            self._cache[year] = {}
            return {}
        try:
            kwargs: dict[str, Any] = {"country": "IN", "years": year}
            if self.state_code and self.state_code in INDIA_STATE_CODES:
                kwargs["subdiv"] = self.state_code
            hols = _holidays_lib.country_holidays(**kwargs)
            self._cache[year] = hols
            return hols
        except Exception as exc:
            logger.debug("Holiday lookup failed for year %d: %s", year, exc)
            self._cache[year] = {}
            return {}

    def is_holiday(self, query_date: date | datetime) -> bool:
        """
        Return True if ``query_date`` is a gazetted Indian holiday.

        Includes national holidays (Republic Day, Independence Day, Gandhi
        Jayanti) and, if a state_code was provided, state-specific holidays.
        """
        d = query_date.date() if isinstance(query_date, datetime) else query_date
        hols = self._get_year_holidays(d.year)
        return d in hols

    def is_long_weekend(self, query_date: date | datetime) -> bool:
        """
        Return True if the date is part of a 3+ day break (weekend + holiday
        combination), which significantly affects travel demand.
        """
        d = query_date.date() if isinstance(query_date, datetime) else query_date
        for delta in range(-2, 3):
            check = d + timedelta(days=delta)
            if self.is_holiday(check) and check.weekday() in (0, 4):
                return True
        return False

    @classmethod
    def for_city(cls, city: str) -> "IndianCalendar":
        """Create a calendar configured for a known Indian city name."""
        code = CITY_TO_STATE.get(city.lower().strip())
        return cls(state_code=code)


# ---------------------------------------------------------------------------
# Weather Context
# ---------------------------------------------------------------------------

@dataclass
class WeatherObservation:
    """Snapshot of weather conditions at a location."""
    lat: float
    lng: float
    rain_mm_per_hr: float = 0.0      # Precipitation intensity (mm/h)
    visibility_m: float = 10_000.0   # Visibility in metres (10km = clear)
    temperature_c: float = 25.0
    humidity_pct: float = 60.0
    description: str = "unknown"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "unavailable"      # 'openweathermap', 'heuristic', 'unavailable'

    @property
    def is_heavy_rain(self) -> bool:
        """True if precipitation exceeds 7.5 mm/hr (IMD 'heavy rain' threshold)."""
        return self.rain_mm_per_hr >= 7.5

    @property
    def is_moderate_rain(self) -> bool:
        """True if precipitation is between 2.5 and 7.5 mm/hr."""
        return 2.5 <= self.rain_mm_per_hr < 7.5

    @property
    def is_low_visibility(self) -> bool:
        """True if visibility is below 1000m (fog / dense smog conditions)."""
        return self.visibility_m < 1000.0


class WeatherClient:
    """
    Lightweight OpenWeatherMap client for fetching live precipitation and
    visibility data at a given coordinate.

    Requires a free OWM API key set via the ``OWM_API_KEY`` environment
    variable or passed directly to the constructor.

    In-process TTL cache (default 10 minutes) prevents redundant API calls
    for nearby coordinates during live inference.

    Parameters
    ----------
    api_key : str, optional
        OpenWeatherMap API key. Defaults to ``OWM_API_KEY`` env var.
    cache_ttl_seconds : int
        Time-to-live for cached weather responses (default 600 = 10 min).
    """

    _OWM_URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(
        self,
        api_key: str | None = None,
        cache_ttl_seconds: int = 600,
    ) -> None:
        self.api_key = (api_key or os.getenv("OWM_API_KEY", "")).strip()
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[str, dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _cache_key(self, lat: float, lng: float) -> str:
        # Round to 2 decimal places (~1.1 km) for cache grouping
        return f"{lat:.2f},{lng:.2f}"

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        entry = self._cache.get(key)
        if not entry:
            return None
        age = (datetime.now(timezone.utc) - entry["ts"]).total_seconds()
        if age > self.cache_ttl_seconds:
            del self._cache[key]
            return None
        return entry["data"]

    def _cache_set(self, key: str, data: dict[str, Any]) -> None:
        self._cache[key] = {"ts": datetime.now(timezone.utc), "data": data}

    def get_current(self, lat: float, lng: float) -> WeatherObservation:
        """
        Fetch current weather conditions at (lat, lng).

        Returns a ``WeatherObservation`` with safe defaults on any failure.
        """
        default = WeatherObservation(lat=lat, lng=lng, source="unavailable")

        if not self.enabled:
            logger.debug("OWM_API_KEY not set — returning default weather observation.")
            return default

        cache_key = self._cache_key(lat, lng)
        cached = self._cache_get(cache_key)
        if cached:
            return WeatherObservation(**cached)

        query = urllib_parse.urlencode({
            "lat": round(lat, 4),
            "lon": round(lng, 4),
            "appid": self.api_key,
            "units": "metric",
        })
        url = f"{self._OWM_URL}?{query}"

        try:
            with urllib_request.urlopen(url, timeout=10) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib_error.URLError, urllib_error.HTTPError) as exc:
            logger.warning("OWM request failed: %s", exc)
            return default

        # --- Parse OWM response ---
        rain = raw.get("rain", {})
        rain_1h = float(rain.get("1h", rain.get("3h", 0.0)))

        main = raw.get("main", {})
        temp_c = float(main.get("temp", 25.0))
        humidity = float(main.get("humidity", 60.0))
        visibility_m = float(raw.get("visibility", 10_000))

        weather_list = raw.get("weather", [{}])
        description = weather_list[0].get("description", "unknown") if weather_list else "unknown"

        obs_dict = {
            "lat": lat,
            "lng": lng,
            "rain_mm_per_hr": rain_1h,
            "visibility_m": visibility_m,
            "temperature_c": temp_c,
            "humidity_pct": humidity,
            "description": description,
            "fetched_at": datetime.now(timezone.utc),
            "source": "openweathermap",
        }
        self._cache_set(cache_key, obs_dict)
        return WeatherObservation(**obs_dict)
