"""
harvest_telemetry.py — Multi-city probe data collector for Pan-India training.

Queries TomTom Traffic Flow API at pre-configured Indian corridor anchor points
on a repeating schedule and stores the enriched observations as partitioned
Parquet files under data/telemetry/.

Each record conforms to the universal RoadSegmentObservation schema defined in
traffic_hybrid/schemas.py and includes:
  - Live speed & free-flow speed (TomTom)
  - Congestion Index (derived)
  - Weather context (OpenWeatherMap, optional)
  - Indian holiday flag (holidays library, optional)
  - Static road metadata from config/india_corridors.json

Usage
-----
    # Run once (collect a single sample for all corridors):
    python tools/harvest_telemetry.py --once

    # Run on a recurring 15-minute schedule:
    python tools/harvest_telemetry.py --interval 15

    # Override output directory:
    python tools/harvest_telemetry.py --once --output data/telemetry

Environment Variables
---------------------
    TOMTOM_API_KEY  : Required. TomTom Developer API key.
    OWM_API_KEY     : Optional. OpenWeatherMap key for weather enrichment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

# Ensure the project root is on sys.path so traffic_hybrid imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from traffic_hybrid.context import IndianCalendar, WeatherClient
from traffic_hybrid.schemas import compute_congestion_index

_LOGS_DIR = _PROJECT_ROOT / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOGS_DIR / "harvest_telemetry.log"

_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("harvest_telemetry")
logger.setLevel(logging.INFO)
logger.handlers.clear()

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_formatter)
logger.addHandler(_console_handler)

_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_formatter)
logger.addHandler(_file_handler)


# ---------------------------------------------------------------------------
# TomTom Flow API helper
# ---------------------------------------------------------------------------

_TOMTOM_FLOW_URL = (
    "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{zoom}/json"
)


def _fetch_tomtom_flow(
    lat: float,
    lng: float,
    api_key: str,
    zoom: int = 12,
    timeout: int = 15,
) -> dict[str, Any] | None:
    """
    Query TomTom Flow Segment Data API for a single (lat, lng) point.

    Returns the parsed 'flowSegmentData' dict on success, or None on error.
    """
    query = urllib_parse.urlencode({
        "key": api_key,
        "point": f"{lat},{lng}",
        "unit": "kmph",
    })
    url = _TOMTOM_FLOW_URL.format(zoom=zoom) + f"?{query}"
    try:
        with urllib_request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("flowSegmentData")
    except urllib_error.HTTPError as exc:
        logger.warning("TomTom HTTP %d for (%.4f, %.4f)", exc.code, lat, lng)
    except urllib_error.URLError as exc:
        logger.warning("TomTom network error for (%.4f, %.4f): %s", lat, lng, exc.reason)
    except Exception as exc:
        logger.warning("TomTom unexpected error for (%.4f, %.4f): %s", lat, lng, exc)
    return None


# ---------------------------------------------------------------------------
# Single corridor sample
# ---------------------------------------------------------------------------

def _sample_corridor(
    corridor: dict[str, Any],
    tomtom_key: str,
    calendar: IndianCalendar,
    weather_client: WeatherClient,
    now: datetime,
) -> dict[str, Any] | None:
    """
    Collect a single observation for one corridor anchor point.
    Returns a flat dict ready for DataFrame construction, or None on failure.
    """
    lat = float(corridor["lat"])
    lng = float(corridor["lng"])
    city = corridor["city"]
    corridor_name = corridor["corridor_name"]
    road_class = corridor.get("road_class", "unknown")

    # --- Live speed data from TomTom ---
    flow = _fetch_tomtom_flow(lat, lng, tomtom_key)
    if flow is None:
        logger.warning("Skipping %s/%s — no TomTom data", city, corridor_name)
        return None

    current_speed = float(flow.get("currentSpeed", 0.0))
    free_flow_speed = float(flow.get("freeFlowSpeed", 0.0))
    current_travel_time = float(flow.get("currentTravelTime", 0.0))
    free_flow_travel_time = float(flow.get("freeFlowTravelTime", 0.0))
    road_closure = bool(flow.get("roadClosure", False))
    confidence = float(flow.get("confidence", 0.0))

    ci = compute_congestion_index(current_speed, free_flow_speed)

    # --- Weather enrichment (non-blocking) ---
    wx = weather_client.get_current(lat, lng)

    # --- Indian calendar context ---
    is_holiday = calendar.for_city(city).is_holiday(now)

    record: dict[str, Any] = {
        # Identity
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "city": city,
        "corridor_name": corridor_name,
        "lat": lat,
        "lng": lng,
        "road_class": road_class,
        # Traffic state
        "current_speed_kmh": round(current_speed, 2),
        "free_flow_speed_kmh": round(free_flow_speed, 2),
        "congestion_index": round(ci, 6),
        "current_travel_time_s": round(current_travel_time, 1),
        "free_flow_travel_time_s": round(free_flow_travel_time, 1),
        "road_closure": road_closure,
        "confidence": round(confidence, 3),
        # Temporal
        "hour": now.hour,
        "day_of_week": now.weekday(),
        "is_weekend": now.weekday() >= 5,
        "is_indian_holiday": is_holiday,
        # Weather
        "rain_mm_per_hr": round(wx.rain_mm_per_hr, 2),
        "visibility_m": round(wx.visibility_m, 0),
        "temperature_c": round(wx.temperature_c, 1),
    }
    return record


# ---------------------------------------------------------------------------
# Batch run: sample all corridors
# ---------------------------------------------------------------------------

def run_batch(
    corridors: list[dict[str, Any]],
    tomtom_key: str,
    output_dir: Path,
    weather_client: WeatherClient,
) -> int:
    """
    Sample all corridor anchors once and append records to a daily Parquet file.

    Returns the number of successfully collected records.
    """
    now = datetime.now(timezone.utc)
    calendar = IndianCalendar()

    records = []
    for corridor in corridors:
        record = _sample_corridor(corridor, tomtom_key, calendar, weather_client, now)
        if record:
            records.append(record)
            logger.info(
                "  ✓ %s / %s — CI=%.3f  speed=%s/%s km/h",
                record["city"],
                record["corridor_name"],
                record["congestion_index"],
                record["current_speed_kmh"],
                record["free_flow_speed_kmh"],
            )

    if not records:
        logger.warning("No records collected in this batch.")
        return 0

    # --- Write to partitioned Parquet ---
    try:
        import pandas as pd  # type: ignore[import]
        df = pd.DataFrame(records)

        # Partition by city and date
        for city, city_df in df.groupby("city"):
            date_str = now.strftime("%Y-%m-%d")
            out_path = output_dir / f"city={city}" / f"{date_str}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Append to existing daily file if it exists
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                city_df = pd.concat([existing, city_df], ignore_index=True)

            city_df.to_parquet(out_path, index=False)
            logger.info("  → Saved %d records to %s", len(city_df), out_path)

    except ImportError:
        # Fallback to JSONL if pandas/pyarrow not available
        jsonl_path = output_dir / f"{now.strftime('%Y-%m-%d')}.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, default=str) + "\n")
        logger.info("  → Appended %d records to %s (JSONL fallback)", len(records), jsonl_path)

    return len(records)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pan-India traffic telemetry harvester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corridors",
        default=str(_PROJECT_ROOT / "config" / "india_corridors.json"),
        help="Path to corridor anchor config JSON (default: config/india_corridors.json)",
    )
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "data" / "telemetry"),
        help="Output directory for Parquet telemetry files (default: data/telemetry/)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Sampling interval in minutes for recurring mode (default: 15)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single sampling pass and exit (no recurring schedule)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    tomtom_key = os.getenv("TOMTOM_API_KEY", "").strip()
    if not tomtom_key:
        logger.error(
            "TOMTOM_API_KEY environment variable is not set.\n"
            "Export it before running: set TOMTOM_API_KEY=your_key"
        )
        sys.exit(1)

    corridors_path = Path(args.corridors)
    if not corridors_path.exists():
        logger.error("Corridors config not found: %s", corridors_path)
        sys.exit(1)

    config = json.loads(corridors_path.read_text(encoding="utf-8"))
    corridors = config.get("corridors", [])
    if not corridors:
        logger.error("No corridors defined in %s", corridors_path)
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    weather_client = WeatherClient()  # Uses OWM_API_KEY env var

    interval_s = args.interval * 60
    logger.info("Pan-India Telemetry Harvester")
    logger.info("Corridors : %d anchor points across %d cities",
                len(corridors),
                len({c["city"] for c in corridors}))
    logger.info("Output    : %s", output_dir.resolve())
    logger.info("TomTom    : key configured ✓")
    logger.info("Weather   : %s", "OpenWeatherMap ✓" if weather_client.enabled else "disabled (set OWM_API_KEY)")
    logger.info("Mode      : %s", "single pass" if args.once else f"every {args.interval} min")
    logger.info("-" * 60)

    if args.once:
        total = run_batch(corridors, tomtom_key, output_dir, weather_client)
        logger.info("Completed. %d records saved.", total)
        return

    # Recurring mode
    while True:
        logger.info("--- Batch start at %s ---", datetime.now().strftime("%H:%M:%S"))
        run_batch(corridors, tomtom_key, output_dir, weather_client)
        logger.info("Next batch in %d minutes. Sleeping...", args.interval)
        time.sleep(interval_s)


if __name__ == "__main__":
    main()
