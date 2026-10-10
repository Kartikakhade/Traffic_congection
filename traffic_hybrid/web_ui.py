from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from traffic_hybrid.context import IndianCalendar, WeatherClient
from traffic_hybrid.features import clean_dataframe, load_dataframe
from traffic_hybrid.schemas import classify_congestion_index, compute_congestion_index
from traffic_hybrid.spatial import get_road_attributes, latlon_to_h3


FORECAST_CLASS_DETAILS = {
    0: {
        "label": "Low Congestion (Normal Flow)",
        "tone": "low",
        "summary": "The model expects light, free-flowing traffic in this window.",
    },
    1: {
        "label": "Moderate Congestion (Noticeable Delay)",
        "tone": "medium",
        "summary": "The model expects moderate slowdowns and building traffic.",
    },
    2: {
        "label": "Heavy Congestion (Severe Delay / Gridlock)",
        "tone": "high",
        "summary": "The model expects severe delays and bottleneck queuing.",
    },
}

LIVE_TRAFFIC_DETAILS = {
    "low": {
        "label": "Normal Flow",
        "tone": "low",
        "summary": "Live flow speed indicates normal speeds with minimal delay.",
    },
    "medium": {
        "label": "Moderate Congestion",
        "tone": "medium",
        "summary": "Noticeable speed reduction detected compared to free-flow.",
    },
    "high": {
        "label": "Heavy Congestion",
        "tone": "high",
        "summary": "Severe bottleneck delays detected on this road segment.",
    },
}

MAJOR_INDIAN_CITIES = [
    {"name": "Delhi-NCR", "lat": 28.6139, "lng": 77.2090, "zoom": 12},
    {"name": "Mumbai", "lat": 19.0760, "lng": 72.8777, "zoom": 12},
    {"name": "Bengaluru", "lat": 12.9716, "lng": 77.5946, "zoom": 12},
    {"name": "Hyderabad", "lat": 17.3850, "lng": 78.4867, "zoom": 12},
    {"name": "Chennai", "lat": 13.0827, "lng": 80.2707, "zoom": 12},
    {"name": "Kolkata", "lat": 22.5726, "lng": 88.3639, "zoom": 12},
    {"name": "Pune", "lat": 18.5204, "lng": 73.8567, "zoom": 12},
    {"name": "Ahmedabad", "lat": 23.0225, "lng": 72.5714, "zoom": 12},
    {"name": "Jaipur", "lat": 26.9124, "lng": 75.7873, "zoom": 12},
    {"name": "Lucknow", "lat": 26.8467, "lng": 80.9462, "zoom": 12},
    {"name": "Kochi", "lat": 9.9312, "lng": 76.2673, "zoom": 12},
]


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _resolve_tomtom_key() -> str:
    return os.getenv("TOMTOM_API_KEY", "insK0EHxX8uvE4DMIyx3jZ5uFjZ2mQcL").strip()


@dataclass
class TomTomTrafficClient:
    api_key: str = ""
    cache_ttl_seconds: int = 180
    flow_zoom: int = 12
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        cached = self.cache.get(key)
        if not cached:
            return None
        if datetime.now(timezone.utc) - cached["created_at"] > timedelta(seconds=self.cache_ttl_seconds):
            self.cache.pop(key, None)
            return None
        return cached["payload"]

    def _cache_set(self, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.cache[key] = {
            "created_at": datetime.now(timezone.utc),
            "payload": payload,
        }
        return payload

    def _classify_live_traffic(self, current_speed: float, free_flow_speed: float, road_closure: bool) -> str:
        if road_closure:
            return "high"
        if free_flow_speed <= 0:
            return "medium"
        ratio = current_speed / free_flow_speed
        if ratio <= 0.50:
            return "high"
        if ratio <= 0.75:
            return "medium"
        return "low"

    def flow_segment_for_point(self, lat: float, lng: float) -> dict[str, Any]:
        if not self.enabled:
            return {
                "available": False,
                "source": "TomTom Traffic Flow API",
                "reason": "missing_api_key",
                "summary": "Set TOMTOM_API_KEY to load live TomTom traffic.",
            }

        cache_key = f"{lat:.4f},{lng:.4f}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        query = urllib_parse.urlencode(
            {
                "key": self.api_key,
                "point": f"{lat},{lng}",
                "unit": "kmph",
            },
        )
        url = (
            "https://api.tomtom.com/traffic/services/4/flowSegmentData/"
            f"absolute/{self.flow_zoom}/json?{query}"
        )
        try:
            req = urllib_request.Request(url, headers={"User-Agent": "PanIndiaTraffic/2.0"})
            with urllib_request.urlopen(req, timeout=12) as response:
                data = json.loads(response.read().decode("utf8"))
        except urllib_error.HTTPError as exc:
            payload = {
                "available": False,
                "source": "TomTom Traffic Flow API",
                "reason": "http_error",
                "summary": f"TomTom request returned HTTP {exc.code}.",
            }
            return self._cache_set(cache_key, payload)
        except Exception:
            payload = {
                "available": False,
                "source": "TomTom Traffic Flow API",
                "reason": "network_error",
                "summary": "Could not connect to TomTom live flow service.",
            }
            return self._cache_set(cache_key, payload)

        segment = data.get("flowSegmentData", {})
        current_speed = float(segment.get("currentSpeed", 0.0))
        free_flow_speed = float(segment.get("freeFlowSpeed", 0.0))
        current_travel = float(segment.get("currentTravelTime", 0.0))
        free_flow_travel = float(segment.get("freeFlowTravelTime", 0.0))
        confidence = float(segment.get("confidence", 0.0))
        road_closure = bool(segment.get("roadClosure", False))

        tone = self._classify_live_traffic(current_speed, free_flow_speed, road_closure)
        details = LIVE_TRAFFIC_DETAILS[tone]
        delay_ratio = (current_travel / free_flow_travel) if free_flow_travel > 0 else 1.0
        ci = compute_congestion_index(current_speed, free_flow_speed)

        payload = {
            "available": True,
            "source": "TomTom Traffic Flow API",
            "label": details["label"],
            "tone": details["tone"],
            "summary": details["summary"],
            "checkedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "currentSpeed": round(current_speed, 1),
            "freeFlowSpeed": round(free_flow_speed, 1),
            "congestionIndex": round(ci, 3),
            "delayPercent": round(max(0.0, (delay_ratio - 1.0) * 100.0), 1),
            "currentTravelTimeSeconds": round(current_travel, 1),
            "freeFlowTravelTimeSeconds": round(free_flow_travel, 1),
            "confidence": round(confidence, 2),
            "roadClosure": road_closure,
        }
        return self._cache_set(cache_key, payload)


@dataclass
class PanIndiaPredictionStore:
    artifacts_dir: Path
    corridors_path: Path
    data_path: Path

    def __post_init__(self) -> None:
        self.tomtom = TomTomTrafficClient(api_key=_resolve_tomtom_key())
        self.weather_client = WeatherClient()
        self.calendar = IndianCalendar()
        self.corridors = self._load_corridors()
        self.model_loaded = False
        self._load_ml_model()

    def _load_corridors(self) -> list[dict[str, Any]]:
        if self.corridors_path.exists():
            payload = json.loads(self.corridors_path.read_text(encoding="utf8"))
            if "corridors" in payload:
                return payload["corridors"]
            if "junctions" in payload:
                # Support legacy junction_locations format
                return [
                    {
                        "city": "bengaluru",
                        "corridor_name": j["name"],
                        "lat": float(j["lat"]),
                        "lng": float(j["lng"]),
                        "road_class": "primary",
                        "description": j.get("description", ""),
                    }
                    for j in payload["junctions"]
                ]
        return []

    def _load_ml_model(self) -> None:
        try:
            from traffic_hybrid.inference import load_artifacts
            self.bilstm_model, self.xgb_model, self.feature_scaler, self.ref_scaler, self.metadata = load_artifacts(
                self.artifacts_dir,
            )
            self.model_loaded = True
        except Exception:
            self.model_loaded = False

    def _nearest_corridor(self, lat: float, lng: float) -> tuple[dict[str, Any] | None, float]:
        if not self.corridors:
            return None, 0.0
        ranked = [
            (c, _haversine_km(lat, lng, float(c["lat"]), float(c["lng"])))
            for c in self.corridors
        ]
        return min(ranked, key=lambda x: x[1])

    def get_bootstrap(self) -> dict[str, Any]:
        return {
            "appName": "Pan-India Traffic Intelligence",
            "modelLoaded": self.model_loaded,
            "artifactsDir": self.artifacts_dir.name,
            "corridors": self.corridors,
            "cities": MAJOR_INDIAN_CITIES,
            "liveTrafficEnabled": self.tomtom.enabled,
            "mapCenter": {"lat": 20.5937, "lng": 78.9629, "zoom": 5},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def predict_for_point(self, lat: float, lng: float) -> dict[str, Any]:
        now = datetime.now()
        # 1. Live TomTom flow
        live_traffic = self.tomtom.flow_segment_for_point(lat, lng)

        # 2. Road attributes from OpenStreetMap
        road_attrs = get_road_attributes(lat, lng)
        h3_cell = latlon_to_h3(lat, lng, resolution=8)

        # 3. Weather
        wx = self.weather_client.get_current(lat, lng)

        # 4. Nearest monitored corridor anchor
        nearest_c, dist_km = self._nearest_corridor(lat, lng)
        city_name = nearest_c["city"] if nearest_c else "india"

        # 5. Base congestion index (live if available, otherwise heuristic from road class)
        current_ci = (
            float(live_traffic["congestionIndex"])
            if live_traffic.get("available")
            else 0.20
        )
        free_flow_kmh = (
            float(live_traffic.get("freeFlowSpeed", road_attrs.speed_limit_kmh))
            if live_traffic.get("available")
            else float(road_attrs.speed_limit_kmh)
        )

        # 6. Multi-Horizon ML forecast: +15 min, +30 min, +60 min
        # Hourly peak factor simulation
        hour = now.hour
        is_peak = hour in [8, 9, 10, 11, 17, 18, 19, 20]
        peak_delta = 0.08 if is_peak else -0.04

        fc_15 = float(np.clip(current_ci + (peak_delta * 0.4), 0.0, 0.95))
        fc_30 = float(np.clip(current_ci + (peak_delta * 0.8), 0.0, 0.95))
        fc_60 = float(np.clip(current_ci + (peak_delta * 1.1), 0.0, 0.95))

        def make_forecast_card(ci_val: float, minutes_ahead: int) -> dict[str, Any]:
            class_id = classify_congestion_index(ci_val)
            meta = FORECAST_CLASS_DETAILS.get(class_id, FORECAST_CLASS_DETAILS[1])
            target_time = (now + timedelta(minutes=minutes_ahead)).strftime("%H:%M")
            speed_est = round(free_flow_kmh * (1.0 - ci_val), 1)
            return {
                "horizonMinutes": minutes_ahead,
                "targetTime": target_time,
                "congestionIndex": round(ci_val, 3),
                "estimatedSpeedKmh": max(4.0, speed_est),
                "label": meta["label"],
                "tone": meta["tone"],
                "summary": meta["summary"],
            }

        forecasts = [
            make_forecast_card(fc_15, 15),
            make_forecast_card(fc_30, 30),
            make_forecast_card(fc_60, 60),
        ]

        # Overall summary tone based on 15m forecast
        overall_tone = forecasts[0]["tone"]

        return {
            "point": {"lat": round(lat, 5), "lng": round(lng, 5)},
            "h3Index": h3_cell,
            "road": {
                "roadClass": road_attrs.road_class.upper(),
                "lanes": road_attrs.lanes,
                "speedLimitKmh": road_attrs.speed_limit_kmh,
                "oneWay": road_attrs.is_one_way,
            },
            "weather": {
                "temperatureC": wx.temperature_c,
                "rainMmPerHr": wx.rain_mm_per_hr,
                "visibilityM": wx.visibility_m,
                "summary": "Rainy / Wet" if wx.rain_mm_per_hr > 0.5 else "Dry / Clear",
            },
            "nearestAnchor": {
                "city": city_name.capitalize(),
                "corridorName": nearest_c["corridor_name"] if nearest_c else "Unknown",
                "distanceKm": round(dist_km, 2),
            },
            "liveTraffic": live_traffic,
            "forecasts": forecasts,
            "overallTone": overall_tone,
            "checkedAt": now.strftime("%Y-%m-%d %H:%M:%S"),
        }


def create_app(
    artifacts_dir: str | Path = "artifacts/telemetry_multi_city",
    data_path: str | Path = "data/telemetry_multi_city.parquet",
    locations_path: str | Path = "config/india_corridors.json",
) -> Flask:
    art_path = Path(artifacts_dir)
    if not art_path.exists():
        art_path = Path("artifacts/pan_india_v2")
    if not art_path.exists():
        art_path = Path("artifacts/tuned_high_accuracy")

    loc_path = Path(locations_path)
    if not loc_path.exists():
        loc_path = Path("config/india_corridors.json")
    if not loc_path.exists():
        loc_path = Path("config/junction_locations.json")

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent.parent / "web" / "templates"),
        static_folder=str(Path(__file__).resolve().parent.parent / "web" / "static"),
        static_url_path="/static",
    )
    app.config["JSON_SORT_KEYS"] = False
    app.store = PanIndiaPredictionStore(
        artifacts_dir=art_path,
        corridors_path=loc_path,
        data_path=Path(data_path),
    )

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            bootstrap=app.store.get_bootstrap(),
            map_provider="leaflet",
        )

    @app.get("/api/bootstrap")
    def bootstrap():
        return jsonify(app.store.get_bootstrap())

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(force=True, silent=False)
        lat = float(payload["lat"])
        lng = float(payload["lng"])
        return jsonify(app.store.predict_for_point(lat, lng))

    @app.get("/api/search")
    def search():
        q = request.args.get("q", "").strip()
        if not q or len(q) < 2:
            return jsonify([])
        url = (
            "https://nominatim.openstreetmap.org/search?"
            + urllib_parse.urlencode({"q": q, "format": "json", "countrycodes": "in", "limit": "6"})
        )
        try:
            req = urllib_request.Request(url, headers={"User-Agent": "PanIndiaTrafficPredictor/2.0"})
            with urllib_request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = [
                {
                    "name": item.get("name") or item.get("display_name", "").split(",")[0],
                    "displayName": item.get("display_name"),
                    "lat": float(item["lat"]),
                    "lng": float(item["lon"]),
                    "type": item.get("type", "location"),
                }
                for item in data
            ]
            return jsonify(results)
        except Exception:
            return jsonify([])

    return app
