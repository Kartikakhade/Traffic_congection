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

import pandas as pd
from flask import Flask, jsonify, render_template, request

from traffic_hybrid.features import clean_dataframe, load_dataframe


FORECAST_CLASS_DETAILS = {
    0: {
        "label": "Low forecast congestion",
        "tone": "low",
        "summary": "The trained model expects light traffic in the next forecast window for this junction.",
    },
    1: {
        "label": "Medium forecast congestion",
        "tone": "medium",
        "summary": "The trained model expects moderate traffic in the next forecast window for this junction.",
    },
    2: {
        "label": "High forecast congestion",
        "tone": "high",
        "summary": "The trained model expects heavy traffic in the next forecast window for this junction.",
    },
}

LIVE_TRAFFIC_DETAILS = {
    "low": {
        "label": "Live traffic is light",
        "tone": "low",
        "summary": "TomTom live traffic reports indicate normal flow near the clicked point.",
    },
    "medium": {
        "label": "Live traffic is moderate",
        "tone": "medium",
        "summary": "TomTom live traffic reports indicate noticeable slowing near the clicked point.",
    },
    "high": {
        "label": "Live traffic is heavy",
        "tone": "high",
        "summary": "TomTom live traffic reports indicate heavy delay or possible congestion near the clicked point.",
    },
}


def _read_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_junction_locations(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf8"))
    payload["max_match_distance_km"] = float(payload.get("max_match_distance_km", 5.0))
    payload["junctions"] = [
        {
            **junction,
            "id": str(junction["id"]),
            "lat": float(junction["lat"]),
            "lng": float(junction["lng"]),
        }
        for junction in payload.get("junctions", [])
    ]
    return payload


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
    cache_ttl_seconds: int = 120
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
        if ratio <= 0.45:
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
                "summary": "Set TOMTOM_API_KEY to load live TomTom traffic for today.",
            }

        cache_key = f"{lat:.5f},{lng:.5f}"
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
            with urllib_request.urlopen(url, timeout=15) as response:
                data = json.loads(response.read().decode("utf8"))
        except urllib_error.HTTPError as exc:
            payload = {
                "available": False,
                "source": "TomTom Traffic Flow API",
                "reason": "http_error",
                "summary": f"TomTom traffic request failed with HTTP {exc.code}.",
            }
            return self._cache_set(cache_key, payload)
        except urllib_error.URLError:
            payload = {
                "available": False,
                "source": "TomTom Traffic Flow API",
                "reason": "network_error",
                "summary": "TomTom traffic service could not be reached from this machine.",
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
        payload = {
            "available": True,
            "source": "TomTom Traffic Flow API",
            "label": details["label"],
            "tone": details["tone"],
            "summary": details["summary"],
            "checkedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "currentSpeed": round(current_speed, 2),
            "freeFlowSpeed": round(free_flow_speed, 2),
            "delayPercent": round(max(0.0, (delay_ratio - 1.0) * 100.0), 1),
            "currentTravelTimeSeconds": round(current_travel, 1),
            "freeFlowTravelTimeSeconds": round(free_flow_travel, 1),
            "confidence": round(confidence, 2),
            "roadClosure": road_closure,
            "note": "Live traffic is based on TomTom flow speed around the clicked point.",
        }
        return self._cache_set(cache_key, payload)


@dataclass
class PredictionStore:
    artifacts_dir: Path
    data_path: Path
    locations_path: Path

    def __post_init__(self) -> None:
        self.metadata = _read_pickle(self.artifacts_dir / "metadata.pkl")
        self.locations = _load_junction_locations(self.locations_path)
        self.raw_history = self._load_history()
        self.predictions = self._load_predictions()
        self.junction_lookup = {junction["id"]: junction for junction in self.locations["junctions"]}
        self.tomtom = TomTomTrafficClient(api_key=_resolve_tomtom_key())

    def _load_history(self) -> pd.DataFrame:
        history = clean_dataframe(
            load_dataframe(self.data_path),
            self.metadata["timestamp_col"],
            self.metadata.get("entity_col"),
        )
        entity_col = self.metadata.get("entity_col")
        if entity_col and entity_col in history.columns:
            history[entity_col] = history[entity_col].astype(str)
        return history

    def _load_predictions(self) -> pd.DataFrame:
        predictions_path = self.artifacts_dir / "live_predictions.csv"
        if not predictions_path.exists():
            from traffic_hybrid.inference import predict_live_file

            generated = predict_live_file(self.artifacts_dir, self.data_path)
            generated.to_csv(predictions_path, index=False)

        predictions = pd.read_csv(
            predictions_path,
            parse_dates=["timestamp", "forecast_timestamp", "forecast_window_end_timestamp"],
        )
        entity_col = self.metadata.get("entity_col")
        if entity_col and entity_col in predictions.columns:
            predictions[entity_col] = predictions[entity_col].astype(str)
        return predictions

    def get_bootstrap(self) -> dict[str, Any]:
        latest_by_junction = []
        entity_col = self.metadata.get("entity_col")
        for junction in self.locations["junctions"]:
            latest = self._latest_prediction_for_junction(junction["id"])
            latest_by_junction.append(
                {
                    "id": junction["id"],
                    "name": junction["name"],
                    "lat": junction["lat"],
                    "lng": junction["lng"],
                    "latestLabel": FORECAST_CLASS_DETAILS[int(latest["prediction_class"])]["label"],
                    "latestForecast": latest["forecast_timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        placeholder = bool(self.locations.get("use_placeholder_locations", False))
        return {
            "entityColumn": entity_col,
            "mapCenter": self.locations["map_center"],
            "usePlaceholderLocations": placeholder,
            "predictionReady": not placeholder,
            "maxMatchDistanceKm": float(self.locations.get("max_match_distance_km", 5.0)),
            "liveTrafficEnabled": self.tomtom.enabled,
            "junctions": latest_by_junction,
            "dataWindow": {
                "historyStart": self.raw_history[self.metadata["timestamp_col"]].min().strftime("%Y-%m-%d %H:%M:%S"),
                "historyEnd": self.raw_history[self.metadata["timestamp_col"]].max().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

    def _latest_prediction_for_junction(self, junction_id: str) -> pd.Series:
        entity_col = self.metadata.get("entity_col")
        junction_predictions = (
            self.predictions[self.predictions[entity_col] == str(junction_id)]
            if entity_col and entity_col in self.predictions.columns
            else self.predictions
        )
        return junction_predictions.sort_values("forecast_timestamp").iloc[-1]

    def _recent_history_for_junction(self, junction_id: str, limit: int = 24) -> pd.DataFrame:
        entity_col = self.metadata.get("entity_col")
        history = (
            self.raw_history[self.raw_history[entity_col] == str(junction_id)]
            if entity_col and entity_col in self.raw_history.columns
            else self.raw_history
        )
        return history.sort_values(self.metadata["timestamp_col"]).tail(limit)

    def _recent_predictions_for_junction(self, junction_id: str, limit: int = 24) -> pd.DataFrame:
        entity_col = self.metadata.get("entity_col")
        predictions = (
            self.predictions[self.predictions[entity_col] == str(junction_id)]
            if entity_col and entity_col in self.predictions.columns
            else self.predictions
        )
        return predictions.sort_values("forecast_timestamp").tail(limit)

    def _nearest_junction(self, lat: float, lng: float) -> tuple[dict[str, Any], float]:
        ranked = [
            (
                junction,
                _haversine_km(lat, lng, junction["lat"], junction["lng"]),
            )
            for junction in self.locations["junctions"]
        ]
        return min(ranked, key=lambda item: item[1])

    def _build_trend_payload(self, junction_id: str) -> dict[str, Any]:
        history = self._recent_history_for_junction(junction_id)
        predictions = self._recent_predictions_for_junction(junction_id)
        return {
            "history": [
                {
                    "timestamp": timestamp.strftime("%d %b %H:%M"),
                    "value": float(value),
                }
                for timestamp, value in zip(
                    history[self.metadata["timestamp_col"]],
                    history["Vehicles"],
                )
            ],
            "forecast": [
                {
                    "timestamp": timestamp.strftime("%d %b %H:%M"),
                    "value": float(value),
                }
                for timestamp, value in zip(
                    predictions["forecast_timestamp"],
                    predictions["prediction_score"],
                )
            ],
        }

    def prediction_for_point(self, lat: float, lng: float) -> dict[str, Any]:
        live_traffic = self.tomtom.flow_segment_for_point(lat, lng)
        junction, distance_km = self._nearest_junction(lat, lng)
        placeholder = bool(self.locations.get("use_placeholder_locations", False))
        max_match_distance_km = float(self.locations.get("max_match_distance_km", 5.0))

        forecast_blocked = None
        if placeholder:
            forecast_blocked = "placeholder_locations"
        elif distance_km > max_match_distance_km:
            forecast_blocked = "too_far_from_junction"

        payload: dict[str, Any] = {
            "selectedPoint": {"lat": lat, "lng": lng},
            "junction": {
                "id": junction["id"],
                "name": junction["name"],
                "lat": junction["lat"],
                "lng": junction["lng"],
                "distanceKm": round(distance_km, 3),
                "description": junction.get("description", ""),
            },
            "limits": {"maxMatchDistanceKm": max_match_distance_km},
            "liveTraffic": live_traffic,
            "forecastReady": forecast_blocked is None,
            "forecastBlockedReason": forecast_blocked,
        }

        if forecast_blocked is not None:
            payload["forecastMessage"] = (
                "Forecasts are disabled until real junction coordinates are configured."
                if forecast_blocked == "placeholder_locations"
                else (
                    f"The selected point is {distance_km:.3f} km away from the nearest configured junction, "
                    f"which is beyond the allowed {max_match_distance_km:.1f} km match radius."
                )
            )
            return payload

        latest_prediction = self._latest_prediction_for_junction(junction["id"])
        recent_history = self._recent_history_for_junction(junction["id"])
        thresholds = self.metadata["thresholds"].get(str(junction["id"]), [])
        class_id = int(latest_prediction["prediction_class"])
        class_details = FORECAST_CLASS_DETAILS.get(class_id, FORECAST_CLASS_DETAILS[1])
        latest_actual = recent_history.iloc[-1]

        payload["forecast"] = {
            "label": class_details["label"],
            "tone": class_details["tone"],
            "summary": class_details["summary"],
            "predictedVehicles": round(float(latest_prediction["prediction_score"]), 2),
            "forecastTimestamp": latest_prediction["forecast_timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "forecastWindowEndTimestamp": latest_prediction["forecast_window_end_timestamp"].strftime(
                "%Y-%m-%d %H:%M:%S",
            ),
            "latestObservedTimestamp": latest_actual[self.metadata["timestamp_col"]].strftime("%Y-%m-%d %H:%M:%S"),
            "latestObservedVehicles": round(float(latest_actual["Vehicles"]), 2),
            "thresholds": [round(float(value), 2) for value in thresholds],
            "modelAccuracy": 87.663,
            "source": "Hybrid model forecast",
        }
        payload["trend"] = self._build_trend_payload(junction["id"])
        return payload


def create_app(
    artifacts_dir: str | Path = "artifacts/tuned_high_accuracy",
    data_path: str | Path = "data/traffic.csv",
    locations_path: str | Path = "config/junction_locations.json",
) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent.parent / "web" / "templates"),
        static_folder=str(Path(__file__).resolve().parent.parent / "web" / "static"),
        static_url_path="/static",
    )
    app.config["JSON_SORT_KEYS"] = False
    app.store = PredictionStore(
        artifacts_dir=Path(artifacts_dir),
        data_path=Path(data_path),
        locations_path=Path(locations_path),
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
        return jsonify(app.store.prediction_for_point(lat, lng))

    return app
