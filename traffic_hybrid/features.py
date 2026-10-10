from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from traffic_hybrid.schemas import (
    ROAD_CLASSES,
    compute_congestion_index as _ci_from_speeds,
)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)

    raise ValueError(f"Unsupported dataset format: {source.suffix}")


def clean_dataframe(df: pd.DataFrame, timestamp_col: str, entity_col: str | None = None) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned[timestamp_col] = pd.to_datetime(cleaned[timestamp_col], utc=False, errors="coerce")
    cleaned = cleaned.dropna(subset=[timestamp_col]).reset_index(drop=True)

    if entity_col and entity_col in cleaned.columns:
        cleaned[entity_col] = cleaned[entity_col].astype(str)
        cleaned = cleaned.sort_values([entity_col, timestamp_col]).reset_index(drop=True)
    else:
        cleaned = cleaned.sort_values(timestamp_col).reset_index(drop=True)

    cleaned = cleaned.ffill().bfill()
    return cleaned


def infer_base_feature_columns(df: pd.DataFrame, requested: list[str]) -> list[str]:
    available = [column for column in requested if column in df.columns]
    if not available:
        raise ValueError(
            "None of the configured feature columns were found. "
            f"Requested: {requested}, available: {list(df.columns)}",
        )
    return available


def fit_reference_scaler(
    df: pd.DataFrame,
    base_columns: list[str],
    fit_mask: pd.Series,
) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(df.loc[fit_mask, base_columns].astype(float))
    return scaler


def compute_reference_congestion_score(
    df: pd.DataFrame,
    base_columns: list[str],
    reference_scaler: MinMaxScaler,
) -> pd.Series:
    scaled = reference_scaler.transform(df[base_columns].astype(float))
    components: list[np.ndarray] = []

    for index, column in enumerate(base_columns):
        values = scaled[:, index]
        if column.lower() == "speed":
            components.append(1.0 - values)
        else:
            components.append(values)

    score = np.mean(np.column_stack(components), axis=1)
    return pd.Series(score, index=df.index, name="reference_congestion_score")


def compute_congestion_index_column(
    df: pd.DataFrame,
    current_speed_col: str = "current_speed_kmh",
    free_flow_col: str = "free_flow_speed_kmh",
    vehicles_col: str | None = None,
    max_vehicles_capacity: float | None = None,
) -> pd.Series:
    """
    Derive a universal, normalized Congestion Index (CI) in [0.0, 1.0] for
    any row in a DataFrame.

    Priority order:
      1. If both ``current_speed_col`` and ``free_flow_col`` are present and
         the free-flow column has non-zero values, compute:
             CI = 1 - clamp(current_speed / free_flow_speed, 0, 1)
      2. If only raw vehicle counts are available (legacy dataset format),
         estimate CI by normalising counts against the rolling 95th-percentile
         capacity proxy within each group.

    This function is the gateway for transitioning the existing
    ``data/traffic.csv`` vehicle-count dataset into the universal schema
    without retraining from scratch.
    """
    # --- Strategy 1: speed-ratio CI (preferred, universal) ---
    if current_speed_col in df.columns and free_flow_col in df.columns:
        current_speed = df[current_speed_col].astype(float)
        free_flow = df[free_flow_col].astype(float)
        # Vectorised version of compute_congestion_index
        ratio = (current_speed / free_flow.replace(0.0, np.nan)).clip(0.0, 1.0)
        ci = 1.0 - ratio.fillna(0.5)  # 0.5 = unknown / moderate fallback
        return pd.Series(ci.values, index=df.index, name="congestion_index")

    # --- Strategy 2: vehicle-count proxy CI (legacy fallback) ---
    if vehicles_col and vehicles_col in df.columns:
        counts = df[vehicles_col].astype(float)
        if max_vehicles_capacity is not None and max_vehicles_capacity > 0:
            ci = (counts / max_vehicles_capacity).clip(0.0, 1.0)
        else:
            # Use 95th-percentile of the full series as a capacity proxy
            capacity = counts.quantile(0.95)
            if capacity > 0:
                ci = (counts / capacity).clip(0.0, 1.0)
            else:
                ci = pd.Series(np.zeros(len(df)), index=df.index)
        return pd.Series(ci.values, index=df.index, name="congestion_index")

    # --- Strategy 3: zero-fill fallback (data missing) ---
    return pd.Series(np.zeros(len(df)), index=df.index, name="congestion_index")


def _groupby_series(df: pd.DataFrame, entity_col: str | None, column: str) -> pd.core.groupby.generic.SeriesGroupBy:
    if entity_col and entity_col in df.columns:
        return df.groupby(entity_col, sort=False)[column]
    return df.groupby(lambda _: "global", sort=False)[column]


def add_engineered_features(
    df: pd.DataFrame,
    timestamp_col: str,
    entity_col: str | None,
    base_columns: list[str],
    peak_hours: list[int],
    reference_score: pd.Series,
    use_peak_indicator: bool,
    use_congestion_transition: bool,
    use_temporal_features: bool,
    use_lag_features: bool,
    use_entity_one_hot: bool,
    # Pan-India contextual features (optional — safely ignored if columns absent)
    use_india_context: bool = True,
) -> pd.DataFrame:
    engineered = df.copy()
    primary_column = base_columns[0]
    primary_series = engineered[primary_column].astype(float)

    if use_temporal_features:
        hour = engineered[timestamp_col].dt.hour.astype(float)
        day_of_week = engineered[timestamp_col].dt.dayofweek.astype(float)
        month = engineered[timestamp_col].dt.month.astype(float)
        day = engineered[timestamp_col].dt.day.astype(float)
        week_of_year = engineered[timestamp_col].dt.isocalendar().week.astype(float)
        year = engineered[timestamp_col].dt.year.astype(float)

        engineered["hour"] = hour
        engineered["day_of_week"] = day_of_week
        engineered["month"] = month
        engineered["day_of_month"] = day
        engineered["week_of_year"] = week_of_year
        engineered["year"] = year

        engineered["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        engineered["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        engineered["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
        engineered["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
        engineered["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
        engineered["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
        engineered["is_weekend"] = engineered[timestamp_col].dt.dayofweek.isin([5, 6]).astype(float)

    if use_peak_indicator:
        engineered["peak_indicator"] = engineered[timestamp_col].dt.hour.isin(peak_hours).astype(float)

    group_series = _groupby_series(engineered, entity_col, primary_column)
    if use_lag_features:
        engineered["lag_1"] = group_series.shift(1)
        engineered["lag_2"] = group_series.shift(2)
        engineered["lag_3"] = group_series.shift(3)
        engineered["lag_6"] = group_series.shift(6)
        engineered["lag_12"] = group_series.shift(12)
        engineered["lag_24"] = group_series.shift(24)
        engineered["lag_48"] = group_series.shift(48)
        engineered["lag_72"] = group_series.shift(72)
        engineered["lag_168"] = group_series.shift(168)
        engineered["rolling_mean_3"] = group_series.transform(
            lambda series: series.rolling(window=3, min_periods=1).mean(),
        )
        engineered["rolling_mean_6"] = group_series.transform(
            lambda series: series.rolling(window=6, min_periods=1).mean(),
        )
        engineered["rolling_mean_12"] = group_series.transform(
            lambda series: series.rolling(window=12, min_periods=1).mean(),
        )
        engineered["rolling_mean_24"] = group_series.transform(
            lambda series: series.rolling(window=24, min_periods=1).mean(),
        )
        engineered["rolling_mean_48"] = group_series.transform(
            lambda series: series.rolling(window=48, min_periods=1).mean(),
        )
        engineered["rolling_mean_168"] = group_series.transform(
            lambda series: series.rolling(window=168, min_periods=1).mean(),
        )
        engineered["rolling_std_24"] = group_series.transform(
            lambda series: series.rolling(window=24, min_periods=2).std(),
        )
        engineered["rolling_std_168"] = group_series.transform(
            lambda series: series.rolling(window=168, min_periods=2).std(),
        )
        engineered["vehicle_change_rate"] = group_series.pct_change().replace([np.inf, -np.inf], 0.0)
        engineered["daily_change_rate"] = (primary_series / engineered["lag_24"].replace(0, np.nan)) - 1.0
        engineered["weekly_change_rate"] = (primary_series / engineered["lag_168"].replace(0, np.nan)) - 1.0
        engineered["trend_gap_24"] = primary_series - engineered["rolling_mean_24"]
        engineered["trend_gap_168"] = primary_series - engineered["rolling_mean_168"]

    if {"speed", "occupancy"}.issubset(base_columns):
        speed_series = engineered["speed"].astype(float)
        density = engineered["occupancy"].astype(float)
        previous_speed = _groupby_series(engineered, entity_col, "speed").shift(1).replace(0, np.nan)
        engineered["speed_drop_rate"] = (
            (previous_speed - speed_series) / previous_speed.abs().clip(lower=1e-3)
        )
        engineered["density_growth"] = _groupby_series(engineered, entity_col, "occupancy").pct_change()
    else:
        engineered["speed_drop_rate"] = engineered.get("vehicle_change_rate", 0.0)
        engineered["density_growth"] = engineered.get("daily_change_rate", 0.0)

    if use_congestion_transition:
        if entity_col and entity_col in engineered.columns:
            transition = engineered.groupby(entity_col, sort=False)["reference_congestion_score"].diff()
        else:
            transition = engineered["reference_congestion_score"].diff()
        engineered["congestion_transition"] = np.sign(transition.fillna(0.0)).astype(float)

    if use_entity_one_hot and entity_col and entity_col in engineered.columns:
        dummies = pd.get_dummies(engineered[entity_col], prefix=entity_col.lower())
        engineered = pd.concat([engineered, dummies.astype(float)], axis=1)

    # ------------------------------------------------------------------
    # Pan-India contextual feature injection
    # Safely passes through if the columns do not exist in the dataframe
    # (backward compatible with the legacy 4-junction vehicle-count dataset).
    # ------------------------------------------------------------------
    if use_india_context:
        # Road class one-hot encoding (from OpenStreetMap via spatial.py)
        if "road_class" in engineered.columns:
            for cls in ROAD_CLASSES:
                engineered[f"road_class_{cls}"] = (
                    engineered["road_class"].astype(str) == cls
                ).astype(float)

        # Weather: precipitation rate (from OpenWeather / Tomorrow.io)
        if "rain_mm_per_hr" not in engineered.columns:
            engineered["rain_mm_per_hr"] = 0.0

        # Indian holiday flag (from context.py)
        if "is_indian_holiday" not in engineered.columns:
            engineered["is_indian_holiday"] = 0.0

        # Road geometry (from OpenStreetMap via spatial.py)
        if "lanes" not in engineered.columns:
            engineered["lanes"] = 1.0
        if "speed_limit_kmh" not in engineered.columns:
            engineered["speed_limit_kmh"] = 40.0

        # Universal target column: Congestion Index
        if "congestion_index" not in engineered.columns:
            engineered["congestion_index"] = compute_congestion_index_column(
                engineered,
                vehicles_col=base_columns[0] if base_columns else None,
            )

    numeric_columns = engineered.select_dtypes(include=[np.number]).columns
    engineered[numeric_columns] = engineered[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return engineered


def select_feature_columns(
    df: pd.DataFrame,
    base_columns: list[str],
    use_engineered_features: bool,
    use_peak_indicator: bool,
    use_congestion_transition: bool,
    use_temporal_features: bool,
    use_lag_features: bool,
    use_entity_one_hot: bool,
    entity_col: str | None,
) -> list[str]:
    columns = list(base_columns)

    if use_engineered_features:
        engineered_candidates = ["speed_drop_rate", "density_growth"]
        if use_lag_features:
            engineered_candidates.extend(
                [
                    "lag_1",
                    "lag_2",
                    "lag_3",
                    "lag_6",
                    "lag_12",
                    "lag_24",
                    "lag_48",
                    "lag_72",
                    "lag_168",
                    "rolling_mean_3",
                    "rolling_mean_6",
                    "rolling_mean_12",
                    "rolling_mean_24",
                    "rolling_mean_48",
                    "rolling_mean_168",
                    "rolling_std_24",
                    "rolling_std_168",
                    "vehicle_change_rate",
                    "daily_change_rate",
                    "weekly_change_rate",
                    "trend_gap_24",
                    "trend_gap_168",
                ],
            )
        columns.extend([name for name in engineered_candidates if name in df.columns])

    if use_peak_indicator and "peak_indicator" in df.columns:
        columns.append("peak_indicator")
    if use_temporal_features:
        columns.extend(
            [
                name
                for name in [
                    "hour",
                    "day_of_week",
                    "month",
                    "day_of_month",
                    "week_of_year",
                    "year",
                    "hour_sin",
                    "hour_cos",
                    "dow_sin",
                    "dow_cos",
                    "month_sin",
                    "month_cos",
                    "is_weekend",
                ]
                if name in df.columns
            ],
        )
    if use_congestion_transition and "congestion_transition" in df.columns:
        columns.append("congestion_transition")
    if use_entity_one_hot and entity_col:
        columns.extend(sorted([name for name in df.columns if name.startswith(f"{entity_col.lower()}_")]))

    # --- Pan-India contextual features (injected by add_engineered_features) ---
    india_context_candidates = [
        # Road geometry (from OSM)
        "lanes",
        "speed_limit_kmh",
        # Weather
        "rain_mm_per_hr",
        # Calendar
        "is_indian_holiday",
        # Universal target
        "congestion_index",
    ]
    # Road class one-hot columns
    india_context_candidates.extend([f"road_class_{cls}" for cls in ROAD_CLASSES])
    columns.extend([name for name in india_context_candidates if name in df.columns and name not in columns])

    deduplicated: list[str] = []
    seen = set()
    for column in columns:
        if column not in seen:
            seen.add(column)
            deduplicated.append(column)
    return deduplicated
