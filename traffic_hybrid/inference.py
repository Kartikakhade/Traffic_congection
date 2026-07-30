from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from xgboost import XGBRegressor

from traffic_hybrid.data import build_tabular_context_vector
from traffic_hybrid.features import (
    add_engineered_features,
    clean_dataframe,
    compute_reference_congestion_score,
    load_dataframe,
)
from traffic_hybrid.metrics import classify
from traffic_hybrid.model import AttentionPooling, LastStepSlice


def _read_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_artifacts(artifacts_dir: str | Path):
    root = Path(artifacts_dir)
    tf.get_logger().setLevel("ERROR")
    bilstm_model = tf.keras.models.load_model(
        root / "bilstm_attention.keras",
        custom_objects={"AttentionPooling": AttentionPooling, "LastStepSlice": LastStepSlice},
    )
    xgb_model = XGBRegressor()
    xgb_model.load_model(root / "xgboost_model.json")
    feature_scaler = _read_pickle(root / "feature_scaler.pkl")
    reference_scaler = _read_pickle(root / "reference_scaler.pkl")
    metadata = _read_pickle(root / "metadata.pkl")
    return bilstm_model, xgb_model, feature_scaler, reference_scaler, metadata


def _predict_group(
    engineered: pd.DataFrame,
    normalized: np.ndarray,
    metadata: dict,
    bilstm_model,
    xgb_model,
    group_value: str,
    group_indices: list[int],
    lambda_weight: float,
    sequence_context_steps: int,
    sequence_context_columns: list[str],
) -> pd.DataFrame:
    window = metadata["window_size"]
    if len(group_indices) < window:
        return pd.DataFrame()

    sequences: list[np.ndarray] = []
    tabular_rows: list[np.ndarray] = []
    timestamps: list[pd.Timestamp] = []
    groups: list[str] = []
    entity_col = metadata.get("entity_col")

    for end_local in range(window - 1, len(group_indices)):
        seq_indices = group_indices[end_local - window + 1 : end_local + 1]
        end_index = group_indices[end_local]
        sequences.append(normalized[seq_indices])
        tabular_rows.append(
            build_tabular_context_vector(
                normalized_features=normalized,
                sequence_indices=seq_indices,
                feature_names=metadata["feature_names"],
                sequence_context_steps=sequence_context_steps,
                sequence_context_columns=sequence_context_columns,
            ),
        )
        timestamps.append(engineered.iloc[end_index][metadata["timestamp_col"]])
        groups.append(str(group_value))

    seq_batch = np.asarray(sequences, dtype=np.float32)
    tab_batch = np.asarray(tabular_rows, dtype=np.float32)
    dl_pred = bilstm_model.predict(seq_batch, batch_size=1024, verbose=0).reshape(-1)
    gb_pred = xgb_model.predict(tab_batch).reshape(-1)
    blended = (lambda_weight * dl_pred) + ((1.0 - lambda_weight) * gb_pred)
    class_ids = classify(
        blended,
        metadata["thresholds"],
        np.asarray(groups, dtype=object) if metadata.get("groupwise_bins") else None,
    )

    forecast_timestamps = [
        timestamp + pd.Timedelta(minutes=metadata["sample_interval_minutes"] * metadata["horizon_steps"])
        for timestamp in timestamps
    ]
    forecast_window_end_timestamps = [
        timestamp
        + pd.Timedelta(
            minutes=metadata["sample_interval_minutes"]
            * (metadata["horizon_steps"] + metadata.get("future_average_steps", 1) - 1),
        )
        for timestamp in timestamps
    ]

    payload = {
        "timestamp": timestamps,
        "forecast_timestamp": forecast_timestamps,
        "forecast_window_end_timestamp": forecast_window_end_timestamps,
        "prediction_score": blended.astype(float),
        "prediction_class": class_ids.astype(int),
    }
    if entity_col:
        payload[entity_col] = [engineered.iloc[group_indices[window - 1 + offset]][entity_col] for offset in range(len(timestamps))]
    return pd.DataFrame(payload)


def predict_live_file(artifacts_dir: str | Path, live_data_path: str | Path) -> pd.DataFrame:
    bilstm_model, xgb_model, feature_scaler, reference_scaler, metadata = load_artifacts(artifacts_dir)
    df = clean_dataframe(load_dataframe(live_data_path), metadata["timestamp_col"], metadata.get("entity_col"))
    reference_score = compute_reference_congestion_score(
        df,
        metadata["base_feature_names"],
        reference_scaler,
    )
    engineered = df.copy()
    engineered["reference_congestion_score"] = reference_score
    engineered = add_engineered_features(
        df=engineered,
        timestamp_col=metadata["timestamp_col"],
        entity_col=metadata.get("entity_col"),
        base_columns=metadata["base_feature_names"],
        peak_hours=metadata["peak_hours"],
        reference_score=reference_score,
        use_peak_indicator="peak_indicator" in metadata["feature_names"],
        use_congestion_transition="congestion_transition" in metadata["feature_names"],
        use_temporal_features=any(
            name in metadata["feature_names"]
            for name in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]
        ),
        use_lag_features=any(
            name in metadata["feature_names"]
            for name in ["lag_1", "lag_2", "lag_3", "lag_24", "rolling_mean_3", "rolling_mean_24"]
        ),
        use_entity_one_hot=any(
            name.startswith(f"{metadata['entity_col'].lower()}_")
            for name in metadata["feature_names"]
        )
        if metadata.get("entity_col")
        else False,
    )
    for feature_name in metadata["feature_names"]:
        if feature_name not in engineered.columns:
            engineered[feature_name] = 0.0

    normalized = feature_scaler.transform(engineered[metadata["feature_names"]].astype(float))
    lambda_weight = metadata["ensemble_weight"]
    entity_col = metadata.get("entity_col")
    sequence_context_steps = int(metadata.get("xgboost_sequence_context_steps", 0))
    sequence_context_columns = metadata.get("xgboost_sequence_context_columns", metadata["base_feature_names"])
    prediction_frames: list[pd.DataFrame] = []

    group_values = (
        engineered[entity_col].astype(str).unique().tolist()
        if entity_col and entity_col in engineered.columns
        else ["global"]
    )

    for group_value in group_values:
        if entity_col and entity_col in engineered.columns:
            group_df = engineered[engineered[entity_col].astype(str) == group_value]
        else:
            group_df = engineered

        group_indices = group_df.index.to_list()
        prediction_frames.append(
            _predict_group(
                engineered=engineered,
                normalized=normalized,
                metadata=metadata,
                bilstm_model=bilstm_model,
                xgb_model=xgb_model,
                group_value=str(group_value),
                group_indices=group_indices,
                lambda_weight=lambda_weight,
                sequence_context_steps=sequence_context_steps,
                sequence_context_columns=sequence_context_columns,
            ),
        )

    result = pd.concat([frame for frame in prediction_frames if not frame.empty], ignore_index=True)
    order = (
        [entity_col, "timestamp", "forecast_timestamp", "prediction_score", "prediction_class"]
        if entity_col
        else ["timestamp", "forecast_timestamp", "prediction_score", "prediction_class"]
    )
    if "forecast_window_end_timestamp" in result.columns:
        order = (
            [entity_col, "timestamp", "forecast_timestamp", "forecast_window_end_timestamp", "prediction_score", "prediction_class"]
            if entity_col
            else ["timestamp", "forecast_timestamp", "forecast_window_end_timestamp", "prediction_score", "prediction_class"]
        )
    return result[order]
