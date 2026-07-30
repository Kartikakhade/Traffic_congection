from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from traffic_hybrid.config import TrainingConfig
from traffic_hybrid.features import (
    add_engineered_features,
    clean_dataframe,
    compute_reference_congestion_score,
    fit_reference_scaler,
    infer_base_feature_columns,
    load_dataframe,
    select_feature_columns,
)
from traffic_hybrid.metrics import derive_thresholds


@dataclass
class PreparedDatasets:
    seq_train: np.ndarray
    seq_val: np.ndarray
    tab_train: np.ndarray
    tab_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    groups_train: np.ndarray
    groups_val: np.ndarray
    timestamps_train: list[str]
    timestamps_val: list[str]
    thresholds: dict[str, list[float]] | list[float]
    feature_names: list[str]
    base_feature_names: list[str]
    feature_scaler: Any
    reference_scaler: Any
    metadata: dict[str, Any]


def _build_feature_scaler(name: str) -> Any:
    normalized = name.lower()
    if normalized == "minmax":
        return MinMaxScaler()
    if normalized == "standard":
        return StandardScaler()
    if normalized == "robust":
        return RobustScaler()
    raise ValueError(f"Unsupported normalization strategy: {name}")


def _build_target_series(
    cfg: TrainingConfig,
    df: pd.DataFrame,
    base_columns: list[str],
    reference_scaler,
):
    entity_col = cfg.data.entity_col
    mode = cfg.data.target_mode
    average_steps = max(1, cfg.data.future_average_steps)

    def build_future_average(series: pd.Series) -> pd.Series:
        shifted = [
            series.shift(-step)
            for step in range(cfg.data.horizon_steps, cfg.data.horizon_steps + average_steps)
        ]
        return pd.concat(shifted, axis=1).mean(axis=1)

    if entity_col and entity_col in df.columns:
        grouped_target = df.groupby(entity_col, sort=False)
    else:
        grouped_target = None

    if mode == "future_target_column":
        if not cfg.data.target_column:
            raise ValueError("`target_column` must be set when using `future_target_column`.")
        if cfg.data.target_column not in df.columns:
            raise ValueError(f"Target column `{cfg.data.target_column}` not found in dataset.")
        base_target = df[cfg.data.target_column].astype(float)
        if average_steps == 1:
            if grouped_target is not None:
                future = grouped_target[cfg.data.target_column].shift(-cfg.data.horizon_steps).astype(float)
            else:
                future = base_target.shift(-cfg.data.horizon_steps).astype(float)
            return future, reference_scaler

        if grouped_target is not None:
            future = grouped_target[cfg.data.target_column].transform(build_future_average).astype(float)
        else:
            future = build_future_average(base_target).astype(float)
        return future, reference_scaler

    if mode == "future_speed":
        if "speed" not in df.columns:
            raise ValueError("`future_speed` mode requires a `speed` column.")
        base_target = df["speed"].astype(float)
        if average_steps == 1:
            if grouped_target is not None:
                future = grouped_target["speed"].shift(-cfg.data.horizon_steps).astype(float)
            else:
                future = base_target.shift(-cfg.data.horizon_steps).astype(float)
            return future, reference_scaler

        if grouped_target is not None:
            future = grouped_target["speed"].transform(build_future_average).astype(float)
        else:
            future = build_future_average(base_target).astype(float)
        return future, reference_scaler

    if mode == "future_congestion_score":
        score = compute_reference_congestion_score(df, base_columns, reference_scaler)
        temp_df = df.copy()
        temp_df["reference_congestion_score"] = score
        if average_steps == 1:
            if grouped_target is not None:
                future = temp_df.groupby(entity_col, sort=False)["reference_congestion_score"].shift(-cfg.data.horizon_steps)
            else:
                future = score.shift(-cfg.data.horizon_steps)
            return future.astype(float), reference_scaler

        if grouped_target is not None:
            future = temp_df.groupby(entity_col, sort=False)["reference_congestion_score"].transform(build_future_average)
        else:
            future = build_future_average(score)
        return future.astype(float), reference_scaler

    raise ValueError(f"Unsupported target mode: {mode}")


def _resolve_split_time(df: pd.DataFrame, cfg: TrainingConfig) -> pd.Timestamp:
    unique_times = sorted(pd.to_datetime(df[cfg.data.timestamp_col]).unique().tolist())
    split_index = max(1, int(len(unique_times) * cfg.data.split_ratio)) - 1
    return pd.Timestamp(unique_times[split_index])


def build_tabular_context_vector(
    normalized_features: np.ndarray,
    sequence_indices: list[int],
    feature_names: list[str],
    sequence_context_steps: int,
    sequence_context_columns: list[str],
) -> np.ndarray:
    tab_vector = normalized_features[sequence_indices[-1]]
    if sequence_context_steps <= 0:
        return tab_vector

    sequence_context_indices = [feature_names.index(column) for column in sequence_context_columns if column in feature_names]
    if not sequence_context_indices:
        return tab_vector

    context_indices = sequence_indices[-sequence_context_steps:]
    context_vector = normalized_features[np.asarray(context_indices)[:, None], sequence_context_indices].reshape(-1)
    return np.concatenate([tab_vector, context_vector.astype(np.float32)], axis=0)


def _build_group_samples(
    df: pd.DataFrame,
    normalized_features: np.ndarray,
    target_values: np.ndarray,
    cfg: TrainingConfig,
    feature_names: list[str],
    sequence_context_columns: list[str],
):
    entity_col = cfg.data.entity_col
    sequence_context_steps = max(0, cfg.xgboost.sequence_context_steps)
    group_values = (
        df[entity_col].astype(str).unique().tolist()
        if entity_col and entity_col in df.columns
        else ["global"]
    )

    samples: list[dict[str, Any]] = []
    for group_value in group_values:
        if entity_col and entity_col in df.columns:
            group_df = df[df[entity_col].astype(str) == group_value]
        else:
            group_df = df

        group_indices = group_df.index.to_list()
        if len(group_indices) < cfg.data.window_size + cfg.data.horizon_steps:
            continue

        first_target_local = cfg.data.window_size - 1
        for target_local in range(first_target_local, len(group_indices)):
            target_global_index = group_indices[target_local]
            input_end_local = target_local
            input_start_local = input_end_local - cfg.data.window_size + 1
            sequence_indices = group_indices[input_start_local : input_end_local + 1]
            tab_vector = build_tabular_context_vector(
                normalized_features=normalized_features,
                sequence_indices=sequence_indices,
                feature_names=feature_names,
                sequence_context_steps=sequence_context_steps,
                sequence_context_columns=sequence_context_columns,
            )

            samples.append(
                {
                    "seq": normalized_features[sequence_indices],
                    "tab": tab_vector,
                    "y": float(target_values[target_global_index]),
                    "timestamp": df.iloc[target_global_index][cfg.data.timestamp_col],
                    "group": str(group_value),
                },
            )

    return samples


def _stack_samples(samples: list[dict[str, Any]]):
    seq = np.asarray([sample["seq"] for sample in samples], dtype=np.float32)
    tab = np.asarray([sample["tab"] for sample in samples], dtype=np.float32)
    y = np.asarray([sample["y"] for sample in samples], dtype=np.float32)
    groups = np.asarray([sample["group"] for sample in samples], dtype=object)
    timestamps = [pd.Timestamp(sample["timestamp"]).strftime("%Y-%m-%d %H:%M:%S") for sample in samples]
    return seq, tab, y, groups, timestamps


def prepare_datasets(cfg: TrainingConfig) -> PreparedDatasets:
    raw_df = load_dataframe(cfg.data.dataset_path)
    df = clean_dataframe(raw_df, cfg.data.timestamp_col, cfg.data.entity_col)
    base_columns = infer_base_feature_columns(df, cfg.data.feature_columns)
    split_time = _resolve_split_time(df, cfg)
    train_mask = df[cfg.data.timestamp_col] <= split_time

    reference_scaler = fit_reference_scaler(df, base_columns, train_mask)
    target_series, reference_scaler = _build_target_series(cfg, df, base_columns, reference_scaler)
    reference_score = (
        target_series
        if cfg.data.target_mode == "future_congestion_score"
        else compute_reference_congestion_score(df, base_columns, reference_scaler)
    )
    engineered_df = df.copy()
    engineered_df["reference_congestion_score"] = reference_score
    engineered_df = add_engineered_features(
        df=engineered_df,
        timestamp_col=cfg.data.timestamp_col,
        entity_col=cfg.data.entity_col,
        base_columns=base_columns,
        peak_hours=cfg.data.peak_hours,
        reference_score=reference_score,
        use_peak_indicator=cfg.preprocessing.use_peak_indicator,
        use_congestion_transition=cfg.preprocessing.use_congestion_transition,
        use_temporal_features=cfg.preprocessing.use_temporal_features,
        use_lag_features=cfg.preprocessing.use_lag_features,
        use_entity_one_hot=cfg.preprocessing.use_entity_one_hot,
    )

    feature_names = select_feature_columns(
        df=engineered_df,
        base_columns=base_columns,
        use_engineered_features=cfg.preprocessing.use_engineered_features,
        use_peak_indicator=cfg.preprocessing.use_peak_indicator,
        use_congestion_transition=cfg.preprocessing.use_congestion_transition,
        use_temporal_features=cfg.preprocessing.use_temporal_features,
        use_lag_features=cfg.preprocessing.use_lag_features,
        use_entity_one_hot=cfg.preprocessing.use_entity_one_hot,
        entity_col=cfg.data.entity_col,
    )
    sequence_context_columns = [
        column
        for column in (cfg.xgboost.sequence_context_columns or base_columns)
        if column in feature_names
    ]

    feature_scaler = _build_feature_scaler(cfg.preprocessing.normalization)
    feature_scaler.fit(engineered_df.loc[train_mask, feature_names].astype(float))
    normalized_features = feature_scaler.transform(engineered_df[feature_names].astype(float))
    target_valid_mask = target_series.notna()
    engineered_df = engineered_df.loc[target_valid_mask].reset_index(drop=True)
    normalized_features = normalized_features[target_valid_mask.to_numpy()]
    target_values = target_series.loc[target_valid_mask].to_numpy(dtype=np.float32)

    all_samples = _build_group_samples(
        df=engineered_df,
        normalized_features=normalized_features,
        target_values=target_values,
        cfg=cfg,
        feature_names=feature_names,
        sequence_context_columns=sequence_context_columns,
    )
    train_samples = [sample for sample in all_samples if pd.Timestamp(sample["timestamp"]) <= split_time]
    val_samples = [sample for sample in all_samples if pd.Timestamp(sample["timestamp"]) > split_time]

    if not train_samples or not val_samples:
        raise ValueError("Temporal split produced an empty train or validation set.")

    seq_train, tab_train, y_train, groups_train, timestamps_train = _stack_samples(train_samples)
    seq_val, tab_val, y_val, groups_val, timestamps_val = _stack_samples(val_samples)

    thresholds = derive_thresholds(
        y_train,
        num_classes=cfg.evaluation.num_classes,
        groups=groups_train if cfg.evaluation.groupwise_bins else None,
    )

    metadata = {
        "timestamp_col": cfg.data.timestamp_col,
        "entity_col": cfg.data.entity_col,
        "window_size": cfg.data.window_size,
        "horizon_steps": cfg.data.horizon_steps,
        "future_average_steps": cfg.data.future_average_steps,
        "peak_hours": cfg.data.peak_hours,
        "feature_names": feature_names,
        "base_feature_names": base_columns,
        "target_mode": cfg.data.target_mode,
        "target_column": cfg.data.target_column,
        "thresholds": thresholds,
        "split_ratio": cfg.data.split_ratio,
        "sample_interval_minutes": cfg.data.sample_interval_minutes,
        "groupwise_bins": cfg.evaluation.groupwise_bins,
        "split_time": split_time.strftime("%Y-%m-%d %H:%M:%S"),
        "xgboost_sequence_context_steps": cfg.xgboost.sequence_context_steps,
        "xgboost_sequence_context_columns": sequence_context_columns,
    }

    return PreparedDatasets(
        seq_train=seq_train,
        seq_val=seq_val,
        tab_train=tab_train,
        tab_val=tab_val,
        y_train=y_train,
        y_val=y_val,
        groups_train=groups_train,
        groups_val=groups_val,
        timestamps_train=timestamps_train,
        timestamps_val=timestamps_val,
        thresholds=thresholds,
        feature_names=feature_names,
        base_feature_names=base_columns,
        feature_scaler=feature_scaler,
        reference_scaler=reference_scaler,
        metadata=metadata,
    )
