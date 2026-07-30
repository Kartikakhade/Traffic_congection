from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def derive_thresholds(
    values: np.ndarray,
    num_classes: int = 3,
    groups: np.ndarray | None = None,
) -> dict[str, list[float]] | list[float]:
    quantiles = np.linspace(0, 1, num_classes + 1)[1:-1]

    if groups is None:
        return [float(threshold) for threshold in np.quantile(values, quantiles).tolist()]

    thresholds: dict[str, list[float]] = {}
    for group in sorted({str(group_value) for group_value in groups.tolist()}):
        mask = groups.astype(str) == group
        thresholds[group] = [float(threshold) for threshold in np.quantile(values[mask], quantiles).tolist()]
    return thresholds


def classify(
    values: np.ndarray,
    thresholds: dict[str, list[float]] | Iterable[float],
    groups: np.ndarray | None = None,
) -> np.ndarray:
    if isinstance(thresholds, dict):
        if groups is None:
            raise ValueError("Group-aware thresholds require group labels during classification.")

        classes = []
        for value, group in zip(values, groups.astype(str)):
            classes.append(np.digitize(value, bins=thresholds[group], right=False))
        return np.asarray(classes)

    return np.digitize(values, bins=list(thresholds), right=False)


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: dict[str, list[float]] | list[float],
    groups: np.ndarray | None = None,
) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean(np.square(y_true - y_pred))))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    accuracy = float(
        np.mean(classify(y_true, thresholds, groups) == classify(y_pred, thresholds, groups)) * 100.0,
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "accuracy": accuracy,
    }


def find_best_ensemble_weight(
    y_true: np.ndarray,
    dl_pred: np.ndarray,
    gb_pred: np.ndarray,
    thresholds: dict[str, list[float]] | list[float],
    groups: np.ndarray | None,
    weight_min: float,
    weight_max: float,
    step: float,
    optimization_metric: str = "rmse",
) -> tuple[float, dict[str, float]]:
    best_weight = weight_min
    best_metrics: dict[str, float] | None = None
    optimize_accuracy = optimization_metric.lower() == "accuracy"
    best_score = float("-inf") if optimize_accuracy else float("inf")

    for weight in np.arange(weight_min, weight_max + (step / 2.0), step):
        blended = (weight * dl_pred) + ((1.0 - weight) * gb_pred)
        metrics = compute_metrics(y_true, blended, thresholds, groups)
        candidate_score = metrics["accuracy"] if optimize_accuracy else metrics["rmse"]
        is_better = candidate_score > best_score if optimize_accuracy else candidate_score < best_score
        if is_better:
            best_score = candidate_score
            best_weight = float(weight)
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("Unable to determine an ensemble weight.")

    return best_weight, best_metrics
