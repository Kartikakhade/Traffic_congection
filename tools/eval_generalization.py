"""
eval_generalization.py — Zero-Shot Cross-City Evaluation Benchmark.

Evaluates the trained Pan-India hybrid model (artifacts/pan_india_v2) on the
live telemetry observations collected across 11 Indian cities (data/telemetry/).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import tensorflow as tf

from traffic_hybrid.data import build_tabular_context_vector
from traffic_hybrid.features import add_engineered_features
from traffic_hybrid.inference import load_artifacts
from traffic_hybrid.metrics import classify


def load_telemetry_dataframe(telemetry_dir: Path) -> pd.DataFrame:
    dfs = []
    for file_path in sorted(telemetry_dir.glob("city=*/*.parquet")):
        dfs.append(pd.read_parquet(file_path))
    if not dfs:
        raise FileNotFoundError(f"No parquet telemetry files found in {telemetry_dir}")
    df = pd.concat(dfs, ignore_index=True)
    df["DateTime"] = pd.to_datetime(df["timestamp"])
    # Synthetic vehicles proxy from speed ratio if Vehicles col not present
    if "Vehicles" not in df.columns:
        # Density proxy: free_flow / current_speed scaled to typical vehicle range (10-120)
        ratio = (df["free_flow_speed_kmh"] / df["current_speed_kmh"].clip(lower=2.0)).clip(1.0, 10.0)
        df["Vehicles"] = (ratio * 12.0).round().astype(float)
    return df.sort_values(["city", "corridor_name", "DateTime"]).reset_index(drop=True)


def evaluate_cities(artifacts_dir: str | Path, telemetry_dir: str | Path) -> dict:
    artifacts_path = Path(artifacts_dir)
    telemetry_path = Path(telemetry_dir)

    bilstm_model, xgb_model, feature_scaler, reference_scaler, metadata = load_artifacts(artifacts_path)
    lambda_weight = float(metadata.get("ensemble_weight", 0.55))
    window = int(metadata.get("window_size", 12))
    feature_names = metadata["feature_names"]
    thresholds = metadata["thresholds"]

    df = load_telemetry_dataframe(telemetry_path)
    cities = sorted(df["city"].unique())

    city_results = {}
    all_actuals = []
    all_preds = []

    sequence_context_steps = int(metadata.get("xgboost_sequence_context_steps", 0))
    sequence_context_columns = metadata.get("xgboost_sequence_context_columns", metadata["base_feature_names"])

    for city in cities:
        city_df = df[df["city"] == city].copy()
        if len(city_df) < window + 2:
            continue

        city_df["reference_congestion_score"] = city_df["congestion_index"].astype(float)
        # Add engineered features using our feature pipeline
        engineered = add_engineered_features(
            df=city_df,
            timestamp_col="DateTime",
            entity_col="corridor_name",
            base_columns=metadata["base_feature_names"],
            peak_hours=metadata["peak_hours"],
            reference_score=city_df["reference_congestion_score"],
            use_peak_indicator=True,
            use_congestion_transition=True,
            use_temporal_features=True,
            use_lag_features=True,
            use_entity_one_hot=False,
        )

        for col in feature_names:
            if col not in engineered.columns:
                engineered[col] = 0.0

        engineered = engineered.reset_index(drop=True)
        normalized = feature_scaler.transform(engineered[feature_names].astype(float))

        city_seqs = []
        city_tabs = []
        city_acts = []

        for corridor in engineered["corridor_name"].unique():
            corridor_mask = (engineered["corridor_name"] == corridor).values
            corridor_indices = np.where(corridor_mask)[0].tolist()
            if len(corridor_indices) < window:
                continue

            for end_idx in range(window - 1, len(corridor_indices)):
                seq_idx = corridor_indices[end_idx - window + 1 : end_idx + 1]
                target_actual = float(engineered.loc[corridor_indices[end_idx], "congestion_index"])

                city_seqs.append(normalized[seq_idx])
                city_tabs.append(build_tabular_context_vector(
                    normalized_features=normalized,
                    sequence_indices=seq_idx,
                    feature_names=feature_names,
                    sequence_context_steps=sequence_context_steps,
                    sequence_context_columns=sequence_context_columns,
                ))
                city_acts.append(target_actual)

        if not city_seqs:
            continue

        seq_batch = np.asarray(city_seqs, dtype=np.float32)
        tab_batch = np.asarray(city_tabs, dtype=np.float32)

        dl_preds = bilstm_model.predict(seq_batch, batch_size=512, verbose=0).reshape(-1)
        gb_preds = xgb_model.predict(tab_batch).reshape(-1)
        pred_arr = np.clip((lambda_weight * dl_preds) + ((1.0 - lambda_weight) * gb_preds), 0.0, 1.0)
        act_arr = np.array(city_acts, dtype=np.float32)

        mae = float(np.mean(np.abs(pred_arr - act_arr)))
        rmse = float(np.sqrt(np.mean((pred_arr - act_arr) ** 2)))

        act_classes = classify(act_arr, thresholds)
        pred_classes = classify(pred_arr, thresholds)
        accuracy = float(np.mean(act_classes == pred_classes) * 100.0)

        city_results[city] = {
            "records": len(act_arr),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "accuracy_percent": round(accuracy, 2),
            "mean_actual_ci": round(float(np.mean(act_arr)), 3),
            "max_actual_ci": round(float(np.max(act_arr)), 3),
        }
        all_actuals.extend(city_acts)
        all_preds.extend(pred_arr.tolist())

    all_act_arr = np.array(all_actuals)
    all_pred_arr = np.array(all_preds)
    overall_mae = float(np.mean(np.abs(all_pred_arr - all_act_arr)))
    overall_rmse = float(np.sqrt(np.mean((all_pred_arr - all_act_arr) ** 2)))
    overall_act_classes = classify(all_act_arr, thresholds)
    overall_pred_classes = classify(all_pred_arr, thresholds)
    overall_acc = float(np.mean(overall_act_classes == overall_pred_classes) * 100.0)

    report = {
        "overall": {
            "total_records": len(all_actuals),
            "mae": round(overall_mae, 4),
            "rmse": round(overall_rmse, 4),
            "accuracy_percent": round(overall_acc, 2),
            "cities_evaluated": len(city_results),
        },
        "by_city": city_results,
    }

    # Save benchmark report
    report_file = artifacts_path / "generalization_benchmark.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Pan-India Zero-Shot Cross-City Generalization Benchmark",
        "",
        f"- **Model**: `{artifacts_path.name}` (BiLSTM + Attention + XGBoost Hybrid)",
        f"- **Total Multi-City Telemetry Records Evaluated**: `{len(all_actuals)}`",
        f"- **Cities Evaluated**: `{len(city_results)}` across North, South, East, West India",
        f"- **Overall 3-Tier Classification Accuracy**: **`{overall_acc:.2f}%`**",
        f"- **Overall MAE on Congestion Index**: **`{overall_mae:.4f}`** (Target: <= 0.08)",
        f"- **Overall RMSE**: **`{overall_rmse:.4f}`**",
        "",
        "## Performance By City",
        "",
        "| City | Evaluated Windows | Mean Actual CI | Max Peak CI | MAE | RMSE | Accuracy |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for c, res in city_results.items():
        lines.append(
            f"| **{c.capitalize()}** | {res['records']} | {res['mean_actual_ci']:.3f} | "
            f"{res['max_actual_ci']:.3f} | {res['mae']:.4f} | {res['rmse']:.4f} | **{res['accuracy_percent']:.1f}%** |"
        )
    md_file = artifacts_path / "generalization_benchmark.md"
    md_file.write_text("\n".join(lines), encoding="utf-8")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generalization on multi-city telemetry.")
    parser.add_argument("--artifacts", default="artifacts/pan_india_v2", help="Path to artifacts dir")
    parser.add_argument("--telemetry", default="data/telemetry", help="Path to telemetry dir")
    args = parser.parse_args()

    report = evaluate_cities(args.artifacts, args.telemetry)
    print("\nBenchmark Complete!")
    print(f"Overall Accuracy : {report['overall']['accuracy_percent']:.2f}%")
    print(f"Overall MAE      : {report['overall']['mae']:.4f}")
    print(f"Overall RMSE     : {report['overall']['rmse']:.4f}")
    print(f"Benchmark saved to: artifacts/pan_india_v2/generalization_benchmark.md")
