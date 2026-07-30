from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path

from xgboost import XGBRegressor

from traffic_hybrid.config import TrainingConfig
from traffic_hybrid.data import prepare_datasets
from traffic_hybrid.metrics import compute_metrics, find_best_ensemble_weight
from traffic_hybrid.model import build_bilstm_attention_model, build_callbacks, set_random_seed


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf8")


def _write_pickle(path: Path, payload) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _write_markdown_report(path: Path, report: dict) -> None:
    lines = [
        "# Training Report",
        "",
        f"- Config: `{report['config_path']}`",
        f"- Paper target accuracy: `{report['paper_target_accuracy']:.1f}%`",
        f"- Selected ensemble weight: `{report['ensemble_weight']:.2f}`",
        "",
        "## Validation Metrics",
        "",
        f"- Accuracy: `{report['validation']['accuracy']:.3f}%`",
        f"- MAE: `{report['validation']['mae']:.4f}`",
        f"- RMSE: `{report['validation']['rmse']:.4f}`",
        f"- MAPE: `{report['validation']['mape']:.4f}%`",
        "",
        "## Paper Target Check",
        "",
        f"- Hit 91.3% target: `{report['validation']['accuracy'] >= report['paper_target_accuracy']}`",
        "",
        "## Notes",
        "",
        "- Reproducing the paper metric depends on matching dataset choice, feature schema, interval size, and temporal split.",
    ]
    path.write_text("\n".join(lines), encoding="utf8")


def train_hybrid_model(cfg: TrainingConfig) -> dict:
    output_dir = cfg.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(cfg.model.random_seed)

    datasets = prepare_datasets(cfg)
    bilstm_model = build_bilstm_attention_model(
        cfg,
        input_shape=(datasets.seq_train.shape[1], datasets.seq_train.shape[2]),
    )
    history = bilstm_model.fit(
        datasets.seq_train,
        datasets.y_train,
        validation_data=(datasets.seq_val, datasets.y_val),
        epochs=cfg.model.epochs,
        batch_size=cfg.model.batch_size,
        verbose=2,
        callbacks=build_callbacks(cfg, output_dir),
    )

    dl_train = bilstm_model.predict(datasets.seq_train, verbose=0).reshape(-1)
    dl_val = bilstm_model.predict(datasets.seq_val, verbose=0).reshape(-1)

    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        max_depth=cfg.xgboost.max_depth,
        learning_rate=cfg.xgboost.learning_rate,
        n_estimators=cfg.xgboost.n_estimators,
        subsample=cfg.xgboost.subsample,
        colsample_bytree=cfg.xgboost.colsample_bytree,
        reg_lambda=cfg.xgboost.reg_lambda,
        random_state=cfg.model.random_seed,
        n_jobs=-1,
    )
    xgb_model.fit(datasets.tab_train, datasets.y_train)

    gb_train = xgb_model.predict(datasets.tab_train)
    gb_val = xgb_model.predict(datasets.tab_val)

    ensemble_weight = cfg.ensemble.weight
    validation_metrics = None
    if cfg.ensemble.search_validation_weight:
        ensemble_weight, validation_metrics = find_best_ensemble_weight(
            y_true=datasets.y_val,
            dl_pred=dl_val,
            gb_pred=gb_val,
            thresholds=datasets.thresholds,
            groups=datasets.groups_val,
            weight_min=cfg.ensemble.search_min,
            weight_max=cfg.ensemble.search_max,
            step=cfg.ensemble.search_step,
            optimization_metric=cfg.ensemble.optimization_metric,
        )

    ensemble_train = (ensemble_weight * dl_train) + ((1.0 - ensemble_weight) * gb_train)
    ensemble_val = (ensemble_weight * dl_val) + ((1.0 - ensemble_weight) * gb_val)

    train_metrics = compute_metrics(
        datasets.y_train,
        ensemble_train,
        datasets.thresholds,
        datasets.groups_train,
    )
    validation_metrics = validation_metrics or compute_metrics(
        datasets.y_val,
        ensemble_val,
        datasets.thresholds,
        datasets.groups_val,
    )

    bilstm_model.save(output_dir / "bilstm_attention.keras")
    xgb_model.save_model(output_dir / "xgboost_model.json")
    _write_pickle(output_dir / "feature_scaler.pkl", datasets.feature_scaler)
    _write_pickle(output_dir / "reference_scaler.pkl", datasets.reference_scaler)
    _write_pickle(output_dir / "metadata.pkl", datasets.metadata | {"ensemble_weight": ensemble_weight})

    history_payload = {key: [float(value) for value in values] for key, values in history.history.items()}
    _write_json(output_dir / "training_history.json", history_payload)

    report = {
        "config_path": cfg.config_path,
        "paper_target_accuracy": cfg.evaluation.paper_target_accuracy,
        "ensemble_weight": ensemble_weight,
        "train": train_metrics,
        "validation": validation_metrics,
        "feature_names": datasets.feature_names,
        "base_feature_names": datasets.base_feature_names,
        "thresholds": datasets.thresholds,
        "metadata": datasets.metadata,
        "resolved_config": asdict(cfg),
    }
    _write_json(output_dir / "metrics.json", report)
    _write_markdown_report(output_dir / "training_report.md", report)

    return report
