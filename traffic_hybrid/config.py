from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset_path: str
    timestamp_col: str = "timestamp"
    entity_col: str | None = None
    feature_columns: list[str] = field(
        default_factory=lambda: ["speed", "volume", "occupancy"],
    )
    target_mode: str = "future_congestion_score"
    target_column: str | None = None
    window_size: int = 12
    horizon_steps: int = 3
    future_average_steps: int = 1
    split_ratio: float = 0.8
    sample_interval_minutes: int = 5
    peak_hours: list[int] = field(default_factory=lambda: [7, 8, 9, 17, 18, 19])


@dataclass
class PreprocessingConfig:
    normalization: str = "minmax"
    use_engineered_features: bool = True
    use_peak_indicator: bool = True
    use_congestion_transition: bool = True
    use_temporal_features: bool = True
    use_lag_features: bool = True
    use_entity_one_hot: bool = True


@dataclass
class ModelConfig:
    hidden_units: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.3
    dense_units: int = 64
    head_dropout: float = 0.0
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 40
    loss: str = "mse"
    early_stopping: bool = False
    early_stopping_patience: int = 7
    use_last_step_skip: bool = False
    use_layer_normalization: bool = False
    reduce_lr_on_plateau: bool = False
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 2
    reduce_lr_min_lr: float = 1e-5
    gradient_clipnorm: float | None = None
    random_seed: int = 42


@dataclass
class XGBoostConfig:
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 300
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    sequence_context_steps: int = 0
    sequence_context_columns: list[str] = field(default_factory=list)


@dataclass
class EnsembleConfig:
    weight: float = 0.7
    search_validation_weight: bool = False
    search_min: float = 0.6
    search_max: float = 0.8
    search_step: float = 0.02
    optimization_metric: str = "rmse"


@dataclass
class EvaluationConfig:
    num_classes: int = 3
    bins_strategy: str = "quantile"
    groupwise_bins: bool = False
    paper_target_accuracy: float = 91.3


@dataclass
class TrainingConfig:
    data: DataConfig
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_dir: str = "artifacts/paper_baseline"
    config_path: str | None = None

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


def _load_section(section_cls: Any, values: dict[str, Any] | None) -> Any:
    return section_cls(**(values or {}))


def load_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf8")) or {}
    return TrainingConfig(
        data=_load_section(DataConfig, payload.get("data")),
        preprocessing=_load_section(PreprocessingConfig, payload.get("preprocessing")),
        model=_load_section(ModelConfig, payload.get("model")),
        xgboost=_load_section(XGBoostConfig, payload.get("xgboost")),
        ensemble=_load_section(EnsembleConfig, payload.get("ensemble")),
        evaluation=_load_section(EvaluationConfig, payload.get("evaluation")),
        output_dir=payload.get("output_dir", "artifacts/paper_baseline"),
        config_path=str(config_path),
    )
