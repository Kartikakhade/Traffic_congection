# Intelligent Traffic Congestion Prediction — Pan-India

A **general-purpose hybrid AI framework** for traffic congestion prediction across Indian cities.
The system combines BiLSTM + Attention, XGBoost, and live TomTom traffic data into a unified ensemble that can be trained on historical CSV data or real-time multi-city telemetry — without being tied to any single city or junction set.

---

## Key Results (from the paper)

| Metric | Value |
|:---|:---|
| Accuracy | `91.3%` |
| MAE | `2.9` |
| RMSE | `4.3` |
| MAPE | `7.2%` |

> Results achieved on the original single-city dataset. Cross-city generalization targets `≥ 85%` accuracy using the `pan_india_v2` config.

---

## What's New (Pan-India Generalization)

- **Universal Congestion Index (CI)**: Replaces the old city-specific congestion score with a normalized, dimensionless target `CI = 1 - speed/free_flow_speed` (range `[0, 1]`), enabling cross-city comparison across any road type.
- **Multi-City Telemetry Pipeline**: Live probe data harvester (`tools/harvest_telemetry.py`) collects TomTom flow data across 11+ Indian cities and stores it as partitioned Parquet files.
- **Indian Context Features**: National and state-specific holiday detection, long weekend identification, and OpenWeatherMap weather enrichment (precipitation, visibility) to account for Indian traffic variance drivers.
- **Spatial Road Attributes**: Dynamic OSM-based road attribute resolution (`spatial.py`) — road class, lane count, speed limit — for any coordinate, without requiring a pre-existing junction list.
- **Zero-Shot City Evaluation**: `tools/eval_generalization.py` benchmarks a trained model against telemetry from unseen cities.

---

## Architecture

```
┌─────────────────────────────────────────┐
│         Data Sources                    │
│  CSV / Parquet telemetry / TomTom API   │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│  Feature Engineering (features.py)      │
│  Lag, rolling stats, cyclic time,       │
│  peak indicator, weather, holidays      │
└────────┬──────────────┬─────────────────┘
         │              │
┌────────▼──────┐  ┌────▼──────────────────┐
│ BiLSTM +      │  │  XGBoost Regressor     │
│ Attention     │  │  (tabular features)    │
│ (sequences)   │  │                        │
└────────┬──────┘  └────┬──────────────────┘
         │              │
┌────────▼──────────────▼─────────────────┐
│  Dynamic Ensemble  (lambda grid search) │
│  Score = λ·BiLSTM + (1-λ)·XGBoost      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│  3-Class Congestion Output              │
│  Low / Medium / High  (quantile bins)   │
└─────────────────────────────────────────┘
```

---

## Repository Structure

```
traffic_congession/
├── traffic_hybrid/           # Core ML library
│   ├── config.py             # Typed config dataclasses
│   ├── data.py               # Dataset prep, sliding windows, train/val split
│   ├── features.py           # Feature engineering (lags, rolling, cyclic, peak)
│   ├── model.py              # BiLSTM + AttentionPooling Keras model
│   ├── training.py           # End-to-end training orchestrator
│   ├── inference.py          # Batch & live inference engine
│   ├── metrics.py            # MAE, RMSE, MAPE, accuracy, ensemble weight search
│   ├── web_ui.py             # Flask backend (TomTom proxy, junction matching)
│   ├── schemas.py            # Universal RoadSegmentObservation schema & CI formula
│   ├── spatial.py            # OSM road attribute resolver (osmnx + fallback)
│   └── context.py            # Indian holidays (state-level) & weather client (OWM)
│
├── configs/                  # YAML training configurations
│   ├── pan_india_v2.yaml          # General-purpose pan-India config (CI target)
│   ├── telemetry_multi_city.yaml  # Config for Parquet telemetry input
│   ├── tuned_high_accuracy.yaml   # Original single-city tuned config
│   ├── upgraded.yaml
│   └── paper_baseline.yaml
│
├── config/                   # Spatial configuration
│   ├── india_corridors.json  # 11+ city corridor anchor points for telemetry harvest
│   └── junction_locations.json    # Legacy single-city junction coordinates (web UI)
│
├── tools/                    # Standalone utility scripts
│   ├── harvest_telemetry.py        # Multi-city live probe data collector
│   ├── prepare_multi_city_dataset.py  # Merge Parquet telemetry into training data
│   └── eval_generalization.py      # Zero-shot cross-city benchmark
│
├── web/                      # Frontend (Leaflet map UI)
├── data/                     # Training data (traffic.csv + telemetry/)
├── artifacts/                # Saved model outputs
├── logs/                     # Harvest & training logs
├── train.py                  # Training entry point
├── webapp.py                 # Web UI entry point
└── predict_live.py           # Batch offline inference entry point
```

---

## Quickstart

### 1. Install Dependencies

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Train the Model

**Option A — Legacy CSV (single-city compatible):**
```bash
python train.py --config configs/tuned_high_accuracy.yaml
```

**Option B — Pan-India generalized config (recommended):**
```bash
python train.py --config configs/pan_india_v2.yaml
```

**Option C — Multi-city Parquet telemetry:**
```bash
python train.py --config configs/telemetry_multi_city.yaml
```

### 3. Offline Batch Inference

```bash
python predict_live.py \
  --artifacts artifacts/pan_india_v2 \
  --input data/traffic.csv \
  --output artifacts/pan_india_v2/live_predictions.csv
```

### 4. Interactive Web UI (Map Based)

```bash
python webapp.py --artifacts artifacts/pan_india_v2 --input data/traffic.csv
```

Open **http://127.0.0.1:5000** — click any point on the map to get live TomTom traffic data and an ML congestion forecast.

---

## Multi-City Telemetry Pipeline

Collect live probe data across Indian cities using TomTom and (optionally) OpenWeatherMap:

```bash
# Set API keys
set TOMTOM_API_KEY=your_tomtom_key
set OWM_API_KEY=your_owm_key        # Optional — enables weather enrichment

# Run a single harvest pass (all corridors in config/india_corridors.json)
python tools/harvest_telemetry.py --once

# Run on a recurring 15-minute schedule
python tools/harvest_telemetry.py --interval 15

# Custom output directory
python tools/harvest_telemetry.py --once --output data/telemetry
```

Collected records are saved as partitioned Parquet files:
`data/telemetry/city=<city>/<YYYY-MM-DD>.parquet`

Each record includes: live speed, free-flow speed, Congestion Index, travel times, weather observations, and Indian holiday flags.

---

## Cross-City Generalization Evaluation

After collecting telemetry and training with `pan_india_v2`, run a zero-shot evaluation across all cities:

```bash
python tools/eval_generalization.py \
  --artifacts artifacts/pan_india_v2 \
  --telemetry data/telemetry
```

---

## Indian Context Features

The system incorporates India-specific signal sources that significantly impact traffic patterns:

| Feature | Source | Notes |
|:---|:---|:---|
| National holidays | `holidays` library | Republic Day, Independence Day, Gandhi Jayanti, etc. |
| State holidays | `holidays` library (state subdivisions) | 28 states + UTs supported |
| Long weekends | Derived | 3+ day breaks adjacent to holidays |
| Rain intensity | OpenWeatherMap (OWM) | mm/hr — IMD heavy rain threshold at 7.5 mm/hr |
| Visibility | OpenWeatherMap | Fog/smog detection (< 1000 m threshold) |

Cities supported for state-level holiday context:
Bengaluru · Mumbai · Pune · Delhi/NCR · Hyderabad · Chennai · Kolkata · Ahmedabad · Jaipur · Lucknow · Kochi

---

## API Keys

| Variable | Required | Purpose |
|:---|:---|:---|
| `TOMTOM_API_KEY` | Required (live UI + telemetry harvest) | Traffic flow speed & delay data |
| `OWM_API_KEY` | Optional | Weather enrichment (rain, visibility) |

Both integrations degrade gracefully when keys are absent — the pipeline runs without live enrichment.

---

## Training Artifacts

After training, the following are saved to `artifacts/<run_name>/`:

| File | Description |
|:---|:---|
| `bilstm_attention.keras` | Trained Keras BiLSTM model |
| `xgboost_model.json` | Trained XGBoost model |
| `feature_scaler.pkl` | Fitted MinMaxScaler |
| `reference_scaler.pkl` | Target scaler reference |
| `metadata.pkl` | Feature names, thresholds, ensemble weight, window sizes |
| `metrics.json` | Accuracy, MAE, RMSE, MAPE |
| `training_report.md` | Human-readable performance summary |

---

## Dataset

**Legacy (single-city):** `data/traffic.csv`
Columns: `DateTime`, `Junction`, `Vehicles`, `ID`

**Multi-city telemetry:** `data/telemetry/city=<name>/<date>.parquet`
Columns: `timestamp`, `city`, `corridor_name`, `lat`, `lng`, `road_class`, `current_speed_kmh`, `free_flow_speed_kmh`, `congestion_index`, `rain_mm_per_hr`, `visibility_m`, `is_indian_holiday`, …

The `pan_india_v2` config works with both formats. Switch `dataset_path` in the YAML to point at either source.

---

## Reproducibility

The paper reports `91.3%` accuracy under its single-city evaluation protocol.
The pan-India generalization target is `≥ 85%` accuracy across unseen cities (set in `pan_india_v2.yaml`).
Actual results vary with dataset coverage, city diversity, and live traffic API availability.

