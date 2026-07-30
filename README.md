# Intelligent Traffic Congestion Prediction

This repository accompanies the research paper:

**"Intelligent Traffic Congestion Prediction Using Deep Learning, Historical Learning & Live Traffic Inference"**  
Authors: Abhigyan Kumar Mahato, Kumar Satwik, Upasana Tiwari  
Department of AIT-CSE, Chandigarh University, India  
Paper PDF: `C:\Users\ABHIGYAN\Documents\research_paper.pdf`

The project implements a hybrid congestion prediction framework that combines:
- BiLSTM with attention to model temporal congestion patterns
- XGBoost to capture non-linear feature interactions
- An ensemble that improves robustness and accuracy
- Live traffic inference (TomTom) to provide real-time context

## Key Results (from the paper)

The proposed hybrid framework reports:
- Accuracy: `91.3%`
- MAE: `2.9`
- RMSE: `4.3`
- MAPE: `7.2%`

These results are attributed to attention focus on critical temporal patterns, ensemble learning to reduce variance, and live traffic integration for responsiveness.

## What This Repo Provides

- Paper-aligned hybrid model training
- Offline evaluation with temporal splits
- Online/live inference workflow
- Interactive map UI for live traffic + forecast
- Ready-to-run scripts and configs

## Architecture (High Level)

1. **Historical learning**  
   BiLSTM + attention learns temporal dependencies and congestion buildup patterns from historical traffic sequences.

2. **Feature-driven learning**  
   XGBoost learns complementary non-linear patterns from engineered traffic features.

3. **Ensemble fusion**  
   Weighted blending of BiLSTM and XGBoost predictions for improved stability.

4. **Live inference**  
   TomTom live traffic flow is queried at the clicked location, while the model forecast is shown for the nearest configured junction.

## Dataset

The paper describes a Kaggle traffic prediction dataset with time-stamped observations including speed, volume, and occupancy.  
In this repo we provide an adapted pipeline for `data/traffic.csv` with columns:
- `DateTime`, `Junction`, `Vehicles`, `ID`

## Quickstart

Install dependencies:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Train the model:

```bash
python train.py --config configs/tuned_high_accuracy.yaml
```

Run offline inference:

```bash
python predict_live.py --artifacts artifacts/tuned_high_accuracy --input data/traffic.csv --output artifacts/tuned_high_accuracy/live_predictions.csv
```

## Interactive Web UI (Map Based)

Start the UI:

```bash
python webapp.py --artifacts artifacts/tuned_high_accuracy --input data/traffic.csv
```

Open:

```text
http://127.0.0.1:5000
```

### Live Traffic (TomTom)

Set your TomTom key:

```bash
TOMTOM_API_KEY=your_tomtom_key
```

Notes:
- Live traffic uses TomTom Flow Segment Data at the clicked point.
- Forecast only appears when the click is near a real configured junction.
- Update real junction coordinates in `config/junction_locations.json`.

## Repository Structure

- `traffic_hybrid/` — model, data prep, inference, web backend
- `configs/` — paper baseline + tuned configs
- `web/` — frontend UI
- `webapp.py` — launch the UI
- `data/traffic.csv` — dataset
- `artifacts/` — saved model outputs

## Reproducibility

The paper reports `91.3%` accuracy under its evaluation protocol and dataset.  
This repo provides a strong reproduction pipeline and a tuned configuration. Actual results may vary with dataset, split, and live traffic availability.

## Citation

If you use this work, please cite the paper:

```text
Mahato, A. K., Satwik, K., & Tiwari, U. (2025).
Intelligent Traffic Congestion Prediction Using Deep Learning, Historical Learning & Live Traffic Inference.
```

## Contact

Corresponding Author: Abhigyan Kumar Mahato  
Email: mastermindagaming@gmail.com
