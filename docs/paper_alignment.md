# Paper Alignment

Paper used:
- `Intelligent Traffic Congestion Prediction Using Deep Learning, Historical Learning & Live Traffic Inference`

Paper-faithful choices implemented in `configs/paper_baseline.yaml`:
- Temporal split for offline training and later-time validation.
- Two-layer `BiLSTM + attention` deep model.
- Parallel `XGBoost` model on engineered traffic features.
- Weighted ensemble with `lambda = 0.7`, which stays inside the paper's `0.6-0.8` range.
- `Adam` optimizer with learning rate `0.001`.
- Dropout regularization.
- Offline training plus rolling-buffer online inference.

Engineered features used because the paper explicitly mentions feature engineering and congestion transitions:
- `peak_indicator`
- `congestion_transition`
- lag features
- rolling traffic statistics
- hourly / weekly seasonal encodings

Required adaptations for your `traffic.csv` dataset:
- The dataset is hourly and multi-junction, not `METR-LA/PeMS` speed-volume-occupancy format.
- Sequence windows are built separately per `Junction`.
- Prediction target is future `Vehicles`.
- Groupwise congestion thresholds are used per junction to make `low / medium / high` congestion states meaningful across very different junction scales.
- Baseline window is `24` hours and horizon is `1` hour because the dataset has strong hourly and daily seasonality.

Optional upgrades implemented in `configs/upgraded.yaml`:
- Early stopping with best-model restoration.
- Gradient clipping for stabler BiLSTM training.
- Validation search for the ensemble weight inside the paper's own `0.6-0.8` range.
- `Huber` loss instead of plain MSE for better robustness to sharp traffic spikes.

Additional accuracy-focused upgrades implemented in `configs/tuned_high_accuracy.yaml`:
- A stronger `BiLSTM + attention` readout that concatenates the attention context with the latest timestep features.
- Layer normalization inside the BiLSTM stack.
- `ReduceLROnPlateau` scheduling so the deep branch can keep improving after the first plateau.
- Extra XGBoost context by appending the recent `Vehicles` history window to the tabular branch.
- Finer ensemble-weight search with `0.01` steps.

Latest strict validation result in this workspace:
- `87.663%` accuracy with [artifacts/tuned_high_accuracy/metrics.json](C:/Users/ABHIGYAN/Documents/New project/artifacts/tuned_high_accuracy/metrics.json)

Important note:
- The code is aligned to the paper and targets the paper's reported `91.3%` accuracy, but that number is only realistically reproducible when the dataset, cleaning, sampling interval, feature schema, and evaluation protocol match the paper setup closely.
