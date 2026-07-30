from __future__ import annotations

import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DateTime"] = pd.to_datetime(out["DateTime"])
    out["Junction"] = out["Junction"].astype(str)
    out = out.sort_values(["Junction", "DateTime"]).reset_index(drop=True)

    grp = out.groupby("Junction", sort=False)["Vehicles"]
    out["lag_1"] = grp.shift(1)
    out["lag_2"] = grp.shift(2)
    out["lag_3"] = grp.shift(3)
    out["lag_24"] = grp.shift(24)
    out["lag_168"] = grp.shift(168)
    out["rolling_mean_3"] = grp.transform(lambda s: s.rolling(3, min_periods=1).mean())
    out["rolling_mean_24"] = grp.transform(lambda s: s.rolling(24, min_periods=1).mean())
    out["rolling_mean_168"] = grp.transform(lambda s: s.rolling(168, min_periods=1).mean())
    out["rolling_std_24"] = grp.transform(lambda s: s.rolling(24, min_periods=2).std())
    out["hour"] = out["DateTime"].dt.hour
    out["dow"] = out["DateTime"].dt.dayofweek
    out["month"] = out["DateTime"].dt.month
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)

    for j in sorted(out["Junction"].unique()):
        out[f"junction_{j}"] = (out["Junction"] == j).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def make_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    out = df.copy()
    out["future_vehicles"] = out.groupby("Junction", sort=False)["Vehicles"].shift(-horizon)
    out = out.dropna(subset=["future_vehicles"]).copy()

    thresholds = {}
    split_time = sorted(out["DateTime"].unique())[int(len(sorted(out["DateTime"].unique())) * 0.8) - 1]
    train_mask = out["DateTime"] <= split_time

    labels = np.zeros(len(out), dtype=int)
    for junction, g in out.groupby("Junction", sort=False):
        junction_mask = out["Junction"] == junction
        vals = out.loc[junction_mask & train_mask, "future_vehicles"].to_numpy()
        q1, q2 = np.quantile(vals, [1 / 3, 2 / 3]).tolist()
        thresholds[junction] = [q1, q2]
        labels[junction_mask] = np.digitize(out.loc[junction_mask, "future_vehicles"], [q1, q2], right=False)

    out["target_class"] = labels
    return out, pd.Timestamp(split_time), thresholds


def main() -> None:
    df = pd.read_csv("data/traffic.csv")
    df = add_features(df)
    df, split_time, thresholds = make_target(df, horizon=1)

    features = [
        "Vehicles",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_24",
        "lag_168",
        "rolling_mean_3",
        "rolling_mean_24",
        "rolling_mean_168",
        "rolling_std_24",
        "hour",
        "dow",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "junction_1",
        "junction_2",
        "junction_3",
        "junction_4",
    ]

    train = df[df["DateTime"] <= split_time].copy()
    val = df[df["DateTime"] > split_time].copy()

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=600,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    model.fit(train[features], train["target_class"])
    pred = model.predict(val[features])
    acc = float((pred == val["target_class"]).mean() * 100.0)
    print(json.dumps({"accuracy": acc, "split_time": str(split_time), "rows_train": len(train), "rows_val": len(val)}, indent=2))


if __name__ == "__main__":
    main()
