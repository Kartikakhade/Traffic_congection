import pandas as pd
from pathlib import Path

root = Path("data/telemetry")
dfs = []
for p in sorted(root.glob("city=*/*.parquet")):
    df_part = pd.read_parquet(p)
    dfs.append(df_part)

if not dfs:
    raise FileNotFoundError("No telemetry parquet files found in data/telemetry")

combined = pd.concat(dfs, ignore_index=True)
combined["timestamp"] = pd.to_datetime(combined["timestamp"])
combined = combined.sort_values(["city", "corridor_name", "timestamp"]).reset_index(drop=True)

out_parquet = Path("data/telemetry_multi_city.parquet")
combined.to_parquet(out_parquet, index=False)

out_csv = Path("data/telemetry_multi_city.csv")
combined.to_csv(out_csv, index=False)

print(f"Combined {len(combined)} records across {combined['city'].nunique()} cities and {combined['corridor_name'].nunique()} corridors.")
print(f"Saved to: {out_parquet.resolve()} ({out_parquet.stat().st_size / 1024:.1f} KB)")
print(f"Saved to: {out_csv.resolve()} ({out_csv.stat().st_size / 1024:.1f} KB)")
