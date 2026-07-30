from __future__ import annotations

import argparse
from pathlib import Path

from traffic_hybrid.inference import predict_live_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling live inference with trained artifacts.")
    parser.add_argument("--artifacts", default="artifacts/paper_baseline", help="Artifact directory.")
    parser.add_argument("--input", required=True, help="Path to live CSV or Parquet data.")
    parser.add_argument("--output", help="Optional output CSV path.")
    args = parser.parse_args()

    print("Running batched live inference. Large files may take around a minute on CPU.")
    predictions = predict_live_file(args.artifacts, args.input)
    print(f"Generated {len(predictions)} predictions.")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"Predictions written to: {output_path}")
    else:
        print(predictions.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
