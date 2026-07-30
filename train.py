from __future__ import annotations

import argparse
from pathlib import Path

from traffic_hybrid.config import load_config
from traffic_hybrid.training import train_hybrid_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the paper-aligned hybrid traffic model.")
    parser.add_argument(
        "--config",
        default="configs/paper_baseline.yaml",
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    report = train_hybrid_model(config)
    print(f"Validation accuracy: {report['validation']['accuracy']:.3f}%")
    print(f"Metrics saved to: {Path(config.output_dir) / 'metrics.json'}")


if __name__ == "__main__":
    main()
