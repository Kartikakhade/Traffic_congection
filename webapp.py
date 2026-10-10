from __future__ import annotations

import argparse

from traffic_hybrid.web_ui import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the interactive traffic prediction web UI.")
    parser.add_argument("--artifacts", default="artifacts/telemetry_multi_city", help="Artifacts directory.")
    parser.add_argument("--input", default="data/telemetry_multi_city.parquet", help="Input traffic dataset.")
    parser.add_argument(
        "--corridors",
        default="config/india_corridors.json",
        help="JSON file with corridor coordinates for the map UI.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind.")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode.")
    args = parser.parse_args()

    app = create_app(
        artifacts_dir=args.artifacts,
        data_path=args.input,
        locations_path=args.corridors,
    )
    print(f"Starting web UI at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
