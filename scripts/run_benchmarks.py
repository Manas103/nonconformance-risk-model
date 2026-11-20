"""Builds features from the generated parquet file, trains the model and
both baselines, calibrates, measures the cold-start slice, and writes
docs/benchmark_output.json.

    python scripts/run_benchmarks.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk_model.features import build_features
from risk_model.train import run

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "production_records.parquet")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "benchmark_output.json")


def main():
    t0 = time.time()
    print("building features with DuckDB...")
    df = build_features(DATA_PATH)
    print(f"  {df.shape[0]} rows, {df.shape[1]} columns, {time.time() - t0:.1f}s")

    print("training LightGBM + baselines, calibrating, measuring cold-start...")
    report, model = run(df)
    report["elapsed_s"] = time.time() - t0

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
