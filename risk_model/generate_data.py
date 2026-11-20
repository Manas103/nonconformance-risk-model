"""Synthetic production-line records, structured like the public Bosch
Production Line Performance dataset but not a copy of it: the real Kaggle
dataset requires an authenticated Kaggle account that accepts the
competition rules, and no such credentials were available in this build
environment (see README "What this is and is not"). This generator
reproduces the dataset's structural character instead: many parts, each
visiting a subset of a fixed set of stations (producing a long-tailed
distribution of unique station-path combinations, many of them rare),
hundreds of per-station measurement columns that are only populated for
the stations a part actually visited (so missingness is structural, not
random), extreme class imbalance, and a time-ordered part index so a
time-based split is meaningful.

Failure probability is a real (if synthetic) function of station path and
measurement values, not pure noise: a handful of station combinations and
a handful of marker measurements are built to carry signal, so a model
that actually learns something can beat chance, and a cold-start split on
unseen station paths has a real reason to generalize worse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_STATIONS = 48
N_MEASUREMENTS_PER_STATION = 16
MIN_STATIONS_VISITED = 4
MAX_STATIONS_VISITED = 14
BASE_FAILURE_RATE = 0.0019  # calibrated so the realized rate lands near 0.58% once the risky-set and marker lifts are applied

# a handful of stations are "risky": visiting several of them together
# raises failure probability well above baseline.
RISKY_STATION_SETS = [
    frozenset({2, 7, 19}),
    frozenset({5, 23, 31, 40}),
    frozenset({11, 12}),
]
RISKY_SET_LIFT = 18.0  # multiplicative odds lift per matched risky set

# a handful of measurement columns (keyed by (station, local_index)) carry
# a real signal: values past a threshold raise failure odds.
MARKER_COLUMNS = [(2, 3), (23, 7), (40, 1), (12, 9)]
MARKER_LIFT = 6.0


def _station_path(rng: np.random.Generator) -> tuple:
    n_visited = rng.integers(MIN_STATIONS_VISITED, MAX_STATIONS_VISITED + 1)
    # a Zipf-like station popularity so some stations are common and the
    # path-combination distribution gets a realistic long tail
    weights = 1.0 / np.arange(1, N_STATIONS + 1)
    weights = weights / weights.sum()
    stations = rng.choice(N_STATIONS, size=n_visited, replace=False, p=weights)
    return tuple(sorted(int(s) for s in stations))


def generate(n_parts: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    col_names = [f"S{st}_M{m}" for st in range(N_STATIONS) for m in range(N_MEASUREMENTS_PER_STATION)]

    for part_idx in range(n_parts):
        path = _station_path(rng)
        row = {"Id": part_idx, "station_path": ",".join(str(s) for s in path)}

        log_odds = np.log(BASE_FAILURE_RATE / (1 - BASE_FAILURE_RATE))
        path_set = set(path)
        for risky in RISKY_STATION_SETS:
            if risky <= path_set:
                log_odds += np.log(RISKY_SET_LIFT)

        marker_hits = 0
        for st in path:
            for m in range(N_MEASUREMENTS_PER_STATION):
                value = rng.normal(0, 1)
                if (st, m) in MARKER_COLUMNS and value > 2.0:
                    marker_hits += 1
                row[f"S{st}_M{m}"] = round(float(value), 4)
        if marker_hits:
            log_odds += np.log(MARKER_LIFT) * marker_hits

        p_fail = 1 / (1 + np.exp(-log_odds))
        row["Response"] = int(rng.random() < p_fail)
        records.append(row)

    df = pd.DataFrame.from_records(records, columns=["Id", "station_path"] + col_names + ["Response"])
    return df


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-parts", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "production_records.parquet"))
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    chunks = []
    n_done = 0
    while n_done < args.n_parts:
        n = min(args.chunk_size, args.n_parts - n_done)
        chunk = generate(n, seed=args.seed + n_done)
        chunk["Id"] = chunk["Id"] + n_done
        chunks.append(chunk)
        n_done += n
        print(f"  generated {n_done}/{args.n_parts}")
    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} rows, {len(df.columns)} columns, failure rate "
          f"{df['Response'].mean():.5f} to {args.out}")


if __name__ == "__main__":
    main()
