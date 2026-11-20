"""Builds model features from the raw production_records table with
DuckDB SQL: station-path structure (visit count, a one-hot presence flag
per station so the model can discover which stations or combinations
matter on its own rather than being told) and a missingness summary
(itself informative: a part that never reached a late station is
structurally different from one that did and passed). The raw per-station
measurement columns are passed through unchanged; LightGBM splits on
missing values natively, so there is no imputation step to leak
information through.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from .generate_data import N_STATIONS


def build_features(parquet_path: str) -> pd.DataFrame:
    con = duckdb.connect()
    station_flags_sql = ", ".join(
        f"(list_contains(string_split(station_path, ','), '{st}')) AS station_{st}_visited"
        for st in range(N_STATIONS)
    )
    measurement_cols = con.execute(
        f"SELECT * FROM read_parquet('{parquet_path}') LIMIT 0"
    ).df().columns
    measurement_cols = [c for c in measurement_cols if c.startswith("S") and "_M" in c]
    measurement_select = ", ".join(f't."{c}"' for c in measurement_cols)

    query = f"""
        SELECT
            t.Id,
            t.station_path,
            len(string_split(t.station_path, ',')) AS station_count,
            (
                list_sum(
                    list_transform([{measurement_select}], lambda x: CASE WHEN x IS NULL THEN 1 ELSE 0 END)
                ) * 1.0 / {len(measurement_cols)}
            ) AS missing_fraction,
            {station_flags_sql},
            {measurement_select},
            t.Response
        FROM read_parquet('{parquet_path}') t
    """
    df = con.execute(query).df()
    con.close()
    return df


def time_ordered_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    """Splits by Id (the dataset's own generation order, standing in for
    time) so no row in train or val was generated after any row in test;
    the same no-future-information-leak discipline a real time-ordered
    production split enforces."""
    df_sorted = df.sort_values("Id").reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (
        df_sorted.iloc[:n_train],
        df_sorted.iloc[n_train : n_train + n_val],
        df_sorted.iloc[n_train + n_val :],
    )


def cold_start_test_mask(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    """True for test rows whose exact station_path combination never
    appears anywhere in train."""
    train_paths = set(train_df["station_path"])
    return ~test_df["station_path"].isin(train_paths)
