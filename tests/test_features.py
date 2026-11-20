import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk_model.features import build_features, cold_start_test_mask, time_ordered_split
from risk_model.generate_data import N_STATIONS, generate


def _sample_parquet(n=800, seed=10):
    df = generate(n, seed=seed)
    fd, path = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    df.to_parquet(path, index=False)
    return path


def test_build_features_adds_station_count_and_missing_fraction():
    path = _sample_parquet()
    try:
        feats = build_features(path)
        assert "station_count" in feats.columns
        assert "missing_fraction" in feats.columns
        assert (feats["missing_fraction"] >= 0).all() and (feats["missing_fraction"] <= 1).all()
    finally:
        os.remove(path)


def test_build_features_adds_one_station_flag_per_station():
    path = _sample_parquet()
    try:
        feats = build_features(path)
        flag_cols = [c for c in feats.columns if c.startswith("station_") and c.endswith("_visited")]
        assert len(flag_cols) == N_STATIONS
    finally:
        os.remove(path)


def test_station_flag_matches_station_path_membership():
    path = _sample_parquet()
    try:
        feats = build_features(path)
        for _, row in feats.head(30).iterrows():
            visited = set(int(s) for s in row["station_path"].split(","))
            for st in range(N_STATIONS):
                assert bool(row[f"station_{st}_visited"]) == (st in visited)
    finally:
        os.remove(path)


def test_time_ordered_split_preserves_order_and_covers_all_rows():
    df = generate(1000, seed=11)
    train, val, test = time_ordered_split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert train["Id"].max() < val["Id"].min()
    assert val["Id"].max() < test["Id"].min()


def test_cold_start_mask_flags_only_unseen_paths():
    df = generate(2000, seed=12)
    train, val, test = time_ordered_split(df)
    mask = cold_start_test_mask(train, test)
    train_paths = set(train["station_path"])
    for is_cold, path in zip(mask, test["station_path"]):
        assert is_cold == (path not in train_paths)
