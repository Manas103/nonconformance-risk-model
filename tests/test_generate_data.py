import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk_model.generate_data import N_MEASUREMENTS_PER_STATION, N_STATIONS, generate


def test_generate_returns_requested_row_count():
    df = generate(500, seed=1)
    assert len(df) == 500


def test_generate_is_deterministic_for_a_fixed_seed():
    df1 = generate(200, seed=5)
    df2 = generate(200, seed=5)
    assert df1.equals(df2)


def test_different_seeds_give_different_data():
    df1 = generate(200, seed=1)
    df2 = generate(200, seed=2)
    assert not df1["Response"].equals(df2["Response"])


def test_measurement_columns_are_null_for_unvisited_stations():
    df = generate(500, seed=2)
    for _, row in df.head(50).iterrows():
        visited = set(int(s) for s in row["station_path"].split(","))
        for st in range(N_STATIONS):
            col = f"S{st}_M0"
            if st in visited:
                assert row[col] == row[col]  # not NaN
            else:
                assert row[col] != row[col]  # is NaN


def test_response_is_binary():
    df = generate(2000, seed=3)
    assert set(df["Response"].unique()) <= {0, 1}


def test_failure_rate_is_low_and_nonzero_at_scale():
    df = generate(20000, seed=4)
    rate = df["Response"].mean()
    assert 0 < rate < 0.05


def test_station_count_matches_station_path_length():
    df = generate(300, seed=6)
    for _, row in df.iterrows():
        n_visited = len(row["station_path"].split(","))
        n_non_null = sum(
            1 for st in range(N_STATIONS) for m in range(N_MEASUREMENTS_PER_STATION)
            if row[f"S{st}_M{m}"] == row[f"S{st}_M{m}"]
        )
        assert n_non_null == n_visited * N_MEASUREMENTS_PER_STATION
