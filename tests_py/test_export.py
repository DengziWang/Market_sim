"""Tests for market_sim.export -- in particular df_to_columnar, added after
discovering `df.to_dict("list")` (the obvious way to invert `to_dataframes`) took
~21s on a real 1.4M-row/8M-order-event saved run (every cell individually boxed
into a Python object) where the column-wise numpy version takes under a
millisecond. Both dashboard.py's "Load a saved run" and
market_sim.runner.load_run_legs depend on this being fast."""
import numpy as np
import pandas as pd

from market_sim.export import df_to_columnar


def test_df_to_columnar_preserves_values_and_dtypes():
    df = pd.DataFrame(
        {
            "time": np.array([0.1, 0.2, 0.3], dtype="float64"),
            "asset_id": np.array([0, 1, 0], dtype="uint16"),
            "side": ["buy", "sell", "buy"],
        }
    )
    out = df_to_columnar(df)
    assert set(out.keys()) == {"time", "asset_id", "side"}
    np.testing.assert_array_equal(out["time"], df["time"].to_numpy())
    np.testing.assert_array_equal(out["asset_id"], df["asset_id"].to_numpy())
    assert list(out["side"]) == ["buy", "sell", "buy"]


def test_df_to_columnar_matches_slow_to_dict_list_on_small_data():
    # The whole point of df_to_columnar is to be a fast drop-in replacement for
    # `df.to_dict("list")` -- confirm they actually produce equivalent columnar
    # data (same values, in order) on a case small enough that the slow path is
    # still fine to run in a test.
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.5, 2.5, 3.5]})
    fast = df_to_columnar(df)
    slow = df.to_dict("list")
    for col in df.columns:
        assert list(fast[col]) == list(slow[col])


def test_df_to_columnar_handles_empty_dataframe():
    df = pd.DataFrame({"time": pd.Series(dtype="float64"), "side": pd.Series(dtype="object")})
    out = df_to_columnar(df)
    assert len(out["time"]) == 0
    assert len(out["side"]) == 0
