"""Tests for market_sim.candles: OHLC candlestick aggregation powering the
dashboard's price chart. The core correctness property that matters most --
`resample_candles` composing several candles into a coarser one must agree with
computing that coarser resolution directly from the raw trades -- is checked
explicitly, since that's what makes the multi-leg overview (composing many legs'
separately-computed candle summaries) trustworthy.
"""
import numpy as np
import pandas as pd
import pytest

from market_sim.candles import candles_from_trades, resample_candles


def test_single_bucket_basic_ohlcv():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    price = np.array([100.0, 105.0, 95.0, 102.0])
    qty = np.array([1.0, 2.0, 3.0, 4.0])
    out = candles_from_trades(time, price, qty, view_start=0.0, view_end=10.0, target_candles=1)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["open"] == 100.0
    assert row["high"] == 105.0
    assert row["low"] == 95.0
    assert row["close"] == 102.0
    assert row["mean"] == pytest.approx((100.0 + 105.0 + 95.0 + 102.0) / 4)
    assert row["volume"] == 10.0
    assert row["n_trades"] == 4


def test_trades_split_across_buckets_by_time():
    # 4 trades spread over [0, 4): 2 buckets of width 2 each -> [0,2), [2,4)
    time = np.array([0.5, 1.5, 2.5, 3.5])
    price = np.array([100.0, 110.0, 90.0, 95.0])
    qty = np.array([1.0, 1.0, 1.0, 1.0])
    out = candles_from_trades(time, price, qty, view_start=0.0, view_end=4.0, target_candles=2)
    assert len(out) == 2
    assert out.iloc[0]["open"] == 100.0 and out.iloc[0]["close"] == 110.0 and out.iloc[0]["high"] == 110.0
    assert out.iloc[1]["open"] == 90.0 and out.iloc[1]["close"] == 95.0 and out.iloc[1]["low"] == 90.0


def test_empty_buckets_forward_fill_from_previous_close():
    # Trades only in the first and last of 5 buckets over [0, 10).
    time = np.array([0.5, 9.5])
    price = np.array([100.0, 120.0])
    qty = np.array([1.0, 1.0])
    out = candles_from_trades(time, price, qty, view_start=0.0, view_end=10.0, target_candles=5)
    assert len(out) == 5
    assert out.iloc[0]["n_trades"] == 1
    # Middle buckets (no trades) should be flat at the previous close, zero volume.
    for i in [1, 2, 3]:
        assert out.iloc[i]["n_trades"] == 0
        assert out.iloc[i]["volume"] == 0.0
        assert out.iloc[i]["open"] == out.iloc[i]["close"] == out.iloc[i]["high"] == out.iloc[i]["low"] == 100.0
    assert out.iloc[4]["n_trades"] == 1
    assert out.iloc[4]["close"] == 120.0


def test_no_nan_candles_even_with_no_trades_before_first():
    time = np.array([5.0])
    price = np.array([50.0])
    qty = np.array([1.0])
    out = candles_from_trades(time, price, qty, view_start=0.0, view_end=10.0, target_candles=10)
    assert not out[["open", "high", "low", "close"]].isna().any().any()
    # Buckets before the first trade should sit flat at that trade's own price.
    assert (out.iloc[:5]["close"] == 50.0).all()


def test_trades_outside_view_range_are_ignored():
    time = np.array([-5.0, 1.0, 2.0, 50.0])
    price = np.array([1.0, 100.0, 110.0, 999.0])
    qty = np.array([1.0, 1.0, 1.0, 1.0])
    out = candles_from_trades(time, price, qty, view_start=0.0, view_end=10.0, target_candles=1)
    assert len(out) == 1
    assert out.iloc[0]["open"] == 100.0
    assert out.iloc[0]["close"] == 110.0
    assert out.iloc[0]["high"] == 110.0  # not 999 -- that trade is outside the view


def test_empty_trades_returns_empty_frame():
    out = candles_from_trades(np.array([]), np.array([]), np.array([]), 0.0, 10.0, 5)
    assert out.empty


def test_zero_width_view_returns_empty_frame():
    out = candles_from_trades(np.array([1.0]), np.array([100.0]), np.array([1.0]), 5.0, 5.0, 5)
    assert out.empty


def test_resample_composing_fine_candles_matches_direct_coarse_computation():
    rng = np.random.default_rng(42)
    n = 5000
    time = np.sort(rng.uniform(0, 1000, size=n))
    price = 100 + np.cumsum(rng.normal(0, 0.1, size=n))
    qty = rng.uniform(0.1, 5.0, size=n)

    fine = candles_from_trades(time, price, qty, view_start=0.0, view_end=1000.0, target_candles=200)
    coarse_via_resample = resample_candles(fine, view_start=0.0, view_end=1000.0, target_candles=20)
    coarse_direct = candles_from_trades(time, price, qty, view_start=0.0, view_end=1000.0, target_candles=20)

    assert len(coarse_via_resample) == len(coarse_direct) == 20
    np.testing.assert_allclose(coarse_via_resample["open"].to_numpy(), coarse_direct["open"].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(coarse_via_resample["high"].to_numpy(), coarse_direct["high"].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(coarse_via_resample["low"].to_numpy(), coarse_direct["low"].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(coarse_via_resample["close"].to_numpy(), coarse_direct["close"].to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(coarse_via_resample["mean"].to_numpy(), coarse_direct["mean"].to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(coarse_via_resample["volume"].to_numpy(), coarse_direct["volume"].to_numpy(), rtol=1e-6)
    np.testing.assert_array_equal(coarse_via_resample["n_trades"].to_numpy(), coarse_direct["n_trades"].to_numpy())


def test_mean_is_plain_average_of_trade_prices_in_bucket():
    # Unweighted mean of the raw trade prices, not close/open, and not
    # volume-weighted -- a genuine time-span average, per its docstring.
    time = np.array([1.0, 2.0, 3.0])
    price = np.array([100.0, 200.0, 300.0])
    qty = np.array([1.0, 50.0, 1.0])  # deliberately lopsided qty -- must not skew "mean"
    out = candles_from_trades(time, price, qty, view_start=0.0, view_end=10.0, target_candles=1)
    assert out.iloc[0]["mean"] == pytest.approx(200.0)  # (100+200+300)/3, not qty-weighted


def test_resample_composes_multiple_concatenated_leg_summaries():
    # Two "legs" covering disjoint time ranges, each summarized into a handful of
    # candles -- concatenating and resampling should behave like one continuous
    # series, not lose or double-count anything at the boundary.
    rng = np.random.default_rng(7)
    leg1_time = np.sort(rng.uniform(0, 500, size=1000))
    leg1_price = 100 + np.cumsum(rng.normal(0, 0.05, size=1000))
    leg1_qty = rng.uniform(0.1, 2.0, size=1000)
    leg1_candles = candles_from_trades(leg1_time, leg1_price, leg1_qty, 0.0, 500.0, target_candles=50)

    leg2_time = np.sort(rng.uniform(500, 1000, size=1000))
    leg2_price = leg1_price[-1] + np.cumsum(rng.normal(0, 0.05, size=1000))
    leg2_qty = rng.uniform(0.1, 2.0, size=1000)
    leg2_candles = candles_from_trades(leg2_time, leg2_price, leg2_qty, 500.0, 1000.0, target_candles=50)

    combined = pd.concat([leg1_candles, leg2_candles], ignore_index=True)
    overview = resample_candles(combined, view_start=0.0, view_end=1000.0, target_candles=20)

    assert len(overview) == 20
    total_volume = overview["volume"].sum()
    expected_total_volume = leg1_qty.sum() + leg2_qty.sum()
    assert total_volume == pytest.approx(expected_total_volume, rel=1e-6)
    # First candle's open should be leg 1's very first trade price; last candle's
    # close should be leg 2's very last trade price -- confirms no boundary loss.
    assert overview.iloc[0]["open"] == pytest.approx(leg1_candles.iloc[0]["open"])
    assert overview.iloc[-1]["close"] == pytest.approx(leg2_candles.iloc[-1]["close"])


def test_single_target_candle_covers_whole_range():
    time = np.array([1.0, 5.0, 9.0])
    price = np.array([10.0, 20.0, 15.0])
    qty = np.array([1.0, 1.0, 1.0])
    out = candles_from_trades(time, price, qty, 0.0, 10.0, target_candles=1)
    assert len(out) == 1
    assert out.iloc[0]["open"] == 10.0
    assert out.iloc[0]["high"] == 20.0
    assert out.iloc[0]["low"] == 10.0
    assert out.iloc[0]["close"] == 15.0
