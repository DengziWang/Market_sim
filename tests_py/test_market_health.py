"""Tests for market_health analysis functions against synthetic data with a known
expected result -- per the non-functional requirement in spec section 10.
"""
import numpy as np
import pandas as pd
import pytest

from market_health import concentration, crisis, efficiency, toxicity, wash_trading


# ---------------------------------------------------------------------------------
# wash_trading
# ---------------------------------------------------------------------------------


def test_benford_accepts_benford_compliant_sizes():
    rng = np.random.default_rng(0)
    # log-uniform samples are a standard way to generate data that closely follows
    # Benford's Law (scale-invariant magnitude distribution).
    sizes = pd.Series(10 ** rng.uniform(0, 4, size=5000))
    result = wash_trading.benford_test(sizes)
    assert result["n"] == 5000
    # Should NOT reject the null (data is Benford-compliant by construction).
    assert result["p_value"] > 0.01


def test_benford_rejects_round_number_only_sizes():
    # All trade sizes exactly 1.0, 2.0, 5.0, 10.0 (classic wash-trading signature):
    # first digits are only {1, 2, 5}, nothing like Benford's Law.
    sizes = pd.Series(np.random.default_rng(1).choice([1.0, 2.0, 5.0, 10.0], size=2000))
    result = wash_trading.benford_test(sizes)
    assert result["p_value"] < 1e-6


def test_round_number_clustering_known_fractions():
    sizes = pd.Series([1.0, 2.0, 3.0, 1.5, 2.5, 4.0])  # 4/6 are integer multiples of 1.0
    result = wash_trading.round_number_clustering(sizes, round_to=1.0)
    assert result["round_fraction"] == pytest.approx(4 / 6)

    continuous = pd.Series(np.random.default_rng(2).uniform(0.001, 1.0, size=5000))
    result2 = wash_trading.round_number_clustering(continuous, round_to=1.0, tolerance=1e-6)
    assert result2["round_fraction"] < 0.01


def test_ground_truth_wash_volume_share_known_split():
    # Trade 0 (notional 100): wash-trading agent 1 is the maker, agent 2 (momentum) is
    # the taker -> exactly one of its two legs is wash-attributed. Trade 1 (notional
    # 300): both maker and taker are non-wash -> contributes 0. Each trade has 2 legs,
    # so total leg-notional = 2*(100+300) = 800, and wash-attributed leg-notional =
    # 100 (only trade 0's maker leg) -> share = 100/800 = 0.125.
    trades = pd.DataFrame(
        {
            "price": [10.0, 10.0],
            "qty": [10.0, 30.0],  # notional 100 and 300
            "maker_agent_id": [1, 2],
            "taker_agent_id": [2, 3],
        }
    )
    agent_state = pd.DataFrame({"agent_id": [1, 2, 3], "bot_type": ["wash_trading", "momentum", "momentum"]})
    share = wash_trading.ground_truth_wash_volume_share(trades, agent_state)
    assert share == pytest.approx(100.0 / 800.0)


# ---------------------------------------------------------------------------------
# concentration
# ---------------------------------------------------------------------------------


def test_hhi_matches_hand_computed_value():
    # hhi() combines maker- and taker-side volume per agent (documented behavior --
    # ignoring taker flow would badly undercount aggressive/taker-only participants).
    # 4 agents in a ring, each maker on one trade (notional 100) and taker on the
    # next -> every agent ends up with combined volume 200 out of a total of 800
    # (each of the 4 trades' 100 notional counted once per leg) -> equal 0.25 shares.
    trades = pd.DataFrame(
        {
            "price": [1.0] * 4,
            "qty": [100.0, 100.0, 100.0, 100.0],
            "maker_agent_id": [1, 2, 3, 4],
            "taker_agent_id": [2, 3, 4, 1],
        }
    )
    result = concentration.hhi(trades, by="agent")
    assert result["hhi"] == pytest.approx(0.25, rel=1e-6)


def test_hhi_fully_concentrated_is_one():
    trades = pd.DataFrame({"price": [1.0], "qty": [100.0], "maker_agent_id": [1], "taker_agent_id": [1]})
    result = concentration.hhi(trades, by="agent")
    assert result["hhi"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------
# efficiency
# ---------------------------------------------------------------------------------


def test_variance_ratio_close_to_one_for_random_walk():
    rng = np.random.default_rng(3)
    n = 5000
    returns = rng.normal(0, 0.001, size=n)
    prices = 100 * np.exp(np.cumsum(returns))
    times = np.arange(n) * 5.0  # 5-second bars, matching the default resample
    series = pd.Series(prices, index=times)
    vr = efficiency.variance_ratio(series, q=4, resample_secs=5.0)
    assert vr == pytest.approx(1.0, abs=0.15)


def test_variance_ratio_below_one_for_mean_reverting_series():
    # Strong negative-autocorrelation (bid-ask-bounce-like) series: alternates around
    # a slow trend -> multi-period variance grows much slower than q * 1-period
    # variance -> ratio well below 1.
    rng = np.random.default_rng(4)
    n = 4000
    noise = rng.normal(0, 0.01, size=n)
    bounce = np.array([0.02 if i % 2 == 0 else -0.02 for i in range(n)])
    log_prices = np.cumsum(noise * 0.05) + bounce
    prices = 100 * np.exp(log_prices)
    times = np.arange(n) * 5.0
    series = pd.Series(prices, index=times)
    vr = efficiency.variance_ratio(series, q=4, resample_secs=5.0)
    assert vr < 0.7


# ---------------------------------------------------------------------------------
# crisis (flash crash detection)
# ---------------------------------------------------------------------------------


def test_flash_crash_detected_with_known_depth_and_recovery():
    times = np.arange(0, 120, 1.0)
    prices = np.full_like(times, 100.0)
    # Crash at t=50: price falls 10% instantly, recovers linearly back to 99 by t=70.
    crash_idx = 50
    recover_idx = 70
    prices[crash_idx:recover_idx] = np.linspace(90.0, 99.0, recover_idx - crash_idx)
    prices[recover_idx:] = 99.5
    series = pd.Series(prices, index=times)

    events = crisis.detect_flash_crashes(series, drawdown_threshold=0.05, window_secs=30.0, recovery_frac=0.95)
    assert len(events) >= 1
    ev = events[0]
    assert ev["depth_frac"] == pytest.approx(0.10, abs=0.02)
    assert ev["peak_time"] == pytest.approx(49.0, abs=2.0)


def test_no_flash_crash_in_flat_series():
    times = np.arange(0, 100, 1.0)
    prices = 100 + np.sin(times / 10.0) * 0.1  # tiny wiggle, no real drawdown
    series = pd.Series(prices, index=times)
    events = crisis.detect_flash_crashes(series, drawdown_threshold=0.05)
    assert events == []


# ---------------------------------------------------------------------------------
# toxicity
# ---------------------------------------------------------------------------------


def test_cancel_to_trade_ratio_known_counts():
    order_events = pd.DataFrame(
        {
            "asset_id": [0, 0, 0, 0, 0],
            "kind": ["submit", "cancel", "cancel", "fill", "partial_fill"],
        }
    )
    ratio = toxicity.cancel_to_trade_ratio(order_events, asset_id=0)
    assert ratio == pytest.approx(2 / 2)


def test_order_flow_imbalance_all_buys_is_one():
    trades = pd.DataFrame(
        {
            "asset_id": [0, 0, 0],
            "time": [0.0, 1.0, 2.0],
            "qty": [1.0, 1.0, 1.0],
            "taker_side": ["buy", "buy", "buy"],
        }
    )
    ofi = toxicity.order_flow_imbalance(trades, asset_id=0, bucket_seconds=10.0)
    assert (ofi == 1.0).all()
