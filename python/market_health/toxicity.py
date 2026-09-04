"""Fragility / toxicity metrics: report section 5."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _signed_volume(trades: pd.DataFrame, asset_id: int) -> pd.DataFrame:
    t = trades[trades["asset_id"] == asset_id].sort_values("time").copy()
    if t.empty:
        return t
    sign = np.where(t["taker_side"] == "buy", 1.0, -1.0)
    t["signed_qty"] = t["qty"] * sign
    return t


def order_flow_imbalance(trades: pd.DataFrame, asset_id: int, bucket_seconds: float = 30.0) -> pd.Series:
    """Net signed (buy - sell) taker volume per time bucket, as a fraction of total
    volume in that bucket -- in [-1, 1]. Abrupt swings are a classic crash precursor."""
    t = _signed_volume(trades, asset_id)
    if t.empty:
        return pd.Series(dtype=float)
    t["bucket"] = (t["time"] // bucket_seconds).astype(int)
    grouped = t.groupby("bucket").agg(net=("signed_qty", "sum"), total=("qty", "sum"))
    grouped = grouped[grouped["total"] > 0]
    return (grouped["net"] / grouped["total"]).rename("ofi")


def vpin(trades: pd.DataFrame, asset_id: int, n_buckets: int = 50) -> pd.Series:
    """Volume-Synchronized Probability of Informed Trading (Easley/Lopez de Prado/
    O'Hara). Splits total traded volume into `n_buckets` equal-VOLUME (not
    equal-time) buckets, then VPIN_i = |buy_vol - sell_vol| / total_vol within a
    rolling window of buckets. Returns the VPIN series indexed by bucket end time."""
    t = _signed_volume(trades, asset_id)
    if len(t) < n_buckets * 2:
        return pd.Series(dtype=float)
    total_volume = t["qty"].sum()
    bucket_size = total_volume / n_buckets
    cum = t["qty"].cumsum()
    t["vbucket"] = (cum / bucket_size).astype(int).clip(upper=n_buckets * 50)  # cap for pathological cases
    agg = t.groupby("vbucket").agg(buy=("signed_qty", lambda s: s[s > 0].sum()), sell=("signed_qty", lambda s: -s[s < 0].sum()), time=("time", "last"))
    agg["total"] = agg["buy"] + agg["sell"]
    agg = agg[agg["total"] > 0]
    if len(agg) < 5:
        return pd.Series(dtype=float)
    window = max(5, len(agg) // 10)
    imbalance = (agg["buy"] - agg["sell"]).abs()
    vpin_series = imbalance.rolling(window).sum() / agg["total"].rolling(window).sum()
    vpin_series.index = agg["time"]
    return vpin_series.dropna().rename("vpin")


def cancel_to_trade_ratio(order_events: pd.DataFrame, asset_id: int) -> float:
    """#cancels / #fills (partial+full) for one asset -- high values suggest
    quote-stuffing/spoofing-like behavior (relevant for MEV/latency-arb bots that may
    submit and cancel aggressively)."""
    sub = order_events[order_events["asset_id"] == asset_id]
    n_cancel = int((sub["kind"] == "cancel").sum())
    n_fill = int(sub["kind"].isin(["fill", "partial_fill"]).sum())
    if n_fill == 0:
        return np.nan
    return n_cancel / n_fill


def cancel_to_trade_ratio_by_agent_type(order_events: pd.DataFrame, agent_state: pd.DataFrame) -> pd.DataFrame:
    """Same ratio, broken down by bot_type (joins order_events.agent_id against
    agent_state's per-agent bot_type)."""
    type_map = agent_state.drop_duplicates("agent_id").set_index("agent_id")["bot_type"]
    oe = order_events.copy()
    oe["bot_type"] = oe["agent_id"].map(type_map)
    grouped = oe.groupby("bot_type")["kind"].value_counts().unstack(fill_value=0)
    n_cancel = grouped.get("cancel", 0)
    n_fill = grouped.get("fill", 0) + grouped.get("partial_fill", 0)
    ratio = (n_cancel / n_fill.replace(0, np.nan)).rename("cancel_to_trade_ratio")
    return ratio.reset_index()
