"""Liquidity metrics: report section 2. Operates only on exported parquet data
(trades / orderbook_snapshots), never re-runs the simulation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _mid_series(snapshots: pd.DataFrame, asset_id: int) -> pd.DataFrame:
    """Best bid/ask (level 0) over time for one asset, from the long-format
    orderbook_snapshots table -- returns columns [time, bid, ask, mid, spread]."""
    sub = snapshots[(snapshots["asset_id"] == asset_id) & (snapshots["level"] == 0) & (snapshots["side"].isin(["bid", "ask"]))]
    piv = sub.pivot_table(index="time", columns="side", values="price", aggfunc="last").sort_index()
    piv = piv.rename(columns={"bid": "bid", "ask": "ask"})
    for c in ("bid", "ask"):
        if c not in piv.columns:
            piv[c] = np.nan
    piv["mid"] = (piv["bid"] + piv["ask"]) / 2.0
    piv["spread"] = piv["ask"] - piv["bid"]
    return piv.reset_index()


def effective_relative_spread(trades: pd.DataFrame, snapshots: pd.DataFrame, asset_id: int) -> pd.Series:
    """RES = (p(t) - m(t)) / m(t) per trade (report eq. in section 2), using the most
    recent snapshot mid price at or before each trade's time."""
    mid = _mid_series(snapshots, asset_id)
    if mid.empty:
        return pd.Series(dtype=float)
    t = trades[trades["asset_id"] == asset_id].sort_values("time")
    if t.empty:
        return pd.Series(dtype=float)
    merged = pd.merge_asof(t, mid[["time", "mid"]], on="time", direction="backward")
    return ((merged["price"] - merged["mid"]) / merged["mid"]).dropna()


def quoted_spread_stats(snapshots: pd.DataFrame, asset_id: int) -> dict:
    mid = _mid_series(snapshots, asset_id)
    if mid.empty or mid["mid"].isna().all():
        return {"mean_abs_spread": np.nan, "mean_rel_spread": np.nan, "median_rel_spread": np.nan}
    rel = (mid["spread"] / mid["mid"]).dropna()
    return {
        "mean_abs_spread": float(mid["spread"].mean()),
        "mean_rel_spread": float(rel.mean()) if not rel.empty else np.nan,
        "median_rel_spread": float(rel.median()) if not rel.empty else np.nan,
    }


def depth_stats(snapshots: pd.DataFrame, asset_id: int, levels: Optional[int] = None) -> dict:
    """Mean resting quantity on each side, summed over the top `levels` levels (all
    exported levels if None)."""
    sub = snapshots[(snapshots["asset_id"] == asset_id) & (snapshots["side"].isin(["bid", "ask"]))]
    if levels is not None:
        sub = sub[sub["level"] < levels]
    if sub.empty:
        return {"mean_bid_depth": np.nan, "mean_ask_depth": np.nan}
    by_time_side = sub.groupby(["time", "side"])["qty"].sum().reset_index()
    return {
        "mean_bid_depth": float(by_time_side[by_time_side.side == "bid"]["qty"].mean()) if (by_time_side.side == "bid").any() else np.nan,
        "mean_ask_depth": float(by_time_side[by_time_side.side == "ask"]["qty"].mean()) if (by_time_side.side == "ask").any() else np.nan,
    }


def amihud_illiquidity(trades: pd.DataFrame, asset_id: int, bucket_seconds: float = 60.0) -> float:
    """Amihud (2002): mean( |return| / dollar_volume ) over time buckets. Larger =
    less liquid (a given amount of trading moves price more)."""
    t = trades[trades["asset_id"] == asset_id].sort_values("time")
    if len(t) < 2:
        return np.nan
    t = t.copy()
    t["bucket"] = (t["time"] // bucket_seconds).astype(int)
    agg = t.groupby("bucket").agg(open=("price", "first"), close=("price", "last"), notional=("price", lambda s: np.nan))
    notional = t.assign(notional=t["price"] * t["qty"]).groupby("bucket")["notional"].sum()
    agg["notional"] = notional
    agg = agg[agg["notional"] > 0]
    if agg.empty:
        return np.nan
    ret = (agg["close"] - agg["open"]).abs() / agg["open"]
    illiq = (ret / agg["notional"]).replace([np.inf, -np.inf], np.nan).dropna()
    return float(illiq.mean()) if not illiq.empty else np.nan


def kyle_lambda(trades: pd.DataFrame, asset_id: int, bucket_seconds: float = 10.0) -> float:
    """Kyle's Lambda: OLS slope of price change on signed net order flow (taker-side
    signed volume) within time buckets. Larger = a unit of order flow moves price
    more (thinner market)."""
    t = trades[trades["asset_id"] == asset_id].sort_values("time")
    if len(t) < 3:
        return np.nan
    t = t.copy()
    if "taker_side" in t.columns:
        sign = np.where(t["taker_side"] == "buy", 1.0, -1.0)
    else:
        # Fallback (older exports without taker_side): classic Lee-Ready tick rule.
        sign = np.sign(t["price"].diff().fillna(0.0))
        sign = np.where(sign == 0, 1.0, sign)
    t["signed_qty"] = t["qty"] * sign
    t["bucket"] = (t["time"] // bucket_seconds).astype(int)
    agg = t.groupby("bucket").agg(price_first=("price", "first"), price_last=("price", "last"), net_flow=("signed_qty", "sum"))
    agg["dprice"] = agg["price_last"] - agg["price_first"]
    agg = agg.dropna()
    if len(agg) < 3 or agg["net_flow"].std() == 0:
        return np.nan
    x = agg["net_flow"].values
    y = agg["dprice"].values
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def resilience(snapshots: pd.DataFrame, asset_id: int, shock_frac: float = 0.02, recovery_frac: float = 0.5, max_horizon_secs: float = 60.0) -> dict:
    """How quickly spread reverts toward its pre-shock level after widening sharply
    (Muranaga & Shimizu framing). A "shock" = spread jumping by >= `shock_frac` of
    mid relative to its trailing median; "recovered" = spread back within
    `recovery_frac` of the pre-shock level. Returns mean/median recovery time (secs)
    and the number of shocks detected, or NaNs if none were found."""
    mid = _mid_series(snapshots, asset_id).dropna(subset=["mid", "spread"])
    if len(mid) < 10:
        return {"n_shocks": 0, "mean_recovery_secs": np.nan, "median_recovery_secs": np.nan}
    rel_spread = (mid["spread"] / mid["mid"]).values
    times = mid["time"].values
    baseline = pd.Series(rel_spread).rolling(20, min_periods=5).median().values
    recoveries = []
    i = 1
    n = len(rel_spread)
    while i < n:
        if not np.isnan(baseline[i]) and rel_spread[i] - baseline[i] >= shock_frac:
            pre_level = baseline[i]
            j = i + 1
            recovered_at = None
            while j < n and times[j] - times[i] <= max_horizon_secs:
                if rel_spread[j] <= pre_level + recovery_frac * shock_frac:
                    recovered_at = times[j]
                    break
                j += 1
            if recovered_at is not None:
                recoveries.append(recovered_at - times[i])
            i = j + 1
        else:
            i += 1
    if not recoveries:
        return {"n_shocks": 0, "mean_recovery_secs": np.nan, "median_recovery_secs": np.nan}
    return {
        "n_shocks": len(recoveries),
        "mean_recovery_secs": float(np.mean(recoveries)),
        "median_recovery_secs": float(np.median(recoveries)),
    }
