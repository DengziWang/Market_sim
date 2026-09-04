"""Price-efficiency / information-content metrics: report section 4."""
from __future__ import annotations

import numpy as np
import pandas as pd

from market_health.volatility import log_returns, mid_price_series


def variance_ratio(price_series: pd.Series, q: int = 4, resample_secs: float = 5.0) -> float:
    """Lo & MacKinlay (1988) variance ratio: Var(q-period return) / (q * Var(1-period
    return)). Under a random walk this is 1; >1 suggests positive autocorrelation
    (momentum/trend), <1 suggests negative autocorrelation (excess mean-reversion,
    e.g. market-maker bid-ask bounce). Computed on a fixed-interval resample so "1
    period" has a well-defined meaning regardless of raw event timing."""
    if price_series.empty or q < 2:
        return np.nan
    idx = pd.to_timedelta(price_series.index, unit="s")
    s = pd.Series(price_series.values, index=idx).resample(f"{resample_secs}s").last().ffill()
    p = np.log(s[s > 0].values)
    if len(p) < q * 4:
        return np.nan
    r1 = np.diff(p)
    var_1 = np.var(r1, ddof=1)
    if var_1 == 0 or np.isnan(var_1):
        return np.nan
    rq = p[q:] - p[:-q]
    var_q = np.var(rq, ddof=1)
    return float(var_q / (q * var_1))


def return_autocorrelation(price_series: pd.Series, lag: int = 1, resample_secs: float = 5.0) -> float:
    idx = pd.to_timedelta(price_series.index, unit="s")
    s = pd.Series(price_series.values, index=idx).resample(f"{resample_secs}s").last().ffill()
    r = log_returns(pd.Series(s.values, index=s.index.total_seconds()))
    if len(r) < lag + 5:
        return np.nan
    return float(r.autocorr(lag))


def fundamental_deviation(snapshots: pd.DataFrame, asset_id: int) -> dict:
    """Mean/std of the market's traded (mid) price relative to the simulation's known
    ground-truth latent fundamental value -- a diagnostic real markets don't have
    (report section 4's "advantage over real markets" note)."""
    sub = snapshots[(snapshots["asset_id"] == asset_id) & (snapshots["side"] == "fundamental")]
    mid = mid_price_series(snapshots, asset_id)
    if sub.empty or mid.empty:
        return {"mean_abs_dev_frac": np.nan, "std_dev_frac": np.nan}
    fair = sub.set_index("time")["price"].sort_index()
    merged = pd.merge_asof(mid.reset_index().rename(columns={"time": "time"}), fair.reset_index().rename(columns={"price": "fair"}), on="time", direction="nearest")
    merged = merged.dropna(subset=["mid", "fair"])
    merged = merged[merged["fair"] > 0]
    if merged.empty:
        return {"mean_abs_dev_frac": np.nan, "std_dev_frac": np.nan}
    dev = (merged["mid"] - merged["fair"]) / merged["fair"]
    return {"mean_abs_dev_frac": float(dev.abs().mean()), "std_dev_frac": float(dev.std())}
