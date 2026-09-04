"""Volatility metrics: report section 3."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mid_price_series(snapshots: pd.DataFrame, asset_id: int) -> pd.Series:
    from market_health.liquidity import _mid_series

    m = _mid_series(snapshots, asset_id)
    return m.set_index("time")["mid"].dropna()


def trade_price_series(trades: pd.DataFrame, asset_id: int) -> pd.Series:
    t = trades[trades["asset_id"] == asset_id].sort_values("time")
    return t.set_index("time")["price"]


def log_returns(price_series: pd.Series) -> pd.Series:
    p = price_series[price_series > 0]
    return np.log(p).diff().dropna()


def realized_volatility(price_series: pd.Series, annualize_seconds: float = None) -> float:
    """Std of log returns over the whole series (per-sample, i.e. per the series'
    own native sampling interval -- pass a resampled series if you want a specific
    bar size). `annualize_seconds`: if given, scales to a per-`annualize_seconds`
    volatility assuming iid returns (sqrt-time scaling) -- e.g. 3600 for hourly."""
    r = log_returns(price_series)
    if len(r) < 2:
        return np.nan
    vol = float(r.std())
    if annualize_seconds and len(price_series) > 1:
        span = price_series.index.max() - price_series.index.min()
        avg_dt = span / max(len(price_series) - 1, 1)
        if avg_dt > 0:
            vol *= np.sqrt(annualize_seconds / avg_dt)
    return vol


def volatility_clustering_autocorr(price_series: pd.Series, max_lag: int = 20, resample_secs: float = 5.0) -> pd.Series:
    """Autocorrelation of |return| at lags 1..max_lag, on a fixed-interval resample
    of the price series. Positive autocorrelation of absolute returns at short lags
    is one of the most robust "stylized facts" of real markets (volatility
    clustering) -- a market with none of this is suspicious, not necessarily good."""
    if price_series.empty:
        return pd.Series(dtype=float)
    idx = pd.to_timedelta(price_series.index, unit="s")
    s = pd.Series(price_series.values, index=idx).resample(f"{resample_secs}s").last().ffill()
    r = np.log(s[s > 0]).diff().dropna().abs()
    if len(r) < max_lag + 2:
        return pd.Series(dtype=float)
    return pd.Series({lag: r.autocorr(lag) for lag in range(1, max_lag + 1)})


def jump_fraction(price_series: pd.Series, jump_z_threshold: float = 4.0) -> dict:
    """Simple threshold-based jump detector: fraction of total squared-return
    variation attributable to returns more than `jump_z_threshold` robust-std away
    from zero (a crude but standard proxy for the continuous/jump decomposition; not
    a full bipower-variation estimator)."""
    r = log_returns(price_series)
    if len(r) < 10:
        return {"jump_variance_fraction": np.nan, "n_jumps": 0}
    mad = (r - r.median()).abs().median() * 1.4826  # robust std estimate
    if mad == 0:
        return {"jump_variance_fraction": np.nan, "n_jumps": 0}
    z = (r - r.median()) / mad
    jumps = r[z.abs() > jump_z_threshold]
    total_var = float((r**2).sum())
    jump_var = float((jumps**2).sum())
    return {
        "jump_variance_fraction": jump_var / total_var if total_var > 0 else np.nan,
        "n_jumps": int(len(jumps)),
    }
