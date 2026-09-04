"""Crisis-resilience metrics: report section 5's flash-crash framing (Paddrik et al.
2012 / Kirilenko et al. 2017) -- threshold-based detection of depth/duration/
recovery-time for significant price drops, computed on the trade-price series (works
even if orderbook_snapshots weren't exported).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def detect_flash_crashes(
    price_series: pd.Series,
    drawdown_threshold: float = 0.05,
    window_secs: float = 30.0,
    recovery_frac: float = 0.8,
    max_recovery_horizon_secs: float = 300.0,
) -> List[dict]:
    """A "flash crash" here: price falls by >= `drawdown_threshold` from its trailing
    `window_secs` running max, within that same window. For each detected crash,
    reports depth (max drawdown), duration (time from peak to trough), and recovery
    time (time from trough until price is back within `recovery_frac` of the
    pre-crash peak, or None if it never recovers within `max_recovery_horizon_secs`).
    """
    if price_series.empty:
        return []
    times = price_series.index.to_numpy(dtype=float)
    prices = price_series.to_numpy(dtype=float)
    n = len(prices)

    events = []
    i = 0
    while i < n:
        # trailing running max within window_secs behind i -- use the *last* index
        # attaining that max (not argmax's first-occurrence default), so a flat
        # pre-crash period reports the peak at the moment right before the drop, not
        # at the start of the flat stretch.
        lo = np.searchsorted(times, times[i] - window_secs, side="left")
        window = prices[lo : i + 1]
        max_val = window.max()
        peak_idx = lo + int(np.max(np.flatnonzero(window == max_val)))
        peak = prices[peak_idx]
        if peak <= 0:
            i += 1
            continue
        drawdown = (peak - prices[i]) / peak
        if drawdown >= drawdown_threshold:
            # extend forward to find the local trough within the window
            j = i
            trough_idx = i
            while j < n and times[j] - times[peak_idx] <= window_secs:
                if prices[j] < prices[trough_idx]:
                    trough_idx = j
                j += 1
            trough = prices[trough_idx]
            depth = (peak - trough) / peak
            duration = times[trough_idx] - times[peak_idx]

            recovery_time = None
            target = peak * recovery_frac
            k = trough_idx
            while k < n and times[k] - times[trough_idx] <= max_recovery_horizon_secs:
                if prices[k] >= target:
                    recovery_time = times[k] - times[trough_idx]
                    break
                k += 1

            events.append(
                {
                    "peak_time": float(times[peak_idx]),
                    "trough_time": float(times[trough_idx]),
                    "peak_price": float(peak),
                    "trough_price": float(trough),
                    "depth_frac": float(depth),
                    "duration_secs": float(duration),
                    "recovery_secs": recovery_time,
                }
            )
            i = max(j, trough_idx + 1)
        else:
            i += 1
    return events


def flash_crash_summary(price_series: pd.Series, **kwargs) -> dict:
    events = detect_flash_crashes(price_series, **kwargs)
    if not events:
        return {"n_flash_crashes": 0, "mean_depth_frac": np.nan, "mean_recovery_secs": np.nan, "unrecovered_count": 0}
    depths = [e["depth_frac"] for e in events]
    recoveries = [e["recovery_secs"] for e in events if e["recovery_secs"] is not None]
    return {
        "n_flash_crashes": len(events),
        "mean_depth_frac": float(np.mean(depths)),
        "mean_recovery_secs": float(np.mean(recoveries)) if recoveries else np.nan,
        "unrecovered_count": sum(1 for e in events if e["recovery_secs"] is None),
    }
