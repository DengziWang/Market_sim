"""OHLC candlestick aggregation for the dashboard's price chart.

Time-window bucketing, not trade-count bucketing: a candle covers a slice of
simulated time, so its volume bar means what a volume bar normally means (how much
happened in that slice), and a quiet period reads as a flat, low-volume candle
rather than being stretched to "fill" the same trade count as a busy one.

The bucket width is *not* a fixed global grid -- it's recomputed for whatever time
range is currently in view: `width = view_duration / target_candles`. Zooming into
a narrower range therefore produces thinner, more numerous candles covering that
span, with no notion of a candle "splitting" that needs to be tracked explicitly --
the same view, redrawn at a smaller width, naturally shows more (thinner) candles.
At a narrow enough view, width can drop below the typical time between trades, at
which point most candles hold exactly one trade (or zero) -- real single-trade
resolution, not a special-cased mode.

Two entry points sharing one bucketing core:
  - `candles_from_trades`: raw trades (time/price/qty) -> a candle DataFrame. The
    normal case -- a loaded run's actual trade data.
  - `resample_candles`: an *existing* candle DataFrame -> a coarser one. Used both
    to downsample a fine series for display, and to compose several separately-
    computed candle series (e.g. one per leg of a `run_scenario_in_legs` sequence)
    into a single overview spanning all of them -- OHLC composes correctly this
    way (open of the merged bucket = open of its first input candle, high/low =
    max/min across inputs, close = close of the last input candle, volume = sum),
    the same math real charting software uses to build higher-timeframe candles
    from lower-timeframe ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CANDLE_COLUMNS = ["bucket_start", "bucket_end", "open", "high", "low", "close", "mean", "volume", "n_trades"]


def _bucket_index(t: np.ndarray, view_start: float, view_end: float, target_buckets: int) -> tuple[np.ndarray, float, int]:
    duration = max(view_end - view_start, 1e-12)
    n_buckets = max(1, int(target_buckets))
    width = duration / n_buckets
    idx = np.floor((t - view_start) / width).astype(np.int64)
    np.clip(idx, 0, n_buckets - 1, out=idx)
    return idx, width, n_buckets


def _empty_candles() -> pd.DataFrame:
    return pd.DataFrame(columns=CANDLE_COLUMNS)


def candles_from_trades(
    time: np.ndarray,
    price: np.ndarray,
    qty: np.ndarray,
    view_start: float,
    view_end: float,
    target_candles: int,
) -> pd.DataFrame:
    """`time` must be sorted ascending (every trade table in this project already
    is). Trades outside `[view_start, view_end]` are ignored. Returns one row per
    bucket covering the *whole* view range (`CANDLE_COLUMNS`), including empty
    buckets (forward-filled from the previous close, `n_trades=0`, `volume=0`) so
    the series has no gaps -- a quiet stretch reads as a flat candle, matching how
    a real exchange chart shows a quiet period rather than leaving a hole.
    """
    if len(time) == 0 or view_end <= view_start:
        return _empty_candles()
    mask = (time >= view_start) & (time <= view_end)
    t, p, q = time[mask], price[mask], qty[mask]
    idx, width, n_buckets = _bucket_index(t, view_start, view_end, target_candles)

    if len(t) == 0:
        # No trades at all in this range -- still return a flat series (a caller
        # showing "no trades in this window" is better served by an empty frame,
        # but callers composing multiple ranges want a uniform shape); leave it to
        # the caller to decide, empty here is the more honest answer.
        return _empty_candles()

    # `idx` is monotonic non-decreasing (t is sorted ascending by contract, and
    # bucket assignment is monotonic in t), so every occupied bucket is one
    # contiguous run in `t`/`p`/`q` -- exactly the precondition for `np.ufunc
    # .reduceat`, the same trick the dashboard's own `_downsample_minmax` uses.
    # This replaces a `groupby(...).agg(...)` that measured ~190ms per ~1.4M-row
    # leg (computing a per-leg summary for the multi-leg overview needs one of
    # these per leg, per asset -- 200 legs was ~45s of pure aggregation, the
    # actual bottleneck once `df.to_dict("list")`-style boxing was already ruled
    # out elsewhere) down to single-digit milliseconds: reduceat is a tight C loop
    # over contiguous runs, not a hash-based grouping.
    starts = np.concatenate(([0], np.flatnonzero(np.diff(idx)) + 1))
    ends = np.concatenate((starts[1:], [len(p)]))
    counts = ends - starts
    agg = pd.DataFrame(
        {
            "open": p[starts],
            "high": np.maximum.reduceat(p, starts),
            "low": np.minimum.reduceat(p, starts),
            "close": p[ends - 1],
            # Plain arithmetic mean of trade prices within the bucket -- the line
            # chart's y-value (see dashboard.py's Line chart-type mode): a genuine
            # time-span average, not a single representative trade (close/open) or
            # a min/max extremum, per the user's explicit preference. Composes
            # exactly under `resample_candles` below via a count-weighted average
            # of sub-bucket means, since that's arithmetically identical to
            # averaging the underlying individual trade prices directly.
            "mean": np.add.reduceat(p, starts) / counts,
            "volume": np.add.reduceat(q, starts),
            "n_trades": counts,
        },
        index=idx[starts],
    )

    out = pd.DataFrame(index=pd.RangeIndex(n_buckets))
    out = out.join(agg)
    out["bucket_start"] = view_start + np.arange(n_buckets) * width
    out["bucket_end"] = out["bucket_start"] + width

    # Forward-fill empty buckets from the previous candle's close; any buckets
    # before the first trade (no previous close yet) fall back to that first
    # trade's open, so there's never a NaN candle at the very start of the view.
    out["close"] = out["close"].ffill().bfill()
    out["open"] = out["open"].fillna(out["close"])
    out["high"] = out["high"].fillna(out["close"])
    out["low"] = out["low"].fillna(out["close"])
    out["mean"] = out["mean"].ffill().bfill()
    out["volume"] = out["volume"].fillna(0.0)
    out["n_trades"] = out["n_trades"].fillna(0).astype(int)
    return out[CANDLE_COLUMNS]


def resample_candles(candles: pd.DataFrame, view_start: float, view_end: float, target_candles: int) -> pd.DataFrame:
    """Re-buckets an already-computed candle series (or several concatenated ones,
    e.g. one per leg -- concatenate first with `pd.concat`, sorted by
    `bucket_start`, then call this once) down to `target_candles` coarser candles.
    Each input candle contributes to whichever new bucket its midpoint falls in;
    composing OHLCV across the candles landing in the same new bucket uses the
    same open-of-first/high-of-max/low-of-min/close-of-last/volume-of-sum rule as
    `candles_from_trades`, just over candles instead of individual trades.
    """
    if candles.empty or view_end <= view_start:
        return _empty_candles()
    c = candles[(candles["bucket_end"] > view_start) & (candles["bucket_start"] < view_end)].copy()
    if c.empty:
        return _empty_candles()
    c = c.sort_values("bucket_start")
    # Count-weighted mean of the sub-buckets' own means -- exactly equal to the
    # plain average of every underlying individual trade price, since each
    # sub-bucket's "mean" is itself already an unweighted average over its
    # n_trades. Computed before groupby so it aggregates as a plain sum like the
    # other columns.
    c["_mean_x_n"] = c["mean"] * c["n_trades"]
    midpoint = (c["bucket_start"] + c["bucket_end"]) / 2.0
    idx, width, n_buckets = _bucket_index(midpoint.to_numpy(), view_start, view_end, target_candles)
    c["bucket"] = idx

    agg = c.groupby("bucket").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        mean_x_n=("_mean_x_n", "sum"),
        volume=("volume", "sum"),
        n_trades=("n_trades", "sum"),
    )
    out = pd.DataFrame(index=pd.RangeIndex(n_buckets))
    out = out.join(agg)
    out["bucket_start"] = view_start + np.arange(n_buckets) * width
    out["bucket_end"] = out["bucket_start"] + width
    out["close"] = out["close"].ffill().bfill()
    out["open"] = out["open"].fillna(out["close"])
    out["high"] = out["high"].fillna(out["close"])
    out["low"] = out["low"].fillna(out["close"])
    out["mean"] = (out["mean_x_n"] / out["n_trades"]).where(out["n_trades"] > 0)
    out["mean"] = out["mean"].ffill().bfill()
    out["volume"] = out["volume"].fillna(0.0)
    out["n_trades"] = out["n_trades"].fillna(0).astype(int)
    return out[CANDLE_COLUMNS]
