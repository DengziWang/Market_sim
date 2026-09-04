"""Spec section 8: single-run visualization. Price/candle + volume + spread, one row
per asset so multiple assets from the same run plot side by side (stacked rows,
shared x-axis).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def _bucketed_volume(trades: pd.DataFrame, asset_id: int, n_buckets: int = 200) -> pd.DataFrame:
    sub = trades[trades["asset_id"] == asset_id]
    if sub.empty:
        return pd.DataFrame({"t": [], "volume": []})
    t_min, t_max = sub["time"].min(), sub["time"].max()
    span = max(t_max - t_min, 1e-9)
    bucket_width = span / n_buckets
    sub = sub.copy()
    sub["bucket"] = ((sub["time"] - t_min) / bucket_width).astype(int).clip(upper=n_buckets - 1)
    sub["notional"] = sub["price"] * sub["qty"]
    grouped = sub.groupby("bucket")["notional"].sum()
    t = t_min + (grouped.index.values + 0.5) * bucket_width
    return pd.DataFrame({"t": t, "volume": grouped.values})


def plot_run(
    result: Dict[str, Any],
    assets: Optional[Sequence[int]] = None,
    asset_names: Optional[Dict[int, str]] = None,
    n_volume_buckets: int = 200,
    figsize_per_asset: float = 2.8,
):
    """Build a matplotlib Figure: for each asset, a price panel (trade prices over
    time, plus best bid/ask band when orderbook snapshots were exported) and a volume
    panel underneath. Returns the Figure (caller decides whether to show/save it)."""
    trades = pd.DataFrame(result["trades"])
    snapshots = pd.DataFrame(result.get("snapshots", {}))

    if trades.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.set_title("No trades occurred in this run")
        return fig

    asset_ids = sorted(assets) if assets is not None else sorted(trades["asset_id"].unique())
    n = len(asset_ids)
    fig, axes = plt.subplots(n, 2, figsize=(12, figsize_per_asset * n), squeeze=False, gridspec_kw={"width_ratios": [3, 1]})

    for row, asset_id in enumerate(asset_ids):
        name = (asset_names or {}).get(asset_id, f"asset {asset_id}")
        price_ax, vol_ax = axes[row][0], axes[row][1]

        sub = trades[trades["asset_id"] == asset_id]
        price_ax.plot(sub["time"], sub["price"], lw=0.6, color="#1f5fa8")
        price_ax.set_ylabel(f"{name}\nprice")
        price_ax.set_title(f"{name}: trade price" + (" & book" if not snapshots.empty else ""))

        if not snapshots.empty:
            snap = snapshots[(snapshots["asset_id"] == asset_id) & (snapshots["level"] == 0)]
            bid = snap[snap["side"] == "bid"].sort_values("time")
            ask = snap[snap["side"] == "ask"].sort_values("time")
            if not bid.empty and not ask.empty:
                price_ax.fill_between(bid["time"], bid["price"], ask.set_index("time").reindex(bid["time"], method="nearest")["price"].values, alpha=0.15, color="grey", label="spread")

        vb = _bucketed_volume(trades, asset_id, n_volume_buckets)
        vol_ax.bar(vb["t"], vb["volume"], width=(vb["t"].iloc[1] - vb["t"].iloc[0]) if len(vb) > 1 else 1.0, color="#888")
        vol_ax.set_ylabel("volume\n(notional)")

        if row == n - 1:
            price_ax.set_xlabel("simulated time (s)")
            vol_ax.set_xlabel("simulated time (s)")

    fig.tight_layout()
    return fig
