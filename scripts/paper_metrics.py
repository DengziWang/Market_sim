"""Computes the exact metrics the paper's own tables report (Section 5's
Tables 6-8, Section 6's Tables 9 and 11) from a saved run's raw exported data,
and renders them into a PDF -- one table per page, matching the paper's own
table layout. Used by human_pct_sweep.py and dump_sim_sweep.py after their
sweep finishes.

Formulas reproduced exactly as the paper defines them (02_Methodology.tex /
03_Results.tex), including the ones a later review found statistically
debatable (e.g. the flash-crash/growth count is a raw per-trade threshold
crossing count, not deduplicated into distinct events, and volatility is a
per-trade mean, not normalized by elapsed time) -- this module's job is
faithful reproduction of what the paper actually reports, not a corrected
methodology.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from market_sim.runner import load_run_legs

from market_health.liquidity import depth_stats, quoted_spread_stats


def _to_df(columnar: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(columnar) if columnar else pd.DataFrame()


def load_run_tables(out_dir: str) -> Dict[str, Any]:
    """Loads one run (possibly multi-leg) and returns its trades/snapshots/
    agent_state as DataFrames plus meta, in the shape the metric functions
    below expect."""
    combined = load_run_legs(out_dir)
    return {
        "trades": _to_df(combined["trades"]),
        "snapshots": _to_df(combined["snapshots"]),
        "agent_state": _to_df(combined["agent_states"]),
        "meta": combined["meta"],
    }


# ---------------------------------------------------------------------------
# Table 6: Liquidity metrics by human share (Section 5.1)
# ---------------------------------------------------------------------------

def liquidity_metrics(run: Dict[str, Any], asset_ids: Sequence[int], session_seconds: float) -> Dict[str, float]:
    trades, snapshots = run["trades"], run["snapshots"]
    session_minutes = session_seconds / 60.0
    vols, spreads, bid_depths, ask_depths = [], [], [], []
    for a in asset_ids:
        t = trades[trades["asset_id"] == a]
        vols.append(t["qty"].sum() / session_minutes if session_minutes > 0 else np.nan)
        spread_stats = quoted_spread_stats(snapshots, a)
        spreads.append(spread_stats["mean_rel_spread"])  # == mean(2*(ask-bid)/(ask+bid)), the paper's formula
        d = depth_stats(snapshots, a)
        bid_depths.append(d["mean_bid_depth"])
        ask_depths.append(d["mean_ask_depth"])
    return {
        "volume_per_min": float(np.nanmean(vols)),
        "mean_rel_spread": float(np.nanmean(spreads)),
        "mean_bid_depth": float(np.nanmean(bid_depths)),
        "mean_ask_depth": float(np.nanmean(ask_depths)),
    }


# ---------------------------------------------------------------------------
# Table 7: Stability metrics by human share (Section 5.2)
# ---------------------------------------------------------------------------

def _count_flash_events(prices: np.ndarray, window_last30s_count: int, threshold: float, direction: str) -> int:
    """Literal reproduction of the paper's formula: for each trade i, compare
    price_i against the running max/min price over the preceding window (the
    paper calls it "last 30s"; implemented here as the preceding N trades,
    since trade density varies by composition and the paper's own count is
    per-trade, not per-wall-clock-second). Counts every trade where the
    condition holds -- NOT deduplicated into distinct events, matching the
    paper's own (raw, trade-indexed) definition exactly."""
    n = len(prices)
    count = 0
    for i in range(n):
        lo = max(0, i - window_last30s_count)
        window = prices[lo:i] if i > lo else prices[lo : i + 1]
        if window.size == 0:
            continue
        ref = window.max() if direction == "crash" else window.min()
        if ref <= 0:
            continue
        if direction == "crash":
            moved = (ref - prices[i]) / ref
        else:
            moved = (prices[i] - ref) / ref
        if moved > threshold:
            count += 1
    return count


def stability_metrics(
    run: Dict[str, Any],
    asset_ids: Sequence[int],
    trades_per_30s: int = 30,
    threshold: float = 0.05,
) -> Dict[str, float]:
    """`trades_per_30s` approximates the paper's "last 30s" trailing window as
    a trade count (see `_count_flash_events`'s doc comment); pass a value
    matched to your run's trade density for a closer match if needed.
    Aggregate equity growth isn't included here -- see `growth_by_bot_type`
    for the paper's actual Table 8 figures, broken down per bot type."""
    trades = run["trades"]

    log_rets: List[float] = []
    n_crashes = n_growths = 0
    for a in asset_ids:
        prices = trades[trades["asset_id"] == a].sort_values("time")["price"].to_numpy(dtype=float)
        if prices.size > 1:
            log_rets.extend(np.abs(np.diff(np.log(prices))).tolist())
        if prices.size > 0:
            n_crashes += _count_flash_events(prices, trades_per_30s, threshold, "crash")
            n_growths += _count_flash_events(prices, trades_per_30s, threshold, "growth")

    mean_volatility = float(np.mean(log_rets)) if log_rets else float("nan")
    ratio = (n_growths / n_crashes) if n_crashes > 0 else float("nan")

    return {
        "mean_volatility": mean_volatility,
        "n_flash_crashes": n_crashes,
        "n_flash_growths": n_growths,
        "ratio": ratio,
    }


def _equity_growth_pct(agent_state: pd.DataFrame, bot_type: str) -> float:
    """(Sum end net worth / Sum start net worth - 1) * 100, over agents of
    `bot_type` -- the paper's equity-growth formula (Section 5.2), reported
    as the % change the paper's tables show, not the raw ratio."""
    df = agent_state[agent_state["bot_type"] == bot_type]
    start = df["initial_net_worth"].sum()
    end = df["equity"].sum()
    if start == 0:
        return float("nan")
    return float((end / start - 1.0) * 100.0)


# ---------------------------------------------------------------------------
# Table 8: Relative net worth growth by human share and bot type (Section 5.3)
# ---------------------------------------------------------------------------

GROWTH_BOT_TYPES = ["human", "market_maker", "momentum", "dca", "mean_reversion"]


def growth_by_bot_type(run: Dict[str, Any]) -> Dict[str, float]:
    agent_state = run["agent_state"]
    return {bt: _equity_growth_pct(agent_state, bt) for bt in GROWTH_BOT_TYPES}


# ---------------------------------------------------------------------------
# Tables 9 & 11: price impact of the institutional-seller dump (Section 6)
# ---------------------------------------------------------------------------

def dump_fall_pct(run: Dict[str, Any], asset_ids: Sequence[int], shock_time_secs: float) -> float:
    """Fall % for one run: price at the post-shock trough vs. price at the
    moment the seller activates, averaged across the seller's target assets
    -- the quantity Tables 9 and 11 both report (negative = price fell,
    positive = price rose)."""
    trades = run["trades"]
    falls = []
    for a in asset_ids:
        t = trades[trades["asset_id"] == a].sort_values("time")
        if t.empty:
            continue
        pre = t[t["time"] <= shock_time_secs]
        post = t[t["time"] > shock_time_secs]
        if pre.empty or post.empty:
            continue
        price_at_shock = float(pre["price"].iloc[-1])
        trough_price = float(post["price"].min())
        if price_at_shock == 0:
            continue
        falls.append((trough_price - price_at_shock) / price_at_shock * 100.0)
    return float(np.mean(falls)) if falls else float("nan")


# ---------------------------------------------------------------------------
# PDF rendering -- one table per page, matplotlib only (already a dependency)
# ---------------------------------------------------------------------------

def render_tables_pdf(
    tables: List[Tuple[str, List[str], List[List[Any]]]],
    out_path: str,
    title: Optional[str] = None,
) -> str:
    """`tables`: list of (table_title, column_headers, rows). Writes one PDF
    with a cover page (if `title` given) followed by one page per table,
    paginating any table with more rows than fit on one page."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows_per_page = 28

    with PdfPages(out_path) as pdf:
        if title:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.5, title, ha="center", va="center", fontsize=18, wrap=True)
            pdf.savefig(fig)
            plt.close(fig)

        for table_title, headers, rows in tables:
            n_pages = max(1, -(-len(rows) // rows_per_page))
            for page in range(n_pages):
                chunk = rows[page * rows_per_page : (page + 1) * rows_per_page]
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis("off")
                page_suffix = f" (page {page + 1}/{n_pages})" if n_pages > 1 else ""
                ax.set_title(table_title + page_suffix, fontsize=13, pad=20)
                tbl = ax.table(cellText=chunk, colLabels=headers, loc="center", cellLoc="center")
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8)
                tbl.scale(1, 1.3)
                tbl.auto_set_column_width(col=list(range(len(headers))))
                pdf.savefig(fig)
                plt.close(fig)
    return out_path


def fmt(x: Any, pct: bool = False, decimals: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if pct:
        return f"{x:+.3f}%" if isinstance(x, float) else f"{x}"
    if isinstance(x, float):
        return f"{x:,.{decimals}f}"
    return str(x)
