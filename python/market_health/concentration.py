"""Concentration/fairness metrics: report section 7 (Herfindahl-Hirschman Index)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _volume_by(trades: pd.DataFrame, key: str) -> pd.Series:
    """Total notional volume attributed to each value of `key` ('maker_agent_id',
    'taker_agent_id', 'maker_type', or 'taker_type'), counting maker and taker legs
    together under a unified column name."""
    notional = trades["price"] * trades["qty"]
    maker_col = "maker_agent_id" if key.endswith("agent_id") else "maker_type"
    taker_col = "taker_agent_id" if key.endswith("agent_id") else "taker_type"
    by_maker = notional.groupby(trades[maker_col]).sum()
    by_taker = notional.groupby(trades[taker_col]).sum()
    return by_maker.add(by_taker, fill_value=0.0)


def hhi(trades: pd.DataFrame, by: str = "agent") -> dict:
    """Herfindahl-Hirschman Index over volume share, `by` either 'agent' (per
    individual agent id) or 'bot_type' (per strategy category). HHI in [0, 1] here
    (fractional shares, not the x10000 convention); higher = more concentrated ->
    less "fair"/competitive, structurally more fragile (report section 7)."""
    if trades.empty:
        return {"hhi": np.nan, "n_participants": 0}
    key = "maker_agent_id" if by == "agent" else "maker_type"
    vol = _volume_by(trades, key)
    total = vol.sum()
    if total <= 0:
        return {"hhi": np.nan, "n_participants": 0}
    shares = vol / total
    return {"hhi": float((shares**2).sum()), "n_participants": int((vol > 0).sum())}


def top_n_share(trades: pd.DataFrame, n: int = 1, by: str = "agent") -> float:
    """Volume share of the top `n` participants -- a more interpretable companion to
    HHI (e.g. spec 5.8's MEV-concentration note: "a single bot can account for a very
    large share")."""
    if trades.empty:
        return np.nan
    key = "maker_agent_id" if by == "agent" else "maker_type"
    vol = _volume_by(trades, key).sort_values(ascending=False)
    total = vol.sum()
    if total <= 0:
        return np.nan
    return float(vol.head(n).sum() / total)
