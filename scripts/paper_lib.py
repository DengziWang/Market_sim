"""Shared scenario-building helpers for the two paper-reproduction programs
(`human_pct_sweep.py`, `dump_sim_sweep.py`). Builds the baseline market
composition described in "The Role of Humans on Market Quality and
Flash-Crash Dynamics in a Simulated Crypto Market": five identical submarkets,
a human share that varies while the relative mix of the four bot types is
held constant, and (for the dump experiment) a pair of institutional_seller
agents reproducing the May 6, 2010 flash-crash mechanism against two of those
assets.

This is a clean reference implementation of the paper's experimental design,
not a byte-for-byte replay of its original (unpublished) run scripts -- the
underlying engine (sim_core) has itself moved on since the paper's numbers
were produced (e.g. institutional_seller's wake_interval_secs default is now
0.1s, not the 0.5s the paper's methodology section describes), so this code
follows the engine's current, documented defaults rather than stale values.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Relative composition of the four bot types among non-human agents, held
# fixed across every human-share point -- taken from the paper's own 60%
# human-share baseline (Table 4): market_maker=157, momentum=63, dca=42,
# mean_reversion=138 out of 400 non-human agents.
BOT_TYPE_SHARE = {
    "market_maker": 157 / 400,
    "momentum": 63 / 400,
    "dca": 42 / 400,
    "mean_reversion": 138 / 400,
}

# Each bot type's subtypes, split as evenly as possible across its group's
# headcount -- matches the paper's Methodology section (3.1), which gives
# each bot type this exact set of subtypes.
SUBTYPES = {
    "market_maker": ["avellaneda_stoikov", "fixed_offset", "inventory_skew", "adaptive_spread"],
    "momentum": ["ma_crossover", "breakout_donchian", "rsi", "ml_online"],
    "mean_reversion": ["bollinger", "ou_fair_value", "zscore"],
    "dca": ["time_based", "value_averaging", "dip_buying"],
}

# Initial cash ranges per bot type, matching the paper's description of
# market-maker ($30k-120k) and OU-fair-value ($3M-8M, a "value fund"
# archetype) capitalization; other subtypes use a retail/mid-size default.
CASH_RANGE = {
    "market_maker": (30_000.0, 120_000.0),
    "momentum": (5_000.0, 50_000.0),
    "dca": (2_000.0, 20_000.0),
    "mean_reversion_ou_fair_value": (3_000_000.0, 8_000_000.0),
    "mean_reversion_default": (5_000.0, 50_000.0),
    "human": (1_000.0, 20_000.0),
}

N_ASSETS = 5


def identical_assets(n_assets: int = N_ASSETS) -> List[Dict[str, Any]]:
    """Five (by default) identically-configured assets -- same tick size,
    starting price, volatility and mean-reversion speed -- so that every
    submarket is an interchangeable replicate rather than a distinct
    real-crypto profile."""
    return [
        {
            "symbol": f"Stock{i + 1}",
            "tick_size": 0.01,
            "initial_price": 1000.0,
            "fundamental_vol": 0.8,
            "fundamental_theta": 0.1,
        }
        for i in range(n_assets)
    ]


def _split_evenly(total: int, n_parts: int) -> List[int]:
    """Splits `total` into `n_parts` non-negative integers as evenly as
    possible (earlier parts get the remainder, e.g. 10 into 3 -> [4, 3, 3])."""
    base, rem = divmod(total, n_parts)
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def build_population(n_total: int, human_pct: float, human_variant: str = "retail") -> List[Dict[str, Any]]:
    """Builds the population group list for one human-share composition.
    `n_total` agents total; `human_pct` (0-100) of them are human, the rest
    split across market_maker/momentum/dca/mean_reversion in the fixed
    `BOT_TYPE_SHARE` ratio, each further split evenly across that bot type's
    subtypes."""
    n_human = round(n_total * human_pct / 100.0)
    n_bots = n_total - n_human
    groups: List[Dict[str, Any]] = []

    if n_human > 0:
        groups.append(
            {
                "bot_type": "human",
                "variant": human_variant,
                "count": n_human,
                "initial_cash_range": list(CASH_RANGE["human"]),
            }
        )

    remaining = n_bots
    bot_types = list(BOT_TYPE_SHARE.keys())
    for idx, bot_type in enumerate(bot_types):
        if idx == len(bot_types) - 1:
            count = remaining  # last type absorbs any rounding remainder
        else:
            count = round(n_bots * BOT_TYPE_SHARE[bot_type])
            count = min(count, remaining)
        remaining -= count
        if count <= 0:
            continue
        for subtype, sub_count in zip(SUBTYPES[bot_type], _split_evenly(count, len(SUBTYPES[bot_type]))):
            if sub_count <= 0:
                continue
            if bot_type == "mean_reversion" and subtype == "ou_fair_value":
                cash_range = CASH_RANGE["mean_reversion_ou_fair_value"]
            elif bot_type == "mean_reversion":
                cash_range = CASH_RANGE["mean_reversion_default"]
            else:
                cash_range = CASH_RANGE[bot_type]
            groups.append(
                {
                    "bot_type": bot_type,
                    "variant": subtype,
                    "count": sub_count,
                    "initial_cash_range": list(cash_range),
                }
            )
    return groups


def institutional_seller_groups(
    target_assets: List[int],
    total_qty: float = 1.0,
    participation_rate: float = 0.09,
    trailing_window_secs: float = 15.0,
    start_time_secs: float = 0.0,
    wake_interval_secs: float = 0.1,
) -> List[Dict[str, Any]]:
    """One institutional_seller group per target asset, reproducing the May 6
    2010 flash-crash's pacing mechanism (9% of trailing volume, no price/time
    limit) against each -- the paper's Section 6 targets two assets at once
    ("both Stock 1 and Stock 2")."""
    return [
        {
            "bot_type": "institutional_seller",
            "variant": "pov",
            "count": 1,
            "target_asset": asset_idx,
            "params": {
                "total_qty": total_qty,
                "participation_rate": participation_rate,
                "trailing_window_secs": trailing_window_secs,
                "start_time_secs": start_time_secs,
                "wake_interval_secs": wake_interval_secs,
            },
        }
        for asset_idx in target_assets
    ]


def build_scenario(
    *,
    human_pct: float,
    seed: int,
    session_seconds: float,
    n_total: int = 1000,
    n_assets: int = N_ASSETS,
    extra_groups: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assembles a full scenario config dict (matching sim_core's
    ScenarioConfig TOML schema) for one (human_pct, seed) point. Market/fees/
    latency/mempool/liquidation sections are left unset so the engine's own
    documented defaults apply (including the 0.15 circuit-breaker band the
    paper's Section 6 relies on)."""
    groups = build_population(n_total, human_pct)
    if extra_groups:
        groups = groups + extra_groups
    return {
        "assets": identical_assets(n_assets),
        "run": {"seed": seed, "session_seconds": session_seconds},
        # `out_dir` here is required by the TOML schema but unused in
        # practice: run_scenario_in_legs/export_run write to their own
        # `out_dir` argument from the Python side, not this field.
        "export": {"out_dir": "unused", "write_snapshots": True},
        "population": {"groups": groups},
    }
