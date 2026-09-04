"""Report section 8: the composite market-health score, plus the per-run metric
collection that feeds it. Deliberately follows the report's own recipe rather than
inventing a different one:

  1. compute every metric per run (`compute_run_metrics`)
  2. z-normalize each metric *relative to the batch* (never in isolation -- a score
     of "0.3" only means something next to other runs)
  3. orient sign so higher-z = healthier
  4. average within each of the 6 report dimensions, then take a configurable
     weighted sum across dimensions (default: equal weight)
  5. `composite_scores` always returns the per-dimension scores alongside the total,
     precisely so the composite is never read in isolation (report section 8, point 4)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from market_health import concentration, crisis, efficiency, liquidity, toxicity, volatility
from market_health import wash_trading as washmod

# Metric name -> how a *higher raw value* relates to health:
#   +1      : higher is healthier
#   -1      : lower is healthier
#   "dist1" : closer to 1 is healthier (distance-from-1 penalized)
#   "dist0" : closer to 0 is healthier
#   absent  : diagnostic only, not folded into the composite score
METRIC_DIRECTION: Dict[str, Any] = {
    "liquidity.mean_rel_spread": -1,
    "liquidity.median_rel_spread": -1,
    "liquidity.mean_bid_depth": 1,
    "liquidity.mean_ask_depth": 1,
    "liquidity.amihud": -1,
    "liquidity.kyle_lambda": -1,
    "liquidity.mean_recovery_secs": -1,
    "volatility.jump_variance_fraction": -1,
    "efficiency.variance_ratio": "dist1",
    "efficiency.return_autocorr_lag1": "dist0",
    "efficiency.mean_abs_dev_frac": -1,
    "toxicity.mean_abs_ofi": -1,
    "toxicity.mean_vpin": -1,
    "toxicity.cancel_to_trade_ratio": -1,
    "wash.chi2": -1,
    "wash.round_fraction": -1,
    "wash.ground_truth_share": -1,
    "concentration.hhi_agent": -1,
    "concentration.hhi_bot_type": -1,
    "concentration.top1_share_agent": -1,
    "concentration.top5_share_agent": -1,
    "crisis.n_flash_crashes": -1,
    "crisis.mean_depth_frac": -1,
    "crisis.mean_recovery_secs": -1,
    "crisis.unrecovered_count": -1,
}

# Metric-name prefix -> which of the report's 6 dimensions it belongs to. Flash-crash
# ("crisis") metrics are folded into "toxicity" (report section 5 treats fragility and
# crisis-proneness as one dimension); "wash" -> "authenticity" (section 6).
DIMENSION_OF: Dict[str, str] = {
    "liquidity": "liquidity",
    "volatility": "volatility",
    "efficiency": "efficiency",
    "toxicity": "toxicity",
    "crisis": "toxicity",
    "wash": "authenticity",
    "concentration": "concentration",
}

DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "liquidity": 1.0,
    "volatility": 1.0,
    "efficiency": 1.0,
    "toxicity": 1.0,
    "authenticity": 1.0,
    "concentration": 1.0,
}


def compute_run_metrics(run: Dict[str, Any], asset_ids: Optional[Sequence[int]] = None) -> Dict[str, float]:
    """All metrics for one already-loaded run (as returned by
    `market_sim.export.load_run` / `market_sim.SimulationRun.result`), per-asset
    metrics averaged across assets into one flat dict. Use the individual
    `market_health.<module>` functions directly for per-asset breakdowns."""
    trades = run["trades"]
    snapshots = run.get("orderbook_snapshots", run.get("snapshots", pd.DataFrame()))
    order_events = run["order_events"]
    agent_state = run["agent_state"]

    if asset_ids is None:
        asset_ids = sorted(trades["asset_id"].unique()) if not trades.empty else []

    rows = []
    for aid in asset_ids:
        m: Dict[str, float] = {}
        if not snapshots.empty:
            m.update({f"liquidity.{k}": v for k, v in liquidity.quoted_spread_stats(snapshots, aid).items()})
            m.update({f"liquidity.{k}": v for k, v in liquidity.depth_stats(snapshots, aid).items()})
            m.update({f"liquidity.{k}": v for k, v in liquidity.resilience(snapshots, aid).items()})
        m["liquidity.amihud"] = liquidity.amihud_illiquidity(trades, aid)
        m["liquidity.kyle_lambda"] = liquidity.kyle_lambda(trades, aid)

        price_series = volatility.trade_price_series(trades, aid)
        m["volatility.realized"] = volatility.realized_volatility(price_series)
        clustering = volatility.volatility_clustering_autocorr(price_series)
        m["volatility.clustering_lag1"] = float(clustering.iloc[0]) if len(clustering) else np.nan
        m.update({f"volatility.{k}": v for k, v in volatility.jump_fraction(price_series).items()})

        m["efficiency.variance_ratio"] = efficiency.variance_ratio(price_series)
        m["efficiency.return_autocorr_lag1"] = efficiency.return_autocorrelation(price_series, lag=1)
        if not snapshots.empty:
            m.update({f"efficiency.{k}": v for k, v in efficiency.fundamental_deviation(snapshots, aid).items()})

        ofi = toxicity.order_flow_imbalance(trades, aid)
        m["toxicity.mean_abs_ofi"] = float(ofi.abs().mean()) if len(ofi) else np.nan
        vp = toxicity.vpin(trades, aid)
        m["toxicity.mean_vpin"] = float(vp.mean()) if len(vp) else np.nan
        m["toxicity.cancel_to_trade_ratio"] = toxicity.cancel_to_trade_ratio(order_events, aid)

        sizes = trades.loc[trades.asset_id == aid, "qty"]
        for fn in (washmod.benford_test, washmod.round_number_clustering, washmod.pareto_tail_check):
            m.update({f"wash.{k}": v for k, v in fn(sizes).items() if k not in ("n", "n_tail")})

        m.update({f"crisis.{k}": v for k, v in crisis.flash_crash_summary(price_series).items()})
        m["_asset_id"] = aid
        rows.append(m)

    df = pd.DataFrame(rows)
    out = df.drop(columns=["_asset_id"], errors="ignore").mean(numeric_only=True).to_dict() if not df.empty else {}

    out["wash.ground_truth_share"] = washmod.ground_truth_wash_volume_share(trades, agent_state) if not trades.empty else np.nan
    out["concentration.hhi_agent"] = concentration.hhi(trades, by="agent")["hhi"]
    out["concentration.hhi_bot_type"] = concentration.hhi(trades, by="bot_type")["hhi"]
    out["concentration.top1_share_agent"] = concentration.top_n_share(trades, n=1, by="agent")
    out["concentration.top5_share_agent"] = concentration.top_n_share(trades, n=5, by="agent")
    return out


def composite_scores(
    batch_metrics: List[Dict[str, float]],
    run_names: Optional[List[str]] = None,
    dimension_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Batch (>= 2 runs, ideally many more) of `compute_run_metrics(...)` outputs ->
    a DataFrame with one column per report dimension (z-scored, oriented so
    higher=healthier) plus `composite_score` (the weighted average of those). A
    dimension with too little variance across the batch to compute (e.g. a
    single-run "batch") is silently dropped from that dimension's average rather than
    poisoning it with NaN -- but always inspect the per-dimension columns, not just
    `composite_score` (report section 8, point 4)."""
    weights = {**DEFAULT_DIMENSION_WEIGHTS, **(dimension_weights or {})}
    df = pd.DataFrame(batch_metrics)
    if run_names is not None:
        df.index = run_names

    z_by_dimension: Dict[str, List[pd.Series]] = {dim: [] for dim in weights}
    for col in df.columns:
        prefix = col.split(".")[0]
        dim = DIMENSION_OF.get(prefix)
        direction = METRIC_DIRECTION.get(col)
        if dim is None or direction is None or dim not in z_by_dimension:
            continue
        vals = df[col].astype(float)
        if direction == -1:
            transformed = -vals
        elif direction == 1:
            transformed = vals
        elif direction == "dist1":
            transformed = -(vals - 1.0).abs()
        elif direction == "dist0":
            transformed = -vals.abs()
        else:
            continue
        std = transformed.std(skipna=True)
        if not std or np.isnan(std):
            continue
        z_by_dimension[dim].append((transformed - transformed.mean()) / std)

    dim_scores = pd.DataFrame(index=df.index)
    for dim, zs in z_by_dimension.items():
        if zs:
            dim_scores[dim] = pd.concat(zs, axis=1).mean(axis=1)

    present_weight = sum(weights[d] for d in dim_scores.columns)
    if present_weight > 0:
        dim_scores["composite_score"] = sum(dim_scores[d] * weights[d] for d in dim_scores.columns) / present_weight
    else:
        dim_scores["composite_score"] = np.nan
    return dim_scores
