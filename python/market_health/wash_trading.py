"""Wash-trading / volume-authenticity metrics: report section 6, following Cong, Li,
Tang & Yang's three statistical signatures (Benford's Law on first digit,
round-number clustering, Pareto tail), plus a ground-truth check unique to a
simulation: agent identity (and hence true bot_type) is known, so the statistical
proxies can be validated against how much volume actually came from
`bot_type == "wash_trading"` agents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Benford's Law expected first-significant-digit distribution, digits 1..9.
_BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def _first_significant_digit(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    x = x[x > 0]
    exponent = np.floor(np.log10(x))
    mantissa = x / (10.0**exponent)
    return np.floor(mantissa).astype(int).clip(1, 9)


def benford_test(trade_sizes: pd.Series) -> dict:
    """Chi-square goodness-of-fit of the first-significant-digit distribution of
    trade sizes against Benford's Law. Authentic trade sizes tend to follow Benford;
    fabricated/wash trades tend to deviate (report section 6, Cong et al.)."""
    sizes = trade_sizes.dropna().to_numpy()
    sizes = sizes[sizes > 0]
    if len(sizes) < 30:
        return {"chi2": np.nan, "p_value": np.nan, "n": len(sizes)}
    digits = _first_significant_digit(sizes)
    observed = np.array([(digits == d).sum() for d in range(1, 10)])
    expected = _BENFORD * len(sizes)
    chi2, p = stats.chisquare(observed, expected)
    return {"chi2": float(chi2), "p_value": float(p), "n": int(len(sizes))}


def round_number_clustering(trade_sizes: pd.Series, round_to: float = 1.0, tolerance: float = 1e-6) -> dict:
    """Fraction of trades whose size is (near-)exactly a multiple of `round_to`, vs.
    the fraction that would be expected under a continuous size distribution (~0 for
    a truly continuous underlying process). A high observed fraction is a classic
    wash-trading/manipulation signature."""
    sizes = trade_sizes.dropna().to_numpy()
    if len(sizes) == 0:
        return {"round_fraction": np.nan, "n": 0}
    remainder = np.abs(sizes / round_to - np.round(sizes / round_to))
    is_round = remainder < tolerance
    return {"round_fraction": float(is_round.mean()), "n": int(len(sizes))}


def pareto_tail_check(trade_sizes: pd.Series, tail_quantile: float = 0.95) -> dict:
    """Fits a power-law (Pareto) tail index (Hill estimator) to the upper
    `1 - tail_quantile` of trade sizes. Authentic markets typically show a
    heavy/power-law tail (small alpha, e.g. 2-4); a wash-trading population using a
    narrow set of fixed sizes will instead show a thin/near-degenerate tail (the
    Hill estimator becomes unstable or alpha is very large)."""
    sizes = trade_sizes.dropna().to_numpy()
    sizes = sizes[sizes > 0]
    if len(sizes) < 50:
        return {"hill_alpha": np.nan, "n_tail": 0}
    threshold = np.quantile(sizes, tail_quantile)
    tail = np.sort(sizes[sizes >= threshold])
    if len(tail) < 10 or threshold <= 0:
        return {"hill_alpha": np.nan, "n_tail": int(len(tail))}
    log_ratios = np.log(tail / threshold)
    log_ratios = log_ratios[log_ratios > 0]
    if len(log_ratios) < 5:
        return {"hill_alpha": np.nan, "n_tail": int(len(tail))}
    alpha = 1.0 / np.mean(log_ratios)
    return {"hill_alpha": float(alpha), "n_tail": int(len(tail))}


def ground_truth_wash_volume_share(trades: pd.DataFrame, agent_state: pd.DataFrame) -> float:
    """The one check only a simulation can do: what fraction of total notional
    volume is *known* (by agent identity) to have come from a wash_trading-type
    agent, counting both maker and taker legs. Compare this against what the
    statistical proxies above suggest to gauge their detection power."""
    if trades.empty:
        return np.nan
    type_map = agent_state.drop_duplicates("agent_id").set_index("agent_id")["bot_type"]
    notional = trades["price"] * trades["qty"]
    maker_is_wash = trades["maker_agent_id"].map(type_map).eq("wash_trading")
    taker_is_wash = trades["taker_agent_id"].map(type_map).eq("wash_trading")
    wash_notional = notional[maker_is_wash].sum() + notional[taker_is_wash].sum()
    total_notional = notional.sum() * 2  # each trade counted once per side, consistent with the numerator
    return float(wash_notional / total_notional) if total_notional > 0 else np.nan
