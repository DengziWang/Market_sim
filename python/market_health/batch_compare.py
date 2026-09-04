"""Spec section 9's last bullet: turn a whole experiment batch (many runs across
different bot compositions x seeds) into a comparison table/plot of market health as
a function of bot composition.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from market_sim.export import load_run
from market_health.score import compute_run_metrics, composite_scores


def load_and_score_runs(run_dirs: Sequence[Tuple[str, int, str]]) -> pd.DataFrame:
    """`run_dirs`: (composition_name, seed, path) triples -- exactly what
    `ExperimentPlan.all_run_dirs()` returns. Loads each run's export, computes its
    metrics, and returns one row per (composition, seed) with all metrics as columns
    plus `composition` and `seed`."""
    rows = []
    for name, seed, path in run_dirs:
        if not os.path.exists(os.path.join(path, "run_metadata.json")):
            continue
        run = load_run(path)
        metrics = compute_run_metrics(run)
        metrics["composition"] = name
        metrics["seed"] = seed
        rows.append(metrics)
    return pd.DataFrame(rows)


def compare_compositions(metrics_df: pd.DataFrame, dimension_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """`metrics_df`: output of `load_and_score_runs`. Computes the composite/
    dimension scores batch-wide (z-normalized across *all* rows, per report section
    8), then aggregates to one row per composition: mean +/- 95%-CI (normal
    approximation, fine for the spec's 20-50-seeds-per-composition default) for each
    dimension and the composite score, plus n_seeds actually available."""
    meta_cols = ["composition", "seed"]
    metric_cols = [c for c in metrics_df.columns if c not in meta_cols]
    scored = composite_scores(metrics_df[metric_cols].to_dict("records"), run_names=metrics_df.index)
    scored["composition"] = metrics_df["composition"].values

    out_rows = []
    for name, g in scored.groupby("composition"):
        row: Dict[str, Any] = {"composition": name, "n_seeds": len(g)}
        for col in [c for c in scored.columns if c != "composition"]:
            vals = g[col].dropna()
            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_ci95"] = np.nan
                continue
            mean = vals.mean()
            sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            row[f"{col}_mean"] = float(mean)
            row[f"{col}_ci95"] = float(1.96 * sem)
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values("composite_score_mean", ascending=False)


def plot_composition_comparison(comparison: pd.DataFrame, dimensions: Optional[List[str]] = None):
    """Bar chart with 95%-CI error bars: market health (composite + each dimension)
    as a function of bot composition -- spec section 9's required comparison plot."""
    import matplotlib.pyplot as plt

    dims = dimensions or [c[:-5] for c in comparison.columns if c.endswith("_mean") and c != "composite_score_mean"]
    cols = dims + ["composite_score"]
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    order = comparison.sort_values("composite_score_mean", ascending=True)["composition"]
    for ax, col in zip(axes, cols):
        d = comparison.set_index("composition").loc[order]
        ax.barh(d.index, d[f"{col}_mean"], xerr=d[f"{col}_ci95"], color="#2a6f97")
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_title(col.replace("_", " "))
    fig.tight_layout()
    return fig
