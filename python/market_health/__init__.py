"""market_health: computes the market-health metrics from `docs/marktgesundheit_bericht.md`
directly on the parquet exports from spec section 7 -- never re-runs the simulation.

    from market_health import compute_run_metrics, composite_scores
    from market_sim import load_run

    run = load_run("runs/realistic_mix/seed_1")
    metrics = compute_run_metrics(run)

For a whole experiment batch, see `market_health.batch_compare`.
"""
from market_health.score import compute_run_metrics, composite_scores, save_metrics, load_metrics, DEFAULT_DIMENSION_WEIGHTS
from market_health.batch_compare import load_and_score_runs, compare_compositions, plot_composition_comparison

__all__ = [
    "compute_run_metrics",
    "composite_scores",
    "save_metrics",
    "load_metrics",
    "DEFAULT_DIMENSION_WEIGHTS",
    "load_and_score_runs",
    "compare_compositions",
    "plot_composition_comparison",
]
