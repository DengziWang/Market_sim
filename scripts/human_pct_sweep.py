#!/usr/bin/env python3
"""Human-share composition sweep -- reproduces the paper's Section 5 experiment
("Impact of human population on the market"): the human share is varied from
0% to 100% of a fixed-size population while the relative mix of the four bot
types (market_maker, momentum, dca, mean_reversion) is held constant (see
paper_lib.BOT_TYPE_SHARE), across five identical submarkets. Every (human
share, seed) run is saved in full to its own directory.

Usage (defaults to a quick, modest-size sweep so it runs out of the box;
bump --session-seconds and --n-total for a closer match to the paper's own
1000-agent, 100,000-second runs):

    python scripts/human_pct_sweep.py --runs 3

    # Full paper-scale sweep, one seed per point:
    python scripts/human_pct_sweep.py --human-pcts 0,10,20,30,40,50,60,70,80,90,100 \\
        --runs 1 --n-total 1000 --session-seconds 100000

Each run's full trade/order-event/snapshot/agent-state history is written to
<out-dir>/human_<pct>pct/seed_<seed>/leg_0001/ (see market_sim.load_run_legs
to read one back). Re-running with the same --out-dir resumes any
interrupted (human_pct, seed) runs rather than redoing them.

Once every run is done, a PDF report (<out-dir>/report.pdf) is generated with
the same tables the paper itself reports for this experiment (Section 5):
liquidity metrics, stability metrics, and relative net worth growth by bot
type, each by human share (seeds averaged per point).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_lib  # noqa: E402
import paper_metrics  # noqa: E402

from market_sim.runner import run_scenario_in_legs  # noqa: E402


def parse_pcts(s: str) -> list:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def build_report(run_dirs: dict, out_dir: str, flash_window_trades: int) -> str:
    """Loads every (human_pct, seed) run back from disk, computes the paper's
    Section-5 metrics for each, averages across seeds per human-share point,
    and writes <out_dir>/report.pdf with the same three tables the paper
    reports: liquidity metrics, stability metrics, and relative net worth
    growth by bot type (each by human share)."""
    asset_ids = list(range(paper_lib.N_ASSETS))
    liquidity_rows, stability_rows, growth_rows = [], [], []

    for pct in sorted(run_dirs.keys()):
        liq_acc, stab_acc, growth_acc = [], [], {bt: [] for bt in paper_metrics.GROWTH_BOT_TYPES}
        for out in run_dirs[pct]:
            run = paper_metrics.load_run_tables(out)
            session_seconds = run["meta"].get("actual_session_seconds", 0.0)
            liq_acc.append(paper_metrics.liquidity_metrics(run, asset_ids, session_seconds))
            stab_acc.append(paper_metrics.stability_metrics(run, asset_ids, trades_per_30s=flash_window_trades))
            g = paper_metrics.growth_by_bot_type(run)
            for bt in paper_metrics.GROWTH_BOT_TYPES:
                growth_acc[bt].append(g[bt])

        def mean_of(key, acc=liq_acc):
            import numpy as np
            return float(np.nanmean([d[key] for d in acc]))

        liquidity_rows.append([
            f"{pct:g}%",
            paper_metrics.fmt(mean_of("volume_per_min")),
            paper_metrics.fmt(mean_of("mean_rel_spread"), decimals=5),
            paper_metrics.fmt(mean_of("mean_bid_depth")),
            paper_metrics.fmt(mean_of("mean_ask_depth")),
        ])

        import numpy as np
        stability_rows.append([
            f"{pct:g}%",
            paper_metrics.fmt(float(np.nanmean([d["mean_volatility"] for d in stab_acc])), decimals=5),
            paper_metrics.fmt(float(np.nanmean([d["n_flash_crashes"] for d in stab_acc])), decimals=1),
            paper_metrics.fmt(float(np.nanmean([d["n_flash_growths"] for d in stab_acc])), decimals=1),
            paper_metrics.fmt(float(np.nanmean([d["ratio"] for d in stab_acc])), decimals=4),
        ])

        growth_rows.append([f"{pct:g}%"] + [paper_metrics.fmt(float(np.nanmean(growth_acc[bt])), pct=True) for bt in paper_metrics.GROWTH_BOT_TYPES])

    tables = [
        ("Liquidity metrics by human share (mean per asset)",
         ["Human %", "Volume/min", "Mean rel. spread", "Mean bid depth", "Mean ask depth"], liquidity_rows),
        ("Stability metrics by human share",
         ["Human %", "Mean volatility", "Crashes (-5%)", "Growths (+5%)", "Ratio"], stability_rows),
        ("Relative net worth growth by human share and bot type",
         ["Human %"] + paper_metrics.GROWTH_BOT_TYPES, growth_rows),
    ]
    out_path = os.path.join(out_dir, "report.pdf")
    return paper_metrics.render_tables_pdf(tables, out_path, title="Human-Share Sweep Report\n(Section 5 tables)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--human-pcts", type=parse_pcts, default="0,10,20,30,40,50,60,70,80,90,100",
                    help="comma-separated human-share percentages to sweep (default: 0..100 in steps of 10)")
    p.add_argument("--runs", type=int, default=1, help="number of seeds (independent runs) per human-share point (default: 1)")
    p.add_argument("--first-seed", type=int, default=1, help="first seed number; seeds used are first-seed..first-seed+runs-1 (default: 1)")
    p.add_argument("--n-total", type=int, default=1000, help="total agent population size (default: 1000, matching the paper)")
    p.add_argument("--session-seconds", type=float, default=20_000.0,
                    help="simulated seconds per run (default: 20,000; the paper itself used 100,000)")
    p.add_argument("--leg-seconds", type=float, default=None,
                    help="split each run into legs of this many seconds (default: min(session-seconds, 20000), bounds memory use)")
    p.add_argument("--out-dir", default="runs/human_pct_sweep", help="base output directory (default: runs/human_pct_sweep)")
    p.add_argument("--no-report", action="store_true", help="skip generating the PDF report after the sweep finishes")
    p.add_argument("--flash-window-trades", type=int, default=30,
                    help="trade-count window approximating the paper's 'last 30s' flash-crash/growth reference window (default: 30)")
    args = p.parse_args()

    leg_seconds = args.leg_seconds or min(args.session_seconds, 20_000.0)
    n_legs = max(1, round(args.session_seconds / leg_seconds))
    seeds = list(range(args.first_seed, args.first_seed + args.runs))

    total_points = len(args.human_pcts) * len(seeds)
    print(f"Human-share sweep: {len(args.human_pcts)} composition(s) x {len(seeds)} seed(s) = {total_points} run(s), "
          f"{args.n_total} agents, {args.session_seconds:,.0f}s each ({n_legs} leg(s) of {leg_seconds:,.0f}s) -> {args.out_dir}/\n")

    t0 = time.time()
    done = 0
    run_dirs = {}  # pct -> [out_dir, ...] (one per seed)
    for pct in args.human_pcts:
        pct_label = f"{pct:g}".replace(".", "_")
        run_dirs[pct] = []
        for seed in seeds:
            out_dir = os.path.join(args.out_dir, f"human_{pct_label}pct", f"seed_{seed}")
            cfg = paper_lib.build_scenario(human_pct=pct, seed=seed, session_seconds=leg_seconds, n_total=args.n_total)
            legs = run_scenario_in_legs(cfg, leg_seconds, n_legs, out_dir, scenario_name=f"human_{pct_label}pct_seed{seed}")
            run_dirs[pct].append(out_dir)
            done += 1
            n_trades = sum(leg.meta.get("n_trades", 0) for leg in legs)
            elapsed = time.time() - t0
            print(f"[{done}/{total_points}] human={pct:g}% seed={seed}: {n_trades:,} trades -> {out_dir} ({elapsed:,.1f}s elapsed)")

    if not args.no_report:
        print("\nBuilding PDF report (loading every run's saved data back)...")
        report_path = build_report(run_dirs, args.out_dir, args.flash_window_trades)
        print(f"Report written to {report_path}")

    print(f"\nDone: {done} run(s), {time.time() - t0:,.1f}s wall time. Data saved under {args.out_dir}/")


if __name__ == "__main__":
    main()
