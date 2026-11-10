#!/usr/bin/env python3
"""Institutional-seller "flash crash" sweep -- reproduces the paper's Section 6
experiment ("Recreating the 2010 Flash Crash"): a two-asset institutional
seller (the May 6, 2010 event's Percentage-of-Volume mechanism: ~9% of
trailing volume, no price/time limit) is dropped onto the human-share
composition sweep's baseline market after a warmup period. For every
(human share, seed) point this runs BOTH a "with dump" and a same-seed
"no dump" control, so price paths can be compared directly and any move can
be attributed to the seller rather than to the ordinary drift of that
composition (the paper's Table 11 methodology, generalized across every
human-share point rather than just 100%).

Usage (defaults to a quick sweep; see --help for every knob):

    python scripts/dump_sim_sweep.py --runs 3

    # Full paper-scale sweep, one seed per point, matching Section 6 exactly:
    python scripts/dump_sim_sweep.py --human-pcts 0,10,20,30,40,50,60,70,80,90,100 \\
        --runs 1 --n-total 1000 --warmup-seconds 1800 --post-shock-seconds 3000

Each run's full history is saved to
<out-dir>/human_<pct>pct/seed_<seed>/{with_dump,no_dump}/leg_0001/. Compare
the two with market_sim.load_run_legs on each directory -- e.g. the fall from
pre-shock price to post-shock trough, with vs. without the seller present.

Once every run is done, a PDF report (<out-dir>/report.pdf) is generated with
the same tables the paper itself reports for this experiment (Section 6):
price impact of the dump by human share (mean/best/worst fall % across
seeds), and price change with vs. without the seller, one row per
(human share, seed).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_lib  # noqa: E402
import paper_metrics  # noqa: E402

from market_sim.runner import run_scenario_in_legs  # noqa: E402


def parse_pcts(s: str) -> list:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def parse_ints(s: str) -> list:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def build_report(run_dirs: dict, out_dir: str, target_assets: list, warmup_seconds: float) -> str:
    """`run_dirs`: {pct: [(seed, with_dump_dir, no_dump_dir), ...]}. Loads
    every run back, computes each seed's fall % (paper_metrics.dump_fall_pct)
    for both the with-dump run and its same-seed no-dump control, and writes
    <out_dir>/report.pdf with the paper's Table 9 (mean/best/worst fall % by
    human share, with-dump) and Table 11 (with- vs. without-seller fall %,
    one row per human share x seed)."""
    table9_rows, table11_rows = [], []

    for pct in sorted(run_dirs.keys()):
        with_falls = []
        for seed, with_dir, no_dir in run_dirs[pct]:
            run_with = paper_metrics.load_run_tables(with_dir)
            run_no = paper_metrics.load_run_tables(no_dir)
            fall_with = paper_metrics.dump_fall_pct(run_with, target_assets, warmup_seconds)
            fall_no = paper_metrics.dump_fall_pct(run_no, target_assets, warmup_seconds)
            with_falls.append(fall_with)
            diff = fall_with - fall_no if not (np.isnan(fall_with) or np.isnan(fall_no)) else float("nan")
            table11_rows.append([f"{pct:g}%", seed, paper_metrics.fmt(fall_with, pct=True), paper_metrics.fmt(fall_no, pct=True), paper_metrics.fmt(diff, pct=True)])

        valid = [f for f in with_falls if not np.isnan(f)]
        mean_fall = float(np.mean(valid)) if valid else float("nan")
        best = max(valid) if valid else float("nan")   # least negative / most positive = "best" (price held up best)
        worst = min(valid) if valid else float("nan")  # most negative = "worst" (biggest fall)
        table9_rows.append([f"{pct:g}%", paper_metrics.fmt(mean_fall, pct=True), paper_metrics.fmt(best, pct=True), paper_metrics.fmt(worst, pct=True)])

    tables = [
        ("Price impact of the dump by human share",
         ["Human %", "Mean fall %", "Best %", "Worst %"], table9_rows),
        ("Price change with vs. without the institutional seller (one row per human share x seed)",
         ["Human %", "Seed", "With-seller fall %", "No-seller fall %", "Difference"], table11_rows),
    ]
    out_path = os.path.join(out_dir, "report.pdf")
    return paper_metrics.render_tables_pdf(tables, out_path, title="Institutional-Seller Dump Sweep Report\n(Section 6 tables)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--human-pcts", type=parse_pcts, default="0,10,20,30,40,50,60,70,80,90,100",
                    help="comma-separated human-share percentages to sweep (default: 0..100 in steps of 10)")
    p.add_argument("--runs", type=int, default=1, help="number of seeds (independent runs) per human-share point (default: 1)")
    p.add_argument("--first-seed", type=int, default=1, help="first seed number; seeds used are first-seed..first-seed+runs-1 (default: 1)")
    p.add_argument("--n-total", type=int, default=1000, help="total agent population size (default: 1000, matching the paper)")
    p.add_argument("--warmup-seconds", type=float, default=1800.0, help="simulated seconds before the seller activates (default: 1800)")
    p.add_argument("--post-shock-seconds", type=float, default=3000.0,
                    help="simulated seconds to keep running after the seller activates (default: 3000, matching the paper's post-dump window)")
    p.add_argument("--target-assets", type=parse_ints, default="0,1",
                    help="0-based asset indices the seller targets at once (default: 0,1, i.e. Stock1 and Stock2)")
    p.add_argument("--total-qty", type=float, default=1.0, help="quantity the seller starts holding per target asset (default: 1.0)")
    p.add_argument("--participation-rate", type=float, default=0.09,
                    help="target fraction of trailing volume the seller paces against (default: 0.09, the real 2010 event's rate)")
    p.add_argument("--leg-seconds", type=float, default=None,
                    help="split each run into legs of this many seconds (default: the full warmup+post-shock duration, one leg)")
    p.add_argument("--out-dir", default="runs/dump_sim_sweep", help="base output directory (default: runs/dump_sim_sweep)")
    p.add_argument("--no-report", action="store_true", help="skip generating the PDF report after the sweep finishes")
    args = p.parse_args()

    session_seconds = args.warmup_seconds + args.post_shock_seconds
    leg_seconds = args.leg_seconds or session_seconds
    n_legs = max(1, round(session_seconds / leg_seconds))
    seeds = list(range(args.first_seed, args.first_seed + args.runs))

    total_runs = len(args.human_pcts) * len(seeds) * 2  # with_dump + no_dump
    print(f"Dump-sim sweep: {len(args.human_pcts)} composition(s) x {len(seeds)} seed(s) x 2 (with/no dump) = {total_runs} run(s), "
          f"{args.n_total} agents, {session_seconds:,.0f}s each (warmup {args.warmup_seconds:,.0f}s + post-shock {args.post_shock_seconds:,.0f}s) "
          f"-> {args.out_dir}/\n")

    t0 = time.time()
    done = 0
    run_dirs = {}  # pct -> [(seed, with_dump_dir, no_dump_dir), ...]
    for pct in args.human_pcts:
        pct_label = f"{pct:g}".replace(".", "_")
        run_dirs[pct] = []
        for seed in seeds:
            base_out = os.path.join(args.out_dir, f"human_{pct_label}pct", f"seed_{seed}")

            seller_groups = paper_lib.institutional_seller_groups(
                target_assets=args.target_assets,
                total_qty=args.total_qty,
                participation_rate=args.participation_rate,
                start_time_secs=args.warmup_seconds,
            )
            variant_dirs = {}
            for variant_name, extra_groups in (("with_dump", seller_groups), ("no_dump", None)):
                out_dir = os.path.join(base_out, variant_name)
                cfg = paper_lib.build_scenario(
                    human_pct=pct, seed=seed, session_seconds=leg_seconds,
                    n_total=args.n_total, extra_groups=extra_groups,
                )
                legs = run_scenario_in_legs(cfg, leg_seconds, n_legs, out_dir,
                                             scenario_name=f"human_{pct_label}pct_seed{seed}_{variant_name}")
                variant_dirs[variant_name] = out_dir
                done += 1
                n_trades = sum(leg.meta.get("n_trades", 0) for leg in legs)
                elapsed = time.time() - t0
                print(f"[{done}/{total_runs}] human={pct:g}% seed={seed} {variant_name}: {n_trades:,} trades -> {out_dir} ({elapsed:,.1f}s elapsed)")
            run_dirs[pct].append((seed, variant_dirs["with_dump"], variant_dirs["no_dump"]))

    print(f"\nDone: {done} run(s), {time.time() - t0:,.1f}s wall time. Data saved under {args.out_dir}/")

    if not args.no_report:
        print("\nBuilding PDF report (loading every run's saved data back)...")
        report_path = build_report(run_dirs, args.out_dir, args.target_assets, args.warmup_seconds)
        print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
