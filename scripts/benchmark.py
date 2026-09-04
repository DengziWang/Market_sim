#!/usr/bin/env python3
"""Spec section 6a: benchmark-first requirement. Run this BEFORE starting any large
batch. It runs a small simulation (default: 300 agents, 2 assets, 5 simulated
minutes), measures actual events/second on this machine, and extrapolates a wall-clock
estimate for a batch you specify -- so you know roughly how long a real experiment
will take before you commit to it.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --agents 500 --session-seconds 600
    python scripts/benchmark.py --estimate-agents 2000 --estimate-session-seconds 3600 \
        --estimate-n-runs 1000 --estimate-workers 5
"""
from __future__ import annotations

import argparse
import platform
import time

from market_sim import validate_config, run_simulation

BENCHMARK_CONFIG_TEMPLATE = """
[run]
seed = {seed}
session_seconds = {session_seconds}
min_trades = 0

[[assets]]
symbol = "A"
tick_size = 0.01
initial_price = 100.0

[[assets]]
symbol = "B"
tick_size = 0.01
initial_price = 50.0

[market]
snapshot_interval_secs = 1.0
snapshot_depth_levels = 5

[export]
out_dir = "runs/_benchmark"
write_snapshots = true

[[population.groups]]
bot_type = "market_maker"
variant = "avellaneda_stoikov"
count = {n_mm}
[population.groups.params]
wake_interval_secs = 0.3

[[population.groups]]
bot_type = "momentum"
variant = "ma_crossover"
count = {n_momentum}

[[population.groups]]
bot_type = "noise"
variant = "zero_intelligence"
count = {n_noise}
[population.groups.params]
wake_interval_secs = 1.0

[[population.groups]]
bot_type = "wash_trading"
variant = "self_ring"
count = {n_wash}
"""


def build_benchmark_config(n_agents: int, session_seconds: float, seed: int) -> str:
    # Rough, fixed composition split -- good enough as a representative mix of
    # requote-heavy (market_maker), periodic (momentum), high-frequency-ish (noise
    # default 1s) and self-crossing (wash_trading) activity patterns.
    n_mm = max(1, round(n_agents * 0.1))
    n_momentum = max(1, round(n_agents * 0.2))
    n_wash = max(1, round(n_agents * 0.1))
    n_noise = max(1, n_agents - n_mm - n_momentum - n_wash)
    return BENCHMARK_CONFIG_TEMPLATE.format(
        seed=seed, session_seconds=session_seconds, n_mm=n_mm, n_momentum=n_momentum, n_noise=n_noise, n_wash=n_wash
    )


def run_benchmark(n_agents: int, session_seconds: float, seed: int = 0) -> dict:
    cfg = build_benchmark_config(n_agents, session_seconds, seed)
    n_assets, n_venues, n_agents_actual = validate_config(cfg)
    t0 = time.perf_counter()
    result = run_simulation(cfg)
    wall_seconds = time.perf_counter() - t0

    meta = result["meta"]
    n_events = meta["n_trades"] + meta["n_order_events"]
    return {
        "n_agents": n_agents_actual,
        "n_assets": n_assets,
        "session_seconds_simulated": meta["actual_session_seconds"],
        "wall_seconds": wall_seconds,
        "n_trades": meta["n_trades"],
        "n_order_events": meta["n_order_events"],
        "events_per_wall_second": n_events / wall_seconds if wall_seconds > 0 else float("inf"),
        "sim_seconds_per_wall_second": meta["actual_session_seconds"] / wall_seconds if wall_seconds > 0 else float("inf"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", type=int, default=300, help="agents in the benchmark run (spec default: 100-500)")
    p.add_argument("--session-seconds", type=float, default=300.0, help="simulated session length for the benchmark run")
    p.add_argument("--repeats", type=int, default=2, help="repeat the timed run this many times and report the median")
    p.add_argument("--estimate-agents", type=int, default=1500, help="agents per run in the batch you're planning")
    p.add_argument("--estimate-session-seconds", type=float, default=1800.0, help="simulated session length per run in the planned batch")
    p.add_argument("--estimate-n-runs", type=int, default=100, help="total number of runs in the planned batch (compositions x seeds)")
    p.add_argument("--estimate-workers", type=int, default=5, help="parallel worker processes for the planned batch")
    args = p.parse_args()

    print(f"Platform: {platform.platform()}, Python {platform.python_version()}")
    print(f"Running {args.repeats} benchmark run(s): {args.agents} agents, {args.session_seconds}s simulated session...\n")

    results = [run_benchmark(args.agents, args.session_seconds, seed=i) for i in range(args.repeats)]
    results.sort(key=lambda r: r["wall_seconds"])
    med = results[len(results) // 2]

    print("Benchmark result (median of {} run(s)):".format(args.repeats))
    for k, v in med.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")

    events_per_sec = med["events_per_wall_second"]
    sim_seconds_per_wall_second = med["sim_seconds_per_wall_second"]
    # Naive linear scaling in agent count is optimistic for very large populations
    # (order-book contention, per-tick agent overhead) but a reasonable first-order
    # estimate -- treat this as a floor on wall-clock time, not a tight bound.
    agent_scale = args.estimate_agents / med["n_agents"]
    est_wall_per_run = (args.estimate_session_seconds / sim_seconds_per_wall_second) * agent_scale
    est_total_wall = est_wall_per_run * args.estimate_n_runs / max(1, args.estimate_workers)

    print("\nExtrapolated estimate for your planned batch:")
    print(f"  {args.estimate_agents} agents x {args.estimate_session_seconds}s simulated x {args.estimate_n_runs} runs, {args.estimate_workers} parallel workers")
    print(f"  ~{est_wall_per_run:,.1f}s per run")
    print(f"  ~{est_total_wall:,.1f}s (~{est_total_wall/60:,.1f} min, ~{est_total_wall/3600:,.2f} h) total wall time")
    print(
        "\nNote (spec 6a/12): if this is dramatically worse than expected (e.g. thermal "
        "throttling under sustained laptop load), reduce agent-count/session-length/seed-count "
        "defaults rather than deciding on your own -- confirm the change with the user first."
    )


if __name__ == "__main__":
    main()
