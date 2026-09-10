#!/usr/bin/env python3
"""Run a scenario as several shorter, separately-saved sessions instead of one huge
one -- see README's "Memory: the actual scaling limit" section for why that matters
at the population/session sizes the spec describes (everything is held in memory at
once for the length of a single run; splitting into legs bounds each leg's own
footprint and discards it once it's written to disk, rather than letting a full
15,000-agent/21,600s session's data all exist simultaneously).

Each leg after the first is a genuine checkpoint/resume continuation, not a fresh
restart -- same order books, every agent's state, both RNG streams carried over --
so only the very first leg sees the cold-start burst of trades every fresh
simulation shows in its first simulated second (empty book, every agent's first
wake jittered into that same second); legs 2+ pick up an already-warmed, already-
trading market.

Usage:
    python scripts/run_in_legs.py scenarios/realistic_mix.toml \
        --leg-seconds 1800 --n-legs 12 --out-dir runs/realistic_mix_legs

    # Then, in Python:
    #   import market_sim as ms
    #   combined = ms.load_run_legs("runs/realistic_mix_legs")   # full history, for analysis
    #   legs = ms.discover_legs("runs/realistic_mix_legs")        # individual leg directories
"""
from __future__ import annotations

import argparse
import time

from market_sim import config as cfgmod
from market_sim.runner import run_scenario_in_legs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario", help="path to a scenario TOML file (e.g. scenarios/realistic_mix.toml)")
    p.add_argument("--leg-seconds", type=float, required=True, help="simulated seconds per leg")
    p.add_argument("--n-legs", type=int, required=True, help="number of legs to run (total simulated time = leg-seconds * n-legs)")
    p.add_argument("--out-dir", required=True, help="base directory; each leg is saved to <out-dir>/leg_0001, leg_0002, ...")
    p.add_argument("--scenario-name", default=None, help="recorded in each leg's run_metadata.json")
    args = p.parse_args()

    cfg = cfgmod.load_config_dict(args.scenario)
    total = args.leg_seconds * args.n_legs
    print(f"Running {args.scenario} as {args.n_legs} leg(s) x {args.leg_seconds:,.0f}s ({total:,.0f}s total simulated) -> {args.out_dir}/leg_NNNN/\n")

    t0 = time.time()

    def on_leg(leg):
        elapsed = time.time() - t0
        print(f"  leg {leg.index}/{args.n_legs}: {leg.meta['n_trades']:,} trades, absolute sim time {leg.meta['actual_session_seconds']:,.0f}s -- saved to {leg.out_dir} ({elapsed:,.1f}s elapsed)")

    legs = run_scenario_in_legs(cfg, args.leg_seconds, args.n_legs, args.out_dir, scenario_name=args.scenario_name, on_leg_saved=on_leg)

    print(f"\nDone: {len(legs)} leg(s), {sum(l.meta['n_trades'] for l in legs):,} total trades, {time.time()-t0:,.1f}s wall time.")
    print(f"Load the full combined history with: market_sim.load_run_legs({args.out_dir!r})")


if __name__ == "__main__":
    main()
