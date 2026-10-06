"""New/changed analysis pieces for the human-fraction-sweep dashboard rebuild
(2026-10-03), designed to slot into the existing report-data JSON schema
(`D.bankrupt_trajectories`, `D.bankrupt_by_type`, `D.proxy`) used by
human_sweep_3_report.html's JS. Kept in the repo (not the scratchpad, which
has been wiped by session restarts before) so it survives.

Three pieces, each addressing one part of the user's request:
  1. `bankrupt_heatmap`: a (composition x leg) grid of mean bankrupt fraction
     across seeds -- replaces the old many-lines picker chart
     (`D.bankrupt_trajectories` + the picker), which the user found unreadable
     with many lines selected at once.
  2. `bankrupt_by_type` terminal rates, PLUS a new `human_bot_ratio` field =
     human_bankrupt_rate / bot_bankrupt_rate (per seed, then mean/std across
     seeds) -- the user's requested "one metric to measure who is more
     successful" (confirmed via clarification to reduce to this exact ratio).
  3. `vshare_at_start`: per-bot-type volume share computed from leg 1 only
     (first 1000 of 150,000s), alongside the existing whole-run
     `vshare_<bt>` -- the user's requested "also display the ratio at the
     start of the simulation" in the volume-share section.

Run standalone against any archive root (old or new bankruptcy-mechanic) to
verify the functions; the real dashboard build imports and calls these.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ALL_BOT_TYPES = [
    "arbitrage", "dca", "grid", "human", "liquidation_trader",
    "market_maker", "mean_reversion", "mev", "momentum",
]
PCTS = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30,
        32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5, 55, 57.5, 60,
        62.5, 65, 67.5, 70, 72.5, 75, 77.5, 80, 82.5, 85, 87.5, 90,
        92.5, 95, 97.5, 100]
SEEDS = list(range(1, 11))


def pct_to_name(pct):
    return f"human_{str(pct).replace('.', '_')}pct"


def bankrupt_fraction_series(root, name, seed, n_legs=150):
    """One value per leg: fraction of agents with bankrupt==True by that leg's
    end (bankrupt is a sticky/permanent flag -- see sim.rs, never reset), read
    straight from each leg's agent_state.parquet (kept in full every leg,
    unlike trades/order_events). Leg paths built directly (no glob) since the
    leg range/naming convention (leg_0001..leg_NNNN, zero-padded) is already
    known -- this runs across hundreds of (scenario, seed) pairs, so skipping
    a directory listing per call is a real, if small, saving at that scale."""
    out = []
    for leg_idx in range(1, n_legs + 1):
        fp = f"{root}/{name}/seed{seed}/leg_{leg_idx:04d}/agent_state.parquet"
        if not os.path.exists(fp):
            break
        ag = pd.read_parquet(fp, columns=["agent_id", "bankrupt"])
        ag = ag.drop_duplicates("agent_id")
        out.append(float(ag["bankrupt"].mean()))
    return out


def bankrupt_fraction_series_by_type(root, name, seed, n_legs=150):
    """Same as bankrupt_fraction_series, but split into (human_series,
    bot_series) -- the fraction of each group bankrupt by that leg's end.
    Needed for the human-vs-bot-over-time comparison chart (the heatmap
    alone can't show this split, only the combined fraction)."""
    human_out, bot_out = [], []
    for leg_idx in range(1, n_legs + 1):
        fp = f"{root}/{name}/seed{seed}/leg_{leg_idx:04d}/agent_state.parquet"
        if not os.path.exists(fp):
            break
        ag = pd.read_parquet(fp, columns=["agent_id", "bot_type", "bankrupt"]).drop_duplicates("agent_id")
        is_human = ag["bot_type"] == "human"
        n_human, n_bot = int(is_human.sum()), int((~is_human).sum())
        human_out.append(float(ag.loc[is_human, "bankrupt"].mean()) if n_human > 0 else None)
        bot_out.append(float(ag.loc[~is_human, "bankrupt"].mean()) if n_bot > 0 else None)
    return human_out, bot_out


def bankrupt_fraction_series_all_types(root, name, seed, n_legs=150):
    """Same per-leg read as bankrupt_fraction_series_by_type, but split into
    one series per individual bot_type (all 9 of ALL_BOT_TYPES, including
    human) instead of just a human/bot binary -- 2026-10-05, for a chart that
    lets bot bankruptcy be broken down by type, not just lumped into one
    aggregate "bot" line (which flattens real differences between e.g.
    market_maker and mev). Returns {bot_type: [150 values or None]}."""
    out = {bt: [] for bt in ALL_BOT_TYPES}
    for leg_idx in range(1, n_legs + 1):
        fp = f"{root}/{name}/seed{seed}/leg_{leg_idx:04d}/agent_state.parquet"
        if not os.path.exists(fp):
            break
        ag = pd.read_parquet(fp, columns=["agent_id", "bot_type", "bankrupt"]).drop_duplicates("agent_id")
        for bt in ALL_BOT_TYPES:
            sub = ag.loc[ag["bot_type"] == bt, "bankrupt"]
            out[bt].append(float(sub.mean()) if len(sub) > 0 else None)
    return out


def build_bankrupt_trajectories_all_types(root, n_legs=150, max_workers=16):
    """Returns {"compositions": [...], "human_pct": [...], "by_type":
    {name: {bot_type: [150 values or None]}}} -- per-leg bankrupt fraction
    for every individual bot type (not just human-vs-bot), averaged across
    seeds, for every composition. Same thread-pool-across-(scenario,seed)
    pattern as build_bankrupt_trajectories_by_type."""
    names = [pct_to_name(p) for p in PCTS]
    jobs = [(name, seed) for name in names for seed in SEEDS]
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(bankrupt_fraction_series_all_types, root, name, seed, n_legs): (name, seed) for name, seed in jobs}
        for fut in futures:
            name, seed = futures[fut]
            results[(name, seed)] = fut.result()

    by_type_out = {}
    for name in names:
        per_type = {}
        for bt in ALL_BOT_TYPES:
            runs = [results[(name, seed)][bt] for seed in SEEDS if len(results[(name, seed)][bt]) == n_legs]
            if not runs:
                per_type[bt] = [None] * n_legs
                continue
            out = []
            for leg in range(n_legs):
                vals = [r[leg] for r in runs if r[leg] is not None]
                out.append(float(np.mean(vals)) if vals else None)
            per_type[bt] = out
        by_type_out[name] = per_type
    return {"compositions": names, "human_pct": PCTS, "by_type": by_type_out}


def build_leg1_total_notional(root, max_workers=16):
    """Returns {name: {mean, std}} -- the whole market's total notional
    traded during leg 1 only (first 1,000s), averaged across seeds. Pairs
    with `proxy[name].total_notional` (the whole-run total) to compute
    relative market-activity growth over the course of a run, per scenario
    (2026-10-05, user request) -- same leg1-vs-whole-run growth definition
    used for the per-bot-type volume growth table, just not broken down by
    type."""
    names = [pct_to_name(p) for p in PCTS]

    def one(name, seed):
        fp = f"{root}/{name}/seed{seed}/leg_0001/agent_state.parquet"
        if not os.path.exists(fp):
            return None
        ag = pd.read_parquet(fp, columns=["agent_id", "cumulative_volume"]).drop_duplicates("agent_id")
        # /2: cumulative_volume counts each trade once for the maker and once
        # for the taker (see AgentCommon's doc comment) -- same halving
        # build_trendbot_report_data.py's _proxy_one_seed already applies to
        # recover the true notional from this same column.
        return float(ag["cumulative_volume"].sum()) / 2.0

    jobs = [(name, seed) for name in names for seed in SEEDS]
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(one, name, seed): (name, seed) for name, seed in jobs}
        for fut in futures:
            name, seed = futures[fut]
            results[(name, seed)] = fut.result()

    out = {}
    for name in names:
        vals = [results[(name, seed)] for seed in SEEDS if results[(name, seed)] is not None]
        out[name] = _mean_std(vals)
    return out


def build_bankrupt_trajectories_by_type(root, n_legs=150, max_workers=16):
    """Returns {"compositions": [...], "human_pct": [...],
    "human": {name: [150 values or null]}, "bot": {name: [150 values or
    null]}} -- per-leg human/bot bankrupt fraction, averaged across seeds,
    for every composition. Powers the picker chart: pick a composition, see
    its human vs. bot bankruptcy trajectories side by side. Same
    thread-pool-across-(scenario,seed) pattern as build_bankrupt_heatmap."""
    names = [pct_to_name(p) for p in PCTS]
    jobs = [(name, seed) for name in names for seed in SEEDS]
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(bankrupt_fraction_series_by_type, root, name, seed, n_legs): (name, seed) for name, seed in jobs}
        for fut in futures:
            name, seed = futures[fut]
            results[(name, seed)] = fut.result()

    human_out, bot_out = {}, {}
    for name in names:
        human_runs = [results[(name, seed)][0] for seed in SEEDS if len(results[(name, seed)][0]) == n_legs]
        bot_runs = [results[(name, seed)][1] for seed in SEEDS if len(results[(name, seed)][1]) == n_legs]

        def avg_per_leg(runs):
            if not runs:
                return [None] * n_legs
            out = []
            for leg in range(n_legs):
                vals = [r[leg] for r in runs if r[leg] is not None]
                out.append(float(np.mean(vals)) if vals else None)
            return out

        human_out[name] = avg_per_leg(human_runs)
        bot_out[name] = avg_per_leg(bot_runs)
    return {"compositions": names, "human_pct": PCTS, "human": human_out, "bot": bot_out}


def build_bankrupt_heatmap(root, n_legs=150, max_workers=16):
    """Returns {"compositions": [...], "human_pct": [...], "grid": [[...]]}
    where grid[i] is composition i's per-leg bankrupt fraction, averaged
    across seeds -- the heatmap's actual plotted data. Parallelized across
    every (scenario, seed) pair with a thread pool -- pandas/pyarrow's
    parquet read releases the GIL for the actual I/O-bound work, same
    pattern this project already used for the dashboard's per-leg OHLC
    summaries (see memory) -- measured serially first (28ms/leg-read,
    ~44min for the full 41-scenario archive) before parallelizing, not
    assumed slow."""
    names = [pct_to_name(p) for p in PCTS]
    jobs = [(name, seed) for name in names for seed in SEEDS]
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(bankrupt_fraction_series, root, name, seed, n_legs): (name, seed) for name, seed in jobs}
        for fut in futures:
            name, seed = futures[fut]
            results[(name, seed)] = fut.result()

    grid = []
    for name in names:
        per_seed = [results[(name, seed)] for seed in SEEDS if len(results[(name, seed)]) == n_legs]
        if not per_seed:
            grid.append([None] * n_legs)
            continue
        arr = np.array(per_seed)  # (n_seeds_available, n_legs)
        grid.append(arr.mean(axis=0).tolist())
    return {"compositions": names, "human_pct": PCTS, "grid": grid}


def _mean_std(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return {"mean": None, "std": None, "per_seed": []}
    arr = np.array(vals)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "per_seed": [float(v) for v in vals]}


def bankrupt_by_type_one_scenario(root, name, n_legs=150):
    human_rates, bot_rates, human_shares, ratios = [], [], [], []
    for seed in SEEDS:
        leg_dir = f"{root}/{name}/seed{seed}/leg_{n_legs:04d}"
        fp = os.path.join(leg_dir, "agent_state.parquet")
        if not os.path.exists(fp):
            continue
        ag = pd.read_parquet(fp, columns=["agent_id", "bot_type", "bankrupt"]).drop_duplicates("agent_id")
        is_human = ag["bot_type"] == "human"
        n_human, n_bot = int(is_human.sum()), int((~is_human).sum())
        b_human = int(ag.loc[is_human, "bankrupt"].sum())
        b_bot = int(ag.loc[~is_human, "bankrupt"].sum())
        h_rate = (b_human / n_human) if n_human > 0 else None
        b_rate = (b_bot / n_bot) if n_bot > 0 else None
        total_b = b_human + b_bot
        h_share = (b_human / total_b) if total_b > 0 else None
        human_rates.append(h_rate)
        bot_rates.append(b_rate)
        human_shares.append(h_share)
        if h_rate is not None and b_rate is not None and b_rate > 0:
            ratios.append(h_rate / b_rate)
    return {
        "human_bankrupt_rate": _mean_std(human_rates),
        "bot_bankrupt_rate": _mean_std(bot_rates),
        "human_share_of_bankruptcies": _mean_std(human_shares),
        "human_bot_ratio": _mean_std(ratios),
    }


def build_bankrupt_by_type(root, n_legs=150):
    names = [pct_to_name(p) for p in PCTS]
    terminal = {name: bankrupt_by_type_one_scenario(root, name, n_legs) for name in names}
    return {"scenarios": names, "human_pct": PCTS, "terminal": terminal}


def _networth_one_seed(root, name, seed, n_legs):
    """Per-bot-type average equity (net worth) at leg 1's end and leg
    n_legs's end, for one (scenario, seed). `equity` is cash + inventory
    value (+ margin value) at that leg's own mark prices -- see
    AgentCommon::equity in sim_core -- so this is a true current net worth
    snapshot, not a cumulative/notional quantity like cumulative_volume."""
    out = {"start": {}, "end": {}}
    for leg_idx, bucket in [(1, "start"), (n_legs, "end")]:
        fp = f"{root}/{name}/seed{seed}/leg_{leg_idx:04d}/agent_state.parquet"
        if not os.path.exists(fp):
            continue
        ag = pd.read_parquet(fp, columns=["agent_id", "bot_type", "equity"]).drop_duplicates("agent_id")
        for bt in ALL_BOT_TYPES:
            sub = ag.loc[ag["bot_type"] == bt, "equity"]
            if len(sub):
                out[bucket][bt] = float(sub.mean())
    return out


def build_networth_by_type(root, n_legs=150, max_workers=16):
    """Returns {name: {bot_type: {start: {mean,std}, end: {mean,std}}}} --
    per-bot-type average agent net worth at the start (leg 1) and end
    (leg n_legs) of the run, mean/std across seeds. Answers "who actually
    made or lost money", as distinct from vshare (who was most active) or
    bankrupt_by_type (who blew up entirely)."""
    names = [pct_to_name(p) for p in PCTS]
    jobs = [(name, seed) for name in names for seed in SEEDS]
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_networth_one_seed, root, name, seed, n_legs): (name, seed) for name, seed in jobs}
        for fut in futures:
            name, seed = futures[fut]
            results[(name, seed)] = fut.result()

    out = {}
    for name in names:
        per_type = {bt: {"start": [], "end": []} for bt in ALL_BOT_TYPES}
        for seed in SEEDS:
            r = results.get((name, seed))
            if r is None:
                continue
            for bt in ALL_BOT_TYPES:
                if bt in r["start"]:
                    per_type[bt]["start"].append(r["start"][bt])
                if bt in r["end"]:
                    per_type[bt]["end"].append(r["end"][bt])
        out[name] = {
            bt: {"start": _mean_std(per_type[bt]["start"]), "end": _mean_std(per_type[bt]["end"])}
            for bt in ALL_BOT_TYPES
        }
    return out


def vshare_from_leg(root, name, seed, leg_index):
    leg_dir = f"{root}/{name}/seed{seed}/leg_{leg_index:04d}"
    fp = os.path.join(leg_dir, "agent_state.parquet")
    if not os.path.exists(fp):
        return None
    ag = pd.read_parquet(fp, columns=["agent_id", "bot_type", "cumulative_volume"]).drop_duplicates("agent_id")
    total = ag["cumulative_volume"].sum()
    if total <= 0:
        return None
    return {bt: float(ag.loc[ag["bot_type"] == bt, "cumulative_volume"].sum() / total) for bt in ALL_BOT_TYPES}


def build_vshare_at_start(root):
    """Volume share by bot_type accumulated during leg 1 only (the first 1000
    of 150,000s) -- `cumulative_volume` is a true running total carried across
    checkpoint/resume, so leg 1's end value is exactly that leg's own volume,
    with nothing from later legs mixed in."""
    names = [pct_to_name(p) for p in PCTS]
    out = {}
    for name in names:
        per_seed = {bt: [] for bt in ALL_BOT_TYPES}
        for seed in SEEDS:
            shares = vshare_from_leg(root, name, seed, 1)
            if shares is None:
                continue
            for bt in ALL_BOT_TYPES:
                per_seed[bt].append(shares[bt])
        out[name] = {f"vshare_{bt}_start": _mean_std(per_seed[bt]) for bt in ALL_BOT_TYPES}
    return {"scenarios": names, "human_pct": PCTS, "by_scenario": out}


def detect_spike_events_one_seed(root, name, seed, n_legs, threshold=0.05, bucket_secs=10.0):
    """Scans this (scenario, seed)'s trades.parquet, leg by leg, for
    bucket-to-bucket price jumps >= `threshold` (fractional, e.g. 0.05 = 5%)
    within the same asset -- flagged as a "spike" (price jumped up) or
    "crash" (price dropped) event.

    Bucketed (last trade price per `bucket_secs`-second window), NOT a naive
    consecutive-trade comparison -- an earlier version compared every pair of
    consecutive trades directly and produced 33,539 "events" from a single
    (scenario, seed) with only 2 legs of data, nearly all of them the same
    instant: agents whose orders cross the book at the same timestamp can
    alternate between two price levels (e.g. hitting both a stale bid and a
    stale ask microseconds apart), which a per-trade comparison reads as a
    real move and immediate reversal -- pure microstructure noise, not a
    "sudden spike/crash" in any meaningful sense. Bucketing to one
    representative price per short window (like an OHLC close) filters that
    out while still catching genuine fast moves, since `bucket_secs` is still
    short relative to how long a real dislocation persists.

    Post-hoc over each leg's already-saved trades rather than instrumented
    into the Rust engine itself (2026-10-06, user request) -- far lower-risk
    to add on top of an actively-running sweep, and every leg's trades are
    already saved regardless. Returns a list of dicts: leg, asset_id, time,
    price_before, price_after, pct_change, direction."""
    events = []
    for leg_idx in range(1, n_legs + 1):
        fp = f"{root}/{name}/seed{seed}/leg_{leg_idx:04d}/trades.parquet"
        if not os.path.exists(fp):
            break
        df = pd.read_parquet(fp, columns=["asset_id", "time", "price"])
        if df.empty:
            continue
        for aid, g in df.groupby("asset_id"):
            g = g.sort_values("time").copy()
            g["bucket"] = (g["time"] // bucket_secs).astype(int)
            bucketed = g.groupby("bucket").agg(time=("time", "last"), price=("price", "last"))
            prices = bucketed["price"].to_numpy()
            times = bucketed["time"].to_numpy()
            if len(prices) < 2:
                continue
            pct_change = (prices[1:] - prices[:-1]) / prices[:-1]
            for i in np.where(np.abs(pct_change) >= threshold)[0]:
                events.append({
                    "leg": leg_idx,
                    "asset_id": int(aid),
                    "time": float(times[i + 1]),
                    "price_before": float(prices[i]),
                    "price_after": float(prices[i + 1]),
                    "pct_change": float(pct_change[i]),
                    "direction": "spike" if pct_change[i] > 0 else "crash",
                })
    return events


def build_spike_events(root, n_legs, threshold=0.05, max_workers=16):
    """Returns {name: {seed: [events...]}} -- spike/crash events for every
    (scenario, seed) in the sweep, same thread-pool-across-(scenario,seed)
    pattern as the other build_* functions here."""
    names = [pct_to_name(p) for p in PCTS]
    jobs = [(name, seed) for name in names for seed in SEEDS]
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(detect_spike_events_one_seed, root, name, seed, n_legs, threshold): (name, seed) for name, seed in jobs}
        for fut in futures:
            name, seed = futures[fut]
            results[(name, seed)] = fut.result()

    out = {}
    for name in names:
        out[name] = {str(seed): results.get((name, seed), []) for seed in SEEDS}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-legs", type=int, default=150)
    ap.add_argument("--quick-check", action="store_true", help="only build for 3 compositions, for fast verification")
    args = ap.parse_args()

    global PCTS
    if args.quick_check:
        PCTS = [0, 50, 100]

    print("building bankrupt_heatmap...")
    heatmap = build_bankrupt_heatmap(args.root, args.n_legs)
    print("building bankrupt_by_type (with human_bot_ratio)...")
    by_type = build_bankrupt_by_type(args.root, args.n_legs)
    print("building vshare_at_start...")
    vshare_start = build_vshare_at_start(args.root)

    out = {"bankrupt_heatmap": heatmap, "bankrupt_by_type": by_type, "vshare_at_start": vshare_start}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")

    for name in by_type["scenarios"]:
        r = by_type["terminal"][name]["human_bot_ratio"]
        print(f"  {name}: human_bot_ratio mean={r['mean']}, std={r['std']}, n_seeds={len(r['per_seed'])}")


if __name__ == "__main__":
    main()
