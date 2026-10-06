"""Builds the full report-data JSON for the NEW trendbot-archive dashboard,
matching human_sweep_3_report.html's embedded schema (scenarios, human_pct,
composite, mh, proxy) plus the three new/changed pieces from
sweep_v3_additions.py (bankrupt_heatmap replacing the old many-lines picker,
bankrupt_by_type with the new human_bot_ratio, vshare_at_start).

Data sources:
  - Main sweep archive (runs/human_sweep_lite_3000agents_150000s_trendbot,
    41 scenarios x 15 seeds x 150 legs): agent_state.parquet (bankrupt flags,
    cumulative_volume -- a true running notional total across the whole
    multi-leg run, per AgentCommon's own doc comment) and run_metadata.json
    (cheap per-leg n_trades, summed across all 150 legs for a whole-run
    total -- no trades.parquet reads needed at all).
  - Market-health sample archive (runs/mh_sample_lite_3000agents_150000s_
    trendbot, 41 scenarios x 16 legs, legs 2-16 each carrying a real
    market_health.json) -- see run_mh_sample_job.py's docstring for why this
    15-sample-per-scenario design, not a full 92,250-leg backfill, is this
    project's actual established method.
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/wangzi/crypto-bot-market-sim/scripts")
import sweep_v3_additions as v3

from market_health import score as market_health

TRENDBOT_ROOT = "/home/wangzi/crypto-bot-market-sim/runs/human_sweep_lite_3000agents_150000s_trendbot"
MH_ROOT = "/home/wangzi/crypto-bot-market-sim/runs/mh_sample_lite_3000agents_150000s_trendbot"
N_LEGS = 150
SEEDS = list(range(1, 16))
v3.SEEDS = SEEDS  # sweep_v3_additions' functions read this module-level global

ALL_BOT_TYPES = v3.ALL_BOT_TYPES
PCTS = v3.PCTS


def log(msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mean_std(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return {"mean": None, "std": None}
    arr = np.array(vals)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def build_mh_and_composite():
    """mh[name][metric] = {mean, std} across the 15 sample legs (2..16);
    composite[name] = {dim: z-score, composite_score} via market_health's own
    batch z-normalization across all 41 scenarios' per-scenario means."""
    names = [v3.pct_to_name(p) for p in PCTS]
    per_scenario_leg_metrics = {}
    for name in names:
        leg_metrics = []
        for leg_idx in range(2, 17):
            fp = os.path.join(MH_ROOT, name, f"leg_{leg_idx:04d}", "market_health.json")
            if os.path.exists(fp):
                leg_metrics.append(json.load(open(fp)))
        per_scenario_leg_metrics[name] = leg_metrics

    mh = {}
    for name in names:
        legs = per_scenario_leg_metrics[name]
        if not legs:
            mh[name] = {}
            continue
        keys = legs[0].keys()
        mh[name] = {k: mean_std([float(leg[k]) for leg in legs if leg.get(k) is not None]) for k in keys}

    # composite_scores needs ONE metrics dict per scenario (its own mean across
    # the 15 sample legs), z-normalized across the 41-scenario batch.
    batch = []
    for name in names:
        legs = per_scenario_leg_metrics[name]
        if not legs:
            batch.append({})
            continue
        keys = legs[0].keys()
        batch.append({k: float(np.nanmean([float(leg.get(k, np.nan)) for leg in legs])) for k in keys})
    comp_df = market_health.composite_scores(batch, run_names=names)
    composite = {name: {col: (None if pd.isna(v) else float(v)) for col, v in comp_df.loc[name].items()} for name in names}
    return mh, composite


def _proxy_one_seed(name, seed):
    """One (scenario, seed)'s whole-run vshare/notional/n_trades -- the
    n_trades sum alone touches 150 tiny run_metadata.json files, so across
    41 scenarios x 15 seeds this is ~92,250 file opens; run via a thread
    pool below (first version ran this serially and was still going after
    30+ minutes -- genuinely slow at this file count, not stuck, but worth
    the same GIL-releasing-I/O parallelization already applied to the
    heatmap)."""
    seed_dir = os.path.join(TRENDBOT_ROOT, name, f"seed{seed}")
    ag_fp = os.path.join(seed_dir, "leg_0150", "agent_state.parquet")
    if not os.path.exists(ag_fp):
        return None
    ag = pd.read_parquet(ag_fp, columns=["agent_id", "bot_type", "cumulative_volume"]).drop_duplicates("agent_id")
    total_cv = ag["cumulative_volume"].sum()
    vshare = {bt: float(ag.loc[ag["bot_type"] == bt, "cumulative_volume"].sum() / total_cv) for bt in ALL_BOT_TYPES} if total_cv > 0 else None
    # Each trade's notional is counted once in the maker's and once in the
    # taker's own cumulative_volume -- see AgentCommon's doc comment -- so
    # summing across every agent double-counts; /2 recovers the actual
    # whole-run total_notional without touching any trades.parquet.
    notional = float(total_cv) / 2.0
    n_trades_sum = 0
    for leg_idx in range(1, N_LEGS + 1):
        meta_fp = os.path.join(seed_dir, f"leg_{leg_idx:04d}", "run_metadata.json")
        if os.path.exists(meta_fp):
            n_trades_sum += json.load(open(meta_fp)).get("n_trades", 0)
    return vshare, notional, n_trades_sum


def build_proxy(max_workers=16):
    """proxy[name] = {vshare_<bt>: {mean, std_across_seeds}, total_notional,
    n_trades} -- whole-run totals, cheap-sourced (see module docstring),
    parallelized across (scenario, seed) pairs."""
    names = [v3.pct_to_name(p) for p in PCTS]
    jobs = [(name, seed) for name in names for seed in SEEDS]
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_proxy_one_seed, name, seed): (name, seed) for name, seed in jobs}
        n_done = 0
        for fut in futures:
            name, seed = futures[fut]
            results[(name, seed)] = fut.result()
            n_done += 1
            if n_done % 100 == 0:
                log(f"  proxy: {n_done}/{len(jobs)} (scenario,seed) pairs done")

    proxy = {}
    for name in names:
        vshare_per_seed = {bt: [] for bt in ALL_BOT_TYPES}
        notional_per_seed = []
        ntrades_per_seed = []
        for seed in SEEDS:
            r = results.get((name, seed))
            if r is None:
                continue
            vshare, notional, n_trades_sum = r
            if vshare is not None:
                for bt in ALL_BOT_TYPES:
                    vshare_per_seed[bt].append(vshare[bt])
            notional_per_seed.append(notional)
            ntrades_per_seed.append(n_trades_sum)
        entry = {f"vshare_{bt}": mean_std(vshare_per_seed[bt]) for bt in ALL_BOT_TYPES}
        entry["total_notional"] = mean_std(notional_per_seed)
        entry["n_trades"] = mean_std(ntrades_per_seed)
        proxy[name] = entry
    return proxy


def main():
    names = [v3.pct_to_name(p) for p in PCTS]

    log("building bankrupt_heatmap (0/3)...")
    bankrupt_heatmap = v3.build_bankrupt_heatmap(TRENDBOT_ROOT, N_LEGS)
    log("building bankrupt_by_type with human_bot_ratio (1/3)...")
    bankrupt_by_type = v3.build_bankrupt_by_type(TRENDBOT_ROOT, N_LEGS)
    log("building vshare_at_start (2/3)...")
    vshare_at_start = v3.build_vshare_at_start(TRENDBOT_ROOT)

    log("building market-health + composite scores...")
    mh, composite = build_mh_and_composite()

    log("building proxy metrics (whole-run vshare/notional/n_trades)...")
    proxy = build_proxy()

    out = {
        "scenarios": names,
        "human_pct": PCTS,
        "composite": composite,
        "mh": mh,
        "proxy": proxy,
        "bankrupt_heatmap": bankrupt_heatmap,
        "bankrupt_by_type": bankrupt_by_type,
        "vshare_at_start": vshare_at_start,
        "meta": {
            "archive": "human_sweep_lite_3000agents_150000s_trendbot",
            "n_seeds": len(SEEDS),
            "n_legs": N_LEGS,
            "note": "Human bot now reads Market::random_trend, a single shared "
                     "crowd-trend line per asset; each agent's sensitivity "
                     "(personal_trend_weight) is drawn independently per agent.",
        },
    }
    out_path = "/tmp/claude-1000/-home-wangzi/c7c92265-59b5-4361-9f81-6b0752df46bd/scratchpad/trendbot_report_data.json"
    with open(out_path, "w") as f:
        json.dump(out, f, default=str)
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
