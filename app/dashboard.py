"""Local interactive dashboard for the bot-dominated crypto market simulation.

Launch with:
    cd ~/crypto-bot-market-sim
    streamlit run app/dashboard.py

Lets you configure everything in sim_core::config::ScenarioConfig (run length,
assets/"stocks", venues, fees, latency, mempool, leveraged-liquidation params, and
the full bot population -- counts, cash, leverage, and each variant's own
parameters), run a simulation, and explore the result: zoomable per-asset
price/volume charts, the market_health metrics, and a searchable/filterable trade
table. Nothing here talks to the network -- it's a thin UI over the same
market_sim/market_health packages the Python API guide uses directly.
"""
from __future__ import annotations

import copy
import glob
import json
import math
import os
import re
import threading
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import toml
from plotly.subplots import make_subplots

from market_sim import CancelToken, ProgressHandle, continue_simulation_cancellable, run_simulation_cancellable
from market_sim import config as cfgmod
from market_sim.candles import candles_from_trades, resample_candles
from market_sim.export import df_to_columnar, export_run, load_run, load_trades_columns
from market_sim.runner import combine_results, discover_legs, run_scenario_in_legs
import market_health
from market_health.score import DIMENSION_OF

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS = {
    "Realistic mix": os.path.join(REPO_ROOT, "scenarios", "realistic_mix.toml"),
    "Pure bot market": os.path.join(REPO_ROOT, "scenarios", "pure_bot_market.toml"),
    "Stress test (wash/MEV)": os.path.join(REPO_ROOT, "scenarios", "stress_wash_mev.toml"),
}
BOT_TYPES = [
    "market_maker", "arbitrage", "momentum", "mean_reversion", "grid", "dca",
    "liquidation_trader", "mev", "wash_trading", "noise", "human",
]
VARIANTS_BY_TYPE = {
    "market_maker": ["avellaneda_stoikov", "fixed_offset", "inventory_skew", "adaptive_spread"],
    "arbitrage": ["cross_venue", "triangular", "stat_arb_pairs", "latency_arb"],
    "momentum": ["ma_crossover", "breakout_donchian", "rsi", "ml_online"],
    "mean_reversion": ["bollinger", "ou_fair_value", "zscore"],
    "grid": ["static", "geometric", "dynamic_atr"],
    "dca": ["time_based", "value_averaging", "dip_buying"],
    "liquidation_trader": ["full", "staggered_twap"],
    "mev": ["sandwich", "frontrun", "backrun"],
    "wash_trading": ["self_ring", "volume_ranking"],
    "noise": ["zero_intelligence", "herding"],
    # Not spec-mandated -- a behavioral-finance-motivated bot (disposition effect,
    # FOMO/panic on both price trend AND sentiment, "hot hand" sizing, irregular
    # attention, round-number bias) rather than a rational/mechanical strategy. One
    # unified variant carries every bias at once. See sim_core/src/bots/human.rs.
    "human": ["retail"],
}

RUNS_ROOT = os.path.join(REPO_ROOT, "runs")

st.set_page_config(page_title="Bot Market Sim", layout="wide")

# ---------------------------------------------------------------------------------
# Rough pre-run size estimate -- see the "estimate before you run" note in README's
# Performance section. Calibrated against a real measurement (realistic_mix.toml
# scaled to 2,000 agents / 3,000s simulated: 6,130,334 order events, ~9.2s wall,
# ~2.5GB peak RSS through the full Rust -> numpy -> pandas pipeline), so it's
# specifically tuned to *this* engine/dashboard's actual memory behavior, not a
# generic guess. It assumes a population mix with roughly realistic_mix.toml's
# average per-agent wake frequency -- a config with e.g. only very-high-frequency
# market_makers would produce more order events per agent-second than this predicts,
# and a config dominated by slow `human`/`dca` bots would produce fewer. Meant to
# catch a "this will likely exhaust this machine's RAM" mistake before it happens
# (see project memory: a 5,000-agent/300,000s run OOM'd the whole desktop, not just
# the simulation), not to be a precise forecast.
_EST_ORDER_EVENTS_PER_AGENT_SECOND = 6_130_334 / (2000 * 3000)
_EST_MB_PER_ORDER_EVENT = 2457 / 6_130_334
_EST_SECONDS_PER_ORDER_EVENT = 9.2 / 6_130_334


def _estimate_run_cost(n_agents: float, session_seconds: float) -> dict:
    est_events = max(0.0, n_agents) * max(0.0, session_seconds) * _EST_ORDER_EVENTS_PER_AGENT_SECOND
    return {
        "order_events": est_events,
        "peak_mb": est_events * _EST_MB_PER_ORDER_EVENT,
        "wall_seconds": est_events * _EST_SECONDS_PER_ORDER_EVENT,
    }


# Below this fraction of total RAM, run as a single session as before. At or above
# it, auto-split into legs (market_sim.run_scenario_in_legs) rather than either
# running one huge session or making the user manually reach for a different
# function -- discussed and confirmed with the user directly, see project memory.
# Deliberately well below _RISKY_RUN_THRESHOLD (0.35, the "confirm before running
# anyway" line further down) -- with auto-splitting in place, a plain "Run
# simulation" click should essentially never need to hit that warning anymore; it
# stays as a safety net for the one case auto-split doesn't cover (`min_trades`
# set -- see _AUTO_SPLIT_TARGET_FRACTION's doc comment) or a population so large
# even a floor-sized leg would still be risky.
_AUTO_SPLIT_THRESHOLD_FRACTION = 0.15
# Each auto-picked leg targets this fraction of total RAM -- comfortably under
# _AUTO_SPLIT_THRESHOLD_FRACTION itself, so a many-leg run doesn't produce legs
# that are *themselves* borderline risky.
_AUTO_SPLIT_TARGET_FRACTION = 0.08
_AUTO_SPLIT_MIN_LEG_SECONDS = 10.0


def _pick_auto_split_leg_seconds(n_agents: float, available_mb: float) -> float:
    """Solves `_estimate_run_cost(n_agents, leg_seconds)["peak_mb"] ==
    _AUTO_SPLIT_TARGET_FRACTION * available_mb` for `leg_seconds` -- the linear
    inverse of `_estimate_run_cost`'s `peak_mb` formula. Floored at
    `_AUTO_SPLIT_MIN_LEG_SECONDS` so an extreme population doesn't produce
    sub-10-second, mostly-overhead legs; a population so large even that floor is
    risky falls through to the existing manual confirm-and-run-anyway warning
    instead (this function doesn't guarantee safety by itself, just picks a
    reasonable size for the common case)."""
    per_second_mb = max(1e-9, n_agents * _EST_ORDER_EVENTS_PER_AGENT_SECOND * _EST_MB_PER_ORDER_EVENT)
    return max(_AUTO_SPLIT_MIN_LEG_SECONDS, (_AUTO_SPLIT_TARGET_FRACTION * available_mb) / per_second_mb)


def _available_ram_mb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / 1e6
    except Exception:  # noqa: BLE001
        # streamlit itself depends on psutil, so this is only a fallback for the
        # unusual case it's somehow missing -- a conservative assumption (8GB) so
        # the warning below errs toward showing up rather than silently not.
        return 8000.0


def _default_cfg() -> dict:
    return cfgmod.load_config_dict(SCENARIOS["Realistic mix"])


def _run_worker(run_fn, holder: dict) -> None:
    """Runs on a background thread -- must never call any `st.*` function (Streamlit
    widget calls are only valid on the main script-running thread); it only talks to
    the Rust engine (via `run_fn`, a zero-arg closure -- either a fresh
    `run_simulation_cancellable(...)` or a `continue_simulation_cancellable(...)` on
    an existing checkpoint) and writes to a plain dict the main thread polls."""
    try:
        holder["result"] = run_fn()
    except Exception as e:  # noqa: BLE001
        holder["error"] = f"{type(e).__name__}: {e}"


def _start_sim_thread(
    make_run_fn,
    config_toml: str,
    *,
    is_continue: bool,
    base_result: dict | None,
    target_seconds: float | None,
    is_leg_run: bool = False,
    leg_progress: dict | None = None,
) -> None:
    """Shared by the sidebar's "Run simulation" button and the Results tab's
    "Continue simulation" button: spins up the background thread and the session
    state `_poll_sim_status` (sidebar) watches regardless of which UI section
    started it. `make_run_fn(token, progress)` builds the zero-arg closure the thread
    calls -- takes the freshly-created `CancelToken`/`ProgressHandle` as plain
    arguments (not read back from `st.session_state` inside the thread, which only
    the main script-running thread should touch). `target_seconds` (the nominal
    simulated-time length of this run -- `session_seconds` for a fresh run, or
    however-far-along-already + `additional_seconds` for a continuation) is what the
    "currently at Xs" display below measures progress against; pass `None` if it
    can't be known upfront.

    `is_leg_run`/`leg_progress`: set when `make_run_fn`'s closure calls
    `run_scenario_in_legs` (auto-split) rather than a single `run_simulation*` --
    `holder["result"]` will be a `list[LegResult]`, not a single result dict, and
    `_poll_sim_status` branches on `sim_is_leg_run` to handle that. `leg_progress`
    is a plain dict the caller's own `on_leg_saved` callback (baked into
    `make_run_fn`'s closure already) writes `leg_index`/`n_legs`/`last_meta` into
    from the background thread -- stored here only so the polling fragment can read
    it; this function doesn't write to it itself.
    """
    st.session_state.run_error = None
    token = CancelToken()
    progress = ProgressHandle()
    holder = {"result": None, "error": None}
    thread = threading.Thread(target=_run_worker, args=(make_run_fn(token, progress), holder), daemon=True)
    st.session_state.sim_token = token
    st.session_state.sim_progress = progress
    st.session_state.sim_target_seconds = target_seconds
    st.session_state.sim_holder = holder
    st.session_state.sim_config_toml = config_toml
    st.session_state.sim_start_time = time.time()
    st.session_state.sim_is_continue = is_continue
    st.session_state.sim_base_result = base_result
    st.session_state.sim_is_leg_run = is_leg_run
    st.session_state.sim_leg_progress = leg_progress
    st.session_state.sim_thread = thread
    thread.start()


def _loaded_run_to_result(loaded: dict) -> dict:
    """market_sim.load_run(...)'s DataFrame-keyed shape -> the same columnar
    dict[str, list/ndarray] shape run_simulation(...)/active_result use everywhere
    else, so the Results tab renders a freshly-run, stopped-early, or reloaded-
    from-disk result identically. Carries over `checkpoint` (present when the saved
    run has one -- see `market_sim.has_checkpoint`) so a loaded run can still be
    continued, not just viewed.

    Uses `df_to_columnar` (`{col: df[col].to_numpy() ...}`), NOT `df.to_dict("list")`
    -- measured ~21s for a real 1.4M-row/8M-order-event saved run with the latter
    (every cell individually boxed into a Python object) vs. under a millisecond for
    the former; see its docstring in market_sim/export.py.
    """
    result = {
        "trades": df_to_columnar(loaded["trades"]),
        "order_events": df_to_columnar(loaded["order_events"]),
        "snapshots": df_to_columnar(loaded["orderbook_snapshots"]),
        "agent_states": df_to_columnar(loaded["agent_state"]),
        "meta": loaded["meta"],
    }
    if "checkpoint" in loaded:
        result["checkpoint"] = loaded["checkpoint"]
    return result


def _discover_saved_runs() -> list:
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(RUNS_ROOT, "**", "run_metadata.json"), recursive=True))


_LEG_DIR_RE = re.compile(r"^leg_\d+$")


def _group_saved_runs(saved: list) -> dict:
    """Groups `_discover_saved_runs()`'s flat directory list by run: a
    `market_sim.run_scenario_in_legs` sequence's `leg_NNNN` directories all group
    under their shared parent (so a 200-leg run shows as one entry with a leg
    picker, not 200 separate dropdown rows each showing an indistinguishable,
    truncated `.../leg_00NN`), everything else is its own single-entry group.
    Returned dict preserves insertion order (Python 3.7+), sorted so the parent of
    a leg group sorts by its own name rather than its first leg's."""
    groups: dict = {}
    for path in saved:
        parent, leaf = os.path.dirname(path), os.path.basename(path)
        key = parent if _LEG_DIR_RE.match(leaf) else path
        groups.setdefault(key, []).append(path)
    for legs in groups.values():
        legs.sort()
    return dict(sorted(groups.items()))


def _leg_group_for_path(path: str | None) -> str | None:
    """If `path` is one leg of a `run_scenario_in_legs` sequence (a `leg_NNNN`
    directory), returns its parent (the group key `_group_saved_runs` would use);
    `None` for a plain single-run path or `None` itself. Recomputed fresh from
    whatever's currently loaded (`last_save_path`) rather than cached in its own
    session_state flag, so it can't go stale the way a separately-tracked "current
    group" flag could (e.g. after loading an unrelated single run following a
    multi-leg one)."""
    if not path:
        return None
    parent, leaf = os.path.dirname(path), os.path.basename(path)
    return parent if _LEG_DIR_RE.match(leaf) else None


def _activate_result(result: dict, config_toml: str, source: str, save_path: str | None = None) -> None:
    st.session_state.active_result = result
    st.session_state.active_config_toml = config_toml
    st.session_state.active_source = source
    # A leg from a run made after `run_scenario_in_legs`'s `compute_market_health`
    # default already has this persisted (`<leg_dir>/market_health.json`) -- reuse
    # it rather than recomputing from `result` (already in memory, so recomputing
    # is "just" ~11-13s of CPU wasted, not a reload, but still no reason to pay it
    # twice). An older leg without one gets computed here AND persisted, same as
    # `_leg_market_health`'s own miss path -- either way this leg never needs
    # computing again after this.
    leg_group = _leg_group_for_path(save_path)
    persisted = market_health.load_metrics(save_path) if leg_group else None
    if persisted is not None:
        st.session_state.metrics = persisted
    else:
        st.session_state.metrics = market_health.compute_run_metrics(result)
        if leg_group:
            try:
                market_health.save_metrics(st.session_state.metrics, save_path)
            except OSError:
                pass
    st.session_state.last_save_path = save_path
    # Every activation defaults to whole-run overview mode when the loaded run is
    # part of a leg group (see _render_price_volume) -- the one exception, drilling
    # into a specific leg's full single-trade-resolution chart via "Load for full
    # detail", sets this back to True itself right after calling this function.
    st.session_state.leg_drilldown = False
    # Remember this leg's metrics (keyed by leg dir) so the Results tab's
    # per-metric line charts across legs (see _render_market_health/
    # _compute_all_leg_metrics) count this leg as already known and don't
    # recompute it -- this dict also grows in bulk via the "Compute market-health
    # for the remaining N legs" button there.
    if leg_group:
        st.session_state.setdefault("leg_metrics_by_dir", {})[save_path] = dict(st.session_state.metrics)
    # Bumped so cached, version-keyed derivations (see _trades_df_cached below) know
    # to recompute -- without this a long session's results stay slow to browse
    # (every widget interaction anywhere re-sent every trade to the browser); see
    # the fragment wrapping and downsampling around the Results tab for the rest of
    # the fix.
    st.session_state.result_version = st.session_state.get("result_version", 0) + 1


# `cache_resource` (not `cache_data`): `cache_data` always hands back a fresh
# deep copy on every call (its whole safety contract), which for a large run's
# multi-hundred-MB-to-multi-GB DataFrame turned into real, repeated cost on every
# widget interaction that re-ran this fragment (a full copy just to read a column).
# Nothing downstream mutates these frames in place -- every filter/search below
# does `df[mask]` (a new frame), never `df.loc[...] = ...` -- so sharing the same
# object across reruns is safe, and this is a single-user local dashboard (see
# README) so there's no other session to leak state to either.
_LOW_CARDINALITY_COLS = {
    "trades": ["taker_side", "maker_type", "maker_variant", "taker_type", "taker_variant"],
    "snapshots": ["side"],
    "agent_states": ["bot_type", "variant"],
}


def _to_categorical(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """A handful of string columns (bot type/variant, buy/sell, ...) repeat across
    every row -- `category` dtype stores each distinct value once and every row as a
    small integer code instead of a full Python string object per cell, which adds
    up fast at millions of rows. Same values, same comparisons/groupby/filtering
    behavior as plain `object` columns everywhere this dashboard uses them; purely a
    more compact encoding, not a data change."""
    for col in _LOW_CARDINALITY_COLS.get(key, []):
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# `max_entries=1`: only the *current* active result's DataFrames need to stay
# cached -- without this, every new run (result_version bump) adds another cache
# entry keyed on the new version without evicting the old one, so a session with
# several large runs would keep *all* of their trades/snapshots/agent_state
# DataFrames resident in memory simultaneously (the previous run's are no longer
# reachable through `st.session_state.active_result` and won't be re-queried at
# their old version number, so there's nothing lost by evicting them).
@st.cache_resource(show_spinner=False, max_entries=1)
def _trades_df_cached(version: int) -> pd.DataFrame:
    return _to_categorical(pd.DataFrame(st.session_state.active_result["trades"]), "trades")


@st.cache_resource(show_spinner=False, max_entries=1)
def _snapshots_df_cached(version: int) -> pd.DataFrame:
    return _to_categorical(pd.DataFrame(st.session_state.active_result.get("snapshots", {})), "snapshots")


@st.cache_resource(show_spinner=False, max_entries=1)
def _agent_state_df_cached(version: int) -> pd.DataFrame:
    return _to_categorical(pd.DataFrame(st.session_state.active_result.get("agent_states", {})), "agent_states")


# How many candles each leg is summarized into for the multi-leg overview -- a
# fixed, modest resolution computed once per leg (cached) and composed across
# however many legs are in view. Doesn't need to be large: the overview's whole
# purpose is a whole-run shape at a glance, with "Load this leg" for full detail.
_LEG_SUMMARY_CANDLES = 300


@st.cache_resource(show_spinner=False, max_entries=2000)
def _leg_candle_summary(leg_dir: str) -> dict:
    """Per-asset OHLC summary for one leg, computed from just the trades columns
    actually needed (`load_trades_columns`, not the full trades table -- skips the
    bot-type/variant/agent-id columns entirely) at a fixed, modest resolution.
    Cached per leg directory so re-viewing/re-zooming the overview doesn't re-read
    parquet or recompute -- only a *new* leg (not seen this session) costs
    anything; `max_entries` is generously above any realistic leg count (each
    entry is tiny -- a few hundred candles x however many assets), so nothing
    gets evicted just from having viewed a lot of legs. Returns `{asset_id:
    DataFrame}` (each `market_sim.candles.CANDLE_COLUMNS` shape, in that leg's
    own absolute simulated-time coordinates -- legs already don't need rebasing
    to concatenate, since `run_scenario_in_legs` never resets the clock between
    legs). Trades within a leg are already time-sorted per asset (the parent
    table is written in execution order and filtering preserves that), so no
    extra sort is needed here.
    """
    df = load_trades_columns(leg_dir)
    out = {}
    for aid, sub in df.groupby("asset_id", sort=False):
        t = sub["time"].to_numpy()
        out[int(aid)] = candles_from_trades(t, sub["price"].to_numpy(), (sub["price"] * sub["qty"]).to_numpy(), float(t.min()), float(t.max()), _LEG_SUMMARY_CANDLES)
    return out


def _warm_leg_summaries(leg_dirs) -> None:
    """Populates `_leg_candle_summary`'s cache for every leg in `leg_dirs`, in
    parallel. Reading + aggregating one leg is dominated by pyarrow's parquet
    read and numpy's `reduceat`-based aggregation (see `candles.py`), both of
    which release the GIL for their actual work, so a thread pool gives a real
    (~1.5x measured on this machine) speedup over a plain loop despite Python's
    GIL -- worth it since this runs synchronously behind a spinner on first
    view of a many-legs run (measured ~200 legs -> ~13-20s either way; this is
    a one-time-per-session cost, not a per-rerun one, courtesy of the cache
    above)."""
    import concurrent.futures

    # No cache-hit check before dispatching: `_leg_candle_summary` is itself
    # cached, so re-calling it for an already-warmed leg just returns the cached
    # value quickly rather than recomputing -- simpler than tracking cache
    # membership separately, and cheap enough not to matter.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_leg_candle_summary, leg_dirs))


@st.cache_resource(show_spinner=False, max_entries=2000)
def _leg_market_health(leg_dir: str) -> dict:
    """One leg's `market_health.compute_run_metrics` output. Checks for a
    persisted `market_health.json` first (`market_health.load_metrics` --
    written automatically by `run_scenario_in_legs` right after each leg since
    `compute_market_health=True` became its default, so most legs from a run
    made after that hit this path and are essentially free, just a JSON read).
    Only on a miss (an older run made before persisting existed, or one made
    with `compute_market_health=False`) does it fall back to the actually heavy
    path: loading every export file (trades, order_events, snapshots,
    agent_state) for the leg -- unlike `_leg_candle_summary` above, which reads
    only 4 narrow columns -- and running the full report metric suite. Measured
    on this machine's real 200-leg run: ~2s to load + ~11-13s to compute per leg
    (~1.2-1.5M trades, ~8M order events each) -- genuinely heavy, which is why
    computing this for *every* leg of an old run is an explicit opt-in action
    (see `_compute_all_leg_metrics`) with a real progress bar, not something
    done silently on every render the way the candle overview is. Persists the
    result on a miss too (not just the in-process `st.cache_resource` above,
    which is lost on dashboard restart), so a leg computed lazily here only
    ever needs it once, same guarantee a fresh run's own automatic computation
    gives."""
    persisted = market_health.load_metrics(leg_dir)
    if persisted is not None:
        return persisted
    loaded = load_run(leg_dir)
    run = {
        "trades": loaded["trades"],
        "order_events": loaded["order_events"],
        "orderbook_snapshots": loaded["orderbook_snapshots"],
        "agent_state": loaded["agent_state"],
    }
    metrics = market_health.compute_run_metrics(run)
    try:
        market_health.save_metrics(metrics, leg_dir)
    except OSError:
        pass  # e.g. a read-only run directory -- the in-process cache above still avoids recomputing this session
    return metrics


def _compute_all_leg_metrics(leg_dirs: tuple, known: dict) -> dict:
    """Computes `_leg_market_health` for every leg in `leg_dirs` not already in
    `known`, showing a live progress bar as legs complete (unlike
    `_warm_leg_summaries`'s silent spinner -- appropriate there since it's a
    ~15-20s one-time cost; this can genuinely take many minutes for a large
    multi-leg run, so a static spinner with no feedback would look hung).
    `max_workers=2`, not 8 like the candle summaries: this is CPU-heavy pandas/
    numpy work (variance ratio, autocorrelation, VPIN, Benford, ...), not
    I/O-bound parquet reads, so oversubscribing threads past the machine's core
    count buys nothing and can even slow things down via contention -- but the
    real reason it's this low is memory, not CPU: each in-flight leg holds its
    full trades/order_events/snapshots/agent_state DataFrames at once (unlike
    the narrow 4-column reads the candle summaries use), and a first attempt at
    4 workers measured RSS climbing to 4.3GB in 15s on this machine with only
    ~5GB free at the time -- real evidence of a genuine OOM risk, not a
    theoretical one, on the same machine this whole project's memory-safety work
    has been about. Returns a NEW dict merging `known` with whatever got
    computed (callers should assign this back into session_state)."""
    import concurrent.futures

    todo = [d for d in leg_dirs if d not in known]
    out = dict(known)
    if not todo:
        return out
    progress_bar = st.progress(0.0)
    status = st.empty()
    status.caption(f"Computing market-health for {len(todo)} legs (0/{len(todo)})...")
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(_leg_market_health, d): d for d in todo}
        for fut in concurrent.futures.as_completed(futures):
            d = futures[fut]
            out[d] = fut.result()
            done += 1
            progress_bar.progress(done / len(todo))
            status.caption(f"Computing market-health for {len(todo)} legs ({done}/{len(todo)})...")
    progress_bar.empty()
    status.empty()
    return out


def _leg_group_time_range(leg_dirs: tuple) -> tuple[float, float]:
    """Whole-group [min, max] simulated time, from the already-cached (or cheaply
    computed) per-leg candle summaries -- no trades.parquet re-read needed, unlike
    the actual search below."""
    _warm_leg_summaries(leg_dirs)
    starts, ends = [], []
    for d in leg_dirs:
        for df in _leg_candle_summary(d).values():
            if not df.empty:
                starts.append(df["bucket_start"].min())
                ends.append(df["bucket_end"].max())
    return (float(min(starts)), float(max(ends))) if starts else (0.0, 1.0)


_TRADE_SEARCH_COLS = ["asset_id", "price", "qty", "taker_side", "maker_agent_id", "taker_agent_id", "maker_type", "maker_variant", "taker_type", "taker_variant"]


def _search_all_legs(leg_dirs: tuple, asset_filter, bot_type_filter, agent_id_val, free_text: str, time_range: tuple, qty_min: float, row_cap: int = 5000, batch_size: int = 16):
    """Streams every leg's `trades.parquet` (all columns -- unlike the narrow
    time/price/qty/asset_id read the candle summaries use, search needs the
    bot-type/agent-id/side columns too), applies the same filters `_render_trade_
    search` uses per leg, and keeps only the matches -- never holds more than
    `batch_size` legs' full trades tables in memory at once (processed in
    batches, not all 200+ dispatched to the thread pool simultaneously), and
    stops scanning further legs once `row_cap` matches are already found (a
    permissive-enough filter, e.g. no free text and no other filters, would
    otherwise match literally every trade in every leg -- reintroducing the same
    unbounded-memory problem splitting into legs exists to avoid). Measured on
    this machine's real 200-leg run: ~0.5s/leg to read a full trades.parquet
    (~100s serially for 200 legs, well under market-health's per-leg cost),
    parallelized here the same way the candle summaries are.

    Returns `(matches_df, legs_scanned, total_legs, stopped_early)`."""
    import concurrent.futures

    def one(leg_dir):
        df = pd.read_parquet(os.path.join(leg_dir, "trades.parquet"))
        f = df[(df["time"] >= time_range[0]) & (df["time"] <= time_range[1]) & (df["qty"] >= qty_min)]
        if asset_filter:
            f = f[f["asset_id"].isin(asset_filter)]
        if bot_type_filter:
            f = f[f["maker_type"].isin(bot_type_filter) | f["taker_type"].isin(bot_type_filter)]
        if agent_id_val is not None:
            f = f[(f["maker_agent_id"] == agent_id_val) | (f["taker_agent_id"] == agent_id_val)]
        if free_text.strip():
            combined = f[_TRADE_SEARCH_COLS[0]].astype(str)
            for col in _TRADE_SEARCH_COLS[1:]:
                combined = combined + " " + f[col].astype(str)
            f = f[combined.str.lower().str.contains(free_text.lower(), regex=False, na=False)]
        return f

    parts, legs_scanned, total = [], 0, 0
    stopped_early = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for i in range(0, len(leg_dirs), batch_size):
            batch = leg_dirs[i : i + batch_size]
            for part in ex.map(one, batch):
                parts.append(part)
                total += len(part)
                legs_scanned += 1
            if total >= row_cap:
                stopped_early = legs_scanned < len(leg_dirs)
                break
    matches = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_TRADE_SEARCH_COLS + ["time"])
    return matches, legs_scanned, len(leg_dirs), stopped_early


@st.cache_resource(show_spinner=False, max_entries=16)
def _multi_leg_candles_for_asset(leg_dirs: tuple, asset_id: int) -> pd.DataFrame:
    """Concatenates every leg's cached per-asset candle summary for one asset into
    one series spanning the whole multi-leg run -- the raw material
    `resample_candles` (called per-view, not cached, since it's cheap once the
    inputs above are) turns into the actual displayed overview at whatever
    resolution the current view needs. Call `_warm_leg_summaries(leg_dirs)`
    first if legs might not be cached yet -- this function itself just reads the
    cache (via `_leg_candle_summary`, which computes-and-caches on a miss, but
    serially, one leg at a time, if nothing warmed it first)."""
    frames = []
    for leg_dir in leg_dirs:
        summary = _leg_candle_summary(leg_dir).get(asset_id)
        if summary is not None and not summary.empty:
            frames.append(summary)
    if not frames:
        return pd.DataFrame(columns=["bucket_start", "bucket_end", "open", "high", "low", "close", "volume", "n_trades"])
    return pd.concat(frames, ignore_index=True).sort_values("bucket_start").reset_index(drop=True)


def _downsample_minmax(t: "pd.Series", y: "pd.Series", max_points: int):
    """For a time series with far more points than can usefully be plotted, keep
    only each bucket's min and max (not a naive every-Nth-point decimation), so
    sharp spikes -- exactly the kind of thing worth spotting in a long run, e.g. a
    liquidation cascade -- survive downsampling instead of being averaged away.
    Returns (t, y) unchanged if already within budget. Requires `t` sorted ascending
    (every call site already sorts by time for plotting anyway) -- that's what lets
    this use a single `reduceat` pass per bucket instead of a hash-based groupby,
    ~500x faster in practice (4.9ms vs 2.4s on a 617k-point series): sorted time
    means every bucket is one contiguous run, not scattered rows to be gathered."""
    n = len(t)
    if n <= max_points or max_points < 2:
        return t, y
    t_arr = t.to_numpy()
    y_arr = y.to_numpy()
    n_buckets = max(1, max_points // 2)
    span = t_arr[-1] - t_arr[0]
    bucket_width = span / n_buckets if span > 0 else 1.0
    bucket_idx = np.clip(((t_arr - t_arr[0]) / bucket_width).astype(np.int64), 0, n_buckets - 1)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(bucket_idx)) + 1))
    seg_t = t_arr[starts]
    min_vals = np.minimum.reduceat(y_arr, starts)
    max_vals = np.maximum.reduceat(y_arr, starts)
    # Both points of a bucket share its start time: at the zoom level where buckets
    # are even visible as distinct, this reads as a small vertical wick (min-to-max
    # excursion within that bucket) rather than a misleading diagonal, which is the
    # right visual for "here's the range trade prices covered in this slice".
    t_out = np.repeat(seg_t, 2)
    y_out = np.empty(2 * len(seg_t), dtype=y_arr.dtype)
    y_out[0::2] = min_vals
    y_out[1::2] = max_vals
    return pd.Series(t_out), pd.Series(y_out)


# Discussed with the user directly: candle count target of ~100 (so 100-200 are
# visible at once), time-window bucketing (not trade-count -- a volume bar under a
# trade-count-bucketed candle would be near-constant across candles, since every
# candle would hold the same number of trades almost by construction, losing the
# "which period was actually busy" signal a volume chart is for).
DEFAULT_TARGET_CANDLES = 100


def _extract_box_x_range(box_list):
    """Plotly's `plotly_selected` box-select event payload (what Streamlit's
    `on_select="rerun"` passes through) isn't formally documented down to the exact
    key names, so this tries the two shapes Plotly.js is known to use rather than
    assuming one -- returns `None` (leaving the current view alone) if neither
    matches, rather than raising."""
    if not box_list:
        return None
    b = box_list[0]
    x = b.get("x")
    if isinstance(x, (list, tuple)) and len(x) == 2:
        try:
            return float(min(x)), float(max(x))
        except (TypeError, ValueError):
            return None
    x0, x1 = b.get("x0"), b.get("x1")
    if x0 is not None and x1 is not None:
        try:
            return float(min(x0, x1)), float(max(x0, x1))
        except (TypeError, ValueError):
            return None
    return None


def _render_candlestick_chart(key_prefix: str, title: str, full_start: float, full_end: float, get_candles, get_overlays=None):
    """One price + volume chart, switchable between candlestick and plain line,
    with drag-to-zoom and explicit numeric start/end controls (a guaranteed-to-work
    alternative/fallback to dragging, and a quick way to jump to an exact time --
    see `_extract_box_x_range`'s doc comment on why drag alone isn't relied on as
    the only way in).

    Both chart types are driven by the same adaptive candle bucketing (`get_candles`
    below) -- "Line" mode connects each bucket's close price instead of drawing an
    OHLC candle, so it still gets more detailed as you zoom in rather than being a
    fixed-resolution line over the whole run. It's not the old raw-per-trade line
    (that view is gone, replaced by candlestick's single-trade resolution at max
    zoom) -- it's the same underlying data, just drawn as a line for readers who
    find OHLC candles harder to scan at a glance.

    `get_candles(view_start, view_end, target_candles) -> DataFrame` (shape:
    `market_sim.candles.CANDLE_COLUMNS`) supplies the candle data -- lets the same
    zoom/render logic serve both the single-run chart (computed directly from
    in-memory trades) and the multi-leg overview (resampled from cached per-leg
    summaries) via different callables. `get_overlays(view_start, view_end) ->
    list[(name, t, v, dash, color)]`, if given, draws thin lines over the candles
    (best bid/ask, fundamental value).

    `key_prefix` should be unique per chart *and* per loaded result (e.g. include
    the asset id and `result_version`) so switching to a different run/asset starts
    at a fresh full-range view rather than keeping a stale zoom from whatever was
    loaded before.
    """
    start_key, end_key, target_key = f"{key_prefix}_view_start", f"{key_prefix}_view_end", f"{key_prefix}_target_candles"
    pending_key = f"{key_prefix}_pending_view"
    # A view-range change requested *within* a run (Reset zoom clicked, or a
    # drag-select resolved below) can't be written to session_state directly --
    # Streamlit refuses once the start/end number_input widgets below have already
    # instantiated in this same run. Both request paths instead stage the new
    # range here and call st.rerun(); this block applies it on the *next* run,
    # before those widgets (re)instantiate, which is allowed.
    pending = st.session_state.pop(pending_key, None)
    if pending is not None:
        st.session_state[start_key], st.session_state[end_key] = pending

    if start_key not in st.session_state:
        st.session_state[start_key] = float(full_start)
    if end_key not in st.session_state:
        st.session_state[end_key] = float(full_end)
    if target_key not in st.session_state:
        st.session_state[target_key] = DEFAULT_TARGET_CANDLES
    # Clamp to the current full range in case it shifted (e.g. session_state
    # entries left over at a stale key) -- same reasoning as run_max_session_seconds
    # elsewhere in this file: fix up the stored value directly, not just `value=`,
    # since a widget with an explicit key ignores `value=` once its entry exists.
    st.session_state[start_key] = min(max(st.session_state[start_key], float(full_start)), float(full_end))
    st.session_state[end_key] = min(max(st.session_state[end_key], st.session_state[start_key] + 1e-9), float(full_end))

    chart_type = st.radio("Chart type", ["Candlestick", "Line"], horizontal=True, key=f"{key_prefix}_chart_type")

    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    step = max((full_end - full_start) / 100, 1.0)
    view_start = c1.number_input("View start (s)", min_value=float(full_start), max_value=float(full_end), step=step, key=start_key)
    view_end = c2.number_input("View end (s)", min_value=float(full_start), max_value=float(full_end), step=step, key=end_key)
    target_candles = c3.number_input("Target candles", min_value=20, max_value=500, step=10, key=target_key, help="Chart aims to show target-2x this many candles for the current view; this is the X in 'X to 2X'.")
    if c4.button("Reset zoom", key=f"{key_prefix}_reset_zoom"):
        st.session_state[pending_key] = (float(full_start), float(full_end))
        st.rerun()

    if view_end <= view_start:
        st.caption("View end must be after view start.")
        return

    candles = get_candles(view_start, view_end, int(target_candles))
    if candles.empty:
        st.caption("No trades in this range.")
        return

    mid = (candles["bucket_start"] + candles["bucket_end"]) / 2.0
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(x=mid, open=candles["open"], high=candles["high"], low=candles["low"], close=candles["close"], name="price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"),
            row=1,
            col=1,
        )
    else:
        # "mean", not "close": a genuine time-span average of the trade prices in
        # each bucket, matching how the line chart worked before the candlestick
        # rebuild (smooth, representative of the whole bucket) rather than
        # plotting a single representative trade (the bucket's last trade, which
        # is what "close" is) or a min/max extremum.
        fig.add_trace(go.Scattergl(x=mid, y=candles["mean"], mode="lines", name="price", line=dict(width=1.5, color="#1f77b4")), row=1, col=1)
    for name, ot, ov, dash, color in get_overlays(view_start, view_end) if get_overlays else []:
        if len(ot):
            fig.add_trace(go.Scattergl(x=ot, y=ov, mode="lines", name=name, line=dict(width=1, dash=dash, color=color)), row=1, col=1)
    fig.add_trace(go.Bar(x=mid, y=candles["volume"], name="volume (notional)", marker_color="grey"), row=2, col=1)
    fig.update_layout(height=440, title=title, legend=dict(orientation="h", y=1.08), margin=dict(t=60, b=20), xaxis_rangeslider_visible=False)
    fig.update_xaxes(title_text="simulated time (s)", row=2, col=1)

    avg_trades = candles["n_trades"].mean() if len(candles) else 0.0
    resolution_note = "single-trade resolution" if avg_trades <= 1.5 else f"~{avg_trades:.1f} trades/candle on average"
    st.caption(f"{len(candles)} candles, {view_end - view_start:,.3g}s span ({resolution_note}). Drag a box on the chart to zoom into a region, or set exact start/end above.")

    event = st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_chart", on_select="rerun", selection_mode="box")
    if event and event.get("selection"):
        box_range = _extract_box_x_range(event["selection"].get("box"))
        if box_range and box_range[1] > box_range[0]:
            new_start, new_end = box_range
            if (round(new_start, 6), round(new_end, 6)) != (round(view_start, 6), round(view_end, 6)):
                st.session_state[pending_key] = (max(float(full_start), new_start), min(float(full_end), new_end))
                st.rerun()


if "cfg" not in st.session_state:
    st.session_state.cfg = _default_cfg()
for _key, _default in [
    ("active_result", None), ("active_config_toml", None), ("active_source", None),
    ("metrics", None), ("last_save_path", None), ("run_error", None),
    ("sim_thread", None), ("sim_token", None), ("sim_holder", None),
    ("sim_config_toml", None), ("sim_start_time", None),
    ("sim_is_continue", False), ("sim_base_result", None),
    ("sim_progress", None), ("sim_target_seconds", None),
    ("sim_is_leg_run", False), ("sim_leg_progress", None),
    ("leg_metrics_by_dir", {}), ("leg_drilldown", False),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

cfg = st.session_state.cfg
cfg.setdefault("venues", [])
cfg.setdefault("fees", {}).setdefault("default_maker_fee", 0.0002)
cfg["fees"].setdefault("default_taker_fee", 0.0005)
cfg.setdefault("latency", {}).setdefault("default_latency_secs", 0.05)
cfg.setdefault("mempool", {}).setdefault("enabled", False)
cfg["mempool"].setdefault("reveal_delay_secs", 0.4)
cfg["mempool"].setdefault("large_order_notional_threshold", 20000.0)
cfg.setdefault("liquidation", {})
cfg["liquidation"].setdefault("maintenance_margin_buffer", 0.8)
cfg["liquidation"].setdefault("sweep_interval_secs", 1.0)
cfg["liquidation"].setdefault("staggered_slices", 5)
cfg["liquidation"].setdefault("full_liquidation_fraction", 0.5)
cfg.setdefault("market", {})
cfg["market"].setdefault("snapshot_interval_secs", 1.0)
cfg["market"].setdefault("snapshot_depth_levels", 10)
cfg["market"].setdefault("mark_price_ema_halflife_secs", 30.0)
cfg.setdefault("export", {}).setdefault("out_dir", "runs/dashboard")
cfg["export"].setdefault("write_snapshots", True)

# ---------------------------------------------------------------------------------
# Sidebar: scenario loader + run button
# ---------------------------------------------------------------------------------
with st.sidebar:
    st.title("Bot Market Sim")
    st.caption("Configure, run, and explore -- all local.")

    chosen = st.selectbox("Load a base scenario", list(SCENARIOS), key="scenario_choice")
    if st.button("Load scenario (discards current edits)", key="load_scenario_btn"):
        st.session_state.cfg = cfgmod.load_config_dict(SCENARIOS[chosen])
        st.rerun()

    st.divider()
    n_agents = sum(int(g.get("count", 0)) for g in cfg.get("population", {}).get("groups", []))
    n_assets = len(cfg.get("assets", []))
    st.metric("Total agents", n_agents)
    st.metric("Assets configured", n_assets)

    st.divider()
    sim_running = st.session_state.sim_thread is not None

    if not sim_running:
        session_seconds = float(cfg.get("run", {}).get("session_seconds", 0.0) or 0.0)
        max_session_seconds = float(cfg.get("run", {}).get("max_session_seconds") or session_seconds)
        has_min_trades = bool(cfg.get("run", {}).get("min_trades"))
        est = _estimate_run_cost(n_agents, session_seconds)
        est_worst = _estimate_run_cost(n_agents, max_session_seconds)
        available_mb = _available_ram_mb()

        # Auto-split: below _AUTO_SPLIT_THRESHOLD_FRACTION, run as a single session
        # as before; at or above it, run as an auto-sized leg sequence instead --
        # discussed and confirmed with the user directly (see project memory), so a
        # plain "Run simulation" click on an oversized config no longer needs to hit
        # the risky-confirm warning below at all, it just transparently becomes a
        # multi-leg run. Doesn't apply when `min_trades` is set: run_scenario_in_legs
        # has no notion of a trade-count floor across an unknown final length (each
        # leg runs for exactly leg_seconds, see its own docstring), so that case
        # keeps going through the single-run path and its own warning.
        auto_split = (not has_min_trades) and est["peak_mb"] > _AUTO_SPLIT_THRESHOLD_FRACTION * available_mb

        if auto_split:
            leg_seconds_raw = _pick_auto_split_leg_seconds(n_agents, available_mb)
            n_legs = max(1, math.ceil(session_seconds / leg_seconds_raw))
            leg_seconds = session_seconds / n_legs  # even legs summing exactly to session_seconds
            est_leg = _estimate_run_cost(n_agents, leg_seconds)
            st.caption(
                f"This session ({session_seconds:,.0f}s, ~{est['peak_mb']:,.0f} MB if run as one) will "
                f"auto-split into **{n_legs} legs** of ~{leg_seconds:,.0f}s each (~{est_leg['peak_mb']:,.0f} MB "
                f"peak per leg) -- see README's \"Splitting a long run into legs\". The Multi-leg overview tab "
                f"will build automatically once it's done."
            )
            # Safety net: the target per-leg budget is chosen well under the risky
            # threshold, but a population so large that even the minimum leg length
            # doesn't fit still needs the same explicit acknowledgement the
            # single-run path would have required.
            risky = est_leg["peak_mb"] > 0.35 * available_mb
            confirmed_large = True
            if risky:
                st.warning(
                    f"⚠️ Even a single auto-split leg is estimated at ~{est_leg['peak_mb']/1000:,.1f} GB -- this "
                    f"population is large enough that {_AUTO_SPLIT_MIN_LEG_SECONDS:.0f}s-minimum legs aren't "
                    f"enough headroom on a ~{available_mb/1000:,.1f} GB machine. Consider fewer agents."
                )
                confirmed_large = st.checkbox("I understand the risk -- run it anyway", key="confirm_large_run")
            if st.button("Run simulation (auto-split)", type="primary", width="stretch", key="run_sim_btn", disabled=risky and not confirmed_large):
                cfg_snapshot = copy.deepcopy(st.session_state.cfg)
                out_dir = os.path.join(RUNS_ROOT, "dashboard_legs", time.strftime("%Y%m%d_%H%M%S"))
                leg_progress = {"leg_index": 0, "n_legs": n_legs, "last_meta": None, "out_dir": out_dir}

                def make_run_fn(token, progress, cfg_snapshot=cfg_snapshot, leg_seconds=leg_seconds, n_legs=n_legs, out_dir=out_dir, leg_progress=leg_progress):
                    def on_leg_saved(leg):
                        # Background thread -- writes to a plain dict only, same
                        # rule as _run_worker itself (no st.* calls here).
                        leg_progress["leg_index"] = leg.index
                        leg_progress["last_meta"] = leg.meta

                    return lambda: run_scenario_in_legs(
                        cfg_snapshot, leg_seconds, n_legs, out_dir, scenario_name="dashboard (auto-split)", cancel_token=token, progress=progress, on_leg_saved=on_leg_saved
                    )

                _start_sim_thread(
                    make_run_fn,
                    cfgmod.dump_config_str(cfg_snapshot),
                    is_continue=False,
                    base_result=None,
                    target_seconds=session_seconds or None,
                    is_leg_run=True,
                    leg_progress=leg_progress,
                )
                st.rerun()
        else:
            max_session_seconds_note = max_session_seconds if max_session_seconds > session_seconds else None
            risky = est_worst["peak_mb"] > 0.35 * available_mb
            st.caption(
                f"Estimated for this config: ~{est['order_events']:,.0f} order events, "
                f"~{est['peak_mb']:,.0f} MB peak, ~{est['wall_seconds']:,.0f}s wall time"
                + (f" (up to ~{est_worst['peak_mb']:,.0f} MB / ~{est_worst['wall_seconds']:,.0f}s if `min_trades` extends to the {max_session_seconds_note:,.0f}s cap)" if max_session_seconds_note else "")
                + ". Rough approximation, not a guarantee -- see README's Performance section."
            )
            confirmed_large = True
            if risky:
                st.warning(
                    f"⚠️ This could use ~{est_worst['peak_mb']/1000:,.1f} GB at peak on a machine with "
                    f"~{available_mb/1000:,.1f} GB total RAM -- likely to make the whole system (not just "
                    f"this tab) unresponsive, or get killed outright, the same way a 5,000-agent/300,000s "
                    f"run did before. Consider a shorter session, fewer agents, or exporting via "
                    f"`market_sim`/`scripts/benchmark.py` directly instead of through the browser."
                )
                confirmed_large = st.checkbox("I understand the risk -- run it anyway", key="confirm_large_run")
            if st.button("Run simulation", type="primary", width="stretch", key="run_sim_btn", disabled=risky and not confirmed_large):
                config_toml = cfgmod.dump_config_str(st.session_state.cfg)
                target_seconds = session_seconds or None
                _start_sim_thread(
                    lambda token, progress: (lambda: run_simulation_cancellable(config_toml, token, progress)),
                    config_toml,
                    is_continue=False,
                    base_result=None,
                    target_seconds=target_seconds,
                )
                st.rerun()

    @st.fragment(run_every=1)
    def _poll_sim_status():
        thread = st.session_state.sim_thread
        if thread is None:
            return
        if thread.is_alive():
            elapsed = time.time() - st.session_state.sim_start_time
            stopping = st.session_state.sim_token.is_cancelled()
            current_sim_time = st.session_state.sim_progress.current_seconds() if st.session_state.sim_progress else None
            target = st.session_state.sim_target_seconds
            if current_sim_time is None:
                sim_progress_note = ""
            elif target:
                # The engine's own event queue is what actually decides how far a run
                # goes (e.g. min_trades-driven extension can push it past the nominal
                # session length), so this fraction can occasionally read >100% --
                # clamped for the progress bar, shown uncapped in the text either way.
                sim_progress_note = f" -- simulated {current_sim_time:,.0f}s / {target:,.0f}s ({100 * current_sim_time / target:,.0f}%)"
            else:
                sim_progress_note = f" -- simulated {current_sim_time:,.0f}s so far"
            if st.session_state.sim_is_leg_run and st.session_state.sim_leg_progress:
                lp = st.session_state.sim_leg_progress
                sim_progress_note += f" -- leg {lp['leg_index']}/{lp['n_legs']}"
            st.info(f"⏳ {'Stopping...' if stopping else 'Running...'} ({elapsed:.1f}s elapsed{sim_progress_note})")
            if current_sim_time is not None and target:
                st.progress(min(1.0, max(0.0, current_sim_time / target)))
            if not stopping:
                if st.button("⏹ Stop simulation", width="stretch", key="stop_sim_btn"):
                    st.session_state.sim_token.cancel()
                    st.rerun(scope="fragment")
            return

        # Thread just finished (this is the first poll to notice) -- pull its result
        # out and clear the running state; a full-app rerun below lets every other
        # tab (esp. Results) pick up the newly active result.
        holder = st.session_state.sim_holder
        config_toml = st.session_state.sim_config_toml
        is_continue = st.session_state.sim_is_continue
        base_result = st.session_state.sim_base_result
        is_leg_run = st.session_state.sim_is_leg_run
        st.session_state.sim_thread = None
        st.session_state.sim_token = None
        st.session_state.sim_progress = None
        st.session_state.sim_target_seconds = None
        st.session_state.sim_is_continue = False
        st.session_state.sim_base_result = None
        st.session_state.sim_is_leg_run = False
        st.session_state.sim_leg_progress = None
        if holder.get("error"):
            st.session_state.run_error = holder["error"]
        elif is_leg_run:
            legs = holder["result"]  # list[LegResult] -- see run_scenario_in_legs
            if legs:
                # Load the last leg into the normal single-run view -- its
                # save_path is a `leg_NNNN` dir, which _leg_group_for_path picks up
                # to switch the Results tab's chart into whole-run overview mode
                # automatically (see _render_price_volume below), full detail on
                # this specific leg still one click away via "Load for full detail".
                loaded = load_run(legs[-1].out_dir)
                result = _loaded_run_to_result(loaded)
                was_cancelled = legs[-1].meta.get("cancelled", False)
                source = f"Auto-split run{' (stopped early)' if was_cancelled else ''} -- showing leg {legs[-1].index}/{len(legs)}"
                _activate_result(result, loaded["meta"].get("config_toml", ""), source, save_path=legs[-1].out_dir)
            else:
                st.session_state.run_error = "Auto-split run produced no legs (cancelled before the first one finished)."
        else:
            result = holder["result"]
            # continue_simulation_cancellable returns only the newly-simulated
            # segment (see sim_core::sim::Simulation::resume's doc comment); combine
            # it with what came before so the Results tab shows one continuous
            # history, not just the latest extension.
            if is_continue and base_result is not None:
                result = combine_results(base_result, result)
            if result["meta"]["cancelled"]:
                # "add function to break current simulation (which is going to be
                # saved)": a stopped run is auto-exported immediately, not just left
                # in memory, since the whole point of stopping is usually that you
                # don't trust it'll finish in reasonable time and still want what it
                # produced.
                ts = time.strftime("%Y%m%d_%H%M%S")
                save_dir = os.path.join(RUNS_ROOT, "dashboard_stopped", f"run_{ts}")
                export_run(result, config_toml, save_dir, scenario_name="dashboard (stopped)", extra_meta={"stopped_via_dashboard": True})
                _activate_result(result, config_toml, "Stopped early (auto-saved below)", save_path=save_dir)
            else:
                source = "Continued run" if is_continue else "Fresh run"
                _activate_result(result, config_toml, source, save_path=None)
        st.rerun()

    _poll_sim_status()

    if st.session_state.run_error:
        st.error(st.session_state.run_error)
    if st.session_state.active_result is not None and not sim_running:
        meta = st.session_state.active_result["meta"]
        cancelled_note = " (stopped early)" if meta.get("cancelled") else ""
        st.success(f"{st.session_state.active_source}: {meta['n_trades']:,} trades{cancelled_note}")

    with st.expander("📂 Load a saved run"):
        saved = _discover_saved_runs()
        if not saved:
            st.caption(f"No saved runs found under {os.path.relpath(RUNS_ROOT, REPO_ROOT)}/ yet.")
        else:
            groups = _group_saved_runs(saved)
            # A `run_scenario_in_legs` sequence groups under one entry here (e.g.
            # "user_5000agents_300000s", not 200 separate "leg_0001".."leg_0200"
            # rows each showing an indistinguishable, truncated path) -- a leg
            # picker appears only for the selected group when it actually has more
            # than one leg.
            group_key = st.selectbox("Saved run", list(groups), format_func=lambda p: os.path.relpath(p, RUNS_ROOT), key="load_run_group")
            legs = groups[group_key]
            if len(legs) > 1:
                leg_num = st.number_input(f"Leg (1-{len(legs)})", min_value=1, max_value=len(legs), value=len(legs), step=1, key="load_run_leg_num")
                pick = legs[leg_num - 1]
            else:
                pick = legs[0]
            # A selectbox/number_input's own rendered label can still get clipped by
            # a narrow sidebar -- st.caption wraps instead of truncating, so the
            # full path is always readable regardless of width.
            st.caption(f"Loading: `{os.path.relpath(pick, RUNS_ROOT)}`")
            if st.button("Load", key="load_run_btn"):
                loaded = load_run(pick)
                result = _loaded_run_to_result(loaded)
                _activate_result(result, loaded["meta"].get("config_toml", ""), f"Loaded from {os.path.relpath(pick, REPO_ROOT)}", save_path=pick)
                st.rerun()

tab_setup, tab_pop, tab_advanced, tab_results = st.tabs(["Run & Market Setup", "Bot Population", "Advanced (raw TOML)", "Results"])

# ---------------------------------------------------------------------------------
# Tab: Run & Market Setup
# ---------------------------------------------------------------------------------
with tab_setup:
    st.subheader("Session")
    c1, c2, c3 = st.columns(3)
    cfg["run"]["seed"] = c1.number_input("Seed", value=int(cfg["run"].get("seed", 1)), step=1, key="run_seed")
    cfg["run"]["session_seconds"] = c2.number_input("Session length (simulated seconds)", value=float(cfg["run"].get("session_seconds", 1800.0)), min_value=1.0, step=60.0, key="run_session_seconds")
    cfg["run"]["min_trades"] = c3.number_input("Minimum trades (extends session if unmet)", value=int(cfg["run"].get("min_trades", 0) or 0), min_value=0, step=1000, key="run_min_trades")
    c4, c5 = st.columns(2)
    session_seconds = cfg["run"]["session_seconds"]
    stored_max = cfg["run"].get("max_session_seconds") or session_seconds * 4
    # min_value tracks the session-length field above, which can change every rerun.
    # A widget with an explicit key ignores `value=` once its session_state entry
    # exists (Streamlit's documented behavior), so the entry itself -- not just the
    # value= default -- must be kept >= min_value, or raising session length past a
    # previously-set max throws StreamlitValueBelowMinError. Streamlit also warns if
    # `value=` and a pre-set session_state entry are given together, so pass `value=`
    # only on the widget's first-ever creation and rely on the direct session_state
    # clamp for every rerun after that.
    key = "run_max_session_seconds"
    if key not in st.session_state:
        st.session_state[key] = max(float(stored_max), session_seconds)
    elif st.session_state[key] < session_seconds:
        st.session_state[key] = max(float(stored_max), session_seconds)
    cfg["run"]["max_session_seconds"] = c4.number_input("Max session length (extension cap)", min_value=session_seconds, step=60.0, key=key)
    cfg["run"]["extension_step_seconds"] = c5.number_input("Extension step", value=float(cfg["run"].get("extension_step_seconds", 1800.0)), min_value=1.0, step=60.0, key="run_extension_step")

    st.subheader("Assets (\"stocks\")")
    st.caption("symbol / tick size / starting price / fundamental-value volatility & mean-reversion speed (the latent \"true value\" process fundamentalist bots and price-efficiency metrics reference)")
    assets_df = pd.DataFrame(cfg["assets"]) if cfg["assets"] else pd.DataFrame(columns=["symbol", "tick_size", "initial_price", "fundamental_vol", "fundamental_theta"])
    for col, default in [("fundamental_vol", 0.02), ("fundamental_theta", 0.1)]:
        if col not in assets_df.columns:
            assets_df[col] = default
    assets_edited = st.data_editor(
        assets_df,
        num_rows="dynamic",
        width="stretch",
        key="assets_editor",
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", required=True),
            "tick_size": st.column_config.NumberColumn("Tick size", min_value=1e-9, format="%.8f"),
            "initial_price": st.column_config.NumberColumn("Initial price", min_value=1e-9),
            "fundamental_vol": st.column_config.NumberColumn("Fundamental vol", min_value=0.0),
            "fundamental_theta": st.column_config.NumberColumn("Fundamental mean-reversion speed", min_value=0.0),
        },
    )
    cfg["assets"] = assets_edited.dropna(subset=["symbol"]).to_dict("records")

    st.subheader("Venues")
    st.caption("2+ venues enables cross-venue arbitrage bots. Leave as one venue (\"primary\") otherwise.")
    venues_df = pd.DataFrame(cfg["venues"]) if cfg["venues"] else pd.DataFrame([{"name": "primary", "inter_venue_latency_secs": 0.05}])
    venues_edited = st.data_editor(venues_df, num_rows="dynamic", width="stretch", key="venues_editor")
    cfg["venues"] = venues_edited.dropna(subset=["name"]).to_dict("records")

    st.subheader("Fees, latency, mempool")
    c1, c2, c3 = st.columns(3)
    cfg["fees"]["default_maker_fee"] = c1.number_input("Default maker fee", value=float(cfg["fees"]["default_maker_fee"]), format="%.5f", step=0.0001, key="fees_maker")
    cfg["fees"]["default_taker_fee"] = c2.number_input("Default taker fee", value=float(cfg["fees"]["default_taker_fee"]), format="%.5f", step=0.0001, key="fees_taker")
    cfg["latency"]["default_latency_secs"] = c3.number_input("Default agent latency (s)", value=float(cfg["latency"]["default_latency_secs"]), min_value=0.0, step=0.01, key="latency_default")
    c1, c2, c3 = st.columns(3)
    cfg["mempool"]["enabled"] = c1.checkbox("Mempool model enabled (needed for MEV bots)", value=bool(cfg["mempool"]["enabled"]), key="mempool_enabled")
    cfg["mempool"]["reveal_delay_secs"] = c2.number_input("Mempool reveal delay (s)", value=float(cfg["mempool"]["reveal_delay_secs"]), min_value=0.0, step=0.05, key="mempool_delay")
    cfg["mempool"]["large_order_notional_threshold"] = c3.number_input("Mempool notional threshold ($)", value=float(cfg["mempool"]["large_order_notional_threshold"]), min_value=0.0, step=1000.0, key="mempool_threshold")
    st.caption("Per-bot-type fee/latency overrides ([fees.by_bot_type] / [latency.by_bot_type]) are available in the Advanced tab.")

    st.subheader("Leveraged liquidation model")
    c1, c2, c3, c4 = st.columns(4)
    cfg["liquidation"]["maintenance_margin_buffer"] = c1.number_input("Maintenance margin buffer", value=float(cfg["liquidation"]["maintenance_margin_buffer"]), min_value=0.01, max_value=1.0, step=0.05, key="liq_mm_buffer")
    cfg["liquidation"]["sweep_interval_secs"] = c2.number_input("Liquidation sweep interval (s)", value=float(cfg["liquidation"]["sweep_interval_secs"]), min_value=0.01, step=0.5, key="liq_sweep_interval")
    cfg["liquidation"]["staggered_slices"] = c3.number_input("Staggered-TWAP slices", value=int(cfg["liquidation"]["staggered_slices"]), min_value=1, step=1, key="liq_slices")
    cfg["liquidation"]["full_liquidation_fraction"] = c4.number_input("Fraction using immediate full liquidation", value=float(cfg["liquidation"]["full_liquidation_fraction"]), min_value=0.0, max_value=1.0, step=0.1, key="liq_full_frac")

    st.subheader("Market microstructure / export")
    c1, c2, c3, c4 = st.columns(4)
    cfg["market"]["snapshot_interval_secs"] = c1.number_input("Orderbook snapshot interval (s)", value=float(cfg["market"]["snapshot_interval_secs"]), min_value=0.01, step=0.5, key="market_snapshot_interval")
    cfg["market"]["snapshot_depth_levels"] = c2.number_input("Snapshot depth (levels)", value=int(cfg["market"]["snapshot_depth_levels"]), min_value=1, step=1, key="market_snapshot_depth")
    cfg["market"]["mark_price_ema_halflife_secs"] = c3.number_input("Mark-price EMA half-life (s)", value=float(cfg["market"]["mark_price_ema_halflife_secs"]), min_value=0.1, step=1.0, key="market_mark_halflife")
    cfg["export"]["write_snapshots"] = c4.checkbox("Export orderbook snapshots", value=bool(cfg["export"]["write_snapshots"]), key="export_write_snapshots")

# ---------------------------------------------------------------------------------
# Tab: Bot Population
# ---------------------------------------------------------------------------------
with tab_pop:
    st.subheader("Bot population groups")
    st.caption(
        "One row per (bot_type, variant) group. `count` is how many agents of that exact "
        "variant to create. Cash range and leverage are optional (blank/0 = use engine "
        "defaults). `params_json` holds that variant's own parameters (wake interval, "
        "spread, thresholds, ...) as JSON -- see sim_core/src/bots/<type>.rs's "
        "`from_params` for each variant's keys and defaults."
    )

    groups = cfg.get("population", {}).get("groups", [])
    rows = []
    for g in groups:
        cash_range = g.get("initial_cash_range")
        lev_range = g.get("leverage_range")
        rows.append(
            {
                "bot_type": g.get("bot_type", "noise"),
                "variant": g.get("variant", ""),
                "count": int(g.get("count", 0)),
                "cash_lo": cash_range[0] if cash_range else None,
                "cash_hi": cash_range[1] if cash_range else None,
                "leveraged_fraction": float(g.get("leveraged_fraction", 0.0)),
                "leverage_lo": lev_range[0] if lev_range else None,
                "leverage_hi": lev_range[1] if lev_range else None,
                "params_json": json.dumps(g.get("params", {}) or {}),
                "notes": g.get("notes", "") or "",
            }
        )
    groups_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["bot_type", "variant", "count", "cash_lo", "cash_hi", "leveraged_fraction", "leverage_lo", "leverage_hi", "params_json", "notes"]
    )

    groups_edited = st.data_editor(
        groups_df,
        num_rows="dynamic",
        width="stretch",
        height=420,
        key="groups_editor",
        column_config={
            "bot_type": st.column_config.SelectboxColumn("Bot type", options=BOT_TYPES, required=True),
            "variant": st.column_config.TextColumn("Variant", required=True, help="See caption above for valid variants per bot_type"),
            "count": st.column_config.NumberColumn("Count", min_value=0, step=1, required=True),
            "cash_lo": st.column_config.NumberColumn("Cash min ($)", min_value=0.0),
            "cash_hi": st.column_config.NumberColumn("Cash max ($)", min_value=0.0),
            "leveraged_fraction": st.column_config.NumberColumn("Leveraged fraction", min_value=0.0, max_value=1.0, step=0.05),
            "leverage_lo": st.column_config.NumberColumn("Leverage min"),
            "leverage_hi": st.column_config.NumberColumn("Leverage max"),
            "params_json": st.column_config.TextColumn("params (JSON)", width="large"),
            "notes": st.column_config.TextColumn("Notes"),
        },
    )

    with st.expander("Valid variants per bot type"):
        st.json(VARIANTS_BY_TYPE)

    scale_col1, scale_col2 = st.columns([1, 3])
    scale_factor = scale_col1.number_input("Scale all counts by", value=1.0, min_value=0.0, step=0.1, key="pop_scale_factor")
    if scale_col2.button("Apply scale factor to every group's count", key="pop_scale_apply"):
        groups_edited["count"] = (groups_edited["count"] * scale_factor).round().astype(int)

    new_groups = []
    parse_errors = []
    for _, row in groups_edited.iterrows():
        if not row.get("bot_type") or not row.get("variant") or pd.isna(row.get("count")):
            continue
        g = {"bot_type": row["bot_type"], "variant": row["variant"], "count": int(row["count"])}
        if pd.notna(row.get("cash_lo")) and pd.notna(row.get("cash_hi")):
            g["initial_cash_range"] = [float(row["cash_lo"]), float(row["cash_hi"])]
        if pd.notna(row.get("leveraged_fraction")) and float(row["leveraged_fraction"]) > 0:
            g["leveraged_fraction"] = float(row["leveraged_fraction"])
        if pd.notna(row.get("leverage_lo")) and pd.notna(row.get("leverage_hi")):
            g["leverage_range"] = [float(row["leverage_lo"]), float(row["leverage_hi"])]
        params_text = row.get("params_json") or "{}"
        try:
            g["params"] = json.loads(params_text) if params_text.strip() else {}
        except json.JSONDecodeError as e:
            parse_errors.append(f"Row '{row['bot_type']}/{row['variant']}': invalid params JSON ({e})")
            g["params"] = {}
        if row.get("notes"):
            g["notes"] = row["notes"]
        new_groups.append(g)

    for err in parse_errors:
        st.error(err)
    cfg.setdefault("population", {})["groups"] = new_groups
    st.metric("Total agents in this population", sum(g["count"] for g in new_groups))

# ---------------------------------------------------------------------------------
# Tab: Advanced (raw TOML) -- full-coverage escape hatch
# ---------------------------------------------------------------------------------
with tab_advanced:
    st.caption(
        "Full control: every field of sim_core::config::ScenarioConfig, including "
        "per-bot-type fee/latency overrides ([fees.by_bot_type], [latency.by_bot_type]) "
        "that the structured tabs don't expose their own widgets for. Edits here take "
        "effect once you click Apply; edits in the other tabs are reflected here "
        "immediately (this text always shows the *current* config)."
    )
    raw_text = st.text_area("Scenario TOML", value=cfgmod.dump_config_str(cfg), height=500, key="raw_toml")
    col1, col2 = st.columns([1, 5])
    if col1.button("Apply raw TOML", key="apply_raw_toml"):
        try:
            st.session_state.cfg = toml.loads(raw_text)
            st.success("Applied.")
            st.rerun()
        except toml.TomlDecodeError as e:
            st.error(f"TOML parse error: {e}")
    col2.download_button("Download this config as .toml", data=raw_text, file_name="scenario.toml", mime="text/plain", key="download_toml")

# ---------------------------------------------------------------------------------
# Tab: Results
# ---------------------------------------------------------------------------------
@st.fragment
def _render_results_tab():
    """The whole tab lives in one fragment: without this, Streamlit's default
    behavior reruns and re-sends *this entire tab's* charts (which can mean
    hundreds of thousands of points across 5 figures for a long session) on *any*
    widget interaction *anywhere in the app*, including totally unrelated ones in
    other tabs. Wrapped like this, it only re-executes when a widget inside it
    changes, or on an explicit full-app rerun (a new run finishing, a load, etc.)."""
    result = st.session_state.active_result
    if result is None:
        st.info("Configure a scenario in the other tabs, then click **Run simulation** in the sidebar -- or load a previously saved run from the sidebar expander.")
        return

    version = st.session_state.result_version
    meta = result["meta"]
    st.caption(f"Source: {st.session_state.active_source}")
    if meta.get("cancelled"):
        st.warning(f"⚠️ This run was stopped early -- only {meta['actual_session_seconds']:,.0f}s of simulated time was reached before stopping.")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Trades", f"{meta['n_trades']:,}")
    m2.metric("Order events", f"{meta['n_order_events']:,}")
    m3.metric("Liquidations", f"{meta['liquidation_events']:,}")
    m4.metric("Bankruptcies", f"{meta['bankruptcies']:,}")
    m5.metric("Simulated session", f"{meta['actual_session_seconds']:,.0f}s")

    st.subheader("Save this run")
    s1, s2 = st.columns([3, 1])
    default_name = f"dashboard_run_{time.strftime('%Y%m%d_%H%M%S')}"
    save_name = s1.text_input("Save as (under runs/)", value=st.session_state.last_save_path or os.path.join("runs", default_name), key="save_run_path")
    if s2.button("💾 Save", width="stretch", key="save_run_btn"):
        out_dir = save_name if os.path.isabs(save_name) else os.path.join(REPO_ROOT, save_name)
        export_run(result, st.session_state.active_config_toml or "", out_dir, scenario_name="dashboard")
        st.session_state.last_save_path = out_dir
        st.success(f"Saved to {os.path.relpath(out_dir, REPO_ROOT)}/ -- reload it later from the sidebar's \"Load a saved run\".")
    if st.session_state.last_save_path:
        st.caption(f"Last saved to: {os.path.relpath(st.session_state.last_save_path, REPO_ROOT)}/")

    st.subheader("Continue this simulation")
    checkpoint = result.get("checkpoint")
    if checkpoint is None:
        st.caption(
            "No checkpoint available for this run (it was saved before checkpoint/resume existed, or the "
            "source didn't capture one), so it can't be continued -- re-run it to get one."
        )
    else:
        st.caption(
            "Loads the full internal engine state this run ended with -- every order book, every agent's "
            "cash/inventory/indicators, both RNG streams, and everything still pending -- and keeps "
            "simulating from exactly there. Not a restart: this is the same as if the session had simply "
            "been configured to run longer in the first place."
        )
        c1, c2 = st.columns([3, 1])
        extra_seconds = c1.number_input("Additional simulated seconds", value=1800.0, min_value=1.0, step=60.0, key="continue_extra_seconds")
        sim_running_now = st.session_state.sim_thread is not None
        est = _estimate_run_cost(meta["n_agents"], extra_seconds)
        available_mb = _available_ram_mb()
        # Combined with whatever this run's history already uses -- combine_results
        # (used to show one continuous view) keeps the old segment's tables around
        # too, so the risk compounds across repeated continuations, not just this leg.
        already_mb = meta["n_order_events"] * _EST_MB_PER_ORDER_EVENT
        risky_continue = (est["peak_mb"] + already_mb) > 0.35 * available_mb
        st.caption(f"Estimated for this extension: ~{est['order_events']:,.0f} more order events, ~{est['peak_mb']:,.0f} MB, ~{est['wall_seconds']:,.0f}s wall time (rough approximation).")
        confirmed_continue = True
        if risky_continue:
            st.warning(f"⚠️ This extension alone could use ~{est['peak_mb']/1000:,.1f} GB on top of what this run already holds, on a machine with ~{available_mb/1000:,.1f} GB total RAM.")
            confirmed_continue = st.checkbox("I understand the risk -- continue anyway", key="confirm_large_continue")
        if c2.button("▶ Continue", width="stretch", key="continue_sim_btn", disabled=sim_running_now or (risky_continue and not confirmed_continue)):
            base_config_toml = st.session_state.active_config_toml or ""
            _start_sim_thread(
                lambda token, progress: (lambda: continue_simulation_cancellable(checkpoint, extra_seconds, token, progress)),
                base_config_toml,
                is_continue=True,
                base_result=result,
                target_seconds=meta["actual_session_seconds"] + extra_seconds,
            )
            st.rerun()
        elif sim_running_now:
            st.caption("A simulation is already running (see the sidebar) -- wait for it to finish before continuing this one.")

    trades = _trades_df_cached(version)
    snapshots = _snapshots_df_cached(version)
    agent_state = _agent_state_df_cached(version)
    asset_names = {i: a.get("symbol", f"asset {i}") for i, a in enumerate(cfg.get("assets", []))}

    @st.fragment
    def _render_price_volume():
        """Its own fragment: rebuilding every asset's Plotly figures is the single
        most expensive thing in this tab, so it should only happen when a chart
        control changes -- not on every keystroke in the unrelated Trade search box
        below, which (before this split) forced the *whole* Results tab, charts
        included, to re-render on every single character typed.

        Two data sources feed the same `_render_candlestick_chart`, chosen by
        whether the currently loaded run is one leg of a `run_scenario_in_legs`
        sequence (`_leg_group_for_path`): a plain run charts directly from its own
        (already fully in-memory) trades, same as always; a multi-leg run instead
        composes the *whole* run from small cached per-leg OHLC summaries (see
        `_leg_candle_summary`) -- so browsing a 200-leg run's price history never
        needs more than one leg's full trade data in memory at once, the same
        principle `run_scenario_in_legs` itself exists for. Both paths share the
        same adaptive zoom/candlestick-or-line rendering, so there's nothing
        chart-mechanic-specific to switch between -- only where the candles come
        from changes."""
        st.subheader("Price & volume (candlestick or line -- drag to zoom in, Reset zoom to back out)")

        leg_group = _leg_group_for_path(st.session_state.last_save_path)
        leg_dirs = None
        if leg_group and not st.session_state.leg_drilldown:
            candidate = _group_saved_runs(_discover_saved_runs()).get(leg_group, [])
            if len(candidate) > 1:
                leg_dirs = tuple(candidate)

        if leg_dirs:
            group_label = os.path.relpath(leg_group, RUNS_ROOT)
            st.caption(
                f"This run has {len(leg_dirs)} legs under `{group_label}` -- showing the *whole* run, composed "
                f"from small cached per-leg summaries ({_LEG_SUMMARY_CANDLES} candles/leg each), not every leg's "
                "full trade data at once. Zoom into a single leg's span for a \"Load for full detail\" button "
                "(single-trade resolution, trade search, and that leg's own market-health metrics)."
            )
            with st.spinner(f"Summarizing {len(leg_dirs)} legs (first time only, ~0.1-0.2s/leg)..."):
                _warm_leg_summaries(leg_dirs)

            for aid in sorted(asset_names):
                candles_all = _multi_leg_candles_for_asset(leg_dirs, aid)
                if candles_all.empty:
                    continue
                full_start, full_end = float(candles_all["bucket_start"].min()), float(candles_all["bucket_end"].max())

                def get_candles(view_start, view_end, target_candles, candles_all=candles_all):
                    return resample_candles(candles_all, view_start, view_end, target_candles)

                chart_key = f"legoverview_{re.sub(r'[^0-9A-Za-z_]', '_', leg_group)}_{aid}"
                _render_candlestick_chart(
                    key_prefix=chart_key,
                    title=asset_names.get(aid, f"asset {aid}"),
                    full_start=full_start,
                    full_end=full_end,
                    get_candles=get_candles,
                )

                # Once the view has narrowed to within a single leg's own span for
                # this asset, offer to load that leg's full trade data -- genuine
                # single-trade-resolution drill-down, trade search, agent state,
                # and that leg's own market-health metrics (see below).
                view_start = st.session_state.get(f"{chart_key}_view_start", full_start)
                view_end = st.session_state.get(f"{chart_key}_view_end", full_end)
                containing_legs = []
                for d in leg_dirs:
                    leg_summary = _leg_candle_summary(d).get(aid)
                    if leg_summary is not None and not leg_summary.empty and leg_summary["bucket_start"].min() <= view_end and leg_summary["bucket_end"].max() >= view_start:
                        containing_legs.append(d)
                if len(containing_legs) == 1:
                    leg_dir = containing_legs[0]
                    if st.button(f"📂 Load {os.path.relpath(leg_dir, RUNS_ROOT)} for full detail", key=f"load_leg_btn_{aid}"):
                        loaded = load_run(leg_dir)
                        result2 = _loaded_run_to_result(loaded)
                        _activate_result(result2, loaded["meta"].get("config_toml", ""), f"Loaded from {os.path.relpath(leg_dir, REPO_ROOT)}", save_path=leg_dir)
                        st.session_state.leg_drilldown = True  # _activate_result just reset this to False
                        st.rerun()
                elif len(containing_legs) > 1:
                    st.caption(f"Current {asset_names.get(aid, aid)} view spans {len(containing_legs)} legs -- zoom in further to load one for full detail.")
            return

        if leg_group:
            st.caption(f"Showing full single-trade-resolution detail for one leg of `{os.path.relpath(leg_group, RUNS_ROOT)}` (`{os.path.relpath(st.session_state.last_save_path, RUNS_ROOT)}`).")
            if st.button("🔭 Back to whole-run overview", key="back_to_overview_btn"):
                st.session_state.leg_drilldown = False
                st.rerun()

        if trades.empty:
            st.warning("No trades occurred in this run.")
            return
        show_fundamental = st.checkbox(
            "Show fundamental value",
            value=False,
            key="show_fundamental",
            help="Off by default: the fundamental process is often far more volatile than the traded price, and including it in the chart drags the y-axis scale out so the actual price line looks flat.",
        )
        show_book = st.checkbox("Show best bid/ask", value=True, key="show_book")

        for aid in sorted(trades["asset_id"].unique()):
            sub = trades[trades["asset_id"] == aid].sort_values("time")
            t_arr = sub["time"].to_numpy()
            price_arr = sub["price"].to_numpy()
            notional_arr = (sub["price"] * sub["qty"]).to_numpy()
            asset_snap = snapshots[snapshots["asset_id"] == aid] if not snapshots.empty else snapshots

            def get_candles(view_start, view_end, target_candles, t_arr=t_arr, price_arr=price_arr, notional_arr=notional_arr):
                return candles_from_trades(t_arr, price_arr, notional_arr, view_start, view_end, target_candles)

            def get_overlays(view_start, view_end, asset_snap=asset_snap):
                overlays = []
                if asset_snap.empty:
                    return overlays
                snap0 = asset_snap[asset_snap["level"] == 0]
                in_view = (snap0["time"] >= view_start) & (snap0["time"] <= view_end)
                snap0 = snap0[in_view]
                if show_book:
                    for side, color in [("bid", "rgba(0,150,0,0.7)"), ("ask", "rgba(200,0,0,0.7)")]:
                        s = snap0[snap0["side"] == side].sort_values("time")
                        if not s.empty:
                            t_b, y_b = _downsample_minmax(s["time"], s["price"], 2000)
                            overlays.append((f"best {side}", t_b, y_b, "solid", color))
                if show_fundamental:
                    fair = snap0[snap0["side"] == "fundamental"].sort_values("time")
                    if not fair.empty:
                        t_f, y_f = _downsample_minmax(fair["time"], fair["price"], 2000)
                        overlays.append(("fundamental value", t_f, y_f, "dot", "grey"))
                return overlays

            _render_candlestick_chart(
                key_prefix=f"chart_{aid}_{version}",
                title=asset_names.get(aid, f"asset {aid}"),
                full_start=float(t_arr.min()),
                full_end=float(t_arr.max()),
                get_candles=get_candles,
                get_overlays=get_overlays,
            )

    _render_price_volume()

    @st.fragment
    def _render_market_health():
        """Its own fragment for the same reason as _render_price_volume/
        _render_trade_search: the "compute all legs" action below can genuinely
        run for minutes (see _leg_market_health's docstring) and shouldn't force
        the price chart or trade search to rebuild alongside it."""
        st.subheader("Market-health metrics")
        leg_group_active = _leg_group_for_path(st.session_state.last_save_path)
        if leg_group_active:
            leg_dirs_active = tuple(_group_saved_runs(_discover_saved_runs()).get(leg_group_active, []))
            # Not just session_state.leg_metrics_by_dir (empty at the start of a
            # fresh dashboard session) -- a run made after run_scenario_in_legs's
            # compute_market_health default already has every leg's metrics
            # persisted on disk, so check for those too (cheap: an os.path.exists
            # plus a small JSON read per leg, only for legs not already known this
            # session) rather than showing a misleading "N legs missing, ~X min"
            # estimate for legs that are actually already instant to load.
            known = {}
            for d in leg_dirs_active:
                if d in st.session_state.leg_metrics_by_dir:
                    known[d] = st.session_state.leg_metrics_by_dir[d]
                    continue
                persisted = market_health.load_metrics(d)
                if persisted is not None:
                    known[d] = persisted
                    st.session_state.leg_metrics_by_dir[d] = persisted
            n_missing = len(leg_dirs_active) - len(known)
            st.caption(
                f"{len(known)}/{len(leg_dirs_active)} legs have market-health metrics computed, shown below as "
                "one line chart per metric across legs (a genuine list of sessions, not one merged whole-run "
                "number -- a few metrics, like autocorrelation-based efficiency and resilience timing, don't "
                "compose cleanly across a leg boundary, so per-leg is the honest, always-correct unit here)."
            )
            if n_missing > 0:
                # ~13-15s/leg measured serially on this machine's real 200-leg run
                # (~2s load + ~11-13s compute), ~2x parallel via _compute_all_leg_metrics
                # -- kept deliberately low (not e.g. 4) after a real memory scare: see
                # that function's docstring.
                est_minutes = n_missing * 14 / 2 / 60
                st.caption(
                    f"Computing all of them is genuinely heavy (~14s/leg of real work, this machine's own "
                    f"measurement) -- estimated ~{est_minutes:.1f} min for the remaining {n_missing} legs, "
                    "hence a button rather than automatic on every view. Cached per leg afterward."
                )
                if st.button(f"📊 Compute market-health for the remaining {n_missing} legs", key="compute_all_leg_metrics_btn"):
                    st.session_state.leg_metrics_by_dir.update(_compute_all_leg_metrics(leg_dirs_active, known))
                    # A plain full-page rerun, not scope="fragment" -- that scope is
                    # only legal during an actual fragment-triggered rerun, not this
                    # first pass through the fragment (Streamlit raises
                    # StreamlitInvalidLayoutContextError otherwise, caught by testing
                    # against a real click rather than assuming). `known`/`hist_df`
                    # below are computed from session_state at the top of this
                    # fragment call, so a full rerun (which re-executes this fragment
                    # too) is what actually picks up the just-computed legs.
                    st.rerun()
            if known:
                hist_df = pd.DataFrame({os.path.basename(d): known[d] for d in sorted(leg_dirs_active) if d in known}).T
                hist_df.index.name = "leg"
                by_dim: dict[str, list[str]] = {}
                for col in hist_df.columns:
                    by_dim.setdefault(DIMENSION_OF.get(col.split(".", 1)[0], "other"), []).append(col)
                for dim in sorted(by_dim):
                    with st.expander(f"{dim} ({len(by_dim[dim])} metrics across {len(hist_df)} legs)"):
                        grid = st.columns(3)
                        for i, metric in enumerate(sorted(by_dim[dim])):
                            with grid[i % 3]:
                                st.caption(metric)
                                st.line_chart(hist_df[[metric]], height=180)
                with st.expander("Raw table (every computed leg x every metric)"):
                    st.dataframe(hist_df, width="stretch", height=300)
            st.markdown(f"**Currently loaded leg** ({os.path.relpath(st.session_state.last_save_path, RUNS_ROOT)}):")
        if st.session_state.metrics:
            metrics_df = pd.DataFrame(sorted(st.session_state.metrics.items()), columns=["metric", "value"])
            st.dataframe(metrics_df, width="stretch", height=300)

    _render_market_health()

    st.subheader("Human involvement by asset")
    if trades.empty:
        st.caption("No trades to break down.")
    else:
        # Vectorized (column-wise boolean ops + groupby), not a per-row loop -- same
        # reasoning as the trade-search free-text fix: stays fast at hundreds of
        # thousands of trades.
        maker_human = trades["maker_type"] == "human"
        taker_human = trades["taker_type"] == "human"
        breakdown = pd.DataFrame(
            {
                "asset_id": trades["asset_id"],
                "fully_human": maker_human & taker_human,
                "partially_human": maker_human ^ taker_human,
                "no_human": ~maker_human & ~taker_human,
            }
        )
        by_asset = breakdown.groupby("asset_id").sum()
        by_asset["total_trades"] = by_asset[["fully_human", "partially_human", "no_human"]].sum(axis=1)
        by_asset["human_involved_pct"] = (100 * (by_asset["fully_human"] + by_asset["partially_human"]) / by_asset["total_trades"]).round(2)
        by_asset = by_asset.rename(index=asset_names).rename_axis("asset")
        by_asset = by_asset[["total_trades", "fully_human", "partially_human", "no_human", "human_involved_pct"]]
        st.caption("`fully_human`: both sides of the trade were human bots. `partially_human`: exactly one side was. `human_involved_pct`: share of that asset's trades touching a human bot at all.")
        st.dataframe(by_asset, width="stretch")

    @st.fragment
    def _render_trade_search():
        """Its own fragment for the same reason as `_render_price_volume` above:
        `text_input` reruns on every keystroke, and before this split that meant
        retyping a search re-rendered every price/volume chart too, on every single
        character -- the main source of the in-browser lag this whole section of
        changes was about fixing."""
        st.subheader("Trade search")

        leg_group_search = _leg_group_for_path(st.session_state.last_save_path)
        leg_dirs_search = None
        if leg_group_search:
            candidate = _group_saved_runs(_discover_saved_runs()).get(leg_group_search, [])
            if len(candidate) > 1:
                leg_dirs_search = tuple(candidate)
        search_all_legs = False
        if leg_dirs_search:
            search_all_legs = st.checkbox(
                f"Search across all {len(leg_dirs_search)} legs, not just the currently loaded one (reads every "
                f"leg's trades.parquet once -- measured ~0.5s/leg on this machine's real 200-leg run -- keeping "
                "only matching rows, and stops scanning once 5,000 matches are found so an unfiltered search "
                "can't try to hold every leg's full trade table in memory at once)",
                key="search_all_legs",
            )

        if trades.empty and not search_all_legs:
            st.caption("No trades to search.")
            return
        # Deliberately no explicit `key=` on the multiselect/slider filters below:
        # their valid options/range come from *this run's* trades and change
        # between runs, so a stale key (from a previous, differently-ranged run)
        # could hold a selection outside the new valid set -- letting Streamlit
        # derive a fresh key from the (also-changing) options/min/max args avoids
        # that rather than needing to reproduce the same clamp-before-render
        # pattern used for max_session_seconds above. Options come from the
        # currently loaded leg even in whole-run search mode -- every leg shares
        # the same bot population/asset config, so that leg's own values are
        # already representative.
        f1, f2, f3, f4 = st.columns(4)
        asset_filter = f1.multiselect("Asset", options=sorted(trades["asset_id"].unique()) if not trades.empty else sorted(asset_names), format_func=lambda a: asset_names.get(a, str(a)))
        bot_types_present = sorted(set(trades["maker_type"]).union(trades["taker_type"])) if not trades.empty else []
        bot_type_filter = f2.multiselect("Bot type (maker or taker)", options=bot_types_present)
        agent_id_filter = f3.text_input("Agent ID (maker or taker)", key="trade_search_agent_id")
        free_text = f4.text_input("Free-text search (any column)", key="trade_search_free_text")

        f5, f6 = st.columns(2)
        if search_all_legs:
            t_min, t_max = _leg_group_time_range(leg_dirs_search)
        else:
            t_min, t_max = float(trades["time"].min()), float(trades["time"].max())
        time_range = f5.slider("Time range (s)", min_value=t_min, max_value=t_max, value=(t_min, t_max))
        qty_min = f6.number_input("Min quantity", value=0.0, min_value=0.0, key="trade_search_min_qty")

        agent_id_val = None
        if agent_id_filter.strip():
            try:
                agent_id_val = int(agent_id_filter.strip())
            except ValueError:
                st.warning("Agent ID must be an integer.")

        if search_all_legs:
            with st.spinner(f"Scanning up to {len(leg_dirs_search)} legs..."):
                filtered, legs_scanned, n_legs, stopped_early = _search_all_legs(leg_dirs_search, asset_filter, bot_type_filter, agent_id_val, free_text, time_range, qty_min)
            st.caption(f"{len(filtered):,} matches, scanned {legs_scanned}/{n_legs} legs" + (" (stopped early -- 5,000-match cap reached; narrow the filters for a complete scan)" if stopped_early else ""))
        else:
            filtered = trades[(trades["time"] >= time_range[0]) & (trades["time"] <= time_range[1]) & (trades["qty"] >= qty_min)]
            if asset_filter:
                filtered = filtered[filtered["asset_id"].isin(asset_filter)]
            if bot_type_filter:
                filtered = filtered[filtered["maker_type"].isin(bot_type_filter) | filtered["taker_type"].isin(bot_type_filter)]
            if agent_id_val is not None:
                filtered = filtered[(filtered["maker_agent_id"] == agent_id_val) | (filtered["taker_agent_id"] == agent_id_val)]
            if free_text.strip():
                # Vectorized (column-wise string ops), not a per-row .apply(axis=1)
                # -- the latter is a Python-level loop and was the slowest part of
                # the whole tab on a large trade table.
                combined = filtered[_TRADE_SEARCH_COLS[0]].astype(str)
                for col in _TRADE_SEARCH_COLS[1:]:
                    combined = combined + " " + filtered[col].astype(str)
                filtered = filtered[combined.str.lower().str.contains(free_text.lower(), regex=False, na=False)]
            st.caption(f"{len(filtered):,} of {len(trades):,} trades match")

        st.dataframe(filtered.sort_values("time").head(5000), width="stretch", height=400)
        if len(filtered) > 5000:
            st.caption("Showing first 5,000 matching rows; narrow the filters to see more precisely, or download all matches below.")
        st.download_button("Download filtered trades as CSV", data=filtered.to_csv(index=False), file_name="trades_filtered.csv", mime="text/csv", key="download_trades_csv")

    _render_trade_search()

    if not agent_state.empty:
        st.subheader("Agent state (end of session)")
        st.dataframe(agent_state, width="stretch", height=300)


with tab_results:
    _render_results_tab()
