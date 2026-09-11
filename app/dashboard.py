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

import glob
import json
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
from market_sim.export import df_to_columnar, export_run, load_run
from market_sim.runner import combine_results
import market_health

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


def _start_sim_thread(make_run_fn, config_toml: str, *, is_continue: bool, base_result: dict | None, target_seconds: float | None) -> None:
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
    can't be known upfront."""
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


def _activate_result(result: dict, config_toml: str, source: str, save_path: str | None = None) -> None:
    st.session_state.active_result = result
    st.session_state.active_config_toml = config_toml
    st.session_state.active_source = source
    st.session_state.metrics = market_health.compute_run_metrics(result)
    st.session_state.last_save_path = save_path
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


if "cfg" not in st.session_state:
    st.session_state.cfg = _default_cfg()
for _key, _default in [
    ("active_result", None), ("active_config_toml", None), ("active_source", None),
    ("metrics", None), ("last_save_path", None), ("run_error", None),
    ("sim_thread", None), ("sim_token", None), ("sim_holder", None),
    ("sim_config_toml", None), ("sim_start_time", None),
    ("sim_is_continue", False), ("sim_base_result", None),
    ("sim_progress", None), ("sim_target_seconds", None),
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
        est = _estimate_run_cost(n_agents, session_seconds)
        est_worst = _estimate_run_cost(n_agents, max_session_seconds)
        available_mb = _available_ram_mb()
        # Threshold: this run's estimated peak memory relative to *total* machine
        # RAM, not just what's free right now -- other things (browser, OS) need
        # headroom too, and this same number is what wrecked the whole desktop last
        # time (see project memory), not just the simulation.
        risky = est_worst["peak_mb"] > 0.35 * available_mb
        st.caption(
            f"Estimated for this config: ~{est['order_events']:,.0f} order events, "
            f"~{est['peak_mb']:,.0f} MB peak, ~{est['wall_seconds']:,.0f}s wall time"
            + (f" (up to ~{est_worst['peak_mb']:,.0f} MB / ~{est_worst['wall_seconds']:,.0f}s if `min_trades` extends to the {max_session_seconds:,.0f}s cap)" if max_session_seconds > session_seconds else "")
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
        st.session_state.sim_thread = None
        st.session_state.sim_token = None
        st.session_state.sim_progress = None
        st.session_state.sim_target_seconds = None
        st.session_state.sim_is_continue = False
        st.session_state.sim_base_result = None
        if holder.get("error"):
            st.session_state.run_error = holder["error"]
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
        """Its own fragment: rebuilding every asset's Plotly figures (each with
        several downsampled traces) is the single most expensive thing in this tab,
        so it should only happen when a chart control changes -- not on every
        keystroke in the unrelated Trade search box below, which (before this split)
        forced the *whole* Results tab, charts included, to re-render on every single
        character typed."""
        st.subheader("Price & volume (zoom: drag to select, double-click to reset, scroll to zoom)")
        if trades.empty:
            st.warning("No trades occurred in this run.")
            return
        c1, c2, c3 = st.columns(3)
        show_fundamental = c1.checkbox(
            "Show fundamental value",
            value=False,
            key="show_fundamental",
            help="Off by default: the fundamental process is often far more volatile than the traded price, and including it in the chart drags the y-axis scale out so the actual price line looks flat.",
        )
        show_book = c2.checkbox("Show best bid/ask", value=True, key="show_book")
        max_points = c3.number_input(
            "Max points per line",
            value=3000,
            min_value=200,
            max_value=50_000,
            step=500,
            key="max_chart_points",
            help="Long sessions can produce hundreds of thousands of trades per asset -- plotting every single one is what makes charts slow to render and pan/zoom. Each line is downsampled to at most this many points, keeping each bucket's min AND max so spikes (e.g. a liquidation cascade) stay visible instead of being smoothed away.",
        )
        total_trades = len(trades)
        if total_trades > max_points:
            st.caption(f"{total_trades:,} trades total -- price/book lines below are downsampled to ~{max_points:,} points each for rendering speed. Volume bars are always pre-aggregated (300 buckets), not affected.")

        for aid in sorted(trades["asset_id"].unique()):
            sub = trades[trades["asset_id"] == aid].sort_values("time")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
            t_ds, y_ds = _downsample_minmax(sub["time"], sub["price"], max_points)
            fig.add_trace(go.Scattergl(x=t_ds, y=y_ds, mode="lines", name="trade price", line=dict(width=1)), row=1, col=1)

            if not snapshots.empty:
                snap = snapshots[(snapshots["asset_id"] == aid) & (snapshots["level"] == 0)]
                if show_book:
                    for side, color in [("bid", "rgba(0,150,0,0.6)"), ("ask", "rgba(200,0,0,0.6)")]:
                        s = snap[snap["side"] == side].sort_values("time")
                        if not s.empty:
                            t_b, y_b = _downsample_minmax(s["time"], s["price"], max_points)
                            fig.add_trace(go.Scattergl(x=t_b, y=y_b, mode="lines", name=f"best {side}", line=dict(width=0.6, color=color)), row=1, col=1)
                if show_fundamental:
                    fair = snap[snap["side"] == "fundamental"].sort_values("time")
                    if not fair.empty:
                        t_f, y_f = _downsample_minmax(fair["time"], fair["price"], max_points)
                        fig.add_trace(go.Scattergl(x=t_f, y=y_f, mode="lines", name="fundamental value", line=dict(width=1, dash="dot", color="grey")), row=1, col=1)

            notional = sub["price"] * sub["qty"]
            n_buckets = 300
            t_min, t_max = sub["time"].min(), sub["time"].max()
            width = max((t_max - t_min) / n_buckets, 1e-9)
            bucket = ((sub["time"] - t_min) / width).astype(int).clip(upper=n_buckets - 1)
            vol_by_bucket = notional.groupby(bucket).sum()
            bucket_t = t_min + (vol_by_bucket.index.values + 0.5) * width
            fig.add_trace(go.Bar(x=bucket_t, y=vol_by_bucket.values, name="volume (notional)", marker_color="grey"), row=2, col=1)

            fig.update_layout(height=420, title=f"{asset_names.get(aid, f'asset {aid}')}", legend=dict(orientation="h", y=1.08), margin=dict(t=60, b=20))
            fig.update_xaxes(title_text="simulated time (s)", row=2, col=1)
            st.plotly_chart(fig, width="stretch", key=f"chart_{aid}")

    _render_price_volume()

    st.subheader("Market-health metrics")
    if st.session_state.metrics:
        metrics_df = pd.DataFrame(sorted(st.session_state.metrics.items()), columns=["metric", "value"])
        st.dataframe(metrics_df, width="stretch", height=300)

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
        if trades.empty:
            st.caption("No trades to search.")
            return
        # Deliberately no explicit `key=` on the multiselect/slider filters below:
        # their valid options/range come from *this run's* trades and change
        # between runs, so a stale key (from a previous, differently-ranged run)
        # could hold a selection outside the new valid set -- letting Streamlit
        # derive a fresh key from the (also-changing) options/min/max args avoids
        # that rather than needing to reproduce the same clamp-before-render
        # pattern used for max_session_seconds above.
        f1, f2, f3, f4 = st.columns(4)
        asset_filter = f1.multiselect("Asset", options=sorted(trades["asset_id"].unique()), format_func=lambda a: asset_names.get(a, str(a)))
        bot_types_present = sorted(set(trades["maker_type"]).union(trades["taker_type"]))
        bot_type_filter = f2.multiselect("Bot type (maker or taker)", options=bot_types_present)
        agent_id_filter = f3.text_input("Agent ID (maker or taker)", key="trade_search_agent_id")
        free_text = f4.text_input("Free-text search (any column)", key="trade_search_free_text")

        f5, f6 = st.columns(2)
        t_min, t_max = float(trades["time"].min()), float(trades["time"].max())
        time_range = f5.slider("Time range (s)", min_value=t_min, max_value=t_max, value=(t_min, t_max))
        qty_min = f6.number_input("Min quantity", value=0.0, min_value=0.0, key="trade_search_min_qty")

        filtered = trades[(trades["time"] >= time_range[0]) & (trades["time"] <= time_range[1]) & (trades["qty"] >= qty_min)]
        if asset_filter:
            filtered = filtered[filtered["asset_id"].isin(asset_filter)]
        if bot_type_filter:
            filtered = filtered[filtered["maker_type"].isin(bot_type_filter) | filtered["taker_type"].isin(bot_type_filter)]
        if agent_id_filter.strip():
            try:
                aid_val = int(agent_id_filter.strip())
                filtered = filtered[(filtered["maker_agent_id"] == aid_val) | (filtered["taker_agent_id"] == aid_val)]
            except ValueError:
                st.warning("Agent ID must be an integer.")
        if free_text.strip():
            # Vectorized (column-wise string ops), not a per-row .apply(axis=1) --
            # the latter is a Python-level loop and was the slowest part of the
            # whole tab on a large trade table.
            search_cols = ["asset_id", "price", "qty", "taker_side", "maker_agent_id", "taker_agent_id", "maker_type", "maker_variant", "taker_type", "taker_variant"]
            combined = filtered[search_cols[0]].astype(str)
            for col in search_cols[1:]:
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
