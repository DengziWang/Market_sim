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

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import toml
from plotly.subplots import make_subplots

from market_sim import config as cfgmod
from market_sim import run_scenario_dict
import market_health

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS = {
    "Realistic mix": os.path.join(REPO_ROOT, "scenarios", "realistic_mix.toml"),
    "Pure bot market": os.path.join(REPO_ROOT, "scenarios", "pure_bot_market.toml"),
    "Stress test (wash/MEV)": os.path.join(REPO_ROOT, "scenarios", "stress_wash_mev.toml"),
}
BOT_TYPES = [
    "market_maker", "arbitrage", "momentum", "mean_reversion", "grid", "dca",
    "liquidation_trader", "mev", "wash_trading", "noise",
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
}

st.set_page_config(page_title="Bot Market Sim", layout="wide")


def _default_cfg() -> dict:
    return cfgmod.load_config_dict(SCENARIOS["Realistic mix"])


if "cfg" not in st.session_state:
    st.session_state.cfg = _default_cfg()
if "run" not in st.session_state:
    st.session_state.run = None
    st.session_state.metrics = None
    st.session_state.run_error = None

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

    chosen = st.selectbox("Load a base scenario", list(SCENARIOS))
    if st.button("Load scenario (discards current edits)"):
        st.session_state.cfg = cfgmod.load_config_dict(SCENARIOS[chosen])
        st.rerun()

    st.divider()
    n_agents = sum(int(g.get("count", 0)) for g in cfg.get("population", {}).get("groups", []))
    n_assets = len(cfg.get("assets", []))
    st.metric("Total agents", n_agents)
    st.metric("Assets configured", n_assets)

    st.divider()
    run_clicked = st.button("Run simulation", type="primary", width="stretch")
    if run_clicked:
        st.session_state.run_error = None
        try:
            with st.spinner("Running simulation..."):
                run = run_scenario_dict(st.session_state.cfg)
                st.session_state.run = run
                with st.spinner("Computing market-health metrics..."):
                    st.session_state.metrics = market_health.compute_run_metrics(run.result)
        except Exception as e:  # noqa: BLE001
            st.session_state.run_error = f"{type(e).__name__}: {e}"
            st.session_state.run = None
        st.rerun()

    if st.session_state.run is not None:
        st.success(f"Last run: {st.session_state.run.meta['n_trades']:,} trades")
    if st.session_state.run_error:
        st.error(st.session_state.run_error)

tab_setup, tab_pop, tab_advanced, tab_results = st.tabs(["Run & Market Setup", "Bot Population", "Advanced (raw TOML)", "Results"])

# ---------------------------------------------------------------------------------
# Tab: Run & Market Setup
# ---------------------------------------------------------------------------------
with tab_setup:
    st.subheader("Session")
    c1, c2, c3 = st.columns(3)
    cfg["run"]["seed"] = c1.number_input("Seed", value=int(cfg["run"].get("seed", 1)), step=1)
    cfg["run"]["session_seconds"] = c2.number_input("Session length (simulated seconds)", value=float(cfg["run"].get("session_seconds", 1800.0)), min_value=1.0, step=60.0)
    cfg["run"]["min_trades"] = c3.number_input("Minimum trades (extends session if unmet)", value=int(cfg["run"].get("min_trades", 0) or 0), min_value=0, step=1000)
    c4, c5 = st.columns(2)
    cfg["run"]["max_session_seconds"] = c4.number_input("Max session length (extension cap)", value=float(cfg["run"].get("max_session_seconds", cfg["run"]["session_seconds"] * 4) or cfg["run"]["session_seconds"] * 4), min_value=cfg["run"]["session_seconds"], step=60.0)
    cfg["run"]["extension_step_seconds"] = c5.number_input("Extension step", value=float(cfg["run"].get("extension_step_seconds", 1800.0)), min_value=1.0, step=60.0)

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
    cfg["fees"]["default_maker_fee"] = c1.number_input("Default maker fee", value=float(cfg["fees"]["default_maker_fee"]), format="%.5f", step=0.0001)
    cfg["fees"]["default_taker_fee"] = c2.number_input("Default taker fee", value=float(cfg["fees"]["default_taker_fee"]), format="%.5f", step=0.0001)
    cfg["latency"]["default_latency_secs"] = c3.number_input("Default agent latency (s)", value=float(cfg["latency"]["default_latency_secs"]), min_value=0.0, step=0.01)
    c1, c2, c3 = st.columns(3)
    cfg["mempool"]["enabled"] = c1.checkbox("Mempool model enabled (needed for MEV bots)", value=bool(cfg["mempool"]["enabled"]))
    cfg["mempool"]["reveal_delay_secs"] = c2.number_input("Mempool reveal delay (s)", value=float(cfg["mempool"]["reveal_delay_secs"]), min_value=0.0, step=0.05)
    cfg["mempool"]["large_order_notional_threshold"] = c3.number_input("Mempool notional threshold ($)", value=float(cfg["mempool"]["large_order_notional_threshold"]), min_value=0.0, step=1000.0)
    st.caption("Per-bot-type fee/latency overrides ([fees.by_bot_type] / [latency.by_bot_type]) are available in the Advanced tab.")

    st.subheader("Leveraged liquidation model")
    c1, c2, c3, c4 = st.columns(4)
    cfg["liquidation"]["maintenance_margin_buffer"] = c1.number_input("Maintenance margin buffer", value=float(cfg["liquidation"]["maintenance_margin_buffer"]), min_value=0.01, max_value=1.0, step=0.05)
    cfg["liquidation"]["sweep_interval_secs"] = c2.number_input("Liquidation sweep interval (s)", value=float(cfg["liquidation"]["sweep_interval_secs"]), min_value=0.01, step=0.5)
    cfg["liquidation"]["staggered_slices"] = c3.number_input("Staggered-TWAP slices", value=int(cfg["liquidation"]["staggered_slices"]), min_value=1, step=1)
    cfg["liquidation"]["full_liquidation_fraction"] = c4.number_input("Fraction using immediate full liquidation", value=float(cfg["liquidation"]["full_liquidation_fraction"]), min_value=0.0, max_value=1.0, step=0.1)

    st.subheader("Market microstructure / export")
    c1, c2, c3, c4 = st.columns(4)
    cfg["market"]["snapshot_interval_secs"] = c1.number_input("Orderbook snapshot interval (s)", value=float(cfg["market"]["snapshot_interval_secs"]), min_value=0.01, step=0.5)
    cfg["market"]["snapshot_depth_levels"] = c2.number_input("Snapshot depth (levels)", value=int(cfg["market"]["snapshot_depth_levels"]), min_value=1, step=1)
    cfg["market"]["mark_price_ema_halflife_secs"] = c3.number_input("Mark-price EMA half-life (s)", value=float(cfg["market"]["mark_price_ema_halflife_secs"]), min_value=0.1, step=1.0)
    cfg["export"]["write_snapshots"] = c4.checkbox("Export orderbook snapshots", value=bool(cfg["export"]["write_snapshots"]))

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
    scale_factor = scale_col1.number_input("Scale all counts by", value=1.0, min_value=0.0, step=0.1)
    if scale_col2.button("Apply scale factor to every group's count"):
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
    if col1.button("Apply raw TOML"):
        try:
            st.session_state.cfg = toml.loads(raw_text)
            st.success("Applied.")
            st.rerun()
        except toml.TomlDecodeError as e:
            st.error(f"TOML parse error: {e}")
    col2.download_button("Download this config as .toml", data=raw_text, file_name="scenario.toml", mime="text/plain")

# ---------------------------------------------------------------------------------
# Tab: Results
# ---------------------------------------------------------------------------------
with tab_results:
    run = st.session_state.run
    if run is None:
        st.info("Configure a scenario in the other tabs, then click **Run simulation** in the sidebar.")
    else:
        meta = run.meta
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Trades", f"{meta['n_trades']:,}")
        m2.metric("Order events", f"{meta['n_order_events']:,}")
        m3.metric("Liquidations", f"{meta['liquidation_events']:,}")
        m4.metric("Bankruptcies", f"{meta['bankruptcies']:,}")
        m5.metric("Simulated session", f"{meta['actual_session_seconds']:,.0f}s")

        trades = pd.DataFrame(run.result["trades"])
        snapshots = pd.DataFrame(run.result.get("snapshots", {}))
        agent_state = pd.DataFrame(run.result.get("agent_states", {}))
        asset_names = {i: a.get("symbol", f"asset {i}") for i, a in enumerate(cfg.get("assets", []))}

        st.subheader("Price & volume (zoom: drag to select, double-click to reset, scroll to zoom)")
        if trades.empty:
            st.warning("No trades occurred in this run.")
        else:
            for aid in sorted(trades["asset_id"].unique()):
                sub = trades[trades["asset_id"] == aid].sort_values("time")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
                fig.add_trace(go.Scattergl(x=sub["time"], y=sub["price"], mode="lines", name="trade price", line=dict(width=1)), row=1, col=1)

                if not snapshots.empty:
                    snap = snapshots[(snapshots["asset_id"] == aid) & (snapshots["level"] == 0)]
                    for side, color in [("bid", "rgba(0,150,0,0.6)"), ("ask", "rgba(200,0,0,0.6)")]:
                        s = snap[snap["side"] == side].sort_values("time")
                        if not s.empty:
                            fig.add_trace(go.Scattergl(x=s["time"], y=s["price"], mode="lines", name=f"best {side}", line=dict(width=0.6, color=color)), row=1, col=1)
                    fair = snap[snap["side"] == "fundamental"].sort_values("time")
                    if not fair.empty:
                        fig.add_trace(go.Scattergl(x=fair["time"], y=fair["price"], mode="lines", name="fundamental value", line=dict(width=1, dash="dot", color="grey")), row=1, col=1)

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

        st.subheader("Market-health metrics")
        if st.session_state.metrics:
            metrics_df = pd.DataFrame(sorted(st.session_state.metrics.items()), columns=["metric", "value"])
            st.dataframe(metrics_df, width="stretch", height=300)

        st.subheader("Trade search")
        if trades.empty:
            st.caption("No trades to search.")
        else:
            f1, f2, f3, f4 = st.columns(4)
            asset_filter = f1.multiselect("Asset", options=sorted(trades["asset_id"].unique()), format_func=lambda a: asset_names.get(a, str(a)))
            bot_types_present = sorted(set(trades["maker_type"]).union(trades["taker_type"]))
            bot_type_filter = f2.multiselect("Bot type (maker or taker)", options=bot_types_present)
            agent_id_filter = f3.text_input("Agent ID (maker or taker)")
            free_text = f4.text_input("Free-text search (any column)")

            f5, f6 = st.columns(2)
            t_min, t_max = float(trades["time"].min()), float(trades["time"].max())
            time_range = f5.slider("Time range (s)", min_value=t_min, max_value=t_max, value=(t_min, t_max))
            qty_min = f6.number_input("Min quantity", value=0.0, min_value=0.0)

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
                mask = filtered.apply(lambda row: free_text.lower() in " ".join(str(v) for v in row.values).lower(), axis=1)
                filtered = filtered[mask]

            st.caption(f"{len(filtered):,} of {len(trades):,} trades match")
            st.dataframe(filtered.sort_values("time").head(5000), width="stretch", height=400)
            if len(filtered) > 5000:
                st.caption("Showing first 5,000 matching rows; narrow the filters to see more precisely, or download all matches below.")
            st.download_button("Download filtered trades as CSV", data=filtered.to_csv(index=False), file_name="trades_filtered.csv", mime="text/csv")

        if not agent_state.empty:
            st.subheader("Agent state (end of session)")
            st.dataframe(agent_state, width="stretch", height=300)
