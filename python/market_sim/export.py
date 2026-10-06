"""Turn a raw `run_simulation(...)` result (columnar dicts from the Rust core) into
the parquet/JSON export files described in spec section 7:

    trades.parquet, orderbook_snapshots.parquet, order_events.parquet,
    agent_state.parquet, run_metadata.json

Kept deliberately dumb (dict -> DataFrame -> parquet): all the actual simulation
logic lives in `sim_core`; this module's only job is the export format.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class ExportPaths:
    out_dir: str
    trades: str
    order_events: str
    orderbook_snapshots: str
    agent_state: str
    run_metadata: str
    checkpoint: str


def export_paths(out_dir: str) -> ExportPaths:
    return ExportPaths(
        out_dir=out_dir,
        trades=os.path.join(out_dir, "trades.parquet"),
        order_events=os.path.join(out_dir, "order_events.parquet"),
        orderbook_snapshots=os.path.join(out_dir, "orderbook_snapshots.parquet"),
        agent_state=os.path.join(out_dir, "agent_state.parquet"),
        run_metadata=os.path.join(out_dir, "run_metadata.json"),
        # Opaque `sim_core::sim::Simulation::to_checkpoint_bytes()` blob -- full
        # internal engine state (order books, every agent, both RNG streams, pending
        # events) at the moment the run stopped. Not present for runs saved before
        # checkpoint/resume existed, or if the caller didn't ask `run_simulation*`
        # for one; see `has_checkpoint`/`continue_run`.
        checkpoint=os.path.join(out_dir, "checkpoint.bin"),
    )


def has_checkpoint(out_dir: str) -> bool:
    """Whether a previously exported run directory has a resumable checkpoint."""
    return os.path.exists(export_paths(out_dir).checkpoint)


def to_dataframes(result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Columnar dict[str, list] (as returned by the Rust extension) -> DataFrames."""
    return {
        "trades": pd.DataFrame(result["trades"]),
        "order_events": pd.DataFrame(result["order_events"]),
        "orderbook_snapshots": pd.DataFrame(result["snapshots"]),
        "agent_state": pd.DataFrame(result["agent_states"]),
    }


def export_run(
    result: Dict[str, Any],
    config_toml: str,
    out_dir: str,
    *,
    scenario_name: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    write_order_events: bool = True,
) -> ExportPaths:
    """Write the full spec-section-7 export set for one run to `out_dir`.

    `config_toml` is stored verbatim in run_metadata.json so the run is exactly
    reproducible (same config text + same seed inside it -> same result, given the
    engine's deterministic ChaCha8 PRNG derivation).

    `write_order_events=False` (2026-10-04) skips writing order_events.parquet --
    measured as THE dominant cost of a run directory (68MB for a single 1000s leg
    of one scenario/seed, confirmed present even in scenarios unrelated to any
    recent engine change -- a pre-existing characteristic, not a regression), and
    confirmed unused by every report/analysis script in this repo (only
    `trades`/`agent_state` are ever read back). Filled a 1TB disk mid-sweep (see
    git history/session notes) before this flag existed. `load_run` below still
    expects the file to exist when reading one back, so an empty placeholder
    frame is written in its place rather than skipping the file entirely --
    keeps every existing reader working unchanged, just on an empty table.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = export_paths(out_dir)
    frames = to_dataframes(result)

    frames["trades"].to_parquet(paths.trades, index=False)
    if write_order_events:
        frames["order_events"].to_parquet(paths.order_events, index=False)
    else:
        frames["order_events"].iloc[0:0].to_parquet(paths.order_events, index=False)
    frames["orderbook_snapshots"].to_parquet(paths.orderbook_snapshots, index=False)
    frames["agent_state"].to_parquet(paths.agent_state, index=False)

    checkpoint = result.get("checkpoint")
    if checkpoint is not None:
        with open(paths.checkpoint, "wb") as f:
            f.write(checkpoint)

    meta = dict(result.get("meta", {}))
    meta["scenario_name"] = scenario_name
    meta["config_toml"] = config_toml
    if extra_meta:
        meta.update(extra_meta)
    with open(paths.run_metadata, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return paths


def df_to_columnar(df: pd.DataFrame) -> Dict[str, Any]:
    """The inverse of building a DataFrame from a `dict[str, list/ndarray]` result
    (`to_dataframes` above) -- `{col: df[col].to_numpy() for col in df.columns}`,
    NOT `df.to_dict("list")`. The two look interchangeable but aren't at any real
    size: `to_dict("list")` boxes every single cell into an individual Python
    object while building a plain list, which measured ~21s for one 1.4M-row/
    8M-order-event saved run (loading it in the dashboard, or via
    `market_sim.load_run_legs`, would hang for that long) -- the column-wise numpy
    version does the same conversion in under a millisecond."""
    return {col: df[col].to_numpy() for col in df.columns}


def load_trades_columns(out_dir: str, columns=("asset_id", "time", "price", "qty")) -> pd.DataFrame:
    """Reads just the given columns of a saved run's `trades.parquet` -- parquet
    supports column projection at the file-read level, so this skips
    deserializing the other columns (the bot-type/variant/agent-id strings and
    ids) entirely rather than reading the full table and discarding most of it.
    For the dashboard's per-leg candlestick summaries (only time/price/qty/
    asset_id are needed), this matters: those other columns are most of a
    trades.parquet's bytes."""
    return pd.read_parquet(export_paths(out_dir).trades, columns=list(columns))


def load_run(out_dir: str) -> Dict[str, Any]:
    """Read a previously exported run back in (DataFrames + metadata dict), without
    re-running the simulation -- what `market_health/` operates on. Also loads the
    `checkpoint` bytes (as `bytes`, key "checkpoint") when the run has one, i.e.
    `has_checkpoint(out_dir)` is true -- pass those to `continue_run`/
    `market_sim.continue_simulation` to keep simulating from exactly where this run
    stopped, rather than starting a new one from scratch."""
    paths = export_paths(out_dir)
    with open(paths.run_metadata) as f:
        meta = json.load(f)
    # order_events.parquet may be missing entirely (2026-10-04) on a leg saved
    # either by the disk-full crash this flag was added to prevent, or by this
    # session's own bulk cleanup of pre-existing order_events files once they
    # were confirmed unused -- `write_order_events=False` normally still writes
    # an empty placeholder (see `export_run`), but neither of those two cases
    # went through that path. Read back as empty rather than erroring, since
    # that's exactly what `write_order_events=False` would have produced here
    # anyway.
    order_events = pd.read_parquet(paths.order_events) if os.path.exists(paths.order_events) else pd.DataFrame()
    out = {
        "trades": pd.read_parquet(paths.trades),
        "order_events": order_events,
        "orderbook_snapshots": pd.read_parquet(paths.orderbook_snapshots),
        "agent_state": pd.read_parquet(paths.agent_state),
        "meta": meta,
    }
    if os.path.exists(paths.checkpoint):
        with open(paths.checkpoint, "rb") as f:
            out["checkpoint"] = f.read()
    return out
