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


def export_paths(out_dir: str) -> ExportPaths:
    return ExportPaths(
        out_dir=out_dir,
        trades=os.path.join(out_dir, "trades.parquet"),
        order_events=os.path.join(out_dir, "order_events.parquet"),
        orderbook_snapshots=os.path.join(out_dir, "orderbook_snapshots.parquet"),
        agent_state=os.path.join(out_dir, "agent_state.parquet"),
        run_metadata=os.path.join(out_dir, "run_metadata.json"),
    )


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
) -> ExportPaths:
    """Write the full spec-section-7 export set for one run to `out_dir`.

    `config_toml` is stored verbatim in run_metadata.json so the run is exactly
    reproducible (same config text + same seed inside it -> same result, given the
    engine's deterministic ChaCha8 PRNG derivation).
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = export_paths(out_dir)
    frames = to_dataframes(result)

    frames["trades"].to_parquet(paths.trades, index=False)
    frames["order_events"].to_parquet(paths.order_events, index=False)
    frames["orderbook_snapshots"].to_parquet(paths.orderbook_snapshots, index=False)
    frames["agent_state"].to_parquet(paths.agent_state, index=False)

    meta = dict(result.get("meta", {}))
    meta["scenario_name"] = scenario_name
    meta["config_toml"] = config_toml
    if extra_meta:
        meta.update(extra_meta)
    with open(paths.run_metadata, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return paths


def load_run(out_dir: str) -> Dict[str, Any]:
    """Read a previously exported run back in (DataFrames + metadata dict), without
    re-running the simulation -- what `market_health/` operates on."""
    paths = export_paths(out_dir)
    with open(paths.run_metadata) as f:
        meta = json.load(f)
    return {
        "trades": pd.read_parquet(paths.trades),
        "order_events": pd.read_parquet(paths.order_events),
        "orderbook_snapshots": pd.read_parquet(paths.orderbook_snapshots),
        "agent_state": pd.read_parquet(paths.agent_state),
        "meta": meta,
    }
