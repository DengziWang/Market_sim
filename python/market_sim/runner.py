"""Single-run convenience layer: run one scenario, export it, or plot it directly
without going through the batch experiment pipeline (spec section 8's `--plot` /
`simulate_and_plot(...)` requirement).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from market_sim import config as cfgmod
from market_sim.export import ExportPaths, export_run
from market_sim.sim_pyo3 import CancelToken, continue_simulation, continue_simulation_cancellable, run_simulation


@dataclass
class SimulationRun:
    result: Dict[str, Any]
    config_toml: str
    out_dir: Optional[str] = None

    @property
    def meta(self) -> Dict[str, Any]:
        return self.result.get("meta", {})

    def export(self, out_dir: str, scenario_name: Optional[str] = None, **extra_meta: Any) -> ExportPaths:
        paths = export_run(self.result, self.config_toml, out_dir, scenario_name=scenario_name, extra_meta=extra_meta or None)
        self.out_dir = out_dir
        return paths

    def plot(self, **kwargs):
        from market_sim.plotting import plot_run

        return plot_run(self.result, **kwargs)


def combine_results(previous: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Concatenate a checkpointed run's earlier `result` with the fresh `result` from
    resuming it (`continue_run`/`continue_simulation` deliberately return only the
    newly-simulated segment, not the full history -- see `Simulation::resume`'s doc
    comment in sim_core). Trade/order-event/snapshot tables are appended end to end
    (times in `new` continue on from where `previous` left off, so this stays in
    chronological order); `agent_states` is NOT concatenated -- it's always a full
    point-in-time snapshot of every agent, so `new`'s copy alone is the current
    truth. Counters (`n_trades`, `liquidation_events`, `bankruptcies`) are summed;
    `checkpoint` and `cancelled` are carried over from `new` (the latest state)."""
    combined_columnar: Dict[str, Any] = {}
    for key in ("trades", "order_events", "snapshots"):
        prev_cols, new_cols = previous.get(key, {}), new.get(key, {})
        combined_columnar[key] = {col: [*prev_cols.get(col, []), *new_cols.get(col, [])] for col in new_cols.keys() or prev_cols.keys()}
    combined_columnar["agent_states"] = new.get("agent_states", previous.get("agent_states", {}))

    prev_meta, new_meta = previous.get("meta", {}), new.get("meta", {})
    combined_meta = dict(new_meta)
    # `actual_session_seconds` is already absolute simulated time since t=0 (not a
    # per-leg delta) on both a fresh run and a resumed one -- sim_core's `resume`
    # extends `session_end` from wherever the checkpoint left off, so `new`'s value
    # already accounts for everything before it. Carrying it over as-is (not adding
    # `previous`'s) is what's correct here.
    combined_meta["actual_session_seconds"] = new_meta.get("actual_session_seconds", 0.0)
    combined_meta["n_trades"] = len(combined_columnar["trades"].get("time", []))
    combined_meta["n_order_events"] = len(combined_columnar["order_events"].get("time", []))
    combined_meta["liquidation_events"] = prev_meta.get("liquidation_events", 0) + new_meta.get("liquidation_events", 0)
    combined_meta["bankruptcies"] = prev_meta.get("bankruptcies", 0) + new_meta.get("bankruptcies", 0)
    combined_columnar["meta"] = combined_meta
    if "checkpoint" in new:
        combined_columnar["checkpoint"] = new["checkpoint"]
    return combined_columnar


def continue_run(
    checkpoint: bytes,
    additional_seconds: float,
    *,
    config_toml: str = "",
    cancel_token: Optional[CancelToken] = None,
    previous_result: Optional[Dict[str, Any]] = None,
) -> SimulationRun:
    """Load a checkpoint (from a prior run's `result["checkpoint"]`, or
    `market_sim.load_run(out_dir)["checkpoint"]` for one already exported to disk)
    and keep simulating for `additional_seconds` more simulated time from exactly
    where it left off -- same order books, every agent's state, both RNG streams,
    pending events, none of it re-seeded (see `Simulation::resume` in sim_core).

    By default the returned run's tables cover only the newly-simulated segment
    (consistent with the underlying `continue_simulation` binding). Pass
    `previous_result` (the dict this checkpoint came from) to get one continuous
    combined history back instead -- see `combine_results`.
    """
    if cancel_token is not None:
        result = continue_simulation_cancellable(checkpoint, additional_seconds, cancel_token)
    else:
        result = continue_simulation(checkpoint, additional_seconds)
    if previous_result is not None:
        result = combine_results(previous_result, result)
    return SimulationRun(result=result, config_toml=config_toml)


def run_scenario_dict(cfg: Dict[str, Any]) -> SimulationRun:
    text = cfgmod.dump_config_str(cfg)
    result = run_simulation(text)
    return SimulationRun(result=result, config_toml=text)


def run_scenario_file(path: str) -> SimulationRun:
    with open(path) as f:
        text = f.read()
    result = run_simulation(text)
    return SimulationRun(result=result, config_toml=text)


def simulate_and_plot(config_path: str, save_path: Optional[str] = None, show: bool = False, **plot_kwargs):
    """One-shot: run a single scenario and immediately produce the price/volume/
    spread chart from spec section 8 -- the quick "does this look sane" entry point
    that doesn't need the full batch export+analysis pipeline."""
    run = run_scenario_file(config_path)
    fig = run.plot(**plot_kwargs)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    return run, fig
