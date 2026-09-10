"""Single-run convenience layer: run one scenario, export it, or plot it directly
without going through the batch experiment pipeline (spec section 8's `--plot` /
`simulate_and_plot(...)` requirement).
"""
from __future__ import annotations

import copy
import glob
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from market_sim import config as cfgmod
from market_sim.export import ExportPaths, export_run, load_run
from market_sim.sim_pyo3 import CancelToken, continue_simulation, continue_simulation_cancellable, run_simulation, run_simulation_cancellable


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


@dataclass
class LegResult:
    index: int
    out_dir: str
    paths: ExportPaths
    meta: Dict[str, Any]


def _leg_dir(out_dir: str, index: int) -> str:
    return os.path.join(out_dir, f"leg_{index:04d}")


def run_scenario_in_legs(
    cfg: Dict[str, Any],
    leg_seconds: float,
    n_legs: int,
    out_dir: str,
    *,
    scenario_name: Optional[str] = None,
    cancel_token: Optional[CancelToken] = None,
    on_leg_saved: Optional[Callable[[LegResult], None]] = None,
) -> List[LegResult]:
    """Runs a scenario as `n_legs` sequential sessions of `leg_seconds` simulated
    seconds each, saving every leg to its own subdirectory
    (`out_dir/leg_0001/`, `leg_0002/`, ...) as it finishes, instead of one huge
    session held entirely in memory at once -- see README's "Memory: the actual
    scaling limit" section for why that matters at the population/session sizes the
    spec describes.

    Leg 1 is a fresh run (`run_simulation`); every leg after that is a genuine
    continuation of the previous leg's checkpoint (`continue_simulation`, see
    `Simulation::resume` in sim_core) -- same order books, every agent's state, both
    RNG streams, nothing re-seeded, not a series of independent restarts. This
    matters for more than just correctness: a *fresh* start's order book is empty
    and every bot's indicators are unwarmed, so essentially the whole population
    takes its first action within the first simulated second (each agent's initial
    wake is jittered uniformly over `[0, 1)`) -- a real, measured cold-start burst
    (in one test, ~9x the steady-state trade rate in the first second, back to
    normal within ~5s) that has nothing to do with the market dynamics being
    studied. Because legs 2+ genuinely continue rather than restart, they pick up an
    already-warmed, already-trading market and do NOT reproduce that burst
    (verified: a continued leg's first-second trade count sits within its own normal
    range, no spike) -- only leg 1 sees it, same as it would in a single
    non-split run of any length.

    Each leg runs for *exactly* `leg_seconds` -- `run.min_trades` /
    `run.max_session_seconds` from `cfg` are dropped (a warning is emitted if either
    was set) since open-ended per-leg extension would defeat the point of a
    predictable, bounded-size chunk. If you need a `min_trades` *floor* across the
    whole sequence, add more legs rather than relying on within-leg extension.

    Memory during the run stays roughly bounded to one leg's data at a time: each
    leg's (large) result dict is exported to disk and then dropped (not
    accumulated), only the small `checkpoint` bytes carry over to the next leg. Pass
    `cancel_token` to stop between legs (checked after each leg finishes, and pushed
    down into that leg's own run so a very long individual leg can also stop mid-way);
    a leg that stopped early is still saved, and the returned list reflects only the
    legs actually completed. `on_leg_saved(leg_result)`, if given, is called after
    each leg is written -- e.g. for a progress readout without loading anything back
    off disk.
    """
    cfg = copy.deepcopy(cfg)
    run_cfg = cfg.setdefault("run", {})
    ignored = [k for k in ("min_trades", "max_session_seconds") if run_cfg.get(k)]
    if ignored:
        warnings.warn(
            f"run_scenario_in_legs: ignoring run.{ignored} from the input config -- "
            "each leg runs for exactly `leg_seconds`, no auto-extension, since that's "
            "what keeps every leg's memory footprint bounded and predictable.",
            stacklevel=2,
        )
    run_cfg.pop("min_trades", None)
    run_cfg.pop("max_session_seconds", None)
    run_cfg["session_seconds"] = leg_seconds
    config_toml = cfgmod.dump_config_str(cfg)

    os.makedirs(out_dir, exist_ok=True)
    results: List[LegResult] = []
    checkpoint: Optional[bytes] = None
    for leg_idx in range(1, n_legs + 1):
        if leg_idx == 1:
            result = run_simulation_cancellable(config_toml, cancel_token) if cancel_token is not None else run_simulation(config_toml)
        else:
            result = (
                continue_simulation_cancellable(checkpoint, leg_seconds, cancel_token)
                if cancel_token is not None
                else continue_simulation(checkpoint, leg_seconds)
            )
        leg_out_dir = _leg_dir(out_dir, leg_idx)
        paths = export_run(
            result,
            config_toml,
            leg_out_dir,
            scenario_name=scenario_name,
            extra_meta={"leg_index": leg_idx, "n_legs": n_legs, "leg_seconds": leg_seconds},
        )
        leg_result = LegResult(index=leg_idx, out_dir=leg_out_dir, paths=paths, meta=dict(result["meta"]))
        results.append(leg_result)
        checkpoint = result["checkpoint"]
        was_cancelled = result["meta"].get("cancelled", False)
        # Drop the reference to this leg's (large) result before starting the next
        # one -- CPython refcounting frees it immediately, not at some later GC
        # cycle, which is what keeps peak memory bounded to ~one leg at a time
        # across a long sequence rather than growing with every leg completed.
        del result
        if on_leg_saved is not None:
            on_leg_saved(leg_result)
        if was_cancelled:
            break
    return results


def discover_legs(out_dir: str) -> List[str]:
    """Lists a `run_scenario_in_legs` sequence's leg directories under `out_dir`, in
    order (`leg_0001`, `leg_0002`, ...). Each is independently loadable with
    `market_sim.load_run`."""
    return sorted(p for p in glob.glob(os.path.join(out_dir, "leg_*")) if os.path.exists(os.path.join(p, "run_metadata.json")))


def load_run_legs(out_dir: str) -> Dict[str, Any]:
    """Loads every leg under `out_dir` (see `discover_legs`) and concatenates them
    into one continuous combined result, the same shape `market_sim.load_run`
    returns for a single run -- for when you actually want the full multi-leg
    history for analysis. Unlike running the legs in the first place, this DOES
    load everything into memory at once (same as loading any single large saved
    run would), so it's only appropriate once you're ready to analyze the combined
    result, not as a step during the run itself."""
    legs = discover_legs(out_dir)
    if not legs:
        raise FileNotFoundError(f"no leg_* subdirectories with a run_metadata.json found under {out_dir}")
    combined = None
    for leg_dir in legs:
        loaded = load_run(leg_dir)
        leg_result = {
            "trades": loaded["trades"].to_dict("list"),
            "order_events": loaded["order_events"].to_dict("list"),
            "snapshots": loaded["orderbook_snapshots"].to_dict("list"),
            "agent_states": loaded["agent_state"].to_dict("list"),
            "meta": loaded["meta"],
        }
        if "checkpoint" in loaded:
            leg_result["checkpoint"] = loaded["checkpoint"]
        combined = leg_result if combined is None else combine_results(combined, leg_result)
    return combined
