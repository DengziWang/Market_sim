"""Read/write scenario TOML configs from Python. The config *schema* itself is
defined by `sim_core::config::ScenarioConfig` (Rust) -- this module is just a thin,
convenient read/write/patch layer so experiment scripts can build variations of a
base scenario without hand-editing TOML each time.
"""
from __future__ import annotations

import copy
from typing import Any, Dict

import toml


def load_config_dict(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return toml.load(f)


def dump_config_str(cfg: Dict[str, Any]) -> str:
    return toml.dumps(cfg)


def save_config(cfg: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        toml.dump(cfg, f)


def with_overrides(cfg: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    """Shallow-merge top-level overrides (e.g. `with_overrides(cfg, run={"seed": 7})`
    merges into `cfg["run"]` rather than replacing it), returning a new dict; `cfg`
    is left untouched. Useful for seed sweeps / parameter grids in the experiment
    runner without repeatedly re-parsing a TOML file."""
    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def with_seed(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    return with_overrides(cfg, run={**cfg.get("run", {}), "seed": seed})


def total_agents(cfg: Dict[str, Any]) -> int:
    return sum(int(g["count"]) for g in cfg.get("population", {}).get("groups", []))
