"""Batch experiment runner: spec section 6 + 6a.

An "experiment" is one or more `ExperimentPoint`s (a bot composition/parameter
combination) x N seeds each -- never a single run in isolation, so results can be
reported as confidence intervals rather than point estimates (spec 6a: 20-50 seeds
per composition by default).

Crash-/stress-test scenarios (spec 6a) aren't a separate code path: they're just an
`ExperimentPoint` whose `overrides` set a short `run.session_seconds` and typically
more seeds -- see `scenarios/stress_wash_mev.toml` and `build_stress_test_plan`.

Parallelism: `n_workers` independent OS processes (default 5, leaving one core free
on the 4-6 core laptop target from spec 6a/11), each running one full simulation at a
time via `multiprocessing`. This is coarse-grained (embarrassingly-parallel across
runs) parallelism; per-run parallelism (Rayon, inside a single simulation) is a
separate, not-yet-wired-up axis -- see README performance section.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from market_sim import config as cfgmod
from market_sim.export import export_run
from market_sim.sim_pyo3 import run_simulation


@dataclass
class ExperimentPoint:
    """One bot-composition/parameter cell of the experiment grid, to be run at each
    seed in `seeds`. `overrides` is shallow-merged onto `base_config` per
    `market_sim.config.with_overrides` (e.g. `{"population": {...}}` to swap the bot
    mix, or `{"run": {"session_seconds": 300}}` for a stress-test window)."""

    name: str
    base_config: Dict[str, Any]
    seeds: Sequence[int]
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentPlan:
    points: List[ExperimentPoint]
    out_dir: str

    def run_dir(self, point_name: str, seed: int) -> str:
        return os.path.join(self.out_dir, point_name, f"seed_{seed}")

    def all_run_dirs(self) -> List[Tuple[str, int, str]]:
        """(point_name, seed, run_dir) for every cell -- handy for the analysis side
        (`market_health.batch_compare`) to discover what a plan produced."""
        return [(p.name, s, self.run_dir(p.name, s)) for p in self.points for s in p.seeds]


def build_plan_from_compositions(
    base_config_path: str,
    compositions: Dict[str, Dict[str, Any]],
    seeds: Sequence[int],
    out_dir: str,
) -> ExperimentPlan:
    """`compositions`: name -> overrides dict, e.g.
    `{"realistic_mix": {}, "no_market_makers": {"population": {...}}}`. Each becomes
    one `ExperimentPoint` run at every seed in `seeds`."""
    base = cfgmod.load_config_dict(base_config_path)
    points = [ExperimentPoint(name=name, base_config=base, seeds=list(seeds), overrides=overrides) for name, overrides in compositions.items()]
    return ExperimentPlan(points=points, out_dir=out_dir)


def build_stress_test_plan(
    base_config_path: str,
    shock_overrides: Dict[str, Any],
    seeds: Sequence[int],
    out_dir: str,
    name: str = "stress_test",
) -> ExperimentPlan:
    """Spec 6a's "short, focused window with more repeats" experiment type. Same
    mechanism as `build_plan_from_compositions`, just named for clarity -- typically
    `shock_overrides` sets a short `run.session_seconds` and/or a heavier
    wash-trading/MEV population (see `scenarios/stress_wash_mev.toml`)."""
    return build_plan_from_compositions(base_config_path, {name: shock_overrides}, seeds, out_dir)


def _run_one(args: Tuple[str, str, str, int]):
    out_dir, config_toml, scenario_name, seed = args
    if os.path.exists(os.path.join(out_dir, "run_metadata.json")):
        return out_dir, True, None  # resume: already completed
    try:
        result = run_simulation(config_toml)
        export_run(result, config_toml, out_dir, scenario_name=scenario_name, extra_meta={"seed": seed})
        return out_dir, True, None
    except Exception as e:  # noqa: BLE001 - reported to the caller, never swallowed
        return out_dir, False, f"{type(e).__name__}: {e}"


def run_experiment(plan: ExperimentPlan, n_workers: int = 5, resume: bool = True) -> Dict[str, Any]:
    """Execute every (composition x seed) cell of `plan` in parallel across
    `n_workers` processes. Each cell writes its own export directory
    (`plan.run_dir(name, seed)`); with `resume=True` (default) a cell whose
    `run_metadata.json` already exists is skipped, so a killed/interrupted batch can
    just be restarted with the same plan and pick up where it left off."""
    os.makedirs(plan.out_dir, exist_ok=True)
    tasks = []
    for point in plan.points:
        cfg = cfgmod.with_overrides(point.base_config, **point.overrides) if point.overrides else point.base_config
        for seed in point.seeds:
            run_cfg = cfgmod.with_seed(cfg, seed)
            out_dir = plan.run_dir(point.name, seed)
            if resume and os.path.exists(os.path.join(out_dir, "run_metadata.json")):
                continue
            tasks.append((out_dir, cfgmod.dump_config_str(run_cfg), point.name, seed))

    results = []
    if tasks:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=max(1, n_workers)) as pool:
            for res in tqdm(pool.imap_unordered(_run_one, tasks), total=len(tasks), desc=f"experiment '{os.path.basename(plan.out_dir)}'"):
                results.append(res)

    failures = [{"run_dir": d, "error": e} for d, ok, e in results if not ok]
    summary = {
        "n_points": len(plan.points),
        "n_cells_total": sum(len(p.seeds) for p in plan.points),
        "n_cells_run_this_call": len(tasks),
        "n_cells_skipped_resume": sum(len(p.seeds) for p in plan.points) - len(tasks),
        "n_failed": len(failures),
        "failures": failures,
    }
    with open(os.path.join(plan.out_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
