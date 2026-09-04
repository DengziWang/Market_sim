"""Single-run convenience layer: run one scenario, export it, or plot it directly
without going through the batch experiment pipeline (spec section 8's `--plot` /
`simulate_and_plot(...)` requirement).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from market_sim import config as cfgmod
from market_sim.export import ExportPaths, export_run
from market_sim.sim_pyo3 import run_simulation


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
