"""market_sim: orchestration, config, export and plotting for the bot-dominated
crypto market simulation. The performance-critical engine itself lives in the
compiled Rust extension `market_sim.sim_pyo3` (see ../../sim_core, ../../sim_pyo3).
"""

from market_sim.sim_pyo3 import CancelToken, run_simulation, run_simulation_cancellable, run_simulation_from_file, validate_config
from market_sim.runner import SimulationRun, run_scenario_file, run_scenario_dict, simulate_and_plot
from market_sim.export import export_run, load_run, to_dataframes
from market_sim.experiment import ExperimentPlan, ExperimentPoint, build_plan_from_compositions, build_stress_test_plan, run_experiment

__all__ = [
    "run_simulation",
    "run_simulation_cancellable",
    "CancelToken",
    "run_simulation_from_file",
    "validate_config",
    "SimulationRun",
    "run_scenario_file",
    "run_scenario_dict",
    "simulate_and_plot",
    "export_run",
    "load_run",
    "to_dataframes",
    "ExperimentPlan",
    "ExperimentPoint",
    "build_plan_from_compositions",
    "build_stress_test_plan",
    "run_experiment",
]

__version__ = "0.1.0"
