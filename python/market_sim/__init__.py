"""market_sim: orchestration, config, export and plotting for the bot-dominated
crypto market simulation. The performance-critical engine itself lives in the
compiled Rust extension `market_sim.sim_pyo3` (see ../../sim_core, ../../sim_pyo3).
"""

from market_sim.sim_pyo3 import (
    CancelToken,
    ProgressHandle,
    continue_simulation,
    continue_simulation_cancellable,
    graft_from_checkpoint,
    run_simulation,
    run_simulation_cancellable,
    run_simulation_from_file,
    run_warmup_then_graft,
    validate_config,
)
from market_sim.runner import (
    LegResult,
    SimulationRun,
    continue_run,
    discover_legs,
    load_run_legs,
    run_scenario_dict,
    run_scenario_file,
    run_scenario_in_legs,
    simulate_and_plot,
)
from market_sim.export import df_to_columnar, export_run, load_run, load_trades_columns, has_checkpoint, to_dataframes
from market_sim.experiment import ExperimentPlan, ExperimentPoint, build_plan_from_compositions, build_stress_test_plan, run_experiment

__all__ = [
    "run_simulation",
    "run_simulation_cancellable",
    "continue_simulation",
    "continue_simulation_cancellable",
    "CancelToken",
    "ProgressHandle",
    "run_simulation_from_file",
    "run_warmup_then_graft",
    "graft_from_checkpoint",
    "validate_config",
    "SimulationRun",
    "run_scenario_file",
    "run_scenario_dict",
    "continue_run",
    "run_scenario_in_legs",
    "LegResult",
    "discover_legs",
    "load_run_legs",
    "simulate_and_plot",
    "export_run",
    "load_run",
    "load_trades_columns",
    "df_to_columnar",
    "has_checkpoint",
    "to_dataframes",
    "ExperimentPlan",
    "ExperimentPoint",
    "build_plan_from_compositions",
    "build_stress_test_plan",
    "run_experiment",
]

__version__ = "0.1.0"
