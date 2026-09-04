//! Thin PyO3 wrapper around `sim_core`. Converts a TOML scenario config into a
//! `Simulation`, runs it (with the GIL released, since the run is pure Rust), and
//! hands the result back to Python as plain columnar `dict[str, list]` structures --
//! `market_sim.export` turns those into the parquet files from spec section 7 using
//! pandas/pyarrow, so no parquet/arrow dependency is needed on the Rust side at all.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sim_core::config::ScenarioConfig;
use sim_core::export::{AgentStateRow, OrderEventRow, SimResult, SnapshotRow, TradeRow};
use sim_core::sim::Simulation;

fn trades_dict<'py>(py: Python<'py>, rows: &[TradeRow]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("time", rows.iter().map(|r| r.time).collect::<Vec<_>>())?;
    d.set_item("asset_id", rows.iter().map(|r| r.asset_id).collect::<Vec<_>>())?;
    d.set_item("venue_id", rows.iter().map(|r| r.venue_id).collect::<Vec<_>>())?;
    d.set_item("price", rows.iter().map(|r| r.price).collect::<Vec<_>>())?;
    d.set_item("qty", rows.iter().map(|r| r.qty).collect::<Vec<_>>())?;
    d.set_item("taker_side", rows.iter().map(|r| r.taker_side).collect::<Vec<_>>())?;
    d.set_item("maker_agent_id", rows.iter().map(|r| r.maker_agent_id).collect::<Vec<_>>())?;
    d.set_item("taker_agent_id", rows.iter().map(|r| r.taker_agent_id).collect::<Vec<_>>())?;
    d.set_item("maker_type", rows.iter().map(|r| r.maker_type).collect::<Vec<_>>())?;
    d.set_item("maker_variant", rows.iter().map(|r| r.maker_variant).collect::<Vec<_>>())?;
    d.set_item("taker_type", rows.iter().map(|r| r.taker_type).collect::<Vec<_>>())?;
    d.set_item("taker_variant", rows.iter().map(|r| r.taker_variant).collect::<Vec<_>>())?;
    d.set_item("maker_fee_paid", rows.iter().map(|r| r.maker_fee_paid).collect::<Vec<_>>())?;
    d.set_item("taker_fee_paid", rows.iter().map(|r| r.taker_fee_paid).collect::<Vec<_>>())?;
    Ok(d)
}

fn order_events_dict<'py>(py: Python<'py>, rows: &[OrderEventRow]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("time", rows.iter().map(|r| r.time).collect::<Vec<_>>())?;
    d.set_item("order_id", rows.iter().map(|r| r.order_id).collect::<Vec<_>>())?;
    d.set_item("agent_id", rows.iter().map(|r| r.agent_id).collect::<Vec<_>>())?;
    d.set_item("asset_id", rows.iter().map(|r| r.asset_id).collect::<Vec<_>>())?;
    d.set_item("venue_id", rows.iter().map(|r| r.venue_id).collect::<Vec<_>>())?;
    d.set_item("kind", rows.iter().map(|r| r.kind).collect::<Vec<_>>())?;
    d.set_item("side", rows.iter().map(|r| r.side).collect::<Vec<_>>())?;
    d.set_item("price", rows.iter().map(|r| r.price).collect::<Vec<_>>())?;
    d.set_item("qty", rows.iter().map(|r| r.qty).collect::<Vec<_>>())?;
    Ok(d)
}

fn snapshots_dict<'py>(py: Python<'py>, rows: &[SnapshotRow]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("time", rows.iter().map(|r| r.time).collect::<Vec<_>>())?;
    d.set_item("asset_id", rows.iter().map(|r| r.asset_id).collect::<Vec<_>>())?;
    d.set_item("venue_id", rows.iter().map(|r| r.venue_id).collect::<Vec<_>>())?;
    d.set_item("side", rows.iter().map(|r| r.side).collect::<Vec<_>>())?;
    d.set_item("level", rows.iter().map(|r| r.level).collect::<Vec<_>>())?;
    d.set_item("price", rows.iter().map(|r| r.price).collect::<Vec<_>>())?;
    d.set_item("qty", rows.iter().map(|r| r.qty).collect::<Vec<_>>())?;
    Ok(d)
}

fn agent_states_dict<'py>(py: Python<'py>, rows: &[AgentStateRow]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("agent_id", rows.iter().map(|r| r.agent_id).collect::<Vec<_>>())?;
    d.set_item("bot_type", rows.iter().map(|r| r.bot_type).collect::<Vec<_>>())?;
    d.set_item("variant", rows.iter().map(|r| r.variant).collect::<Vec<_>>())?;
    d.set_item("asset_id", rows.iter().map(|r| r.asset_id).collect::<Vec<_>>())?;
    d.set_item("inventory", rows.iter().map(|r| r.inventory).collect::<Vec<_>>())?;
    d.set_item("cash", rows.iter().map(|r| r.cash).collect::<Vec<_>>())?;
    d.set_item("equity", rows.iter().map(|r| r.equity).collect::<Vec<_>>())?;
    d.set_item("realized_pnl", rows.iter().map(|r| r.realized_pnl).collect::<Vec<_>>())?;
    d.set_item("cumulative_volume", rows.iter().map(|r| r.cumulative_volume).collect::<Vec<_>>())?;
    d.set_item("bankrupt", rows.iter().map(|r| r.bankrupt).collect::<Vec<_>>())?;
    d.set_item("has_margin_position", rows.iter().map(|r| r.has_margin_position).collect::<Vec<_>>())?;
    d.set_item("margin_leverage", rows.iter().map(|r| r.margin_leverage).collect::<Vec<_>>())?;
    Ok(d)
}

fn result_to_pyobject(py: Python<'_>, result: SimResult, n_agents: u32) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);
    out.set_item("trades", trades_dict(py, &result.trades)?)?;
    out.set_item("order_events", order_events_dict(py, &result.order_events)?)?;
    out.set_item("snapshots", snapshots_dict(py, &result.snapshots)?)?;
    out.set_item("agent_states", agent_states_dict(py, &result.agent_states)?)?;
    let meta = PyDict::new_bound(py);
    meta.set_item("actual_session_seconds", result.actual_session_seconds)?;
    meta.set_item("liquidation_events", result.liquidation_events)?;
    meta.set_item("bankruptcies", result.bankruptcies)?;
    meta.set_item("n_agents", n_agents)?;
    meta.set_item("n_trades", result.trades.len())?;
    meta.set_item("n_order_events", result.order_events.len())?;
    out.set_item("meta", meta)?;
    Ok(out.into())
}

/// Run one simulation from a TOML scenario config string. Returns a dict with keys
/// `trades`, `order_events`, `snapshots`, `agent_states` (each `dict[str, list]`,
/// columnar) and `meta` (session/run summary) -- see `market_sim.export` for turning
/// this into the parquet files from spec section 7.
#[pyfunction]
fn run_simulation(py: Python<'_>, config_toml: &str) -> PyResult<PyObject> {
    let cfg = ScenarioConfig::from_toml_str(config_toml).map_err(|e| PyValueError::new_err(format!("config parse error: {e}")))?;
    let n_agents = cfg.total_agents();
    let sim = Simulation::build(cfg).map_err(PyValueError::new_err)?;
    let result = py.allow_threads(|| sim.run());
    result_to_pyobject(py, result, n_agents)
}

/// Same as `run_simulation`, but reads the TOML config from a file path.
#[pyfunction]
fn run_simulation_from_file(py: Python<'_>, config_path: &str) -> PyResult<PyObject> {
    let text = std::fs::read_to_string(config_path).map_err(|e| PyValueError::new_err(format!("could not read '{config_path}': {e}")))?;
    run_simulation(py, &text)
}

/// Parse-only: returns (n_assets, n_venues, n_agents) without running anything --
/// useful for quickly sanity-checking a config (e.g. before a big batch).
#[pyfunction]
fn validate_config(config_toml: &str) -> PyResult<(usize, usize, u32)> {
    let cfg = ScenarioConfig::from_toml_str(config_toml).map_err(|e| PyValueError::new_err(format!("config parse error: {e}")))?;
    Ok((cfg.assets.len(), cfg.venues.len().max(1), cfg.total_agents()))
}

#[pymodule]
fn sim_pyo3(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(run_simulation_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(validate_config, m)?)?;
    Ok(())
}
