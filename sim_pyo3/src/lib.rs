//! Thin PyO3 wrapper around `sim_core`. Converts a TOML scenario config into a
//! `Simulation`, runs it (with the GIL released, since the run is pure Rust), and
//! hands the result back to Python as plain columnar `dict[str, list]` structures --
//! `market_sim.export` turns those into the parquet files from spec section 7 using
//! pandas/pyarrow, so no parquet/arrow dependency is needed on the Rust side at all.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use sim_core::config::ScenarioConfig;
use sim_core::export::{AgentStateRow, OrderEventRow, SimResult, SnapshotRow, TradeRow};
use sim_core::sim::Simulation;

/// Numeric columns are handed to Python as numpy arrays, not plain lists: a plain
/// `Vec<f64>`/`Vec<u32>`/etc. collected via PyO3's blanket `IntoPy` impl becomes a
/// Python `list` of individually boxed `PyFloat`/`PyLong` objects -- for a run with
/// millions of rows (order_events scales with agent count x session length x wake
/// frequency) that boxing dominated both time and peak memory well before pandas
/// ever got a chance to convert it into a compact typed column anyway. A numpy array
/// is the same bits `Vec<T>` already held, handed over with one allocation instead of
/// one-PyObject-per-element; `pd.DataFrame({...})` accepts numpy arrays directly (see
/// `market_sim.export.to_dataframes`) with no format change on the Python side. Pure
/// representation change -- every value transferred is numerically identical.
fn set_numpy_col<'py, T: numpy::Element>(d: &Bound<'py, PyDict>, key: &str, v: Vec<T>) -> PyResult<()> {
    d.set_item(key, PyArray1::from_vec_bound(d.py(), v))
}

fn trades_dict<'py>(py: Python<'py>, rows: &[TradeRow]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    set_numpy_col(&d, "time", rows.iter().map(|r| r.time).collect::<Vec<_>>())?;
    set_numpy_col(&d, "asset_id", rows.iter().map(|r| r.asset_id).collect::<Vec<_>>())?;
    set_numpy_col(&d, "venue_id", rows.iter().map(|r| r.venue_id).collect::<Vec<_>>())?;
    set_numpy_col(&d, "price", rows.iter().map(|r| r.price).collect::<Vec<_>>())?;
    set_numpy_col(&d, "qty", rows.iter().map(|r| r.qty).collect::<Vec<_>>())?;
    // Strings stay as plain lists (numpy has no compact variable-length string type
    // without extra complexity); the dashboard/export side collapses these few
    // repeated values with pandas `.astype("category")` instead.
    d.set_item("taker_side", rows.iter().map(|r| r.taker_side).collect::<Vec<_>>())?;
    set_numpy_col(&d, "maker_agent_id", rows.iter().map(|r| r.maker_agent_id).collect::<Vec<_>>())?;
    set_numpy_col(&d, "taker_agent_id", rows.iter().map(|r| r.taker_agent_id).collect::<Vec<_>>())?;
    d.set_item("maker_type", rows.iter().map(|r| r.maker_type).collect::<Vec<_>>())?;
    d.set_item("maker_variant", rows.iter().map(|r| r.maker_variant).collect::<Vec<_>>())?;
    d.set_item("taker_type", rows.iter().map(|r| r.taker_type).collect::<Vec<_>>())?;
    d.set_item("taker_variant", rows.iter().map(|r| r.taker_variant).collect::<Vec<_>>())?;
    set_numpy_col(&d, "maker_fee_paid", rows.iter().map(|r| r.maker_fee_paid).collect::<Vec<_>>())?;
    set_numpy_col(&d, "taker_fee_paid", rows.iter().map(|r| r.taker_fee_paid).collect::<Vec<_>>())?;
    Ok(d)
}

fn order_events_dict<'py>(py: Python<'py>, rows: &[OrderEventRow]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    set_numpy_col(&d, "time", rows.iter().map(|r| r.time).collect::<Vec<_>>())?;
    set_numpy_col(&d, "order_id", rows.iter().map(|r| r.order_id).collect::<Vec<_>>())?;
    set_numpy_col(&d, "agent_id", rows.iter().map(|r| r.agent_id).collect::<Vec<_>>())?;
    set_numpy_col(&d, "asset_id", rows.iter().map(|r| r.asset_id).collect::<Vec<_>>())?;
    set_numpy_col(&d, "venue_id", rows.iter().map(|r| r.venue_id).collect::<Vec<_>>())?;
    d.set_item("kind", rows.iter().map(|r| r.kind).collect::<Vec<_>>())?;
    d.set_item("side", rows.iter().map(|r| r.side).collect::<Vec<_>>())?;
    // `Option<f64>::None` (market orders have no limit price) -> NaN: exactly the
    // sentinel pandas already uses to represent a missing float64 value, so this
    // isn't a lossy conversion -- it's the same representation `pd.DataFrame` would
    // produce from the old list-of-`Option[float]` (which pandas maps to NaN) too.
    set_numpy_col(&d, "price", rows.iter().map(|r| r.price.unwrap_or(f64::NAN)).collect::<Vec<_>>())?;
    set_numpy_col(&d, "qty", rows.iter().map(|r| r.qty).collect::<Vec<_>>())?;
    Ok(d)
}

fn snapshots_dict<'py>(py: Python<'py>, rows: &[SnapshotRow]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    set_numpy_col(&d, "time", rows.iter().map(|r| r.time).collect::<Vec<_>>())?;
    set_numpy_col(&d, "asset_id", rows.iter().map(|r| r.asset_id).collect::<Vec<_>>())?;
    set_numpy_col(&d, "venue_id", rows.iter().map(|r| r.venue_id).collect::<Vec<_>>())?;
    d.set_item("side", rows.iter().map(|r| r.side).collect::<Vec<_>>())?;
    set_numpy_col(&d, "level", rows.iter().map(|r| r.level).collect::<Vec<_>>())?;
    set_numpy_col(&d, "price", rows.iter().map(|r| r.price).collect::<Vec<_>>())?;
    set_numpy_col(&d, "qty", rows.iter().map(|r| r.qty).collect::<Vec<_>>())?;
    Ok(d)
}

fn agent_states_dict<'py>(py: Python<'py>, rows: &[AgentStateRow]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    set_numpy_col(&d, "agent_id", rows.iter().map(|r| r.agent_id).collect::<Vec<_>>())?;
    d.set_item("bot_type", rows.iter().map(|r| r.bot_type).collect::<Vec<_>>())?;
    d.set_item("variant", rows.iter().map(|r| r.variant).collect::<Vec<_>>())?;
    set_numpy_col(&d, "asset_id", rows.iter().map(|r| r.asset_id).collect::<Vec<_>>())?;
    set_numpy_col(&d, "inventory", rows.iter().map(|r| r.inventory).collect::<Vec<_>>())?;
    set_numpy_col(&d, "cash", rows.iter().map(|r| r.cash).collect::<Vec<_>>())?;
    set_numpy_col(&d, "equity", rows.iter().map(|r| r.equity).collect::<Vec<_>>())?;
    set_numpy_col(&d, "realized_pnl", rows.iter().map(|r| r.realized_pnl).collect::<Vec<_>>())?;
    set_numpy_col(&d, "cumulative_volume", rows.iter().map(|r| r.cumulative_volume).collect::<Vec<_>>())?;
    set_numpy_col(&d, "bankrupt", rows.iter().map(|r| r.bankrupt).collect::<Vec<_>>())?;
    set_numpy_col(&d, "has_margin_position", rows.iter().map(|r| r.has_margin_position).collect::<Vec<_>>())?;
    // See the `price` comment in order_events_dict -- same None -> NaN reasoning.
    set_numpy_col(&d, "margin_leverage", rows.iter().map(|r| r.margin_leverage.unwrap_or(f64::NAN)).collect::<Vec<_>>())?;
    Ok(d)
}

/// Builds the same result dict as before, plus a `checkpoint` key (`bytes`) holding
/// `sim`'s full internal state at the moment it stopped -- see `run_simulation`'s
/// doc comment. Checkpoint serialization failures are surfaced as a Python
/// exception rather than silently dropping the checkpoint, since a caller relying on
/// checkpoint/resume needs to know if it didn't happen.
fn result_to_pyobject(py: Python<'_>, sim: Simulation, result: SimResult, n_agents: u32) -> PyResult<PyObject> {
    let checkpoint = sim.to_checkpoint_bytes().map_err(PyValueError::new_err)?;
    let out = PyDict::new_bound(py);
    out.set_item("trades", trades_dict(py, &result.trades)?)?;
    out.set_item("order_events", order_events_dict(py, &result.order_events)?)?;
    out.set_item("snapshots", snapshots_dict(py, &result.snapshots)?)?;
    out.set_item("agent_states", agent_states_dict(py, &result.agent_states)?)?;
    out.set_item("checkpoint", PyBytes::new_bound(py, &checkpoint))?;
    let meta = PyDict::new_bound(py);
    meta.set_item("actual_session_seconds", result.actual_session_seconds)?;
    meta.set_item("liquidation_events", result.liquidation_events)?;
    meta.set_item("bankruptcies", result.bankruptcies)?;
    meta.set_item("n_agents", n_agents)?;
    meta.set_item("n_trades", result.trades.len())?;
    meta.set_item("n_order_events", result.order_events.len())?;
    meta.set_item("cancelled", result.cancelled)?;
    out.set_item("meta", meta)?;
    Ok(out.into())
}

/// Run one simulation from a TOML scenario config string. Returns a dict with keys
/// `trades`, `order_events`, `snapshots`, `agent_states` (each `dict[str, list]`,
/// columnar), `meta` (session/run summary) -- see `market_sim.export` for turning
/// this into the parquet files from spec section 7 -- and `checkpoint` (`bytes`):
/// the full internal simulation state at the moment the run stopped, opaque to
/// Python, meant to be saved alongside the result and handed back to
/// `continue_simulation`/`continue_simulation_cancellable` later to keep simulating
/// from exactly this point rather than starting over.
#[pyfunction]
fn run_simulation(py: Python<'_>, config_toml: &str) -> PyResult<PyObject> {
    let cfg = ScenarioConfig::from_toml_str(config_toml).map_err(|e| PyValueError::new_err(format!("config parse error: {e}")))?;
    let n_agents = cfg.total_agents();
    let sim = Simulation::build(cfg).map_err(PyValueError::new_err)?;
    let (sim, result) = py.allow_threads(|| sim.run_cancellable_keep_state(None, None));
    result_to_pyobject(py, sim, result, n_agents)
}

/// A cancellation flag shared between Python threads (not tied to any one
/// simulation run): create one, pass it to `run_simulation_cancellable` running in a
/// background thread, and call `.cancel()` from the main thread at any time -- the
/// run stops at its next periodic check point and still returns whatever it had
/// produced so far (`meta["cancelled"]` is `True`), rather than raising or returning
/// nothing. Checking `.is_cancelled()` does not require the run to have started or
/// finished.
#[pyclass]
struct CancelToken {
    flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[pymethods]
impl CancelToken {
    #[new]
    fn new() -> Self {
        Self { flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)) }
    }

    fn cancel(&self) {
        self.flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn is_cancelled(&self) -> bool {
        self.flag.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// A shared "how far along is this run" cell, independent of any one simulation
/// (like `CancelToken`): create one, pass it to `run_simulation_cancellable`/
/// `continue_simulation_cancellable` running on a background thread, and poll
/// `.current_seconds()` from the main thread at any time -- returns the simulated
/// time (not wall-clock) the run has reached so far, updated from inside the engine
/// at most a few times a second (see `sim_core::sim::Simulation::drain_loop`'s doc
/// comment), not on every event. Meant for a "currently at Xs simulated" progress
/// display on a long run, not fine-grained instrumentation. `.current_seconds()`
/// returns `None` before the run has processed its first periodic checkpoint.
#[pyclass]
struct ProgressHandle {
    time_bits: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

#[pymethods]
impl ProgressHandle {
    #[new]
    fn new() -> Self {
        // f64::NAN's bit pattern is the "no reading yet" sentinel -- current_seconds()
        // maps it back to None rather than showing a misleading 0.0 before the engine
        // has written anything.
        Self { time_bits: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(f64::NAN.to_bits())) }
    }

    fn current_seconds(&self) -> Option<f64> {
        let v = f64::from_bits(self.time_bits.load(std::sync::atomic::Ordering::Relaxed));
        if v.is_nan() {
            None
        } else {
            Some(v)
        }
    }
}

/// Same as `run_simulation`, but accepts a `CancelToken` and checks it periodically
/// while running; if `.cancel()` is called (from another thread -- this function
/// itself blocks the calling thread until the run stops one way or another, same as
/// `run_simulation`) the run stops early and returns its partial result rather than
/// running to completion. Intended to be called from a Python `threading.Thread` so
/// the caller's own thread stays responsive; see `app/dashboard.py`. `progress` is
/// optional -- pass a `ProgressHandle` to be able to poll how far along the run is
/// while it's still going.
#[pyfunction]
#[pyo3(signature = (config_toml, token, progress=None))]
fn run_simulation_cancellable(py: Python<'_>, config_toml: &str, token: &CancelToken, progress: Option<&ProgressHandle>) -> PyResult<PyObject> {
    let cfg = ScenarioConfig::from_toml_str(config_toml).map_err(|e| PyValueError::new_err(format!("config parse error: {e}")))?;
    let n_agents = cfg.total_agents();
    let sim = Simulation::build(cfg).map_err(PyValueError::new_err)?;
    let flag = token.flag.clone();
    let progress_handle = progress.map(|p| p.time_bits.clone());
    let (sim, result) = py.allow_threads(|| sim.run_cancellable_keep_state(Some(flag), progress_handle));
    result_to_pyobject(py, sim, result, n_agents)
}

/// Loads a simulation from a checkpoint produced by a previous `run_simulation`/
/// `run_simulation_cancellable`/`continue_simulation*` call's `checkpoint` bytes, and
/// keeps simulating for `additional_seconds` more simulated time from exactly where
/// it left off -- same order books, every agent's cash/inventory/indicator state,
/// both RNG streams, and pending events, none of it re-seeded. Returns the same dict
/// shape as `run_simulation`, covering only this newly-simulated segment (not the
/// checkpoint's prior history -- concatenate with the earlier result if a single
/// combined view is needed) plus a fresh `checkpoint` for continuing again later.
#[pyfunction]
fn continue_simulation(py: Python<'_>, checkpoint: &[u8], additional_seconds: f64) -> PyResult<PyObject> {
    let sim = Simulation::from_checkpoint_bytes(checkpoint).map_err(PyValueError::new_err)?;
    let n_agents = sim.agents.len() as u32;
    let (sim, result) = py.allow_threads(|| sim.resume_keep_state(additional_seconds, None, None));
    result_to_pyobject(py, sim, result, n_agents)
}

/// Same as `continue_simulation`, but accepts a `CancelToken` (see
/// `run_simulation_cancellable`) so a long continuation can also be stopped early
/// from another thread, and an optional `ProgressHandle`.
#[pyfunction]
#[pyo3(signature = (checkpoint, additional_seconds, token, progress=None))]
fn continue_simulation_cancellable(py: Python<'_>, checkpoint: &[u8], additional_seconds: f64, token: &CancelToken, progress: Option<&ProgressHandle>) -> PyResult<PyObject> {
    let sim = Simulation::from_checkpoint_bytes(checkpoint).map_err(PyValueError::new_err)?;
    let n_agents = sim.agents.len() as u32;
    let flag = token.flag.clone();
    let progress_handle = progress.map(|p| p.time_bits.clone());
    let (sim, result) = py.allow_threads(|| sim.resume_keep_state(additional_seconds, Some(flag), progress_handle));
    result_to_pyobject(py, sim, result, n_agents)
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
    m.add_function(wrap_pyfunction!(run_simulation_cancellable, m)?)?;
    m.add_function(wrap_pyfunction!(continue_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(continue_simulation_cancellable, m)?)?;
    m.add_function(wrap_pyfunction!(validate_config, m)?)?;
    m.add_class::<CancelToken>()?;
    m.add_class::<ProgressHandle>()?;
    Ok(())
}
