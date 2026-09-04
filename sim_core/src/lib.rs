//! `sim_core`: event-driven, multi-asset limit-order-book market simulation engine.
//! See `../../README.md` for the research goal and architecture; this crate is the
//! performance-critical Rust core wrapped by the `sim_pyo3` PyO3 extension.

pub mod agent;
pub mod bots;
pub mod config;
pub mod event;
pub mod export;
pub mod market;
pub mod order;
pub mod orderbook;
pub mod params;
pub mod rng;
pub mod sim;
pub mod types;

pub use config::ScenarioConfig;
pub use export::SimResult;
pub use sim::Simulation;
