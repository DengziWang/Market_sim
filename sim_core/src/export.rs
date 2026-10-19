//! Flat row types collected during a run and handed to the PyO3 layer, which turns
//! them into columnar Python data (`dict[str, list]`) for `pandas`/`pyarrow` to write
//! out as the parquet files described in spec section 7. Kept independent of PyO3 so
//! `sim_core` stays a plain, `cargo test`-able Rust crate.

use crate::types::{AgentId, AssetId, OrderId, SimTime, VenueId};

#[derive(Debug, Clone)]
pub struct TradeRow {
    pub time: SimTime,
    pub asset_id: AssetId,
    pub venue_id: VenueId,
    pub price: f64,
    pub qty: f64,
    pub taker_side: &'static str,
    pub maker_agent_id: AgentId,
    pub taker_agent_id: AgentId,
    pub maker_type: &'static str,
    pub maker_variant: &'static str,
    pub taker_type: &'static str,
    pub taker_variant: &'static str,
    pub maker_fee_paid: f64,
    pub taker_fee_paid: f64,
}

#[derive(Debug, Clone)]
pub struct OrderEventRow {
    pub time: SimTime,
    pub order_id: OrderId,
    pub agent_id: AgentId,
    pub asset_id: AssetId,
    pub venue_id: VenueId,
    pub kind: &'static str,
    pub side: &'static str,
    pub price: Option<f64>,
    pub qty: f64,
}

#[derive(Debug, Clone)]
pub struct SnapshotRow {
    pub time: SimTime,
    pub asset_id: AssetId,
    pub venue_id: VenueId,
    pub side: &'static str,
    pub level: u32,
    pub price: f64,
    pub qty: f64,
}

#[derive(Debug, Clone)]
pub struct AgentStateRow {
    pub agent_id: AgentId,
    pub bot_type: &'static str,
    pub variant: &'static str,
    pub asset_id: AssetId,
    pub inventory: f64,
    pub cash: f64,
    pub equity: f64,
    pub realized_pnl: f64,
    pub cumulative_volume: f64,
    pub bankrupt: bool,
    pub has_margin_position: bool,
    pub margin_leverage: Option<f64>,
    /// Fixed spawn-time baseline the universal 80%-loss bankruptcy sweep compares
    /// `equity` against (bankrupt whenever equity <= 0.2 * initial_net_worth) --
    /// exported so that check is independently verifiable from this file alone,
    /// not just trusted.
    pub initial_net_worth: f64,
}

#[derive(Default)]
pub struct SimResult {
    pub trades: Vec<TradeRow>,
    pub order_events: Vec<OrderEventRow>,
    /// Incremented on every order event regardless of `export.write_order_events`
    /// -- unlike `order_events` itself (one `OrderEventRow` + two heap-allocated
    /// strings per event), this is one `u64` add, so the true count stays
    /// available for `n_order_events` metadata even when collecting the full
    /// per-event rows is disabled to bound memory on an event-heavy run.
    pub n_order_events_total: u64,
    pub snapshots: Vec<SnapshotRow>,
    pub agent_states: Vec<AgentStateRow>,
    pub liquidation_events: u64,
    pub bankruptcies: u64,
    pub actual_session_seconds: SimTime,
    /// True if the run was interrupted (see `Simulation::run_cancellable`) before it
    /// reached its natural end -- every field above still reflects whatever was
    /// produced up to that point, not a failed/empty run.
    pub cancelled: bool,
}
