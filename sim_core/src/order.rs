use crate::types::{AgentId, AssetId, OrderId, OrderType, PriceTicks, Side, SimTime, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderStatus {
    Open,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

#[derive(Debug, Clone)]
pub struct Order {
    pub id: OrderId,
    pub agent_id: AgentId,
    pub asset_id: AssetId,
    pub side: Side,
    pub order_type: OrderType,
    pub tif: TimeInForce,
    pub post_only: bool,
    /// Limit price in ticks. Ignored (None) for Market orders.
    pub price_ticks: Option<PriceTicks>,
    pub qty: f64,
    pub qty_filled: f64,
    pub status: OrderStatus,
    pub submitted_at: SimTime,
    /// Monotonically increasing sequence number, used as the time-priority tiebreaker
    /// within a price level (submission order at matching-engine ingestion time).
    pub seq: u64,
}

impl Order {
    pub fn remaining(&self) -> f64 {
        (self.qty - self.qty_filled).max(0.0)
    }

    pub fn is_done(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Filled | OrderStatus::Cancelled | OrderStatus::Rejected
        ) || self.remaining() <= 1e-12
    }
}

/// A single match between a resting (maker) order and an incoming (taker) order.
#[derive(Debug, Clone)]
pub struct Fill {
    pub asset_id: AssetId,
    pub price_ticks: PriceTicks,
    pub qty: f64,
    pub maker_order_id: OrderId,
    pub maker_agent_id: AgentId,
    pub taker_order_id: OrderId,
    pub taker_agent_id: AgentId,
    pub taker_side: Side,
    pub time: SimTime,
}

/// Record of an order lifecycle event, kept for `order_events.parquet` (cancel-to-trade
/// ratio, quote-stuffing analysis, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderEventKind {
    Submit,
    Cancel,
    Fill,
    PartialFill,
    Reject,
    Expire,
}

#[derive(Debug, Clone)]
pub struct OrderEvent {
    pub time: SimTime,
    pub order_id: OrderId,
    pub agent_id: AgentId,
    pub asset_id: AssetId,
    pub kind: OrderEventKind,
    pub side: Side,
    pub price_ticks: Option<PriceTicks>,
    pub qty: f64,
}
