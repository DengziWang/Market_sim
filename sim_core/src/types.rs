//! Fundamental identifiers and value types shared across the engine.

pub type AssetId = u16;
pub type VenueId = u16;
pub type AgentId = u32;
pub type OrderId = u64;

/// Simulated time, in seconds since session start. f64 is precise enough for
/// sessions of a few hours at sub-millisecond event resolution and is far
/// simpler to work with across the codebase than a fixed-point duration type.
pub type SimTime = f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    pub fn opposite(self) -> Side {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }
    pub fn sign(self) -> f64 {
        match self {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Limit,
    Market,
}

/// Time-in-force. PostOnly is modeled as a flag on Limit orders (see `Order::post_only`)
/// rather than a TIF variant, since it composes with GTC/IOC semantics for *placement*
/// (a post-only order that would cross is rejected/cancelled rather than filled).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    /// Good-Til-Cancelled: rests on the book until filled or explicitly cancelled.
    Gtc,
    /// Immediate-Or-Cancel: fills what it can immediately, cancels the remainder.
    Ioc,
    /// Fill-Or-Kill: fills completely and immediately, or not at all.
    Fok,
}

/// Price is stored as an integer number of ticks internally (deterministic ordering,
/// no float-comparison edge cases in the book), but every public API speaks in f64
/// so config/bot code can work in ordinary decimal prices.
pub type PriceTicks = i64;

#[derive(Debug, Clone, Copy)]
pub struct TickSize(pub f64);

impl TickSize {
    #[inline]
    pub fn to_ticks(&self, price: f64) -> PriceTicks {
        (price / self.0).round() as PriceTicks
    }
    #[inline]
    pub fn to_price(&self, ticks: PriceTicks) -> f64 {
        ticks as f64 * self.0
    }
}
