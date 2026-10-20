//! Agent state shared by every bot type/variant, and the small vocabulary
//! (`Action`, `StepResult`) bots use to talk to the engine.

use rand_chacha::ChaCha8Rng;

use crate::types::{AgentId, AssetId, FxHashMap, OrderId, OrderType, PriceTicks, Side, SimTime, TimeInForce};

/// A leveraged position, held by the configurable sub-population of agents that
/// trade on margin (Phase 1 simplified model: see sim_core::liquidation).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MarginPosition {
    pub asset_id: AssetId,
    pub side: Side,
    pub size: f64,
    pub entry_price: f64,
    pub leverage: f64,
    /// Fraction of the theoretical liquidation distance (1/leverage) actually used,
    /// e.g. 0.8 => at 10x a position liquidates around an ~8% adverse move, not 10%.
    pub maintenance_margin_buffer: f64,
    pub collateral: f64,
    /// How many StaggeredTwap liquidation rounds (see `Sim::run_liquidation_sweep`,
    /// 2026-10-20 "liquidation cascade" fix) have already run against this position
    /// while it stayed liquidatable throughout. A round that closes at most 1/4 of
    /// the remaining size per slice only "tapers off" the way its original design
    /// intended if the adverse price move causing the liquidation stops -- if it
    /// doesn't (the quarter-at-a-time close can't keep up), every slice is a fresh
    /// IOC market order, each one individually bounded by `max_price_move_frac` but
    /// starting from wherever the previous slice left off, so a position that
    /// never recovers emits an unbounded sequence of circuit-breaker-bounded hops
    /// that *compound*: ~15 sequential 15%-capped slices is enough to move price
    /// 8x in under a second, which is exactly what one BTC leg of the
    /// `sweep_175k_btceth_mm` run did (130k -> 1,088k, wiping out 97.6% of that
    /// cell's market_makers). Reset to 0 whenever the position is briefly not
    /// liquidatable; once it reaches 1 (one full round already tried and the
    /// position is still liquidatable), `run_liquidation_sweep` escalates straight
    /// to a single full close instead of scheduling another round -- trading one
    /// circuit-breaker-bounded price step for what would otherwise be an unbounded
    /// compounding sequence of them.
    pub staggered_round: u32,
}

impl MarginPosition {
    /// Liquidation trigger price against the (EMA) mark price.
    pub fn liquidation_price(&self) -> f64 {
        let raw_distance = 1.0 / self.leverage;
        let distance = raw_distance * self.maintenance_margin_buffer;
        match self.side {
            Side::Buy => self.entry_price * (1.0 - distance),
            Side::Sell => self.entry_price * (1.0 + distance),
        }
    }

    pub fn is_liquidatable(&self, mark_price: f64) -> bool {
        match self.side {
            Side::Buy => mark_price <= self.liquidation_price(),
            Side::Sell => mark_price >= self.liquidation_price(),
        }
    }

    pub fn unrealized_pnl(&self, mark_price: f64) -> f64 {
        match self.side {
            Side::Buy => (mark_price - self.entry_price) * self.size,
            Side::Sell => (self.entry_price - mark_price) * self.size,
        }
    }
}

/// Present only for the configurable fraction of agents (of any bot type -- see spec
/// 5.7) that additionally carry a leveraged directional side-bet alongside their
/// primary strategy.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct LeverageCfg {
    pub leverage_range: (f64, f64),
    pub position_frac_of_cash: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LiquidationExecStyle {
    Full,
    /// Staggered/partial execution over several slices (TWAP-like), to reduce impact.
    StaggeredTwap { slices: u32 },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentCommon {
    pub id: AgentId,
    pub cash: f64,
    pub inventory: FxHashMap<AssetId, f64>,
    pub bankrupt: bool,
    /// Set only by `Simulation::graft_population` (2026-10-10, sim.rs) on an
    /// agent's whole prior population when a second population is grafted on
    /// top of it mid-run (e.g. a zero-intelligence warmup phase handed off to
    /// the "real" bot population) -- stops this agent from ever being
    /// stepped again (same early-return shape as `bankrupt`), but leaves its
    /// cash/inventory/resting orders exactly as they were, unlike bankruptcy
    /// which force-liquidates. Existing resting orders stay live in the book
    /// as real starting liquidity for whichever population replaces this
    /// one; this agent just never acts again.
    pub dormant: bool,
    pub margin: Option<MarginPosition>,
    pub liquidation_style: LiquidationExecStyle,
    /// Set only for agents in a leveraged sub-population; drives the generic
    /// "maybe open a leveraged side-bet" piggyback check in `sim.rs`.
    pub leverage_cfg: Option<LeverageCfg>,
    /// Per-agent-class network delay applied to this agent's outgoing orders.
    pub latency: SimTime,
    pub maker_fee: f64,
    pub taker_fee: f64,
    /// Orders this agent believes are currently resting, tracked by the agent itself
    /// (the book is the source of truth; agents reconcile via fill/cancel events).
    pub open_orders: Vec<OrderId>,
    pub rng: ChaCha8Rng,
    /// Realized PnL accumulator (for agent_state export).
    pub realized_pnl: f64,
    /// Cumulative notional traded (maker + taker fills), for concentration (HHI) and
    /// volume-authenticity analysis.
    pub cumulative_volume: f64,
    /// Net worth (cash + inventory value, at the asset's own initial price) at the
    /// moment this agent was spawned -- the fixed baseline the universal drawdown
    /// bankruptcy check (`Sim::run_net_worth_bankruptcy_sweep`) measures an 80% loss
    /// against. Set once at construction, never mutated afterward.
    pub initial_net_worth: f64,
}

impl AgentCommon {
    pub fn inventory_of(&self, asset_id: AssetId) -> f64 {
        *self.inventory.get(&asset_id).unwrap_or(&0.0)
    }

    pub fn add_inventory(&mut self, asset_id: AssetId, delta: f64) {
        *self.inventory.entry(asset_id).or_insert(0.0) += delta;
    }

    pub fn equity(&self, mark_prices: &FxHashMap<AssetId, f64>) -> f64 {
        let inv_value: f64 = self
            .inventory
            .iter()
            .map(|(a, q)| q * mark_prices.get(a).copied().unwrap_or(0.0))
            .sum();
        let margin_value = self
            .margin
            .as_ref()
            .map(|m| {
                let mp = mark_prices.get(&m.asset_id).copied().unwrap_or(m.entry_price);
                m.collateral + m.unrealized_pnl(mp)
            })
            .unwrap_or(0.0);
        self.cash + inv_value + margin_value
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NewOrderRequest {
    pub asset_id: AssetId,
    pub side: Side,
    pub order_type: OrderType,
    pub tif: TimeInForce,
    pub post_only: bool,
    pub price: Option<f64>,
    pub qty: f64,
}

#[derive(Debug, Clone)]
pub enum Action {
    Submit(NewOrderRequest),
    /// The asset the order being cancelled belongs to -- every bot that cancels its
    /// own resting orders already knows this (each is single-asset per resting
    /// order), so carrying it here avoids `Simulation` having to search every
    /// asset's book for the order id on every single cancel (previously
    /// `order_asset_hint`, an O(n_assets) scan -- see git history).
    Cancel(OrderId, AssetId),
    /// A margin-settlement order sourced from a request this *specific* action
    /// already decided was a leveraged entry (2026-10-10) -- unlike the
    /// existing bot-type-wide `is_leveraged_bot` routing in `sim.rs` (every
    /// order from a `liquidation_trader` is a margin order), this lets a bot
    /// that mixes ordinary spot orders with occasional leveraged ones in the
    /// same `step()` call (see `human.rs`'s cash-constrained-conviction
    /// trigger) mark just the one action that needs the margin settlement
    /// path, skipping normal latency/mempool staging the same way every other
    /// margin order does.
    SubmitMargin(NewOrderRequest),
}

pub struct StepResult {
    pub actions: Vec<Action>,
    /// Absolute simulated time this agent wants to be woken next; `None` means it
    /// goes dormant (e.g. bankrupt, or a one-shot stress-test trader).
    pub next_wake: Option<SimTime>,
}

impl StepResult {
    pub fn sleep_only(next_wake: Option<SimTime>) -> Self {
        Self {
            actions: Vec::new(),
            next_wake,
        }
    }
}

/// Suppresses an unused-import warning until PriceTicks is used directly by a bot
/// (most bots work in plain f64 prices via NewOrderRequest).
#[allow(dead_code)]
type _Unused = PriceTicks;
