//! Bot decision logic. Every bot type lives in its own submodule and exposes one
//! `State` struct (immutable params parsed from config + small mutable indicator
//! state) plus a `step(...)` function; `dispatch::step` below is the single place
//! that matches on `BotStrategy` and calls into them, so `sim.rs` only ever talks to
//! this module.

pub mod arbitrage;
pub mod dca;
pub mod grid;
pub mod human;
pub mod liquidation_trader;
pub mod market_maker;
pub mod mean_reversion;
pub mod mev;
pub mod momentum;
pub mod noise;
pub mod wash_trading;

use std::collections::VecDeque;

use crate::agent::{AgentCommon, NewOrderRequest, StepResult};
use crate::market::Market;
use crate::types::{OrderId, SimTime, VenueId};

/// Small fixed-capacity rolling buffer of samples, used by every indicator-based bot
/// (moving averages, RSI, Bollinger, Donchian, z-score, VPIN buckets, ...).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RollingWindow {
    pub buf: VecDeque<f64>,
    pub capacity: usize,
}
impl RollingWindow {
    pub fn new(capacity: usize) -> Self {
        Self { buf: VecDeque::with_capacity(capacity), capacity: capacity.max(1) }
    }
    pub fn push(&mut self, v: f64) {
        if self.buf.len() == self.capacity {
            self.buf.pop_front();
        }
        self.buf.push_back(v);
    }
    pub fn mean(&self) -> Option<f64> {
        if self.buf.is_empty() {
            None
        } else {
            Some(self.buf.iter().sum::<f64>() / self.buf.len() as f64)
        }
    }
    pub fn std(&self) -> Option<f64> {
        let m = self.mean()?;
        if self.buf.len() < 2 {
            return Some(0.0);
        }
        let var = self.buf.iter().map(|x| (x - m).powi(2)).sum::<f64>() / self.buf.len() as f64;
        Some(var.sqrt())
    }
    pub fn min(&self) -> Option<f64> {
        self.buf.iter().cloned().fold(None, |acc, x| Some(acc.map_or(x, |a: f64| a.min(x))))
    }
    pub fn max(&self) -> Option<f64> {
        self.buf.iter().cloned().fold(None, |acc, x| Some(acc.map_or(x, |a: f64| a.max(x))))
    }
    pub fn is_full(&self) -> bool {
        self.buf.len() >= self.capacity
    }
    pub fn len(&self) -> usize {
        self.buf.len()
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub enum BotStrategy {
    MarketMaker(market_maker::State),
    Arbitrage(arbitrage::State),
    Momentum(momentum::State),
    MeanReversion(mean_reversion::State),
    Grid(grid::State),
    Dca(dca::State),
    LiquidationTrader(liquidation_trader::State),
    Mev(mev::State),
    WashTrading(wash_trading::State),
    Noise(noise::State),
    Human(human::State),
}

impl BotStrategy {
    pub fn type_variant_names(&self) -> (&'static str, &'static str) {
        match self {
            BotStrategy::MarketMaker(s) => ("market_maker", s.variant_name()),
            BotStrategy::Arbitrage(s) => ("arbitrage", s.variant_name()),
            BotStrategy::Momentum(s) => ("momentum", s.variant_name()),
            BotStrategy::MeanReversion(s) => ("mean_reversion", s.variant_name()),
            BotStrategy::Grid(s) => ("grid", s.variant_name()),
            BotStrategy::Dca(s) => ("dca", s.variant_name()),
            BotStrategy::LiquidationTrader(s) => ("liquidation_trader", s.variant_name()),
            BotStrategy::Mev(s) => ("mev", s.variant_name()),
            BotStrategy::WashTrading(s) => ("wash_trading", s.variant_name()),
            BotStrategy::Noise(s) => ("noise", s.variant_name()),
            BotStrategy::Human(s) => ("human", s.variant_name()),
        }
    }
}

pub struct StepCtx<'a> {
    pub market: &'a Market,
    pub venue: VenueId,
    pub now: SimTime,
}

pub fn step(agent: &mut AgentCommon, strat: &mut BotStrategy, ctx: &StepCtx) -> StepResult {
    if agent.bankrupt {
        return StepResult::sleep_only(None);
    }
    match strat {
        BotStrategy::MarketMaker(s) => market_maker::step(agent, s, ctx),
        BotStrategy::Arbitrage(s) => arbitrage::step(agent, s, ctx),
        BotStrategy::Momentum(s) => momentum::step(agent, s, ctx),
        BotStrategy::MeanReversion(s) => mean_reversion::step(agent, s, ctx),
        BotStrategy::Grid(s) => grid::step(agent, s, ctx),
        BotStrategy::Dca(s) => dca::step(agent, s, ctx),
        BotStrategy::LiquidationTrader(s) => liquidation_trader::step(agent, s, ctx),
        BotStrategy::Mev(s) => mev::step(agent, s, ctx),
        BotStrategy::WashTrading(s) => wash_trading::step(agent, s, ctx),
        BotStrategy::Noise(s) => noise::step(agent, s, ctx),
        BotStrategy::Human(s) => human::step(agent, s, ctx),
    }
}

/// The asset a strategy instance primarily trades, used by the generic leveraged
/// side-bet piggyback in `sim.rs` (spec 5.7: leverage is independent of bot_type).
/// Multi-asset strategies (arbitrage, market_maker) leverage their first configured leg/asset.
pub fn primary_asset(strat: &BotStrategy) -> crate::types::AssetId {
    match strat {
        BotStrategy::MarketMaker(s) => s.assets[0].asset_id,
        BotStrategy::Arbitrage(s) => s.asset_a,
        BotStrategy::Momentum(s) => s.asset_id,
        BotStrategy::MeanReversion(s) => s.asset_id,
        BotStrategy::Grid(s) => s.asset_id,
        BotStrategy::Dca(s) => s.asset_id,
        BotStrategy::LiquidationTrader(s) => s.asset_id,
        BotStrategy::Mev(s) => s.asset_id,
        BotStrategy::WashTrading(s) => s.asset_id,
        BotStrategy::Noise(s) => s.asset_id,
        BotStrategy::Human(s) => s.asset_id,
    }
}

/// Feedback hook: called once per `Action::Submit` after `sim.rs` gets an order id
/// back from the market, so strategies that requote (cancel their own prior resting
/// orders) know what to cancel next time. Most bot types don't track ids (they only
/// ever place IOC/market orders, or don't need to cancel selectively) and no-op here.
pub fn on_order_id(strat: &mut BotStrategy, req: &NewOrderRequest, id: OrderId) {
    match strat {
        BotStrategy::MarketMaker(s) => market_maker::on_order_id(s, req, id),
        BotStrategy::Grid(s) => grid::on_order_id(s, req, id),
        BotStrategy::Human(s) => human::on_order_id(s, req, id),
        _ => {}
    }
}
