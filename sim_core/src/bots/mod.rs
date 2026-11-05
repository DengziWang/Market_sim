//! Bot decision logic. Every bot type lives in its own submodule and exposes one
//! `State` struct (immutable params parsed from config + small mutable indicator
//! state) plus a `step(...)` function; `dispatch::step` below is the single place
//! that matches on `BotStrategy` and calls into them, so `sim.rs` only ever talks to
//! this module.

pub mod arbitrage;
pub mod dca;
pub mod grid;
pub mod human;
pub mod institutional_seller;
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
use crate::params;
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

/// Self risk-management for the "structural" strategies (arbitrage, mev): a real
/// arbitrageur or MEV searcher cuts size, or stops trading altogether, after
/// sustained losses rather than keep betting the same size into capital that's
/// disappearing. Without this, these bots kept firing full-size trades right up to
/// the engine's forced-liquidation floor (`run_net_worth_bankruptcy_sweep`, 20% of
/// `initial_net_worth`) -- which is what let several variants post strongly
/// positive *average* growth alongside implausibly high bankruptcy rates
/// (2026-10-22: stat_arb_pairs +219%/49.2% bankrupt, frontrun +105%/34.7%,
/// sandwich +102%/37.1% on the full 5,000-agent sweep). A bot that actually
/// de-risks on its own drawdown should rarely need the engine's floor to stop it.
///
/// Trips once equity falls to `derisk_threshold_frac` of the agent's starting net
/// worth: halves its own position-size multiplier (floor 1/8th) and pauses opening
/// *new* positions for `cooldown_secs`, while still allowing the caller to flatten
/// whatever is already open. This is sticky, not a one-shot warning -- getting cut
/// down to a smaller size is itself the lesson, mirroring a risk desk that reduces
/// a strategy's allocation after it blows through a loss limit rather than
/// restoring it to full size the moment the drawdown technically clears.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RiskGovernor {
    pub derisk_threshold_frac: f64,
    pub cooldown_secs: SimTime,
    pub size_multiplier: f64,
    pub paused_until: Option<SimTime>,
}

impl RiskGovernor {
    pub fn from_params(p: &toml::Value) -> Self {
        Self {
            derisk_threshold_frac: params::get_f64(p, "derisk_threshold_frac", 0.5),
            cooldown_secs: params::get_f64(p, "derisk_cooldown_secs", 1800.0),
            size_multiplier: 1.0,
            paused_until: None,
        }
    }

    /// Returns whether new positions may be opened this wake. The caller is
    /// responsible for flattening any already-open position when this returns
    /// `false` -- the governor only gates new entries, it never silently abandons
    /// existing exposure.
    pub fn poll(&mut self, equity: f64, initial_net_worth: f64, now: SimTime) -> bool {
        if let Some(until) = self.paused_until {
            if now < until {
                return false;
            }
            self.paused_until = None;
        }
        if initial_net_worth > 1e-9 && equity <= self.derisk_threshold_frac * initial_net_worth {
            self.size_multiplier = (self.size_multiplier * 0.5).max(0.125);
            self.paused_until = Some(now + self.cooldown_secs);
            return false;
        }
        true
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
    Mev(mev::State),
    WashTrading(wash_trading::State),
    Noise(noise::State),
    Human(human::State),
    InstitutionalSeller(institutional_seller::State),
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
            BotStrategy::Mev(s) => ("mev", s.variant_name()),
            BotStrategy::WashTrading(s) => ("wash_trading", s.variant_name()),
            BotStrategy::Noise(s) => ("noise", s.variant_name()),
            BotStrategy::Human(s) => ("human", s.variant_name()),
            BotStrategy::InstitutionalSeller(s) => ("institutional_seller", s.variant_name()),
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
        BotStrategy::Mev(s) => mev::step(agent, s, ctx),
        BotStrategy::WashTrading(s) => wash_trading::step(agent, s, ctx),
        BotStrategy::Noise(s) => noise::step(agent, s, ctx),
        BotStrategy::Human(s) => human::step(agent, s, ctx),
        BotStrategy::InstitutionalSeller(s) => institutional_seller::step(agent, s, ctx),
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
        BotStrategy::Mev(s) => s.asset_id,
        BotStrategy::WashTrading(s) => s.asset_id,
        BotStrategy::Noise(s) => s.asset_id,
        BotStrategy::Human(s) => s.asset_id,
        BotStrategy::InstitutionalSeller(s) => s.asset_id,
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
