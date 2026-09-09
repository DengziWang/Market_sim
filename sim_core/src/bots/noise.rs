//! Noise/retail trader baseline: 5.10 in the spec. Not a "bot" strategically, but the
//! control-group population every composition should be compared against.

use rand::Rng;

use super::{RollingWindow, StepCtx};
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderType, Side, SimTime, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    ZeroIntelligence,
    Herding,
}

pub struct State {
    pub variant: Variant,
    pub asset_id: AssetId,
    pub wake_interval: SimTime,
    pub price_band_frac: f64,
    pub qty_range: (f64, f64),
    pub herd_strength: f64,
    /// Position cap (both directions -- inventory is kept in [0, max_position], sells
    /// are separately capped to what's actually held). Without *some* bound, a purely
    /// cash-limited buy side is effectively unconstrained for a cheap asset, and a
    /// herding population chasing an uptrend can compound an unbounded one-directional
    /// position exactly as an uncapped sell side did for a downtrend (see git history:
    /// this is the buy-side twin of the earlier short-selling bug).
    pub max_position: f64,
    pub recent_prices: RollingWindow,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::ZeroIntelligence => "zero_intelligence",
            Variant::Herding => "herding",
        }
    }

    pub fn from_params(variant: Variant, asset_id: AssetId, p: &toml::Value) -> Self {
        Self {
            variant,
            asset_id,
            wake_interval: params::get_f64(p, "wake_interval_secs", 8.0),
            price_band_frac: params::get_f64(p, "price_band_frac", 0.01),
            qty_range: params::get_range(p, "qty_range", (0.01, 0.2)),
            herd_strength: params::get_f64(p, "herd_strength", 0.7),
            max_position: params::get_f64(p, "max_position", 20.0),
            recent_prices: RollingWindow::new(5),
        }
    }
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    let mut actions = Vec::new();
    let Some(mid) = ctx.market.mid_price(ctx.venue, s.asset_id) else {
        return StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) };
    };

    let side = match s.variant {
        Variant::ZeroIntelligence => {
            if agent.rng.gen_bool(0.5) { Side::Buy } else { Side::Sell }
        }
        Variant::Herding => {
            s.recent_prices.push(mid);
            let trend = if s.recent_prices.len() >= 2 {
                let first = *s.recent_prices.buf.front().unwrap();
                let last = *s.recent_prices.buf.back().unwrap();
                last - first
            } else {
                0.0
            };
            // Imperfect proxy for "recent order flow": this architecture gives bots
            // access to market-observable state (prices/depth), not a private trade
            // tape, so recent price direction stands in for recent order-flow sign.
            if trend.abs() > 1e-12 && agent.rng.gen_bool(s.herd_strength) {
                if trend > 0.0 { Side::Buy } else { Side::Sell }
            } else if agent.rng.gen_bool(0.5) {
                Side::Buy
            } else {
                Side::Sell
            }
        }
    };

    let offset = agent.rng.gen_range(-s.price_band_frac..s.price_band_frac);
    let price = mid * (1.0 + offset);
    let mut qty = agent.rng.gen_range(s.qty_range.0..=s.qty_range.1.max(s.qty_range.0 + 1e-6));

    // Retail/noise traders don't short spot assets: a sell is capped at (and skipped
    // if there's none of) what the agent actually holds. Without this, a herding
    // trader population facing a downtrend has literally unlimited, uncapitalized
    // sell pressure to add to it -- unlike every capital-/position-bounded bot type,
    // this drove an unbounded price collapse toward zero in testing (see git history).
    let inventory = agent.inventory_of(s.asset_id);
    if side == Side::Sell {
        qty = qty.min(inventory.max(0.0));
    } else {
        qty = qty.min((s.max_position - inventory).max(0.0));
    }

    if price > 0.0 && qty > 1e-9 {
        let affordable = match side {
            Side::Buy => agent.cash >= price * qty,
            Side::Sell => true, // already capped to held inventory above
        };
        if affordable {
            actions.push(Action::Submit(NewOrderRequest {
                asset_id: s.asset_id,
                side,
                order_type: OrderType::Limit,
                tif: TimeInForce::Gtc,
                post_only: false,
                price: Some(price),
                qty,
            }));
        }
    }

    let jitter = agent.rng.gen_range(0.5..1.5);
    StepResult { actions, next_wake: Some(ctx.now + s.wake_interval * jitter) }
}
