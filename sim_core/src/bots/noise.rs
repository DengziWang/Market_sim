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
    let qty = agent.rng.gen_range(s.qty_range.0..=s.qty_range.1.max(s.qty_range.0 + 1e-6));

    if price > 0.0 && qty > 0.0 {
        let affordable = match side {
            Side::Buy => agent.cash >= price * qty,
            Side::Sell => true, // spot short-selling not modeled for noise traders; engine still accepts it (inventory can go negative) but keep it rare by preferring buys when flat
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
