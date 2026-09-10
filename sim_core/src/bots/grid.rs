//! Grid bots: 4.5 in the spec.
//!
//! Simplification: the whole ladder is torn down and rebuilt every `wake_interval`
//! (like the market-maker requote pattern) rather than replacing only the individual
//! level that just filled. This still produces genuine grid-trading behavior (a
//! resting ladder of buy/sell levels around price that adapts to price/volatility)
//! at a fraction of the per-level order-tracking complexity.

use super::{RollingWindow, StepCtx};
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderId, OrderType, Side, SimTime, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    Static,
    Geometric,
    DynamicAtr,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub variant: Variant,
    pub asset_id: AssetId,
    pub wake_interval: SimTime,
    pub order_qty: f64,
    pub n_levels: u32,
    pub spacing_abs: f64,
    pub spacing_pct: f64,
    pub atr_window: RollingWindow,
    pub atr_multiplier: f64,
    pub last_price: Option<f64>,
    pub active_order_ids: Vec<OrderId>,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::Static => "static",
            Variant::Geometric => "geometric",
            Variant::DynamicAtr => "dynamic_atr",
        }
    }

    /// `initial_price`: converts the dollar-notional `order_notional` param into
    /// asset units, so a single default provides comparable *dollar* depth across
    /// assets of very different price scales -- see market_maker.rs's from_params
    /// doc comment for why a fixed unit-quantity default doesn't.
    pub fn from_params(variant: Variant, asset_id: AssetId, initial_price: f64, p: &toml::Value) -> Self {
        let order_qty = params::get_f64(p, "order_notional", 1500.0) / initial_price.max(1e-9);
        // Static grid spacing has the same scale problem as a unit-denominated order
        // size: a fixed absolute price offset (e.g. 5.0) is negligible for a $60k
        // asset but enormous relative to a $0.60 one, leaving huge, effectively-empty
        // gaps between levels. Default it as a fraction of the reference price
        // instead (still overridable directly via `spacing_abs` for a scenario that
        // genuinely wants a fixed absolute grid).
        let spacing_abs = params::get_f64(p, "spacing_abs", initial_price.max(1e-9) * 0.004);
        Self {
            variant,
            asset_id,
            wake_interval: params::get_f64(p, "wake_interval_secs", 10.0),
            order_qty,
            n_levels: params::get_u32(p, "n_levels", 5),
            spacing_abs,
            spacing_pct: params::get_f64(p, "spacing_pct", 0.004),
            atr_window: RollingWindow::new(params::get_u32(p, "atr_window", 14) as usize),
            atr_multiplier: params::get_f64(p, "atr_multiplier", 0.5),
            last_price: None,
            active_order_ids: Vec::new(),
        }
    }
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    let mut actions: Vec<Action> = s.active_order_ids.drain(..).map(Action::Cancel).collect();

    let Some(mid) = ctx.market.mid_price(ctx.venue, s.asset_id) else {
        return StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) };
    };
    if let Some(prev) = s.last_price {
        s.atr_window.push((mid - prev).abs());
    }
    s.last_price = Some(mid);

    let spacing = match s.variant {
        Variant::Static => s.spacing_abs,
        Variant::Geometric => mid * s.spacing_pct,
        Variant::DynamicAtr => (s.atr_window.mean().unwrap_or(mid * s.spacing_pct) * s.atr_multiplier).max(mid * 0.0005),
    };
    if spacing <= 0.0 {
        return StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) };
    }

    let inventory = agent.inventory_of(s.asset_id);
    let max_inventory = s.order_qty * s.n_levels as f64;

    for lvl in 1..=s.n_levels {
        let offset = match s.variant {
            Variant::Geometric => mid * (1.0 - (1.0 - s.spacing_pct).powi(lvl as i32)),
            _ => spacing * lvl as f64,
        };
        let buy_price = mid - offset;
        let sell_price = mid + offset;
        if buy_price > 0.0 && inventory < max_inventory && agent.cash > buy_price * s.order_qty {
            actions.push(Action::Submit(NewOrderRequest {
                asset_id: s.asset_id,
                side: Side::Buy,
                order_type: OrderType::Limit,
                tif: TimeInForce::Gtc,
                post_only: true,
                price: Some(buy_price),
                qty: s.order_qty,
            }));
        }
        if inventory > -max_inventory {
            actions.push(Action::Submit(NewOrderRequest {
                asset_id: s.asset_id,
                side: Side::Sell,
                order_type: OrderType::Limit,
                tif: TimeInForce::Gtc,
                post_only: true,
                price: Some(sell_price),
                qty: s.order_qty,
            }));
        }
    }

    StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) }
}

pub fn on_order_id(s: &mut State, _req: &NewOrderRequest, id: OrderId) {
    s.active_order_ids.push(id);
}
