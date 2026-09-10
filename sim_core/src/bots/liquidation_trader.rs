//! Leveraged directional traders: 5.7 in the spec. This module owns (a) the entry
//! logic for the population of agents that hold a leveraged margin position, and
//! (b) the shared partial-liquidation-quantity formula used by the system-wide
//! liquidation sweep in `sim.rs` (which must apply to *any* agent carrying a margin
//! position, not just this bot type -- see `AgentCommon::margin` /
//! `leveraged_fraction` in config).
//!
//! The two "variants" here are liquidation *execution* styles (immediate full vs.
//! staggered/partial TWAP), matching spec 5.7's explicit ask to compare "soft
//! correction vs. hard cascade" -- the entry-side logic (open a random-side leveraged
//! bet) is intentionally simple since it exists to *produce* liquidation events for
//! the cascade dynamics to be studied, not to be a sophisticated directional strategy.

use rand::Rng;

use super::StepCtx;
use crate::agent::{Action, AgentCommon, LiquidationExecStyle, MarginPosition, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderType, Side, SimTime, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    Full,
    StaggeredTwap,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub variant: Variant,
    pub asset_id: AssetId,
    pub wake_interval: SimTime,
    pub leverage_range: (f64, f64),
    pub position_frac_of_cash: f64,
    pub maintenance_margin_buffer: f64,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::Full => "full",
            Variant::StaggeredTwap => "staggered_twap",
        }
    }

    pub fn from_params(variant: Variant, asset_id: AssetId, maintenance_margin_buffer: f64, p: &toml::Value) -> Self {
        Self {
            variant,
            asset_id,
            wake_interval: params::get_f64(p, "wake_interval_secs", 180.0),
            leverage_range: params::get_range(p, "leverage_range", (2.0, 20.0)),
            position_frac_of_cash: params::get_f64(p, "position_frac_of_cash", 0.6),
            maintenance_margin_buffer,
        }
    }

    pub fn exec_style(&self, staggered_slices: u32) -> LiquidationExecStyle {
        match self.variant {
            Variant::Full => LiquidationExecStyle::Full,
            Variant::StaggeredTwap => LiquidationExecStyle::StaggeredTwap { slices: staggered_slices.max(1) },
        }
    }
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    let mut actions = Vec::new();
    if agent.bankrupt {
        return StepResult::sleep_only(None);
    }
    if let Some(mid) = ctx.market.mid_price(ctx.venue, s.asset_id) {
        if let Some(req) = open_leveraged_position(agent, s.asset_id, s.leverage_range, s.position_frac_of_cash, s.maintenance_margin_buffer, mid) {
            actions.push(Action::Submit(req));
        }
    }
    StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) }
}

/// Shared by the dedicated `liquidation_trader` bot type and by any other bot type's
/// leveraged sub-population (spec 5.7: leverage is independent of bot_type). No-op if
/// the agent already carries a margin position or has insufficient cash.
pub fn open_leveraged_position(
    agent: &mut AgentCommon,
    asset_id: AssetId,
    leverage_range: (f64, f64),
    position_frac_of_cash: f64,
    maintenance_margin_buffer: f64,
    mid: f64,
) -> Option<NewOrderRequest> {
    if agent.margin.is_some() || agent.bankrupt {
        return None;
    }
    let leverage = agent.rng.gen_range(leverage_range.0..=leverage_range.1.max(leverage_range.0 + 1e-6));
    let collateral = (agent.cash * position_frac_of_cash).max(0.0);
    if collateral <= 1.0 {
        return None;
    }
    let side = if agent.rng.gen_bool(0.5) { Side::Buy } else { Side::Sell };
    let notional = collateral * leverage;
    let size = notional / mid.max(1e-9);
    // Collateral is debited from cash directly here (not via the fill's price*qty --
    // that's the leveraged notional, not the margin actually posted). The order below
    // is submitted purely to source real fills/price-impact from the shared order
    // book; `sim.rs` routes margin orders through a settlement path that only applies
    // the fee, never a second cash/inventory adjustment for the notional.
    agent.cash -= collateral;
    agent.margin = Some(MarginPosition {
        asset_id,
        side,
        size,
        entry_price: mid,
        leverage,
        maintenance_margin_buffer,
        collateral,
    });
    Some(NewOrderRequest {
        asset_id,
        side,
        order_type: OrderType::Market,
        tif: TimeInForce::Ioc,
        post_only: false,
        price: None,
        qty: size,
    })
}

/// How much of a margin position must be closed to bring its margin ratio back to a
/// safe level above the maintenance threshold. Simplified model (documented in the
/// README): closing a slice at the current mark price is assumed not to change total
/// equity (collateral + unrealized pnl); the target post-liquidation size is chosen so
/// equity / new_notional == mm_rate * safety_factor.
pub fn required_liquidation_qty(pos: &MarginPosition, mark_price: f64, maintenance_margin_buffer: f64, safety_factor: f64) -> f64 {
    let mm_rate = maintenance_margin_buffer / pos.leverage.max(1e-9);
    let target_ratio = (mm_rate * safety_factor).min(1.0);
    let equity = pos.collateral + pos.unrealized_pnl(mark_price);
    if equity <= 0.0 || target_ratio <= 0.0 {
        return pos.size;
    }
    let target_notional = equity / target_ratio;
    let target_size = (target_notional / mark_price.max(1e-9)).max(0.0);
    (pos.size - target_size).clamp(0.0, pos.size)
}
