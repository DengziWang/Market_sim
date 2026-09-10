//! DCA (Dollar-Cost-Averaging) bots: 4.6 in the spec.

use super::StepCtx;
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderType, Side, SimTime, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    TimeBased,
    ValueAveraging,
    DipBuying,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub variant: Variant,
    pub asset_id: AssetId,
    pub wake_interval: SimTime,
    pub buy_amount_quote: f64,
    pub target_initial_value: f64,
    pub target_growth_per_period: f64,
    pub periods_elapsed: u32,
    pub peak_price: Option<f64>,
    pub dip_threshold: f64,
    pub dip_extra_frac: f64,
    pub last_dip_triggered_at_drawdown: f64,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::TimeBased => "time_based",
            Variant::ValueAveraging => "value_averaging",
            Variant::DipBuying => "dip_buying",
        }
    }

    pub fn from_params(variant: Variant, asset_id: AssetId, p: &toml::Value) -> Self {
        Self {
            variant,
            asset_id,
            wake_interval: params::get_f64(p, "wake_interval_secs", 300.0),
            buy_amount_quote: params::get_f64(p, "buy_amount_quote", 100.0),
            target_initial_value: params::get_f64(p, "target_initial_value", 100.0),
            target_growth_per_period: params::get_f64(p, "target_growth_per_period", 100.0),
            periods_elapsed: 0,
            peak_price: None,
            dip_threshold: params::get_f64(p, "dip_threshold", 0.1),
            dip_extra_frac: params::get_f64(p, "dip_extra_frac", 1.0),
            last_dip_triggered_at_drawdown: 0.0,
        }
    }
}

fn buy_quote_amount(asset_id: AssetId, mid: f64, quote_amount: f64) -> Option<Action> {
    if quote_amount <= 0.0 || mid <= 0.0 {
        return None;
    }
    Some(Action::Submit(NewOrderRequest {
        asset_id,
        side: Side::Buy,
        order_type: OrderType::Market,
        tif: TimeInForce::Ioc,
        post_only: false,
        price: None,
        qty: quote_amount / mid,
    }))
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    let mut actions = Vec::new();
    let Some(mid) = ctx.market.mid_price(ctx.venue, s.asset_id) else {
        return StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) };
    };

    match s.variant {
        Variant::TimeBased => {
            let amount = s.buy_amount_quote.min(agent.cash);
            if let Some(a) = buy_quote_amount(s.asset_id, mid, amount) {
                actions.push(a);
            }
        }
        Variant::ValueAveraging => {
            s.periods_elapsed += 1;
            let target_value = s.target_initial_value + s.target_growth_per_period * s.periods_elapsed as f64;
            let actual_value = agent.inventory_of(s.asset_id) * mid;
            let delta_value = target_value - actual_value;
            if delta_value > 0.0 {
                if let Some(a) = buy_quote_amount(s.asset_id, mid, delta_value.min(agent.cash)) {
                    actions.push(a);
                }
            } else if -delta_value > 1e-6 {
                let sell_qty = (-delta_value / mid).min(agent.inventory_of(s.asset_id).max(0.0));
                if sell_qty > 1e-9 {
                    actions.push(Action::Submit(NewOrderRequest {
                        asset_id: s.asset_id,
                        side: Side::Sell,
                        order_type: OrderType::Market,
                        tif: TimeInForce::Ioc,
                        post_only: false,
                        price: None,
                        qty: sell_qty,
                    }));
                }
            }
        }
        Variant::DipBuying => {
            let peak = s.peak_price.map_or(mid, |p| p.max(mid));
            s.peak_price = Some(peak);
            let drawdown = (peak - mid) / peak.max(1e-9);

            let amount = s.buy_amount_quote.min(agent.cash);
            if let Some(a) = buy_quote_amount(s.asset_id, mid, amount) {
                actions.push(a);
            }
            if drawdown >= s.dip_threshold && drawdown > s.last_dip_triggered_at_drawdown + 1e-6 {
                s.last_dip_triggered_at_drawdown = drawdown;
                let extra = (s.buy_amount_quote * s.dip_extra_frac).min((agent.cash - amount).max(0.0));
                if let Some(a) = buy_quote_amount(s.asset_id, mid, extra) {
                    actions.push(a);
                }
            }
            if drawdown < s.dip_threshold * 0.5 {
                // recovered enough to re-arm the next dip trigger
                s.last_dip_triggered_at_drawdown = 0.0;
            }
        }
    }

    StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) }
}
