//! Mean-reversion bots: 4.4 in the spec (a separate category from momentum).

use super::{RollingWindow, StepCtx};
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderType, Side, SimTime, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Bollinger,
    OuFairValue,
    ZScore,
}

pub struct State {
    pub variant: Variant,
    pub asset_id: AssetId,
    pub wake_interval: SimTime,
    pub order_qty: f64,
    pub allow_short: bool,
    pub window: RollingWindow,
    pub num_std: f64,
    pub deviation_threshold: f64,
    pub entry_z: f64,
    pub exit_z: f64,
    pub in_position: bool,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::Bollinger => "bollinger",
            Variant::OuFairValue => "ou_fair_value",
            Variant::ZScore => "zscore",
        }
    }

    pub fn from_params(variant: Variant, asset_id: AssetId, p: &toml::Value) -> Self {
        let window_n = params::get_u32(p, "window", 20) as usize;
        Self {
            variant,
            asset_id,
            wake_interval: params::get_f64(p, "wake_interval_secs", 3.0),
            order_qty: params::get_f64(p, "order_qty", 0.05),
            allow_short: params::get_bool(p, "allow_short", true),
            window: RollingWindow::new(window_n),
            num_std: params::get_f64(p, "num_std", 2.0),
            deviation_threshold: params::get_f64(p, "deviation_threshold", 0.01),
            entry_z: params::get_f64(p, "entry_z", 1.5),
            exit_z: params::get_f64(p, "exit_z", 0.3),
            in_position: false,
        }
    }
}

fn rebalance(asset_id: AssetId, current: f64, target: f64) -> Option<Action> {
    let delta = target - current;
    if delta.abs() < 1e-9 {
        return None;
    }
    let side = if delta > 0.0 { Side::Buy } else { Side::Sell };
    Some(Action::Submit(NewOrderRequest {
        asset_id,
        side,
        order_type: OrderType::Market,
        tif: TimeInForce::Ioc,
        post_only: false,
        price: None,
        qty: delta.abs(),
    }))
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    let mut actions = Vec::new();
    let Some(mid) = ctx.market.mid_price(ctx.venue, s.asset_id) else {
        return StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) };
    };
    let inventory = agent.inventory_of(s.asset_id);
    let max_pos = s.order_qty;

    match s.variant {
        Variant::Bollinger => {
            s.window.push(mid);
            if let (Some(mean), Some(std)) = (s.window.mean(), s.window.std()) {
                if s.window.is_full() && std > 1e-9 {
                    let upper = mean + s.num_std * std;
                    let lower = mean - s.num_std * std;
                    let target = if mid > upper {
                        if s.allow_short { -max_pos } else { 0.0 }
                    } else if mid < lower {
                        max_pos
                    } else if (mid - mean).abs() < 0.3 * std {
                        0.0
                    } else {
                        inventory
                    };
                    if let Some(a) = rebalance(s.asset_id, inventory, target) {
                        actions.push(a);
                    }
                }
            }
        }
        Variant::OuFairValue => {
            // Uses the simulation's ground-truth latent fundamental value directly --
            // a "fundamentalist" archetype (Chiarella 1992) that expects price to
            // revert toward the known fair value, unlike Bollinger/ZScore which only
            // see the traded price series.
            let fair = *ctx.market.fundamental_value.get(&s.asset_id).unwrap_or(&mid);
            let dev = (mid - fair) / fair.max(1e-9);
            let target = if dev > s.deviation_threshold {
                if s.allow_short { -max_pos } else { 0.0 }
            } else if dev < -s.deviation_threshold {
                max_pos
            } else {
                0.0
            };
            if let Some(a) = rebalance(s.asset_id, inventory, target) {
                actions.push(a);
            }
        }
        Variant::ZScore => {
            s.window.push(mid);
            if let (Some(mean), Some(std)) = (s.window.mean(), s.window.std()) {
                if s.window.is_full() && std > 1e-9 {
                    let z = (mid - mean) / std;
                    if !s.in_position && z > s.entry_z {
                        if let Some(a) = rebalance(s.asset_id, inventory, if s.allow_short { -max_pos } else { 0.0 }) {
                            actions.push(a);
                        }
                        s.in_position = true;
                    } else if !s.in_position && z < -s.entry_z {
                        if let Some(a) = rebalance(s.asset_id, inventory, max_pos) {
                            actions.push(a);
                        }
                        s.in_position = true;
                    } else if s.in_position && z.abs() < s.exit_z {
                        if let Some(a) = rebalance(s.asset_id, inventory, 0.0) {
                            actions.push(a);
                        }
                        s.in_position = false;
                    }
                }
            }
        }
    }

    let _ = agent;
    StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) }
}
