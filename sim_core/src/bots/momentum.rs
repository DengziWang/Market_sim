//! Momentum / trend-following bots: 4.3 in the spec. Each variant recomputes its
//! signal once per wake (the wake interval acts as the indicator's "bar" period,
//! matching how real periodic-poll trend bots operate) and rebalances inventory
//! toward a signed target position via an IOC market order.

use super::{RollingWindow, StepCtx};
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderType, Side, SimTime, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    MaCrossover,
    BreakoutDonchian,
    Rsi,
    MlOnline,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub variant: Variant,
    pub asset_id: AssetId,
    pub wake_interval: SimTime,
    pub order_qty: f64,
    pub allow_short: bool,
    // MA crossover
    pub alpha_short: f64,
    pub alpha_long: f64,
    pub ema_short: Option<f64>,
    pub ema_long: Option<f64>,
    // Donchian breakout
    pub donchian_window: RollingWindow,
    pub current_signal: f64,
    // RSI
    pub rsi_window: RollingWindow,
    pub last_price: Option<f64>,
    pub rsi_overbought: f64,
    pub rsi_oversold: f64,
    // ML online (simple logistic regression on 3 hand features, online SGD)
    pub weights: [f64; 3],
    pub bias: f64,
    pub learning_rate: f64,
    pub short_ret_window: RollingWindow,
    pub medium_ret_window: RollingWindow,
    pub prev_features: Option<[f64; 3]>,
    pub prev_price: Option<f64>,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::MaCrossover => "ma_crossover",
            Variant::BreakoutDonchian => "breakout_donchian",
            Variant::Rsi => "rsi",
            Variant::MlOnline => "ml_online",
        }
    }

    /// `initial_price`: converts the dollar-notional `order_notional` param into
    /// asset units, so a single default position size means comparable *dollar*
    /// exposure across assets of very different price scales -- see
    /// market_maker.rs's from_params doc comment for why a fixed unit-quantity
    /// default doesn't (same issue, this bot type just doesn't provide resting
    /// liquidity so it didn't show up as a price-stability bug -- it still skewed
    /// reported per-asset volume by orders of magnitude).
    pub fn from_params(variant: Variant, asset_id: AssetId, initial_price: f64, p: &toml::Value) -> Self {
        let short_p = params::get_f64(p, "ema_short_period", 5.0);
        let long_p = params::get_f64(p, "ema_long_period", 20.0);
        let donchian_n = params::get_u32(p, "donchian_window", 20) as usize;
        let rsi_n = params::get_u32(p, "rsi_window", 14) as usize;
        let order_qty = params::get_f64(p, "order_notional", 3000.0) / initial_price.max(1e-9);
        Self {
            variant,
            asset_id,
            wake_interval: params::get_f64(p, "wake_interval_secs", 5.0),
            order_qty,
            allow_short: params::get_bool(p, "allow_short", true),
            alpha_short: 2.0 / (short_p + 1.0),
            alpha_long: 2.0 / (long_p + 1.0),
            ema_short: None,
            ema_long: None,
            donchian_window: RollingWindow::new(donchian_n),
            current_signal: 0.0,
            rsi_window: RollingWindow::new(rsi_n),
            last_price: None,
            rsi_overbought: params::get_f64(p, "rsi_overbought", 65.0),
            rsi_oversold: params::get_f64(p, "rsi_oversold", 35.0),
            weights: [0.0, 0.0, 0.0],
            bias: 0.0,
            learning_rate: params::get_f64(p, "learning_rate", 0.05),
            short_ret_window: RollingWindow::new(3),
            medium_ret_window: RollingWindow::new(10),
            prev_features: None,
            prev_price: None,
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
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
    let max_pos = if s.allow_short { s.order_qty } else { s.order_qty };

    match s.variant {
        Variant::MaCrossover => {
            s.ema_short = Some(s.ema_short.map_or(mid, |e| e + s.alpha_short * (mid - e)));
            s.ema_long = Some(s.ema_long.map_or(mid, |e| e + s.alpha_long * (mid - e)));
            if let (Some(short), Some(long)) = (s.ema_short, s.ema_long) {
                let target = if short > long { max_pos } else if s.allow_short { -max_pos } else { 0.0 };
                if let Some(a) = rebalance(s.asset_id, inventory, target) {
                    actions.push(a);
                }
            }
        }
        Variant::BreakoutDonchian => {
            let high = s.donchian_window.max();
            let low = s.donchian_window.min();
            if let (Some(h), Some(l)) = (high, low) {
                if mid > h {
                    s.current_signal = max_pos;
                } else if mid < l {
                    s.current_signal = if s.allow_short { -max_pos } else { 0.0 };
                }
            }
            s.donchian_window.push(mid);
            if let Some(a) = rebalance(s.asset_id, inventory, s.current_signal) {
                actions.push(a);
            }
        }
        Variant::Rsi => {
            if let Some(prev) = s.last_price {
                s.rsi_window.push(mid - prev);
            }
            s.last_price = Some(mid);
            if s.rsi_window.is_full() {
                let (gains, losses): (f64, f64) = s.rsi_window.buf.iter().fold((0.0, 0.0), |(g, l), &d| {
                    if d > 0.0 { (g + d, l) } else { (g, l - d) }
                });
                let avg_gain = gains / s.rsi_window.len() as f64;
                let avg_loss = losses / s.rsi_window.len() as f64;
                let rsi = if avg_loss < 1e-12 {
                    100.0
                } else {
                    100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
                };
                // Momentum-style (trend-confirmation) use of RSI, not mean-reversion:
                // strength readings are traded *with* the trend.
                let target = if rsi > s.rsi_overbought {
                    max_pos
                } else if rsi < s.rsi_oversold {
                    if s.allow_short { -max_pos } else { 0.0 }
                } else {
                    inventory
                };
                if let Some(a) = rebalance(s.asset_id, inventory, target) {
                    actions.push(a);
                }
            }
        }
        Variant::MlOnline => {
            if let Some(prev_price) = s.prev_price {
                let realized_ret = (mid - prev_price) / prev_price.max(1e-9);
                if let Some(pf) = s.prev_features {
                    let label = if realized_ret > 0.0 { 1.0 } else { 0.0 };
                    let pred = sigmoid(pf[0] * s.weights[0] + pf[1] * s.weights[1] + pf[2] * s.weights[2] + s.bias);
                    let err = label - pred;
                    for i in 0..3 {
                        s.weights[i] += s.learning_rate * err * pf[i];
                    }
                    s.bias += s.learning_rate * err;
                }
                s.short_ret_window.push(realized_ret);
                s.medium_ret_window.push(realized_ret);
            }
            let feat = [
                s.short_ret_window.mean().unwrap_or(0.0) * 100.0,
                s.medium_ret_window.mean().unwrap_or(0.0) * 100.0,
                s.medium_ret_window.std().unwrap_or(0.0) * 100.0,
            ];
            let prob_up = sigmoid(feat[0] * s.weights[0] + feat[1] * s.weights[1] + feat[2] * s.weights[2] + s.bias);
            let target = if prob_up > 0.55 {
                max_pos
            } else if prob_up < 0.45 && s.allow_short {
                -max_pos
            } else {
                0.0
            };
            if let Some(a) = rebalance(s.asset_id, inventory, target) {
                actions.push(a);
            }
            s.prev_features = Some(feat);
            s.prev_price = Some(mid);
        }
    }

    let _ = agent;
    StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) }
}
