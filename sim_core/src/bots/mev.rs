//! MEV / sandwich bots: 5.8 in the spec. Only produce activity when
//! `mempool.enabled = true` in the scenario config (otherwise `visible_mempool`/
//! `pending_mempool` are always empty and these bots simply idle).
//!
//! Calibration note (from the spec): this population should be configured as a
//! handful of very active agents (short `wake_interval_secs`), not a large evenly
//! spread population -- real sandwich activity is heavily concentrated in a few
//! dominant bots.

use super::StepCtx;
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderType, Side, SimTime, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Sandwich,
    Frontrun,
    Backrun,
}

#[derive(Debug, Clone)]
pub struct PendingSandwich {
    pub payload_id: u64,
    pub close_side: Side,
    pub qty: f64,
    pub close_at: SimTime,
}

pub struct State {
    pub variant: Variant,
    pub asset_id: AssetId,
    pub wake_interval: SimTime,
    pub min_victim_qty: f64,
    pub size_frac_of_victim: f64,
    pub frontrun_hold_secs: f64,
    pub pending_sandwich: Option<PendingSandwich>,
    pub frontrun_close_at: Option<SimTime>,
    pub frontrun_close_side: Option<Side>,
    pub frontrun_qty: f64,
    pub seen_payloads: Vec<u64>,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::Sandwich => "sandwich",
            Variant::Frontrun => "frontrun",
            Variant::Backrun => "backrun",
        }
    }

    /// `initial_price`: converts the dollar-notional `min_victim_notional` param
    /// into asset units -- see market_maker.rs's from_params doc comment. A fixed
    /// unit threshold is trivially satisfied by any order on a cheap asset (and
    /// unreasonably strict on an expensive one), independent of whether that order
    /// is actually "large" in the dollar-value sense that makes it worth attacking.
    pub fn from_params(variant: Variant, asset_id: AssetId, initial_price: f64, p: &toml::Value) -> Self {
        let min_victim_qty = params::get_f64(p, "min_victim_notional", 20_000.0) / initial_price.max(1e-9);
        Self {
            variant,
            asset_id,
            wake_interval: params::get_f64(p, "wake_interval_secs", 0.05),
            min_victim_qty,
            size_frac_of_victim: params::get_f64(p, "size_frac_of_victim", 0.5),
            frontrun_hold_secs: params::get_f64(p, "frontrun_hold_secs", 2.0),
            pending_sandwich: None,
            frontrun_close_at: None,
            frontrun_close_side: None,
            frontrun_qty: 0.0,
            seen_payloads: Vec::new(),
        }
    }

    fn mark_seen(&mut self, payload_id: u64) {
        self.seen_payloads.push(payload_id);
        if self.seen_payloads.len() > 256 {
            self.seen_payloads.remove(0);
        }
    }
}

fn market_order(asset_id: AssetId, side: Side, qty: f64) -> Action {
    Action::Submit(NewOrderRequest {
        asset_id,
        side,
        order_type: OrderType::Market,
        tif: TimeInForce::Ioc,
        post_only: false,
        price: None,
        qty,
    })
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    let mut actions = Vec::new();

    match s.variant {
        Variant::Sandwich => {
            if let Some(p) = s.pending_sandwich.take() {
                if ctx.now + 1e-9 >= p.close_at {
                    actions.push(market_order(s.asset_id, p.close_side, p.qty));
                } else {
                    s.pending_sandwich = Some(p);
                }
            } else {
                let victim = ctx
                    .market
                    .visible_mempool(ctx.now)
                    .filter(|o| o.asset_id == s.asset_id && o.agent_id != agent.id && o.qty >= s.min_victim_qty && !s.seen_payloads.contains(&o.payload_id))
                    .min_by(|a, b| a.execute_at.partial_cmp(&b.execute_at).unwrap())
                    .cloned();
                if let Some(v) = victim {
                    s.mark_seen(v.payload_id);
                    let qty = v.qty * s.size_frac_of_victim;
                    actions.push(market_order(s.asset_id, v.side, qty));
                    s.pending_sandwich = Some(PendingSandwich {
                        payload_id: v.payload_id,
                        close_side: v.side.opposite(),
                        qty,
                        close_at: v.execute_at + 1e-3,
                    });
                }
            }
        }
        Variant::Frontrun => {
            if let (Some(close_at), Some(side)) = (s.frontrun_close_at, s.frontrun_close_side) {
                if ctx.now + 1e-9 >= close_at {
                    actions.push(market_order(s.asset_id, side, s.frontrun_qty));
                    s.frontrun_close_at = None;
                    s.frontrun_close_side = None;
                }
            } else {
                let victim = ctx
                    .market
                    .visible_mempool(ctx.now)
                    .filter(|o| o.asset_id == s.asset_id && o.agent_id != agent.id && o.qty >= s.min_victim_qty && !s.seen_payloads.contains(&o.payload_id))
                    .next()
                    .cloned();
                if let Some(v) = victim {
                    s.mark_seen(v.payload_id);
                    let qty = v.qty * s.size_frac_of_victim;
                    actions.push(market_order(s.asset_id, v.side, qty));
                    s.frontrun_close_at = Some(ctx.now + s.frontrun_hold_secs);
                    s.frontrun_close_side = Some(v.side.opposite());
                    s.frontrun_qty = qty;
                }
            }
        }
        Variant::Backrun => {
            let hit = ctx
                .market
                .pending_mempool
                .iter()
                .find(|o| o.asset_id == s.asset_id && o.execute_at <= ctx.now && !s.seen_payloads.contains(&o.payload_id))
                .cloned();
            if let Some(v) = hit {
                s.mark_seen(v.payload_id);
                // No front leg: exploit the price impact *after* the large order has
                // landed, trading in the same direction (pure impact/arbitrage capture).
                actions.push(market_order(s.asset_id, v.side, v.qty * s.size_frac_of_victim));
            }
        }
    }

    StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) }
}
