//! MEV / sandwich bots: 5.8 in the spec. Only produce activity when
//! `mempool.enabled = true` in the scenario config (otherwise `visible_mempool`/
//! `pending_mempool` are always empty and these bots simply idle).
//!
//! Calibration note (from the spec): this population should be configured as a
//! handful of very active agents (short `wake_interval_secs`), not a large evenly
//! spread population -- real sandwich activity is heavily concentrated in a few
//! dominant bots.

use super::{RiskGovernor, StepCtx};
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, FxHashMap, OrderType, Side, SimTime, TimeInForce};
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    Sandwich,
    Frontrun,
    Backrun,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PendingSandwich {
    pub payload_id: u64,
    pub close_side: Side,
    pub qty: f64,
    pub close_at: SimTime,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Self de-risking on drawdown -- see `RiskGovernor`'s doc comment.
    pub risk: RiskGovernor,
    /// Fraction of noticed, qualifying victims actually acted on (2026-11-05,
    /// found during random-walk tuning work: every scenario TOML in this
    /// project already sets `reaction_prob` -- e.g. 0.0003 -- but `step()`
    /// below never read or applied it at all, so every mev agent reacted to
    /// *every* qualifying victim unconditionally. Measured consequence: an
    /// 18-agent (1.8%-of-population) mev group was capturing 30-75% of total
    /// trade notional in the realistic composition, vs. this project's own
    /// <1-2% target. Default 1.0 (react to every opportunity) preserves
    /// exactly that prior behavior for any scenario that doesn't set this
    /// param, so this fix only changes behavior where the TOML already
    /// expressed an intent this code wasn't honoring.
    pub reaction_prob: f64,
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
    /// Recalibrated 2026-10-21: the old 20,000/0.05s defaults sat just below this
    /// scenario's mempool-eligibility floor (`large_order_notional_threshold`,
    /// typically 25,000) and polled fast enough that, in a busy 5,000-agent book,
    /// `sandwich`/`frontrun` were effectively attacking almost every mempool-
    /// visible order rather than a genuinely large, rare one -- measured directly:
    /// 28 sandwich agents made ~34,000 trades in just 2,000s. Worse, many of the
    /// ~28 agents per variant were independently targeting the same victims
    /// (crowding), diluting the profit any one of them could actually capture.
    /// Raising the threshold and slowing the wake interval took sandwich from
    /// -X%/60.7% bankrupt to +110%/10.7% bankrupt on the same 25,000s benchmark
    /// (frontrun: similar swing, from 67.9% to 14.3% bankrupt).
    pub fn from_params(variant: Variant, asset_id: AssetId, initial_price: f64, p: &toml::Value) -> Self {
        let min_victim_qty = params::get_f64(p, "min_victim_notional", 80_000.0) / initial_price.max(1e-9);
        Self {
            variant,
            asset_id,
            wake_interval: params::get_f64(p, "wake_interval_secs", 0.3),
            min_victim_qty,
            size_frac_of_victim: params::get_f64(p, "size_frac_of_victim", 0.5),
            frontrun_hold_secs: params::get_f64(p, "frontrun_hold_secs", 2.0),
            pending_sandwich: None,
            frontrun_close_at: None,
            frontrun_close_side: None,
            frontrun_qty: 0.0,
            seen_payloads: Vec::new(),
            risk: RiskGovernor::from_params(p),
            reaction_prob: params::get_f64(p, "reaction_prob", 1.0).clamp(0.0, 1.0),
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

    // Self de-risking (see `RiskGovernor`'s doc comment): gates *new* attacks, not
    // closing a position already in flight -- sandwich/frontrun holds are brief
    // (micro-seconds to `frontrun_hold_secs`), so letting an in-progress close
    // finish is not the open-ended risk a flattening gate exists for. If the mid
    // price is briefly unavailable, don't let that spuriously trip the governor --
    // poll with an effectively-infinite equity so only an already-running cooldown
    // still applies.
    let equity = match ctx.market.mid_price(ctx.venue, s.asset_id) {
        Some(p) => {
            let mut mark_prices = FxHashMap::default();
            mark_prices.insert(s.asset_id, p);
            agent.equity(&mark_prices)
        }
        None => f64::MAX,
    };
    let allowed = s.risk.poll(equity, agent.initial_net_worth, ctx.now);

    match s.variant {
        Variant::Sandwich => {
            if let Some(p) = s.pending_sandwich.take() {
                if ctx.now + 1e-9 >= p.close_at {
                    actions.push(market_order(s.asset_id, p.close_side, p.qty));
                } else {
                    s.pending_sandwich = Some(p);
                }
            } else if allowed {
                let victim = ctx
                    .market
                    .visible_mempool(ctx.now)
                    .filter(|o| o.asset_id == s.asset_id && o.agent_id != agent.id && o.qty >= s.min_victim_qty && !s.seen_payloads.contains(&o.payload_id))
                    .min_by(|a, b| a.execute_at.partial_cmp(&b.execute_at).unwrap())
                    .cloned();
                if let Some(v) = victim {
                    s.mark_seen(v.payload_id);
                    if agent.rng.gen_bool(s.reaction_prob) {
                        let qty = v.qty * s.size_frac_of_victim * s.risk.size_multiplier;
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
        }
        Variant::Frontrun => {
            if let (Some(close_at), Some(side)) = (s.frontrun_close_at, s.frontrun_close_side) {
                if ctx.now + 1e-9 >= close_at {
                    actions.push(market_order(s.asset_id, side, s.frontrun_qty));
                    s.frontrun_close_at = None;
                    s.frontrun_close_side = None;
                }
            } else if allowed {
                let victim = ctx
                    .market
                    .visible_mempool(ctx.now)
                    .filter(|o| o.asset_id == s.asset_id && o.agent_id != agent.id && o.qty >= s.min_victim_qty && !s.seen_payloads.contains(&o.payload_id))
                    .next()
                    .cloned();
                if let Some(v) = victim {
                    s.mark_seen(v.payload_id);
                    if agent.rng.gen_bool(s.reaction_prob) {
                        let qty = v.qty * s.size_frac_of_victim * s.risk.size_multiplier;
                        actions.push(market_order(s.asset_id, v.side, qty));
                        s.frontrun_close_at = Some(ctx.now + s.frontrun_hold_secs);
                        s.frontrun_close_side = Some(v.side.opposite());
                        s.frontrun_qty = qty;
                    }
                }
            }
        }
        Variant::Backrun => {
            if allowed {
                let hit = ctx
                    .market
                    .pending_mempool
                    .iter()
                    .find(|o| o.asset_id == s.asset_id && o.execute_at <= ctx.now && !s.seen_payloads.contains(&o.payload_id))
                    .cloned();
                if let Some(v) = hit {
                    s.mark_seen(v.payload_id);
                    if agent.rng.gen_bool(s.reaction_prob) {
                        // No front leg: exploit the price impact *after* the large order has
                        // landed, trading in the same direction (pure impact/arbitrage capture).
                        actions.push(market_order(s.asset_id, v.side, v.qty * s.size_frac_of_victim * s.risk.size_multiplier));
                    }
                }
            }
        }
    }

    StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) }
}
