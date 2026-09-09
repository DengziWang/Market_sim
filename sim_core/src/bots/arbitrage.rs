//! Arbitrage bots: 4.2 in the spec.
//!
//! Simplification note (triangular): the engine models each asset as a single spot
//! book quoted in an implicit common numeraire (not dedicated pair books for every
//! A/B, B/C, A/C combination). True triangular arbitrage across 3 independent pair
//! books is listed as a natural Phase-2 extension once pair order books exist. Here,
//! triangular arbitrage is approximated as 3 assets that should maintain a configured
//! `peg_ratio` relationship; the bot trades all 3 legs (IOC, leg risk minimized by
//! sizing to the smallest available leg) whenever that relationship drifts.

use super::StepCtx;
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderType, Side, SimTime, TimeInForce, VenueId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    CrossVenue,
    Triangular,
    StatArbPairs,
    LatencyArb,
}

pub struct State {
    pub variant: Variant,
    pub wake_interval: SimTime,
    pub order_qty: f64,
    // cross-venue / latency-arb
    pub asset_a: AssetId,
    pub venue_a: VenueId,
    pub venue_b: VenueId,
    pub transfer_cost_frac: f64,
    pub last_seen_price: Option<f64>,
    pub jump_threshold_frac: f64,
    // triangular
    pub asset_b: AssetId,
    pub asset_c: AssetId,
    pub peg_ratio: f64,
    pub peg_deviation_threshold: f64,
    // stat arb pairs
    pub beta: f64,
    pub spread_window: super::RollingWindow,
    pub entry_z: f64,
    pub exit_z: f64,
    pub in_position: bool,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::CrossVenue => "cross_venue",
            Variant::Triangular => "triangular",
            Variant::StatArbPairs => "stat_arb_pairs",
            Variant::LatencyArb => "latency_arb",
        }
    }

    pub fn from_params(variant: Variant, asset_a: AssetId, asset_b: AssetId, asset_c: AssetId, venue_a: VenueId, venue_b: VenueId, p: &toml::Value) -> Self {
        let z_window = params::get_u32(p, "z_window", 50) as usize;
        Self {
            variant,
            wake_interval: params::get_f64(p, "wake_interval_secs", 0.2),
            order_qty: params::get_f64(p, "order_qty", 0.05),
            asset_a,
            venue_a,
            venue_b,
            transfer_cost_frac: params::get_f64(p, "transfer_cost_frac", 0.0015),
            last_seen_price: None,
            jump_threshold_frac: params::get_f64(p, "jump_threshold_frac", 0.002),
            asset_b,
            asset_c,
            peg_ratio: params::get_f64(p, "peg_ratio", 1.0),
            peg_deviation_threshold: params::get_f64(p, "peg_deviation_threshold", 0.003),
            beta: params::get_f64(p, "beta", 1.0),
            spread_window: super::RollingWindow::new(z_window),
            entry_z: params::get_f64(p, "entry_z", 2.0),
            exit_z: params::get_f64(p, "exit_z", 0.3),
            in_position: false,
        }
    }
}

fn ioc(asset_id: AssetId, side: Side, qty: f64) -> Action {
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

/// Rebalance toward a signed target position, bounded by `s.order_qty` on either
/// side -- same pattern as momentum/mean_reversion. Without this, a strategy that
/// re-evaluates every `wake_interval` while its trigger condition (a persistent
/// mispricing) holds would otherwise keep firing a fresh, uncapped order every wake
/// with no memory of the position it already built up; that compounded into an
/// effectively unbounded one-directional position in testing (see git history) for
/// CrossVenue/Triangular/LatencyArb, which (unlike StatArbPairs) had no such cap.
fn rebalance(asset_id: AssetId, current: f64, target: f64) -> Option<Action> {
    let delta = target - current;
    if delta.abs() < 1e-9 {
        return None;
    }
    let side = if delta > 0.0 { Side::Buy } else { Side::Sell };
    Some(ioc(asset_id, side, delta.abs()))
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    let mut actions = Vec::new();
    match s.variant {
        Variant::CrossVenue => {
            if let (Some(pa), Some(pb)) = (ctx.market.mid_price(s.venue_a, s.asset_a), ctx.market.mid_price(s.venue_b, s.asset_a)) {
                let diff = pb - pa;
                let cost = pa * s.transfer_cost_frac;
                let current = agent.inventory_of(s.asset_a);
                // Target flips sign with which venue is cheaper, but stays 0 (no
                // legitimate signal) once the gap is inside transfer cost -- and
                // rebalance() only trades the *delta* to get there, so a persistent
                // gap does not cause repeated re-entry (both venues assumed
                // pre-funded, i.e. no real-time asset transfer modeled).
                let target = if diff.abs() > cost {
                    if diff > 0.0 { s.order_qty } else { -s.order_qty }
                } else {
                    0.0
                };
                if agent.cash > pa.max(pb) * s.order_qty {
                    if let Some(a) = rebalance(s.asset_a, current, target) {
                        actions.push(a);
                    }
                }
            }
        }
        Variant::Triangular => {
            if let (Some(pa), Some(pb), Some(pc)) = (
                ctx.market.mid_price(ctx.venue, s.asset_a),
                ctx.market.mid_price(ctx.venue, s.asset_b),
                ctx.market.mid_price(ctx.venue, s.asset_c),
            ) {
                let implied_c = pa / pb.max(1e-9) * s.peg_ratio;
                let deviation = (pc - implied_c) / implied_c.max(1e-9);
                let target_c = if deviation.abs() > s.peg_deviation_threshold {
                    if deviation > 0.0 { -s.order_qty } else { s.order_qty }
                } else {
                    0.0
                };
                let target_a = -target_c; // opposite legs of the same cycle
                let target_b = target_c;
                if agent.cash > (pa + pb + pc) * s.order_qty {
                    if let Some(a) = rebalance(s.asset_c, agent.inventory_of(s.asset_c), target_c) {
                        actions.push(a);
                    }
                    if let Some(a) = rebalance(s.asset_a, agent.inventory_of(s.asset_a), target_a) {
                        actions.push(a);
                    }
                    if let Some(a) = rebalance(s.asset_b, agent.inventory_of(s.asset_b), target_b) {
                        actions.push(a);
                    }
                }
            }
        }
        Variant::StatArbPairs => {
            if let (Some(pa), Some(pb)) = (ctx.market.mid_price(ctx.venue, s.asset_a), ctx.market.mid_price(ctx.venue, s.asset_b)) {
                let spread = pa - s.beta * pb;
                s.spread_window.push(spread);
                if let (Some(mean), Some(std)) = (s.spread_window.mean(), s.spread_window.std()) {
                    if std > 1e-9 && s.spread_window.is_full() {
                        let z = (spread - mean) / std;
                        if !s.in_position && z > s.entry_z {
                            actions.push(ioc(s.asset_a, Side::Sell, s.order_qty));
                            actions.push(ioc(s.asset_b, Side::Buy, (s.order_qty * s.beta).max(0.0)));
                            s.in_position = true;
                        } else if !s.in_position && z < -s.entry_z {
                            actions.push(ioc(s.asset_a, Side::Buy, s.order_qty));
                            actions.push(ioc(s.asset_b, Side::Sell, (s.order_qty * s.beta).max(0.0)));
                            s.in_position = true;
                        } else if s.in_position && z.abs() < s.exit_z {
                            // unwind toward flat
                            actions.push(ioc(s.asset_a, Side::Buy, s.order_qty));
                            actions.push(ioc(s.asset_b, Side::Sell, (s.order_qty * s.beta).max(0.0)));
                            s.in_position = false;
                        }
                    }
                }
            }
        }
        Variant::LatencyArb => {
            if let Some(p) = ctx.market.mid_price(ctx.venue, s.asset_a) {
                if let Some(prev) = s.last_seen_price {
                    let ret = (p - prev) / prev.max(1e-9);
                    if ret.abs() > s.jump_threshold_frac {
                        // Exploits the fact this bot polls far more frequently
                        // (short wake_interval, low configured latency) than most of
                        // the population, letting it react to a move before it fully
                        // propagates through slower agents' quotes. Bounded by
                        // order_qty via rebalance(), not fired unconditionally every
                        // wake: its own trades otherwise could look like a fresh
                        // "jump" and re-trigger itself indefinitely in one direction.
                        let target = if ret > 0.0 { s.order_qty } else { -s.order_qty };
                        if let Some(a) = rebalance(s.asset_a, agent.inventory_of(s.asset_a), target) {
                            actions.push(a);
                        }
                    }
                }
                s.last_seen_price = Some(p);
            }
        }
    }
    StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) }
}
