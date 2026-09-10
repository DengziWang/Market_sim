//! Wash-trading bots: 5.9 in the spec.
//!
//! Simplification (self_ring): implemented as a single agent crossing its own resting
//! order (post a passive limit, immediately take it with an IOC order) rather than
//! coordinating a multi-wallet ring across distinct agent ids. This produces the
//! defining economic signature the spec and report care about -- volume with ~zero
//! net position change, at the cost of round-trip fees -- and is ground-truth
//! taggable (agent type is known) for the double-validation the report describes.
//! True separate-wallet ring coordination is a natural later extension.

use rand::Rng;

use super::StepCtx;
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderType, Side, SimTime, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    SelfRing,
    VolumeRanking,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub variant: Variant,
    pub asset_id: AssetId,
    pub wake_interval: SimTime,
    pub qty_round: f64,
    pub target_session_volume: f64,
    pub session_end: SimTime,
    pub cumulative_volume: f64,
    /// volume_ranking's per-wake unit size, fixed once at construction from the
    /// asset's *initial* price (target_session_volume / initial_price / planned
    /// wake count) rather than recomputed from the live price every wake. Dividing a
    /// dollar target by the *live* price is unstable: if price dips at all, the
    /// computed unit quantity grows, and executing that larger self-cross nudges
    /// price down further, which grows the next quantity again -- a real runaway
    /// feedback loop observed in testing (see git history), not just a theoretical
    /// risk. Pacing off a fixed reference price trades a little target-volume
    /// accuracy (if price moves a lot over the session) for not being able to blow up.
    pub unit_qty_per_wake: f64,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::SelfRing => "self_ring",
            Variant::VolumeRanking => "volume_ranking",
        }
    }

    pub fn from_params(variant: Variant, asset_id: AssetId, session_end: SimTime, initial_price: f64, p: &toml::Value) -> Self {
        let wake_interval = params::get_f64(p, "wake_interval_secs", 2.0);
        let target_session_volume = params::get_f64(p, "target_session_volume", 5000.0);
        let planned_wakes = (session_end / wake_interval.max(1e-6)).max(1.0);
        // qty_round (self_ring's own per-round-trip size, and the base the
        // volume_ranking cap below scales from) is also dollar-notional now, for the
        // same reason as every other bot type's sizing -- a fixed unit count is wildly
        // inconsistent in dollar terms across assets of different price scales.
        let qty_round = params::get_f64(p, "notional_round", 200.0) / initial_price.max(1e-9);
        // A dollar target applied blindly to a cheap asset implies an enormous *unit*
        // quantity (e.g. $200k / $0.15 DOGE = ~1.3M units) -- wildly disproportionate
        // to realistic resting liquidity (market-maker quotes are ~0.05-0.6 units in
        // these scenarios) regardless of how correctly the self-cross itself is
        // implemented, and drove a real price-crash cascade in testing (the
        // self-cross's own momentary price prints were enough to trigger adaptive
        // market-makers' volatility withdrawal, removing the real book entirely --
        // see git history). Cap it to a bounded multiple of qty_round; a
        // misconfigured target_session_volume degrades to "trade qty_round-ish size
        // more often" rather than "trade a liquidity-swamping size".
        let uncapped = (target_session_volume / initial_price.max(1e-9)) / planned_wakes;
        let max_unit_qty = params::get_f64(p, "max_qty_per_wake", qty_round * 20.0);
        Self {
            variant,
            asset_id,
            wake_interval,
            qty_round,
            target_session_volume,
            session_end,
            cumulative_volume: 0.0,
            unit_qty_per_wake: uncapped.min(max_unit_qty),
        }
    }
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    let mut actions = Vec::new();
    let Some(mid) = ctx.market.mid_price(ctx.venue, s.asset_id) else {
        return StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) };
    };
    let tick = ctx.market.assets[s.asset_id as usize].tick_size;

    let qty = match s.variant {
        Variant::SelfRing => s.qty_round,
        Variant::VolumeRanking => s.unit_qty_per_wake,
    };
    if qty <= 1e-9 {
        return StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) };
    }

    // Alternate which side rests vs. takes so the agent doesn't accumulate a
    // persistent quoting footprint on one side of the book. `leg1` is priced to
    // become the unconditional new best price on its side (strictly beyond the
    // current best), so time-priority guarantees `leg2`'s immediate IOC market order
    // below matches leg1 itself rather than some unrelated resting order -- i.e. this
    // really is a self-cross, not an accidental trade against a real counterparty.
    // post_only on leg1 is a belt-and-braces guard: if the price nudge is somehow
    // still marketable (e.g. a 1-tick-spread edge case), it is rejected rather than
    // executed against someone else, and no leg2 is sent that wake.
    let book = ctx.market.book(ctx.venue, s.asset_id);
    let best_bid = book.best_bid();
    let best_ask = book.best_ask();
    let leg1_side = if agent.rng.gen_bool(0.5) { Side::Buy } else { Side::Sell };
    let leg1_price = match leg1_side {
        Side::Buy => best_bid.map(|b| b + tick).unwrap_or(mid - tick),
        Side::Sell => best_ask.map(|a| a - tick).unwrap_or(mid + tick),
    };
    let safe = match leg1_side {
        Side::Buy => best_ask.map_or(true, |a| leg1_price < a) && best_bid.map_or(true, |b| leg1_price > b),
        Side::Sell => best_bid.map_or(true, |b| leg1_price > b) && best_ask.map_or(true, |a| leg1_price < a),
    };
    if !safe {
        return StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) };
    }
    actions.push(Action::Submit(NewOrderRequest {
        asset_id: s.asset_id,
        side: leg1_side,
        order_type: OrderType::Limit,
        tif: TimeInForce::Gtc,
        post_only: true,
        price: Some(leg1_price),
        qty,
    }));
    actions.push(Action::Submit(NewOrderRequest {
        asset_id: s.asset_id,
        side: leg1_side.opposite(),
        order_type: OrderType::Market,
        tif: TimeInForce::Ioc,
        post_only: false,
        price: None,
        qty,
    }));

    s.cumulative_volume += qty * mid * 2.0;
    let _ = agent;
    StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) }
}
