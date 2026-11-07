//! Grid bots: 4.5 in the spec.
//!
//! Simplification: the whole ladder is torn down and rebuilt every `wake_interval`
//! (like the market-maker requote pattern) rather than replacing only the individual
//! level that just filled. This still produces genuine grid-trading behavior (a
//! resting ladder of buy/sell levels around price that adapts to price/volatility)
//! at a fraction of the per-level order-tracking complexity.
//!
//! **Passive inventory taper (2026-10-25)**: measured across Sweep XXI (all 11
//! human_sweep_lite compositions), grid was consistently the worst- or
//! second-worst-performing bot type by both growth and bankruptcy rate --
//! bankruptcy as high as 33-38% at human_70pct/human_80pct/human_90pct, 2-3x
//! market_maker's own rate at the same compositions. Fix: shrink the qty
//! quoted on the side that would deepen an already-large inventory lean, the
//! same passive-taper pattern as market_maker's `MIN_TAPER`.
//!
//! A second mechanism was also tried and reverted: skewing the ladder's own
//! center price away from raw `mid` toward the de-risking side (the same way
//! market_maker's `InventorySkew` variant skews its quote reservation), on
//! the theory that a taper alone only slows new accumulation and does
//! nothing about marking an *existing* position's unrealized loss. Measured
//! head-to-head against the taper-only version and the original, both at
//! human_80pct across the same 7 seeds (a first pass compared against only 2
//! seeds and was misleadingly inconclusive -- one of those two happened to
//! land on an unusually low-bankruptcy seed for every variant tried, masking
//! the taper's real effect): original grid bankruptcy 43.9% -> taper-only
//! 34.3% (a real, meaningful drop) -> taper+skew 46.2% (worse than doing
//! nothing). The skew actively hurt because this ladder is a mean-reversion
//! bet: pulling the center down to make sell levels easier to hit during a
//! decline *realizes* the loss on the existing position at a depressed price
//! instead of waiting for the reversion the strategy is designed around, so
//! it locks in exactly the losses patience would have recovered. Taper alone
//! has no such effect -- it only throttles how much *more* exposure gets
//! added, never forces an exit -- which is why it's the one kept here.
//!
//! **Fundamental-value bias (2026-10-25, direct user request: make bots more
//! successful, not just less bankrupt)**: a per-(bot_type, variant) growth
//! breakdown at human_80pct (200-leg full window) showed a sharp, consistent
//! pattern across every bot type in this scenario: whichever variant has
//! genuine information about where price *should* be wins or breaks even
//! (mean_reversion's `ou_fair_value`, which trades against the sim's ground-
//! truth fundamental value, was the only mean_reversion variant near
//! breakeven at +0.99%, against -30% to -35% for the two variants that only
//! see the noisy traded-price series); whichever variant reacts to noisy
//! technical price action alone loses, often badly (every grid variant was
//! strongly negative here too). Fee/rebate magnitudes were checked directly
//! and ruled out (even a generous maker rebate moves equity by a few tenths
//! of a percent, three orders of magnitude too small to matter) -- the gap
//! is real trading P&L, and it tracks informational edge, not transaction
//! cost. Grid (all three variants: Static/Geometric/DynamicAtr) has zero
//! access to fundamental value today; it reads only the traded mid. This
//! gives it the same edge `ou_fair_value` already has: bias the ladder's
//! center toward the known fundamental value by `fundamental_bias_frac`, so
//! when price sits below fair value the ladder leans toward buying more
//! eagerly and selling less eagerly (and the mirror image above fair value)
//! -- a continuous lean toward the correct side of a reversion, not a forced
//! exit. This is a different mechanism from the reverted inventory-skew
//! above despite the superficial resemblance (both shift `center`): that one
//! reacted to this agent's *own position* and forced realizing a loss at a
//! bad price; this one reacts to the *asset's own fundamental value*, same
//! as `ou_fair_value`'s already-validated edge, and never forces an exit --
//! it only leans the existing ladder's framing, keeping the taper above
//! fully intact.

use rand::Rng;

use super::{RollingWindow, StepCtx};
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderId, OrderType, Side, SimTime, TimeInForce};

/// Floor for the passive inventory taper below -- see this module's doc
/// comment. Matches market_maker.rs's own `MIN_TAPER`: never fully stop
/// quoting a side, just shrink it, so the ladder doesn't freeze solid.
const MIN_TAPER: f64 = 0.1;

/// `Variant::Passive` only -- fraction of available cash/inventory actually
/// committed to the ladder on each rebuild. Not literally 100%: leaves a
/// sliver of slack so a fee or float-rounding error at the exact boundary
/// doesn't reject an order outright.
const PASSIVE_COMMIT_FRAC: f64 = 0.98;

/// `Variant::Passive` only -- caps how much gets re-committed to the ladder
/// each rebuild at roughly market_maker's own typical capital scale (direct
/// user request, 2026-11-07: "lower their inventory to about mm level"),
/// fixing an unbounded-reinvestment runaway: recommitting ~98% of *current*
/// (already-grown) cash every rebuild compounds any real edge geometrically
/// -- one full 100,000s sweep25 session took a single grid agent from a
/// $24k seed to $5.25M, ~216x, over about 20 rebuild cycles. Cash or
/// inventory beyond this cap just sits uncommitted instead of continuing to
/// compound the stake. market_maker's own cash_range in this project is
/// [30000, 120000]; 75000 is its midpoint.
const PASSIVE_MAX_COMMIT_NOTIONAL: f64 = 75000.0;

/// `Variant::Passive` only -- its fixed offset schedule, as fractions of mid:
/// 10%, 15%, 20%, ..., up to `n_levels` steps of 5 points starting at 10
/// (direct user request, 2026-11-07, prompted by seeing real BTC orders
/// resting thousands of dollars from mid on a live exchange: "layered with
/// 10 15 20 25 30 35 40 45 50 over and the same under start price level").
/// `lvl` is 1-indexed, same convention as the other three variants' loops.
fn passive_offset_pct(lvl: u32) -> f64 {
    0.05 * (lvl as f64 + 1.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    Static,
    Geometric,
    DynamicAtr,
    /// A real-world "set and forget" retail grid bot (direct user request,
    /// 2026-11-07): a wide, layered ladder at a fixed percentage schedule
    /// (`passive_offset_pct`) rather than uniform spacing, every layer on
    /// each side targeting the same net-worth goal (an earlier
    /// thin-middle/thick-edge version was tried and replaced with this
    /// uniform one, also direct user request). Like the three variants
    /// above, the whole ladder is torn down and rebuilt
    /// every wake -- the user's own framing was "completely changing
    /// prices" every `wake_interval`, which defaults far longer here (500s)
    /// so it mostly just sits. The one behavioral difference from the other
    /// three: submission order. This agent's own resting orders disappear
    /// only by filling (nothing else cancels them mid-interval), so by the
    /// time a rebuild runs, each level that's currently empty is one that
    /// filled since the last rebuild -- "wounded." The farthest wounded
    /// level is resubmitted first, then the next-farthest, and so on
    /// inward, before any level that was still resting; see `step`'s
    /// sorting of `level_order` below.
    Passive,
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
    /// Fraction of the mid/fundamental-value gap the ladder's center leans
    /// toward fundamental value -- see module doc comment's
    /// "Fundamental-value bias" section. 0 = ignore fundamental entirely
    /// (original behavior); 1 = fully re-center on fundamental value.
    pub fundamental_bias_frac: f64,
    /// `Variant::Passive` only -- bounds for this agent's own next-wake
    /// interval, redrawn uniformly every wake (direct user request,
    /// 2026-11-07: 5000s mean, 3500-6500s range -- up from the original flat
    /// 500s). Per-agent jitter, not a single shared interval, so a
    /// population of these bots doesn't all rebuild in lockstep.
    pub passive_wake_lo: SimTime,
    pub passive_wake_hi: SimTime,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::Static => "static",
            Variant::Geometric => "geometric",
            Variant::DynamicAtr => "dynamic_atr",
            Variant::Passive => "passive",
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
        //
        // Widening spacing_pct from 0.4% to 1% was also tried and reverted
        // (2026-10-25): the theory was that 0.4% only gave ~6-10x this
        // scenario's own diffusive per-wake sigma as a margin, too thin once
        // sustained trend persistence is layered on top. Measured head-to-head
        // at human_80pct across 7 seeds against the taper-only version: bot
        // bankruptcy *rose* to 22.3% (worse than even the unfixed original's
        // 15.1%) and market_maker's own growth cratered from -19.3% to -39.5%
        // -- widening grid's spacing reshaped the liquidity landscape those
        // other bot types trade against, with cascading effects nothing in
        // this multi-agent market can be assumed immune to. Reverted to the
        // original 0.4%/0.5 defaults; only the taper (below) is kept.
        let spacing_abs = params::get_f64(p, "spacing_abs", initial_price.max(1e-9) * 0.004);
        // Passive's own defaults differ sharply from the other three
        // variants' (see `Variant::Passive`'s doc comment): a long,
        // mostly-idle wake interval instead of a tight requote loop, and 9
        // levels to match its fixed 10%-50%-by-5 offset schedule rather than
        // the other variants' 5-level default.
        let default_wake = if matches!(variant, Variant::Passive) { 5000.0 } else { 10.0 };
        let default_n_levels = if matches!(variant, Variant::Passive) { 9 } else { 5 };
        let passive_wake_range = params::get_range(p, "wake_interval_range_secs", (3500.0, 6500.0));
        Self {
            variant,
            asset_id,
            wake_interval: params::get_f64(p, "wake_interval_secs", default_wake),
            order_qty,
            n_levels: params::get_u32(p, "n_levels", default_n_levels),
            spacing_abs,
            spacing_pct: params::get_f64(p, "spacing_pct", 0.004),
            atr_window: RollingWindow::new(params::get_u32(p, "atr_window", 14) as usize),
            atr_multiplier: params::get_f64(p, "atr_multiplier", 0.5),
            last_price: None,
            active_order_ids: Vec::new(),
            // 0.3: a modest lean, not a full re-center -- keeps this a grid
            // strategy that follows the traded price, just no longer blind
            // to which side of fair value it's trading on. Unused by
            // Passive, which recenters on raw mid every rebuild instead.
            fundamental_bias_frac: params::get_f64(p, "fundamental_bias_frac", 0.3),
            passive_wake_lo: passive_wake_range.0,
            passive_wake_hi: passive_wake_range.1,
        }
    }
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    let mut actions: Vec<Action> = s.active_order_ids.drain(..).map(|id| Action::Cancel(id, s.asset_id)).collect();
    let is_passive = matches!(s.variant, Variant::Passive);
    // `Variant::Passive` only -- redrawn fresh every wake (direct user
    // request, 2026-11-07: 5000s mean, 3500-6500s range) rather than a fixed
    // interval, so a population of these bots doesn't rebuild in lockstep --
    // see `State::passive_wake_lo`'s doc comment. The other three variants
    // keep their original fixed `wake_interval`.
    let next_wake = Some(if is_passive { ctx.now + agent.rng.gen_range(s.passive_wake_lo..s.passive_wake_hi) } else { ctx.now + s.wake_interval });

    let Some(mid) = ctx.market.mid_price(ctx.venue, s.asset_id) else {
        return StepResult { actions, next_wake };
    };
    if let Some(prev) = s.last_price {
        s.atr_window.push((mid - prev).abs());
    }
    s.last_price = Some(mid);

    let spacing = match s.variant {
        Variant::Static => s.spacing_abs,
        Variant::Geometric => mid * s.spacing_pct,
        Variant::DynamicAtr => (s.atr_window.mean().unwrap_or(mid * s.spacing_pct) * s.atr_multiplier).max(mid * 0.0005),
        // Unused -- Passive computes its own per-level offset directly from
        // `passive_offset_pct` in the loop below, not from one shared value.
        Variant::Passive => 0.0,
    };
    if spacing <= 0.0 && !is_passive {
        return StepResult { actions, next_wake };
    }

    let inventory = agent.inventory_of(s.asset_id);
    let max_inventory = s.order_qty * s.n_levels as f64;

    // Fundamental-value bias -- see module doc comment. Leans the ladder's
    // center toward the known fundamental value: below fair value, center
    // shifts up (buy levels sit closer to the live mid, more fillable --
    // buy more eagerly; sell levels sit further above it -- sell less
    // eagerly), and the mirror image above fair value. Continuous, and
    // never forces an exit the way the reverted inventory-skew did. Passive
    // skips this and recenters on raw `mid` every rebuild instead -- it has
    // no risk-management agenda, just a wide, fixed-shape static ladder.
    let fair = *ctx.market.fundamental_value.get(&s.asset_id).unwrap_or(&mid);
    let center = if is_passive { mid } else { mid + s.fundamental_bias_frac * (fair - mid) };

    // Passive taper -- see module doc comment. inv_norm > 0 means already net
    // long (buying more would deepen that lean), < 0 means already net short.
    // Each side's own qty shrinks linearly as the lean grows, floored at
    // MIN_TAPER rather than cut off entirely, so the ladder stays alive on
    // both sides but stops adding full size to a position already working
    // against it. (Unrelated to `Variant::Passive` despite the name overlap
    // -- this taper applies to every variant except Passive, which uses its
    // own cash/inventory-committed sizing below instead.)
    let inv_norm = (inventory / max_inventory).clamp(-1.0, 1.0);
    let base_buy_qty = s.order_qty * (1.0 - inv_norm.max(0.0)).max(MIN_TAPER);
    let base_sell_qty = s.order_qty * (1.0 - (-inv_norm).max(0.0)).max(MIN_TAPER);

    // `Variant::Passive` only -- "all their orders and cash are committed"
    // (direct user request): rather than a fixed per-level notional (the
    // other three variants' `order_qty`), every rebuild divides essentially
    // all of this agent's current cash (buy side) and current inventory
    // (sell side) evenly across the n_levels orders on that side -- every
    // layer targets the same net-worth goal (direct user request,
    // 2026-11-07, replacing an earlier thin-middle/thick-edge version). Not
    // literally 100% committed, see `PASSIVE_COMMIT_FRAC`.
    let passive_level_weight = 1.0 / s.n_levels.max(1) as f64;
    let passive_sell_inventory_cap = if mid > 0.0 { PASSIVE_MAX_COMMIT_NOTIONAL / mid } else { f64::MAX };
    let passive_buy_notional_total = agent.cash.min(PASSIVE_MAX_COMMIT_NOTIONAL) * PASSIVE_COMMIT_FRAC;
    let passive_sell_qty_total = inventory.max(0.0).min(passive_sell_inventory_cap) * PASSIVE_COMMIT_FRAC;

    // Submission order -- see `Variant::Passive`'s doc comment: "always start
    // restocking orders of the furthest wounded order". The other three
    // variants aren't priority-sensitive (nothing distinguishes a fresh
    // rebuild from the last one), so they keep the original near-to-far order.
    let level_order: Vec<u32> = if is_passive { (1..=s.n_levels).rev().collect() } else { (1..=s.n_levels).collect() };

    for lvl in level_order {
        let offset = match s.variant {
            Variant::Geometric => mid * (1.0 - (1.0 - s.spacing_pct).powi(lvl as i32)),
            Variant::Passive => mid * passive_offset_pct(lvl),
            _ => spacing * lvl as f64,
        };
        // Clamped to mid (never above/below it respectively) so an extreme
        // fundamental deviation can only make a level more or less likely to
        // fill, never cross the live spread -- a post_only order that would
        // cross simply gets rejected by the book, same as market_maker's own
        // post_only quotes.
        let buy_price = (center - offset).min(mid);
        let sell_price = (center + offset).max(mid);
        let (buy_qty, sell_qty) = if is_passive {
            let buy_qty = if buy_price > 0.0 { (passive_buy_notional_total * passive_level_weight) / buy_price } else { 0.0 };
            (buy_qty, passive_sell_qty_total * passive_level_weight)
        } else {
            (base_buy_qty, base_sell_qty)
        };
        let position_caps_ok = is_passive || (inventory < max_inventory);
        if buy_price > 0.0 && buy_qty > 0.0 && position_caps_ok && agent.cash > buy_price * buy_qty {
            actions.push(Action::Submit(NewOrderRequest {
                asset_id: s.asset_id,
                side: Side::Buy,
                order_type: OrderType::Limit,
                tif: TimeInForce::Gtc,
                post_only: true,
                price: Some(buy_price),
                qty: buy_qty,
            }));
        }
        if sell_qty > 0.0 && (is_passive || inventory > -max_inventory) {
            actions.push(Action::Submit(NewOrderRequest {
                asset_id: s.asset_id,
                side: Side::Sell,
                order_type: OrderType::Limit,
                tif: TimeInForce::Gtc,
                post_only: true,
                price: Some(sell_price),
                qty: sell_qty,
            }));
        }
    }

    StepResult { actions, next_wake }
}

pub fn on_order_id(s: &mut State, _req: &NewOrderRequest, id: OrderId) {
    s.active_order_ids.push(id);
}
