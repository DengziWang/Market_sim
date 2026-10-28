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

use super::{RollingWindow, StepCtx};
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderId, OrderType, Side, SimTime, TimeInForce};

/// Floor for the passive inventory taper below -- see this module's doc
/// comment. Matches market_maker.rs's own `MIN_TAPER`: never fully stop
/// quoting a side, just shrink it, so the ladder doesn't freeze solid.
const MIN_TAPER: f64 = 0.1;

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
    /// Fraction of the mid/fundamental-value gap the ladder's center leans
    /// toward fundamental value -- see module doc comment's
    /// "Fundamental-value bias" section. 0 = ignore fundamental entirely
    /// (original behavior); 1 = fully re-center on fundamental value.
    pub fundamental_bias_frac: f64,
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
            // 0.3: a modest lean, not a full re-center -- keeps this a grid
            // strategy that follows the traded price, just no longer blind
            // to which side of fair value it's trading on.
            fundamental_bias_frac: params::get_f64(p, "fundamental_bias_frac", 0.3),
        }
    }
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    let mut actions: Vec<Action> = s.active_order_ids.drain(..).map(|id| Action::Cancel(id, s.asset_id)).collect();

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

    // Fundamental-value bias -- see module doc comment. Leans the ladder's
    // center toward the known fundamental value: below fair value, center
    // shifts up (buy levels sit closer to the live mid, more fillable --
    // buy more eagerly; sell levels sit further above it -- sell less
    // eagerly), and the mirror image above fair value. Continuous, and
    // never forces an exit the way the reverted inventory-skew did.
    let fair = *ctx.market.fundamental_value.get(&s.asset_id).unwrap_or(&mid);
    let center = mid + s.fundamental_bias_frac * (fair - mid);

    // Passive taper -- see module doc comment. inv_norm > 0 means already net
    // long (buying more would deepen that lean), < 0 means already net short.
    // Each side's own qty shrinks linearly as the lean grows, floored at
    // MIN_TAPER rather than cut off entirely, so the ladder stays alive on
    // both sides but stops adding full size to a position already working
    // against it.
    let inv_norm = (inventory / max_inventory).clamp(-1.0, 1.0);
    let buy_qty = s.order_qty * (1.0 - inv_norm.max(0.0)).max(MIN_TAPER);
    let sell_qty = s.order_qty * (1.0 - (-inv_norm).max(0.0)).max(MIN_TAPER);

    for lvl in 1..=s.n_levels {
        let offset = match s.variant {
            Variant::Geometric => mid * (1.0 - (1.0 - s.spacing_pct).powi(lvl as i32)),
            _ => spacing * lvl as f64,
        };
        // Clamped to mid (never above/below it respectively) so an extreme
        // fundamental deviation can only make a level more or less likely to
        // fill, never cross the live spread -- a post_only order that would
        // cross simply gets rejected by the book, same as market_maker's own
        // post_only quotes.
        let buy_price = (center - offset).min(mid);
        let sell_price = (center + offset).max(mid);
        if buy_price > 0.0 && inventory < max_inventory && agent.cash > buy_price * buy_qty {
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
        if inventory > -max_inventory {
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

    StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) }
}

pub fn on_order_id(s: &mut State, _req: &NewOrderRequest, id: OrderId) {
    s.active_order_ids.push(id);
}
