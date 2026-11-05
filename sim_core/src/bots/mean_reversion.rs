//! Mean-reversion bots: 4.4 in the spec (a separate category from momentum).
//!
//! **Per-agent parameter jitter (2026-10-25)**: see momentum.rs's module doc
//! comment for the full reasoning -- `from_params` took no `rng` here either,
//! so every agent of a given variant+asset was an exact clone sharing one
//! `window`/`num_std`/`deviation_threshold`/`entry_z`/`exit_z`/`wake_interval`,
//! reading the same market and reacting in perfect lockstep. Mean-reversion
//! clones are a subtler version of the same bug as momentum clones: this bot
//! type is supposed to be the market's counter-pressure against sustained
//! trends, but a population of identical mean-reverters all enters and exits
//! at exactly the same z-score/deviation and the same instant, which makes
//! their own counter-pressure itself a synchronized, bursty push rather than
//! the smooth, staggered buying-the-dip a real population of value investors
//! with different convictions about "how far is too far" would provide.
//!
//! **Fundamental-value bias for Bollinger/ZScore (2026-10-25, direct user
//! request: make bots more successful)**: a per-variant growth breakdown at
//! human_80pct (200-leg full window) showed `ou_fair_value` -- the one
//! variant here that trades against the sim's ground-truth fundamental
//! value -- near breakeven (+0.99%), against -30% to -35% for Bollinger and
//! ZScore, which see only the noisy traded-price series and have no concept
//! of where price *should* be. The same fix already validated on grid.rs
//! (see that file's own "Fundamental-value bias" doc comment -- same
//! mechanism, same reasoning, reused here): blend each variant's own rolling
//! `mean` with the known fundamental value by `fundamental_bias_frac` before
//! computing bands/z-score, so a population that still reacts primarily to
//! the traded price is no longer *blind* to which side of fair value it's
//! trading on. Continuous, applied to the reference level only -- never
//! forces an exit, so the disposition/entry-exit logic each variant already
//! has is untouched.

use super::{RollingWindow, StepCtx};
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderType, Side, SimTime, TimeInForce};
use rand::Rng;

/// Same jitter helper as momentum.rs (not shared via `super` since the two
/// modules otherwise have no dependency on each other and this is a two-line
/// function) -- draws `default * Uniform(1-frac, 1+frac)`.
fn jittered(rng: &mut impl Rng, default: f64, frac: f64) -> f64 {
    let frac = frac.clamp(0.0, 0.95);
    default * rng.gen_range((1.0 - frac)..(1.0 + frac))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    Bollinger,
    OuFairValue,
    ZScore,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Fraction of the rolling-mean/fundamental-value gap that Bollinger's
    /// and ZScore's own reference level leans toward fundamental value --
    /// see module doc comment's "Fundamental-value bias" section. Unused by
    /// OuFairValue, which already trades directly against fundamental value.
    pub fundamental_bias_frac: f64,
    /// OuFairValue only ("value fund" reconstruction): how strongly position
    /// size scales up as `dev` moves past `deviation_threshold` -- 0.0 means
    /// OuFairValue stays the old fixed-size +-max_pos flip. See `step()`'s
    /// `size_mult` formula.
    pub deviation_scale_factor: f64,
    /// OuFairValue only: hard cap on `size_mult`, so the scale-up above can't
    /// run away unbounded as `dev` grows.
    pub max_position_scale: f64,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::Bollinger => "bollinger",
            Variant::OuFairValue => "ou_fair_value",
            Variant::ZScore => "zscore",
        }
    }

    /// `initial_price`: see momentum.rs's from_params doc comment -- converts a
    /// dollar-notional default position size into asset units per-asset.
    /// `rng`: per-agent jitter -- see this file's module doc comment.
    pub fn from_params(variant: Variant, asset_id: AssetId, initial_price: f64, p: &toml::Value, rng: &mut impl Rng) -> Self {
        let window_n = (jittered(rng, params::get_u32(p, "window", 20) as f64, 0.35).round() as usize).max(2);
        let order_qty = params::get_f64(p, "order_notional", 3000.0) / initial_price.max(1e-9);
        // entry_z jittered first, exit_z derived relative to it (kept comfortably
        // below so "exit" always means a less extreme reading than "entry" for
        // every agent, not just on average).
        let entry_z = jittered(rng, params::get_f64(p, "entry_z", 1.5), 0.3).max(0.5);
        let exit_z = jittered(rng, params::get_f64(p, "exit_z", 0.3), 0.3).clamp(0.05, entry_z - 0.2);
        Self {
            variant,
            asset_id,
            // (2026-11-06, direct user request: "make everyone jitter every
            // time") `wake_interval` stores the un-jittered base; the +/-40%
            // jitter is re-rolled every wake in `step()` instead of drawn
            // once here and fixed for life -- same every-wake desync
            // mechanism market_maker.rs uses. entry_z/exit_z above are
            // unaffected (still once-per-agent): that jitter gives each
            // agent a stable, distinct reading of the signal, a separate
            // concern from wake-clock desync.
            wake_interval: params::get_f64(p, "wake_interval_secs", 3.0).max(0.5),
            order_qty,
            allow_short: params::get_bool(p, "allow_short", true),
            window: RollingWindow::new(window_n),
            num_std: jittered(rng, params::get_f64(p, "num_std", 2.0), 0.3).max(0.5),
            deviation_threshold: jittered(rng, params::get_f64(p, "deviation_threshold", 0.01), 0.3).max(1e-4),
            entry_z,
            exit_z,
            in_position: false,
            // 0.3: same modest-lean default as grid.rs's own fundamental_bias_frac.
            fundamental_bias_frac: params::get_f64(p, "fundamental_bias_frac", 0.3),
            // Off by default (0.0): a scenario has to opt an OuFairValue group
            // into "value fund" behavior explicitly.
            deviation_scale_factor: params::get_f64(p, "deviation_scale_factor", 0.0),
            max_position_scale: params::get_f64(p, "max_position_scale", 10.0).max(1.0),
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
        return StepResult { actions, next_wake: Some(ctx.now + jittered(&mut agent.rng, s.wake_interval, 0.4).max(0.5)) };
    };
    let inventory = agent.inventory_of(s.asset_id);
    let max_pos = s.order_qty;

    match s.variant {
        Variant::Bollinger => {
            s.window.push(mid);
            if let (Some(raw_mean), Some(std)) = (s.window.mean(), s.window.std()) {
                if s.window.is_full() && std > 1e-9 {
                    // Fundamental-value bias -- see module doc comment.
                    let fair = *ctx.market.fundamental_value.get(&s.asset_id).unwrap_or(&raw_mean);
                    let mean = raw_mean + s.fundamental_bias_frac * (fair - raw_mean);
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
            //
            // Value-fund reconstruction (2026-11-01): position size scales with how
            // far past `deviation_threshold` the mispricing has gone, instead of the
            // old fixed +-max_pos flip -- the whole point of configuring this variant
            // as a small, well-capitalized "value fund" rather than just another
            // bounded mean-reverter. `size_mult` is 1.0 at the threshold itself and
            // grows linearly with the excess deviation, capped at `max_position_scale`.
            let fair = *ctx.market.fundamental_value.get(&s.asset_id).unwrap_or(&mid);
            let dev = (mid - fair) / fair.max(1e-9);
            let excess = (dev.abs() - s.deviation_threshold).max(0.0);
            let size_mult = (1.0 + s.deviation_scale_factor * excess / s.deviation_threshold.max(1e-9))
                .min(s.max_position_scale);
            let scaled_pos = max_pos * size_mult;
            let target = if dev > s.deviation_threshold {
                if s.allow_short { -scaled_pos } else { 0.0 }
            } else if dev < -s.deviation_threshold {
                // Cash cap on the buy side -- same bug class this session's rebalance()
                // fixes addressed elsewhere: don't let the scale-up above submit a buy
                // bigger than this agent can actually afford in cash.
                let affordable_target = inventory + (agent.cash / mid.max(1e-9)).max(0.0);
                scaled_pos.min(affordable_target)
            } else {
                0.0
            };
            if let Some(a) = rebalance(s.asset_id, inventory, target) {
                actions.push(a);
            }
        }
        Variant::ZScore => {
            s.window.push(mid);
            if let (Some(raw_mean), Some(std)) = (s.window.mean(), s.window.std()) {
                if s.window.is_full() && std > 1e-9 {
                    // Fundamental-value bias -- see module doc comment.
                    let fair = *ctx.market.fundamental_value.get(&s.asset_id).unwrap_or(&raw_mean);
                    let mean = raw_mean + s.fundamental_bias_frac * (fair - raw_mean);
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

    StepResult { actions, next_wake: Some(ctx.now + jittered(&mut agent.rng, s.wake_interval, 0.4).max(0.5)) }
}
