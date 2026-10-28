//! Momentum / trend-following bots: 4.3 in the spec. Each variant recomputes its
//! signal once per wake (the wake interval acts as the indicator's "bar" period,
//! matching how real periodic-poll trend bots operate) and rebalances inventory
//! toward a signed target position via an IOC market order.
//!
//! **Per-agent parameter jitter (2026-10-25)**: `from_params` used to take no
//! `rng` at all, so every agent of a given variant+asset (scenario counts run
//! 70-150+ per variant) was constructed with byte-for-byte identical lookback
//! windows, thresholds, and `wake_interval`. Since every instance also reads the
//! same market mid-price and starts from the same first-second wake jitter
//! (`sim.rs`), identical parameters mean identical state transitions forever --
//! a whole population of exact clones that flip from +max_pos to -max_pos (or
//! back) in the same ~1s window on every signal change, not 143 independent
//! trend-followers with different opinions about lookback length. Measured
//! directly: human-containing sweeps showed leg-to-leg price-return sign flips
//! at roughly half the rate a random walk would produce (flip_rate ~0.24-0.35
//! vs. the ~0.5 two independent symmetric increments would give), i.e. far more
//! sustained one-directional runs than the underlying fundamental-value process
//! alone would create -- consistent with synchronized, correlated order bursts
//! from clone populations reinforcing whatever direction they're already in.
//! This mirrors the exact reasoning already applied to `market_maker`'s
//! `derisk_inventory_frac`/`markout_horizon_secs` (see that file's own from_params:
//! "every market_maker agent shares the same market, so their fills' markouts"
//! needs jitter to differ) and to `human`'s many `rng.gen_range` propensity
//! draws -- this bot type was simply never given the same treatment. Every
//! period/window/threshold/`wake_interval` below is now drawn per-agent from a
//! range around the configured default (the default is always the *center* of
//! the range, so a scenario's existing tuning is preserved on average) rather
//! than applied as one shared constant.
//!
//! **Whipsaw/churn fixes (2026-10-25, direct user request to make bots more
//! successful)**: measured at human_80pct, momentum was mildly but
//! consistently net-negative (-4% to -9% growth across seeds) despite having
//! no missing risk-management mechanism (position size is already bounded to
//! a single `max_pos`, no accumulation risk the way grid's ladder had).
//! Found two sources of pure fee/spread bleed with zero informational edge,
//! both paid via this module's IOC *market* orders (taker fee + crossing the
//! spread, every single time):
//!
//! 1. **`MaCrossover` had no deadband.** `short > long` flipped the *entire*
//!    position the instant the two EMAs were on either side of each other --
//!    including by floating-point noise. Two continuously-updating EMAs
//!    chatter back and forth across their own crossover point constantly in a
//!    noisy market, and every chatter was a full round-trip (2x `max_pos`) at
//!    taker cost for a move that was never a real trend change. Every other
//!    variant in this file already has some form of hysteresis (RSI/Donchian
//!    hold `inventory` between thresholds; the `rsi_overbought`/`rsi_oversold`
//!    gap itself is a deadband) -- crossover was the one variant missing it.
//!    Fixed with `crossover_deadband_frac`: require the EMA gap to exceed this
//!    fraction of `mid` before flipping at all; inside the band, hold.
//! 2. **`MlOnline` forced flat in its own uncertain zone.** `prob_up` between
//!    0.45 and 0.55 isn't a bearish signal, it's "the model doesn't know" --
//!    yet the original `else { 0.0 }` branch closed any open position there
//!    and reopened once probability cleared 0.55/0.45 again, a full
//!    unnecessary round trip every time a weak online classifier (3 crude
//!    features, slow-learning weights) wobbled near 50%, which is often, by
//!    construction. Fixed to hold `inventory` in that zone instead, matching
//!    the pattern the RSI variant two cases above it already uses -- only the
//!    genuinely bearish-but-can't-short case (`prob_up < 0.45` with
//!    `allow_short == false`) still goes flat, since that's a deliberate
//!    risk-avoidance choice, not noise.
//!
//! **Fundamental-value bias for MaCrossover (2026-10-25)**: the deadband
//! above removed pure noise-churn, but `MaCrossover` was still the worst
//! momentum variant in a per-variant growth breakdown at human_80pct
//! (-16.59% over the full 200-leg window, against +2 to +3% for RSI/
//! Donchian). Same root issue as grid.rs and mean_reversion.rs (see either
//! file's own "Fundamental-value bias" doc comment): this variant's EMA
//! crossover reacts only to the noisy traded-price series, with no concept
//! of which side of fair value it's trading on. Fixed the same way: the
//! crossover gap (`short - long`) gets a `fundamental_bias_frac`-weighted
//! nudge toward favoring long when mid sits below fundamental value (and
//! short when above) -- both terms are in raw price units, so they combine
//! directly, no rescaling needed. Continuous, never forces an exit; the
//! deadband above still applies to the *combined* gap.
//!
//! **Trailing stop for MaCrossover (2026-10-25)**: momentum's pooled growth
//! stayed mildly negative even after the fixes above, and a long ledger of
//! other attempts (more fundamental-value awareness on more variants,
//! population reallocation toward the best variant, confidence-scaled
//! sizing, self-derisking on this agent's own realized P&L, removing an
//! unrelated leveraged side-bet) each either backfired or made no
//! difference -- see this file's git history for the full, individually
//! validated ledger. Every variant in this file, including MaCrossover,
//! holds a position purely until the *opposite* signal fires, with no exit
//! discipline on the position's own trajectory in between -- unlike a real
//! trend-following strategy, which locks in gains with a trailing stop
//! rather than riding a winning position all the way back down to a
//! reversal signal before giving any of it back. This is a different kind
//! of mechanism from everything tried before: not about *when to enter*
//! (deadband, fundamental bias), not about *how much size* (sizing,
//! confidence-scaling, self-derisking, population mix) -- about *when to
//! exit early* based on this specific position's own unrealized P&L path.
//! Tracks `peak_favorable_frac`, the best unrealized return in this
//! position's favor since entry; once price gives back more than
//! `trailing_stop_frac` of that peak, the position flattens immediately,
//! independent of what the crossover signal says. A new entry (post-flatten
//! or on the next signal) resets the tracking.

use super::{RollingWindow, StepCtx};
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderType, Side, SimTime, TimeInForce};
use rand::Rng;

/// Draws `default * Uniform(1-frac, 1+frac)`, i.e. symmetric jitter around the
/// configured default rather than around some arbitrary fixed range -- so a
/// scenario override of the default still centers the jittered population on
/// the value the scenario author actually chose.
fn jittered(rng: &mut impl Rng, default: f64, frac: f64) -> f64 {
    let frac = frac.clamp(0.0, 0.95);
    default * rng.gen_range((1.0 - frac)..(1.0 + frac))
}

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
    /// Fraction of `mid` the EMA gap must exceed before flipping position --
    /// see module doc comment's "Whipsaw/churn fixes" section.
    pub crossover_deadband_frac: f64,
    /// Fraction of the mid/fundamental-value gap nudging MaCrossover's own
    /// EMA gap toward favoring the fundamentally-correct side -- see module
    /// doc comment's "Fundamental-value bias for MaCrossover" section.
    pub fundamental_bias_frac: f64,
    /// Trailing-stop state -- see module doc comment's "Trailing stop"
    /// section. Tracks the best unrealized-return-in-favor seen since the
    /// current position was opened; a drawdown from that peak beyond
    /// `trailing_stop_frac` force-flattens early, independent of the
    /// crossover signal.
    pub position_entry_price: Option<f64>,
    pub position_side: f64,
    pub peak_favorable_frac: f64,
    pub trailing_stop_frac: f64,
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
    ///
    /// `rng`: per-agent jitter on every lookback/threshold/wake_interval below --
    /// see the module doc comment's 2026-10-25 note for why a shared constant
    /// across a whole population of this variant was itself the bug.
    pub fn from_params(variant: Variant, asset_id: AssetId, initial_price: f64, p: &toml::Value, rng: &mut impl Rng) -> Self {
        // +/-35%: wide enough that a population of 100+ agents spans a genuinely
        // different range of lookback lengths (e.g. a 20-period EMA default spans
        // roughly 13-27 periods), not so wide that "momentum, ma_crossover" stops
        // meaning anything -- same philosophy as market_maker's jittered fields.
        let short_p = jittered(rng, params::get_f64(p, "ema_short_period", 5.0), 0.35).max(1.0);
        let long_p = jittered(rng, params::get_f64(p, "ema_long_period", 20.0), 0.35).max(short_p + 1.0);
        let donchian_n = (jittered(rng, params::get_u32(p, "donchian_window", 20) as f64, 0.35).round() as usize).max(2);
        let rsi_n = (jittered(rng, params::get_u32(p, "rsi_window", 14) as f64, 0.35).round() as usize).max(2);
        // 200 (2026-10-25, down from 3000): this bot commits its *entire*
        // position in one direction every signal flip via a single taker
        // IOC order -- unlike grid's own ladder (multiple levels), there's
        // no diversification here, so a large default gave momentum far
        // more per-trade risk concentration than its edge could support.
        // Measured directly at human_80pct on a fixed 10-seed sample (via
        // direct TOML param override, not separate wheel rebuilds per
        // value, to keep the comparison apples-to-apples -- an earlier pass
        // comparing differently-sized seed samples at each step wrongly
        // concluded 750 beat 375): mean pooled momentum growth was 3000ish
        // baseline -3.3%, 750 -1.22% (7/10 seeds negative), 375 -0.76%
        // (6/10), 200 -0.29% (5/10, effectively a coin flip around zero),
        // 100 -1.38% (8/10, worse again). 200 is the measured optimum, not
        // a monotonic "smaller is always better" relationship -- too little
        // size means too little edge/presence to matter, too much means
        // too much risk concentration, and 200 sits at the sweet spot
        // between them. Every reduction step also improved every *other*
        // bot type's own growth (shrinking momentum's footprint reduced how
        // much adverse-selection noise the rest of the market had to absorb
        // from its large one-shot directional bets). This doesn't add
        // edge, it shrinks the magnitude of a strategy whose own average
        // return is uncertain -- the same risk-reduction logic as grid's
        // taper.
        let order_qty = params::get_f64(p, "order_notional", 200.0) / initial_price.max(1e-9);
        // RSI overbought/oversold jittered independently (not symmetric around a
        // shared center) and clamped apart -- real signal-threshold traders don't
        // all agree on exactly where "overbought" starts either.
        let rsi_overbought = jittered(rng, params::get_f64(p, "rsi_overbought", 65.0), 0.1).clamp(55.0, 85.0);
        let rsi_oversold = jittered(rng, params::get_f64(p, "rsi_oversold", 35.0), 0.1).clamp(15.0, 45.0).min(rsi_overbought - 5.0);
        Self {
            variant,
            asset_id,
            // +/-40%: this is the field most directly responsible for synchronized
            // clone bursts (module doc comment) -- the widest jitter of the bunch,
            // since desynchronizing *when* agents act matters as much as *how*
            // they read the signal.
            wake_interval: jittered(rng, params::get_f64(p, "wake_interval_secs", 5.0), 0.4).max(0.5),
            order_qty,
            allow_short: params::get_bool(p, "allow_short", true),
            alpha_short: 2.0 / (short_p + 1.0),
            alpha_long: 2.0 / (long_p + 1.0),
            ema_short: None,
            ema_long: None,
            // 0.1% of mid: small enough that a genuine trend change (the EMAs
            // diverging by more than typical tick-to-tick mid noise) still
            // flips promptly, large enough to filter the two EMAs merely
            // chattering across their own crossover point.
            crossover_deadband_frac: params::get_f64(p, "crossover_deadband_frac", 0.001),
            // 0.3: same modest-lean default as grid.rs's/mean_reversion.rs's
            // own fundamental_bias_frac.
            fundamental_bias_frac: params::get_f64(p, "fundamental_bias_frac", 0.3),
            position_entry_price: None,
            position_side: 0.0,
            peak_favorable_frac: 0.0,
            // 0.5: give back at most half of the best favorable move seen
            // since entry before locking in what's left -- loose enough
            // not to flatten on ordinary noise around the peak, tight
            // enough to actually protect a real run-up.
            trailing_stop_frac: params::get_f64(p, "trailing_stop_frac", 0.5),
            donchian_window: RollingWindow::new(donchian_n),
            current_signal: 0.0,
            rsi_window: RollingWindow::new(rsi_n),
            last_price: None,
            rsi_overbought,
            rsi_oversold,
            weights: [0.0, 0.0, 0.0],
            bias: 0.0,
            learning_rate: jittered(rng, params::get_f64(p, "learning_rate", 0.05), 0.4).max(1e-4),
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
                // Fundamental-value bias -- see module doc comment. Nudges
                // the crossover gap toward favoring long when mid is below
                // fundamental value (and short when above); both terms are
                // raw price units, so they combine directly.
                let fair = *ctx.market.fundamental_value.get(&s.asset_id).unwrap_or(&mid);
                // Deadband -- see module doc comment's "Whipsaw/churn fixes".
                // Only flip once the gap clears a minimum fraction of mid;
                // inside the band, hold whatever position we already have
                // instead of chattering on noise.
                let deadband = mid * s.crossover_deadband_frac;
                let gap = (short - long) + s.fundamental_bias_frac * (fair - mid);
                let signal_target = if gap > deadband {
                    max_pos
                } else if gap < -deadband {
                    if s.allow_short { -max_pos } else { 0.0 }
                } else {
                    inventory
                };
                // Trailing stop -- see module doc comment. Update the peak
                // favorable excursion for whatever position is currently
                // open (track it even while just holding, not only right
                // after an entry), then force-flatten if price has given
                // back more than `trailing_stop_frac` of that peak.
                let mut target = signal_target;
                if s.position_side != 0.0 {
                    if let Some(entry) = s.position_entry_price {
                        let favorable = s.position_side * (mid - entry) / entry.max(1e-9);
                        s.peak_favorable_frac = s.peak_favorable_frac.max(favorable);
                        if s.peak_favorable_frac > 0.0 && favorable < s.peak_favorable_frac * (1.0 - s.trailing_stop_frac) {
                            target = 0.0;
                        }
                    }
                }
                // Track entry/reset whenever the realized position's side
                // changes (a fresh entry after a flip or a trailing-stop
                // flatten) -- same signum-of-inventory-after-rebalance idea
                // as this file's self-derisking attempt, scoped to this
                // single position's own trajectory instead.
                let new_side = if target > 1e-9 { 1.0 } else if target < -1e-9 { -1.0 } else { 0.0 };
                if new_side != s.position_side {
                    s.position_side = new_side;
                    s.position_entry_price = if new_side != 0.0 { Some(mid) } else { None };
                    s.peak_favorable_frac = 0.0;
                }
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
            // Hold inventory in the uncertain zone (0.45-0.55) instead of
            // forcing flat -- see module doc comment's "Whipsaw/churn fixes".
            // "The model doesn't know" is not a bearish signal; liquidating
            // here and re-entering once probability clears 0.55/0.45 again
            // is a pure, repeated round-trip cost with no edge behind it.
            // Only the genuinely bearish-but-can't-short case still goes
            // flat deliberately.
            let target = if prob_up > 0.55 {
                max_pos
            } else if prob_up < 0.45 {
                if s.allow_short { -max_pos } else { 0.0 }
            } else {
                inventory
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
