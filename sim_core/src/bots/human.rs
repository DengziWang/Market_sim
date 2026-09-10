//! Human-behavior bots: not a spec-mandated category, added to model actual
//! behavioral-finance biases rather than a rational (even if simplified) strategy.
//! `noise` (bots/noise.rs) is the spec's "baseline control group" and deliberately
//! stays simple/memoryless (Gode & Sunder zero-intelligence); these bots are the
//! opposite -- they remember their own trades and react in ways a rational or
//! mechanical bot wouldn't.
//!
//! Modeled biases, each with real behavioral-finance backing:
//!   - **Disposition effect**: hold losers too long, sell winners too early --
//!     probably the most robust documented bias in retail trading. Needs the bot's
//!     own average entry price, which nothing elsewhere in the engine tracks; see
//!     `avg_entry_price`'s inference logic in `step()`.
//!   - **Irregular, bursty attention**: a long, jittery background check-in interval
//!     (not a fixed wake_interval like every other bot type), with a much shorter
//!     "can't stop refreshing" interval after a big move/mood swing was just
//!     noticed. The engine's event-driven design can't interrupt an agent mid-sleep,
//!     so "alert-driven" attention is approximated as *becoming* more frequently
//!     attentive after a big move, checked at the agent's own next wake, rather than
//!     a true interrupt. Reaction delay (noticing vs. acting) reuses the existing
//!     per-bot-type `latency` config rather than adding a new mechanism.
//!   - **FOMO / panic (convex reaction)**: response scales faster than linearly with
//!     the *magnitude* of the triggering move/mood, with a dead zone below a
//!     threshold -- unlike `noise::Herding`'s flat linear response probability.
//!   - **"Hot hand" sizing**: bet size scales with the bot's own recent realized
//!     P&L, not just its cash budget.
//!   - **Round-number bias**: trade notional is rounded to a "nice" number (1/2/5 x
//!     10^k), not a raw computed float.
//!
//! Two variants differ only in what drives the directional signal:
//!   - `disposition_chaser`: recent price return (FOMO/panic on price moves).
//!   - `sentiment_herder`: the latent, price-independent `Market::sentiment`
//!     process (see market.rs), read with individual perception noise -- "mood"-
//!     driven herding, not just price-driven.

use rand::Rng;

use super::StepCtx;
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderType, Side, SimTime, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    DispositionChaser,
    SentimentHerder,
}

pub struct State {
    pub variant: Variant,
    pub asset_id: AssetId,
    pub background_wake_min: SimTime,
    pub background_wake_max: SimTime,
    pub alert_wake_interval: SimTime,
    pub signal_threshold: f64,
    pub fomo_exponent: f64,
    pub perception_noise: f64,
    pub order_notional: f64,
    pub max_position: f64,
    pub disposition_loss_hold: f64,
    pub disposition_gain_eagerness: f64,
    pub pnl_sizing_sensitivity: f64,
    pub max_cash_frac_per_trade: f64,
    pub last_price: Option<f64>,
    pub last_wake_time: Option<SimTime>,
    pub last_seen_inventory: f64,
    pub avg_entry_price: Option<f64>,
    pub recent_pnl_frac: f64,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::DispositionChaser => "disposition_chaser",
            Variant::SentimentHerder => "sentiment_herder",
        }
    }

    /// `initial_price`: converts dollar-notional size params into asset units --
    /// see market_maker.rs's from_params doc comment for why that matters.
    pub fn from_params(variant: Variant, asset_id: AssetId, initial_price: f64, p: &toml::Value) -> Self {
        let price = initial_price.max(1e-9);
        Self {
            variant,
            asset_id,
            background_wake_min: params::get_f64(p, "background_wake_min_secs", 60.0),
            background_wake_max: params::get_f64(p, "background_wake_max_secs", 900.0),
            alert_wake_interval: params::get_f64(p, "alert_wake_interval_secs", 20.0),
            // Default threshold means "a return since last check" for
            // disposition_chaser (a fraction) but "a mood reading" for
            // sentiment_herder (already normalized to [-1,1]) -- from_params picks
            // the right default per variant since the two scales aren't comparable.
            signal_threshold: params::get_f64(
                p,
                "signal_threshold",
                match variant {
                    Variant::DispositionChaser => 0.02,
                    Variant::SentimentHerder => 0.15,
                },
            ),
            fomo_exponent: params::get_f64(p, "fomo_exponent", 1.1),
            perception_noise: params::get_f64(p, "perception_noise", 0.25),
            order_notional: params::get_f64(p, "order_notional", 800.0),
            max_position: params::get_f64(p, "max_position_notional", 4000.0) / price,
            disposition_loss_hold: params::get_f64(p, "disposition_loss_hold", 0.35),
            disposition_gain_eagerness: params::get_f64(p, "disposition_gain_eagerness", 1.3),
            pnl_sizing_sensitivity: params::get_f64(p, "pnl_sizing_sensitivity", 0.8),
            // Even a panicking/euphoric person rarely deploys their *entire*
            // remaining cash into one single trade; also directly bounds how much any
            // one agent's single order can contribute to a cascade, regardless of how
            // extreme the computed intensity is.
            max_cash_frac_per_trade: params::get_f64(p, "max_cash_frac_per_trade", 0.15),
            last_price: None,
            last_wake_time: None,
            last_seen_inventory: 0.0,
            avg_entry_price: None,
            recent_pnl_frac: 0.0,
        }
    }
}

/// Round to the nearest "nice" number (1/2/5 x 10^k) -- how a person sizes a trade
/// ("put in $500", not "put in $483.17"), not how a formula's raw output looks.
fn round_to_nice(x: f64) -> f64 {
    if !(x > 0.0) || !x.is_finite() {
        return 0.0;
    }
    let exponent = x.log10().floor();
    let base = 10f64.powf(exponent);
    let fraction = x / base;
    let nice = if fraction < 1.5 {
        1.0
    } else if fraction < 3.5 {
        2.0
    } else if fraction < 7.5 {
        5.0
    } else {
        10.0
    };
    nice * base
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    let mut actions = Vec::new();
    let Some(mid) = ctx.market.mid_price(ctx.venue, s.asset_id) else {
        return StepResult { actions, next_wake: Some(ctx.now + s.background_wake_max) };
    };

    let inventory = agent.inventory_of(s.asset_id);

    // Infer fills from the observed inventory change since last wake (there's no
    // synchronous fill callback here, same constraint every other bot works under),
    // and use it to track a running average entry price and a decaying estimate of
    // recent realized P&L -- the two things the disposition effect and "hot hand"
    // sizing below need, and that nothing generic in AgentCommon tracks.
    let delta_inv = inventory - s.last_seen_inventory;
    if delta_inv > 1e-9 {
        let prev_qty = s.last_seen_inventory.max(0.0);
        let new_qty = prev_qty + delta_inv;
        let prev_cost = s.avg_entry_price.unwrap_or(mid) * prev_qty;
        s.avg_entry_price = Some((prev_cost + delta_inv * mid) / new_qty.max(1e-9));
    } else if delta_inv < -1e-9 {
        if let Some(entry) = s.avg_entry_price {
            let pnl_frac = (mid - entry) / entry.max(1e-9);
            s.recent_pnl_frac = 0.7 * s.recent_pnl_frac + 0.3 * pnl_frac;
        }
        if inventory <= 1e-9 {
            s.avg_entry_price = None;
        }
    }
    s.last_seen_inventory = inventory;

    let raw_signal = match s.variant {
        Variant::DispositionChaser => {
            let prev = s.last_price.unwrap_or(mid);
            let raw_return = (mid - prev) / prev.max(1e-9);
            // Irregular attention means the gap since this agent's own last wake
            // varies a lot (60-900s by default) -- comparing raw return straight to a
            // fixed threshold makes an agent that happened to sleep longer see an
            // artificially bigger "move" for the same underlying drift rate, not a
            // real signal. Normalize by sqrt(elapsed / reference), consistent with
            // how return volatility scales with time in a random walk, so a return
            // observed after the *typical* gap keeps a ~1x multiplier while a return
            // spanning an unusually long gap is scaled back down. Without this, a
            // large population sharing a wide range of check-in intervals could
            // produce a feedback loop: bots with longer gaps overreact, push price
            // further, and *other* bots' next (possibly also-long) gap then spans an
            // even bigger apparent move -- this happened in testing (see git history).
            let reference_interval = (s.background_wake_min + s.background_wake_max) / 2.0;
            let elapsed = ctx.now - s.last_wake_time.unwrap_or(ctx.now);
            let time_norm = (elapsed / reference_interval.max(1e-9)).sqrt().max(0.3);
            raw_return / time_norm
        }
        Variant::SentimentHerder => {
            let true_sentiment = *ctx.market.sentiment.get(&s.asset_id).unwrap_or(&0.0);
            let noise = agent.rng.gen_range(-s.perception_noise..s.perception_noise);
            (true_sentiment + noise).clamp(-1.0, 1.0)
        }
    };
    s.last_price = Some(mid);
    s.last_wake_time = Some(ctx.now);

    // Convex (not linear) response with a dead zone: no reaction to small moves,
    // then intensity grows faster than the signal itself past the threshold --
    // FOMO/panic, not measured trend-following.
    let magnitude = raw_signal.abs();
    let alert = magnitude > s.signal_threshold;
    let excess = (magnitude / s.signal_threshold.max(1e-9) - 1.0).max(0.0);
    // Capped well below the earlier 4.0: at intensity=4 a single agent's notional
    // could reach (1+4)*2.5 = 12.5x its own base size, which combined with a large,
    // synchronized population (many agents reading the same shared sentiment/price
    // signal) was enough to overwhelm book depth on thinner assets in testing --
    // see git history.
    let intensity = excess.powf(s.fomo_exponent).min(1.2);

    if intensity > 0.0 {
        let side = if raw_signal > 0.0 { Side::Buy } else { Side::Sell };
        // "Hot hand": bigger bets after recent wins, smaller/timider after losses.
        let pnl_size_mult = (1.0 + s.pnl_sizing_sensitivity * s.recent_pnl_frac).clamp(0.2, 2.5);
        let notional = round_to_nice(s.order_notional * (1.0 + intensity) * pnl_size_mult);
        let mut qty = notional / mid.max(1e-9);

        match side {
            Side::Sell => {
                // Disposition effect: only modulates exits. Reluctant to sell at a
                // loss (hoping to "get back to even"), eager to sell at a gain (lock
                // it in) -- the opposite of what a rational risk-manager would do.
                if let Some(entry) = s.avg_entry_price {
                    qty *= if mid < entry { s.disposition_loss_hold } else { s.disposition_gain_eagerness };
                }
                // Never short: real retail traders overwhelmingly don't, and (like
                // noise.rs) an unconstrained sell here would be unlimited,
                // uncapitalized short pressure -- see git history for why that
                // matters, not just style.
                qty = qty.min(inventory.max(0.0));
            }
            Side::Buy => {
                qty = qty.min((s.max_position - inventory).max(0.0));
                let cash_budget = agent.cash * s.max_cash_frac_per_trade;
                if cash_budget < mid * qty {
                    qty = (cash_budget / mid.max(1e-9)).max(0.0);
                }
            }
        }

        if qty > 1e-9 {
            actions.push(Action::Submit(NewOrderRequest {
                asset_id: s.asset_id,
                side,
                order_type: OrderType::Market,
                tif: TimeInForce::Ioc,
                post_only: false,
                price: None,
                qty,
            }));
        }
    }

    // Irregular attention: a long, jittery background gap normally, a short one
    // ("can't stop refreshing") right after noticing a big move/mood swing.
    let next_wake = if alert {
        ctx.now + s.alert_wake_interval * agent.rng.gen_range(0.5..1.5)
    } else {
        ctx.now + agent.rng.gen_range(s.background_wake_min..s.background_wake_max)
    };

    StepResult { actions, next_wake: Some(next_wake) }
}
