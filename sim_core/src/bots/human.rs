//! Human-behavior bot: not a spec-mandated category, added to model actual
//! behavioral-finance biases rather than a rational (even if simplified) strategy.
//! `noise` (bots/noise.rs) is the spec's "baseline control group" and deliberately
//! stays simple/memoryless (Gode & Sunder zero-intelligence); this bot is the
//! opposite -- it remembers its own trades and reacts in ways a rational or
//! mechanical bot wouldn't.
//!
//! A single variant carries every modeled bias at once (this used to be split across
//! two variants -- `disposition_chaser` reacting only to price trend,
//! `sentiment_herder` only to mood -- merged into one bot that does both, since a
//! real retail trader isn't cleanly one or the other):
//!   - **Reacts to both price trend and sentiment**: the directional signal blends a
//!     trend/mean-reversion component (see "chartist vs. fundamentalist" below) with
//!     the latent, price-independent `Market::sentiment` process (see market.rs),
//!     each normalized by its own threshold and perceived with individual noise --
//!     see `raw_signal` in `step()`.
//!   - **Chartist-vs-fundamentalist strategy switching (2026-09-13)**: real retail
//!     traders don't consistently trust momentum ("it's going up, buy") or mean-
//!     reversion ("it's overpriced, sell") -- they drift toward whichever has
//!     recently been paying off, the same mechanism Brock & Hommes (1998,
//!     *Heterogeneous beliefs and routes to chaos in a simple asset pricing model*,
//!     JEDC) use to explain endogenous bubbles/crashes from a population that keeps
//!     re-weighting itself toward the currently-winning strategy. Each agent tracks
//!     a rolling "fitness" (realized hypothetical payoff, decayed like
//!     `recent_pnl_frac` below) for a chartist (momentum) and a fundamentalist
//!     (mean-reversion toward the market's own known-but-not-directly-observed-by-
//!     the-agent `fundamental_value`) signal, and blends between them via a
//!     logistic function of the fitness *difference* scaled by `intensity_of_choice`
//!     (Brock-Hommes' own parameter name) -- the sharper the recent gap in which
//!     strategy paid off, the more the agent leans into it. This replaces what used
//!     to be a single fixed momentum-only price component with something that can
//!     genuinely flip character mid-run, the way real retail sentiment does.
//!   - **Disposition effect via Cumulative Prospect Theory (2026-09-13)**: hold
//!     losers too long, sell winners too early -- probably the most robust
//!     documented bias in retail trading (Odean 1998). Previously modeled as two
//!     fixed multipliers (`disposition_loss_hold`/`disposition_gain_eagerness`)
//!     applied identically regardless of how big the gain/loss actually was. Now
//!     uses the real CPT value function from Tversky & Kahneman (1992, *Advances in
//!     Prospect Theory*): `v(r) = r^alpha` for gains, `v(r) = -lambda*(-r)^beta` for
//!     losses, with their own empirically-estimated defaults (alpha=beta=0.88,
//!     lambda=2.25) -- diminishing sensitivity in *both* domains (a bigger loss
//!     doesn't hurt proportionally more once you're already deep in one, which is
//!     the actual mathematical reason CPT predicts "hold and hope" rather than
//!     accepting a small loss) and loss aversion (losses hurt ~2.25x more than an
//!     equal-sized gain feels good). Needs the bot's own average entry price,
//!     which nothing elsewhere in the engine tracks; see `avg_entry_price`'s
//!     inference logic in `step()`.
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

use rand::Rng;

use super::StepCtx;
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderType, Side, SimTime, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    /// The only variant: an ordinary retail trader carrying every bias listed in the
    /// module doc comment at once. Kept as a one-value enum (rather than dropping
    /// `Variant` entirely) so `human` still fits the same `bot_type`/`variant`
    /// dispatch pattern every other bot type uses.
    Retail,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub variant: Variant,
    pub asset_id: AssetId,
    pub background_wake_min: SimTime,
    pub background_wake_max: SimTime,
    pub alert_wake_interval: SimTime,
    /// Return (fraction) since this agent's own last check-in that counts as a full
    /// "one unit" price signal -- see `raw_signal`'s normalization in `step()`.
    pub price_signal_threshold: f64,
    /// Deviation from the (agent-unobserved) fundamental value that counts as a
    /// full "one unit" mean-reversion signal -- the fundamentalist counterpart to
    /// `price_signal_threshold`'s chartist one.
    pub fundamentalist_signal_threshold: f64,
    /// Brock & Hommes' own parameter name: how sharply the population re-weights
    /// toward whichever of chartist/fundamentalist has recently paid off. 0 = always
    /// an even 50/50 blend regardless of fitness; higher = faster, more decisive
    /// switching toward the currently-winning strategy.
    pub intensity_of_choice: f64,
    /// Perceived sentiment reading (already normalized to roughly [-1, 1]) that
    /// counts as a full "one unit" mood signal.
    pub sentiment_signal_threshold: f64,
    /// Blend weight for the two normalized signal components: 1.0 = price-based
    /// (chartist/fundamentalist blend) only, 0.0 = sentiment only, 0.5 (default) =
    /// equal weight to both.
    pub price_weight: f64,
    pub fomo_exponent: f64,
    pub perception_noise: f64,
    pub order_notional: f64,
    pub max_position: f64,
    /// Tversky & Kahneman (1992) CPT value-function parameters -- see the module
    /// doc comment. `cpt_alpha`/`cpt_beta` are the diminishing-sensitivity curvature
    /// for gains/losses respectively (canonical estimate: both 0.88); `cpt_lambda`
    /// is the loss-aversion coefficient (canonical estimate: 2.25).
    pub cpt_alpha: f64,
    pub cpt_beta: f64,
    pub cpt_lambda: f64,
    /// Free scale constants (not from the CPT literature, which doesn't specify how
    /// *this* engine's sell-quantity multiplier should respond to a given CPT value
    /// -- these calibrate the mapping) converting the CPT value into how much the
    /// sell quantity shrinks (losses) or grows (gains) relative to the un-adjusted
    /// signal-driven quantity.
    pub cpt_loss_hold_scale: f64,
    pub cpt_gain_eagerness_scale: f64,
    pub pnl_sizing_sensitivity: f64,
    pub max_cash_frac_per_trade: f64,
    pub p_act_given_signal: f64,
    pub last_price: Option<f64>,
    pub last_wake_time: Option<SimTime>,
    pub last_seen_inventory: f64,
    pub avg_entry_price: Option<f64>,
    pub recent_pnl_frac: f64,
    /// Rolling, decayed fitness (realized hypothetical payoff) of the chartist and
    /// fundamentalist signals -- see the module doc comment's Brock-Hommes section.
    /// Updated each wake from what actually happened since the *previous* one, using
    /// `last_chartist_signal`/`last_fundamentalist_signal` (that previous wake's
    /// signals, i.e. what each strategy "predicted" then).
    pub chartist_fitness: f64,
    pub fundamentalist_fitness: f64,
    pub last_chartist_signal: f64,
    pub last_fundamentalist_signal: f64,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::Retail => "retail",
        }
    }

    /// `initial_price`: converts dollar-notional size params into asset units --
    /// see market_maker.rs's from_params doc comment for why that matters.
    pub fn from_params(variant: Variant, asset_id: AssetId, initial_price: f64, p: &toml::Value) -> Self {
        let price = initial_price.max(1e-9);
        Self {
            variant,
            asset_id,
            // Real people don't watch a chart continuously -- most check in a few
            // times an hour at most, and even less on a quiet asset. 3-40 min
            // (average ~22 min) between voluntary check-ins, vs. an earlier,
            // unrealistically chatty 1-15 min default.
            background_wake_min: params::get_f64(p, "background_wake_min_secs", 180.0),
            background_wake_max: params::get_f64(p, "background_wake_max_secs", 2400.0),
            // And even having noticed something (an alert wake), "check back again
            // in a couple of minutes" is more human than refreshing every 20s.
            alert_wake_interval: params::get_f64(p, "alert_wake_interval_secs", 120.0),
            price_signal_threshold: params::get_f64(p, "price_signal_threshold", 0.02),
            fundamentalist_signal_threshold: params::get_f64(p, "fundamentalist_signal_threshold", 0.02),
            intensity_of_choice: params::get_f64(p, "intensity_of_choice", 4.0),
            sentiment_signal_threshold: params::get_f64(p, "sentiment_signal_threshold", 0.15),
            price_weight: params::get_f64(p, "price_weight", 0.5).clamp(0.0, 1.0),
            fomo_exponent: params::get_f64(p, "fomo_exponent", 1.1),
            perception_noise: params::get_f64(p, "perception_noise", 0.25),
            // 200, not 800: dropped 4x at the user's request (2026-09-12), after
            // measuring that human bots already trade ~150x less *often* than the
            // noise::zero_intelligence baseline -- this makes each individual trade
            // smaller too, on top of being rarer.
            order_notional: params::get_f64(p, "order_notional", 200.0),
            max_position: params::get_f64(p, "max_position_notional", 4000.0) / price,
            cpt_alpha: params::get_f64(p, "cpt_alpha", 0.88),
            cpt_beta: params::get_f64(p, "cpt_beta", 0.88),
            cpt_lambda: params::get_f64(p, "cpt_lambda", 2.25),
            cpt_loss_hold_scale: params::get_f64(p, "cpt_loss_hold_scale", 2.0),
            cpt_gain_eagerness_scale: params::get_f64(p, "cpt_gain_eagerness_scale", 1.5),
            pnl_sizing_sensitivity: params::get_f64(p, "pnl_sizing_sensitivity", 0.8),
            // Even a panicking/euphoric person rarely deploys their *entire*
            // remaining cash into one single trade; also directly bounds how much any
            // one agent's single order can contribute to a cascade, regardless of how
            // extreme the computed intensity is.
            max_cash_frac_per_trade: params::get_f64(p, "max_cash_frac_per_trade", 0.15),
            // Status-quo bias / inertia: even noticing a signal worth reacting to
            // (intensity > 0), a real person often just... doesn't act on it. Without
            // this, every single qualifying wake traded, which is a lot more active
            // than an actual retail investor -- most check-ins end in "nah, I'll hold".
            p_act_given_signal: params::get_f64(p, "p_act_given_signal", 0.35),
            last_price: None,
            last_wake_time: None,
            last_seen_inventory: 0.0,
            avg_entry_price: None,
            recent_pnl_frac: 0.0,
            chartist_fitness: 0.0,
            fundamentalist_fitness: 0.0,
            last_chartist_signal: 0.0,
            last_fundamentalist_signal: 0.0,
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

/// Tversky & Kahneman (1992) cumulative prospect theory value function: diminishing
/// sensitivity in both domains (concave for gains, convex for losses -- curvature
/// `alpha`/`beta`), loss aversion (`lambda`, applied only to the loss side). `r` is
/// a return fraction (e.g. 0.10 = +10%, -0.10 = -10%), not a raw dollar amount --
/// this engine has no natural absolute-dollar reference point, but return fractions
/// are exactly what `avg_entry_price` already gives us for free.
fn cpt_value(r: f64, alpha: f64, beta: f64, lambda: f64) -> f64 {
    if r >= 0.0 {
        r.powf(alpha)
    } else {
        -lambda * (-r).powf(beta)
    }
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
    // recent realized P&L -- the two things the CPT-based disposition effect and
    // "hot hand" sizing below need, and that nothing generic in AgentCommon tracks.
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

    // Realized return since last check-in, time-normalized the same way regardless
    // of which downstream use it feeds (chartist signal below, and the Brock-Hommes
    // fitness update just after) -- see the original comment on why sqrt(elapsed/
    // reference) normalization matters: without it, agents with a longer gap since
    // their last wake see an artificially amplified "move" for the same underlying
    // drift rate, which produced a real feedback loop in testing (see git history).
    let prev_price = s.last_price.unwrap_or(mid);
    let raw_return = (mid - prev_price) / prev_price.max(1e-9);
    let reference_interval = (s.background_wake_min + s.background_wake_max) / 2.0;
    let elapsed = ctx.now - s.last_wake_time.unwrap_or(ctx.now);
    let time_norm = (elapsed / reference_interval.max(1e-9)).sqrt().max(0.3);

    // Brock & Hommes (1998) fitness update: score last wake's chartist and
    // fundamentalist signals against what the price *actually* did since then. A
    // signal that correctly anticipated the direction of `raw_return` gets a
    // positive contribution (scaled by the return's own magnitude, so being right
    // about a big move matters more than being right about noise); decayed like
    // `recent_pnl_frac` above so this tracks recent form, not a lifetime average --
    // consistent with Brock-Hommes' own emphasis on fast strategy-switching being
    // what drives the endogenous bubble/crash dynamics their model produces.
    s.chartist_fitness = 0.7 * s.chartist_fitness + 0.3 * (s.last_chartist_signal.signum() * raw_return);
    s.fundamentalist_fitness = 0.7 * s.fundamentalist_fitness + 0.3 * (s.last_fundamentalist_signal.signum() * raw_return);

    // Chartist (momentum) component: "it's been going up, so it'll keep going up."
    let price_component = (raw_return / time_norm) / s.price_signal_threshold.max(1e-9);

    // Fundamentalist (mean-reversion) component: "it's trading above what it's
    // really worth, so it'll fall back." Sign is flipped relative to the raw
    // deviation -- trading *above* fundamental value is a *bearish* signal here.
    // The agent doesn't literally observe `fundamental_value` (nothing in the
    // engine claims retail traders have insider access to ground truth) so much as
    // proxy it the way a real fundamentalist would -- forming a view from whatever
    // information they have -- but per this project's existing convention (e.g.
    // `efficiency.fundamental_deviation` in market_health already compares traded
    // price to this same ground-truth series to *measure* efficiency), using the
    // engine's actual latent fundamental value here is the direct, honest way to
    // give this strategy a real chance of being right, rather than inventing a
    // separate proxy signal with no principled anchor.
    let fair_value = *ctx.market.fundamental_value.get(&s.asset_id).unwrap_or(&mid);
    let deviation = (mid - fair_value) / fair_value.max(1e-9);
    let fundamentalist_component = -deviation / s.fundamentalist_signal_threshold.max(1e-9);

    // Logistic blend of the two by fitness difference (Brock-Hommes' "intensity of
    // choice" applied to a 2-strategy case simplifies to a sigmoid of the fitness
    // gap) -- 0.5 when the two have performed identically, sliding toward whichever
    // has been winning as the gap grows. f64::exp saturates to 0/+inf at extreme
    // inputs rather than panicking, so this stays well-behaved even if fitness
    // diverges further than typical.
    let fitness_gap = s.intensity_of_choice * (s.chartist_fitness - s.fundamentalist_fitness);
    let chartist_weight = 1.0 / (1.0 + (-fitness_gap).exp());
    let trend_component = chartist_weight * price_component + (1.0 - chartist_weight) * fundamentalist_component;

    // Mood component: the latent, price-independent sentiment process, read with
    // individual perception noise (no two people read "the mood" identically) and
    // likewise normalized by its own threshold.
    let true_sentiment = *ctx.market.sentiment.get(&s.asset_id).unwrap_or(&0.0);
    let perceived_sentiment = (true_sentiment + agent.rng.gen_range(-s.perception_noise..s.perception_noise)).clamp(-1.0, 1.0);
    let sentiment_component = perceived_sentiment / s.sentiment_signal_threshold.max(1e-9);

    // Blended, normalized signal: both components are already on a "1.0 = as strong
    // as this agent's own alert threshold" scale, so `magnitude > 1.0` below is the
    // combined alert condition regardless of which component (or mix of both) drove
    // it there -- a real trader doesn't cleanly separate "the price moved" from
    // "everyone's talking about it", they reinforce each other.
    let raw_signal = s.price_weight * trend_component + (1.0 - s.price_weight) * sentiment_component;
    s.last_price = Some(mid);
    s.last_wake_time = Some(ctx.now);
    s.last_chartist_signal = price_component;
    s.last_fundamentalist_signal = fundamentalist_component;

    // Convex (not linear) response with a dead zone: no reaction below the combined
    // threshold, then intensity grows faster than the signal itself past it --
    // FOMO/panic, not measured trend-following.
    let magnitude = raw_signal.abs();
    let alert = magnitude > 1.0;
    let excess = (magnitude - 1.0).max(0.0);
    // Capped well below an earlier, higher cap: at intensity=4 a single agent's
    // notional could reach (1+4)*2.5 = 12.5x its own base size, which combined with a
    // large, synchronized population (many agents reading the same shared sentiment/
    // price signal) was enough to overwhelm book depth on thinner assets in testing
    // -- see git history.
    let intensity = excess.powf(s.fomo_exponent).min(1.2);
    // Inertia: noticing a signal doesn't mean acting on it -- rolled once per wake,
    // not per potential trade, so it's a real "decided not to today", not a size cut.
    let acts_this_time = intensity > 0.0 && agent.rng.gen_bool(s.p_act_given_signal);

    if acts_this_time {
        let side = if raw_signal > 0.0 { Side::Buy } else { Side::Sell };
        // "Hot hand": bigger bets after recent wins, smaller/timider after losses.
        let pnl_size_mult = (1.0 + s.pnl_sizing_sensitivity * s.recent_pnl_frac).clamp(0.2, 2.5);
        let notional = round_to_nice(s.order_notional * (1.0 + intensity) * pnl_size_mult);
        let mut qty = notional / mid.max(1e-9);

        match side {
            Side::Sell => {
                // Disposition effect via real CPT (see module doc comment and
                // `cpt_value` above): a positive CPT value (gain, diminishing
                // sensitivity) makes selling *more* eager the bigger the gain is --
                // "lock it in" -- while a negative CPT value (loss, also diminishing
                // sensitivity but loss-averse-scaled) makes selling *less* eager the
                // deeper the loss is -- "hold and hope", the actual predicted
                // mechanism behind the disposition effect, not just an assumed
                // constant. Unlike the old fixed multipliers, this is genuinely
                // sensitive to *how big* the gain/loss is, not just its sign.
                if let Some(entry) = s.avg_entry_price {
                    let r = (mid - entry) / entry.max(1e-9);
                    let v = cpt_value(r, s.cpt_alpha, s.cpt_beta, s.cpt_lambda);
                    let sell_mult = if v >= 0.0 {
                        1.0 + s.cpt_gain_eagerness_scale * v
                    } else {
                        1.0 / (1.0 + s.cpt_loss_hold_scale * (-v))
                    };
                    qty *= sell_mult;
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
