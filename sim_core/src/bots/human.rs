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
//!   - **Grouped crowd trend (2026-10-06 revert)**: every human agent is
//!     assigned to one of `n_trend_groups` groups at construction
//!     (`State::trend_group`, drawn directly from this agent's own
//!     construction-time RNG). Each group reads its own independent
//!     `Market::random_trend` line for the asset it trades (see that field's
//!     doc comment) -- agents in the same group read the literal same
//!     number, and disagreement comes entirely from `personal_trend_weight`
//!     (how much each individual leans on it), drawn once per agent so some
//!     agents are far more swept up in the crowd than others. This replaced
//!     a brief social-network-graph experiment (2026-10-05/06) that was
//!     reverted at the user's request; see git history/session notes for
//!     why, and note this is a best-effort reconstruction of the original
//!     group mechanism, not a byte-exact restore (the original code was
//!     never committed).
//!   - **Liquidity-aware order placement (2026-10-04)**: previously this bot
//!     always submitted an unconditional IOC market order -- accepting *any*
//!     available price, with no check at all. Now, before submitting, it looks
//!     at the book: if there's no resting order on the side it would need to hit
//!     (e.g. no resting ask at all when it wants to buy), or the best available
//!     price is too far from what this agent itself believes the asset is worth
//!     (`believed_value`, see below), it posts a resting limit order at
//!     `believed_value` instead of taking the bad/missing price -- GTC and
//!     post-only, so it only ever adds liquidity, never crosses. This is what
//!     lets a 100%-human market trade at all: previously every human order was a
//!     taker order with nothing to take, so a population with zero market-making
//!     bot types produced exactly zero trades for the entire run (measured,
//!     not assumed -- see git history). `believed_value` is derived from this
//!     agent's own `raw_signal` (already personalized: its own perceived
//!     sentiment noise, its own network neighbors' influence, its own
//!     `social_weight`), not the engine's raw ground-truth
//!     `fundamental_value` -- using ground truth directly would mean every agent
//!     silently agreed on the exact same "fair" price, which contradicts the
//!     whole point of this bot's per-agent heterogeneity. At most one resting
//!     buy and one resting sell order per agent at a time (`resting_buy_order`/
//!     `resting_sell_order`) -- placing a new one on a side cancels whatever this
//!     agent still had resting on that same side first, so stale orders from a
//!     check-in long ago never pile up.
//!   - **"Too smart to fail" fix (2026-10-07)**: measured in the `trendgroups_v4`
//!     sweep that human agents went bankrupt at essentially 0% while bots failed
//!     at ~55-70%, across every composition. Root cause wasn't "humans trade
//!     well" so much as every risk control being far tighter, in relative terms,
//!     than a real retail trader's: `order_notional`/`max_position_notional` are
//!     fixed dollar amounts (not scaled to this agent's own capital, which spans
//!     roughly $1k-$20k in practice), `max_cash_frac_per_trade` caps any one trade
//!     at 15% of current cash, and "hot hand" sizing *shrinks* bets after losses
//!     -- the opposite of the loss-chasing/doubling-down that drives most real
//!     retail blowups. Three new per-agent propensity values (below), each drawn
//!     independently and uniformly at construction like `personal_trend_weight`
//!     above, so *some* agents genuinely risk enough to blow up while most stay
//!     close to the original, well-risk-managed behavior -- not a population-wide
//!     toggle, since real retail populations aren't uniformly reckless either.
//!   - **Cash-constrained conviction -> leverage (2026-10-10)**: a fourth
//!     lever on the same "too smart to fail" problem, but NOT via the
//!     generic `LeverageCfg` piggyback (`agent.rs`/`sim.rs`) every other
//!     leveraged bot type uses -- tried that first, and it's wrong for this
//!     bot specifically. That piggyback opens a position in a *random*
//!     direction (`agent.rng.gen_bool(0.5)` in
//!     `liquidation_trader::open_leveraged_position`) on essentially every
//!     wake once none is already open, sized at a flat fraction of cash --
//!     completely decoupled from this agent's own signal, and at a scale
//!     (tens of thousands of dollars per position) wildly out of proportion
//!     to this bot's own normal trade size (`order_notional` defaults to
//!     $200). Measured directly: enabling it for humans reliably produced a
//!     runaway cascade (the random, oversized, directionless flow triggers
//!     FOMO reactions across the rest of the population, which triggers more
//!     of the same, compounding) and simulation wall-clock time exploded
//!     well past normal. A real retail trader doesn't lever up on a coin
//!     flip; they reach for borrowed money when they *believe* in a
//!     direction and are out of cash to act on that belief at spot. So
//!     instead: after the ordinary Buy-side cap above clips `qty` down,
//!     `shortfall_qty` (`desired_qty - qty`, the part this agent wanted to
//!     buy but couldn't afford) becomes a candidate for a leveraged top-up,
//!     gated on two things clearing their own bar at once -- how strongly
//!     this agent currently believes the direction (`raw_signal`'s own
//!     magnitude, i.e. its "reputation" for this asset in the sense of how
//!     much it trusts its own view right now, not a track record) past
//!     `leverage_signal_threshold`, and a per-agent
//!     `leverage_risk_appetite` roll (most agents' draw is low, so most
//!     convicted-and-cash-constrained moments still just... don't get
//!     leveraged, same as a real population where most retail traders never
//!     touch margin at all). Same direction as the spot order (never a
//!     separate bet), sized to exactly the shortfall (never more), and
//!     routed through the shared `liquidation_trader::
//!     open_leveraged_position_directed` core (refactored out of the
//!     piggyback's own entry logic) and the new `Action::SubmitMargin`
//!     (`agent.rs`) so this one action -- and only this one, not every
//!     order this agent submits -- takes the margin settlement path.

use rand::Rng;

use super::liquidation_trader;
use super::StepCtx;
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderId, OrderType, Side, SimTime, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    /// An ordinary retail trader carrying every bias listed in the module doc
    /// comment at once.
    Retail,
    /// Cheaper variant (2026-09-17), added specifically to replace `noise` as the
    /// bulk non-strategic population: keeps every bias `Retail` has *except* the
    /// Brock-Hommes dynamic chartist/fundamentalist fitness-switching, which is
    /// replaced by one fixed blend weight (`lite_blend_weight`). This removes, on
    /// *every* wake (not just acting ones): two EMA fitness updates and one
    /// `f64::exp` logistic call -- see `step()`'s `Variant::Lite` branch. Everything
    /// else (sentiment blend, CPT disposition effect, FOMO convexity, hot-hand
    /// sizing, round-number bias, bursty attention) is identical to `Retail`, so
    /// this is a genuinely simpler *mechanism* for the same *kind* of bot, not a
    /// tuned-down/randomized approximation of one -- unlike widening the wake
    /// interval (rejected: would only suppress an already-too-low human trade
    /// volume further) or a parameter-fuzzed `Retail` (no guarantee of actually
    /// landing near memoryless). `Retail` itself is left byte-for-byte unchanged so
    /// every existing sweep (human_sweep, human_sweep_2) stays exactly reproducible.
    Lite,
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
    /// `Variant::Lite` only: fixed chartist-vs-fundamentalist blend weight, used in
    /// place of the dynamic Brock-Hommes fitness-based switch (`Retail` ignores
    /// this field entirely). Same meaning as the dynamic `chartist_weight` it
    /// replaces: 1.0 = pure chartist/momentum, 0.0 = pure fundamentalist, 0.5 =
    /// even blend.
    pub lite_blend_weight: f64,
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
    /// Drawn once per agent, Uniform(1.0, `aggressive_sizing_max`) -- a direct
    /// multiplier on this agent's own `max_cash_frac_per_trade`/`max_position`,
    /// same convention as `personal_trend_weight` (a ready-to-use scale, not a
    /// 0-1 probability needing a separate scale constant). 1.0 (the floor) =
    /// exactly today's caps; most agents land well below `aggressive_sizing_max`
    /// since the draw is uniform across the whole range, but a minority land
    /// high enough to actually risk a large enough share of net worth to go
    /// bankrupt from trading losses alone -- see module doc comment's "too smart
    /// to fail" section for why this exists.
    pub aggressive_sizing_propensity: f64,
    /// Drawn once per agent, Uniform(0, `revenge_trading_max`). Blends this
    /// agent's own "hot hand" sizing response away from today's risk-*reducing*
    /// shrink-after-losses behavior and toward risk-*increasing* "get-even"
    /// loss-chasing (bet bigger, not smaller, right after a loss) -- 0 = exactly
    /// today's behavior, 1 = a full reversal (grows by the same magnitude a loss
    /// would otherwise shrink it).
    pub revenge_trading_propensity: f64,
    /// Drawn once per agent, Uniform(0, `price_chase_prob_max`) -- the per-attempt
    /// probability (rolled fresh each time the situation arises, not fixed per
    /// agent-level outcome like the two propensities above) that this agent
    /// impulsively crosses at a bad price with a market order instead of patiently
    /// posting a resting limit at its own believed value. 0 = never (today's
    /// always-patient behavior).
    pub price_chase_prob: f64,
    /// Drawn once per agent, Uniform(0, `leverage_risk_appetite_max`) -- how
    /// willing *this* agent is to borrow against its own conviction rather
    /// than wait/save up, when both `leverage_signal_threshold` (how
    /// strongly it currently believes the direction, i.e. its own
    /// `raw_signal`'s magnitude -- "reputation" in the sense of how much it
    /// currently trusts its own bullish/bearish view, not a track record)
    /// and this probability clear their bars at once -- see module doc
    /// comment's "cash-constrained conviction" leverage section. Most agents
    /// land low (leverage stays rare even among convicted agents); a minority
    /// land high enough to actually reach for it regularly.
    pub leverage_risk_appetite: f64,
    /// Drawn once per agent, Uniform(`leverage_min`, `leverage_max`) -- this
    /// agent's own leverage multiplier on the rare occasions it does borrow.
    /// Independent of `leverage_risk_appetite` (how *often* it reaches for
    /// leverage) -- this is how *much* leverage it reaches for when it does.
    pub leverage_amount: f64,
    /// How strongly `raw_signal`'s magnitude must clear the ordinary
    /// act-at-all dead zone (1.0) before leverage is even considered -- not
    /// per-agent (unlike the three fields above), since this represents "how
    /// convicted is convicted enough to borrow," a population-level bar, not
    /// individual heterogeneity.
    pub leverage_signal_threshold: f64,
    pub leverage_maintenance_margin_buffer: f64,
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
    /// This group's `Market::random_trend` value that counts as a full "one
    /// unit" signal, same normalization convention as
    /// `sentiment_signal_threshold`. A fixed per-variant constant (not
    /// randomized) -- the heterogeneity across agents comes from
    /// `personal_trend_weight` below and from which group they landed in,
    /// not from this.
    pub personal_trend_signal_threshold: f64,
    /// How much this agent's group crowd-trend value influences its *own*
    /// final signal, versus its private market-based signal (price/
    /// fundamentalist/sentiment blend, which already carries this agent's
    /// own perception noise -- its "individual random voice"). Drawn once
    /// per agent, independently, at construction (see `from_params`'s `rng`
    /// parameter) -- the source of cross-sectional disagreement even among
    /// agents in the same group: everyone in a group reads the literal same
    /// trend value but weights it completely differently. 0.0 = this agent
    /// ignores the crowd entirely; higher = more swept up in it.
    pub personal_trend_weight: f64,
    /// Which of `Market::n_trend_groups` groups this agent belongs to (see
    /// module doc comment) -- fixed at construction, never changes
    /// afterward.
    pub trend_group: u32,
    /// This agent's own currently-resting buy limit order, if any: `(order_id,
    /// asset_id)`. See the module doc comment's "Liquidity-aware order
    /// placement" section -- placing a new resting buy cancels whatever's here
    /// first, so this is never more than one order.
    pub resting_buy_order: Option<(OrderId, AssetId)>,
    /// Same as `resting_buy_order`, for this agent's resting sell side.
    pub resting_sell_order: Option<(OrderId, AssetId)>,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::Retail => "retail",
            Variant::Lite => "lite",
        }
    }

    /// `initial_price`: converts dollar-notional size params into asset units --
    /// see market_maker.rs's from_params doc comment for why that matters.
    /// `rng`: used to draw this specific agent's own `trend_group` and
    /// `personal_trend_weight` (see those fields' doc comments) -- the same
    /// construction-time RNG `sim.rs`'s `build_population` already uses for this
    /// agent's starting cash/leverage, so the draw is deterministic given a fixed
    /// run seed, same as everything else about population construction.
    /// `n_trend_groups`: passed in by the caller (`sim.rs::build_population`,
    /// via `compute_n_trend_groups`) rather than recomputed here, so every
    /// human agent in the same scenario draws from the same group range.
    pub fn from_params(variant: Variant, asset_id: AssetId, initial_price: f64, p: &toml::Value, n_trend_groups: u32, rng: &mut impl Rng) -> Self {
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
            lite_blend_weight: params::get_f64(p, "lite_blend_weight", 0.5).clamp(0.0, 1.0),
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
            personal_trend_signal_threshold: params::get_f64(p, "personal_trend_signal_threshold", 0.2),
            // Drawn once per agent from Uniform(0, max) -- "differently much", not
            // a single fixed weight applied identically to every agent (that would
            // make every human agent lean on the crowd the same amount, which is
            // just a fixed extra blend term, not genuine heterogeneity). Nonzero max
            // by default (not opt-in via TOML) -- meant to be an always-on source of
            // cross-sectional disagreement, not a dormant feature every existing
            // scenario config would need editing to exercise.
            personal_trend_weight: {
                let max = params::get_f64(p, "personal_trend_weight_max", 0.6).clamp(0.0, 1.0);
                if max > 0.0 { rng.gen_range(0.0..max) } else { 0.0 }
            },
            trend_group: if n_trend_groups > 0 { rng.gen_range(0..n_trend_groups) } else { 0 },
            // See module doc comment's "too smart to fail" section. All three
            // below default to a nonzero max (always-on, like
            // `personal_trend_weight` above) so a sweep gets a heterogeneous
            // minority of genuinely-risky agents without any scenario file
            // needing to opt in -- but each max is itself tunable per scenario.
            aggressive_sizing_propensity: {
                // Calibrated by smoke test against scenarios/human_sweep_lite/
                // human_50pct.toml (2026-10-10): 4.0 produced 0% human
                // bankruptcy (the caps were loosened but the underlying desired
                // trade size, order_notional * (1+intensity) * pnl_size_mult,
                // rarely got big enough to matter even scaled up 4x); 15.0 blew
                // up simulation wall-clock time (oversized trades triggering a
                // cascade of FOMO reactions across the rest of the population --
                // a feedback loop, not a hang). 8.0 landed in between: a 1.28%
                // human bankruptcy rate (up from 0%), normal sim runtime, and a
                // real near-wipeout (one agent's equity measured at $89 against
                // a starting net worth over $10,000) -- a minority of agents
                // genuinely blow up, not "every human is risky."
                let max = params::get_f64(p, "aggressive_sizing_max", 8.0).max(1.0);
                if max > 1.0 { rng.gen_range(1.0..max) } else { 1.0 }
            },
            revenge_trading_propensity: {
                let max = params::get_f64(p, "revenge_trading_max", 1.0).clamp(0.0, 1.0);
                if max > 0.0 { rng.gen_range(0.0..max) } else { 0.0 }
            },
            price_chase_prob: {
                let max = params::get_f64(p, "price_chase_prob_max", 0.3).clamp(0.0, 1.0);
                if max > 0.0 { rng.gen_range(0.0..max) } else { 0.0 }
            },
            leverage_risk_appetite: {
                // Defaulted OFF briefly (2026-10-10), then back ON after
                // isolating the actual cause of two machine crashes that
                // first looked leverage-related. What actually happened: a
                // *single-shot* `run_simulation`-style call keeps every
                // trade/order-event accumulating in memory for the whole run
                // and only marshals the complete Rust result into Python
                // objects at the very end -- for a large, long run that final
                // marshal is a big one-time spike, independent of leverage
                // entirely. Measured directly: a leg-based run (10 x 10,000s
                // legs via `run_scenario_in_legs`, which exports+drops each
                // leg's result individually instead of accumulating one
                // giant result) completed this same scenario at full
                // duration *with* leverage enabled, no crash, memory
                // oscillating in a bounded sawtooth the whole time (14GB ->
                // ~11GB -> back to ~14GB each leg) -- and 394/1250 humans
                // (31.5%) genuinely used leverage at some point, human
                // bankruptcy landed at 1.92%, well within the same range as
                // every other run in this file's history, not a runaway
                // outcome. The earlier single-shot tests that appeared to
                // implicate leverage were measuring the marshal-spike
                // artifact, not this mechanism. Lesson for future testing of
                // this bot (or any large/long scenario): prefer
                // `run_scenario_in_legs` over a single `run_scenario_file`/
                // `run_simulation` call once population/duration gets large
                // -- see `scripts/` sweep tooling, which already does this
                // for exactly this reason. Default matches the value this
                // was actually validated at; override `leverage_risk_appetite_max`
                // in a scenario file to tune it.
                let max = params::get_f64(p, "leverage_risk_appetite_max", 0.5).clamp(0.0, 1.0);
                if max > 0.0 { rng.gen_range(0.0..max) } else { 0.0 }
            },
            leverage_amount: {
                let lo = params::get_f64(p, "leverage_min", 1.5).max(1.0);
                let hi = params::get_f64(p, "leverage_max", 3.0).max(lo + 1e-6);
                rng.gen_range(lo..hi)
            },
            leverage_signal_threshold: params::get_f64(p, "leverage_signal_threshold", 1.5),
            leverage_maintenance_margin_buffer: params::get_f64(p, "leverage_maintenance_margin_buffer", 0.8),
            resting_buy_order: None,
            resting_sell_order: None,
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
    //
    // `Variant::Lite` skips this update entirely (and the logistic blend below) --
    // see the variant doc comment for why.
    if s.variant == Variant::Retail {
        s.chartist_fitness = 0.7 * s.chartist_fitness + 0.3 * (s.last_chartist_signal.signum() * raw_return);
        s.fundamentalist_fitness = 0.7 * s.fundamentalist_fitness + 0.3 * (s.last_fundamentalist_signal.signum() * raw_return);
    }

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
    //
    // `Variant::Lite` skips the fitness-based switch and uses one fixed weight
    // instead -- no `f64::exp` call, and no dependence on the (unupdated, for Lite)
    // fitness state above.
    let chartist_weight = match s.variant {
        Variant::Retail => {
            let fitness_gap = s.intensity_of_choice * (s.chartist_fitness - s.fundamentalist_fitness);
            1.0 / (1.0 + (-fitness_gap).exp())
        }
        Variant::Lite => s.lite_blend_weight,
    };
    let trend_component = chartist_weight * price_component + (1.0 - chartist_weight) * fundamentalist_component;

    // Mood component: the latent, price-independent sentiment process, read with
    // individual perception noise (no two people read "the mood" identically) and
    // likewise normalized by its own threshold.
    let true_sentiment = *ctx.market.sentiment.get(&s.asset_id).unwrap_or(&0.0);
    let perceived_sentiment = (true_sentiment + agent.rng.gen_range(-s.perception_noise..s.perception_noise)).clamp(-1.0, 1.0);
    let sentiment_component = perceived_sentiment / s.sentiment_signal_threshold.max(1e-9);

    // Blended, normalized private signal: every component is already on a "1.0 =
    // as strong as this agent's own alert threshold" scale. This is this agent's
    // *own* view -- price/fundamentalist trend, perceived sentiment (with its own
    // perception noise, its "individual random voice") -- before any outside
    // influence.
    let market_signal = s.price_weight * trend_component + (1.0 - s.price_weight) * sentiment_component;

    // Crowd trend: this agent's group's own shared `Market::random_trend`
    // line (see module doc comment), normalized the same way every other
    // signal component is.
    let group_trend = ctx.market.trend_for(s.asset_id, s.trend_group);
    let trend_component_group = group_trend / s.personal_trend_signal_threshold.max(1e-9);
    let raw_signal = (1.0 - s.personal_trend_weight) * market_signal + s.personal_trend_weight * trend_component_group;
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
        // "Hot hand" <-> "revenge trading": interpolates this agent's own P&L
        // sensitivity from today's risk-*reducing* response (grows after wins,
        // shrinks after losses) at propensity=0, through indifference at 0.5, to
        // a full risk-*increasing* reversal (shrinks after wins, grows after
        // losses -- "get-even" loss-chasing) at propensity=1. See module doc
        // comment's "too smart to fail" section.
        let effective_pnl_sensitivity = s.pnl_sizing_sensitivity * (1.0 - 2.0 * s.revenge_trading_propensity);
        let pnl_size_mult = (1.0 + effective_pnl_sensitivity * s.recent_pnl_frac).clamp(0.2, 2.5);
        // `aggressive_sizing_propensity` has to scale the *desired* trade size
        // itself, not just the caps below -- the caps were rarely the binding
        // constraint to begin with (order_notional * (1+intensity) * pnl_size_mult
        // tops out well under the original max_position/cash caps already), so
        // loosening only the ceiling without this left the fix inert (measured:
        // a smoke test with only the caps loosened still showed 0% human
        // bankruptcy). This is what actually lets a high-draw agent's single
        // trade reach a bankruptcy-capable size.
        let notional = round_to_nice(s.order_notional * (1.0 + intensity) * pnl_size_mult * s.aggressive_sizing_propensity);
        let mut qty = notional / mid.max(1e-9);
        // Captured before the Buy-side cap below clips `qty` down -- the gap
        // between the two is "how much of this trade conviction couldn't be
        // paid for in cash," which is exactly the leverage trigger a few lines
        // down needs. Meaningless for Sell (never capped the same way, and
        // leverage here is buy-side-only, same as the existing never-short
        // rule), but harmless to compute either way.
        let desired_qty = qty;
        let mut leverage_action: Option<Action> = None;

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
                // `aggressive_sizing_propensity` (>= 1.0, see its field comment)
                // loosens both caps together for this specific agent -- most
                // agents' draw is close to the 1.0 floor (today's exact caps),
                // a minority draw high enough to commit a genuinely bankruptcy-
                // capable share of their own capital to a single trade.
                let max_position = s.max_position * s.aggressive_sizing_propensity;
                let cash_frac = (s.max_cash_frac_per_trade * s.aggressive_sizing_propensity).min(1.0);
                qty = qty.min((max_position - inventory).max(0.0));
                let cash_budget = agent.cash * cash_frac;
                if cash_budget < mid * qty {
                    qty = (cash_budget / mid.max(1e-9)).max(0.0);
                }

                // Cash-constrained conviction -> leverage (module doc comment's
                // "cash-constrained conviction" section): this agent wanted
                // `desired_qty`, strongly believes the direction (magnitude
                // clears a *higher* bar than the ordinary act-at-all dead
                // zone -- "reputation" in the sense of how much it currently
                // trusts its own view, not a track record), is one of the
                // minority willing to borrow for it at all
                // (`leverage_risk_appetite`), and spot alone can't cover the
                // full size -- open a leveraged top-up for exactly the
                // shortfall, same direction, not a separate unrelated bet.
                let shortfall_qty = (desired_qty - qty).max(0.0);
                if shortfall_qty > 1e-9 && magnitude > s.leverage_signal_threshold && agent.rng.gen_bool(s.leverage_risk_appetite) {
                    leverage_action = liquidation_trader::open_leveraged_position_directed(
                        agent,
                        s.asset_id,
                        Side::Buy,
                        s.leverage_amount,
                        shortfall_qty * mid,
                        s.leverage_maintenance_margin_buffer,
                        mid,
                    )
                    .map(Action::SubmitMargin);
                }
            }
        }

        if qty > 1e-9 {
            // Liquidity check, added 2026-10-04 (see module doc comment): don't
            // blindly take whatever's resting on the other side. `believed_value`
            // is this agent's own view of what the asset is worth, derived from
            // its own (already personalized) `raw_signal` rather than the
            // engine's raw ground-truth `fair_value` -- using ground truth
            // directly here would mean every agent silently agreed on the same
            // "fair" price, defeating the point of this bot's per-agent
            // heterogeneity. Converted back from signal-space ("how many
            // multiples of my own alert threshold") into a price using the same
            // `price_signal_threshold` the chartist component above already
            // uses as its price-move unit.
            // `raw_signal` itself is unbounded (only the dead-zoned `intensity`
            // derived from it is capped, at 1.2) -- a large value here, now newly
            // reachable since each trend group's line is its own independent
            // random walk rather than one bounded shared value, could otherwise
            // send `believed_value` negative or absurd, which the engine's own
            // sanity guard (`Market::submit`'s `invalid_price`/
            // `far_from_fundamental` checks) would then reject outright.
            //
            // Clamped to `fair_value` (ground truth), NOT `mid` -- measured
            // during development: with zero agents starting with inventory, no
            // sell order (market or limit) is possible until *someone* has
            // already bought (`qty.min(inventory)` below), so the book starts
            // bid-only; `OrderBook::mid_price()` then falls back to the single
            // best bid when no ask exists at all. Clamping relative to `mid` in
            // that regime bounds each new belief relative to the *previous*
            // ratchet-up, not to anything stable -- compounding over hundreds of
            // wake-ups into a runaway spiral (observed: up to ~130x the asset's
            // starting price in a 20,000s smoke test). `fair_value` is the one
            // process in this engine that's provably immune to being corrupted
            // by trading activity itself (see market.rs's doc comment), so
            // anchoring the clamp to it bounds the worst case regardless of how
            // far a one-sided book has already drifted -- while the *base*
            // computation still uses `mid`, so this agent's own signal still
            // moves the belief around within that band.
            let believed_value = (mid * (1.0 + raw_signal * s.price_signal_threshold)).clamp(fair_value * 0.5, fair_value * 2.0);
            let book = ctx.market.book(ctx.venue, s.asset_id);
            let best_opposite = match side {
                Side::Buy => book.best_ask(),
                Side::Sell => book.best_bid(),
            };
            // Directional, not a plain distance check: only the *unfavorable*
            // direction counts as "too far." A buyer only balks at an ask priced
            // too far *above* its own belief (paying a premium); an ask that's a
            // bargain (at or below belief, however far below) is never a reason
            // to avoid it -- symmetric for a seller and the bid. This also keeps
            // the resulting limit order safe: whenever this fires, the believed
            // price this agent would then post at is, by construction, still on
            // its own side of the best opposite price *as observed right now* --
            // a buy-limit at a believed_value strictly below the current
            // best_ask can never cross it at decision time (symmetric for
            // sell/best_bid). This order still goes through this agent's own
            // latency before actually reaching the book (see sim.rs's
            // `apply_action`, `Action::Submit`'s latency delay vs. `Cancel`'s
            // immediate one) -- if the book moves in that window, `post_only`
            // correctly rejects the now-stale cross rather than resting it at a
            // bad price. Measured: ~0.07% of this bot's submits in a 100%-human
            // smoke test, an expected, benign rate given realistic latency, not
            // a sign this check is wrong.
            let too_far = |p: f64| match side {
                Side::Buy => (p - believed_value) / believed_value.max(1e-9) > s.fundamentalist_signal_threshold,
                Side::Sell => (believed_value - p) / believed_value.max(1e-9) > s.fundamentalist_signal_threshold,
            };
            let use_limit = match best_opposite {
                None => true,
                // A price exists but is unattractive -- normally this agent
                // patiently posts a limit at its own belief instead. A roll
                // against this agent's own `price_chase_prob` (see module doc
                // comment's "too smart to fail" section) instead crosses
                // impulsively at the bad price with a market order, same as a
                // real panicking/euphoric trader sometimes does. Most agents'
                // draw is near 0 (today's always-patient behavior); this never
                // fires at all when `best_opposite` is `None` -- there's nothing
                // to impulsively chase.
                Some(p) => too_far(p) && !agent.rng.gen_bool(s.price_chase_prob),
            };

            if use_limit {
                // Cancel-and-replace on this same side only -- see module doc
                // comment and `resting_buy_order`/`resting_sell_order`'s field
                // comments: at most one resting order per side per agent, always
                // freshly priced at this wake's `believed_value`, never a pile-up
                // of stale orders from check-ins long past.
                let slot = match side {
                    Side::Buy => &mut s.resting_buy_order,
                    Side::Sell => &mut s.resting_sell_order,
                };
                if let Some((id, aid)) = slot.take() {
                    actions.push(Action::Cancel(id, aid));
                }
                actions.push(Action::Submit(NewOrderRequest {
                    asset_id: s.asset_id,
                    side,
                    order_type: OrderType::Limit,
                    tif: TimeInForce::Gtc,
                    post_only: true,
                    price: Some(believed_value),
                    qty,
                }));
            } else {
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

        // Pushed regardless of whether the spot order above was itself
        // nonzero (see `desired_qty`/`shortfall_qty` above) -- a fully
        // cash-constrained agent (spot qty clipped all the way to 0) can
        // still have a real shortfall worth leveraging into.
        if let Some(a) = leverage_action {
            actions.push(a);
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

/// Called by `sim.rs` after an order from `Action::Submit` gets an id assigned
/// (same hook market_maker.rs/grid.rs use), so this agent remembers which
/// resting limit order to cancel before replacing it next time -- see
/// `resting_buy_order`/`resting_sell_order`'s field comments. Market/IOC orders
/// fill or expire immediately so there's nothing to remember for those; only a
/// `Limit` request ever needs tracking here.
pub fn on_order_id(s: &mut State, req: &NewOrderRequest, id: OrderId) {
    if req.order_type != OrderType::Limit {
        return;
    }
    match req.side {
        Side::Buy => s.resting_buy_order = Some((id, req.asset_id)),
        Side::Sell => s.resting_sell_order = Some((id, req.asset_id)),
    }
}
