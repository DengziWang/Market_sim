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
//!   - **...but biased, not accurate, self-scoring (2026-10-22)**: Brock-Hommes'
//!     own model tracks fitness as a plain decayed average of realized payoff --
//!     correct for modeling a population of boundedly-rational-but-honest
//!     forecasters, wrong for modeling a single retail trader's notoriously poor
//!     introspective access to their *own* track record (direct user feedback:
//!     "I believe the agent is too rational... it learns from what worked well" --
//!     accurately scoring which of two mental strategies has been paying off is
//!     itself a rational-agent behavior, not a bias). Two real documented biases
//!     replace the plain decay: **salience** (a big, vivid move dominates memory
//!     far more than a quiet accurate running average would -- "flashbulb memory"
//!     -- so the effective learning rate scales up with the size of the realized
//!     move, not a fixed 0.3 regardless) and **confirmation bias** (evidence that
//!     confirms whichever strategy this agent is *currently* leaning toward is
//!     absorbed at the normal rate; evidence that the current lean was wrong is
//!     absorbed much more slowly -- people update faster toward "I was right" than
//!     "I was wrong"). Net effect: fitness tracking is now a noisy, self-reinforcing
//!     impression of "what's been working," not an accurate scorecard -- an agent
//!     can keep leaning chartist for a while after fundamentalist would objectively
//!     have paid off better, exactly the kind of sticky, poorly-calibrated belief
//!     revision a real person shows and a fitness-maximizing agent wouldn't. See
//!     `fitness_learning_rate`/`salience_sensitivity`/`salience_cap`/
//!     `confirmation_bias_strength`'s field comments and the update site in
//!     `step()` for the mechanics.
//!   - **Spontaneous profit-taking, independent of any signal (2026-10-22)**:
//!     every bias above, including the disposition effect just below, only ever
//!     fired as a *modifier* on a trade this agent was already about to make
//!     because `raw_signal` cleared the act-at-all dead zone -- on a flat tape
//!     (no real momentum, no mean-reversion gap, calm sentiment) the agent never
//!     acts at all, so a real open gain just sits there forever (direct user
//!     feedback: "humans also sell when they have a perfectly straight line").
//!     Real disposition-effect selling isn't contingent on a fresh technical
//!     signal; it's driven by the unrealized gain itself -- "it's up, lock it
//!     in" doesn't need a reason. `spontaneous_take_profit` in `step()` rolls
//!     independently of `intensity`/`raw_signal` entirely, using the same CPT
//!     value of the unrealized gain (see below) to scale a per-wake probability
//!     of a plain take-profit sell; see `spontaneous_profit_taking_rate`'s field
//!     comment. Never fires on a loss (that's "hold and hope," not a reason to
//!     act), and never overrides a real opposing signal -- it only fills the gap
//!     *between* signal-driven wakes.
//!   - **More extreme profit-taking at large gains (2026-10-25, direct user
//!     request)**: the plain CPT value above is *concave* in the raw gain
//!     (`cpt_alpha` < 1 by Tversky & Kahneman's own estimate), which actually
//!     compresses the probability gap between a modest winner and a huge one --
//!     a 5% gain and a 50% gain don't differ nearly as much in take-profit odds
//!     as the size difference alone would suggest. `spontaneous_profit_convexity`
//!     raises the CPT value to a power > 1 before it scales the roll, pulling
//!     that gap back open and past linear: a small gain becomes genuinely rare to
//!     act on, while a large one gets realized almost for certain within a wake
//!     or two -- see that field's own comment for the worked examples.
//!   - **Chartist vs. contrarian sub-switching (2026-10-23)**: measured directly
//!     (a diversity test over a real run) that the typical human agent's own
//!     trades were 90-96% one-sided, with ~44-50% of agents never once switching
//!     direction in a ~17-hour window, and that the population as a whole was
//!     84.7% one-sided per leg at the median -- direct feedback: "this is not
//!     realistic." A first attempt (an EMA-smoothed "sticky" chartist view,
//!     replaced by this section) measurably made both numbers *worse*, not
//!     better -- smoothing an individual agent's own noisy momentum reading only
//!     filters out the reversals that would have made its own trading two-sided,
//!     while every agent was still tracking the same underlying tape, just more
//!     slowly, so it didn't create any actual *disagreement* between agents
//!     either. The real lesson from Lux & Marchesi (1999, *Nature* 397; 2000,
//!     *Int. J. Theoretical and Applied Finance* 3) isn't that chartists are
//!     sticky -- Brock-Hommes' chartist-vs-fundamentalist switch above is already
//!     that -- it's that chartists themselves split into sub-groups that
//!     genuinely disagree: optimists who keep believing a move continues, and
//!     pessimists who bet it reverses, with the population's own split evolving
//!     by which sub-strategy has actually been paying off. This reuses that
//!     exact already-working machinery (the same salience/confirmation-bias
//!     fitness update and logistic blend as the chartist-vs-fundamentalist
//!     switch, just one level deeper): a new `contrarian_fitness` tracks how
//!     well *betting against* last wake's momentum signal would have scored
//!     (`-last_chartist_signal.signum() * raw_return`, the exact mirror image of
//!     `chartist_fitness`'s own payoff), and a logistic blend of
//!     `contrarian_fitness - chartist_fitness` decides, per agent per wake, how
//!     much of this agent's chartist reading flips against the raw momentum
//!     signal rather than following it. Because fitness evolves from each
//!     agent's own idiosyncratic wake timing and is sticky under confirmation
//!     bias once established, agents genuinely diverge -- some settle into
//!     "ride the trend," others into "fade it," and either can flip an
//!     individual agent's own trading two-sided over time, which plain
//!     persistence never could. Retail-only, like the switch it extends (`Lite`
//!     keeps using the raw, unmodified `price_component`) -- deliberately *not*
//!     coupled across agents at all (no shared population-level opinion state),
//!     to avoid reintroducing the shared `random_trend` herding amplifier
//!     removed above (2026-10-17).
//!   - **Realization utility near a position's own high (2026-10-23)**: the
//!     spontaneous profit-taking roll above only ever cared about raw unrealized
//!     gain relative to entry price. Barberis & Xiong (2012, *Journal of
//!     Financial Economics* 104) document that real selling clusters specifically
//!     near an investor's own historical high *since opening the position* --
//!     "realization utility," a direct utility kick from locking in a gain right
//!     at its best point, not just a function of the gain's size. `step()` tracks
//!     `position_high_water_mark` (the running high of `mid` since the current
//!     position was opened, resetting whenever the agent goes flat, same
//!     lifecycle as `avg_entry_price`) and adds an extra boost to the
//!     spontaneous-sell probability when price is at or near that high --
//!     on top of, not instead of, the existing unrealized-gain term. This is
//!     what lets a sustained rally get interrupted by real profit-taking right
//!     near its local peaks, not just whenever the cumulative gain happens to be
//!     large; see `realization_utility_scale`/`realization_utility_window`'s
//!     field comments.
//!   - **Rising-trend profit-taking override + fast-crash panic selling
//!     (2026-10-24)**: direct user feedback -- "when a human is on a rising
//!     stock they should sometimes sell. Also if stocks fall really fast
//!     they should sell." Two gaps the biases above didn't cover: (1) every
//!     bias so far that can turn a would-be *buy* into a sell
//!     (`spontaneous_take_profit`) only ever fires on a *quiet* wake, i.e.
//!     one where the signal path didn't already decide to buy -- a strongly
//!     bullish signal always bought more, with nothing modeling a trader who
//!     sells *into* a rally instead of chasing it further ("sell the rip").
//!     `rising_profit_take_prob_scale`/`rising_profit_take_max_prob` fix this
//!     directly: whenever the ordinary signal path was about to buy and this
//!     agent is sitting on an open gain, roll a probability (scaled by the
//!     same CPT value of the unrealized gain used everywhere else in this
//!     file, capped so it never dominates buying outright) to flip that buy
//!     into a profit-take sell instead. (2) the disposition effect below is
//!     specifically "hold and hope" on a loss -- by design, the deeper the
//!     loss, the *less* eager this agent is to sell, which is realistic for
//!     an ordinary bad day but wrong for a real crash: capitulation selling
//!     is a documented override of that exact hold instinct once a drop is
//!     sharp and fast enough, not just "a sell signal that happens to be
//!     bigger." `panic_sell` is computed independently of the ordinary
//!     signal-acts roll (so it can act through a wake inertia would normally
//!     have skipped) off `price_component` -- the raw, belief-independent
//!     price move, not this agent's own chartist/contrarian-adjusted
//!     reading, since panic is a reaction to what the tape actually did, not
//!     to this agent's own theory about it -- once the drop clears a higher
//!     bar than the ordinary alert dead zone (`panic_sell_signal_multiple`)
//!     *and* the fall is still actively in progress (`prev_price_component`
//!     was also negative -- a revision made the same day, direct user
//!     request: "only do if it's really bad and the falling hasn't stopped
//!     yet", since a single severe wake could just as easily be a move that
//!     already bottomed out and started recovering by the time this agent
//!     next checks in -- panicking into a bounce isn't the behavior being
//!     modeled). When it fires, it forces the side to Sell regardless of what the
//!     ordinary signal said (even a contrarian "buy the dip" read) and
//!     applies `panic_sell_qty_mult` on top of, not instead of, the usual
//!     sizing -- bypassing the loss-hold suppression below rather than
//!     feeding into it, since the whole point is that panic overrides "hold
//!     and hope" rather than being a more extreme version of it.
//!   - **Slow-bleed capitulation (2026-10-25)**: measured directly (a diversity
//!     sweep after the fundamental-vol/market_maker-coverage/depth-collapse engine
//!     fixes) that humans still trade overwhelmingly one-sided -- and, cross-
//!     checking against which compositions got worse, specifically the ones with
//!     net-negative growth. The mechanism: `panic_sell` above only fires for an
//!     acute, fast, *ongoing* crash (`panic_sell_signal_multiple` clears a high bar
//!     this wake *and* last wake), and the disposition effect's "hold and hope"
//!     only ever *shrinks* a sell this agent was already about to make -- neither
//!     one gives an agent stuck on the wrong side of a slow, grinding decline (no
//!     single wake severe enough to panic over, but a real, sustained loss) any
//!     way to ever sell, if its own raw_signal (e.g. a fundamentalist-leaning
//!     agent reading "it's cheap, buy more" through the whole decline) never once
//!     points Sell. That agent's entire trade history for the rest of the run is
//!     100% buys -- not because it's rationally averaging down forever, but
//!     because nothing in this file lets a real person's patience for a bad
//!     position run out gradually, only acutely (panic) or never (hold and hope).
//!     Real retail capitulation after a prolonged drawdown -- "I can't watch this
//!     anymore" -- is exactly this gradual-exhaustion case, documented separately
//!     from panic selling in the disposition-effect literature. `wakes_underwater`
//!     (new state, reset whenever the agent goes flat or the position returns to a
//!     gain) counts consecutive wakes spent at an unrealized loss; past
//!     `capitulation_fatigue_threshold_wakes`, a `capitulation_sell` trigger --
//!     independent of `raw_signal`'s direction, same construction as `panic_sell`
//!     -- fires with probability scaling in the *excess* wake count, and, like
//!     `panic_sell_qty_mult`, applies `capitulation_qty_mult` to override (not feed
//!     into) the ordinary loss-hold suppression, since the whole point is that the
//!     hold instinct is what's finally giving out.
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
//!   - **Grouped crowd trend -- removed (2026-10-06 revert, then removed
//!     entirely 2026-10-17)**: used to assign every human agent to one of
//!     `n_trend_groups` groups at construction, each reading its own shared
//!     `Market::random_trend` line identically (disagreement came only from
//!     `personal_trend_weight`, how much each agent leaned on it). Removed
//!     at direct user request after this session's crash investigation
//!     traced correlated human herding partly to this mechanism: a shared,
//!     price-independent signal that many agents react to at the literal
//!     same moment, for a reason that has nothing to do with the market
//!     itself, is exactly the kind of synchronized push that can overwhelm
//!     available liquidity. See `market.rs`'s removal note on `Market::new`.
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
//!     sentiment noise, its own chartist/fundamentalist blend), not the
//!     engine's raw ground-truth `fundamental_value` -- using ground truth
//!     directly would mean every agent
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
//!     retail blowups. Originally three new per-agent propensity values (below),
//!     each drawn independently and uniformly at construction, so *some* agents
//!     genuinely risk enough to blow up while most stay close to the original,
//!     well-risk-managed behavior -- not a population-wide toggle, since real
//!     retail populations aren't uniformly reckless either. Now two:
//!     `revenge_trading_propensity` (bet *bigger*, not smaller, right after a
//!     loss) was removed (2026-10-17, direct user request) alongside the
//!     grouped crowd trend above, for the same reason -- a population-wide
//!     tendency to double down specifically when already losing is fuel for
//!     exactly the kind of correlated herding this session's crash
//!     investigation was chasing, not a source of agent-level heterogeneity
//!     worth keeping around.
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
    /// multiplier on this agent's own `max_cash_frac_per_trade`/`max_position`
    /// (a ready-to-use scale, not a 0-1 probability needing a separate scale
    /// constant). 1.0 (the floor) =
    /// exactly today's caps; most agents land well below `aggressive_sizing_max`
    /// since the draw is uniform across the whole range, but a minority land
    /// high enough to actually risk a large enough share of net worth to go
    /// bankrupt from trading losses alone -- see module doc comment's "too smart
    /// to fail" section for why this exists.
    pub aggressive_sizing_propensity: f64,
    /// Drawn once per agent, Uniform(0, `price_chase_prob_max`) -- the per-attempt
    /// probability (rolled fresh each time the situation arises, not fixed per
    /// agent-level outcome like the two propensities above) that this agent
    /// impulsively crosses at a bad price with a market order instead of patiently
    /// posting a resting limit at its own believed value. 0 = never (today's
    /// always-patient behavior).
    pub price_chase_prob: f64,
    /// Drawn once per agent, Uniform(0, `take_good_price_prob_max`) -- the
    /// per-attempt probability of taking an *attractive* available price
    /// outright with a market order instead of resting a limit at belief
    /// (see module doc comment's "General price-impact patience"). 1.0 =
    /// always takes a good price outright (the old, only behavior); low =
    /// mostly patient even when the price looks fine.
    pub take_good_price_prob: f64,
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
    /// Running high of `mid` since the current position was opened (module doc
    /// comment's "Realization utility" section) -- `None` while flat, same
    /// lifecycle as `avg_entry_price`. Updated every wake in `step()`.
    pub position_high_water_mark: Option<f64>,
    /// Scale on the extra spontaneous-sell urge from being at/near
    /// `position_high_water_mark` (see `step()`'s `spontaneous_take_profit`
    /// roll). 0 = this reference point is ignored entirely (old behavior: only
    /// raw unrealized gain from entry matters).
    pub realization_utility_scale: f64,
    /// How far below `position_high_water_mark` (as a fraction of it) still
    /// counts as "near" it for the realization-utility boost -- 0.15 means the
    /// boost fades out linearly between the high itself and 15% below it, then
    /// contributes nothing further down. Calibrates the *width* of "near a new
    /// high," not whether the effect exists at all (`realization_utility_scale`
    /// does that).
    pub realization_utility_window: f64,
    /// Per-wake probability *scale* for the signal-independent profit-taking urge
    /// (module doc comment's "Spontaneous profit-taking" section) -- the actual
    /// roll each wake is `(this * cpt_value(unrealized_gain)).clamp(0, 1)`, so a
    /// bigger open gain makes the urge fire more often, not just this flat rate
    /// every time. Only ever considered when the signal-driven path this wake
    /// didn't already act (see `step()`), and only for a gain (`cpt_value >= 0`)
    /// -- a loss has no spontaneous-sell counterpart, consistent with "hold and
    /// hope" being the whole point of the loss side of the disposition effect.
    /// 0 = this bias is off entirely (old behavior: profit-taking only ever rides
    /// along with a real signal).
    pub spontaneous_profit_taking_rate: f64,
    /// Module doc comment's "More extreme profit-taking at large gains" section:
    /// exponent applied to the CPT value before it scales the spontaneous
    /// take-profit probability. 1.0 = old behavior (linear in `v`, itself already
    /// concave in the raw gain via `cpt_alpha`); higher = the probability grows
    /// faster than the gain itself, concentrating realized profit-taking onto
    /// genuinely large winners rather than spreading it evenly across any gain.
    pub spontaneous_profit_convexity: f64,
    /// Module doc comment's "Rising-trend profit-taking override" section:
    /// scale on how much the current unrealized gain's CPT value (the same
    /// one used by the disposition effect below) converts into a per-wake
    /// probability of flipping an otherwise-buy signal into a profit-take
    /// sell instead. 0 = this override never fires (old behavior: a bullish
    /// signal always buys).
    pub rising_profit_take_prob_scale: f64,
    /// Ceiling on the probability above -- without one, a large enough gain
    /// could make this override near-certain, which would mean this agent
    /// effectively never buys into its own winners again, not "sometimes
    /// sells into strength."
    pub rising_profit_take_max_prob: f64,
    /// Module doc comment's "fast-crash panic selling" section: how many
    /// multiples of `price_signal_threshold`-normalized return
    /// (`price_component`) a drop must clear, beyond the ordinary act-at-all
    /// dead zone (1.0), before it counts as a "crash" worth panicking over --
    /// not just an everyday down move, which the ordinary signal path
    /// (disposition-suppressed) already handles. Deliberately a high bar
    /// (2026-10-24 revision, direct user request -- "only do if it's really
    /// bad"): panic selling is reserved for genuinely severe drops, not a
    /// slightly-bigger-than-usual one.
    pub panic_sell_signal_multiple: f64,
    /// Previous wake's `price_component` (module doc comment's "fast-crash
    /// panic selling" section, 2026-10-24 revision) -- lets `step()` check
    /// whether a fall is still actively in progress across two consecutive
    /// check-ins, not just whether this single wake's move was severe. Reset
    /// implicitly to 0.0 at construction (no prior wake yet).
    pub prev_price_component: f64,
    /// Scale on how far the drop clears `panic_sell_signal_multiple` (the
    /// "crash excess") that converts into this wake's panic-sell
    /// probability. 0 = panic selling never fires (old behavior: losses are
    /// only ever sold via the disposition-suppressed ordinary signal path).
    pub panic_sell_prob_scale: f64,
    /// Multiplier applied to the sell quantity once panic selling fires, on
    /// top of (applied after, not instead of) the ordinary CPT-based sizing
    /// -- this is what makes panic selling an *override* of "hold and hope"
    /// rather than just another input into it. 1.0 = no extra dump beyond
    /// whatever the normal sizing already produced.
    pub panic_sell_qty_mult: f64,
    /// Mirror of `panic_sell_signal_multiple` (2026-11-05, direct user request
    /// after the crash investigation traced persistent human sell-bias --
    /// price trading below fundamental value 65-77% of the time -- to this
    /// file's override mechanisms all pushing one direction: panic_sell,
    /// capitulation_sell, and the rising-trend profit-take override all only
    /// ever force extra *selling*; nothing forced extra buying). Same
    /// structure as `panic_sell`, mirrored: how many "units" (`price_component`)
    /// a *rise* must clear, past the ordinary act-at-all dead zone, before it
    /// counts as a rally worth chasing into rather than an everyday up move.
    pub panic_buy_signal_multiple: f64,
    /// Scale on how far the rise clears `panic_buy_signal_multiple` (the
    /// "rally excess") that converts into this wake's panic-buy probability.
    /// 0 = panic buying never fires (old behavior: no forced-buy override
    /// existed at all). Mirrors `panic_sell_prob_scale`.
    pub panic_buy_prob_scale: f64,
    /// Multiplier applied to the buy quantity once panic buying fires, on top
    /// of the ordinary sizing -- mirrors `panic_sell_qty_mult`, same "override,
    /// not just another input" reasoning.
    pub panic_buy_qty_mult: f64,
    /// Module doc comment's "Slow-bleed capitulation" section: consecutive wakes
    /// spent with an open position at an unrealized loss (CPT value < 0). Reset to
    /// 0 whenever the agent goes flat or the position's CPT value is >= 0 (back to
    /// even or in a gain) -- this tracks *duration of an unbroken bad stretch*,
    /// not a lifetime count.
    pub wakes_underwater: u32,
    /// How many consecutive underwater wakes (`wakes_underwater`) before fatigue
    /// starts contributing to a capitulation-sell probability at all -- below
    /// this, "hold and hope" is still the whole story, same as before this
    /// mechanism existed. Calibrated in wakes, not seconds, since this agent's own
    /// check-in cadence already varies (background vs. alert wakes).
    pub capitulation_fatigue_threshold_wakes: f64,
    /// Scale on how far `wakes_underwater` clears the threshold above (the
    /// "fatigue excess") that converts into this wake's capitulation-sell
    /// probability -- same shape as `panic_sell_prob_scale` for the acute case.
    /// 0 = this bias never fires (old behavior: a slow bleed can be held forever).
    pub capitulation_fatigue_prob_scale: f64,
    /// Multiplier applied to the sell quantity once capitulation fires, overriding
    /// (not feeding into) the ordinary loss-hold suppression just like
    /// `panic_sell_qty_mult` -- the point of this mechanism is specifically that
    /// the hold instinct runs out, so applying the hold-and-hope shrink on top
    /// would defeat it.
    pub capitulation_qty_mult: f64,
    /// Mirror of `avg_entry_price`/`wakes_underwater`'s machinery, opposite
    /// direction (2026-11-05, direct user request -- see `panic_buy`'s own
    /// comment for the full reasoning): the `mid` price at the moment this
    /// agent most recently went flat (inventory crossed to ~0), so a long
    /// stretch of price climbing *since* can be measured the same way
    /// `avg_entry_price` measures a loss since entry. `None` while holding a
    /// position (mirrors `avg_entry_price` being `None` while flat).
    pub flat_since_price: Option<f64>,
    /// Mirror of `wakes_underwater`: consecutive wakes spent flat (or holding
    /// too little to matter) while price has climbed above `flat_since_price`
    /// -- a long, grinding "I'm missing this rally" stretch, the FOMO
    /// counterpart to a long grinding loss. Reset the moment the agent opens a
    /// real position or price drops back to/below `flat_since_price`.
    pub wakes_missed_rally: u32,
    /// Mirror of `capitulation_fatigue_threshold_wakes`.
    pub fomo_fatigue_threshold_wakes: f64,
    /// Mirror of `capitulation_fatigue_prob_scale`.
    pub fomo_fatigue_prob_scale: f64,
    /// Mirror of `capitulation_qty_mult` -- overrides (not feeds into) the
    /// ordinary buy sizing once FOMO fatigue fires, same "the resistance
    /// finally gives out" reasoning as `capitulation_qty_mult`.
    pub fomo_qty_mult: f64,
    /// Rolling, *biased* fitness (realized hypothetical payoff) of the chartist and
    /// fundamentalist signals -- see the module doc comment's "biased, not accurate,
    /// self-scoring" section. Updated each wake from what actually happened since
    /// the *previous* one, using `last_chartist_signal`/`last_fundamentalist_signal`
    /// (that previous wake's signals, i.e. what each strategy "predicted" then).
    pub chartist_fitness: f64,
    pub fundamentalist_fitness: f64,
    /// Overall damping on the chartist/fundamentalist trend blend below
    /// (2026-11-05, direct user request: a pure-human market with no
    /// market_maker/mean_reversion counterforce runs away into explosive,
    /// high-variance bubbles/crashes once the sell-only override mechanisms
    /// are relaxed enough to stop biasing it downward -- the actual driver
    /// turned out to be unchecked trend-following, not those overrides. 1.0 =
    /// no change from the original, undamped blend; lower values shrink the
    /// whole trend signal's contribution to `raw_signal` uniformly.
    pub trend_damping: f64,
    /// Standing, *structural* counter-trend pull -- always present,
    /// continuously scaled by `price_component` (this wake's own
    /// momentum reading), independent of the *learned* chartist-vs-contrarian
    /// switch above (`contrarian_fitness`, which only engages after
    /// trend-following has provably been losing). 0.0 = no effect (backward
    /// compatible: the learned switch is the only contrarian force, as
    /// before). A positive value sells into a rise and buys into a fall every
    /// wake, proportionally, rather than waiting for fitness-learning to
    /// eventually catch up -- the real-time stabilizer a pure-human market
    /// has nothing else to provide.
    pub counter_trend_scale: f64,
    /// Rolling, *biased* fitness of betting *against* last wake's chartist
    /// signal -- module doc comment's "Chartist vs. contrarian sub-switching"
    /// section. Updated by the exact same salience/confirmation-bias mechanism
    /// as `chartist_fitness`/`fundamentalist_fitness` just above, sharing the
    /// same `confirmation_bias_strength` too -- tried a separate, weaker value
    /// for this switch specifically (2026-10-23), reasoning that a loser
    /// should catch back up faster; confirmed directly it made every diversity
    /// measure worse, not better (a less-biased updater converges *faster*
    /// onto whichever sub-strategy is objectively winning the market's current
    /// trend, concentrating the population rather than spreading it). The bias
    /// staying strong is what keeps genuine disagreement alive. With the
    /// mirror-image payoff (`-last_chartist_signal.signum() * raw_return`), a
    /// logistic blend of this against `chartist_fitness` decides, per wake,
    /// how much this agent's chartist reading flips against the raw momentum
    /// signal rather than following it -- see the blend site in `step()`.
    ///
    /// Initialized to a tiny per-agent random value, Uniform(
    /// -`contrarian_initial_bias`, `contrarian_initial_bias`), *not* a flat
    /// 0.0 -- both fitnesses start at 0 so a hard tie needs breaking somehow,
    /// and always breaking it the same way (confirmed directly: tie always
    /// favored trend-follow) gave trend-following a population-wide first-mover
    /// advantage that confirmation bias then locked in almost permanently, no
    /// matter how well contrarian would objectively have scored. A random
    /// per-agent nudge means roughly half the population starts slightly
    /// contrarian-leaning instead, which is what actually produced the
    /// population-level diversity this mechanism exists for; see `step()`'s
    /// `leans_trend_follow` for where the tie still has to resolve somehow.
    pub contrarian_fitness: f64,
    /// Base learning rate for the fitness update at *unit* salience and on
    /// *confirming* evidence -- the old, plain Brock-Hommes decay used a fixed 0.3
    /// unconditionally; this is that same baseline, now the starting point before
    /// `salience`/`confirmation_bias_strength` scale it up or down per update.
    pub fitness_learning_rate: f64,
    /// How much a *vivid* (large-magnitude) realized move amplifies this update's
    /// learning rate above the baseline -- "flashbulb memory": a single dramatic
    /// win/loss should dominate this agent's impression of which strategy is
    /// working far more than a smooth, size-blind average would. 0 = size never
    /// matters (back to plain Brock-Hommes); higher = more of the agent's belief
    /// about "what's working" comes from its single most dramatic recent outcome.
    pub salience_sensitivity: f64,
    /// Ceiling on the salience multiplier above -- without one, a single extreme
    /// move could push the effective learning rate past 1.0, which would flip the
    /// *sign* of retained old fitness in the convex combination below (nonsense,
    /// not "very forgetful").
    pub salience_cap: f64,
    /// Multiplier applied to this update's learning rate when the realized payoff
    /// *contradicts* whichever strategy this agent is currently leaning toward
    /// (`chartist_fitness >= fundamentalist_fitness` or the reverse) -- see the
    /// module doc comment. 1.0 = symmetric/rational (equally fast to learn you were
    /// right or wrong, the original Brock-Hommes behavior); lower = genuine
    /// confirmation bias, slower to accept the current lean was the wrong one.
    pub confirmation_bias_strength: f64,
    pub last_chartist_signal: f64,
    pub last_fundamentalist_signal: f64,
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
    /// `rng`: used to draw this specific agent's own per-agent propensities (see
    /// their field comments) -- the same construction-time RNG `sim.rs`'s
    /// `build_population` already uses for this agent's starting cash/leverage,
    /// so the draw is deterministic given a fixed run seed, same as everything
    /// else about population construction.
    pub fn from_params(variant: Variant, asset_id: AssetId, initial_price: f64, p: &toml::Value, rng: &mut impl Rng) -> Self {
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
            position_high_water_mark: None,
            // Default 0.12: right at a position's own post-entry high, this
            // adds a flat +0.12 to the spontaneous-sell roll probability --
            // comparable to what a large (50%+) unrealized gain already
            // contributes via the CPT term alone, which is the point: being at
            // a new high should feel like roughly as strong a reason to lock
            // in as a big raw gain does, per Barberis & Xiong.
            realization_utility_scale: params::get_f64(p, "realization_utility_scale", 0.12),
            realization_utility_window: params::get_f64(p, "realization_utility_window", 0.15),
            // Default 0.25 (raised from 0.15, 2026-10-25, alongside the new
            // convexity exponent below): calibrated so the *50%-gain* case stays
            // anchored at roughly the same ~9% per wake this already had, while
            // convexity does the work of pulling small and huge gains apart from
            // that anchor in opposite directions -- see
            // `spontaneous_profit_convexity`'s own comment for the worked
            // examples at the low and high end.
            spontaneous_profit_taking_rate: params::get_f64(p, "spontaneous_profit_taking_rate", 0.25),
            // Default 1.8: with the 0.25 rate above, a modest 5% open gain
            // (cpt_value ~0.07) now fires only ~0.2% of the time (down from ~1%)
            // -- genuinely rare, consistent with "a small gain alone isn't a
            // reason to sell." A 50% gain (cpt_value ~0.54) stays anchored at
            // ~9%, matching the old un-convex behavior there by construction. A
            // 100% gain (cpt_value 1.0) jumps to 25% per wake, and a 200% gain
            // (cpt_value ~1.84) to ~75% -- a genuinely outsized winner gets
            // realized within one or two wakes almost for certain, modeling real
            // profit-taking euphoria on a huge run rather than a flat per-percent
            // rate that treats a 20% gain and a 200% gain as differing only
            // linearly.
            spontaneous_profit_convexity: params::get_f64(p, "spontaneous_profit_convexity", 1.8),
            // Default 0.25: at a modest 5% open gain (cpt_value ~0.07) that's
            // roughly a 1.75% chance per qualifying buy-signal wake of
            // selling into strength instead; at a large 50% gain (cpt_value
            // ~0.59) it's ~15% -- noticeable but still the minority outcome,
            // same calibration philosophy as `spontaneous_profit_taking_rate`.
            rising_profit_take_prob_scale: params::get_f64(p, "rising_profit_take_prob_scale", 0.25),
            // Default 0.45: even at an extreme gain, this agent still buys
            // into its own winner more often than not -- this is "sometimes
            // sells into strength," not "stops buying winners."
            rising_profit_take_max_prob: params::get_f64(p, "rising_profit_take_max_prob", 0.45).clamp(0.0, 1.0),
            // Default 4.0 (raised from an earlier 2.5, 2026-10-24, direct
            // user request -- "only do if it's really bad"): well past the
            // ordinary alert dead zone (1.0), and past where the earlier
            // default sat too -- panic is reserved for a drop that's
            // genuinely severe, not just sharper/faster than the everyday
            // moves the ordinary signal path already reacts to.
            panic_sell_signal_multiple: params::get_f64(p, "panic_sell_signal_multiple", 4.0),
            prev_price_component: 0.0,
            // Default 0.3: one full unit of crash excess beyond the
            // multiple above gives a 30% chance this wake; two units, 60%;
            // clamped at 1.0 beyond that -- a real crash doesn't guarantee
            // this specific agent sells on this specific wake, but makes it
            // likely within a few.
            panic_sell_prob_scale: params::get_f64(p, "panic_sell_prob_scale", 0.3),
            // Default 2.0: once panicking, this agent dumps roughly twice
            // what the ordinary signal-driven sizing alone would have sold
            // -- a real capitulation sale, not a slightly-larger-than-usual
            // one.
            panic_sell_qty_mult: params::get_f64(p, "panic_sell_qty_mult", 2.0).max(1.0),
            // Mirrors of the panic_sell defaults just above, same reasoning,
            // opposite direction -- see State's own field comments.
            panic_buy_signal_multiple: params::get_f64(p, "panic_buy_signal_multiple", 4.0),
            panic_buy_prob_scale: params::get_f64(p, "panic_buy_prob_scale", 0.3),
            panic_buy_qty_mult: params::get_f64(p, "panic_buy_qty_mult", 2.0).max(1.0),
            wakes_underwater: 0,
            // Default 20: at this agent's own average check-in interval
            // (`background_wake_min`/`max`'s midpoint, ~21.5 min at the defaults
            // above), 20 straight underwater wakes is roughly 7 hours of sustained
            // paper loss with no relief -- long enough that this is a real
            // prolonged drawdown, not just an ordinary bad afternoon the acute
            // `panic_sell` dead zone already leaves alone.
            capitulation_fatigue_threshold_wakes: params::get_f64(p, "capitulation_fatigue_threshold_wakes", 20.0),
            // Default 0.05: linear in the fatigue excess, reaching a 100% roll at
            // 20 wakes past the threshold above -- a real person's patience runs
            // out gradually over the following days, not all at once the moment
            // the threshold clears.
            capitulation_fatigue_prob_scale: params::get_f64(p, "capitulation_fatigue_prob_scale", 0.05),
            // Default 1.5: more modest than panic_sell_qty_mult's 2.0 -- this is
            // "finally giving up on a slow bleed," not a frantic crash dump.
            capitulation_qty_mult: params::get_f64(p, "capitulation_qty_mult", 1.5).max(1.0),
            // Mirrors of the capitulation_sell defaults just above, same
            // reasoning, opposite direction -- see State's own field comments.
            flat_since_price: None,
            wakes_missed_rally: 0,
            fomo_fatigue_threshold_wakes: params::get_f64(p, "fomo_fatigue_threshold_wakes", 20.0),
            fomo_fatigue_prob_scale: params::get_f64(p, "fomo_fatigue_prob_scale", 0.05),
            fomo_qty_mult: params::get_f64(p, "fomo_qty_mult", 1.5).max(1.0),
            trend_damping: params::get_f64(p, "trend_damping", 1.0),
            counter_trend_scale: params::get_f64(p, "counter_trend_scale", 0.0),
            chartist_fitness: 0.0,
            fundamentalist_fitness: 0.0,
            contrarian_fitness: {
                let eps = params::get_f64(p, "contrarian_initial_bias", 0.02).abs();
                if eps > 0.0 { rng.gen_range(-eps..eps) } else { 0.0 }
            },
            fitness_learning_rate: params::get_f64(p, "fitness_learning_rate", 0.3),
            // Calibrated so a "big" move (2x price_signal_threshold) roughly
            // doubles the confirming-evidence learning rate (0.3 -> ~0.6):
            // noticeably more vivid-move-dominated than plain Brock-Hommes
            // without being able to single-move overwrite all prior fitness.
            salience_sensitivity: params::get_f64(p, "salience_sensitivity", 1.5),
            salience_cap: params::get_f64(p, "salience_cap", 3.0),
            // Default 0.35: disconfirming evidence is absorbed roughly a third
            // as fast as confirming evidence -- a real, sticky bias, not a
            // rounding-error-sized one, but not so extreme the agent never
            // revises a wrong lean either.
            confirmation_bias_strength: params::get_f64(p, "confirmation_bias_strength", 0.35),
            last_chartist_signal: 0.0,
            last_fundamentalist_signal: 0.0,
            // See module doc comment's "too smart to fail" section. Both
            // below default to a nonzero max (always-on) so a sweep gets a
            // heterogeneous minority of genuinely-risky agents without any
            // scenario file needing to opt in -- but each max is itself
            // tunable per scenario.
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
            price_chase_prob: {
                let max = params::get_f64(p, "price_chase_prob_max", 0.3).clamp(0.0, 1.0);
                if max > 0.0 { rng.gen_range(0.0..max) } else { 0.0 }
            },
            // Default max 0.3, not e.g. 1.0 (which would silently restore the
            // old always-take behavior) -- most agents land well under 30%,
            // so an attractive price gets taken outright only occasionally;
            // the rest of the time this agent rests a limit at belief instead,
            // still a real participant, just not the one pushing price.
            take_good_price_prob: {
                let max = params::get_f64(p, "take_good_price_prob_max", 0.3).clamp(0.0, 1.0);
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
    // Realization-utility reference point (module doc comment): tracks the
    // running high of `mid` for as long as this position stays open, reset
    // once the agent goes flat so the next position starts its own high fresh
    // -- same lifecycle as `avg_entry_price` just above.
    s.position_high_water_mark = if inventory > 1e-9 {
        Some(mid.max(s.position_high_water_mark.unwrap_or(mid)))
    } else {
        None
    };

    // Slow-bleed capitulation tracking (module doc comment): counts consecutive
    // wakes spent holding a position at an unrealized loss. Reset (not just
    // decayed) the moment the agent goes flat or the position is back to even or
    // better -- this measures an unbroken bad stretch, so any relief clears it,
    // the same lifecycle as `avg_entry_price`/`position_high_water_mark` above.
    let currently_underwater = inventory > 1e-9
        && s.avg_entry_price.is_some_and(|entry| mid < entry);
    s.wakes_underwater = if currently_underwater { s.wakes_underwater + 1 } else { 0 };

    // FOMO-fatigue tracking -- mirrors the capitulation-sell tracking just
    // above, opposite direction (see `flat_since_price`/`wakes_missed_rally`'s
    // own field comments). `flat_since_price` is set once, the moment this
    // agent becomes flat, and held fixed (not re-maxed every wake like
    // `position_high_water_mark`) for as long as it stays flat -- mirrors
    // `avg_entry_price` only updating on an actual trade, not every wake.
    s.flat_since_price = if inventory <= 1e-9 {
        Some(s.flat_since_price.unwrap_or(mid))
    } else {
        None
    };
    let currently_missing_rally = inventory <= 1e-9
        && s.flat_since_price.is_some_and(|ref_price| mid > ref_price);
    s.wakes_missed_rally = if currently_missing_rally { s.wakes_missed_rally + 1 } else { 0 };

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

    // Biased fitness update (module doc comment's "biased, not accurate,
    // self-scoring" section): score last wake's chartist and fundamentalist
    // signals against what the price *actually* did since then, same as plain
    // Brock-Hommes -- but absorbed at a *variable* rate, not the original fixed
    // 0.3, so the result is a sticky, salience-skewed impression rather than an
    // accurate running tally.
    //
    // `Variant::Lite` skips this update entirely (and the logistic blend below) --
    // see the variant doc comment for why.
    if s.variant == Variant::Retail {
        let leans_chartist = s.chartist_fitness >= s.fundamentalist_fitness;
        let chartist_payoff = s.last_chartist_signal.signum() * raw_return;
        let fundamentalist_payoff = s.last_fundamentalist_signal.signum() * raw_return;
        // Mirror image of `chartist_payoff` by construction (module doc
        // comment's "Chartist vs. contrarian sub-switching" section): betting
        // against last wake's chartist signal scores exactly the opposite of
        // betting with it.
        let contrarian_payoff = -chartist_payoff;
        let leans_trend_follow = s.chartist_fitness >= s.contrarian_fitness;

        // Salience: this update's realized move, relative to this agent's own
        // "one unit" price-signal threshold, scales the learning rate up the
        // more dramatic it was -- a quiet move teaches almost nothing new, a
        // big one can dominate the agent's whole impression at once.
        let salience = |payoff: f64| 1.0 + s.salience_sensitivity * (payoff.abs() / s.price_signal_threshold.max(1e-9)).min(s.salience_cap);

        // Confirmation bias: full-rate learning when this payoff confirms
        // whichever strategy the agent is *currently* leaning toward;
        // discounted when it says the current lean was wrong.
        let confirms_chartist = leans_chartist == (chartist_payoff >= 0.0);
        let confirms_fundamentalist = !leans_chartist == (fundamentalist_payoff >= 0.0);
        let chartist_rate = (s.fitness_learning_rate * salience(chartist_payoff) * if confirms_chartist { 1.0 } else { s.confirmation_bias_strength }).clamp(0.0, 0.95);
        let fundamentalist_rate = (s.fitness_learning_rate * salience(fundamentalist_payoff) * if confirms_fundamentalist { 1.0 } else { s.confirmation_bias_strength }).clamp(0.0, 0.95);

        s.chartist_fitness = (1.0 - chartist_rate) * s.chartist_fitness + chartist_rate * chartist_payoff;
        s.fundamentalist_fitness = (1.0 - fundamentalist_rate) * s.fundamentalist_fitness + fundamentalist_rate * fundamentalist_payoff;

        // Same salience/confirmation-bias update, one level deeper: whether
        // *this* agent currently leans trend-following or contrarian (not
        // chartist-vs-fundamentalist -- a separate, independent lean; an agent
        // can trust momentum-type reasoning over value reasoning while also
        // thinking the current move specifically is due to reverse). Only
        // `contrarian_fitness` is new state here -- `chartist_fitness` already
        // *is* "how well trend-following has paid off," updated above at
        // `chartist_rate`, so it's reused as-is rather than updated twice.
        let confirms_contrarian = !leans_trend_follow == (contrarian_payoff >= 0.0);
        let contrarian_rate = (s.fitness_learning_rate * salience(contrarian_payoff) * if confirms_contrarian { 1.0 } else { s.confirmation_bias_strength }).clamp(0.0, 0.95);
        s.contrarian_fitness = (1.0 - contrarian_rate) * s.contrarian_fitness + contrarian_rate * contrarian_payoff;
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
    // Capped (2026-10-28): unlike `price_component` above, whose input (a recent
    // return) is naturally bounded by how far price can move in one wake,
    // `deviation` has no such ceiling -- once `mid` has already drifted far from
    // `fair_value` (e.g. after a prior leg traded at this bot's own 0.5x-2x
    // `believed_value` clamp, see that clamp's doc comment below), dividing a
    // 50%+ deviation by the tiny default `fundamentalist_signal_threshold`
    // (0.02) produces a component in the tens, which alone is then enough to
    // slam the *next* wake's `believed_value` straight to the opposite clamp
    // extreme -- turning what should be gentle mean-reversion into a
    // self-sustaining overshoot. Confirmed directly: a `human_27_5pct` leg
    // where no bots happened to trade saw humans alone clear 19 trades at
    // ~0.5x fair value, the clamp bound, with no single-trade or circuit-
    // breaker anomaly behind it -- just this term pushing belief there on its
    // own. Only the magnitude is capped, not the sign, and the uncapped
    // `fundamentalist_component` computed here is still what feeds
    // `last_fundamentalist_signal` for Brock-Hommes fitness tracking below
    // (via its `.signum()`, which this cap can't change) -- this only tames how
    // hard a single already-large deviation can push *this* wake's own belief.
    let fundamentalist_component = (-deviation / s.fundamentalist_signal_threshold.max(1e-9)).clamp(-8.0, 8.0);

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

    // Chartist vs. contrarian sub-blend (module doc comment's "Chartist vs.
    // contrarian sub-switching" section) -- how much this agent's own momentum
    // reading flips against `price_component` rather than following it.
    // *Not* the same symmetric logistic shape as the blend just above: that
    // blend picks between two genuinely independent strategies, so a 50/50 tie
    // is a sensible "average them" default. Trend-follow and contrarian here
    // are exact opposites of the same reading, so a naive 50/50 blend between
    // them cancels to exactly zero -- which would silently erase the momentum
    // signal for every agent at its fitness-neutral starting point (both at
    // 0.0, confirmed directly: broke a disposition-effect test depending on a
    // real sell signal from a fresh, never-stepped agent). Contrarian weight
    // instead ramps up from 0 (pure trend-follow, the original, unmodified
    // default, matching the Lite-variant behavior below) only once
    // `contrarian_fitness` genuinely overtakes `chartist_fitness`, saturating
    // toward 1 (full flip) as that lead grows -- never engaged while
    // trend-following is still winning or tied.
    let chartist_signal = match s.variant {
        Variant::Retail => {
            let contrarian_gap = s.intensity_of_choice * (s.contrarian_fitness - s.chartist_fitness);
            let contrarian_weight = (1.0 / (1.0 + (-contrarian_gap).exp()) - 0.5).max(0.0) * 2.0;
            (1.0 - 2.0 * contrarian_weight) * price_component
        }
        Variant::Lite => price_component,
    };
    // `trend_damping`/`counter_trend_scale` (2026-11-05, direct user request --
    // see their own field comments): the learned chartist-vs-contrarian blend
    // above still runs unchanged; this adds a damping factor on the whole
    // trend blend plus a standing, always-on counter-trend pull scaled
    // directly off `price_component` -- continuous, not a gated override, so
    // it folds into this same blended value like every other component here
    // rather than branching separately later in `step()`.
    let trend_component = s.trend_damping * (chartist_weight * chartist_signal + (1.0 - chartist_weight) * fundamentalist_component)
        - s.counter_trend_scale * price_component;

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

    // Grouped crowd trend removed (2026-10-17, direct user request) -- this
    // used to blend in a shared `Market::random_trend` line that every agent
    // in the same group read identically (see `market.rs`'s removal note).
    // `raw_signal` is now just this agent's own private signal.
    let raw_signal = market_signal;
    s.last_price = Some(mid);
    s.last_wake_time = Some(ctx.now);
    s.last_chartist_signal = price_component;
    s.last_fundamentalist_signal = fundamentalist_component;
    // Fast-crash panic selling (module doc comment) needs to know whether the
    // fall was already underway *before* this wake, not just whether this
    // wake's own move was severe -- captured into a local before this wake's
    // `price_component` overwrites it, so the "still falling" check below
    // compares two genuinely separate sample points rather than one value
    // against itself.
    let prior_price_component = s.prev_price_component;
    s.prev_price_component = price_component;

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
    let signal_acts = intensity > 0.0 && agent.rng.gen_bool(s.p_act_given_signal);

    // Spontaneous profit-taking (module doc comment's "Spontaneous profit-taking"
    // section): only considered when the signal path above *didn't* already give
    // this wake a reason to act -- this fills the gap on an otherwise flat/quiet
    // tape, it doesn't pile onto a wake that's already trading. Fires purely off
    // the unrealized gain on the current position, never a loss (CPT value < 0
    // is "hold and hope," not a trigger to act).
    let spontaneous_take_profit = !signal_acts
        && s.avg_entry_price.is_some_and(|entry| {
            let r = (mid - entry) / entry.max(1e-9);
            let v = cpt_value(r, s.cpt_alpha, s.cpt_beta, s.cpt_lambda);
            // Realization utility (module doc comment's "Realization utility
            // near a position's own high" section): an extra, additive urge to
            // sell specifically near this position's `position_high_water_mark`
            // -- on top of, not instead of, the plain unrealized-gain term just
            // above. Fades linearly to 0 once price has pulled back
            // `realization_utility_window` below that high.
            let near_high = s.position_high_water_mark.map_or(0.0, |hwm| {
                let pullback = ((hwm - mid) / hwm.max(1e-9)).max(0.0);
                (1.0 - (pullback / s.realization_utility_window.max(1e-9)).min(1.0)) * s.realization_utility_scale
            });
            // Convexity on top of the plain CPT value (module doc comment's
            // "More extreme profit-taking at large gains" section, 2026-10-25):
            // `v` alone is already *concave* in the raw gain (`cpt_alpha` < 1),
            // which compresses the probability gap between a modest and a huge
            // winner. Raising it to `spontaneous_profit_convexity` (> 1) widens
            // that gap back out and past linear -- a modest 5-20% gain fires only
            // rarely, while a genuinely large winner (triple digits) becomes a
            // near-certainty to realize *some* of within a few wakes, matching
            // real profit-taking euphoria on outsized winners rather than a flat
            // per-percent-of-gain rate.
            let scaled_v = v.max(0.0).powf(s.spontaneous_profit_convexity);
            v > 0.0 && agent.rng.gen_bool((s.spontaneous_profit_taking_rate * scaled_v + near_high).clamp(0.0, 1.0))
        });

    // Fast-crash panic selling (module doc comment's "Rising-trend profit-taking
    // override + fast-crash panic selling" section): computed independently of
    // `signal_acts`, like `spontaneous_take_profit`, so a sharp enough drop can
    // push this agent to act through a wake the ordinary inertia roll would
    // otherwise have skipped. Uses `price_component` -- the raw, belief-
    // independent price move -- not this agent's own chartist/contrarian-
    // adjusted reading, since panic reacts to what the tape actually did, not
    // to this agent's theory about it. Requires an open long position (never
    // short, same rule as everywhere else in this file).
    //
    // Two gates, both required (2026-10-24 revision, direct user request --
    // "only do if it's really bad and the falling hasn't stopped yet"):
    // `panic_sell_signal_multiple` alone already picks out a genuinely severe
    // move (well past the ordinary alert dead zone), but a single severe wake
    // could just as easily be a move that already bottomed out and started
    // recovering by the time this agent notices -- panicking into a bounce
    // isn't the behavior being modeled. `prior_price_component < 0.0` requires
    // the *previous* wake's move to also have been a fall, so this only fires
    // while the decline is still actively in progress across consecutive
    // check-ins, not off one dramatic-but-possibly-already-over data point.
    // Continuous replacement (2026-11-05, direct user request: a Bernoulli
    // coin-flip gate on each of these four was "single events" that either
    // fire at full multiplier or do nothing that wake -- lumpy, and makes no
    // sense as a model of a crowd's aggregate behavior). Each `_boost` below
    // is the exact continuous *expectation* of what the old
    // `gen_bool(p) -> qty *= mult` mechanism produced on average per wake --
    // `E[mult applied] = (1-p)*1 + p*mult = 1 + (mult-1)*p` -- so the mean
    // effect across many wakes/agents is unchanged, but it's now a smooth,
    // deterministic function of severity/duration instead of a random
    // on/off switch. `severity`/`fatigue` themselves are unchanged from the
    // old gating logic (crash/rally still has to be actively in progress;
    // fatigue still needs `wakes_underwater`/`wakes_missed_rally` past their
    // threshold) -- only the probabilistic trigger is gone, not the
    // preconditions.
    let crash_severity = if inventory > 1e-9 && prior_price_component < 0.0 {
        (-price_component - s.panic_sell_signal_multiple).max(0.0)
    } else {
        0.0
    };
    let crash_boost = (s.panic_sell_qty_mult - 1.0) * (s.panic_sell_prob_scale * crash_severity).min(1.0);

    let underwater_fatigue = if inventory > 1e-9 {
        (s.wakes_underwater as f64 - s.capitulation_fatigue_threshold_wakes).max(0.0)
    } else {
        0.0
    };
    let underwater_boost = (s.capitulation_qty_mult - 1.0) * (s.capitulation_fatigue_prob_scale * underwater_fatigue).min(1.0);

    let rally_severity = if prior_price_component > 0.0 {
        (price_component - s.panic_buy_signal_multiple).max(0.0)
    } else {
        0.0
    };
    let rally_boost = (s.panic_buy_qty_mult - 1.0) * (s.panic_buy_prob_scale * rally_severity).min(1.0);

    // No inventory gate here, mirroring the old `panic_buy`'s own reasoning:
    // FOMO buying is about an agent who *isn't* positioned enough chasing in,
    // not reducing an existing position.
    let missed_rally_fatigue = (s.wakes_missed_rally as f64 - s.fomo_fatigue_threshold_wakes).max(0.0);
    let missed_rally_boost = (s.fomo_qty_mult - 1.0) * (s.fomo_fatigue_prob_scale * missed_rally_fatigue).min(1.0);

    // Combined continuous pressure in each direction -- > 0.0 whenever either
    // of that direction's two components (acute + chronic) is active at all.
    let sell_pressure = crash_boost + underwater_boost;
    let buy_pressure = rally_boost + missed_rally_boost;

    let acts_this_time = signal_acts || spontaneous_take_profit || sell_pressure > 0.0 || buy_pressure > 0.0;

    if acts_this_time {
        // `spontaneous_take_profit` only ever reaches here on its own terms
        // (independent of `signal_acts`, see above), so there's no real
        // `raw_signal` direction to defer to -- always a sell, by
        // construction (module doc comment). `sell_pressure`/`buy_pressure`
        // being continuous rather than boolean, a tie (both zero, or equal
        // and nonzero -- the latter shouldn't happen in practice: the
        // sell-side components require an open long, the buy-side ones'
        // duration counters only run while flat) resolves deterministically
        // to Sell, same as before.
        let mut side = if spontaneous_take_profit || (sell_pressure > 0.0 && sell_pressure >= buy_pressure) {
            Side::Sell
        } else if buy_pressure > 0.0 {
            Side::Buy
        } else if raw_signal > 0.0 {
            Side::Buy
        } else {
            Side::Sell
        };

        // Rising-trend profit-taking override (module doc comment): even
        // when the ordinary signal path was about to buy more, sometimes
        // sell into strength instead -- only considered when this wasn't
        // already forced to Sell by the two cases above, and only when
        // there's an actual open gain to take.
        if side == Side::Buy {
            if let Some(entry) = s.avg_entry_price {
                if inventory > 1e-9 {
                    let r = (mid - entry) / entry.max(1e-9);
                    let v = cpt_value(r, s.cpt_alpha, s.cpt_beta, s.cpt_lambda);
                    if v > 0.0 && agent.rng.gen_bool((s.rising_profit_take_prob_scale * v).min(s.rising_profit_take_max_prob)) {
                        side = Side::Sell;
                    }
                }
            }
        }
        // "Hot hand" sizing: bet size grows after wins, shrinks after losses --
        // see module doc comment. Revenge-trading ("get-even" loss-chasing,
        // betting *bigger* after a loss) used to be a per-agent reversal of
        // this blended in here; removed (2026-10-17, direct user request)
        // after the crash investigation identified it as a direct amplifier
        // of correlated human herding -- see market_maker.rs's module doc
        // comment for the fuller story this was found during.
        let pnl_size_mult = (1.0 + s.pnl_sizing_sensitivity * s.recent_pnl_frac).clamp(0.2, 2.5);
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
                if sell_pressure > 0.0 {
                    // Continuous override on top of the disposition
                    // hold-and-hope suppression just applied above, rather
                    // than feeding into it -- see `sell_pressure`'s own
                    // comment above for why this is `1.0 + pressure` rather
                    // than a gated fixed multiplier.
                    qty *= 1.0 + sell_pressure;
                }
                // Never short: real retail traders overwhelmingly don't, and (like
                // noise.rs) an unconstrained sell here would be unlimited,
                // uncapitalized short pressure -- see git history for why that
                // matters, not just style.
                qty = qty.min(inventory.max(0.0));
            }
            Side::Buy => {
                if buy_pressure > 0.0 {
                    // Mirrors `sell_pressure`'s continuous override above --
                    // applied before the position/cash caps just below so
                    // it's still subject to the same affordability limits as
                    // any other buy, same as the sell side still gets capped
                    // by `inventory` at the end of the Sell branch.
                    qty *= 1.0 + buy_pressure;
                }
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
            // wake-ups into a runaway spiral. `fair_value` is the one process in
            // this engine that's provably immune to being corrupted by trading
            // activity itself (see market.rs's doc comment), so anchoring the
            // clamp to it bounds the worst case regardless of how far a
            // one-sided book has already drifted -- while the *base*
            // computation still uses `mid`, so this agent's own signal still
            // moves the belief around within that band.
            //
            // Tried tightening this band to (0.8, 1.25) (2026-10-24, direct
            // user request: "fine tune the simulation so the worth of the
            // stocks... represent the real world") on the theory that the
            // original (0.5, 2.0) was only ever loose enough to matter
            // because it was sized against a since-fixed bug in `market.rs`'s
            // `tick_mark_and_fundamental` that made `fair_value` ~5,600x more
            // volatile than intended. Measured directly instead of assumed:
            // it made things *worse*, not better -- 7 otherwise-identical
            // human_80pct seeds that previously split between big up-moves
            // and big down-moves all turned into big up-moves (30-99%) with
            // the tighter band. Root cause isn't this clamp: every agent in
            // this engine (human and most bot types) starts with zero
            // inventory and only cash, so the entire population's earliest
            // trades are structurally forced to be buys -- a population-wide
            // bootstrap bias this clamp can't address and that a 25,000s
            // warmup apparently doesn't fully work off. Left at the original
            // (0.5, 2.0) pending an actual fix at the source (e.g. seeding
            // non-zero starting inventory) rather than papering over it here.
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
                Some(p) if too_far(p) => {
                    // Unattractive price -- normally this agent patiently posts a
                    // limit at its own belief instead. A roll against
                    // `price_chase_prob` (see module doc comment's "too smart to
                    // fail" section) instead crosses impulsively at the bad price
                    // with a market order, same as a real panicking/euphoric
                    // trader sometimes does.
                    !agent.rng.gen_bool(s.price_chase_prob)
                }
                // **General price-impact patience (2026-10-17, direct user
                // request)**: an *attractive* price used to always be taken
                // outright with a market order -- correct in the sense that a
                // real trader doesn't ignore a good deal, but measured to be
                // this bot's dominant source of its own price impact (far more
                // common than the `too_far` chase case above, which only fires
                // when the price already looks bad). Most agents now mostly
                // rest a limit at their own belief even on a good price instead
                // of immediately taking it -- still a real trade (it'll likely
                // fill passively once someone else crosses into it), just not
                // one *this* agent pushed the price with. A roll against
                // `take_good_price_prob` still takes it outright sometimes, same
                // heterogeneity convention as `price_chase_prob` (most agents'
                // draw is low; a minority stay closer to the old always-take
                // behavior).
                Some(_) => !agent.rng.gen_bool(s.take_good_price_prob),
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
