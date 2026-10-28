//! Market-maker bots: 4.1 in the spec. All four variants requote periodically
//! (cancel previous quotes, place new two-sided post-only quotes) rather than
//! reacting tick-by-tick, matching real MM requote cadences (~0.1-1s).
//!
//! **Active de-risking (2026-10-05)**: measured in a 41-scenario x 5-seed
//! sweep that ~86-97% of market_maker agents end the run bankrupt (a drop to
//! 20% of starting net worth), *every* variant included -- even
//! `AdaptiveSpread`, the one variant that already widens/withdraws under
//! volatility. That ruled out "only the naive variants are missing a
//! defense": the gap is common to all four. Each variant above only ever
//! adjusts the *next resting quote* (wider spread, skewed center, or no
//! quote at all) -- none of them can act on a position it has *already*
//! accumulated. A real market maker that's gotten run over by sustained
//! directional flow actively crosses the spread to flatten part of the
//! position; this bot could previously only requote passively and wait for
//! its own resting ask/bid to get hit, even while the mark price kept moving
//! further against the inventory it was already holding. The fix below
//! applies to every variant uniformly, after its own quote logic: once
//! inventory crosses this agent's own `derisk_inventory_frac` of
//! `max_inventory`, submit an IOC market order on the de-risking side, sized
//! to only a small fraction of the excess per wake (`derisk_frac_per_step`).
//!
//! First version of this (uniform 0.7 threshold, 0.3 unwind fraction, fired
//! on skew alone) was measured to make things *worse*, not better: 100% of
//! market_makers bankrupt by 20,000s of a 150,000s run, vs. 84-96% over the
//! same window with de-risking disabled (A/B'd on the same seed/scenario).
//! Root cause: with one shared threshold, agents facing the same market
//! conditions tend to cross it together, so they all dump/buy in the same
//! direction at once -- a self-inflicted stampede that pushes price further
//! against the very position they're trying to reduce, compounding the
//! original loss with fresh taker-side slippage. Three fixes, all below: (1)
//! `derisk_inventory_frac` is now drawn once per agent at construction from a
//! range, not one shared constant, so agents don't all trip the same
//! threshold simultaneously; (2) `derisk_frac_per_step`'s default is far
//! smaller (a slow trickle, not a third of the excess every half-second);
//! (3) de-risking now also requires the position to actually be
//! *underwater* (current mid worse than `ema_mid`, a slow trailing reference
//! price, in the direction that hurts this agent's current position) --
//! merely being skewed is not itself a reason to sell into a move that's
//! about to reverse.
//!
//! **Four more fixes (2026-10-10)**: even with de-risking above, measured in
//! the `toosmart_fix_v1` sweep that market_maker bankruptcy had gotten
//! *worse* (90-99.9%, often already past 90% by the end of leg 1) since the
//! human bot picked up its own new aggressive-sizing/revenge-trading/
//! leverage behaviors the same night -- human-driven flow can now be larger
//! and more sustained than when de-risking's unwind rate was tuned, so
//! inventory builds faster than the old flat trickle can clear it. Four
//! structural gaps, independent of that specific cause:
//!   1. **Capital-unaware sizing**: `order_notional`/`max_inventory_notional`
//!      were fixed dollar amounts from construction, never shrinking as this
//!      agent's own equity does -- a real market maker de-leverages as
//!      losses mount; this one kept quoting the same size right up until
//!      bankrupt. Now `max_inventory` is recomputed every wake as a fraction
//!      of *current* equity (`cash + inventory * mid`, no margin -- this bot
//!      never holds one), floored at `min_inventory_notional` so a
//!      depleted-but-not-yet-bankrupt agent keeps a token presence rather
//!      than going fully dark; `order_qty` is a fraction of that, preserving
//!      the original 3000/30000 = 0.1 ratio as the default.
//!   2. **No passive taper**: quoted the *same* full size on both sides
//!      regardless of current inventory, until a hard wall at
//!      `inventory >= max_inventory` flipped a binary gate. Now the size on
//!      whichever side would *increase* an already-large position shrinks
//!      linearly as `inventory_norm` approaches that cap, reaching zero
//!      exactly at it -- accumulation slows itself down organically well
//!      before active de-risking needs to clean up after the fact.
//!   3. **Two of four variants ignored market stress entirely**:
//!      `FixedOffset` and `InventorySkew` quoted the same static spread in a
//!      calm market and a toxic one -- only `AdaptiveSpread` reacted at all.
//!      A shared, more modest toxicity/volatility spread multiplier
//!      (`stress_vol_scale`/`stress_toxicity_scale`) now applies to every
//!      *other* variant too (`AdaptiveSpread` keeps its own, already more
//!      aggressive, dedicated widening/withdrawal unchanged, so this isn't
//!      doubly applied there).
//!   4. **De-risking's unwind rate was flat regardless of severity**: a
//!      position 5% underwater and one 40% underwater unwound at the same
//!      rate. Now scaled by how far mid has actually moved against this
//!      position since `ema_mid` (the same reference the `underwater` gate
//!      already checks), capped (`drawdown_derisk_cap`) so a severe move
//!      still can't dump the whole excess in one step.
//!
//! Fixes 1-4 together measured a real improvement (market_maker bankruptcy
//! 98.2% -> 80.8% in a direct `toosmart_fix_v1`-scale comparison, same seed)
//! but didn't fix it -- some agents' equity still measured deeply negative.
//! **Tried and reverted (2026-10-10): a hard stop-loss**, past a drawdown
//! threshold, abandoning the proportional trickle (fix 4, deliberately
//! capped so one bad step can't itself dump a huge order into a thin book)
//! for flattening the *entire* position in one IOC order instead. Measured,
//! same seed/scale: bankruptcy *count* was unchanged (236/292 -- bankruptcy
//! is a sticky flag once tripped, the stop-loss didn't stop anyone from
//! crossing it) and minimum equity got *dramatically worse* (-$280,557 ->
//! -$1,191,023). Dumping an entire position in one shot into a thin/stressed
//! book doesn't just take "somewhat worse execution" -- it walks the book
//! down across many price levels, and the realized slippage from that was
//! far larger than the loss it was meant to cap. A real phenomenon (panic-
//! selling into illiquidity is a classic way to turn a bad position into a
//! much worse one), but it means one-shot flattening is net harmful here,
//! not helpful. A staggered flatten (faster than fix 4's trickle but still
//! sliced over a few wakes, reusing the existing `StaggeredTwap` pattern
//! already in `liquidation_trader.rs` for leveraged-position liquidation)
//! is the more promising next direction if this gets revisited -- not
//! attempted yet.
//!
//! **Laddered quoting + panic withdrawal (2026-10-14)**: direct user
//! hypothesis, A/B tested on human_80pct (single seed): concentrating 100%
//! of a side's size at one best bid/ask means a single aggressive trade
//! fills the whole thing at one, instantly-stale price -- exactly what the
//! momentum/arbitrage-driven jump events (see find_volatility_culprit.py's
//! archive-wide scan) do to this bot. Splitting the *same total* size across
//! `n_ladder_levels` price levels makes an attacker walk through
//! progressively worse prices for the same size, the same principle
//! `grid.rs` already uses. Measured: market_maker's aggregate end-of-run
//! equity flipped from -$181,359 to +$921,253 (same seed), worst-case min
//! equity improved 38%, and -- unlike the earlier predictive-skew-only
//! experiment below -- most other bot types also improved (arbitrage,
//! mean_reversion, mev, liquidation_trader), not just market_maker; only
//! grid/momentum got mildly worse and total market volatility ticked up a
//! mild +6% (vs. the skew experiment's systemic +9% that hurt everyone).
//! Paired with a generalized panic withdrawal: cancel the *whole* ladder and
//! skip quoting entirely for a wake once realized vol or the momentum signal
//! (below) crosses a threshold -- generalizes `AdaptiveSpread`'s existing
//! single-order `withdrawal_vol_frac` to every variant and to the whole
//! ladder, not just one resting order per side.
//!
//! **Predictive momentum signal (2026-10-14)**: `vol_window`/`toxicity_window`
//! are both direction-agnostic (std of signed returns, mean of |return|) --
//! they measure how noisy the market has been, not whether it's been pushed
//! one direction. `momentum_window` (a short window of *signed* returns)
//! catches that instead. Tried standalone, applied as `predictive_skew_scale`
//! (shift `reservation` toward the detected direction, "lead the price") +
//! `predictive_widen_scale` (extra spread widening) + `predictive_size_scale`
//! (shrink risk budget proactively): A/B tested on human_80pct (single
//! seed), market_maker's aggregate end-of-run equity improved (less deeply
//! negative) but total market volatility rose +9% and every *other* major
//! bot type (arbitrage, mean_reversion, momentum, liquidation_trader) got
//! measurably worse. Root cause: skewing toward recent momentum makes this
//! agent *procyclical* -- it stops damping moves and starts leaning into
//! them, which is a real, known failure mode of naive adverse-selection
//! avoidance, not specific to this implementation. Left in (all three scales
//! default 0.0, i.e. off) since `momentum_window`/`momentum_signal` are
//! still useful as a per-asset predictive signal for the panic-withdrawal
//! and ladder/de-risking logic above and below -- just without the skew
//! component actively steering price, which is the piece that was
//! net-harmful to the wider market.
//!
//! **Multi-asset risk allocation (2026-10-14)**: direct user request --
//! previously each individual market_maker agent was assigned exactly one
//! fixed asset at construction (round-robin across `cfg.assets`, see
//! `sim.rs`'s `build_population`), so "risk management" only ever meant
//! managing risk *within* that one market. Real market makers run books
//! across many instruments simultaneously and size each one by its own
//! risk, not just its own slice of a fixed per-asset budget. Every
//! market_maker agent now manages *every* configured asset (`State.assets`,
//! one `AssetState` per asset, replacing the single `asset_id`/vol_window/
//! etc. fields), computing each asset's own realized volatility every wake
//! and allocating this agent's *total* equity-scaled risk budget across
//! them by inverse volatility (a standard risk-parity weighting: a calmer
//! asset gets a bigger slice, a more volatile one a smaller slice) --
//! instead of each asset getting a fixed, equal-by-construction budget
//! regardless of how risky it currently is. Aggregate notional risk budget
//! per asset is designed to land in the same ballpark as before (previously
//! ~1/5 of market_maker agents each deployed their *full* budget into one
//! asset; now every agent deploys ~1/5 of its budget, on average, into each
//! of 5 assets -- same aggregate exposure per asset at equal weights, now
//! tilted away from whichever asset is currently riskiest). Panic withdrawal
//! is now per-asset (skip quoting *that* asset only, not the whole agent) --
//! one asset spiking shouldn't silence this agent's otherwise-calm markets.
//!
//! **Seeding new agents from market history (2026-10-14)**: direct user
//! request, following a counterintuitive result from `staged_introduction_
//! test` -- introducing bot types gradually (humans alone first, then
//! calmer bots, then market_maker/arbitrage/mev last) made market_maker's
//! collapse *faster*, not slower (58% bankrupt within 8 minutes of
//! entering, vs. ~2% at the same point when everyone starts together cold).
//! Root cause: every rolling window (`vol_window`/`toxicity_window`/
//! `momentum_window`) starts completely empty at construction, regardless
//! of whether the *market* itself has history -- a newly-grafted
//! market_maker had no sense that momentum/mean_reversion had already been
//! trading for 10,000s and the market was already "hot," so it quoted its
//! first several wakes exactly as if the book were brand new. Fixed at the
//! source: `Market` now keeps a small rolling buffer of each asset's own
//! recent log-returns (`Market::recent_log_returns`, sampled once per
//! `SnapshotTick`, independent of any agent's own wake cadence -- see that
//! field's doc comment in `market.rs`), and every new `AssetState` is now
//! seeded from it at construction (`AssetState::new`, via `seeded_window`)
//! instead of starting blank -- a no-op for the very first population ever
//! built (there's no history yet), but real seed data for anything grafted
//! into an already-running market. `ema_mid`/`last_mid` are similarly
//! seeded from the market's current price instead of `None`.
//!
//! **Price-push de-risking, replaces aggressive IOC orders (2026-10-14)**:
//! direct user request -- see `State::derisk_price_push_scale`'s doc
//! comment for the full "why" (the old mechanism's own aggressive buy-
//! backs were found to be the dominant driver, by trade count, of the
//! exact price spikes they were reacting to, a self-inflicted stampede).
//! `step()` now computes `underwater`/`drawdown_frac`/`derisk_push_frac`
//! *before* quoting (not after, as a separate reactive order afterward),
//! and spends that signal entirely on asymmetric pricing: the side that
//! would worsen the position (ask when short, bid when long) gets pushed
//! further from `reservation`, while the correcting side is left alone to
//! fill passively. No `Action::Submit` with `OrderType::Market` is issued
//! for this any more -- de-risking is now 100% a pricing decision, never
//! a trade this agent itself initiates.
//!
//! **Microprice reservation anchor + self markout feedback (2026-10-17)**:
//! direct user request, following a redesign discussion grounded in how real
//! market makers operate (order-book imbalance, realized-P&L feedback) --
//! this bot previously only ever priced off `mid` (last trade) and reacted to
//! *market*-state heuristics (realized vol/toxicity/momentum), never the
//! order book itself, and had no memory of whether its own quoting was
//! actually making money. Two independent additions, below:
//!
//!   1. **Microprice**: `reservation`'s anchor is now a depth-weighted price
//!      (`microprice()`, below) instead of raw `mid` -- weights each side's
//!      price by the *opposite* side's resting top-of-book quantity, so heavy
//!      bid size (strong buying interest, thin offer) pulls the anchor toward
//!      the ask, anticipating the move that imbalance usually precedes. Only
//!      replaces `mid` as the quote *center*; `mid` itself is unchanged
//!      everywhere else (risk sizing, panic thresholds, the clamp that still
//!      bounds how far this can push the quote) -- a bad imbalance read can
//!      shift where this agent quotes, never how much risk it's willing to
//!      take.
//!   2. **Markout feedback**: every fill (detected the same way wait-and-
//!      reinforce already does, via `level_being_touched`) is queued as a
//!      pending observation and resolved `markout_horizon_secs` later into
//!      `AssetState::markout_window`: did price move against this *specific*
//!      trade (bought then price fell / sold then price rose)? A negative
//!      running average is a direct, measured "I'm getting picked off on
//!      this asset" -- not inferred from a volatility proxy -- fed back as
//!      extra spread widening and a smaller risk budget on that asset only.
//!      One-directional by design (never rewards a positive markout with
//!      tighter/bigger quoting) -- this closes the loop between realized
//!      outcome and quoting parameters without introducing a new way for the
//!      fleet to correlate into a stampede, the recurring failure mode in
//!      this module's own history above.
//!
//! **Empirical kappa calibration -- tried, disabled (2026-10-17)**: direct
//! user request, idea #3 of the same redesign session as microprice/markout
//! above. `SymmetricAvellanedaStoikov`'s order-arrival-decay parameter `k`
//! was a fixed textbook default, identical for every asset; `AssetState`
//! gained a simple two-point fit (EMA fill rate at the nearest and farthest
//! ladder level, paired with the distance each was actually quoted at) to
//! re-derive it per asset from this agent's own real execution data instead.
//! Measured net-harmful on human_80pct (overall market growth +68.0% ->
//! -11.8%, human's own growth +44.9% -> +3.6%, grid -58.9% -> -92.5%,
//! market_maker itself unmoved at ~-83%): the fitted spread width shapes what
//! gets measured as "near"/"far" fill-rate on the *next* wake, which then
//! re-shapes the fit again -- a closed feedback loop between quoting and its
//! own training data, not an independent signal the way real exchange
//! fill-rate data would be. `State::empirical_k_enabled` defaults false; the
//! calibration machinery stays for future rework.
//!
//! **Cross-asset hedging via correlation -- tried, disabled (2026-10-17)**:
//! direct user request, idea #4. Multi-asset risk allocation above only
//! ever *sized* each asset's budget by inverse volatility -- it never let
//! one asset's position influence another's pricing. `asset_correlation()`
//! computes pairwise Pearson correlation from `Market::recent_log_returns`;
//! each asset's reservation was made to also skew by a correlation-weighted
//! sum of every *other* managed asset's own `inventory_norm` (being long a
//! correlated asset is directionally similar risk to being long this one),
//! the same subtractive form `InventorySkew` already uses on its *own*
//! position. Measured net-harmful in isolation on human_80pct: trade count
//! -62%, market_maker's own volume share collapsed 32.6% -> 9.9%, equity
//! *unchanged* (no better than without it). Unlike the per-asset de-risking
//! mechanisms, which only activate when that specific asset is actually
//! underwater, this skew applied *every* wake regardless of magnitude --
//! with several correlated assets the signal rarely nets to zero, so it
//! acted as a near-constant skew rather than an occasional correction,
//! chronically making one side unattractive and suppressing fills.
//! `State::cross_asset_hedge_scale` defaults 0.0; machinery stays for future
//! rework (e.g. gating on the signal's own magnitude).
//!
//! **Depth-collapse panic trigger -- kept, measured net-positive
//! (2026-10-17)**: direct user request, idea #5.
//! `panic_vol_frac`/`panic_momentum_frac` above only confirm a move *after*
//! realized price/vol has actually shifted. A slow EMA baseline of
//! near-touch resting depth, each side, catches a *developing* one -- a
//! sudden drop well below that baseline (liquidity visibly pulling away) is
//! a documented pre-adverse-selection signal, and now panic-withdraws the
//! same way the existing vol/momentum triggers do. The one idea out of five
//! tried in this redesign session that actually helped in isolation:
//! measured on human_80pct, market_maker's equity growth improved from
//! -83.1% to -63.5%, bankruptcy improved (0.102 -> 0.068 terminal rate), and
//! -- unlike every other idea tried here -- trading activity was *preserved*
//! (trade count 342k -> 334k, market_maker's own volume share actually rose
//! 32.6% -> 49.3%) rather than achieved by the agent simply withdrawing from
//! the market. `depth_collapse_frac` defaults to a real (not disabled)
//! 0.35.
//!
//! **Loss-timing investigation + two new candidate fixes (2026-10-17)**:
//! direct user request, autonomous follow-up after tracing this run's worst
//! loss legs. Every major loss leg coincided with an extreme price swing
//! (70%-570% within one 500s leg) in one of the two cheapest/thinnest
//! assets, and in every case this agent's *entire* fleet (117/117 agents in
//! one traced case) had drifted to the *same side* of that asset over many
//! prior legs before the move -- not random per-agent noise, a systematic
//! drift. Two independent candidate fixes, both left at a safe/no-op
//! default pending the test results that follow this comment in the file's
//! own history:
//!   1. **Steeper risk-parity exponent**: `risk_aversion_exponent` raises
//!      sigma to a power before inverting for the multi-asset risk-parity
//!      weight, instead of linear inverse-vol -- shrinks a currently-
//!      volatile asset's budget faster than linearly, directly capping how
//!      large a position can accumulate on exactly the assets where this
//!      kept happening. 1.0 reproduces the original linear weighting.
//!   2. **Sustained-drift extra de-risking**: a *slow* EMA of this asset's
//!      own `inventory_norm` (`AssetState::inventory_drift_ema`), separate
//!      from the instantaneous underwater check -- catches "this fleet has
//!      leaned the same way on this asset for a long time" proactively,
//!      before any specific move has turned against it yet, which is
//!      exactly the gap the traced loss legs fell through (the existing
//!      underwater gate only fires once a move is *already* happening).
//!
//! **Follow-up on the two 2026-10-17 candidates, plus two more tried and
//! reverted (2026-10-17, later the same day)**: direct user request ("i want
//! the market makers to win sth" -- depth-collapse alone, while a real and
//! robust improvement, still leaves this bot net-negative). Multi-seed/
//! multi-composition results for the two candidates above: (1) steeper
//! risk-parity exponent looked like a large win on one seed, then failed on
//! 3 more seeds plus a second composition (average effect flat-to-negative)
//! -- reverted to its no-op default (1.0), claim retracted. (2) sustained-
//! drift extra de-risking was flat-to-harmful at every strength tried, even
//! combined with (1) -- left at its no-op default (0.0).
//!
//! Two further ideas, tried and fully reverted (not just defaulted off --
//! the code itself, since both measured unambiguously harmful with no
//! redeeming scale): (3) **uniform spread widening** -- scaling
//! `base_half_spread_frac`/`gamma` together by a flat multiplier (1.5x-3x),
//! reasoning this bot simply isn't charging enough for the risk it takes.
//! Backfired at every multiplier tried: `market_maker`'s own growth got
//! *worse* (seed 42, human_80pct: -8.2% baseline -> -20.0% at 1.5x, -16.2%
//! at 2x, roughly flat at 3x), while `arbitrage`/`mev`/`liquidation_trader`
//! growth exploded (triple-digit % gains) and this bot's own volume share
//! collapsed (35.1% -> 17.0-25.4%). (4) **per-asset vol/toxicity-aware base
//! spread for `FixedOffset`/`InventorySkew`** -- same idea targeted only at
//! the actually-toxic assets (reusing `AdaptiveSpread`'s existing sigma/
//! toxicity widen formula, which already does this for that one variant)
//! instead of a global multiplier. Same failure, slightly worse (-29.8% vs
//! -8.2% baseline, same arbitrage/mev explosion, volume share 35.1% ->
//! 27.2%). Root cause, consistent across both: this bot is in active quote
//! competition with faster/more opportunistic bot types on the very same
//! book. Any wider spread it posts is a bigger target those bots step
//! inside of and capture themselves, not a premium this bot gets paid --
//! `market_maker` ends up quoting less competitively AND still eating
//! whatever adverse-selection/squeeze risk the residual flow brings, a
//! strictly worse position than before. **Pricing-side fixes (charge more)
//! are a dead end here; the direction that worked (depth-collapse) is
//! participation-side (quote less under detectable stress) instead.**
//!
//! **Single order per side, deleted the ladder (2026-10-20)**: direct user
//! request ("only use one ladder/delete this logic"), after the order-event
//! cost investigation below found `market_maker`'s 5-level ladder (added
//! 2026-10-14, see that entry above) was overwhelmingly the dominant source
//! of this sim's order-event/memory/CPU cost -- isolated alone, it generated
//! ~75% of all order-event volume in a representative composition, at 8.76
//! events/agent/sec vs. 0.01-3.65 for every other bot type, simply from
//! cancelling and replacing all 5 levels, both sides, every 0.3-0.5s wake.
//! Rather than just default `n_ladder_levels` to 1 (leaving the Vec/resize/
//! drain machinery, the wait-and-reinforce mechanism, and the empirical-
//! kappa near/far fill-rate calibration in place as dead weight -- the
//! empirical-kappa gate in particular is already unconditionally false at
//! 1 level, since `far_distance_frac` collapses to equal `near_distance_frac`
//! with no second level to measure), all of that was deleted: `bid_levels`/
//! `ask_levels: Vec<LadderLevel>` are now plain `bid_level`/`ask_level:
//! LadderLevel` fields, `n_ladder_levels`/`ladder_level_spacing_frac` are
//! gone from `State`, and the empirical-kappa fields are gone from both
//! `State` and `AssetState`.
//!
//! Before deleting it, re-ran the 2026-10-14 ladder result properly --
//! multi-seed, multi-composition, the discipline this file's other entries
//! already insist on (that original result was a single seed). It didn't
//! hold up: single-order quoting (this change) *outperforms* the 5-level
//! ladder on every one of 7 tested seed/composition combinations (human_80pct
//! seeds 42/7/999, human_40pct seeds 42/7/123), often by a large margin
//! (market_maker's own equity growth, e.g. seed 42: +2.1% laddered -> +41.5%
//! single-order). The 2026-10-14 entry's single-seed result was itself
//! superseded by everything else this file has absorbed since (microprice,
//! depth-collapse, the `max_fundamental_vol` asset-eligibility fix) --
//! re-tested properly, the ladder no longer helps and the extra order flow
//! it generates is pure cost.
//!
//! `wake_interval`'s default also moved from a per-variant mix (0.3s for
//! `SymmetricAvellanedaStoikov`, 0.5s for the other three) to one shared
//! 2.0s default for every variant -- also multi-seed/multi-composition
//! validated (the mixed cadence was *not* equivalent: unifying to a single
//! wake interval, even just 0.5s uniformly, flipped a seed/composition that
//! was net-negative under the old mixed cadence to strongly positive; 2.0s
//! tested equally robust across all 7 combinations above and cuts order-
//! event volume further, since it's the slower of the two full validated
//! candidates). Agents still desynchronize from each other -- see
//! `step()`'s own per-wake `+/- 5%` jitter and the initial wake's uniform
//! `[0,1)` (or `graft_population`'s 60s) stagger -- this only changed the
//! shared target frequency, not whether agents move in lockstep.
//!
//! Net effect measured directly on the worst-case (algo-bot-heavy)
//! composition: market_maker alone, same scaled population, 200s window --
//! 1,020,344 order events at 5 ladder levels down to 119,908 at 1 level +
//! 2.0s wake (-88%), while simultaneously being the single biggest driver of
//! `market_maker`'s own equity turning net-positive across this whole
//! session's investigation.
//!
//! **Pre-funded starting inventory (2026-10-14)**: direct user request,
//! same root cause as the price spikes above but structural rather than
//! behavioral -- `sim.rs`'s `build_population` previously gave every bot
//! type except `noise`/`human` zero starting inventory, which (per that
//! same function's own older doc comment, originally written for `noise`/
//! `human`) makes a bot categorically buy-only until it organically
//! accumulates something to sell, biasing price sharply upward right when
//! a population is freshly built or grafted in. market_maker is now
//! included in that same pre-funded-inventory list (along with grid,
//! momentum, mean_reversion, arbitrage, mev -- every bot type that posts
//! ordinary spot `Side::Sell` orders as part of its own strategy, not
//! through the separate margin/leverage mechanism). See `build_population`
//! in `sim.rs` for the mechanism itself (not duplicated here, since it's
//! applied before this bot's own `State` is ever constructed).
//!
//! **Depth-aware sizing cap (2026-10-20)**: direct user request ("optimize
//! market_maker further ... especially high human percentages"), following
//! the liquidation-cascade fix elsewhere in this session -- with that bug
//! gone, the full 33-cell/200,000s composition sweep still showed
//! `market_maker` consistently losing in `human_50pct` (3/3 seeds negative)
//! and `human_90pct` (2/3 seeds negative), via real (non-bug) bankruptcy
//! concentration events, not gradual erosion. Root cause: every existing
//! sizing mechanism (fix 1's equity-scaled `max_inventory`, predictive/
//! markout/drift shrinks) scales this agent's own risk budget off its own
//! equity or its own recent experience -- never off how much the book it's
//! actually quoting into can absorb. High-human/thin-population
//! compositions have far fewer other market_maker/arbitrage/grid agents
//! resting orders alongside this one, so a budget that's a perfectly
//! reasonable dollar amount can still dwarf the real book: this agent's own
//! quotes get swept disproportionately hard (bigger than what a thin book
//! "expected"), and any de-risking it then needs to do pushes price further
//! in a book that was already thin. `depth_size_cap_frac` caps
//! `max_inventory` to this fraction of the thinner side's
//! `bid_depth_baseline`/`ask_depth_baseline` (idea #5's existing slow EMA of
//! near-touch depth, reused here as a sizing ceiling, not just a panic
//! trigger). A/B tested multi-seed on 5 compositions (`human_0pct/40pct/
//! 50pct/60pct/90pct`, 3 seeds each, 100,000s/200-leg harness): at 0.5, 14 of
//! 15 cells improved or stayed net-positive -- including two (`human_0pct`
//! seed 42, `human_90pct` seed 7) that had real bankruptcy events (48-75% of
//! the population) baseline had been masking as "already fine" at the
//! original 3-composition validation scope. `human_60pct` is the one
//! exception: already strongly positive at baseline (+34% avg), the cap
//! trims that to +20% avg -- still solidly positive, zero bankruptcy either
//! way, just a smaller margin. Net effect clear enough to promote straight
//! to the real default (0.5), not leave opt-in at 0.0 the way several
//! single-composition ideas above were left pending wider validation.

use rand::Rng;

use super::{RollingWindow, StepCtx};
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::market::Market;
use crate::params;
use crate::types::{AssetId, OrderId, OrderType, Side, SimTime, TimeInForce};

/// Builds a `RollingWindow` of the given capacity, pre-filled from `history`
/// (oldest first) instead of starting empty -- see module doc comment's
/// "Seeding new agents from market history". Keeps only the most recent
/// `capacity` entries if `history` holds more than that.
fn seeded_window(capacity: usize, history: &std::collections::VecDeque<f64>) -> RollingWindow {
    let mut w = RollingWindow::new(capacity);
    let skip = history.len().saturating_sub(capacity);
    for &v in history.iter().skip(skip) {
        w.push(v);
    }
    w
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    SymmetricAvellanedaStoikov,
    FixedOffset,
    InventorySkew,
    AdaptiveSpread,
}

/// One ladder level's currently-resting order, if any, plus the qty it was
/// placed with -- the qty is needed *after* the order may have partially
/// filled, to tell "partially eaten" apart from "freshly placed," so kept
/// alongside the id rather than re-derived from the book.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct LadderLevel {
    pub order_id: Option<OrderId>,
    pub qty_placed: f64,
    /// The price this level was actually resting at -- needed for markout
    /// tracking (module doc comment): a post-only limit fill always happens
    /// at the order's own quoted price, so this doubles as "the fill price"
    /// for any quantity that got (partially or fully) consumed this wake.
    pub price_placed: f64,
}

/// Per-asset rolling state and resting-order bookkeeping -- one of these per
/// asset this agent manages (see module doc comment's "Multi-asset risk
/// allocation"). Everything that's a *shared policy* across every asset this
/// agent trades (spread/inventory/de-risking parameters, variant, etc.)
/// stays on `State` itself, not duplicated here.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssetState {
    pub asset_id: AssetId,
    pub vol_window: RollingWindow,
    pub toxicity_window: RollingWindow,
    pub momentum_window: RollingWindow,
    pub ema_mid: Option<f64>,
    pub last_mid: Option<f64>,
    /// This agent's one resting order each side (2026-10-20, direct user
    /// request: deleted the ladder -- see module doc comment's "Single
    /// order per side, deleted the ladder" entry for why multiple levels
    /// were removed rather than just defaulted to one).
    pub bid_level: LadderLevel,
    pub ask_level: LadderLevel,
    /// Fills awaiting markout resolution (module doc comment): `(fill_time,
    /// side, fill_price)`, queued in time order so resolution can just pop
    /// from the front until the next entry isn't old enough yet.
    pub pending_markouts: std::collections::VecDeque<(SimTime, Side, f64)>,
    /// Resolved markout observations, as a signed fraction of each fill's own
    /// price (positive = this fill aged well, negative = picked off) --
    /// fractional so it's comparable across assets at very different price
    /// scales, same reasoning as every other per-asset ratio in this file.
    pub markout_window: RollingWindow,
    /// Slow EMA baseline of total resting depth near the touch, each side
    /// (module doc comment's "Depth-collapse panic trigger", idea #5).
    /// Starts at 0.0 (untrusted, see `step()`'s guard) rather than a guessed
    /// prior -- the first real reading sets it.
    pub bid_depth_baseline: f64,
    pub ask_depth_baseline: f64,
    /// Consecutive wakes this asset has read `depth_collapsed` (module doc
    /// comment's "Bounded re-anchor") -- resets to 0 the moment depth is no
    /// longer collapsed.
    pub depth_collapsed_wakes: u32,
    /// Slow EMA of this asset's own `inventory_norm` (module doc comment's
    /// "Sustained-drift extra de-risking") -- starts at 0.0 (freshly
    /// constructed agent has no position yet either way).
    pub inventory_drift_ema: f64,
}

impl AssetState {
    /// **Seeding new agents from market history (2026-10-14, direct user
    /// request)**: `vol_window`/`toxicity_window`/`momentum_window` start
    /// pre-filled from `market.recent_log_returns` (see that field's doc
    /// comment) instead of empty, and `ema_mid`/`last_mid` start at the
    /// market's actual current price instead of `None` -- a newly
    /// constructed agent (at initial build, where there's no history yet
    /// and this is a no-op, or grafted into an already-running market,
    /// where there very much is) gets to see what's already been
    /// happening, rather than quoting its first several wakes with zero
    /// sense of current volatility/toxicity/momentum regardless of how
    /// "hot" the market already is.
    fn new(asset_id: AssetId, vol_window_len: usize, momentum_window_len: usize, markout_window_len: usize, market: &Market) -> Self {
        let empty = std::collections::VecDeque::new();
        let history = market.recent_log_returns.get(&asset_id).unwrap_or(&empty);
        let toxicity_history: std::collections::VecDeque<f64> = history.iter().map(|v| v.abs()).collect();
        let current_price = market.mark_price_ema.get(&asset_id).copied();
        Self {
            asset_id,
            vol_window: seeded_window(vol_window_len, history),
            toxicity_window: seeded_window(vol_window_len, &toxicity_history),
            momentum_window: seeded_window(momentum_window_len, history),
            ema_mid: current_price,
            last_mid: current_price,
            bid_level: LadderLevel::default(),
            ask_level: LadderLevel::default(),
            // No market history to seed from (unlike vol/toxicity/momentum
            // above) -- markout is about this *agent's own* fills, which
            // don't exist yet for a freshly constructed one either way.
            pending_markouts: std::collections::VecDeque::new(),
            markout_window: RollingWindow::new(markout_window_len),
            bid_depth_baseline: 0.0,
            ask_depth_baseline: 0.0,
            depth_collapsed_wakes: 0,
            inventory_drift_ema: 0.0,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub variant: Variant,
    /// One entry per asset this agent manages -- see module doc comment's
    /// "Multi-asset risk allocation". Every wake, `step()` evaluates all of
    /// them before deciding how to split this agent's total risk budget.
    pub assets: Vec<AssetState>,
    pub wake_interval: SimTime,
    pub base_half_spread_frac: f64,
    pub gamma: f64,
    /// Order-arrival-decay rate used directly in the AS spread formula --
    /// a fixed textbook default, the same for every asset.
    pub k: f64,
    pub horizon_secs: f64,
    /// Scale on the cross-asset hedge skew (module doc comment's
    /// "Cross-asset hedging") -- 0.0 = no effect (today's behavior, every
    /// asset priced independently of every other one this agent manages).
    pub cross_asset_hedge_scale: f64,
    /// Minimum `|hedge_signal|` before the skew applies at all (2026-10-17
    /// rework) -- the original always-on version applied even a small,
    /// noisy correlation reading every wake, which chronically skewed
    /// quoting without the position actually being a problem yet.
    pub cross_asset_hedge_threshold: f64,
    /// Depth-collapse panic trigger (module doc comment, idea #5): panic
    /// withdraw on an asset whose current near-touch depth, either side, has
    /// fallen below this fraction of its own slow EMA baseline -- 0.0 (ratio
    /// can never go negative) safely means never, same disabled-sentinel
    /// convention as `panic_vol_frac`'s `f64::MAX`, just inverted since this
    /// is a ratio, not an absolute level.
    pub depth_collapse_frac: f64,
    /// Consecutive wakes an asset may stay `depth_collapsed` before this
    /// agent force-reanchors `bid_depth_baseline`/`ask_depth_baseline` to
    /// the current (still thin) reading and resumes quoting -- see the
    /// "Bounded re-anchor" comment at this field's one call site for why a
    /// permanent trigger with no escape caused a real, measured permanent
    /// market freeze on a liquidity-floor-less (no `noise`/`human`)
    /// composition. Default 50 wakes (~100s at the 2.0s default
    /// `wake_interval`, far short of one 500s leg) -- short enough that a
    /// composition with any independent order flow recovers the old way
    /// (depth naturally returns before the counter trips) and this never
    /// engages, long enough that a momentary panic-withdrawal still behaves
    /// exactly as before.
    pub depth_collapse_max_wakes: f64,
    /// EMA window (in wakes) for `AssetState::bid_depth_baseline`/
    /// `ask_depth_baseline`.
    pub depth_window: f64,
    /// How many book levels' resting qty to sum for the depth reading.
    pub depth_levels: u32,
    /// Caps `max_inventory` (asset units) to at most this fraction of the
    /// *smaller* of `bid_depth_baseline`/`ask_depth_baseline` -- i.e. how much
    /// typically rests near the touch on the thinner side, not just this
    /// agent's own equity-scaled risk budget. 0.0 would be a no-op (old
    /// behavior); see module doc comment's "Depth-aware sizing cap" for why
    /// the real default is 0.5. Reuses `bid_depth_baseline`/
    /// `ask_depth_baseline` (idea #5's existing slow EMA of near-touch depth)
    /// as the ceiling instead of only a panic trigger -- no new signal, just
    /// a second use of one already measured and already trusted.
    pub depth_size_cap_frac: f64,
    /// Exponent on sigma before inverting for risk-parity weighting (module
    /// doc comment's "Steeper risk-parity exponent") -- 1.0 reproduces the
    /// original linear inverse-vol weighting; higher shrinks a currently-
    /// volatile asset's budget faster than linearly.
    pub risk_aversion_exponent: f64,
    /// Sustained-drift extra de-risking (module doc comment, same loss-
    /// timing investigation): a *slow* EMA of this asset's own
    /// `inventory_norm` (see `AssetState::inventory_drift_ema`), separate
    /// from the instantaneous underwater check above -- catches "this
    /// agent's whole fleet has been one-sided on this asset for many legs,"
    /// which the instantaneous check only reacts to once *this specific*
    /// move has already turned against it. Threshold before it engages at
    /// all; scale on the extra risk-budget shrink past that threshold.
    pub drift_window: f64,
    pub drift_threshold: f64,
    pub drift_size_scale: f64,
    /// Fraction of this agent's *current total* equity (across every asset,
    /// recomputed every wake, see module doc comment's fix 1) its combined
    /// inventory cap is sized to -- not a fixed dollar amount from
    /// construction, so the risk budget shrinks as losses do. Split across
    /// assets by inverse realized volatility, not applied per-asset
    /// independently (see module doc comment's "Multi-asset risk
    /// allocation").
    pub max_inventory_frac_of_equity: f64,
    /// `max_inventory`'s own dollar floor, regardless of how depleted equity
    /// gets -- keeps a token quoting presence rather than going fully dark
    /// well before the agent is actually bankrupt. Applied per-asset.
    pub min_inventory_notional: f64,
    /// Each quote's size as a fraction of the (equity-scaled) `max_inventory`
    /// -- default 0.1 preserves the original fixed 3000/30000 ratio, just
    /// applied to a now-dynamic base.
    pub order_frac_of_max_inventory: f64,
    pub skew_factor: f64,
    /// Shifts `reservation` in the direction of detected short-horizon
    /// momentum (0.0 = no effect, matches old behavior) -- leads the price
    /// instead of trailing it. Left at 0.0 by default: A/B tested and found
    /// to make this agent procyclical and hurt the wider market -- see
    /// module doc comment's "Predictive momentum signal".
    pub predictive_skew_scale: f64,
    /// Extra multiplicative spread widening scaled by |momentum signal|
    /// (0.0 = no effect) -- on top of the existing sigma/toxicity stress
    /// widening, since a one-directional push is specifically more likely
    /// informed/toxic than two-way noise of the same magnitude.
    pub predictive_widen_scale: f64,
    /// Shrinks `max_inventory` itself when momentum is detected (0.0 = no
    /// effect), clamped so it can shrink by at most 90% -- proactive
    /// exposure reduction *before* a position builds up, unlike the
    /// existing active de-risking (further below), which only acts after
    /// the agent is already underwater on a position it already holds.
    pub predictive_size_scale: f64,
    /// Cancel this asset's resting bid/ask (both sides) and skip quoting it
    /// entirely this wake, when that asset's own realized volatility
    /// exceeds this fraction of mid. Generalizes `AdaptiveSpread`'s
    /// existing single-order `withdrawal_vol_frac` to every variant and to
    /// every asset independently. Default disabled (effectively
    /// unreachable) so existing scenarios are unaffected unless explicitly
    /// opted in.
    pub panic_vol_frac: f64,
    /// Same panic withdrawal, triggered instead by that asset's own
    /// `momentum_signal` magnitude (direction-agnostic here -- a sustained
    /// push either way is the trigger) -- catches a *developing*
    /// one-directional move, which `panic_vol_frac` (a std, confirms a move
    /// already happened) reacts to more slowly. Default disabled.
    pub panic_momentum_frac: f64,
    pub withdrawal_vol_frac: f64,
    /// Shared spread-widening reaction to realized volatility/toxicity,
    /// applied to every variant except `AdaptiveSpread` (which already has
    /// its own, more aggressive, dedicated version) -- see module doc
    /// comment's fix 3.
    pub stress_vol_scale: f64,
    pub stress_toxicity_scale: f64,
    /// De-risking's unwind-rate multiplier per unit of drawdown beyond
    /// `ema_mid` -- see module doc comment's fix 4.
    pub drawdown_derisk_scale: f64,
    /// Ceiling on that multiplier, so a severe move still can't dump the
    /// whole excess position in a single step.
    pub drawdown_derisk_cap: f64,
    /// Active de-risking threshold, as a fraction of `max_inventory` -- see
    /// the module doc comment. Below this, inventory is managed only by
    /// requoting (the original behavior, unchanged); above it (AND only
    /// while underwater, see `ema_mid`), this agent starts actively crossing
    /// the spread to reduce the position. Drawn once per agent at
    /// construction from a range, not a single shared constant -- agents
    /// facing the same market conditions must not all cross the identical
    /// threshold at the same moment, which was measured to cause a
    /// self-inflicted stampede (see module doc comment).
    pub derisk_inventory_frac: f64,
    /// **Price-push de-risking (2026-10-14, replaces aggressive IOC
    /// orders)**: direct user request, after the previous de-risking
    /// mechanism (an aggressive IOC market order to flatten part of the
    /// excess, see this struct's and the module's own history) was found
    /// to be a *cause* of runaway price moves, not just a response to one
    /// -- confirmed directly in `staged_introduction_test`: market_maker's
    /// own de-risking buy-backs were, by trade count, the dominant
    /// aggressor pushing XRP/DOGE up hundreds of percent in under 500s,
    /// each round of buying triggering more de-risking buying in a
    /// self-inflicted stampede (117 agents all hit by the same entry shock
    /// at once, so even per-agent randomized thresholds -- see
    /// `derisk_inventory_frac` above -- don't desynchronize them). Instead
    /// of crossing the spread to flatten, this agent now only pushes the
    /// price of the side that would *worsen* its position further away
    /// (raises the ask when short, lowers the bid when long -- see
    /// `step()`'s `derisk_push_frac`), stopping the bleeding without
    /// itself being the trade that moves the market. The position can
    /// still unwind, just passively (a fill on the *other*, un-pushed
    /// side), or simply stop getting worse until the move itself pauses.
    /// This field is the scale on that push, applied on top of the same
    /// drawdown-severity multiplier (`drawdown_derisk_scale`/
    /// `drawdown_derisk_cap`) the old mechanism used -- a position deeply
    /// underwater pushes its bad side's price much further than one just
    /// over the threshold.
    pub derisk_price_push_scale: f64,
    /// Lookback before a fill's own markout (did price move against this
    /// specific trade afterward?) gets resolved into
    /// `AssetState::markout_window` -- see module doc comment's markout
    /// feedback. Long enough to capture real post-trade drift, short enough
    /// to feed back within this bot's own risk horizon.
    pub markout_horizon_secs: f64,
    /// Extra spread-widening multiplier per unit of negative realized
    /// markout (0 = no effect; markout only ever penalizes, see module doc
    /// comment -- a positive running average doesn't tighten anything).
    pub markout_widen_scale: f64,
    /// Same signal, shrinking `max_inventory` instead -- trade less on an
    /// asset this agent is structurally getting picked off on, not just
    /// price itself wider. Capped like `predictive_size_scale`.
    pub markout_size_scale: f64,
    /// **Fundamental-value reservation anchor (2026-10-27, direct user
    /// request: "market_maker should be one of the best bots at realistic
    /// [50-80%] human ratios")**: every variant's reservation price was
    /// anchored purely on `micro`/`mid` -- *market-observed* signals, with
    /// no pull toward the asset's true (simulation-known) fundamental
    /// value, unlike `grid.rs`'s and edge `mean_reversion`'s own
    /// `fundamental_bias_frac`, which measurably cut `grid`'s bankruptcy by
    /// leaning its quotes back toward fair value instead of chasing wherever
    /// the noisy traded price currently sits. `center` (below) blends
    /// `micro` toward `fair` by this fraction before any variant computes
    /// its reservation from it -- a market_maker whose quotes are pulled
    /// toward fair value doesn't accumulate as large a one-sided position
    /// during a sustained move away from it (the same "loss-timing
    /// investigation" pattern this file's history keeps tracing: the whole
    /// fleet drifting the same way on a price swing away from fundamentals),
    /// since its own bid/ask center lags the move rather than tracking it
    /// tick-for-tick. Distinct from `predictive_skew_scale` (measured
    /// net-harmful, see that field's doc comment): that one *leads* recent
    /// momentum (procyclical, leans into a move); this one pulls *back*
    /// toward a value that doesn't move with short-term noise at all
    /// (mean-reverting by construction, same direction as the de-risking/
    /// taper machinery already in this file, not opposed to it). Defaults
    /// 0.0 (old behavior, pure micro anchor) -- opt-in per scenario, same
    /// convention as every other experimental knob in this struct.
    pub fundamental_bias_frac: f64,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::SymmetricAvellanedaStoikov => "avellaneda_stoikov",
            Variant::FixedOffset => "fixed_offset",
            Variant::InventorySkew => "inventory_skew",
            Variant::AdaptiveSpread => "adaptive_spread",
        }
    }

    /// `rng`: used once, here, to draw this specific agent's own
    /// `derisk_inventory_frac` (see that field's doc comment) from a range --
    /// the same construction-time RNG `sim.rs`'s `build_population` already
    /// uses for this agent's starting cash/leverage, so the draw is
    /// deterministic given a fixed run seed.
    ///
    /// `asset_ids`: every asset this agent will manage -- `sim.rs` now passes
    /// *all* of the scenario's configured assets to every market_maker
    /// agent (see module doc comment's "Multi-asset risk allocation"), not
    /// one fixed asset per agent as before.
    ///
    /// No `initial_price`/notional-to-asset-unit conversion at construction
    /// any more (2026-10-10, module doc comment's fix 1) -- `max_inventory`/
    /// `order_qty` are now recomputed every wake in `step()` from *current*
    /// equity and the *current* mid price, not fixed at construction time.
    pub fn from_params(variant: Variant, asset_ids: Vec<AssetId>, p: &toml::Value, rng: &mut impl Rng, market: &Market) -> Self {
        let vol_window_len = params::get_u32(p, "vol_window", 30) as usize;
        let momentum_window_len = params::get_u32(p, "momentum_window", 6) as usize;
        let markout_window_len = params::get_u32(p, "markout_window", 60) as usize;
        let derisk_frac_min = params::get_f64(p, "derisk_inventory_frac_min", 0.6);
        let derisk_frac_max = params::get_f64(p, "derisk_inventory_frac_max", 0.9);
        Self {
            variant,
            assets: asset_ids
                .into_iter()
                .map(|aid| AssetState::new(aid, vol_window_len, momentum_window_len, markout_window_len, market))
                .collect(),
            // 2.0s default (2026-10-20, direct user request): multi-seed/
            // multi-composition validated as the best refresh cadence for
            // single-order quoting (see module doc comment's "Single order
            // per side, deleted the ladder") -- the old per-variant mix
            // (AS at 0.3s, everyone else at 0.5s) is gone; every variant now
            // shares this one default unless a scenario overrides it.
            wake_interval: params::get_f64(p, "wake_interval_secs", 2.0),
            base_half_spread_frac: params::get_f64(p, "base_half_spread_frac", 0.0008),
            gamma: params::get_f64(p, "gamma", 0.1),
            k: params::get_f64(p, "k", 1.5),
            horizon_secs: params::get_f64(p, "horizon_secs", 60.0),
            // 0.5/0.1 preserve the original fixed 30,000/3,000 ratio (max
            // inventory = 10x one order), just scaled off current equity
            // instead of a constant -- a market_maker with ~$60k equity gets
            // roughly the same $30k cap as before; one that's already lost
            // half its capital gets roughly half the cap, not the same one.
            max_inventory_frac_of_equity: params::get_f64(p, "max_inventory_frac_of_equity", 0.5),
            min_inventory_notional: params::get_f64(p, "min_inventory_notional", 1_000.0),
            order_frac_of_max_inventory: params::get_f64(p, "order_frac_of_max_inventory", 0.1),
            skew_factor: params::get_f64(p, "skew_factor", 0.0005),
            // All three default to 0.0 (no effect) -- A/B tested and found
            // net-harmful to the wider market at nonzero skew specifically,
            // see module doc comment's "Predictive momentum signal".
            predictive_skew_scale: params::get_f64(p, "predictive_skew_scale", 0.0),
            predictive_widen_scale: params::get_f64(p, "predictive_widen_scale", 0.0),
            predictive_size_scale: params::get_f64(p, "predictive_size_scale", 0.0),
            // f64::MAX, not e.g. 1.0 -- sigma/momentum_signal are small
            // fractional-per-wake quantities (see their own doc comments),
            // so even a generous-looking disabled sentinel like 1.0 could in
            // principle still be reachable in an extreme/thin-sample case.
            // f64::MAX is genuinely unreachable.
            panic_vol_frac: params::get_f64(p, "panic_vol_frac", f64::MAX),
            panic_momentum_frac: params::get_f64(p, "panic_momentum_frac", f64::MAX),
            withdrawal_vol_frac: params::get_f64(p, "withdrawal_vol_frac", 0.01),
            derisk_inventory_frac: if derisk_frac_max > derisk_frac_min { rng.gen_range(derisk_frac_min..derisk_frac_max) } else { derisk_frac_min },
            derisk_price_push_scale: params::get_f64(p, "derisk_price_push_scale", 2.0),
            // Scaled to land in the same ballpark as AdaptiveSpread's own
            // dedicated 50.0/100.0 widening, but more modest (this applies
            // to variants that otherwise have zero stress reaction at all,
            // not as their primary defense).
            stress_vol_scale: params::get_f64(p, "stress_vol_scale", 15.0),
            stress_toxicity_scale: params::get_f64(p, "stress_toxicity_scale", 30.0),
            drawdown_derisk_scale: params::get_f64(p, "drawdown_derisk_scale", 20.0),
            drawdown_derisk_cap: params::get_f64(p, "drawdown_derisk_cap", 5.0),
            markout_horizon_secs: params::get_f64(p, "markout_horizon_secs", 20.0),
            // Jittered per agent (same rng stream as derisk_inventory_frac above,
            // same reasoning): measured A/B on human_80pct without this, every
            // market_maker agent shares the same market, so their fills' markouts
            // are highly correlated -- a uniform shared scale means the whole
            // fleet widens/shrinks in near-lockstep whenever the shared market
            // condition turns against them, which is exactly the self-inflicted
            // liquidity-withdrawal stampede this module's history keeps running
            // into from other mechanisms. Measured: overall market growth over 40
            // legs went from +17.9% (baseline, no markout feedback) to -11.4%
            // (uniform scale) to +9.5%-ish range once jittered -- trade count and
            // bankruptcy rate moved the same direction. Jittering the *response*
            // scale (not the underlying signal, which is correctly per-agent
            // already) desynchronizes magnitude/timing without losing the signal.
            // Defaults 0.0 (no effect) for now -- see the long comment above:
            // measured net-harmful on human_80pct even after jittering
            // (overall market growth +17.9% baseline -> -1.9%, trade count
            // -30%), the same "individually-reasonable reaction, correlated
            // across the whole fleet, hurts the wider market" pattern as
            // `predictive_skew_scale` elsewhere in this file. Left in,
            // disabled, for future tuning (e.g. splitting markout by side
            // instead of combining buy/sell into one rolling mean, or a
            // longer/smoother window) rather than ripped out.
            markout_widen_scale: params::get_f64(p, "markout_widen_scale", 0.0) * rng.gen_range(0.5..1.5),
            markout_size_scale: params::get_f64(p, "markout_size_scale", 0.0) * rng.gen_range(0.5..1.5),
            fundamental_bias_frac: params::get_f64(p, "fundamental_bias_frac", 0.0),
            // Defaults 0.0 -- measured net-harmful in isolation on
            // human_80pct: trade count -62% (342k -> 130k), market_maker's
            // own volume share collapsed 32.6% -> 9.9%, equity *unchanged*
            // (-83.1% -> -84.4%, no better). Unlike the per-asset de-risking
            // mechanisms, which only activate when that specific asset is
            // actually underwater, this skew applies *every* wake regardless
            // of magnitude -- with several correlated assets, the signal
            // rarely nets to zero, so it acts as a near-constant skew rather
            // than an occasional correction, chronically making one side
            // unattractive and suppressing fills. Left in, disabled, for
            // future rework (e.g. gating it on the hedge signal's own
            // magnitude, not applying it unconditionally).
            cross_asset_hedge_scale: params::get_f64(p, "cross_asset_hedge_scale", 0.0),
            cross_asset_hedge_threshold: params::get_f64(p, "cross_asset_hedge_threshold", 0.4),
            depth_collapse_frac: params::get_f64(p, "depth_collapse_frac", 0.35),
            depth_collapse_max_wakes: params::get_f64(p, "depth_collapse_max_wakes", 50.0),
            depth_window: params::get_f64(p, "depth_window", 50.0),
            depth_levels: params::get_u32(p, "depth_levels", 3),
            depth_size_cap_frac: params::get_f64(p, "depth_size_cap_frac", 0.5),
            risk_aversion_exponent: params::get_f64(p, "risk_aversion_exponent", 1.0),
            drift_window: params::get_f64(p, "drift_window", 200.0),
            drift_threshold: params::get_f64(p, "drift_threshold", 0.3),
            drift_size_scale: params::get_f64(p, "drift_size_scale", 0.0),
        }
    }
}

/// Per-asset snapshot computed in the first pass over `s.assets`, before this
/// agent's total risk budget is known (that needs every asset's mid price
/// and realized vol first) -- see `step()`'s two-pass structure.
struct AssetSnapshot {
    mid: f64,
    /// See module doc comment's "Microprice reservation anchor".
    micro: f64,
    /// This asset's true (simulation-known) fundamental value -- see module
    /// doc comment's "Fundamental-value reservation anchor". Falls back to
    /// `mid` when the market doesn't expose one for this asset, same
    /// fallback `grid.rs`'s own fundamental-bias fix already uses.
    fair: f64,
    sigma: f64,
    toxicity: f64,
    momentum_signal: f64,
    inventory: f64,
}

/// Microprice (depth-weighted mid): weights each side's price by the
/// *opposite* side's resting top-of-book quantity, so heavy bid size (strong
/// buying interest, thin offer) pulls the anchor toward the ask --
/// anticipating the move that imbalance usually precedes -- and vice versa.
/// See module doc comment. Falls back to `mid` whenever either side of the
/// book is empty (nothing to weight against -- includes this agent's own
/// just-placed-last-wake resting orders, so it's never reading a truly empty
/// book while this agent itself is quoting).
fn microprice(ctx: &StepCtx, asset_id: AssetId, mid: f64) -> f64 {
    let bid = ctx.market.depth(ctx.venue, asset_id, Side::Buy, 1);
    let ask = ctx.market.depth(ctx.venue, asset_id, Side::Sell, 1);
    match (bid.first(), ask.first()) {
        (Some(&(bid_px, bid_qty)), Some(&(ask_px, ask_qty))) if bid_qty + ask_qty > 1e-9 => {
            (bid_px * ask_qty + ask_px * bid_qty) / (bid_qty + ask_qty)
        }
        _ => mid,
    }
}

/// Pearson correlation between two assets' recent log-returns (module doc
/// comment's "Cross-asset hedging"), from `Market::recent_log_returns` --
/// every asset ticks at the same cadence (see that field's doc comment), so
/// index `i` in one deque lines up with index `i` in another; both are read
/// most-recent-first, consistently, so the pairing survives regardless.
/// Returns 0.0 (no usable signal) on too little overlapping history or zero
/// variance on either side, rather than a NaN/division-by-zero.
fn asset_correlation(market: &Market, a: AssetId, b: AssetId) -> f64 {
    let empty = std::collections::VecDeque::new();
    let ra = market.recent_log_returns.get(&a).unwrap_or(&empty);
    let rb = market.recent_log_returns.get(&b).unwrap_or(&empty);
    let n = ra.len().min(rb.len());
    if n < 5 {
        return 0.0;
    }
    let xs: Vec<f64> = ra.iter().rev().take(n).copied().collect();
    let ys: Vec<f64> = rb.iter().rev().take(n).copied().collect();
    let mean_x = xs.iter().sum::<f64>() / n as f64;
    let mean_y = ys.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..n {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= 1e-18 || var_y <= 1e-18 {
        return 0.0;
    }
    (cov / (var_x.sqrt() * var_y.sqrt())).clamp(-1.0, 1.0)
}

/// Has this asset's resting bid/ask experienced *any* fill since it was
/// placed (fully or partially) -- used for markout tracking (module doc
/// comment): even a partial fill means the price moved into this order,
/// which is enough to record a post-trade markout observation for it.
fn level_being_touched(ctx: &StepCtx, asset_id: AssetId, level: &LadderLevel) -> bool {
    if level.qty_placed <= 1e-9 {
        return false;
    }
    match level.order_id.and_then(|id| ctx.market.book(ctx.venue, asset_id).orders.get(&id)) {
        Some(order) => order.remaining() < level.qty_placed - 1e-9,
        None => true,
    }
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    let mut actions = Vec::new();
    for asset in s.assets.iter_mut() {
        // Markout tracking (module doc comment): record a pending observation
        // if this asset's resting bid/ask received any fill since it was
        // placed, before the reset below erases that information. Each
        // order lives exactly one wake (cancelled and replaced every step),
        // so this can't double-count the same fill across multiple wakes.
        if asset.bid_level.qty_placed > 1e-9 && level_being_touched(ctx, asset.asset_id, &asset.bid_level) {
            asset.pending_markouts.push_back((ctx.now, Side::Buy, asset.bid_level.price_placed));
        }
        if asset.ask_level.qty_placed > 1e-9 && level_being_touched(ctx, asset.asset_id, &asset.ask_level) {
            asset.pending_markouts.push_back((ctx.now, Side::Sell, asset.ask_level.price_placed));
        }

        if let Some(id) = asset.bid_level.order_id {
            actions.push(Action::Cancel(id, asset.asset_id));
        }
        if let Some(id) = asset.ask_level.order_id {
            actions.push(Action::Cancel(id, asset.asset_id));
        }
        asset.bid_level = LadderLevel::default();
        asset.ask_level = LadderLevel::default();
    }

    // Pass 1: update every asset's rolling windows/reference price and read
    // off this wake's per-asset signals. Needed in full before pass 2 below,
    // since this agent's *total* equity (summed inventory value across every
    // asset) and the inverse-volatility weighting both need every asset's
    // numbers first -- neither can be decided one asset at a time.
    let mut snaps: Vec<Option<AssetSnapshot>> = Vec::with_capacity(s.assets.len());
    for asset in s.assets.iter_mut() {
        let Some(mid) = ctx.market.mid_price(ctx.venue, asset.asset_id) else {
            snaps.push(None);
            continue;
        };
        if let Some(prev) = asset.last_mid {
            if prev > 0.0 {
                let log_ret = (mid / prev).ln();
                asset.vol_window.push(log_ret);
                asset.toxicity_window.push(((mid - prev) / prev).abs());
                asset.momentum_window.push(log_ret);
            }
        }
        asset.last_mid = Some(mid);

        // Slow trailing reference for the de-risking "is this position
        // actually underwater" check below -- see module doc comment. Time
        // constant tied to this agent's own `horizon_secs`, not a separate
        // tunable: a requote-interval-scaled discrete EMA, so it reacts on
        // the same timescale this agent already reasons about risk on.
        asset.ema_mid = Some(match asset.ema_mid {
            None => mid,
            Some(prev) => {
                let alpha = s.wake_interval / (s.horizon_secs + s.wake_interval).max(1e-9);
                prev + alpha * (mid - prev)
            }
        });

        // Resolve any pending markout observations whose horizon has
        // elapsed (module doc comment) -- the queue is time-ordered, so it's
        // safe to stop at the first entry that isn't ready yet.
        while let Some(&(t, _, _)) = asset.pending_markouts.front() {
            if ctx.now - t < s.markout_horizon_secs {
                break;
            }
            let (_, side, fill_price) = asset.pending_markouts.pop_front().unwrap();
            let frac = match side {
                Side::Buy => (mid - fill_price) / fill_price.max(1e-9),
                Side::Sell => (fill_price - mid) / fill_price.max(1e-9),
            };
            asset.markout_window.push(frac);
        }
        let micro = microprice(ctx, asset.asset_id, mid);
        let fair = *ctx.market.fundamental_value.get(&asset.asset_id).unwrap_or(&mid);

        let sigma = asset.vol_window.std().unwrap_or(0.0).max(1e-9);
        let toxicity = asset.toxicity_window.mean().unwrap_or(0.0);
        let momentum_signal = asset.momentum_window.mean().unwrap_or(0.0);
        let inventory = agent.inventory_of(asset.asset_id);
        snaps.push(Some(AssetSnapshot { mid, micro, fair, sigma, toxicity, momentum_signal, inventory }));
    }

    // This agent's *total* equity across every asset it holds -- replaces
    // the old single-asset `agent.cash + inventory * mid`. No margin term
    // (unlike `AgentCommon::equity`'s general version): this bot never
    // holds one.
    let total_equity = (agent.cash
        + snaps.iter().flatten().map(|snap| snap.inventory * snap.mid).sum::<f64>())
    .max(0.0);
    let total_risk_budget_notional = (total_equity * s.max_inventory_frac_of_equity).max(s.min_inventory_notional);

    // Inverse-volatility (risk-parity) weighting across every asset with a
    // live price this wake -- see module doc comment's "Multi-asset risk
    // allocation". A calmer asset gets a larger slice of the shared budget,
    // a currently-more-volatile one a smaller slice. Degrades gracefully to
    // equal weighting when every live asset's sigma sits at the same floor
    // (e.g. this agent's first wake, before any window has filled).
    // `risk_aversion_exponent` (module doc comment's "Steeper risk-parity
    // exponent", idea from the loss-timing investigation): raising sigma to
    // a power > 1 before inverting makes the budget shrink *faster* than
    // linearly for a currently-volatile asset -- 1.0 reproduces the original
    // linear weighting exactly.
    let inv_vol: Vec<f64> = snaps.iter().map(|snap| snap.as_ref().map_or(0.0, |snap| 1.0 / snap.sigma.powf(s.risk_aversion_exponent))).collect();
    let inv_vol_sum: f64 = inv_vol.iter().sum();

    let next_wake = ctx.now + s.wake_interval + agent.rng.gen_range(-0.05..0.05) * s.wake_interval;
    let n_assets_managed = s.assets.len().max(1);

    // Cross-asset hedging pre-pass (module doc comment's "Cross-asset
    // hedging"): every asset's own `inventory_norm`, computed once here
    // (duplicating the formula the main loop below computes for *its own*
    // asset -- this is the one place this agent needs to know every *other*
    // asset's position too, before quoting any of them) so the main loop
    // can read a correlated asset's position without needing `&mut` access
    // to the whole `s.assets` collection it's already iterating mutably.
    // `asset_ids` is the same, for the same reason (`other.asset_id` would
    // otherwise need a second borrow of `s.assets` inside the main loop).
    let asset_ids: Vec<AssetId> = s.assets.iter().map(|a| a.asset_id).collect();
    let inventory_norms: Vec<f64> = s
        .assets
        .iter()
        .enumerate()
        .map(|(i, asset)| {
            let Some(snap) = &snaps[i] else { return 0.0 };
            let weight = if inv_vol_sum > 0.0 { inv_vol[i] / inv_vol_sum } else { 1.0 / n_assets_managed as f64 };
            let max_inventory_notional = (total_risk_budget_notional * weight).max(s.min_inventory_notional);
            let mut max_inventory = max_inventory_notional / snap.mid.max(1e-9);
            max_inventory *= 1.0 - (s.predictive_size_scale * snap.momentum_signal.abs()).min(0.9);
            let markout_mean = asset.markout_window.mean().unwrap_or(0.0);
            max_inventory *= 1.0 - ((-markout_mean).max(0.0) * s.markout_size_scale).min(0.9);
            snap.inventory / max_inventory.max(1e-9)
        })
        .collect();

    for (i, asset) in s.assets.iter_mut().enumerate() {
        let Some(snap) = &snaps[i] else { continue };
        let mid = snap.mid;
        let micro = snap.micro;
        // Fundamental-value reservation anchor (module doc comment,
        // `fundamental_bias_frac`'s own doc comment): blend `micro` toward
        // `fair` before any variant below anchors its reservation on it.
        // 0.0 (default) reproduces the old pure-`micro` anchor exactly.
        let center = micro + s.fundamental_bias_frac * (snap.fair - micro);
        let sigma = snap.sigma;
        let toxicity = snap.toxicity;
        let momentum_signal = snap.momentum_signal;
        let inventory = snap.inventory;

        // Depth-collapse detection (module doc comment, idea #5): a sudden
        // drop in near-touch depth well below its own slow baseline --
        // liquidity pulling away -- is a documented pre-adverse-selection
        // signal that can arrive *before* realized price/vol has moved
        // enough to trip the checks below. Same known caveat as microprice:
        // `depth()` includes this agent's own not-yet-cancelled resting
        // orders from last wake (cancels for this wake haven't reached
        // `ctx.market` yet either way), so this partly measures "did
        // *everyone's* resting liquidity thin out," not purely external
        // flow. Checked against the baseline *before* updating it below, so
        // today's reading can't water down the very comparison it's
        // checked against.
        let bid_depth: f64 = ctx.market.depth(ctx.venue, asset.asset_id, Side::Buy, s.depth_levels as usize).iter().map(|&(_, q)| q).sum();
        let ask_depth: f64 = ctx.market.depth(ctx.venue, asset.asset_id, Side::Sell, s.depth_levels as usize).iter().map(|&(_, q)| q).sum();
        let depth_collapsed = asset.bid_depth_baseline > 1e-9
            && asset.ask_depth_baseline > 1e-9
            && (bid_depth < asset.bid_depth_baseline * s.depth_collapse_frac || ask_depth < asset.ask_depth_baseline * s.depth_collapse_frac);
        // Bounded re-anchor (2026-10-24, direct user request: "fine tune the
        // simulation so the worth of the stocks... represent the real
        // world"), found by tracing a *permanent* market freeze on a
        // 0%-human, no-noise-trader composition over a full 100,000s sweep:
        // the EMA baseline below keeps chasing the depressed depth reading
        // even while `depth_collapsed` is true, so on a book with no
        // independent liquidity floor (no `noise`/`human` group injecting
        // flow regardless of what market_maker does), every MM variant can
        // withdraw from an asset at once and never come back -- baseline and
        // live depth decay together, `depth_collapsed` stays true forever,
        // and the asset shows literally zero trades for 90+ consecutive
        // 500s legs (measured directly: Sweep XVIII validation run,
        // human_0pct, both seeds). This was validated as net-positive on
        // human_80pct (module doc comment's "Depth-collapse panic trigger"
        // entry), a composition with enough independent (human) order flow
        // that depth always recovered on its own within a few wakes -- that
        // recovery path was never actually exercised by code, just by
        // circumstance, so it silently assumed a liquidity floor this sim
        // doesn't always have. A real exchange halts and *reopens* trading
        // after a liquidity shock rather than closing it permanently; this
        // mirrors that by re-anchoring the baseline to the current (still
        // thin) reading after `depth_collapse_max_wakes` consecutive
        // collapsed wakes, which lets this agent resume quoting at the new,
        // lower-liquidity regime instead of waiting forever for depth to
        // return to a pre-shock level that may never come back on its own.
        if depth_collapsed {
            asset.depth_collapsed_wakes += 1;
        } else {
            asset.depth_collapsed_wakes = 0;
        }
        let depth_collapsed = if depth_collapsed && (asset.depth_collapsed_wakes as f64) > s.depth_collapse_max_wakes {
            asset.bid_depth_baseline = bid_depth;
            asset.ask_depth_baseline = ask_depth;
            asset.depth_collapsed_wakes = 0;
            false
        } else {
            depth_collapsed
        };
        let depth_alpha = 2.0 / (s.depth_window + 1.0);
        asset.bid_depth_baseline += depth_alpha * (bid_depth - asset.bid_depth_baseline);
        asset.ask_depth_baseline += depth_alpha * (ask_depth - asset.ask_depth_baseline);

        // Panic withdrawal (default thresholds unreachable, see their doc
        // comments) -- per asset: a spike in one market shouldn't silence
        // this agent's otherwise-calm others. Cancel actions for this
        // asset's resting orders were already queued above; this just skips
        // placing any new ones (and skips de-risking) for *this* asset.
        if sigma / mid.max(1e-9) > s.panic_vol_frac || momentum_signal.abs() > s.panic_momentum_frac || depth_collapsed {
            continue;
        }

        let weight = if inv_vol_sum > 0.0 { inv_vol[i] / inv_vol_sum } else { 1.0 / n_assets_managed as f64 };
        let max_inventory_notional = (total_risk_budget_notional * weight).max(s.min_inventory_notional);
        let max_inventory = max_inventory_notional / mid.max(1e-9);
        // Predictive proactive shrink (0.0 scale = no effect, old behavior):
        // reduce the risk budget itself *before* a position builds up,
        // rather than only reacting once already underwater on one already
        // held (see State::predictive_size_scale's doc comment). Capped at
        // a 90% shrink so this never fully zeroes max_inventory (which
        // inventory_norm divides by, below).
        let max_inventory = max_inventory * (1.0 - (s.predictive_size_scale * momentum_signal.abs()).min(0.9));
        // Markout feedback, size side (module doc comment): shrink the risk
        // budget further when this asset's own recent fills have been
        // getting picked off (negative realized markout). No effect when
        // markout is zero (no fills resolved yet) or positive.
        let markout_mean = asset.markout_window.mean().unwrap_or(0.0);
        let max_inventory = max_inventory * (1.0 - ((-markout_mean).max(0.0) * s.markout_size_scale).min(0.9));
        // Sustained-drift extra de-risking (module doc comment): shrink the
        // budget further when this asset's own *slow* position average
        // (from last wake, updated below) has been persistently one-sided
        // past `drift_threshold` -- catches "this agent's whole fleet has
        // leaned the same way on this asset for many legs" proactively,
        // unlike the instantaneous underwater check below, which only
        // reacts once the *current* move has already turned against it.
        let drift_excess = (asset.inventory_drift_ema.abs() - s.drift_threshold).max(0.0);
        let max_inventory = max_inventory * (1.0 - (drift_excess * s.drift_size_scale).min(0.9));
        // Depth-aware sizing cap (module doc comment, `depth_size_cap_frac`):
        // never let the equity-sized budget above exceed what the thinner
        // side of the book typically has resting near the touch. Baseline
        // readings start at 0.0 (untrusted, same guard `depth_collapsed`
        // uses) so a freshly constructed agent isn't capped to zero before
        // it has ever observed the book. Floored at the same
        // `min_inventory_notional` fix 1 already uses to keep a token
        // quoting presence rather than going fully dark -- `depth()`
        // includes this agent's *own* resting orders (same caveat as
        // microprice/depth-collapse above), so in a market where this
        // agent is itself most of the near-touch depth (very few other
        // resting-order bot types), shrinking its own size would shrink the
        // very baseline the cap is measured against next wake, spiralling
        // toward zero with nothing to arrest it -- caught by
        // `checkpoint_test.rs`'s tiny fixed population (3 market_maker
        // agents, few other resting-order bot types) going fully silent.
        let depth_cap = asset.bid_depth_baseline.min(asset.ask_depth_baseline);
        let min_inventory_floor = s.min_inventory_notional / mid.max(1e-9);
        let max_inventory = if s.depth_size_cap_frac > 0.0 && depth_cap > 1e-9 {
            max_inventory.min(depth_cap * s.depth_size_cap_frac).max(min_inventory_floor)
        } else {
            max_inventory
        };
        let order_qty = max_inventory * s.order_frac_of_max_inventory;
        // Shared by every variant below that reacts to inventory: raw asset units (as
        // `inventory` is) scale inversely with price via `max_inventory`'s own
        // dollar-notional derivation above -- the same risk budget is 0.5 BTC units but
        // 200,000 DOGE units. Normalizing first, to [-1, 1]-ish, keeps every variant's
        // inventory reaction a *fraction* of mid regardless of which asset it's
        // quoting, not an asset-price-dependent one.
        let inventory_norm = inventory / max_inventory.max(1e-9);
        let drift_alpha = 2.0 / (s.drift_window + 1.0);
        asset.inventory_drift_ema += drift_alpha * (inventory_norm - asset.inventory_drift_ema);

        // De-risking signal, computed here (before quoting) rather than
        // after it -- see module doc comment's "Price-push de-risking
        // (replaces aggressive IOC orders)". Gated on actually being
        // underwater (not just skewed): a long only de-risks when mid has
        // dropped below `ema_mid`, a short only when mid has risen above it
        // -- merely holding a big position that's currently *working*
        // (price moved in this agent's favor since it was built) is not a
        // reason to price itself out of adding to it.
        let underwater = if inventory > 0.0 {
            asset.ema_mid.map_or(false, |e| mid < e)
        } else {
            asset.ema_mid.map_or(false, |e| mid > e)
        };
        // How far mid has moved against this position since `ema_mid` --
        // the same reference `underwater` above already gates on. Drives
        // the drawdown-scaled push below: a position 5% underwater and one
        // 40% underwater push by very different amounts, not the same flat
        // rate, capped so a severe move still can't blow the push out to
        // something absurd (the final per-side clamp below is the real
        // backstop either way).
        let drawdown_frac = asset.ema_mid.map_or(0.0, |e| {
            if inventory > 0.0 { ((e - mid) / e.max(1e-9)).max(0.0) } else { ((mid - e) / e.max(1e-9)).max(0.0) }
        });
        let derisk_push_frac = if underwater && inventory_norm.abs() > s.derisk_inventory_frac {
            let excess_frac = inventory_norm.abs() - s.derisk_inventory_frac;
            let drawdown_multiplier = (1.0 + drawdown_frac * s.drawdown_derisk_scale).min(s.drawdown_derisk_cap);
            excess_frac * drawdown_multiplier * s.derisk_price_push_scale
        } else {
            0.0
        };

        let (reservation, half_spread) = match s.variant {
            Variant::SymmetricAvellanedaStoikov => {
                // Rolling-horizon approximation of Avellaneda-Stoikov (2008): the terminal
                // horizon T-t is replaced by a fixed rolling horizon, standard practice for
                // an always-on (non-terminating) market.
                //
                // Dimensional fix: the textbook formula's sigma is an *absolute* price
                // volatility (dS, dollars/sqrt(time)), and its inventory term uses q in
                // raw units against that same absolute scale. `sigma` here is a fractional
                // (log-return) volatility instead. Naively mixing these treats a
                // price-scale-independent quantity as if it were already in price units,
                // adding e.g. a fixed ~1.3 absolute-price spread term regardless of whether
                // the asset is worth $60,000 or $0.15 -- negligible for the former, many
                // times the entire price for the latter (this produced actual
                // sub-half-price/over-double-price quotes in testing; see git history).
                // Fixed by working entirely in *fractional* terms (`inventory_norm` above)
                // and converting to a price only once, at the end, by multiplying by mid.
                // `center` (microprice, optionally blended toward fundamental value --
                // module doc comment's `fundamental_bias_frac`), not raw `mid`, anchors
                // the reservation -- order-book imbalance leads price, so centering on
                // it instead of the last trade should reduce adverse selection on its
                // own, independent of the inventory-skew term below.
                let reservation = center * (1.0 - inventory_norm * s.gamma * sigma.powi(2) * s.horizon_secs);
                // The order-arrival-intensity term (2/gamma)*ln(1+gamma/k) is dimensionless
                // by construction (a pure function of gamma/k), not a price or even a
                // fraction of one -- rescaled so the module's own defaults land in the same
                // ballpark as fixed_offset's default 0.0008 base half-spread fraction.
                const K_TERM_SCALE: f64 = 800.0;
                let spread_frac = s.gamma * sigma.powi(2) * s.horizon_secs + (2.0 / s.gamma) * (1.0 + s.gamma / s.k).ln() / K_TERM_SCALE;
                (reservation, (spread_frac * mid / 2.0).max(mid * 1e-5))
            }
            Variant::FixedOffset => (center, mid * s.base_half_spread_frac),
            Variant::InventorySkew => {
                // Linear-skew approximation of the Gueant-Lehalle-Fernandez-Tapia inventory
                // adjustment: push the mid reference against the current inventory so quotes
                // asymmetrically encourage inventory back toward zero.
                //
                // Dimensional fix (same bug class, same fix, as Avellaneda-Stoikov above --
                // found second, while investigating why BTC stayed calm but the cheaper/
                // higher-unit-count assets didn't): this used raw `inventory` directly, so
                // `skew_factor * mid * inventory` at a given *dollar* inventory position is
                // the same absolute number regardless of asset (`mid * inventory` toward
                // `max_inventory_notional` cancels price out) -- negligible against BTC's
                // price, many multiples of DOGE's entire price at even a perfectly ordinary
                // (e.g. half-of-max) inventory position, not an extreme one. Fixed the same
                // way: react to `inventory_norm` (a fraction of max_inventory), not raw units.
                let skewed = center * (1.0 - s.skew_factor * inventory_norm);
                (skewed, mid * s.base_half_spread_frac)
            }
            Variant::AdaptiveSpread => {
                let widen = 1.0 + (sigma / mid.max(1e-9)) * 50.0 + toxicity * 100.0;
                (center, mid * s.base_half_spread_frac * widen)
            }
        };

        // Liquidity withdrawal under turbulence (adaptive-spread variant): pull quotes
        // entirely rather than just widening, once realized vol crosses the threshold.
        if s.variant == Variant::AdaptiveSpread && sigma / mid.max(1e-9) > s.withdrawal_vol_frac {
            continue;
        }

        // Shared stress widening (module doc comment's fix 3): FixedOffset and
        // InventorySkew otherwise never react to volatility/toxicity at all --
        // AdaptiveSpread keeps its own, already more aggressive, dedicated
        // version above rather than applying this on top of it. Deliberately
        // more modest than AdaptiveSpread's 50.0/100.0: this is a floor of
        // protection for variants that had none, not meant to make them behave
        // identically to the variant whose whole purpose is reacting to stress.
        let half_spread = if s.variant == Variant::AdaptiveSpread {
            half_spread
        } else {
            let stress_widen = 1.0 + (sigma / mid.max(1e-9)) * s.stress_vol_scale + toxicity * s.stress_toxicity_scale;
            half_spread * stress_widen
        };

        // Predictive skew/widen (0.0 scales = no effect, old behavior) -- applied
        // uniformly to every variant, including AdaptiveSpread, since this reacts
        // to *direction* (momentum_signal), which nothing above measures at all.
        let reservation = reservation + mid * momentum_signal * s.predictive_skew_scale;
        let half_spread = half_spread * (1.0 + s.predictive_widen_scale * momentum_signal.abs());
        // Cross-asset hedging (module doc comment): a correlation-weighted
        // sum of every *other* managed asset's own position -- being long a
        // correlated asset is directionally similar risk to being long this
        // one, so it skews this asset's reservation the same way
        // InventorySkew's own `skew_factor * inventory_norm` term skews on
        // *its own* position (subtracted, same sign convention). Entirely a
        // pricing decision on *this* asset, same as every other de-risking
        // mechanism in this file -- never a trade this agent initiates.
        let hedge_signal: f64 = if s.cross_asset_hedge_scale > 0.0 {
            asset_ids
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != i)
                .map(|(k, &other_id)| asset_correlation(ctx.market, asset.asset_id, other_id) * inventory_norms[k])
                .sum()
        } else {
            0.0
        };
        // Gated (2026-10-17 rework, same loss-timing investigation): only
        // the *excess* past `cross_asset_hedge_threshold` applies -- the
        // original always-on version measured net-harmful (chronically
        // skewed quoting even on a small/noisy signal, suppressing fills
        // without actually improving equity, see module doc comment).
        let hedge_excess = (hedge_signal.abs() - s.cross_asset_hedge_threshold).max(0.0) * hedge_signal.signum();
        let reservation = reservation - mid * hedge_excess * s.cross_asset_hedge_scale;
        // Markout feedback, spread side: same signal as the size shrink
        // above, applied as extra widening -- a direct, measured "I'm
        // getting picked off here," on top of the vol/toxicity proxies.
        let half_spread = half_spread * (1.0 + (-markout_mean).max(0.0) * s.markout_widen_scale);

        // Defensive clamps: an extreme (e.g. early-session, thin-sample) volatility or
        // toxicity estimate must never be able to produce a non-positive, absurdly wide,
        // or wildly mis-centered quote. Without this, adaptive_spread in particular could
        // occasionally emit a negative bid price. 10% half-spread (20% total) is already
        // a very wide, stressed quote for any real market maker -- a tighter belt-and-
        // braces bound than the 50% this used before the Avellaneda-Stoikov dimensional
        // fix above, which was wide enough to let a formula bug still produce a
        // barely-clamped, badly wrong quote rather than actually catching it.
        //
        // The `reservation` clamp is the same idea, closing a real gap: unlike
        // `half_spread` above, `reservation` (Avellaneda-Stoikov specifically) had no
        // bound at all until this fix. `inventory_norm * gamma * sigma^2 * horizon_secs`
        // is unbounded, so a momentary spike in this bot's own rolling realized-
        // volatility estimate (`sigma`, from recent observed returns -- not the asset's
        // configured `fundamental_vol`) could push the reservation many multiples of mid
        // away from the real market price. That produces a real, wildly mispriced quote
        // that another bot immediately trades on -- which feeds a large realized return
        // back into `sigma` on this bot's *next* wake, an actual measured runaway
        // feedback loop: a real 5,000-agent/300,000s run had DOGE's traded price jump
        // 17.6x in a single trade this way (repeatedly, on the thinner/higher-vol
        // assets specifically -- see git history). Same 10%-of-mid bound as the spread
        // above: an aggressive but not fully clamped-to-nothing skew for a stressed
        // market, not a multiple of the price.
        let reservation = reservation.clamp(mid * 0.9, mid * 1.1);
        let half_spread = half_spread.max(mid * 1e-6).min(mid * 0.1);

        // Price-push de-risking (module doc comment's "Price-push
        // de-risking (replaces aggressive IOC orders)"): push the price of
        // the side that would *worsen* this position further away -- raise
        // the ask when short, lower the bid when long -- instead of
        // crossing the spread with an aggressive order to flatten it.
        //
        // Direct user follow-up: the side that would *reduce* the position
        // isn't just left alone any more -- it's actively *sweetened* too
        // (the bid raised when short, the ask lowered when long), so this
        // agent is more competitive for exactly the flow it needs to cover
        // itself. Still entirely a pricing decision, never a trade this
        // agent initiates itself: a more attractive resting limit order
        // only gets filled if some *other* agent chooses to cross into it,
        // so this can't recreate the old mechanism's self-inflicted
        // stampede (that came specifically from this agent being the
        // taker, not from where its resting prices sat). Floored well
        // above zero (`mid * 1e-6`, the same floor `half_spread` itself
        // already uses) so an extreme push can narrow the sweetened side
        // a great deal without it ever inverting through `reservation`.
        let mut bid_half_spread = half_spread;
        let mut ask_half_spread = half_spread;
        if derisk_push_frac > 0.0 {
            let sweeten = (1.0 - derisk_push_frac).max(0.05);
            if inventory > 0.0 {
                // Long & underwater: discourage buying more, sweeten selling.
                bid_half_spread = (half_spread * (1.0 + derisk_push_frac)).min(mid * 0.1);
                ask_half_spread = (half_spread * sweeten).max(mid * 1e-6);
            } else {
                // Short & underwater: discourage selling more, sweeten buying.
                ask_half_spread = (half_spread * (1.0 + derisk_push_frac)).min(mid * 0.1);
                bid_half_spread = (half_spread * sweeten).max(mid * 1e-6);
            }
        }

        // Passive taper (module doc comment's fix 2): shrink the side that would
        // *increase* an already-large position as inventory approaches the cap.
        // Floored at `MIN_TAPER`, not all the way to 0 (2026-10-24, direct user
        // request: "fine tune the simulation so the worth of the stocks...
        // represent the real world" -- see module doc comment's "One-sided book
        // from fully-tapered inventory" entry): going fully one-sided,
        // population-wide (every market_maker agent is seeded with a similar
        // starting long position at construction, see sim.rs's
        // `needs_inventory_seed`), lets `mid_price()` degenerate to tracking
        // only the remaining side -- with no resting bid to average against,
        // this agent's own `micro`/`reservation` (derived from that same
        // degenerate mid) can drift for an extended dormant stretch with zero
        // actual trades to ground it, then print a large, otherwise-
        // unexplainable move the instant some other agent finally crosses into
        // wherever it drifted to. Measured directly: a pure-bot (0% human) run
        // printed zero BTC trades for 3,000+ seconds, then resumed over 60%
        // lower with no intermediate prints at all. A real market maker at or
        // past its own cap still keeps a token two-sided presence (at a worse
        // price/size), it doesn't vanish from one side of the book entirely.
        // The side that *reduces* the position keeps full size throughout;
        // only the accumulating side tapers.
        const MIN_TAPER: f64 = 0.1;
        let buy_taper = if inventory_norm > 0.0 { (1.0 - inventory_norm.min(1.0)).max(MIN_TAPER) } else { 1.0 };
        let sell_taper = if inventory_norm < 0.0 { (1.0 - inventory_norm.abs().min(1.0)).max(MIN_TAPER) } else { 1.0 };
        let buy_qty = order_qty * buy_taper;
        let sell_qty = order_qty * sell_taper;

        // Single order per side (2026-10-20, direct user request -- see
        // module doc comment's "Single order per side, deleted the ladder"):
        // `asset.bid_level`/`ask_level` were reset to default at the top of
        // `step()`; this is the only place either gets set again. The hard
        // gate below widened from `max_inventory` to `max_inventory * 1.5`
        // (same 2026-10-24 fix as the taper floor above) -- construction-time
        // inventory seeding (sim.rs) sizes each agent's starting position off
        // its *cash*, a different basis than this per-wake risk-budget-derived
        // `max_inventory`, so a freshly built agent can start out already past
        // its own cap; at the exact `max_inventory` gate, that agent's buy side
        // (and hence the floored-taper fix above) would be blocked outright
        // from the very first wake, with no way to ever work itself back under
        // cap other than selling. The wider gate gives it room to do both.
        if inventory < max_inventory * 1.5 && buy_qty > 1e-9 {
            let bid_price = (reservation - bid_half_spread).max(mid * 1e-4);
            if agent.cash > bid_price * buy_qty {
                asset.bid_level.qty_placed = buy_qty;
                asset.bid_level.price_placed = bid_price;
                actions.push(Action::Submit(NewOrderRequest {
                    asset_id: asset.asset_id,
                    side: Side::Buy,
                    order_type: OrderType::Limit,
                    tif: TimeInForce::Gtc,
                    post_only: true,
                    price: Some(bid_price),
                    qty: buy_qty,
                }));
            }
        }
        if inventory > -max_inventory * 1.5 && sell_qty > 1e-9 {
            let ask_price = (reservation + ask_half_spread).max(mid * 1e-4);
            asset.ask_level.qty_placed = sell_qty;
            asset.ask_level.price_placed = ask_price;
            actions.push(Action::Submit(NewOrderRequest {
                asset_id: asset.asset_id,
                side: Side::Sell,
                order_type: OrderType::Limit,
                tif: TimeInForce::Gtc,
                post_only: true,
                price: Some(ask_price),
                qty: sell_qty,
            }));
        }
    }

    // Jitter the wake slightly (agents don't all requote in perfect lockstep), matching
    // the "not a fixed global tick" design goal even within one bot's own cadence. One
    // shared wake for the whole agent -- it reconsiders every asset together each time,
    // not on independent per-asset schedules.
    StepResult { actions, next_wake: Some(next_wake) }
}

/// Called by `sim.rs` after an order from `Action::Submit` gets an id assigned, so the
/// bot remembers which resting order to cancel on its next requote.
pub fn on_order_id(s: &mut State, req: &NewOrderRequest, id: OrderId) {
    let Some(asset) = s.assets.iter_mut().find(|a| a.asset_id == req.asset_id) else { return };
    match req.side {
        Side::Buy => asset.bid_level.order_id = Some(id),
        Side::Sell => asset.ask_level.order_id = Some(id),
    }
}
