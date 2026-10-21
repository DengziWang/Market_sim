//! Arbitrage bots: 4.2 in the spec.
//!
//! Simplification note (triangular): the engine models each asset as a single spot
//! book quoted in an implicit common numeraire (not dedicated pair books for every
//! A/B, B/C, A/C combination). True triangular arbitrage across 3 independent pair
//! books is listed as a natural Phase-2 extension once pair order books exist. Here,
//! triangular arbitrage is approximated as 3 assets that should maintain a configured
//! `peg_ratio` relationship; the bot trades all 3 legs (IOC, leg risk minimized by
//! sizing to the smallest available leg) whenever that relationship drifts.

use super::StepCtx;
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderType, Side, SimTime, TimeInForce, VenueId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    CrossVenue,
    Triangular,
    StatArbPairs,
    LatencyArb,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub variant: Variant,
    pub wake_interval: SimTime,
    /// Per-leg unit quantities, fixed once at construction from `order_notional`
    /// divided by *each asset's own configured reference price* -- not recomputed
    /// from the live price every wake. A first attempt did compute it live (to
    /// properly dollar-balance legs at very different price scales instead of just
    /// applying one shared unit quantity everywhere), but that reintroduced the exact
    /// runaway feedback loop already fixed once for wash_trading's volume_ranking:
    /// dividing a dollar target by a *falling* price grows the quantity, and
    /// executing a larger quantity pushes price down further. Fixing the divisor at
    /// construction keeps every other bot type's sizing safe and this should be no
    /// different (see git history for the regression this caused).
    pub qty_a: f64,
    pub qty_b: f64,
    pub qty_c: f64,
    // cross-venue / latency-arb
    pub asset_a: AssetId,
    pub venue_a: VenueId,
    pub venue_b: VenueId,
    pub transfer_cost_frac: f64,
    pub last_seen_price: Option<f64>,
    pub jump_threshold_frac: f64,
    pub cooldown_secs: SimTime,
    pub last_trade_at: Option<SimTime>,
    // triangular
    pub asset_b: AssetId,
    pub asset_c: AssetId,
    pub peg_ratio: f64,
    pub peg_deviation_threshold: f64,
    pub peg_stop_threshold: f64,
    // stat arb pairs
    pub beta: f64,
    pub spread_window: super::RollingWindow,
    pub entry_z: f64,
    pub exit_z: f64,
    pub stop_z: f64,
    pub in_position: bool,
    /// +1.0 while long the spread (bought asset_a, sold asset_b), -1.0 while short
    /// it (sold asset_a, bought asset_b), 0.0 while flat. Needed because the old
    /// exit always did "buy a, sell b" regardless of which side was actually open --
    /// correct only for the short-spread entry, so unwinding a long-spread position
    /// silently doubled it instead of closing it (2026-10-21 bug fix).
    pub position_sign: f64,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::CrossVenue => "cross_venue",
            Variant::Triangular => "triangular",
            Variant::StatArbPairs => "stat_arb_pairs",
            Variant::LatencyArb => "latency_arb",
        }
    }

    /// `price_a`/`price_b`/`price_c`: the three assets' configured reference
    /// (initial) prices -- see `qty_a`/`qty_b`/`qty_c`'s doc comment. `peg_ratio`
    /// defaults to whatever value makes the triangular-arbitrage relationship exactly
    /// hold at those reference prices (`price_c * price_b / price_a`), rather than a
    /// flat 1.0 -- three unrelated assets (e.g. BTC, ETH, SOL) essentially never
    /// satisfy `price_c == price_a/price_b` by coincidence, so an uncalibrated
    /// peg_ratio made the "deviation" signal permanently, enormously past threshold
    /// and the bot trade every single wake regardless of any real mispricing.
    ///
    /// `cash`: this agent's own starting capital, used to size `order_notional`
    /// (2026-09-12, at the user's request after a bot-by-bot realism review found
    /// every arbitrageur trading the same flat $3,000/leg regardless of whether it
    /// was capitalized at $30,000 or $100,000 -- unlike a real arbitrageur, who
    /// sizes trades to their own capital and the opportunity, not a fixed ticket
    /// size). Default `size_frac_of_cash` (10%) reproduces the old flat $3,000
    /// default almost exactly at this scenario's $30,000 capital floor, and scales
    /// up from there for better-capitalized agents.
    #[allow(clippy::too_many_arguments)]
    pub fn from_params(variant: Variant, asset_a: AssetId, asset_b: AssetId, asset_c: AssetId, venue_a: VenueId, venue_b: VenueId, price_a: f64, price_b: f64, price_c: f64, cash: f64, p: &toml::Value) -> Self {
        // Per-variant defaults below (2026-10-21 recalibration): a single shared
        // default across all four variants is what let latency_arb and
        // stat_arb_pairs each fire on ordinary short-horizon noise instead of a
        // genuine signal -- measured directly (see each field's own comment) and
        // fixed per-variant, not by changing one knob everyone shares.
        let default_z_window = match variant {
            // 50 samples at the old 0.2s wake default is a 10-second lookback --
            // far too short to tell a real spread dislocation from ordinary
            // short-horizon clustering. Measured directly: widening this (and
            // slowing the wake cadence below) took stat_arb_pairs from -85%
            // growth/100% bankrupt to +1.0%/0% bankrupt on the same 25,000s run.
            Variant::StatArbPairs => 2000,
            _ => 50,
        };
        let default_wake_interval = match variant {
            Variant::StatArbPairs => 5.0,
            // Measured directly (2026-10-21): of {2s, 1s, 5s} tried against
            // {0.5%, 0.3%, 0.8%} entry thresholds on the same 25,000s run, 2s +
            // 0.5% gave the best (least negative) result, -14.7% vs -39.6% and
            // -18.4% for the other two combinations.
            Variant::Triangular => 2.0,
            _ => 0.2,
        };
        let z_window = params::get_u32(p, "z_window", default_z_window) as usize;
        let size_frac_of_cash = params::get_f64(p, "size_frac_of_cash", 0.10);
        let order_notional = params::get_f64(p, "order_notional", cash * size_frac_of_cash);
        let default_peg_ratio = price_c.max(1e-9) * price_b.max(1e-9) / price_a.max(1e-9);
        Self {
            variant,
            wake_interval: params::get_f64(p, "wake_interval_secs", default_wake_interval),
            qty_a: order_notional / price_a.max(1e-9),
            qty_b: order_notional / price_b.max(1e-9),
            qty_c: order_notional / price_c.max(1e-9),
            asset_a,
            venue_a,
            venue_b,
            transfer_cost_frac: params::get_f64(p, "transfer_cost_frac", 0.0015),
            last_seen_price: None,
            // Measured directly on this scenario (2026-10-21): the old 0.002/5.0
            // default whipsawed on ordinary bid-ask bounce in a busy 5,000-agent
            // book (70 agents made ~210,000 taker trades in just 2,000s, an
            // almost exact 50/50 buy/sell split -- not a directional edge). This
            // default took the same 25,000s run from -79% growth/99% bankrupt to
            // -5.6%/0% bankrupt.
            jump_threshold_frac: params::get_f64(p, "jump_threshold_frac", 0.02),
            cooldown_secs: params::get_f64(p, "cooldown_secs", 60.0),
            last_trade_at: None,
            asset_b,
            asset_c,
            peg_ratio: params::get_f64(p, "peg_ratio", default_peg_ratio),
            // Measured directly (2026-10-21): the old 0.003 threshold at the old
            // 0.2s wake traded on noise around a peg fixed at construction time,
            // not a real, recalibrated relationship -- see `peg_stop_threshold`'s
            // doc comment for why the peg itself can't be trusted to hold forever
            // either. Widening the entry threshold alongside the slower wake
            // cadence above reduced (did not eliminate) the resulting loss.
            peg_deviation_threshold: params::get_f64(p, "peg_deviation_threshold", 0.005),
            // Stop-loss, not a tighter entry: the 3-asset "peg" is calibrated once
            // at construction from each asset's *initial* price (see `peg_ratio`'s
            // own doc comment) and these assets' fundamentals evolve independently
            // (no real cross-asset correlation is modeled) -- so over a long run the
            // live ratio can drift away from that initial snapshot for good, not
            // just dip and revert. Trading the gap is still worth doing (a real,
            // if non-stationary, signal), but holding a position open forever
            // waiting for a reversion that may never come is what was driving
            // triangular's bankruptcy rate. Bail out at `peg_stop_threshold`
            // (several multiples of the entry threshold) instead of holding
            // indefinitely.
            // Measured directly alongside the entry threshold/wake changes above:
            // 0.01 (paired with 0.005 entry, 2s wake) beat both a tighter (0.008)
            // and a looser (0.012) stop tried in the same sweep.
            peg_stop_threshold: params::get_f64(p, "peg_stop_threshold", 0.01),
            beta: params::get_f64(p, "beta", 1.0),
            spread_window: super::RollingWindow::new(z_window),
            // Measured directly (2026-10-21): 2.5 alongside the wider z_window/
            // slower wake above (vs the old 2.0) was part of the combination that
            // took stat_arb_pairs from -85%/100% bankrupt to +1.0%/0% bankrupt.
            entry_z: params::get_f64(p, "entry_z", 2.5),
            exit_z: params::get_f64(p, "exit_z", 0.3),
            // Same reasoning as `peg_stop_threshold` above: two independently-
            // modeled assets have no guaranteed cointegration, so a z-score that
            // keeps growing instead of reverting may mean the "pair" relationship
            // itself broke, not that a bigger reversion is coming. Cut the loss
            // rather than wait indefinitely.
            stop_z: params::get_f64(p, "stop_z", 4.0),
            in_position: false,
            position_sign: 0.0,
        }
    }
}

fn ioc(asset_id: AssetId, side: Side, qty: f64) -> Action {
    Action::Submit(NewOrderRequest {
        asset_id,
        side,
        order_type: OrderType::Market,
        tif: TimeInForce::Ioc,
        post_only: false,
        price: None,
        qty,
    })
}

/// Rebalance toward a signed target position, bounded by the relevant leg's fixed
/// qty_a/qty_b/qty_c on either side -- same pattern as momentum/mean_reversion.
/// Without this, a strategy that
/// re-evaluates every `wake_interval` while its trigger condition (a persistent
/// mispricing) holds would otherwise keep firing a fresh, uncapped order every wake
/// with no memory of the position it already built up; that compounded into an
/// effectively unbounded one-directional position in testing (see git history) for
/// CrossVenue/Triangular/LatencyArb, which (unlike StatArbPairs) had no such cap.
fn rebalance(asset_id: AssetId, current: f64, target: f64) -> Option<Action> {
    let delta = target - current;
    if delta.abs() < 1e-9 {
        return None;
    }
    let side = if delta > 0.0 { Side::Buy } else { Side::Sell };
    Some(ioc(asset_id, side, delta.abs()))
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    let mut actions = Vec::new();
    match s.variant {
        Variant::CrossVenue => {
            // `Market::mid_price` falls back to the asset's static `initial_price`
            // whenever a venue's book is empty -- the right default for a bot that
            // just needs *some* reference to quote around, but fatal here: no bot
            // type in this scenario ever trades on a non-primary venue, so venue_b's
            // book is permanently empty and that fallback would silently hand this
            // bot a frozen, 100,000s-stale "price" to arbitrage the live primary
            // price against. That isn't noise, it's a one-way bet that price returns
            // to where it started -- and it was the real cause of cross_venue's
            // 47-82% bankruptcy rate (2026-10-21), not merely the missing second
            // leg below. Read the book's own (fallback-free) mid price instead, and
            // simply don't trade while venue_b has no genuine resting liquidity.
            if let (Some(pa), Some(pb)) = (ctx.market.mid_price(s.venue_a, s.asset_a), ctx.market.book(s.venue_b, s.asset_a).mid_price()) {
                let qty = s.qty_a;
                let diff = pb - pa;
                let cost = pa * s.transfer_cost_frac;
                let current = agent.inventory_of(s.asset_a);
                // Target flips sign with which venue is cheaper, but stays 0 (no
                // legitimate signal) once the gap is inside transfer cost -- and
                // rebalance() only trades the *delta* to get there, so a persistent
                // gap does not cause repeated re-entry (both venues assumed
                // pre-funded, i.e. no real-time asset transfer modeled).
                let target = if diff.abs() > cost {
                    if diff > 0.0 { qty } else { -qty }
                } else {
                    0.0
                };
                if agent.cash > pa.max(pb) * qty {
                    if let Some(a) = rebalance(s.asset_a, current, target) {
                        actions.push(a);
                    }
                }
            }
        }
        Variant::Triangular => {
            if let (Some(pa), Some(pb), Some(pc)) = (
                ctx.market.mid_price(ctx.venue, s.asset_a),
                ctx.market.mid_price(ctx.venue, s.asset_b),
                ctx.market.mid_price(ctx.venue, s.asset_c),
            ) {
                let implied_c = pa / pb.max(1e-9) * s.peg_ratio;
                let deviation = (pc - implied_c) / implied_c.max(1e-9);
                // Each leg's *unit* quantity is fixed (see qty_a/b/c's doc comment),
                // derived from the same dollar notional at each asset's own reference
                // price, so the three legs are dollar-hedged against each other at
                // that reference -- not exactly at the live price, but far safer than
                // recomputing from a price this very trade might be moving.
                let sign = if deviation.abs() > s.peg_stop_threshold {
                    // Past the stop: treat a deviation this large as the peg
                    // relationship itself having drifted, not a dislocation to
                    // fade, and flatten instead of continuing to bet against it.
                    0.0
                } else if deviation.abs() > s.peg_deviation_threshold {
                    if deviation > 0.0 { -1.0 } else { 1.0 }
                } else {
                    0.0
                };
                let target_c = sign * s.qty_c;
                let target_a = -sign * s.qty_a; // opposite leg of the same cycle
                let target_b = sign * s.qty_b;
                // Crude solvency gate: each leg is worth roughly `order_notional`
                // (qty_x * price_x), so require a multiple of headroom above one
                // leg's worth rather than trying to net the (partially offsetting)
                // buy/sell legs exactly.
                if agent.cash > 3.0 * s.qty_a * pa {
                    if let Some(a) = rebalance(s.asset_c, agent.inventory_of(s.asset_c), target_c) {
                        actions.push(a);
                    }
                    if let Some(a) = rebalance(s.asset_a, agent.inventory_of(s.asset_a), target_a) {
                        actions.push(a);
                    }
                    if let Some(a) = rebalance(s.asset_b, agent.inventory_of(s.asset_b), target_b) {
                        actions.push(a);
                    }
                }
            }
        }
        Variant::StatArbPairs => {
            if let (Some(pa), Some(pb)) = (ctx.market.mid_price(ctx.venue, s.asset_a), ctx.market.mid_price(ctx.venue, s.asset_b)) {
                let qty_a = s.qty_a;
                // Dollar-match the hedge leg, not unit-match it (2026-10-21 fix):
                // `qty_a` is a unit count sized in *asset_a's* units at construction
                // (order_notional / price_a) -- reusing that same number of units
                // for asset_b made the "beta-hedged" leg's actual dollar exposure
                // off by a factor of price_a/price_b, which for this scenario's
                // assets spans 5 orders of magnitude (BTC ~$60,000 vs DOGE ~$0.15).
                // A pair assigned (asset_a=DOGE, asset_b=BTC) was never remotely
                // market-neutral; it was a massive naked bet dominated by whichever
                // leg happened to have the higher price. Scale by the live price
                // ratio so the b leg is worth `beta` times the a leg's dollar size,
                // which is what a hedge ratio is supposed to mean.
                let qty_b = (s.qty_a * pa / pb.max(1e-9) * s.beta).max(0.0);
                let spread = pa - s.beta * pb;
                s.spread_window.push(spread);
                if let (Some(mean), Some(std)) = (s.spread_window.mean(), s.spread_window.std()) {
                    if std > 1e-9 && s.spread_window.is_full() {
                        let z = (spread - mean) / std;
                        // Stop-loss: if z keeps growing past several multiples of
                        // the entry threshold instead of reverting, the spread's
                        // historical mean/std (both estimated from the same rolling
                        // window being blown through) may no longer describe a real
                        // mean-reverting relationship between two independently-
                        // modeled assets -- bail rather than hold indefinitely.
                        let stop_hit = s.in_position && z.abs() > s.stop_z;
                        if !s.in_position && z > s.entry_z {
                            actions.push(ioc(s.asset_a, Side::Sell, qty_a));
                            actions.push(ioc(s.asset_b, Side::Buy, qty_b));
                            s.in_position = true;
                            s.position_sign = -1.0;
                        } else if !s.in_position && z < -s.entry_z {
                            actions.push(ioc(s.asset_a, Side::Buy, qty_a));
                            actions.push(ioc(s.asset_b, Side::Sell, qty_b));
                            s.in_position = true;
                            s.position_sign = 1.0;
                        } else if s.in_position && (z.abs() < s.exit_z || stop_hit) {
                            // Unwind by trading the *opposite* of how this position
                            // was entered, not an always-the-same-direction close --
                            // see `position_sign`'s doc comment for the bug this fixes.
                            if s.position_sign > 0.0 {
                                actions.push(ioc(s.asset_a, Side::Sell, qty_a));
                                actions.push(ioc(s.asset_b, Side::Buy, qty_b));
                            } else {
                                actions.push(ioc(s.asset_a, Side::Buy, qty_a));
                                actions.push(ioc(s.asset_b, Side::Sell, qty_b));
                            }
                            s.in_position = false;
                            s.position_sign = 0.0;
                        }
                    }
                }
            }
        }
        Variant::LatencyArb => {
            if let Some(p) = ctx.market.mid_price(ctx.venue, s.asset_a) {
                if let Some(prev) = s.last_seen_price {
                    let ret = (p - prev) / prev.max(1e-9);
                    // Cooldown (2026-10-21 fix): measured directly on this scenario,
                    // this bot was re-triggering on ordinary tick-to-tick bid-ask
                    // bounce rather than genuine jumps -- 70 agents made ~210,000
                    // taker trades in just 2,000s (nearly a third of the *entire
                    // market's* trade count), an almost exact 50/50 buy/sell split,
                    // i.e. pure whipsaw, not a real directional edge. Requiring a
                    // cooldown since the last trade stops it from flip-flopping on
                    // every single noisy sample while still reacting to a fresh jump
                    // once the cooldown has passed.
                    let off_cooldown = s.last_trade_at.map_or(true, |t| ctx.now - t >= s.cooldown_secs);
                    if ret.abs() > s.jump_threshold_frac && off_cooldown {
                        // Exploits the fact this bot polls far more frequently
                        // (short wake_interval, low configured latency) than most of
                        // the population, letting it react to a move before it fully
                        // propagates through slower agents' quotes. Bounded by
                        // order_qty via rebalance(), not fired unconditionally every
                        // wake: its own trades otherwise could look like a fresh
                        // "jump" and re-trigger itself indefinitely in one direction.
                        let target = if ret > 0.0 { s.qty_a } else { -s.qty_a };
                        if let Some(a) = rebalance(s.asset_a, agent.inventory_of(s.asset_a), target) {
                            actions.push(a);
                            s.last_trade_at = Some(ctx.now);
                        }
                    }
                }
                s.last_seen_price = Some(p);
            }
        }
    }
    StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) }
}
