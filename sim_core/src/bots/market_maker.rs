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

use rand::Rng;

use super::{RollingWindow, StepCtx};
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderId, OrderType, Side, SimTime, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    SymmetricAvellanedaStoikov,
    FixedOffset,
    InventorySkew,
    AdaptiveSpread,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub variant: Variant,
    pub asset_id: AssetId,
    pub wake_interval: SimTime,
    pub order_qty: f64,
    pub base_half_spread_frac: f64,
    pub gamma: f64,
    pub k: f64,
    pub horizon_secs: f64,
    pub max_inventory: f64,
    pub skew_factor: f64,
    pub vol_window: RollingWindow,
    pub toxicity_window: RollingWindow,
    pub withdrawal_vol_frac: f64,
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
    /// How much of the excess (inventory beyond `derisk_inventory_frac`) gets
    /// actively unwound per wake -- a small fraction, not all of it at once,
    /// so a single bad step doesn't itself dump a large market order into a
    /// thin book and move price further against this agent.
    pub derisk_frac_per_step: f64,
    /// Slow EMA of mid, the reference de-risking compares the current price
    /// against to decide whether this agent's position is actually
    /// underwater (not just skewed) -- see the module doc comment. A long
    /// position is "underwater" when mid has dropped below this trailing
    /// average since it was built; a short, when mid has risen above it.
    pub ema_mid: Option<f64>,
    pub last_mid: Option<f64>,
    pub last_bid_id: Option<OrderId>,
    pub last_ask_id: Option<OrderId>,
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

    /// `initial_price`: the asset's configured reference price, used to convert
    /// dollar-notional-denominated size params (`order_notional`,
    /// `max_inventory_notional`) into asset units. Without this, a single
    /// unit-denominated `order_qty` default applied uniformly across assets spanning
    /// a wide price range (e.g. BTC at $60k next to DOGE at $0.15) provides wildly
    /// different *dollar* depth per asset -- a fixed 0.05-unit quote is $3,000 of
    /// resting liquidity on BTC but $0.0075 on DOGE, leaving cheap assets
    /// structurally paper-thin against any real-sized order. This produced a genuine
    /// price-collapse cascade in testing (see git history) once a cheap asset's
    /// market-maker depth was orders of magnitude too thin relative to other bots'
    /// (also now notional-scaled) order sizes.
    /// `rng`: used once, here, to draw this specific agent's own
    /// `derisk_inventory_frac` (see that field's doc comment) from a range --
    /// the same construction-time RNG `sim.rs`'s `build_population` already
    /// uses for this agent's starting cash/leverage, so the draw is
    /// deterministic given a fixed run seed.
    pub fn from_params(variant: Variant, asset_id: AssetId, initial_price: f64, p: &toml::Value, rng: &mut impl Rng) -> Self {
        let vol_window_len = params::get_u32(p, "vol_window", 30) as usize;
        let price = initial_price.max(1e-9);
        let order_qty = params::get_f64(p, "order_notional", 3000.0) / price;
        let max_inventory = params::get_f64(p, "max_inventory_notional", 30_000.0) / price;
        let derisk_frac_min = params::get_f64(p, "derisk_inventory_frac_min", 0.6);
        let derisk_frac_max = params::get_f64(p, "derisk_inventory_frac_max", 0.9);
        Self {
            variant,
            asset_id,
            wake_interval: params::get_f64(p, "wake_interval_secs", 0.5),
            order_qty,
            base_half_spread_frac: params::get_f64(p, "base_half_spread_frac", 0.0008),
            gamma: params::get_f64(p, "gamma", 0.1),
            k: params::get_f64(p, "k", 1.5),
            horizon_secs: params::get_f64(p, "horizon_secs", 60.0),
            max_inventory,
            skew_factor: params::get_f64(p, "skew_factor", 0.0005),
            vol_window: RollingWindow::new(vol_window_len),
            toxicity_window: RollingWindow::new(vol_window_len),
            withdrawal_vol_frac: params::get_f64(p, "withdrawal_vol_frac", 0.01),
            derisk_inventory_frac: if derisk_frac_max > derisk_frac_min { rng.gen_range(derisk_frac_min..derisk_frac_max) } else { derisk_frac_min },
            derisk_frac_per_step: params::get_f64(p, "derisk_frac_per_step", 0.04),
            ema_mid: None,
            last_mid: None,
            last_bid_id: None,
            last_ask_id: None,
        }
    }
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    let mut actions = Vec::new();
    if let Some(id) = s.last_bid_id.take() {
        actions.push(Action::Cancel(id, s.asset_id));
    }
    if let Some(id) = s.last_ask_id.take() {
        actions.push(Action::Cancel(id, s.asset_id));
    }

    let Some(mid) = ctx.market.mid_price(ctx.venue, s.asset_id) else {
        return StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) };
    };
    if let Some(prev) = s.last_mid {
        if prev > 0.0 {
            s.vol_window.push((mid / prev).ln());
            s.toxicity_window.push(((mid - prev) / prev).abs());
        }
    }
    s.last_mid = Some(mid);

    // Slow trailing reference for the de-risking "is this position actually
    // underwater" check below -- see module doc comment. Time constant tied
    // to this agent's own `horizon_secs` (already its reservation-price
    // horizon), not a separate tunable: a requote-interval-scaled discrete
    // EMA, so it reacts on the same timescale this agent already reasons
    // about risk on, not tick-by-tick noise.
    s.ema_mid = Some(match s.ema_mid {
        None => mid,
        Some(prev) => {
            let alpha = s.wake_interval / (s.horizon_secs + s.wake_interval).max(1e-9);
            prev + alpha * (mid - prev)
        }
    });

    let sigma = s.vol_window.std().unwrap_or(0.0).max(1e-9);
    let inventory = agent.inventory_of(s.asset_id);
    // Shared by every variant below that reacts to inventory: raw asset units (as
    // `inventory` is) scale inversely with price via `max_inventory`'s own
    // dollar-notional derivation (see `from_params`) -- the same $30,000 of risk
    // is 0.5 BTC units but 200,000 DOGE units. Normalizing first, to [-1, 1]-ish,
    // keeps every variant's inventory reaction a *fraction* of mid regardless of
    // which asset it's quoting, not an asset-price-dependent one.
    let inventory_norm = inventory / s.max_inventory.max(1e-9);

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
            let reservation = mid * (1.0 - inventory_norm * s.gamma * sigma.powi(2) * s.horizon_secs);
            // The order-arrival-intensity term (2/gamma)*ln(1+gamma/k) is dimensionless
            // by construction (a pure function of gamma/k), not a price or even a
            // fraction of one -- rescaled so the module's own defaults land in the same
            // ballpark as fixed_offset's default 0.0008 base half-spread fraction.
            const K_TERM_SCALE: f64 = 800.0;
            let spread_frac = s.gamma * sigma.powi(2) * s.horizon_secs + (2.0 / s.gamma) * (1.0 + s.gamma / s.k).ln() / K_TERM_SCALE;
            (reservation, (spread_frac * mid / 2.0).max(mid * 1e-5))
        }
        Variant::FixedOffset => (mid, mid * s.base_half_spread_frac),
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
            let skewed = mid * (1.0 - s.skew_factor * inventory_norm);
            (skewed, mid * s.base_half_spread_frac)
        }
        Variant::AdaptiveSpread => {
            let toxicity = s.toxicity_window.mean().unwrap_or(0.0);
            let widen = 1.0 + (sigma / mid.max(1e-9)) * 50.0 + toxicity * 100.0;
            (mid, mid * s.base_half_spread_frac * widen)
        }
    };

    // Liquidity withdrawal under turbulence (adaptive-spread variant): pull quotes
    // entirely rather than just widening, once realized vol crosses the threshold.
    if s.variant == Variant::AdaptiveSpread && sigma / mid.max(1e-9) > s.withdrawal_vol_frac {
        return StepResult { actions, next_wake: Some(ctx.now + s.wake_interval) };
    }

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
    let bid_price = (reservation - half_spread).max(mid * 1e-4);
    let ask_price = (reservation + half_spread).max(mid * 1e-4);

    if inventory < s.max_inventory && agent.cash > bid_price * s.order_qty {
        actions.push(Action::Submit(NewOrderRequest {
            asset_id: s.asset_id,
            side: Side::Buy,
            order_type: OrderType::Limit,
            tif: TimeInForce::Gtc,
            post_only: true,
            price: Some(bid_price),
            qty: s.order_qty,
        }));
    }
    if inventory > -s.max_inventory {
        actions.push(Action::Submit(NewOrderRequest {
            asset_id: s.asset_id,
            side: Side::Sell,
            order_type: OrderType::Limit,
            tif: TimeInForce::Gtc,
            post_only: true,
            price: Some(ask_price),
            qty: s.order_qty,
        }));
    }

    // Active de-risking -- see module doc comment. Runs for every variant,
    // on top of whatever quotes it just posted above: requoting alone only
    // changes what happens on this agent's *next* fill, it does nothing
    // about inventory already sitting on the books from past fills. Gated on
    // actually being underwater (not just skewed): a long only de-risks when
    // mid has dropped below `ema_mid`, a short only when mid has risen above
    // it -- merely holding a big position that's currently *working* (price
    // moved in this agent's favor since it was built) is not a reason to
    // sell into further upside or buy back a profitable short.
    let underwater = if inventory > 0.0 {
        s.ema_mid.map_or(false, |e| mid < e)
    } else {
        s.ema_mid.map_or(false, |e| mid > e)
    };
    if underwater && inventory_norm.abs() > s.derisk_inventory_frac {
        let excess_frac = inventory_norm.abs() - s.derisk_inventory_frac;
        let derisk_qty = (s.max_inventory * excess_frac * s.derisk_frac_per_step).min(inventory.abs());
        if derisk_qty > 1e-9 {
            // Long -> sell to reduce; short -> buy to reduce.
            let side = if inventory > 0.0 { Side::Sell } else { Side::Buy };
            actions.push(Action::Submit(NewOrderRequest {
                asset_id: s.asset_id,
                side,
                order_type: OrderType::Market,
                tif: TimeInForce::Ioc,
                post_only: false,
                price: None,
                qty: derisk_qty,
            }));
        }
    }

    // Jitter the wake slightly (agents don't all requote in perfect lockstep), matching
    // the "not a fixed global tick" design goal even within one bot's own cadence.
    let jitter = agent.rng.gen_range(-0.05..0.05) * s.wake_interval;
    StepResult { actions, next_wake: Some(ctx.now + s.wake_interval + jitter) }
}

/// Called by `sim.rs` after an order from `Action::Submit` gets an id assigned, so the
/// bot remembers which resting order to cancel on its next requote.
pub fn on_order_id(s: &mut State, req: &NewOrderRequest, id: OrderId) {
    match req.side {
        Side::Buy => s.last_bid_id = Some(id),
        Side::Sell => s.last_ask_id = Some(id),
    }
}
