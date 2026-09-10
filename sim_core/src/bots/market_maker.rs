//! Market-maker bots: 4.1 in the spec. All four variants requote periodically
//! (cancel previous quotes, place new two-sided post-only quotes) rather than
//! reacting tick-by-tick, matching real MM requote cadences (~0.1-1s).

use rand::Rng;

use super::{RollingWindow, StepCtx};
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderId, OrderType, Side, SimTime, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    SymmetricAvellanedaStoikov,
    FixedOffset,
    InventorySkew,
    AdaptiveSpread,
}

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
    pub fn from_params(variant: Variant, asset_id: AssetId, initial_price: f64, p: &toml::Value) -> Self {
        let vol_window_len = params::get_u32(p, "vol_window", 30) as usize;
        let price = initial_price.max(1e-9);
        let order_qty = params::get_f64(p, "order_notional", 3000.0) / price;
        let max_inventory = params::get_f64(p, "max_inventory_notional", 30_000.0) / price;
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
            last_mid: None,
            last_bid_id: None,
            last_ask_id: None,
        }
    }
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    let mut actions = Vec::new();
    if let Some(id) = s.last_bid_id.take() {
        actions.push(Action::Cancel(id));
    }
    if let Some(id) = s.last_ask_id.take() {
        actions.push(Action::Cancel(id));
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

    let sigma = s.vol_window.std().unwrap_or(0.0).max(1e-9);
    let inventory = agent.inventory_of(s.asset_id);

    let (reservation, half_spread) = match s.variant {
        Variant::SymmetricAvellanedaStoikov => {
            // Rolling-horizon approximation of Avellaneda-Stoikov (2008): the terminal
            // horizon T-t is replaced by a fixed rolling horizon, standard practice for
            // an always-on (non-terminating) market.
            //
            // Dimensional fix: the textbook formula's sigma is an *absolute* price
            // volatility (dS, dollars/sqrt(time)), and its inventory term uses q in
            // raw units against that same absolute scale. `sigma` here is a fractional
            // (log-return) volatility instead, and inventory is in raw asset units
            // whose *scale* (via max_inventory, itself now dollar-notional-derived --
            // see from_params) varies enormously by asset price. Naively mixing these
            // treats a price-scale-independent quantity as if it were already in price
            // units, adding e.g. a fixed ~1.3 absolute-price spread term regardless of
            // whether the asset is worth $60,000 or $0.15 -- negligible for the former,
            // many times the entire price for the latter (this produced actual
            // sub-half-price/over-double-price quotes in testing; see git history).
            // Fixed by working entirely in *fractional* terms (inventory normalized by
            // max_inventory, to [-1, 1]-ish) and converting to a price only once, at
            // the end, by multiplying by mid.
            let inventory_norm = inventory / s.max_inventory.max(1e-9);
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
            let skewed = mid - s.skew_factor * mid * inventory;
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
    // toxicity estimate must never be able to produce a non-positive or absurdly wide
    // quote. Without this, adaptive_spread in particular could occasionally emit a
    // negative bid price. 10% half-spread (20% total) is already a very wide, stressed
    // quote for any real market maker -- a tighter belt-and-braces bound than the
    // 50% this used before the Avellaneda-Stoikov dimensional fix above, which was
    // wide enough to let a formula bug still produce a barely-clamped, badly wrong
    // quote rather than actually catching it.
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
