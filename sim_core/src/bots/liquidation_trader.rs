//! Shared margin-entry and partial-liquidation-quantity primitives, used by any
//! agent that can carry a leveraged `MarginPosition` -- the generic
//! `leveraged_fraction` side-bet piggyback and `human.rs`'s cash-constrained-
//! conviction top-up today (see `AgentCommon::margin` in `agent.rs`). The
//! system-wide liquidation sweep in `sim.rs` applies to *any* agent carrying a
//! margin position, regardless of which bot type opened it.
//!
//! There used to be a dedicated `liquidation_trader` bot type here whose entire
//! "strategy" was opening a random-direction leveraged bet purely to manufacture
//! liquidation/cascade events to study (spec 5.7's "soft correction vs. hard
//! cascade" comparison). Removed 2026-10-21 at the user's request: a bot whose
//! only behavior is "borrow randomly, then get liquidated" isn't a trading
//! strategy, and conflated the leverage mechanism itself with that one
//! degenerate way of using it. The liquidation *execution* styles (`Full` vs.
//! `StaggeredTwap`) and the quantity formula below are unaffected -- they apply
//! to whatever agent ends up over-leveraged, dedicated bot type or not.

use rand::Rng;

use crate::agent::{AgentCommon, MarginPosition, NewOrderRequest};
use crate::types::{AssetId, OrderType, Side, TimeInForce};

/// No-op if the agent already carries a margin position or has insufficient cash.
pub fn open_leveraged_position(
    agent: &mut AgentCommon,
    asset_id: AssetId,
    leverage_range: (f64, f64),
    position_frac_of_cash: f64,
    maintenance_margin_buffer: f64,
    mid: f64,
) -> Option<NewOrderRequest> {
    // Checked here too (not just in the shared core below) to keep this
    // function's RNG draw sequence byte-for-byte unchanged from before the
    // 2026-10-10 refactor -- every existing leveraged-sub-population sweep
    // stays exactly reproducible.
    if agent.margin.is_some() || agent.bankrupt {
        return None;
    }
    let leverage = agent.rng.gen_range(leverage_range.0..=leverage_range.1.max(leverage_range.0 + 1e-6));
    let collateral = (agent.cash * position_frac_of_cash).max(0.0);
    if collateral <= 1.0 {
        return None;
    }
    let side = if agent.rng.gen_bool(0.5) { Side::Buy } else { Side::Sell };
    let notional = collateral * leverage;
    open_leveraged_position_directed(agent, asset_id, side, leverage, notional, maintenance_margin_buffer, mid)
}

/// Shared core behind `open_leveraged_position` above: debits collateral, sets
/// `agent.margin`, and builds the sourcing order. Split out (2026-10-10) so a
/// bot can open a leveraged position *in a direction and size it already
/// decided for its own reasons* -- e.g. `human.rs`'s "I believe in this and
/// I'm out of cash to buy more of it at spot" trigger -- rather than only ever
/// through the random-direction/fixed-fraction entry above, which exists
/// specifically to produce liquidation events for spec 5.7's cascade dynamics,
/// not to model a directional conviction.
pub fn open_leveraged_position_directed(
    agent: &mut AgentCommon,
    asset_id: AssetId,
    side: Side,
    leverage: f64,
    notional: f64,
    maintenance_margin_buffer: f64,
    mid: f64,
) -> Option<NewOrderRequest> {
    if agent.margin.is_some() || agent.bankrupt || notional <= 0.0 || leverage <= 0.0 {
        return None;
    }
    let collateral = notional / leverage.max(1e-9);
    if collateral <= 1.0 || collateral > agent.cash {
        return None;
    }
    let size = notional / mid.max(1e-9);
    // Collateral is debited from cash directly here (not via the fill's price*qty --
    // that's the leveraged notional, not the margin actually posted). The order below
    // is submitted purely to source real fills/price-impact from the shared order
    // book; `sim.rs` routes margin orders through a settlement path that only applies
    // the fee, never a second cash/inventory adjustment for the notional.
    agent.cash -= collateral;
    agent.margin = Some(MarginPosition {
        asset_id,
        side,
        size,
        entry_price: mid,
        leverage,
        maintenance_margin_buffer,
        collateral,
        staggered_round: 0,
    });
    Some(NewOrderRequest {
        asset_id,
        side,
        order_type: OrderType::Market,
        tif: TimeInForce::Ioc,
        post_only: false,
        price: None,
        qty: size,
    })
}

/// How much of a margin position must be closed to bring its margin ratio back to a
/// safe level above the maintenance threshold. Simplified model (documented in the
/// README): closing a slice at the current mark price is assumed not to change total
/// equity (collateral + unrealized pnl); the target post-liquidation size is chosen so
/// equity / new_notional == mm_rate * safety_factor.
pub fn required_liquidation_qty(pos: &MarginPosition, mark_price: f64, maintenance_margin_buffer: f64, safety_factor: f64) -> f64 {
    let mm_rate = maintenance_margin_buffer / pos.leverage.max(1e-9);
    let target_ratio = (mm_rate * safety_factor).min(1.0);
    let equity = pos.collateral + pos.unrealized_pnl(mark_price);
    if equity <= 0.0 || target_ratio <= 0.0 {
        return pos.size;
    }
    let target_notional = equity / target_ratio;
    let target_size = (target_notional / mark_price.max(1e-9)).max(0.0);
    (pos.size - target_size).clamp(0.0, pos.size)
}
