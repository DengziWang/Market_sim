//! Institutional "flash crash" seller: a one-shot Percentage-of-Volume (PoV)
//! execution algorithm modeled on the May 6, 2010 flash crash (SEC/CFTC joint
//! report, "Findings Regarding the Market Events of May 6, 2010"). The real
//! seller did not dump a fixed quantity instantly -- it ran an automated
//! execution algorithm targeting ~9% of the asset's own trailing 1-minute
//! trade volume, with no price or time limit, which fed back on itself (HFT
//! "hot potato" trading inflated measured volume, which let the algorithm
//! sell faster still) and let a multi-hour order finish in ~20 minutes. This
//! bot reproduces that pacing mechanism -- not a lump-sum dump -- against a
//! fixed total quantity.
//!
//! Stays completely dormant (no orders, no further wakes scheduled until the
//! trigger) until `start_time_secs`, so it can sit in the population from the
//! very first leg without disturbing a warmup period -- avoids needing any
//! mid-run order-injection plumbing (none exists in this engine; see
//! `sim_pyo3::continue_simulation`, which only ever extends a checkpointed
//! run by a duration, never mutates its agent roster). Once triggered it
//! sells until `total_qty` is exhausted, then stops for good (no re-arming).

use super::StepCtx;
use crate::agent::{Action, AgentCommon, NewOrderRequest, StepResult};
use crate::params;
use crate::types::{AssetId, OrderType, Side, SimTime, TimeInForce};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Variant {
    /// The only variant today -- kept as an enum (rather than a bare struct)
    /// for parity with every other bot module's `Variant`/`parse_*_variant!`
    /// pattern in `sim.rs`, and in case a non-volume-pegged ("Twap") variant
    /// is ever wanted for comparison.
    Pov,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    pub variant: Variant,
    pub asset_id: AssetId,
    /// Fixed total quantity to sell over the life of this run. Deliberately an
    /// absolute quantity, not volume-relative -- only the *pacing* below is
    /// pegged to trailing volume, matching the real mechanism. How much is
    /// still left to sell is deliberately NOT tracked as a separate mutable
    /// counter here -- see `step`'s doc comment on why it's derived fresh
    /// from the agent's own inventory every wake instead.
    pub total_qty: f64,
    /// Target participation rate relative to trailing volume (0.09 = the real
    /// event's ~9%).
    pub participation_rate: f64,
    /// Trailing window the participation rate is measured against, in seconds
    /// (60.0 = the real event's 1-minute window).
    pub trailing_window_secs: SimTime,
    /// Sim time at which this bot wakes up and starts selling. Before this,
    /// every wake is a no-op that reschedules straight to this time rather
    /// than idling on `wake_interval_secs` -- cheaper, and keeps the warmup
    /// period's trailing-volume/book-depth baseline completely undisturbed.
    pub start_time_secs: SimTime,
    /// How often (once active) this bot re-samples trailing volume and submits
    /// its next slice. Not the real event's own internal cadence (not
    /// disclosed at that granularity in the report) -- a few seconds is fine
    /// given the participation rate, not the wake cadence, sets the pacing.
    pub wake_interval_secs: SimTime,
}

impl State {
    pub fn variant_name(&self) -> &'static str {
        match self.variant {
            Variant::Pov => "pov",
        }
    }

    pub fn from_params(variant: Variant, asset_id: AssetId, p: &toml::Value) -> Self {
        Self {
            variant,
            asset_id,
            total_qty: params::get_f64(p, "total_qty", 0.0).max(0.0),
            participation_rate: params::get_f64(p, "participation_rate", 0.09).max(0.0),
            trailing_window_secs: params::get_f64(p, "trailing_window_secs", 60.0).max(1.0),
            start_time_secs: params::get_f64(p, "start_time_secs", 0.0).max(0.0),
            wake_interval_secs: params::get_f64(p, "wake_interval_secs", 3.0).max(0.1),
        }
    }
}

pub fn step(agent: &mut AgentCommon, s: &mut State, ctx: &StepCtx) -> StepResult {
    // How much has actually sold so far is read back from the agent's own
    // inventory (updated by `Sim::settle` with the *actual* fill quantity on
    // every trade), not a self-reported running counter decremented by the
    // *requested* order size. An IOC market order against a thin book or the
    // circuit breaker routinely fills far less than requested -- an earlier
    // version of this bot decremented its own counter by the request size
    // regardless, so it would believe a single mostly-unfilled order had
    // exhausted the whole total and go dormant after selling a tiny fraction
    // of `total_qty` (confirmed directly: logged <1% filled before stopping).
    // This agent starts with zero inventory in this asset and only ever
    // submits Sell orders on it, so `-inventory_of` is exactly cumulative
    // actual sold volume, with no possibility of drifting out of sync.
    let sold_so_far = (-agent.inventory_of(s.asset_id)).max(0.0);
    let remaining = (s.total_qty - sold_so_far).max(0.0);
    if remaining <= 1e-9 {
        return StepResult { actions: Vec::new(), next_wake: None };
    }
    if ctx.now < s.start_time_secs {
        return StepResult { actions: Vec::new(), next_wake: Some(s.start_time_secs) };
    }

    // Pegged to this asset's own trailing trade volume -- the real mechanism's
    // feedback loop (more volume -> sell faster -> more volume) falls out of
    // this naturally, with no special-casing: `trailing_volume` already
    // includes this bot's own just-filled trades from prior wakes within the
    // window, exactly like it would for the real algorithm reading back the
    // tape.
    let trailing_volume = ctx.market.trailing_volume(s.asset_id, ctx.now, s.trailing_window_secs);
    let rate_per_sec = s.participation_rate * trailing_volume / s.trailing_window_secs;
    let qty = (rate_per_sec * s.wake_interval_secs).min(remaining).max(0.0);

    let mut actions = Vec::new();
    if qty > 1e-9 {
        actions.push(Action::Submit(NewOrderRequest {
            asset_id: s.asset_id,
            side: Side::Sell,
            order_type: OrderType::Market,
            tif: TimeInForce::Ioc,
            post_only: false,
            price: None,
            qty,
        }));
    }

    StepResult { actions, next_wake: Some(ctx.now + s.wake_interval_secs) }
}
