//! Regression tests for two real bugs found by inspecting an actual large run
//! (5,000 agents, 300,000s): the Avellaneda-Stoikov and InventorySkew market
//! makers both mixed a dollar/raw-unit-scale quantity into a formula meant to
//! produce a *fractional* (relative-to-mid) price adjustment, without first
//! normalizing by `max_inventory` -- the same bug class in two different
//! variants, the second found while investigating why the first fix alone
//! didn't fully explain why BTC stayed calm but the cheaper, higher-unit-count
//! assets (DOGE especially) didn't. See market_maker.rs's `reservation` clamp
//! and the `InventorySkew` arm's `inventory_norm` fix for the full writeup.
//!
//! Both tests directly drive `bots::market_maker::step` with manually-crafted
//! state (rather than hoping a full stochastic run happens to reproduce the
//! same rare/not-so-rare event) -- deterministic and targeted, the same
//! condition-triggering approach `matching_engine_tests.rs`'s circuit-breaker
//! tests use at the order-book level.

use sim_core::agent::Action;
use sim_core::bots::{step as bots_step, BotStrategy, StepCtx};
use sim_core::config::ScenarioConfig;
use sim_core::sim::Simulation;

const CONFIG: &str = r#"
[run]
seed = 1
session_seconds = 10.0

[[assets]]
symbol = "DOGE"
tick_size = 0.00001
initial_price = 0.15
fundamental_vol = 0.04
fundamental_theta = 0.12

[export]
out_dir = "/tmp/does_not_matter"

[[population.groups]]
bot_type = "market_maker"
variant = "avellaneda_stoikov"
count = 1
"#;

#[test]
fn avellaneda_stoikov_reservation_stays_bounded_under_an_extreme_volatility_spike() {
    let parsed = ScenarioConfig::from_toml_str(CONFIG).expect("config must parse");
    let mut sim = Simulation::build(parsed).expect("population must build");

    let mid = sim.market.mid_price(0, 0).expect("book should have a fallback mid price at t=0");
    assert!((mid - 0.15).abs() < 1e-9, "sanity: mid should start at initial_price");

    // Craft exactly the condition the bug report found: a handful of large
    // log-returns already sitting in this bot's rolling volatility window (as if
    // it had just observed a wild few trades), plus inventory pinned near its max
    // -- both terms the unbounded `reservation` formula multiplies together.
    let BotStrategy::MarketMaker(state) = &mut sim.strategies[0] else {
        panic!("population group 0 must be the avellaneda_stoikov market maker configured above");
    };
    // Alternating +/-2.5, not a constant: `RollingWindow::std` is a genuine
    // standard deviation, so a repeated constant has zero variance regardless of
    // its magnitude -- these need to actually *vary* to produce a large sigma,
    // same as real (wild, not just large) observed returns would.
    for i in 0..state.vol_window.capacity {
        state.vol_window.push(if i % 2 == 0 { 2.5 } else { -2.5 });
    }
    sim.agents[0].add_inventory(0, state.max_inventory * 0.999);

    let ctx = StepCtx { market: &sim.market, venue: 0, now: 0.0 };
    let result = bots_step(&mut sim.agents[0], &mut sim.strategies[0], &ctx);

    let mut saw_a_quote = false;
    for action in &result.actions {
        if let Action::Submit(req) = action {
            let price = req.price.expect("market maker quotes are limit orders with an explicit price");
            saw_a_quote = true;
            assert!(
                price >= mid * 0.75 && price <= mid * 1.25,
                "quote price {price} should stay within a generous band around mid {mid} even under an \
                 extreme volatility spike -- reservation (and hence both sides of the quote) must be \
                 clamped, not just half_spread"
            );
        }
    }
    assert!(saw_a_quote, "expected at least one Submit action (bid and/or ask) from this wake");
}

const INVENTORY_SKEW_CONFIG: &str = r#"
[run]
seed = 1
session_seconds = 10.0

[[assets]]
symbol = "DOGE"
tick_size = 0.00001
initial_price = 0.15
fundamental_vol = 0.04
fundamental_theta = 0.12

[export]
out_dir = "/tmp/does_not_matter"

[[population.groups]]
bot_type = "market_maker"
variant = "inventory_skew"
count = 1
"#;

#[test]
fn inventory_skew_stays_bounded_at_an_entirely_ordinary_inventory_level() {
    // Deliberately NOT an extreme edge case (not near-max inventory, no crafted
    // volatility spike) -- the whole point of this bug was that it broke DOGE at
    // a perfectly ordinary inventory level, precisely because raw DOGE units for
    // a given dollar position vastly outnumber raw BTC units for the same dollar
    // position. Half of max_inventory is an unremarkable, everyday position for
    // an active market maker to be sitting at.
    let parsed = ScenarioConfig::from_toml_str(INVENTORY_SKEW_CONFIG).expect("config must parse");
    let mut sim = Simulation::build(parsed).expect("population must build");
    let mid = sim.market.mid_price(0, 0).expect("book should have a fallback mid price at t=0");

    let BotStrategy::MarketMaker(state) = &mut sim.strategies[0] else {
        panic!("population group 0 must be the inventory_skew market maker configured above");
    };
    let half_max = state.max_inventory * 0.5;
    sim.agents[0].add_inventory(0, half_max);

    let ctx = StepCtx { market: &sim.market, venue: 0, now: 0.0 };
    let result = bots_step(&mut sim.agents[0], &mut sim.strategies[0], &ctx);

    let mut saw_a_quote = false;
    for action in &result.actions {
        if let Action::Submit(req) = action {
            let price = req.price.expect("market maker quotes are limit orders with an explicit price");
            saw_a_quote = true;
            assert!(
                price > 0.0 && price >= mid * 0.9 && price <= mid * 1.1,
                "quote price {price} should stay close to mid {mid} at a perfectly ordinary (50%-of-max) \
                 inventory level -- the pre-fix formula used raw asset units directly, and DOGE's raw \
                 units for a given dollar position vastly outnumber BTC's for the same position, so this \
                 same inventory level produced a deeply negative reservation price on DOGE specifically"
            );
        }
    }
    assert!(saw_a_quote, "expected at least one Submit action (bid and/or ask) from this wake");
}
