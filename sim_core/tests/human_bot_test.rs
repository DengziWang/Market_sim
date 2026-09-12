//! Regression tests for the two behavioral-finance upgrades to `human` bots
//! (2026-09-13, at the user's request after research into which established
//! models better simulate retail trader behavior than the original hand-tuned
//! heuristic): a real Cumulative Prospect Theory (Tversky & Kahneman 1992) value
//! function replacing the old fixed disposition-effect multipliers, and
//! Brock-Hommes (1998) fitness-based chartist/fundamentalist strategy switching.
//! Both drive `bots::human::step` directly with manually-crafted state, the same
//! deterministic, targeted approach `market_maker_test.rs` uses.

use sim_core::agent::Action;
use sim_core::bots::{step as bots_step, BotStrategy, StepCtx};
use sim_core::config::ScenarioConfig;
use sim_core::sim::Simulation;

const CONFIG: &str = r#"
[run]
seed = 1
session_seconds = 10.0

[[assets]]
symbol = "A"
tick_size = 0.01
initial_price = 100.0
fundamental_vol = 0.02
fundamental_theta = 0.1

[export]
out_dir = "/tmp/does_not_matter"

[[population.groups]]
bot_type = "human"
variant = "retail"
count = 1
initial_cash_range = [1000000.0, 1000000.0]
"#;

fn build() -> Simulation {
    let parsed = ScenarioConfig::from_toml_str(CONFIG).expect("config must parse");
    Simulation::build(parsed).expect("population must build")
}

/// Extract the sell quantity from a step, given a pre-set inventory and
/// avg_entry_price -- asserts the bot actually chose to sell (raw_signal must be
/// negative) so the caller doesn't misread "no action" as "zero qty".
fn sell_qty_after_price_drop(entry_multiple: f64) -> f64 {
    let mut sim = build();
    let BotStrategy::Human(state) = &mut sim.strategies[0] else {
        panic!("population group 0 must be the human bot configured above");
    };
    // Force a deterministic, real "act" and a downward price move (sell signal):
    // p_act_given_signal=1.0 makes the roll deterministic; last_price above the
    // current mid (100.0) with a tiny threshold makes the chartist signal
    // strongly negative regardless of the RNG-perceived sentiment noise term.
    state.p_act_given_signal = 1.0;
    state.price_signal_threshold = 0.001;
    state.sentiment_signal_threshold = 100.0; // effectively mutes the sentiment component
    state.last_price = Some(120.0);
    state.last_wake_time = Some(-1000.0); // typical-gap elapsed since "last wake"
    state.avg_entry_price = Some(100.0 * entry_multiple);
    sim.agents[0].add_inventory(0, 10.0);
    state.last_seen_inventory = 10.0; // matches the inventory already added, so the
                                       // fill-inference logic doesn't (mis)read this
                                       // as a fresh buy on this very step

    let ctx = StepCtx { market: &sim.market, venue: 0, now: 0.0 };
    let result = bots_step(&mut sim.agents[0], &mut sim.strategies[0], &ctx);
    for action in &result.actions {
        if let Action::Submit(req) = action {
            assert_eq!(req.side, sim_core::types::Side::Sell, "expected a sell given a downward price move");
            return req.qty;
        }
    }
    panic!("expected a Submit action (deterministic act_given_signal=1.0 and a well-above-threshold signal)");
}

#[test]
fn cpt_disposition_effect_makes_selling_more_reluctant_the_deeper_the_loss() {
    let sell_at_gain = sell_qty_after_price_drop(0.8); // mid=100 > entry=80: a gain
    let sell_at_small_loss = sell_qty_after_price_drop(1.05); // a modest loss
    let sell_at_big_loss = sell_qty_after_price_drop(1.5); // a much deeper loss

    assert!(sell_at_gain > sell_at_small_loss, "selling a winner ({sell_at_gain}) should be more eager than selling a small loser ({sell_at_small_loss})");
    assert!(
        sell_at_small_loss > sell_at_big_loss,
        "CPT's diminishing sensitivity in the loss domain should make a deeper loss ({sell_at_big_loss}) *more* reluctant to sell than a shallow one ({sell_at_small_loss}) -- \
         the old fixed-multiplier model couldn't distinguish these at all, since it only looked at the sign of the P&L, not its magnitude"
    );
}

#[test]
fn chartist_vs_fundamentalist_fitness_actually_changes_which_signal_wins() {
    // Same raw market condition both times: price sits *above* its fundamental
    // value (a fundamentalist would say "overpriced, sell") while having just
    // risen (a chartist would say "momentum, buy") -- only the fitness balance
    // differs between the two runs, so any difference in the resulting trade
    // direction is attributable to the Brock-Hommes switching mechanism, not to
    // the underlying signal itself.
    fn traded_side(chartist_fitness: f64, fundamentalist_fitness: f64) -> sim_core::types::Side {
        let mut sim = build();
        // With an empty book, `mid_price` falls back to `initial_price` -- which is
        // also exactly where `fundamental_value` starts, so there'd be zero
        // deviation to react to. Set it below the (soon-to-be) mid directly so
        // "price sits above fundamental value" is actually true for this test.
        sim.market.fundamental_value.insert(0, 90.0);
        let BotStrategy::Human(state) = &mut sim.strategies[0] else {
            panic!("population group 0 must be the human bot configured above");
        };
        state.p_act_given_signal = 1.0;
        state.price_signal_threshold = 0.001;
        state.fundamentalist_signal_threshold = 0.001;
        state.sentiment_signal_threshold = 100.0;
        state.intensity_of_choice = 50.0; // decisive switching for a clean test
        state.chartist_fitness = chartist_fitness;
        state.fundamentalist_fitness = fundamentalist_fitness;
        state.last_chartist_signal = 0.0; // no fitness update should fire this step
        state.last_fundamentalist_signal = 0.0; // (elapsed-since-last-wake path below)
        state.last_price = Some(95.0); // price rose 95 -> mid (fundamental_value starts
                                        // at initial_price=100, so mid=100 > fundamental
                                        // -- "overpriced" -- while also being a rise
                                        // from last_price -- "momentum")
        state.last_wake_time = Some(-1000.0);
        // The fundamentalist-dominant case resolves to a Sell -- give the agent
        // something to sell, since `human` (correctly) never shorts, the same way
        // `sell_qty_after_price_drop` above does.
        sim.agents[0].add_inventory(0, 10.0);
        state.last_seen_inventory = 10.0;

        let ctx = StepCtx { market: &sim.market, venue: 0, now: 0.0 };
        let result = bots_step(&mut sim.agents[0], &mut sim.strategies[0], &ctx);
        for action in &result.actions {
            if let Action::Submit(req) = action {
                return req.side;
            }
        }
        panic!("expected a Submit action");
    }

    let chartist_dominant = traded_side(10.0, 0.0);
    let fundamentalist_dominant = traded_side(0.0, 10.0);

    assert_eq!(chartist_dominant, sim_core::types::Side::Buy, "a chartist-dominant agent should follow the recent upward momentum and buy");
    assert_eq!(
        fundamentalist_dominant,
        sim_core::types::Side::Sell,
        "a fundamentalist-dominant agent, facing the exact same market condition, should instead sell because price sits above fundamental value -- \
         confirms fitness actually drives which strategy the agent follows, not just theoretically"
    );
}
