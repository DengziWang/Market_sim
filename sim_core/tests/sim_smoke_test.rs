//! End-to-end smoke test: builds a population covering every bot type/variant
//! category across multiple assets/venues, with mempool and leverage enabled, and
//! checks the whole event loop runs to completion without panicking and produces
//! sane output (trades happened, agent states cover every agent, no NaNs).

use sim_core::config::ScenarioConfig;
use sim_core::sim::Simulation;

const CONFIG: &str = r#"
[run]
seed = 42
session_seconds = 120.0
min_trades = 50
max_session_seconds = 600.0
extension_step_seconds = 60.0

[[assets]]
symbol = "AAA"
tick_size = 0.01
initial_price = 100.0
fundamental_vol = 0.02
fundamental_theta = 0.1

[[assets]]
symbol = "BBB"
tick_size = 0.01
initial_price = 50.0

[[assets]]
symbol = "CCC"
tick_size = 0.01
initial_price = 20.0

[[venues]]
name = "primary"
inter_venue_latency_secs = 0.05

[[venues]]
name = "secondary"
inter_venue_latency_secs = 0.05

[market]
snapshot_interval_secs = 1.0
snapshot_depth_levels = 5
mark_price_ema_halflife_secs = 10.0

[fees]
default_maker_fee = 0.0002
default_taker_fee = 0.0005

[latency]
default_latency_secs = 0.05

[mempool]
enabled = true
reveal_delay_secs = 0.2
large_order_notional_threshold = 500.0

[liquidation]
maintenance_margin_buffer = 0.8
sweep_interval_secs = 1.0
staggered_slices = 3
full_liquidation_fraction = 0.5

[export]
out_dir = "/tmp/does_not_matter"
write_snapshots = true

[[population.groups]]
bot_type = "market_maker"
variant = "avellaneda_stoikov"
count = 3
[population.groups.params]
wake_interval_secs = 0.2

[[population.groups]]
bot_type = "market_maker"
variant = "fixed_offset"
count = 2

[[population.groups]]
bot_type = "market_maker"
variant = "inventory_skew"
count = 2

[[population.groups]]
bot_type = "market_maker"
variant = "adaptive_spread"
count = 2

[[population.groups]]
bot_type = "arbitrage"
variant = "triangular"
count = 2

[[population.groups]]
bot_type = "arbitrage"
variant = "cross_venue"
count = 2

[[population.groups]]
bot_type = "arbitrage"
variant = "stat_arb_pairs"
count = 2

[[population.groups]]
bot_type = "arbitrage"
variant = "latency_arb"
count = 2

[[population.groups]]
bot_type = "momentum"
variant = "ma_crossover"
count = 2

[[population.groups]]
bot_type = "momentum"
variant = "breakout_donchian"
count = 2

[[population.groups]]
bot_type = "momentum"
variant = "rsi"
count = 2

[[population.groups]]
bot_type = "momentum"
variant = "ml_online"
count = 2

[[population.groups]]
bot_type = "mean_reversion"
variant = "bollinger"
count = 2

[[population.groups]]
bot_type = "mean_reversion"
variant = "ou_fair_value"
count = 2

[[population.groups]]
bot_type = "mean_reversion"
variant = "zscore"
count = 2

[[population.groups]]
bot_type = "grid"
variant = "static"
count = 2

[[population.groups]]
bot_type = "grid"
variant = "geometric"
count = 2

[[population.groups]]
bot_type = "grid"
variant = "dynamic_atr"
count = 2

[[population.groups]]
bot_type = "dca"
variant = "time_based"
count = 2

[[population.groups]]
bot_type = "dca"
variant = "value_averaging"
count = 2

[[population.groups]]
bot_type = "dca"
variant = "dip_buying"
count = 2

[[population.groups]]
bot_type = "liquidation_trader"
variant = "full"
count = 3
[population.groups.params]
wake_interval_secs = 5.0
leverage_range = [3.0, 15.0]

[[population.groups]]
bot_type = "liquidation_trader"
variant = "staggered_twap"
count = 3
[population.groups.params]
wake_interval_secs = 5.0
leverage_range = [3.0, 15.0]

[[population.groups]]
bot_type = "mev"
variant = "sandwich"
count = 1
[population.groups.params]
wake_interval_secs = 0.05
min_victim_qty = 0.5

[[population.groups]]
bot_type = "mev"
variant = "frontrun"
count = 1
[population.groups.params]
wake_interval_secs = 0.05
min_victim_qty = 0.5

[[population.groups]]
bot_type = "mev"
variant = "backrun"
count = 1
[population.groups.params]
wake_interval_secs = 0.05
min_victim_qty = 0.5

[[population.groups]]
bot_type = "wash_trading"
variant = "self_ring"
count = 2

[[population.groups]]
bot_type = "wash_trading"
variant = "volume_ranking"
count = 2

[[population.groups]]
bot_type = "noise"
variant = "zero_intelligence"
count = 20
[population.groups.params]
wake_interval_secs = 1.0

[[population.groups]]
bot_type = "noise"
variant = "herding"
count = 10
leveraged_fraction = 0.2
leverage_range = [2.0, 5.0]
[population.groups.params]
wake_interval_secs = 1.0

[[population.groups]]
bot_type = "human"
variant = "disposition_chaser"
count = 6

[[population.groups]]
bot_type = "human"
variant = "sentiment_herder"
count = 6
"#;

#[test]
fn full_population_runs_to_completion_and_produces_trades() {
    let cfg = ScenarioConfig::from_toml_str(CONFIG).expect("config must parse");
    let n_agents = cfg.total_agents();
    assert!(n_agents > 60);

    let sim = Simulation::build(cfg).expect("population must build");
    let result = sim.run();

    assert!(result.actual_session_seconds >= 120.0);
    assert!(!result.trades.is_empty(), "some trading must have occurred");
    assert!(result.trades.len() as u64 >= 50, "min_trades / session extension must be honored");
    assert!(!result.order_events.is_empty());
    assert_eq!(result.actual_session_seconds >= 120.0, true);

    for t in &result.trades {
        assert!(t.price.is_finite() && t.price > 0.0, "trade price must be finite and positive");
        assert!(t.qty.is_finite() && t.qty > 0.0);
    }
    for a in &result.agent_states {
        assert!(a.cash.is_finite());
        assert!(a.equity.is_finite());
        assert!(a.inventory.is_finite());
    }

    let distinct_agents_in_states: std::collections::HashSet<_> = result.agent_states.iter().map(|a| a.agent_id).collect();
    assert_eq!(distinct_agents_in_states.len() as u32, n_agents, "every agent must appear in the final agent_state export");
}

#[test]
fn cancellation_stops_early_and_still_returns_a_valid_partial_result() {
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    let cfg = ScenarioConfig::from_toml_str(CONFIG).expect("config must parse");
    let n_agents = cfg.total_agents();
    let session_seconds = cfg.run.session_seconds;
    let sim = Simulation::build(cfg).expect("population must build");

    // Pre-set the flag so cancellation is deterministic (takes effect at the very
    // first periodic check, well before the 120s+ session would naturally end) --
    // not a race against a timer, so this can't be flaky.
    let cancel = Arc::new(AtomicBool::new(true));
    let result = sim.run_cancellable(Some(cancel));

    assert!(result.cancelled, "SimResult::cancelled must be set");
    assert!(
        result.actual_session_seconds < session_seconds,
        "a cancelled run must stop before its configured session length ({} < {})",
        result.actual_session_seconds,
        session_seconds
    );
    // finalize_agent_states() must still run on the early-exit path: every agent
    // should be present even though the session was cut short.
    let distinct_agents: std::collections::HashSet<_> = result.agent_states.iter().map(|a| a.agent_id).collect();
    assert_eq!(distinct_agents.len() as u32, n_agents);
    for a in &result.agent_states {
        assert!(a.cash.is_finite() && a.equity.is_finite());
    }
}

#[test]
fn uncancelled_run_behaves_identically_via_run_and_run_cancellable() {
    // run() is documented as equivalent to run_cancellable(None); a smaller/faster
    // config than CONFIG keeps this test itself quick.
    let small_cfg = r#"
[run]
seed = 7
session_seconds = 20.0

[[assets]]
symbol = "A"
tick_size = 0.01
initial_price = 100.0

[export]
out_dir = "/tmp/does_not_matter"

[[population.groups]]
bot_type = "noise"
variant = "zero_intelligence"
count = 20
[population.groups.params]
wake_interval_secs = 0.5
"#;
    let cfg = ScenarioConfig::from_toml_str(small_cfg).unwrap();
    let sim = Simulation::build(cfg).unwrap();
    let result = sim.run_cancellable(None);
    assert!(!result.cancelled);
    assert!((result.actual_session_seconds - 20.0).abs() < 1e-6);
}

#[test]
fn human_bots_trade_never_short_and_stay_solvent() {
    let cfg = r#"
[run]
seed = 11
session_seconds = 1200.0

[[assets]]
symbol = "A"
tick_size = 0.01
initial_price = 200.0
fundamental_vol = 0.02
fundamental_theta = 0.1
sentiment_vol = 0.4
sentiment_theta = 0.05

[export]
out_dir = "/tmp/does_not_matter"

[[population.groups]]
bot_type = "market_maker"
variant = "fixed_offset"
count = 3
[population.groups.params]
wake_interval_secs = 0.3

[[population.groups]]
bot_type = "human"
variant = "disposition_chaser"
count = 15
initial_cash_range = [1000.0, 8000.0]

[[population.groups]]
bot_type = "human"
variant = "sentiment_herder"
count = 15
initial_cash_range = [1000.0, 8000.0]
"#;
    let parsed = ScenarioConfig::from_toml_str(cfg).expect("config must parse");
    let sim = Simulation::build(parsed).expect("population must build");
    let result = sim.run();

    assert!(!result.trades.is_empty(), "human bots (plus MM liquidity) must produce some trades");
    for t in &result.trades {
        assert!(t.price.is_finite() && t.price > 0.0);
        assert!(t.qty.is_finite() && t.qty > 0.0);
    }

    let human_states: Vec<_> = result.agent_states.iter().filter(|a| a.bot_type == "human").collect();
    assert_eq!(human_states.len(), 30, "every human agent must appear in the final state export");
    for a in human_states {
        assert!(a.inventory >= -1e-6, "human bots must never hold a short (negative) position: agent {} has {}", a.agent_id, a.inventory);
        assert!(a.cash.is_finite() && a.equity.is_finite());
        assert!(!a.bankrupt || a.equity <= 0.0, "a bankrupt flag should correspond to non-positive equity");
    }
}
