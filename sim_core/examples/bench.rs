//! Pure-Rust benchmark harness, no PyO3/Python involved -- isolates the engine's
//! own compute cost from marshaling overhead (already optimized separately, see
//! sim_pyo3's numpy-array result conversion). Used to find real optimization
//! targets via `perf record`/`perf report`, not guesses.
//!
//! Usage: cargo run --release --example bench -- [n_agents] [session_seconds]

use std::time::Instant;

use sim_core::config::ScenarioConfig;
use sim_core::sim::Simulation;

fn scaled_config(scale: u32, session_seconds: f64) -> String {
    format!(
        r#"
[run]
seed = 1
session_seconds = {session_seconds}

[[assets]]
symbol = "BTC"
tick_size = 0.5
initial_price = 60000.0
fundamental_vol = 0.015
fundamental_theta = 0.08

[[assets]]
symbol = "ETH"
tick_size = 0.05
initial_price = 3000.0
fundamental_vol = 0.02
fundamental_theta = 0.08

[[assets]]
symbol = "SOL"
tick_size = 0.01
initial_price = 150.0
fundamental_vol = 0.03
fundamental_theta = 0.1

[[assets]]
symbol = "XRP"
tick_size = 0.0001
initial_price = 0.6
fundamental_vol = 0.025
fundamental_theta = 0.1

[[assets]]
symbol = "DOGE"
tick_size = 0.00001
initial_price = 0.15
fundamental_vol = 0.04
fundamental_theta = 0.12

[[venues]]
name = "primary"
inter_venue_latency_secs = 0.05

[[venues]]
name = "secondary"
inter_venue_latency_secs = 0.08

[market]
snapshot_interval_secs = 1.0
snapshot_depth_levels = 10
mark_price_ema_halflife_secs = 30.0

[fees]
default_maker_fee = 0.0002
default_taker_fee = 0.0005

[latency]
default_latency_secs = 0.08

[mempool]
enabled = true
reveal_delay_secs = 0.4
large_order_notional_threshold = 25000.0

[liquidation]
maintenance_margin_buffer = 0.8
sweep_interval_secs = 1.0
staggered_slices = 5
full_liquidation_fraction = 0.5

[export]
out_dir = "/tmp/does_not_matter"
write_snapshots = true

[[population.groups]]
bot_type = "market_maker"
variant = "avellaneda_stoikov"
count = {mm1}
[population.groups.params]
wake_interval_secs = 0.3

[[population.groups]]
bot_type = "market_maker"
variant = "fixed_offset"
count = {mm2}

[[population.groups]]
bot_type = "market_maker"
variant = "inventory_skew"
count = {mm3}

[[population.groups]]
bot_type = "market_maker"
variant = "adaptive_spread"
count = {mm4}

[[population.groups]]
bot_type = "arbitrage"
variant = "cross_venue"
count = {arb1}

[[population.groups]]
bot_type = "arbitrage"
variant = "triangular"
count = {arb2}

[[population.groups]]
bot_type = "momentum"
variant = "ma_crossover"
count = {mom1}
leveraged_fraction = 0.15
leverage_range = [2.0, 8.0]

[[population.groups]]
bot_type = "momentum"
variant = "rsi"
count = {mom2}

[[population.groups]]
bot_type = "mean_reversion"
variant = "bollinger"
count = {mr1}

[[population.groups]]
bot_type = "liquidation_trader"
variant = "staggered_twap"
count = {liq1}
[population.groups.params]
wake_interval_secs = 300.0
leverage_range = [3.0, 20.0]

[[population.groups]]
bot_type = "noise"
variant = "zero_intelligence"
count = {noise1}
initial_cash_range = [500.0, 15000.0]

[[population.groups]]
bot_type = "noise"
variant = "herding"
count = {noise2}
initial_cash_range = [500.0, 15000.0]

[[population.groups]]
bot_type = "human"
variant = "retail"
count = {human1}
initial_cash_range = [1000.0, 20000.0]
"#,
        session_seconds = session_seconds,
        mm1 = 10 * scale,
        mm2 = 10 * scale,
        mm3 = 10 * scale,
        mm4 = 10 * scale,
        arb1 = 5 * scale,
        arb2 = 5 * scale,
        mom1 = 10 * scale,
        mom2 = 10 * scale,
        mr1 = 10 * scale,
        liq1 = 15 * scale,
        noise1 = 200 * scale,
        noise2 = 44 * scale,
        human1 = 40 * scale,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scale: u32 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1);
    let session_seconds: f64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(3000.0);

    let text = scaled_config(scale, session_seconds);
    let cfg = ScenarioConfig::from_toml_str(&text).expect("config must parse");
    let n_agents = cfg.total_agents();
    let sim = Simulation::build(cfg).expect("population must build");

    let t0 = Instant::now();
    let result = sim.run();
    let elapsed = t0.elapsed();

    let n_events = result.trades.len() + result.order_events.len();
    println!("agents={n_agents} session_seconds={session_seconds}");
    println!("trades={} order_events={} snapshots={}", result.trades.len(), result.order_events.len(), result.snapshots.len());
    println!("elapsed={:.3}s events/sec={:.0}", elapsed.as_secs_f64(), n_events as f64 / elapsed.as_secs_f64());
}
