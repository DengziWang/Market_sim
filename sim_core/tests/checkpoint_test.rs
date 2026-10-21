//! Checkpoint/resume: save a simulation's full internal state mid-flight, reload it
//! in a fresh `Simulation` value, and confirm continuing from there behaves exactly
//! like continuing the original (never-serialized) instance would have -- i.e. the
//! round trip through bincode loses nothing that affects future evolution.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use sim_core::config::ScenarioConfig;
use sim_core::sim::Simulation;

const CONFIG: &str = r#"
[run]
seed = 123
session_seconds = 300.0

[[assets]]
symbol = "A"
tick_size = 0.01
initial_price = 100.0
fundamental_vol = 0.02
fundamental_theta = 0.1
sentiment_vol = 0.4
sentiment_theta = 0.05

[[assets]]
symbol = "B"
tick_size = 0.01
initial_price = 50.0

[market]
snapshot_interval_secs = 1.0

[liquidation]
sweep_interval_secs = 2.0

[export]
out_dir = "/tmp/does_not_matter"
write_snapshots = true

[[population.groups]]
bot_type = "market_maker"
variant = "fixed_offset"
count = 3
[population.groups.params]
wake_interval_secs = 0.3

[[population.groups]]
bot_type = "noise"
variant = "zero_intelligence"
count = 15
[population.groups.params]
wake_interval_secs = 0.5

[[population.groups]]
bot_type = "momentum"
variant = "rsi"
count = 7
leveraged_fraction = 1.0
leverage_range = [3.0, 12.0]

[[population.groups]]
bot_type = "human"
variant = "retail"
count = 10
"#;

#[test]
fn checkpoint_round_trip_preserves_deterministic_continuation() {
    // Byte-for-byte equality isn't a meaningful check here: `Market`/`AgentCommon`
    // use plain `HashMap`s, whose iteration order (and therefore CBOR map encoding
    // order) depends on a randomized per-instance hasher seed, not insertion order
    // -- re-serializing a freshly deserialized copy can legitimately produce
    // differently-ordered (but equally correct) bytes. The property that actually
    // matters is checked instead: restoring the *same* checkpoint bytes twice and
    // resuming each copy identically (same seed carried in the checkpoint, same
    // additional_seconds) must produce bit-identical trade histories, since nothing
    // about continuation should depend on which physical HashMap layout happened to
    // come out of deserialization.
    let cfg = ScenarioConfig::from_toml_str(CONFIG).expect("config must parse");
    let sim = Simulation::build(cfg).expect("population must build");
    let (sim, _result) = sim.run_cancellable_keep_state(None, None);
    let bytes = sim.to_checkpoint_bytes().expect("checkpoint must serialize");

    // 300s, not 120s (2026-10-20): this tiny fixed population's resting-bid
    // depth is chronically near-zero with occasional brief recoveries (CONFIG
    // has no asset-sizing/eligibility knobs to correct it, unlike the real
    // scenario files), making trade activity bursty -- confirmed directly
    // that *this is true of this CONFIG regardless of any later behavioral
    // change*: trade count in the baseline (pre-market_maker-depth-cap)
    // trajectory is already flat at 26 across 120/180/240s before jumping to
    // 260 at 300s, i.e. a ~120s quiet period already existed, just at a
    // different point in time. A behavior-neutral-on-average change to any
    // bot elsewhere in the sim (not specific to the depth-aware sizing cap
    // added the same day) can shift exactly where that quiet period falls
    // without indicating anything wrong -- which a 120s window is long
    // enough to land entirely inside, tripping the liveness assert below for
    // reasons unrelated to what it's meant to catch. 300s has a comfortable
    // margin in both trajectories (260 vs. 283 trades, measured directly).
    let restored_a = Simulation::from_checkpoint_bytes(&bytes).expect("checkpoint must deserialize");
    let restored_b = Simulation::from_checkpoint_bytes(&bytes).expect("checkpoint must deserialize");
    let result_a = restored_a.resume(300.0, None);
    let result_b = restored_b.resume(300.0, None);

    assert!(!result_a.trades.is_empty(), "resumed leg must produce trades");
    assert_eq!(result_a.trades.len(), result_b.trades.len(), "two restores of the same checkpoint must continue identically");
    for (ta, tb) in result_a.trades.iter().zip(result_b.trades.iter()) {
        assert_eq!(ta.time, tb.time);
        assert!((ta.price - tb.price).abs() < 1e-9);
        assert!((ta.qty - tb.qty).abs() < 1e-9);
        assert_eq!(ta.maker_agent_id, tb.maker_agent_id);
        assert_eq!(ta.taker_agent_id, tb.taker_agent_id);
    }
}

#[test]
fn resuming_a_checkpoint_continues_producing_valid_trades() {
    let cfg = ScenarioConfig::from_toml_str(CONFIG).expect("config must parse");
    let sim = Simulation::build(cfg).expect("population must build");
    let (sim, first_result) = sim.run_cancellable_keep_state(None, None);
    assert!(!first_result.trades.is_empty(), "first leg must have produced trades");
    let n_agents_before = first_result.agent_states.iter().map(|a| a.agent_id).collect::<std::collections::HashSet<_>>().len();

    let bytes = sim.to_checkpoint_bytes().expect("checkpoint must serialize");
    let restored = Simulation::from_checkpoint_bytes(&bytes).expect("checkpoint must deserialize");

    let second_result = restored.resume(300.0, None);
    assert!(!second_result.trades.is_empty(), "resumed leg must also produce trades");
    assert!(
        (second_result.actual_session_seconds - (first_result.actual_session_seconds + 300.0)).abs() < 1e-6,
        "resumed session must report a session length extended by exactly `additional_seconds`"
    );
    for t in &second_result.trades {
        assert!(t.price.is_finite() && t.price > 0.0);
        assert!(t.qty.is_finite() && t.qty > 0.0);
    }
    let distinct_agents_after = second_result.agent_states.iter().map(|a| a.agent_id).collect::<std::collections::HashSet<_>>().len();
    assert_eq!(distinct_agents_after, n_agents_before, "every agent must still be present after resuming");
}

#[test]
fn zero_second_resume_on_a_completed_run_is_a_pure_no_op() {
    // A resume() extending the session by exactly 0 seconds should immediately hit
    // the new EndOfSession with no agent ever getting an extra wake -- confirms the
    // reused (not re-seeded) event queue behaves correctly at this boundary.
    let cfg = ScenarioConfig::from_toml_str(CONFIG).expect("config must parse");
    let sim = Simulation::build(cfg).expect("population must build");
    let (sim, first) = sim.run_cancellable_keep_state(None, None);

    let bytes = sim.to_checkpoint_bytes().unwrap();
    let restored = Simulation::from_checkpoint_bytes(&bytes).unwrap();
    let after_reload = restored.resume(0.0, None);

    assert!(after_reload.trades.is_empty(), "a zero-second resume should produce no additional trades");
    assert!((after_reload.actual_session_seconds - first.actual_session_seconds).abs() < 1e-6);
}

#[test]
fn cancelled_run_can_still_be_checkpointed_and_resumed() {
    let cfg = ScenarioConfig::from_toml_str(CONFIG).expect("config must parse");
    let sim = Simulation::build(cfg).expect("population must build");

    let cancel = Arc::new(AtomicBool::new(true));
    let (sim, cancelled_result) = sim.run_cancellable_keep_state(Some(cancel), None);
    assert!(cancelled_result.cancelled);
    assert!(cancelled_result.actual_session_seconds < 300.0);

    let bytes = sim.to_checkpoint_bytes().expect("a cancelled sim must still checkpoint");
    let restored = Simulation::from_checkpoint_bytes(&bytes).expect("must deserialize");
    let resumed = restored.resume(60.0, None);
    assert!(!resumed.cancelled);
    assert!(
        (resumed.actual_session_seconds - (cancelled_result.actual_session_seconds + 60.0)).abs() < 1e-6
    );
}

#[test]
fn progress_handle_reaches_the_final_simulated_time() {
    // A plain, non-flaky check that doesn't depend on wall-clock timing: whatever
    // the progress handle last saw by the time the run returns must be exactly the
    // final `actual_session_seconds` -- drain_loop's doc comment promises a final
    // write after the loop specifically so a caller doesn't see a stale value up to
    // ~500ms behind the true end.
    let cfg = ScenarioConfig::from_toml_str(CONFIG).expect("config must parse");
    let sim = Simulation::build(cfg).expect("population must build");
    let progress = Arc::new(AtomicU64::new(f64::to_bits(-1.0)));
    let (_sim, result) = sim.run_cancellable_keep_state(None, Some(progress.clone()));

    let seen = f64::from_bits(progress.load(Ordering::Relaxed));
    assert!((seen - result.actual_session_seconds).abs() < 1e-6, "progress handle must end exactly at the final simulated time");
}

#[test]
fn progress_handle_advances_monotonically_during_a_run() {
    // Best-effort timing check (run on a background thread, sampled from the main
    // thread every 50ms): a large-enough population/session that the run takes
    // measurable wall-clock time should produce a strictly-increasing sequence of
    // intermediate progress reads, never exceeding the eventual final value.
    let cfg = r#"
[run]
seed = 77
session_seconds = 1800.0

[[assets]]
symbol = "A"
tick_size = 0.01
initial_price = 100.0

[export]
out_dir = "/tmp/does_not_matter"

[[population.groups]]
bot_type = "noise"
variant = "zero_intelligence"
count = 300
[population.groups.params]
wake_interval_secs = 0.2
"#;
    let parsed = ScenarioConfig::from_toml_str(cfg).expect("config must parse");
    let sim = Simulation::build(parsed).expect("population must build");
    let progress = Arc::new(AtomicU64::new(f64::to_bits(0.0)));

    let progress_for_thread = progress.clone();
    let handle = std::thread::spawn(move || sim.run_cancellable_keep_state(None, Some(progress_for_thread)));

    let mut samples = Vec::new();
    for _ in 0..100 {
        std::thread::sleep(std::time::Duration::from_millis(50));
        samples.push(f64::from_bits(progress.load(Ordering::Relaxed)));
        if handle.is_finished() {
            break;
        }
    }
    let (_sim, result) = handle.join().expect("worker thread must not panic");

    for w in samples.windows(2) {
        assert!(w[1] >= w[0], "progress must never go backwards: {:?}", samples);
    }
    assert!(samples.iter().any(|&s| s > 0.0), "expected at least one nonzero intermediate progress read; samples={:?}", samples);
    assert!(samples.iter().all(|&s| s <= result.actual_session_seconds + 1e-6), "no intermediate read should exceed the final time");
}
