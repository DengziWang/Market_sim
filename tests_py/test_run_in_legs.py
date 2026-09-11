"""Tests for market_sim.run_scenario_in_legs: splitting one long simulation into
several shorter, separately-saved sessions via checkpoint/resume (requested so a
huge run's memory footprint stays bounded, and so only the very first leg sees the
cold-start burst every fresh simulation exhibits -- see README's "Memory: the actual
scaling limit" section and sim_core/src/sim.rs's initial-wake seeding)."""
import os

import numpy as np
import pytest

import market_sim as ms

SMALL_CFG = {
    "run": {"seed": 5},
    "assets": [{"symbol": "A", "tick_size": 0.01, "initial_price": 100.0, "fundamental_vol": 0.02, "fundamental_theta": 0.1}],
    "export": {"out_dir": "runs/_test_legs"},
    "population": {
        "groups": [
            {"bot_type": "market_maker", "variant": "fixed_offset", "count": 5, "params": {"wake_interval_secs": 0.2}},
            {"bot_type": "noise", "variant": "zero_intelligence", "count": 20, "params": {"wake_interval_secs": 0.3}},
        ]
    },
}


def test_run_scenario_in_legs_produces_one_directory_per_leg(tmp_path):
    out_dir = str(tmp_path / "legs")
    legs = ms.run_scenario_in_legs(SMALL_CFG, leg_seconds=20.0, n_legs=3, out_dir=out_dir)

    assert len(legs) == 3
    for i, leg in enumerate(legs, start=1):
        assert leg.index == i
        assert leg.out_dir == os.path.join(out_dir, f"leg_{i:04d}")
        assert os.path.exists(leg.paths.run_metadata)
        assert os.path.exists(leg.paths.checkpoint), "every leg should have a checkpoint, including the last"

    # Absolute simulated time accumulates leg over leg (20s, 40s, 60s), not reset.
    assert legs[0].meta["actual_session_seconds"] == pytest.approx(20.0)
    assert legs[1].meta["actual_session_seconds"] == pytest.approx(40.0)
    assert legs[2].meta["actual_session_seconds"] == pytest.approx(60.0)


def test_resume_picks_up_from_an_interrupted_sequence(tmp_path):
    out_dir = str(tmp_path / "legs")
    first = ms.run_scenario_in_legs(SMALL_CFG, leg_seconds=15.0, n_legs=2, out_dir=out_dir)
    assert len(first) == 2

    # Simulate a crash between calls: ask for more legs than exist yet, with the
    # default resume=True -- should continue from leg 2's checkpoint, not restart.
    second = ms.run_scenario_in_legs(SMALL_CFG, leg_seconds=15.0, n_legs=4, out_dir=out_dir)
    assert len(second) == 4
    assert second[0].out_dir == first[0].out_dir
    assert second[1].out_dir == first[1].out_dir
    assert second[2].meta["actual_session_seconds"] == pytest.approx(45.0)
    assert second[3].meta["actual_session_seconds"] == pytest.approx(60.0)

    # Calling again with the same n_legs is a no-op -- nothing new to compute.
    third = ms.run_scenario_in_legs(SMALL_CFG, leg_seconds=15.0, n_legs=4, out_dir=out_dir)
    assert len(third) == 4
    assert [l.out_dir for l in third] == [l.out_dir for l in second]


def test_discover_legs_and_load_run_legs_roundtrip(tmp_path):
    out_dir = str(tmp_path / "legs")
    legs = ms.run_scenario_in_legs(SMALL_CFG, leg_seconds=20.0, n_legs=3, out_dir=out_dir)

    discovered = ms.discover_legs(out_dir)
    assert discovered == [leg.out_dir for leg in legs]

    combined = ms.load_run_legs(out_dir)
    total_trades = sum(leg.meta["n_trades"] for leg in legs)
    assert len(combined["trades"]["time"]) == total_trades
    assert combined["meta"]["actual_session_seconds"] == pytest.approx(60.0)
    # agent_states is a point-in-time snapshot (not concatenated) -- should match the
    # last leg's, not accumulate across legs.
    assert len(combined["agent_states"]["agent_id"]) == legs[-1].meta["n_agents"]


def test_load_run_legs_raises_on_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        ms.load_run_legs(str(tmp_path / "nothing_here"))


def test_ignored_run_fields_warn_but_dont_crash(tmp_path):
    cfg = {**SMALL_CFG, "run": {**SMALL_CFG["run"], "min_trades": 99999, "max_session_seconds": 999.0}}
    out_dir = str(tmp_path / "legs")
    with pytest.warns(UserWarning, match="ignoring run"):
        legs = ms.run_scenario_in_legs(cfg, leg_seconds=15.0, n_legs=2, out_dir=out_dir)
    # Each leg still runs for exactly leg_seconds, not extended toward min_trades.
    assert legs[0].meta["actual_session_seconds"] == pytest.approx(15.0)
    assert legs[1].meta["actual_session_seconds"] == pytest.approx(30.0)


def test_only_the_first_leg_shows_the_cold_start_burst():
    """The actual behavioral claim: a fresh start's initial trade rate is far above
    its own steady-state rate (every agent's first wake is jittered into the first
    simulated second, hitting an empty book); a *continued* leg picks up an
    already-warmed market and doesn't reproduce that. Uses a bigger, more active
    population than SMALL_CFG so the steady-state rate is stable enough to compare
    against -- still small enough to run quickly."""
    cfg = {
        "run": {"seed": 3, "session_seconds": 60.0},
        "assets": [{"symbol": "A", "tick_size": 0.01, "initial_price": 100.0, "fundamental_vol": 0.02, "fundamental_theta": 0.1}],
        "export": {"out_dir": "runs/_test_legs_burst"},
        "population": {
            "groups": [
                {"bot_type": "market_maker", "variant": "fixed_offset", "count": 10, "params": {"wake_interval_secs": 0.1}},
                {"bot_type": "noise", "variant": "zero_intelligence", "count": 60, "params": {"wake_interval_secs": 0.2}},
            ]
        },
    }

    def first_second_ratio(trades_time: list) -> float:
        t = np.asarray(trades_time, dtype=float)
        if len(t) == 0:
            return 0.0
        t = t - t.min()
        first_sec = (t < 1.0).sum()
        steady = t[t >= 3.0]
        span = max(t.max() - 3.0, 1e-9)
        steady_rate = len(steady) / span if len(steady) else 1e-9
        return first_sec / max(steady_rate, 1e-9)

    r1 = ms.run_scenario_dict(cfg).result
    ratio_leg1 = first_second_ratio(r1["trades"]["time"])

    r2 = ms.continue_simulation(r1["checkpoint"], 60.0)
    ratio_leg2 = first_second_ratio(r2["trades"]["time"])

    # A relative comparison rather than fixed absolute thresholds: at this small a
    # population the exact ratio is somewhat seed-/noise-sensitive (a real run at
    # realistic_mix.toml's 500-agent scale showed leg 1 at ~10x vs. leg 2's ~1x --
    # see project memory), but leg 1 clearing its own market from a cold, empty book
    # should reliably dominate a continuation's steady-state-ish start by a wide
    # margin regardless of exact scale.
    assert ratio_leg1 > 2.0 * max(ratio_leg2, 0.5), f"expected leg 1's cold start to clearly dominate leg 2's, got leg1={ratio_leg1:.2f} leg2={ratio_leg2:.2f}"
