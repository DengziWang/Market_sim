#!/usr/bin/env python3
"""One-off driver for the specific run the user described (5,000 agents,
300,000 simulated seconds), split into 200 legs of 1,500s each via
run_scenario_in_legs so it stays memory-bounded on this machine (measured at
this exact scale before launching: ~2.9GB peak RSS per 1,500s/5,000-agent leg,
~10s pure compute -- see terminal history / project memory for the calibration
run this was based on).

Self-healing: if a leg sequence run raises (crash, transient error, etc.), this
retries by calling run_scenario_in_legs again -- resume=True (the default)
means it picks up from the last successfully-saved leg's checkpoint rather than
restarting from scratch. Caps retries and logs clearly either way, so an
unattended run either finishes or fails loudly with a clear reason after
genuinely exhausting its retries, rather than silently stopping or silently
looping forever.

Also sets a hard address-space limit (resource.setrlimit) well above the
measured per-leg peak but well below total system RAM, as a safety net: if
something unexpectedly balloons (e.g. a leg somehow needs much more than
calibrated), this process gets a clean MemoryError to log and retry from,
instead of the system-wide OOM-kill that crashed the whole desktop before.
"""
from __future__ import annotations

import resource
import sys
import time
import traceback

# 10GB: ~3.4x the measured ~2.9GB per-leg peak, well under this machine's ~15GB
# total RAM (leaving room for the OS, the dashboard if it's running, etc.).
resource.setrlimit(resource.RLIMIT_AS, (10 * 1024**3, 10 * 1024**3))

from market_sim import config as cfgmod  # noqa: E402
from market_sim.runner import run_scenario_in_legs  # noqa: E402

OUT_DIR = "runs/user_5000agents_300000s"
LEG_SECONDS = 1500.0
N_LEGS = 200  # 200 * 1500 = 300,000s total
MAX_ATTEMPTS = 8
RETRY_DELAY_SECONDS = 10

LOG_PATH = "runs/user_5000agents_300000s.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def build_cfg():
    cfg = cfgmod.load_config_dict("scenarios/realistic_mix.toml")
    for g in cfg["population"]["groups"]:
        g["count"] = int(g["count"] * 10)  # 500 -> 5,000, same mix/proportions
    return cfg


def on_leg_saved(leg):
    log(f"leg {leg.index}/{N_LEGS} saved -> {leg.out_dir} | trades={leg.meta['n_trades']:,} order_events={leg.meta['n_order_events']:,} abs_sim_time={leg.meta['actual_session_seconds']:,.0f}s")


def main() -> int:
    cfg = build_cfg()
    n_agents = sum(g["count"] for g in cfg["population"]["groups"])
    log(f"starting: {n_agents} agents, {LEG_SECONDS:.0f}s/leg x {N_LEGS} legs = {LEG_SECONDS * N_LEGS:,.0f}s total simulated -> {OUT_DIR}")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            legs = run_scenario_in_legs(
                cfg,
                LEG_SECONDS,
                N_LEGS,
                OUT_DIR,
                scenario_name="realistic_mix_x10_300000s",
                on_leg_saved=on_leg_saved,
                resume=True,
            )
            total_trades = sum(l.meta["n_trades"] for l in legs)
            log(f"DONE: {len(legs)} legs, {total_trades:,} total trades, {legs[-1].meta['actual_session_seconds']:,.0f}s total simulated time.")
            return 0
        except MemoryError:
            log(f"attempt {attempt}/{MAX_ATTEMPTS}: hit the {10}GB safety cap (MemoryError) -- this shouldn't happen at the calibrated scale; retrying in {RETRY_DELAY_SECONDS}s (resume picks up from the last saved leg).")
        except Exception:  # noqa: BLE001
            log(f"attempt {attempt}/{MAX_ATTEMPTS}: failed with an exception:\n{traceback.format_exc()}retrying in {RETRY_DELAY_SECONDS}s (resume picks up from the last saved leg).")
        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_DELAY_SECONDS)

    log(f"FAILED: exhausted {MAX_ATTEMPTS} attempts without finishing. Whatever legs completed are saved under {OUT_DIR} -- re-run this script to keep trying (it will resume, not restart).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
