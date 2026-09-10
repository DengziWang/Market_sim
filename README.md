# crypto-bot-market-sim

A research simulation for one question: **can a crypto market that's dominated by
algorithmic bots function in a healthy way**, and how does that depend on the mix of
bot strategies present? See `docs/marktgesundheit_bericht.md` for the market-health
framework this project measures against, and `scenarios/` for three example
compositions (a realistic mix, a pure-bot market, and an adversarial stress test).

## Architecture

- **`sim_core/`** (Rust): the event-driven engine -- multi-asset/multi-venue limit
  order book with strict price-time priority, a min-heap event queue (agents only
  wake on their own timer or a relevant event, never a fixed global tick), all bot
  decision logic, the simplified leveraged-liquidation model, and a deterministic
  ChaCha8-seeded PRNG per agent. Pure Rust, `cargo test`-able with no Python
  involved. See `sim_core/src/`.
- **`sim_pyo3/`**: a thin PyO3 wrapper crate (built by `maturin`) that runs a
  `sim_core::Simulation` from Python and hands the result back as plain columnar
  `dict[str, list]` data -- no parquet/arrow dependency on the Rust side at all.
- **`python/market_sim/`**: orchestration -- config read/write, single-run helpers
  (`run_scenario_file`, `simulate_and_plot`), the parquet/JSON export (spec section
  7), plotting, and the batch experiment runner (spec section 6).
- **`python/market_health/`**: the analysis module (spec section 9) -- computes every
  metric from `docs/marktgesundheit_bericht.md` directly from the exported parquet
  files, never by re-running the simulation, plus the composite market-health score
  and batch composition-comparison plots.
- **`scenarios/`**: three example scenario configs (spec section 10).
- **`scripts/benchmark.py`**: the mandatory benchmark-first script (spec 6a) -- run
  this before any large batch.
- **`tests_py/`**: Python tests for `market_health` against synthetic data with a
  known expected result.

## Setup

Requires a Rust toolchain and Python 3.9+. If Python has no `pip`/`venv` available
(true in the environment this was built in -- Debian/Ubuntu images sometimes ship
`python3-minimal` without `ensurepip`), bootstrap `pip` first:

```bash
# Rust toolchain (skip if you already have one)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
source "$HOME/.cargo/env"

# pip, if `python3 -m pip --version` fails
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --user
export PATH="$HOME/.local/bin:$PATH"   # put this in your shell rc too

python3 -m pip install --user maturin
```

Then build and install the extension. `maturin develop` needs a virtualenv or conda
env; without one, build a wheel and install it directly:

```bash
cd crypto-bot-market-sim

# with a venv (recommended if you have one):
#   python3 -m venv .venv && source .venv/bin/activate && pip install -e .
#   maturin develop --release

# without a venv (what this repo was actually built/tested with):
maturin build --release
python3 -m pip install --user --force-reinstall --no-deps target/wheels/market_sim-*.whl
python3 -m pip install --user pandas pyarrow numpy matplotlib scipy pyyaml toml tqdm pytest
```

Re-run `maturin build --release` + reinstall after *any* change under `sim_core/`,
`sim_pyo3/`, **or** `python/` -- the wheel is a snapshot, not a live link (this is a
consequence of not using `maturin develop`/editable installs; switch to a venv if
you're iterating a lot).

### Verify the install

```bash
cargo test -p sim_core                      # Rust: matching engine + full-population smoke test
python3 -m pytest tests_py/                  # Python: market_health against synthetic data
python3 -c "from market_sim import validate_config; print(validate_config(open('scenarios/realistic_mix.toml').read()))"
```

## Quickstart

```python
from market_sim import run_scenario_file

run = run_scenario_file("scenarios/realistic_mix.toml")
print(run.meta)                              # {'actual_session_seconds': ..., 'n_trades': ..., ...}
run.export("runs/realistic_mix/seed_1")       # writes trades.parquet, order_events.parquet, ...
fig = run.plot()                              # price/volume per asset (spec section 8)
fig.savefig("run.png")
```

or from the CLI for a one-shot look:

```bash
python3 -c "from market_sim import simulate_and_plot; simulate_and_plot('scenarios/realistic_mix.toml', save_path='run.png')"
```

### Batch experiments (spec section 6/6a)

**Always run `scripts/benchmark.py` on your actual machine before a big batch** --
see the Performance section below for why.

```python
from market_sim import build_plan_from_compositions, run_experiment

plan = build_plan_from_compositions(
    base_config_path="scenarios/realistic_mix.toml",
    compositions={
        "realistic_mix": {},
        "pure_bot_market": {"population": __import__("market_sim.config", fromlist=["load_config_dict"]).load_config_dict("scenarios/pure_bot_market.toml")["population"]},
    },
    seeds=range(20, 40),        # 20 seeds per composition (spec 6a: 20-50 default)
    out_dir="runs/experiment_1",
)
summary = run_experiment(plan, n_workers=5)   # 5 workers: spec 6a's "leave one core free" default
print(summary)
```

Then analyze the whole batch:

```python
from market_health import load_and_score_runs, compare_compositions, plot_composition_comparison

metrics = load_and_score_runs(plan.all_run_dirs())
comparison = compare_compositions(metrics)     # mean +/- 95% CI per composition, per dimension
plot_composition_comparison(comparison).savefig("comparison.png")
```

A killed/interrupted `run_experiment` call can just be re-run with the same `plan` --
completed cells (those with a `run_metadata.json`) are skipped (`resume=True` is the
default).

### Crash-/stress-test experiments

Spec 6a treats these as a distinct experiment *type* (short focused window, more
repeats) rather than a special code path -- `build_stress_test_plan` is a thin
wrapper around the same mechanism, and `scenarios/stress_wash_mev.toml` is a ready-
made short-session, wash/MEV-heavy example to point it at.

## Dashboard (local, interactive)

`app/dashboard.py` is a local Streamlit UI over the same `market_sim`/`market_health`
packages above -- no separate backend, nothing leaves your machine. It gives you:

- Every field of the scenario config as widgets: session length/seed, the asset
  ("stock") list, venues, fees/latency/mempool/liquidation, and the full bot
  population as an editable table (bot type, variant, count, cash range, leverage,
  and each variant's own JSON params) -- plus a raw-TOML tab as a full-coverage
  escape hatch for anything the structured widgets don't have a dedicated field for.
- Zoomable/pannable per-asset price+volume charts (Plotly: drag to zoom, double-click
  to reset, scroll to zoom) with best bid/ask and the fundamental-value reference
  overlaid when snapshots are exported.
- The full `market_health` metric set for the run.
- A searchable/filterable trade table (asset, bot type, agent id, time range, min
  quantity, free-text) with CSV export of the filtered rows.
- **Run in the background + stop early**: clicking Run starts the simulation on a
  background thread (the Rust engine supports cooperative cancellation --
  `Simulation::run_cancellable`/PyO3 `CancelToken`) so the UI stays responsive; a
  **Stop simulation** button appears while it's running. Stopping doesn't discard
  anything -- the partial result is auto-exported to `runs/dashboard_stopped/` the
  moment it stops, and immediately usable in the Results tab.
- **Save / load runs**: the Results tab has a Save button (writes the spec-section-7
  export set to a directory you name), and the sidebar's "Load a saved run" expander
  reloads any previously exported run (from the dashboard, `run.export(...)`, or a
  batch experiment) back into the Results tab without re-running anything.
- **Continue a simulation from where it stopped** (not a restart -- see below): the
  Results tab's "Continue this simulation" section is available on any run (fresh,
  stopped-early, or loaded from disk) that has a checkpoint.

```bash
python3 -m pip install --user streamlit plotly   # once
cd crypto-bot-market-sim
streamlit run app/dashboard.py
```

Opens at `http://localhost:8501` in your browser. Pick a base scenario in the
sidebar, edit whatever you want across the tabs, click **Run simulation**, then look
at the **Results** tab.

### Checkpoint / resume

Every `run_simulation`/`run_simulation_cancellable` call (and the dashboard's Run
button, which just wraps the cancellable one) returns a `checkpoint: bytes` alongside
the usual `trades`/`agent_states`/etc. -- the *entire* internal engine state at the
moment the run stopped: every order book, every agent's cash/inventory/indicator
state and RNG stream, the market-level RNG stream, and every event still pending
(next agent wakes, the next snapshot/liquidation tick, ...). `market_sim.continue_run`
(Python) / `Simulation::resume` (Rust) load that back and keep simulating for however
many more seconds you ask for -- genuinely picking up mid-session, not restarting from
a summary or re-seeding anything. This is what lets a session be extended after the
fact instead of needing to guess the right length upfront.

```python
import market_sim as ms

run = ms.run_scenario_file("scenarios/realistic_mix.toml")
run.export("runs/leg1")                        # writes checkpoint.bin alongside the usual parquet files

# ... later, maybe in a different process: reload and keep going.
loaded = ms.load_run("runs/leg1")               # checkpoint.bin -> loaded["checkpoint"] if present
more = ms.continue_run(loaded["checkpoint"], additional_seconds=1800.0)
more.export("runs/leg2")                        # covers only the new segment -- see below
```

A couple of things worth knowing:

- The checkpoint excludes accumulated trade/order-event/snapshot history (by design --
  otherwise every checkpoint would grow with the whole run instead of staying
  proportional to live state). `continue_run`/`continue_simulation` therefore return
  only the *newly*-simulated segment; pass `previous_result=` (Python) or use
  `combine_results` yourself to get one continuous combined table back. The dashboard
  does this automatically -- continuing a run there always shows the full history.
- The wire format is CBOR (self-describing, unlike bincode -- needed because bot
  params are a free-form dynamic TOML value and the order book's maps use tuple keys,
  neither of which bincode's non-self-describing format can round-trip).
- Works on a cancelled/stopped-early run too, not just a completed one -- "Stop
  simulation" already auto-saves, and that saved run's checkpoint can be continued
  the same way as any other.

## Configuration format

Everything is a TOML file matching `sim_core::config::ScenarioConfig` (see
`sim_core/src/config.rs` for the authoritative schema/defaults). Top level:

```toml
[run]            # seed, session_seconds, min_trades, max_session_seconds, extension_step_seconds
[[assets]]        # symbol, tick_size, initial_price, fundamental_vol, fundamental_theta (repeat per asset)
[[venues]]        # name, inter_venue_latency_secs (repeat per venue; omit for a single "primary" venue)
[market]          # snapshot_interval_secs, snapshot_depth_levels, mark_price_ema_halflife_secs
[fees]            # default_maker_fee/default_taker_fee, [fees.by_bot_type]
[latency]         # default_latency_secs, [latency.by_bot_type]
[mempool]         # enabled, reveal_delay_secs, large_order_notional_threshold
[liquidation]     # maintenance_margin_buffer, sweep_interval_secs, staggered_slices, full_liquidation_fraction
[export]          # out_dir, write_snapshots
[[population.groups]]   # bot_type, variant, count, initial_cash_range, leveraged_fraction,
                         # leverage_range, [population.groups.params] (variant-specific), notes
```

`bot_type` is one of `market_maker`, `arbitrage`, `momentum`, `mean_reversion`,
`grid`, `dca`, `liquidation_trader`, `mev`, `wash_trading`, `noise`, `human` -- see
`sim_core/src/bots/` for each type's variants and their `params` keys/defaults
(every module has them in a doc comment at the top of its `State::from_params`).

**Bot-type weighting**: `scenarios/realistic_mix.toml` is the finalized realistic-
composition template (headcount/volume anchors documented in the comments at the top
of the file, including `human` bots -- see below); `pure_bot_market.toml` and
`stress_wash_mev.toml` are deliberately extreme by design (not meant to be
"realistic"), see their own header comments.

### `human` bots (not spec-mandated)

Added on request to model actual behavioral-finance biases -- disposition effect
(hold losers, sell winners early), convex FOMO/panic response with a dead zone
(not linear trend-following), "hot hand" position sizing off the bot's own recent
realized P&L, irregular bursty attention (long jittery background check-ins, much
shorter ones right after noticing a big move), and round-number trade sizes --
rather than a rational or mechanical strategy. `noise` remains the spec's
deliberately-simple memoryless baseline; `human` is the opposite: it remembers its
own trades and reacts the way research says people actually do, not randomly.

A single variant (`retail`) carries every bias listed above at once -- this used to
be split across two variants (`disposition_chaser` reacting only to price trend,
`sentiment_herder` only to mood), merged into one bot since a real retail trader
isn't cleanly one or the other. Its directional signal blends both: a recent-price-
return component (time-normalized, so an agent that happened to sleep longer doesn't
see an artificially bigger "move" for the same underlying drift) and a
`Market::sentiment` component (a latent, price-independent process -- same
Ornstein-Uhlenbeck shape as the fundamental-value process, but mean-reverting to
neutral mood rather than to a price -- read with individual perception noise, not
identically by everyone). Each is normalized by its own threshold before blending
(`price_signal_threshold`/`sentiment_signal_threshold`, default weighted 50/50 via
`price_weight`), so "the price moved" and "everyone's talking about it" reinforce
each other the way they would for an actual person instead of being handled as
mutually exclusive triggers. Capped fairly conservatively by default (see
`sim_core/src/bots/human.rs`'s `from_params`) after testing showed a large,
synchronized population of these bots reading a shared signal can otherwise
overwhelm book depth on thin assets -- tune `fomo_exponent`, `max_cash_frac_per_trade`
up if you deliberately want more extreme herding/bubble dynamics to study.

Trades noticeably less often than every other bot type by design: real retail
traders don't watch a chart continuously. Check-ins are spaced 3-40 min apart on
average (`background_wake_min_secs`/`background_wake_max_secs`) rather than the
seconds-to-low-minutes every mechanical bot uses, and even a qualifying signal only
triggers an actual trade `p_act_given_signal` of the time (default 0.35) -- noticing
something isn't the same as acting on it. In practice this comes out to roughly one
trade per human agent per half-hour session; tune both up if you want a more active
population.

The dashboard's Results tab has a **"Human involvement by asset"** table breaking
down, per asset, how many trades had a human bot on *both* sides
(`fully_human`), on *exactly one* side (`partially_human`), or on neither.

## Performance

Measured with `scripts/benchmark.py --release` on the machine this was built on --
its hostname identifies it as the actual Acer Nitro AN517-51 target hardware from
spec section 11 (`Linux-6.8.0-138-generic-x86_64`, 12 logical cores visible via
`nproc`), so these numbers should be directly usable for planning. Still: laptops
throttle under sustained load in a way a short benchmark run won't catch, so if you
run a long batch and it's much slower than this predicts, that's the likely cause --
see the note at the end of this section.

| agents | simulated session | wall time | events/sec |
|---|---|---|---|
| 300  | 300s  | ~0.2s | ~1.6M/s |
| 3000 | 600s  | ~5.2s | ~1.25M/s |

That's roughly **30-50x the "several zehntausend events/sec" target** from a single
un-parallelized run, on this machine. Two things this does *not* yet include:

1. **Per-run parallelism (Rayon)**: not wired up. The spec's Rayon requirement is
   satisfied at the *batch* level (`market_sim.run_experiment`'s multiprocessing
   pool runs independent simulations across worker processes -- the actual
   embarrassingly-parallel axis, since a batch is many independent runs). A single
   simulation's event loop is inherently sequential (each event can depend on the
   book state left by the previous one), so intra-run Rayon parallelism would need
   splitting work per-asset at best, which wasn't necessary given the numbers above.
2. Numbers will be lower with `write_snapshots = true` at a tight
   `snapshot_interval_secs` on many assets, and with `mempool.enabled = true` (extra
   bookkeeping per large order) -- the benchmark script's default config exercises a
   representative but not maximal mix; adjust it or your scenario file if your
   composition is unusually snapshot- or mempool-heavy.

Given these numbers, the spec 4/6a agent-count defaults (1,000-3,000 for batch runs,
5,000-15,000 for confirmation runs) look comfortably achievable in terms of raw
**event throughput** on this hardware. **Memory is the actual binding constraint**,
not throughput -- see the next section, which matters more in practice: the numbers
above are how *fast* the engine computes events, not how much it costs to hold the
resulting trades/order_events/snapshots afterward, and holding all of it is what
exhausted this machine's RAM (see below). Still re-run `scripts/benchmark.py` before
a big batch if it's been a while (thermal state, other load on the machine, etc. all
matter), and per spec 6a/12: if a real batch run turns out much worse than this
predicts, reduce the defaults and confirm the change with the user rather than
deciding alone.

### Memory: the actual scaling limit

Every trade/order-event/snapshot produced during a run is held in memory for the
whole run, in triplicate -- Rust's own result buffers, a numpy-array-backed Python
dict (`sim_pyo3`'s return value), and (for the dashboard, `market_sim.export`, and
`market_health.compute_run_metrics`) a pandas DataFrame built from that dict. None of
this is streamed to disk incrementally; it all has to fit in RAM at once. Order
events dominate (every submit/cancel/fill/reject, not just trades -- typically
5-6x the trade count), and their count scales with `agents x session_seconds x
average wake frequency`, same as event throughput above -- but throughput measures
*speed*, not the *size* of what gets retained, which is the thing that actually
exhausted a real 16GB machine's RAM (not just the simulation process -- `systemd-oomd`
killed the browser, the window manager, and the terminal session too under the
resulting memory pressure) on a 5,000-agent, 300,000-simulated-second run: an order
of magnitude past this project's own documented scenario cap (`realistic_mix.toml`'s
`max_session_seconds = 21600.0`, "spec 6a upper bound").

Measured on this hardware (`scenarios/realistic_mix.toml` scaled to 2,000 agents /
3,000s simulated -- 1,015,194 trades, 6,130,334 order events): **~9s wall time,
~2.5GB peak RSS** for a full run through Rust -> numpy -> pandas (down from ~22s /
~4GB before an optimization pass that returns numeric columns as numpy arrays
instead of Python lists of boxed floats/ints -- a representation change only, same
values, see `sim_pyo3::result_to_pyobject`). Extrapolating linearly to the spec's own
stated upper range (15,000 agents x 21,600s, ~54x this measurement) implies **tens of
GB** -- not achievable on a 16GB machine regardless of how fast the event loop itself
is, with the current engine architecture (everything retained in memory, nothing
streamed to disk incrementally). If you need the full spec-described scale for real,
that requires a streaming/chunked export path (write parquet incrementally instead of
building one giant in-memory result first) -- a real architecture change with
tradeoffs (e.g. what stays available for live in-dashboard exploration vs. what's
only on disk afterward), not yet built; ask before assuming a direction there.

**In the meantime**: the dashboard estimates a run's likely peak memory/wall-time
before you click Run (calibrated from the measurement above) and requires an explicit
"I understand the risk" acknowledgement past a conservative threshold (currently
~35% of total system RAM) -- so an oversized config gets caught before it repeats the
crash above, not after. The same estimate/gate applies to "Continue this simulation".
For genuinely large runs, prefer fewer agents, a shorter session, `write_snapshots =
false`, and/or running via `market_sim`/`scripts/benchmark.py` directly (still fully
in-memory today, but at least without a browser tab and Streamlit's own overhead
alongside it) over the interactive dashboard.

## Known simplifications (read before treating results as findings)

Documented in more detail at each site in the code; the significant ones:

- **Triangular arbitrage** operates on 3 assets that should maintain a configured
  `peg_ratio`, not on dedicated A/B, B/C, A/C pair order books (the engine models
  each asset as a single spot book against a common numeraire). Real pair-based
  triangular arbitrage is a natural extension once pair books exist.
- **Wash-trading `self_ring`** is one agent crossing its own resting order (proven
  safe against accidentally trading a real counterparty -- see the comment in
  `sim_core/src/bots/wash_trading.rs`), not coordinated multi-wallet ring trading
  across distinct agent ids.
- **Grid bots** tear down and rebuild their whole ladder every `wake_interval`
  (like a market-maker requote) rather than replacing only the level that just filled.
- **Liquidation**: Phase-1 simplified margin model only (leverage + maintenance-
  margin buffer + EMA mark price + partial/staggered liquidation) -- no funding rate,
  funding-arbitrage bots, insurance fund, or auto-deleveraging (spec 5.7's explicit
  scope cut; the architecture -- `AgentCommon::margin`, `LiquidationExecStyle` --
  is meant to take a Phase-2 funding-rate extension without a core rewrite).
- **Cancel latency**: cancels apply immediately; only new-order submission is delayed
  by an agent's configured latency (and by the mempool model for large orders).
- **MEV bots' "mempool"**: a configurable notional threshold decides which orders are
  staged through the public pending-order queue at all; below it, orders land
  directly (real mempools don't filter by size, but modeling every order that way
  would make *every* agent latency-sensitive to *every* other agent's flow, which
  isn't the phenomenon spec 5.8 is asking to isolate).
- **Kyle's Lambda / order-flow-imbalance / VPIN** use the trade table's `taker_side`
  directly (this simulation actually knows it); `Amihud` and `variance_ratio` use a
  fixed time-bucket/resample rather than a tick-imbalance clock.
- **Order sizing is dollar-notional, not unit-based**, for every bot type (`order_notional`
  params, converted to units per-asset at population-build time; DCA and liquidation
  traders already sized this way naturally). Without this, the *same* configured
  size meant $3,000 of BTC and $0.0075 of DOGE -- comparable trade counts across
  assets but wildly skewed reported volume, and for market-makers specifically, thin
  enough resting depth on cheap assets to destabilize price. See git history
  (2026-09-10) for the debugging trail, which also found and fixed a real dimensional
  bug in the Avellaneda-Stoikov market-maker (mixing a fractional log-return
  volatility with an absolute-price spread term) and an unstable arbitrage sizing
  formula (dividing a dollar target by a live, potentially-crashing price).
- **Matching-time circuit breaker** (`market.max_price_move_frac`, default 15%): a
  single order's matching won't cross more than that fraction away from the best
  opposing price it started against, and a resting limit order more than 50x (or
  under 2%) of the latent fundamental value is rejected outright -- real exchanges
  have an equivalent (price bands / fat-finger protection). Without either of these,
  one bot's bug (or, legitimately, a large order hitting a thin book) could move a
  printed price by orders of magnitude in a single fill. `max_price_move_frac` is
  itself a research variable -- set it very large to study an unprotected market.

## Tests

```bash
cargo test -p sim_core          # matching engine (price-time priority, partial fills,
                                 # IOC/FOK/post-only edge cases) + a full-population
                                 # end-to-end smoke test across every bot type/variant
python3 -m pytest tests_py/     # market_health metrics against synthetic data with a
                                 # known expected result
```
