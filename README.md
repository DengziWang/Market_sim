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

### Human-fraction sweep

`scenarios/human_sweep/human_{50,40,30,20,15,10,5,2_5,1,0}pct.toml` -- 10 scenarios
(2026-09-12, at the user's request), each 1,000 agents / 10,000s simulated / seed 1,
differing *only* in what fraction of the population is `human` (the `0pct` point
keeps the `human` group in the config with `count = 0`, i.e. zero of those agents
actually get built, rather than omitting the group -- verified this runs cleanly,
same as any other count). Every non-human bot type/variant is rescaled to keep its
relative share of the non-human population exactly what it is in `realistic_mix.toml`
(the base template they're generated from), so the sweep isolates "how much of the
market is human" as the one variable -- not a side effect of some other bot type's
headcount also shifting. Same seed across every point deliberately: this is a
composition sweep, not a seed sweep (add multiple seeds per point yourself, e.g. via
`build_plan_from_compositions`, for real confidence intervals per spec 6a's
20-50-seeds recommendation -- not done here since only one seed per point was
asked for). All 10 are also selectable from the dashboard's "Load a base scenario"
dropdown (`SCENARIOS`, prefixed "Human sweep:").

Regenerate (e.g. after editing `realistic_mix.toml`'s non-human composition) via
the generator script's logic in git history (2026-09-12) -- not a permanent CLI
tool, since a sweep's shape (which fractions, what varies) is itself something to
decide per-question rather than a fixed pipeline.

At 1,000 agents / 10,000s, `_estimate_run_cost` puts each point at ~4.1GB peak RAM
if run as one session -- above this project's dashboard's 15%-of-RAM auto-split
threshold on a 16GB machine, so running one from the dashboard will automatically
split it into legs (see "Splitting a long run into legs" above) rather than risking
the memory problem that feature exists to avoid.

## Dashboard (local, interactive)

`app/dashboard.py` is a local Streamlit UI over the same `market_sim`/`market_health`
packages above -- no separate backend, nothing leaves your machine. It gives you:

- Every field of the scenario config as widgets: session length/seed, the asset
  ("stock") list, venues, fees/latency/mempool/liquidation, and the full bot
  population as an editable table (bot type, variant, count, cash range, leverage,
  and each variant's own JSON params) -- plus a raw-TOML tab as a full-coverage
  escape hatch for anything the structured widgets don't have a dedicated field for.
- **Candlestick price+volume charts**, one per asset, with adaptive zoom: drag a box
  on the chart (or set exact start/end numerically) to zoom into a region -- candle
  width recomputes for whatever time span is in view (`span / target candle count`,
  ~100-200 candles visible at any zoom level by default), so narrowing the view
  produces more, thinner candles covering that span, down to genuine single-trade
  resolution once the window gets that narrow. Best bid/ask and the fundamental-value
  reference overlay as thin lines when snapshots are exported. See "Multi-leg
  overview" below for viewing an entire `run_scenario_in_legs` sequence this way too.
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

### Splitting a long run into legs

For a run long/large enough that holding its full history in memory at once is the
problem (see "Memory: the actual scaling limit" above), `market_sim.run_scenario_in_legs`
runs a scenario as several shorter sessions instead of one huge one, saving each to
its own directory as it finishes and discarding it from memory before starting the
next -- built on checkpoint/resume above, not a new engine capability.

```python
import market_sim as ms

cfg = ms.config.load_config_dict("scenarios/realistic_mix.toml")
legs = ms.run_scenario_in_legs(cfg, leg_seconds=1800.0, n_legs=12, out_dir="runs/realistic_mix_legs")
# -> runs/realistic_mix_legs/leg_0001/, leg_0002/, ..., leg_0012/, each a normal
#    export_run() directory (trades.parquet, ..., checkpoint.bin)

combined = ms.load_run_legs("runs/realistic_mix_legs")   # full history, for analysis
legs = ms.discover_legs("runs/realistic_mix_legs")        # just the directory list
```

Or from the command line: `python scripts/run_in_legs.py scenarios/realistic_mix.toml --leg-seconds 1800 --n-legs 12 --out-dir runs/realistic_mix_legs`.

**The dashboard does this automatically** -- clicking "Run simulation" on a config
estimated at or above ~15% of the machine's total RAM (well under the ~35% line that
used to require an explicit "run it anyway" confirmation) transparently runs it as an
auto-sized leg sequence instead of asking you to reach for a different function: leg
length is picked so each leg's own estimate lands around ~8% of total RAM, saved to
`runs/dashboard_legs/<timestamp>/`. The existing "current simulated second" progress
display keeps working seamlessly across leg boundaries (`ProgressHandle` reports
*absolute* simulated time, so a leg continuation's first reading already starts where
the previous leg left off) with a "leg N/M" counter alongside it. When it finishes,
the last leg loads into the Results tab immediately, which shows the whole run's
overview chart automatically since it's a multi-leg run (see "Candlestick chart +
multi-leg overview" below) -- no extra click, no separate tab to switch to.
`min_trades` being set disables auto-split (a trade-count floor
across an unknown final split doesn't have a sensible meaning) and falls back to the
manual single-run confirmation instead.

Measured (see project memory): across 8 legs of a real scenario, peak process memory
stayed flat (~200MB) rather than growing with leg count -- each leg's large result is
exported and dropped (Python reference-counting frees it immediately) before the next
one starts, so only ~one leg's data plus the small checkpoint (a few MB, bounded by
population size, not session length) exists in memory at a time. `load_run_legs`
loads everything back for analysis, same as loading any single large saved run would
-- appropriate once you're ready to look at the combined result, not during the run.

Only leg 1 runs `min_trades`/`max_session_seconds`-free at exactly `leg_seconds` --
those two `[run]` fields are dropped from the input config if set (a warning is
raised) since open-ended per-leg extension would defeat having a predictable,
bounded-size chunk; add more legs instead if you need a trade-count floor across the
whole sequence.

**Bonus finding**: only leg 1 shows the cold-start burst every fresh simulation has
(every agent's very first wake is jittered into the first simulated second, so
essentially the whole population acts at once against a still-empty order book --
measured on a real run: ~9-10x the steady-state trade rate in the first second, back
to normal within ~5s). Legs 2+ are genuine continuations (checkpoint/resume, not
independent restarts), so they pick up an already-warmed, already-trading market and
don't reproduce it -- verified directly: a continued leg's first-second trade rate
sits within its own normal range, no spike. If you're analyzing a `run_scenario_in_legs`
sequence and want to exclude the cold start entirely, drop leg 1's data specifically
(or start your analysis window a few seconds into it) rather than treating every leg
the same way.

### Candlestick chart + multi-leg overview (one chart, one tab)

The dashboard's price chart (Results tab) is a candlestick chart with adaptive,
time-window-based zoom (`market_sim.candles`): a candle covers a slice of simulated
time, not a fixed count of trades, so its volume bar means what a volume bar
normally means -- how much happened in that slice -- rather than being
near-constant across candles the way trade-count-based bucketing would be (a real
concern raised and fixed during design: with trade-count buckets, every candle
holds the same number of trades almost by construction, so the volume chart loses
the "which period was actually busy" signal). A "Chart type" toggle switches
between candlestick and a plain line -- the line traces each bucket's `mean`
price, a genuine time-span average of every trade in that bucket (not a single
representative trade like `open`/`close`, not a `high`/`low` extremum, and not a
volume-weighted average either -- a plain arithmetic mean, matching how the line
chart looked before the candlestick rebuild). Both modes share the same
bucketing, so neither ever plots raw per-trade points -- only the current view's
target-candle-count-bounded buckets.

Bucket width isn't a fixed global grid -- it's recomputed for whatever time range
is currently in view: `width = view_span / target_candle_count` (default target
100, so 100-200 candles visible at any zoom level). Dragging a box on the chart (or
typing exact start/end values -- the always-reliable alternative, since Plotly's
own scroll/toolbar zoom isn't observable server-side, only a drag-*selection* is)
narrows the view, which shrinks the bucket width and produces more, thinner
candles covering that narrower span -- down to genuine single-trade resolution
once the window is narrow enough. Empty buckets forward-fill from the previous
close (flat candle, zero volume) rather than leaving a gap, same as a real
exchange chart showing a quiet period.

**Multi-leg runs use the same chart, in the same Results tab** -- there's no
separate "overview" tab to switch to. Whenever the currently loaded run is one leg
of a `run_scenario_in_legs` sequence, the chart defaults to showing the *whole* run
(e.g. all 300,000s of a 200-leg run) composed from small, cached per-leg OHLC
summaries, not every leg's full trade data at once: each leg contributes a 300-
candle summary computed from just the `time`/`price`/`qty` columns of that leg's
`trades.parquet` (`market_sim.load_trades_columns`, not the full table), and these
compose into one series via `market_sim.candles.resample_candles`, the same OHLC
math real charting software uses to build 1-hour candles from 1-minute ones.
Summarizing a many-legs run takes a few/several seconds the first time per session
(parallelized across legs with a thread pool; measured ~15-20s for 200 legs) --
cached per leg afterward, so re-viewing or re-zooming is instant. The chart-type
toggle and adaptive zoom both work identically in this composed mode.

Once you've zoomed to within a single leg's span, a "Load for full detail" button
switches that chart specifically into single-leg mode: that leg's actual trade
data, genuine single-trade-resolution zoom, and (see below) whole-run trade search
and market-health metrics. A "Back to whole-run overview" button returns to the
composed view without re-reading anything (the per-leg summaries are already
cached). Loading any *other* run (a fresh run, a different saved run, a different
leg picked from the sidebar) resets back to whole-run-overview-by-default for
whatever multi-leg group it belongs to, if any.

**Market-health metrics across a multi-leg run**: rather than one merged whole-run
composite (which runs into real boundary-handling issues for some metrics --
autocorrelation-based efficiency measures and resilience/recovery timing right at
a leg boundary need special treatment to stay exactly correct), each leg keeps its
own already-correct `market_health.compute_run_metrics` output, shown as **one
line chart per metric across every leg** (grouped by report dimension, in
expanders) -- a genuine list of sessions, not one number that hides how it
changes over the run's whole length.

This is real work, not the ~15-20s/200-legs cost the candle overview has:
`compute_run_metrics` needs every export file (not just `trades.parquet`),
measured on this machine's real 200-leg run at ~2s to load + ~11-13s to compute
*per leg*. `run_scenario_in_legs` pays this cost automatically now
(`compute_market_health=True`, its default) -- right after exporting each leg,
while its data is still in memory (no reload needed), computing and persisting
that leg's metrics to `<leg_dir>/market_health.json` (`market_health.save_metrics`
/ `load_metrics`). This is a deliberate tradeoff: it genuinely adds that same
~11-13s/leg to the run itself, paid once upfront instead of never or lazily
later -- pass `compute_market_health=False` to skip it for a scan where
market-health doesn't matter. The dashboard checks for this file first and only
falls back to the actually-heavy recompute-from-scratch path (also persisting the
result once it does, so it's never paid twice) for an older run made before this
existed, or one made with the flag off -- shown as an explicit "Compute
market-health for the remaining N legs" button with an honest time estimate,
parallelized (`ThreadPoolExecutor`, 2 workers -- deliberately conservative: a
first attempt at 4 measured RSS climbing to 4.3GB in 15s on this machine, real
evidence of OOM risk given this project's actual crash history, not just a CPU-
contention guess). Loading a leg for full detail also checks/persists that one
leg's metrics the same way.

**Trade search across an entire multi-leg run**: a checkbox on the Trade search
table extends it from "this one loaded leg" to "every leg" -- each leg's
`trades.parquet` is read and filtered in turn (not just the four candle-summary
columns; search needs bot-type/agent-id/side too), keeping only the matching rows,
never holding more than a small batch of legs' full trade tables in memory at
once. Measured ~0.5s/leg to read a full `trades.parquet` (~100s serially for 200
legs, well under the market-health cost above) -- and since an unfiltered or very
permissive search would otherwise match literally every trade in every leg
(reintroducing the exact unbounded-memory problem splitting into legs exists to
avoid), the scan stops as soon as 5,000 matches are found, same as the existing
single-leg display cap, with a note that the result may be incomplete so you know
to narrow the filters rather than trusting a partial count silently.

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

**Composition sources (2026-09-13)**: the original weights (headcount percentages)
were this project's own estimates, not sourced from real data -- flagged and
re-derived at the user's request. Real, cited findings and what changed as a
result:

- Retail's share of crypto trading *volume* has shrunk hard even as headcount stays
  dominant: on Coinbase, retail volume share fell from ~80% (2018) to ~18% (Q1 2024)
  ([source](https://finance.yahoo.com/news/does-stock-crypto-investor-divide-signal-121645815.html)),
  and institutions were >70% of volume industry-wide in H1 2026
  ([BitKE / Wintermute report](https://bitcoinke.io/2026/07/the-wintermute-h1-2026-report/)).
  Measuring this project's *original* config directly (not assumed) found `noise` +
  `human`'s ~58% combined headcount contributing only ~2.1% of volume -- far below
  even the most institution-dominated real venue. Fix: sped up and enlarged `noise`
  (`wake_interval_secs` 8.0 -> 2.0, `notional_range` [50,500] -> [75,750]) --
  `human` deliberately left untouched, since its low activity models a specific,
  real behavioral trait (infrequent, irregular checking), not a bug. Re-measured
  after the change: retail volume share landed at ~19.3%, right in the real range.
- Bitwise's SEC filing found ~95% of volume on *unregulated* crypto exchanges was
  wash-traded, versus a real, legitimate market on ~10 major regulated exchanges
  ([Cointelegraph](https://cointelegraph.com/news/bitwise-tells-us-sec-that-95-of-volume-on-unregulated-crypto-exchanges-is-suspect)).
  This didn't change anything here -- `wash_trading` was already removed from this
  template entirely (see git history, "isn't wash trading illegal" discussion) --
  but it's relevant if you ever build a variant meant to model a low-quality venue
  specifically (closer to `stress_wash_mev.toml`'s territory).
- MEV, measured directly rather than assumed: the original config's `mev` (1.2%
  headcount, `wake_interval_secs=0.02`) contributed ~26% of total volume --
  disproportionate next to any real analog. Real Ethereum MEV data (~89% of blocks
  via MEV-Boost, but MEV is only ~26% of *validator revenue*, and sandwich-attack
  revenue specifically has shrunk to ~$740K/30 days after mitigations --
  [survey via Binance Square](https://www.binance.com/en/square/post/13660663730586))
  doesn't map cleanly onto this project's continuous-limit-order-book structure
  anyway (that's Ethereum-mempool MEV, a different market structure entirely), so
  there's no precise target number to hit -- just "not a quarter of all volume from
  1.2% of agents." Fix: `wake_interval_secs` 0.02 -> 0.1, added
  `size_frac_of_victim` 0.5 (the prior unnamed default) -> 0.25. Re-measured: mev's
  volume share dropped to ~2.4%.
- `market_maker` (measured ~47% -> ~40% of volume from 8% headcount after the above
  changes shifted the mix) and `arbitrage` (~21% -> ~33%) weren't touched --
  broadly consistent with "small headcount, majority of volume" for market makers,
  but no precise real citation was found for either specific number, so these are
  still this project's own judgment, not sourced.

All percentages above are from actually running the scenario and measuring
maker+taker notional volume by bot type (not assumed) -- see git history for the
exact measurement script if you want to re-run this check after any future
compositional change.

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

### Engine micro-optimizations (verified behavior-preserving)

`sim_core/examples/bench.rs` is a pure-Rust benchmark (no PyO3/Python overhead)
built to safely speed up the event loop itself, added when optimizing without
being allowed to change simulation behavior at all -- run it directly:

```
cargo build --release --example bench -p sim_core
./target/release/examples/bench <n_agents_scale> <session_seconds>   # e.g. `2 1000` -> 758 agents
```

It prints `trades=`/`order_events=`/`snapshots=` counts alongside `elapsed=`/
`events/sec=`, which is the actual verification methodology used for every change
below: `git stash` the change, run the benchmark at a fixed seed/scale, `git stash
pop`, rebuild, run again, and `diff` the two outputs. Only `elapsed`/`events/sec`
are allowed to differ -- any change to the trade/order_event/snapshot counts means
the change altered behavior, not just speed, and gets reverted. Two changes passed
this bar:

1. **`Action::Cancel` carries its `AssetId` directly.** Every bot that cancels one
   of its own resting orders already knows which asset that order is on (each
   resting order is single-asset), so `Action::Cancel(OrderId, AssetId)` lets
   `Simulation::apply_action` cancel directly. Previously it only carried the
   `OrderId` and `Simulation` had to find the asset itself via `order_asset_hint`,
   an O(n_assets) scan across every asset's order-book `HashMap` on *every single
   cancel* -- cheap with a handful of assets, but pure waste since the caller
   already had the answer.
2. **FxHash instead of SipHash for small dense-integer-keyed maps.** `std::HashMap`'s
   default hasher (SipHash) is deliberately DoS-resistant, which costs real speed
   for the per-asset and per-`(venue, asset)` maps in `market.rs`
   (`books`/`mark_price_ema`/`fundamental_value`/`sentiment`/`last_trade_price`) and
   `agent.rs` (`AgentCommon.inventory`) -- typically 1-10 distinct keys, looked up
   millions of times a run, never keyed by untrusted input. `types::FxHashMap`
   swaps in `rustc-hash`'s `FxHasher` (the same one used throughout the Rust
   compiler for this exact reason) as a drop-in hasher-only change: identical
   `HashMap` API and semantics, no data-structure or logic change.

Measured at scale=2/session=1000s (758 agents), `trades=35434 order_events=791278
snapshots=113040` identical before and after both changes:

| | elapsed | events/sec |
|---|---|---|
| before | 0.374s | 2,208,026 |
| after | 0.327s | 2,527,424 |

~12.6% faster wall time, ~14.5% higher throughput, for zero change to simulated
trading behavior (same trades, same prices, same order events, byte-for-byte).
Verified with the full Rust test suite (`cargo test`, 24 tests including the
checkpoint CBOR round-trip tests) and the full Python suite (`pytest tests_py/`)
passing unchanged after rebuilding the `sim_pyo3` wheel.

Not pursued this round (identified but lower confidence/payoff, left for a future
pass if more speed is needed): `OrderBook.orders: HashMap<OrderId, Order>` --
order IDs are large monotonically-increasing `u64`s rather than small dense keys,
so FxHash would still help some, but a `Vec`-based approach would need more design
work given orders get removed (creating gaps); `Simulation.pending_delayed:
HashMap<u64, (AgentId, NewOrderRequest)>`, left as `std::HashMap` since it wasn't
confirmed hot.

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
- **Avellaneda-Stoikov's reservation price is now clamped** (2026-09-12), closing a
  second, later-discovered gap in the same bot: `half_spread` already had a
  defensive clamp (the previous point), but the reservation price it's centered on
  didn't -- `inventory_norm * gamma * sigma^2 * horizon_secs` (`sigma` here is this
  bot's own rolling realized-volatility estimate from recent observed returns, not
  the asset's configured `fundamental_vol`) is unbounded, so a momentary spike in
  that estimate could push the reservation many multiples of mid away from the real
  market price -- a real, wildly mispriced quote that another bot immediately trades
  on, feeding an even larger realized return back into the estimate on this bot's
  *next* wake. Found by inspecting a real 5,000-agent/300,000s run at the user's
  prompting ("why is BTC stable but the others sometimes near-zero or way up") --
  DOGE's traded price jumped 17.6x in a single trade this way, repeatedly, on the
  thinner/higher-vol assets specifically (the latent fundamental value itself stayed
  perfectly well-behaved throughout, confirming this was a bot-logic bug, not a
  reference-price or data issue). Fixed by clamping the reservation to within 10% of
  mid, same bound already used for the spread. Regression test:
  `sim_core/tests/market_maker_test.rs` drives the bot directly with a manually
  crafted extreme volatility window (deterministic, not hoping to reproduce the rare
  stochastic event in a full run) and asserts the resulting quote stays bounded;
  verified via `git stash` that it fails without the fix and passes with it.
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
