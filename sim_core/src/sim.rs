//! Orchestration: builds the agent population from a `ScenarioConfig`, drives the
//! event-driven main loop, applies fills/fees/latency/mempool staging, runs the
//! liquidation sweep, and produces a `SimResult`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rand::Rng;

use crate::agent::{Action, AgentCommon, LeverageCfg, LiquidationExecStyle, NewOrderRequest, StepResult};
use crate::bots::{self, arbitrage, dca, grid, human, liquidation_trader, market_maker, mean_reversion, mev, momentum, noise, wash_trading, BotStrategy, StepCtx};
use crate::config::ScenarioConfig;
use crate::event::{EventKind, EventQueue};
use crate::export::{AgentStateRow, OrderEventRow, SimResult, SnapshotRow, TradeRow};
use crate::market::Market;
use crate::order::{Fill, OrderEventKind};
use crate::params;
use crate::rng;
use crate::types::{AgentId, AssetId, FxHashMap, OrderId, OrderType, Side, SimTime, TimeInForce, VenueId};

const PRIMARY_VENUE: VenueId = 0;

/// Everything needed to resume a simulation exactly where it left off: order books,
/// every agent's cash/inventory/indicator state, both RNG streams (market-level and
/// per-agent), and the still-pending event queue (agent wakes, periodic ticks, etc.
/// scheduled but not yet due). `result` (accumulated trades/events/snapshots so far)
/// is deliberately excluded -- see `Simulation::resume`'s doc comment -- so a
/// checkpoint stays proportional to *state*, not to how long the run has already
/// gone on for.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Simulation {
    pub market: Market,
    pub agents: Vec<AgentCommon>,
    pub strategies: Vec<BotStrategy>,
    pub cfg: ScenarioConfig,
    events: EventQueue,
    market_rng: rand_chacha::ChaCha8Rng,
    now: SimTime,
    session_end: SimTime,
    last_periodic_tick: SimTime,
    pending_delayed: FxHashMap<u64, (AgentId, NewOrderRequest)>,
    #[serde(skip, default)]
    result: SimResult,
}

fn side_str(s: Side) -> &'static str {
    match s {
        Side::Buy => "buy",
        Side::Sell => "sell",
    }
}
fn kind_str(k: OrderEventKind) -> &'static str {
    match k {
        OrderEventKind::Submit => "submit",
        OrderEventKind::Cancel => "cancel",
        OrderEventKind::Fill => "fill",
        OrderEventKind::PartialFill => "partial_fill",
        OrderEventKind::Reject => "reject",
        OrderEventKind::Expire => "expire",
    }
}

impl Simulation {
    pub fn build(cfg: ScenarioConfig) -> Result<Self, String> {
        let market = Market::new(cfg.assets.clone(), cfg.venues.clone(), cfg.market.clone(), cfg.fees.clone(), cfg.mempool.clone());
        let (agents, strategies) = build_population(&cfg, 0, &market)?;
        let now = 0.0;
        Ok(Self {
            market,
            agents,
            strategies,
            session_end: cfg.run.session_seconds,
            cfg,
            events: EventQueue::new(),
            market_rng: rng::market_rng(0),
            now,
            last_periodic_tick: 0.0,
            pending_delayed: FxHashMap::default(),
            result: SimResult::default(),
        })
    }

    /// Runs to completion (natural end-of-session, or `min_trades`-driven extension
    /// up to `max_session_seconds`). Equivalent to `run_cancellable(None)`.
    pub fn run(self) -> SimResult {
        self.run_cancellable(None)
    }

    /// Same as `run`, but checks `cancel` periodically and stops early (returning
    /// whatever was produced so far, with `SimResult::cancelled = true`) once it's
    /// set. `cancel` is a plain `Arc<AtomicBool>` rather than anything PyO3-specific
    /// so `sim_core` stays usable standalone -- the pyo3 layer's `CancelToken` is
    /// just a thin wrapper around one, set from another thread (or another Python
    /// thread, via `py.allow_threads` releasing the GIL for the duration of this
    /// call) while this runs.
    pub fn run_cancellable(self, cancel: Option<Arc<std::sync::atomic::AtomicBool>>) -> SimResult {
        self.run_cancellable_keep_state(cancel, None).1
    }

    /// Same as `run_cancellable`, but also returns the `Simulation` in its final
    /// state (order books, agents, RNG streams, pending event queue), for callers
    /// that want to checkpoint it (`to_checkpoint_bytes`) rather than discard it, and
    /// accepts an optional `progress` handle -- see `drain_loop`'s doc comment.
    pub fn run_cancellable_keep_state(mut self, cancel: Option<Arc<std::sync::atomic::AtomicBool>>, progress: Option<Arc<AtomicU64>>) -> (Self, SimResult) {
        let seed = self.cfg.run.seed;
        self.market_rng = rng::market_rng(seed);

        for (i, _) in self.strategies.iter().enumerate() {
            let jitter = self.market_rng.gen_range(0.0..1.0);
            self.events.push(jitter, EventKind::AgentWake { agent_id: i as AgentId });
        }
        self.events.push(0.0, EventKind::SnapshotTick);
        self.events.push(self.cfg.liquidation.sweep_interval_secs, EventKind::LiquidationSweep);
        self.events.push(self.session_end, EventKind::EndOfSession);

        self.drain_loop(cancel, progress)
    }

    /// Continues a previously-checkpointed simulation (see `to_checkpoint_bytes` /
    /// `from_checkpoint_bytes`) for `additional_seconds` more simulated time, with NO
    /// re-seeding of anything: the order books, every agent's cash/inventory/
    /// indicator state, both RNG streams (market-level and per-agent), and the
    /// not-yet-due event queue (pending agent wakes, periodic snapshot/liquidation
    /// ticks) all carry over exactly as checkpointed.
    ///
    /// This works with no special-casing because every bot and every periodic system
    /// event is self-perpetuating -- each `AgentWake`/`SnapshotTick`/
    /// `LiquidationSweep` handler reschedules its own successor -- and the original
    /// run loop only ever exits by `break`ing out on `EndOfSession`. So a checkpoint
    /// taken from a completed (or cancelled) run already has a queue full of events
    /// scheduled beyond that point (next agent wakes, next snapshot tick, ...) that
    /// were simply never popped; resuming just needs to push one new `EndOfSession`
    /// further out and keep draining the same queue.
    ///
    /// Produces a *fresh* `SimResult` covering only the newly-simulated segment --
    /// trades/order_events/snapshots from before the checkpoint are not duplicated
    /// here. A caller wanting one combined history should concatenate the original
    /// run's `SimResult` with this one (e.g. the dashboard does this when showing a
    /// continued run).
    pub fn resume(self, additional_seconds: SimTime, cancel: Option<Arc<std::sync::atomic::AtomicBool>>) -> SimResult {
        self.resume_keep_state(additional_seconds, cancel, None).1
    }

    /// Same as `resume`, but also returns the `Simulation` in its final state, for
    /// callers that want to checkpoint it again (e.g. to continue a second time), and
    /// accepts an optional `progress` handle -- see `drain_loop`'s doc comment.
    pub fn resume_keep_state(mut self, additional_seconds: SimTime, cancel: Option<Arc<std::sync::atomic::AtomicBool>>, progress: Option<Arc<AtomicU64>>) -> (Self, SimResult) {
        self.result = SimResult::default();
        self.session_end = self.now + additional_seconds.max(0.0);
        self.events.push(self.session_end, EventKind::EndOfSession);
        self.drain_loop(cancel, progress)
    }

    /// Hands this simulation's market off from whatever population built it
    /// so far to a *new* one, mid-run -- e.g. a zero-intelligence warmup
    /// phase (real trade/price history and resting liquidity already in the
    /// live order book, built from actual trading rather than an empty book)
    /// handed off to the "real" bot population once it's ready to go, rather
    /// than that population's own first wakes being the very first orders
    /// the book has ever seen (2026-10-10, measured: that cold-start moment
    /// produces a real, large burst -- the whole population's first wake is
    /// jittered over just the first second).
    ///
    /// Every agent currently in the simulation is marked `dormant` (see that
    /// field's doc comment in `agent.rs`) -- they never act again, but their
    /// cash/inventory/resting orders are untouched, so whatever liquidity
    /// they'd already built stays live in the book as the new population's
    /// starting point. The new population is appended (never replaces
    /// existing agent ids/indices -- `build_population`'s own
    /// `starting_agent_id` keeps the two populations' ids, and hence their
    /// deterministic per-agent RNG streams via `rng::agent_rng`, from ever
    /// colliding), and each new agent gets its own first-wake event jittered
    /// over `GRAFT_WAKE_SPREAD_SECS` from *now* -- deliberately much wider
    /// than the 1-second window `run_cancellable_keep_state` uses for a fresh
    /// `build` at t=0: a direct first measurement (300 noise + 296 real
    /// agents, 500s warmup) showed that even with warm resting liquidity
    /// already in the book, cramming every new agent's first action into the
    /// same 1-second window still produced a real burst (2,235 trades in the
    /// graft's first 5 seconds vs. tens per 5 seconds shortly after) --
    /// spreading new agents' first wakes out removes that specific cause
    /// (everyone acting at once), separate from the already-warm book
    /// removing the *empty-book* cause a fresh start has.
    ///
    /// `new_cfg`'s `assets`/`venues`/`market`/`fees`/`mempool`/`liquidation`/
    /// `run.seed` are ignored -- the already-running `Market` and `self.cfg`
    /// own those; only `new_cfg.population.groups` is read. (Used to also
    /// note a trend-group graft fidelity gap here -- moot now that trend
    /// groups are gone entirely, see `market.rs`'s removal note.)
    pub fn graft_population(&mut self, new_cfg: &ScenarioConfig) -> Result<(), String> {
        for agent in self.agents.iter_mut() {
            agent.dormant = true;
        }
        let starting_agent_id = self.agents.len() as AgentId;
        let (new_agents, new_strategies) = build_population(new_cfg, starting_agent_id, &self.market)?;
        let n_new = new_strategies.len();
        self.agents.extend(new_agents);
        self.strategies.extend(new_strategies);
        const GRAFT_WAKE_SPREAD_SECS: f64 = 60.0;
        for i in 0..n_new {
            let agent_id = starting_agent_id + i as AgentId;
            let jitter = self.market_rng.gen_range(0.0..GRAFT_WAKE_SPREAD_SECS);
            self.events.push(self.now + jitter, EventKind::AgentWake { agent_id });
        }
        Ok(())
    }

    /// Serializes the full simulation state (order books, agents, RNG streams,
    /// pending event queue, config -- everything needed by `resume`) to a compact
    /// binary blob (CBOR). Excludes `result` (see `resume`'s doc comment): a
    /// checkpoint's size reflects live state, not how much trading history has
    /// accumulated.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, String> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf).map_err(|e| format!("checkpoint serialization failed: {e}"))?;
        Ok(buf)
    }

    /// Restores a simulation previously saved with `to_checkpoint_bytes`, ready to be
    /// driven further via `resume`.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, String> {
        ciborium::from_reader(bytes).map_err(|e| format!("checkpoint deserialization failed: {e}"))
    }

    /// Shared event-draining main loop used by both a fresh `run_cancellable` (after
    /// it seeds the initial events) and `resume` (which reuses an already-populated
    /// queue). See `run_cancellable`'s doc comment for `cancel` semantics.
    ///
    /// `progress`, if given, is written with the current simulated time (`self.now`,
    /// as `f64::to_bits`) at most a few times a second -- a way for another thread
    /// (the PyO3 layer's `ProgressHandle`, backing the dashboard's "current
    /// simulated second" display while a long run is in progress) to see how far
    /// along a run is without needing it to actually finish. A plain `Arc<AtomicU64>`
    /// rather than anything PyO3-specific, same reasoning as `cancel`.
    fn drain_loop(mut self, cancel: Option<Arc<std::sync::atomic::AtomicBool>>, progress: Option<Arc<AtomicU64>>) -> (Self, SimResult) {
        let mut events_since_check: u32 = 0;
        let mut last_progress_write = Instant::now();
        while let Some(ev) = self.events.pop() {
            if cancel.is_some() || progress.is_some() {
                events_since_check += 1;
                // Checked periodically, not every iteration: cheap individually, but
                // this loop can run millions of times/sec, so even a relaxed atomic
                // load/store adds up if done every single iteration. Cancellation (or
                // a progress update) being noticed a few hundred events late makes no
                // practical difference at these rates.
                if events_since_check >= 512 {
                    events_since_check = 0;
                    if let Some(flag) = &cancel {
                        if flag.load(Ordering::Relaxed) {
                            self.result.cancelled = true;
                            break;
                        }
                    }
                    if let Some(p) = &progress {
                        // Further throttled to wall-clock time, not just event count:
                        // event throughput varies a lot with population size/session
                        // length, and the UI polls this at most once a second anyway
                        // (see app/dashboard.py) -- no need to pay for an atomic store
                        // more often than that regardless of how fast events drain.
                        let now_wall = Instant::now();
                        if now_wall.duration_since(last_progress_write).as_millis() >= 500 {
                            last_progress_write = now_wall;
                            p.store(self.now.to_bits(), Ordering::Relaxed);
                        }
                    }
                }
            }
            self.now = ev.time;
            match ev.kind {
                EventKind::AgentWake { agent_id } => self.handle_agent_wake(agent_id),
                EventKind::DelayedOrderArrival { payload_id, .. } => self.handle_delayed_arrival(payload_id),
                EventKind::MempoolExecute { payload_id, .. } => self.handle_mempool_execute(payload_id),
                EventKind::MarkPriceTick { .. } => {}
                EventKind::LiquidationSweep => {
                    self.run_liquidation_sweep();
                    self.run_net_worth_bankruptcy_sweep();
                    // Piggybacks on this same periodic tick rather than its own EventKind --
                    // see `run_order_expiry_sweep`'s doc comment for why this exists.
                    self.run_order_expiry_sweep();
                    self.events.push(self.now + self.cfg.liquidation.sweep_interval_secs, EventKind::LiquidationSweep);
                }
                EventKind::LiquidationSlice { agent_id, asset_id, slice_no } => self.handle_liquidation_slice(agent_id, asset_id, slice_no),
                EventKind::SnapshotTick => {
                    let dt = (self.now - self.last_periodic_tick).max(1e-6);
                    self.last_periodic_tick = self.now;
                    let mut mrng = std::mem::replace(&mut self.market_rng, rng::market_rng(0));
                    self.market.tick_mark_and_fundamental(dt, &mut mrng);
                    self.market_rng = mrng;
                    if self.cfg.export.write_snapshots {
                        self.record_snapshot();
                    }
                    self.events.push(self.now + self.cfg.market.snapshot_interval_secs.max(0.01), EventKind::SnapshotTick);
                }
                EventKind::EndOfSession => {
                    if let Some(min_trades) = self.cfg.run.min_trades {
                        let have = self.result.trades.len() as u64;
                        let cap = self.cfg.run.max_session_seconds.unwrap_or(f64::INFINITY);
                        if have < min_trades && self.session_end < cap {
                            self.session_end = (self.session_end + self.cfg.run.extension_step_seconds).min(cap);
                            self.events.push(self.session_end, EventKind::EndOfSession);
                            continue;
                        }
                    }
                    break;
                }
            }
        }

        // Final write so the UI promptly sees the true end time rather than being up
        // to ~500ms stale from the last periodic update above.
        if let Some(p) = &progress {
            p.store(self.now.to_bits(), Ordering::Relaxed);
        }
        self.result.actual_session_seconds = self.now;
        self.finalize_agent_states();
        let result = std::mem::take(&mut self.result);
        (self, result)
    }

    fn handle_agent_wake(&mut self, agent_id: AgentId) {
        let idx = agent_id as usize;
        if self.agents[idx].bankrupt || self.agents[idx].dormant {
            return;
        }
        let ctx = StepCtx { market: &self.market, venue: PRIMARY_VENUE, now: self.now };
        let StepResult { actions, next_wake } = bots::step(&mut self.agents[idx], &mut self.strategies[idx], &ctx);

        // wash_trading's self-cross (bots/wash_trading.rs) prices its resting leg
        // against the book state *at decision time* and relies on its own IOC leg
        // matching that same resting order immediately afterward. Latency breaks that:
        // the two legs would be independently delayed, and if the book moved in
        // between, the resting leg can get silently rejected while the IOC leg still
        // fires unconditionally -- a naked, unbounded market order with nothing to
        // absorb it (this produced a genuine runaway price crash in testing; see git
        // history). So wash_trading orders always bypass configured latency/mempool
        // staging, the same way margin orders do.
        let is_wash_trading = matches!(self.strategies[idx], BotStrategy::WashTrading(_));
        for a in actions {
            match (is_wash_trading, a) {
                (true, Action::Submit(req)) => self.apply_submit(agent_id, req),
                // Per-action margin flag (not bot-type-wide) -- see `Action::SubmitMargin`'s
                // doc comment: a bot that mixes ordinary and leveraged orders in the same
                // step marks only the leveraged one.
                (_, Action::SubmitMargin(req)) => self.apply_submit_margin(agent_id, req),
                (_, a) => self.apply_action(agent_id, a),
            }
        }

        // Generic leveraged side-bet piggyback (spec 5.7: independent of bot_type).
        if let Some(lev) = self.agents[idx].leverage_cfg {
            let asset_id = bots::primary_asset(&self.strategies[idx]);
            if let Some(mid) = self.market.mid_price(PRIMARY_VENUE, asset_id) {
                if let Some(req) = liquidation_trader::open_leveraged_position(
                    &mut self.agents[idx],
                    asset_id,
                    lev.leverage_range,
                    lev.position_frac_of_cash,
                    self.cfg.liquidation.maintenance_margin_buffer,
                    mid,
                ) {
                    self.apply_submit_margin(agent_id, req);
                }
            }
        }

        if let Some(t) = next_wake {
            self.events.push(t, EventKind::AgentWake { agent_id });
        }
    }

    fn apply_action(&mut self, agent_id: AgentId, action: Action) {
        match action {
            Action::Cancel(order_id, asset_id) => {
                // Cancels are applied immediately (no latency modeled on cancels --
                // documented simplification, see README).
                if let Some(ev) = self.market.cancel(PRIMARY_VENUE, asset_id, order_id, self.now) {
                    self.push_order_event(ev, PRIMARY_VENUE);
                }
            }
            Action::Submit(req) => {
                let latency = self.agents[agent_id as usize].latency;
                let mid = self.market.mid_price(PRIMARY_VENUE, req.asset_id).unwrap_or(req.price.unwrap_or(0.0));
                let notional = req.price.unwrap_or(mid) * req.qty;
                if self.cfg.mempool.enabled && notional >= self.cfg.mempool.large_order_notional_threshold {
                    let payload_id = self.market.queue_mempool_order(
                        PRIMARY_VENUE,
                        agent_id,
                        req.asset_id,
                        req.side,
                        req.order_type,
                        req.tif,
                        req.price,
                        req.qty,
                        self.now,
                        latency,
                    );
                    self.events.push(self.now + self.cfg.mempool.reveal_delay_secs.max(latency), EventKind::MempoolExecute { venue_id: PRIMARY_VENUE, payload_id });
                } else if latency > 1e-9 {
                    let payload_id = self.market.alloc_payload_id();
                    self.pending_delayed.insert(payload_id, (agent_id, req));
                    self.events.push(self.now + latency, EventKind::DelayedOrderArrival { agent_id, payload_id });
                } else {
                    self.apply_submit(agent_id, req);
                }
            }
            // Reachable only if some future call site routes a raw `Action` here
            // without going through the per-action dispatch `step_agent` already
            // handles (see that match's own `Action::SubmitMargin` arm) -- kept
            // for exhaustiveness/defensiveness, not a normal path today.
            Action::SubmitMargin(req) => self.apply_submit_margin(agent_id, req),
        }
    }

    fn handle_delayed_arrival(&mut self, payload_id: u64) {
        if let Some((agent_id, req)) = self.pending_delayed.remove(&payload_id) {
            // This order was submitted (and its latency delay started) before the
            // agent potentially went bankrupt in between -- the universal
            // net-worth sweep (`run_net_worth_bankruptcy_sweep`) can mark any
            // agent bankrupt mid-stream now, not just at a tightly-controlled
            // margin-close call site, so an order already "in flight" when that
            // happens needs to be dropped here rather than landing on the book
            // for a dormant agent and silently reintroducing a position that was
            // just guaranteed flat (measured directly in testing before this
            // check was added).
            if self.agents[agent_id as usize].bankrupt {
                return;
            }
            self.apply_submit(agent_id, req);
        }
    }

    fn handle_mempool_execute(&mut self, payload_id: u64) {
        if let Some(p) = self.market.take_pending_mempool(payload_id) {
            // Same reasoning as `handle_delayed_arrival` -- drop an in-flight
            // mempool order for an agent that went bankrupt while it sat pending.
            if self.agents[p.agent_id as usize].bankrupt {
                return;
            }
            let req = NewOrderRequest {
                asset_id: p.asset_id,
                side: p.side,
                order_type: p.order_type,
                tif: p.tif,
                post_only: false,
                price: p.price,
                qty: p.qty,
            };
            self.apply_submit(p.agent_id, req);
        }
    }

    fn apply_submit(&mut self, agent_id: AgentId, req: NewOrderRequest) {
        self.apply_submit_inner(agent_id, req, false);
    }

    /// Used only for the two margin-order sources (leveraged-position entry, and
    /// liquidation exits): submitted with no latency, and settled without touching
    /// spot cash/inventory for the submitting agent (see module docs on `Order` /
    /// `open_leveraged_position` for why -- the collateral/pnl bookkeeping happens
    /// directly at the call site, not through generic spot settlement).
    fn apply_submit_margin(&mut self, agent_id: AgentId, req: NewOrderRequest) {
        self.apply_submit_inner(agent_id, req, true);
    }

    fn apply_submit_inner(&mut self, agent_id: AgentId, req: NewOrderRequest, taker_is_margin: bool) {
        let (order_id, result) = self.market.submit(
            PRIMARY_VENUE,
            agent_id,
            req.asset_id,
            req.side,
            req.order_type,
            req.tif,
            req.post_only,
            req.price,
            req.qty,
            self.now,
        );
        bots::on_order_id(&mut self.strategies[agent_id as usize], &req, order_id);
        for ev in &result.events {
            self.push_order_event(ev.clone(), PRIMARY_VENUE);
        }
        for fill in &result.fills {
            self.apply_fill(PRIMARY_VENUE, fill, taker_is_margin);
        }
    }

    fn push_order_event(&mut self, ev: crate::order::OrderEvent, venue_id: VenueId) {
        self.result.n_order_events_total += 1;
        if !self.cfg.export.write_order_events {
            return;
        }
        self.result.order_events.push(OrderEventRow {
            time: ev.time,
            order_id: ev.order_id,
            agent_id: ev.agent_id,
            asset_id: ev.asset_id,
            venue_id,
            kind: kind_str(ev.kind),
            side: side_str(ev.side),
            price: ev.price_ticks.map(|t| t as f64 * self.market.assets[ev.asset_id as usize].tick_size),
            qty: ev.qty,
        });
    }

    fn apply_fill(&mut self, venue_id: VenueId, fill: &Fill, taker_is_margin: bool) {
        let price = fill.price_ticks as f64 * self.market.assets[fill.asset_id as usize].tick_size;
        let (maker_type, maker_variant) = self.strategies[fill.maker_agent_id as usize].type_variant_names();
        let (taker_type, taker_variant) = self.strategies[fill.taker_agent_id as usize].type_variant_names();
        let maker_fee_rate = self.agents[fill.maker_agent_id as usize].maker_fee;
        let taker_fee_rate = self.agents[fill.taker_agent_id as usize].taker_fee;
        let notional = price * fill.qty;
        let maker_fee_paid = notional * maker_fee_rate;
        let taker_fee_paid = notional * taker_fee_rate;

        // Maker is always a normal spot counterparty in this model (margin orders are
        // always the aggressive/taker leg -- see open_leveraged_position/
        // execute_liquidation_qty).
        self.settle(fill.maker_agent_id, fill.asset_id, fill.taker_side.opposite(), fill.qty, price, maker_fee_paid);
        if taker_is_margin {
            self.agents[fill.taker_agent_id as usize].cash -= taker_fee_paid;
            self.agents[fill.taker_agent_id as usize].cumulative_volume += notional;
        } else {
            self.settle(fill.taker_agent_id, fill.asset_id, fill.taker_side, fill.qty, price, taker_fee_paid);
        }

        self.result.trades.push(TradeRow {
            time: fill.time,
            asset_id: fill.asset_id,
            venue_id,
            price,
            qty: fill.qty,
            taker_side: side_str(fill.taker_side),
            maker_agent_id: fill.maker_agent_id,
            taker_agent_id: fill.taker_agent_id,
            maker_type,
            maker_variant,
            taker_type,
            taker_variant,
            maker_fee_paid,
            taker_fee_paid,
        });
    }

    fn settle(&mut self, agent_id: AgentId, asset_id: AssetId, side: Side, qty: f64, price: f64, fee: f64) {
        let a = &mut self.agents[agent_id as usize];
        match side {
            Side::Buy => {
                a.cash -= price * qty;
                a.add_inventory(asset_id, qty);
            }
            Side::Sell => {
                a.cash += price * qty;
                a.add_inventory(asset_id, -qty);
            }
        }
        a.cash -= fee;
        a.cumulative_volume += price * qty;
    }

    fn run_liquidation_sweep(&mut self) {
        let mark_prices = self.market.mark_price_ema.clone();
        for idx in 0..self.agents.len() {
            if self.agents[idx].bankrupt {
                continue;
            }
            let Some(pos) = self.agents[idx].margin.clone() else { continue };
            let mark = *mark_prices.get(&pos.asset_id).unwrap_or(&pos.entry_price);
            if !pos.is_liquidatable(mark) {
                if let Some(m) = self.agents[idx].margin.as_mut() {
                    m.staggered_round = 0;
                }
                continue;
            }
            let style = self.agents[idx].liquidation_style;
            match style {
                LiquidationExecStyle::Full => {
                    self.execute_liquidation_qty(idx as AgentId, pos.asset_id, pos.size);
                }
                LiquidationExecStyle::StaggeredTwap { slices } => {
                    // One round already ran against this exact position and it is
                    // *still* liquidatable -- the adverse move is outpacing a
                    // quarter-at-a-time taper, not the ordinary case the staggered
                    // style is for. Escalate straight to a single full close (same
                    // as `Full` above) instead of scheduling another round: see
                    // `MarginPosition::staggered_round`'s doc comment for why
                    // repeating this indefinitely is a liquidation-cascade bug, not
                    // a feature.
                    if pos.staggered_round >= 1 {
                        self.execute_liquidation_qty(idx as AgentId, pos.asset_id, pos.size);
                        continue;
                    }
                    let required = liquidation_trader::required_liquidation_qty(&pos, mark, self.cfg.liquidation.maintenance_margin_buffer, 1.25);
                    if required <= 1e-9 {
                        continue;
                    }
                    if let Some(m) = self.agents[idx].margin.as_mut() {
                        m.staggered_round += 1;
                    }
                    // Slices re-derive what's still needed (bounded to at most 1/4 of
                    // the remaining position each time, see handle_liquidation_slice)
                    // at execution time, rather than committing to a fixed amount now,
                    // so a partially-recovered position naturally tapers off.
                    let stagger_interval = (self.cfg.liquidation.sweep_interval_secs / slices as f64).max(0.05);
                    for k in 0..slices {
                        self.events.push(self.now + stagger_interval * (k as f64 + 1.0), EventKind::LiquidationSlice { agent_id: idx as AgentId, asset_id: pos.asset_id, slice_no: k });
                    }
                }
            }
        }
    }

    fn handle_liquidation_slice(&mut self, agent_id: AgentId, asset_id: AssetId, _slice_no: u32) {
        let idx = agent_id as usize;
        let Some(pos) = self.agents[idx].margin.clone() else { return };
        if pos.asset_id != asset_id {
            return;
        }
        let mark = *self.market.mark_price_ema.get(&asset_id).unwrap_or(&pos.entry_price);
        let required = liquidation_trader::required_liquidation_qty(&pos, mark, self.cfg.liquidation.maintenance_margin_buffer, 1.25);
        let slice_qty = required.min(pos.size / 4.0).max(0.0);
        if slice_qty > 1e-9 {
            self.execute_liquidation_qty(agent_id, asset_id, slice_qty);
        }
    }

    fn execute_liquidation_qty(&mut self, agent_id: AgentId, asset_id: AssetId, qty: f64) {
        let idx = agent_id as usize;
        // Snapshot the position as it stood *before* this closing slice: pnl and the
        // collateral-released share are both computed against this original size, not
        // the (not yet mutated) live position, to avoid a fraction-of-a-fraction bug.
        let Some(pos) = self.agents[idx].margin.clone() else { return };
        let close_qty = qty.min(pos.size).max(0.0);
        if close_qty <= 1e-9 {
            return;
        }
        let closing_side = pos.side.opposite();
        self.apply_submit_margin(
            agent_id,
            NewOrderRequest { asset_id, side: closing_side, order_type: OrderType::Market, tif: TimeInForce::Ioc, post_only: false, price: None, qty: close_qty },
        );
        self.result.liquidation_events += 1;

        let mark = *self.market.mark_price_ema.get(&asset_id).unwrap_or(&pos.entry_price);
        let pnl_per_unit = match pos.side {
            Side::Buy => mark - pos.entry_price,
            Side::Sell => pos.entry_price - mark,
        };
        let realized = pnl_per_unit * close_qty;
        let collateral_released = pos.collateral * (close_qty / pos.size);

        let a = &mut self.agents[idx];
        a.realized_pnl += realized;
        a.cash += collateral_released + realized;
        if let Some(m) = a.margin.as_mut() {
            m.size -= close_qty;
            m.collateral -= collateral_released;
            if m.size <= 1e-9 {
                a.margin = None;
            }
        }
        // No bankruptcy check here anymore -- that used to be the only place
        // bankruptcy could ever be declared (margin closed + cash<=0), but it
        // only ever flattened *this* margin position, never any spot inventory
        // the same agent separately held (e.g. a grid/momentum agent with both
        // a primary spot strategy and a leveraged side-bet) -- measured
        // directly in testing: agents went "bankrupt" via this path with large
        // spot positions still completely untouched. `run_net_worth_bankruptcy_
        // sweep`, which runs on the same periodic cadence right after
        // `run_liquidation_sweep` (and independently after each staggered
        // `handle_liquidation_slice`, at most one `sweep_interval_secs` later),
        // is now the single place bankruptcy gets declared, always through the
        // complete cancel-orders + close-margin + flatten-spot path -- see
        // `force_liquidate_and_bankrupt`.
    }

    /// Flattens every nonzero spot holding this agent has, in two stages: a
    /// real IOC market order first (so a counterparty that exists gets a
    /// genuinely realistic fill, with normal slippage/impact through
    /// `Sim::settle`), then force-closes whatever that didn't fill at the
    /// current mark price directly. The second stage is not optional --
    /// measured directly in testing, a thin/crashed book can leave an IOC
    /// order with a zero fill on every single attempt (a short agent
    /// repeatedly failing to buy back even one unit while its mark-to-market
    /// loss grew without bound) -- so "sells everything" has to be a
    /// guarantee that completes within this one call, not a best-effort order
    /// that might silently never find a counterparty.
    fn flatten_spot_inventory(&mut self, agent_id: AgentId) {
        let idx = agent_id as usize;
        // Collect first: iterating `inventory` borrows `self.agents[idx]`
        // immutably, but submitting orders below needs `&mut self`.
        let holdings: Vec<(AssetId, f64)> = self.agents[idx]
            .inventory
            .iter()
            .filter(|&(_, &qty)| qty.abs() > 1e-9)
            .map(|(&aid, &qty)| (aid, qty))
            .collect();
        for (asset_id, qty) in holdings {
            let side = if qty > 0.0 { Side::Sell } else { Side::Buy };
            self.apply_submit(
                agent_id,
                NewOrderRequest { asset_id, side, order_type: OrderType::Market, tif: TimeInForce::Ioc, post_only: false, price: None, qty: qty.abs() },
            );
        }
        let mark_prices = self.market.mark_price_ema.clone();
        let leftover: Vec<(AssetId, f64)> = self.agents[idx]
            .inventory
            .iter()
            .filter(|&(_, &qty)| qty.abs() > 1e-9)
            .map(|(&aid, &qty)| (aid, qty))
            .collect();
        for (asset_id, qty) in leftover {
            let mark = *mark_prices.get(&asset_id).unwrap_or(&0.0);
            let a = &mut self.agents[idx];
            a.cash += qty * mark; // qty>0 (long): credits cash as if sold; qty<0 (short): debits cash as if bought back
            a.inventory.insert(asset_id, 0.0);
        }
    }

    /// Cancels every order this agent still has resting in the market, across
    /// every venue and asset. NOTE: deliberately does not use
    /// `AgentCommon::open_orders` -- nothing in this codebase actually pushes
    /// into that field (confirmed by grep: it's only ever initialized empty),
    /// so it's always an empty vec and relying on it silently cancels
    /// nothing. Instead this scans each book's own `orders` map directly for
    /// this agent's ids -- cheap, since there are only a handful of
    /// venues/assets and each book's order count is bounded. Called once, at
    /// the moment an agent is marked bankrupt: without this, a stale resting
    /// order could still get filled by someone else's incoming order later
    /// and silently reintroduce inventory into an agent
    /// `flatten_spot_inventory` just guaranteed was flat -- measured directly
    /// in testing before this was added.
    fn cancel_all_open_orders(&mut self, agent_id: AgentId) {
        let n_venues = self.market.n_venues();
        let n_assets = self.market.n_assets();
        for venue in 0..n_venues as VenueId {
            for asset_id in 0..n_assets as AssetId {
                let order_ids: Vec<OrderId> = self
                    .market
                    .book(venue, asset_id)
                    .orders
                    .values()
                    .filter(|o| o.agent_id == agent_id)
                    .map(|o| o.id)
                    .collect();
                for order_id in order_ids {
                    if let Some(ev) = self.market.cancel(venue, asset_id, order_id, self.now) {
                        self.push_order_event(ev, venue);
                    }
                }
            }
        }
    }

    /// Force-cancels every resting GTC order older than
    /// `cfg.market.max_resting_order_age_secs`, emitted as an `Expire` order event
    /// (not `Cancel`, so cancel-to-trade-ratio analysis can tell the two apart).
    /// Exists because `noise`/`wash_trading` place a fresh GTC order on every wake
    /// without ever cancelling the previous one if it didn't fill -- unlike
    /// `market_maker`/`grid`/`human`, which explicitly cancel-and-replace. Over a
    /// long session that leaves unbounded dead resting liquidity in the book:
    /// confirmed as the cause of a 2026-10-06 anomaly where one job's
    /// checkpoint.bin reached 852MB (vs. ~2.6MB normal) and wall-clock time per
    /// leg climbed steadily while trade volume *fell* -- more and more stale
    /// orders to scan, fewer of them anywhere near the live price. This is a
    /// backstop (same category as a real exchange's max-open-orders or
    /// cancel-on-disconnect protections), not a fix for the underlying bot
    /// behavior -- see this field's doc comment in `config.rs` for the reasoning
    /// behind the 3600s default.
    fn run_order_expiry_sweep(&mut self) {
        let max_age = self.cfg.market.max_resting_order_age_secs;
        if max_age <= 0.0 {
            return;
        }
        let now = self.now;
        let n_venues = self.market.n_venues();
        let n_assets = self.market.n_assets();
        for venue in 0..n_venues as VenueId {
            for asset_id in 0..n_assets as AssetId {
                let expired_ids: Vec<OrderId> = self
                    .market
                    .book(venue, asset_id)
                    .orders
                    .values()
                    .filter(|o| now - o.submitted_at > max_age)
                    .map(|o| o.id)
                    .collect();
                for order_id in expired_ids {
                    if let Some(mut ev) = self.market.cancel(venue, asset_id, order_id, now) {
                        ev.kind = OrderEventKind::Expire;
                        self.push_order_event(ev, venue);
                    }
                }
            }
        }
    }

    /// Universal "blew up" handler, called from `run_net_worth_bankruptcy_sweep`
    /// the tick an agent's equity breaches the 80%-loss threshold, and the
    /// *only* place bankruptcy is ever declared: force-closes any margin
    /// position (reusing the same `execute_liquidation_qty` path the ordinary
    /// margin-liquidation sweep uses -- passing the position's full size closes
    /// it in one shot, regardless of the agent's own staggered/full liquidation
    /// style), cancels every resting order (see `cancel_all_open_orders`),
    /// flattens every spot holding (guaranteed complete -- see
    /// `flatten_spot_inventory`), and marks the agent bankrupt.
    fn force_liquidate_and_bankrupt(&mut self, agent_id: AgentId) {
        let idx = agent_id as usize;
        if self.agents[idx].bankrupt {
            return;
        }
        self.cancel_all_open_orders(agent_id);
        if let Some(pos) = self.agents[idx].margin.clone() {
            self.execute_liquidation_qty(agent_id, pos.asset_id, pos.size);
        }
        self.flatten_spot_inventory(agent_id);
        self.agents[idx].bankrupt = true;
        self.result.bankruptcies += 1;
    }

    /// Runs on the same periodic cadence as the margin-liquidation sweep
    /// (`sweep_interval_secs`). Any non-bankrupt agent whose current equity has
    /// fallen to <=20% of its initial net worth (an 80% loss, measured against
    /// the fixed spawn-time baseline -- see `AgentCommon::initial_net_worth`)
    /// is force-liquidated and marked bankrupt here. Unlike the original
    /// margin-liquidation-sweep bankruptcy check, this applies to every agent
    /// -- spot-only bots and human/retail included, not just the margin
    /// sub-population mid-close -- since ordinary spot settlement
    /// (`Sim::settle`) never itself checks solvency.
    ///
    /// Skips `dormant` agents (2026-10-21 perf fix): a `graft_population` call
    /// leaves the entire pre-graft population in `self.agents` forever, marked
    /// dormant -- they never wake or trade again, so there's no reason to keep
    /// computing `equity()` for them here every `sweep_interval_secs` tick for
    /// the rest of the run. This is also a correctness fix, not just speed: a
    /// dormant agent's mark-to-market equity still drifts with price even
    /// though it can never manage that risk, so without this skip a dormant
    /// agent can get flagged bankrupt from pure passive drift -- exactly the
    /// measurement artifact that made every composition-sweep report before
    /// Sweep XI blend live trading outcomes with frozen warmup leftovers.
    fn run_net_worth_bankruptcy_sweep(&mut self) {
        let mark_prices = self.market.mark_price_ema.clone();
        let mut newly_breaching: Vec<AgentId> = Vec::new();
        for idx in 0..self.agents.len() {
            let agent_id = idx as AgentId;
            let a = &self.agents[idx];
            if a.bankrupt || a.dormant {
                continue;
            }
            if a.initial_net_worth <= 0.0 {
                continue;
            }
            if a.equity(&mark_prices) <= 0.2 * a.initial_net_worth {
                newly_breaching.push(agent_id);
            }
        }
        for agent_id in newly_breaching {
            self.force_liquidate_and_bankrupt(agent_id);
        }
    }

    fn record_snapshot(&mut self) {
        for aid in 0..self.market.n_assets() as AssetId {
            let book = self.market.book(PRIMARY_VENUE, aid);
            for (level, (price, qty)) in book.depth(Side::Buy, self.cfg.market.snapshot_depth_levels).into_iter().enumerate() {
                self.result.snapshots.push(SnapshotRow { time: self.now, asset_id: aid, venue_id: PRIMARY_VENUE, side: "bid", level: level as u32, price, qty });
            }
            for (level, (price, qty)) in book.depth(Side::Sell, self.cfg.market.snapshot_depth_levels).into_iter().enumerate() {
                self.result.snapshots.push(SnapshotRow { time: self.now, asset_id: aid, venue_id: PRIMARY_VENUE, side: "ask", level: level as u32, price, qty });
            }
            // Ground-truth series only the simulation (not a real market) can supply
            // directly -- see report section 4's "price vs. known fundamental value"
            // efficiency metric. Piggybacked onto the same long-format table (qty=0,
            // level=0) rather than a separate export file.
            let fair = *self.market.fundamental_value.get(&aid).unwrap_or(&0.0);
            let mark = *self.market.mark_price_ema.get(&aid).unwrap_or(&0.0);
            let sentiment = *self.market.sentiment.get(&aid).unwrap_or(&0.0);
            self.result.snapshots.push(SnapshotRow { time: self.now, asset_id: aid, venue_id: PRIMARY_VENUE, side: "fundamental", level: 0, price: fair, qty: 0.0 });
            self.result.snapshots.push(SnapshotRow { time: self.now, asset_id: aid, venue_id: PRIMARY_VENUE, side: "mark", level: 0, price: mark, qty: 0.0 });
            // Reuses `price` as a plain numeric carrier (not an actual price) for the
            // sentiment value, same trick as fundamental/mark above -- long-format,
            // no new export file needed.
            self.result.snapshots.push(SnapshotRow { time: self.now, asset_id: aid, venue_id: PRIMARY_VENUE, side: "sentiment", level: 0, price: sentiment, qty: 0.0 });
            // The old "trend" diagnostic row (one shared crowd-trend line per
            // asset) no longer applies -- replaced 2026-10-05 by the social-
            // network signal (`Market::human_signal`), which is per-agent, not
            // per-asset, so it doesn't fit this per-asset snapshot row
            // unchanged. Dropped here rather than exporting something
            // misleading; a proper per-asset view (e.g. mean signal among
            // agents trading this asset) would be a reasonable follow-up.
        }
    }

    fn finalize_agent_states(&mut self) {
        let mark_prices = self.market.mark_price_ema.clone();
        for (idx, a) in self.agents.iter().enumerate() {
            let (bot_type, variant) = self.strategies[idx].type_variant_names();
            let equity = a.equity(&mark_prices);
            if a.inventory.is_empty() {
                self.result.agent_states.push(AgentStateRow {
                    agent_id: a.id,
                    bot_type,
                    variant,
                    asset_id: 0,
                    inventory: 0.0,
                    cash: a.cash,
                    equity,
                    realized_pnl: a.realized_pnl,
                    cumulative_volume: a.cumulative_volume,
                    bankrupt: a.bankrupt,
                    has_margin_position: a.margin.is_some(),
                    margin_leverage: a.margin.as_ref().map(|m| m.leverage),
                    initial_net_worth: a.initial_net_worth,
                });
            } else {
                for (&asset_id, &qty) in &a.inventory {
                    self.result.agent_states.push(AgentStateRow {
                        agent_id: a.id,
                        bot_type,
                        variant,
                        asset_id,
                        inventory: qty,
                        cash: a.cash,
                        equity,
                        realized_pnl: a.realized_pnl,
                        cumulative_volume: a.cumulative_volume,
                        bankrupt: a.bankrupt,
                        has_margin_position: a.margin.is_some(),
                        margin_leverage: a.margin.as_ref().map(|m| m.leverage),
                        initial_net_worth: a.initial_net_worth,
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// Population builder
// ---------------------------------------------------------------------------------

/// `starting_agent_id`: 0 for a normal fresh build (`Simulation::build`'s own
/// call); nonzero when `graft_population` (below) calls this a second time to
/// append a new population onto an already-running simulation -- agent ids
/// (and the per-agent deterministic RNG streams derived from them, see
/// `rng::agent_rng` below) must never collide with the existing population's.
fn build_population(cfg: &ScenarioConfig, starting_agent_id: AgentId, market: &Market) -> Result<(Vec<AgentCommon>, Vec<BotStrategy>), String> {
    let n_assets = cfg.assets.len();
    if n_assets == 0 {
        return Err("scenario must configure at least 1 asset".to_string());
    }
    let n_venues = cfg.venues.len().max(1);

    let mut agents = Vec::new();
    let mut strategies = Vec::new();
    let mut agent_id: AgentId = starting_agent_id;

    for group in &cfg.population.groups {
        let params = group.params_table();
        let (cash_lo, cash_hi) = group.initial_cash_range.unwrap_or((5_000.0, 50_000.0));
        let (maker_fee, taker_fee) = cfg
            .fees
            .by_bot_type
            .get(&group.bot_type)
            .copied()
            .unwrap_or((cfg.fees.default_maker_fee, cfg.fees.default_taker_fee));
        let latency = cfg.latency.by_bot_type.get(&group.bot_type).copied().unwrap_or(cfg.latency.default_latency_secs);

        for i in 0..group.count {
            let mut rng = rng::agent_rng(cfg.run.seed, agent_id);
            let mut cash = if cash_hi > cash_lo { rng.gen_range(cash_lo..cash_hi) } else { cash_lo };
            let asset_a = (i as usize % n_assets) as AssetId;
            let asset_b = ((i as usize + 1) % n_assets) as AssetId;
            let asset_c = ((i as usize + 2) % n_assets) as AssetId;
            let venue_b: VenueId = if n_venues > 1 { 1 } else { 0 };

            let strat = match group.bot_type.as_str() {
                // Every market_maker agent manages every configured asset up to
                // `max_fundamental_vol` (module doc comment's 2026-10-17 "Asset
                // eligibility by volatility tier" entry), not one fixed asset
                // round-robined per agent -- see market_maker.rs's own module doc
                // comment's "Multi-asset risk allocation". Default `f64::MAX`
                // reproduces the old unconditional "every asset" behavior exactly.
                "market_maker" => {
                    let max_vol = params::get_f64(&params, "max_fundamental_vol", f64::MAX);
                    let eligible: Vec<AssetId> = (0..n_assets).filter(|&a| cfg.assets[a].fundamental_vol <= max_vol).map(|a| a as AssetId).collect();
                    BotStrategy::MarketMaker(market_maker::State::from_params(parse_mm_variant(&group.variant)?, eligible, &params, &mut rng, market))
                }
                "arbitrage" => BotStrategy::Arbitrage(arbitrage::State::from_params(
                    parse_arb_variant(&group.variant)?,
                    asset_a,
                    asset_b,
                    asset_c,
                    PRIMARY_VENUE,
                    venue_b,
                    cfg.assets[asset_a as usize].initial_price,
                    cfg.assets[asset_b as usize].initial_price,
                    cfg.assets[asset_c as usize].initial_price,
                    cash,
                    &params,
                )),
                "momentum" => BotStrategy::Momentum(momentum::State::from_params(parse_momentum_variant(&group.variant)?, asset_a, cfg.assets[asset_a as usize].initial_price, &params)),
                "mean_reversion" => BotStrategy::MeanReversion(mean_reversion::State::from_params(parse_mr_variant(&group.variant)?, asset_a, cfg.assets[asset_a as usize].initial_price, &params)),
                "grid" => BotStrategy::Grid(grid::State::from_params(parse_grid_variant(&group.variant)?, asset_a, cfg.assets[asset_a as usize].initial_price, &params)),
                "dca" => BotStrategy::Dca(dca::State::from_params(parse_dca_variant(&group.variant)?, asset_a, cfg.run.session_seconds, &params)),
                "mev" => BotStrategy::Mev(mev::State::from_params(parse_mev_variant(&group.variant)?, asset_a, cfg.assets[asset_a as usize].initial_price, &params)),
                "wash_trading" => BotStrategy::WashTrading(wash_trading::State::from_params(
                    parse_wash_variant(&group.variant)?,
                    asset_a,
                    cfg.run.session_seconds,
                    cfg.assets[asset_a as usize].initial_price,
                    &params,
                )),
                "noise" => BotStrategy::Noise(noise::State::from_params(parse_noise_variant(&group.variant)?, asset_a, cfg.assets[asset_a as usize].initial_price, &params)),
                "human" => {
                    BotStrategy::Human(human::State::from_params(parse_human_variant(&group.variant)?, asset_a, cfg.assets[asset_a as usize].initial_price, &params, &mut rng))
                }
                other => return Err(format!("unknown bot_type '{other}' in population config")),
            };

            let liquidation_style = if rng.gen_bool(cfg.liquidation.full_liquidation_fraction.clamp(0.0, 1.0)) {
                LiquidationExecStyle::Full
            } else {
                LiquidationExecStyle::StaggeredTwap { slices: cfg.liquidation.staggered_slices }
            };

            let leverage_cfg = if group.leveraged_fraction > 0.0 && rng.gen_bool(group.leveraged_fraction.clamp(0.0, 1.0)) {
                Some(LeverageCfg {
                    leverage_range: group.leverage_range.unwrap_or((2.0, 10.0)),
                    position_frac_of_cash: 0.3,
                })
            } else {
                None
            };

            // Noise/retail agents start with an existing spot holding (like a real
            // trader who already owns some of the asset), not zero -- otherwise the
            // *entire* population is structurally buy-only until it organically
            // accumulates inventory (sells are correctly gated to what's actually
            // held; see bots/noise.rs), which biased price sharply upward at the
            // start of every session in testing (see git history). 20-60% of the
            // agent's starting budget is pre-converted into inventory at the asset's
            // initial price; the rest stays as cash.
            let mut inventory = FxHashMap::default();
            // **2026-10-14, direct user request**: every bot type that posts
            // ordinary spot `Side::Sell` orders as a normal part of its own
            // strategy (not through the separate margin/leverage mechanism,
            // which `liquidation_trader` uses exclusively and needs no spot
            // inventory for) hits the *exact same* structural bug `noise`/
            // `human` were already fixed for above -- confirmed directly in
            // `staged_introduction_test`: grafting market_maker/arbitrage/mev
            // in cold, all at zero inventory, produced +1,732%/+460%/+665%
            // price spikes in under 500s, because every one of them was
            // categorically buy-only (sells gated to actual holdings) until
            // it organically accumulated something to sell. `dca` is
            // deliberately excluded (a buy-only accumulator by design, not a
            // bug) and `liquidation_trader` needs no spot seed (its
            // directional exposure is entirely margin-based).
            let needs_inventory_seed = matches!(
                group.bot_type.as_str(),
                "noise" | "human" | "market_maker" | "grid" | "momentum" | "mean_reversion" | "arbitrage" | "mev"
            );
            if needs_inventory_seed {
                // 2026-10-15, direct user request: raised from 0.2..0.6 -- the
                // smaller seed still left buy capacity (bounded by cash) well
                // ahead of sell capacity (bounded by this starting stake), so
                // the same upward-bias dynamic kept reappearing, just milder.
                // This is one shared code path for both the initial
                // population (`Simulation::build`) and every later wave
                // (`graft_population`), so the larger seed applies to fresh
                // agents either way, not just at session start.
                let frac = rng.gen_range(0.5..0.9);
                let mut value = cash * frac;
                if group.bot_type == "human" {
                    // `human`'s own position cap (`max_position_notional`,
                    // default $4000 -- see human.rs) is far smaller relative to
                    // typical cash than `noise`'s is: an unconstrained 20-60%-
                    // of-cash seed would start a meaningful share of human agents
                    // already pinned at (or past) their own buy cap before the
                    // session even begins, for no behavioral reason -- just an
                    // artifact of how much cash they happened to be given. Half
                    // the cap leaves room for this agent's own signal to still
                    // move it either direction from the start.
                    let max_position_notional = params::get_f64(&params, "max_position_notional", 4000.0);
                    value = value.min(max_position_notional * 0.5);
                }
                if group.bot_type == "market_maker" {
                    // Spread evenly across *every* asset this agent manages
                    // (see market_maker.rs's "Multi-asset risk allocation"),
                    // not just `asset_a` -- a single-asset seed would leave
                    // it buy-only on the other 4 assets it also quotes.
                    let per_asset_value = value / n_assets as f64;
                    cash -= value;
                    for aid in 0..n_assets as AssetId {
                        let initial_price = cfg.assets[aid as usize].initial_price.max(1e-9);
                        inventory.insert(aid, per_asset_value / initial_price);
                    }
                } else {
                    let initial_price = cfg.assets[asset_a as usize].initial_price.max(1e-9);
                    inventory.insert(asset_a, value / initial_price);
                    cash -= value;
                }
            }

            // cash + any pre-funded inventory (only `noise`, above), valued at each
            // asset's own initial price -- the fixed baseline the universal 80%-loss
            // bankruptcy sweep compares current equity against (see
            // Sim::run_net_worth_bankruptcy_sweep). Computed here, once, from the
            // same (already-finalized) cash/inventory this agent is actually
            // constructed with, not re-derived later.
            let initial_net_worth = cash + inventory.iter().map(|(&aid, &qty)| qty * cfg.assets[aid as usize].initial_price).sum::<f64>();

            agents.push(AgentCommon {
                id: agent_id,
                cash,
                inventory,
                bankrupt: false,
                dormant: false,
                margin: None,
                liquidation_style,
                leverage_cfg,
                latency,
                maker_fee,
                taker_fee,
                open_orders: Vec::new(),
                rng,
                realized_pnl: 0.0,
                cumulative_volume: 0.0,
                initial_net_worth,
            });
            strategies.push(strat);
            agent_id += 1;
        }
    }

    Ok((agents, strategies))
}

macro_rules! variant_parser {
    ($fn_name:ident, $enum_ty:ty, { $($name:literal => $variant:expr),+ $(,)? }) => {
        fn $fn_name(s: &str) -> Result<$enum_ty, String> {
            match s {
                $($name => Ok($variant),)+
                other => Err(format!(concat!("unknown ", stringify!($enum_ty), " variant '{}'"), other)),
            }
        }
    };
}

variant_parser!(parse_mm_variant, market_maker::Variant, {
    "avellaneda_stoikov" => market_maker::Variant::SymmetricAvellanedaStoikov,
    "fixed_offset" => market_maker::Variant::FixedOffset,
    "inventory_skew" => market_maker::Variant::InventorySkew,
    "adaptive_spread" => market_maker::Variant::AdaptiveSpread,
});
variant_parser!(parse_arb_variant, arbitrage::Variant, {
    "cross_venue" => arbitrage::Variant::CrossVenue,
    "triangular" => arbitrage::Variant::Triangular,
    "stat_arb_pairs" => arbitrage::Variant::StatArbPairs,
    "latency_arb" => arbitrage::Variant::LatencyArb,
});
variant_parser!(parse_momentum_variant, momentum::Variant, {
    "ma_crossover" => momentum::Variant::MaCrossover,
    "breakout_donchian" => momentum::Variant::BreakoutDonchian,
    "rsi" => momentum::Variant::Rsi,
    "ml_online" => momentum::Variant::MlOnline,
});
variant_parser!(parse_mr_variant, mean_reversion::Variant, {
    "bollinger" => mean_reversion::Variant::Bollinger,
    "ou_fair_value" => mean_reversion::Variant::OuFairValue,
    "zscore" => mean_reversion::Variant::ZScore,
});
variant_parser!(parse_grid_variant, grid::Variant, {
    "static" => grid::Variant::Static,
    "geometric" => grid::Variant::Geometric,
    "dynamic_atr" => grid::Variant::DynamicAtr,
});
variant_parser!(parse_dca_variant, dca::Variant, {
    "time_based" => dca::Variant::TimeBased,
    "value_averaging" => dca::Variant::ValueAveraging,
    "dip_buying" => dca::Variant::DipBuying,
});
variant_parser!(parse_mev_variant, mev::Variant, {
    "sandwich" => mev::Variant::Sandwich,
    "frontrun" => mev::Variant::Frontrun,
    "backrun" => mev::Variant::Backrun,
});
variant_parser!(parse_wash_variant, wash_trading::Variant, {
    "self_ring" => wash_trading::Variant::SelfRing,
    "volume_ranking" => wash_trading::Variant::VolumeRanking,
});
variant_parser!(parse_noise_variant, noise::Variant, {
    "zero_intelligence" => noise::Variant::ZeroIntelligence,
    "herding" => noise::Variant::Herding,
});
variant_parser!(parse_human_variant, human::Variant, {
    "retail" => human::Variant::Retail,
    "lite" => human::Variant::Lite,
});
