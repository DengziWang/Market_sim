//! Orchestration: builds the agent population from a `ScenarioConfig`, drives the
//! event-driven main loop, applies fills/fees/latency/mempool staging, runs the
//! liquidation sweep, and produces a `SimResult`.

use std::collections::HashMap;
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
use crate::rng;
use crate::types::{AgentId, AssetId, FxHashMap, OrderType, Side, SimTime, TimeInForce, VenueId};

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
    pending_delayed: HashMap<u64, (AgentId, NewOrderRequest)>,
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
        let (agents, strategies) = build_population(&cfg)?;
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
            pending_delayed: HashMap::new(),
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
        if self.agents[idx].bankrupt {
            return;
        }
        let ctx = StepCtx { market: &self.market, venue: PRIMARY_VENUE, now: self.now };
        let StepResult { actions, next_wake } = bots::step(&mut self.agents[idx], &mut self.strategies[idx], &ctx);

        // The dedicated liquidation_trader bot type only ever emits margin-entry
        // orders (see liquidation_trader::step) -- route them through the margin
        // settlement path (fee-only; the collateral/position bookkeeping already
        // happened directly in open_leveraged_position) rather than normal spot
        // settlement, and skip the latency/mempool staging normal orders get.
        let is_leveraged_bot = matches!(self.strategies[idx], BotStrategy::LiquidationTrader(_));
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
            match (is_leveraged_bot, is_wash_trading, a) {
                (true, _, Action::Submit(req)) => self.apply_submit_margin(agent_id, req),
                (_, true, Action::Submit(req)) => self.apply_submit(agent_id, req),
                (_, _, a) => self.apply_action(agent_id, a),
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
        }
    }

    fn handle_delayed_arrival(&mut self, payload_id: u64) {
        if let Some((agent_id, req)) = self.pending_delayed.remove(&payload_id) {
            self.apply_submit(agent_id, req);
        }
    }

    fn handle_mempool_execute(&mut self, payload_id: u64) {
        if let Some(p) = self.market.take_pending_mempool(payload_id) {
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
                continue;
            }
            let style = self.agents[idx].liquidation_style;
            match style {
                LiquidationExecStyle::Full => {
                    self.execute_liquidation_qty(idx as AgentId, pos.asset_id, pos.size);
                }
                LiquidationExecStyle::StaggeredTwap { slices } => {
                    let required = liquidation_trader::required_liquidation_qty(&pos, mark, self.cfg.liquidation.maintenance_margin_buffer, 1.25);
                    if required <= 1e-9 {
                        continue;
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
        if a.margin.is_none() && a.cash <= 0.0 {
            a.bankrupt = true;
            self.result.bankruptcies += 1;
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
                    });
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// Population builder
// ---------------------------------------------------------------------------------

fn build_population(cfg: &ScenarioConfig) -> Result<(Vec<AgentCommon>, Vec<BotStrategy>), String> {
    let n_assets = cfg.assets.len();
    if n_assets == 0 {
        return Err("scenario must configure at least 1 asset".to_string());
    }
    let n_venues = cfg.venues.len().max(1);

    let mut agents = Vec::new();
    let mut strategies = Vec::new();
    let mut agent_id: AgentId = 0;

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
                "market_maker" => BotStrategy::MarketMaker(market_maker::State::from_params(parse_mm_variant(&group.variant)?, asset_a, cfg.assets[asset_a as usize].initial_price, &params)),
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
                "liquidation_trader" => {
                    BotStrategy::LiquidationTrader(liquidation_trader::State::from_params(parse_liq_variant(&group.variant)?, asset_a, cfg.liquidation.maintenance_margin_buffer, &params))
                }
                "mev" => BotStrategy::Mev(mev::State::from_params(parse_mev_variant(&group.variant)?, asset_a, cfg.assets[asset_a as usize].initial_price, &params)),
                "wash_trading" => BotStrategy::WashTrading(wash_trading::State::from_params(
                    parse_wash_variant(&group.variant)?,
                    asset_a,
                    cfg.run.session_seconds,
                    cfg.assets[asset_a as usize].initial_price,
                    &params,
                )),
                "noise" => BotStrategy::Noise(noise::State::from_params(parse_noise_variant(&group.variant)?, asset_a, cfg.assets[asset_a as usize].initial_price, &params)),
                "human" => BotStrategy::Human(human::State::from_params(parse_human_variant(&group.variant)?, asset_a, cfg.assets[asset_a as usize].initial_price, &params)),
                other => return Err(format!("unknown bot_type '{other}' in population config")),
            };

            let liquidation_style = if let BotStrategy::LiquidationTrader(s) = &strat {
                s.exec_style(cfg.liquidation.staggered_slices)
            } else if rng.gen_bool(cfg.liquidation.full_liquidation_fraction.clamp(0.0, 1.0)) {
                LiquidationExecStyle::Full
            } else {
                LiquidationExecStyle::StaggeredTwap { slices: cfg.liquidation.staggered_slices }
            };

            let leverage_cfg = if group.bot_type != "liquidation_trader" && group.leveraged_fraction > 0.0 && rng.gen_bool(group.leveraged_fraction.clamp(0.0, 1.0)) {
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
            if group.bot_type == "noise" {
                let frac = rng.gen_range(0.2..0.6);
                let initial_price = cfg.assets[asset_a as usize].initial_price.max(1e-9);
                let value = cash * frac;
                inventory.insert(asset_a, value / initial_price);
                cash -= value;
            }

            agents.push(AgentCommon {
                id: agent_id,
                cash,
                inventory,
                bankrupt: false,
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
variant_parser!(parse_liq_variant, liquidation_trader::Variant, {
    "full" => liquidation_trader::Variant::Full,
    "staggered_twap" => liquidation_trader::Variant::StaggeredTwap,
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
});
