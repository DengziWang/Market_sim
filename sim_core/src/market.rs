//! Multi-venue, multi-asset market state: order books, EMA mark price (for
//! liquidation checks), a latent fundamental-value process per asset (Ornstein-
//! Uhlenbeck; ground truth for price-efficiency metrics and fundamentalist bots),
//! and an optional simplified public mempool for MEV bots.

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use crate::config::{AssetConfig, FeeConfig, MarketConfig, MempoolConfig, VenueConfig};
use crate::order::{Order, OrderEvent};
use crate::orderbook::{OrderBook, SubmitResult};
use crate::types::{AgentId, AssetId, FxHashMap, OrderId, OrderType, PriceTicks, Side, SimTime, TimeInForce, VenueId};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PendingMempoolOrder {
    pub payload_id: u64,
    pub venue_id: VenueId,
    pub asset_id: AssetId,
    pub agent_id: AgentId,
    pub side: Side,
    pub order_type: OrderType,
    pub tif: TimeInForce,
    pub price: Option<f64>,
    pub qty: f64,
    pub visible_at: SimTime,
    pub execute_at: SimTime,
    pub consumed: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Market {
    pub assets: Vec<AssetConfig>,
    pub venues: Vec<VenueConfig>,
    pub cfg: MarketConfig,
    pub fees: FeeConfig,
    pub mempool_cfg: MempoolConfig,
    books: FxHashMap<(VenueId, AssetId), OrderBook>,
    pub mark_price_ema: FxHashMap<AssetId, f64>,
    pub fundamental_value: FxHashMap<AssetId, f64>,
    /// Collective market mood per asset, roughly in [-1, 1]. See
    /// `AssetConfig::sentiment_vol`/`sentiment_theta`'s doc comment.
    pub sentiment: FxHashMap<AssetId, f64>,
    pub last_trade_price: FxHashMap<(VenueId, AssetId), f64>,
    pub pending_mempool: Vec<PendingMempoolOrder>,
    /// **Market-wide recent-return memory (2026-10-14, direct user request)**:
    /// a rolling window of this asset's own recent log-returns, sampled once
    /// per `SnapshotTick` (same cadence `tick_mark_and_fundamental` already
    /// updates `mark_price_ema` on -- see that function), independent of any
    /// one agent's own wake cadence. Exists so a *newly constructed* agent
    /// (at initial build, or grafted into an already-running market) can
    /// seed its own risk-sensing rolling windows from what's actually been
    /// happening in the market, instead of starting with a blank slate --
    /// measured directly in `staged_introduction_test`: market_maker agents
    /// grafted in after momentum/mean_reversion had already been trading for
    /// 10,000s collapsed *faster* than when everyone started cold together
    /// (58% bankrupt within 8 minutes of entering vs. ~2% at the same point
    /// in a simultaneous start), because they had zero sense that the
    /// market was already "hot." Capacity matches market_maker's own
    /// default `vol_window` length (30) -- see `MARKET_HISTORY_CAPACITY`.
    pub recent_log_returns: FxHashMap<AssetId, std::collections::VecDeque<f64>>,
    /// The previous `SnapshotTick`'s own sample (same "last trade, else mid,
    /// else initial price" fallback `tick_mark_and_fundamental` already uses
    /// for `mark_price_ema`), kept separately so `recent_log_returns` holds
    /// genuine tick-to-tick returns rather than comparing against the
    /// smoothed EMA.
    last_tick_sample: FxHashMap<AssetId, f64>,
    next_order_id: OrderId,
    next_payload_id: u64,
}

/// See `Market::recent_log_returns`'s doc comment.
const MARKET_HISTORY_CAPACITY: usize = 30;

impl Market {
    /// **Grouped "crowd trend" removed (2026-10-17, direct user request)**:
    /// used to also take `n_trend_groups: u32` here and seed a
    /// `random_trend` map, a shared per-group Ornstein-Uhlenbeck line every
    /// human agent in that group read identically (`human::State::
    /// trend_group`/`personal_trend_weight`). Removed along with
    /// `revenge_trading_propensity` (see `human.rs`'s module doc comment)
    /// after this session's crash investigation identified both as direct
    /// contributors to correlated human herding: a shared group signal
    /// means many agents lean the same way at the same moment for a reason
    /// that has nothing to do with price, which is exactly the kind of
    /// synchronized push that overwhelms available liquidity. `sentiment`
    /// above stays -- it's shared too, but read with independent per-agent
    /// perception noise (`human.rs`'s `perceived_sentiment`), not a bare
    /// identical number every group member reacts to the same way.
    pub fn new(assets: Vec<AssetConfig>, venues: Vec<VenueConfig>, cfg: MarketConfig, fees: FeeConfig, mempool_cfg: MempoolConfig) -> Self {
        let mut books = FxHashMap::default();
        let mut mark_price_ema = FxHashMap::default();
        let mut fundamental_value = FxHashMap::default();
        let mut sentiment = FxHashMap::default();
        for (vid, _venue) in venues.iter().enumerate() {
            for (aid, asset) in assets.iter().enumerate() {
                books.insert((vid as VenueId, aid as AssetId), OrderBook::new(aid as AssetId, asset.tick_size, cfg.max_price_move_frac));
            }
        }
        let mut last_tick_sample = FxHashMap::default();
        for (aid, asset) in assets.iter().enumerate() {
            mark_price_ema.insert(aid as AssetId, asset.initial_price);
            fundamental_value.insert(aid as AssetId, asset.initial_price);
            sentiment.insert(aid as AssetId, 0.0);
            last_tick_sample.insert(aid as AssetId, asset.initial_price);
        }
        Self {
            assets,
            venues,
            cfg,
            fees,
            mempool_cfg,
            books,
            mark_price_ema,
            fundamental_value,
            sentiment,
            last_trade_price: FxHashMap::default(),
            pending_mempool: Vec::new(),
            recent_log_returns: FxHashMap::default(),
            last_tick_sample,
            next_order_id: 1,
            next_payload_id: 1,
        }
    }

    pub fn n_assets(&self) -> usize {
        self.assets.len()
    }
    pub fn n_venues(&self) -> usize {
        self.venues.len()
    }

    pub fn book(&self, venue: VenueId, asset: AssetId) -> &OrderBook {
        self.books.get(&(venue, asset)).expect("unknown (venue, asset)")
    }
    fn book_mut(&mut self, venue: VenueId, asset: AssetId) -> &mut OrderBook {
        self.books.get_mut(&(venue, asset)).expect("unknown (venue, asset)")
    }

    /// The book's own mid price, or the asset's configured `initial_price` as a
    /// reference-price fallback while the book is empty (one-sided, or has no orders
    /// at all -- always true at t=0). Without this fallback every bot deadlocks at
    /// startup: every strategy needs *some* price to quote/trade around, and an empty
    /// book can never produce one on its own.
    pub fn mid_price(&self, venue: VenueId, asset: AssetId) -> Option<f64> {
        self.book(venue, asset).mid_price().or_else(|| self.assets.get(asset as usize).map(|a| a.initial_price))
    }

    fn alloc_order_id(&mut self) -> OrderId {
        let id = self.next_order_id;
        self.next_order_id += 1;
        id
    }

    pub fn alloc_payload_id(&mut self) -> u64 {
        let id = self.next_payload_id;
        self.next_payload_id += 1;
        id
    }

    /// Submit a new order directly (no latency/mempool staging -- callers in `sim.rs`
    /// apply per-agent-class latency by delaying *when* this is invoked, not here).
    #[allow(clippy::too_many_arguments)]
    pub fn submit(
        &mut self,
        venue: VenueId,
        agent_id: AgentId,
        asset_id: AssetId,
        side: Side,
        order_type: OrderType,
        tif: TimeInForce,
        post_only: bool,
        price: Option<f64>,
        qty: f64,
        time: SimTime,
    ) -> (OrderId, SubmitResult) {
        let id = self.alloc_order_id();
        let tick_size = self.assets[asset_id as usize].tick_size;
        let mut price_ticks: Option<PriceTicks> = price.map(|p| (p / tick_size).round() as PriceTicks);
        // Last-line-of-defense sanity guards: no strategy bug (bad vol estimate, stale
        // reference price, dimensional/units mistake, etc.) should ever be able to
        // push a non-positive, non-finite, or wildly-off-market price onto the book.
        // Reject rather than let it corrupt the book -- this matters even beyond the
        // matching-time circuit breaker (`OrderBook::max_price_move_frac`), which only
        // guards *crossing* trades: a bad price that doesn't cross anything just rests
        // quietly and becomes the new best bid/ask, corrupting the reference every
        // subsequent order (including well-behaved ones) sees. The band is checked
        // against the latent fundamental value, not the traded mid price, specifically
        // because it's a ground-truth process immune to being corrupted by trading
        // activity itself (unlike mid price, which a bad order could already be in the
        // middle of throwing off).
        let invalid_price = matches!(order_type, OrderType::Limit) && price_ticks.map_or(true, |t| t <= 0);
        let far_from_fundamental = matches!(order_type, OrderType::Limit)
            && price.map_or(false, |p| {
                self.fundamental_value.get(&asset_id).map_or(false, |&fv| fv > 0.0 && (p < fv * 0.02 || p > fv * 50.0))
            });
        let invalid_price = invalid_price || far_from_fundamental;
        let invalid_qty = !qty.is_finite() || qty <= 0.0;
        if invalid_price || invalid_qty {
            price_ticks = price_ticks.filter(|t| *t > 0);
            let ev = OrderEvent {
                time,
                order_id: id,
                agent_id,
                asset_id,
                kind: crate::order::OrderEventKind::Reject,
                side,
                price_ticks,
                qty,
            };
            return (id, SubmitResult { fills: Vec::new(), events: vec![ev] });
        }
        let order = Order {
            id,
            agent_id,
            asset_id,
            side,
            order_type,
            tif,
            post_only,
            price_ticks,
            qty,
            qty_filled: 0.0,
            status: crate::order::OrderStatus::Open,
            submitted_at: time,
            seq: 0,
        };
        let result = self.book_mut(venue, asset_id).submit(order, time);
        for f in &result.fills {
            self.last_trade_price.insert((venue, asset_id), self.assets[asset_id as usize].tick_size * f.price_ticks as f64);
        }
        (id, result)
    }

    pub fn cancel(&mut self, venue: VenueId, asset_id: AssetId, order_id: OrderId, time: SimTime) -> Option<OrderEvent> {
        self.book_mut(venue, asset_id).cancel(order_id, time)
    }

    pub fn depth(&self, venue: VenueId, asset: AssetId, side: Side, levels: usize) -> Vec<(f64, f64)> {
        self.book(venue, asset).depth(side, levels)
    }

    /// Advance the EMA mark price, the latent fundamental value process, and the
    /// latent sentiment process for every asset (primary venue only feeds the mark
    /// price -- see module docs). `dt` is the time elapsed since the previous tick.
    pub fn tick_mark_and_fundamental(&mut self, dt: SimTime, rng: &mut ChaCha8Rng) {
        let halflife = self.cfg.mark_price_ema_halflife_secs.max(1e-6);
        let alpha = 1.0 - 0.5f64.powf(dt / halflife);
        for aid in 0..self.assets.len() as AssetId {
            let sample = self
                .last_trade_price
                .get(&(0, aid))
                .copied()
                .or_else(|| self.mid_price(0, aid))
                .unwrap_or(self.assets[aid as usize].initial_price);
            let prev = *self.mark_price_ema.get(&aid).unwrap();
            self.mark_price_ema.insert(aid, prev + alpha * (sample - prev));

            // See `Market::recent_log_returns`'s doc comment.
            let prev_sample = *self.last_tick_sample.get(&aid).unwrap_or(&sample);
            if prev_sample > 0.0 && sample > 0.0 {
                let log_ret = (sample / prev_sample).ln();
                let buf = self.recent_log_returns.entry(aid).or_default();
                if buf.len() >= MARKET_HISTORY_CAPACITY {
                    buf.pop_front();
                }
                buf.push_back(log_ret);
            }
            self.last_tick_sample.insert(aid, sample);

            let asset = &self.assets[aid as usize];
            let f_prev = *self.fundamental_value.get(&aid).unwrap();
            let z: f64 = rng.sample(StandardNormal);
            let f_next = f_prev
                + asset.fundamental_theta * (asset.initial_price - f_prev) * dt
                + asset.fundamental_vol * f_prev * dt.sqrt() * z;
            self.fundamental_value.insert(aid, f_next.max(asset.tick_size));

            // Independent draw and independent process from fundamental value above:
            // mean-reverts to neutral (0), not to any price level, and is deliberately
            // not clamped to a price-derived floor (it's a normalized mood, not a price).
            let s_prev = *self.sentiment.get(&aid).unwrap();
            let z2: f64 = rng.sample(StandardNormal);
            let s_next = s_prev + asset.sentiment_theta * (0.0 - s_prev) * dt + asset.sentiment_vol * dt.sqrt() * z2;
            self.sentiment.insert(aid, s_next.clamp(-1.0, 1.0));
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn queue_mempool_order(
        &mut self,
        venue: VenueId,
        agent_id: AgentId,
        asset_id: AssetId,
        side: Side,
        order_type: OrderType,
        tif: TimeInForce,
        price: Option<f64>,
        qty: f64,
        now: SimTime,
        extra_delay: SimTime,
    ) -> u64 {
        let payload_id = self.alloc_payload_id();
        self.pending_mempool.push(PendingMempoolOrder {
            payload_id,
            venue_id: venue,
            asset_id,
            agent_id,
            side,
            order_type,
            tif,
            price,
            qty,
            visible_at: now,
            execute_at: now + self.mempool_cfg.reveal_delay_secs.max(extra_delay),
            consumed: false,
        });
        payload_id
    }

    pub fn take_pending_mempool(&mut self, payload_id: u64) -> Option<PendingMempoolOrder> {
        if let Some(pos) = self.pending_mempool.iter().position(|p| p.payload_id == payload_id) {
            Some(self.pending_mempool.remove(pos))
        } else {
            None
        }
    }

    pub fn visible_mempool(&self, now: SimTime) -> impl Iterator<Item = &PendingMempoolOrder> {
        self.pending_mempool.iter().filter(move |p| !p.consumed && p.visible_at <= now)
    }
}
