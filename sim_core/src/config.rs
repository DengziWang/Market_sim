//! Scenario configuration: everything about a single run that is NOT hardcoded --
//! read from TOML by both Rust (to build the simulation) and Python (to write/inspect
//! it), per the "no code changes to try a different bot mix" requirement.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub seed: u64,
    /// Nominal simulated session length. Actual length may be extended (see
    /// `max_session_seconds`) to satisfy `min_trades`.
    pub session_seconds: f64,
    #[serde(default)]
    pub min_trades: Option<u64>,
    #[serde(default)]
    pub max_session_seconds: Option<f64>,
    /// Extend the session in steps of this many seconds while under `min_trades`.
    #[serde(default = "default_extension_step")]
    pub extension_step_seconds: f64,
}
fn default_extension_step() -> f64 {
    1800.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetConfig {
    pub symbol: String,
    pub tick_size: f64,
    pub initial_price: f64,
    /// Annualized-style volatility parameter driving the latent fundamental-value
    /// process (Ornstein-Uhlenbeck) that fundamentalist/mean-reversion bots and the
    /// price-efficiency metrics (deviation from known fundamental) reference.
    #[serde(default = "default_fundamental_vol")]
    pub fundamental_vol: f64,
    #[serde(default = "default_fundamental_theta")]
    pub fundamental_theta: f64,
    /// Same Ornstein-Uhlenbeck shape as the fundamental-value process, but modeling
    /// something psychologically different: collective market *mood* (bullish <-> 0
    /// <-> bearish, normalized to roughly [-1, 1]), not the asset's true value.
    /// `human` bots read this alongside price trend (see bots/human.rs). Independent
    /// of fundamental_value -- its own random draw, mean-reverts
    /// to neutral (0) rather than to any price level, and its vol/theta are
    /// deliberately named/defaulted differently since mood realistically swings more
    /// and persists longer than a well-anchored fundamental value would.
    #[serde(default = "default_sentiment_vol")]
    pub sentiment_vol: f64,
    #[serde(default = "default_sentiment_theta")]
    pub sentiment_theta: f64,
}
fn default_fundamental_vol() -> f64 {
    0.02
}
fn default_fundamental_theta() -> f64 {
    0.1
}
fn default_sentiment_vol() -> f64 {
    0.4
}
fn default_sentiment_theta() -> f64 {
    0.05
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueConfig {
    pub name: String,
    /// One-way latency (seconds) for information/orders to travel between this venue
    /// and any other venue -- used by cross-venue arbitrage bots.
    #[serde(default = "default_inter_venue_latency")]
    pub inter_venue_latency_secs: f64,
}
fn default_inter_venue_latency() -> f64 {
    0.05
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConfig {
    #[serde(default = "default_snapshot_interval")]
    pub snapshot_interval_secs: f64,
    #[serde(default = "default_snapshot_depth")]
    pub snapshot_depth_levels: usize,
    #[serde(default = "default_mark_halflife")]
    pub mark_price_ema_halflife_secs: f64,
    /// Circuit breaker: the largest fraction a single order's matching is allowed to
    /// move price away from the best opposing price it started against. Once matching
    /// would cross that band, the order stops taking further liquidity there (an IOC/
    /// market order's unfilled remainder is discarded as usual; a limit order's
    /// remainder rests per its TIF) rather than sweeping arbitrarily deep into a thin
    /// book. Real exchanges have an equivalent (price bands / fat-finger protection);
    /// without one here, a single large order hitting a thinly-quoted asset could
    /// move its printed price by 10-1000x in one fill regardless of how reasonably
    /// any individual bot's order size was chosen. Set very large (e.g. 1e9) to
    /// effectively disable this as its own research variable.
    #[serde(default = "default_max_price_move_frac")]
    pub max_price_move_frac: f64,
    /// A resting GTC limit order older than this (by `now - submitted_at`) is
    /// force-cancelled (emitted as an `Expire` order event, not `Cancel`) the next
    /// time the expiry sweep runs. Several bot types (noise, wash_trading) place a
    /// fresh GTC order on every wake without ever cancelling the previous one if it
    /// didn't fill -- unlike market_maker/grid/human, which explicitly cancel their
    /// own resting order before replacing it. Over a long session this leaves
    /// unbounded stale litter in the book: confirmed in testing as the cause of a
    /// 2026-10-06 anomaly where one job's checkpoint.bin reached 852MB (vs. ~2.6MB
    /// normal) and its per-leg wall-clock time climbed steadily *while trade volume
    /// fell* -- the smoking gun for "more and more dead resting orders to scan,
    /// fewer of them still near the live price." 3600s (1 hour) is chosen to sit
    /// comfortably above every strategy's own legitimate intentional resting
    /// period (the longest is human's `background_wake_max`, 2400s by default) so
    /// it never interferes with normal cancel-and-replace behavior, while still
    /// being short enough to bound backlog growth within a small fraction of a
    /// typical multi-hour session. Set very large (e.g. 1e12) to effectively
    /// disable.
    #[serde(default = "default_max_resting_order_age")]
    pub max_resting_order_age_secs: f64,
}
fn default_snapshot_interval() -> f64 {
    1.0
}
fn default_snapshot_depth() -> usize {
    10
}
fn default_mark_halflife() -> f64 {
    30.0
}
fn default_max_price_move_frac() -> f64 {
    0.15
}
fn default_max_resting_order_age() -> f64 {
    3600.0
}
impl Default for MarketConfig {
    fn default() -> Self {
        Self {
            snapshot_interval_secs: default_snapshot_interval(),
            snapshot_depth_levels: default_snapshot_depth(),
            mark_price_ema_halflife_secs: default_mark_halflife(),
            max_price_move_frac: default_max_price_move_frac(),
            max_resting_order_age_secs: default_max_resting_order_age(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeConfig {
    #[serde(default = "default_maker_fee")]
    pub default_maker_fee: f64,
    #[serde(default = "default_taker_fee")]
    pub default_taker_fee: f64,
    #[serde(default)]
    pub by_bot_type: HashMap<String, (f64, f64)>,
}
fn default_maker_fee() -> f64 {
    0.0002
}
fn default_taker_fee() -> f64 {
    0.0005
}
impl Default for FeeConfig {
    fn default() -> Self {
        Self {
            default_maker_fee: default_maker_fee(),
            default_taker_fee: default_taker_fee(),
            by_bot_type: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyConfig {
    #[serde(default)]
    pub default_latency_secs: f64,
    #[serde(default)]
    pub by_bot_type: HashMap<String, f64>,
}
impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            default_latency_secs: 0.05,
            by_bot_type: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MempoolConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_reveal_delay")]
    pub reveal_delay_secs: f64,
    /// Orders with notional (price*qty, or qty*mid for market orders) at or above this
    /// are staged through the public pending-order queue instead of landing directly;
    /// smaller "retail-sized" orders never appear in the mempool model.
    #[serde(default = "default_mempool_threshold")]
    pub large_order_notional_threshold: f64,
}
fn default_reveal_delay() -> f64 {
    0.4
}
fn default_mempool_threshold() -> f64 {
    20_000.0
}
impl Default for MempoolConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            reveal_delay_secs: default_reveal_delay(),
            large_order_notional_threshold: default_mempool_threshold(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidationConfig {
    #[serde(default = "default_mm_buffer")]
    pub maintenance_margin_buffer: f64,
    #[serde(default = "default_sweep_interval")]
    pub sweep_interval_secs: f64,
    #[serde(default = "default_staggered_slices")]
    pub staggered_slices: u32,
    /// Fraction of liquidation events executed as immediate full liquidation vs.
    /// staggered/partial TWAP-style execution.
    #[serde(default = "default_full_frac")]
    pub full_liquidation_fraction: f64,
}
fn default_mm_buffer() -> f64 {
    0.8
}
fn default_sweep_interval() -> f64 {
    1.0
}
fn default_staggered_slices() -> u32 {
    5
}
fn default_full_frac() -> f64 {
    0.5
}
impl Default for LiquidationConfig {
    fn default() -> Self {
        Self {
            maintenance_margin_buffer: default_mm_buffer(),
            sweep_interval_secs: default_sweep_interval(),
            staggered_slices: default_staggered_slices(),
            full_liquidation_fraction: default_full_frac(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotGroupConfig {
    /// One of: market_maker, arbitrage, momentum, mean_reversion, grid, dca,
    /// liquidation_exec, mev, wash_trading, noise.
    pub bot_type: String,
    pub variant: String,
    pub count: u32,
    #[serde(default)]
    pub initial_cash_range: Option<(f64, f64)>,
    /// Fraction of this group that additionally carries a leveraged margin position
    /// (independent of bot_type, per spec 5.7 -- momentum/grid/etc. bots can be
    /// leveraged too).
    #[serde(default)]
    pub leveraged_fraction: f64,
    #[serde(default)]
    pub leverage_range: Option<(f64, f64)>,
    /// Free-form variant-specific parameters, read via `sim_core::params` helpers.
    #[serde(default)]
    pub params: Option<toml::Value>,
    /// Purely documentary: lets a scenario file explain/flag a still-provisional
    /// weight (e.g. "TODO: confirm with user") without affecting simulation behavior.
    #[serde(default)]
    pub notes: Option<String>,
}

impl BotGroupConfig {
    /// Always-present params table, even when the scenario file omits `[*.params]`.
    pub fn params_table(&self) -> toml::Value {
        self.params
            .clone()
            .unwrap_or_else(|| toml::Value::Table(Default::default()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationConfig {
    pub groups: Vec<BotGroupConfig>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    pub out_dir: String,
    #[serde(default)]
    pub write_snapshots: bool,
    /// 2026-10-15: an event-heavy, low-human-fraction composition can generate
    /// tens of millions of order events in a single 500s leg; collecting each
    /// one as an `OrderEventRow` (two heap-allocated strings per row) to hand
    /// back to Python regardless of whether the caller ever writes it to
    /// parquet was blowing past 11GB RSS for a single simulation. Defaults to
    /// `true` (the old, always-collect behavior) for backward compatibility --
    /// set `false` in scenarios/callers that already pass
    /// `write_order_events=False` to `export_run` so the engine never
    /// materializes data nobody asked to keep.
    #[serde(default = "default_true")]
    pub write_order_events: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioConfig {
    pub run: RunConfig,
    pub assets: Vec<AssetConfig>,
    #[serde(default)]
    pub venues: Vec<VenueConfig>,
    #[serde(default)]
    pub market: MarketConfig,
    #[serde(default)]
    pub fees: FeeConfig,
    #[serde(default)]
    pub latency: LatencyConfig,
    #[serde(default)]
    pub mempool: MempoolConfig,
    pub population: PopulationConfig,
    #[serde(default)]
    pub liquidation: LiquidationConfig,
    pub export: ExportConfig,
}

impl ScenarioConfig {
    pub fn from_toml_str(s: &str) -> Result<Self, toml::de::Error> {
        let mut cfg: ScenarioConfig = toml::from_str(s)?;
        if cfg.venues.is_empty() {
            cfg.venues.push(VenueConfig {
                name: "primary".to_string(),
                inter_venue_latency_secs: default_inter_venue_latency(),
            });
        }
        Ok(cfg)
    }

    pub fn total_agents(&self) -> u32 {
        self.population.groups.iter().map(|g| g.count).sum()
    }
}
