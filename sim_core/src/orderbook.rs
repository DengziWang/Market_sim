//! Single-asset limit order book: continuous double-auction matching with strict
//! price-time priority. No batch/call-auction phase.
//!
//! Cancel/replace is implemented at the agent level as cancel() + submit() of a new
//! order (this is how every bot in `bots/` uses it) rather than as a single atomic op
//! in the book -- that keeps the matching core small while still giving agents the
//! full cancel/replace *behaviour* they need (e.g. market-maker requoting).

use std::collections::{BTreeMap, HashMap, VecDeque};

use crate::order::{Fill, Order, OrderEvent, OrderEventKind, OrderStatus};
use crate::types::{AssetId, OrderId, OrderType, PriceTicks, Side, SimTime, TickSize, TimeInForce};

#[derive(Debug, Clone, Copy)]
pub struct MakerFee(pub f64);
#[derive(Debug, Clone, Copy)]
pub struct TakerFee(pub f64);

pub struct OrderBook {
    pub asset_id: AssetId,
    pub tick_size: TickSize,
    pub bids: BTreeMap<PriceTicks, VecDeque<OrderId>>,
    pub asks: BTreeMap<PriceTicks, VecDeque<OrderId>>,
    pub orders: HashMap<OrderId, Order>,
    next_seq: u64,
}

pub struct SubmitResult {
    pub fills: Vec<Fill>,
    pub events: Vec<OrderEvent>,
}

impl OrderBook {
    pub fn new(asset_id: AssetId, tick_size: f64) -> Self {
        Self {
            asset_id,
            tick_size: TickSize(tick_size),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            orders: HashMap::new(),
            next_seq: 0,
        }
    }

    pub fn best_bid_ticks(&self) -> Option<PriceTicks> {
        self.bids.keys().next_back().copied()
    }
    pub fn best_ask_ticks(&self) -> Option<PriceTicks> {
        self.asks.keys().next().copied()
    }
    pub fn best_bid(&self) -> Option<f64> {
        self.best_bid_ticks().map(|t| self.tick_size.to_price(t))
    }
    pub fn best_ask(&self) -> Option<f64> {
        self.best_ask_ticks().map(|t| self.tick_size.to_price(t))
    }
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((b + a) / 2.0),
            (Some(b), None) => Some(b),
            (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((a - b).max(0.0)),
            _ => None,
        }
    }

    /// Aggregated resting quantity per price level, best-first, up to `levels` levels.
    pub fn depth(&self, side: Side, levels: usize) -> Vec<(f64, f64)> {
        let iter_book = match side {
            Side::Buy => &self.bids,
            Side::Sell => &self.asks,
        };
        let ids_at = |ids: &VecDeque<OrderId>| -> f64 {
            ids.iter()
                .filter_map(|id| self.orders.get(id))
                .map(|o| o.remaining())
                .sum()
        };
        match side {
            Side::Buy => iter_book
                .iter()
                .rev()
                .take(levels)
                .map(|(p, ids)| (self.tick_size.to_price(*p), ids_at(ids)))
                .collect(),
            Side::Sell => iter_book
                .iter()
                .take(levels)
                .map(|(p, ids)| (self.tick_size.to_price(*p), ids_at(ids)))
                .collect(),
        }
    }

    fn book_mut(&mut self, side: Side) -> &mut BTreeMap<PriceTicks, VecDeque<OrderId>> {
        match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        }
    }

    fn rest(&mut self, order: Order) {
        let side = order.side;
        let price = order.price_ticks.expect("resting order must have a price");
        let id = order.id;
        self.orders.insert(id, order);
        self.book_mut(side).entry(price).or_default().push_back(id);
    }

    /// Remove an order from its book side (used by cancel and after full fill).
    fn unlink(&mut self, side: Side, price: PriceTicks, id: OrderId) {
        let book = self.book_mut(side);
        if let Some(level) = book.get_mut(&price) {
            if let Some(pos) = level.iter().position(|&x| x == id) {
                level.remove(pos);
            }
            if level.is_empty() {
                book.remove(&price);
            }
        }
    }

    pub fn cancel(&mut self, order_id: OrderId, time: SimTime) -> Option<OrderEvent> {
        let (side, price, agent_id, asset_id) = {
            let o = self.orders.get(&order_id)?;
            if o.is_done() {
                return None;
            }
            (o.side, o.price_ticks, o.agent_id, o.asset_id)
        };
        if let Some(p) = price {
            self.unlink(side, p, order_id);
        }
        // Terminal: drop from `self.orders` rather than retaining a Cancelled-status
        // entry forever (keeps long-running sessions with heavy requoting bounded).
        let remaining_qty = self
            .orders
            .remove(&order_id)
            .map(|o| o.remaining())
            .unwrap_or(0.0);
        Some(OrderEvent {
            time,
            order_id,
            agent_id,
            asset_id,
            kind: OrderEventKind::Cancel,
            side,
            price_ticks: price,
            qty: remaining_qty,
        })
    }

    /// Would a limit order at `price_ticks`/`side` cross the book (i.e. be immediately
    /// marketable)? Used for the post-only check.
    fn would_cross(&self, side: Side, price_ticks: PriceTicks) -> bool {
        match side {
            Side::Buy => self.best_ask_ticks().map_or(false, |a| price_ticks >= a),
            Side::Sell => self.best_bid_ticks().map_or(false, |b| price_ticks <= b),
        }
    }

    /// Total resting quantity reachable by an incoming order of `side` at/within
    /// `limit_ticks` (None = unbounded, i.e. a market order): used for the FOK
    /// pre-check without mutating the book.
    fn reachable_liquidity(&self, side: Side, limit_ticks: Option<PriceTicks>) -> f64 {
        let opposite = match side {
            Side::Buy => &self.asks,
            Side::Sell => &self.bids,
        };
        let mut total = 0.0;
        match side {
            Side::Buy => {
                for (p, ids) in opposite.iter() {
                    if let Some(l) = limit_ticks {
                        if *p > l {
                            break;
                        }
                    }
                    total += ids.iter().filter_map(|id| self.orders.get(id)).map(|o| o.remaining()).sum::<f64>();
                }
            }
            Side::Sell => {
                for (p, ids) in opposite.iter().rev() {
                    if let Some(l) = limit_ticks {
                        if *p < l {
                            break;
                        }
                    }
                    total += ids.iter().filter_map(|id| self.orders.get(id)).map(|o| o.remaining()).sum::<f64>();
                }
            }
        }
        total
    }

    /// Submit a new order. `order_id` must already be globally unique (assigned by
    /// the caller/Market) and `order.seq` is assigned here for time priority.
    #[allow(clippy::too_many_arguments)]
    pub fn submit(&mut self, mut order: Order, time: SimTime) -> SubmitResult {
        let mut events = Vec::new();
        let mut fills = Vec::new();

        order.seq = self.next_seq;
        self.next_seq += 1;
        order.status = OrderStatus::Open;

        events.push(OrderEvent {
            time,
            order_id: order.id,
            agent_id: order.agent_id,
            asset_id: order.asset_id,
            kind: OrderEventKind::Submit,
            side: order.side,
            price_ticks: order.price_ticks,
            qty: order.qty,
        });

        // Post-only: reject outright if it would cross.
        if order.post_only {
            if let Some(p) = order.price_ticks {
                if self.would_cross(order.side, p) {
                    order.status = OrderStatus::Rejected;
                    events.push(OrderEvent {
                        time,
                        order_id: order.id,
                        agent_id: order.agent_id,
                        asset_id: order.asset_id,
                        kind: OrderEventKind::Reject,
                        side: order.side,
                        price_ticks: order.price_ticks,
                        qty: order.qty,
                    });
                    return SubmitResult { fills, events };
                }
            }
        }

        // FOK pre-check: must be fully fillable right now, or reject with zero fills.
        if order.tif == TimeInForce::Fok {
            let available = self.reachable_liquidity(order.side, order.price_ticks);
            if available + 1e-12 < order.qty {
                order.status = OrderStatus::Rejected;
                events.push(OrderEvent {
                    time,
                    order_id: order.id,
                    agent_id: order.agent_id,
                    asset_id: order.asset_id,
                    kind: OrderEventKind::Reject,
                    side: order.side,
                    price_ticks: order.price_ticks,
                    qty: order.qty,
                });
                return SubmitResult { fills, events };
            }
        }

        let opposite_side = order.side.opposite();

        loop {
            if order.remaining() <= 1e-12 {
                break;
            }
            let best_opp = match order.side {
                Side::Buy => self.best_ask_ticks(),
                Side::Sell => self.best_bid_ticks(),
            };
            let Some(opp_price) = best_opp else { break };

            let marketable = match order.order_type {
                OrderType::Market => true,
                OrderType::Limit => match order.side {
                    Side::Buy => order.price_ticks.unwrap() >= opp_price,
                    Side::Sell => order.price_ticks.unwrap() <= opp_price,
                },
            };
            if !marketable {
                break;
            }

            let maker_id = {
                let level = self.book_mut(opposite_side).get_mut(&opp_price).unwrap();
                *level.front().unwrap()
            };
            let fill_qty = {
                let maker = self.orders.get(&maker_id).unwrap();
                order.remaining().min(maker.remaining())
            };

            // Execute fill at the resting (maker) price -- standard price-time-priority
            // continuous double auction semantics.
            {
                let maker = self.orders.get_mut(&maker_id).unwrap();
                maker.qty_filled += fill_qty;
                maker.status = if maker.remaining() <= 1e-12 {
                    OrderStatus::Filled
                } else {
                    OrderStatus::PartiallyFilled
                };
            }
            order.qty_filled += fill_qty;
            order.status = if order.remaining() <= 1e-12 {
                OrderStatus::Filled
            } else {
                OrderStatus::PartiallyFilled
            };

            let (maker_agent, maker_done) = {
                let maker = self.orders.get(&maker_id).unwrap();
                (maker.agent_id, maker.is_done())
            };

            fills.push(Fill {
                asset_id: self.asset_id,
                price_ticks: opp_price,
                qty: fill_qty,
                maker_order_id: maker_id,
                maker_agent_id: maker_agent,
                taker_order_id: order.id,
                taker_agent_id: order.agent_id,
                taker_side: order.side,
                time,
            });
            let fill_kind = if maker_done {
                OrderEventKind::Fill
            } else {
                OrderEventKind::PartialFill
            };
            events.push(OrderEvent {
                time,
                order_id: maker_id,
                agent_id: maker_agent,
                asset_id: self.asset_id,
                kind: fill_kind,
                side: opposite_side,
                price_ticks: Some(opp_price),
                qty: fill_qty,
            });

            if maker_done {
                self.unlink(opposite_side, opp_price, maker_id);
                self.orders.remove(&maker_id);
            }
        }

        let taker_fill_kind = if order.qty_filled > 1e-12 {
            if order.remaining() <= 1e-12 {
                Some(OrderEventKind::Fill)
            } else {
                Some(OrderEventKind::PartialFill)
            }
        } else {
            None
        };
        if let Some(kind) = taker_fill_kind {
            events.push(OrderEvent {
                time,
                order_id: order.id,
                agent_id: order.agent_id,
                asset_id: order.asset_id,
                kind,
                side: order.side,
                price_ticks: order.price_ticks,
                qty: order.qty_filled,
            });
        }

        // Post-match disposition of any remainder.
        if order.remaining() > 1e-12 {
            match (order.order_type, order.tif) {
                (OrderType::Limit, TimeInForce::Gtc) => {
                    self.rest(order);
                }
                _ => {
                    // Market orders never rest; IOC/FOK remainders are discarded.
                    let remainder = order.remaining();
                    order.status = if order.qty_filled > 1e-12 {
                        OrderStatus::PartiallyFilled
                    } else {
                        OrderStatus::Cancelled
                    };
                    events.push(OrderEvent {
                        time,
                        order_id: order.id,
                        agent_id: order.agent_id,
                        asset_id: order.asset_id,
                        kind: OrderEventKind::Expire,
                        side: order.side,
                        price_ticks: order.price_ticks,
                        qty: remainder,
                    });
                    // Terminal (non-resting) order: not retained in `self.orders`, mirroring
                    // real order books. Callers get everything they need via `fills`/`events`.
                }
            }
        }
        // else: order fully filled without ever resting -- nothing to retain either.

        SubmitResult { fills, events }
    }
}
