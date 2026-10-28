//! Matching-engine correctness tests: price-time priority, partial fills, TIF/
//! post-only edge cases. Per the spec's non-functional requirements.

use sim_core::order::{Order, OrderStatus};
use sim_core::orderbook::OrderBook;
use sim_core::types::{OrderType, Side, TimeInForce};

fn limit_order(id: u64, agent: u32, side: Side, price_ticks: i64, qty: f64, tif: TimeInForce) -> Order {
    Order {
        id,
        agent_id: agent,
        asset_id: 0,
        side,
        order_type: OrderType::Limit,
        tif,
        post_only: false,
        price_ticks: Some(price_ticks),
        qty,
        qty_filled: 0.0,
        status: OrderStatus::Open,
        submitted_at: 0.0,
        seq: 0,
    }
}

fn market_order(id: u64, agent: u32, side: Side, qty: f64, tif: TimeInForce) -> Order {
    Order {
        id,
        agent_id: agent,
        asset_id: 0,
        side,
        order_type: OrderType::Market,
        tif,
        post_only: false,
        price_ticks: None,
        qty,
        qty_filled: 0.0,
        status: OrderStatus::Open,
        submitted_at: 0.0,
        seq: 0,
    }
}

#[test]
fn price_priority_best_bid_fills_first() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    // Two resting bids at different prices; a market sell should hit the higher one.
    book.submit(limit_order(1, 1, Side::Buy, 10_000, 1.0, TimeInForce::Gtc), 0.0, None);
    book.submit(limit_order(2, 2, Side::Buy, 10_001, 1.0, TimeInForce::Gtc), 0.0, None);

    let result = book.submit(market_order(3, 3, Side::Sell, 1.0, TimeInForce::Ioc), 1.0, None);
    assert_eq!(result.fills.len(), 1);
    assert_eq!(result.fills[0].maker_order_id, 2, "higher bid (order 2) must fill before the lower one");
    assert_eq!(result.fills[0].price_ticks, 10_001);
}

#[test]
fn time_priority_within_same_price_level() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Buy, 10_000, 1.0, TimeInForce::Gtc), 0.0, None);
    book.submit(limit_order(2, 2, Side::Buy, 10_000, 1.0, TimeInForce::Gtc), 0.0, None);

    let result = book.submit(market_order(3, 3, Side::Sell, 1.0, TimeInForce::Ioc), 1.0, None);
    assert_eq!(result.fills[0].maker_order_id, 1, "earlier order at the same price must fill first");
}

#[test]
fn partial_fill_across_two_makers_at_same_level() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Buy, 10_000, 0.5, TimeInForce::Gtc), 0.0, None);
    book.submit(limit_order(2, 2, Side::Buy, 10_000, 0.5, TimeInForce::Gtc), 0.0, None);

    let result = book.submit(market_order(3, 3, Side::Sell, 0.7, TimeInForce::Ioc), 1.0, None);
    assert_eq!(result.fills.len(), 2);
    assert_eq!(result.fills[0].maker_order_id, 1);
    assert!((result.fills[0].qty - 0.5).abs() < 1e-9, "first maker fully consumed");
    assert_eq!(result.fills[1].maker_order_id, 2);
    assert!((result.fills[1].qty - 0.2).abs() < 1e-9, "second maker only partially filled");

    // The partially filled maker should still be resting with its remainder.
    let remaining = book.orders.get(&2).expect("partially filled order stays on book").remaining();
    assert!((remaining - 0.3).abs() < 1e-9);
}

#[test]
fn ioc_cancels_unfilled_remainder_without_resting() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 0.3, TimeInForce::Gtc), 0.0, None);

    let result = book.submit(market_order(2, 2, Side::Buy, 1.0, TimeInForce::Ioc), 1.0, None);
    assert!((result.fills[0].qty - 0.3).abs() < 1e-9);
    assert!(book.best_ask_ticks().is_none(), "maker fully consumed");
    assert!(!book.orders.contains_key(&2), "IOC remainder must not rest on the book");
}

#[test]
fn fok_rejects_entirely_when_not_fully_fillable() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 0.3, TimeInForce::Gtc), 0.0, None);

    let result = book.submit(limit_order(2, 2, Side::Buy, 10_000, 1.0, TimeInForce::Fok), 1.0, None);
    assert!(result.fills.is_empty(), "FOK must not partially fill");
    assert!(!book.orders.contains_key(&2));
    // The resting sell order must be completely untouched.
    let resting = book.orders.get(&1).unwrap();
    assert!((resting.remaining() - 0.3).abs() < 1e-9);
}

#[test]
fn fok_fills_completely_when_liquidity_suffices() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 0.6, TimeInForce::Gtc), 0.0, None);
    book.submit(limit_order(2, 2, Side::Sell, 10_001, 0.6, TimeInForce::Gtc), 0.0, None);

    let result = book.submit(limit_order(3, 3, Side::Buy, 10_001, 1.0, TimeInForce::Fok), 1.0, None);
    let filled: f64 = result.fills.iter().map(|f| f.qty).sum();
    assert!((filled - 1.0).abs() < 1e-9);
}

#[test]
fn post_only_rejects_when_it_would_cross() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 1.0, TimeInForce::Gtc), 0.0, None);

    let mut crossing = limit_order(2, 2, Side::Buy, 10_000, 1.0, TimeInForce::Gtc);
    crossing.post_only = true;
    let result = book.submit(crossing, 1.0, None);
    assert!(result.fills.is_empty(), "post-only must never execute");
    assert!(!book.orders.contains_key(&2));
    // The resting sell is untouched.
    assert!((book.best_ask().unwrap() - 100.0).abs() < 1e-6);
}

#[test]
fn post_only_rests_when_it_does_not_cross() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 1.0, TimeInForce::Gtc), 0.0, None);

    let mut non_crossing = limit_order(2, 2, Side::Buy, 9_999, 1.0, TimeInForce::Gtc);
    non_crossing.post_only = true;
    let result = book.submit(non_crossing, 1.0, None);
    assert!(result.fills.is_empty());
    assert!(book.orders.contains_key(&2), "non-crossing post-only order must rest");
    assert_eq!(book.best_bid_ticks(), Some(9_999));
}

#[test]
fn cancel_removes_order_from_book_and_matching() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Buy, 10_000, 1.0, TimeInForce::Gtc), 0.0, None);
    let ev = book.cancel(1, 1.0);
    assert!(ev.is_some());
    assert!(book.best_bid_ticks().is_none());

    // A subsequent sell must find no counterparty.
    let result = book.submit(market_order(2, 2, Side::Sell, 1.0, TimeInForce::Ioc), 2.0, None);
    assert!(result.fills.is_empty());
}

#[test]
fn market_order_sweeps_multiple_price_levels() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 0.4, TimeInForce::Gtc), 0.0, None);
    book.submit(limit_order(2, 2, Side::Sell, 10_001, 0.4, TimeInForce::Gtc), 0.0, None);
    book.submit(limit_order(3, 3, Side::Sell, 10_002, 0.4, TimeInForce::Gtc), 0.0, None);

    let result = book.submit(market_order(4, 4, Side::Buy, 1.0, TimeInForce::Ioc), 1.0, None);
    assert_eq!(result.fills.len(), 3);
    assert!((result.fills[0].qty - 0.4).abs() < 1e-9);
    assert!((result.fills[1].qty - 0.4).abs() < 1e-9);
    assert!((result.fills[2].qty - 0.2).abs() < 1e-9);
    assert_eq!(result.fills[0].price_ticks, 10_000);
    assert_eq!(result.fills[2].price_ticks, 10_002);
}

#[test]
fn circuit_breaker_stops_a_market_order_from_sweeping_beyond_the_price_band() {
    // 15% band around a mark price of 100.0 (10_000 ticks). Best ask starts at
    // 10_000 ticks; a second, much worse ask sits at 12_000 ticks (+20% from the
    // mark, outside the band). A large market buy should only take the first
    // level and then stop, leaving the rest of its IOC quantity unfilled, rather
    // than sweeping into the 20%-away level.
    let mut book = OrderBook::new(0, 0.01, 0.15);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 1.0, TimeInForce::Gtc), 0.0, None);
    book.submit(limit_order(2, 2, Side::Sell, 12_000, 1.0, TimeInForce::Gtc), 0.0, None);

    let result = book.submit(market_order(3, 3, Side::Buy, 2.0, TimeInForce::Ioc), 1.0, Some(100.0));
    assert_eq!(result.fills.len(), 1, "must not cross into the level beyond the band");
    assert_eq!(result.fills[0].price_ticks, 10_000);
    assert!((result.fills[0].qty - 1.0).abs() < 1e-9);
    // The far level must still be resting, untouched.
    assert_eq!(book.best_ask_ticks(), Some(12_000));
}

#[test]
fn circuit_breaker_allows_a_fill_within_the_band() {
    let mut book = OrderBook::new(0, 0.01, 0.15);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 1.0, TimeInForce::Gtc), 0.0, None);
    book.submit(limit_order(2, 2, Side::Sell, 11_000, 1.0, TimeInForce::Gtc), 0.0, None); // +10% from the 100.0 mark, inside the band

    let result = book.submit(market_order(3, 3, Side::Buy, 2.0, TimeInForce::Ioc), 1.0, Some(100.0));
    assert_eq!(result.fills.len(), 2, "a level within the band must still be reachable");
    let total: f64 = result.fills.iter().map(|f| f.qty).sum();
    assert!((total - 2.0).abs() < 1e-9);
}

#[test]
fn circuit_breaker_uses_the_passed_mark_price_not_the_books_own_best_price() {
    // Regression test for two bugs found and fixed in turn on 2026-10-24 (direct
    // user request: "fine tune the simulation so the worth of the stocks...
    // represent the real world"):
    //
    // (1) The circuit breaker's reference used to be derived from the book's OWN
    // best opposing price at the start of each order's matching -- always
    // "wherever the book currently is". A rapid sequence of separate orders
    // could each individually stay within the band relative to that fresh
    // starting point, yet compound past it cumulatively, since every new order
    // got its own fresh reference at wherever the previous one left off.
    // Measured: a real human_80pct run printed a +230% BTC move within one 500s
    // window this way -- a burst of `momentum`/`mean_reversion` taker orders
    // arriving right as `market_maker`'s own depth-collapse panic trigger had
    // pulled its quotes, each individual sweep legal under the per-order band,
    // nothing capping their sum.
    //
    // (2) The first fix tried was a periodic, order-book-local anchor refreshed
    // every `price_protection_window_secs` instead of every order. That stopped
    // single-window compounding but introduced a *worse*, structural bug:
    // whichever side of the book trades more often (in this engine's
    // population, buys, since every agent starts at zero inventory and must buy
    // before it can ever sell) saturates its own side of the band quickly and
    // then sits capped for the rest of the window, while the rarer,
    // un-saturated side keeps drifting the price unopposed. Measured: every one
    // of 7 otherwise-varied seeds ground 57-95% lower within one ~2.78-hour
    // window this way -- a uniform structural ratchet, not genuine variation.
    //
    // The actual fix: a reference the book doesn't control or reset at all.
    // `Market::submit` now passes in its own continuously-EMA-smoothed
    // `mark_price_ema`, external to the book, which can't get stuck favoring
    // either side since it drifts toward wherever price actually trades, in
    // *either* direction, every tick -- the same style real exchanges use for
    // mark-price-based price protection (e.g. Binance).
    //
    // Proof this is really wired up (not silently falling back to the book's
    // own best price): a resting ask at 100.0 with a deliberately stale/lagging
    // mark price of 80.0 passed in -- 80.0 * 1.15 = 92.0, below the 100.0 ask --
    // must block the order even though this is the very first order to ever
    // touch this level (nothing for a book-local reference to have "already
    // reset" against; only an externally-supplied reference can explain the
    // block).
    let mut book = OrderBook::new(0, 0.01, 0.15);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 1.0, TimeInForce::Gtc), 0.0, None);

    let result = book.submit(market_order(2, 2, Side::Buy, 1.0, TimeInForce::Ioc), 1.0, Some(80.0));
    assert!(result.fills.is_empty(), "a stale mark price's band must block even the very first touch of this level");
    assert_eq!(book.best_ask_ticks(), Some(10_000), "the untouched ask must still be resting");

    // The same resting ask is reachable once the mark price has caught up.
    let result2 = book.submit(market_order(3, 3, Side::Buy, 1.0, TimeInForce::Ioc), 2.0, Some(95.0));
    assert_eq!(result2.fills.len(), 1, "a mark price close enough to the ask must let the order through");
}

#[test]
fn fok_precheck_agrees_with_the_circuit_breaker() {
    // Same setup as the sweep test: only 1.0 unit is actually reachable within the
    // band, so an FOK order for 2.0 must be rejected outright (not partially filled,
    // and not accepted on a stale liquidity count that ignores the band).
    let mut book = OrderBook::new(0, 0.01, 0.15);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 1.0, TimeInForce::Gtc), 0.0, None);
    book.submit(limit_order(2, 2, Side::Sell, 12_000, 1.0, TimeInForce::Gtc), 0.0, None);

    let result = book.submit(market_order(3, 3, Side::Buy, 2.0, TimeInForce::Fok), 1.0, Some(100.0));
    assert!(result.fills.is_empty(), "FOK must reject rather than partially fill");
}

fn assert_depth_close(actual: Vec<(f64, f64)>, expected: &[(f64, f64)]) {
    assert_eq!(actual.len(), expected.len(), "actual={actual:?} expected={expected:?}");
    for (&(ap, aq), &(ep, eq)) in actual.iter().zip(expected) {
        assert!((ap - ep).abs() < 1e-6, "price: actual={actual:?} expected={expected:?}");
        assert!((aq - eq).abs() < 1e-9, "qty: actual={actual:?} expected={expected:?}");
    }
}

#[test]
fn depth_reflects_every_kind_of_book_mutation() {
    // `depth()` had no direct test coverage before this; walks it through resting,
    // aggregation at a shared level, cancel, partial fill, and full fill in turn.
    let mut book = OrderBook::new(0, 0.01, 1e9);

    assert_eq!(book.depth(Side::Buy, 10), vec![]);

    // New resting orders at two price levels.
    book.submit(limit_order(1, 1, Side::Buy, 10_000, 1.0, TimeInForce::Gtc), 0.0, None);
    book.submit(limit_order(2, 2, Side::Buy, 9_999, 2.0, TimeInForce::Gtc), 0.0, None);
    assert_depth_close(book.depth(Side::Buy, 10), &[(100.0, 1.0), (99.99, 2.0)]);

    // A second order joining an existing level aggregates into it.
    book.submit(limit_order(3, 3, Side::Buy, 10_000, 0.5, TimeInForce::Gtc), 0.0, None);
    assert_depth_close(book.depth(Side::Buy, 10), &[(100.0, 1.5), (99.99, 2.0)]);

    // Cancelling order 1 leaves only order 3's 0.5 at the top level.
    book.cancel(1, 1.0);
    assert_depth_close(book.depth(Side::Buy, 10), &[(100.0, 0.5), (99.99, 2.0)]);

    // A partial fill against the top level reduces it in place.
    let r = book.submit(market_order(4, 4, Side::Sell, 0.3, TimeInForce::Ioc), 2.0, None);
    assert_eq!(r.fills.len(), 1);
    assert_depth_close(book.depth(Side::Buy, 10), &[(100.0, 0.2), (99.99, 2.0)]);

    // Fully filling what's left at the top level empties it, removing the level.
    let r = book.submit(market_order(5, 5, Side::Sell, 0.2, TimeInForce::Ioc), 3.0, None);
    assert_eq!(r.fills.len(), 1);
    assert_depth_close(book.depth(Side::Buy, 10), &[(99.99, 2.0)]);

    // `levels` slices correctly either way.
    assert_depth_close(book.depth(Side::Buy, 1), &[(99.99, 2.0)]);
    assert_depth_close(book.depth(Side::Buy, 100), &[(99.99, 2.0)]);
}

#[test]
fn mid_price_and_spread() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Buy, 10_000, 1.0, TimeInForce::Gtc), 0.0, None);
    book.submit(limit_order(2, 2, Side::Sell, 10_010, 1.0, TimeInForce::Gtc), 0.0, None);
    assert!((book.mid_price().unwrap() - 100.05).abs() < 1e-6);
    assert!((book.spread().unwrap() - 0.10).abs() < 1e-6);
}
