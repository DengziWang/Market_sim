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
    book.submit(limit_order(1, 1, Side::Buy, 10_000, 1.0, TimeInForce::Gtc), 0.0);
    book.submit(limit_order(2, 2, Side::Buy, 10_001, 1.0, TimeInForce::Gtc), 0.0);

    let result = book.submit(market_order(3, 3, Side::Sell, 1.0, TimeInForce::Ioc), 1.0);
    assert_eq!(result.fills.len(), 1);
    assert_eq!(result.fills[0].maker_order_id, 2, "higher bid (order 2) must fill before the lower one");
    assert_eq!(result.fills[0].price_ticks, 10_001);
}

#[test]
fn time_priority_within_same_price_level() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Buy, 10_000, 1.0, TimeInForce::Gtc), 0.0);
    book.submit(limit_order(2, 2, Side::Buy, 10_000, 1.0, TimeInForce::Gtc), 0.0);

    let result = book.submit(market_order(3, 3, Side::Sell, 1.0, TimeInForce::Ioc), 1.0);
    assert_eq!(result.fills[0].maker_order_id, 1, "earlier order at the same price must fill first");
}

#[test]
fn partial_fill_across_two_makers_at_same_level() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Buy, 10_000, 0.5, TimeInForce::Gtc), 0.0);
    book.submit(limit_order(2, 2, Side::Buy, 10_000, 0.5, TimeInForce::Gtc), 0.0);

    let result = book.submit(market_order(3, 3, Side::Sell, 0.7, TimeInForce::Ioc), 1.0);
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
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 0.3, TimeInForce::Gtc), 0.0);

    let result = book.submit(market_order(2, 2, Side::Buy, 1.0, TimeInForce::Ioc), 1.0);
    assert!((result.fills[0].qty - 0.3).abs() < 1e-9);
    assert!(book.best_ask_ticks().is_none(), "maker fully consumed");
    assert!(!book.orders.contains_key(&2), "IOC remainder must not rest on the book");
}

#[test]
fn fok_rejects_entirely_when_not_fully_fillable() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 0.3, TimeInForce::Gtc), 0.0);

    let result = book.submit(limit_order(2, 2, Side::Buy, 10_000, 1.0, TimeInForce::Fok), 1.0);
    assert!(result.fills.is_empty(), "FOK must not partially fill");
    assert!(!book.orders.contains_key(&2));
    // The resting sell order must be completely untouched.
    let resting = book.orders.get(&1).unwrap();
    assert!((resting.remaining() - 0.3).abs() < 1e-9);
}

#[test]
fn fok_fills_completely_when_liquidity_suffices() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 0.6, TimeInForce::Gtc), 0.0);
    book.submit(limit_order(2, 2, Side::Sell, 10_001, 0.6, TimeInForce::Gtc), 0.0);

    let result = book.submit(limit_order(3, 3, Side::Buy, 10_001, 1.0, TimeInForce::Fok), 1.0);
    let filled: f64 = result.fills.iter().map(|f| f.qty).sum();
    assert!((filled - 1.0).abs() < 1e-9);
}

#[test]
fn post_only_rejects_when_it_would_cross() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 1.0, TimeInForce::Gtc), 0.0);

    let mut crossing = limit_order(2, 2, Side::Buy, 10_000, 1.0, TimeInForce::Gtc);
    crossing.post_only = true;
    let result = book.submit(crossing, 1.0);
    assert!(result.fills.is_empty(), "post-only must never execute");
    assert!(!book.orders.contains_key(&2));
    // The resting sell is untouched.
    assert!((book.best_ask().unwrap() - 100.0).abs() < 1e-6);
}

#[test]
fn post_only_rests_when_it_does_not_cross() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 1.0, TimeInForce::Gtc), 0.0);

    let mut non_crossing = limit_order(2, 2, Side::Buy, 9_999, 1.0, TimeInForce::Gtc);
    non_crossing.post_only = true;
    let result = book.submit(non_crossing, 1.0);
    assert!(result.fills.is_empty());
    assert!(book.orders.contains_key(&2), "non-crossing post-only order must rest");
    assert_eq!(book.best_bid_ticks(), Some(9_999));
}

#[test]
fn cancel_removes_order_from_book_and_matching() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Buy, 10_000, 1.0, TimeInForce::Gtc), 0.0);
    let ev = book.cancel(1, 1.0);
    assert!(ev.is_some());
    assert!(book.best_bid_ticks().is_none());

    // A subsequent sell must find no counterparty.
    let result = book.submit(market_order(2, 2, Side::Sell, 1.0, TimeInForce::Ioc), 2.0);
    assert!(result.fills.is_empty());
}

#[test]
fn market_order_sweeps_multiple_price_levels() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 0.4, TimeInForce::Gtc), 0.0);
    book.submit(limit_order(2, 2, Side::Sell, 10_001, 0.4, TimeInForce::Gtc), 0.0);
    book.submit(limit_order(3, 3, Side::Sell, 10_002, 0.4, TimeInForce::Gtc), 0.0);

    let result = book.submit(market_order(4, 4, Side::Buy, 1.0, TimeInForce::Ioc), 1.0);
    assert_eq!(result.fills.len(), 3);
    assert!((result.fills[0].qty - 0.4).abs() < 1e-9);
    assert!((result.fills[1].qty - 0.4).abs() < 1e-9);
    assert!((result.fills[2].qty - 0.2).abs() < 1e-9);
    assert_eq!(result.fills[0].price_ticks, 10_000);
    assert_eq!(result.fills[2].price_ticks, 10_002);
}

#[test]
fn circuit_breaker_stops_a_market_order_from_sweeping_beyond_the_price_band() {
    // 15% band. Best ask starts at 10_000 ticks; a second, much worse ask sits at
    // 12_000 ticks (+20%, outside the band). A large market buy should only take the
    // first level and then stop, leaving the rest of its IOC quantity unfilled,
    // rather than sweeping into the 20%-away level.
    let mut book = OrderBook::new(0, 0.01, 0.15);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 1.0, TimeInForce::Gtc), 0.0);
    book.submit(limit_order(2, 2, Side::Sell, 12_000, 1.0, TimeInForce::Gtc), 0.0);

    let result = book.submit(market_order(3, 3, Side::Buy, 2.0, TimeInForce::Ioc), 1.0);
    assert_eq!(result.fills.len(), 1, "must not cross into the level beyond the band");
    assert_eq!(result.fills[0].price_ticks, 10_000);
    assert!((result.fills[0].qty - 1.0).abs() < 1e-9);
    // The far level must still be resting, untouched.
    assert_eq!(book.best_ask_ticks(), Some(12_000));
}

#[test]
fn circuit_breaker_allows_a_fill_within_the_band() {
    let mut book = OrderBook::new(0, 0.01, 0.15);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 1.0, TimeInForce::Gtc), 0.0);
    book.submit(limit_order(2, 2, Side::Sell, 11_000, 1.0, TimeInForce::Gtc), 0.0); // +10%, inside the band

    let result = book.submit(market_order(3, 3, Side::Buy, 2.0, TimeInForce::Ioc), 1.0);
    assert_eq!(result.fills.len(), 2, "a level within the band must still be reachable");
    let total: f64 = result.fills.iter().map(|f| f.qty).sum();
    assert!((total - 2.0).abs() < 1e-9);
}

#[test]
fn fok_precheck_agrees_with_the_circuit_breaker() {
    // Same setup as the sweep test: only 1.0 unit is actually reachable within the
    // band, so an FOK order for 2.0 must be rejected outright (not partially filled,
    // and not accepted on a stale liquidity count that ignores the band).
    let mut book = OrderBook::new(0, 0.01, 0.15);
    book.submit(limit_order(1, 1, Side::Sell, 10_000, 1.0, TimeInForce::Gtc), 0.0);
    book.submit(limit_order(2, 2, Side::Sell, 12_000, 1.0, TimeInForce::Gtc), 0.0);

    let result = book.submit(market_order(3, 3, Side::Buy, 2.0, TimeInForce::Fok), 1.0);
    assert!(result.fills.is_empty(), "FOK must reject rather than partially fill");
}

#[test]
fn mid_price_and_spread() {
    let mut book = OrderBook::new(0, 0.01, 1e9);
    book.submit(limit_order(1, 1, Side::Buy, 10_000, 1.0, TimeInForce::Gtc), 0.0);
    book.submit(limit_order(2, 2, Side::Sell, 10_010, 1.0, TimeInForce::Gtc), 0.0);
    assert!((book.mid_price().unwrap() - 100.05).abs() < 1e-6);
    assert!((book.spread().unwrap() - 0.10).abs() < 1e-6);
}
