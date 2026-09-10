//! Event-driven scheduling core: a min-heap priority queue keyed by simulated time.
//! Agents are woken only when *their own* next-activation time arrives (or an event
//! targeted at them fires), never on a fixed global tick -- this is what lets a
//! session with thousands of agents but sparse, heterogeneous activation rates
//! (market-makers every ~0.1-1s, DCA bots every few minutes) run fast.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::types::{AgentId, AssetId, OrderId, SimTime, VenueId};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum EventKind {
    /// An agent's own timer fired: it gets a chance to act.
    AgentWake { agent_id: AgentId },
    /// A resting order, submitted with network latency, becomes visible/live on the
    /// book only at this time (models per-agent-class network delay).
    DelayedOrderArrival { agent_id: AgentId, payload_id: u64 },
    /// A pending order queued in the public mempool model actually lands on the book
    /// now, after its reveal delay. It was visible to MEV bots from the moment it was
    /// queued (immediately, like a real public mempool) until this event fires.
    MempoolExecute { venue_id: VenueId, payload_id: u64 },
    /// Periodic recomputation of the EMA mark price used for liquidation checks.
    MarkPriceTick { asset_id: AssetId },
    /// Periodic liquidation-eligibility sweep over leveraged positions.
    LiquidationSweep,
    /// Periodic orderbook snapshot export tick.
    SnapshotTick,
    /// End of the simulated session.
    EndOfSession,
    /// A specific staged partial-liquidation slice (TWAP-style liquidation execution).
    LiquidationSlice { agent_id: AgentId, asset_id: AssetId, slice_no: u32 },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScheduledEvent {
    pub time: SimTime,
    /// Insertion order, used as a deterministic tiebreaker for equal times so that
    /// results depend only on (seed, config), never on heap implementation details.
    pub seq: u64,
    pub kind: EventKind,
}

impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.seq == other.seq
    }
}
impl Eq for ScheduledEvent {}
impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed for a min-heap via std BinaryHeap (which is max-heap by default).
        other
            .time
            .partial_cmp(&self.time)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}
impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct EventQueue {
    heap: BinaryHeap<ScheduledEvent>,
    next_seq: u64,
}

impl EventQueue {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_seq: 0,
        }
    }

    pub fn push(&mut self, time: SimTime, kind: EventKind) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.heap.push(ScheduledEvent { time, seq, kind });
    }

    pub fn pop(&mut self) -> Option<ScheduledEvent> {
        self.heap.pop()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }
}

impl Default for EventQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Suppresses "unused" warnings for identifiers only referenced through EventKind
/// variants in early build stages; harmless once bots/market wire everything up.
#[allow(dead_code)]
type _Unused = OrderId;
