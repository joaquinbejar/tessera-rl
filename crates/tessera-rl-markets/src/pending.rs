// SPDX-License-Identifier: MIT OR Apache-2.0

//! The latency queue (`docs/reviews/381/DESIGN.md`, section 1): a submitted
//! order or cancel waits here until step `ready`, then the environment
//! releases it to the book. Items are released in arrival order: the order
//! the environment pushed them, which within a step is the seeded arrival
//! permutation (section 2), never the storage index.

#![deny(clippy::float_arithmetic)]

use tessera_rl_core::{Pool, Reserved};
use tessera_rl_macros::EnvState;

use crate::error::PendingError;
use crate::exposure::Exposure;
use crate::limits::{MAX_AGENTS, PENDING_CAP};
use crate::order::{Kind, Side};

/// One queued item, 24 bytes of plain data.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Pending {
    /// Limit, market or cancel.
    pub kind: Kind,
    /// The order's side; `Buy` for a cancel.
    pub side: Side,
    /// The submitting agent slot.
    pub owner: u8,
    /// Zero.
    pub pad: Reserved<1>,
    /// Limit price, in ticks; 0 for market orders and cancels.
    pub price: u32,
    /// Quantity, in lots; 0 for cancels.
    pub qty: u32,
    /// The id of the resting order to cancel; 0 otherwise.
    pub target: u32,
    /// The step at which it is released.
    pub ready: u32,
    /// Arrival rank: the queue's counter when it was pushed.
    pub arrival: u32,
}

const _: () = assert!(size_of::<Pending>() == 24);

impl Pending {
    /// A limit order.
    #[must_use]
    #[inline]
    pub const fn limit(owner: u8, side: Side, price: u32, qty: u32) -> Self {
        Self {
            kind: Kind::Limit,
            side,
            owner,
            pad: Reserved::ZERO,
            price,
            qty,
            target: 0,
            ready: 0,
            arrival: 0,
        }
    }

    /// A market order.
    #[must_use]
    #[inline]
    pub const fn market(owner: u8, side: Side, qty: u32) -> Self {
        Self {
            kind: Kind::Market,
            ..Self::limit(owner, side, 0, qty)
        }
    }

    /// A cancel of the resting order with id `target`.
    #[must_use]
    #[inline]
    pub const fn cancel(owner: u8, target: u32) -> Self {
        Self {
            kind: Kind::Cancel,
            target,
            ..Self::limit(owner, Side::Buy, 0, 0)
        }
    }
}

/// The queue: at most [`PENDING_CAP`] items and the arrival counter.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct PendingQueue {
    items: Pool<Pending, PENDING_CAP>,
    next_arrival: u32,
}

impl PendingQueue {
    /// An empty queue.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queued items.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether nothing is queued.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Every queued item, in pool index order (not arrival order).
    #[inline]
    pub fn items(&self) -> impl Iterator<Item = &Pending> {
        self.items.iter().map(|(_, p)| p)
    }

    /// Queues `item` for release at step `ready`, after every item already
    /// queued. `item.ready` and `item.arrival` are overwritten.
    ///
    /// # Errors
    ///
    /// [`PendingError`] for an invalid owner, a full queue or an exhausted
    /// arrival counter; the queue is then unchanged.
    pub fn push(&mut self, item: Pending, ready: u32) -> Result<(), PendingError> {
        if usize::from(item.owner) >= MAX_AGENTS {
            return Err(PendingError::Owner(item.owner));
        }
        let Some(after) = self.next_arrival.checked_add(1) else {
            return Err(PendingError::ArrivalExhausted);
        };
        let queued = Pending {
            ready,
            arrival: self.next_arrival,
            ..item
        };
        self.items.insert(queued).ok_or(PendingError::Full)?;
        self.next_arrival = after;
        Ok(())
    }

    /// Removes and returns the earliest-arrived item whose `ready` step is
    /// at most `now`, or `None` if none is due. Call it until `None` to
    /// release a step's items in arrival order.
    #[inline]
    pub fn pop_ready(&mut self, now: u32) -> Option<Pending> {
        let index = self
            .items
            .iter()
            .filter(|(_, p)| p.ready <= now)
            .min_by_key(|(_, p)| p.arrival)
            .map(|(i, _)| i)?;
        self.items.remove(index)
    }

    /// Items queued by `owner`, of every kind.
    #[must_use]
    #[inline]
    pub fn count(&self, owner: u8) -> usize {
        self.items().filter(|p| p.owner == owner).count()
    }

    /// The commitment of `owner`'s queued orders (cancels count for
    /// nothing): limit orders as resting orders at their price, market
    /// orders as lots only, a market buy's cash at `market_buy_price` per
    /// lot (review r1 of #381, correction 2: the caller picks the bound,
    /// for example `max_tick`). `None` on overflow, which the queue's
    /// capacity cannot reach.
    #[must_use]
    pub fn exposure(&self, owner: u8, market_buy_price: u32) -> Option<Exposure> {
        self.items()
            .filter(|p| p.owner == owner)
            .try_fold(Exposure::default(), |e, p| match p.kind {
                Kind::Limit => e.with_order(p.side, true, p.qty, p.price),
                Kind::Market => e.with_order(p.side, false, p.qty, market_buy_price),
                Kind::Cancel => Some(e),
            })
    }

    /// Drops every item of `owner` (bankruptcy); returns how many.
    #[inline]
    pub fn drop_owner(&mut self, owner: u8) -> usize {
        let mut n = 0;
        for index in 0..PENDING_CAP {
            if self.items.get(index).is_some_and(|p| p.owner == owner) {
                self.items.remove(index);
                n += 1;
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::{Pending, PendingQueue};
    use crate::error::PendingError;
    use crate::limits::PENDING_CAP;
    use crate::order::{Kind, Side};

    #[test]
    fn test_pending_size_is_twenty_four_bytes() {
        assert_eq!(size_of::<Pending>(), 24);
    }

    #[test]
    fn test_pending_pop_ready_in_arrival_order_not_index_order() {
        let mut q = PendingQueue::new();
        assert_eq!(q.push(Pending::market(0, Side::Buy, 1), 2), Ok(()));
        assert_eq!(q.push(Pending::cancel(1, 9), 1), Ok(()));
        assert_eq!(q.push(Pending::limit(2, Side::Sell, 5, 3), 1), Ok(()));
        // The next push reuses index 1, below owner 2's index 2, but arrives last.
        assert_eq!(
            q.pop_ready(1).map(|p| (p.owner, p.kind)),
            Some((1, Kind::Cancel))
        );
        assert_eq!(q.push(Pending::limit(3, Side::Buy, 4, 1), 1), Ok(()));
        assert_eq!(q.pop_ready(1).map(|p| p.owner), Some(2));
        assert_eq!(q.pop_ready(1).map(|p| p.owner), Some(3));
        assert_eq!(q.pop_ready(1), None);
        assert_eq!(q.pop_ready(2).map(|p| (p.owner, p.arrival)), Some((0, 0)));
        assert!(q.is_empty());
    }

    #[test]
    fn test_pending_full_queue_refuses_unchanged() {
        let mut q = PendingQueue::new();
        for i in 0..PENDING_CAP {
            let owner = u8::try_from(i % 16).unwrap_or(0);
            assert_eq!(q.push(Pending::market(owner, Side::Buy, 1), 0), Ok(()));
        }
        let before = q;
        assert_eq!(
            q.push(Pending::market(0, Side::Buy, 1), 0),
            Err(PendingError::Full)
        );
        assert_eq!(q, before);
        assert_eq!(
            q.push(Pending::market(16, Side::Buy, 1), 0),
            Err(PendingError::Owner(16))
        );
        assert_eq!(q.drop_owner(3), PENDING_CAP / 16);
        assert_eq!(q.count(3), 0);
    }

    #[test]
    fn test_pending_exposure_counts_limits_and_bounds_market_buys() {
        let mut q = PendingQueue::new();
        assert_eq!(q.push(Pending::limit(4, Side::Buy, 10, 2), 1), Ok(()));
        assert_eq!(q.push(Pending::market(4, Side::Buy, 3), 1), Ok(()));
        assert_eq!(q.push(Pending::market(4, Side::Sell, 5), 1), Ok(()));
        assert_eq!(q.push(Pending::cancel(4, 0), 1), Ok(()));
        assert_eq!(q.push(Pending::limit(5, Side::Buy, 10, 7), 1), Ok(()));
        let e = q.exposure(4, 100);
        assert_eq!(
            e.map(|e| (e.orders, e.lots, e.buy_cash)),
            Some(([1, 0], [5, 5], 2 * 10 + 3 * 100))
        );
        assert_eq!(q.count(4), 4);
    }
}
