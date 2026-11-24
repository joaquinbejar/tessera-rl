// SPDX-License-Identifier: MIT OR Apache-2.0

//! The limit order book: price-time priority matching, immediate-or-cancel
//! market orders, cancels, self-trade prevention and the best levels
//! (`docs/reviews/381/DESIGN.md`, section 1, rule 4).
//!
//! The book is a `Pool<Order, BOOK_CAP>` matched by linear scans: no sorted
//! index, so the state is one plain array that snapshots in one `memcpy`.
//! No call allocates; fills go into a caller-provided [`Fills`] buffer.
//!
//! Every refusal ([`BookError`]) is decided before any mutation. Within a
//! call, arithmetic is checked or bounded by the constants of
//! [`crate::limits`]: a price is at most `2^16`, a quantity at most `2^10`,
//! and the book holds at most `BOOK_CAP` orders, so per-level lots fit
//! `u32` and costs fit `u64`.

#![deny(clippy::float_arithmetic)]

use tessera_rl_core::{Pool, Reserved};
use tessera_rl_macros::EnvState;

use crate::error::BookError;
use crate::exposure::Exposure;
use crate::fill::{Fill, Fills};
use crate::limits::{BOOK_CAP, MAX_AGENTS, MAX_LOT, MAX_OPEN, MAX_PRICE};
use crate::order::{Order, Side};

/// One aggregated price level: anonymous, as the observation shows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Level {
    /// Price, in ticks.
    pub price: u32,
    /// Lots resting at that price, summed over orders.
    pub lots: u32,
}

/// What one incoming order did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Outcome {
    /// Lots filled against resting orders; the fills are in [`Fills`].
    pub filled: u32,
    /// The id (`seq`) of the remainder if it rested; limit orders only.
    pub rested: Option<u32>,
    /// Lots dropped: the unfilled remainder of a market order.
    pub dropped: u32,
    /// The owner's resting orders cancelled by self-trade prevention.
    pub stp_orders: u32,
    /// Their lots.
    pub stp_lots: u32,
}

/// The cost of sweeping the opposite side for a market order, without
/// mutating the book (review r1 of #381, correction 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Sweep {
    /// Lots the visible levels can fill, at most the requested quantity.
    pub lots: u32,
    /// What those lots cost, in tick-lots.
    pub cost: u64,
    /// The worst price reached; `None` when nothing fills.
    pub worst: Option<u32>,
}

impl Sweep {
    /// The cost of `qty` lots when the lots the visible levels cannot fill
    /// are priced at `bound` (for a buy, `max_tick` is an upper bound on
    /// any fill): `cost + (qty - lots) x bound`. `None` on overflow, or if
    /// `qty` is below the swept lots (a sweep of a larger quantity).
    #[must_use]
    #[inline]
    pub fn cost_bounded(&self, qty: u32, bound: u32) -> Option<u64> {
        let rest = u64::from(qty.checked_sub(self.lots)?);
        rest.checked_mul(u64::from(bound))?.checked_add(self.cost)
    }

    /// The worst price a market order of `qty` lots could reach: the worst
    /// visible price if the visible levels fill it, otherwise `bound`.
    #[must_use]
    #[inline]
    pub fn worst_bounded(&self, qty: u32, bound: u32) -> u32 {
        match self.worst {
            Some(worst) if self.lots >= qty => worst,
            _ => bound,
        }
    }
}

/// The book: resting orders of both sides and the id counter.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Book {
    orders: Pool<Order, BOOK_CAP>,
    next_seq: u32,
}

/// Whether `price` is a valid price.
const fn valid_price(price: u32) -> bool {
    price >= 1 && price <= MAX_PRICE
}

/// Whether `qty` is a valid order quantity.
const fn valid_qty(qty: u32) -> bool {
    qty >= 1 && qty <= MAX_LOT
}

impl Book {
    /// An empty book whose first order id is `0`.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Resting orders on both sides.
    #[must_use]
    #[inline]
    pub fn len(&self) -> usize {
        self.orders.len()
    }

    /// Whether no order rests.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }

    /// The id the next resting order gets.
    #[must_use]
    #[inline]
    pub const fn next_seq(&self) -> u32 {
        self.next_seq
    }

    /// Every resting order, in pool index order (not priority order).
    #[inline]
    pub fn orders(&self) -> impl Iterator<Item = &Order> {
        self.orders.iter().map(|(_, o)| o)
    }

    /// The resting order with id `seq`, if any.
    #[must_use]
    #[inline]
    pub fn order(&self, seq: u32) -> Option<&Order> {
        self.orders().find(|o| o.seq == seq)
    }

    /// The pool index of the best resting order on `side`: best price,
    /// then lowest `seq`.
    fn best_index(&self, side: Side) -> Option<usize> {
        let mut best: Option<(usize, &Order)> = None;
        for (i, o) in self.orders.iter() {
            if o.side != side {
                continue;
            }
            match best {
                Some((_, b)) if !o.precedes(b) => {}
                _ => best = Some((i, o)),
            }
        }
        best.map(|(i, _)| i)
    }

    /// The best price on `side`: the highest bid or the lowest ask.
    #[must_use]
    #[inline]
    pub fn best(&self, side: Side) -> Option<u32> {
        self.next_level(side, None, None).map(|l| l.price)
    }

    /// The best level on `side` strictly behind price `after` (`None`: the
    /// best level), skipping the orders of `skip`.
    fn next_level(&self, side: Side, after: Option<u32>, skip: Option<u8>) -> Option<Level> {
        let mut price: Option<u32> = None;
        for o in self.orders() {
            if o.side != side || skip == Some(o.owner) {
                continue;
            }
            if after.is_some_and(|a| !side.better(a, o.price)) {
                continue;
            }
            if price.is_none_or(|p| side.better(o.price, p)) {
                price = Some(o.price);
            }
        }
        let price = price?;
        let mut lots: u32 = 0;
        for o in self.orders() {
            if o.side == side && o.price == price && skip != Some(o.owner) {
                // At most BOOK_CAP x MAX_LOT = 2^17 lots: fits u32.
                lots += o.qty;
            }
        }
        Some(Level { price, lots })
    }

    /// Writes the best `out.len()` levels of `side` into `out`, best first,
    /// and returns how many exist; the entries past that are reset to
    /// `Level::default()`.
    #[inline]
    pub fn levels(&self, side: Side, out: &mut [Level]) -> usize {
        let mut after = None;
        let mut n = 0;
        for slot in out.iter_mut() {
            match self.next_level(side, after, None) {
                Some(level) => {
                    *slot = level;
                    after = Some(level.price);
                    n += 1;
                }
                None => *slot = Level::default(),
            }
        }
        n
    }

    /// What a market order of `qty` lots on `taker` by `owner` would fill
    /// within the best `max_levels` opposite levels, skipping the owner's
    /// own orders (self-trade prevention cancels them instead of filling
    /// them). The book is not changed.
    #[must_use]
    pub fn sweep(&self, taker: Side, owner: u8, qty: u32, max_levels: usize) -> Sweep {
        let side = taker.opposite();
        let mut out = Sweep::default();
        let mut after = None;
        for _ in 0..max_levels {
            if out.lots >= qty {
                break;
            }
            let Some(level) = self.next_level(side, after, Some(owner)) else {
                break;
            };
            // `out.lots < qty` here, and every cost is below
            // 2^17 lots x 2^16 ticks: plain arithmetic cannot overflow.
            let take = level.lots.min(qty - out.lots);
            out.lots += take;
            out.cost += u64::from(take) * u64::from(level.price);
            out.worst = Some(level.price);
            after = Some(level.price);
        }
        out
    }

    /// The resting orders of `owner` on both sides, by id (oldest first),
    /// written into `out`; returns how many. This is the order the cancel
    /// head indexes.
    #[inline]
    pub fn own_orders(&self, owner: u8, out: &mut [Order; MAX_OPEN]) -> usize {
        let mut n = 0;
        for o in self.orders().filter(|o| o.owner == owner) {
            // More than MAX_OPEN orders is excluded by `limit`.
            debug_assert!(
                n < MAX_OPEN,
                "owner {owner} holds more than MAX_OPEN orders"
            );
            let Some(dst) = out.get_mut(n) else {
                break;
            };
            *dst = *o;
            // Insertion by id; at most MAX_OPEN entries.
            let mut at = n;
            while at > 0 && out.get(at - 1).is_some_and(|p| p.seq > o.seq) {
                out.swap(at - 1, at);
                at -= 1;
            }
            n += 1;
        }
        for dst in out.iter_mut().skip(n) {
            *dst = Order::default();
        }
        n
    }

    /// Resting orders of `owner`, both sides.
    #[must_use]
    #[inline]
    pub fn open_count(&self, owner: u8) -> usize {
        self.orders().filter(|o| o.owner == owner).count()
    }

    /// The commitment of `owner`'s resting orders: per side, the orders
    /// and their lots, and the cash of the bids at their prices. `None` on
    /// overflow, which `MAX_OPEN` resting orders cannot reach.
    #[must_use]
    pub fn exposure(&self, owner: u8) -> Option<Exposure> {
        self.orders()
            .filter(|o| o.owner == owner)
            .try_fold(Exposure::default(), |e, o| {
                e.with_order(o.side, true, o.qty, o.price)
            })
    }

    /// Checks the common fields of an incoming order.
    fn check(owner: u8, qty: u32) -> Result<(), BookError> {
        if usize::from(owner) >= MAX_AGENTS {
            return Err(BookError::Owner(owner));
        }
        if !valid_qty(qty) {
            return Err(BookError::Qty(qty));
        }
        Ok(())
    }

    /// Submits a limit order of `owner` for `qty` lots at `price` on
    /// `side`: it fills against the best opposite orders while they cross,
    /// each at the resting price; the remainder rests with the next id.
    /// The owner's own crossing orders are cancelled instead of filled
    /// (self-trade prevention). `fills` is cleared, then receives this
    /// order's fills in order.
    ///
    /// # Errors
    ///
    /// [`BookError`] for an invalid owner, price or quantity, an owner
    /// already holding `MAX_OPEN` resting orders, or an exhausted id
    /// counter; the book and `fills` are then unchanged.
    pub fn limit(
        &mut self,
        owner: u8,
        side: Side,
        price: u32,
        qty: u32,
        fills: &mut Fills,
    ) -> Result<Outcome, BookError> {
        Self::check(owner, qty)?;
        if !valid_price(price) {
            return Err(BookError::Price(price));
        }
        if self.open_count(owner) >= MAX_OPEN {
            return Err(BookError::OpenLimit(owner));
        }
        let Some(after) = self.next_seq.checked_add(1) else {
            return Err(BookError::SeqExhausted);
        };
        fills.clear();
        let mut out = self.take(owner, side, Some(price), qty, fills);
        // `take` fills at most `qty`.
        let remaining = qty - out.filled;
        if remaining > 0 {
            let order = Order {
                price,
                qty: remaining,
                seq: self.next_seq,
                owner,
                side,
                pad: Reserved::ZERO,
            };
            // The owner holds fewer than MAX_OPEN orders and every owner is
            // below MAX_AGENTS, so a slot is free (BOOK_CAP = 16 x 8).
            let inserted = self.orders.insert(order).is_some();
            debug_assert!(inserted, "the book overflowed");
            if inserted {
                out.rested = Some(self.next_seq);
                self.next_seq = after;
            } else {
                out.dropped = remaining;
            }
        }
        Ok(out)
    }

    /// Submits a market order of `owner` for `qty` lots on `side`:
    /// immediate or cancel. It fills against the best opposite orders at
    /// their prices until filled or the side is empty; the remainder is
    /// dropped. Self-trade prevention as for [`Book::limit`]. `fills` is
    /// cleared, then receives this order's fills.
    ///
    /// # Errors
    ///
    /// [`BookError`] for an invalid owner or quantity; the book and
    /// `fills` are then unchanged.
    pub fn market(
        &mut self,
        owner: u8,
        side: Side,
        qty: u32,
        fills: &mut Fills,
    ) -> Result<Outcome, BookError> {
        Self::check(owner, qty)?;
        fills.clear();
        let mut out = self.take(owner, side, None, qty, fills);
        out.dropped = qty - out.filled;
        Ok(out)
    }

    /// Submits an immediate-or-cancel order of `owner` for `qty` lots on
    /// `side` with protection price `limit`: it fills like [`Book::limit`]
    /// while the opposite orders cross `limit`, and the remainder is
    /// dropped instead of resting. A market order with a price cap. No id
    /// is used and `MAX_OPEN` does not apply, since nothing rests.
    ///
    /// # Errors
    ///
    /// [`BookError`] for an invalid owner, price or quantity; the book and
    /// `fills` are then unchanged.
    pub fn ioc(
        &mut self,
        owner: u8,
        side: Side,
        limit: u32,
        qty: u32,
        fills: &mut Fills,
    ) -> Result<Outcome, BookError> {
        Self::check(owner, qty)?;
        if !valid_price(limit) {
            return Err(BookError::Price(limit));
        }
        fills.clear();
        let mut out = self.take(owner, side, Some(limit), qty, fills);
        out.dropped = qty - out.filled;
        Ok(out)
    }

    /// The matching loop of an incoming order: `limit` is `None` for a
    /// market order. Bounded: every iteration removes a resting order or
    /// ends the loop, so at most `BOOK_CAP + 1` iterations.
    fn take(
        &mut self,
        owner: u8,
        side: Side,
        limit: Option<u32>,
        qty: u32,
        fills: &mut Fills,
    ) -> Outcome {
        let mut out = Outcome::default();
        let mut remaining = qty;
        for _ in 0..=BOOK_CAP {
            if remaining == 0 {
                break;
            }
            let Some(index) = self.best_index(side.opposite()) else {
                break;
            };
            let Some(resting) = self.orders.get_mut(index) else {
                break;
            };
            if limit.is_some_and(|l| !side.crosses(l, resting.price)) {
                break;
            }
            if resting.owner == owner {
                let lots = resting.qty;
                self.orders.remove(index);
                // At most BOOK_CAP orders of MAX_LOT lots.
                out.stp_orders += 1;
                out.stp_lots += lots;
                continue;
            }
            let lots = remaining.min(resting.qty);
            let fill = Fill {
                price: resting.price,
                qty: lots,
                maker_seq: resting.seq,
                maker: resting.owner,
                taker: owner,
                taker_side: side,
                pad: Reserved::ZERO,
            };
            resting.qty -= lots;
            let emptied = resting.qty == 0;
            if emptied {
                self.orders.remove(index);
            }
            remaining -= lots;
            out.filled += lots;
            let pushed = fills.push(fill);
            debug_assert!(pushed, "more than FILLS_CAP fills from one order");
        }
        out
    }

    /// Cancels `owner`'s resting order `seq`. Returns it, or `None` if no
    /// such order rests (already filled, cancelled, or another owner's).
    #[inline]
    pub fn cancel(&mut self, owner: u8, seq: u32) -> Option<Order> {
        let index = self
            .orders
            .iter()
            .find(|(_, o)| o.seq == seq && o.owner == owner)
            .map(|(i, _)| i)?;
        self.orders.remove(index)
    }

    /// Rewrites the owner of every resting order, in pool index order, to
    /// `owner(order)`. Prices, sizes, ids and priority are unchanged. Used
    /// to anonymize an information state and to determinize one; the
    /// caller keeps the per-owner limits.
    #[inline]
    pub fn set_owners(&mut self, mut owner: impl FnMut(&Order) -> u8) {
        for (_, o) in self.orders.iter_mut() {
            o.owner = owner(o);
        }
    }

    /// Cancels every resting order of `owner` (bankruptcy); returns how
    /// many.
    #[inline]
    pub fn cancel_all(&mut self, owner: u8) -> usize {
        let mut n = 0;
        for index in 0..BOOK_CAP {
            if self.orders.get(index).is_some_and(|o| o.owner == owner) {
                self.orders.remove(index);
                n += 1;
            }
        }
        n
    }

    /// Whether the book is crossed: a bid at or above an ask. Never true
    /// after any call; exposed for tests and debug checks.
    #[must_use]
    #[inline]
    pub fn is_crossed(&self) -> bool {
        match (self.best(Side::Buy), self.best(Side::Sell)) {
            (Some(bid), Some(ask)) => bid >= ask,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Book, Level, Outcome, Sweep};
    use crate::error::BookError;
    use crate::fill::Fills;
    use crate::limits::{MAX_LOT, MAX_OPEN, MAX_PRICE};
    use crate::order::{Order, Side};

    fn rest(book: &mut Book, owner: u8, side: Side, price: u32, qty: u32) -> u32 {
        let mut f = Fills::new();
        let out = book.limit(owner, side, price, qty, &mut f);
        assert_eq!(f.as_slice(), &[]);
        out.ok().and_then(|o| o.rested).unwrap_or(u32::MAX)
    }

    #[test]
    fn test_book_limit_rests_without_cross() {
        let mut b = Book::new();
        assert_eq!(rest(&mut b, 0, Side::Buy, 10, 3), 0);
        assert_eq!(rest(&mut b, 1, Side::Sell, 12, 4), 1);
        assert_eq!(b.best(Side::Buy), Some(10));
        assert_eq!(b.best(Side::Sell), Some(12));
        assert!(!b.is_crossed());
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn test_book_limit_fills_price_time_priority() {
        let mut b = Book::new();
        let first = rest(&mut b, 1, Side::Sell, 10, 2);
        let second = rest(&mut b, 2, Side::Sell, 10, 2);
        let better = rest(&mut b, 3, Side::Sell, 9, 1);
        let mut f = Fills::new();
        let out = b.limit(0, Side::Buy, 10, 4, &mut f);
        assert_eq!(
            out,
            Ok(Outcome {
                filled: 4,
                ..Outcome::default()
            })
        );
        let got: Vec<_> = f
            .as_slice()
            .iter()
            .map(|x| (x.maker_seq, x.price, x.qty))
            .collect();
        assert_eq!(got, [(better, 9, 1), (first, 10, 2), (second, 10, 1)]);
        assert_eq!(b.order(second).map(|o| o.qty), Some(1));
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_book_limit_remainder_rests_after_partial_fill() {
        let mut b = Book::new();
        rest(&mut b, 1, Side::Sell, 10, 2);
        let mut f = Fills::new();
        let out = b.limit(0, Side::Buy, 11, 5, &mut f);
        assert_eq!(out.map(|o| (o.filled, o.rested.is_some())), Ok((2, true)));
        assert_eq!(b.best(Side::Buy), Some(11));
        assert_eq!(b.best(Side::Sell), None);
        assert!(!b.is_crossed());
    }

    #[test]
    fn test_book_market_walks_levels_and_drops_remainder() {
        let mut b = Book::new();
        rest(&mut b, 1, Side::Buy, 10, 2);
        rest(&mut b, 2, Side::Buy, 8, 3);
        let mut f = Fills::new();
        let out = b.market(0, Side::Sell, 7, &mut f);
        assert_eq!(
            out,
            Ok(Outcome {
                filled: 5,
                dropped: 2,
                ..Outcome::default()
            })
        );
        assert_eq!(f.as_slice().len(), 2);
        assert!(b.is_empty());
    }

    #[test]
    fn test_book_market_on_empty_side_fills_nothing() {
        let mut b = Book::new();
        let mut f = Fills::new();
        let out = b.market(0, Side::Buy, 3, &mut f);
        assert_eq!(out.map(|o| (o.filled, o.dropped)), Ok((0, 3)));
    }

    #[test]
    fn test_book_self_trade_cancels_own_resting_order() {
        let mut b = Book::new();
        let own = rest(&mut b, 0, Side::Sell, 10, 2);
        let other = rest(&mut b, 1, Side::Sell, 11, 2);
        let mut f = Fills::new();
        let out = b.limit(0, Side::Buy, 11, 1, &mut f);
        assert_eq!(
            out,
            Ok(Outcome {
                filled: 1,
                stp_orders: 1,
                stp_lots: 2,
                ..Outcome::default()
            })
        );
        assert_eq!(b.order(own), None);
        assert_eq!(b.order(other).map(|o| o.qty), Some(1));
        assert_eq!(f.as_slice().first().map(|x| x.maker), Some(1));
    }

    #[test]
    fn test_book_self_trade_keeps_own_non_crossing_order() {
        let mut b = Book::new();
        let own = rest(&mut b, 0, Side::Sell, 12, 2);
        let mut f = Fills::new();
        let out = b.limit(0, Side::Buy, 11, 1, &mut f);
        assert_eq!(out.map(|o| o.stp_orders), Ok(0));
        assert!(b.order(own).is_some());
    }

    #[test]
    fn test_book_cancel_by_owner_and_id_only() {
        let mut b = Book::new();
        let id = rest(&mut b, 3, Side::Buy, 10, 1);
        assert_eq!(b.cancel(4, id), None);
        assert_eq!(b.cancel(3, id).map(|o| o.seq), Some(id));
        assert_eq!(b.cancel(3, id), None);
        assert!(b.is_empty());
    }

    #[test]
    fn test_book_open_limit_refuses_ninth_order_unchanged() {
        let mut b = Book::new();
        for i in 0..MAX_OPEN {
            let price = u32::try_from(i).map_or(1, |p| p + 1);
            rest(&mut b, 5, Side::Buy, price, 1);
        }
        let before = b;
        let mut f = Fills::new();
        assert_eq!(
            b.limit(5, Side::Buy, 1, 1, &mut f),
            Err(BookError::OpenLimit(5))
        );
        assert_eq!(b, before);
        assert_eq!(b.cancel_all(5), MAX_OPEN);
        assert!(b.is_empty());
    }

    #[test]
    fn test_book_refuses_invalid_inputs() {
        let mut b = Book::new();
        let mut f = Fills::new();
        assert_eq!(
            b.limit(16, Side::Buy, 1, 1, &mut f),
            Err(BookError::Owner(16))
        );
        assert_eq!(
            b.limit(0, Side::Buy, 0, 1, &mut f),
            Err(BookError::Price(0))
        );
        assert_eq!(
            b.limit(0, Side::Buy, MAX_PRICE + 1, 1, &mut f),
            Err(BookError::Price(MAX_PRICE + 1))
        );
        assert_eq!(b.limit(0, Side::Buy, 1, 0, &mut f), Err(BookError::Qty(0)));
        assert_eq!(
            b.market(0, Side::Buy, MAX_LOT + 1, &mut f),
            Err(BookError::Qty(MAX_LOT + 1))
        );
        assert!(b.is_empty());
    }

    #[test]
    fn test_book_levels_aggregate_best_first() {
        let mut b = Book::new();
        rest(&mut b, 0, Side::Buy, 10, 1);
        rest(&mut b, 1, Side::Buy, 10, 2);
        rest(&mut b, 2, Side::Buy, 8, 4);
        rest(&mut b, 3, Side::Buy, 9, 5);
        let mut out = [Level { price: 7, lots: 7 }; 4];
        assert_eq!(b.levels(Side::Buy, &mut out), 3);
        assert_eq!(
            out,
            [
                Level { price: 10, lots: 3 },
                Level { price: 9, lots: 5 },
                Level { price: 8, lots: 4 },
                Level::default(),
            ]
        );
        assert_eq!(b.levels(Side::Sell, &mut out), 0);
    }

    #[test]
    fn test_book_sweep_prices_market_order_and_skips_own() {
        let mut b = Book::new();
        rest(&mut b, 1, Side::Sell, 10, 2);
        rest(&mut b, 0, Side::Sell, 11, 5);
        rest(&mut b, 2, Side::Sell, 12, 2);
        rest(&mut b, 3, Side::Sell, 13, 2);
        let s = b.sweep(Side::Buy, 0, 3, 10);
        assert_eq!(
            s,
            Sweep {
                lots: 3,
                cost: 2 * 10 + 12,
                worst: Some(12),
            }
        );
        assert_eq!(s.worst_bounded(3, 99), 12);
        // Two visible levels hold 4 lots; the other 6 are priced at the bound.
        let short = b.sweep(Side::Buy, 0, 10, 2);
        assert_eq!(short.lots, 4);
        assert_eq!(short.cost_bounded(10, 100), Some(2 * 10 + 2 * 12 + 6 * 100));
        assert_eq!(short.worst_bounded(10, 100), 100);
        let before = b;
        let mut f = Fills::new();
        let out = b.market(0, Side::Buy, 3, &mut f);
        assert_eq!(out.map(|o| o.filled), Ok(s.lots));
        let cost: u64 = f
            .as_slice()
            .iter()
            .map(|x| u64::from(x.qty) * u64::from(x.price))
            .sum();
        assert_eq!(cost, s.cost);
        assert_ne!(b, before);
    }

    #[test]
    fn test_book_own_orders_sorted_by_id_and_exposure() {
        let mut b = Book::new();
        let first = rest(&mut b, 2, Side::Buy, 10, 3);
        rest(&mut b, 1, Side::Buy, 10, 1);
        let sell = rest(&mut b, 2, Side::Sell, 20, 4);
        let bid9 = rest(&mut b, 2, Side::Buy, 9, 2);
        b.cancel(2, first);
        let bid8 = rest(&mut b, 2, Side::Buy, 8, 1);
        let mut out = [Order::default(); MAX_OPEN];
        assert_eq!(b.own_orders(2, &mut out), 3);
        let ids: Vec<_> = out.iter().take(3).map(|o| o.seq).collect();
        assert_eq!(ids, [sell, bid9, bid8]);
        assert_eq!(out.get(3), Some(&Order::default()));
        let exposure = b.exposure(2);
        assert_eq!(
            exposure.map(|x| (x.orders, x.lots, x.buy_cash)),
            Some(([2, 1], [3, 4], 9 * 2 + 8))
        );
        assert_eq!(b.open_count(2), 3);
    }

    #[test]
    fn test_book_limit_price_bounds_accept_one_and_max_refuse_outside() {
        let mut b = Book::new();
        let mut f = Fills::new();
        assert!(b.limit(0, Side::Buy, 1, 1, &mut f).is_ok());
        assert!(b.limit(1, Side::Sell, MAX_PRICE, 1, &mut f).is_ok());
        for price in [0, MAX_PRICE + 1] {
            assert!(matches!(
                b.limit(2, Side::Buy, price, 1, &mut f),
                Err(BookError::Price(p)) if p == price
            ));
        }
    }
}
