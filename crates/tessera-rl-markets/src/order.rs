// SPDX-License-Identifier: MIT OR Apache-2.0

//! The resting order record and the two enums stored in state: [`Side`]
//! and [`Kind`] (`docs/reviews/381/DESIGN.md`, section 7).

#![deny(clippy::float_arithmetic)]

use tessera_rl_core::Reserved;
use tessera_rl_macros::{EnvState, StateEnum};

/// The side of an order. `Buy` is the default, so an empty pool entry is
/// all zero.
#[derive(StateEnum, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Side {
    /// A bid.
    #[default]
    Buy = 0,
    /// An ask.
    Sell = 1,
}

impl Side {
    /// Both sides, in index order.
    pub const ALL: [Self; 2] = [Self::Buy, Self::Sell];

    /// The other side.
    #[must_use]
    #[inline]
    pub const fn opposite(self) -> Self {
        match self {
            Self::Buy => Self::Sell,
            Self::Sell => Self::Buy,
        }
    }

    /// 0 for `Buy`, 1 for `Sell`: the index of per-side arrays.
    #[must_use]
    #[inline]
    pub const fn index(self) -> usize {
        match self {
            Self::Buy => 0,
            Self::Sell => 1,
        }
    }

    /// Whether a resting order at `resting` on this side has priority over
    /// one at `other` on the same side by price alone: the higher bid, the
    /// lower ask.
    #[must_use]
    #[inline]
    pub const fn better(self, resting: u32, other: u32) -> bool {
        match self {
            Self::Buy => resting > other,
            Self::Sell => resting < other,
        }
    }

    /// Whether an incoming order on this side with limit `limit` crosses a
    /// resting order at `resting` on the opposite side.
    #[must_use]
    #[inline]
    pub const fn crosses(self, limit: u32, resting: u32) -> bool {
        match self {
            Self::Buy => resting <= limit,
            Self::Sell => resting >= limit,
        }
    }
}

/// What an agent submitted: the item kind of the latency queue.
#[derive(StateEnum, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Kind {
    /// A limit order: fills while it crosses, the remainder rests.
    #[default]
    Limit = 0,
    /// A market order: immediate or cancel, the remainder is dropped.
    Market = 1,
    /// A cancel of one of the agent's resting orders, by id.
    Cancel = 2,
}

/// One resting order, 16 bytes of plain data (ADR-0029).
///
/// `seq` is the order id and its time priority: the book gives each order
/// that rests the next value of its counter, so ids are unique within an
/// episode and a lower `seq` arrived earlier.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Order {
    /// Limit price, in ticks, in `[1, MAX_PRICE]`.
    pub price: u32,
    /// Remaining quantity, in lots, in `[1, MAX_LOT]` while resting.
    pub qty: u32,
    /// Id and time priority.
    pub seq: u32,
    /// The agent slot that owns it.
    pub owner: u8,
    /// Bid or ask.
    pub side: Side,
    /// Zero.
    pub pad: Reserved<2>,
}

const _: () = assert!(size_of::<Order>() == 16);

impl Order {
    /// Whether `self` has priority over `other`, both on `self.side`:
    /// better price first, then lower `seq` (price-time priority).
    #[must_use]
    #[inline]
    pub const fn precedes(&self, other: &Self) -> bool {
        if self.price == other.price {
            self.seq < other.seq
        } else {
            self.side.better(self.price, other.price)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Kind, Order, Side};
    use tessera_rl_core::{Reserved, StateEnum};

    #[test]
    fn test_order_size_is_sixteen_bytes() {
        assert_eq!(size_of::<Order>(), 16);
        assert_eq!(align_of::<Order>(), 4);
    }

    #[test]
    fn test_order_default_is_all_zero() {
        let o = Order::default();
        assert_eq!(
            o,
            Order {
                price: 0,
                qty: 0,
                seq: 0,
                owner: 0,
                side: Side::Buy,
                pad: Reserved::ZERO,
            }
        );
    }

    #[test]
    fn test_side_repr_round_trips_and_rejects_unknown() {
        for s in Side::ALL {
            assert_eq!(Side::from_repr(s.to_repr()), Some(s));
        }
        assert_eq!(Side::from_repr(2), None);
        assert_eq!(Kind::from_repr(2), Some(Kind::Cancel));
        assert_eq!(Kind::from_repr(3), None);
    }

    #[test]
    fn test_order_precedes_price_then_seq() {
        let bid = |price, seq| Order {
            price,
            qty: 1,
            seq,
            side: Side::Buy,
            ..Order::default()
        };
        assert!(bid(11, 9).precedes(&bid(10, 1)));
        assert!(bid(10, 1).precedes(&bid(10, 2)));
        assert!(!bid(10, 2).precedes(&bid(10, 1)));
        let ask = |price, seq| Order {
            side: Side::Sell,
            ..bid(price, seq)
        };
        assert!(ask(10, 9).precedes(&ask(11, 1)));
    }

    #[test]
    fn test_side_crosses_at_equal_price() {
        assert!(Side::Buy.crosses(10, 10));
        assert!(!Side::Buy.crosses(10, 11));
        assert!(Side::Sell.crosses(10, 10));
        assert!(!Side::Sell.crosses(10, 9));
    }
}
