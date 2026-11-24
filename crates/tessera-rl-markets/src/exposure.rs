// SPDX-License-Identifier: MIT OR Apache-2.0

//! The worst-case commitment of one agent's open and pending orders, which
//! the masks of PR 2 check against the position and cash limits (review r1
//! of #381, correction 1): an action is legal only if the limit holds even
//! when every open and pending order on that side fills.

#![deny(clippy::float_arithmetic)]

use crate::order::Side;

/// One agent's open or pending orders, summed per side. Per-side arrays
/// are indexed by [`Side::index`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Exposure {
    /// Orders that rest or may rest: resting orders, and pending limit
    /// orders. Pending market orders never rest and are not counted here.
    pub orders: [u32; 2],
    /// Lots that could still fill: resting and pending limit orders, and
    /// pending market orders.
    pub lots: [u32; 2],
    /// Cash, in tick-lots, that the buy lots above could cost: each limit
    /// at its price, each pending market buy at the caller's bound price.
    pub buy_cash: u64,
}

impl Exposure {
    /// Resting or possibly resting orders on both sides: the count that
    /// `MAX_OPEN` bounds.
    #[must_use]
    #[inline]
    pub fn total_orders(&self) -> u32 {
        self.orders.iter().sum()
    }

    /// Orders on `side`.
    #[must_use]
    #[inline]
    pub fn orders_on(&self, side: Side) -> u32 {
        self.orders.get(side.index()).copied().unwrap_or(0)
    }

    /// Lots on `side`.
    #[must_use]
    #[inline]
    pub fn lots_on(&self, side: Side) -> u32 {
        self.lots.get(side.index()).copied().unwrap_or(0)
    }

    /// The sum of two exposures, or `None` on overflow (which the config
    /// bounds exclude: at most `MAX_OPEN + PENDING_CAP` orders of
    /// `MAX_LOT` lots at `MAX_PRICE`).
    #[must_use]
    #[inline]
    pub fn checked_add(&self, other: &Self) -> Option<Self> {
        Some(Self {
            orders: [
                self.orders[0].checked_add(other.orders[0])?,
                self.orders[1].checked_add(other.orders[1])?,
            ],
            lots: [
                self.lots[0].checked_add(other.lots[0])?,
                self.lots[1].checked_add(other.lots[1])?,
            ],
            buy_cash: self.buy_cash.checked_add(other.buy_cash)?,
        })
    }

    /// Adds one order of `qty` lots on `side`, costing `price` per lot if
    /// it is a buy; `rests` says whether it counts in `orders`. `None` on
    /// overflow.
    #[must_use]
    #[inline]
    pub(crate) fn with_order(self, side: Side, rests: bool, qty: u32, price: u32) -> Option<Self> {
        let mut out = self;
        let i = side.index();
        if rests {
            let o = out.orders.get_mut(i)?;
            *o = o.checked_add(1)?;
        }
        let l = out.lots.get_mut(i)?;
        *l = l.checked_add(qty)?;
        if side == Side::Buy {
            let cost = u64::from(qty).checked_mul(u64::from(price))?;
            out.buy_cash = out.buy_cash.checked_add(cost)?;
        }
        Some(out)
    }
}
