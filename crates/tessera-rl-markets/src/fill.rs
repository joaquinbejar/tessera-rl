// SPDX-License-Identifier: MIT OR Apache-2.0

//! A trade between an incoming order and a resting one, and the fixed
//! buffer the book reports them into.

#![deny(clippy::float_arithmetic)]

use tessera_rl_core::Reserved;

use crate::limits::FILLS_CAP;
use crate::order::Side;

/// One fill, at the resting (maker) order's price.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Fill {
    /// Price, in ticks.
    pub price: u32,
    /// Lots traded.
    pub qty: u32,
    /// The id of the resting order.
    pub maker_seq: u32,
    /// The resting order's owner.
    pub maker: u8,
    /// The incoming order's owner.
    pub taker: u8,
    /// The incoming order's side: the aggressor side.
    pub taker_side: Side,
    /// Zero.
    pub pad: Reserved<1>,
}

impl Fill {
    /// The buying agent.
    #[must_use]
    #[inline]
    pub const fn buyer(&self) -> u8 {
        match self.taker_side {
            Side::Buy => self.taker,
            Side::Sell => self.maker,
        }
    }

    /// The selling agent.
    #[must_use]
    #[inline]
    pub const fn seller(&self) -> u8 {
        match self.taker_side {
            Side::Buy => self.maker,
            Side::Sell => self.taker,
        }
    }

    /// `price x qty`, in tick-lots; at most `2^26`.
    #[must_use]
    #[inline]
    pub fn notional(&self) -> i64 {
        // At most 2^16 x 2^10: no overflow.
        i64::from(self.price) * i64::from(self.qty)
    }
}

/// The fills of one incoming order. Holds [`FILLS_CAP`] fills, the most
/// one order can produce, so the book never drops one; no allocation.
#[derive(Debug, Clone)]
pub struct Fills {
    items: [Fill; FILLS_CAP],
    len: usize,
}

impl Default for Fills {
    fn default() -> Self {
        Self::new()
    }
}

impl Fills {
    /// An empty buffer.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self {
            items: [Fill {
                price: 0,
                qty: 0,
                maker_seq: 0,
                maker: 0,
                taker: 0,
                taker_side: Side::Buy,
                pad: Reserved::ZERO,
            }; FILLS_CAP],
            len: 0,
        }
    }

    /// Empties the buffer.
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Appends `fill`; `false` if the buffer is full (never for the fills
    /// of one order).
    #[inline]
    pub(crate) fn push(&mut self, fill: Fill) -> bool {
        match self.items.get_mut(self.len) {
            Some(slot) => {
                *slot = fill;
                self.len += 1;
                true
            }
            None => false,
        }
    }

    /// The fills, in the order they happened.
    #[must_use]
    #[inline]
    pub fn as_slice(&self) -> &[Fill] {
        self.items.get(..self.len).unwrap_or(&[])
    }
}
