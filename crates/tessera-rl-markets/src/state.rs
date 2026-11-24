// SPDX-License-Identifier: MIT OR Apache-2.0

//! The plain-data state of `tessera/LimitOrderBook-v1` (ADR-0029) and the
//! public prices every agent can read from it.

#![deny(clippy::float_arithmetic)]

use tessera_rl_core::{Bool8, Reserved};
use tessera_rl_macros::EnvState;

use crate::account::Account;
use crate::book::Book;
use crate::limits::MAX_AGENTS;
use crate::order::Side;
use crate::pending::PendingQueue;

/// Fills of one step kept for the `trades` observation leaf.
pub const MAX_TRADES: usize = 16;

/// One fill of this step, as every agent sees it: anonymous.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Trade {
    /// Price, in ticks.
    pub price: u32,
    /// Lots.
    pub qty: u32,
    /// The aggressor (incoming order's) side.
    pub side: Side,
    /// Zero.
    pub pad: Reserved<3>,
}

const _: () = assert!(size_of::<Trade>() == 12);

/// One instance. The all-zero default is the state before the first reset.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct LobState {
    /// Cash and inventory per agent slot.
    pub accounts: [Account; MAX_AGENTS],
    /// Equity marked at the mid after the previous step (or the reset),
    /// per slot: the reward is the change from it.
    pub last_equity: [i64; MAX_AGENTS],
    /// The resting orders.
    pub book: Book,
    /// Orders and cancels waiting out the latency.
    pub pending: PendingQueue,
    /// The first [`MAX_TRADES`] fills of the last step, in order.
    pub trades: [Trade; MAX_TRADES],
    /// The fundamental value, in ticks.
    pub fundamental: u32,
    /// Steps played in this episode.
    pub step: u32,
    /// Price of the last fill of the episode; 0 before the first. Read by
    /// the observation only, never by the mid.
    pub last_trade: u32,
    /// Entries of `trades` in use.
    pub trade_count: u32,
    /// The fundamental at reset: the reference price before any trade.
    pub reset_price: u32,
    /// Whether each slot is alive (bankruptcy ends a slot).
    pub alive: [Bool8; MAX_AGENTS],
    /// The mid at the end of the previous step (the initial price at
    /// reset): the mid while one side of the book is empty. 0 before the
    /// first reset.
    pub last_mid: u32,
}

const _: () = assert!(size_of::<LobState>().is_multiple_of(8));

/// The public prices at one moment, read by masks, the action decoder and
/// the observation (`docs/reviews/381/DESIGN.md`, section 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Touch {
    /// The best bid.
    pub bid: Option<u32>,
    /// The best ask.
    pub ask: Option<u32>,
    /// The mid: `(bid + ask) / 2` rounded down when both sides quote,
    /// otherwise the mid at the end of the previous step, otherwise (before
    /// the first reset) the fundamental at reset. Never the fundamental of
    /// a later step: rewards and bankruptcy are marked at it.
    pub mid: u32,
}

impl LobState {
    /// The touch and mid of the current book.
    #[must_use]
    pub fn touch(&self) -> Touch {
        let bid = self.book.best(Side::Buy);
        let ask = self.book.best(Side::Sell);
        let mid = match (bid, ask) {
            // Both at most 2^16: the sum fits u32.
            (Some(b), Some(a)) => u32::midpoint(b, a),
            _ if self.last_mid > 0 => self.last_mid,
            _ => self.reset_price,
        };
        Touch { bid, ask, mid }
    }

    /// Whether `slot` is alive.
    #[must_use]
    #[inline]
    pub fn is_alive(&self, slot: usize) -> bool {
        self.alive.get(slot).is_some_and(|a| a.get())
    }

    /// The account of `slot`, or an empty one out of range.
    #[must_use]
    #[inline]
    pub fn account(&self, slot: usize) -> Account {
        self.accounts.get(slot).copied().unwrap_or_default()
    }

    /// The equity of `slot` marked at the mid ([`Touch::mid`]), in
    /// tick-lots. Rewards and bankruptcy use it, so neither reveals the
    /// hidden fundamental before settlement.
    #[must_use]
    #[inline]
    pub fn mark(&self, slot: usize) -> i64 {
        self.account(slot).equity(self.touch().mid)
    }
}

impl Touch {
    /// The reference price of a limit order on `side`: the best price of
    /// that side, or the mid when it is empty.
    #[must_use]
    #[inline]
    pub fn base(&self, side: Side) -> u32 {
        match side {
            Side::Buy => self.bid,
            Side::Sell => self.ask,
        }
        .unwrap_or(self.mid)
    }

    /// `price - mid`, in ticks.
    #[must_use]
    #[inline]
    pub fn offset(&self, price: u32) -> i32 {
        // Both at most 2^16: the difference fits i32.
        i32::try_from(i64::from(price) - i64::from(self.mid)).unwrap_or(0)
    }
}
