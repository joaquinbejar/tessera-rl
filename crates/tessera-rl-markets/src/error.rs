// SPDX-License-Identifier: MIT OR Apache-2.0

//! Refusals of the book and the latency queue. Every refusal is decided
//! before any mutation, so a refused call leaves the state unchanged. The
//! environment's masks and validation boundary (PR 2) keep a legal step
//! from reaching them; they exist so that the book itself never panics and
//! never corrupts its state.

use thiserror::Error;

/// Why the book refused an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum BookError {
    /// The owner is not an agent slot (`>= MAX_AGENTS`).
    #[error("owner {0} is not an agent slot")]
    Owner(u8),
    /// The price is outside `[1, MAX_PRICE]`.
    #[error("price {0} is outside [1, MAX_PRICE]")]
    Price(u32),
    /// The quantity is outside `[1, MAX_LOT]`.
    #[error("quantity {0} is outside [1, MAX_LOT]")]
    Qty(u32),
    /// The owner already holds `MAX_OPEN` resting orders.
    #[error("owner {0} already holds MAX_OPEN resting orders")]
    OpenLimit(u8),
    /// The order id counter would wrap (the config bounds exclude it).
    #[error("the order id counter is exhausted")]
    SeqExhausted,
}

/// Why the latency queue refused an item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PendingError {
    /// The owner is not an agent slot (`>= MAX_AGENTS`).
    #[error("owner {0} is not an agent slot")]
    Owner(u8),
    /// The queue holds `PENDING_CAP` items.
    #[error("the latency queue is full")]
    Full,
    /// The arrival counter would wrap (the config bounds exclude it).
    #[error("the arrival counter is exhausted")]
    ArrivalExhausted,
}
