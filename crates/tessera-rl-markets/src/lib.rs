// SPDX-License-Identifier: MIT OR Apache-2.0

//! The market environments of phase 4 (SPEC 16): `tessera/LimitOrderBook-v1`,
//! a continuous double auction with price-time priority and up to 16
//! trading agents (`docs/reviews/381/DESIGN.md`).
//!
//! This crate is built in stages (design section 9). Stage 1, this one: the
//! configuration ([`config`]), the order record ([`order`]), the book and
//! its matching ([`book`]), fills and accounts ([`fill`], [`account`]),
//! the worst-case commitment the masks will check ([`exposure`]), the
//! latency queue ([`pending`]) and the fundamental's random walk
//! ([`fundamental`]). The `Environment` comes in stage 2.
//!
//! Everything here is integer arithmetic: no float enters a transition
//! (ADR-0008), which `clippy::float_arithmetic` enforces per module.

pub mod account;
pub mod book;
pub mod config;
pub mod error;
pub mod exposure;
pub mod fill;
pub mod fundamental;
pub mod limits;
pub mod order;
pub mod pending;

pub use account::{Account, apply_fill};
pub use book::{Book, Level, Outcome, Sweep};
pub use config::LobConfig;
pub use error::{BookError, PendingError};
pub use exposure::Exposure;
pub use fill::{Fill, Fills};
pub use fundamental::RandomWalk;
pub use order::{Kind, Order, Side};
pub use pending::{Pending, PendingQueue};
