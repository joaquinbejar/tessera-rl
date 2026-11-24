// SPDX-License-Identifier: MIT OR Apache-2.0

//! League machinery for Tessera (ADR-0026, ADR-0088).
//!
//! - [`Population`]: the policies a learner plays against, with dense
//!   [`PolicyId`]s in insertion order, never removed, only retired.
//! - [`Results`]: per-instance rings of `(PolicyId, Outcome)` match results,
//!   written by the instance's own worker without allocation.
//! - [`elo`] and [`trueskill`]: rating updates.
//! - [`Matchmaking`] and [`Table`]: weights over the population, converted to
//!   cumulative `u64` thresholds that a single integer draw indexes.
//! - [`League`]: the state a barrier updates. It applies results in instance
//!   order, then oldest-first within an instance, inside the canonical
//!   floating-point scope, then recomputes the table.
//!
//! Everything here changes only at barriers, outside the per-instance
//! determinism contract unless frozen (ADR-0009): for the same inputs and
//! the same barrier sequence the state is the same bits, at any worker
//! count and on every platform of the tier.

mod error;
mod league;
mod matchmaking;
mod outcome;
mod population;
mod results;

pub mod elo;
pub mod trueskill;

pub use error::LeagueError;
pub use league::{League, LeagueConfig, LeagueReport, PairCounts};
pub use matchmaking::{Matchmaking, PfspWeighting, Table};
pub use outcome::Outcome;
pub use population::{Policy, PolicyId, PolicyKind, Population};
pub use results::Results;

/// `ln(10)` as pinned `f64` bits, `0x40026BB1BBB55516` (ADR-0088 item 4.3).
pub const LN10: f64 = f64::from_bits(0x4002_6BB1_BBB5_5516);
