// SPDX-License-Identifier: MIT OR Apache-2.0

//! The RTS environment of phase 4, `tessera/RTS-v1`: two teams on a square
//! grid, entity-centric orders and simultaneous turns.
//!
//! This crate is built in stages (`docs/reviews/rts-design.md`, section 10).
//! Stage 2: the grid ([`world`]), the seeded map generator ([`mapgen`]),
//! the unit table ([`units`]), the move resolution of a tick ([`resolve`]),
//! the flow fields ([`flow`]), the environment itself ([`env`](mod@env))
//! and the scripted bots ([`bots`]). Stage 3: fog of war and explored
//! memory ([`fog`]) and the fogged observation.
//!
//! Everything here is integer arithmetic: no float enters a transition
//! (ADR-0008), which `clippy::float_arithmetic` enforces per module.

pub mod bots;
pub mod env;
pub mod flow;
pub mod fog;
pub mod mapgen;
pub mod resolve;
pub mod units;
pub mod world;

pub use env::{Rts, RtsConfig, RtsObs, RtsState};
