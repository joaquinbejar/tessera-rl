// SPDX-License-Identifier: MIT OR Apache-2.0

//! The RTS environment of phase 4, `tessera/RTS-v1`: two teams on a square
//! grid, entity-centric orders and simultaneous turns.
//!
//! This crate is built in stages (`docs/reviews/rts-design.md`, section 10).
//! Stage 2 starts here with the pieces that do not depend on the engine:
//! the grid ([`world`]), the seeded map generator ([`mapgen`]), the unit
//! table ([`units`]) and the move resolution of a tick ([`resolve`]). The
//! `Environment` itself, its masks and its bots follow in later changes.
//!
//! Everything here is integer arithmetic: no float enters a transition
//! (ADR-0008), which `clippy::float_arithmetic` enforces per module.

pub mod mapgen;
pub mod resolve;
pub mod units;
pub mod world;
