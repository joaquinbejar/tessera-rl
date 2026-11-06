// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase 2 M0 reference tools for Connect Four (ADR-0042), as a library
//! for the CLI in `main.rs` and for the equivalence tests of the Connect
//! Four environment (P2-M1b). The pinned modules are unchanged; only this
//! file exposes them.

#![expect(
    clippy::cast_possible_wrap,
    reason = "board quantities (plies, cells, scores) are at most 42 and fit in i32"
)]

pub mod best_response;
pub mod board;
pub mod bot;
pub mod calibrate;
pub mod openings;
pub mod rng;
pub mod solver;
