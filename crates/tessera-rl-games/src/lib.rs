// SPDX-License-Identifier: MIT OR Apache-2.0

//! Board and card game environments for phase 2: Connect Four.
//!
//! Environment ids and versions follow SPEC section 16 and ADR-0023.

pub mod connect4;

pub use connect4::{Connect4, Connect4Config, Connect4Obs, Connect4State};
