// SPDX-License-Identifier: MIT OR Apache-2.0

//! Board and card game environments for phase 2: Connect Four,
//! Othello, Go 9x9 and Leduc hold'em.
//!
//! Environment ids and versions follow SPEC section 16 and ADR-0023.

pub mod connect4;
pub mod go9;
pub mod leduc;
pub mod othello;

pub use connect4::{Connect4, Connect4Config, Connect4Obs, Connect4State};
pub use go9::{Go9, Go9Config, Go9Obs, Go9State};
pub use leduc::{Leduc, LeducConfig, LeducInfo, LeducObs, LeducState};
pub use othello::{Othello, OthelloConfig, OthelloObs, OthelloState};
