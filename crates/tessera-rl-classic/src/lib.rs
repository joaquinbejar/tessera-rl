// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic environments: CartPole, Tag and the grid worlds.
//!
//! Environment ids and versions follow SPEC section 16 and ADR-0023.

pub mod cartpole;
pub mod grid;
pub mod keydoor;
pub mod tag;

pub use cartpole::{CartPole, CartPoleConfig, CartPoleObs, CartPoleState};
pub use grid::{Grid, GridConfig, GridObs, GridState};
pub use keydoor::{KeyDoor, KeyDoorConfig, KeyDoorObs, KeyDoorState};
pub use tag::{Tag, TagConfig, TagObs, TagState};
