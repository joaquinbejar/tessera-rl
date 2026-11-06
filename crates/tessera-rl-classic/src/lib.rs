// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic environments: CartPole, Tag, the grid worlds and snake.
//!
//! Environment ids and versions follow SPEC section 16 and ADR-0023.

pub mod cartpole;
pub mod grid;
pub mod keydoor;
pub mod snake;
pub mod tag;

pub use cartpole::{CartPole, CartPoleConfig, CartPoleObs, CartPoleState};
pub use grid::{Grid, GridConfig, GridObs, GridState};
pub use keydoor::{KeyDoor, KeyDoorConfig, KeyDoorObs, KeyDoorState};
pub use snake::{Snake, SnakeConfig, SnakeObs, SnakeState};
pub use tag::{Tag, TagConfig, TagObs, TagState};
