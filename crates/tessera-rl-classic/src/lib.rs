// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic environments: CartPole, Tag, the grid worlds, snake, paddle duel and brick wall.
//!
//! Environment ids and versions follow SPEC section 16 and ADR-0023.

pub mod brickwall;
pub mod cartpole;
pub mod grid;
pub mod keydoor;
pub mod paddleduel;
pub mod snake;
pub mod tag;

pub use brickwall::{BrickWall, BrickWallConfig, BrickWallObs, BrickWallState};
pub use cartpole::{CartPole, CartPoleConfig, CartPoleObs, CartPoleState};
pub use grid::{Grid, GridConfig, GridObs, GridState};
pub use keydoor::{KeyDoor, KeyDoorConfig, KeyDoorObs, KeyDoorState};
pub use paddleduel::{PaddleDuel, PaddleDuelConfig, PaddleDuelObs, PaddleDuelState};
pub use snake::{Snake, SnakeConfig, SnakeObs, SnakeState};
pub use tag::{Tag, TagConfig, TagObs, TagState};
