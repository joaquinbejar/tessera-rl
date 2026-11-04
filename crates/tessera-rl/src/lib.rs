// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Tessera
//!
//! Tessera is a Rust engine for writing and running reinforcement learning
//! environments, with cheap snapshot, restore and fork, bit-exact determinism
//! and a native multi-agent model. It is exposed to Python through Gymnasium,
//! PettingZoo and PufferLib compatible interfaces.
//!
//! This crate is the facade: it re-exports the workspace crates, following
//! the layered workspace layout of SPEC section 4.1 and ADR-0021.
//!
//! Most authors need only the `prelude`:
//!
//! ```
//! use tessera_rl::prelude::*;
//! ```
//!
//! # Features
//!
//! - `debug-checks`: strict checks at the validation boundary (SPEC 5.5),
//!   forwarded to the engine; `engine::DEBUG_CHECKS` reports whether it
//!   is on.

pub use tessera_rl_classic as classic;
pub use tessera_rl_core as core;
pub use tessera_rl_engine as engine;
pub use tessera_rl_games as games;
pub use tessera_rl_macros as macros;
pub use tessera_rl_num as num;
pub use tessera_rl_search as search;

pub mod prelude;
