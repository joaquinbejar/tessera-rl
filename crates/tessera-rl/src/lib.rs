// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Tessera
//!
//! Tessera is a Rust engine for writing and running reinforcement learning
//! environments, with cheap snapshot, restore and fork, bit-exact determinism
//! and a native multi-agent model. It is exposed to Python through Gymnasium,
//! PettingZoo and PufferLib compatible interfaces.
//!
//! This crate is the facade: it re-exports the workspace crates.
//!
//! Status: pre-implementation (phase 1a scaffold).

pub use tessera_rl_classic as classic;
pub use tessera_rl_core as core;
pub use tessera_rl_engine as engine;
pub use tessera_rl_macros as macros;
pub use tessera_rl_num as num;
