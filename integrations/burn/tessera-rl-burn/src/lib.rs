// SPDX-License-Identifier: MIT OR Apache-2.0

//! Burn tensors from Tessera pool outputs, and actions back (SPEC 14.2,
//! #367).
//!
//! Every function copies once, from the engine's borrowed output views into
//! a new tensor on `device`: the engine's zero-copy contract (SPEC 5.3) ends
//! at its views, and the copy is this adapter's. Rows are the pool's
//! `[instance, agent]` slots in order, so `rows = num_envs * MAX_AGENTS`.

mod tensors;

pub use tensors::{AdapterError, actions, f32_leaf, flags, masks, rewards};
