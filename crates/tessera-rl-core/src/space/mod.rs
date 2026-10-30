// SPDX-License-Identifier: MIT OR Apache-2.0

//! Observation and action spaces (SPEC 11, ADR-0014).
//!
//! Spaces are construction-time descriptions: they may allocate, and they
//! never appear in state.

mod definition;
mod dtype;

pub use definition::{MAX_DEPTH, Space};
pub use dtype::DType;
