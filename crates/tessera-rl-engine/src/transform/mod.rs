// SPDX-License-Identifier: MIT OR Apache-2.0

//! Observation transforms (ADR-0078): an engine pipeline after
//! `E::observe`, frame-stack state in an arena beside the records, and
//! running normalization as pool context.

mod error;
pub(crate) mod kernels;
pub(crate) mod plan;
mod pool_api;
pub(crate) mod rows;
pub(crate) mod runtime;
mod spec;
pub(crate) mod stats;

pub use error::TransformError;
pub use runtime::TransformContext;
pub use spec::{
    FRAME_STACK_DEPTH, MAX_NORMALIZE_ACCUMULATOR_BYTES, MAX_TRANSFORMED_ROW_BYTES, Transform,
    Transforms,
};
pub use stats::Moments;
