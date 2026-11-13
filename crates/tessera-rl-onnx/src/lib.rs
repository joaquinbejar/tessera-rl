// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ONNX search evaluator (ADR-0057): ONNX Runtime loaded at run time
//! through `ort`, behind the optional `onnx` feature.
//!
//! So far the crate holds the runtime loader of ADR-0057 section 4:
//! [`load_runtime`] refuses to load anything unless `ORT_DISABLE_TELEMETRY=1`
//! is already set (or [`TelemetryPolicy::Accept`] is passed), loads only the
//! library at the explicit path, checks its exact version, and keeps one
//! runtime per process. [`OnnxEvaluator`] runs a model on that runtime as
//! a search evaluator, on the CPU execution provider only, and records its
//! [`Provenance`].

#[cfg(feature = "onnx")]
mod error;
#[cfg(feature = "onnx")]
mod evaluator;
#[cfg(feature = "onnx")]
mod hash;
#[cfg(feature = "onnx")]
mod layout;
#[cfg(feature = "onnx")]
mod loader;
#[cfg(feature = "onnx")]
mod provenance;
#[cfg(feature = "onnx")]
mod sys;
#[cfg(feature = "onnx")]
mod telemetry;

#[cfg(feature = "onnx")]
pub use error::OnnxError;
#[cfg(feature = "onnx")]
pub use evaluator::{
    CPU_PROVIDER, DEFAULT_MAX_MODEL_BYTES, OPTIMIZATION_LEVEL, OnnxEvaluator, OnnxOptions,
};
#[cfg(feature = "onnx")]
pub use loader::{PINNED_VERSION, Runtime, load_runtime};
#[cfg(feature = "onnx")]
pub use provenance::Provenance;
#[cfg(feature = "onnx")]
pub use telemetry::{TELEMETRY_VARIABLE, TelemetryPolicy};
