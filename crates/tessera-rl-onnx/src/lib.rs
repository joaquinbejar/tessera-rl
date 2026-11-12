// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ONNX search evaluator (ADR-0057): ONNX Runtime loaded at run time
//! through `ort`, behind the optional `onnx` feature.
//!
//! This crate holds, so far, only the audited version read of
//! ADR-0057 section 4 (`sys`). The loader, which checks
//! `ORT_DISABLE_TELEMETRY` before any load, and the evaluator follow in
//! later changes; until then nothing here is public.

#[cfg(feature = "onnx")]
mod error;
#[cfg(feature = "onnx")]
mod sys;
