// SPDX-License-Identifier: MIT OR Apache-2.0

//! The XLA FFI path for JAX handles (ADR-0095), experimental. Audited
//! `unsafe` module (AGENTS.md).
//!
//! - `registry`: the fixed slot array that lets a handler, which runs on an
//!   XLA thread without the interpreter, reach a pool by (slot,
//!   generation), with the step-versus-close protocol of ADR-0095 point 3.
//! - `sys`: `#[repr(C)]` bindings for the subset of the XLA FFI C API 0.3
//!   the handler uses, checked against the pinned header by layout tests.

pub mod registry;
mod sync;
pub mod sys;
/// Test-only allocation counting (feature `test-hooks`).
#[cfg(feature = "test-hooks")]
pub mod testing;
