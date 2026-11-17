// SPDX-License-Identifier: MIT OR Apache-2.0

//! Python bindings for Tessera (`tessera_rl._native`), phase 1b.
//!
//! Every C entry point that CPython can call is a typed row of
//! `tessera_rl_pyffi` (ADR-0062 item 8), whose generic shims run behind the
//! boundary thunks (ADR-0041): they canonicalize the caller's controls
//! before the first Rust frame and restore them after the last. The PyO3
//! macros are not available (the `macros` feature is off), so no raw
//! function pointer can be registered by accident.

// No `unsafe` anywhere in the bindings: the CPython boundary is
// `tessera_rl_pyffi::ffi`, the audited module (ADR-0062 item 8).
#![forbid(unsafe_code)]
#![expect(
    clippy::doc_markdown,
    reason = "CPython, PyO3 and NumPy are names in prose"
)]

mod api;
mod async_api;
mod batched;
#[cfg(feature = "test-hooks")]
mod continuous_probe;
mod env;
mod errors;
mod ffi;
#[cfg(feature = "test-hooks")]
mod multi_probe;
#[cfg(test)]
mod observe_panic;
mod onnx_api;
mod search;
mod search_api;
mod search_group_api;
mod split;
mod view;
