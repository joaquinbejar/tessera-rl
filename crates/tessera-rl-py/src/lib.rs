// SPDX-License-Identifier: MIT OR Apache-2.0

//! Python bindings for Tessera (`tessera_rl._native`), phase 1b.
//!
//! Every C entry point that CPython can call is registered through the
//! boundary thunk table of the `ffi` module (ADR-0041): the PyO3 macros are not
//! available (the `macros` feature is off), so no raw function pointer can
//! be registered by accident. The thunks are the native floating-point
//! boundary (ADR-0041): they canonicalize the caller's controls before the
//! first Rust frame and restore them after the last.

#![expect(
    clippy::doc_markdown,
    reason = "CPython, PyO3 and NumPy are names in prose"
)]

mod api;
mod env;
mod errors;
mod ffi;
mod view;
