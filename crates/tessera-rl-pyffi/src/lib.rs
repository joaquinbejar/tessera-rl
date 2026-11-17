// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CPython boundary shared by Tessera's Python extensions (ADR-0041,
//! ADR-0044, ADR-0062 item 8).
//!
//! An extension writes safe Rust only: its types are [`ffi::object`] bodies
//! and method traits, its module a [`ffi::module_def`] definition registered
//! with [`export_module!`], its buffers [`ffi::buffer`] exports, and its
//! arguments and results go through [`ffi::capi`]'s safe wrappers under a
//! call-scoped [`ffi::capi::Gil`]. No public item of this crate is `unsafe`
//! or relies on its caller for memory safety.
//!
//! - [`ffi`]: the audited module. Privately: the floating-point entry
//!   thunks and their templates, the generic shims that turn CPython's raw
//!   pointers into typed values, the callouts, the C tables and DLPack.
//!   Publicly: the typed layer above, the boundary policy, the fork
//!   generation and the reentrancy guard holder.
//! - [`errors`]: the exception classes and the mapping from engine errors.
//!
//! Each extension (cdylib) links its own copy: the floating-point policy,
//! the counters, the boundary frame and the exception classes are per
//! extension.
//!
//! Experimental until 1.0: this API serves Tessera's own extension crates
//! and may change in any 0.x release.

#![expect(
    clippy::doc_markdown,
    reason = "CPython, PyO3 and NumPy are names in prose"
)]

pub mod ffi;

pub mod errors;

/// Items the exported macros name. Not public API.
#[doc(hidden)]
pub mod __private {
    pub use inventory;
}
