// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CPython boundary shared by Tessera's Python extensions (ADR-0041,
//! ADR-0044, ADR-0062 item 8).
//!
//! - [`ffi`]: the audited module: the floating-point entry thunks and their
//!   templates, the C-API wrappers and callouts, C table rows, DLPack
//!   exports, the fork generation and the reentrancy guard holder.
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

#[doc(hidden)]
pub mod ffi;

pub mod errors;

/// Items the exported macros name. Not public API.
#[doc(hidden)]
pub mod __private {
    pub use tessera_rl_num::fp;
}
