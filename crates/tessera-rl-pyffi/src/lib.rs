// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CPython boundary shared by Tessera's Python extensions (ADR-0041,
//! ADR-0044, ADR-0062 item 8).
//!
//! An extension writes safe Rust only: its types are [`ffi::object`] bodies
//! and method traits, its module a [`ffi::module_def`] definition registered
//! with [`export_module!`], its buffers [`ffi::buffer`] exports, and its
//! arguments and results go through [`ffi::capi`]'s safe wrappers under a
//! call-scoped [`ffi::capi::Gil`]. No public item of this crate is `unsafe`
//! or relies on its caller for memory safety, with two documented
//! exceptions, both outside Rust's view:
//!
//! - **Borrowed exports.** A read through a buffer or DLPack export that
//!   races the next write of its outputs (the next engine call, or the next
//!   gather) reads bytes being overwritten. SPEC 5.3 leaves that race to the
//!   caller, as for every borrowed output; `is_stale` reports the end of the
//!   borrow window. A write through a legacy (unversioned) DLPack capsule,
//!   which has no read-only flag, is the consumer's as well: it changes
//!   the outputs it borrows. No Rust reference to the exported bytes
//!   exists.
//! - **Interpreter finalization.** A daemon thread inside
//!   [`ffi::capi::Gil::detach`] when the interpreter finalizes is ended by
//!   CPython 3.11 to 3.13 with `pthread_exit` when it tries to reattach
//!   (CPython's documented behaviour for daemon threads). On glibc,
//!   `pthread_exit` is a forced unwind that runs its Rust frames' drops; on
//!   macOS it does not unwind them. Code that must run to completion should
//!   not run in a daemon thread.
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
