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
//!   (CPython's documented behaviour for daemon threads). No frame of
//!   `Gil::detach` or above is unwound: pyo3-ffi 0.30.1 wraps
//!   `PyEval_RestoreThread` in a guard that parks the thread forever when
//!   dropped (`HangThread`, pyo3-ffi `src/ceval.rs` and `src/impl_/mod.rs`),
//!   so on glibc the forced unwind of `pthread_exit` stops there and the
//!   thread never ends; on macOS `pthread_exit` ends the thread without
//!   unwinding. Either way the drops of those frames do not run. Code that
//!   must run to completion should not run in a daemon thread.
//!
//! - [`ffi`]: the audited module. Privately: the floating-point entry
//!   thunks and their templates, the generic shims that turn CPython's raw
//!   pointers into typed values, the callouts, the C tables and DLPack.
//!   Publicly: the typed layer above, the boundary policy, the fork
//!   generation and the reentrancy guard holder.
//! - [`errors`]: the exception classes and the mapping from engine errors.
//! - `host` (feature `host`): host protocol 1 (SPEC 14.4 draft, ADR-0062,
//!   ADR-0065) for a third-party host environment's wheel. One
//!   `export_host_env!(Env, module = c"pkg._tessera_host")` in the wheel's
//!   cdylib crate exports `make(*, num_envs, seed, auto_reset,
//!   max_episode_steps, workers, cpu_set, **config)`, which returns the host
//!   object: `host_protocol`, `tessera_rl_min` and `tessera_rl_below`;
//!   `layout()`, `reset(seed)` and `step(actions)` returning the outputs as
//!   read-only views (buffer protocol and DLPack, alive after `close`),
//!   `close()`, `is_poisoned()`, `_tessera_current_generation()` and
//!   `_tessera_set_fp_policy(policy)`. The outputs include ADR-0065's
//!   `aborted`, `abort_code` and `pending_reset`, and the layout its
//!   `initial_agents` and `spawns`.
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

/// The XLA FFI path for JAX handles (ADR-0095), experimental.
pub mod xla;

/// Host protocol 1 (feature `host`).
#[cfg(feature = "host")]
pub mod host;

/// Items the exported macros name. Not public API.
#[doc(hidden)]
pub mod __private {
    pub use inventory;
    #[cfg(feature = "host")]
    pub use tessera_rl_core::HostEnvironment;
}
