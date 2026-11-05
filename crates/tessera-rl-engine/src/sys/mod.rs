// SPDX-License-Identifier: MIT OR Apache-2.0

//! Threads, parking and the panic boundary of Sync mode (ADR-0005).
//! Audited `unsafe` module.
//!
//! The only `unsafe` here is in `gang`: a type-erased call of the batch
//! job and the dereference of the job pointer the caller publishes for the
//! duration of one epoch. Its invariants are documented there.
//!
//! Waits use the native address-wait primitive where it is enabled
//! (`native`), otherwise park and unpark.
//!
//! The Async id rings (`ring`) and the token id buffers (`id_buffers`)
//! are built on atomics only and need no `unsafe` (ADR-0047).
//!
//! Pinning uses the `core_affinity` crate, approved by the coordinator as a
//! dependency for this purpose (best effort, failures logged).

mod affinity;
mod gang;
pub(crate) mod id_buffers;
#[cfg(not(loom))]
mod native;
mod panics;
pub(crate) mod ring;
pub(crate) mod sync;

pub(crate) use affinity::available_cores;
#[cfg(test)]
pub(crate) use gang::Faults;
pub(crate) use gang::{Gang, GangOptions, Job};
#[cfg(not(loom))]
pub(crate) use native::Native;
pub(crate) use panics::catch;

/// Whether [`crate::WaitBackend::Native`] has a real address wait here; without
/// one it falls back to park.
#[must_use]
pub fn native_wait_available() -> bool {
    native::Native::detect().is_some()
}
