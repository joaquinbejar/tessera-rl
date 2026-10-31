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
//! Pinning uses the `core_affinity` crate, approved by the coordinator as a
//! dependency for this purpose (best effort, failures logged).

mod affinity;
mod gang;
mod panics;
#[cfg(not(loom))]
mod native;
pub(crate) mod sync;

pub(crate) use affinity::available_cores;
#[cfg(test)]
pub(crate) use gang::Faults;
pub(crate) use gang::{Gang, GangOptions};
pub(crate) use panics::catch;
#[cfg(not(loom))]
pub(crate) use native::Native;
