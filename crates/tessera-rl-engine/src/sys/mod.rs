// SPDX-License-Identifier: MIT OR Apache-2.0

//! Threads, parking and the panic boundary of Sync mode (ADR-0005).
//! Audited `unsafe` module.
//!
//! The only `unsafe` here is in `gang`: a type-erased call of the batch
//! job and the dereference of the job pointer the caller publishes for the
//! duration of one epoch. Its invariants are documented there.

mod affinity;
mod gang;
pub(crate) mod sync;

pub(crate) use gang::{Gang, GangOptions};
