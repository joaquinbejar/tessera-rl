// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replay files (ADR-0072): a call log over the persisted-snapshot header,
//! with keyframes and divergence checks.
//!
//! A [`Recorder`] wraps an [`EnvPool`](crate::EnvPool) and logs every call
//! that changes state; [`Replay::open`] checks a file and [`Replay::run`]
//! re-issues its calls into a pool, reporting the first failing check as
//! [`ReplayError::Diverged`].
//!
//! A file with keyframes holds full state, so it is privileged exactly like a
//! persisted snapshot (SPEC 9.3, ADR-0035). The checksums detect accidents;
//! they are not authentication.
//!
//! First-party [`EnvPool`](crate::EnvPool)s get keyframes and checkpoints.
//! Host pools (with the `host` feature) get `from_reset_only` files with
//! outputs hashes, `Rejected` events (ADR-0069) and refusal rows (ADR-0070),
//! since host snapshots (ADR-0073) do not exist yet. Async `Recv` events and
//! adaptive-config events are reserved and refused on read with a typed
//! error.

#[cfg(not(loom))]
mod async_run;
#[cfg(not(loom))]
mod async_writer;
mod error;
mod format;
#[cfg(feature = "host")]
mod host_run;
#[cfg(feature = "host")]
mod host_writer;
#[cfg(feature = "host")]
mod outputs_hash;
mod reader;
mod run;
mod writer;

#[cfg(not(loom))]
pub use async_writer::AsyncRecorder;
pub use error::{DivergenceKind, ReplayError};
pub use format::{EventTag, FORMAT_VERSION, REPLAY_MAGIC, StateWriteOp};
#[cfg(feature = "host")]
pub use host_writer::HostRecorder;
pub use reader::{
    Event, Events, HostFileKind, HostInfo, Ids, RecordedReseed, RefusalRow, Replay, ReplayHeader,
};
pub use run::ReplaySummary;
pub use writer::{DetachedRecorder, Recorder, ReplayWriterOptions};
