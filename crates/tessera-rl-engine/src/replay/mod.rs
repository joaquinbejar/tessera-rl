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
//! This implementation covers first-party [`EnvPool`](crate::EnvPool)s.
//! Host pools, `Rejected` events, refusal rows, outputs hashes, Async `Recv`
//! events and adaptive-config events are reserved in the format and refused
//! on read with a typed error.

mod error;
mod format;
mod reader;
mod run;
mod writer;

pub use error::{DivergenceKind, ReplayError};
pub use format::{EventTag, FORMAT_VERSION, REPLAY_MAGIC, StateWriteOp};
pub use reader::{Event, Events, Ids, RecordedReseed, Replay, ReplayHeader};
pub use run::ReplaySummary;
pub use writer::{Recorder, ReplayWriterOptions};
