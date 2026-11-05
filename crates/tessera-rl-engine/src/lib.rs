// SPDX-License-Identifier: MIT OR Apache-2.0

//! Instance records, output ownership, scheduler, execution modes and
//! snapshot, restore and fork (ADR-0002, ADR-0005, ADR-0036).
//!
//! Milestones 5a to 5d: record storage, construction, the validation
//! boundary, Inline and Sync modes, auto-reset, poisoning, and snapshot,
//! restore and fork.
//!
//! `unsafe` code lives only in the audited modules `buffer` (disjoint chunk
//! views for workers) and `sys` (the worker gang).

mod buffer;
mod entry;
mod error;
#[cfg(test)]
mod fault_tests;
mod layout;
mod masks;
mod options;
mod outputs;
mod pool;
mod record;
mod registry;
mod snapshot;
mod sys;
mod validate;

/// The Async id rings and token id buffers (ADR-0047), exposed only for
/// the engine's own benchmarks and allocation tests. Not a stable API.
#[doc(hidden)]
pub mod rings {
    pub use crate::sys::id_buffers::{BufferError, BufferHandle, IdBufferPool};
    pub use crate::sys::ring::{CompletionRing, RingError, Signal, SubmitRing};
}

pub use buffer::{OutputDtype, OutputField, OutputRegion, OutputStorage};
pub use error::EngineError;
pub use layout::MAX_LEAVES;
pub use masks::{IllegalActionPolicy, MaskProvider, Masked, NoMasks};
pub use options::{AutoReset, CHUNK_RANGE, PoolOptions, SyncOptions, WaitBackend};
pub use outputs::Outputs;
pub use pool::{EnvPool, Reseed};
pub use record::{DecodedRecord, PoolContext, Record};
pub use validate::ActionInput;

/// Whether this build has the `debug-checks` feature (SPEC 5.5): strict
/// legality checks at the validation boundary, and an invalid agent event
/// poisons the pool. Lets a downstream crate confirm that the feature it
/// asked for on the facade reached the engine.
pub const DEBUG_CHECKS: bool = cfg!(feature = "debug-checks");
