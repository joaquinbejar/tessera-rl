// SPDX-License-Identifier: MIT OR Apache-2.0

//! Instance records, output ownership, scheduler, execution modes and
//! snapshot, restore and fork (ADR-0002, ADR-0005, ADR-0036).
//!
//! Milestones 5a to 5d: record storage, construction, the validation
//! boundary, Inline and Sync modes, auto-reset, poisoning, and snapshot,
//! restore and fork.
//!
//! `unsafe` code lives only in the audited modules `buffer` (disjoint chunk
//! views for workers) and `sys` (the worker gang, and the live-watch slot).

mod archive;
mod buffer;
#[cfg(test)]
mod chunk_clear_tests;
#[cfg(feature = "debug-checks")]
pub mod debug;
mod entry;
mod error;
#[cfg(test)]
mod fast_path_facts_tests;
#[cfg(test)]
mod fast_path_mixed_tests;
#[cfg(test)]
mod fast_path_tests;
#[cfg(test)]
mod fault_tests;
#[cfg(feature = "host")]
mod host;
mod layout;
mod masks;
#[cfg(not(loom))]
mod modes;
#[cfg(feature = "debug-checks")]
mod nan_check;
mod options;
mod outputs;
mod persist;
#[cfg(test)]
mod pin_tests;
mod pool;
mod pool_state;
mod record;
mod registry;
#[cfg(feature = "replay")]
pub mod replay;
mod set_zero;
mod snapshot;
mod sys;
pub mod transform;
mod validate;
mod watch;

/// The Async id rings and token id buffers (ADR-0047), exposed only for
/// the engine's own benchmarks and allocation tests. Not a stable API.
#[doc(hidden)]
pub mod rings {
    pub use crate::sys::id_buffers::{BufferError, BufferHandle, IdBufferPool};
    pub use crate::sys::ring::{CompletionRing, RingError, Signal, SubmitRing};
}

#[cfg(not(loom))]
pub use buffer::batch::{BatchRow, BatchToken, BatchWaiter};
#[cfg(not(loom))]
pub use buffer::pipeline::{HalfToken, Waiter};
pub use buffer::{
    OUTPUT_ALIGN, OUTPUT_MIN_ALIGN, OutputDtype, OutputField, OutputRegion, OutputStorage,
};
pub use error::{EngineError, NanSite};
#[cfg(feature = "host")]
pub use host::{
    DynHostPool, FactoryError, HOST_MAGIC, HostFactory, HostLayout, HostLeaf, HostPool, HostRecord,
    MAX_HOST_AGENTS, MAX_HOST_HEADS, MAX_HOST_SNAPSHOT_BYTES, MakeOptions, RecordDecodeError,
    make_host,
};
pub use transform::{Transform, TransformContext, TransformError, Transforms};

/// `check_snapshot_contract` (ADR-0073 item 8) and the `cargo tessera check`
/// functions (#351) for host environment crates.
#[cfg(feature = "testing")]
pub use host::testing as host_testing;

/// Not public API: what `register_host_factory!` expands to names.
#[cfg(feature = "host")]
#[doc(hidden)]
pub mod __private {
    pub use inventory;
    pub use tessera_rl_core::HostEnvironment;
}
pub use archive::{Archive, MAX_ARCHIVE_ENTRIES, ResetSource, ResetSourceContext};
pub use layout::{MAX_GROUPED_HEADS, MAX_LEAVES, MaskHeadShape};
pub use masks::{IllegalActionPolicy, MaskProvider, Masked, NoMasks, PersistMask};
#[cfg(not(loom))]
pub use modes::async_mode::{
    AsyncBuildError, AsyncIntoPoolError, AsyncOptions, AsyncPool, IdGenerations, SendBatchError,
};
#[cfg(not(loom))]
pub use modes::pipelined::{
    CycleState, HalfGenerations, IntoPoolError, IntoPoolErrorKind, PipelinedBuildError,
    PipelinedOptions, PipelinedPool, SendError,
};
pub use options::{
    AutoReset, CHUNK_RANGE, MAX_ADAPTIVE_CAP, PoolOptions, Schedule, SyncOptions, WaitBackend,
    WaitPolicy,
};
pub use outputs::Outputs;
pub use persist::{
    ASSET_FORMAT_VERSION, BUILD_ID, DecodeLimits, FORMAT_VERSION, MAGIC, PersistError,
};
pub use pool::{EnvPool, Reseed};
pub use pool_state::PoolState;
pub use record::{DecodedRecord, PoolContext, Record};
pub use registry::attach_shared_registry;
#[cfg(feature = "replay")]
pub use replay::{
    PointKind, Recorder, Replay, ReplayCursor, ReplayError, ReplayPoint, ReplaySummary,
    ReplayWriterOptions,
};
#[cfg(not(loom))]
pub use sys::native_wait_available;
pub use sys::watch::{WatchError, WatchHandle};
pub use sys::{default_workers, physical_cores};
pub use validate::ActionInput;

/// Whether this build has the `debug-checks` feature (SPEC 5.5): strict
/// legality checks at the validation boundary; an invalid agent event, a
/// NaN in state or rewards, or an allocation inside a chunk step (with
/// `debug::TrackingAllocator` installed) poisons the pool (ADR-0055). Lets a downstream crate confirm that the feature it
/// asked for on the facade reached the engine.
pub const DEBUG_CHECKS: bool = cfg!(feature = "debug-checks");
