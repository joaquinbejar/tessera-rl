// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pool construction options.

use core::num::NonZeroU32;
use std::sync::Arc;

use tessera_rl_num::identity::Seed;

use crate::buffer::OutputAllocator;
use crate::error::EngineError;
use crate::masks::IllegalActionPolicy;

/// Smallest and largest chunk sizes (SPEC 5.1).
pub const CHUNK_RANGE: core::ops::RangeInclusive<u32> = 1..=1024;

/// What happens after an episode ends (ADR-0028).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AutoReset {
    /// The step that ends the episode returns the final observation and
    /// the episode flags; on the next step the instance resets instead of
    /// stepping (its actions are ignored), returns its initial observation
    /// and clears its flags. The Gymnasium vector default.
    #[default]
    NextStep,
    /// The instance resets in the same pass: the final observation goes to
    /// the terminal observation buffer, the initial one to the observation
    /// buffer, and the flags are set. PufferLib and EnvPool style. Because
    /// any instance may reset in a pass, the boundary checks the
    /// `episode_index` headroom of every instance on every step.
    SameStep,
    /// The instance stays frozen until the caller resets it; stepping it
    /// is an error.
    Disabled,
}

/// Options of an [`EnvPool`](crate::EnvPool).
///
/// `Clone`, not `Copy`, since it holds an optional output allocator
/// (ADR-0084). Equality compares the allocator by identity: the data
/// pointer of its `Arc`, without the vtable.
#[derive(Clone)]
pub struct PoolOptions {
    /// Number of instances.
    pub num_envs: NonZeroU32,
    /// Instance `i` starts with seed `seed + i`, modulo 2^64 (SPEC 8.1).
    pub seed: Seed,
    /// Instances per chunk, the unit of scheduling.
    pub chunk_size: NonZeroU32,
    /// Truncates an episode after this many transitions.
    pub max_episode_steps: Option<NonZeroU32>,
    /// Behaviour after an episode ends.
    pub auto_reset: AutoReset,
    /// Policy for illegal actions; applies only to masked pools. When
    /// options enter hashes or replays, the penalty is recorded by its bits.
    pub illegal_action: IllegalActionPolicy,
    /// Largest mask buffer allowed, in bytes (SPEC 11.3); checked before
    /// allocating. Applies only to masked pools.
    pub mask_budget_bytes: u64,
    /// How chunks are given to workers (ADR-0076). Never changes what a
    /// step computes.
    pub schedule: Schedule,
    /// The SIMD level of the lane path of a `LaneStep` environment
    /// (ADR-0071); `None` selects the widest level this CPU has. It changes
    /// speed only, never results. A level this CPU lacks is an error at
    /// construction, never a silent fallback. Ignored by environments
    /// without a lane step.
    pub simd_level: Option<tessera_rl_simd::Level>,
    /// Where the output storage comes from (ADR-0084): `None`, the default,
    /// is the engine's own allocation; `Some` asks this allocator once, at
    /// construction, for the whole storage. Never changes what a step
    /// computes or what a snapshot holds.
    ///
    /// Invariant (the `UnwindSafe` and `RefUnwindSafe` impls below rely on
    /// it): the engine calls the allocator only once, at construction, and
    /// keeps nothing of it but the storage it returned. Calling it again
    /// after construction, or keeping per-call allocator state, breaks
    /// those impls.
    pub output_allocator: Option<Arc<dyn OutputAllocator>>,
}

impl core::fmt::Debug for PoolOptions {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PoolOptions")
            .field("num_envs", &self.num_envs)
            .field("seed", &self.seed)
            .field("chunk_size", &self.chunk_size)
            .field("max_episode_steps", &self.max_episode_steps)
            .field("auto_reset", &self.auto_reset)
            .field("illegal_action", &self.illegal_action)
            .field("mask_budget_bytes", &self.mask_budget_bytes)
            .field("schedule", &self.schedule)
            .field("simd_level", &self.simd_level)
            .field("output_allocator", &self.output_allocator.is_some())
            .finish()
    }
}

// The allocator trait object would make `PoolOptions` neither `UnwindSafe`
// nor `RefUnwindSafe`, a public API regression from the `Copy` options of
// before ADR-0084. The engine calls the allocator only once, at
// construction, and holds no state of it that a caught panic could leave
// half-updated, so both hold as before (the facade API snapshot checks it).
impl core::panic::UnwindSafe for PoolOptions {}
impl core::panic::RefUnwindSafe for PoolOptions {}

impl PartialEq for PoolOptions {
    fn eq(&self, other: &Self) -> bool {
        // Destructured, so a new field cannot be left out of the comparison.
        let Self {
            num_envs,
            seed,
            chunk_size,
            max_episode_steps,
            auto_reset,
            illegal_action,
            mask_budget_bytes,
            schedule,
            simd_level,
            output_allocator,
        } = self;
        let allocators = match (output_allocator, &other.output_allocator) {
            (None, None) => true,
            // Data pointers only: two `Arc<dyn _>` of one allocation may
            // carry different vtable pointers (one per codegen unit), so
            // the metadata is not part of the identity.
            (Some(a), Some(b)) => {
                core::ptr::eq(Arc::as_ptr(a).cast::<()>(), Arc::as_ptr(b).cast::<()>())
            }
            _ => false,
        };
        *num_envs == other.num_envs
            && *seed == other.seed
            && *chunk_size == other.chunk_size
            && *max_episode_steps == other.max_episode_steps
            && *auto_reset == other.auto_reset
            && *illegal_action == other.illegal_action
            && *mask_budget_bytes == other.mask_budget_bytes
            && *schedule == other.schedule
            && *simd_level == other.simd_level
            && allocators
    }
}

/// How a worker pool gives chunks to its workers (ADR-0076, SPEC 6.1).
///
/// Outputs, records and state hashes are bit-identical under both: every
/// output row is written by the chunk that owns it, and nothing a step
/// computes or reports depends on which worker ran a chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Schedule {
    /// Worker `w` steps the chunks `c` with `c % workers == w` (ADR-0005):
    /// an instance always runs on the same worker, which keeps its cache
    /// warm. The default.
    #[default]
    Static,
    /// Workers claim chunks until every chunk is taken, balancing variable
    /// per-instance cost (ADR-0087): worker `w` first claims the chunks
    /// `Static` gives it, from its own cursor, then takes the chunks other
    /// workers have not claimed yet, in a fixed rotation. Outputs are
    /// bit-identical to `Static`.
    ///
    /// Use it for variable step cost only, and avoid small chunks. The
    /// `schedule_step` bench, on a quiet 8-core x86-64 host (indicative,
    /// not a reference machine), measured the earlier single-counter
    /// claiming (ADR-0076), which owner-first claiming replaces:
    /// - uniform cost at the default 64 instances per chunk: about 2%
    ///   faster than `Static` on 4 workers and about 6% slower on 8, where
    ///   the claimed chunks lose the per-worker cache affinity;
    /// - one instance per chunk: 12% slower on 4 workers and 38% on 8,
    ///   from contention on the counter;
    /// - a heavy-tailed per-step cost on a loaded host: up to about 3 times
    ///   faster on 8 workers.
    Dynamic,
}

/// Default mask budget: 1 GiB.
pub const DEFAULT_MASK_BUDGET: u64 = 1 << 30;

impl PoolOptions {
    /// `num_envs` instances seeded from `seed`, chunks of 64, no step limit,
    /// `NextStep` auto-reset.
    #[must_use]
    pub const fn new(num_envs: NonZeroU32, seed: Seed) -> Self {
        Self {
            num_envs,
            seed,
            chunk_size: NonZeroU32::MIN.saturating_add(63),
            max_episode_steps: None,
            auto_reset: AutoReset::NextStep,
            illegal_action: IllegalActionPolicy::Noop,
            mask_budget_bytes: DEFAULT_MASK_BUDGET,
            schedule: Schedule::Static,
            simd_level: None,
            output_allocator: None,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), EngineError> {
        if !CHUNK_RANGE.contains(&self.chunk_size.get()) {
            return Err(EngineError::Options {
                option: "chunk_size",
                reason: "must be in 1..=1024",
            });
        }
        if let IllegalActionPolicy::Penalize(r) = self.illegal_action
            && !r.is_finite()
        {
            return Err(EngineError::Options {
                option: "illegal_action",
                reason: "the penalty must be finite",
            });
        }
        Ok(())
    }
}

/// How worker threads block between batches (ADR-0005).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaitBackend {
    /// The native address wait where it measured faster than park (Linux
    /// `futex`); park and unpark elsewhere (macOS showed no gain, Windows
    /// has not run in CI yet). To be re-evaluated on reference hardware.
    #[default]
    Auto,
    /// Portable `thread::park` and `unpark`.
    Park,
    /// The native address wait: `futex`, macOS 14.4+
    /// `os_sync_wait_on_address`, or Windows `WaitOnAddress` (opt-in: not yet
    /// run on Windows). Falls back to park, with a warning, where none
    /// exists.
    Native,
}

/// What a worker does while it waits for its next batch (ADR-0074).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaitPolicy {
    /// Spin [`SyncOptions::spin`] times, then block (ADR-0005). The
    /// default.
    #[default]
    Fixed,
    /// After the fixed spins, keep spinning while the current wait is
    /// shorter than `min(cap, 2 x the EWMA of this worker's past waits)`,
    /// then block. A worker that spins through the caller's gap keeps its
    /// caches warm, at the cost of about one core per worker for the gap
    /// (ADR-0074). `cap` is at most [`MAX_ADAPTIVE_CAP`]; zero behaves as
    /// [`WaitPolicy::Fixed`].
    Adaptive {
        /// Upper bound on the adaptive spinning of one wait.
        cap: core::time::Duration,
    },
}

/// The largest cap of [`WaitPolicy::Adaptive`]: one second.
pub const MAX_ADAPTIVE_CAP: core::time::Duration = core::time::Duration::from_secs(1);

/// Options of Sync mode (ADR-0005, ADR-0006).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncOptions {
    /// Worker threads.
    pub workers: core::num::NonZeroUsize,
    /// Spins before a waiting thread parks (about 20 microseconds at the
    /// default).
    pub spin: u32,
    /// Logical cores to pin workers to, in order. `None` pins worker `w` to
    /// core `w` when no other engine with workers is active, and leaves
    /// workers unpinned (with a warning) otherwise.
    pub cpu_set: Option<Vec<usize>>,
    /// How workers block between batches.
    pub wait: WaitBackend,
    /// How long workers spin before they block (ADR-0074). Named
    /// `wait_policy` because `wait` is the blocking primitive.
    pub wait_policy: WaitPolicy,
}

/// One worker per physical core this process may use (SPEC 6.1,
/// [`crate::physical_cores`]), with the [`SyncOptions::new`] defaults. A pool
/// still steps inline until it is given workers (`EnvPool::with_workers`).
impl Default for SyncOptions {
    fn default() -> Self {
        Self::new(crate::physical_cores())
    }
}

impl SyncOptions {
    /// `workers` workers with the default spin budget and default pinning.
    #[must_use]
    pub const fn new(workers: core::num::NonZeroUsize) -> Self {
        Self {
            workers,
            spin: 2_000,
            cpu_set: None,
            wait: WaitBackend::Auto,
            wait_policy: WaitPolicy::Fixed,
        }
    }
}
