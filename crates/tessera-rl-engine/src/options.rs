// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pool construction options.

use core::num::NonZeroU32;

use tessera_rl_num::identity::Seed;

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
#[derive(Debug, Clone, Copy, PartialEq)]
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
        }
    }
}
