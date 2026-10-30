// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pool construction options.

use core::num::NonZeroU32;

use tessera_rl_num::identity::Seed;

use crate::error::EngineError;

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
    /// buffer, and the flags are set. PufferLib and EnvPool style.
    SameStep,
    /// The instance stays frozen until the caller resets it; stepping it
    /// is an error.
    Disabled,
}

/// Options of an [`EnvPool`](crate::EnvPool).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

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
        }
    }

    pub(crate) fn validate(&self) -> Result<(), EngineError> {
        if !CHUNK_RANGE.contains(&self.chunk_size.get()) {
            return Err(EngineError::Options {
                option: "chunk_size",
                reason: "must be in 1..=1024",
            });
        }
        Ok(())
    }
}
