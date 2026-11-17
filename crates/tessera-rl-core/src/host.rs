// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-driven environments (ADR-0062): environments that own their state,
//! which may live on the heap.
//!
//! A [`HostEnvironment`] is a separate capability from
//! [`Environment`](crate::Environment). It has no plain-data state, so it
//! has no snapshot, restore, fork, state hash, persistence or search, and
//! no determinism tier: same seeds and actions give the same outputs only
//! if the author's code makes it so. The engine still validates actions,
//! owns the output buffers and their borrow windows, handles auto-reset and
//! poisoning, applies the seeding rules of ADR-0007 to the seeds it passes
//! to [`HostEnvironment::reset`], and runs every call under canonical
//! floating-point controls (ADR-0062 items 2 and 3).
//!
//! Author code may allocate, lock, block, do I/O or read the clock. It runs
//! with the Python GIL released and must not call into Python.
//!
//! Experimental until 1.0 (ADR-0062 item 9): a breaking change to this
//! trait or its types bumps the 0.x minor version.

use thiserror::Error;

use crate::agent::{AgentSlot, TurnStructure};
use crate::env::EnvConfig;
use crate::error::{ConfigError, SpaceError};
use crate::io::{Actions, MaskWriter, ObsWriter};
use crate::space::Space;
use crate::step_out::StepOut;

/// The seed of one reset of one instance, after the engine applied the
/// seeding rules of ADR-0007 (`pool_seed + i`, `reset(seed=s)` gives
/// `s + i`, a list gives `s_i`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HostSeed {
    seed: u64,
    episode_index: u32,
}

impl HostSeed {
    /// A seed for episode `episode_index` of an instance seeded with `seed`.
    #[must_use]
    pub const fn new(seed: u64, episode_index: u32) -> Self {
        Self {
            seed,
            episode_index,
        }
    }

    /// The instance's seed.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// The episode count since the instance was last seeded: 0 on the first
    /// reset after seeding, 1 on the next auto-reset, and so on. An author
    /// who wants a different stream per episode derives it from both.
    #[must_use]
    pub const fn episode_index(&self) -> u32 {
        self.episode_index
    }
}

/// A host environment that could not be built.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum HostError {
    /// The configuration is invalid.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// A space is invalid or unsupported.
    #[error(transparent)]
    Space(#[from] SpaceError),
    /// The environment refused to start, for a reason of its own.
    #[error("host environment construction failed: {reason}")]
    Construction {
        /// What failed.
        reason: String,
    },
}

/// An environment that owns its state (ADR-0062).
///
/// One value of the implementing type is one instance; the engine's host
/// pool owns `num_envs` of them, built once at construction with
/// [`HostEnvironment::new`]. The views passed to the methods are built by
/// the engine only: observation leaves are zero on entry (ADR-0058), and
/// `StepOut` rows have `MAX_AGENTS` entries.
///
/// **Validation is shape and range only** (ADR-0028): every action is
/// checked against its head's range, but not against the environment's own
/// mask. A host pool has no no-op to substitute, so `step` can receive an
/// action that [`HostEnvironment::mask`] marked illegal, and must handle it.
pub trait HostEnvironment: Send + Sized + 'static {
    /// The versioned id, `<namespace>/<Name>-v<N>` (ADR-0023). The
    /// namespace `tessera` is reserved for first-party environments.
    const ENV_ID: &'static str;
    /// The outputs version, bumped with any change to the outputs, as
    /// `Persist::ENV_VERSION`.
    const ENV_VERSION: u32;
    /// Number of agent slots (1 for single-agent).
    const MAX_AGENTS: usize;
    /// Whether every agent sees the whole state.
    const PERFECT_INFORMATION: bool = true;
    /// The turn structure (ADR-0013); `Sequential` and `Mixed` must
    /// override [`HostEnvironment::acting`].
    const TURNS: TurnStructure = TurnStructure::Simultaneous;

    /// The environment's configuration.
    type Config: EnvConfig;

    /// Builds one instance; it may allocate.
    ///
    /// # Errors
    ///
    /// [`HostError`] when the configuration or the environment's own setup
    /// fails.
    fn new(cfg: &Self::Config) -> Result<Self, HostError>;

    /// The observation space of one agent.
    ///
    /// # Errors
    ///
    /// [`SpaceError`] for an invalid configuration.
    fn observation_space(cfg: &Self::Config) -> Result<Space, SpaceError>;

    /// The action space of one agent.
    ///
    /// # Errors
    ///
    /// [`SpaceError`] for an invalid configuration.
    fn action_space(cfg: &Self::Config) -> Result<Space, SpaceError>;

    /// Starts an episode with `seed`. Every agent slot is alive afterwards.
    ///
    /// What `out` means depends on why the reset runs:
    ///
    /// - an explicit reset (`reset`, `reset_all`) and a `NextStep`
    ///   auto-reset (the call after the episode ended): the rewards and
    ///   flags written to `out` are that instance's outputs of the call;
    /// - a `SameStep` auto-reset (in the call that ended the episode):
    ///   `out` is scratch, and what the reset writes is discarded, because
    ///   the instance's rows hold the ended step's rewards and flags;
    /// - `Disabled`: the engine never resets an instance by itself.
    ///
    /// An episode end reported through `out` is ignored in every case.
    fn reset(&mut self, seed: HostSeed, out: &mut StepOut<'_>);

    /// Applies one step of validated actions.
    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut StepOut<'_>);

    /// Writes `agent`'s observation into leaves that are zero on entry.
    fn observe(&self, agent: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>);

    /// Writes `agent`'s legal-action mask; every action is legal by default.
    /// The mask is an output: actions are not checked against it.
    fn mask(&self, _agent: AgentSlot, mask: &mut MaskWriter<'_>) {
        mask.fill(true);
    }

    /// Whether `agent` acts in the next step; every agent does by default.
    #[must_use]
    fn acting(&self, _agent: AgentSlot) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{HostError, HostSeed};
    use crate::error::SpaceError;

    #[test]
    fn test_host_seed_new_round_trips() {
        let s = HostSeed::new(u64::MAX, 3);
        assert_eq!((s.seed(), s.episode_index()), (u64::MAX, 3));
    }

    #[test]
    fn test_host_error_from_space_error_keeps_message() {
        let e = HostError::from(SpaceError::Empty);
        assert_eq!(e.to_string(), SpaceError::Empty.to_string());
    }
}
