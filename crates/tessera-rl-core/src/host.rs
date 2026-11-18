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
use crate::host_out::HostOut;
use crate::io::{Actions, MaskWriter, ObsWriter};
use crate::space::Space;

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

/// A joint action a host environment refuses ([`HostEnvironment::validate`],
/// ADR-0065 item 2): a code and a static reason the engine reports. Making
/// one allocates nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionReject {
    /// The environment's own code.
    pub code: u32,
    /// Why, for the error message.
    pub reason: &'static str,
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
    ///
    /// Under `Simultaneous` every alive agent acts and `acting` is never
    /// called. An environment that needs acting subsets declares `Mixed`
    /// and overrides `acting`. A non-acting alive slot's action is not
    /// required: it is zeroed and never validated. The slot is still
    /// observed (and masked) every call, its reward is whatever `step`
    /// writes (0 by default), it stays alive, and its `terminated` and
    /// `truncated` flags are as written. This matches `EnvPool` and
    /// ADR-0013.
    const TURNS: TurnStructure = TurnStructure::Simultaneous;
    /// Whether the environment writes action masks
    /// ([`HostEnvironment::mask`]). Without masks the pool has no mask
    /// buffer (a mask length of 0) and never calls `mask`.
    const MASKS: bool = false;
    /// Whether the environment fills empty slots with `agent_spawned`
    /// (ADR-0065). With `false`, a spawn is an invalid agent event, for any
    /// slot: it is not applied, and it is counted and marked as the other
    /// invalid events are, so slots `initial_agents..` stay empty.
    const SPAWNS: bool = false;
    /// Whether the engine calls [`HostEnvironment::validate`] before each
    /// step (ADR-0065 item 2). With `false` the check is compiled out.
    const VALIDATES: bool = false;

    /// The environment's configuration.
    type Config: EnvConfig;

    /// Builds one instance; it may allocate.
    ///
    /// # Errors
    ///
    /// [`HostError`] when the configuration or the environment's own setup
    /// fails.
    fn new(cfg: &Self::Config) -> Result<Self, HostError>;

    /// Number of agent slots alive at the start of each episode: slots
    /// `0..n` start alive and acting, the rest start empty and may be filled
    /// by `agent_spawned` when [`HostEnvironment::SPAWNS`] is `true`
    /// (ADR-0065). [`HostEnvironment::MAX_AGENTS`] stays the capacity. The
    /// engine checks `1..=MAX_AGENTS` at construction. Defaults to
    /// `MAX_AGENTS`.
    #[must_use]
    fn initial_agents(cfg: &Self::Config) -> usize {
        let _ = cfg;
        Self::MAX_AGENTS
    }

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

    /// Starts an episode with `seed`. Slots `0..initial_agents` are alive
    /// afterwards, and the others empty.
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
    /// An episode end reported through `out` is ignored in every case, and
    /// so are the `died` and `spawned` flags: they never change which agent
    /// slots are alive after a reset (`0..initial_agents`). A reset may
    /// abort ([`HostOut::abort`]): the instance then stays pending a reset.
    fn reset(&mut self, seed: HostSeed, out: &mut HostOut<'_>);

    /// Applies one step of validated actions. `out.abort(code)` ends the
    /// episode as aborted ([`HostOut`], ADR-0065).
    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut HostOut<'_>);

    /// Checks this instance's joint action before any instance of the call
    /// steps (ADR-0065 item 2). Called only when
    /// [`HostEnvironment::VALIDATES`] is `true`, for the instances that step
    /// in the call (not pending a reset, with at least one acting agent),
    /// on the validated actions `step` will see (slots that do not act are
    /// zero). A rejection fails the whole call with
    /// `EngineError::ActionRejected` and mutates nothing the engine owns;
    /// the first rejection in instance order is reported.
    ///
    /// It must be pure: no interior mutability, no I/O, no clock or
    /// randomness, nothing but `self` and `actions` (the same contract as
    /// `observe`, documented and not checked). It runs with the interpreter
    /// released, under canonical floating-point controls; a panic poisons
    /// the pool.
    ///
    /// # Errors
    ///
    /// [`ActionReject`] for a joint action the environment refuses.
    fn validate(&self, actions: &Actions<'_, ()>) -> Result<(), ActionReject> {
        let _ = actions;
        Ok(())
    }

    /// Writes `agent`'s observation into leaves that are zero on entry.
    ///
    /// Typed writes need no derive: declare the leaf ids to match
    /// [`HostEnvironment::observation_space`], for example
    /// `const POS: LeafId<f32> = LeafId::new(0);`, and write through
    /// `obs.leaf_mut(POS)` ([`crate::leaf::LeafMut::set`] or
    /// [`crate::leaf::LeafMut::copy_from`]). Neither `LeafId` nor `LeafMut`
    /// checks the leaf's declared dtype, only its byte length, so the ids
    /// must name each leaf with the element type its space declares.
    fn observe(&self, agent: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>);

    /// Writes `agent`'s legal-action mask; every action is legal by default.
    /// Called only when [`HostEnvironment::MASKS`] is `true`. The mask is an
    /// output: actions are not checked against it.
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
