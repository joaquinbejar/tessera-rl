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

use crate::adapters::Adapters;
use crate::agent::{AgentSlot, TurnStructure};
use crate::env::EnvConfig;
use crate::error::{ConfigError, SpaceError};
use crate::host_out::HostOut;
use crate::io::{Actions, MaskWriter, ObsWriter};
use crate::refusals::Refusals;
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
    /// The PettingZoo interfaces the environment declares (ADR-0080 item 3):
    /// derived from `TURNS`. Host protocols 1 and 2 carry only the turn
    /// structure, so [`register_host!`](crate::register_host) refuses any
    /// other value until a protocol version carries this one.
    const ADAPTERS: Adapters = Adapters::for_turns(Self::TURNS);
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
    /// Whether the engine calls [`HostEnvironment::on_rejected`] on the
    /// instance a rejected call reports (ADR-0069). It needs
    /// [`HostEnvironment::VALIDATES`]: `true` without it fails to compile
    /// when a pool of the type is built. With `false` the notice is
    /// compiled out and a rejection mutates nothing (ADR-0065).
    const NOTIFIES_REJECTION: bool = false;

    /// Whether [`HostEnvironment::refuse`] is called: per-agent refusal
    /// (ADR-0070). When `false`, the hook, the `refused` and `refuse_code`
    /// outputs and their storage are compiled out.
    const REFUSES_AGENTS: bool = false;

    /// The environment's configuration.
    type Config: EnvConfig;

    /// Builds one instance; it may allocate.
    ///
    /// Each pool build calls it once per instance, in index order
    /// (instance 0 first), one call after the other on the thread that
    /// builds the pool; the `k`-th call of that build builds pool instance
    /// `k`. Every call of a build comes before any method is called on any
    /// of its instances. The functions that take only the configuration
    /// (`EnvConfig::validate`, [`HostEnvironment::initial_agents`] and the
    /// spaces) run before the first call and are not ordered against it. An
    /// environment that keeps per-instance resources can rely on that order;
    /// it cannot tell one build from the next (two pools from one
    /// configuration, or a build retried after a failure), so a count it
    /// keeps across builds runs on. If a call fails, no later instance of
    /// that build is built and the instances already built are dropped.
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

    /// Tells the instance that its joint action was refused (ADR-0069).
    /// Called only when [`HostEnvironment::NOTIFIES_REJECTION`] is `true`,
    /// once per rejected call, on the one instance the call reports (the
    /// lowest rejected index), with the [`ActionReject`] the call reports.
    /// It runs on the calling thread after validation, before the call
    /// returns `EngineError::ActionRejected`, under canonical floating-point
    /// controls; a panic poisons the pool. It is not called when a
    /// `validate` that is not pure accepts on the engine's re-run. In
    /// Inline the instances after the reported one are not validated.
    ///
    /// It may write this instance's own state only, so that snapshots
    /// carry the notice. Nothing else happens in a rejected call: no
    /// instance steps, no episode ends or resets, no agent changes and the
    /// RNG step index does not move, so it must not draw randomness. A
    /// consequence (a penalty, a termination) belongs in the next `step`.
    fn on_rejected(&mut self, reject: &ActionReject) {
        let _ = reject;
    }

    /// Refuses some agents' actions in this call while every other agent,
    /// and every other instance, still steps (ADR-0070). Called only when
    /// [`HostEnvironment::REFUSES_AGENTS`] is `true`, once per instance that
    /// steps in the call, after [`HostEnvironment::validate`] accepted the
    /// whole call and immediately before [`HostEnvironment::step`], on the
    /// worker that steps the instance.
    ///
    /// `step` then runs with the full joint action and reads the verdicts
    /// through [`Actions::is_refused`] and [`Actions::refuse_code`]. It must
    /// not act on a refused agent's action; what a refusal costs (nothing,
    /// a penalty, a lost turn) is the environment's documented rule. The
    /// refusals are reported in the `refused` and `refuse_code` outputs.
    ///
    /// Only alive, acting agents may be refused ([`Refusals::refuse`]):
    /// refusing another slot skips the instance's step and poisons the pool
    /// after the call, in every build. It must be pure, as
    /// [`HostEnvironment::validate`]: no `StepRng` or `ObsRng`, no I/O, no
    /// clock. A panic poisons the pool.
    fn refuse(&self, actions: &Actions<'_, ()>, out: &mut Refusals<'_>) {
        let _ = (actions, out);
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
