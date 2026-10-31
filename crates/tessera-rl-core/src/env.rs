// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Environment` trait and its capability traits (SPEC 10, ADR-0012,
//! ADR-0035).

use tessera_rl_num::rng::{ObsRng, StepRng};

use crate::agent::{AgentSlot, Tier};
use crate::encoding::Canonical;
use crate::error::{ConfigError, InvariantViolation, SpaceError};
use crate::header::RecordLayout;
use crate::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use crate::space::Space;
use crate::state_init::StateInit;
use crate::step_out::StepOut;

/// Plain-data state (ADR-0029): `#[repr(C)]`, padding-free, a valid initial
/// value ([`StateInit`]) and a canonical encoding.
///
/// Implemented by `#[derive(EnvState)]`, which checks the layout rules at
/// compile time. Implementing it by hand must uphold the same rules.
pub trait EnvState: Canonical + StateInit + Copy + 'static {}

/// An environment configuration, validated at construction.
pub trait EnvConfig: Sized + 'static {
    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] naming the invalid field.
    fn validate(&self) -> Result<(), ConfigError> {
        Ok(())
    }
}

/// An environment: one trait for single-agent and multi-agent games.
///
/// `reset`, `step` and `observe` are infallible: all fallible work happens
/// at construction and at the validation boundary (ADR-0028).
pub trait Environment: Sized + 'static {
    /// Author state of one instance.
    type State: EnvState;
    /// Configuration shared by all instances of a pool.
    type Config: EnvConfig;
    /// Observation type (a marker until the `Observation` derive exists).
    type Obs: 'static;
    /// Action type (a marker until typed action heads exist).
    type Action: 'static;
    /// Record header: `RecordHeader<MAX_AGENTS, Adaptive>` (ADR-0036). The
    /// engine checks at compile time that its agent count equals
    /// `MAX_AGENTS`; use `NoAdaptive` when the config has no adaptive
    /// fields.
    type Header: RecordLayout;

    /// Number of agent slots; 1 for single-agent environments.
    const MAX_AGENTS: usize;
    /// The declared determinism tier.
    const TIER: Tier;
    /// Whether every agent observes the full state (ADR-0035).
    const PERFECT_INFORMATION: bool = true;

    /// The observation space for `cfg`.
    ///
    /// # Errors
    ///
    /// Returns a [`SpaceError`] if `cfg` yields an invalid space.
    fn observation_space(cfg: &Self::Config) -> Result<Space, SpaceError>;

    /// The action space for `cfg`.
    ///
    /// # Errors
    ///
    /// Returns a [`SpaceError`] if `cfg` yields an invalid space.
    fn action_space(cfg: &Self::Config) -> Result<Space, SpaceError>;

    /// Starts an episode. `state` holds the initial value or the previous
    /// episode.
    fn reset(cfg: &Self::Config, rng: &mut StepRng, state: &mut Self::State);

    /// Advances one transition.
    fn step(
        cfg: &Self::Config,
        state: &mut Self::State,
        actions: &Actions<'_, Self::Action>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    );

    /// Writes the observation of `agent`.
    fn observe(
        cfg: &Self::Config,
        state: &Self::State,
        agent: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, Self::Obs>,
    );
}

/// Legal-action masks, written in the observation pass (ADR-0014), and
/// the no-op that replaces an illegal action.
pub trait ActionMask: Environment {
    /// Writes the mask of `agent`: one entry per choice of each discrete
    /// head, heads concatenated in order.
    fn action_mask(
        cfg: &Self::Config,
        state: &Self::State,
        agent: AgentSlot,
        out: &mut MaskWriter<'_>,
    );

    /// Writes the environment's no-op action for `agent`, used in place of
    /// an illegal action (ADR-0028). There is no default: the author
    /// defines it.
    fn noop(cfg: &Self::Config, state: &Self::State, agent: AgentSlot, out: &mut ActionWriter<'_>);
}

/// Imperfect information: the per-agent information state (ADR-0035).
///
/// Required when [`Environment::PERFECT_INFORMATION`] is `false`.
pub trait ImperfectInformation: Environment {
    /// Fixed-size, plain-data information state of one agent.
    type InfoState: EnvState;

    /// Returns what `agent` knows, read from state.
    fn info_state(cfg: &Self::Config, state: &Self::State, agent: AgentSlot) -> Self::InfoState;
}

/// Samples full states consistent with an information state, for
/// player-perspective search (ADR-0035).
pub trait Determinize: ImperfectInformation {
    /// Writes into `state` a full author state consistent with `info`. It
    /// sees only the information state, never the hidden state.
    fn sample(
        cfg: &Self::Config,
        info: &Self::InfoState,
        rng: &mut StepRng,
        state: &mut Self::State,
    );
}

/// Rendering (SPEC 17). A placeholder: the frame and scene types are
/// defined in phase 2.
pub trait Render: Environment {}

/// A scripted policy for opponents and baselines.
pub trait Heuristic: Environment {
    /// Writes the action of `agent`.
    fn act(
        cfg: &Self::Config,
        state: &Self::State,
        agent: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    );
}

/// Immutable shared data, content-addressed (SPEC 17).
pub trait EnvAssets: Environment {
    /// The asset bundle, shared read-only by every instance.
    type Assets: Send + Sync + 'static;
}

/// Game invariants checked when decoding persisted snapshots (ADR-0029).
pub trait Validate: Environment {
    /// Checks `state` against `cfg`.
    ///
    /// # Errors
    ///
    /// Returns the first [`InvariantViolation`] found.
    fn validate(cfg: &Self::Config, state: &Self::State) -> Result<(), InvariantViolation>;
}
