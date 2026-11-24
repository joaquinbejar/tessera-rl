// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lane kernels over groups gathered from Array-of-Structs chunks
//! (ADR-0071).
//!
//! An environment that keeps the default Array-of-Structs storage implements
//! [`LaneStep`] and sets [`Environment::LANE_STEP`] to
//! `Some(lane_hook::<Self>())`. The engine then steps eligible groups of
//! [`LANES`] consecutive instances of a chunk through [`LaneStep::lane_step`]
//! and every other instance through [`Environment::step`]. Bookkeeping,
//! observe and resets stay per instance. `LaneEnvironment` (ADR-0002,
//! Struct-of-Arrays storage) is a different, still unimplemented mechanism.

use tessera_rl_simd::Resolved;

use crate::env::Environment;
use crate::step_out::EpisodeEnd;

/// Instances per lane group.
pub const LANES: usize = 8;

/// What one lane group's step reports, lane by lane: what
/// [`Environment::step`] would report for that instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaneOut {
    /// The reward of the instance's only agent (`set_reward`).
    pub reward: [f32; LANES],
    /// The episode end, if the step ended the episode (`end_episode`).
    pub end: [Option<EpisodeEnd>; LANES],
}

impl Default for LaneOut {
    fn default() -> Self {
        Self {
            reward: [0.0; LANES],
            end: [None; LANES],
        }
    }
}

/// A step of [`LANES`] instances at once (ADR-0071).
///
/// # Contract
///
/// For every lane, the new state, the reward and the episode end MUST equal,
/// bit for bit, what [`Environment::step`] produces for that instance with
/// the same action. A lane step reports no agent events. The engine only
/// calls it for single-agent, simultaneous, unmasked environments with one
/// discrete action head and no continuous head, and only for groups in
/// which no instance is pending reset and the agent is alive.
pub trait LaneStep: Environment {
    /// Whether [`Environment::step`] draws from its `StepRng`. A lane step
    /// receives no RNG, so `true` is refused at compile time by
    /// [`lane_hook`] (ADR-0071 decision 2).
    const STEP_DRAWS_RNG: bool;

    /// Steps `states` in place at `level`. `actions[l]` is lane `l`'s
    /// discrete action, already validated against the action space. The
    /// caller holds the canonical FP mode.
    fn lane_step(
        cfg: &Self::Config,
        level: Resolved,
        states: &mut [Self::State; LANES],
        actions: &[i32; LANES],
        out: &mut LaneOut,
    );
}

/// The engine's entry point for a [`LaneStep`] environment.
pub type LaneHook<E> = fn(
    &<E as Environment>::Config,
    Resolved,
    &mut [<E as Environment>::State; LANES],
    &[i32; LANES],
    &mut LaneOut,
);

/// The value of [`Environment::LANE_STEP`] for a [`LaneStep`] environment.
/// Fails to compile when the environment's step draws RNG values.
#[must_use]
pub const fn lane_hook<E: LaneStep>() -> LaneHook<E> {
    const {
        assert!(
            !E::STEP_DRAWS_RNG,
            "LaneStep with STEP_DRAWS_RNG = true is not supported (ADR-0071)"
        );
    }
    E::lane_step
}
