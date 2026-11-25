// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lane kernels over the columns of a lane-wise state (ADR-0090 decision
//! 4).
//!
//! An environment whose state carries `#[state(lanes)]` implements
//! [`LaneEnvironment`] and sets [`Environment::LANE_ENV`] to
//! `Some(lane_env_hooks::<Self>())`. The engine then steps and observes
//! eligible groups of [`LANES`] consecutive instances of a chunk directly
//! on their columns, with no gather or scatter, and every other instance
//! (tails, pending resets, dead agents) through the one-instance gather,
//! [`Environment::step`] and [`Environment::observe`], and a scatter.

use tessera_rl_simd::Resolved;

use crate::env::{EnvState, Environment};
use crate::lane_field::LaneTuple;
use crate::lane_step::{LANES, LaneOut};

/// One group of [`LANES`] instances of `E`'s columns.
pub type LaneGroup<'a, E> =
    <<<E as Environment>::State as EnvState>::Fields as LaneTuple>::Group<'a>;

/// A step and an observation of [`LANES`] instances at once, on their
/// columns (ADR-0090 decision 4).
///
/// # Contract
///
/// For every lane, the new state, the reward and the episode end MUST equal,
/// bit for bit, what [`Environment::step`] produces for that instance with
/// the same action, and the observation bytes MUST equal what
/// [`Environment::observe`] writes: the same floating-point operations in
/// the same order and precision, no fused multiply-add, transcendental
/// functions only from the frozen math (ADR-0090 4.8). The engine only calls
/// these for single-agent, simultaneous, unmasked environments with one
/// discrete action head and no continuous head, without observation
/// transforms, and only for groups in which no instance is pending reset
/// and the agent is alive. A lane step reports no agent events.
pub trait LaneEnvironment: Environment {
    /// Whether [`Environment::step`] draws from its `StepRng`; `true` is
    /// refused at compile time by [`lane_env_hooks`].
    const STEP_DRAWS_RNG: bool;

    /// Whether [`Environment::observe`] draws from its `ObsRng`; `true` is
    /// refused at compile time by [`lane_env_hooks`].
    const OBSERVE_DRAWS_RNG: bool;

    /// Steps the group in place at `level`. `actions[l]` is lane `l`'s
    /// discrete action, already validated. The caller holds the canonical
    /// floating-point mode.
    fn step_lanes(
        cfg: &Self::Config,
        level: Resolved,
        lanes: &mut LaneGroup<'_, Self>,
        actions: &[i32; LANES],
        out: &mut LaneOut,
    );

    /// Writes the group's observations. `leaves[j]` is leaf `j`'s rows of
    /// the [`LANES`] instances, in instance order, each row the leaf's byte
    /// length; the rows are zeroed before the call.
    fn observe_lanes(cfg: &Self::Config, lanes: &LaneGroup<'_, Self>, leaves: &mut [&mut [u8]]);
}

/// The engine's entry points for a [`LaneEnvironment`].
pub struct LaneEnvHooks<E: Environment> {
    /// [`LaneEnvironment::step_lanes`].
    pub step: fn(&E::Config, Resolved, &mut LaneGroup<'_, E>, &[i32; LANES], &mut LaneOut),
    /// [`LaneEnvironment::observe_lanes`].
    pub observe: fn(&E::Config, &LaneGroup<'_, E>, &mut [&mut [u8]]),
}

impl<E: Environment> Clone for LaneEnvHooks<E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: Environment> Copy for LaneEnvHooks<E> {}

/// The value of [`Environment::LANE_ENV`] for a [`LaneEnvironment`]. Fails to
/// compile when the state is not lane-wise or a kernel would draw RNG
/// values (ADR-0090 4.2).
#[must_use]
pub const fn lane_env_hooks<E: LaneEnvironment>() -> LaneEnvHooks<E> {
    const {
        assert!(
            <E::State as EnvState>::LANES,
            "LaneEnvironment needs a #[state(lanes)] state (ADR-0090)"
        );
        assert!(
            !E::STEP_DRAWS_RNG && !E::OBSERVE_DRAWS_RNG,
            "LaneEnvironment kernels draw no RNG values (ADR-0090 4.2)"
        );
    }
    LaneEnvHooks {
        step: E::step_lanes,
        observe: E::observe_lanes,
    }
}
