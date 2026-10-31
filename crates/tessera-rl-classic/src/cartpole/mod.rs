// SPDX-License-Identifier: MIT OR Apache-2.0

//! CartPole (`tessera/CartPole-v1`): Gymnasium `CartPole-v1` dynamics,
//! rewards and termination, single agent, cross-platform tier.
//!
//! - Observation: `Box(f32, [4])`, `[x, x_dot, theta, theta_dot]`, with
//!   Gymnasium's bounds (`±4.8`, `±inf`, `±24°` as f32, `±inf`).
//! - Action: `Discrete(2)`; 0 pushes left, 1 pushes right.
//! - Reward: 1 per step, including the terminating one.
//! - Reset: each component uniform in `[-0.05, 0.05)`.
//! - Ends: terminated past the thresholds. Truncation is the engine's
//!   step limit (SPEC 5.4) with `TimeLimit` semantics: `CartPole-v1` is this
//!   environment with `PoolOptions::max_episode_steps` set to
//!   [`MAX_EPISODE_STEPS`]; a terminal step at the limit reports both flags.
//!
//! Sources: Gymnasium v1.0.0,
//! <https://github.com/Farama-Foundation/Gymnasium/blob/v1.0.0/gymnasium/envs/classic_control/cartpole.py>;
//! A. G. Barto, R. S. Sutton and C. W. Anderson, "Neuronlike adaptive
//! elements that can solve difficult learning control problems", IEEE
//! Transactions on Systems, Man, and Cybernetics SMC-13(5), 1983,
//! <https://doi.org/10.1109/TSMC.1983.6313077>.

pub mod physics;

use tessera_rl_core::io::{ActionWriter, Actions, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, Heuristic, InvariantViolation, LeafId, NoAdaptive,
    Observation, RecordHeader, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState};
use tessera_rl_num::rng::StepRng;

/// The step limit of `CartPole-v1`, to set as the pool's
/// `max_episode_steps`.
pub const MAX_EPISODE_STEPS: u32 = 500;

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/CartPole-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;

/// Half-width of the reset distribution.
const RESET_BOUND: f64 = 0.05;

/// CartPole configuration: `CartPole-v1` has no parameters.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct CartPoleConfig {}

/// CartPole state: the four physical components.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct CartPoleState {
    /// `[x, x_dot, theta, theta_dot]`.
    pub s: [f64; 4],
}

/// CartPole observation: one `f32` box leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CartPoleObs;

impl CartPoleObs {
    /// The `[x, x_dot, theta, theta_dot]` leaf.
    pub const STATE: LeafId<f32> = LeafId::new(0);
}

/// Gymnasium's float32 observation bounds: the thresholds doubled, rounded
/// to `f32`, and infinity for the velocities.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the bounds are defined as float32 values"
)]
fn obs_high() -> Vec<f64> {
    [
        physics::X_THRESHOLD * 2.0,
        f64::INFINITY,
        physics::THETA_THRESHOLD * 2.0,
        f64::INFINITY,
    ]
    .map(|v| f64::from(v as f32))
    .to_vec()
}

impl Observation<CartPoleConfig> for CartPoleObs {
    const LEAVES: usize = 1;

    fn space(_: &CartPoleConfig) -> Result<Space, SpaceError> {
        let high = obs_high();
        Ok(Space::Box {
            shape: vec![4],
            dtype: DType::F32,
            bounds: BoxBounds::Float {
                low: high.iter().map(|v| -v).collect(),
                high,
            },
        })
    }
}

/// The CartPole environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CartPole;

impl Environment for CartPole {
    type State = CartPoleState;
    type Config = CartPoleConfig;
    type Obs = CartPoleObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &CartPoleConfig) -> Result<Space, SpaceError> {
        CartPoleObs::space(cfg)
    }

    fn action_space(_: &CartPoleConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(_: &CartPoleConfig, rng: &mut StepRng, state: &mut CartPoleState) {
        for v in &mut state.s {
            *v = -RESET_BOUND + 2.0 * RESET_BOUND * rng.unit_f64();
        }
    }

    fn step(
        _: &CartPoleConfig,
        state: &mut CartPoleState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let push_right = actions.discrete(AgentSlot::FIRST, 0) == Some(1);
        state.s = physics::step(state.s, push_right);
        out.set_reward(AgentSlot::FIRST, 1.0);
        if physics::is_terminal(state.s) {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "observations are float32, as in Gymnasium"
    )]
    fn observe(
        _: &CartPoleConfig,
        state: &CartPoleState,
        _: AgentSlot,
        _: &mut tessera_rl_num::rng::ObsRng,
        out: &mut ObsWriter<'_, '_, CartPoleObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(CartPoleObs::STATE) {
            for (i, v) in state.s.iter().enumerate() {
                leaf.set(i, *v as f32);
            }
        }
    }
}

/// Imported states must be finite.
impl Validate for CartPole {
    fn validate(_: &CartPoleConfig, state: &CartPoleState) -> Result<(), InvariantViolation> {
        if !state.s.iter().all(|v| v.is_finite()) {
            return Err(InvariantViolation {
                code: "a non-finite physical component",
            });
        }
        Ok(())
    }

    /// ADR-0040: once reset, the single slot is always alive.
    fn validate_record(
        cfg: &CartPoleConfig,
        header: &tessera_rl_core::HeaderView<'_>,
        state: &CartPoleState,
    ) -> Result<(), InvariantViolation> {
        Self::validate(cfg, state)?;
        let alive = header.agent(0).is_some_and(|a| a.alive);
        if header.phase() != tessera_rl_core::Phase::NeverReset && !alive {
            return Err(InvariantViolation {
                code: "a reset record whose agent is not alive",
            });
        }
        Ok(())
    }
}

/// Scripted baseline: push toward the side the pole is falling to.
impl Heuristic for CartPole {
    fn act(
        _: &CartPoleConfig,
        state: &CartPoleState,
        _: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let [_, _, theta, theta_dot] = state.s;
        out.set_discrete(0, i32::from(theta + 0.5 * theta_dot > 0.0));
    }
}
