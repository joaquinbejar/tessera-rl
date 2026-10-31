// SPDX-License-Identifier: MIT OR Apache-2.0

//! CartPole (`tessera/CartPole-v1`): Gymnasium `CartPole-v1` dynamics,
//! rewards and termination, single agent, cross-platform tier.
//!
//! - Observation: `Box(f32, [4])`, `[x, x_dot, theta, theta_dot]`, with
//!   Gymnasium's bounds (`±4.8`, `±f32::MAX`, `±24°`, `±f32::MAX`).
//! - Action: `Discrete(2)`; 0 pushes left, 1 pushes right.
//! - Reward: 1 per step, including the terminating one.
//! - Reset: each component uniform in `[-0.05, 0.05)`.
//! - Ends: terminated past the thresholds; truncated after `max_steps`.

pub mod physics;

use tessera_rl_core::io::{ActionWriter, Actions, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, Heuristic, InvariantViolation, LeafId, NoAdaptive,
    Observation, RecordHeader, Reserved, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState};
use tessera_rl_num::rng::StepRng;

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/CartPole-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;

/// Half-width of the reset distribution.
const RESET_BOUND: f64 = 0.05;

/// CartPole configuration.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct CartPoleConfig {
    /// Steps after which the episode is truncated (500 in `CartPole-v1`).
    #[config(default = 500, range = 1..=1_000_000)]
    pub max_steps: u32,
}

/// CartPole state: the four physical components and the step count.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct CartPoleState {
    /// `[x, x_dot, theta, theta_dot]`.
    pub s: [f64; 4],
    /// Steps taken in this episode.
    pub steps: u32,
    /// Padding to a multiple of 8 bytes.
    pub reserved: Reserved<4>,
}

/// CartPole observation: one `f32` box leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CartPoleObs;

impl CartPoleObs {
    /// The `[x, x_dot, theta, theta_dot]` leaf.
    pub const STATE: LeafId<f32> = LeafId::new(0);
}

/// Gymnasium's float32 observation bounds: the thresholds doubled, rounded
/// to `f32`, and the largest finite `f32` for the velocities.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the bounds are defined as float32 values"
)]
fn obs_high() -> Vec<f64> {
    [
        physics::X_THRESHOLD * 2.0,
        f64::from(f32::MAX),
        physics::THETA_THRESHOLD * 2.0,
        f64::from(f32::MAX),
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
        state.steps = 0;
    }

    fn step(
        cfg: &CartPoleConfig,
        state: &mut CartPoleState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let push_right = actions.discrete(AgentSlot::FIRST, 0) == Some(1);
        state.s = physics::step(state.s, push_right);
        // Bounded by max_steps, checked below.
        state.steps += 1;
        out.set_reward(AgentSlot::FIRST, 1.0);
        if physics::is_terminal(state.s) {
            out.end_episode(EpisodeEnd::Terminated);
        } else if state.steps >= cfg.max_steps {
            out.end_episode(EpisodeEnd::Truncated);
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

/// Imported states must be finite and within one episode.
impl Validate for CartPole {
    fn validate(cfg: &CartPoleConfig, state: &CartPoleState) -> Result<(), InvariantViolation> {
        if !state.s.iter().all(|v| v.is_finite()) {
            return Err(InvariantViolation {
                code: "a non-finite physical component",
            });
        }
        if state.steps > cfg.max_steps {
            return Err(InvariantViolation {
                code: "steps beyond max_steps",
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
