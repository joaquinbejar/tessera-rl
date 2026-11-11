// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pendulum (`tessera/Pendulum-v1`): Gymnasium `Pendulum-v1` dynamics and
//! rewards, single agent, cross-platform tier.
//!
//! - Observation: `Box(f32, [3])`, `[cos theta, sin theta, theta_dot]`, with
//!   Gymnasium's bounds.
//! - Action: `Box(f32, [1])`, the torque in `[-2, 2]`. The engine refuses
//!   NaN, infinite or out-of-range values; the Gymnasium adapter clips first.
//! - Reward: `-(angle^2 + 0.1 * theta_dot^2 + 0.001 * u^2)`.
//! - Reset: `theta` uniform in `[-pi, pi]`, `theta_dot` uniform in `[-1, 1]`.
//! - Ends: never terminated. Truncation is the engine's step limit with
//!   `TimeLimit` semantics: `Pendulum-v1` is this environment with
//!   `PoolOptions::max_episode_steps` set to [`MAX_EPISODE_STEPS`].
//!
//! The reward's equivalence with Gymnasium is bounded, not exact (ADR-0050).
//!
//! Sources: Gymnasium v1.4.0,
//! <https://github.com/Farama-Foundation/Gymnasium/blob/v1.4.0/gymnasium/envs/classic_control/pendulum.py>.

pub mod physics;
mod validate;

use tessera_rl_core::io::{ActionWriter, Actions, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, Environment, Heuristic, LeafId, NoAdaptive, Observation, RecordHeader, SpaceError,
    StepOut, Tier,
};
use tessera_rl_macros::{EnvConfig, EnvState};
use tessera_rl_num::math;
use tessera_rl_num::rng::StepRng;

/// The step limit of `Pendulum-v1`, to set as the pool's `max_episode_steps`.
pub const MAX_EPISODE_STEPS: u32 = 200;

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/Pendulum-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;

// Registration under the versioned id (SPEC 10, ADR-0054).
tessera_rl_core::register!(Pendulum, id = "tessera/Pendulum-v1");

/// Pendulum configuration: version 1 has no parameters (`g` is fixed at 10).
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct PendulumConfig {}

/// Pendulum state.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct PendulumState {
    /// `[theta, theta_dot]`.
    pub s: [f64; 2],
}

/// Pendulum observation: one `f32` box leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PendulumObs;

impl PendulumObs {
    /// The `[cos theta, sin theta, theta_dot]` leaf.
    pub const STATE: LeafId<f32> = LeafId::new(0);
}

impl Observation<PendulumConfig> for PendulumObs {
    const LEAVES: usize = 1;

    fn space(_: &PendulumConfig) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![3],
            dtype: DType::F32,
            bounds: BoxBounds::Float {
                low: vec![-1.0, -1.0, -physics::MAX_SPEED],
                high: vec![1.0, 1.0, physics::MAX_SPEED],
            },
        })
    }
}

/// The pendulum environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pendulum;

impl Environment for Pendulum {
    type State = PendulumState;
    type Config = PendulumConfig;
    type Obs = PendulumObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &PendulumConfig) -> Result<Space, SpaceError> {
        PendulumObs::space(cfg)
    }

    fn action_space(_: &PendulumConfig) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![1],
            dtype: DType::F32,
            bounds: BoxBounds::Float {
                low: vec![f64::from(-physics::MAX_TORQUE)],
                high: vec![f64::from(physics::MAX_TORQUE)],
            },
        })
    }

    fn reset(_: &PendulumConfig, rng: &mut StepRng, state: &mut PendulumState) {
        let pi = core::f64::consts::PI;
        let theta = -pi + (pi - (-pi)) * rng.unit_f64();
        let theta_dot = -1.0 + (1.0 - (-1.0)) * rng.unit_f64();
        state.s = [theta, theta_dot];
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the public reward is float32"
    )]
    fn step(
        _: &PendulumConfig,
        state: &mut PendulumState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        // The engine validates the action against the box.
        let u = actions.continuous(AgentSlot::FIRST, 0).unwrap_or(0.0);
        let reward = physics::reward(state.s, u);
        state.s = physics::step(state.s, u);
        out.set_reward(AgentSlot::FIRST, reward as f32);
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "observations are float32, as in Gymnasium"
    )]
    fn observe(
        _: &PendulumConfig,
        state: &PendulumState,
        _: AgentSlot,
        _: &mut tessera_rl_num::rng::ObsRng,
        out: &mut ObsWriter<'_, '_, PendulumObs>,
    ) {
        let [theta, theta_dot] = state.s;
        if let Some(mut leaf) = out.leaf_mut(PendulumObs::STATE) {
            leaf.set(0, math::cos(theta) as f32);
            leaf.set(1, math::sin(theta) as f32);
            leaf.set(2, theta_dot as f32);
        }
    }
}

/// Scripted baseline: energy pumping far from the top, PD control near it.
impl Heuristic for Pendulum {
    #[expect(clippy::cast_possible_truncation, reason = "the torque is float32")]
    fn act(
        _: &PendulumConfig,
        state: &PendulumState,
        _: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let [theta, theta_dot] = state.s;
        let angle = physics::angle_normalize(theta);
        let torque = if angle.abs() < 0.5 {
            -(12.0 * angle + 3.0 * theta_dot)
        } else {
            // Energy of a uniform rod: 1/6 speed^2 + 5 cos(theta), 5 at the top.
            let energy = theta_dot * theta_dot / 6.0 + 5.0 * math::cos(theta);
            let direction = if theta_dot >= 0.0 { 1.0 } else { -1.0 };
            (5.0 - energy) * direction
        };
        let limit = f64::from(physics::MAX_TORQUE);
        out.set_continuous(0, torque.clamp(-limit, limit) as f32);
    }
}

impl tessera_rl_core::Persist for Pendulum {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
