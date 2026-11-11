// SPDX-License-Identifier: MIT OR Apache-2.0

//! Acrobot (`tessera/Acrobot-v1`): Gymnasium `Acrobot-v1` dynamics, rewards
//! and termination, single agent, cross-platform tier.
//!
//! - Observation: `Box(f32, [6])`, `[cos t1, sin t1, cos t2, sin t2, dt1,
//!   dt2]`, with Gymnasium's bounds (1 and `4 pi`, `9 pi` as f32).
//! - Action: `Discrete(3)`; torque -1, 0 or +1 on the joint between links.
//! - Reward: -1 per step, 0 on the terminating one.
//! - Reset: each component uniform in `[-0.1, 0.1]`, rounded to f32 as
//!   Gymnasium's reset does, then widened.
//! - Ends: terminated when the free end is above the bar by one link
//!   length. Truncation is the engine's step limit with `TimeLimit`
//!   semantics: `Acrobot-v1` is this environment with
//!   `PoolOptions::max_episode_steps` set to [`MAX_EPISODE_STEPS`].
//!
//! Sources: Gymnasium v1.4.0,
//! <https://github.com/Farama-Foundation/Gymnasium/blob/v1.4.0/gymnasium/envs/classic_control/acrobot.py>;
//! R. S. Sutton and A. G. Barto, "Reinforcement Learning: An Introduction",
//! MIT Press, 1998.

pub mod physics;
mod validate;

use tessera_rl_core::io::{ActionWriter, Actions, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, Heuristic, LeafId, NoAdaptive, Observation, RecordHeader,
    SpaceError, StepOut, Tier,
};
use tessera_rl_macros::{EnvConfig, EnvState};
use tessera_rl_num::math;
use tessera_rl_num::rng::StepRng;

/// The step limit of `Acrobot-v1`, to set as the pool's `max_episode_steps`.
pub const MAX_EPISODE_STEPS: u32 = 500;

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/Acrobot-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;

// Registration under the versioned id (SPEC 10, ADR-0054).
tessera_rl_core::register!(Acrobot, id = "tessera/Acrobot-v1");

/// Half-width of the reset distribution.
const RESET_BOUND: f64 = 0.1;

/// Acrobot configuration: version 1 has no parameters.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct AcrobotConfig {}

/// Acrobot state: the two joint angles and their velocities.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct AcrobotState {
    /// `[theta1, theta2, dtheta1, dtheta2]`.
    pub s: [f64; 4],
}

/// Acrobot observation: one `f32` box leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AcrobotObs;

impl AcrobotObs {
    /// The six-component leaf.
    pub const STATE: LeafId<f32> = LeafId::new(0);
}

/// Gymnasium's float32 upper bounds, widened back to f64.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the bounds are defined as float32 values"
)]
fn obs_high() -> Vec<f64> {
    [1.0, 1.0, 1.0, 1.0, physics::MAX_VEL_1, physics::MAX_VEL_2]
        .map(|v: f64| f64::from(v as f32))
        .to_vec()
}

impl Observation<AcrobotConfig> for AcrobotObs {
    const LEAVES: usize = 1;

    fn space(_: &AcrobotConfig) -> Result<Space, SpaceError> {
        let high = obs_high();
        Ok(Space::Box {
            shape: vec![6],
            dtype: DType::F32,
            bounds: BoxBounds::Float {
                low: high.iter().map(|v| -v).collect(),
                high,
            },
        })
    }
}

/// The acrobot environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Acrobot;

impl Environment for Acrobot {
    type State = AcrobotState;
    type Config = AcrobotConfig;
    type Obs = AcrobotObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &AcrobotConfig) -> Result<Space, SpaceError> {
        AcrobotObs::space(cfg)
    }

    fn action_space(_: &AcrobotConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 3 })
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "Gymnasium rounds the reset state to float32"
    )]
    fn reset(_: &AcrobotConfig, rng: &mut StepRng, state: &mut AcrobotState) {
        for v in &mut state.s {
            let draw = -RESET_BOUND + 2.0 * RESET_BOUND * rng.unit_f64();
            *v = f64::from(draw as f32);
        }
    }

    fn step(
        _: &AcrobotConfig,
        state: &mut AcrobotState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        // The engine validates the action against `Discrete(3)`.
        let action = actions.discrete(AgentSlot::FIRST, 0).unwrap_or(1);
        state.s = physics::step(state.s, action);
        if physics::is_terminal(state.s) {
            out.set_reward(AgentSlot::FIRST, 0.0);
            out.end_episode(EpisodeEnd::Terminated);
        } else {
            out.set_reward(AgentSlot::FIRST, -1.0);
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "observations are float32, as in Gymnasium"
    )]
    fn observe(
        _: &AcrobotConfig,
        state: &AcrobotState,
        _: AgentSlot,
        _: &mut tessera_rl_num::rng::ObsRng,
        out: &mut ObsWriter<'_, '_, AcrobotObs>,
    ) {
        let [t1, t2, d1, d2] = state.s;
        let values = [
            math::cos(t1),
            math::sin(t1),
            math::cos(t2),
            math::sin(t2),
            d1,
            d2,
        ];
        if let Some(mut leaf) = out.leaf_mut(AcrobotObs::STATE) {
            for (i, v) in values.iter().enumerate() {
                leaf.set(i, *v as f32);
            }
        }
    }
}

/// Scripted baseline: torque in the direction of `dtheta2`, the velocity
/// of the actuated joint, so the motor always adds energy.
impl Heuristic for Acrobot {
    fn act(
        _: &AcrobotConfig,
        state: &AcrobotState,
        _: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let [_, _, _, dtheta2] = state.s;
        out.set_discrete(0, if dtheta2 >= 0.0 { 2 } else { 0 });
    }
}

impl tessera_rl_core::Persist for Acrobot {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
