// SPDX-License-Identifier: MIT OR Apache-2.0

//! Mountain car (`tessera/MountainCar-v1`): Gymnasium `MountainCar-v0`
//! dynamics, rewards and termination, single agent, cross-platform tier.
//!
//! - Observation: `Box(f32, [2])`, `[position, velocity]`, with
//!   Gymnasium's bounds (`[-1.2, -0.07]` to `[0.6, 0.07]` as f32).
//! - Action: `Discrete(3)`; 0 accelerates left, 1 does nothing, 2
//!   accelerates right.
//! - Reward: -1 per step, including the terminating one.
//! - Reset: position uniform in `[-0.6, -0.4)`, velocity 0.
//! - Ends: terminated at the goal (`position >= 0.5` with
//!   `velocity >= 0`). Truncation is the engine's step limit with
//!   `TimeLimit` semantics: `MountainCar-v1` is this environment with
//!   `PoolOptions::max_episode_steps` set to [`MAX_EPISODE_STEPS`].
//!
//! Tessera's version 1 mirrors Gymnasium's `MountainCar-v0`; Tessera
//! versions are its own (SPEC 16.1).
//!
//! Sources: Gymnasium v1.4.0,
//! <https://github.com/Farama-Foundation/Gymnasium/blob/v1.4.0/gymnasium/envs/classic_control/mountain_car.py>;
//! A. W. Moore, "Efficient Memory-based Learning for Robot Control", doctoral
//! thesis, University of Cambridge, 1990.

pub mod physics;
mod validate;

use tessera_rl_core::io::{ActionWriter, Actions, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, Heuristic, LeafId, NoAdaptive, Observation, RecordHeader,
    SpaceError, StepOut, Tier,
};
use tessera_rl_macros::{EnvConfig, EnvState};
use tessera_rl_num::rng::StepRng;

/// The step limit of `MountainCar-v0`, to set as the pool's
/// `max_episode_steps`.
pub const MAX_EPISODE_STEPS: u32 = 200;

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/MountainCar-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;

/// The reset interval of the position, `[low, high)`.
const RESET_LOW: f64 = -0.6;
const RESET_HIGH: f64 = -0.4;

/// Mountain car configuration: version 1 has no parameters.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct MountainCarConfig {}

/// Mountain car state: position and velocity.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct MountainCarState {
    /// `[position, velocity]`.
    pub s: [f64; 2],
}

/// Mountain car observation: one `f32` box leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MountainCarObs;

impl MountainCarObs {
    /// The `[position, velocity]` leaf.
    pub const STATE: LeafId<f32> = LeafId::new(0);
}

/// Gymnasium's float32 bounds, widened back to f64.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the bounds are defined as float32 values"
)]
fn bounds() -> (Vec<f64>, Vec<f64>) {
    let f = |v: f64| f64::from(v as f32);
    (
        vec![f(physics::MIN_POSITION), f(-physics::MAX_SPEED)],
        vec![f(physics::MAX_POSITION), f(physics::MAX_SPEED)],
    )
}

impl Observation<MountainCarConfig> for MountainCarObs {
    const LEAVES: usize = 1;

    fn space(_: &MountainCarConfig) -> Result<Space, SpaceError> {
        let (low, high) = bounds();
        Ok(Space::Box {
            shape: vec![2],
            dtype: DType::F32,
            bounds: BoxBounds::Float { low, high },
        })
    }
}

/// The mountain car environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MountainCar;

impl Environment for MountainCar {
    type State = MountainCarState;
    type Config = MountainCarConfig;
    type Obs = MountainCarObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &MountainCarConfig) -> Result<Space, SpaceError> {
        MountainCarObs::space(cfg)
    }

    fn action_space(_: &MountainCarConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 3 })
    }

    fn reset(_: &MountainCarConfig, rng: &mut StepRng, state: &mut MountainCarState) {
        let position = RESET_LOW + (RESET_HIGH - RESET_LOW) * rng.unit_f64();
        state.s = [position, 0.0];
    }

    fn step(
        _: &MountainCarConfig,
        state: &mut MountainCarState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        // The engine validates the action against `Discrete(3)`.
        let action = actions.discrete(AgentSlot::FIRST, 0).unwrap_or(1);
        state.s = physics::step(state.s, action);
        out.set_reward(AgentSlot::FIRST, -1.0);
        if physics::is_terminal(state.s) {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "observations are float32, as in Gymnasium"
    )]
    fn observe(
        _: &MountainCarConfig,
        state: &MountainCarState,
        _: AgentSlot,
        _: &mut tessera_rl_num::rng::ObsRng,
        out: &mut ObsWriter<'_, '_, MountainCarObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(MountainCarObs::STATE) {
            for (i, v) in state.s.iter().enumerate() {
                leaf.set(i, *v as f32);
            }
        }
    }
}

/// Scripted baseline: accelerate in the direction of the velocity
/// (energy pumping); right when at rest.
impl Heuristic for MountainCar {
    fn act(
        _: &MountainCarConfig,
        state: &MountainCarState,
        _: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let [_, velocity] = state.s;
        out.set_discrete(0, if velocity >= 0.0 { 2 } else { 0 });
    }
}

impl tessera_rl_core::Persist for MountainCar {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
