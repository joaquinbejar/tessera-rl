// SPDX-License-Identifier: MIT OR Apache-2.0

//! A `LaneEnvironment` whose step draws RNG values is refused at compile
//! time by `lane_env_hooks` (ADR-0090 decision 4.2).

use tessera_rl::classic::cartpole::{CartPole, CartPoleConfig, CartPoleObs, CartPoleState};
use tessera_rl::core::io::{Actions, ObsWriter};
use tessera_rl::core::space::Space;
use tessera_rl::core::{
    AgentSlot, Environment, LANES, LaneEnvHooks, LaneEnvironment, LaneGroup, LaneOut, NoAdaptive,
    RecordHeader, SpaceError, StepOut, Tier, lane_env_hooks,
};
use tessera_rl::num::rng::{ObsRng, StepRng};
use tessera_rl::simd::Resolved;
use tessera_rl_macros::EnvState;

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
#[state(lanes)]
struct S {
    x: f64,
    x_dot: f64,
    theta: f64,
    theta_dot: f64,
}

impl S {
    const fn rows(self) -> CartPoleState {
        CartPoleState {
            s: [self.x, self.x_dot, self.theta, self.theta_dot],
        }
    }
}

struct Drawing;

impl Environment for Drawing {
    type State = S;
    type Config = CartPoleConfig;
    type Obs = CartPoleObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;
    const LANE_ENV: Option<LaneEnvHooks<Self>> = Some(lane_env_hooks::<Self>());

    fn observation_space(cfg: &CartPoleConfig) -> Result<Space, SpaceError> {
        CartPole::observation_space(cfg)
    }

    fn action_space(cfg: &CartPoleConfig) -> Result<Space, SpaceError> {
        CartPole::action_space(cfg)
    }

    fn reset(_: &CartPoleConfig, _: &mut StepRng, state: &mut S) {
        *state = S::default();
    }

    fn step(
        _: &CartPoleConfig,
        _: &mut S,
        _: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
    }

    fn observe(
        cfg: &CartPoleConfig,
        state: &S,
        agent: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, CartPoleObs>,
    ) {
        CartPole::observe(cfg, &state.rows(), agent, rng, out);
    }
}

impl LaneEnvironment for Drawing {
    const STEP_DRAWS_RNG: bool = true;
    const OBSERVE_DRAWS_RNG: bool = false;

    fn step_lanes(
        _: &CartPoleConfig,
        _: Resolved,
        _: &mut LaneGroup<'_, Self>,
        _: &[i32; LANES],
        _: &mut LaneOut,
    ) {
    }

    fn observe_lanes(_: &CartPoleConfig, _: &LaneGroup<'_, Self>, _: &mut [&mut [u8]]) {}
}

fn main() {
    let _ = <Drawing as Environment>::LANE_ENV;
}
