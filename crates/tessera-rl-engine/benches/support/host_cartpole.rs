// SPDX-License-Identifier: MIT OR Apache-2.0

//! CartPole as a host environment (ADR-0062), for the host pool's
//! instruction counts: the same physics, spaces, reward and termination as
//! the first-party `CartPole`, with the state held by the instance.
//!
//! Its reset draws the four components from the seed and episode index with
//! splitmix64, not from `StepRng` (host environments have none); the draw is
//! fixed by its inputs, so every run reaches the same states.

use tessera_rl_classic::CartPoleConfig;
use tessera_rl_classic::cartpole::{CartPoleObs, physics};
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::leaf::LeafId;
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, EpisodeEnd, HostEnvironment, HostError, HostSeed, Observation, SpaceError, StepOut,
};

/// The reset range of every component, as the first-party `CartPole`.
const RESET_BOUND: f64 = 0.05;

/// The observation's single leaf: the four components as `f32`.
const STATE: LeafId<f32> = LeafId::new(0);

/// splitmix64: one step of the reset draw.
const fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// CartPole, host-driven.
#[derive(Debug, Default)]
pub struct HostCartPole {
    s: [f64; 4],
}

impl HostEnvironment for HostCartPole {
    const ENV_ID: &'static str = "bench/HostCartPole-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 1;

    type Config = CartPoleConfig;

    fn new(_: &CartPoleConfig) -> Result<Self, HostError> {
        Ok(Self::default())
    }

    fn observation_space(cfg: &CartPoleConfig) -> Result<Space, SpaceError> {
        CartPoleObs::space(cfg)
    }

    fn action_space(_: &CartPoleConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "53 random bits to a unit f64, exactly"
    )]
    fn reset(&mut self, seed: HostSeed, _: &mut StepOut<'_>) {
        let mut s = seed.seed() ^ (u64::from(seed.episode_index()) << 32);
        for v in &mut self.s {
            let unit = (splitmix(&mut s) >> 11) as f64 / (1_u64 << 53) as f64;
            *v = -RESET_BOUND + 2.0 * RESET_BOUND * unit;
        }
    }

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut StepOut<'_>) {
        let push_right = actions.discrete(AgentSlot::FIRST, 0) == Some(1);
        self.s = physics::step(self.s, push_right);
        out.set_reward(AgentSlot::FIRST, 1.0);
        if physics::is_terminal(self.s) {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "observations are float32, as in Gymnasium"
    )]
    fn observe(&self, _: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>) {
        if let Some(mut leaf) = obs.leaf_mut(STATE) {
            for (i, v) in self.s.iter().enumerate() {
                leaf.set(i, *v as f32);
            }
        }
    }
}
