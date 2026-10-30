// SPDX-License-Identifier: MIT OR Apache-2.0

//! A two-agent test environment: each agent adds its action plus a random
//! bonus to its counter; agent 1 dies when its counter passes `kill_at`;
//! the episode terminates after `length` steps.

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Bool8, EnvState, Environment, EpisodeEnd, NoAdaptive, ObsLeaf, Observation,
    RecordHeader, SpaceError, StepOut, Tier,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Counts {
    pub value: [i32; 2],
    pub steps: u32,
    pub dead: Bool8,
    pub reserved: tessera_rl_core::Reserved<3>,
}

#[derive(EnvConfig, Debug, Clone)]
pub struct CountConfig {
    #[config(default = 5, range = 1..=1000)]
    pub length: u32,
    #[config(default = 1000)]
    pub kill_at: i32,
}

#[derive(Observation)]
pub struct CountObs {
    #[space(box(low = -1_000_000, high = 1_000_000, shape = [2]))]
    pub values: ObsLeaf<i32>,
}

pub struct Counter;

pub const AGENTS: [AgentSlot; 2] = [
    AgentSlot::FIRST,
    match AgentSlot::new(1, 2) {
        Some(slot) => slot,
        None => AgentSlot::FIRST,
    },
];

impl Environment for Counter {
    type State = Counts;
    type Config = CountConfig;
    type Obs = CountObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = 2;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        CountObs::space(cfg)
    }

    fn action_space(_: &CountConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 3 })
    }

    fn reset(_: &CountConfig, rng: &mut StepRng, state: &mut Counts) {
        state.value = [rng.next_u32().cast_signed() % 100, 0];
    }

    fn step(
        cfg: &CountConfig,
        state: &mut Counts,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        for agent in AGENTS {
            if agent.index() == 1 && state.dead.get() {
                continue;
            }
            let bonus = i32::from(rng.next_u32().is_multiple_of(2));
            let inc = actions.discrete(agent, 0).unwrap_or(0) + bonus;
            if let Some(v) = state.value.get_mut(agent.index()) {
                *v += inc;
                if agent.index() == 1 && *v > cfg.kill_at {
                    state.dead = Bool8::TRUE;
                    out.agent_died(agent);
                }
            }
            #[expect(clippy::cast_precision_loss, reason = "small test values")]
            out.set_reward(agent, inc as f32);
        }
        state.steps += 1;
        if state.steps >= cfg.length {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &CountConfig,
        state: &Counts,
        agent: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, CountObs>,
    ) {
        let noise = i32::from(rng.next_u32().is_multiple_of(2));
        if let Some(mut leaf) = out.leaf_mut(CountObs::VALUES) {
            let own = state.value.get(agent.index()).copied().unwrap_or(0);
            leaf.set(0, own);
            leaf.set(1, noise);
        }
    }
}

const _: fn() = || {
    fn is_state<S: EnvState>() {}
    is_state::<Counts>();
};
