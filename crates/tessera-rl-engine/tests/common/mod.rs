// SPDX-License-Identifier: MIT OR Apache-2.0

//! A two-agent test environment: each agent adds its action plus a random
//! bonus to its counter; agent 1 dies when its counter passes `kill_at`;
//! the episode terminates after `length` steps.

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Bool8, EnvState, Environment, EpisodeEnd, NoAdaptive, ObsLeaf,
    Observation, RecordHeader, SpaceError, StepOut, Tier,
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

/// Test switches are independent flags by design.
#[expect(clippy::struct_excessive_bools, reason = "independent test switches")]
#[derive(EnvConfig, Debug, Clone)]
pub struct CountConfig {
    #[config(default = 5, range = 1..=1000)]
    pub length: u32,
    #[config(default = 1000)]
    pub kill_at: i32,
    /// Panic in `step` when the step counter reaches this value (0: never).
    #[config(default = 0)]
    pub panic_at: u32,
    /// Panic in `reset`.
    #[config(default = false)]
    pub panic_on_reset: bool,
    /// Panic in `ActionMask::noop`.
    #[config(default = false)]
    pub panic_in_noop: bool,
    /// Panic in `ActionMask::action_mask` once a step has run.
    #[config(default = false)]
    pub panic_in_mask: bool,
    /// Reward every agent with -0.0 instead of its increment.
    #[config(default = false)]
    pub negative_zero_reward: bool,
    /// Number of discrete choices per head.
    #[config(default = 3, range = 3..=2_147_483_648)]
    pub actions_n: u32,
    /// Number of discrete action heads per agent; only head 0 is used.
    #[config(default = 1, range = 1..=4)]
    pub action_heads: u32,
    /// Invalid spawns (SPEC 9.1): agent 1 requests a spawn in the step it
    /// dies, and agent 0 (alive) at step `spawn_alive_at` (0: never).
    #[config(default = false)]
    pub spawn_when_dying: bool,
    #[config(default = 0)]
    pub spawn_alive_at: u32,
    /// Agent 1 reports its death again in the step after it died.
    #[config(default = false)]
    pub die_twice: bool,
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

    fn action_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        if cfg.action_heads == 1 {
            return Ok(Space::Discrete { n: cfg.actions_n });
        }
        let heads = usize::try_from(cfg.action_heads).unwrap_or(1);
        Ok(Space::MultiDiscrete {
            nvec: vec![cfg.actions_n; heads],
        })
    }

    fn reset(cfg: &CountConfig, rng: &mut StepRng, state: &mut Counts) {
        // Panic injection for the poisoning tests.
        assert!(!cfg.panic_on_reset, "injected reset panic");
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
                if cfg.die_twice {
                    out.agent_died(agent);
                }
                continue;
            }
            let bonus = i32::from(rng.next_u32().is_multiple_of(2));
            let inc = actions.discrete(agent, 0).unwrap_or(0) + bonus;
            if let Some(v) = state.value.get_mut(agent.index()) {
                *v += inc;
                if agent.index() == 1 && *v > cfg.kill_at {
                    state.dead = Bool8::TRUE;
                    out.agent_died(agent);
                    if cfg.spawn_when_dying {
                        out.agent_spawned(agent);
                    }
                }
            }
            #[expect(clippy::cast_precision_loss, reason = "small test values")]
            let reward = if cfg.negative_zero_reward {
                -0.0
            } else {
                inc as f32
            };
            out.set_reward(agent, reward);
        }
        state.steps += 1;
        if cfg.spawn_alive_at != 0 && state.steps == cfg.spawn_alive_at {
            out.agent_spawned(AgentSlot::FIRST);
        }
        if cfg.panic_at != 0 && state.steps == cfg.panic_at {
            panic_now();
        }
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

/// Action 2 is illegal for agent 0 and legal for agent 1; the no-op is 0.
impl ActionMask for Counter {
    fn action_mask(cfg: &CountConfig, state: &Counts, agent: AgentSlot, out: &mut MaskWriter<'_>) {
        // Panic injection for the poisoning tests.
        assert!(
            !(cfg.panic_in_mask && state.steps > 0),
            "injected mask panic"
        );
        out.fill(true);
        if agent.index() == 0 {
            out.set(2, false);
        }
    }

    fn noop(cfg: &CountConfig, _: &Counts, _: AgentSlot, out: &mut ActionWriter<'_>) {
        assert!(!cfg.panic_in_noop, "injected noop panic");
        out.set_discrete(0, 0);
    }
}

#[expect(clippy::panic, reason = "panic injection for poisoning tests")]
fn panic_now() {
    panic!("injected step panic");
}

const _: fn() = || {
    fn is_state<S: EnvState>() {}
    is_state::<Counts>();
};
