// SPDX-License-Identifier: MIT OR Apache-2.0

//! The crate-internal agent event buffers across a chunk step (issue #152,
//! G1). `died`, `spawned` and `invalid_agent_event` have no public
//! accessor, so they are read here through the internal view. The chunk
//! clear fills them once per chunk: an event set in one step must not
//! survive into the next, neither in a stepped instance nor in one that
//! `NextStep`-resets in the middle of the chunk.

use core::num::NonZeroU32;

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Bool8, Environment, EpisodeEnd, NoAdaptive, ObsLeaf, Observation, RecordHeader,
    SpaceError, StepOut, Tier,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

use crate::{ActionInput, AutoReset, EnvPool, PoolOptions, Reseed};

/// Agent 0's action bits; agent 1's action is ignored.
const QUIET: i32 = 0;
const KILL: i32 = 1;
const SPAWN: i32 = 2;
const END: i32 = 4;

const AGENTS: usize = 2;
const SECOND: AgentSlot = match AgentSlot::new(1, 2) {
    Some(s) => s,
    None => AgentSlot::FIRST,
};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Quiet {
    steps: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct QuietConfig {}

#[derive(Observation)]
struct QuietObs {
    #[space(box(low = 0, high = 1, shape = [1]))]
    zero: ObsLeaf<i32>,
}

/// Two agents; agent 0's bits kill, spawn or end on slot 1.
struct Events;

impl Environment for Events {
    type State = Quiet;
    type Config = QuietConfig;
    type Obs = QuietObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = AGENTS;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &QuietConfig) -> Result<Space, SpaceError> {
        QuietObs::space(cfg)
    }

    fn action_space(_: &QuietConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 8 })
    }

    fn reset(_: &QuietConfig, _: &mut StepRng, state: &mut Quiet) {
        *state = Quiet::default();
    }

    fn step(
        _: &QuietConfig,
        state: &mut Quiet,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        state.steps += 1;
        let bits = actions.discrete(AgentSlot::FIRST, 0).unwrap_or(0);
        if bits & KILL != 0 {
            out.agent_died(SECOND);
        }
        if bits & SPAWN != 0 {
            out.agent_spawned(SECOND);
        }
        if bits & END != 0 {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &QuietConfig,
        _: &Quiet,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, QuietObs>,
    ) {
    }
}

const N: usize = 5;

fn step(p: &mut EnvPool<Events>, bits: [i32; N]) {
    let actions: Vec<i32> = bits.iter().flat_map(|&b| [b, QUIET]).collect();
    p.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap();
}

/// `(died, spawned, invalid_agent_event)` of every agent row, as bools.
fn events(p: &mut EnvPool<Events>) -> [Vec<bool>; 3] {
    let view = p.outputs.view_all();
    let read = |b: &[Bool8]| b.iter().map(|f| f.get()).collect::<Vec<_>>();
    [
        read(view.died),
        read(view.spawned),
        read(view.invalid_agent_event),
    ]
}

/// The rows of `(instance, slot)` pairs that are set, all others clear.
fn only(set: &[(usize, usize)]) -> Vec<bool> {
    (0..N * AGENTS)
        .map(|k| set.contains(&(k / AGENTS, k % AGENTS)))
        .collect()
}

#[test]
fn test_chunk_clear_agent_events_do_not_survive_a_step_or_a_mid_chunk_reset() {
    let options = PoolOptions {
        auto_reset: AutoReset::NextStep,
        ..PoolOptions::new(NonZeroU32::new(5).unwrap(), Seed::new(3))
    };
    let mut p = EnvPool::<Events>::new(QuietConfig {}, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(events(&mut p), [only(&[]), only(&[]), only(&[])]);

    // 0 kills slot 1; 1 kills it and ends; 2 spawns on the occupied slot
    // (invalid); 3 ends; 4 kills.
    step(&mut p, [KILL, KILL | END, SPAWN, END, KILL]);
    assert_eq!(
        events(&mut p),
        [only(&[(0, 1), (1, 1), (4, 1)]), only(&[]), only(&[(2, 1)])],
        "events of the first step"
    );

    // 1 and 3 reset instead of stepping, in the middle of the chunk; 4
    // spawns on its now empty slot; 0 and 2 are quiet.
    step(&mut p, [QUIET, KILL, QUIET, KILL, SPAWN]);
    assert_eq!(
        events(&mut p),
        [only(&[]), only(&[(4, 1)]), only(&[])],
        "the first step's events are gone; only the valid spawn is set"
    );

    step(&mut p, [QUIET; N]);
    assert_eq!(events(&mut p), [only(&[]), only(&[]), only(&[])]);
}
