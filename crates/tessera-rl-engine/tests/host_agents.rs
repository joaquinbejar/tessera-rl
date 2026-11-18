// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host pools with a runtime agent count (ADR-0065):
//! `HostEnvironment::initial_agents` and `SPAWNS`.

#![cfg(feature = "host")]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers unwrap and slice: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, EnvConfig, HostEnvironment, HostError, HostOut, HostSeed, SpaceError,
};
use tessera_rl_engine::{ActionInput, EngineError, HostPool, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

const SLOTS: usize = 4;

/// Action of one agent: nothing, or spawn into slot `SPAWN_BASE + k`.
const NOTHING: i32 = 0;
const SPAWN_BASE: i32 = 1;

#[derive(Debug, Clone, Copy)]
struct SquadConfig {
    initial: usize,
}

impl EnvConfig for SquadConfig {}

/// Up to four agents, `initial` of them alive at each reset; an action
/// `SPAWN_BASE + k` spawns slot `k`.
#[derive(Debug)]
struct Squad<const S: bool>;

impl<const S: bool> HostEnvironment for Squad<S> {
    const ENV_ID: &'static str = "test/Squad-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = SLOTS;
    const SPAWNS: bool = S;
    type Config = SquadConfig;

    fn new(_: &SquadConfig) -> Result<Self, HostError> {
        Ok(Self)
    }

    fn initial_agents(cfg: &SquadConfig) -> usize {
        cfg.initial
    }

    fn observation_space(_: &SquadConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn action_space(_: &SquadConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 8 })
    }

    fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {}

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut HostOut<'_>) {
        for slot in 0..SLOTS {
            let agent = AgentSlot::new(u16::try_from(slot).unwrap(), SLOTS).unwrap();
            let a = actions.discrete(agent, 0).unwrap_or(0);
            if a >= SPAWN_BASE {
                let target = u16::try_from(a - SPAWN_BASE).unwrap();
                if let Some(t) = AgentSlot::new(target, SLOTS) {
                    out.agent_spawned(t);
                }
            }
        }
    }

    fn observe(&self, _: AgentSlot, _: &mut ObsWriter<'_, '_, ()>) {}
}

fn options(n: u32) -> PoolOptions {
    PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(1))
}

fn alive(out: &tessera_rl_engine::Outputs, i: usize) -> Vec<bool> {
    out.alive()[i * SLOTS..(i + 1) * SLOTS]
        .iter()
        .map(|f| f.get())
        .collect()
}

fn acting(out: &tessera_rl_engine::Outputs, i: usize) -> Vec<bool> {
    out.acting()[i * SLOTS..(i + 1) * SLOTS]
        .iter()
        .map(|f| f.get())
        .collect()
}

fn step<const S: bool>(p: &mut HostPool<Squad<S>, SLOTS>, actions: &[i32]) {
    p.step(ActionInput {
        discrete: actions,
        continuous: &[],
    })
    .unwrap();
}

#[test]
fn test_host_initial_agents_out_of_range_refused_at_construction() {
    for initial in [0, SLOTS + 1] {
        let err =
            HostPool::<Squad<false>, SLOTS>::new(&SquadConfig { initial }, options(1)).unwrap_err();
        assert!(
            matches!(err, EngineError::Config(ref e) if e.field == "initial_agents"),
            "{initial}: {err:?}"
        );
    }
}

#[test]
fn test_host_initial_agents_reset_makes_only_the_first_slots_alive() {
    let mut p =
        HostPool::<Squad<false>, SLOTS>::new(&SquadConfig { initial: 2 }, options(3)).unwrap();
    let out = p.reset_all(Reseed::Keep).unwrap();
    for i in 0..3 {
        assert_eq!(alive(out, i), [true, true, false, false]);
        assert_eq!(acting(out, i), [true, true, false, false]);
    }
    let out = p.reset(&[1], Reseed::Keep).unwrap();
    assert_eq!(alive(out, 1), [true, true, false, false]);
}

#[test]
fn test_host_spawns_fill_empty_slots_when_declared() {
    let mut p =
        HostPool::<Squad<true>, SLOTS>::new(&SquadConfig { initial: 2 }, options(1)).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    // Agent 0 spawns slot 3.
    step(&mut p, &[SPAWN_BASE + 3, NOTHING, NOTHING, NOTHING]);
    let out = p.outputs().unwrap();
    assert_eq!(alive(out, 0), [true, true, false, true]);
    // A reset empties it again.
    let out = p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(alive(out, 0), [true, true, false, false]);
}

#[test]
fn test_host_spawn_without_spawns_is_an_invalid_event_never_applied() {
    let mut p =
        HostPool::<Squad<false>, SLOTS>::new(&SquadConfig { initial: 2 }, options(1)).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let result = p.step(ActionInput {
        discrete: &[SPAWN_BASE + 3, NOTHING, NOTHING, NOTHING],
        continuous: &[],
    });
    if cfg!(feature = "debug-checks") {
        // An invalid agent event poisons under debug-checks.
        assert!(
            matches!(
                result,
                Err(EngineError::InvalidAgentEvent {
                    instance: 0,
                    agent: 3
                })
            ),
            "{result:?}"
        );
        return;
    }
    let out = result.unwrap();
    assert_eq!(alive(out, 0), [true, true, false, false]);
    assert!(!out.alive()[3].get());
}
