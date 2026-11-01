// SPDX-License-Identifier: MIT OR Apache-2.0

//! Mixed turns (ADR-0013): any subset of the alive agents acts, including
//! none (a step that reads no action), and imported records must carry the
//! acting flags the environment names.
//!
//! The fixture cycles through three phases: both agents act, only agent 0
//! acts, then nobody acts (a chance step). Each acting agent adds its
//! action to its own counter.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Bool8, Environment, EpisodeEnd, NoAdaptive, ObsLeaf, Observation, RecordHeader,
    SpaceError, StepOut, Tier, TurnStructure,
};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, PoolOptions, Reseed,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Phases {
    phase: u32,
    sums: [u32; 2],
    steps: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct PhaseConfig {
    #[config(default = 9, range = 1..=100)]
    length: u32,
}

#[derive(Observation)]
struct PhaseObs {
    #[space(box(low = 0, high = 1000, shape = [1]))]
    phase: ObsLeaf<i32>,
}

struct PhaseEnv;

const AGENTS: [AgentSlot; 2] = [
    AgentSlot::FIRST,
    match AgentSlot::new(1, 2) {
        Some(s) => s,
        None => AgentSlot::FIRST,
    },
];

/// Who acts in `phase`: both, agent 0 only, nobody.
fn acts(phase: u32, agent: AgentSlot) -> bool {
    match phase % 3 {
        0 => true,
        1 => agent.index() == 0,
        _ => false,
    }
}

impl Environment for PhaseEnv {
    type State = Phases;
    type Config = PhaseConfig;
    type Obs = PhaseObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = 2;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Mixed;

    fn observation_space(cfg: &PhaseConfig) -> Result<Space, SpaceError> {
        PhaseObs::space(cfg)
    }

    fn action_space(_: &PhaseConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 4 })
    }

    fn acting(_: &PhaseConfig, state: &Phases, agent: AgentSlot) -> bool {
        acts(state.phase, agent)
    }

    fn reset(_: &PhaseConfig, _: &mut StepRng, state: &mut Phases) {
        *state = Phases::default();
    }

    fn step(
        cfg: &PhaseConfig,
        state: &mut Phases,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        for agent in AGENTS {
            if acts(state.phase, agent) {
                let a = u32::try_from(actions.discrete(agent, 0).unwrap_or(0)).unwrap_or(0);
                if let Some(s) = state.sums.get_mut(agent.index()) {
                    *s += a;
                }
            }
        }
        state.phase += 1;
        state.steps += 1;
        if state.steps >= cfg.length {
            out.end_episode(EpisodeEnd::Truncated);
        }
    }

    fn observe(
        _: &PhaseConfig,
        state: &Phases,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, PhaseObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(PhaseObs::PHASE) {
            leaf.set(0, i32::try_from(state.phase % 3).unwrap_or(0));
        }
    }
}

impl tessera_rl_core::Validate for PhaseEnv {
    fn validate(
        cfg: &PhaseConfig,
        state: &Phases,
    ) -> Result<(), tessera_rl_core::InvariantViolation> {
        if state.steps > cfg.length {
            return Err(tessera_rl_core::InvariantViolation {
                code: "steps beyond the episode length",
            });
        }
        Ok(())
    }
}

fn pool() -> EnvPool<PhaseEnv> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(5))
    };
    let mut p = EnvPool::new(PhaseConfig::default(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn step(p: &mut EnvPool<PhaseEnv>, act: [i32; 2]) {
    p.step(ActionInput {
        discrete: &act,
        continuous: &[],
    })
    .unwrap();
}

fn acting(p: &EnvPool<PhaseEnv>) -> Vec<bool> {
    p.outputs().acting().iter().map(|f| f.get()).collect()
}

#[test]
fn test_turns_mixed_acting_subset_follows_the_environment() {
    let mut p = pool();
    assert_eq!(acting(&p), [true, true]);
    step(&mut p, [1, 2]);
    assert_eq!(acting(&p), [true, false]);
    step(&mut p, [1, 3]);
    assert_eq!(acting(&p), [false, false]);
    step(&mut p, [3, 3]);
    assert_eq!(acting(&p), [true, true]);
    // Only acting agents' actions counted: 1 + 1 for agent 0, 2 for agent 1.
    assert_eq!(p.snapshot(0).unwrap().state().sums, [2, 2]);
}

#[test]
fn test_turns_mixed_step_with_no_acting_agent_runs_and_reads_nothing() {
    let (mut a, mut b) = (pool(), pool());
    for p in [&mut a, &mut b] {
        step(p, [0, 0]);
        step(p, [0, 0]);
    }
    assert_eq!(acting(&a), [false, false]);
    // Nobody acts: any (valid) actions give the same next state.
    step(&mut a, [0, 0]);
    step(&mut b, [3, 3]);
    assert_eq!(a.state_hash(0).unwrap(), b.state_hash(0).unwrap());
    assert!(!a.is_poisoned());
}

#[test]
fn test_turns_mixed_import_rejects_acting_flags_outside_the_turn_order() {
    let mut p = pool();
    step(&mut p, [0, 0]);
    // Phase 1: only agent 0 acts; a record that makes agent 1 act is invalid.
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut header = *good.header();
    header.agents[1].acting = Bool8::TRUE;
    let bad = DecodedRecord::from_parts(header, *good.state()).unwrap();
    assert!(matches!(
        p.import(&[0], &[bad]),
        Err(EngineError::InvalidRecord { index: 0, .. })
    ));
    assert!(p.import(&[0], &[good]).is_ok());
    assert_eq!(acting(&p), [true, false]);
}
