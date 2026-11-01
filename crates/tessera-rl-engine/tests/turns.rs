// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sequential turns (ADR-0013): the acting mask follows the environment,
//! only the acting agent's action is required and read, masks are
//! validated only for it, the terminal reward reaches both agents in the
//! step the game ends, and snapshots continue the turn order.
//!
//! The fixture is a race to a target: two players alternate adding 1 or 2
//! to a shared total; whoever reaches the target exactly wins.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Bool8, Environment, EpisodeEnd, NoAdaptive, ObsLeaf, Observation,
    RecordHeader, SpaceError, StepOut, Tier, TurnStructure,
};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, IllegalActionPolicy, Masked, PoolOptions, Reseed,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Race {
    total: u32,
    to_move: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct RaceConfig {
    #[config(default = 7, range = 2..=100)]
    target: u32,
    /// Fault injection: name both agents as acting (a Sequential violation).
    #[config(default = false)]
    both_act: bool,
}

#[derive(Observation)]
struct RaceObs {
    #[space(box(low = 0, high = 1000, shape = [2]))]
    values: ObsLeaf<i32>,
}

struct RaceEnv;

const P0: AgentSlot = AgentSlot::FIRST;
const P1: AgentSlot = match AgentSlot::new(1, 2) {
    Some(s) => s,
    None => AgentSlot::FIRST,
};

impl Environment for RaceEnv {
    type State = Race;
    type Config = RaceConfig;
    type Obs = RaceObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = 2;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &RaceConfig) -> Result<Space, SpaceError> {
        RaceObs::space(cfg)
    }

    fn action_space(_: &RaceConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn acting(cfg: &RaceConfig, state: &Race, agent: AgentSlot) -> bool {
        cfg.both_act || agent.index() == state.to_move as usize
    }

    fn reset(_: &RaceConfig, _: &mut StepRng, state: &mut Race) {
        *state = Race::default();
    }

    fn step(
        cfg: &RaceConfig,
        state: &mut Race,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let (me, other) = if state.to_move == 0 {
            (P0, P1)
        } else {
            (P1, P0)
        };
        let add = u32::try_from(actions.discrete(me, 0).unwrap_or(0)).unwrap_or(0) + 1;
        state.total = (state.total + add).min(cfg.target);
        if state.total == cfg.target {
            out.set_reward(me, 1.0);
            out.set_reward(other, -1.0);
            out.end_episode(EpisodeEnd::Terminated);
        } else {
            state.to_move ^= 1;
        }
    }

    fn observe(
        _: &RaceConfig,
        state: &Race,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, RaceObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(RaceObs::VALUES) {
            leaf.set(0, i32::try_from(state.total).unwrap_or(0));
            leaf.set(1, i32::from(agent.index() == state.to_move as usize));
        }
    }
}

/// The race's invariants: the total never passes the target, and the
/// player to move is 0 or 1.
impl tessera_rl_core::Validate for RaceEnv {
    fn validate(cfg: &RaceConfig, state: &Race) -> Result<(), tessera_rl_core::InvariantViolation> {
        if state.total > cfg.target || state.to_move > 1 {
            return Err(tessera_rl_core::InvariantViolation {
                code: "total past the target or no player to move",
            });
        }
        Ok(())
    }
}

/// Adding 2 is illegal when it would pass the target; the no-op adds 1.
impl ActionMask for RaceEnv {
    fn action_mask(cfg: &RaceConfig, state: &Race, _: AgentSlot, out: &mut MaskWriter<'_>) {
        out.fill(true);
        if state.total + 2 > cfg.target {
            out.set(1, false);
        }
    }

    fn noop(_: &RaceConfig, _: &Race, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, 0);
    }
}

fn pool(cfg: RaceConfig) -> EnvPool<RaceEnv, Masked> {
    let options = PoolOptions {
        illegal_action: IllegalActionPolicy::Noop,
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(3))
    };
    let mut p = EnvPool::with_masks(cfg, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn race() -> EnvPool<RaceEnv, Masked> {
    pool(RaceConfig::default())
}

fn step(p: &mut EnvPool<RaceEnv, Masked>, act: [i32; 2]) {
    p.step(ActionInput {
        discrete: &act,
        continuous: &[],
    })
    .unwrap();
}

fn acting(p: &EnvPool<RaceEnv, Masked>) -> Vec<bool> {
    p.outputs().acting().iter().map(|f| f.get()).collect()
}

fn total(p: &EnvPool<RaceEnv, Masked>) -> u32 {
    let rec = p.snapshot(0).unwrap();
    rec.state().total
}

#[test]
fn test_turns_acting_mask_alternates_with_the_player_to_move() {
    let mut p = race();
    assert_eq!(acting(&p), [true, false]);
    step(&mut p, [0, 0]);
    assert_eq!(acting(&p), [false, true]);
    step(&mut p, [0, 0]);
    assert_eq!(acting(&p), [true, false]);
}

#[test]
fn test_turns_non_acting_action_is_ignored() {
    let (mut a, mut b) = (race(), race());
    // Player 0 acts; player 1's action differs between the pools.
    step(&mut a, [1, 0]);
    step(&mut b, [1, 1]);
    assert_eq!(total(&a), total(&b));
    assert_eq!(total(&a), 2);
    assert_eq!(a.outputs().rewards(), b.outputs().rewards());
}

#[cfg(not(feature = "debug-checks"))]
#[test]
fn test_turns_mask_is_checked_only_for_the_acting_agent() {
    let mut p = pool(RaceConfig {
        target: 2,
        both_act: false,
    });
    // Player 0 adds 1 (total 1): adding 2 is now illegal for everyone.
    step(&mut p, [0, 1]);
    // Player 1 acts with the illegal action; player 0 (not acting) too.
    step(&mut p, [1, 1]);
    let h = p.header(0).unwrap();
    assert_eq!(h.agents[0].illegal_actions, 0, "not acting: not counted");
    assert_eq!(
        h.agents[1].illegal_actions, 1,
        "acting: replaced by the no-op"
    );
}

/// With `debug-checks` an illegal action is an error, reported for the
/// acting agent only.
#[cfg(feature = "debug-checks")]
#[test]
fn test_turns_debug_checks_report_only_the_acting_agent() {
    use tessera_rl_engine::EngineError;

    let mut p = pool(RaceConfig {
        target: 2,
        both_act: false,
    });
    step(&mut p, [0, 1]);
    let err = p
        .step(ActionInput {
            discrete: &[1, 1],
            continuous: &[],
        })
        .map(|_| ());
    assert!(
        matches!(err, Err(EngineError::IllegalAction { agent: 1, .. })),
        "{err:?}"
    );
}

#[test]
fn test_turns_terminal_reward_reaches_both_agents_in_the_final_step() {
    let mut p = race();
    // 2 + 2 + 2 by turns reaches 6; player 1 then adds 1 and wins at 7.
    for _ in 0..3 {
        step(&mut p, [1, 1]);
    }
    step(&mut p, [0, 0]);
    let out = p.outputs();
    assert_eq!(out.rewards(), [-1.0, 1.0]);
    assert!(out.agent_terminated().iter().all(|f| f.get()));
    assert_eq!(out.episode_terminated(), [Bool8::TRUE]);
}

#[test]
fn test_turns_snapshot_and_restore_continue_the_turn_order() {
    let mut p = race();
    step(&mut p, [0, 0]);
    let rec = p.snapshot(0).unwrap();
    let mut seen = Vec::new();
    for act in [[1, 1], [0, 0], [1, 1]] {
        step(&mut p, act);
        seen.push((acting(&p), p.outputs().rewards().to_vec()));
    }
    p.restore(0, &rec).unwrap();
    assert_eq!(acting(&p), [false, true]);
    for (k, act) in [[1, 1], [0, 0], [1, 1]].into_iter().enumerate() {
        step(&mut p, act);
        assert_eq!((acting(&p), p.outputs().rewards().to_vec()), seen[k]);
    }
}

#[cfg(debug_assertions)]
#[test]
fn test_turns_sequential_violation_poisons_in_debug_builds() {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(3))
    };
    let mut p = EnvPool::<RaceEnv, Masked>::with_masks(
        RaceConfig {
            target: 7,
            both_act: true,
        },
        options,
    )
    .unwrap();
    assert!(p.reset_all(Reseed::Keep).is_err());
    assert!(p.is_poisoned());
}

#[test]
fn test_turns_import_rejects_acting_flags_outside_the_turn_order() {
    use tessera_rl_engine::{DecodedRecord, EngineError};

    let mut p = race();
    step(&mut p, [0, 0]);
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    // Player 1 is to move; a record that makes player 0 act too is invalid.
    let mut header = *good.header();
    header.agents[0].acting = Bool8::TRUE;
    let bad = DecodedRecord::from_parts(header, *good.state()).unwrap();
    assert!(matches!(
        p.import(&[0], &[bad]),
        Err(EngineError::InvalidRecord { index: 0, .. })
    ));
    // The untouched record imports.
    assert!(p.import(&[0], &[good]).is_ok());
    assert_eq!(acting(&p), [false, true]);
}
