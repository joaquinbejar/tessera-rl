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

use tessera_rl_core::Bool8;
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, IllegalActionPolicy, Masked, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;

#[path = "common/race.rs"]
mod race;

use race::{RaceConfig, RaceEnv};

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
    p.outputs()
        .unwrap()
        .acting()
        .iter()
        .map(|f| f.get())
        .collect()
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
    assert_eq!(total(&a), 2);
    // The whole state and the engine header are identical.
    assert_eq!(
        a.snapshot(0).unwrap().state(),
        b.snapshot(0).unwrap().state()
    );
    assert_eq!(a.header(0), b.header(0));
    assert_eq!(a.state_hash(0).unwrap(), b.state_hash(0).unwrap());
    assert_eq!(
        a.outputs().unwrap().rewards(),
        b.outputs().unwrap().rewards()
    );
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
    let out = p.outputs().unwrap();
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
        seen.push((acting(&p), p.outputs().unwrap().rewards().to_vec()));
    }
    p.restore(0, &rec).unwrap();
    assert_eq!(acting(&p), [false, true]);
    for (k, act) in [[1, 1], [0, 0], [1, 1]].into_iter().enumerate() {
        step(&mut p, act);
        assert_eq!(
            (acting(&p), p.outputs().unwrap().rewards().to_vec()),
            seen[k]
        );
    }
}

#[test]
fn test_turns_sequential_violation_poisons_in_every_build() {
    use tessera_rl_engine::EngineError;

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
    assert!(matches!(
        p.reset_all(Reseed::Keep),
        Err(EngineError::TurnOrder { instance: 0, .. })
    ));
    assert!(p.is_poisoned());
}

#[test]
fn test_turns_terminal_record_imports_as_pending_reset() {
    use tessera_rl_engine::DecodedRecord;

    let mut p = race();
    for act in [[1, 1], [1, 1], [1, 1], [0, 0]] {
        step(&mut p, act);
    }
    // AutoReset::Disabled: the finished game waits for a reset; no agent
    // acts, which the Sequential check allows for a pending reset.
    assert!(p.header(0).unwrap().engine.pending_reset.get());
    let rec = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut q = race();
    assert!(q.import(&[0], &[rec]).is_ok());
    assert!(q.header(0).unwrap().engine.pending_reset.get());
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

#[test]
fn test_turns_import_validates_the_state_before_asking_who_acts() {
    use tessera_rl_engine::{DecodedRecord, EngineError};

    let mut p = race();
    step(&mut p, [0, 0]);
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    // An invalid state: no player 2. `acting` asserts on it, so asking it
    // before `validate_record` would unwind; the import must fail cleanly.
    let mut state = *good.state();
    state.to_move = 2;
    let bad = DecodedRecord::from_parts(*good.header(), state).unwrap();
    let before = p.state_hash(0).unwrap();
    assert!(matches!(
        p.import(&[0], &[bad]),
        Err(EngineError::Invariant { index: 0, .. })
    ));
    assert!(!p.is_poisoned());
    assert_eq!(p.state_hash(0).unwrap(), before, "atomic: nothing written");
}

#[test]
fn test_turns_import_rejects_a_sequential_record_with_no_acting_agent() {
    use tessera_rl_engine::{DecodedRecord, EngineError};

    let mut p = race();
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    // Player 0 to move, but its slot is dead: every flag matches `acting`
    // (all false), yet nobody can move. The import must reject it.
    let mut header = *good.header();
    header.agents[0].alive = Bool8::FALSE;
    header.agents[0].acting = Bool8::FALSE;
    let bad = DecodedRecord::from_parts(header, *good.state()).unwrap();
    let before = p.state_hash(0).unwrap();
    assert!(matches!(
        p.import(&[0], &[bad]),
        Err(EngineError::InvalidRecord { index: 0, .. })
    ));
    assert!(!p.is_poisoned());
    assert_eq!(p.state_hash(0).unwrap(), before, "atomic: nothing written");
}
