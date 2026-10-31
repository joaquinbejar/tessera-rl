// SPDX-License-Identifier: MIT OR Apache-2.0

//! Auto-reset modes, episode and agent flags, poisoning and recovery
//! (ADR-0028).

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod common;

use core::num::NonZeroU32;

use common::{CountConfig, Counter};
use tessera_rl_core::{Bool8, StateHash};
use tessera_rl_engine::{ActionInput, AutoReset, EngineError, EnvPool, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

const ACT: [i32; 2] = [1, 1];

fn pool(mode: AutoReset, cfg: CountConfig) -> EnvPool<Counter> {
    let options = PoolOptions {
        auto_reset: mode,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(11))
    };
    let mut p = EnvPool::new(cfg, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn short() -> CountConfig {
    CountConfig {
        length: 2,
        ..CountConfig::default()
    }
}

fn step(p: &mut EnvPool<Counter>) -> Result<&tessera_rl_engine::Outputs, EngineError> {
    p.step(ActionInput {
        discrete: &ACT,
        continuous: &[],
    })
}

fn episode(p: &EnvPool<Counter>) -> u32 {
    p.header(0).unwrap().engine.identity.episode_index()
}

#[test]
fn test_next_step_returns_final_then_resets_on_the_following_step() {
    let mut p = pool(AutoReset::NextStep, short());
    step(&mut p).unwrap();
    let out = step(&mut p).unwrap();
    assert_eq!(out.episode_terminated(), [Bool8::TRUE]);
    let final_obs = out.observation(0).unwrap().to_vec();
    assert_ne!(final_obs, vec![0; 16]);
    assert_eq!(episode(&p), 0);

    // The next step resets instead of stepping: flags cleared, no reward,
    // a new episode with a fresh state.
    let out = step(&mut p).unwrap();
    assert_eq!(out.episode_terminated(), [Bool8::FALSE]);
    assert_eq!(out.rewards(), [0.0, 0.0]);
    assert_eq!(episode(&p), 1);
    assert_eq!(p.state(0).unwrap().steps, 0);
    // Actions of the resetting instance were ignored, even invalid ones.
    let mut q = pool(AutoReset::NextStep, short());
    step(&mut q).unwrap();
    step(&mut q).unwrap();
    assert!(
        q.step(ActionInput {
            discrete: &[9, 9],
            continuous: &[]
        })
        .is_ok()
    );
}

#[test]
fn test_same_step_resets_in_the_same_pass_with_terminal_observation() {
    let (mut next, mut same) = (
        pool(AutoReset::NextStep, short()),
        pool(AutoReset::SameStep, short()),
    );
    step(&mut next).unwrap();
    step(&mut same).unwrap();
    let final_next = step(&mut next).unwrap().observation(0).unwrap().to_vec();
    let out = step(&mut same).unwrap();
    assert_eq!(out.episode_terminated(), [Bool8::TRUE]);
    assert_eq!(out.terminal_observation(0).unwrap(), final_next.as_slice());
    let initial_same = out.observation(0).unwrap().to_vec();
    assert_eq!(episode(&same), 1);
    assert_eq!(same.state(0).unwrap().steps, 0);
    // The initial observation equals the one after a NextStep reset.
    let initial_next = step(&mut next).unwrap().observation(0).unwrap().to_vec();
    assert_eq!(initial_same, initial_next);
    assert_eq!(
        StateHash::of(next.state(0).unwrap()),
        StateHash::of(same.state(0).unwrap())
    );
    // The following step is a normal step.
    assert_eq!(
        step(&mut same).unwrap().episode_terminated(),
        [Bool8::FALSE]
    );
    assert_eq!(same.state(0).unwrap().steps, 1);
}

#[test]
fn test_disabled_freezes_until_reset() {
    let mut p = pool(AutoReset::Disabled, short());
    step(&mut p).unwrap();
    step(&mut p).unwrap();
    assert_eq!(
        step(&mut p).map(|_| ()),
        Err(EngineError::PendingReset { id: 0 })
    );
    p.reset(&[0], Reseed::Keep).unwrap();
    assert!(step(&mut p).is_ok());
}

#[test]
fn test_terminal_buffer_exists_only_in_same_step() {
    assert!(
        pool(AutoReset::NextStep, short())
            .outputs()
            .terminal_observation(0)
            .is_none()
    );
    assert!(
        pool(AutoReset::SameStep, short())
            .outputs()
            .terminal_observation(0)
            .is_some()
    );
}

#[test]
fn test_agent_death_does_not_end_the_episode() {
    let cfg = CountConfig {
        kill_at: 0,
        length: 10,
        ..CountConfig::default()
    };
    let mut p = pool(AutoReset::NextStep, cfg);
    let out = step(&mut p).unwrap();
    assert_eq!(out.agent_terminated(), [Bool8::FALSE, Bool8::TRUE]);
    assert_eq!(out.episode_terminated(), [Bool8::FALSE]);
    // terminated is reported once: the next step clears it.
    let out = step(&mut p).unwrap();
    assert_eq!(out.agent_terminated(), [Bool8::FALSE, Bool8::FALSE]);
    assert_eq!(out.alive(), [Bool8::TRUE, Bool8::FALSE]);
}

#[test]
fn test_panic_in_step_poisons_and_full_reset_recovers() {
    let cfg = CountConfig {
        panic_at: 1,
        length: 10,
        ..CountConfig::default()
    };
    let mut p = pool(AutoReset::NextStep, cfg);
    assert_eq!(step(&mut p).map(|_| ()), Err(EngineError::Panicked));
    assert!(p.is_poisoned());
    assert_eq!(step(&mut p).map(|_| ()), Err(EngineError::Poisoned));
    assert_eq!(
        p.reset(&[0], Reseed::Keep).map(|_| ()),
        Err(EngineError::Poisoned)
    );
    p.reset_all(Reseed::Keep).unwrap();
    assert!(!p.is_poisoned());
}

#[test]
fn test_panic_during_recovery_stays_poisoned() {
    let cfg = CountConfig {
        panic_on_reset: true,
        ..CountConfig::default()
    };
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(1));
    let mut p = EnvPool::<Counter>::new(cfg, options).unwrap();
    assert_eq!(
        p.reset_all(Reseed::Keep).map(|_| ()),
        Err(EngineError::Panicked)
    );
    assert!(p.is_poisoned());
    assert_eq!(
        p.reset_all(Reseed::Keep).map(|_| ()),
        Err(EngineError::Panicked)
    );
    assert!(p.is_poisoned());
}

#[test]
fn test_fp_environment_restored_after_panic() {
    let cfg = CountConfig {
        panic_at: 1,
        ..CountConfig::default()
    };
    let mut p = pool(AutoReset::NextStep, cfg);
    let _ = step(&mut p);
    assert!(tessera_rl_num::fp::is_canonical());
}
