// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inline mode, seeding and the validation boundary (SPEC 6, 8.1;
//! ADR-0006, ADR-0007, ADR-0028).

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod common;

use core::num::NonZeroU32;

use common::{CountConfig, Counter};
use tessera_rl_core::{Bool8, StateHash};
use tessera_rl_engine::{ActionInput, AutoReset, EngineError, EnvPool, PoolOptions, Reseed};
use tessera_rl_num::identity::{LogicalId, Seed};

fn pool(n: u32, seed: u64) -> EnvPool<Counter> {
    let options = PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed));
    EnvPool::new(CountConfig::default(), options).unwrap()
}

fn actions(values: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: values,
        continuous: &[],
    }
}

fn hashes(p: &EnvPool<Counter>, n: u32) -> Vec<u64> {
    (0..n)
        .map(|i| StateHash::of(p.state(i).unwrap()).get())
        .collect()
}

/// Steps `steps` times with the same actions and returns the observations
/// and rewards of every step.
fn trajectory(p: &mut EnvPool<Counter>, act: &[i32], steps: usize) -> Vec<(Vec<u8>, Vec<f32>)> {
    (0..steps)
        .map(|_| {
            let out = p.step(actions(act)).unwrap();
            (out.observation(0).unwrap().to_vec(), out.rewards().to_vec())
        })
        .collect()
}

#[test]
fn test_pool_step_before_reset_is_pending_reset() {
    let mut p = pool(2, 1);
    assert_eq!(
        p.step(actions(&[0; 4])).map(|_| ()),
        Err(EngineError::PendingReset { id: 0 })
    );
}

#[test]
fn test_pool_same_seed_same_trajectory() {
    let (mut a, mut b) = (pool(3, 42), pool(3, 42));
    a.reset_all(Reseed::Keep).unwrap();
    b.reset_all(Reseed::Keep).unwrap();
    let act = [1, 2, 0, 1, 2, 2];
    assert_eq!(trajectory(&mut a, &act, 4), trajectory(&mut b, &act, 4));
    assert_eq!(hashes(&a, 3), hashes(&b, 3));
}

#[test]
fn test_pool_seeds_are_pool_seed_plus_index_wrapping() {
    let p = pool(2, u64::MAX);
    let seed = |i| p.header(i).unwrap().engine.identity.seed().get();
    assert_eq!((seed(0), seed(1)), (u64::MAX, 0));
}

#[test]
fn test_pool_duplicate_seeds_identical_until_lineage_differs() {
    let mut p = pool(2, 0);
    p.reset_all(Reseed::List(&[Seed::new(7), Seed::new(7)]))
        .unwrap();
    let out = p.step(actions(&[1, 1, 1, 1])).unwrap();
    let (obs, rewards) = (out.observation(0).unwrap(), out.rewards());
    // Each instance has 2 agents of 8 observation bytes.
    assert_eq!(obs[..16], obs[16..32]);
    assert_eq!(rewards[..2], rewards[2..4]);
    assert_eq!(hashes(&p, 2)[0], hashes(&p, 2)[1]);

    p.set_lineage(1, LogicalId::new(1)).unwrap();
    p.reset_all(Reseed::List(&[Seed::new(7), Seed::new(7)]))
        .unwrap();
    let mut differ = false;
    for _ in 0..4 {
        p.step(actions(&[1, 1, 1, 1])).unwrap();
        differ |= hashes(&p, 2)[0] != hashes(&p, 2)[1];
    }
    assert!(differ);
}

#[test]
fn test_pool_reseeding_one_instance_leaves_others_unchanged() {
    let (mut a, mut b) = (pool(2, 5), pool(2, 5));
    a.reset_all(Reseed::Keep).unwrap();
    b.reset_all(Reseed::Keep).unwrap();
    let act = [1, 0, 2, 1];
    a.step(actions(&act)).unwrap();
    b.step(actions(&act)).unwrap();
    // Only b's instance 1 is reseeded and reset; instance 0 must match a.
    b.reset(&[1], Reseed::Base(Seed::new(99))).unwrap();
    a.step(actions(&act)).unwrap();
    let a0 = hashes(&a, 2)[0];
    b.step(actions(&act)).unwrap();
    assert_eq!(hashes(&b, 2)[0], a0);
    assert_eq!(b.header(1).unwrap().engine.identity.seed().get(), 99);
    assert_eq!(b.header(1).unwrap().engine.identity.episode_index(), 0);
}

#[test]
fn test_pool_episode_index_advances_and_reseed_restarts_it() {
    let mut p = pool(1, 3);
    p.reset_all(Reseed::Keep).unwrap();
    let episode = |p: &EnvPool<Counter>| p.header(0).unwrap().engine.identity.episode_index();
    assert_eq!(episode(&p), 0);
    p.reset_all(Reseed::Keep).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(episode(&p), 2);
    p.reset_all(Reseed::Base(Seed::new(3))).unwrap();
    assert_eq!(episode(&p), 0);
}

#[test]
fn test_pool_changing_first_seed_of_list_changes_only_first_instance() {
    let (mut a, mut b) = (pool(2, 0), pool(2, 0));
    a.reset_all(Reseed::List(&[Seed::new(1), Seed::new(2)]))
        .unwrap();
    b.reset_all(Reseed::List(&[Seed::new(9), Seed::new(2)]))
        .unwrap();
    assert_eq!(hashes(&a, 2)[1], hashes(&b, 2)[1]);
}

#[test]
fn test_validation_rejects_without_mutating() {
    let mut p = pool(2, 1);
    p.reset_all(Reseed::Keep).unwrap();
    let before = hashes(&p, 2);
    let header_before = *p.header(1).unwrap();
    assert_eq!(
        p.step(actions(&[0, 0, 0, 3])).map(|_| ()),
        Err(EngineError::DiscreteOutOfRange {
            instance: 1,
            agent: 1,
            head: 0,
            value: 3,
            n: 3
        })
    );
    assert!(matches!(
        p.step(actions(&[0, 0, 0])).map(|_| ()),
        Err(EngineError::ActionLength {
            buffer: "discrete",
            expected: 4,
            actual: 3
        })
    ));
    assert_eq!(hashes(&p, 2), before);
    assert_eq!(*p.header(1).unwrap(), header_before);
}

#[test]
fn test_reset_rejects_duplicate_and_unknown_ids_and_bad_seed_lists() {
    let mut p = pool(3, 1);
    assert_eq!(
        p.reset(&[0, 2, 0], Reseed::Keep).map(|_| ()),
        Err(EngineError::DuplicateInstance { id: 0 })
    );
    assert_eq!(
        p.reset(&[5], Reseed::Keep).map(|_| ()),
        Err(EngineError::InstanceOutOfRange { id: 5, num_envs: 3 })
    );
    assert_eq!(
        p.reset(&[0, 1], Reseed::List(&[Seed::new(1)])).map(|_| ()),
        Err(EngineError::SeedCount {
            seeds: 1,
            instances: 2
        })
    );
    // The scratch marks were cleared: a valid call works.
    assert!(p.reset(&[0, 2], Reseed::Keep).is_ok());
}

#[test]
fn test_episode_terminates_and_sets_flags_and_stats() {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(2))
    };
    let mut p = EnvPool::<Counter>::new(CountConfig::default(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    for _ in 0..4 {
        let out = p.step(actions(&[1, 1])).unwrap();
        assert_eq!(out.episode_terminated(), [Bool8::FALSE]);
    }
    let out = p.step(actions(&[1, 1])).unwrap();
    assert_eq!(out.episode_terminated(), [Bool8::TRUE]);
    assert_eq!(out.episode_length(), [5.0]);
    assert!(out.episode_return()[0] >= 10.0);
    assert_eq!(
        p.step(actions(&[1, 1])).map(|_| ()),
        Err(EngineError::PendingReset { id: 0 })
    );
}

#[test]
fn test_step_limit_truncates() {
    let options = PoolOptions {
        max_episode_steps: NonZeroU32::new(2),
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(0))
    };
    let mut p = EnvPool::<Counter>::new(CountConfig::default(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p.step(actions(&[0, 0])).unwrap();
    let out = p.step(actions(&[0, 0])).unwrap();
    assert_eq!(
        (out.episode_truncated(), out.episode_terminated()),
        ([Bool8::TRUE].as_slice(), [Bool8::FALSE].as_slice())
    );
}

#[test]
fn test_agent_death_clears_alive_and_skips_its_validation() {
    let cfg = CountConfig {
        kill_at: 2,
        ..CountConfig::default()
    };
    let mut p =
        EnvPool::<Counter>::new(cfg, PoolOptions::new(NonZeroU32::MIN, Seed::new(4))).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let out = p.step(actions(&[0, 2])).unwrap();
    assert_eq!(out.alive(), [Bool8::TRUE, Bool8::FALSE]);
    assert_eq!(out.agent_terminated(), [Bool8::FALSE, Bool8::TRUE]);
    assert_eq!(out.observation(0).unwrap()[8..], [0; 8]);
    // The dead agent's action is not validated.
    assert!(p.step(actions(&[0, 99])).is_ok());
}

#[test]
fn test_invalid_options_and_config_are_rejected() {
    let bad = PoolOptions {
        chunk_size: NonZeroU32::new(2048).unwrap(),
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(0))
    };
    assert!(matches!(
        EnvPool::<Counter>::new(CountConfig::default(), bad),
        Err(EngineError::Options { .. })
    ));
    let cfg = CountConfig {
        length: 0,
        ..CountConfig::default()
    };
    assert!(matches!(
        EnvPool::<Counter>::new(cfg, PoolOptions::new(NonZeroU32::MIN, Seed::new(0))),
        Err(EngineError::Config(_))
    ));
}

#[test]
fn test_episode_end_flushes_final_flags_to_alive_agents_only() {
    // Episodes last 5 steps (the default length).
    let cfg = CountConfig {
        kill_at: 0,
        ..CountConfig::default()
    };
    let options = PoolOptions::new(NonZeroU32::MIN, Seed::new(4));
    let mut p = EnvPool::<Counter>::new(cfg, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    // Agent 1 dies on the first step and reports terminated once.
    let out = p.step(actions(&[0, 1])).unwrap();
    assert_eq!(out.agent_terminated(), [Bool8::FALSE, Bool8::TRUE]);
    for _ in 0..3 {
        let out = p.step(actions(&[0, 0])).unwrap();
        assert_eq!(out.agent_terminated(), [Bool8::FALSE, Bool8::FALSE]);
    }
    // The episode ends on the fifth step: only the living agent 0 gets
    // the final flag; dead agent 1 does not report again.
    let out = p.step(actions(&[0, 0])).unwrap();
    assert_eq!(out.episode_terminated(), [Bool8::TRUE]);
    assert_eq!(out.agent_terminated(), [Bool8::TRUE, Bool8::FALSE]);
}

#[test]
fn test_step_limit_truncates_alive_agents() {
    let options = PoolOptions {
        max_episode_steps: NonZeroU32::new(1),
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(0))
    };
    let mut p = EnvPool::<Counter>::new(CountConfig::default(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let out = p.step(actions(&[0, 0])).unwrap();
    assert_eq!(out.agent_truncated(), [Bool8::TRUE, Bool8::TRUE]);
    assert_eq!(out.agent_terminated(), [Bool8::FALSE, Bool8::FALSE]);
}

#[test]
fn test_set_lineage_applies_immediately_in_a_running_episode() {
    let cfg = CountConfig {
        length: 100,
        ..CountConfig::default()
    };
    let mut p = EnvPool::<Counter>::new(
        cfg,
        PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(0)),
    )
    .unwrap();
    p.reset_all(Reseed::List(&[Seed::new(3), Seed::new(3)]))
        .unwrap();
    p.step(actions(&[1, 1, 1, 1])).unwrap();
    assert_eq!(hashes(&p, 2)[0], hashes(&p, 2)[1]);
    p.set_lineage(1, LogicalId::new(5)).unwrap();
    let mut differ = false;
    for _ in 0..20 {
        p.step(actions(&[1, 1, 1, 1])).unwrap();
        differ |= hashes(&p, 2)[0] != hashes(&p, 2)[1];
    }
    assert!(differ);
}

fn spawn_pool(cfg: CountConfig) -> EnvPool<Counter> {
    EnvPool::<Counter>::new(cfg, PoolOptions::new(NonZeroU32::MIN, Seed::new(4))).unwrap()
}

/// SPEC 9.1: a spawn on a dying or alive slot is never applied; without
/// `debug-checks` it is ignored and counted.
#[cfg(not(feature = "debug-checks"))]
#[test]
fn test_invalid_spawns_are_ignored_and_counted() {
    let cfg = CountConfig {
        kill_at: 2,
        spawn_when_dying: true,
        spawn_alive_at: 2,
        length: 3,
        ..CountConfig::default()
    };
    let mut p = spawn_pool(cfg);
    p.reset_all(Reseed::Keep).unwrap();
    // Step 1: agent 1 dies and asks to respawn in the same step.
    let out = p.step(actions(&[0, 2])).unwrap();
    assert_eq!(out.alive(), [Bool8::TRUE, Bool8::FALSE]);
    assert_eq!(out.generations(), [0, 0]);
    let header = p.header(0).unwrap();
    assert_eq!(header.agents[1].invalid_spawns, 1);
    // Step 2: agent 0 is alive and asks to spawn.
    let out = p.step(actions(&[0, 0])).unwrap();
    assert_eq!(out.generations(), [0, 0]);
    assert_eq!(p.header(0).unwrap().agents[0].invalid_spawns, 1);
    // Step 3 ends the episode: the statistic sums both.
    let out = p.step(actions(&[0, 0])).unwrap();
    assert_eq!(out.episode_invalid_spawns(), [2.0]);
    // Counters restart with the episode.
    p.reset_all(Reseed::Keep).unwrap();
    assert!(
        p.header(0)
            .unwrap()
            .agents
            .iter()
            .all(|r| r.invalid_spawns == 0)
    );
}

/// With `debug-checks` an invalid spawn is an environment bug: the call
/// fails with a typed error and the pool is poisoned.
#[cfg(feature = "debug-checks")]
#[test]
fn test_invalid_spawn_fails_and_poisons_under_debug_checks() {
    for cfg in [
        CountConfig {
            kill_at: 2,
            spawn_when_dying: true,
            ..CountConfig::default()
        },
        CountConfig {
            spawn_alive_at: 1,
            ..CountConfig::default()
        },
    ] {
        let dying = cfg.spawn_when_dying;
        let mut p = spawn_pool(cfg);
        p.reset_all(Reseed::Keep).unwrap();
        let agent = u32::from(dying);
        assert_eq!(
            p.step(actions(&[0, 2])).map(|_| ()),
            Err(EngineError::InvalidSpawn { instance: 0, agent })
        );
        assert!(p.is_poisoned());
    }
}
