// SPDX-License-Identifier: MIT OR Apache-2.0

//! Heuristic slots of a policy map (ADR-0079 items 2, 3 and 10, stage A):
//! the engine computes their actions on the worker, equal to `act` at any
//! worker count and schedule, from a stream apart from the environment's;
//! submitted rows are never read; an invalid computed action poisons; the
//! actions are pure across restore and fork; the map digest is part of the
//! pool context; Pipelined and Async refuse heuristic slots until stage B.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

mod common;

use core::num::{NonZeroU32, NonZeroUsize};

use common::{CountConfig, Counter, Counts};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic, PolicyGroup, RecordLayout};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, EngineError, EnvPool, Masked, PipelinedOptions, PolicyMap,
    PoolContext, PoolOptions, Reseed, Schedule, SyncOptions,
};
use tessera_rl_num::identity::{RngIdentity, Seed};
use tessera_rl_num::rng::{RngAuthority, StepRng};

const N: u32 = 16;
/// `N` as a count.
const NU: usize = 16;
const A: usize = 2;
/// `kill_at` of a config whose heuristic writes an out-of-range action.
const OUT_OF_RANGE: i32 = -1_000;
/// `kill_at` of a config whose heuristic writes action 2, illegal for
/// agent 0 under `Counter`'s mask.
const ILLEGAL: i32 = -2_000;

/// A test heuristic that draws from its RNG, so the heuristic stream is
/// observable: `below(2)` plus the agent index (agent 0 never plays the
/// illegal 2). The two switch values of `kill_at` make it write an invalid
/// action instead.
impl Heuristic for Counter {
    fn act(
        cfg: &CountConfig,
        _: &Counts,
        agent: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let draw = rng.below_u32(NonZeroU32::new(2).unwrap_or(NonZeroU32::MIN));
        let action = match cfg.kill_at {
            OUT_OF_RANGE => 7,
            ILLEGAL => 2,
            _ => i32::try_from(draw).unwrap_or(0) + i32::try_from(agent.index()).unwrap_or(0),
        };
        out.set_discrete(0, action);
    }
}

fn cfg() -> CountConfig {
    CountConfig {
        length: 7,
        ..CountConfig::default()
    }
}

fn options(schedule: Schedule) -> PoolOptions {
    PoolOptions {
        chunk_size: NonZeroU32::new(4).unwrap(),
        schedule,
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(31))
    }
}

fn plain(cfg: CountConfig) -> EnvPool<Counter, Masked> {
    EnvPool::with_masks(cfg, options(Schedule::Static)).unwrap()
}

fn with_workers(p: EnvPool<Counter, Masked>, workers: Option<usize>) -> EnvPool<Counter, Masked> {
    match workers {
        Some(w) => p
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap(),
        None => p,
    }
}

/// A pool whose slot 1 is a heuristic slot.
fn heuristic(
    cfg: CountConfig,
    workers: Option<usize>,
    schedule: Schedule,
) -> EnvPool<Counter, Masked> {
    let p = EnvPool::with_masks(cfg, options(schedule))
        .unwrap()
        .with_heuristic_slots(&[1])
        .unwrap();
    with_workers(p, workers)
}

/// What the engine must compute for `slot` of instance `i` in the next
/// step: `act` with the heuristic RNG at the next step index, or zeros
/// when the slot does not act.
fn expected(p: &EnvPool<Counter, Masked>, i: u32, slot: usize) -> i32 {
    let header = p.header(i).unwrap();
    let engine = header.engine();
    let row = &header.agents()[slot];
    if engine.pending_reset.get() || !(row.alive.get() && row.acting.get()) {
        return 0;
    }
    let id = engine.identity;
    let next = RngIdentity::from_parts(
        id.seed(),
        id.branch_salt(),
        id.logical_id(),
        id.episode_index(),
        engine.episode_steps + 1,
    );
    let slot16 = u16::try_from(slot).unwrap();
    let mut rng = StepRng::heuristic(&RngAuthority::engine_only(), &next, slot16);
    let mut action = [0];
    let agent = AgentSlot::new(slot16, A).unwrap();
    Counter::act(
        p.config(),
        &p.state(i).unwrap(),
        agent,
        &mut rng,
        &mut ActionWriter::new(&mut action, &mut []),
    );
    action[0]
}

fn expected_all(p: &EnvPool<Counter, Masked>, slot: usize) -> Vec<i32> {
    (0..N).map(|i| expected(p, i, slot)).collect()
}

/// Learner actions in slot 0, `fill` in slot 1.
fn actions(t: usize, fill: i32) -> Vec<i32> {
    (0..NU * A)
        .map(|k| {
            if k % A == 0 {
                i32::from((k + t).is_multiple_of(2))
            } else {
                fill
            }
        })
        .collect()
}

fn step(p: &mut EnvPool<Counter, Masked>, acts: &[i32]) -> Result<(), EngineError> {
    p.step(ActionInput {
        discrete: acts,
        continuous: &[],
    })
    .map(|_| ())
}

fn hashes(p: &EnvPool<Counter, Masked>) -> Vec<u64> {
    (0..N).map(|i| p.state_hash(i).unwrap().get()).collect()
}

/// Runs 20 steps with garbage in the heuristic rows; returns each step's
/// heuristic actions of slot 1 and the final state hashes.
fn run(workers: Option<usize>, schedule: Schedule) -> (Vec<Vec<i32>>, Vec<u64>) {
    let mut p = heuristic(cfg(), workers, schedule);
    p.reset_all(Reseed::Keep).unwrap();
    let mut seen = Vec::new();
    for t in 0..20 {
        let want = expected_all(&p, 1);
        step(&mut p, &actions(t, i32::MIN)).unwrap();
        let got: Vec<i32> = p
            .heuristic_actions()
            .unwrap()
            .chunks(A)
            .map(|row| row[1])
            .collect();
        assert_eq!(got, want, "step {t}, workers {workers:?}, {schedule:?}");
        assert!(p.heuristic_actions().unwrap().chunks(A).all(|r| r[0] == 0));
        seen.push(got);
    }
    (seen, hashes(&p))
}

#[test]
fn test_heuristic_slots_actions_equal_act_at_every_worker_count_and_schedule() {
    let inline = run(None, Schedule::Static);
    for workers in [1, 2, 7] {
        assert_eq!(run(Some(workers), Schedule::Static), inline);
        assert_eq!(run(Some(workers), Schedule::Dynamic), inline);
    }
}

#[test]
fn test_heuristic_slots_environment_stream_unchanged_by_a_heuristic_slot() {
    // A pool without a map, fed the heuristic's actions, and a pool with a
    // heuristic slot fed garbage, step identically: the heuristic draws
    // nothing from the environment's own stream.
    let mut a = plain(cfg());
    let mut b = heuristic(cfg(), None, Schedule::Static);
    a.reset_all(Reseed::Keep).unwrap();
    b.reset_all(Reseed::Keep).unwrap();
    for t in 0..20 {
        let mut fed = actions(t, 0);
        for (i, h) in expected_all(&a, 1).into_iter().enumerate() {
            fed[i * A + 1] = h;
        }
        step(&mut a, &fed).unwrap();
        step(&mut b, &actions(t, -77)).unwrap();
        assert_eq!(hashes(&a), hashes(&b), "step {t}");
        assert_eq!(
            a.outputs().unwrap().rewards(),
            b.outputs().unwrap().rewards()
        );
    }
}

#[test]
fn test_heuristic_slots_garbage_rows_change_nothing_and_raise_nothing() {
    let mut zeros = heuristic(cfg(), Some(2), Schedule::Static);
    let mut garbage = heuristic(cfg(), Some(2), Schedule::Static);
    zeros.reset_all(Reseed::Keep).unwrap();
    garbage.reset_all(Reseed::Keep).unwrap();
    for t in 0..12 {
        step(&mut zeros, &actions(t, 0)).unwrap();
        let fill = [i32::MIN, -1, 3, 99, i32::MAX][t % 5];
        step(&mut garbage, &actions(t, fill)).unwrap();
        assert_eq!(hashes(&zeros), hashes(&garbage));
        assert_eq!(zeros.heuristic_actions(), garbage.heuristic_actions());
    }
    // `check_actions` skips the rows too.
    garbage
        .check_actions(ActionInput {
            discrete: &actions(0, i32::MIN),
            continuous: &[],
        })
        .unwrap();
}

#[test]
fn test_heuristic_slots_out_of_range_action_poisons_in_every_mode() {
    for workers in [None, Some(3)] {
        let bad = CountConfig {
            kill_at: OUT_OF_RANGE,
            ..cfg()
        };
        let mut p = heuristic(bad, workers, Schedule::Static);
        p.reset_all(Reseed::Keep).unwrap();
        assert_eq!(
            step(&mut p, &actions(0, 0)),
            Err(EngineError::HeuristicAction {
                instance: 0,
                slot: 1
            })
        );
        assert!(p.is_poisoned());
        // A full reset recovers, as after any poisoning.
        p.reset_all(Reseed::Keep).unwrap();
        assert!(!p.is_poisoned());
    }
}

#[test]
fn test_heuristic_slots_action_illegal_under_the_mask_poisons() {
    let bad = CountConfig {
        kill_at: ILLEGAL,
        ..cfg()
    };
    let mut p = EnvPool::<Counter, Masked>::with_masks(bad, options(Schedule::Static))
        .unwrap()
        .with_heuristic_slots(&[0])
        .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(
        step(&mut p, &actions(0, 1)),
        Err(EngineError::HeuristicAction {
            instance: 0,
            slot: 0
        })
    );
    assert!(p.is_poisoned());
}

#[test]
fn test_heuristic_slots_pure_across_restore_and_fork() {
    let mut p = heuristic(cfg(), Some(2), Schedule::Static);
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p, &actions(0, 0)).unwrap();
    let saved = p.snapshot(0).unwrap();
    step(&mut p, &actions(1, 0)).unwrap();
    let first = p.heuristic_actions().unwrap()[..A].to_vec();
    p.restore(0, &saved).unwrap();
    step(&mut p, &actions(1, 0)).unwrap();
    assert_eq!(&p.heuristic_actions().unwrap()[..A], first.as_slice());
    // Forks of one source act alike in the same step.
    p.fork(0, &[5, 9]).unwrap();
    step(&mut p, &actions(2, 0)).unwrap();
    let rows = p.heuristic_actions().unwrap();
    assert_eq!(rows[5 * A + 1], rows[1]);
    assert_eq!(rows[9 * A + 1], rows[1]);
}

#[test]
fn test_heuristic_slots_context_digest_mismatch_refused_and_mutates_nothing() {
    let mut p = heuristic(cfg(), None, Schedule::Static);
    p.reset_all(Reseed::Keep).unwrap();
    let ctx = p.pool_context().unwrap();
    assert_eq!(ctx.policy_map, p.policy_map().digest());
    assert_ne!(ctx.policy_map, 0);
    let mut q = plain(cfg());
    q.reset_all(Reseed::Keep).unwrap();
    let before = q.pool_context().unwrap();
    let mut other = ctx.clone();
    other.branch_counter = 5;
    assert_eq!(
        q.restore_pool_context(other, None),
        Err(EngineError::PolicyMapMismatch {
            expected: 0,
            found: ctx.policy_map
        })
    );
    assert_eq!(q.pool_context().unwrap(), before);
    assert!(p.restore_pool_context(PoolContext::new(3), None).is_err());
    assert_eq!(p.pool_context().unwrap(), ctx);
    p.restore_pool_context(ctx.clone(), None).unwrap();
}

#[test]
fn test_heuristic_actions_exist_only_with_a_heuristic_slot() {
    assert_eq!(plain(cfg()).heuristic_actions(), None);
    let external = plain(cfg())
        .with_policy_map(PolicyMap::new(A).assign(1, PolicyGroup::External).unwrap())
        .unwrap();
    assert_eq!(external.heuristic_actions(), None);
    assert_eq!(
        external.policy_groups(),
        &[PolicyGroup::Learner, PolicyGroup::External]
    );
    let h = heuristic(cfg(), None, Schedule::Static);
    assert_eq!(h.heuristic_actions().map(<[i32]>::len), Some(NU * A));
    assert_eq!(
        h.policy_groups(),
        &[PolicyGroup::Learner, PolicyGroup::Heuristic]
    );
}

#[test]
fn test_policy_map_setters_refuse_bad_maps_and_late_calls() {
    let refused = |r: Result<EnvPool<Counter, Masked>, EngineError>| {
        assert!(matches!(r, Err(EngineError::PolicyMap { .. })));
    };
    refused(plain(cfg()).with_policy_map(PolicyMap::new(3)));
    refused(
        plain(cfg()).with_policy_map(PolicyMap::new(A).assign(0, PolicyGroup::Heuristic).unwrap()),
    );
    refused(plain(cfg()).with_heuristic_slots(&[2]));
    refused(plain(cfg()).with_heuristic_slots(&[1, 1]));
    refused(heuristic(cfg(), None, Schedule::Static).with_heuristic_slots(&[0]));
    refused(
        heuristic(cfg(), None, Schedule::Static)
            .with_policy_map(PolicyMap::new(A).assign(0, PolicyGroup::External).unwrap()),
    );
    let mut p = plain(cfg());
    p.reset_all(Reseed::Keep).unwrap();
    refused(p.with_heuristic_slots(&[1]));
}

#[test]
fn test_split_modes_refuse_heuristic_slots_and_accept_learner_maps() {
    let w = NonZeroUsize::new(2).unwrap();
    let e = heuristic(cfg(), None, Schedule::Static)
        .pipelined(PipelinedOptions::new(w))
        .err()
        .unwrap();
    assert_eq!(
        e.error,
        EngineError::HeuristicSlotsUnsupported { mode: "Pipelined" }
    );
    assert!(e.pool.is_some());
    let e = heuristic(cfg(), None, Schedule::Static)
        .async_mode(AsyncOptions::new(w, NonZeroUsize::new(4).unwrap()))
        .err()
        .unwrap();
    assert_eq!(
        e.error,
        EngineError::HeuristicSlotsUnsupported { mode: "Async" }
    );
    assert!(e.pool.is_some());
    let external = || {
        plain(cfg())
            .with_policy_map(PolicyMap::new(A).assign(1, PolicyGroup::External).unwrap())
            .unwrap()
    };
    assert!(external().pipelined(PipelinedOptions::new(w)).is_ok());
    assert!(
        external()
            .async_mode(AsyncOptions::new(w, NonZeroUsize::new(4).unwrap()))
            .is_ok()
    );
}
