// SPDX-License-Identifier: MIT OR Apache-2.0

//! Heuristic slots in the split-phase modes (ADR-0079 items 3 and 10, #347
//! stage B1): in Pipelined and Async mode the engine computes the
//! heuristic slots' actions on the workers as in Sync, so every instance's
//! trajectory equals Sync's for the same seed and map at 1, 2 and 7
//! workers, whatever order Async ids are sent in; submitted rows for
//! heuristic slots are never read; `heuristic_actions` holds the returned
//! unit's rows; an invalid computed action poisons the pool with its
//! pool-global instance.

#![cfg(not(loom))]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

mod common;

use core::num::{NonZeroU32, NonZeroUsize};

use common::{CountConfig, Counter, Counts};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, EngineError, EnvPool, Masked, PipelinedOptions, PoolOptions,
    Reseed, Schedule,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::StepRng;

const N: u32 = 16;
/// `N` as a count.
const NU: usize = 16;
const A: usize = 2;
const STEPS: usize = 20;
const BATCH: usize = 4;
/// `kill_at` of a config whose heuristic writes an out-of-range action.
const OUT_OF_RANGE: i32 = -1_000;

/// The heuristic of `tests/heuristic_slots.rs`: it draws from its RNG, so
/// the heuristic stream is observable (agent 0 never plays the illegal 2);
/// `kill_at == OUT_OF_RANGE` makes it write an out-of-range action.
impl Heuristic for Counter {
    fn act(
        cfg: &CountConfig,
        _: &Counts,
        agent: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let draw = rng.below_u32(NonZeroU32::new(2).unwrap_or(NonZeroU32::MIN));
        let action = if cfg.kill_at == OUT_OF_RANGE {
            7
        } else {
            i32::try_from(draw).unwrap_or(0) + i32::try_from(agent.index()).unwrap_or(0)
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

/// A pool whose slot 1 is a heuristic slot, with no workers yet.
fn heuristic(cfg: CountConfig, schedule: Schedule) -> EnvPool<Counter, Masked> {
    EnvPool::with_masks(cfg, options(schedule))
        .unwrap()
        .with_heuristic_slots(&[1])
        .unwrap()
}

fn workers(w: usize) -> NonZeroUsize {
    NonZeroUsize::new(w).unwrap()
}

/// Instance `i`'s learner action at its step `t`.
fn learner(i: usize, t: usize) -> i32 {
    i32::from((i + t).is_multiple_of(2))
}

/// The action rows of `ids` at step `t`: the learner in slot 0, `fill` in
/// the heuristic slot 1.
fn rows(ids: impl Iterator<Item = usize>, t: usize, fill: i32) -> Vec<i32> {
    ids.flat_map(|i| [learner(i, t), fill]).collect()
}

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

fn hashes(p: &EnvPool<Counter, Masked>) -> Vec<u64> {
    (0..N).map(|i| p.state_hash(i).unwrap().get()).collect()
}

/// What one mode produced: per step, the heuristic row (slot 0 and slot 1)
/// of every instance; then the final state hashes.
type Trajectory = (Vec<Vec<[i32; A]>>, Vec<u64>);

/// The heuristic rows of every instance of `all`.
fn instance_rows(all: &[i32]) -> Vec<[i32; A]> {
    all.chunks(A).map(|r| [r[0], r[1]]).collect()
}

/// The Sync reference: inline, valid rows (zeros) in the heuristic slot.
fn sync_run() -> Trajectory {
    let mut p = heuristic(cfg(), Schedule::Static);
    p.reset_all(Reseed::Keep).unwrap();
    let mut seen = Vec::new();
    for t in 0..STEPS {
        p.step(input(&rows(0..NU, t, 0))).unwrap();
        seen.push(instance_rows(p.heuristic_actions().unwrap()));
    }
    (seen, hashes(&p))
}

/// Pipelined, `fill` in the heuristic rows: after each `recv` the returned
/// half's rows of `heuristic_actions` are recorded.
fn pipelined_run(w: usize, schedule: Schedule, fill: i32) -> Trajectory {
    let mut pool = heuristic(cfg(), schedule)
        .pipelined(PipelinedOptions::new(workers(w)))
        .unwrap();
    let halves = pool.halves();
    let range = |h: usize| {
        usize::try_from(halves[h].start).unwrap()..usize::try_from(halves[h].end).unwrap()
    };
    let [mut a, mut b] = pool.reset_all(Reseed::Keep).unwrap();
    let mut seen = Vec::new();
    for t in 0..STEPS {
        pool.send(a, input(&rows(range(0), t, fill)))
            .map_err(|e| e.error)
            .unwrap();
        pool.send(b, input(&rows(range(1), t, fill)))
            .map_err(|e| e.error)
            .unwrap();
        let mut step = vec![[0; A]; NU];
        a = pool.recv().unwrap();
        assert_eq!(a.ids(), halves[0]);
        let now = instance_rows(pool.heuristic_actions().unwrap());
        step[range(0)].copy_from_slice(&now[range(0)]);
        b = pool.recv().unwrap();
        assert_eq!(b.ids(), halves[1]);
        let now = instance_rows(pool.heuristic_actions().unwrap());
        step[range(1)].copy_from_slice(&now[range(1)]);
        seen.push(step);
    }
    drop((a, b));
    let p = pool.into_pool().ok().unwrap();
    (seen, hashes(&p))
}

/// Async, `fill` in the heuristic rows: each round sends every id once, in
/// a fresh random order split into random-sized sends, so batches mix ids
/// of different sends; the rows of each returned id are recorded at its
/// `recv`.
fn async_run(w: usize, fill: i32) -> Trajectory {
    let mut pool = heuristic(cfg(), Schedule::Static)
        .async_mode(AsyncOptions::new(workers(w), NonZeroUsize::new(BATCH).unwrap()))
        .unwrap();
    drop(pool.reset_all(Reseed::Keep).unwrap());
    let mut ids: Vec<u32> = (0..N).collect();
    let mut state = 0x9e37_79b9_u64 ^ u64::try_from(w).unwrap();
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        usize::try_from(state >> 33).unwrap()
    };
    let mut seen = Vec::new();
    for t in 0..STEPS {
        for k in (1..NU).rev() {
            ids.swap(k, next() % (k + 1));
        }
        let mut sent = 0;
        while sent < NU {
            let take = (1 + next() % 5).min(NU - sent);
            let subset = &ids[sent..sent + take];
            let acts = rows(subset.iter().map(|&i| usize::try_from(i).unwrap()), t, fill);
            pool.send(subset, input(&acts)).unwrap();
            sent += take;
        }
        let mut step = vec![[i32::MIN; A]; NU];
        let mut back = 0;
        while back < NU {
            let token = pool.recv().unwrap();
            let now = instance_rows(pool.heuristic_actions().unwrap());
            for id in token.ids() {
                let i = usize::try_from(id).unwrap();
                step[i] = now[i];
            }
            back += token.len();
        }
        seen.push(step);
    }
    let p = pool.into_pool().ok().unwrap();
    (seen, hashes(&p))
}

#[test]
fn test_heuristic_split_pipelined_equals_sync_at_every_worker_count() {
    let sync = sync_run();
    // Slot 0 is a learner: its row of `heuristic_actions` stays zero.
    assert!(sync.0.iter().flatten().all(|r| r[0] == 0));
    for w in [1, 2, 7] {
        for schedule in [Schedule::Static, Schedule::Dynamic] {
            assert_eq!(
                pipelined_run(w, schedule, i32::MIN),
                sync,
                "workers {w}, {schedule:?}"
            );
        }
    }
}

#[test]
fn test_heuristic_split_async_shuffled_sends_equal_sync_at_every_worker_count() {
    let sync = sync_run();
    for w in [1, 2, 7] {
        assert_eq!(async_run(w, i32::MIN), sync, "workers {w}");
    }
}

#[test]
fn test_heuristic_split_garbage_in_heuristic_rows_changes_nothing_per_mode() {
    let sync = sync_run();
    for fill in [i32::MIN, -1, 9, i32::MAX] {
        assert_eq!(pipelined_run(2, Schedule::Static, fill), sync, "{fill}");
        assert_eq!(async_run(2, fill), sync, "{fill}");
    }
}

#[test]
fn test_heuristic_split_pipelined_invalid_action_poisons_with_the_pool_instance() {
    let bad = CountConfig {
        kill_at: OUT_OF_RANGE,
        ..cfg()
    };
    let mut pool = heuristic(bad, Schedule::Static)
        .pipelined(PipelinedOptions::new(workers(2)))
        .unwrap();
    let halves = pool.halves();
    let [a, b] = pool.reset_all(Reseed::Keep).unwrap();
    // The second half first: its fault names its first pool instance.
    let len = usize::try_from(halves[1].end - halves[1].start).unwrap();
    pool.send(b, input(&vec![0; len * A]))
        .map_err(|e| e.error)
        .unwrap();
    assert_eq!(
        pool.recv().err(),
        Some(EngineError::HeuristicAction {
            instance: halves[1].start,
            slot: 1
        })
    );
    assert!(pool.is_poisoned());
    drop(a);
    // A full reset recovers, as after any poisoning.
    drop(pool.reset_all(Reseed::Keep).unwrap());
    assert!(!pool.is_poisoned());
}

#[test]
fn test_heuristic_split_async_invalid_action_poisons_with_the_lowest_instance() {
    let bad = CountConfig {
        kill_at: OUT_OF_RANGE,
        ..cfg()
    };
    let mut pool = heuristic(bad, Schedule::Static)
        .async_mode(AsyncOptions::new(workers(2), NonZeroUsize::new(2).unwrap()))
        .unwrap();
    drop(pool.reset_all(Reseed::Keep).unwrap());
    pool.send(&[11, 5], input(&[0; 2 * A])).unwrap();
    assert_eq!(
        pool.recv().err(),
        Some(EngineError::HeuristicAction {
            instance: 5,
            slot: 1
        })
    );
    assert!(pool.is_poisoned());
    // A full reset recovers, and forgets the fault of the batch.
    drop(pool.reset_all(Reseed::Keep).unwrap());
    assert!(!pool.is_poisoned());
}

#[test]
fn test_heuristic_split_heuristic_actions_none_without_a_heuristic_slot() {
    let plain = || {
        EnvPool::<Counter, Masked>::with_masks(cfg(), options(Schedule::Static)).unwrap()
    };
    let pipelined = plain().pipelined(PipelinedOptions::new(workers(1))).unwrap();
    assert!(pipelined.heuristic_actions().is_none());
    assert!(pipelined.heuristic_continuous().is_none());
    let asynced = plain()
        .async_mode(AsyncOptions::new(workers(1), NonZeroUsize::new(BATCH).unwrap()))
        .unwrap();
    assert!(asynced.heuristic_actions().is_none());
    assert!(asynced.heuristic_continuous().is_none());
}

#[test]
fn test_heuristic_split_into_pool_keeps_the_last_heuristic_actions() {
    let mut pool = heuristic(cfg(), Schedule::Static)
        .async_mode(AsyncOptions::new(workers(2), NonZeroUsize::new(NU).unwrap()))
        .unwrap();
    drop(pool.reset_all(Reseed::Keep).unwrap());
    let ids: Vec<u32> = (0..N).collect();
    pool.send(&ids, input(&rows(0..NU, 0, i32::MIN))).unwrap();
    drop(pool.recv().unwrap());
    let last = pool.heuristic_actions().unwrap().to_vec();
    let p = pool.into_pool().ok().unwrap();
    assert_eq!(p.heuristic_actions().unwrap(), last.as_slice());
    assert_eq!(instance_rows(&last), sync_run().0[0]);
}
