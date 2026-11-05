// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pipelined mode (ADR-0006, ADR-0047): every half batch is bit-identical
//! to the same rows of a Sync run with the same actions, for any worker
//! count; tokens, cycles, poisoning, recovery and ownership.

#![cfg(not(loom))]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers unwrap and slice: a failure is the assertion"
)]

mod common;

use core::num::{NonZeroU32, NonZeroUsize};
use core::time::Duration;

use common::{CountConfig, Counter};
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, EnvPool, HalfToken, IntoPoolErrorKind, Masked,
    PipelinedOptions, PipelinedPool, PoolOptions, Record, Reseed,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 100;
const CHUNK: u32 = 8;

type Pool = EnvPool<Counter, Masked>;
type Piped = PipelinedPool<Counter, Masked>;
type Token = HalfToken<Counter, Masked>;

fn options(mode: AutoReset) -> PoolOptions {
    PoolOptions {
        auto_reset: mode,
        chunk_size: NonZeroU32::new(CHUNK).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(77))
    }
}

fn cfg() -> CountConfig {
    CountConfig {
        length: 7,
        kill_at: 4,
        ..CountConfig::default()
    }
}

fn sync_pool(mode: AutoReset) -> Pool {
    EnvPool::with_masks(cfg(), options(mode)).unwrap()
}

fn piped(mode: AutoReset, workers: usize, cfg: CountConfig) -> Piped {
    EnvPool::with_masks(cfg, options(mode))
        .unwrap()
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(workers).unwrap()))
        .unwrap()
}

/// Actions of step `t` for every instance, `[N, 2]`. Agent 0 sometimes
/// acts illegally (action 2) to exercise the noop, except under
/// `debug-checks`, which rejects it.
fn actions(t: usize) -> Vec<i32> {
    let n = usize::try_from(N).unwrap();
    (0..n * 2)
        .map(|k| {
            let a = i32::try_from((k * 7 + t * 3) % 3).unwrap();
            if tessera_rl_engine::DEBUG_CHECKS && k % 2 == 0 && a == 2 {
                1
            } else {
                a
            }
        })
        .collect()
}

/// Everything a token returns, as bits.
fn token_frame(t: &Token) -> Vec<u64> {
    let mut v: Vec<u64> = t
        .observation(0)
        .unwrap()
        .iter()
        .map(|&b| u64::from(b))
        .collect();
    v.extend(t.rewards().iter().map(|r| u64::from(r.to_bits())));
    for flags in [
        t.agent_terminated(),
        t.agent_truncated(),
        t.alive(),
        t.masks(),
    ] {
        v.extend(flags.iter().map(|f| u64::from(f.get())));
    }
    v.extend(t.episode_terminated().iter().map(|f| u64::from(f.get())));
    v.extend(t.episode_return().iter().map(|r| u64::from(r.to_bits())));
    if let Some(term) = t.terminal_observation(0) {
        v.extend(term.iter().map(|&b| u64::from(b)));
    }
    v
}

/// The same rows of a Sync pool's outputs.
fn pool_frame(pool: &Pool, ids: core::ops::Range<u32>) -> Vec<u64> {
    let out = pool.outputs();
    let a = 2;
    let (s, e) = (
        usize::try_from(ids.start).unwrap(),
        usize::try_from(ids.end).unwrap(),
    );
    let leaf = out.observation(0).unwrap();
    let per = leaf.len() / usize::try_from(N).unwrap();
    let mut v: Vec<u64> = leaf[s * per..e * per]
        .iter()
        .map(|&b| u64::from(b))
        .collect();
    v.extend(
        out.rewards()[s * a..e * a]
            .iter()
            .map(|r| u64::from(r.to_bits())),
    );
    let masks = out.masks();
    let mper = masks.len() / usize::try_from(N).unwrap();
    for (flags, per) in [
        (out.agent_terminated(), a),
        (out.agent_truncated(), a),
        (out.alive(), a),
        (masks, mper),
    ] {
        v.extend(flags[s * per..e * per].iter().map(|f| u64::from(f.get())));
    }
    v.extend(
        out.episode_terminated()[s..e]
            .iter()
            .map(|f| u64::from(f.get())),
    );
    v.extend(
        out.episode_return()[s..e]
            .iter()
            .map(|r| u64::from(r.to_bits())),
    );
    if let Some(term) = out.terminal_observation(0) {
        v.extend(term[s * per..e * per].iter().map(|&b| u64::from(b)));
    }
    v
}

fn half_actions(all: &[i32], ids: &core::ops::Range<u32>) -> Vec<i32> {
    let (s, e) = (
        usize::try_from(ids.start).unwrap() * 2,
        usize::try_from(ids.end).unwrap() * 2,
    );
    all[s..e].to_vec()
}

fn send(p: &mut Piped, t: Token, a: &[i32]) {
    p.send(
        t,
        ActionInput {
            discrete: a,
            continuous: &[],
        },
    )
    .map_err(|e| e.error)
    .unwrap();
}

#[test]
fn test_pipelined_matches_sync_bit_for_bit_for_1_2_7_and_64_workers() {
    for mode in [AutoReset::NextStep, AutoReset::SameStep] {
        // The Sync reference.
        let mut reference = sync_pool(mode);
        reference.reset_all(Reseed::Keep).unwrap();
        let mut frames: Vec<[Vec<u64>; 2]> = Vec::new();
        let halves = {
            let p = piped(mode, 1, cfg());
            p.halves()
        };
        frames.push([
            pool_frame(&reference, halves[0].clone()),
            pool_frame(&reference, halves[1].clone()),
        ]);
        for t in 0..20 {
            let a = actions(t);
            reference
                .step(ActionInput {
                    discrete: &a,
                    continuous: &[],
                })
                .unwrap();
            frames.push([
                pool_frame(&reference, halves[0].clone()),
                pool_frame(&reference, halves[1].clone()),
            ]);
        }
        let hashes: Vec<u64> = (0..N)
            .map(|i| reference.state_hash(i).unwrap().get())
            .collect();
        for workers in [1, 2, 7, 64] {
            let mut p = piped(mode, workers, cfg());
            let [a0, b0] = p.reset_all(Reseed::Keep).unwrap();
            assert_eq!(token_frame(&a0), frames[0][0], "{workers}, {mode:?}");
            assert_eq!(token_frame(&b0), frames[0][1], "{workers}, {mode:?}");
            let (mut ta, mut tb) = (a0, b0);
            for t in 0..20 {
                let a = actions(t);
                send(&mut p, ta, &half_actions(&a, &halves[0]));
                send(&mut p, tb, &half_actions(&a, &halves[1]));
                // FIFO: A then B.
                ta = p.recv().unwrap();
                assert_eq!(ta.half(), 0);
                assert_eq!(
                    token_frame(&ta),
                    frames[t + 1][0],
                    "{workers}, {mode:?}, t {t}"
                );
                tb = p.recv().unwrap();
                assert_eq!(tb.half(), 1);
                assert_eq!(
                    token_frame(&tb),
                    frames[t + 1][1],
                    "{workers}, {mode:?}, t {t}"
                );
            }
            drop((ta, tb));
            let pool = p.into_pool().map_err(|e| e.kind).unwrap();
            let got: Vec<u64> = (0..N).map(|i| pool.state_hash(i).unwrap().get()).collect();
            assert_eq!(got, hashes, "{workers}, {mode:?}");
            assert_eq!(pool.workers(), workers);
        }
    }
}

#[test]
fn test_pipelined_half_a_stays_readable_while_half_b_runs() {
    let mut p = piped(AutoReset::NextStep, 4, cfg());
    let [a, b] = p.reset_all(Reseed::Keep).unwrap();
    let halves = p.halves();
    send(&mut p, a, &half_actions(&actions(0), &halves[0]));
    let a = p.recv().unwrap();
    let before = token_frame(&a);
    send(&mut p, b, &half_actions(&actions(0), &halves[1]));
    // Half A's results are untouched while B is in flight or done.
    assert_eq!(token_frame(&a), before);
    let b = p.recv().unwrap();
    assert_eq!(token_frame(&a), before);
    assert_eq!(b.ids(), halves[1].clone());
    assert!(b.step_counter(halves[1].start).is_some());
    assert!(b.step_counter(halves[0].start).is_none());
}

#[test]
fn test_pipelined_recv_with_nothing_in_flight_is_an_error() {
    let mut p = piped(AutoReset::NextStep, 2, cfg());
    let _tokens = p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(p.recv().map(|_| ()), Err(EngineError::NothingInFlight));
    assert_eq!(p.try_recv().map(|_| ()), Err(EngineError::NothingInFlight));
}

#[test]
fn test_pipelined_too_few_chunks_is_half_too_small() {
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(CHUNK).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(CHUNK).unwrap(), Seed::new(1))
    };
    let r = Pool::with_masks(cfg(), options)
        .unwrap()
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(1).unwrap()));
    assert!(matches!(
        r,
        Err(EngineError::HalfTooSmall { chunk_size: CHUNK })
    ));
}

#[test]
fn test_pipelined_token_of_another_pool_is_wrong_pool() {
    let mut p = piped(AutoReset::NextStep, 1, cfg());
    let mut q = piped(AutoReset::NextStep, 1, cfg());
    let _mine = p.reset_all(Reseed::Keep).unwrap();
    let [foreign, _] = q.reset_all(Reseed::Keep).unwrap();
    let halves = p.halves();
    let err = p
        .send(
            foreign,
            ActionInput {
                discrete: &half_actions(&actions(0), &halves[0]),
                continuous: &[],
            },
        )
        .unwrap_err();
    assert_eq!(err.error, EngineError::WrongPool);
    assert!(err.token.is_some(), "the token is handed back");
}

#[test]
fn test_pipelined_invalid_actions_hand_the_token_back_unchanged() {
    let mut p = piped(AutoReset::NextStep, 2, cfg());
    let [a, _b] = p.reset_all(Reseed::Keep).unwrap();
    let before = token_frame(&a);
    let generation = p.generation(0).unwrap();
    let err = p
        .send(
            a,
            ActionInput {
                discrete: &[0; 3],
                continuous: &[],
            },
        )
        .unwrap_err();
    assert!(
        matches!(err.error, EngineError::ActionLength { .. }),
        "{:?}",
        err.error
    );
    let a = err.token.unwrap();
    assert_eq!(token_frame(&a), before);
    assert_eq!(
        p.generation(0).unwrap(),
        generation,
        "a refused send moves no generation"
    );
    assert_eq!(p.recv().map(|_| ()), Err(EngineError::NothingInFlight));
}

#[test]
fn test_pipelined_send_moves_the_generation_before_any_write() {
    let mut p = piped(AutoReset::NextStep, 2, cfg());
    let [a, _b] = p.reset_all(Reseed::Keep).unwrap();
    let generation = p.generation(0).unwrap();
    let h = p.halves()[0].clone();
    send(&mut p, a, &half_actions(&actions(0), &h));
    assert!(p.generation(0).unwrap() > generation);
    let _a = p.recv().unwrap();
}

#[test]
fn test_pipelined_a_half_cannot_run_two_cycles_ahead() {
    let mut p = piped(AutoReset::NextStep, 2, cfg());
    let [mut a, _b] = p.reset_all(Reseed::Keep).unwrap();
    let h = p.halves()[0].clone();
    for _ in 0..2 {
        send(&mut p, a, &half_actions(&actions(0), &h));
        a = p.recv().unwrap();
    }
    let err = p
        .send(
            a,
            ActionInput {
                discrete: &half_actions(&actions(0), &h),
                continuous: &[],
            },
        )
        .unwrap_err();
    assert_eq!(
        err.error,
        EngineError::CycleAhead {
            cycle: 2,
            needed: 0
        }
    );
    assert!(err.token.is_some());
}

/// Runs four cycles with half B one or two sends behind half A and
/// returns the committed contexts and the final state hashes.
fn interleaving(late_b: bool) -> (Vec<Option<f64>>, Vec<u64>) {
    let mut p = piped(AutoReset::NextStep, 3, cfg());
    let [mut a, mut b] = p.reset_all(Reseed::Keep).unwrap();
    let halves = p.halves();
    let mut contexts = Vec::new();
    for t in 0..4 {
        let act = actions(t);
        if late_b {
            send(&mut p, a, &half_actions(&act, &halves[0]));
            a = p.recv().unwrap();
            send(&mut p, b, &half_actions(&act, &halves[1]));
            b = p.recv().unwrap();
        } else {
            send(&mut p, b, &half_actions(&act, &halves[1]));
            send(&mut p, a, &half_actions(&act, &halves[0]));
            b = p.recv().unwrap();
            a = p.recv().unwrap();
        }
        contexts.push(p.context_version(u64::try_from(t).unwrap()));
    }
    drop((a, b));
    let pool = p.into_pool().map_err(|e| e.kind).unwrap();
    (
        contexts,
        (0..N).map(|i| pool.state_hash(i).unwrap().get()).collect(),
    )
}

#[test]
fn test_pipelined_two_interleavings_give_identical_bytes_and_contexts() {
    let (ca, ha) = interleaving(false);
    let (cb, hb) = interleaving(true);
    assert!(ca.iter().all(Option::is_some));
    assert_eq!(
        ca.iter().map(|c| c.map(f64::to_bits)).collect::<Vec<_>>(),
        cb.iter().map(|c| c.map(f64::to_bits)).collect::<Vec<_>>()
    );
    assert_eq!(ha, hb);
}

#[test]
fn test_pipelined_writers_refuse_while_a_token_lives_and_into_pool_hands_back() {
    let mut p = piped(AutoReset::NextStep, 2, cfg());
    let [a, b] = p.reset_all(Reseed::Keep).unwrap();
    assert!(matches!(
        p.reset_all(Reseed::Keep).map(|_| ()),
        Err(EngineError::HalfBorrowed { .. })
    ));
    let Err(err) = p.into_pool() else {
        panic!("into_pool must refuse with a live token");
    };
    assert_eq!(err.kind, IntoPoolErrorKind::LiveTokens);
    let mut p = *err.pool;
    // The returned wrapper still works.
    let halves = p.halves();
    send(&mut p, a, &half_actions(&actions(0), &halves[0]));
    send(&mut p, b, &half_actions(&actions(0), &halves[1]));
    assert!(matches!(
        p.reset_all(Reseed::Keep).map(|_| ()),
        Err(EngineError::NotQuiescent)
    ));
    let a = p.recv().unwrap();
    let b = p.recv().unwrap();
    drop((a, b));
    let _tokens = p.reset_all(Reseed::Keep).unwrap();
}

#[test]
fn test_pipelined_idle_token_after_a_drop_and_its_errors() {
    let mut p = piped(AutoReset::NextStep, 2, cfg());
    let [a, b] = p.reset_all(Reseed::Keep).unwrap();
    assert!(matches!(
        p.idle_token(0),
        Err(EngineError::HalfBorrowed { half: 0 })
    ));
    assert!(matches!(
        p.idle_token(2),
        Err(EngineError::InvalidHalf { half: 2 })
    ));
    drop(a);
    let a = p.idle_token(0).unwrap();
    let h = p.halves()[0].clone();
    send(&mut p, a, &half_actions(&actions(0), &h));
    assert!(matches!(
        p.idle_token(0),
        Err(EngineError::HalfInFlight { half: 0 })
    ));
    let _a = p.recv().unwrap();
    drop(b);
}

#[test]
fn test_pipelined_token_outlives_the_wrapper() {
    let mut p = piped(AutoReset::NextStep, 2, cfg());
    let [a, b] = p.reset_all(Reseed::Keep).unwrap();
    let before = token_frame(&a);
    drop(p);
    // The rows stay readable through the token after the workers are gone.
    assert_eq!(token_frame(&a), before);
    drop((a, b));
}

#[test]
fn test_pipelined_worker_panic_poisons_and_reset_all_recovers() {
    let mut p = piped(
        AutoReset::NextStep,
        4,
        CountConfig {
            panic_at: 1,
            length: 9,
            ..CountConfig::default()
        },
    );
    let [a, b] = p.reset_all(Reseed::Keep).unwrap();
    let halves = p.halves();
    send(&mut p, a, &half_actions(&actions(0), &halves[0]));
    send(&mut p, b, &half_actions(&actions(0), &halves[1]));
    assert_eq!(p.recv().map(|_| ()), Err(EngineError::Panicked));
    assert!(p.is_poisoned());
    assert_eq!(p.recv().map(|_| ()), Err(EngineError::Poisoned));
    // Recovery drains the half still in flight.
    let _tokens = p.reset_all(Reseed::Keep).unwrap();
    assert!(!p.is_poisoned());
}

#[test]
fn test_pipelined_disabled_mode_refuses_a_send_after_the_episode_ends() {
    let mut p = piped(AutoReset::Disabled, 2, CountConfig { length: 1, ..cfg() });
    let [mut a, _b] = p.reset_all(Reseed::Keep).unwrap();
    let h = p.halves()[0].clone();
    send(&mut p, a, &half_actions(&actions(0), &h));
    a = p.recv().unwrap();
    let err = p
        .send(
            a,
            ActionInput {
                discrete: &half_actions(&actions(1), &h),
                continuous: &[],
            },
        )
        .unwrap_err();
    assert!(
        matches!(err.error, EngineError::PendingReset { .. }),
        "{:?}",
        err.error
    );
}

#[test]
fn test_pipelined_snapshot_and_restore_carry_the_cycle_state() {
    let template = sync_pool(AutoReset::NextStep).snapshot(0).unwrap();
    let mut p = piped(AutoReset::NextStep, 2, cfg());
    let [mut a, mut b] = p.reset_all(Reseed::Keep).unwrap();
    let halves = p.halves();
    for t in 0..3 {
        send(&mut p, a, &half_actions(&actions(t), &halves[0]));
        send(&mut p, b, &half_actions(&actions(t), &halves[1]));
        a = p.recv().unwrap();
        b = p.recv().unwrap();
    }
    let n = usize::try_from(N).unwrap();
    let mut records: Vec<Record<Counter>> = vec![template; n];
    let cycles = p.snapshot_all(&mut records).unwrap();
    let context = p.context_version(2);
    drop((a, b));
    let fresh = p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(p.context_version(2), None);
    drop(fresh);
    p.restore_all(&records, &cycles).unwrap();
    assert_eq!(
        p.context_version(2).map(f64::to_bits),
        context.map(f64::to_bits)
    );
}

#[test]
fn test_pipelined_waiter_and_recv_timeout() {
    let mut p = piped(AutoReset::NextStep, 2, cfg());
    let waiter = p.waiter();
    let [a, _b] = p.reset_all(Reseed::Keep).unwrap();
    assert!(!waiter.wait_ready(Duration::from_millis(1)));
    let h = p.halves()[0].clone();
    send(&mut p, a, &half_actions(&actions(0), &h));
    let w = waiter.clone();
    let seen = std::thread::spawn(move || w.wait_ready(Duration::from_secs(10)))
        .join()
        .unwrap();
    assert!(seen);
    let a = p.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(a.is_some());
}

#[test]
fn test_pipelined_already_sync_pool_is_refused() {
    let p = sync_pool(AutoReset::NextStep)
        .with_workers(tessera_rl_engine::SyncOptions::new(
            NonZeroUsize::new(1).unwrap(),
        ))
        .unwrap();
    assert!(matches!(
        p.pipelined(PipelinedOptions::new(NonZeroUsize::new(1).unwrap())),
        Err(EngineError::Options { .. })
    ));
}
