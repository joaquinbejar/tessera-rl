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

/// Everything a token returns, as bits, plus the step counters.
fn token_frame<E: tessera_rl_core::Environment, M: tessera_rl_engine::MaskProvider<E>>(
    t: &HalfToken<E, M>,
) -> Vec<u64> {
    let mut v: Vec<u64> = Vec::new();
    if let Some(obs) = t.observation(0) {
        v.extend(obs.iter().map(|&b| u64::from(b)));
    }
    v.extend(t.rewards().iter().map(|r| u64::from(r.to_bits())));
    for flags in [
        t.agent_terminated(),
        t.agent_truncated(),
        t.alive(),
        t.acting(),
        t.masks(),
        t.episode_terminated(),
        t.episode_truncated(),
    ] {
        v.extend(flags.iter().map(|f| u64::from(f.get())));
    }
    v.extend(t.generations().iter().map(|&g| u64::from(g)));
    for stats in [
        t.episode_return(),
        t.episode_length(),
        t.episode_illegal_actions(),
        t.episode_invalid_agent_events(),
    ] {
        v.extend(stats.iter().map(|r| u64::from(r.to_bits())));
    }
    if let Some(term) = t.terminal_observation(0) {
        v.extend(term.iter().map(|&b| u64::from(b)));
    }
    for id in t.ids() {
        let (e, s) = t.step_counter(id).unwrap();
        v.extend([u64::from(e), u64::from(s)]);
    }
    v
}

/// The same rows of a Sync pool's outputs, in the order of `token_frame`.
fn pool_frame<E, M>(pool: &EnvPool<E, M>, ids: core::ops::Range<u32>) -> Vec<u64>
where
    E: tessera_rl_core::Environment,
    M: tessera_rl_engine::MaskProvider<E>,
    E::Config: Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
{
    let out = pool.outputs().unwrap();
    let n = out.episode_return().len();
    let (s, e) = (
        usize::try_from(ids.start).unwrap(),
        usize::try_from(ids.end).unwrap(),
    );
    let rows = |len: usize| {
        let per = len / n;
        s * per..e * per
    };
    let mut v: Vec<u64> = Vec::new();
    if let Some(obs) = out.observation(0) {
        v.extend(obs[rows(obs.len())].iter().map(|&b| u64::from(b)));
    }
    let rewards = out.rewards();
    v.extend(
        rewards[rows(rewards.len())]
            .iter()
            .map(|r| u64::from(r.to_bits())),
    );
    for flags in [
        out.agent_terminated(),
        out.agent_truncated(),
        out.alive(),
        out.acting(),
        out.masks(),
        out.episode_terminated(),
        out.episode_truncated(),
    ] {
        v.extend(flags[rows(flags.len())].iter().map(|f| u64::from(f.get())));
    }
    let gens = out.generations();
    v.extend(gens[rows(gens.len())].iter().map(|&g| u64::from(g)));
    for stats in [
        out.episode_return(),
        out.episode_length(),
        out.episode_illegal_actions(),
        out.episode_invalid_agent_events(),
    ] {
        v.extend(
            stats[rows(stats.len())]
                .iter()
                .map(|r| u64::from(r.to_bits())),
        );
    }
    if let Some(term) = out.terminal_observation(0) {
        v.extend(term[rows(term.len())].iter().map(|&b| u64::from(b)));
    }
    for id in ids {
        let identity = tessera_rl_core::RecordLayout::engine(pool.header(id).unwrap()).identity;
        v.extend([
            u64::from(identity.episode_index()),
            u64::from(identity.step_index()),
        ]);
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
    let Err(err) = r else {
        panic!("one chunk cannot be split");
    };
    assert_eq!(err.error, EngineError::HalfTooSmall { chunk_size: CHUNK });
    // The pool is handed back, still usable.
    let mut pool = *err.pool.unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
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

/// A held token of the idle half stays readable while the other half panics.
#[test]
fn test_pipelined_idle_half_token_survives_the_other_halfs_panic() {
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
    let before = b.rewards().to_vec();
    send(&mut p, a, &half_actions(&actions(0), &halves[0]));
    assert_eq!(p.recv().map(|_| ()), Err(EngineError::Panicked));
    assert!(p.is_poisoned());
    assert_eq!(b.rewards(), before.as_slice());
    drop(b);
    drop(p.reset_all(Reseed::Keep).unwrap());
    assert!(!p.is_poisoned());
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
    let Err(err) = p.pipelined(PipelinedOptions::new(NonZeroUsize::new(1).unwrap())) else {
        panic!("a Sync pool cannot be pipelined");
    };
    assert!(matches!(err.error, EngineError::Options { .. }));
    assert!(err.pool.is_some(), "the pool is handed back");
}

/// Under `debug-checks`, an action the mask forbids is refused by Sync's
/// step and by a Pipelined send alike, and the send changes nothing.
#[cfg(feature = "debug-checks")]
#[test]
fn test_pipelined_debug_checks_refuse_what_sync_refuses_and_mutate_nothing() {
    // Action 2 for agent 0 of instance 1 is masked illegal by Counter.
    let mut bad = actions(0);
    bad[2] = 2;
    let mut sync = sync_pool(AutoReset::NextStep);
    sync.reset_all(Reseed::Keep).unwrap();
    let sync_err = sync
        .step(ActionInput {
            discrete: &bad,
            continuous: &[],
        })
        .map(|_| ())
        .unwrap_err();
    assert!(
        matches!(sync_err, EngineError::IllegalAction { instance: 1, .. }),
        "{sync_err:?}"
    );
    let mut p = piped(AutoReset::NextStep, 2, cfg());
    let [a, _b] = p.reset_all(Reseed::Keep).unwrap();
    let halves = p.halves();
    let before = token_frame(&a);
    let generation = p.generation(0).unwrap();
    let err = p
        .send(
            a,
            ActionInput {
                discrete: &half_actions(&bad, &halves[0]),
                continuous: &[],
            },
        )
        .unwrap_err();
    assert_eq!(err.error, sync_err);
    let a = err.token.unwrap();
    assert_eq!(token_frame(&a), before);
    assert_eq!(p.generation(0).unwrap(), generation);
    assert_eq!(p.recv().map(|_| ()), Err(EngineError::NothingInFlight));
    // The token still sends legal actions.
    send(&mut p, a, &half_actions(&actions(0), &halves[0]));
    let _a = p.recv().unwrap();
}

// Codex's #82 reproduction probes (attachment 987), as regression tests.

#[test]
fn test_review_failed_recovery_keeps_poison() {
    let mut p = piped(
        AutoReset::NextStep,
        2,
        CountConfig {
            panic_at: 1,
            ..cfg()
        },
    );
    let [a, b] = p.reset_all(Reseed::Keep).unwrap();
    drop(b);
    let h = p.halves()[0].clone();
    send(&mut p, a, &half_actions(&actions(0), &h));
    assert_eq!(p.recv().map(|_| ()), Err(EngineError::Panicked));
    let before = p.generation(0).unwrap();
    assert!(p.reset_all(Reseed::Each(&[])).is_err());
    assert!(p.is_poisoned(), "failed reset cleared poison");
    assert_eq!(p.generation(0).unwrap(), before);
}

#[test]
fn test_review_into_pool_keeps_poison() {
    let mut p = piped(
        AutoReset::NextStep,
        2,
        CountConfig {
            panic_at: 1,
            ..cfg()
        },
    );
    let [a, b] = p.reset_all(Reseed::Keep).unwrap();
    drop(b);
    let h = p.halves()[0].clone();
    send(&mut p, a, &half_actions(&actions(0), &h));
    assert_eq!(p.recv().map(|_| ()), Err(EngineError::Panicked));
    let pool = p.into_pool().map_err(|e| e.kind).unwrap();
    assert!(pool.is_poisoned(), "conversion cleared poison");
    assert!(pool.state(0).is_none(), "partially written state exposed");
}

#[test]
fn test_review_failed_restore_keeps_cycles_and_generations() {
    let template = sync_pool(AutoReset::NextStep).snapshot(0).unwrap();
    let mut p = piped(AutoReset::NextStep, 2, cfg());
    let [a, b] = p.reset_all(Reseed::Keep).unwrap();
    let mut records = vec![template; N as usize];
    let initial = p.snapshot_all(&mut records).unwrap();
    let h = p.halves();
    send(&mut p, a, &half_actions(&actions(0), &h[0]));
    send(&mut p, b, &half_actions(&actions(0), &h[1]));
    drop(p.recv().unwrap());
    drop(p.recv().unwrap());
    let before = p.snapshot_all(&mut records).unwrap();
    let generation_before = p.generation(0).unwrap();
    assert!(p.restore_all(&[], &initial).is_err());
    let after = p.snapshot_all(&mut records).unwrap();
    assert_eq!(before, after, "failed restore changed cycles");
    assert_eq!(p.generation(0).unwrap(), generation_before);
}

#[test]
fn test_pipelined_failed_reset_on_a_healthy_pool_keeps_generation_and_cycles() {
    let mut p = piped(AutoReset::NextStep, 2, cfg());
    let tokens = p.reset_all(Reseed::Keep).unwrap();
    drop(tokens);
    let before = (p.generation(0).unwrap(), p.generation(1).unwrap());
    assert!(p.reset_all(Reseed::Each(&[])).is_err());
    assert_eq!((p.generation(0).unwrap(), p.generation(1).unwrap()), before);
    assert!(!p.is_poisoned());
    let _tokens = p.reset_all(Reseed::Keep).unwrap();
}

#[test]
fn test_pipelined_idle_token_on_a_poisoned_pool_is_refused() {
    let mut p = piped(
        AutoReset::NextStep,
        2,
        CountConfig {
            panic_at: 1,
            ..cfg()
        },
    );
    let [a, b] = p.reset_all(Reseed::Keep).unwrap();
    drop(b);
    let h = p.halves()[0].clone();
    send(&mut p, a, &half_actions(&actions(0), &h));
    assert_eq!(p.recv().map(|_| ()), Err(EngineError::Panicked));
    assert_eq!(p.idle_token(1).map(|_| ()), Err(EngineError::Poisoned));
}

#[test]
fn test_pipelined_recovery_clears_the_ready_flags() {
    let mut p = piped(
        AutoReset::NextStep,
        2,
        CountConfig {
            panic_at: 1,
            ..cfg()
        },
    );
    let waiter = p.waiter();
    let [a, b] = p.reset_all(Reseed::Keep).unwrap();
    let halves = p.halves();
    send(&mut p, a, &half_actions(&actions(0), &halves[0]));
    send(&mut p, b, &half_actions(&actions(0), &halves[1]));
    assert_eq!(p.recv().map(|_| ()), Err(EngineError::Panicked));
    // Recovery drains half B: no half is ready afterwards.
    let tokens = p.reset_all(Reseed::Keep).unwrap();
    assert!(!waiter.wait_ready(Duration::from_millis(1)));
    drop(tokens);
}

/// A token read and dropped on another thread, then `idle_token` and a
/// send on this thread: the token's end orders its reads before the
/// worker's writes (under Miri, any race on the real rows is reported).
#[test]
fn test_pipelined_token_dropped_on_another_thread_then_reclaimed_and_sent() {
    let mut p = piped(AutoReset::NextStep, 2, cfg());
    let [a, b] = p.reset_all(Reseed::Keep).unwrap();
    let reader = std::thread::spawn(move || {
        let sum: f32 = a.rewards().iter().sum();
        let frame = token_frame(&a);
        drop(a);
        (sum, frame.len())
    });
    let (_, len) = reader.join().unwrap();
    assert!(len > 0);
    let a = p.idle_token(0).unwrap();
    let h = p.halves()[0].clone();
    send(&mut p, a, &half_actions(&actions(0), &h));
    let a = p.recv().unwrap();
    assert_eq!(a.half(), 0);
    drop(b);
}

/// The wrapper moves (into a box, through a function) while a half is in
/// flight; the job keeps running on the shared allocation.
#[test]
fn test_pipelined_wrapper_moved_while_a_half_is_in_flight() {
    fn relay(p: Box<Piped>) -> Box<Piped> {
        p
    }
    let mut p = piped(AutoReset::NextStep, 2, cfg());
    let [a, b] = p.reset_all(Reseed::Keep).unwrap();
    let h = p.halves()[0].clone();
    send(&mut p, a, &half_actions(&actions(0), &h));
    let mut moved = relay(Box::new(p));
    let a = moved.recv().unwrap();
    assert_eq!(a.half(), 0);
    let mut back = *moved;
    drop((a, b));
    let _tokens = back.reset_all(Reseed::Keep).unwrap();
}

mod drift {
    //! A one-agent environment with one continuous action head: the state
    //! accumulates the action, the reward is the action, and episodes end
    //! after five steps.

    use tessera_rl_core::io::{Actions, ObsWriter};
    use tessera_rl_core::space::{BoxBounds, DType, Space};
    use tessera_rl_core::{
        AgentSlot, EpisodeEnd, NoAdaptive, RecordHeader, SpaceError, StepOut, Tier,
    };
    use tessera_rl_macros::{EnvConfig, EnvState};
    use tessera_rl_num::rng::{ObsRng, StepRng};

    #[derive(EnvState, Debug, Clone, Copy, Default)]
    #[repr(C)]
    pub struct Pos {
        pub x: f32,
        pub steps: u32,
    }

    #[derive(EnvConfig, Debug, Clone)]
    pub struct DriftConfig {}

    pub struct Drift;

    impl tessera_rl_core::Environment for Drift {
        type State = Pos;
        type Config = DriftConfig;
        type Obs = ();
        type Action = ();
        type Header = RecordHeader<1, NoAdaptive>;

        const MAX_AGENTS: usize = 1;
        const TIER: Tier = Tier::CrossPlatform;

        fn observation_space(_: &DriftConfig) -> Result<Space, SpaceError> {
            Ok(Space::Discrete { n: 2 })
        }

        fn action_space(_: &DriftConfig) -> Result<Space, SpaceError> {
            Ok(Space::Box {
                shape: vec![1],
                dtype: DType::F32,
                bounds: BoxBounds::Float {
                    low: vec![-1.0],
                    high: vec![1.0],
                },
            })
        }

        fn reset(_: &DriftConfig, rng: &mut StepRng, state: &mut Pos) {
            state.x = f32::from(u8::try_from(rng.next_u32() % 4).unwrap_or(0)) * 0.25;
            state.steps = 0;
        }

        fn step(
            _: &DriftConfig,
            state: &mut Pos,
            actions: &Actions<'_, ()>,
            _: &mut StepRng,
            out: &mut StepOut<'_>,
        ) {
            let a = actions.continuous(AgentSlot::FIRST, 0).unwrap_or(0.0);
            state.x += a;
            state.steps += 1;
            out.set_reward(AgentSlot::FIRST, a);
            if state.steps >= 5 {
                out.end_episode(EpisodeEnd::Terminated);
            }
        }

        fn observe(
            _: &DriftConfig,
            _: &Pos,
            _: AgentSlot,
            _: &mut ObsRng,
            _: &mut ObsWriter<'_, '_, ()>,
        ) {
        }
    }
}

#[test]
fn test_pipelined_continuous_head_matches_sync_bit_for_bit() {
    use drift::{Drift, DriftConfig};
    let count = 40_u32;
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(8).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(count).unwrap(), Seed::new(5))
    };
    let conts = |t: usize| -> Vec<f32> {
        (0..usize::try_from(count).unwrap())
            .map(|k| f32::from(u8::try_from((k * 3 + t) % 5).unwrap()) * 0.25 - 0.5)
            .collect()
    };
    let mut reference = EnvPool::<Drift>::new(DriftConfig {}, options).unwrap();
    reference.reset_all(Reseed::Keep).unwrap();
    let mut piped_pool = EnvPool::<Drift>::new(DriftConfig {}, options)
        .unwrap()
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(3).unwrap()))
        .unwrap();
    let halves = piped_pool.halves();
    let [mut a, mut b] = piped_pool.reset_all(Reseed::Keep).unwrap();
    let span = |r: &core::ops::Range<u32>| {
        usize::try_from(r.start).unwrap()..usize::try_from(r.end).unwrap()
    };
    for t in 0..12 {
        let c = conts(t);
        reference
            .step(ActionInput {
                discrete: &[],
                continuous: &c,
            })
            .unwrap();
        for (tok, h) in [(a, &halves[0]), (b, &halves[1])] {
            piped_pool
                .send(
                    tok,
                    ActionInput {
                        discrete: &[],
                        continuous: &c[span(h)],
                    },
                )
                .map_err(|e| e.error)
                .unwrap();
        }
        a = piped_pool.recv().unwrap();
        b = piped_pool.recv().unwrap();
        assert_eq!(
            token_frame(&a),
            pool_frame(&reference, halves[0].clone()),
            "t {t}"
        );
        assert_eq!(
            token_frame(&b),
            pool_frame(&reference, halves[1].clone()),
            "t {t}"
        );
    }
    drop((a, b));
    let pool = piped_pool.into_pool().map_err(|e| e.kind).unwrap();
    for i in 0..count {
        assert_eq!(
            pool.state_hash(i).unwrap(),
            reference.state_hash(i).unwrap()
        );
    }
}

// Codex's #82 r2 probe (attachment 989), as a regression test.

#[test]
fn test_review_pipelined_conversion_preserves_existing_poison() {
    let mut pool = EnvPool::<Counter, Masked>::with_masks(
        CountConfig {
            panic_at: 1,
            ..cfg()
        },
        options(AutoReset::NextStep),
    )
    .unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    assert_eq!(
        pool.step(ActionInput {
            discrete: &actions(0),
            continuous: &[]
        })
        .map(|_| ()),
        Err(EngineError::Panicked)
    );
    assert!(pool.is_poisoned());
    let p = pool
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
        .unwrap();
    assert!(p.is_poisoned(), "incoming EnvPool poison was lost");
    assert_eq!(p.idle_token(0).map(|_| ()), Err(EngineError::Poisoned));
}

#[test]
fn test_pipelined_poisoned_conversion_recovers_with_reset_all() {
    let mut pool = EnvPool::<Counter, Masked>::with_masks(
        CountConfig {
            panic_at: 1,
            ..cfg()
        },
        options(AutoReset::NextStep),
    )
    .unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    assert!(
        pool.step(ActionInput {
            discrete: &actions(0),
            continuous: &[],
        })
        .is_err()
    );
    let mut p = pool
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
        .unwrap();
    assert!(p.is_poisoned());
    let _tokens = p.reset_all(Reseed::Keep).unwrap();
    assert!(!p.is_poisoned());
}

/// Under `debug-checks`, a value out of the head's range reports the same
/// error in Pipelined as in Sync (the range check runs before legality).
#[cfg(feature = "debug-checks")]
#[test]
fn test_pipelined_debug_checks_out_of_range_value_reports_what_sync_reports() {
    let mut bad = actions(0);
    bad[2] = 99;
    let mut sync = sync_pool(AutoReset::NextStep);
    sync.reset_all(Reseed::Keep).unwrap();
    let sync_err = sync
        .step(ActionInput {
            discrete: &bad,
            continuous: &[],
        })
        .map(|_| ())
        .unwrap_err();
    let mut p = piped(AutoReset::NextStep, 2, cfg());
    let [a, _b] = p.reset_all(Reseed::Keep).unwrap();
    let halves = p.halves();
    let err = p
        .send(
            a,
            ActionInput {
                discrete: &half_actions(&bad, &halves[0]),
                continuous: &[],
            },
        )
        .unwrap_err();
    assert!(
        !matches!(sync_err, EngineError::IllegalAction { .. }),
        "{sync_err:?}"
    );
    assert_eq!(err.error, sync_err);
}
