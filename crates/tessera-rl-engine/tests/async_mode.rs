// SPDX-License-Identifier: MIT OR Apache-2.0

//! Async mode (ADR-0006, ADR-0047): every instance's trajectory is
//! bit-identical to Sync with the same actions, for any worker count and
//! any send order; tokens, errors, writers, poisoning and ownership.

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
    ActionInput, AsyncOptions, AsyncPool, AutoReset, BatchToken, EngineError, EnvPool, Masked,
    PoolOptions, Record, Reseed,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 100;
const CHUNK: u32 = 8;
const BATCH: usize = 10;
const STEPS: usize = 12;

type Pool = EnvPool<Counter, Masked>;
type Asy = AsyncPool<Counter, Masked>;
type Token = BatchToken<Counter, Masked>;

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

fn asy_with(mode: AutoReset, workers: usize, batch: usize, cfg: CountConfig) -> Asy {
    EnvPool::with_masks(cfg, options(mode))
        .unwrap()
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(workers).unwrap(),
            NonZeroUsize::new(batch).unwrap(),
        ))
        .unwrap()
}

fn asy(mode: AutoReset, workers: usize) -> Asy {
    asy_with(mode, workers, BATCH, cfg())
}

/// Actions of step `t` for instance `i`, `[2]`.
fn row(t: usize, i: usize) -> [i32; 2] {
    let pick = |k: usize| {
        let a = i32::try_from((k * 7 + t * 3) % 3).unwrap();
        if tessera_rl_engine::DEBUG_CHECKS && k.is_multiple_of(2) && a == 2 {
            1
        } else {
            a
        }
    };
    [pick(i * 2), pick(i * 2 + 1)]
}

fn all_actions(t: usize) -> Vec<i32> {
    (0..usize::try_from(N).unwrap())
        .flat_map(|i| row(t, i))
        .collect()
}

/// Everything a token returns for `id`, as bits, plus the step counter.
fn token_frame(t: &Token, id: u32) -> Vec<u64> {
    let mut v: Vec<u64> = Vec::new();
    v.extend(t.observation(0, id).unwrap().iter().map(|&b| u64::from(b)));
    v.extend(
        t.rewards(id)
            .unwrap()
            .iter()
            .map(|r| u64::from(r.to_bits())),
    );
    for flags in [
        t.agent_terminated(id).unwrap(),
        t.agent_truncated(id).unwrap(),
        t.alive(id).unwrap(),
        t.acting(id).unwrap(),
        t.masks(id).unwrap(),
    ] {
        v.extend(flags.iter().map(|f| u64::from(f.get())));
    }
    v.push(u64::from(t.episode_terminated(id).unwrap().get()));
    v.push(u64::from(t.episode_truncated(id).unwrap().get()));
    v.extend(t.generations(id).unwrap().iter().map(|&g| u64::from(g)));
    for stat in [
        t.episode_return(id),
        t.episode_length(id),
        t.episode_illegal_actions(id),
        t.episode_invalid_agent_events(id),
    ] {
        v.push(u64::from(stat.unwrap().to_bits()));
    }
    if let Some(term) = t.terminal_observation(0, id) {
        v.extend(term.iter().map(|&b| u64::from(b)));
    }
    let (e, s) = t.step_counter(id).unwrap();
    v.extend([u64::from(e), u64::from(s)]);
    v
}

/// The same row of a Sync pool's outputs, in the order of `token_frame`.
fn pool_frame(pool: &Pool, id: u32) -> Vec<u64> {
    let out = pool.outputs();
    let n = out.episode_return().len();
    let i = usize::try_from(id).unwrap();
    let rows = |len: usize| {
        let per = len / n;
        i * per..(i + 1) * per
    };
    let mut v: Vec<u64> = Vec::new();
    let obs = out.observation(0).unwrap();
    v.extend(obs[rows(obs.len())].iter().map(|&b| u64::from(b)));
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
    ] {
        v.extend(flags[rows(flags.len())].iter().map(|f| u64::from(f.get())));
    }
    v.push(u64::from(out.episode_terminated()[i].get()));
    v.push(u64::from(out.episode_truncated()[i].get()));
    let gens = out.generations();
    v.extend(gens[rows(gens.len())].iter().map(|&g| u64::from(g)));
    for stats in [
        out.episode_return(),
        out.episode_length(),
        out.episode_illegal_actions(),
        out.episode_invalid_agent_events(),
    ] {
        v.push(u64::from(stats[i].to_bits()));
    }
    if let Some(term) = out.terminal_observation(0) {
        v.extend(term[rows(term.len())].iter().map(|&b| u64::from(b)));
    }
    let identity = tessera_rl_core::RecordLayout::engine(pool.header(id).unwrap()).identity;
    v.extend([
        u64::from(identity.episode_index()),
        u64::from(identity.step_index()),
    ]);
    v
}

fn send(p: &mut Asy, ids: &[u32], a: &[i32]) {
    p.send(
        ids,
        ActionInput {
            discrete: a,
            continuous: &[],
        },
    )
    .unwrap();
}

fn rows_for(ids: &[u32], step: &[usize]) -> Vec<i32> {
    ids.iter()
        .flat_map(|&id| {
            let i = usize::try_from(id).unwrap();
            row(step[i], i)
        })
        .collect()
}

/// A small deterministic generator for the send order.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

#[test]
fn test_async_matches_sync_per_instance_for_1_2_7_and_64_workers() {
    for mode in [AutoReset::NextStep, AutoReset::SameStep] {
        // The Sync reference: frames[t][i] after t steps.
        let mut reference = sync_pool(mode);
        reference.reset_all(Reseed::Keep).unwrap();
        let mut frames: Vec<Vec<Vec<u64>>> =
            vec![(0..N).map(|i| pool_frame(&reference, i)).collect()];
        for t in 0..STEPS {
            let a = all_actions(t);
            reference
                .step(ActionInput {
                    discrete: &a,
                    continuous: &[],
                })
                .unwrap();
            frames.push((0..N).map(|i| pool_frame(&reference, i)).collect());
        }
        let hashes: Vec<u64> = (0..N)
            .map(|i| reference.state_hash(i).unwrap().get())
            .collect();
        for workers in [1, 2, 7, 64] {
            // Batches of one, so every id comes back as soon as it is done
            // and the run never waits on a batch it cannot fill.
            let mut p = asy_with(mode, workers, 1, cfg());
            let first = p.reset_all(Reseed::Keep).unwrap();
            for id in 0..N {
                assert_eq!(
                    token_frame(&first, id),
                    frames[0][usize::try_from(id).unwrap()]
                );
            }
            drop(first);
            let n = usize::try_from(N).unwrap();
            let mut step = vec![0_usize; n];
            let mut idle: Vec<u32> = (0..N).collect();
            let mut rng = Lcg(u64::try_from(workers).unwrap() * 31 + 7);
            let mut done = 0;
            while done < n {
                // Send a random subset of the idle ids, in a random order.
                let mut subset = Vec::new();
                let take = 1 + usize::try_from(rng.next()).unwrap() % idle.len().max(1);
                for _ in 0..take.min(idle.len()) {
                    let k = usize::try_from(rng.next()).unwrap() % idle.len();
                    subset.push(idle.swap_remove(k));
                }
                if !subset.is_empty() {
                    let a = rows_for(&subset, &step);
                    send(&mut p, &subset, &a);
                }
                let Some(t) = p.try_recv().unwrap() else {
                    if idle.is_empty() || subset.is_empty() {
                        // Everything is out: block for a batch.
                        let t = p.recv().unwrap();
                        check(&t, &mut step, &frames, &mut idle, &mut done, workers, mode);
                    }
                    continue;
                };
                check(&t, &mut step, &frames, &mut idle, &mut done, workers, mode);
            }
            // Drain what is still out, then compare the final states.
            while p.recv().is_ok() {}
            let pool = p.into_pool().map_err(|_| ()).unwrap();
            let got: Vec<u64> = (0..N).map(|i| pool.state_hash(i).unwrap().get()).collect();
            let want: Vec<u64> = (0..N)
                .map(|i| {
                    if step[usize::try_from(i).unwrap()] == STEPS {
                        hashes[usize::try_from(i).unwrap()]
                    } else {
                        got[usize::try_from(i).unwrap()]
                    }
                })
                .collect();
            assert_eq!(got, want, "{workers}, {mode:?}");
            assert_eq!(pool.workers(), workers);
        }
    }
}

/// Checks a received batch against the Sync frames and returns its ids
/// to the idle set (or retires them after the last step).
fn check(
    t: &Token,
    step: &mut [usize],
    frames: &[Vec<Vec<u64>>],
    idle: &mut Vec<u32>,
    done: &mut usize,
    workers: usize,
    mode: AutoReset,
) {
    assert_eq!(t.len(), 1);
    for id in t.ids() {
        let i = usize::try_from(id).unwrap();
        step[i] += 1;
        assert_eq!(
            token_frame(t, id),
            frames[step[i]][i],
            "{workers}, {mode:?}, id {id}, step {}",
            step[i]
        );
        if step[i] == STEPS {
            *done += 1;
        } else {
            idle.push(id);
        }
    }
}

#[test]
fn test_async_recv_errors_never_wait() {
    let mut p = asy(AutoReset::NextStep, 2);
    drop(p.reset_all(Reseed::Keep).unwrap());
    assert_eq!(p.recv().map(|_| ()), Err(EngineError::NothingInFlight));
    assert_eq!(p.try_recv().map(|_| ()), Err(EngineError::NothingInFlight));
    send(&mut p, &[0, 1, 2], &rows_for(&[0, 1, 2], &[0; 100]));
    assert_eq!(
        p.recv().map(|_| ()),
        Err(EngineError::BatchUnreachable {
            in_flight: 3,
            ready: 0,
            batch_size: 10
        })
    );
}

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

#[test]
fn test_async_send_errors_change_nothing() {
    let mut p = asy(AutoReset::NextStep, 2);
    let all = p.reset_all(Reseed::Keep).unwrap();
    let a = rows_for(&[0, 1], &[0; 100]);
    // Covered by the reset_all token.
    assert_eq!(
        p.send(&[0, 1], input(&a)),
        Err(EngineError::Borrowed { id: 0 })
    );
    drop(all);
    assert_eq!(p.send(&[], input(&[])), Err(EngineError::EmptySend));
    assert!(p.send(&[1, 1], input(&a)).is_err());
    assert!(p.send(&[0, 500], input(&a)).is_err());
    assert!(matches!(
        p.send(&[0, 1], input(&a[..2])),
        Err(EngineError::ActionLength { .. })
    ));
    let g = p.generation(0).unwrap();
    send(&mut p, &[0, 1], &a);
    assert!(
        p.generation(0).unwrap() > g,
        "the generation moved at the send"
    );
    assert_eq!(
        p.send(&[1, 2], input(&a)),
        Err(EngineError::InFlight { id: 1 })
    );
}

#[test]
fn test_async_tokens_across_recv_in_both_drop_orders() {
    for first_dropped in [true, false] {
        let mut p = asy_with(AutoReset::NextStep, 3, 2, cfg());
        drop(p.reset_all(Reseed::Keep).unwrap());
        let ids: Vec<u32> = (0..4).collect();
        send(&mut p, &ids, &rows_for(&ids, &[0; 100]));
        let a = p.recv().unwrap();
        let a_ids: Vec<u32> = a.ids().collect();
        let a_frame: Vec<Vec<u64>> = a_ids.iter().map(|&id| token_frame(&a, id)).collect();
        let b = p.recv().unwrap();
        let b_ids: Vec<u32> = b.ids().collect();
        // A's ids and rows are untouched by B's recv.
        assert_eq!(a.ids().collect::<Vec<_>>(), a_ids);
        for (k, &id) in a_ids.iter().enumerate() {
            assert_eq!(token_frame(&a, id), a_frame[k]);
        }
        // B can be sent again while A stays protected.
        let (kept, sent) = if first_dropped { (b, a) } else { (a, b) };
        let sent_ids: Vec<u32> = sent.ids().collect();
        let kept_ids: Vec<u32> = kept.ids().collect();
        drop(sent);
        send(&mut p, &sent_ids, &rows_for(&sent_ids, &[1; 100]));
        assert_eq!(
            p.send(
                &kept_ids,
                ActionInput {
                    discrete: &rows_for(&kept_ids, &[1; 100]),
                    continuous: &[]
                }
            ),
            Err(EngineError::Borrowed { id: kept_ids[0] })
        );
        let c = p.recv().unwrap();
        assert_eq!(c.ids().collect::<Vec<_>>().len(), 2);
        drop((kept, c));
        let _ = b_ids;
    }
}

#[test]
fn test_async_forgotten_token_keeps_its_ids_borrowed() {
    let mut p = asy_with(AutoReset::NextStep, 2, 2, cfg());
    drop(p.reset_all(Reseed::Keep).unwrap());
    let ids: Vec<u32> = (0..4).collect();
    send(&mut p, &ids, &rows_for(&ids, &[0; 100]));
    let a = p.recv().unwrap();
    let lost: Vec<u32> = a.ids().collect();
    core::mem::forget(a);
    let b = p.recv().unwrap();
    let other: Vec<u32> = b.ids().collect();
    drop(b);
    assert_eq!(
        p.send(
            &lost,
            ActionInput {
                discrete: &rows_for(&lost, &[1; 100]),
                continuous: &[]
            }
        ),
        Err(EngineError::Borrowed { id: lost[0] })
    );
    // The other ids keep cycling, with buffers to spare.
    for round in 0..20 {
        send(&mut p, &other, &rows_for(&other, &[round + 1; 100]));
        drop(p.recv().unwrap());
    }
}

#[test]
fn test_async_writers_refuse_while_a_token_lives_and_into_pool_hands_back() {
    let mut p = asy(AutoReset::NextStep, 2);
    let all = p.reset_all(Reseed::Keep).unwrap();
    assert!(matches!(
        p.reset_all(Reseed::Keep),
        Err(EngineError::Borrowed { .. })
    ));
    assert!(matches!(
        p.reset(&[5], Reseed::Keep),
        Err(EngineError::Borrowed { .. })
    ));
    let p = match p.into_pool() {
        Err(e) => *e.pool,
        Ok(_) => panic!("a live token blocks into_pool"),
    };
    drop(all);
    let mut p = p;
    let ids: Vec<u32> = (0..10).collect();
    send(&mut p, &ids, &rows_for(&ids, &[0; 100]));
    drop(p.recv().unwrap());
    p.into_pool().map_err(|_| ()).unwrap();
}

#[test]
fn test_async_reset_and_restore_ids_are_delivered_by_recv() {
    let mut p = asy_with(AutoReset::NextStep, 2, 2, cfg());
    drop(p.reset_all(Reseed::Keep).unwrap());
    let record: Record<Counter> = p.snapshot(3).unwrap();
    p.reset(&[3, 4], Reseed::Keep).unwrap();
    let t = p.recv().unwrap();
    let mut got: Vec<u32> = t.ids().collect();
    got.sort_unstable();
    assert_eq!(got, vec![3, 4]);
    drop(t);
    p.restore(&[3], &[record]).unwrap();
    send(&mut p, &[9], &rows_for(&[9], &[0; 100]));
    let t = p.recv().unwrap();
    let mut got: Vec<u32> = t.ids().collect();
    got.sort_unstable();
    assert_eq!(got, vec![3, 9]);
    // A reset id is in flight until received.
    drop(t);
    p.reset(&[7], Reseed::Keep).unwrap();
    assert_eq!(
        p.reset(&[7], Reseed::Keep),
        Err(EngineError::InFlight { id: 7 })
    );
}

#[test]
fn test_async_token_outlives_the_wrapper() {
    let mut p = asy(AutoReset::NextStep, 2);
    let all = p.reset_all(Reseed::Keep).unwrap();
    let before = token_frame(&all, 42);
    drop(p);
    assert_eq!(token_frame(&all, 42), before);
}

#[test]
fn test_async_worker_panic_poisons_and_reset_all_recovers() {
    let mut p = asy_with(
        AutoReset::NextStep,
        4,
        BATCH,
        CountConfig {
            panic_at: 1,
            length: 9,
            ..CountConfig::default()
        },
    );
    drop(p.reset_all(Reseed::Keep).unwrap());
    let ids: Vec<u32> = (0..N).collect();
    send(&mut p, &ids, &rows_for(&ids, &[0; 100]));
    let mut seen = None;
    for _ in 0..20 {
        match p.recv() {
            Ok(t) => drop(t),
            Err(e) => {
                seen = Some(e);
                break;
            }
        }
    }
    assert_eq!(seen, Some(EngineError::Panicked));
    assert!(p.is_poisoned());
    assert_eq!(p.recv().map(|_| ()), Err(EngineError::Poisoned));
    drop(p.reset_all(Reseed::Keep).unwrap());
    assert!(!p.is_poisoned());
}

#[test]
fn test_async_send_batch_resends_a_token() {
    let mut p = asy(AutoReset::NextStep, 2);
    let all = p.reset_all(Reseed::Keep).unwrap();
    let a = all_actions(0);
    p.send_batch(
        all,
        ActionInput {
            discrete: &a,
            continuous: &[],
        },
    )
    .map_err(|e| e.error)
    .unwrap();
    let mut seen = 0;
    while seen < usize::try_from(N).unwrap() {
        seen += p.recv().unwrap().len();
    }
    let mut q = asy(AutoReset::NextStep, 1);
    let foreign = q.reset_all(Reseed::Keep).unwrap();
    let err = p
        .send_batch(
            foreign,
            ActionInput {
                discrete: &a,
                continuous: &[],
            },
        )
        .unwrap_err();
    assert_eq!(err.error, EngineError::WrongPool);
    assert!(err.token.is_some(), "the foreign token is handed back");
}

#[test]
fn test_async_refused_send_batch_hands_the_token_back_unchanged() {
    let mut p = asy(AutoReset::NextStep, 2);
    drop(p.reset_all(Reseed::Keep).unwrap());
    let ids: Vec<u32> = (0..10).collect();
    send(&mut p, &ids, &rows_for(&ids, &[0; 100]));
    let t = p.recv().unwrap();
    let got: Vec<u32> = t.ids().collect();
    let before: Vec<Vec<u64>> = got.iter().map(|&id| token_frame(&t, id)).collect();
    let g = p.generation(got[0]).unwrap();
    // Too few actions: refused, and the token comes back.
    let err = p
        .send_batch(
            t,
            ActionInput {
                discrete: &[0, 0],
                continuous: &[],
            },
        )
        .unwrap_err();
    assert!(matches!(err.error, EngineError::ActionLength { .. }));
    let t = err.token.unwrap();
    assert_eq!(t.ids().collect::<Vec<_>>(), got);
    for (k, &id) in got.iter().enumerate() {
        assert_eq!(token_frame(&t, id), before[k], "unchanged");
    }
    assert_eq!(p.generation(got[0]).unwrap(), g, "nothing moved");
    // Its ids are still covered, and the token can be resent.
    assert_eq!(
        p.send(&got, input(&rows_for(&got, &[1; 100]))),
        Err(EngineError::Borrowed { id: got[0] })
    );
    p.send_batch(t, input(&rows_for(&got, &[1; 100])))
        .map_err(|e| e.error)
        .unwrap();
    drop(p.recv().unwrap());
}

#[test]
fn test_async_waiter_and_recv_timeout() {
    let mut p = asy(AutoReset::NextStep, 2);
    drop(p.reset_all(Reseed::Keep).unwrap());
    let w = p.waiter();
    let ids: Vec<u32> = (0..10).collect();
    send(&mut p, &ids, &rows_for(&ids, &[0; 100]));
    let t = loop {
        if let Some(t) = p.recv_timeout(Duration::from_millis(50)).unwrap() {
            break t;
        }
    };
    assert_eq!(t.len(), BATCH);
    assert!(!w.wait_ready(Duration::from_millis(1)));
}

#[test]
fn test_async_batch_larger_than_the_pool_is_refused_and_handed_back() {
    let r = Pool::with_masks(cfg(), options(AutoReset::NextStep))
        .unwrap()
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(1).unwrap(),
            NonZeroUsize::new(101).unwrap(),
        ));
    let Err(err) = r else {
        panic!("a batch larger than the pool");
    };
    assert!(matches!(err.error, EngineError::Options { .. }));
    let mut pool = *err.pool.unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
}

#[cfg(feature = "debug-checks")]
#[test]
fn test_async_debug_checks_refuse_what_sync_refuses_and_mutate_nothing() {
    let mut p = asy(AutoReset::NextStep, 2);
    drop(p.reset_all(Reseed::Keep).unwrap());
    let mut reference = sync_pool(AutoReset::NextStep);
    reference.reset_all(Reseed::Keep).unwrap();
    // Find an action the Sync step refuses.
    let mut bad = all_actions(0);
    let mut refused = None;
    for k in 0..bad.len() {
        let mut probe = bad.clone();
        probe[k] = 2;
        if let Err(e) = reference.step(ActionInput {
            discrete: &probe,
            continuous: &[],
        }) && matches!(e, EngineError::IllegalAction { .. })
        {
            bad = probe;
            refused = Some((k / 2, e));
            break;
        }
    }
    let (id, sync_err) = refused.unwrap();
    let id = u32::try_from(id).unwrap();
    let g = p.generation(id).unwrap();
    let i = usize::try_from(id).unwrap();
    let err = p
        .send(
            &[id],
            ActionInput {
                discrete: &bad[i * 2..i * 2 + 2],
                continuous: &[],
            },
        )
        .unwrap_err();
    assert_eq!(err, sync_err);
    assert_eq!(p.generation(id).unwrap(), g, "nothing changed");
}
