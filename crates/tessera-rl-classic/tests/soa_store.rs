// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Struct-of-Arrays state store (ADR-0090, head 1) against the
//! Array-of-Structs store: `SoaCartPole` (CartPole's kernels on a lane-wise
//! state, one column per field) must give, after every call, the outputs,
//! records, `state(id)`, state hashes, persisted files, forks and replay
//! files that `CartPole` (the same kernels on its own Array-of-Structs
//! state) gives, byte for byte, in Inline, Sync and Pipelined modes. Records cross
//! the two stores in both directions. Async refuses a lane-wise pool, and a
//! lane-wise pool's `chunk_size` must be a multiple of 8.

#![cfg(not(loom))]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

#[path = "../benches/support/soa_cartpole.rs"]
mod soa_cartpole;

use soa_cartpole::{SoaCartPole, SoaCartPoleState};
use tessera_rl_classic::cartpole::{CartPole, CartPoleConfig, CartPoleState};
use tessera_rl_core::Bool8;
use tessera_rl_engine::replay::{Recorder, Replay, ReplayWriterOptions};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AutoReset, DecodeLimits, DecodedRecord, EngineError, EnvPool,
    Outputs, PipelinedOptions, PoolOptions, Reseed, SyncOptions, Transforms,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_simd::{Level, Resolved};

#[derive(Debug, Clone, Copy)]
struct Case {
    n: u32,
    chunk: u32,
    workers: usize,
    auto_reset: AutoReset,
    limit: Option<u32>,
    steps: u32,
    /// The lane level of the lane-wise pool (ADR-0090 decision 4); `None`
    /// takes the best one.
    level: Option<Level>,
}

fn options(case: Case) -> PoolOptions {
    PoolOptions {
        chunk_size: NonZeroU32::new(case.chunk).unwrap(),
        max_episode_steps: case.limit.and_then(NonZeroU32::new),
        auto_reset: case.auto_reset,
        simd_level: case.level,
        ..PoolOptions::new(NonZeroU32::new(case.n).unwrap(), Seed::new(29))
    }
}

macro_rules! pool {
    ($env:ty, $case:expr) => {{
        let case: Case = $case;
        let p = EnvPool::<$env>::new(CartPoleConfig {}, options(case)).unwrap();
        if case.workers == 0 {
            p
        } else {
            p.with_workers(SyncOptions::new(NonZeroUsize::new(case.workers).unwrap()))
                .unwrap()
        }
    }};
}

macro_rules! records {
    ($p:expr, $n:expr) => {
        (0..$n)
            .map(|i| $p.snapshot(i).unwrap().to_bytes())
            .collect::<Vec<Vec<u8>>>()
    };
}

fn action(t: u32, i: u32) -> i32 {
    let h = (u64::from(i).wrapping_mul(0x9e37_79b9) ^ u64::from(t).wrapping_mul(0x85eb_ca6b))
        .wrapping_mul(0xc2b2_ae35);
    i32::try_from((h >> 31) & 1).unwrap()
}

fn flags(v: &[Bool8]) -> Vec<u64> {
    v.iter().map(|b| u64::from(b.get())).collect()
}

fn bits(v: &[f32]) -> Vec<u64> {
    v.iter().map(|x| u64::from(x.to_bits())).collect()
}

fn outputs(o: &Outputs, same_step: bool) -> Vec<Vec<u64>> {
    let mut all = vec![
        bits(o.rewards()),
        flags(o.agent_terminated()),
        flags(o.agent_truncated()),
        flags(o.episode_terminated()),
        flags(o.episode_truncated()),
        bits(o.episode_return()),
        bits(o.episode_length()),
        flags(o.alive()),
        flags(o.acting()),
        flags(o.pending_reset()),
        o.observation(0)
            .unwrap()
            .iter()
            .map(|&b| u64::from(b))
            .collect(),
    ];
    if same_step {
        all.push(
            o.terminal_observation(0)
                .unwrap()
                .iter()
                .map(|&b| u64::from(b))
                .collect(),
        );
    }
    all
}

/// The same states, both ways: `state(id)` gathers the columns.
fn states_match(soa: &EnvPool<SoaCartPole>, plain: &EnvPool<CartPole>, n: u32) {
    for i in 0..n {
        let a = soa.state(i).unwrap();
        let b = plain.state(i).unwrap();
        assert_eq!(
            a.rows().s.map(f64::to_bits),
            b.s.map(f64::to_bits),
            "state {i}"
        );
        assert_eq!(
            soa.state_hash(i).unwrap(),
            plain.state_hash(i).unwrap(),
            "hash {i}"
        );
    }
}

fn differential(case: Case) {
    let mut soa = pool!(SoaCartPole, case);
    let mut plain = pool!(CartPole, case);
    let same = case.auto_reset == AutoReset::SameStep;
    assert_eq!(
        outputs(soa.reset_all(Reseed::Keep).unwrap(), same),
        outputs(plain.reset_all(Reseed::Keep).unwrap(), same),
        "{case:?}: reset"
    );
    states_match(&soa, &plain, case.n);
    let ids: Vec<u32> = (0..case.n).collect();
    for t in 0..case.steps {
        let acts: Vec<i32> = (0..case.n).map(|i| action(t, i)).collect();
        let input = ActionInput {
            discrete: &acts,
            continuous: &[],
        };
        let x = outputs(soa.step(input).unwrap(), same);
        let y = outputs(plain.step(input).unwrap(), same);
        assert_eq!(x, y, "{case:?}: outputs at step {t}");
        if t % 41 == 40 {
            assert_eq!(
                records!(soa, case.n),
                records!(plain, case.n),
                "{case:?}: {t}"
            );
            states_match(&soa, &plain, case.n);
            assert_eq!(
                soa.encode(&ids).unwrap(),
                plain.encode(&ids).unwrap(),
                "{case:?}: persisted at {t}"
            );
            // Fork and fork_branched scatter a gathered record.
            let dst: Vec<u32> = (case.n / 2..case.n).collect();
            soa.fork(0, &dst).unwrap();
            plain.fork(0, &dst).unwrap();
            if case.n > 2 {
                soa.fork_branched(1, &[2], None).unwrap();
                plain.fork_branched(1, &[2], None).unwrap();
            }
            // A partial reset mixes pending and active instances in groups.
            let partial: Vec<u32> = (0..case.n).step_by(5).collect();
            soa.reset(&partial, Reseed::Keep).unwrap();
            plain.reset(&partial, Reseed::Keep).unwrap();
        }
        if t % 67 == 66 {
            // Records cross the stores through their bytes, both ways.
            let from_plain = plain.encode(&ids).unwrap();
            let from_soa = soa.encode(&ids).unwrap();
            let into_soa = soa.decode(&from_plain, DecodeLimits::default()).unwrap();
            let into_plain = plain.decode(&from_soa, DecodeLimits::default()).unwrap();
            soa.import_all(&into_soa).unwrap();
            plain.import_all(&into_plain).unwrap();
            // And one restore from a typed record.
            let r = plain.snapshot(0).unwrap();
            let rec =
                DecodedRecord::from_parts(*r.header(), SoaCartPoleState::of(*r.state())).unwrap();
            soa.import(&[0], &[rec]).unwrap();
        }
    }
    assert_eq!(
        records!(soa, case.n),
        records!(plain, case.n),
        "{case:?}: final"
    );
    states_match(&soa, &plain, case.n);
}

fn matrix(ns: &[u32], chunks: &[u32], workers: &[usize], steps: u32) {
    for &n in ns {
        for &chunk in chunks {
            for &workers in workers {
                for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
                    for limit in [None, Some(40)] {
                        // Every level this CPU has: each lane kernel must
                        // match the reference (ADR-0090 4.8).
                        for level in Resolved::available().map(Resolved::level) {
                            differential(Case {
                                n,
                                chunk,
                                workers,
                                auto_reset,
                                limit,
                                steps,
                                level: Some(level),
                            });
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_soa_store_small_matches_aos() {
    matrix(
        &[1, 7, 8, 9, 63, 64, 65],
        &[8, 16, 64, 1024],
        &[0, 1, 2],
        300,
    );
}

#[test]
fn test_soa_store_large_matches_aos() {
    matrix(&[1000, 8192], &[8, 64, 1024], &[0, 2], 120);
}

#[test]
fn test_soa_store_pipelined_matches_aos() {
    const N: u32 = 64;
    macro_rules! run {
        ($env:ty) => {{
            let case = Case {
                n: N,
                chunk: 8,
                workers: 0,
                auto_reset: AutoReset::SameStep,
                limit: Some(30),
                steps: 0,
                level: None,
            };
            let mut p = EnvPool::<$env>::new(CartPoleConfig {}, options(case))
                .unwrap()
                .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
                .unwrap();
            let halves = p.halves();
            let [mut ta, mut tb] = p.reset_all(Reseed::Keep).unwrap();
            for t in 0..200 {
                for token in [ta, tb] {
                    let h = usize::try_from(token.half()).unwrap();
                    let a: Vec<i32> = halves[h].clone().map(|i| action(t, i)).collect();
                    assert!(
                        p.send(
                            token,
                            ActionInput {
                                discrete: &a,
                                continuous: &[],
                            },
                        )
                        .is_ok()
                    );
                }
                ta = p.recv().unwrap();
                tb = p.recv().unwrap();
            }
            drop((ta, tb));
            let mut out = vec![
                EnvPool::<$env>::new(CartPoleConfig {}, options(case))
                    .unwrap()
                    .snapshot(0)
                    .unwrap();
                N as usize
            ];
            p.snapshot_all(&mut out).unwrap();
            out.iter().map(|r| r.to_bytes()).collect::<Vec<_>>()
        }};
    }
    assert_eq!(run!(SoaCartPole), run!(CartPole));
}

#[test]
fn test_soa_store_replay_files_are_identical_and_cross_replay() {
    macro_rules! record {
        ($env:ty) => {{
            let case = Case {
                n: 16,
                chunk: 8,
                workers: 2,
                auto_reset: AutoReset::SameStep,
                limit: Some(25),
                steps: 0,
                level: None,
            };
            let mut p = pool!($env, case);
            let options = ReplayWriterOptions {
                keyframe_every: 16,
                checkpoint_every: 4,
                outputs_hash: false,
            };
            let mut rec = Recorder::new(&mut p, Vec::new(), options).unwrap();
            rec.reset_all(Reseed::Base(Seed::new(5))).unwrap();
            for t in 0..60 {
                let a: Vec<i32> = (0..16).map(|i| action(t, i)).collect();
                rec.step(ActionInput {
                    discrete: &a,
                    continuous: &[],
                })
                .unwrap();
                if t == 30 {
                    rec.fork_branched(0, &[1, 2], None).unwrap();
                    rec.reset(&[3], Reseed::List(&[Seed::new(8)])).unwrap();
                }
            }
            let bytes = rec.finish().unwrap();
            let hashes: Vec<u64> = (0..16).map(|i| p.state_hash(i).unwrap().get()).collect();
            (bytes, hashes, case)
        }};
    }
    let (soa_file, soa_hashes, case) = record!(SoaCartPole);
    let (plain_file, plain_hashes, _) = record!(CartPole);
    assert_eq!(soa_file, plain_file);
    assert_eq!(soa_hashes, plain_hashes);
    // Each store replays the file with no divergence.
    let mut soa = pool!(SoaCartPole, case);
    Replay::open(&plain_file, DecodeLimits::default())
        .unwrap()
        .run(&mut soa)
        .unwrap();
    let mut plain = pool!(CartPole, case);
    Replay::open(&soa_file, DecodeLimits::default())
        .unwrap()
        .run(&mut plain)
        .unwrap();
    let a: Vec<u64> = (0..16).map(|i| soa.state_hash(i).unwrap().get()).collect();
    let b: Vec<u64> = (0..16)
        .map(|i| plain.state_hash(i).unwrap().get())
        .collect();
    assert_eq!(a, plain_hashes);
    assert_eq!(b, plain_hashes);
}

#[test]
fn test_soa_store_async_is_refused() {
    let case = Case {
        n: 16,
        chunk: 8,
        workers: 0,
        auto_reset: AutoReset::SameStep,
        limit: None,
        steps: 0,
        level: None,
    };
    let p = pool!(SoaCartPole, case);
    let err = p
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(8).unwrap(),
        ))
        .err()
        .unwrap();
    assert!(matches!(err.error, EngineError::LaneEnvUnsupported));
    assert!(err.pool.is_some());
}

#[test]
fn test_soa_store_chunk_size_must_be_a_multiple_of_8() {
    for chunk in [1, 4, 12, 63] {
        let options = PoolOptions {
            chunk_size: NonZeroU32::new(chunk).unwrap(),
            ..PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(1))
        };
        assert!(
            matches!(
                EnvPool::<SoaCartPole>::new(CartPoleConfig {}, options.clone()),
                Err(EngineError::Options {
                    option: "chunk_size",
                    ..
                })
            ),
            "chunk {chunk}"
        );
        // The Array-of-Structs store accepts it.
        EnvPool::<CartPole>::new(CartPoleConfig {}, options).unwrap();
    }
    for chunk in [8, 16, 1024] {
        let options = PoolOptions {
            chunk_size: NonZeroU32::new(chunk).unwrap(),
            ..PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(1))
        };
        EnvPool::<SoaCartPole>::new(CartPoleConfig {}, options).unwrap();
    }
}

#[test]
fn test_soa_store_state_is_gathered_by_value() {
    let case = Case {
        n: 9,
        chunk: 8,
        workers: 0,
        auto_reset: AutoReset::SameStep,
        limit: None,
        steps: 0,
        level: None,
    };
    let mut soa = pool!(SoaCartPole, case);
    let _ = soa.reset_all(Reseed::Keep).unwrap();
    let s = CartPoleState {
        s: [0.25, -0.5, 0.01, 0.125],
    };
    let r = soa.snapshot(8).unwrap();
    let rec = DecodedRecord::from_parts(*r.header(), SoaCartPoleState::of(s)).unwrap();
    soa.import(&[8], &[rec]).unwrap();
    assert_eq!(soa.state(8).unwrap(), SoaCartPoleState::of(s));
    assert_eq!(soa.state(9), None);
}

#[test]
fn test_soa_store_heuristic_slots_are_refused() {
    // The heuristic pass reads a chunk's state rows, which a lane-wise
    // state does not have; the row store accepts the same slots.
    let options = PoolOptions::new(NonZeroU32::new(16).unwrap(), Seed::new(3));
    let soa = EnvPool::<SoaCartPole>::new(CartPoleConfig {}, options.clone()).unwrap();
    assert!(matches!(
        soa.with_heuristic_slots(&[0]),
        Err(EngineError::PolicyMap { .. })
    ));
    let plain = EnvPool::<CartPole>::new(CartPoleConfig {}, options).unwrap();
    assert!(plain.with_heuristic_slots(&[0]).is_ok());
}

#[test]
fn test_soa_store_steps_full_groups_through_the_lane_kernels() {
    // 16 instances, chunk 8: two full groups per step take step_lanes.
    let before = soa_cartpole::LANE_STEPS.load(core::sync::atomic::Ordering::Relaxed);
    let options = PoolOptions {
        chunk_size: NonZeroU32::new(8).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(16).unwrap(), Seed::new(5))
    };
    let mut soa = EnvPool::<SoaCartPole>::new(CartPoleConfig {}, options).unwrap();
    let _ = soa.reset_all(Reseed::Keep).unwrap();
    let acts = vec![1i32; 16];
    let _ = soa
        .step(ActionInput {
            discrete: &acts,
            continuous: &[],
        })
        .unwrap();
    let after = soa_cartpole::LANE_STEPS.load(core::sync::atomic::Ordering::Relaxed);
    assert!(
        after >= before + 2,
        "lane groups stepped: {}",
        after - before
    );
}

#[test]
fn test_soa_store_with_transforms_matches_aos() {
    // Observation transforms take the per-instance observe (ADR-0090 4.4):
    // the lane-wise pool with a normalization and a frame stack must match
    // CartPole with the same transforms, byte for byte.
    use tessera_rl_core::Environment;
    use tessera_rl_core::space::Space;
    let leaf = match CartPole::observation_space(&CartPoleConfig {}) {
        Ok(Space::Dict(entries)) => entries.first().map(|e| e.0.clone()).unwrap_or_default(),
        _ => String::new(),
    };
    let transforms = || {
        Transforms::new()
            .normalize(leaf.clone(), 5.0)
            .frame_stack(leaf.clone(), 4)
    };
    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
        let case = Case {
            n: 32,
            chunk: 8,
            workers: 0,
            auto_reset,
            limit: Some(30),
            steps: 0,
            level: None,
        };
        let mut soa = pool!(SoaCartPole, case)
            .with_transforms(transforms())
            .unwrap();
        let mut plain = pool!(CartPole, case).with_transforms(transforms()).unwrap();
        let same = auto_reset == AutoReset::SameStep;
        assert_eq!(
            outputs(soa.reset_all(Reseed::Keep).unwrap(), same),
            outputs(plain.reset_all(Reseed::Keep).unwrap(), same)
        );
        for t in 0..120 {
            let acts: Vec<i32> = (0..32).map(|i| action(t, i)).collect();
            let input = ActionInput {
                discrete: &acts,
                continuous: &[],
            };
            assert_eq!(
                outputs(soa.step(input).unwrap(), same),
                outputs(plain.step(input).unwrap(), same),
                "{auto_reset:?} step {t}"
            );
        }
    }
}
