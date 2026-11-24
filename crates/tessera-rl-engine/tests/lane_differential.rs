// SPDX-License-Identifier: MIT OR Apache-2.0

//! The engine's lane path against the scalar step (ADR-0071 decision 6):
//! `Walk` (a `LaneStep` environment) at every level this CPU has must give,
//! after every step, the outputs, headers and states `Plain` (the same
//! environment without a lane step) gives, byte for byte, plus equal
//! snapshot, restore, fork and persisted bytes. Sync (Inline and with
//! workers) and Pipelined take the lane path; Async is in
//! `lane_async.rs`.

#![cfg(not(loom))]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod lanes;

use core::num::{NonZeroU32, NonZeroUsize};

use lanes::{Plain, Walk, WalkConfig};
use tessera_rl_core::{Bool8, RecordLayout};
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, NoMasks, Outputs, PipelinedOptions, PipelinedPool,
    PoolOptions, Record, Reseed, SyncOptions,
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
}

fn options(case: Case, level: Option<Level>) -> PoolOptions {
    PoolOptions {
        chunk_size: NonZeroU32::new(case.chunk.min(case.n.max(1)).max(1)).unwrap(),
        max_episode_steps: case.limit.and_then(NonZeroU32::new),
        auto_reset: case.auto_reset,
        simd_level: level,
        ..PoolOptions::new(NonZeroU32::new(case.n).unwrap(), Seed::new(11))
    }
}

macro_rules! pool {
    ($env:ty, $case:expr, $level:expr) => {{
        let case: Case = $case;
        let p = EnvPool::<$env>::new(WalkConfig {}, options(case, $level)).unwrap();
        if case.workers == 0 {
            p
        } else {
            p.with_workers(SyncOptions::new(NonZeroUsize::new(case.workers).unwrap()))
                .unwrap()
        }
    }};
}

/// Instance `i`'s action at step `t`, in 0..3, mixed so that neighbouring
/// lanes diverge.
fn action(t: u32, i: u32) -> i32 {
    let h = (u64::from(i).wrapping_mul(0x9e37_79b9) ^ u64::from(t).wrapping_mul(0x85eb_ca6b))
        .wrapping_mul(0xc2b2_ae35);
    i32::try_from((h >> 29) % 3).unwrap()
}

fn actions(t: u32, ids: impl Iterator<Item = u32>) -> Vec<i32> {
    ids.map(|i| action(t, i)).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn flags(v: &[Bool8]) -> Vec<bool> {
    v.iter().map(|b| b.get()).collect()
}

/// Every output a step writes, as comparable values.
fn outputs(o: &Outputs, same_step: bool) -> Vec<Vec<u64>> {
    let w = |v: Vec<u32>| v.into_iter().map(u64::from).collect::<Vec<_>>();
    let f = |v: Vec<bool>| v.into_iter().map(u64::from).collect::<Vec<_>>();
    let mut all = vec![
        w(bits(o.rewards())),
        f(flags(o.agent_terminated())),
        f(flags(o.agent_truncated())),
        f(flags(o.episode_terminated())),
        f(flags(o.episode_truncated())),
        w(bits(o.episode_return())),
        w(bits(o.episode_length())),
        w(bits(o.episode_illegal_actions())),
        w(bits(o.episode_invalid_agent_events())),
        f(flags(o.alive())),
        f(flags(o.acting())),
        f(flags(o.pending_reset())),
        w(o.generations().to_vec()),
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

macro_rules! records {
    ($p:expr, $n:expr) => {
        (0..$n)
            .map(|i| $p.snapshot(i).unwrap().to_bytes())
            .collect::<Vec<Vec<u8>>>()
    };
}

/// Runs `case` on `Walk` at `level` and on `Plain`, comparing after every
/// step; snapshot, fork, restore and persisted bytes every 97 steps.
fn differential(case: Case, level: Level) {
    let mut lane = pool!(Walk, case, Some(level));
    let mut plain = pool!(Plain, case, None);
    let same = case.auto_reset == AutoReset::SameStep;
    let a = outputs(lane.reset_all(Reseed::Keep).unwrap(), same);
    let b = outputs(plain.reset_all(Reseed::Keep).unwrap(), same);
    assert_eq!(a, b, "{case:?} {level}: reset");
    for t in 0..case.steps {
        let acts = actions(t, 0..case.n);
        let input = ActionInput {
            discrete: &acts,
            continuous: &[],
        };
        let a = outputs(lane.step(input).unwrap(), same);
        let b = outputs(plain.step(input).unwrap(), same);
        assert_eq!(a, b, "{case:?} {level}: outputs at step {t}");
        if t % 97 == 96 {
            assert_eq!(
                records!(lane, case.n),
                records!(plain, case.n),
                "{case:?} {level}: records at {t}"
            );
            let ids: Vec<u32> = (0..case.n).collect();
            assert_eq!(
                lane.encode(&ids).unwrap(),
                plain.encode(&ids).unwrap(),
                "{case:?} {level}: persisted at {t}"
            );
            // Fork the first instance over the last half, and restore the
            // last instance's record into the first.
            let dst: Vec<u32> = (case.n / 2..case.n).collect();
            lane.fork(0, &dst).unwrap();
            plain.fork(0, &dst).unwrap();
            let last = case.n - 1;
            let ra = lane.snapshot(last).unwrap();
            let rb = plain.snapshot(last).unwrap();
            lane.restore(0, &ra).unwrap();
            plain.restore(0, &rb).unwrap();
            assert_eq!(
                records!(lane, case.n),
                records!(plain, case.n),
                "{case:?} {level}: fork at {t}"
            );
        }
    }
    assert_eq!(
        records!(lane, case.n),
        records!(plain, case.n),
        "{case:?} {level}: final records"
    );
}

fn levels() -> Vec<Level> {
    Resolved::available().map(Resolved::level).collect()
}

fn matrix(ns: &[u32], chunks: &[u32], workers: &[usize], steps: u32) {
    for level in levels() {
        for &n in ns {
            for &chunk in chunks {
                for &w in workers {
                    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
                        for limit in [None, Some(7)] {
                            differential(
                                Case {
                                    n,
                                    chunk,
                                    workers: w,
                                    auto_reset,
                                    limit,
                                    steps,
                                },
                                level,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_lane_path_small_pools_every_level_match_scalar() {
    matrix(
        &[1, 7, 8, 9, 63, 64, 65],
        &[1, 8, 12, 64, 1024],
        &[0, 1, 2],
        300,
    );
}

#[test]
fn test_lane_path_large_pools_every_level_match_scalar() {
    matrix(&[1000, 8192], &[8, 64, 1024], &[0, 2], 120);
}

/// The long runs of ADR-0071 decision 6 (10k steps); run by the gate
/// with `--ignored`.
#[test]
#[ignore = "long: the 10k-step differential, run explicitly"]
fn test_lane_path_long_runs_every_level_match_scalar() {
    matrix(
        &[1, 7, 8, 9, 63, 64, 65, 1000, 8192],
        &[1, 8, 64, 1024],
        &[0, 1, 2],
        10_000,
    );
}

/// Pipelined takes the lane path: its records after every cycle equal a
/// scalar Sync pool's.
#[test]
fn test_lane_path_pipelined_every_level_matches_scalar() {
    for level in levels() {
        for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
            let case = Case {
                n: 64,
                chunk: 16,
                workers: 0,
                auto_reset,
                limit: Some(9),
                steps: 400,
            };
            let mut piped: PipelinedPool<Walk, NoMasks> =
                EnvPool::<Walk>::new(WalkConfig {}, options(case, Some(level)))
                    .unwrap()
                    .pipelined(PipelinedOptions::new(NonZeroUsize::new(2).unwrap()))
                    .unwrap();
            let mut plain = pool!(Plain, case, None);
            let template = EnvPool::<Walk>::new(WalkConfig {}, options(case, None))
                .unwrap()
                .snapshot(0)
                .unwrap();
            drop(piped.reset_all(Reseed::Keep).unwrap());
            plain.reset_all(Reseed::Keep).unwrap();
            let halves = piped.halves();
            for t in 0..case.steps {
                for (h, range) in halves.iter().enumerate() {
                    let token = piped.idle_token(u32::try_from(h).unwrap()).unwrap();
                    let acts = actions(t, range.clone());
                    piped
                        .send(
                            token,
                            ActionInput {
                                discrete: &acts,
                                continuous: &[],
                            },
                        )
                        .map_err(|e| e.error)
                        .unwrap();
                }
                for _ in 0..2 {
                    drop(piped.recv().unwrap());
                }
                let acts = actions(t, 0..case.n);
                plain
                    .step(ActionInput {
                        discrete: &acts,
                        continuous: &[],
                    })
                    .unwrap();
                let mut got = vec![template; 64];
                piped.snapshot_all(&mut got).unwrap();
                let got: Vec<Vec<u8>> = got.iter().map(Record::to_bytes).collect();
                assert_eq!(
                    got,
                    records!(plain, case.n),
                    "pipelined {level} {auto_reset:?} at {t}"
                );
            }
        }
    }
}

/// A level this CPU lacks is an error at construction, never a fallback.
#[test]
fn test_lane_path_unavailable_level_is_a_construction_error() {
    let missing = [Level::Avx2, Level::Sse42, Level::Neon]
        .into_iter()
        .find(|l| Resolved::force(*l).is_err());
    let Some(level) = missing else { return };
    let case = Case {
        n: 8,
        chunk: 8,
        workers: 0,
        auto_reset: AutoReset::NextStep,
        limit: None,
        steps: 0,
    };
    let err = EnvPool::<Walk>::new(WalkConfig {}, options(case, Some(level))).err();
    assert_eq!(
        err,
        Some(tessera_rl_engine::EngineError::SimdLevel(
            tessera_rl_simd::LevelError::Unavailable(level)
        ))
    );
    // The same for an environment without a lane step.
    assert!(EnvPool::<Plain>::new(WalkConfig {}, options(case, Some(level))).is_err());
}

/// A group with a dead agent (restored from a record) takes the scalar
/// step: both pools step it alike, and the dead agent's episode keeps its
/// record bytes equal to the scalar pool's.
#[test]
fn test_lane_path_dead_agent_group_matches_scalar() {
    for level in levels() {
        let case = Case {
            n: 16,
            chunk: 16,
            workers: 0,
            auto_reset: AutoReset::NextStep,
            limit: None,
            steps: 50,
        };
        let mut lane = pool!(Walk, case, Some(level));
        let mut plain = pool!(Plain, case, None);
        lane.reset_all(Reseed::Keep).unwrap();
        plain.reset_all(Reseed::Keep).unwrap();
        let mut a = lane.snapshot(3).unwrap();
        let mut b = plain.snapshot(3).unwrap();
        for row in a.header_mut().agents_mut() {
            row.alive = Bool8::FALSE;
            row.acting = Bool8::FALSE;
        }
        for row in b.header_mut().agents_mut() {
            row.alive = Bool8::FALSE;
            row.acting = Bool8::FALSE;
        }
        lane.restore(3, &a).unwrap();
        plain.restore(3, &b).unwrap();
        for t in 0..case.steps {
            let acts = actions(t, 0..case.n);
            let input = ActionInput {
                discrete: &acts,
                continuous: &[],
            };
            let x = lane.step(input).map(|o| outputs(o, false));
            let y = plain.step(input).map(|o| outputs(o, false));
            assert_eq!(x, y, "{level}: dead agent at step {t}");
            assert_eq!(
                records!(lane, case.n),
                records!(plain, case.n),
                "{level}: dead agent records at {t}"
            );
        }
    }
}
