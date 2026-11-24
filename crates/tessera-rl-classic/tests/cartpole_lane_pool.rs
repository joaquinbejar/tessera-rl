// SPDX-License-Identifier: MIT OR Apache-2.0

//! CartPole's lane path in the pool against its scalar step (ADR-0071
//! decision 6): `CartPole` at every level this CPU has must give, after every
//! step, the outputs and records `ScalarCartPole` (the same kernels without
//! a lane step) gives, byte for byte, plus equal fork, restore and persisted
//! bytes. States outside the kernel's fast domain are restored into the
//! pools so that the kernel's scalar fallback lanes run too.

#![cfg(not(loom))]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

#[path = "../benches/support/scalar_cartpole.rs"]
mod scalar_cartpole;

use scalar_cartpole::ScalarCartPole;
use tessera_rl_classic::cartpole::{CartPole, CartPoleConfig, CartPoleState};
use tessera_rl_core::Bool8;
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EnvPool, Outputs, PoolOptions, Reseed, SyncOptions,
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
        chunk_size: NonZeroU32::new(case.chunk.min(case.n)).unwrap(),
        max_episode_steps: case.limit.and_then(NonZeroU32::new),
        auto_reset: case.auto_reset,
        simd_level: level,
        ..PoolOptions::new(NonZeroU32::new(case.n).unwrap(), Seed::new(23))
    }
}

macro_rules! pool {
    ($env:ty, $case:expr, $level:expr) => {{
        let case: Case = $case;
        let p = EnvPool::<$env>::new(CartPoleConfig {}, options(case, $level)).unwrap();
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

/// States outside the kernel's fast domain: large magnitudes and angles off
/// the polynomial path.
const WILD: [[f64; 4]; 4] = [
    [1.0e12, -3.0, 0.1, 2.0],
    [0.0, 5.0e10, -0.2, 0.0],
    [0.5, 0.5, 7.0, -1.0e11],
    [-2.0, 1.0, -40.0, 3.0],
];

fn differential(case: Case, level: Level) {
    let mut lane = pool!(CartPole, case, Some(level));
    let mut plain = pool!(ScalarCartPole, case, None);
    let same = case.auto_reset == AutoReset::SameStep;
    assert_eq!(
        outputs(lane.reset_all(Reseed::Keep).unwrap(), same),
        outputs(plain.reset_all(Reseed::Keep).unwrap(), same),
        "{case:?} {level}: reset"
    );
    for t in 0..case.steps {
        if t % 53 == 5 {
            // Wild states into every third instance, the same in both pools.
            for (k, i) in (0..case.n).step_by(3).enumerate() {
                let s = WILD.get(k % WILD.len()).copied().unwrap();
                let a = DecodedRecord::from_parts(
                    *lane.snapshot(i).unwrap().header_mut(),
                    CartPoleState { s },
                )
                .unwrap();
                let b = DecodedRecord::from_parts(
                    *plain.snapshot(i).unwrap().header_mut(),
                    CartPoleState { s },
                )
                .unwrap();
                lane.import(&[i], &[a]).unwrap();
                plain.import(&[i], &[b]).unwrap();
            }
        }
        let acts: Vec<i32> = (0..case.n).map(|i| action(t, i)).collect();
        let input = ActionInput {
            discrete: &acts,
            continuous: &[],
        };
        let x = outputs(lane.step(input).unwrap(), same);
        let y = outputs(plain.step(input).unwrap(), same);
        assert_eq!(x, y, "{case:?} {level}: outputs at step {t}");
        if t % 97 == 96 {
            assert_eq!(
                records!(lane, case.n),
                records!(plain, case.n),
                "{case:?} {level}: {t}"
            );
            let ids: Vec<u32> = (0..case.n).collect();
            assert_eq!(
                lane.encode(&ids).unwrap(),
                plain.encode(&ids).unwrap(),
                "{case:?} {level}: persisted at {t}"
            );
            let dst: Vec<u32> = (case.n / 2..case.n).collect();
            lane.fork(0, &dst).unwrap();
            plain.fork(0, &dst).unwrap();
        }
    }
    assert_eq!(
        records!(lane, case.n),
        records!(plain, case.n),
        "{case:?} {level}: final"
    );
}

fn levels() -> Vec<Level> {
    Resolved::available().map(Resolved::level).collect()
}

fn matrix(ns: &[u32], chunks: &[u32], workers: &[usize], steps: u32) {
    for level in levels() {
        for &n in ns {
            for &chunk in chunks {
                for &workers in workers {
                    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
                        for limit in [None, Some(40)] {
                            differential(
                                Case {
                                    n,
                                    chunk,
                                    workers,
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
fn test_cartpole_lane_pool_small_every_level_matches_scalar() {
    matrix(
        &[1, 7, 8, 9, 63, 64, 65],
        &[1, 8, 12, 64, 1024],
        &[0, 1, 2],
        400,
    );
}

#[test]
fn test_cartpole_lane_pool_large_every_level_matches_scalar() {
    matrix(&[1000, 8192], &[8, 64, 1024], &[0, 2], 150);
}

/// The 10k-step runs (ADR-0071 decision 6), run by the gate with `--ignored`.
#[test]
#[ignore = "long: the 10k-step differential, run explicitly"]
fn test_cartpole_lane_pool_long_every_level_matches_scalar() {
    matrix(
        &[1, 7, 8, 9, 63, 64, 65, 1000, 8192],
        &[1, 8, 64, 1024],
        &[0, 1, 2],
        10_000,
    );
}
