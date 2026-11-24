// SPDX-License-Identifier: MIT OR Apache-2.0

//! The 1.2x stop of ADR-0071 decision 7: CartPole's full pool step with the
//! lane path (`CartPole`, at the widest level, or the level
//! `TESSERA_RL_SIMD` names) against the per-instance step of the same
//! kernels without a lane step (`ScalarCartPole`, the erratum's baseline).
//! Same-step auto-reset, steady state (300 settling steps), xorshift actions;
//! N = 8192 (binding) and 1024 (reported), Inline and two pinned workers.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

#[path = "support/scalar_cartpole.rs"]
mod scalar_cartpole;

use core::num::{NonZeroU32, NonZeroUsize};
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use scalar_cartpole::ScalarCartPole;
use tessera_rl_classic::cartpole::{CartPole, CartPoleConfig};
use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, PoolOptions, Reseed, SyncOptions};
use tessera_rl_num::identity::Seed;
use tessera_rl_simd::Resolved;

/// 256 action rows of `n` instances, from a fixed xorshift.
fn actions(n: usize) -> Vec<Vec<i32>> {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..256)
        .map(|_| {
            (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    i32::from(x & 1 == 1)
                })
                .collect()
        })
        .collect()
}

fn level() -> Option<tessera_rl_simd::Level> {
    std::env::var("TESSERA_RL_SIMD")
        .ok()
        .map(|n| Resolved::from_name(Some(&n)).unwrap().level())
}

macro_rules! steady {
    ($env:ty, $n:expr, $workers:expr) => {{
        let n: u32 = $n;
        let options = PoolOptions {
            auto_reset: AutoReset::SameStep,
            simd_level: level(),
            ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(1))
        };
        let mut pool = EnvPool::<$env>::new(CartPoleConfig::default(), options).unwrap();
        if $workers > 0 {
            pool = pool
                .with_workers(SyncOptions::new(NonZeroUsize::new($workers).unwrap()))
                .unwrap();
        }
        pool.reset_all(Reseed::Keep).unwrap();
        let acts = actions(usize::try_from(n).unwrap());
        for row in acts.iter().cycle().take(300) {
            pool.step(ActionInput {
                discrete: row,
                continuous: &[],
            })
            .unwrap();
        }
        (pool, acts)
    }};
}

macro_rules! run {
    ($group:expr, $id:expr, $pool:expr, $acts:expr) => {{
        let pool = &mut $pool;
        let mut rows = $acts.iter().cycle();
        $group.bench_function($id, |b| {
            b.iter(|| {
                let row = rows.next().unwrap();
                let out = pool
                    .step(ActionInput {
                        discrete: row,
                        continuous: &[],
                    })
                    .unwrap();
                black_box(out.rewards().first().copied())
            });
        });
    }};
}

fn bench(c: &mut Criterion) {
    let lanes = level().map_or_else(Resolved::best, |l| Resolved::force(l).unwrap());
    for n in [8192u32, 1024] {
        let mut group = c.benchmark_group(format!("cartpole_lane_stop_{n}"));
        for workers in [0usize, 2] {
            let (mut lane, acts) = steady!(CartPole, n, workers);
            run!(
                group,
                BenchmarkId::new(format!("lane_{}", lanes.level()), format!("w{workers}")),
                lane,
                acts
            );
            drop(lane);
            let (mut scalar, acts) = steady!(ScalarCartPole, n, workers);
            run!(
                group,
                BenchmarkId::new("scalar_step", format!("w{workers}")),
                scalar,
                acts
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
