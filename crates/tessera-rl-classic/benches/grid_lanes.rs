// SPDX-License-Identifier: MIT OR Apache-2.0

//! `tessera/GridLanes-v1`'s full pool step (the lane kernels on the column
//! store, at the widest level or the one `TESSERA_RL_SIMD` names) against
//! `tessera/Grid-v1`'s (the same dynamics, one instance at a time on the row
//! store), ADR-0090 stage 2, #345. Report-only: no bar binds (decision 6
//! binds stage 1). Same-step auto-reset, steady state (300 settling steps),
//! xorshift actions, Grid's default config; N = 8192 and 1024, Inline and
//! two workers, chunk 1024.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::{NonZeroU32, NonZeroUsize};
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use tessera_rl_classic::{Grid, GridConfig, GridLanes};
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
                    i32::try_from(x & 3).unwrap()
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
            chunk_size: NonZeroU32::new(1024).unwrap(),
            simd_level: level(),
            ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(1))
        };
        let mut pool = EnvPool::<$env>::new(GridConfig::default(), options).unwrap();
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
        let mut group = c.benchmark_group(format!("grid_lanes_{n}"));
        for workers in [0usize, 2] {
            let (mut soa, acts) = steady!(GridLanes, n, workers);
            run!(
                group,
                BenchmarkId::new(
                    format!("grid_lanes_{}", lanes.level()),
                    format!("w{workers}")
                ),
                soa,
                acts
            );
            drop(soa);
            let (mut grid, acts) = steady!(Grid, n, workers);
            run!(
                group,
                BenchmarkId::new("grid", format!("w{workers}")),
                grid,
                acts
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
