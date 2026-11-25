// SPDX-License-Identifier: MIT OR Apache-2.0

//! Survival-v1 costs (STAGE1-r1): inline pool step at 16, 64 and 128 agents
//! (map sides 32, 64 and 96), reset (map generation included, since
//! auto-reset pays it) at both ends, and snapshot and restore of one
//! instance at both ends. The state is the same size at every config.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_engine::{ActionInput, EnvPool, NoMasks, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;
use tessera_rl_world::survival::MAX_AGENTS;
use tessera_rl_world::{Survival, SurvivalConfig};

const N: u32 = 64;

fn cfg(agents: u32, size: u32) -> SurvivalConfig {
    SurvivalConfig {
        agents,
        size,
        ..SurvivalConfig::default()
    }
}

fn bench(c: &mut Criterion) {
    let options = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1));
    let mut group = c.benchmark_group("world_pool_64");
    for (agents, size) in [(16, 32), (64, 64), (128, 96)] {
        let mut pool = EnvPool::<Survival, NoMasks>::new(cfg(agents, size), options).unwrap();
        pool.reset_all(Reseed::Keep).unwrap();
        // Scattered moves and acts, two heads per agent slot.
        let actions: Vec<i32> = (0..N as usize * MAX_AGENTS * 2)
            .map(|i| i32::try_from((i * 7 + i / 2) % if i % 2 == 0 { 5 } else { 7 }).unwrap())
            .collect();
        group.bench_function(format!("step_{agents}_agents_size_{size}"), |b| {
            b.iter(|| {
                let out = pool
                    .step(ActionInput {
                        discrete: &actions,
                        continuous: &[],
                    })
                    .unwrap();
                black_box(out.rewards());
            });
        });
        if agents != 64 {
            group.bench_function(format!("reset_all_{agents}_agents_size_{size}"), |b| {
                b.iter(|| black_box(pool.reset_all(Reseed::Keep).unwrap().rewards().len()));
            });
            let saved = pool.snapshot(0).unwrap();
            group.bench_function(format!("snapshot_{agents}_agents"), |b| {
                b.iter(|| black_box(pool.snapshot(0).unwrap()));
            });
            group.bench_function(format!("restore_{agents}_agents"), |b| {
                b.iter(|| pool.restore(0, &saved).unwrap());
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
