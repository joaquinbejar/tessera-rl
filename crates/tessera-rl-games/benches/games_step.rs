// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inline step cost of Connect Four and Leduc hold'em: full pool steps (auto-reset on) with
//! masks, the acting-agent update and the turn check.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_games::{Connect4, Connect4Config, Leduc, LeducConfig};
use tessera_rl_num::identity::Seed;

const N: u32 = 1024;

fn bench(c: &mut Criterion) {
    let options = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1));
    let mut group = c.benchmark_group("games_pool_step_1024");
    // Both slots send a column; only the acting one is read, and a full
    // column is replaced by the no-op.
    let actions: Vec<i32> = (0..N * 2).map(|i| ((i * 3) % 7).cast_signed()).collect();
    let mut pool = EnvPool::<Connect4, Masked>::with_masks(Connect4Config {}, options).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    group.bench_function("connect4", |b| {
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
    // Leduc: calls and raises; a raise past the limit is the no-op.
    let bets: Vec<i32> = (0..N * 2).map(|i| (1 + i % 2).cast_signed()).collect();
    let mut leduc = EnvPool::<Leduc, Masked>::with_masks(LeducConfig {}, options).unwrap();
    leduc.reset_all(Reseed::Keep).unwrap();
    group.bench_function("leduc", |b| {
        b.iter(|| {
            let out = leduc
                .step(ActionInput {
                    discrete: &bets,
                    continuous: &[],
                })
                .unwrap();
            black_box(out.rewards());
        });
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
