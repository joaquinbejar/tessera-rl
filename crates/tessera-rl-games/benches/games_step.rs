// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inline step cost of Connect Four, Othello, Go 9x9 and Leduc hold'em: full pool steps (auto-reset on) with
//! masks, the acting-agent update and the turn check.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_engine::{ActionInput, DecodeLimits, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_games::{
    Connect4, Connect4Config, Go9, Go9Config, Leduc, LeducConfig, Othello, OthelloConfig,
};
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
    // Othello: scattered squares; an illegal one is the no-op.
    let squares: Vec<i32> = (0..N * 2).map(|i| ((i * 37) % 65).cast_signed()).collect();
    let mut othello = EnvPool::<Othello, Masked>::with_masks(OthelloConfig {}, options).unwrap();
    othello.reset_all(Reseed::Keep).unwrap();
    group.bench_function("othello", |b| {
        b.iter(|| {
            let out = othello
                .step(ActionInput {
                    discrete: &squares,
                    continuous: &[],
                })
                .unwrap();
            black_box(out.rewards());
        });
    });
    // Go 9x9: scattered points; an illegal one is the pass.
    let points: Vec<i32> = (0..N * 2).map(|i| ((i * 37) % 82).cast_signed()).collect();
    let mut go = EnvPool::<Go9, Masked>::with_masks(Go9Config::default(), options).unwrap();
    go.reset_all(Reseed::Keep).unwrap();
    group.bench_function("go9", |b| {
        b.iter(|| {
            let out = go
                .step(ActionInput {
                    discrete: &points,
                    continuous: &[],
                })
                .unwrap();
            black_box(out.rewards());
        });
    });
    group.finish();
    // What the exact superko history costs search: the state is about 3.9 KB.
    let mut forks = c.benchmark_group("go9_record");
    let dst: Vec<u32> = (1..N).collect();
    forks.bench_function("snapshot", |b| {
        b.iter(|| black_box(go.snapshot(0).unwrap()));
    });
    forks.bench_function("fork_1023", |b| {
        b.iter(|| go.fork(0, black_box(&dst)).unwrap());
    });
    forks.finish();
}

fn persist(c: &mut Criterion) {
    let options = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1));
    let mut pool = EnvPool::<Go9, Masked>::with_masks(Go9Config::default(), options).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let ids: Vec<u32> = (0..N).collect();
    let file = pool.encode(&ids).unwrap();
    let mut group = c.benchmark_group("persist_go9_1024");
    group.bench_function("encode", |b| {
        b.iter(|| black_box(pool.encode(&ids).unwrap()));
    });
    group.bench_function("decode", |b| {
        b.iter(|| black_box(pool.decode(&file, DecodeLimits::default()).unwrap()));
    });
    group.finish();
}

criterion_group!(benches, bench, persist);
criterion_main!(benches);
