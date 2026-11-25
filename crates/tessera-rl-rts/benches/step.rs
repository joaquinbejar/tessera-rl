// SPDX-License-Identifier: MIT OR Apache-2.0

//! The RTS step through the engine: a pool of 64 instances at the default
//! size and unit slots, every unit given the no-op, so the cost is the
//! masks, the ticks and the observation; then the same step with a
//! heuristic slot (#586). Local only, no claim (SPEC 16.1).

#![expect(clippy::unwrap_used, reason = "bench setup")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;
use tessera_rl_rts::env::HEADS_PER_UNIT;
use tessera_rl_rts::{Rts, RtsConfig};

const N: u32 = 64;

fn bench(c: &mut Criterion) {
    let cfg = RtsConfig::default();
    let units = usize::from(cfg.max_units);
    let options = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1));
    let mut pool = EnvPool::<Rts, Masked>::with_masks(cfg, options).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let actions = vec![0i32; usize::try_from(N).unwrap() * 2 * units * HEADS_PER_UNIT];
    c.bench_function("rts/step/n64/noop", |b| {
        b.iter(|| {
            let out = pool
                .step(ActionInput {
                    discrete: black_box(&actions),
                    continuous: &[],
                })
                .unwrap();
            black_box(out.masks().len());
        });
    });
}

/// The same step with agent slot 1 a heuristic slot (#586): the engine
/// builds slot 1's information state (a fogged `RtsState` copy) and runs
/// Rush on it every step; slot 1's input rows are ignored. Against
/// `rts/step/n64/noop`, the difference is the heuristic slot's cost.
fn bench_info_heuristic(c: &mut Criterion) {
    let cfg = RtsConfig::default();
    let units = usize::from(cfg.max_units);
    let options = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1));
    let mut pool = EnvPool::<Rts, Masked>::with_masks(cfg, options)
        .unwrap()
        .with_info_heuristic_slots(&[1])
        .unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let actions = vec![0i32; usize::try_from(N).unwrap() * 2 * units * HEADS_PER_UNIT];
    c.bench_function("rts/step/n64/info_heuristic_slot1", |b| {
        b.iter(|| {
            let out = pool
                .step(ActionInput {
                    discrete: black_box(&actions),
                    continuous: &[],
                })
                .unwrap();
            black_box(out.masks().len());
        });
    });
}

criterion_group!(benches, bench, bench_info_heuristic);
criterion_main!(benches);
