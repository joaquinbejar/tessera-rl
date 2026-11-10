// SPDX-License-Identifier: MIT OR Apache-2.0

//! The RTS step through the engine: a pool of 64 instances at the default
//! size and unit slots, every unit given the no-op, so the cost is the
//! masks, the ticks and the observation. Local only, no claim (SPEC 16.1).

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

criterion_group!(benches, bench);
criterion_main!(benches);
