// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inline step cost of the classic environments: the CartPole physics
//! kernel alone, and full pool steps (auto-reset on) for CartPole and Tag.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_classic::cartpole::physics;
use tessera_rl_classic::{CartPole, CartPoleConfig, Tag, TagConfig};
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

const N: u32 = 1024;

fn options() -> PoolOptions {
    PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1))
}

fn bench(c: &mut Criterion) {
    c.bench_function("cartpole_physics_step", |b| {
        let mut s = [0.01, 0.0, 0.02, 0.0];
        let mut right = false;
        b.iter(|| {
            s = physics::step(black_box(s), right);
            right = !right;
            black_box(physics::is_terminal(s))
        });
    });

    let mut group = c.benchmark_group("classic_pool_step_1024");
    let cartpole_actions: Vec<i32> = (0..N).map(|i| i32::from(i % 2 == 0)).collect();
    let mut cartpole = EnvPool::<CartPole>::new(CartPoleConfig::default(), options()).unwrap();
    cartpole.reset_all(Reseed::Keep).unwrap();
    group.bench_function("cartpole", |b| {
        b.iter(|| {
            let out = cartpole
                .step(ActionInput {
                    discrete: &cartpole_actions,
                    continuous: &[],
                })
                .unwrap();
            black_box(out.rewards());
        });
    });

    let tag_actions: Vec<i32> = (0..N * 8).map(|i| (i % 5).cast_signed()).collect();
    let mut tag = EnvPool::<Tag, Masked>::with_masks(TagConfig::default(), options()).unwrap();
    tag.reset_all(Reseed::Keep).unwrap();
    group.bench_function("tag", |b| {
        b.iter(|| {
            // Unmasked moves off the grid are replaced by the no-op.
            let out = tag
                .step(ActionInput {
                    discrete: &tag_actions,
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
