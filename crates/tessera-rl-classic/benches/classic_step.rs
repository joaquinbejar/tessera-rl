// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inline step cost of the classic environments: the CartPole physics
//! kernel alone, and full pool steps (auto-reset on) for CartPole, Tag and the
//! reference grid world.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_classic::cartpole::physics;
use tessera_rl_classic::{
    CartPole, CartPoleConfig, Grid, GridConfig, KeyDoor, KeyDoorConfig, Snake, SnakeConfig, Tag,
    TagConfig,
};
use tessera_rl_engine::{ActionInput, EnvPool, Masked, NoMasks, PoolOptions, Reseed};
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
    let grid_actions: Vec<i32> = (0..N).map(|i| (i % 4).cast_signed()).collect();
    let mut grid = EnvPool::<Grid, NoMasks>::new(GridConfig::default(), options()).unwrap();
    grid.reset_all(Reseed::Keep).unwrap();
    group.bench_function("grid", |b| {
        b.iter(|| {
            let out = grid
                .step(ActionInput {
                    discrete: &grid_actions,
                    continuous: &[],
                })
                .unwrap();
            black_box(out.rewards());
        });
    });
    // Key and door: a seeded policy over each row's mask, so agents move, take
    // the key and open doors; the step limit forces a reset every 32 steps, each
    // generating a new world. The policy's own cost is part of the figure.
    let keydoor_options = PoolOptions {
        max_episode_steps: NonZeroU32::new(32),
        ..options()
    };
    let keydoor_cfg = KeyDoorConfig { size: 8, agents: 4 };
    let mut keydoor = EnvPool::<KeyDoor, Masked>::with_masks(keydoor_cfg, keydoor_options).unwrap();
    keydoor.reset_all(Reseed::Keep).unwrap();
    let mut policy = vec![4i32; (N * 4) as usize];
    let mut draw = 0x1234_5678_u64;
    group.bench_function("keydoor", |b| {
        b.iter(|| {
            for (row, action) in keydoor
                .outputs()
                .unwrap()
                .masks()
                .chunks(5)
                .zip(policy.iter_mut())
            {
                draw = draw.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
                let pick = usize::try_from((draw >> 33) % 5).unwrap();
                let first = (0..5)
                    .map(|k| (pick + k) % 5)
                    .find(|&a| row.get(a).is_some_and(|f| f.get()));
                *action = i32::try_from(first.unwrap_or(4)).unwrap();
            }
            let out = keydoor
                .step(ActionInput {
                    discrete: &policy,
                    continuous: &[],
                })
                .unwrap();
            black_box(out.rewards());
        });
    });
    snake(&mut group);
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);

/// Snake: two agents on the default board under a seeded policy over each
/// row's mask; deaths end episodes, which reset at once. Both observations,
/// as `pixels` is the default users measure.
fn snake(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    for (name, obs) in [
        ("snake_pixels", tessera_rl_classic::snake::OBS_PIXELS),
        ("snake_state", tessera_rl_classic::snake::OBS_STATE),
    ] {
        let cfg = SnakeConfig {
            agents: 2,
            obs,
            ..SnakeConfig::default()
        };
        let mut snake = EnvPool::<Snake, Masked>::with_masks(cfg, options()).unwrap();
        snake.reset_all(Reseed::Keep).unwrap();
        let mut policy = vec![1i32; (N * 2) as usize];
        let mut draw = 0x9abc_def0_u64;
        group.bench_function(name, |b| {
            b.iter(|| {
                for (row, action) in snake
                    .outputs()
                    .unwrap()
                    .masks()
                    .chunks(4)
                    .zip(policy.iter_mut())
                {
                    draw = draw.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
                    let pick = usize::try_from((draw >> 33) % 4).unwrap();
                    let first = (0..4)
                        .map(|k| (pick + k) % 4)
                        .find(|&a| row.get(a).is_some_and(|f| f.get()));
                    *action = i32::try_from(first.unwrap_or(0)).unwrap();
                }
                let out = snake
                    .step(ActionInput {
                        discrete: &policy,
                        continuous: &[],
                    })
                    .unwrap();
                black_box(out.rewards());
            });
        });
    }
}
