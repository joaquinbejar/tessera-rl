// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inline step cost of the classic environments: the CartPole physics
//! kernel alone, and full pool steps (auto-reset on) for CartPole, Tag, the
//! reference grid world and the other classic environments.

#![expect(clippy::unwrap_used, reason = "bench setup unwraps")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_classic::cartpole::physics;
use tessera_rl_classic::{
    Acrobot, AcrobotConfig, BrickWall, BrickWallConfig, CartPole, CartPoleConfig, Gather,
    GatherConfig, Grid, GridConfig, KeyDoor, KeyDoorConfig, MountainCar, MountainCarConfig,
    PaddleDuel, PaddleDuelConfig, Pendulum, PendulumConfig, Snake, SnakeConfig, Tag, TagConfig,
};
use tessera_rl_engine::{
    ActionInput, EnvPool, Masked, NoMasks, PolicyMap, PoolOptions, Reseed, Transforms,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 1024;

fn options() -> PoolOptions {
    PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1))
}

/// Gather (ADR-0082): the item set's absent rows are zeroed after every
/// observation, so this includes that pass.
fn bench_gather(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    actions: &[i32],
) {
    let mut gather = EnvPool::<Gather, NoMasks>::new(GatherConfig::default(), options()).unwrap();
    gather.reset_all(Reseed::Keep).unwrap();
    group.bench_function("gather", |b| {
        b.iter(|| {
            let out = gather
                .step(ActionInput {
                    discrete: actions,
                    continuous: &[],
                })
                .unwrap();
            black_box(out.rewards());
        });
    });
}

/// The cost of the policy map (ADR-0079): `heuristic_none` (no map) and
/// `heuristic_map_learner_only` (an explicit map, every slot a learner)
/// take the same step path, one branch per chunk; `tag_heuristic_slot` makes
/// Tag's last evader a heuristic slot, whose pass runs once per chunk.
fn bench_policy_map(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    actions: &[i32],
) {
    let cfg = TagConfig::default;
    let last = cfg().agents() - 1;
    let none = EnvPool::<Tag, Masked>::with_masks(cfg(), options()).unwrap();
    let learner_only = EnvPool::<Tag, Masked>::with_masks(cfg(), options())
        .unwrap()
        .with_policy_map(PolicyMap::new(tessera_rl_classic::tag::MAX_AGENTS))
        .unwrap();
    let heuristic = EnvPool::<Tag, Masked>::with_masks(cfg(), options())
        .unwrap()
        .with_info_heuristic_slots(&[last])
        .unwrap();
    for (name, mut pool) in [
        ("heuristic_none", none),
        ("heuristic_map_learner_only", learner_only),
        ("tag_heuristic_slot", heuristic),
    ] {
        pool.reset_all(Reseed::Keep).unwrap();
        group.bench_function(name, |b| {
            b.iter(|| {
                let out = pool
                    .step(ActionInput {
                        discrete: actions,
                        continuous: &[],
                    })
                    .unwrap();
                black_box(out.rewards());
            });
        });
    }
}

/// Gather with transforms on its item set (ADR-0092), against the same
/// pool without them: `full` keeps all 8 items, `sparse` 1 (7 of 8 slots
/// absent, the re-zero worst case). Auto-reset keeps the presence pattern.
fn bench_gather_transforms(c: &mut Criterion) {
    let mut group = c.benchmark_group("gather_transforms_1024");
    let actions: Vec<i32> = (0..N).map(|i| i32::try_from(i % 4).unwrap()).collect();
    let lists = [
        ("none", Transforms::new()),
        ("normalize", Transforms::new().normalize("items", 5.0)),
        ("stack4", Transforms::new().frame_stack("items", 4)),
        (
            "normalize_stack4",
            Transforms::new()
                .normalize("items", 5.0)
                .frame_stack("items", 4),
        ),
    ];
    for (density, items) in [("full", 8u8), ("sparse", 1u8)] {
        for (name, t) in &lists {
            let cfg = GatherConfig {
                items,
                ..GatherConfig::default()
            };
            let mut p = EnvPool::<Gather, NoMasks>::new(cfg, options())
                .unwrap()
                .with_transforms(t.clone())
                .unwrap();
            p.reset_all(Reseed::Keep).unwrap();
            group.bench_function(format!("{density}/{name}"), |b| {
                b.iter(|| {
                    let out = p
                        .step(ActionInput {
                            discrete: &actions,
                            continuous: &[],
                        })
                        .unwrap();
                    black_box(out.rewards());
                });
            });
        }
    }
    group.finish();
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
    bench_gather(&mut group, &grid_actions);
    bench_policy_map(&mut group, &tag_actions);
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
    brickwall(&mut group);
    paddleduel(&mut group);
    mountaincar(&mut group);
    pendulum(&mut group);
    acrobot(&mut group);
    group.finish();
}

criterion_group!(benches, bench);
criterion_group!(transforms, bench_gather_transforms);
criterion_main!(benches, transforms);

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

/// Brick wall: the default config under a seeded random policy, frameskip 4;
/// misses end episodes, which reset at once. Both observations.
fn brickwall(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    for (name, obs) in [
        (
            "brickwall_pixels",
            tessera_rl_classic::brickwall::OBS_PIXELS,
        ),
        ("brickwall_state", tessera_rl_classic::brickwall::OBS_STATE),
    ] {
        let cfg = BrickWallConfig {
            obs,
            ..BrickWallConfig::default()
        };
        let mut pool = EnvPool::<BrickWall, NoMasks>::new(cfg, options()).unwrap();
        pool.reset_all(Reseed::Keep).unwrap();
        let mut policy = vec![1i32; N as usize];
        let mut draw = 0x0bad_cafe_u64;
        group.bench_function(name, |b| {
            b.iter(|| {
                for action in &mut policy {
                    draw = draw.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
                    *action = i32::try_from((draw >> 33) % 3).unwrap();
                }
                let out = pool
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

/// Paddle duel: two agents under a seeded random policy, frameskip 4; points
/// end episodes at the default score, which reset at once. Both observations.
fn paddleduel(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    for (name, obs) in [
        (
            "paddleduel_pixels",
            tessera_rl_classic::paddleduel::OBS_PIXELS,
        ),
        (
            "paddleduel_state",
            tessera_rl_classic::paddleduel::OBS_STATE,
        ),
    ] {
        let cfg = PaddleDuelConfig {
            agents: 2,
            obs,
            ..PaddleDuelConfig::default()
        };
        let mut pool = EnvPool::<PaddleDuel, NoMasks>::new(cfg, options()).unwrap();
        pool.reset_all(Reseed::Keep).unwrap();
        let mut policy = vec![1i32; (N * 2) as usize];
        let mut draw = 0x00de_cade_u64;
        group.bench_function(name, |b| {
            b.iter(|| {
                for action in &mut policy {
                    draw = draw.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
                    *action = i32::try_from((draw >> 33) % 3).unwrap();
                }
                let out = pool
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

/// Mountain car: a seeded random policy; episodes truncate at 200 and
/// reset at once.
fn mountaincar(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    let mut pool =
        EnvPool::<MountainCar, NoMasks>::new(MountainCarConfig::default(), options()).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let mut policy = vec![1i32; N as usize];
    let mut draw = 0x0dd_ba11_u64;
    group.bench_function("mountaincar", |b| {
        b.iter(|| {
            for action in &mut policy {
                draw = draw.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
                *action = i32::try_from((draw >> 33) % 3).unwrap();
            }
            let out = pool
                .step(ActionInput {
                    discrete: &policy,
                    continuous: &[],
                })
                .unwrap();
            black_box(out.rewards());
        });
    });
}

/// Acrobot: a seeded random policy; episodes truncate at 500 and
/// reset at once.
fn acrobot(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    let mut pool = EnvPool::<Acrobot, NoMasks>::new(AcrobotConfig::default(), options()).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let mut policy = vec![1i32; N as usize];
    let mut draw = 0x0dd_ba11_u64;
    group.bench_function("acrobot", |b| {
        b.iter(|| {
            for action in &mut policy {
                draw = draw.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
                *action = i32::try_from((draw >> 33) % 3).unwrap();
            }
            let out = pool
                .step(ActionInput {
                    discrete: &policy,
                    continuous: &[],
                })
                .unwrap();
            black_box(out.rewards());
        });
    });
}

/// Pendulum: a fixed cycle of torques; episodes truncate and reset.
fn pendulum(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    let mut pool = EnvPool::<Pendulum, NoMasks>::new(PendulumConfig::default(), options()).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let policy = vec![1.5f32; N as usize];
    group.bench_function("pendulum", |b| {
        b.iter(|| {
            let out = pool
                .step(ActionInput {
                    discrete: &[],
                    continuous: &policy,
                })
                .unwrap();
            black_box(out.rewards());
        });
    });
}
