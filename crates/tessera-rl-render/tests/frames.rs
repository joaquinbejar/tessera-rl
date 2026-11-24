// SPDX-License-Identifier: MIT OR Apache-2.0

//! Golden frame digests (ADR-0083 item 3): each environment's frame after a
//! seeded reset, and mid-episode frames of CartPole and Connect4, rendered
//! from a snapshot record through the real path (`EnvPool::snapshot` ->
//! `Record::state` -> `Renderer`), as XXH3-64 of the RGB8 bytes. A scene
//! built without floats, or with frozen math under the canonical controls
//! `Renderer` sets, gives the same frame on every platform and under any
//! caller floating-point controls.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::{BrickWall, CartPole, Grid, KeyDoor, PaddleDuel, Snake};
use tessera_rl_engine::{ActionInput, EnvPool, Masked, NoMasks, PoolOptions, Reseed};
use tessera_rl_games::{Connect4, Go9, Othello};
use tessera_rl_num::fp::hostile;
use tessera_rl_num::identity::Seed;
use tessera_rl_render::Renderer;

fn digest(bytes: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(bytes)
}

/// Resets `pool`, snapshots instance 0 and renders its state with `$env`'s
/// renderer: `(digest, overflowed)`.
macro_rules! frame_of {
    ($env:ty, $pool:expr, $cfg:expr) => {
        frame_of!($env, $pool, $cfg, (&[] as &[&[i32]]))
    };
    ($env:ty, $pool:expr, $cfg:expr, $steps:expr) => {{
        $pool.reset_all(Reseed::Keep).unwrap();
        for &acts in $steps.iter() {
            $pool
                .step(ActionInput {
                    discrete: acts,
                    continuous: &[],
                })
                .unwrap();
        }
        let record = $pool.snapshot(0).unwrap();
        let mut renderer = Renderer::<$env>::new(&$cfg).unwrap();
        let mut frame = vec![0; renderer.frame_len()];
        let overflowed = renderer
            .frame_into(&$cfg, record.state(), &mut frame)
            .unwrap();
        (digest(&frame), overflowed)
    }};
}

fn options() -> PoolOptions {
    PoolOptions::new(NonZeroU32::new(1).unwrap(), Seed::new(7))
}

macro_rules! plain {
    ($env:ty) => {{
        let cfg = <$env as tessera_rl_core::Environment>::Config::default();
        let mut pool = EnvPool::<$env, NoMasks>::new(cfg.clone(), options()).unwrap();
        frame_of!($env, pool, cfg)
    }};
    ($env:ty, $steps:expr) => {{
        let cfg = <$env as tessera_rl_core::Environment>::Config::default();
        let mut pool = EnvPool::<$env, NoMasks>::new(cfg.clone(), options()).unwrap();
        frame_of!($env, pool, cfg, $steps)
    }};
}

macro_rules! masked {
    ($env:ty) => {{
        let cfg = <$env as tessera_rl_core::Environment>::Config::default();
        let mut pool = EnvPool::<$env, Masked>::with_masks(cfg.clone(), options()).unwrap();
        frame_of!($env, pool, cfg)
    }};
    ($env:ty, $steps:expr) => {{
        let cfg = <$env as tessera_rl_core::Environment>::Config::default();
        let mut pool = EnvPool::<$env, Masked>::with_masks(cfg.clone(), options()).unwrap();
        frame_of!($env, pool, cfg, $steps)
    }};
}

/// CartPole pushed right fifteen times.
const CARTPOLE_STEPS: [&[i32]; 15] = [&[1]; 15];

/// Connect4: columns 3, 3, 2, 4, 3 (both slots carry the column; only the
/// player to move is read).
const CONNECT4_STEPS: [&[i32]; 5] = [&[3, 3], &[3, 3], &[2, 2], &[4, 4], &[3, 3]];

fn all_frames() -> Vec<(&'static str, (u64, bool))> {
    vec![
        ("CartPole", plain!(CartPole)),
        ("Grid", plain!(Grid)),
        ("GridKeyDoor", masked!(KeyDoor)),
        ("Snake", masked!(Snake)),
        ("BrickWall", plain!(BrickWall)),
        ("PaddleDuel", plain!(PaddleDuel)),
        ("Connect4", masked!(Connect4)),
        ("Othello", masked!(Othello)),
        ("Go9", masked!(Go9)),
        ("CartPole mid-episode", plain!(CartPole, CARTPOLE_STEPS)),
        ("Connect4 mid-episode", masked!(Connect4, CONNECT4_STEPS)),
    ]
}

#[test]
fn test_frames_seeded_reset_match_golden_digests() {
    let got = all_frames();
    for (name, (_, overflowed)) in &got {
        assert!(!overflowed, "{name} overflowed its scene");
    }
    let digests: Vec<(&str, u64)> = got.iter().map(|(n, (d, _))| (*n, *d)).collect();
    assert_eq!(digests, GOLDEN, "\nnew digests: {digests:#x?}");
}

#[test]
fn test_frames_hostile_caller_controls_same_digests() {
    // The caller's controls flush denormals and round toward zero;
    // Renderer computes under the canonical ones, so nothing changes.
    let (got, _) = hostile::run(
        |c| hostile::toward_zero(hostile::flush_to_zero(c)),
        all_frames,
    );
    let digests: Vec<(&str, u64)> = got.iter().map(|(n, (d, _))| (*n, *d)).collect();
    assert_eq!(digests, GOLDEN);
}

const GOLDEN: [(&str, u64); 11] = [
    ("CartPole", 0x9422_48ac_90ee_18df),
    ("Grid", 0x8d2d_7042_c8f6_823d),
    ("GridKeyDoor", 0xec3a_d7cf_0366_f41c),
    ("Snake", 0xc4bc_9801_dc81_f6ed),
    ("BrickWall", 0x8f6d_7bc6_3761_9ba6),
    ("PaddleDuel", 0xe9c3_1ace_509c_04ec),
    ("Connect4", 0xdb02_24b5_0157_2054),
    ("Othello", 0xde81_f232_3c8c_55e5),
    ("Go9", 0x8d37_a8d6_069d_1590),
    ("CartPole mid-episode", 0x2977_459a_79c9_27c7),
    ("Connect4 mid-episode", 0x02f1_5086_ca8f_9163),
];
