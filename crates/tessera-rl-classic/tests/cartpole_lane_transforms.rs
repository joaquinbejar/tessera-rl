// SPDX-License-Identifier: MIT OR Apache-2.0

//! Observation transforms on the lane path (ADR-0071 with ADR-0078): a
//! CartPole pool with `Normalize` and `FrameStack` on its leaf, with the
//! lane path at every level this CPU has, must give the observations,
//! terminal observations, frame stacks, records and transform context
//! (identity, normalization statistics, freeze flag) of the same pool
//! without a lane step, after every step. Same-step resets refill the frame
//! stacks on both paths through the shared post-step code.

#![cfg(not(loom))]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

#[path = "../benches/support/scalar_cartpole.rs"]
mod scalar_cartpole;

use core::num::{NonZeroU32, NonZeroUsize};

use scalar_cartpole::ScalarCartPole;
use tessera_rl_classic::cartpole::{CartPole, CartPoleConfig};
use tessera_rl_core::Environment;
use tessera_rl_core::space::Space;
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, PoolOptions, Reseed, SyncOptions, Transforms,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_simd::{Level, Resolved};

const N: u32 = 64;

fn options(auto_reset: AutoReset, level: Option<Level>) -> PoolOptions {
    PoolOptions {
        chunk_size: NonZeroU32::new(16).unwrap(),
        max_episode_steps: NonZeroU32::new(40),
        auto_reset,
        simd_level: level,
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(31))
    }
}

/// The leaf name of CartPole's observation space ("" for a lone Box).
fn leaf() -> String {
    match CartPole::observation_space(&CartPoleConfig {}) {
        Ok(Space::Dict(entries)) => entries.first().map(|e| e.0.clone()).unwrap_or_default(),
        _ => String::new(),
    }
}

fn transforms() -> Transforms {
    Transforms::new()
        .normalize(leaf(), 5.0)
        .frame_stack(leaf(), 4)
}

macro_rules! pool {
    ($env:ty, $auto_reset:expr, $level:expr, $workers:expr) => {{
        let mut p = EnvPool::<$env>::new(CartPoleConfig {}, options($auto_reset, $level)).unwrap();
        if $workers > 0 {
            p = p
                .with_workers(SyncOptions::new(NonZeroUsize::new($workers).unwrap()))
                .unwrap();
        }
        let mut p = p.with_transforms(transforms()).unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        p
    }};
}

/// Everything a step exposes that the transforms touch, as bytes.
macro_rules! seen {
    ($p:expr, $same:expr) => {{
        let p = &$p;
        let o = p.outputs().unwrap();
        let mut v: Vec<Vec<u8>> = vec![o.observation(0).unwrap().to_vec()];
        if $same {
            v.push(o.terminal_observation(0).unwrap().to_vec());
        }
        v.push(
            o.rewards()
                .iter()
                .flat_map(|r| r.to_bits().to_le_bytes())
                .collect(),
        );
        let mut stacks = vec![0u8; p.transform_state_bytes() * N as usize];
        p.snapshot_transforms_all(&mut stacks).unwrap();
        v.push(stacks);
        for i in 0..N {
            v.push(p.snapshot(i).unwrap().to_bytes());
        }
        (v, p.pool_context().unwrap().adaptive.normalization)
    }};
}

fn action(t: u32, i: u32) -> i32 {
    i32::try_from((i.wrapping_mul(2_654_435_761) ^ t.wrapping_mul(40_503)) >> 7 & 1).unwrap()
}

#[test]
fn test_cartpole_lane_transforms_framestack_normalize_every_level_match_scalar() {
    for level in Resolved::available().map(Resolved::level) {
        for workers in [0usize, 2] {
            for auto_reset in [AutoReset::SameStep, AutoReset::NextStep] {
                let same = auto_reset == AutoReset::SameStep;
                let mut lane = pool!(CartPole, auto_reset, Some(level), workers);
                let mut plain = pool!(ScalarCartPole, auto_reset, None, workers);
                assert_eq!(
                    seen!(lane, same),
                    seen!(plain, same),
                    "{level} w{workers}: reset"
                );
                for t in 0..600 {
                    let acts: Vec<i32> = (0..N).map(|i| action(t, i)).collect();
                    let input = ActionInput {
                        discrete: &acts,
                        continuous: &[],
                    };
                    lane.step(input).unwrap();
                    plain.step(input).unwrap();
                    if t % 50 == 49 {
                        lane.epoch_barrier().unwrap();
                        plain.epoch_barrier().unwrap();
                    }
                    assert_eq!(
                        seen!(lane, same),
                        seen!(plain, same),
                        "{level} w{workers} {auto_reset:?}: step {t}"
                    );
                }
            }
        }
    }
}

const K: usize = 4;
/// One CartPole frame: four `f32`.
const FRAME: usize = 16;

/// Frame stacks shift by one frame per step (ADR-0078 item 4): for every
/// instance that neither ended nor was reset in a step, its exported frames
/// `0..K-1` equal its previous row's frames `1..K` (oldest first), on the
/// lane path and without it. This checks the shared post-step stack mode
/// itself, which an equality of the two paths cannot see.
#[test]
fn test_cartpole_lane_transforms_framestack_shifts_one_frame_per_step() {
    let stacked = || Transforms::new().frame_stack(leaf(), u32::try_from(K).unwrap());
    for level in Resolved::available().map(Resolved::level) {
        for auto_reset in [AutoReset::SameStep, AutoReset::NextStep] {
            let mut lane =
                EnvPool::<CartPole>::new(CartPoleConfig {}, options(auto_reset, Some(level)))
                    .unwrap()
                    .with_transforms(stacked())
                    .unwrap();
            let mut plain =
                EnvPool::<ScalarCartPole>::new(CartPoleConfig {}, options(auto_reset, None))
                    .unwrap()
                    .with_transforms(stacked())
                    .unwrap();
            lane.reset_all(Reseed::Keep).unwrap();
            plain.reset_all(Reseed::Keep).unwrap();
            let mut prev = [
                lane.outputs().unwrap().observation(0).unwrap().to_vec(),
                plain.outputs().unwrap().observation(0).unwrap().to_vec(),
            ];
            let mut pending = vec![false; N as usize];
            let mut checked = 0usize;
            for t in 0..300 {
                let acts: Vec<i32> = (0..N).map(|i| action(t, i)).collect();
                let input = ActionInput {
                    discrete: &acts,
                    continuous: &[],
                };
                lane.step(input).unwrap();
                plain.step(input).unwrap();
                for (k, o) in [lane.outputs().unwrap(), plain.outputs().unwrap()]
                    .into_iter()
                    .enumerate()
                {
                    let row = o.observation(0).unwrap();
                    for i in 0..N as usize {
                        let ended = o.episode_terminated().get(i).is_some_and(|b| b.get())
                            || o.episode_truncated().get(i).is_some_and(|b| b.get());
                        let reset = pending.get(i).copied().unwrap_or(false);
                        if ended || reset {
                            continue;
                        }
                        let base = i * K * FRAME;
                        let now = row.get(base..base + (K - 1) * FRAME).unwrap();
                        let before = prev[k].get(base + FRAME..base + K * FRAME).unwrap();
                        assert_eq!(
                            now, before,
                            "{level} {auto_reset:?} path {k}: instance {i} at step {t}"
                        );
                        checked += 1;
                    }
                    prev[k] = row.to_vec();
                }
                // Under NextStep an ended instance resets (Fresh) on the next step.
                let o = plain.outputs().unwrap();
                for (i, p) in pending.iter_mut().enumerate() {
                    *p = auto_reset == AutoReset::NextStep
                        && (o.episode_terminated().get(i).is_some_and(|b| b.get())
                            || o.episode_truncated().get(i).is_some_and(|b| b.get()));
                }
            }
            assert!(checked > 10_000, "too few shifts checked: {checked}");
        }
    }
}
