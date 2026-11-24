// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gather (`tessera/Gather-v1`) with transforms on its `Set` leaves
//! (ADR-0092): masked normalization pooled over slots, absent slots zero,
//! and a set-wide frame stack with its `frame_age` leaf, against a pool
//! without transforms stepped with the same actions.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_classic::gather::MAX_ITEMS;
use tessera_rl_classic::{Gather, GatherConfig};
use tessera_rl_core::space::Space;
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, EnvPool, NoMasks, PoolOptions, Reseed, SyncOptions,
    TransformError, Transforms,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 6;
const K: usize = 3;

fn cfg() -> GatherConfig {
    GatherConfig {
        size: 6,
        items: 8,
        max_steps: 10_000,
    }
}

fn options() -> PoolOptions {
    PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(77))
    }
}

fn transforms() -> Transforms {
    Transforms::new()
        .normalize("items", 1_000.0)
        .frame_stack("items", u32::try_from(K).unwrap())
}

fn transformed(workers: usize) -> EnvPool<Gather, NoMasks> {
    let mut p = EnvPool::new(cfg(), options())
        .unwrap()
        .with_transforms(transforms())
        .unwrap();
    if workers > 0 {
        p = p
            .with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
            .unwrap();
    }
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn actions(t: usize) -> Vec<i32> {
    (0..N as usize)
        .map(|i| i32::try_from((t * 7 + i * 3) % 4).unwrap())
        .collect()
}

fn step(p: &mut EnvPool<Gather, NoMasks>, t: usize) {
    p.step(ActionInput {
        discrete: &actions(t),
        continuous: &[],
    })
    .unwrap();
}

/// Instance `i`'s raw `(items, mask)`: `[MAX, 2]` i32 and `[MAX]`.
fn raw_frame(p: &EnvPool<Gather, NoMasks>, i: usize) -> (Vec<i32>, Vec<u8>) {
    let out = p.outputs().unwrap();
    let items = &out.observation(1).unwrap()[i * 64..(i + 1) * 64];
    let mask = &out.observation(2).unwrap()[i * MAX_ITEMS..(i + 1) * MAX_ITEMS];
    let items = items
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect();
    (items, mask.to_vec())
}

#[test]
fn test_gather_transforms_stacked_set_exports_item_age_then_mask() {
    let p = transformed(0);
    let Space::Dict(entries) = p.observation_space().unwrap() else {
        panic!("a dict");
    };
    let Space::Set { inner, max } = &entries[1].1 else {
        panic!("{:?}", entries[1].1);
    };
    assert_eq!(usize::try_from(*max).unwrap(), K * MAX_ITEMS);
    let Space::Dict(keys) = &**inner else {
        panic!("{inner:?}");
    };
    let keys: Vec<&str> = keys.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["item", "frame_age"]);
}

/// Every exported leaf against the frames of a pool without transforms:
/// items normalized (identity before the first barrier, clip 1000) and
/// stacked oldest first, `frame_age` `k - 1 - j / MAX` on present slots,
/// absent slots zero everywhere, and statistics counting present slots
/// only.
#[test]
fn test_gather_transforms_stack_and_masked_normalize_match_raw_frames() {
    let mut p = transformed(0);
    let mut raw = EnvPool::<Gather, NoMasks>::new(cfg(), options()).unwrap();
    raw.reset_all(Reseed::Keep).unwrap();
    let n = N as usize;
    let mut history: Vec<Vec<(Vec<i32>, Vec<u8>)>> =
        (0..n).map(|i| vec![raw_frame(&raw, i); K]).collect();
    let mut present = 0usize;
    let mut changed = false;
    for h in &history {
        present += h[0].1.iter().filter(|&&b| b != 0).count();
    }
    for t in 0..=30 {
        if t > 0 {
            step(&mut p, t);
            step(&mut raw, t);
            for (i, h) in history.iter_mut().enumerate() {
                let frame = raw_frame(&raw, i);
                present += frame.1.iter().filter(|&&b| b != 0).count();
                changed |= frame.1 != h[K - 1].1;
                h.remove(0);
                h.push(frame);
            }
        }
        let out = p.outputs().unwrap();
        let slots = K * MAX_ITEMS;
        for (i, h) in history.iter().enumerate() {
            let items = &out.observation(1).unwrap()[i * slots * 8..(i + 1) * slots * 8];
            let age = &out.observation(2).unwrap()[i * slots..(i + 1) * slots];
            let mask = &out.observation(3).unwrap()[i * slots..(i + 1) * slots];
            for j in 0..slots {
                let (frame, slot) = (j / MAX_ITEMS, j % MAX_ITEMS);
                let (raw_items, raw_mask) = &h[frame];
                assert_eq!(mask[j], raw_mask[slot], "t {t} i {i} j {j}");
                let here = raw_mask[slot] != 0;
                let want_age = if here {
                    u8::try_from(K - 1 - frame).unwrap()
                } else {
                    0
                };
                assert_eq!(age[j], want_age, "t {t} i {i} j {j}");
                for e in 0..2 {
                    let at = (j * 2 + e) * 4;
                    let got = f32::from_ne_bytes(items[at..at + 4].try_into().unwrap());
                    #[expect(clippy::cast_precision_loss, reason = "grid coordinates")]
                    let want = if here {
                        raw_items[slot * 2 + e] as f32
                    } else {
                        0.0
                    };
                    assert_eq!(got.to_bits(), want.to_bits(), "t {t} i {i} j {j} e {e}");
                }
            }
        }
    }
    assert!(changed, "the run must collect an item");
    p.epoch_barrier().unwrap();
    let context = p.pool_context().unwrap();
    let stats = &context.adaptive.normalization.normalization;
    assert_eq!(stats.len(), 2, "pooled over slots");
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let want = present as f64;
    assert!(
        stats.iter().all(|m| m.count.to_bits() == want.to_bits()),
        "{stats:?} vs {present}"
    );
}

#[test]
fn test_gather_transforms_identical_across_workers() {
    let run = |workers: usize| {
        let mut p = transformed(workers);
        let mut seen = Vec::new();
        for t in 1..=12 {
            step(&mut p, t);
            if t % 4 == 0 {
                p.epoch_barrier().unwrap();
            }
            let out = p.outputs().unwrap();
            seen.push(
                (0..4)
                    .map(|l| out.observation(l).unwrap().to_vec())
                    .collect::<Vec<_>>(),
            );
        }
        (seen, p.pool_context().unwrap())
    };
    let inline = run(0);
    for w in [1, 2, 5] {
        assert_eq!(run(w), inline, "{w} workers");
    }
}

#[test]
fn test_gather_transforms_partial_set_stack_is_refused() {
    let p = EnvPool::<Gather, NoMasks>::new(cfg(), options()).unwrap();
    assert!(matches!(
        p.with_transforms(Transforms::new().frame_stack("items_mask", 2)),
        Err(EngineError::Transform(
            TransformError::PartialSetStack { .. }
        ))
    ));
}
