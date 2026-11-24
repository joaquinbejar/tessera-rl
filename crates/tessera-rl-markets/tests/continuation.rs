// SPDX-License-Identifier: MIT OR Apache-2.0

//! Continuation tests (SPEC 9.1; ADR-0010) for `tessera/LimitOrderBook-v1`:
//! snapshot one instance mid-episode with orders queued under latency 2,
//! run K steps, then restore the record into the same slot, another slot
//! and another pool built with a different seed, replay the same actions,
//! and require identical rewards, masks, observations, flags and state
//! hashes.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures: a failure is the assertion"
)]

mod support;

use support::{Pool, Rng, config, input, legal_actions, pool};
use tessera_rl_engine::{MAX_LEAVES, Reseed};
use tessera_rl_markets::actions::{HEADS, MASK_LEN};
use tessera_rl_markets::limits::MAX_AGENTS;

const N: u32 = 4;
const SRC: u32 = 1;
const T: u32 = 17;
const K: u32 = 40;

/// What instance `id` reported after the last call, and its state hash.
#[derive(Debug, Clone, PartialEq)]
struct Frame {
    rewards: Vec<u32>,
    masks: Vec<bool>,
    obs: Vec<u8>,
    flags: Vec<bool>,
    state: u64,
}

fn frame(p: &Pool, id: u32) -> Frame {
    let out = p.outputs().unwrap();
    let i = usize::try_from(id).unwrap();
    let rows = i * MAX_AGENTS..(i + 1) * MAX_AGENTS;
    let n = out.rewards().len() / MAX_AGENTS;
    let mut obs = Vec::new();
    for leaf in 0..MAX_LEAVES {
        let Some(bytes) = out.observation(leaf) else {
            break;
        };
        let per = bytes.len() / n;
        obs.extend_from_slice(&bytes[i * per..(i + 1) * per]);
    }
    let mut flags: Vec<bool> = out.alive()[rows.clone()].iter().map(|f| f.get()).collect();
    flags.push(out.episode_terminated()[i].get());
    flags.push(out.episode_truncated()[i].get());
    Frame {
        rewards: out.rewards()[rows.clone()]
            .iter()
            .map(|r| r.to_bits())
            .collect(),
        masks: out.masks()[rows.start * MASK_LEN..rows.end * MASK_LEN]
            .iter()
            .map(|m| m.get())
            .collect(),
        obs,
        flags,
        state: p.state_hash(id).unwrap().get(),
    }
}

/// `acts` with instance `dst`'s row replaced by `row`.
fn with_row(mut acts: Vec<i32>, dst: u32, row: &[i32]) -> Vec<i32> {
    let d = usize::try_from(dst).unwrap();
    let w = MAX_AGENTS * HEADS;
    acts[d * w..(d + 1) * w].copy_from_slice(row);
    acts
}

fn cfg() -> tessera_rl_markets::LobConfig {
    config(6, 2)
}

#[test]
fn test_continuation_restores_identically_in_every_slot_and_pool() {
    let width = MAX_AGENTS * HEADS;
    let src = usize::try_from(SRC).unwrap();
    let mut base = pool(cfg(), N, 77, None);
    let mut masks = base.reset_all(Reseed::Keep).unwrap().masks().to_vec();
    let mut rng = Rng(0x0381_c0de);
    for _ in 0..T {
        let acts = legal_actions(&mut rng, &masks, N as usize);
        masks = base.step(input(&acts)).unwrap().masks().to_vec();
    }
    // Latency 2: the snapshot carries queued items.
    assert!(!base.state(SRC).unwrap().pending.is_empty());
    let record = base.snapshot(SRC).unwrap();
    let mut rows = Vec::new();
    let mut frames = Vec::new();
    for _ in 0..K {
        let acts = legal_actions(&mut rng, &masks, N as usize);
        rows.push(acts[src * width..(src + 1) * width].to_vec());
        masks = base.step(input(&acts)).unwrap().masks().to_vec();
        frames.push(frame(&base, SRC));
    }

    let replay = |p: &mut Pool, dst: u32, seed: u64| {
        p.restore(dst, &record).unwrap();
        let mut other = Rng(seed);
        let mut m = p.outputs().unwrap().masks().to_vec();
        for (k, row) in rows.iter().enumerate() {
            let a = with_row(legal_actions(&mut other, &m, N as usize), dst, row);
            m = p.step(input(&a)).unwrap().masks().to_vec();
            assert_eq!(
                frame(p, dst),
                frames[k],
                "step {k} after restoring into {dst}"
            );
        }
    };
    // Same slot, another slot of the same pool, and a fresh pool with a
    // different seed whose other instances play other actions.
    let mut same = pool(cfg(), N, 77, None);
    same.reset_all(Reseed::Keep).unwrap();
    replay(&mut same, SRC, 1);
    let mut slot = pool(cfg(), N, 77, None);
    slot.reset_all(Reseed::Keep).unwrap();
    replay(&mut slot, 3, 2);
    let mut other = pool(cfg(), N, 991, None);
    other.reset_all(Reseed::Keep).unwrap();
    replay(&mut other, 0, 3);
}
