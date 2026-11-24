// SPDX-License-Identifier: MIT OR Apache-2.0

//! One pool step of `tessera/LimitOrderBook-v1` with masks, at 8 and 16
//! agents and latency 0 and 2, actions drawn from the masks before the
//! timed loop. Local only, no claim (SPEC 16.1).

#![expect(clippy::unwrap_used, clippy::indexing_slicing, reason = "bench setup")]

use core::num::NonZeroU32;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use tessera_rl_core::Bool8;
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_markets::actions::{CANCEL_ROWS, HEADS, KINDS, MASK_LEN, OFFSET_ROWS, SIZE_ROWS};
use tessera_rl_markets::limits::MAX_AGENTS;
use tessera_rl_markets::{LimitOrderBook, LobConfig};
use tessera_rl_num::identity::Seed;

const INSTANCES: u32 = 64;

/// The first legal choice of each head after a pseudo-random start.
fn pick(row: &[Bool8], start: usize) -> i32 {
    let n = row.len();
    (0..n)
        .map(|k| (start + k) % n)
        .find(|&i| row[i].get())
        .map_or(0, |i| i32::try_from(i).unwrap())
}

fn actions(masks: &[Bool8], salt: usize) -> Vec<i32> {
    let mut out = Vec::new();
    for agent in 0..masks.len() / MASK_LEN {
        let row = &masks[agent * MASK_LEN..(agent + 1) * MASK_LEN];
        let seed = agent.wrapping_mul(2_654_435_761).wrapping_add(salt);
        let kind = pick(&row[..KINDS as usize], seed);
        let kind_at = usize::try_from(kind).unwrap();
        let offset = pick(
            &row[OFFSET_ROWS + kind_at * 9..OFFSET_ROWS + kind_at * 9 + 9],
            seed >> 3,
        );
        let offset_at = usize::try_from(offset).unwrap();
        let base = SIZE_ROWS + kind_at * 36 + offset_at * 4;
        let size = pick(&row[base..base + 4], seed >> 5);
        let cancel = pick(
            &row[CANCEL_ROWS + kind_at * 8..CANCEL_ROWS + kind_at * 8 + 8],
            seed >> 7,
        );
        out.extend([kind, offset, size, cancel]);
    }
    out
}

fn bench(c: &mut Criterion) {
    for (agents, latency) in [(8u32, 0u32), (8, 2), (16, 0), (16, 2)] {
        let cfg = LobConfig {
            agents,
            latency,
            ..LobConfig::default()
        };
        let options = PoolOptions::new(NonZeroU32::new(INSTANCES).unwrap(), Seed::new(5));
        let mut p = EnvPool::<LimitOrderBook, Masked>::with_masks(cfg, options).unwrap();
        let mut masks = p.reset_all(Reseed::Keep).unwrap().masks().to_vec();
        // Warm the book up, then time steps with fresh legal actions.
        for t in 0..200 {
            let a = actions(&masks, t);
            masks = p
                .step(ActionInput {
                    discrete: &a,
                    continuous: &[],
                })
                .unwrap()
                .masks()
                .to_vec();
        }
        let a = actions(&masks, 7);
        assert_eq!(
            a.len(),
            usize::try_from(INSTANCES).unwrap() * MAX_AGENTS * HEADS
        );
        c.bench_function(
            &format!("lob_step_{agents}_agents_latency_{latency}"),
            |b| {
                b.iter(|| {
                    let out = p.step(ActionInput {
                        discrete: black_box(&a),
                        continuous: &[],
                    });
                    black_box(out.map_or(0, |o| o.rewards().len()))
                });
            },
        );
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
