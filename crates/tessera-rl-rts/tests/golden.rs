// SPDX-License-Identifier: MIT OR Apache-2.0

//! The stage 2 golden: a seeded legal-play trajectory, folding the state
//! hash, the rewards and the whole mask row of every step, identical inline
//! and with 1 and 3 workers. Pinned on macOS arm64; the Linux runs check it.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures"
)]

use core::num::{NonZeroU32, NonZeroUsize};

use tessera_rl_core::Bool8;
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, IllegalActionPolicy, Masked, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_rts::env::{MASK_PER_UNIT, N_PRODUCTS, N_TYPES, WINDOW_CELLS};
use tessera_rl_rts::{Rts, RtsConfig};

type Pool = EnvPool<Rts, Masked>;

fn cfg(size: u16, max_units: u16) -> RtsConfig {
    RtsConfig {
        size,
        decision_ticks: 4,
        max_units,
        step_limit: 400,
        seat_swap: false,
    }
}

fn pool(c: RtsConfig, n: u32, seed: u64, workers: Option<usize>) -> Pool {
    let options = PoolOptions {
        auto_reset: AutoReset::NextStep,
        illegal_action: IllegalActionPolicy::Noop,
        chunk_size: NonZeroU32::new(1).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed))
    };
    let mut p: Pool = EnvPool::with_masks(c, options).unwrap();
    if let Some(w) = workers {
        p = p
            .with_workers(SyncOptions::new(NonZeroUsize::new(w).unwrap()))
            .unwrap();
    }
    p
}

/// Resets every instance once and returns the masks.
fn ready(p: &mut Pool) -> Vec<Bool8> {
    p.reset_all(Reseed::Keep).unwrap().masks().to_vec()
}

/// A seeded generator for tests.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        usize::try_from(self.0 % u64::try_from(n).unwrap()).unwrap()
    }

    /// A uniformly drawn true index of `row`; every row has one.
    fn pick(&mut self, row: &[Bool8]) -> i32 {
        let legal: Vec<usize> = (0..row.len()).filter(|&i| row[i].get()).collect();
        i32::try_from(legal[self.below(legal.len())]).unwrap()
    }
}

/// A legal action row for every agent of every instance, drawn from masks.
fn legal_actions(rng: &mut Rng, masks: &[Bool8], n: usize, max_units: usize) -> Vec<i32> {
    let types = usize::from(N_TYPES);
    let cells = usize::from(WINDOW_CELLS);
    let products = usize::from(N_PRODUCTS);
    let per_agent = max_units * MASK_PER_UNIT;
    let mut out = Vec::new();
    for a in 0..n * 2 {
        let agent = &masks[a * per_agent..(a + 1) * per_agent];
        for u in 0..max_units {
            let block = &agent[u * MASK_PER_UNIT..(u + 1) * MASK_PER_UNIT];
            let t = rng.pick(&block[..types]);
            let tu = usize::try_from(t).unwrap();
            let target = rng.pick(&block[types + tu * cells..types + (tu + 1) * cells]);
            let base = types + types * cells;
            let product = rng.pick(&block[base + tu * products..base + (tu + 1) * products]);
            out.extend([t, target, product]);
        }
    }
    out
}

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

fn fnv(hash: u64, word: u64) -> u64 {
    word.to_le_bytes().iter().fold(hash, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn golden_run(workers: Option<usize>) -> u64 {
    let c = cfg(12, 6);
    let mut p = pool(c, 2, 2026, workers);
    let mut rng = Rng(0x7e55_u64);
    let mut masks = ready(&mut p);
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for _ in 0..96 {
        let a = legal_actions(&mut rng, &masks, 2, 6);
        let out = p.step(input(&a)).unwrap();
        for r in out.rewards() {
            hash = fnv(hash, u64::from(r.to_bits()));
        }
        masks = out.masks().to_vec();
        hash = masks.iter().fold(hash, |h, m| fnv(h, u64::from(m.get())));
        for i in 0..2 {
            hash = fnv(hash, p.state_hash(i).unwrap().get());
        }
    }
    hash
}

/// Pinned on macOS arm64; any change to the rules, the masks or the
/// encoding moves it.
const RTS_GOLDEN: u64 = 0x73d2_5fff_be44_e95e;

#[test]
fn test_rts_golden_is_pinned_and_identical_across_workers() {
    let inline = golden_run(None);
    assert_eq!(inline, RTS_GOLDEN, "{inline:#x}");
    assert_eq!(golden_run(Some(1)), inline);
    assert_eq!(golden_run(Some(3)), inline);
}
