// SPDX-License-Identifier: MIT OR Apache-2.0

//! A step of the RTS allocates nothing (SPEC 5.5): actions recorded from one
//! run are replayed in a second pool under a counting allocator.
//!
//! The counting allocator of `support/counting.rs` attributes allocator
//! calls by thread (issue #43). This binary holds a single measuring test,
//! which runs each scenario (the former tests, by name) with its own
//! objects and windows, and reports every failing scenario together.

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
        opponent: 0,
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

#[path = "support/counting.rs"]
mod counting;
use counting::window;

fn test_rts_step_does_not_allocate() {
    let c = cfg(16, 12);
    let mut p = pool(c.clone(), 2, 17, None);
    let mut rng = Rng(3);
    let mut masks = ready(&mut p);
    let mut rows = Vec::new();
    for _ in 0..60 {
        let a = legal_actions(&mut rng, &masks, 2, 12);
        masks = p.step(input(&a)).unwrap().masks().to_vec();
        rows.push(a);
    }
    let mut q = pool(c, 2, 17, None);
    ready(&mut q);
    let ((), t) = window(|| {
        for a in &rows {
            let _ = std::hint::black_box(q.step(input(a)).unwrap());
        }
    });
    assert_eq!(t.calls, 0, "a step allocated {} times", t.calls);
    assert_eq!(q.state_hash(0).unwrap(), p.state_hash(0).unwrap());
}

/// The scenarios this binary runs.
const SCENARIOS: usize = 1;

#[test]
fn test_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[(
            "test_rts_step_does_not_allocate",
            test_rts_step_does_not_allocate,
        )],
    );
}
