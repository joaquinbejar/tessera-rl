// SPDX-License-Identifier: MIT OR Apache-2.0

//! An Async pool with a heuristic slot (ADR-0079 item 3.4, #347 stage B1)
//! never allocates after construction, on any thread: random subsets in a
//! random order with `batch_size < N`, the heuristic pass, the per-id fault
//! cells, their read at `recv` and the copy to `heuristic_actions`
//! included.
//!
//! The counting allocator counts on every thread while enabled, so this
//! binary holds a single test.

#![cfg(not(loom))]

mod common;

#[path = "common/alloc_diag.rs"]
mod alloc_diag;

use core::num::{NonZeroU32, NonZeroUsize};

use alloc_diag::Tagged;
use common::{CountConfig, Counts};
use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AutoReset, EnvPool, Masked, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::StepRng;

const N: usize = 64;
const BATCH: usize = 4;

/// A heuristic that draws from its RNG: action 1 or 2, legal for agent 1.
impl Heuristic for Tagged {
    fn act(
        _: &CountConfig,
        _: &Counts,
        _: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let draw = rng.below_u32(NonZeroU32::new(2).unwrap_or(NonZeroU32::MIN));
        out.set_discrete(0, 1 + i32::try_from(draw).unwrap_or(0));
    }
}

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

#[test]
fn test_async_step_with_a_heuristic_slot_does_not_allocate_on_any_thread() {
    let options = PoolOptions {
        auto_reset: AutoReset::SameStep,
        chunk_size: NonZeroU32::new(8).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(1))
    };
    let cfg = CountConfig {
        length: 4,
        ..CountConfig::default()
    };
    let mut pool = EnvPool::<Tagged, Masked>::with_masks(cfg, options)
        .unwrap()
        .with_heuristic_slots(&[1])
        .unwrap()
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(4).unwrap(),
            NonZeroUsize::new(BATCH).unwrap(),
        ))
        .unwrap();
    // Slot 1's rows are garbage: never read.
    let actions: Vec<i32> = (0..N * 2).map(|k| if k % 2 == 0 { 1 } else { -9 }).collect();
    let mut ids: Vec<u32> = (0..u32::try_from(N).unwrap()).collect();
    let mut rng = 0x9e37_79b9_u64;
    let mut next = move || {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        usize::try_from(rng >> 33).unwrap()
    };
    // Warm up: the first jobs may initialize thread-local runtime state.
    drop(pool.reset_all(Reseed::Keep).unwrap());
    pool.send(&ids, input(&actions)).unwrap();
    for _ in 0..N / BATCH {
        drop(pool.recv().unwrap());
    }
    alloc_diag::start();
    for _ in 0..100 {
        // A random order, sent in random-sized subsets.
        for k in (1..N).rev() {
            ids.swap(k, next() % (k + 1));
        }
        let mut sent = 0;
        while sent < N {
            let take = (1 + next() % 7).min(N - sent);
            let subset = &ids[sent..sent + take];
            pool.send(subset, input(&actions[..take * 2])).unwrap();
            sent += take;
        }
        let mut back = 0;
        while back < N {
            let t = pool.recv().unwrap();
            std::hint::black_box(pool.heuristic_actions());
            back += t.len();
        }
    }
    let n = alloc_diag::stop();
    assert_eq!(
        n,
        0,
        "allocations in the measured window:\n{}",
        alloc_diag::report()
    );
    assert!(!pool.is_poisoned());
}
