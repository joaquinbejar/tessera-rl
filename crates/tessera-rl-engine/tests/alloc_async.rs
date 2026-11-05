// SPDX-License-Identifier: MIT OR Apache-2.0

//! Async mode never allocates after construction, on any thread (SPEC 5.5,
//! ADR-0047): random subsets in a random order with `batch_size < N`, one
//! batch held the whole time (its ids stalled) while the others cycle, with
//! masks and `SameStep` auto-reset, then a reset of some ids and a full
//! reset.
//!
//! The counting allocator counts on every thread while enabled, so this
//! binary holds a single test.

#![cfg(not(loom))]

mod common;

#[path = "common/alloc_diag.rs"]
mod alloc_diag;

use core::num::{NonZeroU32, NonZeroUsize};

use alloc_diag::Tagged;
use common::CountConfig;
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AutoReset, EnvPool, Masked, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;

const N: usize = 64;
const BATCH: usize = 4;

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

#[test]
fn test_async_send_recv_and_reset_do_not_allocate_on_any_thread() {
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
        .async_mode(AsyncOptions::new(
            NonZeroUsize::new(4).unwrap(),
            NonZeroUsize::new(BATCH).unwrap(),
        ))
        .unwrap();
    let actions = vec![1i32; N * 2];
    let mut ids: Vec<u32> = (0..u32::try_from(N).unwrap()).collect();
    let mut rng = 0x9e37_79b9_u64;
    let mut next = move || {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        usize::try_from(rng >> 33).unwrap()
    };
    // Warm up, and keep one batch held for the whole run: its ids stall.
    drop(pool.reset_all(Reseed::Keep).unwrap());
    pool.send(&ids, input(&actions)).unwrap();
    let held = pool.recv().unwrap();
    let mut free = Vec::with_capacity(N);
    for _ in 0..(N / BATCH - 1) {
        let t = pool.recv().unwrap();
        free.extend(t.ids());
    }
    ids.clear();
    ids.extend_from_slice(&free);
    let cycling = ids.len();
    alloc_diag::start();
    for _ in 0..100 {
        // A random order, sent in random-sized subsets.
        for k in (1..cycling).rev() {
            ids.swap(k, next() % (k + 1));
        }
        let mut sent = 0;
        while sent < cycling {
            let take = (1 + next() % 7).min(cycling - sent);
            let subset = &ids[sent..sent + take];
            pool.send(subset, input(&actions[..take * 2])).unwrap();
            sent += take;
        }
        let mut back = 0;
        while back < cycling {
            let t = pool.recv().unwrap();
            std::hint::black_box(t.rewards(t.ids().next().unwrap()));
            back += t.len();
        }
    }
    // Writers need no live token (ADR-0047 section 3).
    drop(held);
    pool.reset(&ids[..BATCH], Reseed::Keep).unwrap();
    drop(pool.recv().unwrap());
    let token = std::hint::black_box(pool.reset_all(Reseed::Keep));
    let n = alloc_diag::stop();
    drop(token);
    assert_eq!(
        n,
        0,
        "allocations in the measured window:\n{}",
        alloc_diag::report()
    );
}
