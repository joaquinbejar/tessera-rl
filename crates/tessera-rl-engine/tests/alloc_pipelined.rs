// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pipelined mode never allocates after construction, on any thread
//! (SPEC 5.5, ADR-0047): 100 cycles of send and recv on both halves, with
//! masks and `SameStep` auto-reset, then a full reset.
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
    ActionInput, AutoReset, EnvPool, Masked, PipelinedOptions, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

#[test]
fn test_pipelined_send_recv_and_reset_do_not_allocate_on_any_thread() {
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
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(4).unwrap()))
        .unwrap();
    let halves = pool.halves();
    let len = |h: usize| usize::try_from(halves[h].end - halves[h].start).unwrap() * 2;
    let (act_a, act_b) = (vec![1i32; len(0)], vec![1i32; len(1)]);
    // Warm up: the first jobs may initialize thread-local runtime state.
    let [a, b] = pool.reset_all(Reseed::Keep).unwrap();
    pool.send(a, input(&act_a)).map_err(|e| e.error).unwrap();
    pool.send(b, input(&act_b)).map_err(|e| e.error).unwrap();
    let mut a = pool.recv().unwrap();
    let mut b = pool.recv().unwrap();
    alloc_diag::start();
    for _ in 0..100 {
        pool.send(a, input(&act_a)).map_err(|e| e.error).unwrap();
        pool.send(b, input(&act_b)).map_err(|e| e.error).unwrap();
        a = pool.recv().unwrap();
        b = pool.recv().unwrap();
        std::hint::black_box(a.rewards());
    }
    drop((a, b));
    let tokens = std::hint::black_box(pool.reset_all(Reseed::Keep));
    let n = alloc_diag::stop();
    drop(tokens);
    assert_eq!(
        n,
        0,
        "allocations in the measured window:\n{}",
        alloc_diag::report()
    );
}
