// SPDX-License-Identifier: MIT OR Apache-2.0

//! A Pipelined pool with a heuristic slot (ADR-0079 item 3.4, #347 stage
//! B1) never allocates after construction, on any thread: the heuristic
//! pass, its fault cell, the fault read at `recv` and the copy to
//! `heuristic_actions` included.
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
    ActionInput, AutoReset, EnvPool, Masked, PipelinedOptions, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::StepRng;

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

/// `n` action values, agent 0 then agent 1: slot 1's are garbage, never
/// read.
fn rows(n: usize) -> Vec<i32> {
    (0..n).map(|k| if k % 2 == 0 { 1 } else { -9 }).collect()
}

fn input(a: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: a,
        continuous: &[],
    }
}

#[test]
fn test_pipelined_step_with_a_heuristic_slot_does_not_allocate_on_any_thread() {
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
        .pipelined(PipelinedOptions::new(NonZeroUsize::new(4).unwrap()))
        .unwrap();
    let halves = pool.halves();
    let len = |h: usize| usize::try_from(halves[h].end - halves[h].start).unwrap() * 2;
    let (act_a, act_b) = (rows(len(0)), rows(len(1)));
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
        std::hint::black_box(pool.heuristic_actions());
    }
    let n = alloc_diag::stop();
    assert_eq!(
        n,
        0,
        "allocations in the measured window:\n{}",
        alloc_diag::report()
    );
    assert!(!pool.is_poisoned());
    drop((a, b));
}
