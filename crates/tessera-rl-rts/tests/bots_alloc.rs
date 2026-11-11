// SPDX-License-Identifier: MIT OR Apache-2.0

//! A scripted bot's act never allocates: it runs inside the step when a
//! scripted opponent plays (SPEC zero-allocation hot path). This file
//! installs a counting global allocator for this test binary only; its
//! `unsafe` only forwards to the system allocator.

#![expect(clippy::many_single_char_names, reason = "test fixtures")]

use core::num::NonZeroU32;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use tessera_rl_core::AgentSlot;
use tessera_rl_core::io::ActionWriter;
use tessera_rl_engine::{ActionInput, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::{LogicalId, RngIdentity, Seed, StreamId};
use tessera_rl_num::rng::{RngAuthority, StepRng};
use tessera_rl_rts::bots::Bot;
use tessera_rl_rts::env::HEADS_PER_UNIT;
use tessera_rl_rts::{Rts, RtsConfig};

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

struct Counting;

#[expect(unsafe_code, reason = "test-only counting allocator")]
// SAFETY: every method forwards to `System` with the same arguments; the
// counters are const-initialized thread locals, which never allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: forwarded unchanged; the caller upholds `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn count() {
    let _ = COUNTING.try_with(|on| {
        if on.get() {
            let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        }
    });
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocations(f: impl FnOnce()) -> u64 {
    ALLOCATIONS.with(|n| n.set(0));
    COUNTING.with(|on| on.set(true));
    f();
    COUNTING.with(|on| on.set(false));
    ALLOCATIONS.with(Cell::get)
}

#[test]
fn test_bots_alloc_counter_detects_heap_use() {
    assert!(allocations(|| drop(std::hint::black_box(vec![1u8; 16]))) > 0);
}

#[test]
fn test_bots_act_never_allocates() {
    let c = RtsConfig {
        size: 16,
        decision_ticks: 4,
        max_units: 16,
        step_limit: 600,
        seat_swap: false,
        opponent: 0,
    };
    let options = PoolOptions::new(NonZeroU32::new(1).unwrap(), Seed::new(14));
    let mut p: EnvPool<Rts, Masked> = EnvPool::with_masks(c.clone(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let identity = RngIdentity::new(Seed::new(1), LogicalId::new(0));
    let mut r = StepRng::new(&RngAuthority::engine_only(), &identity, StreamId::DYNAMICS);
    let n = usize::from(c.max_units) * HEADS_PER_UNIT;
    let (mut heads, mut a) = (vec![0; n], vec![0; 2 * n]);
    for _ in 0..200 {
        let s = *p.state(0).unwrap();
        for bot in [Bot::Rush, Bot::Prospector, Bot::RandomLegal] {
            for agent in [0, 1] {
                let slot = AgentSlot::new(agent, 2).unwrap();
                let mut w = ActionWriter::new(&mut heads, &mut []);
                assert_eq!(
                    allocations(|| bot.act(&c, &s, slot, &mut r, &mut w)),
                    0,
                    "{bot:?}"
                );
            }
        }
        for (agent, bot) in [(0u16, Bot::Rush), (1, Bot::Prospector)] {
            let slot = AgentSlot::new(agent, 2).unwrap();
            let lo = usize::from(agent) * n;
            let mut w = ActionWriter::new(&mut a[lo..lo + n], &mut []);
            bot.act(&c, &s, slot, &mut r, &mut w);
        }
        p.step(ActionInput {
            discrete: &a,
            continuous: &[],
        })
        .unwrap();
    }
}
