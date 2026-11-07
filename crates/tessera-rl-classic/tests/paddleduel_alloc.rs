// SPDX-License-Identifier: MIT OR Apache-2.0

//! The paddle duel hot path never allocates after construction (SPEC 5.5,
//! ADR-0004), serves, points and auto-reset included, in both observations. This file installs a counting
//! global allocator for this test binary only; its `unsafe` only forwards to the
//! system allocator.

use core::num::NonZeroU32;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use tessera_rl_classic::paddleduel::{OBS_PIXELS, OBS_STATE};
use tessera_rl_classic::{PaddleDuel, PaddleDuelConfig};
use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, NoMasks, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

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
fn test_counter_detects_heap_use() {
    assert!(allocations(|| drop(std::hint::black_box(vec![1u8; 16]))) > 0);
}

#[test]
fn test_paddleduel_reset_and_steps_through_auto_reset_do_not_allocate() {
    for agents in [1, 2] {
        for obs in [OBS_PIXELS, OBS_STATE] {
            for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
                let options = PoolOptions {
                    auto_reset,
                    ..PoolOptions::new(NonZeroU32::new(8).unwrap(), Seed::new(1))
                };
                let cfg = PaddleDuelConfig {
                    agents,
                    frameskip: 8,
                    max_score: 1,
                    bounce: 1,
                    obs,
                };
                let mut pool = EnvPool::<PaddleDuel, NoMasks>::new(cfg, options).unwrap();
                // Slot 0 always up: it misses balls aimed low, so one point
                // ends each episode within the loop.
                let actions = [0i32; 16];
                let mut ended = false;
                let n = allocations(|| {
                    let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
                    for _ in 0..80 {
                        let input = ActionInput {
                            discrete: &actions,
                            continuous: &[],
                        };
                        let out = pool.step(input).unwrap();
                        ended |= out.episode_terminated().iter().any(|f| f.get());
                    }
                });
                assert_eq!(n, 0, "{agents} {obs} {auto_reset:?}");
                assert!(ended, "no episode ended, so auto-reset was not exercised");
            }
        }
    }
}
