// SPDX-License-Identifier: MIT OR Apache-2.0

//! The mountain car hot path never allocates after construction (SPEC 5.5,
//! ADR-0004), terminations, truncations and auto-reset included. This file installs a counting
//! global allocator for this test binary only; its `unsafe` only forwards to the
//! system allocator.

use core::num::NonZeroU32;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use tessera_rl_classic::{MountainCar, MountainCarConfig};
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
fn test_mountaincar_reset_and_steps_through_auto_reset_do_not_allocate() {
    for auto_reset in [AutoReset::NextStep, AutoReset::SameStep] {
        let options = PoolOptions {
            auto_reset,
            max_episode_steps: NonZeroU32::new(150),
            ..PoolOptions::new(NonZeroU32::new(8).unwrap(), Seed::new(1))
        };
        let mut pool =
            EnvPool::<MountainCar, NoMasks>::new(MountainCarConfig::default(), options).unwrap();
        let mut actions = [2i32; 8];
        let mut ended = false;
        let n = allocations(|| {
            let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
            for t in 0..400 {
                // The baseline on half the instances (some reach the goal),
                // rocking on the rest (truncated at 150).
                for (i, a) in actions.iter_mut().enumerate().take(4) {
                    let v = pool
                        .state(u32::try_from(i).unwrap_or(0))
                        .map_or(0.0, |s| s.s[1]);
                    *a = if v >= 0.0 { 2 } else { 0 };
                }
                for a in actions.iter_mut().skip(4) {
                    *a = 1 + i32::from(t % 2 == 0) - i32::from(t % 2 == 1);
                }
                let input = ActionInput {
                    discrete: &actions,
                    continuous: &[],
                };
                let out = pool.step(input).unwrap();
                ended |= out.episode_terminated().iter().any(|f| f.get());
            }
        });
        assert_eq!(n, 0, "{auto_reset:?}");
        assert!(
            ended,
            "no episode terminated, so auto-reset was not exercised"
        );
    }
}
