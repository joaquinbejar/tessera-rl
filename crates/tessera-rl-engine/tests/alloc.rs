// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Inline hot path never allocates after construction (SPEC 5.5,
//! ADR-0004).
//!
//! This file installs a counting global allocator for this test binary
//! only; it is test code, not one of the audited `unsafe` modules, and its
//! `unsafe` only forwards to the system allocator.

mod common;

use core::num::NonZeroU32;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use common::{CountConfig, Counter};
use tessera_rl_engine::{ActionInput, EnvPool, PoolOptions, Reseed};
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
fn test_inline_reset_and_step_do_not_allocate() {
    let cfg = CountConfig {
        length: 1000,
        ..CountConfig::default()
    };
    let options = PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(1));
    let mut pool = EnvPool::<Counter>::new(cfg, options).unwrap();
    let actions = vec![1i32; 128];
    let ids = [3u32, 9, 27];
    let seeds = [Seed::new(5), Seed::new(6), Seed::new(7)];
    let n = allocations(|| {
        let _ = std::hint::black_box(pool.reset_all(Reseed::Keep));
        for _ in 0..20 {
            let input = ActionInput {
                discrete: &actions,
                continuous: &[],
            };
            let _ = std::hint::black_box(pool.step(input));
        }
        let _ = std::hint::black_box(pool.reset(&ids, Reseed::List(&seeds)));
        let _ = std::hint::black_box(pool.reset(&ids, Reseed::Base(Seed::new(2))));
    });
    assert_eq!(n, 0);
    assert!(pool.outputs().rewards().iter().any(|&r| r > 0.0));
}
