// SPDX-License-Identifier: MIT OR Apache-2.0

//! A pool built by `make_host` allocates nothing per step through the box
//! (ADR-0065 item 3), counted with a global allocator of this binary.

use core::num::NonZeroU32;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

use tessera_rl_core::ConfigSource;
use tessera_rl_engine::{ActionInput, MakeOptions, PoolOptions, Reseed, make_host};
use tessera_rl_num::identity::Seed;
use tessera_rl_test_host_env::COUNTER_ID;

struct Counting;

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards to the system allocator, counting calls.
#[expect(unsafe_code, reason = "a counting test allocator")]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract, forwarded.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn test_factory_pool_steps_allocate_nothing() {
    assert_eq!(tessera_rl_test_host_glue::linked(), COUNTER_ID);
    let options = MakeOptions::new(PoolOptions::new(NonZeroU32::new(8).unwrap(), Seed::new(1)));
    let mut pool = make_host(COUNTER_ID, &ConfigSource::empty(), &options).unwrap();
    pool.reset_all(Reseed::Keep).unwrap();
    let actions = vec![1i32; 16];
    // One warm-up step, then count.
    pool.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap();
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..50 {
        pool.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        })
        .unwrap();
    }
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed) - before, 0);
}
