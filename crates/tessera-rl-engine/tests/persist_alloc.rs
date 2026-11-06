// SPDX-License-Identifier: MIT OR Apache-2.0

//! A refused persisted file allocates nothing before its bounds are checked
//! (ADR-0011). One test: the counter is global.

mod common;

use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;
use core::num::NonZeroU32;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::System;

use common::{CountConfig, Counter};
use tessera_rl_core::Persist;
use tessera_rl_core::encoding::checksum;
use tessera_rl_engine::{DecodeLimits, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

struct Counting;
static BIGGEST: AtomicUsize = AtomicUsize::new(0);

std::thread_local! {
    /// Only the test's own thread counts, not the harness's.
    static COUNT_HERE: Cell<bool> = const { Cell::new(false) };
}

// SAFETY: forwards to `System`; only records the size.
#[expect(unsafe_code, reason = "a counting allocator for this test")]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNT_HERE.with(Cell::get) {
            BIGGEST.fetch_max(l.size(), Ordering::Relaxed);
        }
        // SAFETY: the caller's layout.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: the caller's pointer and layout.
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

#[test]
fn test_persist_refused_counts_never_reach_the_allocator() {
    let options = PoolOptions::new(NonZeroU32::new(4).unwrap(), Seed::new(1));
    let mut p = EnvPool::<Counter, Masked>::with_masks(CountConfig::default(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let good = p.encode_with(&[0], "fixture", None).unwrap();
    for (max_instances, count) in [(1 << 24, 5u32), (u32::MAX, u32::MAX)] {
        let mut bad = good.clone();
        bad[40..44].copy_from_slice(&count.to_le_bytes());
        let n = bad.len() - 8;
        let sum = checksum(&bad[..n]);
        bad[n..].copy_from_slice(&sum.to_le_bytes());
        BIGGEST.store(0, Ordering::Relaxed);
        let limits = DecodeLimits {
            max_instances,
            max_decoded_bytes: u64::MAX,
        };
        COUNT_HERE.with(|c| c.set(true));
        let refused = p.decode_with(&bad, limits, None).is_err();
        COUNT_HERE.with(|c| c.set(false));
        assert!(refused);
        assert_eq!(BIGGEST.load(Ordering::Relaxed), 0, "count {count}");
    }
}

impl Persist for Counter {
    const ENV_ID: &'static str = "tessera/Counter-v1";
    const ENV_VERSION: u32 = 1;
}
