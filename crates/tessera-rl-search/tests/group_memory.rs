// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `SearchGroup` memory bound (ADR-0045 section 1) is an upper bound on
//! the bytes a group and its trees actually retain, measured with a
//! counting allocator. This file installs a counting global allocator for
//! this test binary only; it is test code, not one of the audited `unsafe`
//! modules, and its `unsafe` only forwards to the system allocator.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;
use core::sync::atomic::{AtomicI64, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};

use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Record, Reseed};
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{
    DEFAULT_GROUP_BUDGET, Eviction, GUMBEL_BYTES_PER_ACTION, GumbelConfig, Mcts, PuctConfig,
    RootSelection, SearchGroup, SearchOptions, SearchPool,
};

/// Live heap bytes of the whole process; the tests that measure it hold
/// `SERIAL`, so they run one at a time.
static LIVE: AtomicI64 = AtomicI64::new(0);

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Counting;

fn delta(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

#[expect(unsafe_code, reason = "test-only counting allocator")]
// SAFETY: every method forwards to `System` with the same arguments and
// only adds an atomic counter update, which never allocates.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(delta(layout.size()), Ordering::Relaxed);
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(delta(layout.size()), Ordering::Relaxed);
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(delta(layout.size()), Ordering::Relaxed);
        // SAFETY: forwarded unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_add(delta(new_size) - delta(layout.size()), Ordering::Relaxed);
        // SAFETY: forwarded unchanged; the caller upholds `realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn record(moves: &str) -> Record<Connect4> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(21))
    };
    let mut p = EnvPool::<Connect4, Masked>::with_masks(Connect4Config {}, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    for ch in moves.chars() {
        let me = p.state(0).unwrap().to_move();
        let mut act = [0, 0];
        act[me] = i32::try_from(ch.to_digit(10).unwrap()).unwrap();
        p.step(ActionInput {
            discrete: &act,
            continuous: &[],
        })
        .unwrap();
    }
    p.snapshot(0).unwrap()
}

fn config(simulations: u32) -> PuctConfig {
    let mut c = PuctConfig::default();
    c.simulations = simulations;
    c
}

/// The root record is built before counting starts.
fn measured(root: &Record<Connect4>, trees: usize, capacity: u32, batch: u32) -> (i64, u64) {
    measured_with(root, trees, capacity, batch, false)
}

fn measured_with(
    root: &Record<Connect4>,
    trees: usize,
    capacity: u32,
    batch: u32,
    gumbel: bool,
) -> (i64, u64) {
    let before = LIVE.load(Ordering::Relaxed);
    let mut cfg = config(32);
    cfg.batch_size = batch;
    if gumbel {
        cfg.root = RootSelection::Gumbel(GumbelConfig::default());
    }
    let slots = (0..trees)
        .map(|_| {
            let mut o = SearchOptions::new(
                NonZeroU32::new(capacity).unwrap(),
                NonZeroU32::new(batch).unwrap(),
            );
            o.eviction = Eviction::Oldest;
            let pool = SearchPool::new(Connect4Config {}, o, root, EnvPool::with_masks).unwrap();
            Mcts::new(pool, cfg).unwrap()
        })
        .collect();
    let mut o = SearchOptions::new(
        NonZeroU32::new(capacity).unwrap(),
        NonZeroU32::new(batch).unwrap(),
    );
    o.eviction = Eviction::Oldest;
    let shapes = vec![(o, cfg); trees];
    let predicted =
        SearchGroup::<Connect4, Masked>::bound_bytes(&Connect4Config {}, &shapes).unwrap();
    let g = SearchGroup::new(&Connect4Config {}, slots, DEFAULT_GROUP_BUDGET).unwrap();
    assert_eq!(
        g.memory_bytes(),
        predicted,
        "checked before and after construction alike"
    );
    let live = LIVE.load(Ordering::Relaxed) - before;
    let bound = g.memory_bytes();
    drop(g);
    (live, bound)
}

#[test]
fn test_group_memory_bound_covers_the_retained_heap() {
    let _serial = SERIAL.lock().unwrap();
    let root = record("");
    for (trees, capacity, batch) in [
        (1, 8, 8),
        (4, 8, 8),
        (4, 64, 1),
        (2, 1_024, 8),
        (3, 4_096, 16),
        (8, 8_192, 8),
    ] {
        let (live, bound) = measured(&root, trees, capacity, batch);
        assert!(
            i64::try_from(bound).unwrap() >= live,
            "trees {trees} capacity {capacity} batch {batch}: bound {bound} < retained {live}"
        );
    }
}

/// A Gumbel slot's bound exceeds the same slot's PUCT bound by exactly
/// `GUMBEL_BYTES_PER_ACTION` per action, and still covers the retained
/// heap.
#[test]
fn test_group_memory_gumbel_term_is_exact_and_covers_the_retained_heap() {
    let _serial = SERIAL.lock().unwrap();
    let root = record("");
    for (trees, capacity, batch) in [(1, 8, 8), (4, 64, 1), (3, 4_096, 16)] {
        let (puct_live, puct) = measured_with(&root, trees, capacity, batch, false);
        let (live, bound) = measured_with(&root, trees, capacity, batch, true);
        let t = u64::try_from(trees).unwrap();
        // 128 bytes per action, 7 actions, written as numbers.
        assert_eq!(bound - puct, 128 * 7 * t, "trees {trees}");
        assert_eq!(GUMBEL_BYTES_PER_ACTION, 128);
        // The Gumbel term alone covers what a Gumbel tree retains beyond
        // the same PUCT tree (72 bytes per action, measured 504 per tree).
        let extra = live - puct_live;
        assert!(
            extra > 0 && u64::try_from(extra).unwrap() <= GUMBEL_BYTES_PER_ACTION * 7 * t,
            "trees {trees}: Gumbel retains {extra} more bytes, term {}",
            GUMBEL_BYTES_PER_ACTION * 7 * t
        );
        assert!(
            i64::try_from(bound).unwrap() >= live,
            "trees {trees} capacity {capacity} batch {batch}: bound {bound} < retained {live}"
        );
    }
}
