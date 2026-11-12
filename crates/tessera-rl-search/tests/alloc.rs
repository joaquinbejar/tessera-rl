// SPDX-License-Identifier: MIT OR Apache-2.0

//! Expansion, eviction, re-rooting and clearing never allocate after
//! construction (ADR-0004, ADR-0043 section 1).
//!
//! This file installs a counting global allocator for this test binary
//! only; it is test code, not one of the audited `unsafe` modules, and its
//! `unsafe` only forwards to the system allocator.

use core::num::NonZeroU32;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use tessera_rl_engine::{EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{Eviction, SearchOptions, SearchPool};

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
fn test_search_expand_evict_reroot_and_clear_do_not_allocate() {
    let mut p = EnvPool::<Connect4, Masked>::with_masks(
        Connect4Config {},
        PoolOptions::new(NonZeroU32::MIN, Seed::new(1)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let root_record = p.snapshot(0).unwrap();
    let mut o = SearchOptions::new(NonZeroU32::new(20).unwrap(), NonZeroU32::new(7).unwrap());
    o.eviction = Eviction::Oldest;
    let mut s = SearchPool::new(Connect4Config {}, o, &root_record, EnvPool::with_masks).unwrap();
    let root = s.root();
    let mut out = [root; 7];
    let mut requests = [(root, 0); 7];
    let n = allocations(|| {
        // Level one, then enough grandchildren to force evictions.
        for (a, r) in (0u32..).zip(requests.iter_mut()) {
            *r = (root, a);
        }
        s.expand(&requests, &mut out).unwrap();
        let children = out;
        for c in children {
            for (a, r) in (0u32..).zip(requests.iter_mut()) {
                *r = (c, a);
            }
            // Fails with a stale handle once c itself was evicted.
            let _ = s.expand(&requests, &mut out);
        }
        s.hold(root).unwrap();
        s.release(root).unwrap();
        let first = s.child(root, 6).unwrap().unwrap_or(root);
        s.reroot(first).unwrap();
        s.clear().unwrap();
        s.set_root(&root_record).unwrap();
        std::hint::black_box(s.tree_hash());
    });
    assert_eq!(n, 0, "the search pool allocated {n} times");
    assert_eq!(s.live_nodes(), 1);
}

struct Uniform;

impl tessera_rl_search::Evaluator<Connect4> for Uniform {
    fn evaluate(
        &mut self,
        _: &[tessera_rl_engine::Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), tessera_rl_search::EvaluatorError> {
        priors.fill(1.0 / 7.0);
        values.fill(0.0);
        Ok(())
    }
}

#[test]
fn test_search_mcts_run_with_eviction_and_noise_does_not_allocate() {
    let mut p = EnvPool::<Connect4, Masked>::with_masks(
        Connect4Config {},
        PoolOptions::new(NonZeroU32::MIN, Seed::new(1)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let root = p.snapshot(0).unwrap();
    let mut o = SearchOptions::new(NonZeroU32::new(256).unwrap(), NonZeroU32::new(8).unwrap());
    o.eviction = Eviction::Oldest;
    let pool = SearchPool::new(Connect4Config {}, o, &root, EnvPool::with_masks).unwrap();
    let mut cfg = tessera_rl_search::PuctConfig::default();
    cfg.simulations = 600;
    cfg.noise = Some(tessera_rl_search::RootNoise {
        alpha: 0.3,
        epsilon: 0.25,
    });
    let mut m = tessera_rl_search::Mcts::new(pool, cfg).unwrap();
    let n = allocations(|| {
        for _ in 0..3 {
            m.run(&mut Uniform).unwrap();
            std::hint::black_box(m.select_action(1.0).unwrap());
            m.pool_mut().set_root(&root).unwrap();
        }
    });
    assert_eq!(n, 0, "a search call allocated {n} times");
}

/// The Gumbel root (draws, top-k, halving, completed Q, `pi'`, the root
/// re-evaluation after a reroot) allocates nothing per call.
#[test]
#[expect(clippy::many_single_char_names, reason = "as the tests above")]
fn test_search_mcts_gumbel_run_with_reroot_does_not_allocate() {
    let mut p = EnvPool::<Connect4, Masked>::with_masks(
        Connect4Config {},
        PoolOptions::new(NonZeroU32::MIN, Seed::new(1)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let root = p.snapshot(0).unwrap();
    let mut o = SearchOptions::new(NonZeroU32::new(256).unwrap(), NonZeroU32::new(8).unwrap());
    o.eviction = Eviction::Oldest;
    let pool = SearchPool::new(Connect4Config {}, o, &root, EnvPool::with_masks).unwrap();
    let mut cfg = tessera_rl_search::PuctConfig::default();
    cfg.simulations = 300;
    cfg.root = tessera_rl_search::RootSelection::Gumbel(tessera_rl_search::GumbelConfig::default());
    let mut m = tessera_rl_search::Mcts::new(pool, cfg).unwrap();
    let n = allocations(|| {
        for _ in 0..3 {
            m.run(&mut Uniform).unwrap();
            let a = m.select_action(0.0).unwrap();
            std::hint::black_box(m.improved_policy());
            let r = m.pool().root();
            let child = m.pool().child(r, a).unwrap().unwrap();
            m.pool_mut().reroot(child).unwrap();
            m.run(&mut Uniform).unwrap();
            m.pool_mut().set_root(&root).unwrap();
        }
    });
    assert_eq!(n, 0, "a Gumbel search call allocated {n} times");
}

/// A native evaluator on the batch path: it reads every observation and
/// mask (as a Python bridge would) and allocates nothing.
struct Reads;

impl tessera_rl_search::Evaluator<Connect4> for Reads {
    fn evaluate(
        &mut self,
        _: &[tessera_rl_engine::Record<Connect4>],
        _: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), tessera_rl_search::EvaluatorError> {
        Err(tessera_rl_search::EvaluatorError::NoLegalMass { leaf: 0 })
    }

    fn evaluate_batch(
        &mut self,
        batch: &mut tessera_rl_search::LeafBatch<'_, Connect4>,
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), tessera_rl_search::EvaluatorError> {
        let ones = batch
            .observations()
            .iter()
            .map(|&b| u64::from(b))
            .sum::<u64>()
            + batch.legal().iter().map(|&l| u64::from(l)).sum::<u64>();
        std::hint::black_box(ones);
        priors.fill(1.0 / 7.0);
        values.fill(0.0);
        Ok(())
    }
}

#[test]
#[expect(clippy::many_single_char_names, reason = "test setup")]
fn test_search_mcts_batch_evaluator_path_does_not_allocate_and_leaves_no_virtual_loss() {
    let mut p = EnvPool::<Connect4, Masked>::with_masks(
        Connect4Config {},
        PoolOptions::new(NonZeroU32::MIN, Seed::new(1)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let root = p.snapshot(0).unwrap();
    let mut o = SearchOptions::new(NonZeroU32::new(128).unwrap(), NonZeroU32::new(8).unwrap());
    o.eviction = Eviction::Oldest;
    let pool = SearchPool::new(Connect4Config {}, o, &root, EnvPool::with_masks).unwrap();
    let mut cfg = tessera_rl_search::PuctConfig::default();
    cfg.simulations = 400;
    let mut m = tessera_rl_search::Mcts::new(pool, cfg).unwrap();
    let n = allocations(|| {
        for _ in 0..3 {
            m.run(&mut Reads).unwrap();
            m.pool_mut().set_root(&root).unwrap();
        }
        m.run(&mut Reads).unwrap();
    });
    assert_eq!(n, 0, "the batch evaluator path allocated {n} times");
    let r = m.pool().root();
    for a in 0..7 {
        assert_eq!(m.pool().edge_stats(r, a).unwrap().virtual_loss, 0);
    }
}
