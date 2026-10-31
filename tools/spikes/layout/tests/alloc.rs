// SPDX-License-Identifier: MIT OR Apache-2.0

//! The measured spike paths allocate nothing in the harness: stepping with
//! a pregenerated action table, the fork-heavy cycle and whole-pool copies
//! (codex, R-PR12). Allocations there would dilute the layout ratios.
//!
//! This file installs a counting global allocator for this test binary
//! only; its `unsafe` only forwards to the system allocator.

use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
use std::cell::Cell;

use tessera_rl_classic::{CartPole, CartPoleConfig, Tag, TagConfig};
use tessera_rl_spike_layout::kernel::{SpikeEnv, Work};
use tessera_rl_spike_layout::layouts::{Blocks, Columns, Combined, Layout, Parallel};
use tessera_rl_spike_layout::workload::{ActionTable, fork_heavy, run};

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

struct Counting;

#[expect(unsafe_code, reason = "test-only counting allocator")]
// SAFETY: every method forwards to `System` with the same arguments; the
// counters are const-initialized thread locals, which never allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
        count();
        // SAFETY: forwarded unchanged; the caller upholds `alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: AllocLayout, new_size: usize) -> *mut u8 {
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

fn zero_alloc<E: SpikeEnv, L: Layout<E>>(cfg: &E::Config, choices: u32) {
    const N: usize = 65;
    let mut layout = L::new(cfg, N, 3);
    let mut copy = L::new(cfg, N, 4);
    let mut work = Work::new::<E>(cfg, N);
    let mut table = ActionTable::new(N * E::MAX_AGENTS, choices, 16);
    // Warm up: episodes end and reset inside the measured window too.
    run::<E, L>(&mut layout, cfg, &mut work, &mut table, 0, 50, 20);
    let n = allocations(|| {
        run::<E, L>(&mut layout, cfg, &mut work, &mut table, 50, 100, 20);
        fork_heavy::<E, L>(&mut layout, cfg, &mut work, &mut table, 64, 10, 20);
        layout.copy_all(&mut copy);
        copy.copy_all(&mut layout);
        for i in 0..N {
            if let Some((h, s)) = layout.get(i) {
                copy.set(i, &h, &s);
            }
        }
    });
    assert_eq!(n, 0, "{} allocated {n} times", L::NAME);
}

#[test]
fn test_spike_measured_paths_do_not_allocate() {
    let cp = CartPoleConfig::default();
    zero_alloc::<CartPole, Parallel<CartPole>>(&cp, 2);
    zero_alloc::<CartPole, Combined<CartPole>>(&cp, 2);
    zero_alloc::<CartPole, Columns<CartPole>>(&cp, 2);
    zero_alloc::<CartPole, Blocks<CartPole>>(&cp, 2);
    for tag in [
        TagConfig::default(),
        TagConfig {
            pursuers: 4,
            evaders: 4,
            slip: 300,
            ..TagConfig::default()
        },
    ] {
        zero_alloc::<Tag, Parallel<Tag>>(&tag, 5);
        zero_alloc::<Tag, Combined<Tag>>(&tag, 5);
        zero_alloc::<Tag, Columns<Tag>>(&tag, 5);
        zero_alloc::<Tag, Blocks<Tag>>(&tag, 5);
    }
}
