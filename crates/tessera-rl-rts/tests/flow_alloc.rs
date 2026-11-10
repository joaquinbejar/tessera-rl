// SPDX-License-Identifier: MIT OR Apache-2.0

//! Building and descending a flow field never allocates (`rts-design.md`
//! section 5; SPEC zero-allocation hot path). This file installs a counting
//! global allocator for this test binary only; its `unsafe` only forwards
//! to the system allocator.

#![expect(clippy::unwrap_used, reason = "test fixtures")]

use core::num::NonZeroU32;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use tessera_rl_rts::flow::Field;
use tessera_rl_rts::mapgen::generate;
use tessera_rl_rts::world::{Grid, Team, Terrain};

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

fn draw(seed: u64) -> impl FnMut(NonZeroU32) -> u32 {
    let mut s = seed | 1;
    move |bound| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        u32::try_from(s % u64::from(bound.get())).unwrap()
    }
}

#[test]
fn test_flow_alloc_counter_detects_heap_use() {
    assert!(allocations(|| drop(std::hint::black_box(vec![1u8; 16]))) > 0);
}

#[test]
fn test_flow_alloc_build_and_descend_do_not_allocate() {
    let g = Grid::new(32).unwrap();
    let map = generate(g, draw(5));
    let mut field = Box::new(Field::default());
    let ore = [0u16, 17, 400, 1023];
    let n = allocations(|| {
        for _ in 0..8 {
            field.build(g, map.cores, |c| {
                map.terrain[usize::from(c)] == Terrain::Open as u8
            });
            field.build(g, ore, |c| {
                map.terrain[usize::from(c)] == Terrain::Open as u8
            });
            for c in 0..g.cells() {
                let _ = std::hint::black_box(field.descend(g, c, Team::One));
            }
        }
    });
    assert_eq!(n, 0);
}
