// SPDX-License-Identifier: MIT OR Apache-2.0

//! Building and descending a flow field never allocates (`rts-design.md`
//! section 5; SPEC zero-allocation hot path). The counting allocator of
//! `support/counting.rs` attributes allocator calls by thread (issue #43).
//! This binary holds a single measuring test, which runs each scenario (the
//! former tests, by name) with its own objects and windows, and reports
//! every failing scenario together.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures"
)]

use core::num::NonZeroU32;

use tessera_rl_rts::flow::Field;
use tessera_rl_rts::mapgen::generate;
use tessera_rl_rts::world::{Grid, Team, Terrain};

#[path = "support/counting.rs"]
mod counting;
use counting::window;

/// Allocation calls of the counted threads while `f` runs.
fn allocations(f: impl FnOnce()) -> u64 {
    window(f).1.calls
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

fn test_flow_alloc_counter_detects_heap_use() {
    assert!(allocations(|| drop(std::hint::black_box(vec![1u8; 16]))) > 0);
}

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

/// The scenarios this binary runs.
const SCENARIOS: usize = 2;

#[test]
fn test_flow_alloc_scenarios_do_not_allocate() {
    counting::scenarios(
        SCENARIOS,
        &[
            (
                "test_flow_alloc_counter_detects_heap_use",
                test_flow_alloc_counter_detects_heap_use,
            ),
            (
                "test_flow_alloc_build_and_descend_do_not_allocate",
                test_flow_alloc_build_and_descend_do_not_allocate,
            ),
        ],
    );
}
