// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `SearchGroup` memory bound (ADR-0045 section 1) is an upper bound on
//! the bytes a group and its trees actually retain, measured with a
//! counting allocator. The Gumbel term has its own binary,
//! `group_memory_gumbel.rs`. This binary holds a single measuring test, so no other test's thread
//! runs inside a measurement window; the window fails if any thread other
//! than the measuring one and the harness `main` allocates
//! (`support/counting.rs`).

#![expect(
    clippy::unwrap_used,
    reason = "test helpers: a failure is the assertion"
)]

#[path = "support/counting.rs"]
mod counting;
use counting::{measured, record};

#[test]
fn test_group_memory_bound_covers_the_retained_heap() {
    let root = record();
    for (trees, capacity, batch) in [
        (1, 8, 8),
        (4, 8, 8),
        (4, 64, 1),
        (2, 1_024, 8),
        (3, 4_096, 16),
        (8, 8_192, 8),
    ] {
        let (live, bound) = measured(&root, trees, capacity, batch, false);
        assert!(
            i64::try_from(bound).unwrap() >= live,
            "trees {trees} capacity {capacity} batch {batch}: bound {bound} < retained {live}"
        );
    }
}
