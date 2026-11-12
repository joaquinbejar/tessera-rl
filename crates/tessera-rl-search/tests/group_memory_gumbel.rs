// SPDX-License-Identifier: MIT OR Apache-2.0

//! A Gumbel slot's `SearchGroup` bound exceeds the same PUCT slot's bound
//! by exactly `GUMBEL_BYTES_PER_ACTION` per action, and that term covers
//! what one more Gumbel tree retains beyond one more PUCT tree. This binary holds a single measuring test, so no other test's thread
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
use tessera_rl_engine::Record;
use tessera_rl_games::Connect4;
use tessera_rl_search::GUMBEL_BYTES_PER_ACTION;

/// Gumbel retained bytes minus PUCT retained bytes for `trees` trees, with
/// the literal bound asserts.
fn gumbel_extra(root: &Record<Connect4>, trees: usize, capacity: u32, batch: u32) -> i64 {
    let (puct_live, puct) = measured(root, trees, capacity, batch, false);
    let (live, bound) = measured(root, trees, capacity, batch, true);
    let t = u64::try_from(trees).unwrap();
    // 128 bytes per action, 7 actions, written as numbers.
    assert_eq!(bound - puct, 128 * 7 * t, "trees {trees}");
    assert!(
        i64::try_from(bound).unwrap() >= live,
        "trees {trees} capacity {capacity} batch {batch}: bound {bound} < retained {live}"
    );
    live - puct_live
}

/// A Gumbel slot's bound exceeds the same slot's PUCT bound by exactly
/// `GUMBEL_BYTES_PER_ACTION` per action, and that term covers what one
/// more Gumbel tree retains beyond one more PUCT tree.
#[test]
fn test_group_memory_gumbel_term_is_exact_and_covers_the_retained_heap() {
    assert_eq!(GUMBEL_BYTES_PER_ACTION, 128);
    let root = record();
    for (capacity, batch) in [(8, 8), (64, 1), (4_096, 16)] {
        // (G - P) at two trees minus (G - P) at one tree cancels any
        // constant on either side.
        let per_tree =
            gumbel_extra(&root, 2, capacity, batch) - gumbel_extra(&root, 1, capacity, batch);
        assert!(
            per_tree > 0 && u64::try_from(per_tree).unwrap() <= GUMBEL_BYTES_PER_ACTION * 7,
            "capacity {capacity} batch {batch}: a Gumbel tree retains {per_tree} more bytes, \
             term {}",
            GUMBEL_BYTES_PER_ACTION * 7
        );
        // Pins the present footprint of the Gumbel buffers (4 x 28 + 5 x 56
        // + 1 x 112 bytes), not a bound: update it with those buffers.
        assert_eq!(per_tree, 504, "capacity {capacity} batch {batch}");
    }
}
