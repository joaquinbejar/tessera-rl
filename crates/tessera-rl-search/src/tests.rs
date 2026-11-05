// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exhaustion of the pool's counters (ADR-0043 section 1), with the
//! counters seeded near their limits: each case is a typed error or a
//! documented retirement, and an error leaves the tree unchanged.

use core::num::NonZeroU32;

use tessera_rl_engine::{EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_num::identity::Seed;

use crate::{Eviction, NodeHandle, SearchError, SearchOptions, SearchPool};

fn search(capacity: u32, eviction: Eviction) -> SearchPool<Connect4, Masked> {
    let mut p = EnvPool::<Connect4, Masked>::with_masks(
        Connect4Config {},
        PoolOptions::new(NonZeroU32::MIN, Seed::new(3)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let mut o = SearchOptions::new(NonZeroU32::new(capacity).unwrap(), NonZeroU32::MIN);
    o.eviction = eviction;
    SearchPool::new(
        Connect4Config {},
        o,
        &p.snapshot(0).unwrap(),
        EnvPool::with_masks,
    )
    .unwrap()
}

fn expand(
    s: &mut SearchPool<Connect4, Masked>,
    h: NodeHandle,
    a: u32,
) -> Result<NodeHandle, SearchError> {
    let mut out = [h];
    s.expand(&[(h, a)], &mut out)?;
    Ok(out[0])
}

#[test]
fn test_search_counter_exhaustion_is_a_typed_error_and_atomic() {
    let mut s = search(4, Eviction::Strict);
    s.next_alloc = u64::MAX;
    let root = s.root();
    let before = s.tree_hash();
    assert_eq!(
        expand(&mut s, root, 0).unwrap_err(),
        SearchError::CounterExhausted
    );
    assert_eq!(s.tree_hash(), before);
    let r = *s.record(root).unwrap();
    assert_eq!(s.set_root(&r).unwrap_err(), SearchError::CounterExhausted);
    assert_eq!(s.tree_hash(), before);
}

#[test]
fn test_search_generation_exhaustion_retires_the_slot() {
    let mut s = search(2, Eviction::Oldest);
    let root = s.root();
    let free = *s.free.last().unwrap();
    s.nodes[free as usize].generation = u32::MAX;
    let child = expand(&mut s, root, 0).unwrap();
    assert_eq!((child.slot(), child.generation()), (free, u32::MAX));
    // Freeing a slot at the last generation retires it: it never wraps
    // to a generation an old handle could match.
    s.clear().unwrap();
    assert_eq!(s.free.len(), 0);
    assert!(matches!(
        s.record(child),
        Err(SearchError::StaleHandle { .. })
    ));
    // The retired slot no longer counts: nothing is free or evictable.
    assert_eq!(expand(&mut s, root, 0).unwrap_err(), SearchError::Exhausted);
}

#[test]
fn test_search_hold_and_visit_counters_report_exhaustion() {
    let mut s = search(2, Eviction::Strict);
    let root = s.root();
    s.nodes[root.slot() as usize].holds = u32::MAX;
    assert_eq!(s.hold(root).unwrap_err(), SearchError::HoldsExhausted);
    s.add_visits(root, 0, u32::MAX, 0.0).unwrap();
    let before = s.tree_hash();
    assert_eq!(
        s.add_visits(root, 0, 1, 1.0).unwrap_err(),
        SearchError::VisitsExhausted
    );
    assert_eq!(s.tree_hash(), before);
}

#[test]
fn test_search_set_root_with_no_usable_slot_changes_nothing() {
    // One slot, the root's, at the last generation: freeing it would
    // retire it and leave no slot for the new root (R38-02).
    let mut s = search(1, Eviction::Strict);
    let root = s.root();
    s.nodes[root.slot() as usize].generation = u32::MAX;
    let root = s.root();
    let record = *s.record(root).unwrap();
    let before = s.tree_hash();
    assert_eq!(
        s.set_root(&record).unwrap_err(),
        SearchError::CapacityExceeded { capacity: 1 }
    );
    assert_eq!(s.tree_hash(), before);
    assert_eq!(s.root(), root);
    assert!(s.record(root).is_ok());
}

/// The MCTS route of C1: a batch of terminal leaves only stages no request
/// and commits; the commit must not replay the previous eviction.
#[test]
fn test_search_commit_after_an_empty_stage_replays_nothing() {
    let mut s = search(3, Eviction::Oldest);
    let root = s.root();
    let _a = expand(&mut s, root, 0).unwrap();
    let b = expand(&mut s, root, 1).unwrap();
    let c = expand(&mut s, root, 2).unwrap();
    // A held (pending) node, edge statistics and the free list are in the
    // tree hash, with every record, node and handle generation.
    s.hold(c).unwrap();
    s.add_visits(root, 1, 3, 0.5).unwrap();
    let (free, records) = (
        s.free.clone(),
        (s.record(b).unwrap().hash(), s.record(c).unwrap().hash()),
    );
    let before = s.tree_hash();
    s.stage(&[]).unwrap();
    s.commit_staged(&mut []);
    assert_eq!(s.tree_hash(), before);
    assert_eq!(s.free, free, "free list");
    assert_eq!(
        (s.record(b).unwrap().hash(), s.record(c).unwrap().hash()),
        records
    );
    assert_eq!(s.edge_stats(root, 1).unwrap().visits, 3);
    s.release(c).unwrap();
}

/// The privileged label alone changes the tree hash: one tree, one
/// record, only the mode differs.
#[test]
fn test_search_privileged_label_alone_changes_the_tree_hash() {
    let mut s = search(4, Eviction::Strict);
    let root = s.root();
    let _child = expand(&mut s, root, 3).unwrap();
    let unlabelled = s.tree_hash();
    s.mode = crate::determinize::RootMode::Privileged;
    assert!(s.is_privileged());
    assert_ne!(s.tree_hash(), unlabelled);
    s.mode = crate::determinize::RootMode::Full;
    assert_eq!(s.tree_hash(), unlabelled);
}
