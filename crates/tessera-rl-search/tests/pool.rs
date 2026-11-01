// SPDX-License-Identifier: MIT OR Apache-2.0

//! The bounded `SearchPool` contract (ADR-0043 section 1) on Connect Four:
//! construction and the memory budget, batched expansion against the
//! reference board, terminal children, every rejected request leaving the
//! tree unchanged, strict capacity, deterministic eviction of the oldest
//! evictable subtree with its edge statistics kept, holds, re-rooting,
//! clearing, and determinism.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::{Tag, TagConfig};
use tessera_rl_connect4_ref::board::Position;
use tessera_rl_engine::{AutoReset, EnvPool, Masked, PoolOptions, Record, Reseed};
use tessera_rl_games::{Connect4, Connect4Config, Connect4State};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{Eviction, NodeHandle, SearchError, SearchOptions, SearchPool};

type Pool = SearchPool<Connect4, Masked>;

/// The record of a Connect Four game after `moves` (column digits, 0-based).
fn record(moves: &str) -> Record<Connect4> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(9))
    };
    let mut p = EnvPool::<Connect4, Masked>::with_masks(Connect4Config {}, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    for ch in moves.chars() {
        let me = p.state(0).unwrap().to_move();
        let mut act = [0, 0];
        act[me] = i32::try_from(ch.to_digit(10).unwrap()).unwrap();
        p.step(tessera_rl_engine::ActionInput {
            discrete: &act,
            continuous: &[],
        })
        .unwrap();
    }
    p.snapshot(0).unwrap()
}

fn options(capacity: u32, batch: u32, eviction: Eviction) -> SearchOptions {
    let mut o = SearchOptions::new(
        NonZeroU32::new(capacity).unwrap(),
        NonZeroU32::new(batch).unwrap(),
    );
    o.eviction = eviction;
    o
}

fn search_at(moves: &str, capacity: u32, batch: u32, eviction: Eviction) -> Pool {
    SearchPool::new(
        Connect4Config {},
        options(capacity, batch, eviction),
        &record(moves),
        EnvPool::with_masks,
    )
    .unwrap()
}

fn search(capacity: u32, batch: u32, eviction: Eviction) -> Pool {
    search_at("", capacity, batch, eviction)
}

/// Expands `(parent, action)` pairs; returns the children.
fn expand(s: &mut Pool, requests: &[(NodeHandle, u32)]) -> Vec<NodeHandle> {
    let mut out = vec![s.root(); requests.len()];
    s.expand(requests, &mut out).unwrap();
    out
}

/// Expands exactly `K` requests.
fn expand_n<const K: usize>(s: &mut Pool, requests: &[(NodeHandle, u32)]) -> [NodeHandle; K] {
    <[NodeHandle; K]>::try_from(expand(s, requests)).unwrap()
}

/// The environment state of the reference position after `moves`
/// (0-based column digits).
fn reference(moves: &str) -> Connect4State {
    let one_based: String = moves
        .chars()
        .map(|c| char::from_digit(c.to_digit(10).unwrap() + 1, 10).unwrap())
        .collect();
    let p = Position::from_moves(&one_based).unwrap();
    let mut s = Connect4State {
        moves: p.moves(),
        ..Connect4State::default()
    };
    let me = s.to_move();
    s.stones[me] = p.current();
    s.stones[1 - me] = p.opponent();
    s
}

/// The error of an expansion, and that it left the tree unchanged.
fn rejected(s: &mut Pool, requests: &[(NodeHandle, u32)], out_len: usize) -> SearchError {
    let before = s.tree_hash();
    let mut out = vec![s.root(); out_len];
    let err = s.expand(requests, &mut out).unwrap_err();
    assert_eq!(s.tree_hash(), before, "{err}: the tree changed");
    err
}

#[test]
fn test_search_pool_starts_with_only_the_root() {
    let s = search(16, 7, Eviction::Strict);
    assert_eq!(
        (s.live_nodes(), s.capacity(), s.batch(), s.actions()),
        (1, 16, 7, 7)
    );
    let root = s.root();
    assert_eq!(s.parent(root).unwrap(), None);
    assert!(!s.is_terminal(root).unwrap());
    assert_eq!(*s.record(root).unwrap().state(), Connect4State::default());
}

#[test]
fn test_search_pool_budget_error_names_the_bytes_per_node() {
    let mut o = options(1_000, 7, Eviction::Strict);
    o.budget_bytes = 1_000;
    let err =
        SearchPool::<Connect4, Masked>::new(Connect4Config {}, o, &record(""), EnvPool::with_masks)
            .err()
            .unwrap();
    let per_node = SearchPool::<Connect4, Masked>::bytes_per_node(7);
    assert_eq!(
        err,
        SearchError::Budget {
            bytes: per_node * 1_000,
            per_node,
            budget: 1_000
        }
    );
    assert!(err.to_string().contains("per node"));
}

#[test]
fn test_search_pool_rejects_a_batch_above_the_capacity() {
    let err = SearchPool::<Connect4, Masked>::new(
        Connect4Config {},
        options(4, 5, Eviction::Strict),
        &record(""),
        EnvPool::with_masks,
    )
    .err()
    .unwrap();
    assert!(matches!(
        err,
        SearchError::Options {
            option: "batch",
            ..
        }
    ));
}

#[test]
fn test_search_pool_rejects_imperfect_information_citing_adr_0035() {
    let mut p = EnvPool::<Tag, Masked>::with_masks(
        TagConfig::default(),
        PoolOptions::new(NonZeroU32::MIN, Seed::new(1)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let err = SearchPool::<Tag, Masked>::new(
        TagConfig::default(),
        options(8, 2, Eviction::Strict),
        &p.snapshot(0).unwrap(),
        EnvPool::with_masks,
    )
    .err()
    .unwrap();
    assert_eq!(err, SearchError::ImperfectInformation);
    assert!(err.to_string().contains("ADR-0035"));
}

#[test]
fn test_search_pool_batched_children_match_the_reference_board() {
    let mut s = search_at("33", 64, 7, Eviction::Strict);
    let root = s.root();
    let requests: Vec<_> = (0..7).map(|a| (root, a)).collect();
    let children = expand(&mut s, &requests);
    assert_eq!(s.live_nodes(), 8);
    for (a, &c) in children.iter().enumerate() {
        let moves = format!("33{a}");
        assert_eq!(*s.record(c).unwrap().state(), reference(&moves), "{moves}");
        assert_eq!(
            s.parent(c).unwrap(),
            Some((root, u32::try_from(a).unwrap()))
        );
        assert_eq!(s.child(root, u32::try_from(a).unwrap()).unwrap(), Some(c));
        assert!(
            s.edge_stats(root, u32::try_from(a).unwrap())
                .unwrap()
                .expanded
        );
    }
    // A second level, mixing parents in one batch.
    let grand = expand(
        &mut s,
        &[(children[0], 3), (children[6], 0), (children[3], 3)],
    );
    assert_eq!(*s.record(grand[0]).unwrap().state(), reference("3303"));
    assert_eq!(*s.record(grand[1]).unwrap().state(), reference("3360"));
    assert_eq!(*s.record(grand[2]).unwrap().state(), reference("3333"));
}

#[test]
fn test_search_pool_terminal_child_keeps_its_final_state_and_cannot_expand() {
    let mut s = search_at("010101", 8, 2, Eviction::Strict);
    let root = s.root();
    let [win, other] = expand_n(&mut s, &[(root, 0), (root, 2)]);
    assert!(s.is_terminal(win).unwrap());
    assert!(!s.is_terminal(other).unwrap());
    // Not reset: the scratch pool runs with auto-reset disabled.
    assert_eq!(s.record(win).unwrap().state().moves, 7);
    assert_eq!(s.rewards(win).unwrap(), [1.0, -1.0]);
    assert_eq!(s.rewards(other).unwrap(), [0.0, 0.0]);
    assert_eq!(rejected(&mut s, &[(win, 1)], 1), SearchError::TerminalNode);
}

#[test]
fn test_search_pool_rejected_requests_leave_the_tree_unchanged() {
    let mut s = search_at("333333", 8, 3, Eviction::Strict);
    let root = s.root();
    let [c] = expand_n(&mut s, &[(root, 0)]);
    let stale = c;
    // Column 3 is full: the mask rejects it.
    assert_eq!(
        rejected(&mut s, &[(root, 3)], 1),
        SearchError::IllegalAction { action: 3 }
    );
    assert_eq!(
        rejected(&mut s, &[(root, 7)], 1),
        SearchError::ActionOutOfRange {
            action: 7,
            actions: 7
        }
    );
    assert_eq!(
        rejected(&mut s, &[(root, 0)], 1),
        SearchError::AlreadyExpanded { action: 0 }
    );
    assert_eq!(
        rejected(&mut s, &[(root, 1), (root, 1)], 2),
        SearchError::DuplicateRequest { index: 1 }
    );
    assert_eq!(
        rejected(&mut s, &[(root, 1), (root, 2), (root, 4), (root, 5)], 4),
        SearchError::BatchTooLarge {
            requests: 4,
            batch: 3
        }
    );
    assert_eq!(
        rejected(&mut s, &[(root, 1)], 2),
        SearchError::OutputLength {
            expected: 1,
            got: 2
        }
    );
    // A valid request after an invalid one in the same call: nothing of
    // the valid one is applied.
    assert_eq!(
        rejected(&mut s, &[(root, 1), (root, 3)], 2),
        SearchError::IllegalAction { action: 3 }
    );
    assert!(!s.edge_stats(root, 1).unwrap().expanded);
    s.clear().unwrap();
    assert!(matches!(
        rejected(&mut s, &[(stale, 1)], 1),
        SearchError::StaleHandle { .. }
    ));
}

#[test]
fn test_search_pool_strict_mode_reports_capacity_exceeded() {
    let mut s = search(3, 2, Eviction::Strict);
    let root = s.root();
    expand(&mut s, &[(root, 0), (root, 1)]);
    assert_eq!(
        rejected(&mut s, &[(root, 2)], 1),
        SearchError::CapacityExceeded { capacity: 3 }
    );
}

#[test]
fn test_search_pool_evicts_the_oldest_subtree_and_keeps_its_edge_statistics() {
    let mut s = search(5, 2, Eviction::Oldest);
    let root = s.root();
    // Allocation order: a (1), b (2), then a's child a0 (3), b's child b0 (4).
    let [a, b] = expand_n(&mut s, &[(root, 0), (root, 1)]);
    let [a0, b0] = expand_n(&mut s, &[(a, 0), (b, 0)]);
    s.add_visits(root, 0, 3, 1.5).unwrap();
    s.add_visits(a, 0, 2, -1.0).unwrap();
    assert_eq!(s.live_nodes(), 5);
    // Full: expanding the root's column 2 evicts a with a0 (the oldest).
    let [c] = expand_n(&mut s, &[(root, 2)]);
    assert_eq!(s.live_nodes(), 4);
    for h in [a, a0] {
        assert!(matches!(s.record(h), Err(SearchError::StaleHandle { .. })));
    }
    let edge = s.edge_stats(root, 0).unwrap();
    assert!(!edge.expanded);
    assert_eq!((edge.visits, edge.value), (3, 1.5), "kept for selection");
    assert!(s.record(b0).is_ok() && s.record(c).is_ok());
    // A later visit re-expands the edge: a new node and a new handle.
    let [a2] = expand_n(&mut s, &[(root, 0)]);
    assert_ne!(a2, a);
    assert_eq!(*s.record(a2).unwrap().state(), reference("0"));
    assert_eq!(s.edge_stats(root, 0).unwrap().visits, 3);
}

#[test]
fn test_search_pool_never_evicts_held_nodes_their_ancestors_or_the_request_parents() {
    let mut s = search(4, 1, Eviction::Oldest);
    let root = s.root();
    let [a] = expand_n(&mut s, &[(root, 0)]);
    let [a0] = expand_n(&mut s, &[(a, 0)]);
    let [b] = expand_n(&mut s, &[(root, 1)]);
    // Holding a0 protects a; b is then the only evictable node.
    s.hold(a0).unwrap();
    let [c] = expand_n(&mut s, &[(root, 2)]);
    assert!(matches!(s.record(b), Err(SearchError::StaleHandle { .. })));
    assert!(s.record(a).is_ok() && s.record(a0).is_ok());
    // c's own expansion protects c (a request parent); a0 is held: full.
    assert_eq!(rejected(&mut s, &[(c, 0)], 1), SearchError::Exhausted);
    // Releasing a0 makes the a subtree evictable again.
    s.release(a0).unwrap();
    assert_eq!(s.release(a0).unwrap_err(), SearchError::NotHeld);
    expand(&mut s, &[(c, 0)]);
    assert!(matches!(s.record(a), Err(SearchError::StaleHandle { .. })));
}

#[test]
fn test_search_pool_reroot_keeps_the_subtree_and_frees_the_rest() {
    let mut s = search(16, 3, Eviction::Strict);
    let root = s.root();
    let [a, b, c] = expand_n(&mut s, &[(root, 0), (root, 1), (root, 2)]);
    let [a0, a1] = expand_n(&mut s, &[(a, 0), (a, 1)]);
    s.add_visits(a, 1, 4, 2.0).unwrap();
    // A held node outside the new subtree blocks it, atomically.
    s.hold(b).unwrap();
    let before = s.tree_hash();
    assert_eq!(
        s.reroot(a).unwrap_err(),
        SearchError::Held { slot: b.slot() }
    );
    assert_eq!(s.tree_hash(), before);
    s.release(b).unwrap();
    s.reroot(a).unwrap();
    assert_eq!(s.root(), a);
    assert_eq!(s.parent(a).unwrap(), None);
    assert_eq!(s.live_nodes(), 3);
    for h in [root, b, c] {
        assert!(s.record(h).is_err());
    }
    assert!(s.record(a0).is_ok() && s.record(a1).is_ok());
    assert_eq!(s.edge_stats(a, 1).unwrap().visits, 4);
}

#[test]
fn test_search_pool_clear_and_set_root() {
    let mut s = search(8, 2, Eviction::Strict);
    let root = s.root();
    let [a, _] = expand_n(&mut s, &[(root, 0), (root, 1)]);
    s.add_visits(root, 0, 1, 1.0).unwrap();
    s.clear().unwrap();
    assert_eq!(s.live_nodes(), 1);
    assert_eq!(s.edge_stats(root, 0).unwrap().visits, 0);
    assert!(s.record(a).is_err());
    let new = s.set_root(&record("4")).unwrap();
    assert!(s.record(root).is_err());
    assert_eq!(*s.record(new).unwrap().state(), reference("4"));
    assert_eq!(s.live_nodes(), 1);
}

/// The same expansions give the same tree whatever the batch split.
#[test]
fn test_search_pool_same_expansions_give_the_same_tree_for_any_batch() {
    let run = |batch: u32| {
        let mut s = search(40, batch, Eviction::Strict);
        let root = s.root();
        let mut requests: Vec<_> = (0..7).map(|a| (root, a)).collect();
        let mut children = Vec::new();
        for chunk in requests.chunks(usize::try_from(batch).unwrap()) {
            children.extend(expand(&mut s, chunk));
        }
        requests = children
            .iter()
            .take(3)
            .flat_map(|&c| (0..7).map(move |a| (c, a)))
            .collect();
        for chunk in requests.chunks(usize::try_from(batch).unwrap()) {
            expand(&mut s, chunk);
        }
        (s.tree_hash(), s.live_nodes())
    };
    let (h1, n1) = run(1);
    assert_eq!(n1, 29);
    for batch in [3, 7] {
        assert_eq!(run(batch), (h1, n1), "batch {batch}");
    }
}

/// A partial batch whose padding rows reuse a parent where column 0 is
/// full: strict checks (`debug-checks`) accept the padding (R38-01).
#[test]
fn test_search_pool_partial_batch_padding_is_legal_under_strict_checks() {
    let mut s = search_at("000000", 16, 7, Eviction::Strict);
    let root = s.root();
    let [c] = expand_n(&mut s, &[(root, 3)]);
    assert_eq!(*s.record(c).unwrap().state(), reference("0000003"));
}

#[test]
fn test_search_pool_add_visits_rejects_non_finite_values() {
    let mut s = search(4, 1, Eviction::Strict);
    let root = s.root();
    let before = s.tree_hash();
    for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            s.add_visits(root, 0, 1, v).unwrap_err(),
            SearchError::NonFiniteValue
        );
    }
    assert_eq!(s.tree_hash(), before);
}

/// `add_visits` computes under the canonical controls: a caller in round
/// toward zero gets the same tree as one in round to nearest (ADR-0037).
#[test]
fn test_search_pool_add_visits_is_canonical_under_hostile_rounding() {
    use tessera_rl_num::fp::hostile;
    let run = |hostile_mode: bool| {
        let mut s = search(4, 1, Eviction::Strict);
        let root = s.root();
        let mut add = || {
            for v in [0.1, 0.2, 0.7, -0.3, 1.0 / 3.0] {
                s.add_visits(root, 0, 1, v).unwrap();
            }
        };
        if hostile_mode {
            hostile::run(hostile::toward_zero, add);
        } else {
            add();
        }
        (
            s.tree_hash(),
            s.edge_stats(root, 0).unwrap().value.to_bits(),
        )
    };
    let canonical = run(false);
    assert_eq!(run(true), canonical);
    // The probe can tell: the same sums in round toward zero differ.
    let (direct, _) = hostile::run(hostile::toward_zero, || {
        let mut w = 0.0_f64;
        for v in [0.1, 0.2, 0.7, -0.3, 1.0 / 3.0] {
            w = std::hint::black_box(w) + v;
        }
        w.to_bits()
    });
    assert_ne!(direct, canonical.1, "round toward zero changes this sum");
}
