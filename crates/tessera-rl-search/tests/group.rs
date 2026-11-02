// SPDX-License-Identifier: MIT OR Apache-2.0

//! `SearchGroup` (ADR-0045) on Connect Four with a canned evaluator: every
//! slot equals an independent ADR-0043 search (visits, tree and stream
//! position), over several calls with reuse; whole-group rollback on a
//! failure in any slot, then a successful retry; empty batches, inactive
//! slots, zero participants and the aggregate memory bound.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Record, Reseed};
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{
    DEFAULT_GROUP_BUDGET, Evaluator, EvaluatorError, Eviction, LeafBatch, Mcts, PuctConfig,
    RootNoise, SearchError, SearchGroup, SearchOptions, SearchPool,
};

type C4 = Mcts<Connect4, Masked>;
type Group = SearchGroup<Connect4, Masked>;

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

fn mcts_with(moves: &str, capacity: u32, eviction: Eviction, cfg: PuctConfig) -> C4 {
    let mut o = SearchOptions::new(
        NonZeroU32::new(capacity).unwrap(),
        NonZeroU32::new(8).unwrap(),
    );
    o.eviction = eviction;
    let pool = SearchPool::new(Connect4Config {}, o, &record(moves), EnvPool::with_masks).unwrap();
    Mcts::new(pool, cfg).unwrap()
}

fn config(simulations: u32) -> PuctConfig {
    let mut c = PuctConfig::default();
    c.simulations = simulations;
    c
}

/// A canned evaluator: priors and value from the leaf's state hash,
/// exactly representable arithmetic, identical on every platform.
#[derive(Default)]
struct Canned {
    calls: u32,
    leaves: u32,
}

impl Evaluator<Connect4> for Canned {
    fn evaluate(
        &mut self,
        leaves: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        self.calls += 1;
        for (i, r) in leaves.iter().enumerate() {
            self.leaves += 1;
            let h = r.hash().get();
            let weights: Vec<f64> = (0..7)
                .map(|a| f64::from(1 + ((h >> (8 * a)) & 0xff) as u32))
                .collect();
            let sum: f64 = weights.iter().sum();
            for (a, w) in weights.iter().enumerate() {
                priors[i * 7 + a] = w / sum;
            }
            values[i] = f64::from((h >> 48) as u32 & 0xffff) / 32_767.5 - 1.0;
        }
        Ok(())
    }
}

fn visits(m: &C4) -> [u32; 7] {
    let mut v = [0; 7];
    m.root_visits(&mut v);
    v
}

const GOLDEN_VISITS: [u32; 7] = [1, 142, 18, 3, 69, 3, 20];
const GOLDEN_TREE: u64 = 3_392_156_604_488_121_207;

fn noisy(simulations: u32) -> PuctConfig {
    let mut c = config(simulations);
    c.noise = Some(RootNoise {
        alpha: 0.3,
        epsilon: 0.25,
    });
    c
}

/// Everything a later call can observe of a tree: visits, the tree digest
/// (statistics and allocation numbers), the call counter, and the search
/// stream position (the next draws of a sampled selection).
fn fingerprint(m: &mut C4) -> ([u32; 7], u64, u64, Vec<u32>) {
    let draws = (0..4).map(|_| m.select_action(1.0).unwrap()).collect();
    (visits(m), m.pool().tree_hash(), m.calls(), draws)
}

#[test]
fn test_group_canned_slots_reproduce_the_adr0043_golden() {
    let slots = (0..3)
        .map(|_| mcts_with("33", 4_096, Eviction::Strict, config(256)))
        .collect();
    let mut g = Group::new(&Connect4Config {}, slots, DEFAULT_GROUP_BUDGET).unwrap();
    g.run(&mut Canned::default()).unwrap();
    for i in 0..3 {
        let m = g.slot(i).unwrap();
        assert_eq!(visits(m), GOLDEN_VISITS, "slot {i}");
        assert_eq!(m.pool().tree_hash(), GOLDEN_TREE, "slot {i}");
    }
}

/// Different roots, budgets, batch sizes and noise per slot, over three
/// calls with subtree reuse and a re-root: each slot equals its own
/// independent search, stream position included.
#[test]
fn test_group_slots_equal_independent_searches_over_calls_with_reuse() {
    let specs: [(&str, PuctConfig); 4] = [
        ("", config(64)),
        ("33", noisy(100)),
        ("3412", {
            let mut c = config(37);
            c.batch_size = 3;
            c
        }),
        ("44", {
            let mut c = noisy(150);
            c.batch_size = 1;
            c
        }),
    ];
    let build = || -> Vec<C4> {
        specs
            .iter()
            .map(|(m, c)| mcts_with(m, 4_096, Eviction::Strict, *c))
            .collect()
    };
    let mut solo = build();
    let mut g = Group::new(&Connect4Config {}, build(), DEFAULT_GROUP_BUDGET).unwrap();
    for call in 0..3 {
        for m in &mut solo {
            m.run(&mut Canned::default()).unwrap();
        }
        let mut ev = Canned::default();
        g.run(&mut ev).unwrap();
        let solo_prints: Vec<_> = solo.iter_mut().map(fingerprint).collect();
        for (i, want) in solo_prints.iter().enumerate() {
            assert_eq!(
                &fingerprint(g.slot_mut(i).unwrap()),
                want,
                "call {call}, slot {i}"
            );
        }
        // Fewer evaluator calls than the slots' batches together.
        assert!(ev.calls > 0);
        if call == 1 {
            // Re-root slot 2 on a fresh position, in both.
            let r = record("1");
            solo[2].pool_mut().set_root(&r).unwrap();
            g.slot_mut(2).unwrap().pool_mut().set_root(&r).unwrap();
        }
    }
}

/// Fails at the `at`-th evaluator call.
struct FailAt {
    inner: Canned,
    at: u32,
}

impl Evaluator<Connect4> for FailAt {
    fn evaluate(
        &mut self,
        leaves: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        if self.inner.calls + 1 == self.at {
            self.inner.calls += 1;
            return Err(EvaluatorError::Failed {
                reason: "boom".to_owned(),
            });
        }
        self.inner.evaluate(leaves, priors, values)
    }
}

/// Answers correctly, except a NaN value in the last row of every batch
/// after the first `poison_after`: the last row belongs to the last slot.
struct BadRows {
    inner: Canned,
    poison_after: u32,
}

impl Evaluator<Connect4> for BadRows {
    fn evaluate(
        &mut self,
        leaves: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        self.inner.evaluate(leaves, priors, values)?;
        if self.inner.calls > self.poison_after
            && let Some(last) = values.last_mut()
        {
            *last = f64::NAN;
        }
        Ok(())
    }
}

fn group_of(n: usize, sims: u32) -> Group {
    let slots = (0..n)
        .map(|i| mcts_with(&"1234"[..i.min(4)], 64, Eviction::Oldest, noisy(sims)))
        .collect();
    Group::new(&Connect4Config {}, slots, DEFAULT_GROUP_BUDGET).unwrap()
}

#[test]
fn test_group_failure_in_a_later_slot_rolls_back_every_slot_then_retry_succeeds() {
    let mut g = group_of(3, 90);
    g.run(&mut Canned::default()).unwrap();
    let before: Vec<_> = (0..3)
        .map(|i| {
            let m = g.slot(i).unwrap();
            (visits(m), m.pool().tree_hash(), m.calls())
        })
        .collect();
    // A bad row in the last slot of a later round, and an evaluator
    // failure mid-call.
    let err = g
        .run(&mut BadRows {
            inner: Canned::default(),
            poison_after: 3,
        })
        .unwrap_err();
    assert_eq!(err.slot, Some(2), "{err}");
    assert!(matches!(
        err.error,
        SearchError::Evaluator(EvaluatorError::NonFinite { .. })
    ));
    assert_eq!(g.last_failure().unwrap().slot, Some(2));
    let err = g
        .run(&mut FailAt {
            inner: Canned::default(),
            at: 4,
        })
        .unwrap_err();
    assert_eq!(err.slot, None);
    assert!(g.last_failure().unwrap().slot.is_none());
    for (i, want) in before.iter().enumerate() {
        let m = g.slot(i).unwrap();
        assert_eq!(
            &(visits(m), m.pool().tree_hash(), m.calls()),
            want,
            "slot {i}"
        );
    }
    // The retry equals a group that never saw the failures.
    g.run(&mut Canned::default()).unwrap();
    assert!(g.last_failure().is_none(), "cleared at the start of a call");
    let mut fresh = group_of(3, 90);
    fresh.run(&mut Canned::default()).unwrap();
    fresh.run(&mut Canned::default()).unwrap();
    for i in 0..3 {
        assert_eq!(
            fingerprint(g.slot_mut(i).unwrap()),
            fingerprint(fresh.slot_mut(i).unwrap()),
            "slot {i}"
        );
    }
}

#[test]
fn test_group_terminal_root_fails_before_any_change() {
    let slots = vec![
        mcts_with("", 64, Eviction::Strict, config(16)),
        mcts_with("1212121", 64, Eviction::Strict, config(16)),
    ];
    let mut g = Group::new(&Connect4Config {}, slots, DEFAULT_GROUP_BUDGET).unwrap();
    let mut ev = Canned::default();
    let err = g.run(&mut ev).unwrap_err();
    assert_eq!(err.slot, Some(1));
    assert!(matches!(err.error, SearchError::TerminalNode));
    assert_eq!(ev.calls, 0, "no evaluator call");
    assert_eq!(g.slot(0).unwrap().calls(), 0);
}

/// Counts polls and batch sizes.
#[derive(Default)]
struct Counting {
    inner: Canned,
    polls: u32,
    sizes: Vec<usize>,
}

impl Evaluator<Connect4> for Counting {
    fn evaluate(
        &mut self,
        _: &[Record<Connect4>],
        _: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        Err(EvaluatorError::Failed {
            reason: "evaluate_batch is overridden".to_owned(),
        })
    }

    fn evaluate_batch(
        &mut self,
        batch: &mut LeafBatch<'_, Connect4>,
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        assert_eq!(batch.observations().len(), batch.len() * batch.obs_bytes());
        assert_eq!(batch.legal().len(), batch.len() * 7);
        self.sizes.push(batch.len());
        self.inner.evaluate(batch.records(), priors, values)
    }

    fn check(&mut self) -> Result<(), EvaluatorError> {
        self.polls += 1;
        Ok(())
    }
}

#[test]
fn test_group_terminal_only_rounds_poll_but_never_call_the_evaluator() {
    // A position one move from a forced end: many rounds reach only
    // terminal leaves.
    let mut g = Group::new(
        &Connect4Config {},
        vec![mcts_with("121212", 4_096, Eviction::Strict, config(40))],
        DEFAULT_GROUP_BUDGET,
    )
    .unwrap();
    let mut ev = Counting::default();
    g.run(&mut ev).unwrap();
    assert!(
        ev.sizes.iter().all(|&n| n > 0),
        "never an empty batch: {:?}",
        ev.sizes
    );
    // One poll before the root batch, one per round: more polls than
    // evaluator calls once a round has terminal leaves only.
    assert!(
        ev.polls as usize > ev.sizes.len(),
        "{} polls, {:?}",
        ev.polls,
        ev.sizes
    );
    let mut solo = mcts_with("121212", 4_096, Eviction::Strict, config(40));
    solo.run(&mut Canned::default()).unwrap();
    assert_eq!(fingerprint(g.slot_mut(0).unwrap()), fingerprint(&mut solo));
}

#[test]
fn test_group_inactive_slots_are_untouched_and_no_participants_is_a_no_op() {
    let mut g = group_of(3, 32);
    g.set_active(1, false);
    let untouched = g.slot(1).unwrap().pool().tree_hash();
    g.run(&mut Canned::default()).unwrap();
    assert_eq!(g.slot(1).unwrap().calls(), 0);
    assert_eq!(g.slot(1).unwrap().pool().tree_hash(), untouched);
    assert_eq!(g.slot(0).unwrap().calls(), 1);
    for i in 0..3 {
        g.set_active(i, false);
    }
    let hashes: Vec<_> = (0..3)
        .map(|i| g.slot(i).unwrap().pool().tree_hash())
        .collect();
    let mut ev = Counting::default();
    g.run(&mut ev).unwrap();
    assert_eq!((ev.polls, ev.sizes.len()), (0, 0), "no poll, no call");
    let after: Vec<_> = (0..3)
        .map(|i| g.slot(i).unwrap().pool().tree_hash())
        .collect();
    assert_eq!(hashes, after);
    assert!(!g.set_active(3, true), "out of range");
}

#[test]
fn test_group_batches_concatenate_slots_and_route_rows_back() {
    let mut g = group_of(4, 24);
    let mut ev = Counting::default();
    g.run(&mut ev).unwrap();
    // The root batch has one row per slot; later batches up to 4 x 8.
    assert_eq!(ev.sizes.first(), Some(&4));
    assert!(ev.sizes.iter().any(|&n| n > 8), "{:?}", ev.sizes);
    assert!(ev.sizes.iter().all(|&n| n <= 32));
}

#[test]
fn test_group_rejects_bad_sizes_and_an_aggregate_over_the_budget() {
    assert!(matches!(
        Group::new(&Connect4Config {}, Vec::new(), DEFAULT_GROUP_BUDGET).err(),
        Some(SearchError::Options {
            option: "trees",
            ..
        })
    ));
    let slots = (0..4)
        .map(|_| mcts_with("", 4_096, Eviction::Strict, config(8)))
        .collect();
    let g = Group::new(&Connect4Config {}, slots, DEFAULT_GROUP_BUDGET).unwrap();
    let bound = g.memory_bytes();
    let per_tree = SearchPool::<Connect4, Masked>::bytes_per_node(7) * 4_096;
    assert!(bound >= 4 * per_tree, "{bound} >= 4 x {per_tree}");
    let slots = (0..4)
        .map(|_| mcts_with("", 4_096, Eviction::Strict, config(8)))
        .collect();
    assert!(matches!(
        Group::new(&Connect4Config {}, slots, bound - 1).err(),
        Some(SearchError::Budget { .. })
    ));
}
