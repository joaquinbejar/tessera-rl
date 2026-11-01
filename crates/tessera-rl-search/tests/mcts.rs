// SPDX-License-Identifier: MIT OR Apache-2.0

//! The PUCT helper (ADR-0043 sections 2 to 4) on Connect Four, with a
//! canned deterministic evaluator: exact visit counts (pinned goldens,
//! with and without eviction), the visit budget, terminal values,
//! virtual-loss bookkeeping, rollback of failed calls, evaluator
//! validation, root noise and action selection.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Record, Reseed};
use tessera_rl_games::{Connect4, Connect4Config};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{
    Evaluator, EvaluatorError, Eviction, Mcts, PuctConfig, RootNoise, SearchError, SearchOptions,
    SearchPool,
};

type C4 = Mcts<Connect4, Masked>;

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

/// Uniform priors and a zero value: search driven by terminal values.
struct Uniform;

impl Evaluator<Connect4> for Uniform {
    fn evaluate(
        &mut self,
        _: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        priors.fill(1.0 / 7.0);
        values.fill(0.0);
        Ok(())
    }
}

fn visits(m: &C4) -> [u32; 7] {
    let mut v = [0; 7];
    m.root_visits(&mut v);
    v
}

#[test]
fn test_mcts_spends_exactly_the_simulation_budget() {
    let mut m = mcts_with("", 4_096, Eviction::Strict, config(200));
    let mut ev = Canned::default();
    m.run(&mut ev).unwrap();
    assert_eq!(visits(&m).iter().sum::<u32>(), 200);
    assert_eq!(m.calls(), 1);
    // Every virtual loss was removed.
    let root = m.pool().root();
    for a in 0..7 {
        assert_eq!(m.pool().edge_stats(root, a).unwrap().virtual_loss, 0);
    }
    // A second call adds another budget on the reused tree.
    m.run(&mut ev).unwrap();
    assert_eq!(visits(&m).iter().sum::<u32>(), 400);
}

/// Golden visit counts with the canned evaluator (ADR-0043 section 4):
/// identical on every platform.
#[test]
fn test_mcts_canned_visit_counts_match_the_golden() {
    let mut m = mcts_with("33", 4_096, Eviction::Strict, config(256));
    m.run(&mut Canned::default()).unwrap();
    assert_eq!(visits(&m), GOLDEN_VISITS, "visits");
    assert_eq!(m.pool().tree_hash(), GOLDEN_TREE, "tree");
}

/// The same with a capacity small enough to evict.
#[test]
fn test_mcts_canned_visit_counts_under_eviction_match_the_golden() {
    let mut m = mcts_with("33", 48, Eviction::Oldest, config(256));
    m.run(&mut Canned::default()).unwrap();
    assert_eq!(m.pool().live_nodes(), 48);
    assert_eq!(visits(&m), GOLDEN_EVICTED_VISITS, "visits");
    assert_eq!(m.pool().tree_hash(), GOLDEN_EVICTED_TREE, "tree");
}

// Pinned on macOS arm64; the same on Linux x86-64 and arm64 (see the PR).
const GOLDEN_VISITS: [u32; 7] = [1, 142, 18, 3, 69, 3, 20];
const GOLDEN_TREE: u64 = 3_392_156_604_488_121_207;
const GOLDEN_EVICTED_VISITS: [u32; 7] = [1, 157, 3, 4, 86, 2, 3];
const GOLDEN_EVICTED_TREE: u64 = 7_631_716_779_206_222_123;

#[test]
fn test_mcts_is_deterministic_run_to_run() {
    let run = || {
        let mut m = mcts_with("3", 4_096, Eviction::Strict, config(300));
        m.run(&mut Canned::default()).unwrap();
        (visits(&m), m.pool().tree_hash())
    };
    assert_eq!(run(), run());
}

#[test]
fn test_mcts_finds_the_win_in_one_and_the_forced_block() {
    // Slot 0 to move wins in column 0.
    let mut m = mcts_with("010101", 4_096, Eviction::Strict, config(200));
    m.run(&mut Uniform).unwrap();
    assert_eq!(m.select_action(0.0).unwrap(), 0);
    // Slot 1 to move must block column 0.
    let mut m = mcts_with("01010", 4_096, Eviction::Strict, config(400));
    m.run(&mut Uniform).unwrap();
    assert_eq!(m.select_action(0.0).unwrap(), 0);
}

/// Fails on its `at`-th call, or answers like `Canned`.
struct FailAt {
    inner: Canned,
    at: u32,
    error: EvaluatorError,
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
            return Err(self.error.clone());
        }
        self.inner.evaluate(leaves, priors, values)
    }
}

#[test]
fn test_mcts_failed_call_rolls_the_tree_and_counter_back() {
    let mut m = mcts_with("", 64, Eviction::Oldest, config(120));
    m.run(&mut Canned::default()).unwrap();
    let (tree, calls) = (m.pool().tree_hash(), m.calls());
    // Fails mid-call, after batches were committed and nodes evicted.
    for fail_at in [1, 3, 9] {
        let mut ev = FailAt {
            inner: Canned::default(),
            at: fail_at,
            error: EvaluatorError::Failed {
                reason: "boom".to_owned(),
            },
        };
        let err = m.run(&mut ev).unwrap_err();
        assert!(matches!(
            err,
            SearchError::Evaluator(EvaluatorError::Failed { .. })
        ));
        assert_eq!(m.pool().tree_hash(), tree, "fail at {fail_at}");
        assert_eq!(m.calls(), calls);
    }
    // The next call behaves as if the failures never happened.
    let mut fresh = mcts_with("", 64, Eviction::Oldest, config(120));
    fresh.run(&mut Canned::default()).unwrap();
    fresh.run(&mut Canned::default()).unwrap();
    m.run(&mut Canned::default()).unwrap();
    assert_eq!(m.pool().tree_hash(), fresh.pool().tree_hash());
}

/// Returns a fixed response for every leaf.
struct Fixed {
    priors: [f64; 7],
    value: f64,
}

impl Evaluator<Connect4> for Fixed {
    fn evaluate(
        &mut self,
        leaves: &[Record<Connect4>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        for i in 0..leaves.len() {
            priors[i * 7..i * 7 + 7].copy_from_slice(&self.priors);
            values[i] = self.value;
        }
        Ok(())
    }
}

#[test]
fn test_mcts_rejects_invalid_evaluator_responses_without_changes() {
    let u = 1.0 / 7.0;
    let cases = [
        ([u; 7], f64::NAN, "nan value"),
        (
            [f64::INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            0.0,
            "inf prior",
        ),
        ([u; 7], 1.5, "value range"),
        ([-0.1, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1], 0.0, "negative prior"),
        ([0.2; 7], 0.0, "prior sum"),
        ([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 0.0, "no legal mass"),
    ];
    for (priors, value, what) in cases {
        // Column 3 is full at the root: all mass on it has no legal mass.
        let mut m = mcts_with("333333", 4_096, Eviction::Strict, config(32));
        let before = m.pool().tree_hash();
        let err = m.run(&mut Fixed { priors, value }).unwrap_err();
        assert!(matches!(err, SearchError::Evaluator(_)), "{what}: {err}");
        assert_eq!(m.pool().tree_hash(), before, "{what}");
        assert_eq!(m.calls(), 0);
    }
    // Within tolerance, and illegal mass is dropped and renormalized.
    let mut m = mcts_with("333333", 4_096, Eviction::Strict, config(32));
    let p = [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1 + 5e-7];
    m.run(&mut Fixed {
        priors: p,
        value: 0.0,
    })
    .unwrap();
    let root = m.pool().root();
    let s = m.pool().edge_stats(root, 3).unwrap();
    assert_eq!((s.prior, s.visits), (0.0, 0));
    let legal: f64 = (0..7)
        .map(|a| m.pool().edge_stats(root, a).unwrap().prior)
        .sum();
    assert!((legal - 1.0).abs() < 1e-12);
}

#[test]
fn test_mcts_root_noise_is_reproducible_and_keyed_by_the_call_counter() {
    let mut cfg = config(64);
    cfg.noise = Some(RootNoise {
        alpha: 0.3,
        epsilon: 0.25,
    });
    let run = |calls: u32| {
        let mut m = mcts_with("", 4_096, Eviction::Strict, cfg);
        let mut out = Vec::new();
        for _ in 0..calls {
            m.run(&mut Canned::default()).unwrap();
            out.push(visits(&m));
            m.pool_mut().clear().unwrap();
        }
        out
    };
    let a = run(3);
    assert_eq!(a, run(3));
    // Each call draws fresh noise: identical trees after a clear differ.
    assert!(a[0] != a[1] || a[1] != a[2]);
    let mut plain = mcts_with("", 4_096, Eviction::Strict, config(64));
    plain.run(&mut Canned::default()).unwrap();
    assert_ne!(visits(&plain), a[0], "noise changes the search");
}

#[test]
fn test_mcts_action_selection_greedy_and_sampled() {
    let mut m = mcts_with("", 4_096, Eviction::Strict, config(200));
    assert!(m.select_action(1.0).is_err(), "no call yet");
    m.run(&mut Canned::default()).unwrap();
    let v = visits(&m);
    let best = (0..7)
        .max_by_key(|&a| (v[a], core::cmp::Reverse(a)))
        .unwrap();
    assert_eq!(m.select_action(0.0).unwrap(), u32::try_from(best).unwrap());
    // Sampling is reproducible and only picks visited actions.
    let draws = |m: &mut C4| {
        (0..20)
            .map(|_| m.select_action(1.0).unwrap())
            .collect::<Vec<_>>()
    };
    let mut n = mcts_with("", 4_096, Eviction::Strict, config(200));
    n.run(&mut Canned::default()).unwrap();
    let (x, y) = (draws(&mut m), draws(&mut n));
    assert_eq!(x, y);
    assert!(x.iter().all(|&a| v[a as usize] > 0));
    assert!(m.select_action(-1.0).is_err());
}

#[test]
fn test_mcts_rejects_invalid_configs_and_terminal_roots() {
    let pool = || {
        SearchPool::new(
            Connect4Config {},
            SearchOptions::new(NonZeroU32::new(64).unwrap(), NonZeroU32::new(4).unwrap()),
            &record(""),
            EnvPool::with_masks,
        )
        .unwrap()
    };
    let mut c = PuctConfig::default();
    c.batch_size = 5;
    assert!(Mcts::new(pool(), c).is_err());
    c = PuctConfig::default();
    c.batch_size = 4;
    c.noise = Some(RootNoise {
        alpha: 0.0,
        epsilon: 0.25,
    });
    assert!(Mcts::new(pool(), c).is_err());
    c.noise = None;
    c.c_puct = f64::NAN;
    assert!(Mcts::new(pool(), c).is_err());
    let mut m = mcts_with("0101010", 64, Eviction::Strict, config(8));
    assert_eq!(
        m.run(&mut Canned::default()).unwrap_err(),
        SearchError::TerminalNode
    );
}

/// The whole call computes under the canonical controls: a caller in round
/// toward zero gets the golden visits, tree and sampled actions (ADR-0037,
/// ADR-0043 section 2).
#[test]
fn test_mcts_is_canonical_under_hostile_rounding() {
    use tessera_rl_num::fp::hostile;
    let mut cfg = config(256);
    cfg.noise = Some(RootNoise {
        alpha: 0.3,
        epsilon: 0.25,
    });
    let run = |hostile_mode: bool| {
        let mut m = mcts_with("33", 4_096, Eviction::Strict, cfg);
        let mut go = || {
            m.run(&mut Canned::default()).unwrap();
            (0..8)
                .map(|_| m.select_action(1.0).unwrap())
                .collect::<Vec<_>>()
        };
        let picks = if hostile_mode {
            hostile::run(hostile::toward_zero, go).0
        } else {
            go()
        };
        (visits(&m), m.pool().tree_hash(), picks)
    };
    assert_eq!(run(true), run(false));
    // And without noise, the hostile run reproduces the golden.
    let mut m = mcts_with("33", 4_096, Eviction::Strict, config(256));
    hostile::run(hostile::toward_zero, || {
        m.run(&mut Canned::default()).unwrap();
    });
    assert_eq!(visits(&m), GOLDEN_VISITS);
    assert_eq!(m.pool().tree_hash(), GOLDEN_TREE);
}
