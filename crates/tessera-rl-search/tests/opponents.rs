// SPDX-License-Identifier: MIT OR Apache-2.0

//! Search against heuristic opponents (ADR-0093, #491) on `PaddleDuel`, a
//! perfect-information simultaneous environment: slot 0 searches, slot 1
//! is an ADR-0079 heuristic slot of the scratch pool. Every ply is the
//! searcher's (no negation), a child is the live pool's step with the
//! same heuristic slot, and the construction checks are typed.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::{PaddleDuel, PaddleDuelConfig};
use tessera_rl_core::PolicyGroup;
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, EnvPool, NoMasks, PoolOptions, Record, Reseed,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{
    Evaluator, EvaluatorError, Eviction, Mcts, PuctConfig, SearchError, SearchOptions, SearchPool,
};

type Pool = SearchPool<PaddleDuel, NoMasks>;

fn cfg() -> PaddleDuelConfig {
    PaddleDuelConfig {
        agents: 2,
        ..PaddleDuelConfig::default()
    }
}

/// A scratch builder with slot 1 a heuristic slot.
fn with_opponent(
    cfg: PaddleDuelConfig,
    options: PoolOptions,
) -> Result<EnvPool<PaddleDuel, NoMasks>, EngineError> {
    EnvPool::new(cfg, options)?.with_heuristic_slots(&[1])
}

/// A live pool with the same heuristic slot, after `steps` of the
/// searcher's action 0.
fn live(steps: usize) -> EnvPool<PaddleDuel, NoMasks> {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(3))
    };
    let mut p = with_opponent(cfg(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    for _ in 0..steps {
        p.step(ActionInput {
            discrete: &[0, 0],
            continuous: &[],
        })
        .unwrap();
    }
    p
}

fn options(searcher: Option<usize>) -> SearchOptions {
    let mut o = SearchOptions::new(NonZeroU32::new(256).unwrap(), NonZeroU32::new(8).unwrap());
    o.eviction = Eviction::Strict;
    o.searcher = searcher;
    o
}

fn pool() -> Pool {
    let root = live(2).snapshot(0).unwrap();
    SearchPool::new(cfg(), options(Some(0)), &root, with_opponent).unwrap()
}

/// Every leaf is worth +0.5 to the player to move.
struct Half;

impl Evaluator<PaddleDuel> for Half {
    fn evaluate(
        &mut self,
        _: &[Record<PaddleDuel>],
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        let n = values.len();
        let actions = priors.len() / n.max(1);
        priors.fill(1.0 / f64::from(u32::try_from(actions).unwrap()));
        values.fill(0.5);
        Ok(())
    }
}

#[test]
fn test_opponents_searcher_is_set_and_backup_never_negates() {
    let s = pool();
    assert_eq!(s.searcher(), Some(0));
    let actions = s.actions();
    let mut c = PuctConfig::default();
    c.simulations = 64;
    let mut m = Mcts::new(s, c).unwrap();
    m.run(&mut Half).unwrap();
    let root = m.pool().root();
    let mut visited = 0;
    for a in 0..actions {
        let e = m.pool().edge_stats(root, a).unwrap();
        if e.visits > 0 {
            visited += 1;
            // Every ply is the searcher's: +0.5 at every depth, never -0.5.
            assert_eq!(e.value, 0.5 * f64::from(e.visits), "action {a}");
        }
    }
    assert!(visited > 1);
}

#[test]
fn test_opponents_child_equals_the_live_step_with_the_heuristic_slot() {
    let mut s = pool();
    let root = s.root();
    let mut out = [root; 3];
    s.expand(&[(root, 0), (root, 1), (root, 2)], &mut out)
        .unwrap();
    for (a, child) in out.iter().enumerate() {
        let mut p = live(2);
        // Garbage in the heuristic slot's row is never read (ADR-0079).
        p.step(ActionInput {
            discrete: &[i32::try_from(a).unwrap(), 2],
            continuous: &[],
        })
        .unwrap();
        assert_eq!(
            s.record(*child).unwrap().hash(),
            p.snapshot(0).unwrap().hash(),
            "action {a}"
        );
    }
}

#[test]
fn test_opponents_search_is_reproducible() {
    let run = || {
        let mut c = PuctConfig::default();
        c.simulations = 48;
        let mut m = Mcts::new(pool(), c).unwrap();
        m.run(&mut Half).unwrap();
        m.pool().tree_hash()
    };
    assert_eq!(run(), run());
}

#[test]
fn test_opponents_construction_checks_are_typed() {
    let root = live(0).snapshot(0).unwrap();
    // No searcher and no determinization agent.
    assert_eq!(
        SearchPool::new(cfg(), options(None), &root, with_opponent).err(),
        Some(SearchError::SearcherRequired)
    );
    // The searcher is the heuristic slot.
    assert_eq!(
        SearchPool::new(cfg(), options(Some(1)), &root, with_opponent).err(),
        Some(SearchError::SearcherNotLearner {
            slot: 1,
            group: PolicyGroup::Heuristic
        })
    );
    // Out of range.
    assert!(matches!(
        SearchPool::new(cfg(), options(Some(2)), &root, with_opponent).err(),
        Some(SearchError::Options {
            option: "searcher",
            ..
        })
    ));
    // A default map is today's search: the searcher option is ignored and
    // two acting agents are NotSingleActor at expansion.
    let mut plain = SearchPool::new(cfg(), options(Some(0)), &root, EnvPool::new).unwrap();
    assert_eq!(plain.searcher(), None);
    let r = plain.root();
    let mut out = [r];
    assert!(matches!(
        plain.expand(&[(r, 0)], &mut out),
        Err(SearchError::NotSingleActor { acting: 2 })
    ));
}
