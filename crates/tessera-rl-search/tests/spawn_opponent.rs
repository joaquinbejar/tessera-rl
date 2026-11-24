// SPDX-License-Identifier: MIT OR Apache-2.0

//! ADR-0093 decision 3 erratum: a slot's group is checked when it is alive
//! at the root or acting at an expanded node. `Spawner`'s slot 2 is a
//! `Learner` that is not alive at the root (allowed) and spawns after the
//! first step; expanding the node where it acts is `NotSearchable`, before
//! the scratch step: the tree is unchanged, the scratch pool is not
//! poisoned, and the error is the same with and without workers.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers: a failure is the assertion"
)]

#[path = "support/spawner.rs"]
mod spawner;

use core::num::{NonZeroU32, NonZeroUsize};

use spawner::{Spawner, SpawnerConfig};
use tessera_rl_core::PolicyGroup;
use tessera_rl_engine::{
    AutoReset, EngineError, EnvPool, NoMasks, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{SearchError, SearchOptions, SearchPool};

type Pool = SearchPool<Spawner, NoMasks>;

fn opponent(cfg: SpawnerConfig, o: PoolOptions) -> Result<EnvPool<Spawner, NoMasks>, EngineError> {
    EnvPool::new(cfg, o)?.with_heuristic_slots(&[1])
}

fn opponent_workers(
    cfg: SpawnerConfig,
    o: PoolOptions,
) -> Result<EnvPool<Spawner, NoMasks>, EngineError> {
    EnvPool::new(cfg, o)?
        .with_workers(SyncOptions::new(NonZeroUsize::new(2).unwrap()))?
        .with_heuristic_slots(&[1])
}

fn search(
    build: fn(SpawnerConfig, PoolOptions) -> Result<EnvPool<Spawner, NoMasks>, EngineError>,
) -> Pool {
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(2))
    };
    let mut live = opponent(SpawnerConfig::default(), options).unwrap();
    live.reset_all(Reseed::Keep).unwrap();
    let mut o = SearchOptions::new(NonZeroU32::new(16).unwrap(), NonZeroU32::new(2).unwrap());
    o.searcher = Some(0);
    // Slot 2 is a Learner but not alive at the root: the pool builds.
    SearchPool::new(
        SpawnerConfig::default(),
        o,
        &live.snapshot(0).unwrap(),
        build,
    )
    .unwrap()
}

fn spawned_learner_is_refused_at_expansion(mut s: Pool) -> SearchError {
    let root = s.root();
    let mut out = [root];
    // The first step spawns slot 2.
    s.expand(&[(root, 0)], &mut out).unwrap();
    let child = out[0];
    assert!(s.record(child).unwrap().header().agents[2].alive.get());
    let before = s.tree_hash();
    let mut again = [child];
    let err = s.expand(&[(child, 0)], &mut again).unwrap_err();
    assert_eq!(
        s.tree_hash(),
        before,
        "a refused expansion leaves the tree unchanged"
    );
    // The scratch pool is not poisoned: another expansion still works.
    let mut other = [root];
    s.expand(&[(root, 1)], &mut other).unwrap();
    err
}

#[test]
fn test_spawned_learner_is_not_searchable_at_expansion() {
    let inline = spawned_learner_is_refused_at_expansion(search(opponent));
    assert_eq!(
        inline,
        SearchError::NotSearchable {
            slot: 2,
            group: PolicyGroup::Learner
        }
    );
    let workers = spawned_learner_is_refused_at_expansion(search(opponent_workers));
    assert_eq!(workers, inline);
}

/// ADR-0093 decision 4: at a node where the searcher is not acting, the
/// node has the single edge 0, whose step runs the heuristics only.
#[test]
fn test_searcher_not_acting_has_a_single_edge() {
    let cfg = SpawnerConfig {
        alternate: 1,
        spawn: 0,
    };
    let options = PoolOptions {
        auto_reset: AutoReset::Disabled,
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(2))
    };
    let mut live = opponent(cfg.clone(), options).unwrap();
    live.reset_all(Reseed::Keep).unwrap();
    let mut o = SearchOptions::new(NonZeroU32::new(16).unwrap(), NonZeroU32::new(2).unwrap());
    o.searcher = Some(0);
    let mut s = SearchPool::new(cfg, o, &live.snapshot(0).unwrap(), opponent).unwrap();
    let root = s.root();
    let mut out = [root];
    s.expand(&[(root, 1)], &mut out).unwrap();
    let child = out[0];
    // Step 1: the searcher does not act; only joint 0 is an edge.
    assert!(!s.record(child).unwrap().header().agents[0].acting.get());
    let mut next = [child];
    assert_eq!(
        s.expand(&[(child, 1)], &mut next).unwrap_err(),
        SearchError::IllegalAction { action: 1 }
    );
    s.expand(&[(child, 0)], &mut next).unwrap();
    // The heuristic played 0 and the searcher's row was not read: the
    // total is still the root step's 1.
    assert_eq!(s.record(next[0]).unwrap().state().total, 1);
}
