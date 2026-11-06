// SPDX-License-Identifier: MIT OR Apache-2.0

//! A failed scratch step (a contained kernel panic) poisons the scratch
//! pool: the expansion fails with the tree unchanged, and the next valid
//! expansion succeeds (R38 recovery).

use core::num::NonZeroU32;

use tessera_rl_engine::{EngineError, EnvPool, NoMasks, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;
use tessera_rl_search::{NodeHandle, SearchError, SearchOptions, SearchPool};

#[path = "support/fragile.rs"]
mod fragile;
use fragile::{CountConfig, Fragile};

#[test]
fn test_search_pool_recovers_the_scratch_pool_after_a_kernel_panic() {
    let mut p = EnvPool::<Fragile>::new(
        CountConfig {},
        PoolOptions::new(NonZeroU32::MIN, Seed::new(1)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let mut s = SearchPool::<Fragile, NoMasks>::new(
        CountConfig {},
        SearchOptions::new(NonZeroU32::new(8).unwrap(), NonZeroU32::new(2).unwrap()),
        &p.snapshot(0).unwrap(),
        EnvPool::new,
    )
    .unwrap();
    let root = s.root();
    let before = s.tree_hash();
    let mut out = [root; 2];
    let err = s.expand(&[(root, 1), (root, 2)], &mut out).unwrap_err();
    assert_eq!(err, SearchError::Engine(EngineError::Panicked));
    assert_eq!(
        s.tree_hash(),
        before,
        "the failed expansion changed the tree"
    );
    // The next valid expansion works, and gives the children a fresh pool
    // would give.
    let mut good: [NodeHandle; 2] = [root; 2];
    s.expand(&[(root, 0), (root, 1)], &mut good).unwrap();
    assert_eq!(s.record(good[0]).unwrap().state().total, 0);
    assert_eq!(s.record(good[1]).unwrap().state().total, 1);
    assert_eq!(s.record(good[1]).unwrap().state().to_move, 1);
}
