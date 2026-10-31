// SPDX-License-Identifier: MIT OR Apache-2.0

//! ADR-0037 (c, d): every public `EnvPool` method that runs engine or
//! environment code enters the canonical guard exactly once per call, so a
//! call from a flush-to-zero thread raises `fp::non_canonical_entries` by
//! exactly one; the results equal those of a canonical caller. A future
//! method that forgets `entry::canonical` fails here.
//!
//! The counter is process-wide, so this binary has a single test.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod common;

use core::num::NonZeroU32;

use common::{CountConfig, Counter};
use tessera_rl_engine::{ActionInput, DecodedRecord, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::fp;
use tessera_rl_num::identity::{LogicalId, Seed};

const ACT: ActionInput<'static> = ActionInput {
    discrete: &[1, 0, 0, 1],
    continuous: &[],
};

/// Runs `f` and checks it entered the guard exactly once.
fn once<R>(name: &str, f: impl FnOnce() -> R) -> R {
    let before = fp::non_canonical_entries();
    let result = f();
    let entered = fp::non_canonical_entries() - before;
    // A canonical caller enters zero times; a hostile one exactly once.
    if !fp::is_canonical() {
        assert_eq!(entered, 1, "{name} entered the guard {entered} times");
    }
    result
}

/// Every public method that runs engine or environment code, once each.
fn entries() -> Vec<u64> {
    let options = PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(9));
    let plain = once("new", || {
        EnvPool::<Counter>::new(CountConfig::default(), options)
    })
    .unwrap();
    drop(plain);
    let mut p = once("with_masks", || {
        EnvPool::<Counter, Masked>::with_masks(CountConfig::default(), options)
    })
    .unwrap();
    once("reset_all", || p.reset_all(Reseed::Keep).map(|_| ())).unwrap();
    once("reset", || {
        p.reset(&[1], Reseed::Base(Seed::new(3))).map(|_| ())
    })
    .unwrap();
    once("check_actions", || p.check_actions(ACT)).unwrap();
    once("step", || p.step(ACT).map(|_| ())).unwrap();
    let snap = once("snapshot", || p.snapshot(0)).unwrap();
    let mut all = vec![snap; 2];
    once("snapshot_all", || p.snapshot_all(&mut all)).unwrap();
    once("restore", || p.restore(1, &snap)).unwrap();
    once("restore_all", || p.restore_all(&all)).unwrap();
    once("fork", || p.fork(0, &[1])).unwrap();
    once("fork_branched", || p.fork_branched(0, &[1], Some(&[7]))).unwrap();
    let decoded = DecodedRecord::from_parts(*snap.header(), *snap.state()).unwrap();
    once("import", || p.import(&[1], &[decoded])).unwrap();
    once("import_all", || p.import_all(&[decoded, decoded])).unwrap();
    once("set_lineage", || p.set_lineage(1, LogicalId::new(5))).unwrap();
    let ctx = once("pool_context", || p.pool_context()).unwrap();
    once("restore_pool_context", || p.restore_pool_context(ctx)).unwrap();
    once("state_hash", || p.state_hash(0)).unwrap();
    once("step", || p.step(ACT).map(|_| ())).unwrap();
    (0..2)
        .map(|i| p.snapshot(i).unwrap().hash().get())
        .collect()
}

#[test]
fn test_every_public_method_enters_the_guard_exactly_once() {
    let canonical = entries();
    let hostile = fp::with_hostile_flush_to_zero(entries);
    assert_eq!(hostile, canonical, "results differ under flush-to-zero");
}
