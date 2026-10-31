// SPDX-License-Identifier: MIT OR Apache-2.0

//! ADR-0037 (c, d): every public entry point reaches the canonical guard
//! (each call from a flush-to-zero thread is counted by
//! `fp::non_canonical_entries`), and the results equal those of a
//! canonical caller.

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

/// Runs every public entry point once, counting the calls.
fn entries() -> (Vec<u64>, u64) {
    let mut calls = 0;
    let mut count = |r: bool| {
        assert!(r);
        calls += 1;
    };
    let options = PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(9));
    let p = EnvPool::<Counter, Masked>::with_masks(CountConfig::default(), options);
    count(p.is_ok());
    let mut p = p.unwrap();
    count(p.reset_all(Reseed::Keep).is_ok());
    count(p.reset(&[1], Reseed::Base(Seed::new(3))).is_ok());
    count(p.check_actions(ACT).is_ok());
    count(p.step(ACT).is_ok());
    let snap = p.snapshot(0);
    count(snap.is_ok());
    let snap = snap.unwrap();
    count(p.restore(1, &snap).is_ok());
    count(p.fork(0, &[1]).is_ok());
    count(p.fork_branched(0, &[1], Some(&[7])).is_ok());
    let decoded = DecodedRecord::from_parts(*snap.header(), *snap.state()).unwrap();
    count(p.import(&[1], &[decoded]).is_ok());
    count(p.set_lineage(1, LogicalId::new(5)).is_ok());
    count(p.state_hash(0).is_ok());
    count(p.step(ACT).is_ok());
    let hashes = (0..2).map(|i| p.state_hash(i).unwrap().get()).collect();
    (hashes, calls)
}

#[test]
fn test_every_entry_point_reaches_the_guard_from_a_flush_to_zero_thread() {
    let (canonical, _) = entries();
    let before = fp::non_canonical_entries();
    let (hostile, calls) = fp::with_hostile_flush_to_zero(entries);
    let counted = fp::non_canonical_entries() - before;
    assert_eq!(hostile, canonical, "results differ under flush-to-zero");
    // The two final state_hash calls are entries too.
    assert!(
        counted >= calls + 2,
        "{counted} counted for {calls} entry calls"
    );
}
