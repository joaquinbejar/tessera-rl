// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Reseed::Each`: one optional seed per reset instance, in one call
//! (Gymnasium's per-environment `None`). The whole request is validated
//! before anything is written (ADR-0028), and a full reset with it
//! recovers a poisoned pool.

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod common;

use core::num::NonZeroU32;

use common::{CountConfig, Counter};
use tessera_rl_engine::{ActionInput, EngineError, EnvPool, Masked, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

type Pool = EnvPool<Counter, Masked>;

fn pool(cfg: CountConfig) -> Pool {
    let options = PoolOptions::new(NonZeroU32::new(3).unwrap(), Seed::new(1));
    let mut p = EnvPool::with_masks(cfg, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn step(p: &mut Pool) -> Result<(), EngineError> {
    let actions = vec![0; p.outputs().unwrap().rewards().len()];
    p.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .map(|_| ())
}

#[test]
fn test_reseed_each_matches_list_and_keep_per_instance() {
    let mut each = pool(CountConfig::default());
    let mut list = pool(CountConfig::default());
    let mut keep = pool(CountConfig::default());
    for p in [&mut each, &mut list, &mut keep] {
        step(p).unwrap();
    }
    each.reset(&[0, 2], Reseed::Each(&[Some(Seed::new(7)), None]))
        .unwrap();
    list.reset(&[0], Reseed::List(&[Seed::new(7)])).unwrap();
    keep.reset(&[2], Reseed::Keep).unwrap();
    assert_eq!(each.state_hash(0).unwrap(), list.state_hash(0).unwrap());
    assert_eq!(each.state_hash(2).unwrap(), keep.state_hash(2).unwrap());
    assert_eq!(each.state_hash(1).unwrap(), list.state_hash(1).unwrap());
}

#[test]
fn test_reseed_each_rejects_the_whole_request_before_writing() {
    let mut p = pool(CountConfig::default());
    step(&mut p).unwrap();
    let hashes: Vec<_> = (0..3).map(|i| p.state_hash(i).unwrap()).collect();
    let generation = p.outputs().unwrap().storage().generation();
    let each = [Some(Seed::new(7)), None];
    assert!(matches!(
        p.reset(&[0, 99], Reseed::Each(&each)),
        Err(EngineError::InstanceOutOfRange { .. })
    ));
    assert_eq!(
        p.reset(&[0, 0], Reseed::Each(&each)).map(|_| ()),
        Err(EngineError::DuplicateInstance { id: 0 })
    );
    assert!(matches!(
        p.reset(&[0, 1, 2], Reseed::Each(&each)),
        Err(EngineError::SeedCount { .. })
    ));
    let after: Vec<_> = (0..3).map(|i| p.state_hash(i).unwrap()).collect();
    assert_eq!(after, hashes, "no record changed");
    assert_eq!(
        p.outputs().unwrap().storage().generation(),
        generation,
        "no output written"
    );
}

#[test]
fn test_reseed_each_full_reset_recovers_a_poisoned_pool() {
    let mut p = pool(CountConfig {
        panic_at: 1,
        ..CountConfig::default()
    });
    assert!(step(&mut p).is_err());
    assert!(p.is_poisoned());
    assert_eq!(
        p.reset(&[0], Reseed::Each(&[None])).map(|_| ()),
        Err(EngineError::Poisoned)
    );
    p.reset_all(Reseed::Each(&[
        Some(Seed::new(5)),
        None,
        Some(Seed::new(6)),
    ]))
    .unwrap();
    assert!(!p.is_poisoned());
}
