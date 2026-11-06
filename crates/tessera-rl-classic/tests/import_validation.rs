// SPDX-License-Identifier: MIT OR Apache-2.0

//! Header-aware import validation (ADR-0040): imports reject records whose
//! engine header and environment state disagree, atomically, and accept
//! every record the engine itself produces.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::needless_pass_by_value,
    clippy::panic,
    reason = "test helpers: a failure is the assertion"
)]

use core::num::NonZeroU32;

use tessera_rl_classic::{CartPole, CartPoleConfig, Tag, TagConfig, TagState};
use tessera_rl_core::{Bool8, NoAdaptive, RecordHeader};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, Masked, PoolOptions, Reseed,
};
use tessera_rl_num::identity::Seed;

type TagHeader = RecordHeader<8, NoAdaptive>;

fn tag_pool(n: u32, cfg: TagConfig, auto_reset: AutoReset) -> EnvPool<Tag, Masked> {
    let options = PoolOptions {
        auto_reset,
        max_episode_steps: NonZeroU32::new(200),
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(5))
    };
    EnvPool::with_masks(cfg, options).unwrap()
}

fn stay(p: &mut EnvPool<Tag, Masked>) {
    let acts = vec![0; p.outputs().unwrap().rewards().len()];
    p.step(ActionInput {
        discrete: &acts,
        continuous: &[],
    })
    .unwrap();
}

fn parts(p: &EnvPool<Tag, Masked>, id: u32) -> (TagHeader, TagState) {
    let r = p.snapshot(id).unwrap();
    (*r.header(), *r.state())
}

fn record(header: TagHeader, state: TagState) -> DecodedRecord<Tag> {
    DecodedRecord::from_parts(header, state).unwrap()
}

fn invariant(result: Result<(), EngineError>) -> bool {
    matches!(result, Err(EngineError::Invariant { .. }))
}

/// A small grid where stationary evaders get caught quickly, with an
/// evader caught but the episode still running.
fn with_capture() -> (EnvPool<Tag, Masked>, TagConfig) {
    let cfg = TagConfig {
        width: 3,
        height: 3,
        pursuers: 1,
        evaders: 2,
        slip: 1000,
        ..TagConfig::default()
    };
    // Random moves (slip 1000) until exactly one evader is caught.
    for seed in 0..500u64 {
        let options = PoolOptions {
            max_episode_steps: NonZeroU32::new(200),
            ..PoolOptions::new(NonZeroU32::MIN, Seed::new(seed))
        };
        let mut p = EnvPool::<Tag, Masked>::with_masks(cfg.clone(), options).unwrap();
        p.reset_all(Reseed::Keep).unwrap();
        for _ in 0..50 {
            stay(&mut p);
            let s = p.state(0).unwrap();
            let caught = s.caught[1..3].iter().filter(|c| c.get()).count();
            let ended = p.outputs().unwrap().episode_terminated()[0].get();
            if caught == 1 && !ended {
                return (p, cfg);
            }
            if ended {
                break;
            }
        }
    }
    panic!("no seed gave a single capture");
}

#[test]
fn test_import_rejects_a_dead_slot_whose_agent_is_not_caught() {
    // The fuzz finding: the header says dead, the state says uncaught.
    let mut p = tag_pool(2, TagConfig::default(), AutoReset::NextStep);
    p.reset_all(Reseed::Keep).unwrap();
    stay(&mut p);
    let (mut h, s) = parts(&p, 0);
    h.agents[1].alive = Bool8::FALSE;
    h.agents[1].acting = Bool8::FALSE;
    assert!(invariant(p.import(&[1], &[record(h, s)])));
}

#[test]
fn test_import_rejects_an_alive_slot_whose_agent_is_caught() {
    let (p, _) = with_capture();
    let (h, s) = parts(&p, 0);
    let e = (1..3).find(|&e| s.caught[e].get()).unwrap();
    let mut h2 = h;
    h2.agents[e].alive = Bool8::TRUE;
    h2.agents[e].acting = Bool8::TRUE;
    let mut q = p;
    assert!(invariant(q.import(&[0], &[record(h2, s)])));
    // The consistent record is accepted.
    q.import(&[0], &[record(h, s)]).unwrap();
}

#[test]
fn test_import_rejects_an_active_unused_slot() {
    let mut p = tag_pool(1, TagConfig::default(), AutoReset::NextStep);
    p.reset_all(Reseed::Keep).unwrap();
    let (mut h, s) = parts(&p, 0);
    // Default Tag uses slots 0..4.
    h.agents[6].alive = Bool8::TRUE;
    assert!(invariant(p.import(&[0], &[record(h, s)])));
    let (mut h, s) = parts(&p, 0);
    h.agents[6].alive = Bool8::TRUE;
    h.agents[6].acting = Bool8::TRUE;
    assert!(invariant(p.import(&[0], &[record(h, s)])));
}

#[test]
fn test_import_and_import_all_are_atomic_when_a_later_record_fails() {
    let mut p = tag_pool(2, TagConfig::default(), AutoReset::NextStep);
    p.reset_all(Reseed::Keep).unwrap();
    for _ in 0..5 {
        stay(&mut p);
    }
    let before = [p.snapshot(0).unwrap().hash(), p.snapshot(1).unwrap().hash()];
    let rewards = p.outputs().unwrap().rewards().to_vec();
    let (h, s) = parts(&p, 1);
    let good = record(h, s);
    let (mut bad_h, bad_s) = parts(&p, 0);
    let e = (1..4).find(|&e| bad_h.agents[e].alive.get()).unwrap();
    bad_h.agents[e].alive = Bool8::FALSE;
    bad_h.agents[e].acting = Bool8::FALSE;
    let bad = record(bad_h, bad_s);
    // Scalar import: the first record is valid, the second is not.
    assert!(invariant(p.import(&[0, 1], &[good, bad])));
    // Full import.
    assert!(invariant(p.import_all(&[good, bad])));
    assert_eq!(
        [p.snapshot(0).unwrap().hash(), p.snapshot(1).unwrap().hash()],
        before
    );
    assert_eq!(p.outputs().unwrap().rewards(), rewards.as_slice());
    assert!(!p.is_poisoned());
}

#[test]
fn test_import_accepts_every_record_the_engine_produces() {
    let mut accepted = 0;
    // Never reset.
    let p = tag_pool(1, TagConfig::default(), AutoReset::NextStep);
    let (h, s) = parts(&p, 0);
    let mut q = tag_pool(1, TagConfig::default(), AutoReset::NextStep);
    q.import(&[0], &[record(h, s)]).unwrap();
    accepted += 1;
    // Ordinary, mid-episode.
    let mut p = tag_pool(1, TagConfig::default(), AutoReset::NextStep);
    p.reset_all(Reseed::Keep).unwrap();
    stay(&mut p);
    let (h, s) = parts(&p, 0);
    p.import(&[0], &[record(h, s)]).unwrap();
    accepted += 1;
    // Captured: one evader dead, the episode running.
    let (mut p, _) = with_capture();
    let (h, s) = parts(&p, 0);
    p.import(&[0], &[record(h, s)]).unwrap();
    accepted += 1;
    // Terminated (pending reset): every evader caught.
    let cfg = TagConfig {
        width: 3,
        height: 3,
        pursuers: 1,
        evaders: 1,
        slip: 1000,
        ..TagConfig::default()
    };
    let mut p = tag_pool(1, cfg, AutoReset::NextStep);
    p.reset_all(Reseed::Keep).unwrap();
    for _ in 0..400 {
        stay(&mut p);
        if p.outputs().unwrap().episode_terminated()[0].get() {
            break;
        }
    }
    assert!(
        p.outputs().unwrap().episode_terminated()[0].get(),
        "no termination"
    );
    let (h, s) = parts(&p, 0);
    p.import(&[0], &[record(h, s)]).unwrap();
    accepted += 1;
    // Truncated (pending reset): evaders alive, the step limit reached.
    let options = PoolOptions {
        max_episode_steps: NonZeroU32::new(1),
        ..PoolOptions::new(NonZeroU32::MIN, Seed::new(5))
    };
    let mut p = EnvPool::<Tag, Masked>::with_masks(
        TagConfig {
            width: 16,
            height: 16,
            slip: 0,
            ..TagConfig::default()
        },
        options,
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    stay(&mut p);
    assert!(p.outputs().unwrap().episode_truncated()[0].get());
    let (h, s) = parts(&p, 0);
    p.import(&[0], &[record(h, s)]).unwrap();
    accepted += 1;
    assert_eq!(accepted, 5);
}

#[test]
fn test_cartpole_import_requires_its_slot_alive_once_reset() {
    let mut p = EnvPool::<CartPole>::new(
        CartPoleConfig::default(),
        PoolOptions::new(NonZeroU32::MIN, Seed::new(1)),
    )
    .unwrap();
    // A never-reset record is accepted as is.
    let r = p.snapshot(0).unwrap();
    let never = DecodedRecord::from_parts(*r.header(), *r.state()).unwrap();
    p.import(&[0], &[never]).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let r = p.snapshot(0).unwrap();
    let mut h = *r.header();
    h.agents[0].alive = Bool8::FALSE;
    h.agents[0].acting = Bool8::FALSE;
    let dead = DecodedRecord::from_parts(h, *r.state()).unwrap();
    assert!(matches!(
        p.import(&[0], &[dead]),
        Err(EngineError::Invariant { .. })
    ));
}
