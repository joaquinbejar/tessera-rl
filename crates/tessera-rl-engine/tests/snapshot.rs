// SPDX-License-Identifier: MIT OR Apache-2.0

//! Snapshot, restore, import and fork (ADR-0010, ADR-0007, ADR-0029,
//! ADR-0039), including the normative continuation test: same slot,
//! another slot, another pool.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers unwrap and index: a failure is the assertion"
)]

mod common;

use core::num::NonZeroU32;

use common::{CountConfig, Counter, Counts};
use tessera_rl_core::encoding::{decode_from_slice, encode_to_vec};
use tessera_rl_engine::{
    ActionInput, AutoReset, DecodedRecord, EngineError, EnvPool, Masked, PoolContext, PoolOptions,
    Reseed,
};
use tessera_rl_num::identity::Seed;

type Pool = EnvPool<Counter, Masked>;

fn long() -> CountConfig {
    CountConfig {
        length: 50,
        ..CountConfig::default()
    }
}

fn pool_with(n: u32, seed: u64, cfg: CountConfig, mode: AutoReset) -> Pool {
    let options = PoolOptions {
        auto_reset: mode,
        ..PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed))
    };
    let mut p = EnvPool::with_masks(cfg, options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

fn pool(n: u32, seed: u64) -> Pool {
    pool_with(n, seed, long(), AutoReset::NextStep)
}

/// Everything the continuation test compares for instance `i`; floats as
/// bits.
#[derive(Debug, PartialEq)]
struct Frame {
    obs: Vec<u8>,
    rewards: Vec<u32>,
    flags: Vec<bool>,
    masks: Vec<bool>,
    generations: Vec<u32>,
    stats: Vec<u32>,
    hash: u64,
}

fn frame(p: &Pool, i: usize) -> Frame {
    let out = p.outputs();
    let rows = |len: usize| i * len..(i + 1) * len;
    let mask_len = out.masks().len() / out.rewards().len();
    let bits = |b: &[tessera_rl_core::Bool8]| b.iter().map(|f| f.get()).collect::<Vec<_>>();
    let mut flags = bits(&out.agent_terminated()[rows(2)]);
    flags.extend(bits(&out.agent_truncated()[rows(2)]));
    flags.extend(bits(&out.alive()[rows(2)]));
    flags.extend(bits(&out.acting()[rows(2)]));
    flags.push(out.episode_terminated()[i].get());
    flags.push(out.episode_truncated()[i].get());
    Frame {
        obs: out.observation(0).unwrap()[rows(16)].to_vec(),
        rewards: out.rewards()[rows(2)].iter().map(|r| r.to_bits()).collect(),
        flags,
        masks: bits(&out.masks()[rows(2 * mask_len)]),
        generations: out.generations()[rows(2)].to_vec(),
        stats: vec![
            out.episode_return()[i].to_bits(),
            out.episode_length()[i].to_bits(),
            out.episode_illegal_actions()[i].to_bits(),
        ],
        hash: p.state_hash(u32::try_from(i).unwrap()).unwrap().get(),
    }
}

/// Runs the script on instance `i` (others act 0); returns its frames.
fn run(p: &mut Pool, i: usize, script: &[[i32; 2]]) -> Vec<Frame> {
    let n = usize::try_from(p.options().num_envs.get()).unwrap();
    script
        .iter()
        .map(|act| {
            let mut actions = vec![0; n * 2];
            actions[i * 2..i * 2 + 2].copy_from_slice(act);
            p.step(ActionInput {
                discrete: &actions,
                continuous: &[],
            })
            .unwrap();
            frame(p, i)
        })
        .collect()
}

/// Legal for both agents (action 2 is illegal for agent 0), so the script
/// also runs under `debug-checks`.
const SCRIPT: [[i32; 2]; 8] = [
    [1, 0],
    [0, 2],
    [1, 1],
    [1, 2],
    [0, 0],
    [0, 2],
    [1, 1],
    [0, 1],
];

/// Snapshots instance 0 after `before` steps, then checks that it
/// continues identically in the same slot, another slot and another pool
/// with another seed.
fn continuation(cfg: &CountConfig, mode: AutoReset, before: usize) {
    let mut a = pool_with(3, 1, cfg.clone(), mode);
    run(&mut a, 0, &SCRIPT[..before]);
    let snap = a.snapshot(0).unwrap();
    let expected = run(&mut a, 0, &SCRIPT);

    a.restore(0, &snap).unwrap();
    assert_eq!(run(&mut a, 0, &SCRIPT), expected, "same slot");

    a.restore(2, &snap).unwrap();
    assert_eq!(run(&mut a, 2, &SCRIPT), expected, "other slot");

    let mut b = pool_with(4, 999, cfg.clone(), mode);
    b.restore(1, &snap).unwrap();
    assert_eq!(run(&mut b, 1, &SCRIPT), expected, "other pool, other seed");
}

#[test]
fn test_continuation_mid_episode() {
    continuation(&long(), AutoReset::NextStep, 2);
}

#[test]
fn test_continuation_across_an_episode_boundary() {
    let short = CountConfig {
        length: 3,
        ..CountConfig::default()
    };
    for before in [2, 3] {
        continuation(&short, AutoReset::NextStep, before);
        continuation(&short, AutoReset::SameStep, before);
    }
}

#[test]
fn test_continuation_across_death_and_respawn() {
    let cfg = CountConfig {
        kill_at: 0,
        respawn: true,
        length: 50,
        ..CountConfig::default()
    };
    // Snapshots before the death, between the death and the respawn, and
    // after the respawn.
    for before in [0, 1, 2] {
        continuation(&cfg, AutoReset::NextStep, before);
    }
}

#[test]
fn test_edge_fixtures_survive_snapshot_and_encoding() {
    let mut p = pool(1, 4);
    run(&mut p, 0, &SCRIPT[..1]);
    let snap = p.snapshot(0).unwrap();
    // After an odd step the bias is the smallest subnormal; after reset
    // -0.0; the phase enum has no zero discriminant.
    assert_eq!(snap.state().bias.to_bits(), 1);
    let bytes = snap.to_bytes();
    let decoded: DecodedRecord<Counter> = decode_from_slice(&bytes).unwrap();
    assert_eq!(decoded.state(), snap.state());
    p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(
        p.snapshot(0).unwrap().state().bias.to_bits(),
        (-0.0f32).to_bits()
    );
    let zero_state = encode_to_vec(&Counts {
        phase: common::Phase::Endgame,
        ..Counts::default()
    });
    let mut corrupt = zero_state;
    // value (8) + steps (4) + bias (4) + dead (1): the phase is byte 17.
    corrupt[17] = 0;
    assert!(decode_from_slice::<Counts>(&corrupt).is_err());
}

#[test]
fn test_restore_recomputes_observations_and_masks() {
    let mut a = pool(2, 3);
    run(&mut a, 0, &SCRIPT[..1]);
    let before = frame(&a, 0);
    let snap = a.snapshot(0).unwrap();
    a.restore(1, &snap).unwrap();
    let copied = frame(&a, 1);
    assert_eq!(
        (copied.obs, copied.masks, copied.hash),
        (before.obs, before.masks, before.hash)
    );
    assert_eq!(a.outputs().rewards()[2..4], [0.0, 0.0]);
}

#[test]
fn test_restore_clears_terminal_observation_rows() {
    let cfg = CountConfig {
        length: 2,
        ..CountConfig::default()
    };
    let mut a = pool_with(2, 5, cfg.clone(), AutoReset::SameStep);
    run(&mut a, 0, &SCRIPT[..2]);
    // Instance 0 just ended an episode: its terminal rows are set.
    assert!(
        a.outputs().terminal_observation(0).unwrap()[..16]
            .iter()
            .any(|&b| b != 0)
    );
    let snap = a.snapshot(0).unwrap();
    a.restore(0, &snap).unwrap();
    a.restore(1, &snap).unwrap();
    let mut b = pool_with(3, 9, cfg, AutoReset::SameStep);
    b.restore(2, &snap).unwrap();
    assert_eq!(a.outputs().terminal_observation(0).unwrap()[..32], [0; 32]);
    assert_eq!(
        b.outputs().terminal_observation(0).unwrap()[32..48],
        [0; 16]
    );
}

#[test]
fn test_fork_gives_identical_futures() {
    let mut a = pool(3, 5);
    run(&mut a, 0, &SCRIPT[..2]);
    a.fork(0, &[1, 2]).unwrap();
    let act = [1, 2, 1, 2, 1, 2];
    for _ in 0..4 {
        a.step(ActionInput {
            discrete: &act,
            continuous: &[],
        })
        .unwrap();
        let h: Vec<u64> = (0..3).map(|i| a.state_hash(i).unwrap().get()).collect();
        assert!(h.iter().all(|&x| x == h[0]));
    }
}

#[test]
fn test_fork_branched_diverges_deterministically() {
    let branch = |seed| {
        let mut p = pool(3, seed);
        p.fork_branched(0, &[1, 2], None).unwrap();
        let act = [1; 6];
        let mut hashes = Vec::new();
        for _ in 0..10 {
            p.step(ActionInput {
                discrete: &act,
                continuous: &[],
            })
            .unwrap();
            hashes.push(
                (0..3)
                    .map(|i| p.state_hash(i).unwrap().get())
                    .collect::<Vec<_>>(),
            );
        }
        (hashes, p.pool_context().unwrap())
    };
    let (a, ctx) = branch(7);
    let (b, _) = branch(7);
    assert_eq!(a, b, "deterministic");
    assert_eq!(ctx, PoolContext { branch_counter: 1 });
    let last = a.last().unwrap();
    assert!(last[0] != last[1] && last[1] != last[2], "branches diverge");
}

#[test]
fn test_fork_branched_labels_and_counter_errors() {
    let mut p = pool(3, 1);
    let before = p.state_hash(1).unwrap();
    assert_eq!(
        p.fork_branched(0, &[1, 2], Some(&[4, 4])),
        Err(EngineError::DuplicateLabel { label: 4 })
    );
    assert_eq!(
        p.fork_branched(0, &[1, 2], Some(&[4])),
        Err(EngineError::LabelCount {
            labels: 1,
            destinations: 2
        })
    );
    p.restore_pool_context(PoolContext {
        branch_counter: u64::MAX,
    })
    .unwrap();
    assert_eq!(
        p.fork_branched(0, &[1], None),
        Err(EngineError::BranchCounterExhausted)
    );
    assert_eq!(p.state_hash(1).unwrap(), before, "nothing mutated");
    assert_eq!(
        p.fork(0, &[1, 1]),
        Err(EngineError::DuplicateInstance { id: 1 })
    );
}

fn poisoned_pool() -> Pool {
    let cfg = CountConfig {
        panic_at: 1,
        ..CountConfig::default()
    };
    let options = PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(1));
    let mut bad = EnvPool::<Counter, Masked>::with_masks(cfg, options).unwrap();
    bad.reset_all(Reseed::Keep).unwrap();
    assert!(
        bad.step(ActionInput {
            discrete: &[0; 4],
            continuous: &[]
        })
        .is_err()
    );
    bad
}

#[test]
fn test_poisoned_pool_accepts_only_full_restore_and_import() {
    let p = pool(2, 1);
    let mut records = [p.snapshot(0).unwrap(); 2];
    p.snapshot_all(&mut records).unwrap();
    let mut bad = poisoned_pool();
    assert_eq!(bad.restore(0, &records[0]), Err(EngineError::Poisoned));
    assert_eq!(bad.fork(0, &[1]), Err(EngineError::Poisoned));
    assert_eq!(bad.fork_branched(0, &[1], None), Err(EngineError::Poisoned));
    assert_eq!(bad.snapshot(0).map(|_| ()), Err(EngineError::Poisoned));
    assert_eq!(bad.state_hash(0).map(|_| ()), Err(EngineError::Poisoned));
    assert_eq!(
        bad.snapshot_all(&mut records.clone()),
        Err(EngineError::Poisoned)
    );
    assert_eq!(bad.pool_context(), Err(EngineError::Poisoned));
    assert_eq!(
        bad.restore_pool_context(PoolContext { branch_counter: 0 }),
        Err(EngineError::Poisoned)
    );
    let decoded = records.map(DecodedRecord::from);
    assert_eq!(bad.import(&[0], &decoded[..1]), Err(EngineError::Poisoned));
    assert_eq!(
        bad.restore_all(&records[..1]),
        Err(EngineError::RecordCount {
            records: 1,
            instances: 2
        })
    );
    bad.restore_all(&records).unwrap();
    assert!(!bad.is_poisoned());
    assert_eq!(bad.state_hash(0).unwrap(), p.state_hash(0).unwrap());
}

#[test]
fn test_import_validates_game_invariants_and_headers_atomically() {
    let mut p = pool(2, 1);
    run(&mut p, 0, &SCRIPT[..2]);
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let before = p.state_hash(1).unwrap();

    // A game invariant: steps past the episode length (50).
    let mut state = *good.state();
    state.steps = 999;
    let bad_state = DecodedRecord::from_parts(*good.header(), state);
    assert!(matches!(
        p.import(&[1], &[bad_state]),
        Err(EngineError::Invariant { index: 0, .. })
    ));

    // An inconsistent header: an episode end without a pending reset.
    let mut header = *good.header();
    header.engine.end_kind = 1;
    let bad_header = DecodedRecord::from_parts(header, *good.state());
    assert!(matches!(
        p.import(&[0, 1], &[good, bad_header]),
        Err(EngineError::InvalidRecord { index: 1, .. })
    ));

    // Atomic: the valid first record was not written either.
    assert_eq!(p.state_hash(1).unwrap(), before);
    p.import(&[1], &[good]).unwrap();
    assert_eq!(p.state_hash(1).unwrap(), p.state_hash(0).unwrap());
}

#[test]
fn test_import_all_failure_keeps_the_poison() {
    let p = pool(2, 1);
    let good = DecodedRecord::from(p.snapshot(0).unwrap());
    let mut state = *good.state();
    state.steps = 999;
    let bad = DecodedRecord::from_parts(*good.header(), state);
    let mut poisoned = poisoned_pool();
    assert!(poisoned.import_all(&[good, bad]).is_err());
    assert!(poisoned.is_poisoned());
    poisoned.import_all(&[good, good]).unwrap();
    assert!(!poisoned.is_poisoned());
}

#[test]
fn test_record_encoding_round_trips_and_matches_state_hash() {
    let mut p = pool(1, 2);
    run(&mut p, 0, &SCRIPT[..3]);
    let record = p.snapshot(0).unwrap();
    let decoded: DecodedRecord<Counter> = decode_from_slice(&record.to_bytes()).unwrap();
    let mut q = pool(1, 50);
    q.import(&[0], &[decoded]).unwrap();
    assert_eq!(q.state_hash(0).unwrap(), p.state_hash(0).unwrap());
    assert_eq!(
        tessera_rl_engine::Record::<Counter>::record_size(),
        40 + 2 * 16 + 20
    );
}
