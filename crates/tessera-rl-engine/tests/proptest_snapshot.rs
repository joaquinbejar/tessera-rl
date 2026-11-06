// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property tests of snapshots (ADR-0024 layer 4): records round-trip
//! through bytes and import, `fork` continues identically under any
//! actions, and `fork_branched` is deterministic per label and diverges
//! across labels.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "property tests: a failure is the assertion"
)]

mod common;

use core::num::NonZeroU32;

use common::{CountConfig, Counter};
use proptest::prelude::*;
use tessera_rl_core::StateHash;
use tessera_rl_core::encoding::{decode_from_slice, encode_to_vec};
use tessera_rl_engine::{ActionInput, DecodedRecord, EnvPool, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;

fn pool(n: u32, seed: u64) -> EnvPool<Counter> {
    let cfg = CountConfig {
        length: 1_000,
        ..CountConfig::default()
    };
    let mut p = EnvPool::new(
        cfg,
        PoolOptions::new(NonZeroU32::new(n).unwrap(), Seed::new(seed)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p
}

/// Legal Counter actions (agent 0 may not take 2), two agents per instance.
fn script(instances: usize) -> impl Strategy<Value = Vec<Vec<i32>>> {
    proptest::collection::vec(
        proptest::collection::vec((0..2i32, 0..3i32), instances)
            .prop_map(|rows| rows.into_iter().flat_map(|(a, b)| [a, b]).collect()),
        1..30,
    )
}

fn step(p: &mut EnvPool<Counter>, acts: &[i32]) {
    p.step(ActionInput {
        discrete: acts,
        continuous: &[],
    })
    .unwrap();
}

/// The behavioral trajectory of instances 1 and 2 after `fork_branched`
/// from instance 0: per step, each one's author-state hash, observation row
/// and rewards (the header and its lineage are excluded).
type Trajectory = Vec<[(u64, Vec<u8>, Vec<u32>); 2]>;

fn branched(seed: u64, labels: [u64; 2], ignore_rng: bool) -> Trajectory {
    let cfg = CountConfig {
        length: 1_000,
        ignore_rng,
        ..CountConfig::default()
    };
    let mut p = EnvPool::<Counter>::new(
        cfg,
        PoolOptions::new(NonZeroU32::new(3).unwrap(), Seed::new(seed)),
    )
    .unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    p.fork_branched(0, &[1, 2], Some(&labels)).unwrap();
    (0..40)
        .map(|_| {
            step(&mut p, &[1; 6]);
            let out = p.outputs().unwrap();
            let obs = out.observation(0).unwrap();
            let per = obs.len() / 3;
            [1usize, 2].map(|i| {
                (
                    StateHash::of(p.state(u32::try_from(i).unwrap()).unwrap()).get(),
                    obs[i * per..(i + 1) * per].to_vec(),
                    out.rewards()[i * 2..i * 2 + 2]
                        .iter()
                        .map(|r| r.to_bits())
                        .collect(),
                )
            })
        })
        .collect()
}

/// Whether the two branches differ at some step.
fn diverges(t: &Trajectory) -> bool {
    t.iter().any(|[a, b]| a != b)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn test_record_bytes_and_import_round_trip(seed in any::<u64>(), steps in script(2)) {
        let mut p = pool(2, seed);
        for acts in &steps {
            step(&mut p, acts);
        }
        let record = p.snapshot(0).unwrap();
        let bytes = record.to_bytes();
        let decoded = decode_from_slice::<DecodedRecord<Counter>>(&bytes).unwrap();
        prop_assert_eq!(encode_to_vec(&decoded), bytes);
        p.import(&[1], &[decoded]).unwrap();
        prop_assert_eq!(p.state_hash(1).unwrap(), record.hash());
        prop_assert_eq!(p.snapshot(1).unwrap().hash(), record.hash());
    }

    #[test]
    fn test_fork_continues_identically_under_any_actions(
        seed in any::<u64>(), before in script(3), after in script(3),
    ) {
        let mut p = pool(3, seed);
        for acts in &before {
            step(&mut p, acts);
        }
        p.fork(0, &[2]).unwrap();
        for acts in &after {
            // Instance 2 replays instance 0's actions.
            let mut acts = acts.clone();
            acts[4] = acts[0];
            acts[5] = acts[1];
            step(&mut p, &acts);
            prop_assert_eq!(p.state_hash(0).unwrap(), p.state_hash(2).unwrap());
        }
    }

    #[test]
    fn test_fork_branched_is_deterministic_per_label_and_diverges_across(
        seed in any::<u64>(), label in any::<u64>(), other in any::<u64>(),
    ) {
        prop_assume!(label != other);
        let a = branched(seed, [label, other], false);
        let b = branched(seed, [label, other], false);
        prop_assert_eq!(&a, &b, "deterministic per label");
        // Behavior, not the header: the branches' trajectories (author
        // state, observations, rewards) must differ at some step.
        prop_assert!(diverges(&a), "branches never diverge");
    }

    /// Negative control: an environment that ignores its RNG cannot
    /// diverge, and the checker must see that.
    #[test]
    fn test_fork_branched_divergence_check_rejects_an_rng_free_env(
        seed in any::<u64>(), label in any::<u64>(), other in any::<u64>(),
    ) {
        prop_assume!(label != other);
        prop_assert!(!diverges(&branched(seed, [label, other], true)));
    }
}
