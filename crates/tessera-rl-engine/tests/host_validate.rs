// SPDX-License-Identifier: MIT OR Apache-2.0

//! Joint-action validation in host pools (ADR-0065 item 2): `validate`
//! runs before any instance steps, a rejection mutates nothing, and Sync
//! reports the same rejection as Inline.

#![cfg(feature = "host")]
#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test helpers unwrap and slice: a failure is the assertion"
)]

mod host_script;

use core::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use host_script::{NOTHING, REJECT_CODE, SLOTS, Script, ScriptConfig, Shared, TERMINATE};
use tessera_rl_engine::{
    ActionInput, EngineError, HostPool, Outputs, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;

const N: usize = 8;

type Pool<const T: u8, const V: bool> = HostPool<Script<T, V>, SLOTS>;

fn pool<const T: u8, const V: bool>(shared: &Arc<Shared>, workers: usize) -> Pool<T, V> {
    let mut o = PoolOptions::new(
        NonZeroU32::new(u32::try_from(N).unwrap()).unwrap(),
        Seed::new(4),
    );
    // Two instances per chunk, so Sync spreads them over the workers.
    o.chunk_size = NonZeroU32::new(2).unwrap();
    let p = Pool::<T, V>::new(ScriptConfig::new(shared), o).unwrap();
    if workers > 0 {
        p.with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
            .unwrap()
    } else {
        p
    }
}

fn step<const T: u8, const V: bool>(
    p: &mut Pool<T, V>,
    special: &[(usize, usize, i32)],
) -> Result<(), EngineError> {
    let mut a = vec![NOTHING; N * SLOTS];
    for &(i, s, v) in special {
        a[i * SLOTS + s] = v;
    }
    p.step(ActionInput {
        discrete: &a,
        continuous: &[],
    })
    .map(|_| ())
}

fn total(counts: &[std::sync::atomic::AtomicU64; host_script::MAX_INSTANCES]) -> u64 {
    counts.iter().map(|c| c.load(Ordering::Relaxed)).sum()
}

fn snapshot(out: &Outputs) -> Vec<Vec<u8>> {
    let b = |f: &[tessera_rl_core::Bool8]| f.iter().map(|x| u8::from(x.get())).collect::<Vec<_>>();
    vec![
        out.observation(0).unwrap().to_vec(),
        out.rewards()
            .iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect(),
        b(out.agent_truncated()),
        b(out.acting()),
        b(out.masks()),
        b(out.pending_reset()),
        b(out.aborted()),
    ]
}

fn rejected(r: Result<(), EngineError>) -> (u32, u32, &'static str) {
    match r {
        Err(EngineError::ActionRejected {
            instance,
            code,
            reason,
        }) => (instance, code, reason),
        other => {
            // Fails with the actual result.
            assert_eq!(format!("{other:?}"), "Err(ActionRejected { .. })");
            (u32::MAX, 0, "")
        }
    }
}

#[test]
fn test_host_validate_rejection_mutates_nothing() {
    for workers in [0, 2] {
        let shared = Shared::new();
        let mut p = pool::<0, true>(&shared, workers);
        p.reset_all(Reseed::Keep).unwrap();
        step(&mut p, &[]).unwrap();
        let before = snapshot(p.outputs().unwrap());
        let generation = p.outputs().unwrap().storage().generation();
        let steps = total(&shared.steps);
        shared.reject.store(1 << 3, Ordering::Relaxed);
        let r = step(&mut p, &[]);
        let err = r.clone().unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("instance 3: the script refused (code {})", REJECT_CODE + 3)
        );
        assert_eq!(rejected(r), (3, REJECT_CODE + 3, "the script refused"));
        // No instance stepped, and the outputs, including their write
        // generation, are the previous call's.
        assert_eq!(total(&shared.steps), steps, "workers={workers}");
        assert_eq!(snapshot(p.outputs().unwrap()), before);
        assert_eq!(p.outputs().unwrap().storage().generation(), generation);
        assert!(!p.is_poisoned());
        // The pool keeps working once the action is accepted.
        shared.reject.store(0, Ordering::Relaxed);
        step(&mut p, &[]).unwrap();
        assert_eq!(total(&shared.steps), steps + u64::try_from(N).unwrap());
    }
}

#[test]
fn test_host_validate_sync_reports_the_lowest_rejection_as_inline() {
    // Chunks of two over two workers: instance 6 is in worker 1's chunk 3,
    // instance 1 in worker 0's chunk 0.
    for mask in [1u32 << 6, (1 << 6) | (1 << 1), (1 << 7) | (1 << 2)] {
        let mut results = Vec::new();
        for workers in [0, 2] {
            let shared = Shared::new();
            let mut p = pool::<0, true>(&shared, workers);
            p.reset_all(Reseed::Keep).unwrap();
            let before = snapshot(p.outputs().unwrap());
            shared.reject.store(mask, Ordering::Relaxed);
            let r = rejected(step(&mut p, &[]));
            assert_eq!(total(&shared.steps), 0, "no chunk steps");
            assert_eq!(snapshot(p.outputs().unwrap()), before);
            results.push(r);
        }
        assert_eq!(results[0], results[1], "{mask:#b}");
        assert_eq!(results[0].0, mask.trailing_zeros());
    }
}

#[test]
fn test_host_validate_skips_pending_rows_and_non_acting_instances() {
    // Pending rows: an ended episode under NextStep is reset, not validated.
    let shared = Shared::new();
    let mut p = pool::<0, true>(&shared, 0);
    p.reset_all(Reseed::Keep).unwrap();
    step(&mut p, &[(5, 0, TERMINATE)]).unwrap();
    let v = Shared::get(&shared.validates, 5);
    step(&mut p, &[]).unwrap();
    assert_eq!(Shared::get(&shared.validates, 5), v);
    assert_eq!(Shared::get(&shared.validates, 4), 2);
    // Mixed with no acting agent: not validated, and it still steps.
    let shared = Shared::new();
    let mut p = pool::<2, true>(&shared, 0);
    p.reset_all(Reseed::Keep).unwrap();
    shared.mixed_acting.store(false, Ordering::Relaxed);
    step(&mut p, &[]).unwrap();
    let steps = Shared::get(&shared.steps, 0);
    let validates = Shared::get(&shared.validates, 0);
    step(&mut p, &[]).unwrap();
    assert_eq!(Shared::get(&shared.validates, 0), validates);
    assert_eq!(Shared::get(&shared.steps, 0), steps + 1);
    // VALIDATES = false: never called, even when it would reject.
    let shared = Shared::new();
    let mut p = pool::<0, false>(&shared, 2);
    p.reset_all(Reseed::Keep).unwrap();
    shared.reject.store(u32::MAX, Ordering::Relaxed);
    step(&mut p, &[]).unwrap();
    assert_eq!(total(&shared.validates), 0);
}

#[test]
fn test_host_validate_fp_controls_are_put_back_and_counted() {
    for workers in [0, 2] {
        let shared = Shared::new();
        let mut p = pool::<0, true>(&shared, workers);
        p.reset_all(Reseed::Keep).unwrap();
        let before = p.host_fp_resets();
        shared.hostile_validate.store(true, Ordering::Relaxed);
        step(&mut p, &[]).unwrap();
        // One per instance's validate.
        assert_eq!(p.host_fp_resets() - before, u64::try_from(N).unwrap());
        assert!(tessera_rl_num::fp::is_canonical());
    }
}

#[test]
fn test_host_validate_panic_poisons() {
    for workers in [0, 2] {
        let shared = Shared::new();
        let mut p = pool::<0, true>(&shared, workers);
        p.reset_all(Reseed::Keep).unwrap();
        shared.panic_validate.store(1 << 4, Ordering::Relaxed);
        let r = step(&mut p, &[]);
        assert!(matches!(r, Err(EngineError::Panicked)), "{r:?}");
        assert!(p.is_poisoned());
        assert_eq!(total(&shared.steps), 0);
    }
}

#[test]
fn test_host_validate_impure_rerun_is_still_a_rejection() {
    // Sync: a worker rejects instance 5, and the serial re-run on this
    // thread accepts it (an impure `validate`). The call is still rejected,
    // with code 0 and ADR-0065's reason, and nothing steps.
    for workers in [2, 3] {
        let shared = Shared::new();
        let mut p = pool::<0, true>(&shared, workers);
        p.reset_all(Reseed::Keep).unwrap();
        let before = snapshot(p.outputs().unwrap());
        shared.reject_once.store(1 << 5, Ordering::Relaxed);
        assert_eq!(
            rejected(step(&mut p, &[])),
            (
                5,
                0,
                "validate was not pure: the re-run accepted an action a worker rejected"
            )
        );
        assert_eq!(total(&shared.steps), 0);
        assert_eq!(snapshot(p.outputs().unwrap()), before);
        assert!(!p.is_poisoned());
    }
}
