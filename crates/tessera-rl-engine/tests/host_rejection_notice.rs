// SPDX-License-Identifier: MIT OR Apache-2.0

//! The rejection notice of host pools (ADR-0069): with
//! `NOTIFIES_REJECTION`, a rejected call runs `on_rejected` once, on the
//! instance it reports, in Inline and in Sync; without it a rejection
//! mutates nothing (ADR-0065).

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

use host_script::{NOTHING, REJECT_CODE, SLOTS, Script, ScriptConfig, Shared};
use tessera_rl_core::io::{Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionReject, AgentSlot, HostEnvironment, HostError, HostOut, HostSeed, SpaceError,
};
use tessera_rl_engine::{ActionInput, EngineError, HostPool, PoolOptions, Reseed, SyncOptions};
use tessera_rl_num::identity::Seed;

const N: usize = 8;
/// `on_rejected` panics for this instance's code.
const PANIC_ID: u32 = 7;
/// `on_rejected` leaves hostile FP controls for this instance's code.
const HOSTILE_ID: u32 = 4;
/// The first observation value carries `REFUSED_UNIT * refusals`.
const REFUSED_UNIT: i32 = 1000;

/// The scripted environment (validating, simultaneous), plus a count of
/// the notices it received, kept in its own state and shown in its first
/// observation value. `N_` is `NOTIFIES_REJECTION`.
#[derive(Debug)]
struct Notice<const N_: bool> {
    inner: Script<0, true>,
    refused: i32,
}

impl<const N_: bool> HostEnvironment for Notice<N_> {
    const ENV_ID: &'static str = "test/Notice-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = SLOTS;
    const MASKS: bool = true;
    const SPAWNS: bool = true;
    const VALIDATES: bool = true;
    const NOTIFIES_REJECTION: bool = N_;
    type Config = ScriptConfig;

    fn new(cfg: &ScriptConfig) -> Result<Self, HostError> {
        Ok(Self {
            inner: Script::new(cfg)?,
            refused: 0,
        })
    }

    fn initial_agents(cfg: &ScriptConfig) -> usize {
        Script::<0, true>::initial_agents(cfg)
    }

    fn observation_space(cfg: &ScriptConfig) -> Result<Space, SpaceError> {
        Script::<0, true>::observation_space(cfg)
    }

    fn action_space(cfg: &ScriptConfig) -> Result<Space, SpaceError> {
        Script::<0, true>::action_space(cfg)
    }

    fn reset(&mut self, seed: HostSeed, out: &mut HostOut<'_>) {
        self.inner.reset(seed, out);
    }

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut HostOut<'_>) {
        self.inner.step(actions, out);
    }

    fn validate(&self, actions: &Actions<'_, ()>) -> Result<(), ActionReject> {
        self.inner.validate(actions)
    }

    fn on_rejected(&mut self, reject: &ActionReject) {
        assert!(
            reject.code != REJECT_CODE + PANIC_ID,
            "on_rejected panicked"
        );
        if reject.code == REJECT_CODE + HOSTILE_ID {
            tessera_rl_num::fp::hostile::set(tessera_rl_num::fp::hostile::flush_to_zero);
        }
        self.refused += 1;
    }

    fn observe(&self, agent: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>) {
        self.inner.observe(agent, obs);
        if let Some(leaf) = obs.leaf(0)
            && let Some(first) = leaf.as_chunks_mut::<4>().0.first_mut()
        {
            let v = i32::from_le_bytes(*first) + REFUSED_UNIT * self.refused;
            first.copy_from_slice(&v.to_le_bytes());
        }
    }

    fn mask(&self, agent: AgentSlot, mask: &mut MaskWriter<'_>) {
        self.inner.mask(agent, mask);
    }
}

/// The notice count travels with the script's own bytes (ADR-0069 item 7:
/// snapshot, restore and fork carry it like any other state; ADR-0073).
impl<const N_: bool> tessera_rl_core::HostSnapshot for Notice<N_> {
    const SNAPSHOT_VERSION: u32 = 1;

    fn max_snapshot_len(cfg: &ScriptConfig) -> usize {
        Script::<0, true>::max_snapshot_len(cfg) + 4
    }

    fn save(&self, out: &mut Vec<u8>) {
        self.inner.save(out);
        out.extend_from_slice(&self.refused.to_le_bytes());
    }

    fn check_snapshot(
        cfg: &ScriptConfig,
        bytes: &[u8],
    ) -> Result<(), tessera_rl_core::SnapshotReject> {
        let Some(split) = bytes.len().checked_sub(4) else {
            return Err(tessera_rl_core::SnapshotReject {
                code: 1,
                reason: "no notice count",
            });
        };
        Script::<0, true>::check_snapshot(cfg, bytes.get(..split).unwrap_or(&[]))
    }

    fn load(&mut self, bytes: &[u8]) {
        let split = bytes.len().saturating_sub(4);
        self.inner.load(bytes.get(..split).unwrap_or(&[]));
        self.refused = bytes
            .get(split..)
            .and_then(|b| b.try_into().ok())
            .map_or(0, i32::from_le_bytes);
    }
}

type Pool<const N_: bool> = HostPool<Notice<N_>, SLOTS>;

fn pool<const N_: bool>(shared: &Arc<Shared>, workers: usize) -> Pool<N_> {
    let mut o = PoolOptions::new(
        NonZeroU32::new(u32::try_from(N).unwrap()).unwrap(),
        Seed::new(4),
    );
    // Two instances per chunk, so Sync spreads them over the workers.
    o.chunk_size = NonZeroU32::new(2).unwrap();
    let p = Pool::<N_>::new(ScriptConfig::new(shared), o).unwrap();
    if workers > 0 {
        p.with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
            .unwrap()
    } else {
        p
    }
}

fn step<const N_: bool>(p: &mut Pool<N_>) -> Result<(), EngineError> {
    let a = vec![NOTHING; N * SLOTS];
    p.step(ActionInput {
        discrete: &a,
        continuous: &[],
    })
    .map(|_| ())
}

fn reported(r: Result<(), EngineError>) -> (u32, u32) {
    match r {
        Err(EngineError::ActionRejected { instance, code, .. }) => (instance, code),
        other => {
            // Fails with the actual result.
            assert_eq!(format!("{other:?}"), "Err(ActionRejected { .. })");
            (u32::MAX, 0)
        }
    }
}

/// The refusals each instance recorded, read from agent 0's first
/// observation value after an accepted step.
fn refusals<const N_: bool>(p: &mut Pool<N_>, shared: &Shared) -> Vec<i32> {
    shared.reject.store(0, Ordering::Relaxed);
    shared.reject_once.store(0, Ordering::Relaxed);
    step(p).unwrap();
    let obs = p.outputs().unwrap().observation(0).unwrap();
    (0..N)
        .map(|i| {
            let at = i * SLOTS * 3 * 4;
            let v = i32::from_le_bytes(obs[at..at + 4].try_into().unwrap());
            (v - 1 - i32::try_from(i).unwrap()) / REFUSED_UNIT
        })
        .collect()
}

#[test]
fn test_host_rejection_notice_inline_and_sync_notify_the_reported_instance() {
    for mask in [1u32 << 6, (1 << 6) | (1 << 1), (1 << 7) | (1 << 2), 1 << 0] {
        let mut results = Vec::new();
        for workers in [0, 2, 3] {
            let shared = Shared::new();
            let mut p = pool::<true>(&shared, workers);
            p.reset_all(Reseed::Keep).unwrap();
            shared.reject.store(mask, Ordering::Relaxed);
            let (instance, code) = reported(step(&mut p));
            assert_eq!(
                instance,
                mask.trailing_zeros(),
                "{mask:#b} workers={workers}"
            );
            assert_eq!(code, REJECT_CODE + instance);
            assert!(!p.is_poisoned());
            let counts = refusals(&mut p, &shared);
            let mut want = vec![0; N];
            want[usize::try_from(instance).unwrap()] = 1;
            assert_eq!(counts, want, "{mask:#b} workers={workers}");
            results.push((instance, code, counts));
        }
        assert!(results.windows(2).all(|w| w[0] == w[1]), "{mask:#b}");
    }
}

#[test]
fn test_host_rejection_notice_each_rejected_call_notifies_once() {
    for workers in [0, 2] {
        let shared = Shared::new();
        let mut p = pool::<true>(&shared, workers);
        p.reset_all(Reseed::Keep).unwrap();
        shared.reject.store(1 << 3, Ordering::Relaxed);
        for _ in 0..3 {
            assert_eq!(reported(step(&mut p)).0, 3);
        }
        let counts = refusals(&mut p, &shared);
        assert_eq!(counts[3], 3, "workers={workers}");
        assert_eq!(counts.iter().sum::<i32>(), 3);
    }
}

#[test]
fn test_host_rejection_notice_off_mutates_nothing() {
    for workers in [0, 2] {
        let shared = Shared::new();
        let mut p = pool::<false>(&shared, workers);
        p.reset_all(Reseed::Keep).unwrap();
        shared.reject.store(1 << 3, Ordering::Relaxed);
        assert_eq!(reported(step(&mut p)).0, 3);
        assert_eq!(refusals(&mut p, &shared), vec![0; N], "workers={workers}");
    }
}

#[test]
fn test_host_rejection_notice_impure_rerun_does_not_notify() {
    // A worker rejects instance 5 and the serial re-run accepts it: the
    // call is rejected with code 0 and no `ActionReject` to deliver.
    for workers in [2, 3] {
        let shared = Shared::new();
        let mut p = pool::<true>(&shared, workers);
        p.reset_all(Reseed::Keep).unwrap();
        shared.reject_once.store(1 << 5, Ordering::Relaxed);
        assert_eq!(reported(step(&mut p)), (5, 0));
        assert!(!p.is_poisoned());
        assert_eq!(refusals(&mut p, &shared), vec![0; N], "workers={workers}");
    }
}

#[test]
fn test_host_rejection_notice_panic_poisons() {
    for workers in [0, 2] {
        let shared = Shared::new();
        let mut p = pool::<true>(&shared, workers);
        p.reset_all(Reseed::Keep).unwrap();
        shared.reject.store(1 << PANIC_ID, Ordering::Relaxed);
        let r = step(&mut p);
        assert!(matches!(r, Err(EngineError::Panicked)), "{r:?}");
        assert!(p.is_poisoned());
    }
}

#[test]
fn test_host_rejection_notice_fp_controls_are_put_back_and_counted() {
    for workers in [0, 2] {
        let shared = Shared::new();
        let mut p = pool::<true>(&shared, workers);
        p.reset_all(Reseed::Keep).unwrap();
        let before = p.host_fp_resets();
        shared.reject.store(1 << HOSTILE_ID, Ordering::Relaxed);
        assert_eq!(reported(step(&mut p)).0, HOSTILE_ID);
        // One for the notice; `validate` leaves the controls alone here.
        assert_eq!(p.host_fp_resets() - before, 1, "workers={workers}");
        assert!(tessera_rl_num::fp::is_canonical());
    }
}

/// ADR-0069 item 10 and ADR-0073 item 8: a snapshot taken after a notice,
/// restored, reproduces the next step and the notice count.
#[test]
fn test_host_rejection_notice_snapshot_after_notice_restores_next_step() {
    let shared = Shared::new();
    let mut p = pool::<true>(&shared, 0);
    p.reset_all(Reseed::Keep).unwrap();
    shared.reject_once.store(1 << 1, Ordering::Relaxed);
    assert_eq!(reported(step(&mut p)), (1, REJECT_CODE + 1));
    let record = p.snapshot(1).unwrap();
    let next = refusals(&mut p, &shared);
    assert_eq!(next[1], 1);
    let expected = (row(&p), p.state_hash(1).unwrap());
    p.restore(1, &record).unwrap();
    let again = refusals(&mut p, &shared);
    assert_eq!(again, next);
    assert_eq!((row(&p), p.state_hash(1).unwrap(),), expected);
}

/// Instance 1's observation rows.
fn row(p: &Pool<true>) -> Vec<u8> {
    let at = SLOTS * 3 * 4;
    p.outputs().unwrap().observation(0).unwrap()[at..2 * at].to_vec()
}
