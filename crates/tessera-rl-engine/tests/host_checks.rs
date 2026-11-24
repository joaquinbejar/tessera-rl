// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `cargo tessera check` functions (#351): each passes on the test
//! environments and fails, with the expected error, on a probe environment
//! that breaks its rule.

#![cfg(feature = "testing")]
#![expect(clippy::unwrap_used, reason = "a failure is the assertion")]

mod host_support;
mod host_walk;

use core::num::{NonZeroU32, NonZeroUsize};
use core::sync::atomic::{AtomicU64, Ordering};

use host_support::{AGENTS, Book, BookConfig};
use host_walk::{Walk, WalkConfig};
use tessera_rl_core::config::{CanonicalConfig, write_entry};
use tessera_rl_core::encoding::{Encoder, Sink};
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, EnvConfig, EpisodeEnd, HostEnvironment, HostError, HostOut, HostSeed, HostSnapshot,
    SnapshotReject, SpaceError,
};
use tessera_rl_engine::EngineError;
use tessera_rl_engine::host_testing::{
    CheckError, CheckRun, check_continuation, check_determinism, check_invariants,
    check_snapshot_continuation, check_spaces,
};

fn run(calls: usize) -> CheckRun {
    CheckRun::new(NonZeroU32::new(4).unwrap(), 11, calls)
        .with_workers(NonZeroUsize::new(2).unwrap())
}

/// What the probe gets wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    /// Reads a process-wide counter: runs differ.
    Global,
    /// `reset` keeps the running total.
    Sticky,
    /// Observes 101 at step 2, above the bound 100.
    Escape,
    /// Spawns slot 1 without `SPAWNS`.
    Spawn,
    /// `save` drops the running total.
    Forgetful,
}

#[derive(Debug, Clone, Copy)]
struct ProbeConfig {
    fault: Fault,
}

impl EnvConfig for ProbeConfig {}

impl CanonicalConfig for ProbeConfig {
    const ADAPTIVE_FIELDS: &'static [&'static str] = &[];
    const STATIC_ENTRIES: u16 = 1;

    fn write_entries<S: Sink>(&self, out: &mut Encoder<S>) {
        write_entry(out, "fault", &(self.fault as u32));
    }
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Two slots, one alive at reset; four steps per episode.
#[derive(Debug)]
struct Probe {
    fault: Fault,
    step: u32,
    total: i32,
}

fn slot(i: u16) -> AgentSlot {
    AgentSlot::new(i, 2).unwrap()
}

impl HostEnvironment for Probe {
    const ENV_ID: &'static str = "test/Probe-v1";
    const ENV_VERSION: u32 = 1;
    const MAX_AGENTS: usize = 2;
    type Config = ProbeConfig;

    fn new(cfg: &ProbeConfig) -> Result<Self, HostError> {
        Ok(Self {
            fault: cfg.fault,
            step: 0,
            total: 0,
        })
    }

    fn initial_agents(_: &ProbeConfig) -> usize {
        1
    }

    fn observation_space(_: &ProbeConfig) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![2],
            dtype: DType::I32,
            bounds: BoxBounds::Int {
                low: vec![0],
                high: vec![100],
            },
        })
    }

    fn action_space(_: &ProbeConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 3 })
    }

    fn reset(&mut self, _: HostSeed, _: &mut HostOut<'_>) {
        self.step = 0;
        if self.fault != Fault::Sticky {
            self.total = 0;
        }
    }

    fn step(&mut self, actions: &Actions<'_, ()>, out: &mut HostOut<'_>) {
        self.step += 1;
        self.total = (self.total + actions.discrete(slot(0), 0).unwrap_or(0)) % 50;
        if self.fault == Fault::Global {
            let n = COUNTER.fetch_add(1, Ordering::Relaxed) % 7;
            self.total = (self.total + i32::try_from(n).unwrap()) % 50;
        }
        if self.fault == Fault::Spawn && self.step == 1 {
            out.agent_spawned(slot(1));
        }
        out.set_reward(slot(0), 1.0);
        if self.step >= 4 {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(&self, _: AgentSlot, obs: &mut ObsWriter<'_, '_, ()>) {
        let first = if self.fault == Fault::Escape && self.step == 2 {
            101
        } else {
            i32::try_from(self.step).unwrap()
        };
        if let Some(leaf) = obs.leaf(0) {
            for (bytes, v) in leaf
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip([first, self.total])
            {
                bytes.copy_from_slice(&v.to_le_bytes());
            }
        }
    }
}

impl HostSnapshot for Probe {
    const SNAPSHOT_VERSION: u32 = 1;

    fn max_snapshot_len(_: &ProbeConfig) -> usize {
        8
    }

    fn save(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.step.to_le_bytes());
        let total = if self.fault == Fault::Forgetful {
            0
        } else {
            self.total
        };
        out.extend_from_slice(&total.to_le_bytes());
    }

    fn check_snapshot(_: &ProbeConfig, bytes: &[u8]) -> Result<(), SnapshotReject> {
        if bytes.len() == 8 {
            Ok(())
        } else {
            Err(SnapshotReject {
                code: 1,
                reason: "eight bytes",
            })
        }
    }

    fn load(&mut self, bytes: &[u8]) {
        let word = |at: usize| -> [u8; 4] {
            bytes
                .get(at..at + 4)
                .and_then(|b| b.try_into().ok())
                .unwrap_or([0; 4])
        };
        self.step = u32::from_le_bytes(word(0));
        self.total = i32::from_le_bytes(word(4));
    }
}

fn probe(fault: Fault) -> impl Fn() -> ProbeConfig {
    move || ProbeConfig { fault }
}

#[test]
fn test_check_determinism_book_and_walk_pass_with_one_hash_per_call() {
    let report = check_determinism::<Book, AGENTS>(&BookConfig::default, &run(12)).unwrap();
    assert_eq!(report.call_hashes.len(), 13);
    assert_eq!(report.workers, 2);
    let again = check_determinism::<Book, AGENTS>(&BookConfig::default, &run(12)).unwrap();
    assert_eq!(
        report.call_hashes, again.call_hashes,
        "the hashes are goldens"
    );
    check_determinism::<Walk, 1>(&|| WalkConfig { length: 64 }, &run(12)).unwrap();
}

#[test]
fn test_check_determinism_action_seed_changes_the_hashes() {
    let a = check_determinism::<Book, AGENTS>(&BookConfig::default, &run(8)).unwrap();
    let b = check_determinism::<Book, AGENTS>(&BookConfig::default, &run(8).with_action_seed(99))
        .unwrap();
    assert_eq!(
        a.call_hashes[0], b.call_hashes[0],
        "the reset takes no action"
    );
    assert_ne!(a.call_hashes, b.call_hashes);
}

#[test]
fn test_check_determinism_global_state_diverges() {
    let err = check_determinism::<Probe, 2>(&probe(Fault::Global), &run(8)).unwrap_err();
    assert!(
        matches!(
            err,
            CheckError::Diverged {
                left: "one Inline run",
                right: "another Inline run",
                what: "observations",
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn test_check_continuation_book_passes() {
    let report = check_continuation::<Book, AGENTS>(&BookConfig::default, &run(10), 7).unwrap();
    assert_eq!((report.at, report.call_hashes.len()), (7, 11));
    check_continuation::<Probe, 2>(&probe(Fault::None), &run(10), 3).unwrap();
}

#[test]
fn test_check_continuation_sticky_reset_diverges() {
    let err = check_continuation::<Probe, 2>(&probe(Fault::Sticky), &run(10), 3).unwrap_err();
    assert!(matches!(err, CheckError::Diverged { .. }), "{err:?}");
}

#[test]
fn test_check_snapshot_continuation_walk_and_probe_pass() {
    check_snapshot_continuation::<Walk, 1>(&|| WalkConfig { length: 64 }, &run(10), 5).unwrap();
    check_snapshot_continuation::<Probe, 2>(&probe(Fault::None), &run(10), 2).unwrap();
}

#[test]
fn test_check_snapshot_continuation_forgetful_save_diverges() {
    let err =
        check_snapshot_continuation::<Probe, 2>(&probe(Fault::Forgetful), &run(10), 2).unwrap_err();
    assert!(matches!(err, CheckError::Diverged { .. }), "{err:?}");
}

#[test]
fn test_check_spaces_book_passes_and_counts_values() {
    let report = check_spaces::<Book, AGENTS>(&BookConfig::default, &run(6)).unwrap();
    // At most 7 calls x 4 instances x 2 agents x 4 values; the reset alone
    // checks every alive agent.
    assert_eq!(report.calls, 7);
    assert!((4 * 2 * 4..=7 * 4 * 2 * 4).contains(&report.values));
}

#[test]
fn test_check_spaces_value_above_bound_is_outside() {
    let err = check_spaces::<Probe, 2>(&probe(Fault::Escape), &run(6)).unwrap_err();
    assert!(
        matches!(
            err,
            CheckError::OutsideSpace {
                call: 2,
                element: 0,
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn test_check_spaces_nan_reward_is_not_finite() {
    let cfg = || BookConfig {
        nan_reward_at: 2,
        ..BookConfig::default()
    };
    let err = check_spaces::<Book, AGENTS>(&cfg, &run(6)).unwrap_err();
    // Under `debug-checks` the engine poisons the pool on the NaN first.
    if tessera_rl_engine::DEBUG_CHECKS {
        assert!(
            matches!(err, CheckError::Engine(EngineError::NanDetected { .. })),
            "{err:?}"
        );
    } else {
        assert!(
            matches!(err, CheckError::RewardNotFinite { call: 2, .. }),
            "{err:?}"
        );
    }
}

#[test]
fn test_check_invariants_book_and_probe_pass() {
    let report = check_invariants::<Book, AGENTS>(&BookConfig::default, &run(12)).unwrap();
    assert!(report.episodes_ended > 0);
    assert_eq!((report.aborts, report.fp_resets), (0, 0));
    check_invariants::<Probe, 2>(&probe(Fault::None), &run(12)).unwrap();
}

#[test]
fn test_check_invariants_spawn_without_spawns_is_invalid() {
    let err = check_invariants::<Probe, 2>(&probe(Fault::Spawn), &run(12)).unwrap_err();
    // Under `debug-checks` the engine poisons the pool on the event itself.
    if tessera_rl_engine::DEBUG_CHECKS {
        assert!(
            matches!(
                err,
                CheckError::Engine(EngineError::InvalidAgentEvent { .. })
            ),
            "{err:?}"
        );
    } else {
        assert!(
            matches!(err, CheckError::Invariant { call: 4, .. }),
            "{err:?}"
        );
    }
}
