// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sync mode: bit-identical to Inline for any worker count, worker panics
//! poison, shutdown joins (ADR-0005, ADR-0006, SPEC 8.5).

#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod common;

use core::num::{NonZeroU32, NonZeroUsize};

use common::{CountConfig, Counter};
use tessera_rl_engine::{
    ActionInput, AutoReset, EngineError, EnvPool, Masked, PoolOptions, Reseed, SyncOptions,
    WaitBackend,
};
use tessera_rl_num::identity::Seed;

const N: u32 = 100;

fn pool(mode: AutoReset, workers: Option<usize>, cfg: CountConfig) -> EnvPool<Counter, Masked> {
    pool_on(mode, workers, cfg, WaitBackend::Auto)
}

fn pool_on(
    mode: AutoReset,
    workers: Option<usize>,
    cfg: CountConfig,
    wait: WaitBackend,
) -> EnvPool<Counter, Masked> {
    let options = PoolOptions {
        auto_reset: mode,
        chunk_size: NonZeroU32::new(8).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(77))
    };
    let p = EnvPool::with_masks(cfg, options).unwrap();
    match workers {
        Some(w) => p
            .with_workers(SyncOptions {
                wait,
                ..SyncOptions::new(NonZeroUsize::new(w).unwrap())
            })
            .unwrap(),
        None => p,
    }
}

/// Everything a step returns, plus every state hash.
fn snapshot_outputs(p: &EnvPool<Counter, Masked>) -> Vec<u64> {
    let out = p.outputs().unwrap();
    let mut v: Vec<u64> = out
        .observation(0)
        .unwrap()
        .iter()
        .map(|&b| u64::from(b))
        .collect();
    v.extend(out.rewards().iter().map(|r| u64::from(r.to_bits())));
    for flags in [
        out.agent_terminated(),
        out.agent_truncated(),
        out.alive(),
        out.masks(),
    ] {
        v.extend(flags.iter().map(|f| u64::from(f.get())));
    }
    v.extend(out.episode_terminated().iter().map(|f| u64::from(f.get())));
    v.extend(out.episode_return().iter().map(|r| u64::from(r.to_bits())));
    if let Some(term) = out.terminal_observation(0) {
        v.extend(term.iter().map(|&b| u64::from(b)));
    }
    v.extend((0..N).map(|i| p.state_hash(i).unwrap().get()));
    v
}

fn trajectory(mode: AutoReset, workers: Option<usize>) -> Vec<Vec<u64>> {
    trajectory_on(mode, workers, WaitBackend::Auto)
}

fn trajectory_on(mode: AutoReset, workers: Option<usize>, wait: WaitBackend) -> Vec<Vec<u64>> {
    let cfg = CountConfig {
        length: 7,
        kill_at: 4,
        ..CountConfig::default()
    };
    let mut p = pool_on(mode, workers, cfg, wait);
    assert_eq!(p.workers(), workers.unwrap_or(0));
    if workers.is_some() {
        // Native by default only on Linux; on request on macOS 14.4+ and
        // Windows too.
        let native_expected = match wait {
            WaitBackend::Park => false,
            WaitBackend::Auto => cfg!(target_os = "linux"),
            WaitBackend::Native => cfg!(any(target_os = "linux", target_os = "macos", windows)),
        };
        assert_eq!(p.uses_native_wait(), native_expected, "{wait:?}");
    }
    p.reset_all(Reseed::Keep).unwrap();
    let mut frames = vec![snapshot_outputs(&p)];
    let n = usize::try_from(N).unwrap();
    for t in 0..20 {
        // Agent 0 sometimes acts illegally (action 2) to exercise the noop.
        let actions: Vec<i32> = (0..n * 2)
            .map(|k| i32::try_from((k * 7 + t * 3) % 3).unwrap())
            .collect();
        #[cfg(feature = "debug-checks")]
        let actions: Vec<i32> = actions
            .iter()
            .enumerate()
            .map(|(k, &a)| if k % 2 == 0 && a == 2 { 1 } else { a })
            .collect();
        p.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        })
        .unwrap();
        frames.push(snapshot_outputs(&p));
    }
    frames
}

#[test]
fn test_sync_matches_inline_bit_for_bit_for_1_2_7_and_64_workers() {
    for mode in [AutoReset::NextStep, AutoReset::SameStep] {
        let inline = trajectory(mode, None);
        for workers in [1, 2, 7, 64] {
            assert_eq!(
                trajectory(mode, Some(workers)),
                inline,
                "{workers} workers, {mode:?}"
            );
        }
    }
}

#[test]
fn test_every_wait_backend_matches_inline() {
    let inline = trajectory(AutoReset::NextStep, None);
    for wait in [WaitBackend::Park, WaitBackend::Native, WaitBackend::Auto] {
        for workers in [1, 3, 8] {
            assert_eq!(
                trajectory_on(AutoReset::NextStep, Some(workers), wait),
                inline,
                "{wait:?}, {workers} workers"
            );
        }
    }
}

#[test]
fn test_worker_panic_poisons_and_full_reset_recovers() {
    let cfg = CountConfig {
        panic_at: 1,
        length: 9,
        ..CountConfig::default()
    };
    let mut p = pool(AutoReset::NextStep, Some(4), cfg);
    p.reset_all(Reseed::Keep).unwrap();
    let actions = vec![0; usize::try_from(N).unwrap() * 2];
    assert_eq!(
        p.step(ActionInput {
            discrete: &actions,
            continuous: &[]
        })
        .map(|_| ()),
        Err(EngineError::Panicked)
    );
    assert!(p.is_poisoned());
    assert_eq!(
        p.step(ActionInput {
            discrete: &actions,
            continuous: &[]
        })
        .map(|_| ()),
        Err(EngineError::Poisoned)
    );
    p.reset_all(Reseed::Keep).unwrap();
    assert!(!p.is_poisoned());
}

#[test]
fn test_dropping_a_sync_pool_joins_its_workers() {
    for _ in 0..20 {
        let p = pool(AutoReset::NextStep, Some(8), CountConfig::default());
        drop(p);
    }
}

#[test]
fn test_cpu_set_needs_one_core_per_worker() {
    let p = pool(AutoReset::NextStep, None, CountConfig::default());
    let options = SyncOptions {
        cpu_set: Some(vec![0]),
        ..SyncOptions::new(NonZeroUsize::new(2).unwrap())
    };
    assert!(matches!(
        p.with_workers(options),
        Err(EngineError::Options {
            option: "cpu_set",
            ..
        })
    ));
}

#[test]
fn test_workers_run_in_the_canonical_fp_environment() {
    let mut p = pool(AutoReset::NextStep, Some(2), CountConfig::default());
    p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(p.fp_resets(), 0);
}

#[test]
fn test_cpu_set_rejects_core_ids_the_os_does_not_report() {
    let p = pool(AutoReset::NextStep, None, CountConfig::default());
    let options = SyncOptions {
        cpu_set: Some(vec![usize::MAX]),
        ..SyncOptions::new(NonZeroUsize::new(1).unwrap())
    };
    assert!(matches!(
        p.with_workers(options),
        Err(EngineError::Options {
            option: "cpu_set",
            ..
        })
    ));
}

/// Runs `f` on a helper thread and fails if it does not finish in time.
fn within_timeout(seconds: u64, f: impl FnOnce() + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    assert!(
        rx.recv_timeout(std::time::Duration::from_secs(seconds))
            .is_ok(),
        "timed out: the pool hung"
    );
}

/// codex's regression: a kernel panic whose payload panics on drop (after
/// 300 ms) must neither escape nor kill a worker; the pool is poisoned and
/// `reset_all` recovers, in Sync and in Inline mode.
#[test]
fn test_panic_payload_whose_drop_panics_does_not_hang_or_escape() {
    for workers in [Some(2), None] {
        within_timeout(30, move || {
            let cfg = CountConfig {
                bomb_at: 2,
                length: 50,
                ..CountConfig::default()
            };
            let mut p = pool(AutoReset::NextStep, workers, cfg);
            p.reset_all(Reseed::Keep).unwrap();
            let actions = vec![0; usize::try_from(N).unwrap() * 2];
            let input = || ActionInput {
                discrete: &actions,
                continuous: &[],
            };
            assert!(p.step(input()).is_ok());
            assert_eq!(p.step(input()).map(|_| ()), Err(EngineError::Panicked));
            assert!(p.is_poisoned());
            p.reset_all(Reseed::Keep).unwrap();
            assert!(!p.is_poisoned());
            assert!(p.step(input()).is_ok());
        });
    }
}
