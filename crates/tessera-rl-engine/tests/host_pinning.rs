// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host pool pinning (ADR-0062 item 2, ADR-0005): a host pool pins only to
//! an explicit `cpu_set`, and only then registers as an active engine. An
//! unpinned host pool does not register, so an `EnvPool` started after it
//! pins as it would alone. One test in its own binary: the active-engine
//! registry is process-wide.

#![cfg(all(not(loom), feature = "host"))]
#![expect(clippy::unwrap_used, reason = "a failure is the assertion")]

mod common;
mod host_support;

use core::num::{NonZeroU32, NonZeroUsize};

use common::{CountConfig, Counter};
use host_support::{Book, BookConfig};
use tessera_rl_core::HostEnvironment;
use tessera_rl_engine::{EngineError, EnvPool, HostPool, PoolOptions, SyncOptions};
use tessera_rl_num::identity::Seed;

fn options() -> PoolOptions {
    PoolOptions::new(NonZeroU32::new(4).unwrap(), Seed::new(0))
}

fn two() -> SyncOptions {
    SyncOptions::new(NonZeroUsize::new(2).unwrap())
}

fn env_pool() -> EnvPool<Counter> {
    EnvPool::new(CountConfig::default(), options())
        .unwrap()
        .with_workers(two())
        .unwrap()
}

type Host = HostPool<Book, { Book::MAX_AGENTS }>;

fn host(sync: SyncOptions) -> Result<Host, EngineError> {
    Host::new(BookConfig::default(), options())
        .unwrap()
        .with_workers(sync)
}

#[test]
fn test_host_pinning_only_explicit_cpu_set_pins_and_registers() {
    // What an EnvPool does alone: it pins where the OS reports cores.
    let alone = env_pool().pinned_workers();
    if cfg!(target_os = "linux") {
        assert!(alone, "Linux reports cores: a lone EnvPool pins");
    }
    // An unpinned host pool neither pins nor registers.
    let unpinned = host(two()).unwrap();
    assert!(!unpinned.pinned_workers());
    assert_eq!(
        env_pool().pinned_workers(),
        alone,
        "the host pool registered"
    );
    drop(unpinned);
    // An explicit cpu_set pins the host pool and registers it: an EnvPool
    // started after it runs unpinned, as beside any other engine.
    let explicit = SyncOptions {
        cpu_set: Some(vec![0, 1]),
        ..two()
    };
    match host(explicit) {
        Ok(pinned) => {
            assert!(pinned.pinned_workers());
            assert!(!env_pool().pinned_workers(), "a pinned host pool registers");
        }
        // No core ids reported (macOS): nothing can pin here.
        Err(EngineError::Options {
            option: "cpu_set", ..
        }) => {
            assert_ne!(std::env::consts::OS, "linux", "Linux reports core ids");
            assert!(!alone);
        }
        Err(e) => panic!("{e}"),
    }
}
