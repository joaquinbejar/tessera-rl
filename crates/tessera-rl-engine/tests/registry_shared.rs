// SPDX-License-Identifier: MIT OR Apache-2.0

//! The shared active-engine registry (ADR-0005, #349): once a counter is
//! attached, engines count in it; a registration made before the attach
//! keeps decrementing its own counter; a second, different counter is
//! refused. One test in its own binary: the registry is process-wide.

#![cfg(not(loom))]
#![expect(clippy::unwrap_used, reason = "a failure is the assertion")]

mod common;

use core::num::{NonZeroU32, NonZeroUsize};
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{CountConfig, Counter};
use tessera_rl_engine::{EnvPool, PoolOptions, SyncOptions, attach_shared_registry};
use tessera_rl_num::identity::Seed;

fn env_pool() -> EnvPool<Counter> {
    EnvPool::new(
        CountConfig::default(),
        PoolOptions::new(NonZeroU32::new(4).unwrap(), Seed::new(0)),
    )
    .unwrap()
    .with_workers(SyncOptions::new(NonZeroUsize::new(1).unwrap()))
    .unwrap()
}

#[test]
fn test_registry_shared_counter_counts_engines_and_keeps_earlier_registrations_apart() {
    let shared: &'static AtomicUsize = Box::leak(Box::new(AtomicUsize::new(0)));
    // Registered on this binary's own counter, before the attach.
    let before = env_pool();
    assert!(attach_shared_registry(shared));
    assert!(attach_shared_registry(shared), "the same counter again");
    let other: &'static AtomicUsize = Box::leak(Box::new(AtomicUsize::new(0)));
    assert!(!attach_shared_registry(other), "a different counter is refused");
    assert_eq!(shared.load(Ordering::Acquire), 0);
    // Another wheel's engine, counted in the shared counter only.
    shared.fetch_add(1, Ordering::AcqRel);
    let after = env_pool();
    assert_eq!(shared.load(Ordering::Acquire), 2);
    // It saw the other wheel's engine: unpinned.
    assert!(!after.pinned_workers());
    drop(before);
    assert_eq!(shared.load(Ordering::Acquire), 2, "the earlier engine left its own counter");
    drop(after);
    assert_eq!(shared.load(Ordering::Acquire), 1);
    assert_eq!(other.load(Ordering::Acquire), 0);
    shared.fetch_sub(1, Ordering::AcqRel);
}
