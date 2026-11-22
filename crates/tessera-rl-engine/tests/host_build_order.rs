// SPDX-License-Identifier: MIT OR Apache-2.0

//! How `HostPool::new` builds its instances (issue #285, the
//! `HostEnvironment::new` doc): once each, in index order, on the thread
//! that builds the pool, before any method on an instance; the `k`-th call
//! of a build builds instance `k`; a failure stops the build and drops the
//! instances already built.

#![cfg(feature = "host")]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod host_script;

use core::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;

use host_script::{SLOTS, Script, ScriptConfig, Shared};
use tessera_rl_engine::{EngineError, HostPool, PoolOptions, Reseed, SyncOptions};
use tessera_rl_num::identity::Seed;

const N: u32 = 6;

type Pool = HostPool<Script<0, false>, SLOTS>;

fn build(shared: &Arc<Shared>) -> Result<Pool, EngineError> {
    let o = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(3));
    Pool::new(&ScriptConfig::new(shared), o)
}

fn calls(shared: &Shared, i: usize) -> u64 {
    [
        &shared.resets,
        &shared.steps,
        &shared.observes,
        &shared.masks,
        &shared.actings,
        &shared.validates,
    ]
    .iter()
    .map(|c| Shared::get(c, i))
    .sum()
}

#[test]
fn test_host_build_order_new_once_each_on_the_caller_thread_before_any_instance_method() {
    let shared = Shared::new();
    let pool = build(&shared).unwrap();
    assert_eq!(shared.built(), N, "one `new` per instance");
    let threads = shared.built_on.lock().unwrap().clone();
    assert_eq!(threads.len(), usize::try_from(N).unwrap());
    assert!(
        threads.iter().all(|t| *t == std::thread::current().id()),
        "every `new` runs on the thread that builds the pool"
    );
    for i in 0..usize::try_from(N).unwrap() {
        assert_eq!(
            calls(&shared, i),
            0,
            "instance {i}: no call before the build ends"
        );
    }
    drop(pool);
}

#[test]
fn test_host_build_order_kth_call_builds_instance_k_inline_and_sync() {
    for workers in [0, 2] {
        let shared = Shared::new();
        let mut pool = build(&shared).unwrap();
        if workers > 0 {
            pool = pool
                .with_workers(SyncOptions::new(NonZeroUsize::new(workers).unwrap()))
                .unwrap();
        }
        // The script counts each call under the number of the `new` call
        // that built the instance: resetting instance `k` alone must count
        // under `k`.
        for k in 0..N {
            pool.reset(&[k], Reseed::Keep).unwrap();
            for i in 0..N {
                let want = u64::from(i <= k);
                let got = Shared::get(&shared.resets, usize::try_from(i).unwrap());
                assert_eq!(got, want, "workers {workers}: reset of {k}, instance {i}");
            }
        }
    }
}

#[test]
fn test_host_build_order_failed_new_stops_the_build() {
    let shared = Shared::new();
    // The third call fails: instances 0 and 1 were built, nothing after.
    shared
        .fail_new
        .store(3, core::sync::atomic::Ordering::Relaxed);
    let err = build(&shared).err().unwrap();
    assert!(
        matches!(err, EngineError::HostConstruction { .. }),
        "{err:?}"
    );
    assert_eq!(shared.built(), 3, "no `new` after the failed one");
    assert_eq!(
        shared.drops.load(core::sync::atomic::Ordering::Relaxed),
        2,
        "the two instances already built are dropped"
    );
}
