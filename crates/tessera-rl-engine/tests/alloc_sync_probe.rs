// SPDX-License-Identifier: MIT OR Apache-2.0

//! Positive probes of the allocation diagnostics in `common/alloc_diag.rs`
//! (the assertion of `alloc_sync.rs` relies on them): known allocations in
//! the step kernel on several workers are counted and attributed, a count
//! past the table is reported as truncated, and allocations under the
//! reentrancy guard are not counted. One test, because the counter is
//! process-wide (it first checks `Tagged` against `Counter`, before any
//! counted window).

mod common;

#[path = "common/alloc_diag.rs"]
mod alloc_diag;

use core::num::{NonZeroU32, NonZeroUsize};
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

use alloc_diag::{ALLOCATE_IN_STEP, EVENTS, RECORDING, Tagged};
use common::CountConfig;
use tessera_rl_engine::{
    ActionInput, AutoReset, EnvPool, Masked, PoolOptions, Reseed, SyncOptions,
};
use tessera_rl_num::identity::Seed;

/// `Tagged` stays in step with `Counter` (TR, #43): the same spaces,
/// initial agents and constants, for the default config and for a
/// multi-head one. Called before any counted window.
fn tagged_matches_counter() {
    use tessera_rl_core::Environment;
    let multi = CountConfig {
        action_heads: 2,
        ..CountConfig::default()
    };
    for cfg in [CountConfig::default(), multi] {
        assert_eq!(
            Tagged::observation_space(&cfg),
            common::Counter::observation_space(&cfg)
        );
        assert_eq!(
            Tagged::action_space(&cfg),
            common::Counter::action_space(&cfg)
        );
        assert_eq!(
            Tagged::initial_agents(&cfg),
            common::Counter::initial_agents(&cfg)
        );
    }
    assert_eq!(Tagged::MAX_AGENTS, common::Counter::MAX_AGENTS);
    assert_eq!(Tagged::TIER, common::Counter::TIER);
    assert_eq!(Tagged::TURNS, common::Counter::TURNS);
    assert_eq!(
        Tagged::PERFECT_INFORMATION,
        common::Counter::PERFECT_INFORMATION
    );
}

#[test]
fn test_alloc_diagnostics_count_attribute_truncate_and_guard() {
    tagged_matches_counter();
    let options = PoolOptions {
        auto_reset: AutoReset::SameStep,
        chunk_size: NonZeroU32::new(1).unwrap(),
        ..PoolOptions::new(NonZeroU32::new(64).unwrap(), Seed::new(1))
    };
    let cfg = CountConfig {
        length: 4,
        ..CountConfig::default()
    };
    let mut pool = EnvPool::<Tagged, Masked>::with_masks(cfg, options)
        .unwrap()
        .with_workers(SyncOptions::new(NonZeroUsize::new(4).unwrap()))
        .unwrap();
    let actions = vec![1i32; 128];
    pool.reset_all(Reseed::Keep).unwrap();
    let step = |pool: &mut EnvPool<Tagged, Masked>| {
        pool.step(ActionInput {
            discrete: &actions,
            continuous: &[],
        })
        .unwrap();
    };
    step(&mut pool);

    // Known allocations in the step kernel, one per instance: all counted,
    // the table holds the first EVENTS, each attributed to the step kernel.
    ALLOCATE_IN_STEP.store(true, Ordering::Relaxed);
    alloc_diag::start();
    step(&mut pool);
    let n = alloc_diag::stop();
    ALLOCATE_IN_STEP.store(false, Ordering::Relaxed);
    let (events, dropped) = alloc_diag::recorded();
    assert_eq!(n, 64, "one allocation per instance");
    assert_eq!(events.len(), EVENTS, "the table is full");
    assert_eq!(dropped, 64 - EVENTS as u64, "the rest are truncated");
    assert!(
        events
            .iter()
            .all(|e| e.size == 3 && e.kernel == 1 && e.kernel_calls > 0)
    );
    let report = alloc_diag::report();
    assert!(report.contains("in step"), "{report}");
    assert!(
        report.contains(&format!("and {} more", 64 - EVENTS)),
        "{report}"
    );

    // Several threads take part (the caller and the workers): attributed by
    // tag and name. Run steps until more than one thread has recorded.
    ALLOCATE_IN_STEP.store(true, Ordering::Relaxed);
    let mut threads = BTreeSet::new();
    for _ in 0..50 {
        alloc_diag::start();
        step(&mut pool);
        alloc_diag::stop();
        for e in alloc_diag::recorded().0 {
            threads.insert((e.tag, e.thread));
        }
        if threads.len() > 1 {
            break;
        }
    }
    ALLOCATE_IN_STEP.store(false, Ordering::Relaxed);
    assert!(
        threads.len() > 1,
        "attributed to one thread only: {threads:?}"
    );
    assert!(
        threads
            .iter()
            .any(|(_, name)| name.starts_with("tessera-worker-")),
        "no worker named: {threads:?}"
    );

    // No allocation without the probe: the window is clean.
    alloc_diag::start();
    step(&mut pool);
    assert_eq!(alloc_diag::stop(), 0, "{}", alloc_diag::report());

    // The reentrancy guard: an allocation while this thread records is not
    // counted (it is the diagnostic's own), and one after it is.
    alloc_diag::start();
    RECORDING.with(|r| r.set(true));
    drop(std::hint::black_box(vec![0u8; 5]));
    RECORDING.with(|r| r.set(false));
    let guarded = alloc_diag::stop();
    assert_eq!(guarded, 0, "guarded allocations are not counted");
    alloc_diag::start();
    drop(std::hint::black_box(vec![0u8; 5]));
    let after = alloc_diag::stop();
    assert_eq!(after, 1);
    let (events, _) = alloc_diag::recorded();
    assert_eq!(events.first().map(|e| (e.size, e.kernel)), Some((5, 0)));
}
