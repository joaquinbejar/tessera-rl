// SPDX-License-Identifier: MIT OR Apache-2.0

//! A refused persisted file allocates nothing before its bounds are checked
//! (ADR-0011). One test: the counter is global.

#![expect(clippy::indexing_slicing, reason = "tests: a failure is the assertion")]

mod common;

use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;
use core::num::NonZeroU32;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::System;

use common::{CountConfig, Counter};
use tessera_rl_core::encoding::checksum;
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, InvariantViolation, NoAdaptive, Observation, Persist, RecordHeader,
    SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_engine::{DecodeLimits, EnvPool, Masked, NoMasks, PoolOptions, Reseed};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};

struct Counting;
static BIGGEST: AtomicUsize = AtomicUsize::new(0);

std::thread_local! {
    /// Only the test's own thread counts, not the harness's.
    static COUNT_HERE: Cell<bool> = const { Cell::new(false) };
}

// SAFETY: forwards to `System`; only records the size.
#[expect(unsafe_code, reason = "a counting allocator for this test")]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNT_HERE.with(Cell::get) {
            BIGGEST.fetch_max(l.size(), Ordering::Relaxed);
        }
        // SAFETY: the caller's layout.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: the caller's pointer and layout.
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct SameState {
    v: f64,
}

#[derive(EnvConfig, Debug, Clone)]
struct SameConfig {
    #[config(default = 1)]
    size: u32,
}

#[derive(Observation)]
struct SameObs {
    #[space(box(low = 0, high = 1, shape = [1]))]
    value: tessera_rl_core::ObsLeaf<i32>,
}

struct Same;

impl Environment for Same {
    type State = SameState;
    type Config = SameConfig;
    type Obs = SameObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;
    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::SamePlatform;
    fn observation_space(cfg: &SameConfig) -> Result<Space, SpaceError> {
        SameObs::space(cfg)
    }
    fn action_space(_: &SameConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }
    fn reset(_: &SameConfig, _: &mut StepRng, _: &mut SameState) {}
    fn step(
        _: &SameConfig,
        _: &mut SameState,
        _: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
    }
    fn observe(
        _: &SameConfig,
        _: &SameState,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, SameObs>,
    ) {
    }
}

impl Validate for Same {
    fn validate(_: &SameConfig, _: &SameState) -> Result<(), InvariantViolation> {
        Ok(())
    }
}

impl Persist for Same {
    const ENV_ID: &'static str = "tessera/Same-v1";
    const ENV_VERSION: u32 = 1;
}

/// Decodes `good` with each refused count; the test's thread must not allocate.
fn refused_counts(good: &[u8], decode: &dyn Fn(&[u8], DecodeLimits) -> bool) {
    for (max_instances, count) in [(1 << 24, 5u32), (u32::MAX, u32::MAX)] {
        let mut bad = good.to_vec();
        bad[40..44].copy_from_slice(&count.to_le_bytes());
        let n = bad.len() - 8;
        let sum = checksum(&bad[..n]);
        bad[n..].copy_from_slice(&sum.to_le_bytes());
        BIGGEST.store(0, Ordering::Relaxed);
        let limits = DecodeLimits {
            max_instances,
            max_decoded_bytes: u64::MAX,
        };
        COUNT_HERE.with(|c| c.set(true));
        let refused = decode(&bad, limits);
        COUNT_HERE.with(|c| c.set(false));
        assert!(refused);
        assert_eq!(BIGGEST.load(Ordering::Relaxed), 0, "count {count}");
    }
}

#[test]
fn test_persist_refused_counts_never_reach_the_allocator_on_either_tier() {
    let options = PoolOptions::new(NonZeroU32::new(4).unwrap(), Seed::new(1));
    let mut p = EnvPool::<Counter, Masked>::with_masks(CountConfig::default(), options).unwrap();
    p.reset_all(Reseed::Keep).unwrap();
    let good = p.encode_with(&[0], "fixture", None).unwrap();
    refused_counts(&good, &|b, l| p.decode_with(b, l, None).is_err());

    let mut q = EnvPool::<Same, NoMasks>::new(SameConfig::default(), options).unwrap();
    q.reset_all(Reseed::Keep).unwrap();
    let good = q.encode_with(&[0], "fixture", Some("build")).unwrap();
    refused_counts(&good, &|b, l| q.decode_with(b, l, Some("build")).is_err());
}
