// SPDX-License-Identifier: MIT OR Apache-2.0

//! A refused persisted file allocates nothing before its bounds are checked
//! (ADR-0011). The counting allocator of `support/counting.rs` attributes
//! allocator calls by thread (issue #43); this binary holds one test.

#![expect(clippy::indexing_slicing, reason = "tests: a failure is the assertion")]

mod common;

#[path = "support/counting.rs"]
mod counting;

use core::num::NonZeroU32;

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

/// Decodes `good` with each refused count; no counted thread may allocate.
fn refused_counts(good: &[u8], decode: &dyn Fn(&[u8], DecodeLimits) -> bool) {
    for (max_instances, count) in [(1 << 24, 5u32), (u32::MAX, u32::MAX)] {
        let mut bad = good.to_vec();
        bad[40..44].copy_from_slice(&count.to_le_bytes());
        let n = bad.len() - 8;
        let sum = checksum(&bad[..n]);
        bad[n..].copy_from_slice(&sum.to_le_bytes());
        let limits = DecodeLimits {
            max_instances,
            max_decoded_bytes: u64::MAX,
        };
        let (refused, tally) = counting::window(|| decode(&bad, limits));
        assert!(refused);
        assert_eq!(tally.calls, 0, "count {count}: {tally:?}");
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
