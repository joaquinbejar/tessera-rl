// SPDX-License-Identifier: MIT OR Apache-2.0

//! An environment that reads its adaptive values never takes the lane path
//! (ADR-0091 decision 4, amending ADR-0071): a lane step receives no
//! adaptive values, so such a pool steps every instance through
//! `Environment::step_with`. `AdaptiveWalk` is `Walk` with
//! `READS_ADAPTIVE = true` and the same lane step; its pool runs no lane
//! step, and its records equal `Plain`'s (no lane step) after every step.
//! Alone in its binary, so that the process-wide lane-step counter sees no
//! other test.

#![cfg(not(loom))]
#![expect(
    clippy::unwrap_used,
    reason = "test helpers unwrap: a failure is the assertion"
)]

mod lanes;

use core::num::NonZeroU32;
use core::sync::atomic::Ordering;

use lanes::{LANE_CALLS, Plain, Walk, WalkConfig, WalkObs, WalkState};
use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    Adaptive, AgentSlot, Environment, LANES, LaneHook, LaneOut, LaneStep, NoAdaptive, RecordHeader,
    SpaceError, StepOut, Tier, lane_hook,
};
use tessera_rl_engine::{ActionInput, AutoReset, EnvPool, PoolOptions, Reseed};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};
use tessera_rl_simd::Resolved;

/// `Walk`, declaring that it reads its adaptive values.
struct AdaptiveWalk;

impl Environment for AdaptiveWalk {
    type State = WalkState;
    type Config = WalkConfig;
    type Obs = WalkObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;
    const READS_ADAPTIVE: bool = true;
    const LANE_STEP: Option<LaneHook<Self>> = Some(lane_hook::<Self>());

    fn observation_space(cfg: &WalkConfig) -> Result<Space, SpaceError> {
        Walk::observation_space(cfg)
    }

    fn action_space(cfg: &WalkConfig) -> Result<Space, SpaceError> {
        Walk::action_space(cfg)
    }

    fn reset(cfg: &WalkConfig, rng: &mut StepRng, state: &mut WalkState) {
        Walk::reset(cfg, rng, state);
    }

    fn reset_with(cfg: &WalkConfig, _: &Adaptive<Self>, rng: &mut StepRng, state: &mut WalkState) {
        Walk::reset(cfg, rng, state);
    }

    fn step(
        cfg: &WalkConfig,
        state: &mut WalkState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        Walk::step(cfg, state, actions, rng, out);
    }

    fn step_with(
        cfg: &WalkConfig,
        _: &Adaptive<Self>,
        state: &mut WalkState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        Walk::step(cfg, state, actions, rng, out);
    }

    fn observe(
        cfg: &WalkConfig,
        state: &WalkState,
        agent: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, WalkObs>,
    ) {
        Walk::observe(cfg, state, agent, rng, out);
    }
}

impl LaneStep for AdaptiveWalk {
    const STEP_DRAWS_RNG: bool = false;

    fn lane_step(
        cfg: &WalkConfig,
        level: Resolved,
        states: &mut [WalkState; LANES],
        actions: &[i32; LANES],
        out: &mut LaneOut,
    ) {
        Walk::lane_step(cfg, level, states, actions, out);
    }
}

const N: u32 = 64;

fn options() -> PoolOptions {
    PoolOptions {
        chunk_size: NonZeroU32::new(16).unwrap(),
        max_episode_steps: NonZeroU32::new(9),
        auto_reset: AutoReset::SameStep,
        ..PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(11))
    }
}

#[test]
fn test_lane_path_reads_adaptive_env_steps_scalar_and_matches() {
    let mut adaptive = EnvPool::<AdaptiveWalk>::new(WalkConfig {}, options()).unwrap();
    let mut plain = EnvPool::<Plain>::new(WalkConfig {}, options()).unwrap();
    adaptive.reset_all(Reseed::Keep).unwrap();
    plain.reset_all(Reseed::Keep).unwrap();
    let before = LANE_CALLS.load(Ordering::Relaxed);
    for t in 0..300u32 {
        let acts: Vec<i32> = (0..N)
            .map(|i| i32::try_from((i * 7 + t * 5 + (t >> 2)) % 3).unwrap())
            .collect();
        let input = ActionInput {
            discrete: &acts,
            continuous: &[],
        };
        adaptive.step(input).unwrap();
        plain.step(input).unwrap();
        let a: Vec<Vec<u8>> = (0..N)
            .map(|i| adaptive.snapshot(i).unwrap().to_bytes())
            .collect();
        let b: Vec<Vec<u8>> = (0..N)
            .map(|i| plain.snapshot(i).unwrap().to_bytes())
            .collect();
        assert_eq!(a, b, "records at step {t}");
    }
    assert_eq!(
        LANE_CALLS.load(Ordering::Relaxed),
        before,
        "an adaptive-reading environment took the lane path"
    );
}
