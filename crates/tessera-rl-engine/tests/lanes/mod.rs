// SPDX-License-Identifier: MIT OR Apache-2.0

//! A test-only lane environment for the engine's lane path (ADR-0071).
//!
//! `Walk` is a one-agent walk on a line: the discrete action (0, 1, 2)
//! changes the velocity by -0.25, 0 or +0.25, the position moves by half the
//! velocity, the reward is the position as `f32`, and the episode terminates
//! when the position leaves [-4, 4]. Its step draws no RNG; its reset does.
//! `Walk` implements `LaneStep` with lane types at the pool's level; `Plain`
//! is the same environment without a lane step, the scalar reference. Both
//! have the same state and header, so their records compare byte for byte.

#![allow(dead_code, reason = "each test binary uses part of the module")]

use core::sync::atomic::{AtomicU64, Ordering};

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, InvariantViolation, LANES, LaneHook, LaneOut, LaneStep,
    LeafId, NoAdaptive, Observation, Persist, RecordHeader, SpaceError, StepOut, Tier, Validate,
    lane_hook,
};
use tessera_rl_macros::{EnvConfig, EnvState};
use tessera_rl_num::math::lane::F64Lanes;
use tessera_rl_num::rng::{ObsRng, StepRng};
use tessera_rl_simd::{Kernel, Resolved};

/// Lane steps run by `Walk` in this process (for the Async test).
pub static LANE_CALLS: AtomicU64 = AtomicU64::new(0);

const BOUND: f64 = 4.0;

#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct WalkConfig {}

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct WalkState {
    pub pos: f64,
    pub vel: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WalkObs;

impl WalkObs {
    pub const STATE: LeafId<f32> = LeafId::new(0);
}

impl Observation<WalkConfig> for WalkObs {
    const LEAVES: usize = 1;

    fn space(_: &WalkConfig) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![2],
            dtype: DType::F32,
            bounds: BoxBounds::Float {
                low: vec![f64::NEG_INFINITY; 2],
                high: vec![f64::INFINITY; 2],
            },
        })
    }
}

/// The scalar transition; the reference the lanes must match.
fn walk(state: &mut WalkState, action: i32) -> (f32, bool) {
    state.vel += f64::from(action - 1) * 0.25;
    state.pos += state.vel * 0.5;
    #[expect(clippy::cast_possible_truncation, reason = "the reward is f32")]
    let reward = state.pos as f32;
    (reward, state.pos.abs() > BOUND)
}

macro_rules! walk_env {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        pub struct $name;

        impl Validate for $name {
            fn validate(_: &WalkConfig, s: &WalkState) -> Result<(), InvariantViolation> {
                if s.pos.is_finite() && s.vel.is_finite() {
                    Ok(())
                } else {
                    Err(InvariantViolation { code: "non-finite" })
                }
            }
        }

        impl Persist for $name {
            const ENV_ID: &'static str = "test/Walk-v1";
            const ENV_VERSION: u32 = 1;
        }
    };
}

walk_env!(Walk);
walk_env!(Plain);

macro_rules! walk_kernels {
    () => {
        type State = WalkState;
        type Config = WalkConfig;
        type Obs = WalkObs;
        type Action = ();
        type Header = RecordHeader<1, NoAdaptive>;

        const MAX_AGENTS: usize = 1;
        const TIER: Tier = Tier::CrossPlatform;

        fn observation_space(cfg: &WalkConfig) -> Result<Space, SpaceError> {
            WalkObs::space(cfg)
        }

        fn action_space(_: &WalkConfig) -> Result<Space, SpaceError> {
            Ok(Space::Discrete { n: 3 })
        }

        fn reset(_: &WalkConfig, rng: &mut StepRng, state: &mut WalkState) {
            state.pos = f64::from(rng.next_u32() % 13) * 0.5 - 3.0;
            state.vel = f64::from(rng.next_u32() % 5) * 0.125 - 0.25;
        }

        fn step(
            _: &WalkConfig,
            state: &mut WalkState,
            actions: &Actions<'_, ()>,
            _: &mut StepRng,
            out: &mut StepOut<'_>,
        ) {
            let action = actions.discrete(AgentSlot::FIRST, 0).unwrap_or(1);
            let (reward, end) = walk(state, action);
            out.set_reward(AgentSlot::FIRST, reward);
            if end {
                out.end_episode(EpisodeEnd::Terminated);
            }
        }

        #[expect(clippy::cast_possible_truncation, reason = "f32 observations")]
        fn observe(
            _: &WalkConfig,
            state: &WalkState,
            _: AgentSlot,
            _: &mut ObsRng,
            out: &mut ObsWriter<'_, '_, WalkObs>,
        ) {
            if let Some(mut leaf) = out.leaf_mut(WalkObs::STATE) {
                let _ = leaf.copy_from(&[state.pos as f32, state.vel as f32]);
            }
        }
    };
}

impl Environment for Walk {
    walk_kernels!();
    const LANE_STEP: Option<LaneHook<Self>> = Some(lane_hook::<Self>());
}

impl Environment for Plain {
    walk_kernels!();
}

struct WalkLanes<'a> {
    states: &'a mut [WalkState; LANES],
    actions: &'a [i32; LANES],
    out: &'a mut LaneOut,
}

impl Kernel for WalkLanes<'_> {
    type Output = ();

    #[expect(
        clippy::inline_always,
        reason = "must inline into each target_feature kernel"
    )]
    #[inline(always)]
    fn run<L: F64Lanes>(self) {
        let push = L::from_array(self.actions.map(|a| f64::from(a - 1)));
        let vel = L::from_array(self.states.map(|s| s.vel)) + push * L::splat(0.25);
        let pos = L::from_array(self.states.map(|s| s.pos)) + vel * L::splat(0.5);
        let ended = L::mask_bits(L::gt(pos.abs(), L::splat(BOUND)));
        let (pos_a, vel_a) = (pos.to_array(), vel.to_array());
        self.out.reward = pos.to_f32_array();
        for (l, s) in self.states.iter_mut().enumerate() {
            s.pos = pos_a.get(l).copied().unwrap_or(0.0);
            s.vel = vel_a.get(l).copied().unwrap_or(0.0);
            if let Some(end) = self.out.end.get_mut(l) {
                *end = (ended >> l & 1 == 1).then_some(EpisodeEnd::Terminated);
            }
        }
    }
}

impl LaneStep for Walk {
    const STEP_DRAWS_RNG: bool = false;

    fn lane_step(
        _: &WalkConfig,
        level: Resolved,
        states: &mut [WalkState; LANES],
        actions: &[i32; LANES],
        out: &mut LaneOut,
    ) {
        LANE_CALLS.fetch_add(1, Ordering::Relaxed);
        level.dispatch(WalkLanes {
            states,
            actions,
            out,
        });
    }
}
