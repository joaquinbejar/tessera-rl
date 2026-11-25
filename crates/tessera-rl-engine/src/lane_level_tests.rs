// SPDX-License-Identifier: MIT OR Apache-2.0

//! The lane path's shape conditions (ADR-0071 decision 3): a pool takes
//! the lane path only for an environment with a lane step, one agent,
//! simultaneous turns, no masks, one discrete head and no continuous head.
//! Each test changes one condition of an otherwise eligible environment and
//! checks that the pool has no lane level; the first one is the control.

use core::num::NonZeroU32;

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    ActionMask, AgentSlot, LANES, LaneOut, NoAdaptive, Observation, RecordHeader, SpaceError,
    StepOut, Tier, TurnStructure,
};
use tessera_rl_macros::{EnvConfig, EnvState};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};
use tessera_rl_simd::Resolved;

use crate::{EnvPool, Masked, PoolOptions};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
struct Unit {
    v: u32,
}

#[derive(EnvConfig, Debug, Clone)]
struct UnitConfig {}

struct UnitObs;

impl Observation<UnitConfig> for UnitObs {
    const LEAVES: usize = 1;

    fn space(_: &UnitConfig) -> Result<Space, SpaceError> {
        Ok(Space::Box {
            shape: vec![1],
            dtype: DType::F32,
            bounds: BoxBounds::Float {
                low: vec![0.0],
                high: vec![1.0],
            },
        })
    }
}

fn no_lanes(_: &UnitConfig, _: Resolved, _: &mut [Unit; LANES], _: &[i32; LANES], _: &mut LaneOut) {
}

/// What the probe's action space is.
const DISCRETE: u8 = 0;
const TWO_HEADS: u8 = 1;
const WITH_CONTINUOUS: u8 = 2;

/// An environment with a lane step and the given shape.
struct Probe<const A: usize, const SEQ: bool, const SPACE: u8, const LANE: bool>;

impl<const A: usize, const SEQ: bool, const SPACE: u8, const LANE: bool>
    tessera_rl_core::Environment for Probe<A, SEQ, SPACE, LANE>
{
    type State = Unit;
    type Config = UnitConfig;
    type Obs = UnitObs;
    type Action = ();
    type Header = RecordHeader<A, NoAdaptive>;

    const MAX_AGENTS: usize = A;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = if SEQ {
        TurnStructure::Sequential
    } else {
        TurnStructure::Simultaneous
    };
    const LANE_STEP: Option<tessera_rl_core::LaneHook<Self>> =
        if LANE { Some(no_lanes) } else { None };

    fn observation_space(cfg: &UnitConfig) -> Result<Space, SpaceError> {
        UnitObs::space(cfg)
    }

    fn action_space(_: &UnitConfig) -> Result<Space, SpaceError> {
        Ok(match SPACE {
            TWO_HEADS => Space::MultiDiscrete { nvec: vec![2, 2] },
            WITH_CONTINUOUS => Space::Tuple(vec![
                Space::Discrete { n: 2 },
                Space::Box {
                    shape: vec![1],
                    dtype: DType::F32,
                    bounds: BoxBounds::Float {
                        low: vec![0.0],
                        high: vec![1.0],
                    },
                },
            ]),
            _ => Space::Discrete { n: 2 },
        })
    }

    fn acting(_: &UnitConfig, _: &Unit, agent: AgentSlot) -> bool {
        agent == AgentSlot::FIRST
    }

    fn reset(_: &UnitConfig, _: &mut StepRng, _: &mut Unit) {}

    fn step(
        _: &UnitConfig,
        _: &mut Unit,
        _: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
    }

    fn observe(
        _: &UnitConfig,
        _: &Unit,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, UnitObs>,
    ) {
    }
}

impl<const A: usize, const SEQ: bool, const SPACE: u8, const LANE: bool> ActionMask
    for Probe<A, SEQ, SPACE, LANE>
{
    fn action_mask(_: &UnitConfig, _: &Unit, _: AgentSlot, _: &mut MaskWriter<'_>) {}

    fn noop(_: &UnitConfig, _: &Unit, _: AgentSlot, _: &mut ActionWriter<'_>) {}
}

fn options() -> PoolOptions {
    PoolOptions::new(NonZeroU32::MIN.saturating_add(7), Seed::new(1))
}

fn lanes<const A: usize, const SEQ: bool, const SPACE: u8, const LANE: bool>() -> bool {
    EnvPool::<Probe<A, SEQ, SPACE, LANE>>::new(UnitConfig {}, options()).map_or_else(
        |e| panic!("construction failed: {e}"),
        |p| p.lanes.is_some(),
    )
}

#[test]
fn test_lane_level_eligible_shape_has_a_level() {
    assert!(lanes::<1, false, DISCRETE, true>());
}

#[test]
fn test_lane_level_without_lane_step_has_none() {
    assert!(!lanes::<1, false, DISCRETE, false>());
}

#[test]
fn test_lane_level_two_agents_has_none() {
    assert!(!lanes::<2, false, DISCRETE, true>());
}

#[test]
fn test_lane_level_sequential_turns_has_none() {
    assert!(!lanes::<1, true, DISCRETE, true>());
}

#[test]
fn test_lane_level_two_discrete_heads_has_none() {
    assert!(!lanes::<1, false, TWO_HEADS, true>());
}

#[test]
fn test_lane_level_continuous_head_has_none() {
    assert!(!lanes::<1, false, WITH_CONTINUOUS, true>());
}

#[test]
fn test_lane_level_masked_pool_has_none() {
    let pool =
        EnvPool::<Probe<1, false, DISCRETE, true>, Masked>::with_masks(UnitConfig {}, options())
            .unwrap_or_else(|e| panic!("construction failed: {e}"));
    assert!(pool.lanes.is_none());
}

/// A lane-wise state (ADR-0090): it keeps no rows, so the lane step, which
/// gathers rows, is never used for it.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
#[state(lanes)]
struct UnitLanes {
    v: u32,
}

fn no_lanes_soa(
    _: &UnitConfig,
    _: Resolved,
    _: &mut [UnitLanes; LANES],
    _: &[i32; LANES],
    _: &mut LaneOut,
) {
}

/// The eligible shape with a lane step, over a lane-wise state.
struct SoaProbe;

impl tessera_rl_core::Environment for SoaProbe {
    type State = UnitLanes;
    type Config = UnitConfig;
    type Obs = UnitObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;
    const LANE_STEP: Option<tessera_rl_core::LaneHook<Self>> = Some(no_lanes_soa);

    fn observation_space(cfg: &UnitConfig) -> Result<Space, SpaceError> {
        UnitObs::space(cfg)
    }

    fn action_space(_: &UnitConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(_: &UnitConfig, _: &mut StepRng, _: &mut UnitLanes) {}

    fn step(
        _: &UnitConfig,
        _: &mut UnitLanes,
        _: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
    }

    fn observe(
        _: &UnitConfig,
        _: &UnitLanes,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, UnitObs>,
    ) {
    }
}

#[test]
fn test_row_lane_hook_is_none_for_a_lane_wise_state() {
    // ADR-0098 decision 3 (a): SoaProbe sets LANE_STEP over a lane-wise
    // state; the row arm must never take it, or it would walk an empty
    // row store and step no instance.
    assert!(crate::pool::row_lane_hook::<SoaProbe>().is_none());
    assert!(crate::pool::row_lane_hook::<Probe<1, false, DISCRETE, true>>().is_some());
}

#[test]
fn test_lane_level_lane_wise_state_has_none() {
    let pool = EnvPool::<SoaProbe>::new(UnitConfig {}, options())
        .unwrap_or_else(|e| panic!("construction failed: {e}"));
    assert!(pool.lanes.is_none());
}

/// The eligible shape with a lane step, for an environment that reads its
/// adaptive values (ADR-0091 decision 4): a lane step receives none.
struct AdaptiveProbe;

impl tessera_rl_core::Environment for AdaptiveProbe {
    type State = Unit;
    type Config = UnitConfig;
    type Obs = UnitObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;
    const READS_ADAPTIVE: bool = true;
    const LANE_STEP: Option<tessera_rl_core::LaneHook<Self>> = Some(no_lanes);

    fn observation_space(cfg: &UnitConfig) -> Result<Space, SpaceError> {
        UnitObs::space(cfg)
    }

    fn action_space(_: &UnitConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn reset(_: &UnitConfig, _: &mut StepRng, _: &mut Unit) {}

    fn step(
        _: &UnitConfig,
        _: &mut Unit,
        _: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
    }

    fn observe(
        _: &UnitConfig,
        _: &Unit,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, UnitObs>,
    ) {
    }
}

#[test]
fn test_lane_level_reads_adaptive_has_none() {
    let pool = EnvPool::<AdaptiveProbe>::new(UnitConfig {}, options())
        .unwrap_or_else(|e| panic!("construction failed: {e}"));
    assert!(pool.lanes.is_none());
}
