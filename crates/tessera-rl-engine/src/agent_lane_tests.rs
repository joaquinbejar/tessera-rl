// SPDX-License-Identifier: MIT OR Apache-2.0

//! The agent-lane step path (ADR-0100 decisions 1.5 and 1.6): a pool takes
//! `Environment::AGENT_LANE` when it is set, `agent_lanes` is `Auto`, the
//! environment does not read adaptive values and its state keeps the
//! Array-of-Structs store, at the level `lane_level`'s rule resolves; in
//! every other case it runs the reference step. The probe's hook writes the
//! level it was given into the state, and its reference step writes 1, so a
//! step shows which one ran.

use core::num::NonZeroU32;

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{
    ActionMask, AgentSlot, NoAdaptive, Observation, RecordHeader, SpaceError, StepOut, Tier,
};
use tessera_rl_macros::{EnvConfig, EnvState};
use tessera_rl_num::identity::Seed;
use tessera_rl_num::rng::{ObsRng, StepRng};
use tessera_rl_simd::{Level, Resolved};

use crate::{ActionInput, AgentLanes, EnvPool, Masked, PoolOptions, Reseed};

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

/// What the reference step writes.
const REFERENCE: u32 = 1;

/// What the hook writes for `level`: 100 plus a code per level.
const fn hooked(level: Level) -> u32 {
    100 + match level {
        Level::Scalar => 1,
        Level::Sse42 => 2,
        Level::Avx2 => 3,
        Level::Neon => 4,
    }
}

fn hook(
    _: &UnitConfig,
    _: &NoAdaptive,
    state: &mut Unit,
    _: &Actions<'_, ()>,
    _: &mut StepRng,
    _: &mut StepOut<'_>,
    level: Resolved,
) {
    state.v = hooked(level.level());
}

/// A probe with `A` agents, an agent-lane hook when `HOOK`, adaptive reads
/// when `ADAPT` and two discrete heads when `HEADS`.
struct Probe<const A: usize, const HOOK: bool, const ADAPT: bool, const HEADS: bool>;

impl<const A: usize, const HOOK: bool, const ADAPT: bool, const HEADS: bool>
    tessera_rl_core::Environment for Probe<A, HOOK, ADAPT, HEADS>
{
    type State = Unit;
    type Config = UnitConfig;
    type Obs = UnitObs;
    type Action = ();
    type Header = RecordHeader<A, NoAdaptive>;

    const MAX_AGENTS: usize = A;
    const TIER: Tier = Tier::CrossPlatform;
    const READS_ADAPTIVE: bool = ADAPT;
    const AGENT_LANE: Option<tessera_rl_core::AgentLaneHook<Self>> =
        if HOOK { Some(hook) } else { None };

    fn observation_space(cfg: &UnitConfig) -> Result<Space, SpaceError> {
        UnitObs::space(cfg)
    }

    fn action_space(_: &UnitConfig) -> Result<Space, SpaceError> {
        Ok(if HEADS {
            Space::MultiDiscrete { nvec: vec![2, 2] }
        } else {
            Space::Discrete { n: 2 }
        })
    }

    fn reset(_: &UnitConfig, _: &mut StepRng, state: &mut Unit) {
        state.v = 0;
    }

    fn step(
        _: &UnitConfig,
        state: &mut Unit,
        _: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
        state.v = REFERENCE;
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

impl<const A: usize, const HOOK: bool, const ADAPT: bool, const HEADS: bool> ActionMask
    for Probe<A, HOOK, ADAPT, HEADS>
{
    fn action_mask(_: &UnitConfig, _: &Unit, _: AgentSlot, _: &mut MaskWriter<'_>) {}

    fn noop(_: &UnitConfig, _: &Unit, _: AgentSlot, _: &mut ActionWriter<'_>) {}
}

const N: u32 = 3;

fn options(agent_lanes: AgentLanes, level: Option<Level>) -> PoolOptions {
    let mut o = PoolOptions::new(NonZeroU32::new(N).unwrap_or(NonZeroU32::MIN), Seed::new(1));
    o.agent_lanes = agent_lanes;
    o.simd_level = level;
    o
}

/// Steps a fresh pool once after a reset and returns instance 0's `v`.
fn stepped<E, M>(mut p: EnvPool<E, M>, heads: usize) -> u32
where
    E: tessera_rl_core::Environment<State = Unit>,
    M: crate::MaskProvider<E>,
    E::Config: Sync,
    E::Header: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
{
    p.reset_all(Reseed::Keep)
        .unwrap_or_else(|e| panic!("reset: {e}"));
    let rows = usize::try_from(N).unwrap_or(0) * E::MAX_AGENTS * heads;
    let actions = vec![0; rows];
    p.step(ActionInput {
        discrete: &actions,
        continuous: &[],
    })
    .unwrap_or_else(|e| panic!("step: {e}"));
    p.state(0).map_or(0, |s| s.v)
}

fn run<const A: usize, const HOOK: bool, const ADAPT: bool, const HEADS: bool>(
    agent_lanes: AgentLanes,
    level: Option<Level>,
) -> u32 {
    let p =
        EnvPool::<Probe<A, HOOK, ADAPT, HEADS>>::new(UnitConfig {}, options(agent_lanes, level))
            .unwrap_or_else(|e| panic!("construction failed: {e}"));
    stepped(p, if HEADS { 2 } else { 1 })
}

#[test]
fn test_agent_lane_auto_takes_the_hook_at_the_best_level() {
    assert_eq!(
        run::<1, true, false, false>(AgentLanes::Auto, None),
        hooked(Resolved::best().level())
    );
}

#[test]
fn test_agent_lane_auto_takes_the_hook_at_a_forced_level() {
    for level in Resolved::available() {
        assert_eq!(
            run::<1, true, false, false>(AgentLanes::Auto, Some(level.level())),
            hooked(level.level()),
            "{}",
            level.level()
        );
    }
}

#[test]
fn test_agent_lane_reference_never_takes_the_hook() {
    assert_eq!(
        run::<1, true, false, false>(AgentLanes::Reference, None),
        REFERENCE
    );
}

#[test]
fn test_agent_lane_absent_hook_steps_the_reference() {
    assert_eq!(
        run::<1, false, false, false>(AgentLanes::Auto, None),
        REFERENCE
    );
}

#[test]
fn test_agent_lane_reads_adaptive_is_not_eligible() {
    assert_eq!(
        run::<1, true, true, false>(AgentLanes::Auto, None),
        REFERENCE
    );
}

#[test]
fn test_agent_lane_many_agents_and_heads_are_eligible() {
    assert_eq!(
        run::<4, true, false, true>(AgentLanes::Auto, None),
        hooked(Resolved::best().level())
    );
}

#[test]
fn test_agent_lane_masked_pool_is_eligible() {
    let p = EnvPool::<Probe<4, true, false, false>, Masked>::with_masks(
        UnitConfig {},
        options(AgentLanes::Auto, None),
    )
    .unwrap_or_else(|e| panic!("construction failed: {e}"));
    assert_eq!(stepped(p, 1), hooked(Resolved::best().level()));
}

#[test]
fn test_agent_lane_forced_unavailable_level_is_an_error() {
    let missing = [Level::Scalar, Level::Sse42, Level::Avx2, Level::Neon]
        .into_iter()
        .find(|&l| Resolved::force(l).is_err());
    if let Some(level) = missing {
        // An error whether or not the pool would take the hook.
        for agent_lanes in [AgentLanes::Auto, AgentLanes::Reference] {
            assert!(
                EnvPool::<Probe<1, true, false, false>>::new(
                    UnitConfig {},
                    options(agent_lanes, Some(level))
                )
                .is_err()
            );
        }
    }
}

#[test]
fn test_agent_lane_level_is_fixed_at_construction() {
    let p = EnvPool::<Probe<1, true, false, false>>::new(
        UnitConfig {},
        options(AgentLanes::Auto, None),
    )
    .unwrap_or_else(|e| panic!("construction failed: {e}"));
    assert_eq!(p.agent_lanes, Some(Resolved::best()));
    let q = EnvPool::<Probe<1, true, false, false>>::new(
        UnitConfig {},
        options(AgentLanes::Reference, None),
    )
    .unwrap_or_else(|e| panic!("construction failed: {e}"));
    assert_eq!(q.agent_lanes, None);
}

fn no_lanes(
    _: &UnitConfig,
    _: Resolved,
    _: &mut [Unit; tessera_rl_core::LANES],
    _: &[i32; tessera_rl_core::LANES],
    _: &mut tessera_rl_core::LaneOut,
) {
}

/// A probe that sets both `AGENT_LANE` and `LANE_STEP`: never built into a
/// pool (the construction assert refuses it), only checked.
struct BothHooks;

impl tessera_rl_core::Environment for BothHooks {
    type State = Unit;
    type Config = UnitConfig;
    type Obs = UnitObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;
    const AGENT_LANE: Option<tessera_rl_core::AgentLaneHook<Self>> = Some(hook);
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
fn test_agent_lane_excludes_the_other_lane_hooks() {
    assert!(!crate::pool::agent_lane_exclusive::<BothHooks>());
    assert!(crate::pool::agent_lane_exclusive::<
        Probe<1, true, false, false>,
    >());
    assert!(crate::pool::agent_lane_exclusive::<
        Probe<1, false, false, false>,
    >());
}
