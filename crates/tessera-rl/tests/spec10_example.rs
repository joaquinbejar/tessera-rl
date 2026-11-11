// SPDX-License-Identifier: MIT OR Apache-2.0

//! The SPEC 10 authoring example, verbatim between the markers, so it
//! cannot drift from the real API again (ADR-0053, issue #64).

// --- SPEC 10 example: begin ---
use tessera_rl::prelude::*;

/// Grid state. 8 + 4 + 2 + 1 + 1 = 16 bytes, alignment 4: no padding.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct GridState {
    pos: [Fx32; 2],
    goal: [i16; 2],
    steps: u16,
    reached_goal: Bool8,
    out_of_bounds: Bool8,
}

/// Grid configuration.
#[derive(EnvConfig, Debug, Clone)]
pub struct GridConfig {
    #[config(default = 16, range = 4..=64)]
    size: u16,
    #[config(default = 200, range = 1..=10_000)]
    max_steps: u16,
}

/// Grid observation.
#[derive(Observation)]
pub struct GridObs {
    // Shape resolved from the config at construction.
    #[space(box(low = 0, high = 255, dtype = u8, shape = |c: &GridConfig| [3, u32::from(c.size), u32::from(c.size)]))]
    pixels: ObsLeaf<u8>,
}

/// The reference grid world.
pub struct Grid;

impl Environment for Grid {
    type State = GridState;
    type Config = GridConfig;
    type Obs = GridObs;
    // A marker: the action space is the value `action_space` returns (ADR-0053).
    type Action = ();
    // One agent slot, no adaptive config fields (ADR-0036).
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &GridConfig) -> Result<Space, SpaceError> {
        GridObs::space(cfg)
    }

    fn action_space(_: &GridConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 4 })
    }

    fn reset(_: &GridConfig, _: &mut StepRng, _: &mut GridState) {
        // Draw the goal.
    }

    fn step(
        _: &GridConfig,
        _: &mut GridState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
        // Validated at the boundary: `Some(0..=3)` for an alive, acting agent.
        let _move = actions.discrete(AgentSlot::FIRST, 0);
        // Move; write rewards; call `out.end_episode(..)` when the episode ends.
    }

    fn observe(
        _: &GridConfig,
        _: &GridState,
        _: AgentSlot,
        _: &mut ObsRng,
        _: &mut ObsWriter<'_, '_, GridObs>,
    ) {
        // Write the pixel planes.
    }
}
// --- SPEC 10 example: end ---

use core::num::NonZeroU32;

#[test]
fn test_spec10_example_builds_a_pool_and_steps() -> Result<(), EngineError> {
    let n = NonZeroU32::new(2).ok_or(EngineError::Poisoned)?;
    let mut pool =
        EnvPool::<Grid, NoMasks>::new(GridConfig::default(), PoolOptions::new(n, Seed::new(7)))?;
    pool.reset_all(Reseed::Keep)?;
    pool.step(ActionInput {
        discrete: &[0, 3],
        continuous: &[],
    })?;
    Ok(())
}
