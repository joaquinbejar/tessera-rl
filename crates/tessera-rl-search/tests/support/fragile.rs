// SPDX-License-Identifier: MIT OR Apache-2.0

//! An environment whose action 2 panics in the kernel.

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, NoAdaptive, ObsLeaf, Observation, RecordHeader, SpaceError, StepOut,
    Tier, TurnStructure,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Count {
    pub total: u32,
    pub to_move: u32,
}

#[derive(EnvConfig, Debug, Clone)]
pub struct CountConfig {}

#[derive(Observation)]
pub struct CountObs {
    #[space(box(low = 0, high = 1000, shape = [1]))]
    total: ObsLeaf<i32>,
}

/// Two players add 0 or 1 in turns; action 2 panics in the kernel.
pub struct Fragile;

impl Environment for Fragile {
    type State = Count;
    type Config = CountConfig;
    type Obs = CountObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = 2;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &CountConfig) -> Result<Space, SpaceError> {
        CountObs::space(cfg)
    }

    fn action_space(_: &CountConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 3 })
    }

    fn acting(_: &CountConfig, state: &Count, agent: AgentSlot) -> bool {
        agent.index() == state.to_move as usize
    }

    fn reset(_: &CountConfig, _: &mut StepRng, state: &mut Count) {
        *state = Count::default();
    }

    fn step(
        _: &CountConfig,
        state: &mut Count,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        _: &mut StepOut<'_>,
    ) {
        let me = AgentSlot::new(u16::try_from(state.to_move).unwrap_or(0), 2)
            .unwrap_or(AgentSlot::FIRST);
        let a = actions.discrete(me, 0).unwrap_or(0);
        assert!(a != 2, "fixture: action 2 panics");
        state.total += u32::try_from(a).unwrap_or(0);
        state.to_move ^= 1;
    }

    fn observe(
        _: &CountConfig,
        state: &Count,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, CountObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(CountObs::TOTAL) {
            leaf.set(0, i32::try_from(state.total).unwrap_or(0));
        }
    }
}

impl tessera_rl_core::Validate for Fragile {
    fn validate(_: &CountConfig, _: &Count) -> Result<(), tessera_rl_core::InvariantViolation> {
        Ok(())
    }
}
