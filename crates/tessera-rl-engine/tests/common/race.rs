// SPDX-License-Identifier: MIT OR Apache-2.0

//! A turn-taking test environment (ADR-0013), shared by the turn tests and
//! the turn benchmarks: a race to a target. Two players alternate adding 1
//! or 2 to a shared total; whoever reaches the target exactly wins. Adding
//! 2 is masked when it would pass the target.

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Environment, EpisodeEnd, NoAdaptive, ObsLeaf, Observation, RecordHeader,
    SpaceError, StepOut, Tier, TurnStructure,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Race {
    pub total: u32,
    pub to_move: u32,
}

#[derive(EnvConfig, Debug, Clone)]
pub struct RaceConfig {
    #[config(default = 7, range = 2..=100)]
    pub target: u32,
    /// Fault injection: name both agents as acting (a Sequential violation).
    #[config(default = false)]
    pub both_act: bool,
}

#[derive(Observation)]
pub struct RaceObs {
    #[space(box(low = 0, high = 1000, shape = [2]))]
    pub values: ObsLeaf<i32>,
}

pub struct RaceEnv;

pub const P0: AgentSlot = AgentSlot::FIRST;
pub const P1: AgentSlot = match AgentSlot::new(1, 2) {
    Some(s) => s,
    None => AgentSlot::FIRST,
};

impl Environment for RaceEnv {
    type State = Race;
    type Config = RaceConfig;
    type Obs = RaceObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = 2;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &RaceConfig) -> Result<Space, SpaceError> {
        RaceObs::space(cfg)
    }

    fn action_space(_: &RaceConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn acting(cfg: &RaceConfig, state: &Race, agent: AgentSlot) -> bool {
        // Relies on the invariant `Validate` checks: the engine never asks
        // about an invalid state (an imported record is validated first).
        assert!(state.to_move <= 1, "acting called on an invalid state");
        cfg.both_act || agent.index() == state.to_move as usize
    }

    fn reset(_: &RaceConfig, _: &mut StepRng, state: &mut Race) {
        *state = Race::default();
    }

    fn step(
        cfg: &RaceConfig,
        state: &mut Race,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let (me, other) = if state.to_move == 0 {
            (P0, P1)
        } else {
            (P1, P0)
        };
        let add = u32::try_from(actions.discrete(me, 0).unwrap_or(0)).unwrap_or(0) + 1;
        state.total = (state.total + add).min(cfg.target);
        if state.total == cfg.target {
            out.set_reward(me, 1.0);
            out.set_reward(other, -1.0);
            out.end_episode(EpisodeEnd::Terminated);
        } else {
            state.to_move ^= 1;
        }
    }

    fn observe(
        _: &RaceConfig,
        state: &Race,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, RaceObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(RaceObs::VALUES) {
            leaf.set(0, i32::try_from(state.total).unwrap_or(0));
            leaf.set(1, i32::from(agent.index() == state.to_move as usize));
        }
    }
}

/// The race's invariants: the total never passes the target, and the
/// player to move is 0 or 1.
impl tessera_rl_core::Validate for RaceEnv {
    fn validate(cfg: &RaceConfig, state: &Race) -> Result<(), tessera_rl_core::InvariantViolation> {
        if state.total > cfg.target || state.to_move > 1 {
            return Err(tessera_rl_core::InvariantViolation {
                code: "total past the target or no player to move",
            });
        }
        Ok(())
    }
}

/// Adding 2 is illegal when it would pass the target; the no-op adds 1.
impl ActionMask for RaceEnv {
    fn action_mask(cfg: &RaceConfig, state: &Race, _: AgentSlot, out: &mut MaskWriter<'_>) {
        out.fill(true);
        if state.total + 2 > cfg.target {
            out.set(1, false);
        }
    }

    fn noop(_: &RaceConfig, _: &Race, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, 0);
    }
}
