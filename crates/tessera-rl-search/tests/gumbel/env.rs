// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ladder: a two-player sequential test game for the Gumbel root, whose
//! evaluator value at every leaf depends only on the root action that
//! leads to it, so each root edge receives a known value per visit.
//!
//! - `Discrete(9)`; action 8 is illegal everywhere.
//! - The state holds the first action taken (`first`, 0 at the root), the
//!   depth and the player to move.
//! - `alternate = 1`: the mover changes every move. `alternate = 0`: it
//!   changes only when the new depth is even, so the root's mover also
//!   moves at depth 1 (the same player moves again).
//! - Never terminal within the tests (an episode ends at depth 200 with
//!   reward 0).

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Environment, EpisodeEnd, NoAdaptive, ObsLeaf, Observation, RecordHeader,
    SpaceError, StepOut, Tier, TurnStructure,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

pub const ACTIONS: u32 = 9;
pub const ILLEGAL: u32 = 8;
const MAX_DEPTH: u32 = 200;

#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct LadderConfig {
    /// 1: the mover alternates every move; 0: only at even depths.
    #[config(default = 1, range = 0..=1)]
    pub alternate: u32,
}

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct LadderState {
    pub first: u32,
    pub depth: u32,
    pub mover: u32,
}

#[derive(Observation)]
pub struct LadderObs {
    /// First action, depth, mover.
    #[space(box(low = 0, high = 1000, dtype = u32, shape = [3]))]
    pub view: ObsLeaf<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ladder;

fn slot(i: u32) -> Option<AgentSlot> {
    AgentSlot::new(u16::try_from(i).ok()?, 2)
}

impl Environment for Ladder {
    type State = LadderState;
    type Config = LadderConfig;
    type Obs = LadderObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = 2;
    const TIER: Tier = Tier::CrossPlatform;
    const PERFECT_INFORMATION: bool = true;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &LadderConfig) -> Result<Space, SpaceError> {
        LadderObs::space(cfg)
    }

    fn action_space(_: &LadderConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: ACTIONS })
    }

    fn acting(_: &LadderConfig, state: &LadderState, agent: AgentSlot) -> bool {
        state.depth < MAX_DEPTH && agent.index() == state.mover as usize
    }

    fn reset(_: &LadderConfig, _: &mut StepRng, state: &mut LadderState) {
        *state = LadderState::default();
    }

    fn step(
        cfg: &LadderConfig,
        state: &mut LadderState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let Some(mover) = slot(state.mover) else {
            return;
        };
        let action = actions
            .discrete(mover, 0)
            .and_then(|a| u32::try_from(a).ok())
            .unwrap_or(0);
        if state.depth == 0 {
            state.first = action + 1;
        }
        state.depth += 1;
        if cfg.alternate == 1 || state.depth.is_multiple_of(2) {
            state.mover = 1 - state.mover;
        }
        if state.depth >= MAX_DEPTH {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &LadderConfig,
        state: &LadderState,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, LadderObs>,
    ) {
        let Some(mut leaf) = out.leaf_mut(LadderObs::VIEW) else {
            return;
        };
        for (k, v) in [state.first, state.depth, state.mover]
            .into_iter()
            .enumerate()
        {
            leaf.set(k, v);
        }
    }
}

impl ActionMask for Ladder {
    fn action_mask(_: &LadderConfig, _: &LadderState, _: AgentSlot, out: &mut MaskWriter<'_>) {
        for a in 0..ACTIONS {
            out.set(a as usize, a != ILLEGAL);
        }
    }

    fn noop(_: &LadderConfig, _: &LadderState, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, 0);
    }
}
