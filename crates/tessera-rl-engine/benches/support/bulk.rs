// SPDX-License-Identifier: MIT OR Apache-2.0

//! A single-agent environment with a 4 KiB state, for the snapshot-cost
//! bench: the SPEC 9.2 targets are per KB, and the test counter's state is
//! too small to show a per-byte cost. Each step touches one word, so the
//! cost of a snapshot is dominated by the copy, not by the environment.

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, EpisodeEnd, NoAdaptive, ObsLeaf, Observation, RecordHeader, Reserved,
    SpaceError, StepOut, Tier,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

/// Rows of the payload; each row is 32 words (256 bytes).
pub const ROWS: usize = 16;

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct BulkState {
    pub words: [[u64; 32]; ROWS],
    pub steps: u32,
    pub reserved: Reserved<4>,
}

#[derive(EnvConfig, Debug, Clone)]
pub struct BulkConfig {
    #[config(default = 1000, range = 1..=1_000_000)]
    pub length: u32,
}

#[derive(Observation)]
pub struct BulkObs {
    #[space(box(low = 0, high = 1_000_000, shape = [1]))]
    pub steps: ObsLeaf<i32>,
}

pub struct Bulk;

impl Environment for Bulk {
    type State = BulkState;
    type Config = BulkConfig;
    type Obs = BulkObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &BulkConfig) -> Result<Space, SpaceError> {
        BulkObs::space(cfg)
    }

    fn action_space(_cfg: &BulkConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn initial_agents(_cfg: &BulkConfig) -> usize {
        1
    }

    fn reset(_cfg: &BulkConfig, rng: &mut StepRng, state: &mut BulkState) {
        // Fill every word, so no byte of the payload is trivially zero.
        for row in &mut state.words {
            for word in row.iter_mut() {
                *word = u64::from(rng.next_u32());
            }
        }
        state.steps = 0;
    }

    fn step(
        cfg: &BulkConfig,
        state: &mut BulkState,
        _actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let row = usize::try_from(state.steps).unwrap_or(0) % ROWS;
        if let Some(word) = state.words.get_mut(row).and_then(|r| r.first_mut()) {
            *word ^= u64::from(rng.next_u32());
        }
        state.steps += 1;
        out.set_reward(AgentSlot::FIRST, 0.0);
        if state.steps >= cfg.length {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _cfg: &BulkConfig,
        state: &BulkState,
        _agent: AgentSlot,
        _rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, BulkObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(BulkObs::STEPS) {
            leaf.set(0, state.steps.cast_signed());
        }
    }
}
