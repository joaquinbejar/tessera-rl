// SPDX-License-Identifier: MIT OR Apache-2.0

//! A one-agent environment with conditional `MultiDiscrete([3, 4, 2])`
//! actions (ADR-0086): `kind` (0 pass, 1 add, 2 bonus), `amount` under
//! `kind` (used by add and bonus) and `bonus` under `kind` (used by bonus
//! only). Masks: bonus is illegal from a total of 20, add takes amounts
//! 0 to 2 only. Three steps per episode.
//!
//! `wide` changes the action space for construction tests: 1 gives
//! `MultiDiscrete([64, 65])`, 2 gives `Discrete(5000)`. `cheat` makes pass
//! read the inactive `amount` head, breaking the `head_active` contract.

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
pub struct Tally {
    pub total: u32,
    pub steps: u32,
}

#[derive(EnvConfig, Debug, Clone)]
pub struct PickerConfig {
    pub wide: u32,
    pub cheat: u32,
}

#[derive(Observation)]
pub struct TallyObs {
    #[space(box(low = 0, high = 1000, shape = [2]))]
    tally: ObsLeaf<i32>,
}

pub struct Picker;

/// Head sizes of the conditional space.
pub const NVEC: [u32; 3] = [3, 4, 2];

impl Environment for Picker {
    type State = Tally;
    type Config = PickerConfig;
    type Obs = TallyObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &PickerConfig) -> Result<Space, SpaceError> {
        TallyObs::space(cfg)
    }

    fn action_space(cfg: &PickerConfig) -> Result<Space, SpaceError> {
        Ok(match cfg.wide {
            1 => Space::MultiDiscrete { nvec: vec![64, 65] },
            2 => Space::Discrete { n: 5000 },
            _ => Space::MultiDiscrete {
                nvec: NVEC.to_vec(),
            },
        })
    }

    fn acting(_: &PickerConfig, _: &Tally, _: AgentSlot) -> bool {
        true
    }

    fn reset(_: &PickerConfig, _: &mut StepRng, state: &mut Tally) {
        *state = Tally::default();
    }

    fn step(
        cfg: &PickerConfig,
        state: &mut Tally,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let me = AgentSlot::FIRST;
        let head = |h: usize| u32::try_from(actions.discrete(me, h).unwrap_or(0)).unwrap_or(0);
        let gain = match head(0) {
            1 => head(1) + 1,
            2 => 10 * (head(2) + 1) + head(1),
            _ if cfg.cheat != 0 => head(1),
            _ => 0,
        };
        state.total += gain;
        state.steps += 1;
        out.set_reward(me, f32::from(u16::try_from(gain).unwrap_or(0)));
        if state.steps >= 3 {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &PickerConfig,
        state: &Tally,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, TallyObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(TallyObs::TALLY) {
            leaf.set(0, i32::try_from(state.total).unwrap_or(0));
            leaf.set(1, i32::try_from(state.steps).unwrap_or(0));
        }
    }
}

impl ActionMask for Picker {
    fn action_mask(_: &PickerConfig, state: &Tally, _: AgentSlot, out: &mut MaskWriter<'_>) {
        out.fill(true);
        // kind 2 (bonus) from a total of 20.
        out.set(2, state.total < 20);
        // amount 3 under kind 1 (add): row 3 + 1 x 4.
        out.set(3 + 4 + 3, false);
    }

    fn noop(_: &PickerConfig, _: &Tally, _: AgentSlot, out: &mut ActionWriter<'_>) {
        for h in 0..3 {
            out.set_discrete(h, 0);
        }
    }

    fn head_parents(cfg: &PickerConfig, head: usize) -> [Option<u16>; 2] {
        match (cfg.wide, head) {
            (0, 1 | 2) => [Some(0), None],
            _ => [None, None],
        }
    }

    fn head_active(cfg: &PickerConfig, head: usize, parents: [u32; 2]) -> bool {
        match (cfg.wide, head) {
            (0, 1) => parents[0] != 0,
            (0, 2) => parents[0] == 2,
            _ => true,
        }
    }
}

impl tessera_rl_core::Validate for Picker {
    fn validate(_: &PickerConfig, _: &Tally) -> Result<(), tessera_rl_core::InvariantViolation> {
        Ok(())
    }
}
