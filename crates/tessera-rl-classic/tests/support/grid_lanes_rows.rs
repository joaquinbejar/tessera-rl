// SPDX-License-Identifier: MIT OR Apache-2.0

//! `GridLanesRows`: `GridLanes` on an Array-of-Structs state. Its state has
//! the same six fields in the same order, so its canonical encoding, and with
//! it every record, persisted file, replay file and hash, is byte-identical
//! to `GridLanes`'; only the in-memory store differs. It proves the record
//! interchange between the two stores (ADR-0090 decision 3).

use tessera_rl_classic::{GridConfig, GridLanes, GridLanesState, GridObs, GridOutcome};
use tessera_rl_core::io::{ActionWriter, Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Environment, HeaderView, Heuristic, InvariantViolation, NoAdaptive, Persist,
    RecordHeader, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_macros::EnvState;
use tessera_rl_num::Fx32;
use tessera_rl_num::rng::{ObsRng, StepRng};

/// `GridLanesState`'s fields on the row store.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct GridRowsState {
    /// Agent column.
    pub col: Fx32,
    /// Agent row.
    pub row: Fx32,
    /// Goal column.
    pub goal_col: i32,
    /// Goal row.
    pub goal_row: i32,
    /// Steps taken.
    pub steps: u32,
    /// Where the agent stands.
    pub outcome: GridOutcome,
}

impl GridRowsState {
    /// The same state on the lane-wise type.
    pub const fn lanes(self) -> GridLanesState {
        GridLanesState {
            col: self.col,
            row: self.row,
            goal_col: self.goal_col,
            goal_row: self.goal_row,
            steps: self.steps,
            outcome: self.outcome,
        }
    }

    /// The state of a `GridLanesState`.
    pub const fn of(s: GridLanesState) -> Self {
        Self {
            col: s.col,
            row: s.row,
            goal_col: s.goal_col,
            goal_row: s.goal_row,
            steps: s.steps,
            outcome: s.outcome,
        }
    }
}

/// `GridLanes` on the Array-of-Structs store.
pub struct GridLanesRows;

impl Environment for GridLanesRows {
    type State = GridRowsState;
    type Config = GridConfig;
    type Obs = GridObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &GridConfig) -> Result<Space, SpaceError> {
        GridLanes::observation_space(cfg)
    }

    fn action_space(cfg: &GridConfig) -> Result<Space, SpaceError> {
        GridLanes::action_space(cfg)
    }

    fn reset(cfg: &GridConfig, rng: &mut StepRng, state: &mut GridRowsState) {
        let mut s = state.lanes();
        GridLanes::reset(cfg, rng, &mut s);
        *state = GridRowsState::of(s);
    }

    fn step(
        cfg: &GridConfig,
        state: &mut GridRowsState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let mut s = state.lanes();
        GridLanes::step(cfg, &mut s, actions, rng, out);
        *state = GridRowsState::of(s);
    }

    fn observe(
        cfg: &GridConfig,
        state: &GridRowsState,
        agent: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, GridObs>,
    ) {
        GridLanes::observe(cfg, &state.lanes(), agent, rng, out);
    }
}

impl Validate for GridLanesRows {
    fn validate(cfg: &GridConfig, s: &GridRowsState) -> Result<(), InvariantViolation> {
        GridLanes::validate(cfg, &s.lanes())
    }

    fn validate_record(
        cfg: &GridConfig,
        header: &HeaderView<'_>,
        s: &GridRowsState,
    ) -> Result<(), InvariantViolation> {
        GridLanes::validate_record(cfg, header, &s.lanes())
    }
}

impl Persist for GridLanesRows {
    const ENV_ID: &'static str = GridLanes::ENV_ID;
    const ENV_VERSION: u32 = GridLanes::ENV_VERSION;
}

impl Heuristic for GridLanesRows {
    fn act(
        cfg: &GridConfig,
        state: &GridRowsState,
        agent: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        GridLanes::act(cfg, &state.lanes(), agent, rng, out);
    }
}
