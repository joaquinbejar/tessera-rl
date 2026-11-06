// SPDX-License-Identifier: MIT OR Apache-2.0

//! The reference grid world (`tessera/Grid-v1`), SPEC 10 and goal G6: walk to a
//! goal drawn at reset. +1 on the goal, -1 off the grid, truncated at `max_steps`.

mod checks;

use core::num::NonZeroU32;

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    AgentSlot, Bool8, Environment, EpisodeEnd, NoAdaptive, ObsLeaf, Observation, RecordHeader,
    SpaceError, StepOut, Tier,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::Fx32;
use tessera_rl_num::rng::{ObsRng, StepRng};

/// Environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/Grid-v1";
/// Environment version.
pub const ENV_VERSION: u32 = 1;
const ONE: Fx32 = Fx32::from_raw(1 << 16);

/// Grid state, as SPEC section 10 writes it: 16 bytes, no padding.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct GridState {
    /// Agent column and row, always whole cells.
    pub pos: [Fx32; 2],
    /// Goal column and row.
    pub goal: [i16; 2],
    /// Steps taken in this episode.
    pub steps: u16,
    /// Whether the agent stands on the goal.
    pub reached_goal: Bool8,
    /// Whether the agent left the grid.
    pub out_of_bounds: Bool8,
}

impl GridState {
    /// The agent's column and row as integers.
    #[must_use]
    pub const fn cell(&self) -> (i32, i32) {
        (self.pos[0].to_raw() >> 16, self.pos[1].to_raw() >> 16)
    }
}

/// Grid configuration.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct GridConfig {
    /// Cells on a side.
    #[config(default = 16, range = 4..=64)]
    pub size: u16,
    /// Steps before the episode is truncated.
    #[config(default = 200, range = 1..=10_000)]
    pub max_steps: u16,
}

/// Grid observation.
#[derive(Observation)]
pub struct GridObs {
    /// `[channel, row, column]`, row-major.
    #[space(box(low = 0, high = 255, dtype = u8, shape = |c: &GridConfig| [3, u32::from(c.size), u32::from(c.size)]))]
    pub pixels: ObsLeaf<u8>,
}

/// The reference grid world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Grid;

impl Environment for Grid {
    type State = GridState;
    type Config = GridConfig;
    type Obs = GridObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &GridConfig) -> Result<Space, SpaceError> {
        GridObs::space(cfg)
    }

    fn action_space(_: &GridConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 4 })
    }

    /// The agent starts at the top left; the goal is uniform on the rest.
    fn reset(cfg: &GridConfig, rng: &mut StepRng, s: &mut GridState) {
        let n = u32::from(cfg.size);
        let others = NonZeroU32::new(n * n - 1).unwrap_or(NonZeroU32::MIN);
        let k = rng.below_u32(others) + 1;
        let coord = |v: u32| i16::try_from(v).unwrap_or(0);
        *s = GridState {
            goal: [coord(k % n), coord(k / n)],
            ..GridState::default()
        };
    }

    fn step(
        cfg: &GridConfig,
        s: &mut GridState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let Some(agent) = AgentSlot::new(0, 1) else {
            return;
        };
        let [col, row] = s.pos;
        s.pos = match actions.discrete(agent, 0) {
            Some(0) => [col, row.checked_sub(ONE).unwrap_or(row)],
            Some(1) => [col, row.checked_add(ONE).unwrap_or(row)],
            Some(2) => [col.checked_add(ONE).unwrap_or(col), row],
            _ => [col.checked_sub(ONE).unwrap_or(col), row],
        };
        s.steps += 1;
        let (col, row) = s.cell();
        let size = i32::from(cfg.size);
        let off = !(0..size).contains(&col) || !(0..size).contains(&row);
        let won = (col, row) == (i32::from(s.goal[0]), i32::from(s.goal[1]));
        s.out_of_bounds = Bool8::from(off);
        s.reached_goal = Bool8::from(won);
        if off || won {
            out.set_reward(agent, if won { 1.0 } else { -1.0 });
            out.end_episode(EpisodeEnd::Terminated);
        } else if s.steps >= cfg.max_steps {
            out.end_episode(EpisodeEnd::Truncated);
        }
    }

    fn observe(
        cfg: &GridConfig,
        s: &GridState,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, GridObs>,
    ) {
        let Some(mut leaf) = out.leaf_mut(GridObs::PIXELS) else {
            return;
        };
        let n = i32::from(cfg.size);
        let cells = usize::from(cfg.size) * usize::from(cfg.size);
        let goal = (i32::from(s.goal[0]), i32::from(s.goal[1]));
        for (k, (row, col)) in (0..n).flat_map(|r| (0..n).map(move |c| (r, c))).enumerate() {
            let edge = row == 0 || col == 0 || row == n - 1 || col == n - 1;
            let lit = |on: bool| if on { 255 } else { 0 };
            leaf.set(k, lit((col, row) == s.cell()));
            leaf.set(cells + k, lit((col, row) == goal));
            leaf.set(2 * cells + k, lit(edge));
        }
    }
}
