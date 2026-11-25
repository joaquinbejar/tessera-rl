// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gather (`tessera/Gather-v1`), the `Set` observation example of ADR-0082:
//! an agent walks a grid and collects up to [`MAX_ITEMS`] items. +1 per
//! item; terminated when every item is collected, truncated at `max_steps`.
//!
//! The observation is `{agent: Box[2] i32, items: Set<Box[2] i32, 8>}`:
//! slot `i` holds item `i`'s `[column, row]` while it is on the grid, and is
//! absent (mask false, row zero) once collected or when `i >= items`.
//!
//! `items` is adaptive (`#[config(adaptive)]`, ADR-0081): a pool with
//! targets places each episode's items from its slot's target, read at
//! reset, so a curriculum can change the difficulty between episodes.
//! Version 2: version 1 had a fixed `items`; at the default, with no
//! targets, both give the same trajectories, and only the record and
//! header encoding differ.

mod checks;

use core::num::NonZeroU32;

use tessera_rl_core::io::{Actions, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    Adaptive, AgentSlot, Bool8, Environment, EpisodeEnd, ObsLeaf, ObsSet, Observation,
    RecordHeader, SpaceError, StepOut, Tier,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

/// Environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/Gather-v1";
/// Environment version.
pub const ENV_VERSION: u32 = 2;
/// Capacity of the item set (ADR-0082 point 9).
pub const MAX_ITEMS: usize = 8;

/// Gather state: 48 bytes, no padding.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct GatherState {
    /// Agent column and row.
    pub agent: [i16; 2],
    /// Item columns and rows, by slot.
    pub items: [[i16; 2]; MAX_ITEMS],
    /// Whether each item is still on the grid.
    pub present: [Bool8; MAX_ITEMS],
    /// Steps taken in this episode.
    pub steps: u16,
    /// Items still on the grid.
    pub left: u8,
    /// Items collected on the last step.
    pub collected: u8,
}

/// Gather configuration.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct GatherConfig {
    /// Cells on a side.
    #[config(default = 8, range = 4..=16)]
    pub size: u16,
    /// Items placed at reset, at most [`MAX_ITEMS`]. Adaptive: a pool with
    /// targets reads each slot's target at reset (ADR-0081).
    #[config(default = 8, range = 1..=8, adaptive)]
    pub items: u8,
    /// Steps before the episode is truncated.
    #[config(default = 100, range = 1..=10_000)]
    pub max_steps: u16,
}

/// Gather observation (ADR-0082): `agent`, then `items` and `items_mask`.
#[derive(Observation)]
pub struct GatherObs {
    /// Agent `[column, row]`.
    #[space(box(low = 0, high = |c: &GatherConfig| i32::from(c.size) - 1, shape = [2]))]
    pub agent: ObsLeaf<i32>,
    /// Items on the grid, `[column, row]` per slot.
    #[space(set(max = MAX_ITEMS, inner(box(low = 0, high = |c: &GatherConfig| i32::from(c.size) - 1, shape = [2]))))]
    pub items: ObsSet<i32>,
}

/// The Gather environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Gather;

/// Column and row of cell `k` on a side of `n`.
fn cell(k: u32, n: u32) -> [i16; 2] {
    let coord = |v: u32| i16::try_from(v).unwrap_or(0);
    [coord(k % n), coord(k / n)]
}

/// Places the agent uniformly on the grid and `items` items uniformly on
/// the other cells (items may share a cell). The draw order is version 1's.
fn place(cfg: &GatherConfig, items: u8, rng: &mut StepRng, s: &mut GatherState) {
    let n = u32::from(cfg.size);
    let cells = NonZeroU32::new(n * n).unwrap_or(NonZeroU32::MIN);
    let others = NonZeroU32::new(n * n - 1).unwrap_or(NonZeroU32::MIN);
    let start = rng.below_u32(cells);
    *s = GatherState {
        agent: cell(start, n),
        ..GatherState::default()
    };
    let count = usize::from(items).min(MAX_ITEMS);
    for (item, present) in s.items.iter_mut().zip(s.present.iter_mut()).take(count) {
        let k = rng.below_u32(others);
        *item = cell(if k >= start { k + 1 } else { k }, n);
        *present = Bool8::TRUE;
    }
    s.left = u8::try_from(count).unwrap_or(0);
}

impl Environment for Gather {
    /// Parallel natively and AEC through cycle buffering (SPEC 12, ADR-0080).
    const ADAPTERS: tessera_rl_core::Adapters =
        tessera_rl_core::Adapters::PZ_PARALLEL.union(tessera_rl_core::Adapters::PZ_AEC);
    type State = GatherState;
    type Config = GatherConfig;
    type Obs = GatherObs;
    type Action = ();
    type Header = RecordHeader<1, GatherConfigAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;
    const READS_ADAPTIVE: bool = true;

    fn observation_space(cfg: &GatherConfig) -> Result<Space, SpaceError> {
        GatherObs::space(cfg)
    }

    fn action_space(_: &GatherConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 4 })
    }

    /// The agent is uniform on the grid; each item is uniform on the other
    /// cells (items may share a cell). `cfg.items` items.
    fn reset(cfg: &GatherConfig, rng: &mut StepRng, s: &mut GatherState) {
        place(cfg, cfg.items, rng, s);
    }

    /// As [`Self::reset`], with the slot's adaptive `items` (its target in a
    /// pool with targets, else the config's).
    fn reset_with(
        cfg: &GatherConfig,
        adaptive: &Adaptive<Self>,
        rng: &mut StepRng,
        s: &mut GatherState,
    ) {
        place(cfg, adaptive.items, rng, s);
    }

    fn step(
        cfg: &GatherConfig,
        s: &mut GatherState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let Some(agent) = AgentSlot::new(0, 1) else {
            return;
        };
        let last = i16::try_from(cfg.size).unwrap_or(i16::MAX) - 1;
        let [col, row] = s.agent;
        s.agent = match actions.discrete(agent, 0) {
            Some(0) => [col, (row - 1).max(0)],
            Some(1) => [col, (row + 1).min(last)],
            Some(2) => [(col + 1).min(last), row],
            _ => [(col - 1).max(0), row],
        };
        s.steps += 1;
        let mut collected = 0u8;
        for (item, present) in s.items.iter().zip(s.present.iter_mut()) {
            if present.get() && *item == s.agent {
                *present = Bool8::FALSE;
                collected += 1;
            }
        }
        s.collected = collected;
        s.left -= collected;
        if collected > 0 {
            out.set_reward(agent, f32::from(collected));
        }
        if s.left == 0 {
            out.end_episode(EpisodeEnd::Terminated);
        } else if s.steps >= cfg.max_steps {
            out.end_episode(EpisodeEnd::Truncated);
        }
    }

    fn observe(
        _: &GatherConfig,
        s: &GatherState,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, GatherObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(GatherObs::AGENT) {
            leaf.set(0, i32::from(s.agent[0]));
            leaf.set(1, i32::from(s.agent[1]));
        }
        let Some(mut set) = out.set_mut(GatherObs::ITEMS) else {
            return;
        };
        for (slot, (item, present)) in s.items.iter().zip(s.present.iter()).enumerate() {
            if !present.get() {
                continue;
            }
            set.set(slot, true);
            if let Some(mut row) = set.item(slot) {
                row.set(0, i32::from(item[0]));
                row.set(1, i32::from(item[1]));
            }
        }
    }
}
