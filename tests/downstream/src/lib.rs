// SPDX-License-Identifier: MIT OR Apache-2.0

//! A toy single-agent quoting environment written as a downstream crate
//! would write it: through the facade `tessera-rl` only, with the derive
//! macros resolving their paths through it.
//!
//! The agent quotes a bid and an ask around a mid price that moves by a
//! random tick each step. Each side is filled when the move reaches it.
//! Prices are integer ticks, sizes are lots and cash is in ticks times lots,
//! so the accounting is exact; the reward is the change in marked value.
//!
//! - Action: `MultiDiscrete([3, 3])`: bid offset and ask offset, 1 to 3
//!   ticks from the mid.
//! - Observation: `Box(i32, [3])`: inventory, mid minus its start, steps.
//! - Ends: terminated after `length` steps.

use tessera_rl::core::io::{Actions, ObsWriter};
use tessera_rl::core::space::Space;
use tessera_rl::core::{
    AgentSlot, Environment, EpisodeEnd, InvariantViolation, NoAdaptive, ObsLeaf, Observation,
    RecordHeader, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl::macros::{EnvConfig, EnvState, Observation};
use tessera_rl::num::rng::{ObsRng, StepRng};

/// The mid price at reset, in ticks.
pub const START_MID: i64 = 10_000;

/// Configuration of the toy market.
#[derive(EnvConfig, Debug, Clone)]
pub struct QuoteConfig {
    /// Steps per episode.
    #[config(default = 32, range = 1..=10_000)]
    pub length: u32,
    /// Inventory bound, in lots; quotes that would pass it are not filled.
    #[config(default = 10, range = 1..=1000)]
    pub max_inventory: i32,
}

/// Plain-data state: integers only.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct QuoteState {
    /// Mid price, in ticks.
    pub mid: i64,
    /// Cash, in ticks times lots.
    pub cash: i64,
    /// Inventory, in lots.
    pub inventory: i32,
    /// Steps in this episode.
    pub steps: u32,
}

/// The observation: one `i32` box leaf.
#[derive(Observation)]
pub struct QuoteObs {
    /// `[inventory, mid - START_MID, steps]`.
    #[space(box(low = -1_000_000, high = 1_000_000, shape = [3]))]
    pub book: ObsLeaf<i32>,
}

/// The toy quoting environment.
pub struct Quote;

impl QuoteState {
    /// Cash plus inventory marked at the mid, in ticks times lots.
    #[must_use]
    pub fn marked_value(&self) -> i64 {
        self.cash + i64::from(self.inventory) * self.mid
    }
}

impl Environment for Quote {
    type State = QuoteState;
    type Config = QuoteConfig;
    type Obs = QuoteObs;
    type Action = ();
    type Header = RecordHeader<1, NoAdaptive>;

    const MAX_AGENTS: usize = 1;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &QuoteConfig) -> Result<Space, SpaceError> {
        QuoteObs::space(cfg)
    }

    fn action_space(_: &QuoteConfig) -> Result<Space, SpaceError> {
        Ok(Space::MultiDiscrete { nvec: vec![3, 3] })
    }

    fn reset(_: &QuoteConfig, _: &mut StepRng, state: &mut QuoteState) {
        *state = QuoteState {
            mid: START_MID,
            ..QuoteState::default()
        };
    }

    fn step(
        cfg: &QuoteConfig,
        state: &mut QuoteState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let agent = AgentSlot::FIRST;
        let bid_offset = i64::from(actions.discrete(agent, 0).unwrap_or(0)) + 1;
        let ask_offset = i64::from(actions.discrete(agent, 1).unwrap_or(0)) + 1;
        let before = state.marked_value();
        let (bid, ask) = (state.mid - bid_offset, state.mid + ask_offset);
        // A move of -3..=3 ticks.
        let moved = i64::from(rng.next_u32() % 7) - 3;
        let next = state.mid + moved;
        if next <= bid && state.inventory < cfg.max_inventory {
            state.inventory += 1;
            state.cash -= bid;
        }
        if next >= ask && state.inventory > -cfg.max_inventory {
            state.inventory -= 1;
            state.cash += ask;
        }
        state.mid = next;
        state.steps += 1;
        #[expect(clippy::cast_precision_loss, reason = "small per-step changes")]
        out.set_reward(agent, (state.marked_value() - before) as f32);
        if state.steps >= cfg.length {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &QuoteConfig,
        state: &QuoteState,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, QuoteObs>,
    ) {
        if let Some(mut leaf) = out.leaf_mut(QuoteObs::BOOK) {
            leaf.set(0, state.inventory);
            leaf.set(1, i32::try_from(state.mid - START_MID).unwrap_or(i32::MAX));
            leaf.set(2, i32::try_from(state.steps).unwrap_or(i32::MAX));
        }
    }
}

/// Imported states must respect the inventory bound and the episode length.
impl Validate for Quote {
    fn validate(cfg: &QuoteConfig, state: &QuoteState) -> Result<(), InvariantViolation> {
        if state.inventory.abs() > cfg.max_inventory {
            return Err(InvariantViolation {
                code: "inventory beyond its bound",
            });
        }
        if state.steps > cfg.length {
            return Err(InvariantViolation {
                code: "steps beyond the episode length",
            });
        }
        Ok(())
    }
}
