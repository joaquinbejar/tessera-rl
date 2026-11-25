// SPDX-License-Identifier: MIT OR Apache-2.0

//! Imperfect information (ADR-0035) and the player-only heuristic
//! (ADR-0079 item 4.2) of `tessera/RTS-v1` (#586).
//!
//! An agent's information state is its side's view, [`crate::fog::seen`]:
//! own units, terrain where explored, own ore, and the enemy units on cells
//! the side sees now, without the facts the observation never shows and
//! packed in cell order (`rts-design.md` section 14, ruling 5). The
//! scripted bots already read only that view, so [`InfoHeuristic`] is
//! [`crate::bots::Bot::Rush`] on it, the same reference bot as
//! [`tessera_rl_core::Heuristic`]: `act` equals `act_info` on
//! `info_state`.
//!
//! No `Determinize`: the search pool still refuses the environment
//! (`PERFECT_INFORMATION` is `false`).

#![deny(clippy::float_arithmetic)]

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, ImperfectInformation, InfoHeuristic};
use tessera_rl_num::rng::StepRng;

use crate::bots::Bot;
use crate::env::{Rts, RtsConfig, RtsState, side_of_agent};

impl ImperfectInformation for Rts {
    /// The side view: a fogged copy of the state, plain data on the stack.
    type InfoState = RtsState;

    fn info_state(cfg: &RtsConfig, state: &RtsState, agent: AgentSlot) -> RtsState {
        crate::fog::seen(cfg, state, side_of_agent(cfg, agent.index()))
    }
}

impl InfoHeuristic for Rts {
    fn act_info(
        cfg: &RtsConfig,
        info: &RtsState,
        agent: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        Bot::Rush.act_view(cfg, info, agent, rng, out);
    }
}
