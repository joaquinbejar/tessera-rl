// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scripted baseline (SPEC 16.1): nearest feasible neighbour. The
//! vehicle goes to the nearest unserved customer its load fits, ties broken
//! by the lowest node; with none it goes back to the depot.

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic};
use tessera_rl_num::rng::StepRng;

use super::{DEPOT, Vrp, VrpConfig, VrpState};

/// The node vehicle `v` goes to.
#[must_use]
pub fn act(state: &VrpState, v: usize) -> usize {
    let here = state.node_of(v);
    // `min_by_key` keeps the first minimum: the lowest node.
    (1..=state.customer_count())
        .filter(|&n| state.legal(v, n))
        .min_by_key(|&n| state.distance(here, n))
        .unwrap_or_else(|| {
            if state.legal(v, DEPOT) {
                DEPOT
            } else {
                state.noop(v)
            }
        })
}

/// Nearest feasible neighbour through `Heuristic` (ADR-0079).
impl Heuristic for Vrp {
    fn act(
        _: &VrpConfig,
        state: &VrpState,
        agent: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        out.set_discrete(0, i32::try_from(act(state, agent.index())).unwrap_or(0));
    }
}
