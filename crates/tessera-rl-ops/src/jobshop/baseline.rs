// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scripted dispatching rules (SPEC 16.1's scripted baseline). Each picks a
//! legal job, breaking ties by the lowest job index; with no legal job
//! each returns the no-op. SPT is the published baseline and the
//! `Heuristic` of the environment.

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic};
use tessera_rl_num::rng::StepRng;

use super::{JobShop, JobShopConfig, JobShopState};

/// A dispatching rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// Shortest processing time: the shortest next operation.
    Spt,
    /// Most work remaining: the job with the most work left.
    Mwkr,
    /// First in, first out: the job that has been ready longest.
    Fifo,
}

/// The job `rule` dispatches in `state`.
#[must_use]
pub fn act(rule: Rule, state: &JobShopState) -> usize {
    let key = |j: usize| -> Option<u64> {
        let (_, duration) = state.next(j)?;
        Some(match rule {
            Rule::Spt => u64::from(duration),
            Rule::Mwkr => u64::from(u32::MAX - state.remaining_work(j)),
            Rule::Fifo => u64::from(state.job_ready.get(j).copied().unwrap_or(0)),
        })
    };
    // `min_by_key` keeps the first minimum: the lowest job index.
    (0..state.job_count())
        .filter_map(|j| key(j).map(|k| (k, j)))
        .min_by_key(|&(k, _)| k)
        .map_or_else(|| state.noop(), |(_, j)| j)
}

/// The SPT rule through `Heuristic` (ADR-0079).
impl Heuristic for JobShop {
    fn act(
        _: &JobShopConfig,
        state: &JobShopState,
        _: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        out.set_discrete(0, i32::try_from(act(Rule::Spt, state)).unwrap_or(0));
    }
}
