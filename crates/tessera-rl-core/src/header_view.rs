// SPDX-License-Identifier: MIT OR Apache-2.0

//! A read-only, lifecycle-aware view of a record header for
//! `Validate::validate_record` (ADR-0040): per-slot liveness and
//! generations, the episode step counter and the record's phase. RNG
//! identity and statistics are not exposed.

use crate::header::{AgentRow, RecordLayout};

/// Where a record is in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Phase {
    /// Constructed or reseeded, never reset: no episode has started, so
    /// active-episode invariants do not apply.
    NeverReset,
    /// An episode is running.
    Active,
    /// The episode ended and the instance waits for its reset.
    PendingReset,
}

/// One agent slot as seen by the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct AgentView {
    /// The slot holds a living agent.
    pub alive: bool,
    /// The agent acts this step.
    pub acting: bool,
    /// The slot's occupant generation.
    pub generation: u32,
}

/// A read-only view of a record header.
#[derive(Debug, Clone, Copy)]
pub struct HeaderView<'a> {
    agents: &'a [AgentRow],
    episode_steps: u32,
    phase: Phase,
}

impl<'a> HeaderView<'a> {
    /// Engine-only: the view of `header`.
    #[doc(hidden)]
    #[must_use]
    pub fn of<H: RecordLayout>(header: &'a H) -> Self {
        let engine = header.engine();
        let phase = if engine.fresh.get() {
            Phase::NeverReset
        } else if engine.pending_reset.get() {
            Phase::PendingReset
        } else {
            Phase::Active
        };
        Self {
            agents: header.agents(),
            episode_steps: engine.episode_steps,
            phase,
        }
    }

    /// The record's phase.
    #[must_use]
    pub const fn phase(&self) -> Phase {
        self.phase
    }

    /// Transitions taken in the current episode.
    #[must_use]
    pub const fn episode_steps(&self) -> u32 {
        self.episode_steps
    }

    /// Number of agent slots.
    #[must_use]
    pub const fn agents(&self) -> usize {
        self.agents.len()
    }

    /// Slot `index`, or `None` past the last slot.
    #[must_use]
    pub fn agent(&self, index: usize) -> Option<AgentView> {
        self.agents.get(index).map(|row| AgentView {
            alive: row.alive.get(),
            acting: row.acting.get(),
            generation: row.generation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{HeaderView, Phase};
    use crate::bool8::Bool8;
    use crate::header::{NoAdaptive, RecordHeader};
    use crate::state_init::StateInit;

    #[test]
    fn test_header_view_phase_follows_fresh_and_pending() {
        let mut h = RecordHeader::<2, NoAdaptive>::state_default();
        h.engine.fresh = Bool8::TRUE;
        h.engine.pending_reset = Bool8::TRUE;
        assert_eq!(HeaderView::of(&h).phase(), Phase::NeverReset);
        h.engine.fresh = Bool8::FALSE;
        assert_eq!(HeaderView::of(&h).phase(), Phase::PendingReset);
        h.engine.pending_reset = Bool8::FALSE;
        h.engine.episode_steps = 7;
        h.agents[1].alive = Bool8::TRUE;
        h.agents[1].generation = 3;
        let view = HeaderView::of(&h);
        assert_eq!(view.phase(), Phase::Active);
        assert_eq!(view.episode_steps(), 7);
        assert_eq!(view.agents(), 2);
        let slot = view.agent(1).unwrap();
        assert!(slot.alive && !slot.acting && slot.generation == 3);
        assert!(view.agent(2).is_none());
    }
}
