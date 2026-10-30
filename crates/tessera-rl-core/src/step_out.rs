// SPDX-License-Identifier: MIT OR Apache-2.0

//! `StepOut`: what a transition reports for one instance (SPEC 5.2,
//! ADR-0012, ADR-0028).

use crate::agent::AgentSlot;
use crate::bool8::Bool8;

/// How an episode ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpisodeEnd {
    /// A terminal state of the game was reached.
    Terminated,
    /// The episode was cut short (a time limit or similar).
    Truncated,
}

/// Per-agent outputs of one transition of one instance.
///
/// The engine builds it over this instance's rows of the output buffers,
/// cleared before the step. Episode end and per-agent death are separate:
/// only [`StepOut::end_episode`] (or the step limit) ends the episode
/// (ADR-0028).
///
/// Slots are validated when created ([`AgentSlot::new`]), so every
/// accessor's index is in range. As a safety net, an out-of-range slot is
/// ignored, and debug builds assert.
#[derive(Debug)]
pub struct StepOut<'a> {
    rewards: &'a mut [f32],
    terminated: &'a mut [Bool8],
    truncated: &'a mut [Bool8],
    died: &'a mut [Bool8],
    spawned: &'a mut [Bool8],
    episode_end: Option<EpisodeEnd>,
}

#[inline]
fn slot_mut<T>(buf: &mut [T], slot: AgentSlot) -> Option<&mut T> {
    let item = buf.get_mut(slot.index());
    debug_assert!(item.is_some(), "agent slot {} out of range", slot.index());
    item
}

impl<'a> StepOut<'a> {
    /// Engine-only: a view over one instance's output rows, all of length
    /// `MAX_AGENTS`.
    #[doc(hidden)]
    #[must_use]
    #[inline]
    pub fn new(
        rewards: &'a mut [f32],
        terminated: &'a mut [Bool8],
        truncated: &'a mut [Bool8],
        died: &'a mut [Bool8],
        spawned: &'a mut [Bool8],
    ) -> Self {
        debug_assert!(
            [terminated.len(), truncated.len(), died.len(), spawned.len()]
                .iter()
                .all(|&n| n == rewards.len()),
            "StepOut rows have different lengths"
        );
        Self {
            rewards,
            terminated,
            truncated,
            died,
            spawned,
            episode_end: None,
        }
    }

    /// Sets the reward of `agent` for this step.
    #[inline]
    pub fn set_reward(&mut self, agent: AgentSlot, reward: f32) {
        if let Some(r) = slot_mut(self.rewards, agent) {
            *r = reward;
        }
    }

    /// Adds to the reward of `agent` for this step.
    #[inline]
    pub fn add_reward(&mut self, agent: AgentSlot, reward: f32) {
        if let Some(r) = slot_mut(self.rewards, agent) {
            *r += reward;
        }
    }

    /// Marks `agent` as terminated (it reached a terminal state of its own).
    #[inline]
    pub fn terminate_agent(&mut self, agent: AgentSlot) {
        if let Some(f) = slot_mut(self.terminated, agent) {
            *f = Bool8::TRUE;
        }
    }

    /// Marks `agent` as truncated.
    #[inline]
    pub fn truncate_agent(&mut self, agent: AgentSlot) {
        if let Some(f) = slot_mut(self.truncated, agent) {
            *f = Bool8::TRUE;
        }
    }

    /// Reports that `agent` died this step. The engine clears its alive
    /// flag and bumps its slot generation; the episode continues.
    #[inline]
    pub fn agent_died(&mut self, agent: AgentSlot) {
        if let Some(f) = slot_mut(self.died, agent) {
            *f = Bool8::TRUE;
        }
    }

    /// Reports that `agent` spawned this step.
    #[inline]
    pub fn agent_spawned(&mut self, agent: AgentSlot) {
        if let Some(f) = slot_mut(self.spawned, agent) {
            *f = Bool8::TRUE;
        }
    }

    /// Ends the episode. A later call in the same step overrides an
    /// earlier one.
    #[inline]
    pub fn end_episode(&mut self, end: EpisodeEnd) {
        self.episode_end = Some(end);
    }

    /// Returns how the episode ended this step, if it did.
    #[must_use]
    #[inline]
    pub const fn episode_end(&self) -> Option<EpisodeEnd> {
        self.episode_end
    }
}

#[cfg(test)]
mod tests {
    use super::{EpisodeEnd, StepOut};
    use crate::agent::AgentSlot;
    use crate::bool8::Bool8;

    #[test]
    fn test_step_out_writes_per_agent_rows() {
        let mut rewards = [0.0f32; 2];
        let mut terminated = [Bool8::FALSE; 2];
        let mut truncated = [Bool8::FALSE; 2];
        let mut died = [Bool8::FALSE; 2];
        let mut spawned = [Bool8::FALSE; 2];
        let mut out = StepOut::new(
            &mut rewards,
            &mut terminated,
            &mut truncated,
            &mut died,
            &mut spawned,
        );
        let second = AgentSlot::new(1, 2).unwrap();
        out.set_reward(second, 1.5);
        out.add_reward(second, 0.5);
        out.agent_died(second);
        out.terminate_agent(AgentSlot::FIRST);
        assert_eq!(out.episode_end(), None);
        out.end_episode(EpisodeEnd::Truncated);
        assert_eq!(out.episode_end(), Some(EpisodeEnd::Truncated));
        assert_eq!(rewards, [0.0, 2.0]);
        assert_eq!(terminated, [Bool8::TRUE, Bool8::FALSE]);
        assert_eq!(died, [Bool8::FALSE, Bool8::TRUE]);
        assert_eq!(truncated, [Bool8::FALSE; 2]);
        assert_eq!(spawned, [Bool8::FALSE; 2]);
    }
}
