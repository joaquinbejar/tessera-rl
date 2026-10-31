// SPDX-License-Identifier: MIT OR Apache-2.0

//! Output buffers (SPEC 5.2), allocated once at construction.
//!
//! Per-agent buffers are `[N, A]` row-major; per-instance buffers `[N]`.
//! Observations are one byte buffer per leaf, `[N, A, leaf bytes]`, holding
//! little-endian elements of the leaf dtype. Results borrowed through
//! `&self` stay valid until the next call that takes `&mut self` (the
//! borrow window of ADR-0003 for Sync and Inline modes).

use std::sync::Arc;

use tessera_rl_core::Bool8;

use crate::buffer::{Buf, OutputField, OutputStorage, StorageBuilder};

use crate::layout::{Leaf, MAX_LEAVES};

/// Output buffers of a pool, all in one shared [`OutputStorage`]. The
/// buffers are private: `Outputs::view_all` is the only way to write them,
/// and it starts a new write generation first.
#[derive(Debug)]
pub struct Outputs {
    pub(crate) agents: usize,
    observations: Vec<Buf<u8>>,
    terminal_observations: Vec<Buf<u8>>,
    pub(crate) leaf_bytes: Vec<usize>,
    rewards: Buf<f32>,
    agent_terminated: Buf<Bool8>,
    agent_truncated: Buf<Bool8>,
    died: Buf<Bool8>,
    spawned: Buf<Bool8>,
    /// Invalid agent events of the last call, `[N, A]`.
    invalid_agent_event: Buf<Bool8>,
    alive: Buf<Bool8>,
    acting: Buf<Bool8>,
    generations: Buf<u32>,
    episode_terminated: Buf<Bool8>,
    episode_truncated: Buf<Bool8>,
    episode_return: Buf<f32>,
    episode_length: Buf<f32>,
    episode_illegal_actions: Buf<f32>,
    episode_invalid_agent_events: Buf<f32>,
    masks: Buf<Bool8>,
    pub(crate) mask_len: usize,
}

impl Outputs {
    pub(crate) fn new(
        num_envs: usize,
        agents: usize,
        leaves: &[Leaf],
        terminal: bool,
        mask_len: usize,
    ) -> Option<Self> {
        let rows = num_envs.checked_mul(agents)?;
        let mut b = StorageBuilder::default();
        let obs = leaves
            .iter()
            .enumerate()
            .map(|(n, leaf)| {
                b.reserve::<u8>(OutputField::Observation(n), rows.checked_mul(leaf.bytes)?)
            })
            .collect::<Option<Vec<_>>>()?;
        let term = if terminal {
            leaves
                .iter()
                .enumerate()
                .map(|(n, leaf)| {
                    b.reserve::<u8>(
                        OutputField::TerminalObservation(n),
                        rows.checked_mul(leaf.bytes)?,
                    )
                })
                .collect::<Option<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let rewards = b.reserve::<f32>(OutputField::Rewards, rows)?;
        let agent_terminated = b.reserve::<Bool8>(OutputField::AgentTerminated, rows)?;
        let agent_truncated = b.reserve::<Bool8>(OutputField::AgentTruncated, rows)?;
        let died = b.reserve::<Bool8>(OutputField::Died, rows)?;
        let spawned = b.reserve::<Bool8>(OutputField::Spawned, rows)?;
        let invalid_agent_event = b.reserve::<Bool8>(OutputField::InvalidAgentEvent, rows)?;
        let alive = b.reserve::<Bool8>(OutputField::Alive, rows)?;
        let acting = b.reserve::<Bool8>(OutputField::Acting, rows)?;
        let generations = b.reserve::<u32>(OutputField::Generations, rows)?;
        let episode_terminated = b.reserve::<Bool8>(OutputField::EpisodeTerminated, num_envs)?;
        let episode_truncated = b.reserve::<Bool8>(OutputField::EpisodeTruncated, num_envs)?;
        let episode_return = b.reserve::<f32>(OutputField::EpisodeReturn, num_envs)?;
        let episode_length = b.reserve::<f32>(OutputField::EpisodeLength, num_envs)?;
        let episode_illegal_actions =
            b.reserve::<f32>(OutputField::EpisodeIllegalActions, num_envs)?;
        let episode_invalid_agent_events =
            b.reserve::<f32>(OutputField::EpisodeInvalidAgentEvents, num_envs)?;
        let masks = b.reserve::<Bool8>(OutputField::Masks, rows.checked_mul(mask_len)?)?;
        let mut r = b.finish()?;
        Some(Self {
            agents,
            observations: obs
                .into_iter()
                .map(|slot| r.take(slot))
                .collect::<Option<_>>()?,
            terminal_observations: term
                .into_iter()
                .map(|slot| r.take(slot))
                .collect::<Option<_>>()?,
            leaf_bytes: leaves.iter().map(|l| l.bytes).collect(),
            rewards: r.take(rewards)?,
            agent_terminated: r.take(agent_terminated)?,
            agent_truncated: r.take(agent_truncated)?,
            died: r.take(died)?,
            spawned: r.take(spawned)?,
            invalid_agent_event: r.take(invalid_agent_event)?,
            alive: r.take(alive)?,
            acting: r.take(acting)?,
            generations: r.take(generations)?,
            episode_terminated: r.take(episode_terminated)?,
            episode_truncated: r.take(episode_truncated)?,
            episode_return: r.take(episode_return)?,
            episode_length: r.take(episode_length)?,
            episode_illegal_actions: r.take(episode_illegal_actions)?,
            episode_invalid_agent_events: r.take(episode_invalid_agent_events)?,
            masks: r.take(masks)?,
            mask_len,
        })
    }

    /// The shared storage of every output buffer, for read-only borrowers:
    /// region pointers, lengths, dtypes and the write generation. Holding
    /// the `Arc` keeps the memory alive after the pool is dropped.
    #[must_use]
    pub fn storage(&self) -> Arc<OutputStorage> {
        Arc::clone(self.rewards.storage())
    }

    /// Observation bytes of leaf `leaf`, `[N, A, leaf bytes]`.
    #[must_use]
    pub fn observation(&self, leaf: usize) -> Option<&[u8]> {
        self.observations.get(leaf).map(|b| &**b)
    }

    /// Final observations of episodes that ended in the last call, leaf
    /// `leaf`, `[N, A, leaf bytes]`. Only in `SameStep` auto-reset mode;
    /// rows of instances whose episode did not end are stale.
    #[must_use]
    pub fn terminal_observation(&self, leaf: usize) -> Option<&[u8]> {
        self.terminal_observations.get(leaf).map(|b| &**b)
    }

    /// Invalid agent events of the last call, `[N, A]`.
    #[cfg(any(test, feature = "debug-checks"))]
    pub(crate) fn invalid_agent_event(&self) -> &[Bool8] {
        &self.invalid_agent_event
    }

    /// Rewards of the last call, `[N, A]`.
    #[must_use]
    pub fn rewards(&self) -> &[f32] {
        &self.rewards
    }

    /// Per-agent terminated flags, `[N, A]`.
    #[must_use]
    pub fn agent_terminated(&self) -> &[Bool8] {
        &self.agent_terminated
    }

    /// Per-agent truncated flags, `[N, A]`.
    #[must_use]
    pub fn agent_truncated(&self) -> &[Bool8] {
        &self.agent_truncated
    }

    /// Alive mask, `[N, A]`.
    #[must_use]
    pub fn alive(&self) -> &[Bool8] {
        &self.alive
    }

    /// Acting mask, `[N, A]`.
    #[must_use]
    pub fn acting(&self) -> &[Bool8] {
        &self.acting
    }

    /// Slot generations, `[N, A]`.
    #[must_use]
    pub fn generations(&self) -> &[u32] {
        &self.generations
    }

    /// Episode terminated flags, `[N]`.
    #[must_use]
    pub fn episode_terminated(&self) -> &[Bool8] {
        &self.episode_terminated
    }

    /// Episode truncated flags, `[N]`.
    #[must_use]
    pub fn episode_truncated(&self) -> &[Bool8] {
        &self.episode_truncated
    }

    /// Return of episodes that ended in the last call, `[N]`: the sum of the
    /// per-agent returns, which stay available per agent in the record
    /// header (`AgentRow::episode_return`) for adapters. Zero elsewhere. In
    /// `SameStep` mode the header row already belongs to the new episode
    /// when the call returns, so read the ended episode's return here.
    #[must_use]
    pub fn episode_return(&self) -> &[f32] {
        &self.episode_return
    }

    /// Illegal actions replaced in episodes that ended in the last call
    /// (sum over agents), `[N]`; zero elsewhere.
    #[must_use]
    pub fn episode_illegal_actions(&self) -> &[f32] {
        &self.episode_illegal_actions
    }

    /// Invalid agent events in episodes that ended in the last call (sum
    /// over agents), `[N]`; zero elsewhere. They were never applied: spawns
    /// on a slot that was alive or died in the same step (SPEC 9.1), and
    /// deaths of agents that were already dead.
    #[must_use]
    pub fn episode_invalid_agent_events(&self) -> &[f32] {
        &self.episode_invalid_agent_events
    }

    /// Action masks for the current state, `[N, A, mask length]`: one entry
    /// per choice of each discrete head. Empty for pools without masks.
    #[must_use]
    pub fn masks(&self) -> &[Bool8] {
        &self.masks
    }

    /// Length of episodes that ended in the last call, `[N]`; zero elsewhere.
    #[must_use]
    pub fn episode_length(&self) -> &[f32] {
        &self.episode_length
    }
}

/// Mutable views over the rows of a range of instances in every output
/// buffer, indexed from the first instance of the range. Kernels write
/// through a view, so a chunk of instances can be processed without access
/// to the rest of the pool (SPEC 5.1 chunks).
#[derive(Debug)]
pub(crate) struct OutputsView<'a> {
    pub(crate) agents: usize,
    pub(crate) mask_len: usize,
    pub(crate) leaf_bytes: &'a [usize],
    pub(crate) observations: [&'a mut [u8]; MAX_LEAVES],
    pub(crate) terminal_observations: [&'a mut [u8]; MAX_LEAVES],
    pub(crate) rewards: &'a mut [f32],
    pub(crate) agent_terminated: &'a mut [Bool8],
    pub(crate) agent_truncated: &'a mut [Bool8],
    pub(crate) died: &'a mut [Bool8],
    pub(crate) spawned: &'a mut [Bool8],
    pub(crate) invalid_agent_event: &'a mut [Bool8],
    pub(crate) alive: &'a mut [Bool8],
    pub(crate) acting: &'a mut [Bool8],
    pub(crate) generations: &'a mut [u32],
    pub(crate) episode_terminated: &'a mut [Bool8],
    pub(crate) episode_truncated: &'a mut [Bool8],
    pub(crate) episode_return: &'a mut [f32],
    pub(crate) episode_length: &'a mut [f32],
    pub(crate) episode_illegal_actions: &'a mut [f32],
    pub(crate) episode_invalid_agent_events: &'a mut [f32],
    pub(crate) masks: &'a mut [Bool8],
}

impl Outputs {
    /// A view over every instance. This is the only way to write the
    /// outputs, so it starts a new write generation first (storage
    /// invariant 4).
    pub(crate) fn view_all(&mut self) -> OutputsView<'_> {
        self.rewards.storage().begin_write();
        let mut observations: [&mut [u8]; MAX_LEAVES] = Default::default();
        for (slot, buf) in observations.iter_mut().zip(self.observations.iter_mut()) {
            *slot = &mut **buf;
        }
        let mut terminal_observations: [&mut [u8]; MAX_LEAVES] = Default::default();
        for (slot, buf) in terminal_observations
            .iter_mut()
            .zip(self.terminal_observations.iter_mut())
        {
            *slot = &mut **buf;
        }
        OutputsView {
            agents: self.agents,
            mask_len: self.mask_len,
            leaf_bytes: &self.leaf_bytes,
            observations,
            terminal_observations,
            rewards: &mut self.rewards,
            agent_terminated: &mut self.agent_terminated,
            agent_truncated: &mut self.agent_truncated,
            died: &mut self.died,
            spawned: &mut self.spawned,
            invalid_agent_event: &mut self.invalid_agent_event,
            alive: &mut self.alive,
            acting: &mut self.acting,
            generations: &mut self.generations,
            episode_terminated: &mut self.episode_terminated,
            episode_truncated: &mut self.episode_truncated,
            episode_return: &mut self.episode_return,
            episode_length: &mut self.episode_length,
            episode_illegal_actions: &mut self.episode_illegal_actions,
            episode_invalid_agent_events: &mut self.episode_invalid_agent_events,
            masks: &mut self.masks,
        }
    }
}
