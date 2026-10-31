// SPDX-License-Identifier: MIT OR Apache-2.0

//! Output buffers (SPEC 5.2), allocated once at construction.
//!
//! Per-agent buffers are `[N, A]` row-major; per-instance buffers `[N]`.
//! Observations are one byte buffer per leaf, `[N, A, leaf bytes]`, holding
//! little-endian elements of the leaf dtype. Results borrowed through
//! `&self` stay valid until the next call that takes `&mut self` (the
//! borrow window of ADR-0003 for Sync and Inline modes).

use tessera_rl_core::Bool8;

use crate::layout::Leaf;

/// Output buffers of a pool.
#[derive(Debug)]
pub struct Outputs {
    pub(crate) agents: usize,
    pub(crate) observations: Vec<Vec<u8>>,
    pub(crate) terminal_observations: Vec<Vec<u8>>,
    pub(crate) leaf_bytes: Vec<usize>,
    pub(crate) rewards: Vec<f32>,
    pub(crate) agent_terminated: Vec<Bool8>,
    pub(crate) agent_truncated: Vec<Bool8>,
    pub(crate) died: Vec<Bool8>,
    pub(crate) spawned: Vec<Bool8>,
    /// Spawns rejected in the last call (SPEC 9.1), `[N, A]`.
    pub(crate) invalid_spawn: Vec<Bool8>,
    pub(crate) alive: Vec<Bool8>,
    pub(crate) acting: Vec<Bool8>,
    pub(crate) generations: Vec<u32>,
    pub(crate) episode_terminated: Vec<Bool8>,
    pub(crate) episode_truncated: Vec<Bool8>,
    pub(crate) episode_return: Vec<f32>,
    pub(crate) episode_length: Vec<f32>,
    pub(crate) episode_illegal_actions: Vec<f32>,
    pub(crate) episode_invalid_spawns: Vec<f32>,
    pub(crate) masks: Vec<tessera_rl_core::Bool8>,
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
        let observations = leaves
            .iter()
            .map(|leaf| rows.checked_mul(leaf.bytes).map(|len| vec![0u8; len]))
            .collect::<Option<Vec<_>>>()?;
        let terminal_observations = if terminal {
            observations.clone()
        } else {
            Vec::new()
        };
        Some(Self {
            agents,
            observations,
            terminal_observations,
            leaf_bytes: leaves.iter().map(|l| l.bytes).collect(),
            rewards: vec![0.0; rows],
            agent_terminated: vec![Bool8::FALSE; rows],
            agent_truncated: vec![Bool8::FALSE; rows],
            died: vec![Bool8::FALSE; rows],
            spawned: vec![Bool8::FALSE; rows],
            invalid_spawn: vec![Bool8::FALSE; rows],
            alive: vec![Bool8::FALSE; rows],
            acting: vec![Bool8::FALSE; rows],
            generations: vec![0; rows],
            episode_terminated: vec![Bool8::FALSE; num_envs],
            episode_truncated: vec![Bool8::FALSE; num_envs],
            episode_return: vec![0.0; num_envs],
            episode_length: vec![0.0; num_envs],
            episode_illegal_actions: vec![0.0; num_envs],
            episode_invalid_spawns: vec![0.0; num_envs],
            masks: vec![Bool8::FALSE; rows.checked_mul(mask_len)?],
            mask_len,
        })
    }

    /// Observation bytes of leaf `leaf`, `[N, A, leaf bytes]`.
    #[must_use]
    pub fn observation(&self, leaf: usize) -> Option<&[u8]> {
        self.observations.get(leaf).map(Vec::as_slice)
    }

    /// Final observations of episodes that ended in the last call, leaf
    /// `leaf`, `[N, A, leaf bytes]`. Only in `SameStep` auto-reset mode;
    /// rows of instances whose episode did not end are stale.
    #[must_use]
    pub fn terminal_observation(&self, leaf: usize) -> Option<&[u8]> {
        self.terminal_observations.get(leaf).map(Vec::as_slice)
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

    /// Spawns not applied in episodes that ended in the last call, because
    /// their slot was alive or died in the same step (SPEC 9.1; sum over
    /// agents), `[N]`; zero elsewhere.
    #[must_use]
    pub fn episode_invalid_spawns(&self) -> &[f32] {
        &self.episode_invalid_spawns
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
