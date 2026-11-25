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
use crate::error::EngineError;

use crate::layout::{Leaf, MAX_LEAVES, SetSpans};

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
    /// The observation's sets, whose absent item rows the observation pass
    /// zeroes (ADR-0082 point 3); none unless the pool sets them.
    pub(crate) sets: SetSpans,
    /// Host pools only (ADR-0065): `None` for every other pool, whose
    /// storage layout is unchanged.
    host: Option<HostBufs>,
    /// The privileged state leaves (ADR-0079 item 7), `[N, leaf bytes]`
    /// each, laid out after every other buffer; empty unless the pool
    /// opted in.
    privileged: Vec<Buf<u8>>,
    /// Bytes per instance of each privileged leaf.
    privileged_bytes: Vec<usize>,
    /// Set by every [`Outputs::view_all`]: this engine call has started a
    /// write generation, so the privileged rows may be written in it too.
    written: bool,
}

/// The sizes of a layout besides its leaves.
#[derive(Debug, Clone, Copy)]
struct Parts {
    num_envs: usize,
    agents: usize,
    terminal: bool,
    mask_len: usize,
    /// `Some(refuses)` for a host pool.
    host: Option<bool>,
}

/// The privileged state rows of one engine call (ADR-0079 item 7), handed
/// out only after [`Outputs::view_all`] started that call's write
/// generation (storage invariant 4).
#[derive(Debug)]
pub(crate) struct PrivilegedRows<'a> {
    /// One buffer per leaf, `[N, leaf bytes]`.
    pub(crate) leaves: &'a mut [Buf<u8>],
    /// Bytes per instance of each leaf.
    pub(crate) bytes: &'a [usize],
}

/// The host-only per-instance outputs (ADR-0065 item 1).
#[derive(Debug)]
struct HostBufs {
    aborted: Buf<Bool8>,
    abort_code: Buf<u32>,
    pending_reset: Buf<Bool8>,
    /// `[N, A]` for a refusing host pool, empty otherwise (ADR-0070).
    refused: Buf<Bool8>,
    /// `[N, A]` for a refusing host pool, empty otherwise (ADR-0070).
    refuse_code: Buf<u32>,
}

impl Outputs {
    pub(crate) fn new(
        num_envs: usize,
        agents: usize,
        leaves: &[Leaf],
        terminal: bool,
        mask_len: usize,
    ) -> Result<Self, EngineError> {
        Self::new_with(num_envs, agents, leaves, terminal, mask_len, &[])
    }

    /// As [`Outputs::new`], plus one privileged state region per leaf of
    /// `privileged`, `[num_envs, leaf bytes]` (ADR-0079 item 7), laid out
    /// after every other buffer, so the other offsets are those of a pool
    /// without them.
    pub(crate) fn new_with(
        num_envs: usize,
        agents: usize,
        leaves: &[Leaf],
        terminal: bool,
        mask_len: usize,
        privileged: &[Leaf],
    ) -> Result<Self, EngineError> {
        let b = StorageBuilder::new()?;
        let parts = Parts {
            num_envs,
            agents,
            terminal,
            mask_len,
            host: None,
        };
        Self::lay_out(b, &parts, leaves, privileged).ok_or(EngineError::Options {
            option: "num_envs",
            reason: "buffers too large",
        })
    }

    /// As [`Outputs::new`], plus the host-only rows `aborted`, `abort_code`
    /// and `pending_reset` (ADR-0065), and for a refusing environment
    /// `refused` and `refuse_code` (ADR-0070), laid out after every other
    /// buffer.
    #[cfg(feature = "host")]
    pub(crate) fn new_host(
        num_envs: usize,
        agents: usize,
        leaves: &[Leaf],
        terminal: bool,
        mask_len: usize,
        refuses: bool,
    ) -> Result<Self, EngineError> {
        let b = StorageBuilder::new()?;
        let parts = Parts {
            num_envs,
            agents,
            terminal,
            mask_len,
            host: Some(refuses),
        };
        Self::lay_out(b, &parts, leaves, &[]).ok_or(EngineError::Options {
            option: "num_envs",
            reason: "buffers too large",
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one reservation and one take per output region, in layout order"
    )]
    fn lay_out(
        mut b: StorageBuilder,
        parts: &Parts,
        leaves: &[Leaf],
        privileged: &[Leaf],
    ) -> Option<Self> {
        let Parts {
            num_envs,
            agents,
            terminal,
            mask_len,
            host,
        } = *parts;
        let rows = num_envs.checked_mul(agents)?;
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
        // Last, so the other buffers' offsets are those of a pool without
        // them.
        let host = match host {
            Some(refuses) => {
                // Without refusals the two buffers are empty (ADR-0070).
                let refusal_rows = if refuses { rows } else { 0 };
                Some((
                    b.reserve::<Bool8>(OutputField::Aborted, num_envs)?,
                    b.reserve::<u32>(OutputField::AbortCode, num_envs)?,
                    b.reserve::<Bool8>(OutputField::PendingReset, num_envs)?,
                    b.reserve::<Bool8>(OutputField::Refused, refusal_rows)?,
                    b.reserve::<u32>(OutputField::RefuseCode, refusal_rows)?,
                ))
            }
            None => None,
        };
        // After the host rows: a pool without the opt-in keeps its layout.
        let privileged_slots = privileged
            .iter()
            .enumerate()
            .map(|(n, leaf)| {
                b.reserve::<u8>(
                    OutputField::PrivilegedState(n),
                    num_envs.checked_mul(leaf.bytes)?,
                )
            })
            .collect::<Option<Vec<_>>>()?;
        let mut r = b.finish()?;
        let host = match host {
            Some((aborted, abort_code, pending_reset, refused, refuse_code)) => Some(HostBufs {
                aborted: r.take(aborted)?,
                abort_code: r.take(abort_code)?,
                pending_reset: r.take(pending_reset)?,
                refused: r.take(refused)?,
                refuse_code: r.take(refuse_code)?,
            }),
            None => None,
        };
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
            sets: SetSpans::default(),
            host,
            privileged: privileged_slots
                .into_iter()
                .map(|slot| r.take(slot))
                .collect::<Option<_>>()?,
            privileged_bytes: privileged.iter().map(|l| l.bytes).collect(),
            written: false,
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

    /// Privileged state bytes of leaf `leaf`, `[N, leaf bytes]`, one row
    /// per instance (ADR-0079 item 7); `None` without the opt-in. A pending
    /// instance's row is zero. Valid as the observations are: until the
    /// next call that takes `&mut self` (ADR-0003).
    #[must_use]
    pub fn privileged_state(&self, leaf: usize) -> Option<&[u8]> {
        self.privileged.get(leaf).map(|b| &**b)
    }

    /// The observation leaves, as laid out.
    #[must_use]
    pub(crate) fn leaf_layout(&self) -> Vec<Leaf> {
        self.leaf_bytes.iter().map(|&bytes| Leaf { bytes }).collect()
    }

    /// The privileged leaves, as laid out (empty without the opt-in).
    #[must_use]
    pub(crate) fn privileged_leaves(&self) -> Vec<Leaf> {
        self.privileged_bytes
            .iter()
            .map(|&bytes| Leaf { bytes })
            .collect()
    }

    /// The privileged rows, if a [`Outputs::view_all`] started a write
    /// generation since the last take: only then may they be written
    /// (storage invariant 4). Clears that mark.
    pub(crate) fn take_privileged(&mut self) -> Option<PrivilegedRows<'_>> {
        if !core::mem::replace(&mut self.written, false) {
            return None;
        }
        Some(PrivilegedRows {
            leaves: &mut self.privileged,
            bytes: &self.privileged_bytes,
        })
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

    /// Host pools only (ADR-0065): instances whose episode the last call
    /// aborted, `[N]`. Empty for other pools. An aborted instance's
    /// observation rows are zeros, which may lie outside the observation
    /// space, and its masks are false. `aborted` alone never means pending;
    /// read [`Outputs::pending_reset`].
    #[must_use]
    pub fn aborted(&self) -> &[Bool8] {
        self.host.as_ref().map_or(&[], |h| &*h.aborted)
    }

    /// Host pools only (ADR-0065): the abort codes of the last call, `[N]`,
    /// 0 where `aborted` is false. Empty for other pools.
    #[must_use]
    pub fn abort_code(&self) -> &[u32] {
        self.host.as_ref().map_or(&[], |h| &*h.abort_code)
    }

    /// Host pools only (ADR-0065): whether each instance is pending a reset,
    /// `[N]`, the engine's own state: true for every instance at
    /// construction, then that of each instance after the last call that
    /// ran it. Empty for other pools.
    #[must_use]
    pub fn pending_reset(&self) -> &[Bool8] {
        self.host.as_ref().map_or(&[], |h| &*h.pending_reset)
    }

    /// Refusing host pools only (ADR-0070): agents whose action the last
    /// call refused, `[N, A]`. After every call that is not rejected, true
    /// only for an agent refused in that call; a rejected call leaves the
    /// previous call's rows. Empty for other pools.
    #[must_use]
    pub fn refused(&self) -> &[Bool8] {
        self.host.as_ref().map_or(&[], |h| &*h.refused)
    }

    /// Refusing host pools only (ADR-0070): the refuse codes of the last
    /// call, `[N, A]`, 0 where `refused` is false. Empty for other pools.
    #[must_use]
    pub fn refuse_code(&self) -> &[u32] {
        self.host.as_ref().map_or(&[], |h| &*h.refuse_code)
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
    /// The observation's sets (ADR-0082 point 3).
    pub(crate) sets: SetSpans,
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
    /// Host pools only (ADR-0065); empty for other pools.
    pub(crate) aborted: &'a mut [Bool8],
    /// Host pools only (ADR-0065); empty for other pools.
    pub(crate) abort_code: &'a mut [u32],
    /// Host pools only (ADR-0065); empty for other pools.
    pub(crate) pending_reset: &'a mut [Bool8],
    /// Refusing host pools only (ADR-0070), `[N, A]`; empty otherwise.
    pub(crate) refused: &'a mut [Bool8],
    /// Refusing host pools only (ADR-0070), `[N, A]`; empty otherwise.
    pub(crate) refuse_code: &'a mut [u32],
    /// The transform parts (ADR-0078); empty for a pool without transforms.
    pub(crate) transform: crate::transform::rows::TransformRows<'a>,
    /// Per-instance episode-statistics partials (ADR-0081 item 3); empty
    /// outside a step pass or while the subsystem is frozen.
    pub(crate) episode_partials: &'a mut [crate::adaptive::EpisodePartial],
    /// Per-instance aborted-episode counts of a host pool (ADR-0081 item
    /// 8); empty otherwise.
    pub(crate) episode_aborts: &'a mut [u64],
    /// Per instance, the league outcome written at episode end (ADR-0088
    /// item 4.1); empty without a league.
    pub(crate) league_outcomes: &'a mut [i8],
    /// The league's opponent slots as a bit mask; 0 without a league.
    pub(crate) league_slots: u64,
    /// The pool index of the view's first instance (0 for a pool-wide
    /// view): pool-level per-slot data, such as curriculum targets
    /// (ADR-0081 item 5.3), is indexed `base + i`.
    pub(crate) base: usize,
}

impl OutputsView<'_> {
    /// A view over no output rows, for a dispatch that writes no outputs
    /// (the host validation pass, ADR-0065 item 2): it starts no write
    /// generation, so borrowed outputs stay current.
    #[cfg(feature = "host")]
    pub(crate) fn empty(agents: usize) -> Self {
        OutputsView {
            agents,
            mask_len: 0,
            leaf_bytes: &[],
            sets: SetSpans::default(),
            observations: Default::default(),
            terminal_observations: Default::default(),
            rewards: &mut [],
            agent_terminated: &mut [],
            agent_truncated: &mut [],
            died: &mut [],
            spawned: &mut [],
            invalid_agent_event: &mut [],
            alive: &mut [],
            acting: &mut [],
            generations: &mut [],
            episode_terminated: &mut [],
            episode_truncated: &mut [],
            episode_return: &mut [],
            episode_length: &mut [],
            episode_illegal_actions: &mut [],
            episode_invalid_agent_events: &mut [],
            masks: &mut [],
            aborted: &mut [],
            abort_code: &mut [],
            pending_reset: &mut [],
            refused: &mut [],
            refuse_code: &mut [],
            transform: crate::transform::rows::TransformRows::default(),
            episode_partials: &mut [],
            episode_aborts: &mut [],
            league_outcomes: &mut [],
            league_slots: 0,
            base: 0,
        }
    }
}

impl Outputs {
    /// A view over every instance. This is the only way to write the
    /// outputs, so it starts a new write generation first (storage
    /// invariant 4).
    pub(crate) fn view_all(&mut self) -> OutputsView<'_> {
        self.rewards.storage().begin_write();
        self.written = true;
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
        let (aborted, abort_code, pending_reset, refused, refuse_code) = match &mut self.host {
            Some(h) => (
                &mut *h.aborted,
                &mut *h.abort_code,
                &mut *h.pending_reset,
                &mut *h.refused,
                &mut *h.refuse_code,
            ),
            None => (
                &mut [][..],
                &mut [][..],
                &mut [][..],
                &mut [][..],
                &mut [][..],
            ),
        };
        OutputsView {
            agents: self.agents,
            mask_len: self.mask_len,
            leaf_bytes: &self.leaf_bytes,
            sets: self.sets,
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
            aborted,
            abort_code,
            pending_reset,
            refused,
            refuse_code,
            transform: crate::transform::rows::TransformRows::default(),
            episode_partials: &mut [],
            episode_aborts: &mut [],
            league_outcomes: &mut [],
            league_slots: 0,
            base: 0,
        }
    }
}
