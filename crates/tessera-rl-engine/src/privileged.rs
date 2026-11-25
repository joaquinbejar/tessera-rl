// SPDX-License-Identifier: MIT OR Apache-2.0

//! The privileged state buffer (ADR-0079 item 7): one row per instance of
//! `PrivilegedState::write_state`, for centralized critics.
//!
//! The rows are regions of the pool's shared output storage, laid out after
//! every other buffer, so they are outputs under ADR-0003 as the
//! observations are: the same borrow windows, the same write generation,
//! the same owned copies. Workers never see them. Every engine call that
//! writes the outputs (it starts a write generation through
//! `Outputs::view_all`: reset, step, restore, fork, import) rewrites every
//! row on the caller's thread when the call's passes are done, before it
//! returns. Each row is zeroed first (ADR-0058); a pending instance's row
//! stays zero. No new `unsafe`: the rows are written through their `Buf`s.
//!
//! The routine is chosen once, at the opt-in, as one `fn` pointer per pool
//! ([`PrivilegedFn`]); inside, every call to `write_state` is static.
//! Search never builds or reads the buffer (item 7.3): a search scratch
//! pool with it is refused.

use tessera_rl_core::io::ObsWriter;
use tessera_rl_core::space::Space;
use tessera_rl_core::{Environment, PrivilegedState, RecordLayout};

use crate::error::EngineError;
use crate::layout::{MAX_LEAVES, obs_leaves};
use crate::masks::MaskProvider;
use crate::options::AutoReset;
use crate::outputs::{Outputs, PrivilegedRows};
use crate::pool::{EnvPool, run_kernels};

/// The privileged pass of one call: every instance's row.
pub(crate) type PrivilegedFn<E> = fn(
    &<E as Environment>::Config,
    &[<E as Environment>::Header],
    &[<E as Environment>::State],
    &mut PrivilegedRows<'_>,
);

/// What the pool's privileged pass runs, chosen at the opt-in.
pub(crate) struct PrivilegedRoute<E: Environment> {
    /// The pass, called once per engine call that writes outputs.
    pub(crate) write: PrivilegedFn<E>,
    /// `PrivilegedState::state_space` of the pool's config.
    pub(crate) space: Space,
}

impl<E: Environment> core::fmt::Debug for PrivilegedRoute<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PrivilegedRoute")
            .field("space", &self.space)
            .finish_non_exhaustive()
    }
}

/// Writes every instance's row: zeroed, then `write_state` unless the
/// instance is pending a reset. No allocation: the leaf views live on the
/// stack.
pub(crate) fn write_rows<E: PrivilegedState>(
    cfg: &E::Config,
    headers: &[E::Header],
    states: &[E::State],
    rows: &mut PrivilegedRows<'_>,
) {
    for (i, (header, state)) in headers.iter().zip(states.iter()).enumerate() {
        let mut views: [&mut [u8]; MAX_LEAVES] = Default::default();
        let mut used = 0;
        for ((buf, &bytes), view) in rows
            .leaves
            .iter_mut()
            .zip(rows.bytes.iter())
            .zip(views.iter_mut())
        {
            let Some(start) = i.checked_mul(bytes) else {
                continue;
            };
            let Some(end) = start.checked_add(bytes) else {
                continue;
            };
            if let Some(row) = buf.get_mut(start..end) {
                row.fill(0);
                *view = row;
            }
            used += 1;
        }
        if header.engine().pending_reset.get() {
            continue;
        }
        if let Some(leaves) = views.get_mut(..used) {
            E::write_state(cfg, state, &mut ObsWriter::new(leaves));
        }
    }
}

impl<E: Environment, M: MaskProvider<E>> EnvPool<E, M> {
    /// Whether the pool has the privileged state buffer.
    #[must_use]
    pub const fn has_privileged_state(&self) -> bool {
        self.privileged.is_some()
    }

    /// The space of one instance's privileged row (`state_space`), if the
    /// pool has the buffer.
    #[must_use]
    pub fn privileged_state_space(&self) -> Option<&Space> {
        self.privileged.as_ref().map(|r| &r.space)
    }

    /// Writes the privileged rows after a call that wrote the outputs: a
    /// no-op without the opt-in, for a call that wrote no output, or on a
    /// poisoned pool, whose rows have no defined contents. A panic in
    /// `write_state` poisons the pool.
    pub(crate) fn finish_privileged(&mut self) -> Result<(), EngineError> {
        let Self {
            cfg,
            headers,
            states,
            outputs,
            poisoned,
            privileged,
            ..
        } = self;
        let Some(route) = privileged.as_ref() else {
            return Ok(());
        };
        let Some(mut rows) = outputs.take_privileged() else {
            return Ok(());
        };
        if *poisoned {
            return Ok(());
        }
        let write = route.write;
        run_kernels(poisoned, || write(cfg, headers, states, &mut rows))
    }
}

impl<E: PrivilegedState, M: MaskProvider<E>> EnvPool<E, M> {
    /// Adds the privileged state buffer (ADR-0079 item 7), before the first
    /// reset, once: one row per instance of `E::write_state`, `[N, state
    /// bytes]` per leaf of `E::state_space`, read with
    /// `Outputs::privileged_state`. It is written by every call that writes
    /// the observations, and a pending instance's row is zero. Sync and
    /// Inline modes only: Pipelined and Async refuse a pool with it.
    ///
    /// # Errors
    ///
    /// [`EngineError::PrivilegedState`] after the first reset, on a second
    /// call, or for a lane-wise state; [`EngineError::Space`] or
    /// [`EngineError::UnsupportedSpace`] for the space; [`EngineError`] for
    /// the output buffers.
    pub fn with_privileged_state(mut self) -> Result<Self, EngineError> {
        if self.privileged.is_some() {
            return Err(EngineError::PrivilegedState {
                reason: "the buffer is set once",
            });
        }
        if self.headers.iter().any(|h| !h.engine().fresh.get()) {
            return Err(EngineError::PrivilegedState {
                reason: "the buffer is set before the first reset",
            });
        }
        if <E::State as tessera_rl_core::EnvState>::LANES {
            return Err(EngineError::PrivilegedState {
                reason: "the buffer needs an Array-of-Structs state, not #[state(lanes)]",
            });
        }
        let space = E::state_space(&self.cfg)?;
        space.validate()?;
        let layout = obs_leaves(&space)?;
        if !layout.sets.as_slice().is_empty() {
            return Err(EngineError::PrivilegedState {
                reason: "the state space cannot hold a Set",
            });
        }
        let num_envs = self.headers.len();
        let mut outputs = Outputs::new_with(
            num_envs,
            E::MAX_AGENTS,
            &self.outputs.leaf_layout(),
            self.options.auto_reset == AutoReset::SameStep,
            self.outputs.mask_len,
            &layout.leaves,
        )?;
        outputs.sets = self.outputs.sets;
        self.outputs = outputs;
        self.privileged = Some(PrivilegedRoute {
            write: write_rows::<E>,
            space,
        });
        Ok(self)
    }
}
