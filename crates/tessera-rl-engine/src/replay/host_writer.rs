// SPDX-License-Identifier: MIT OR Apache-2.0

//! The recorder of a host pool (ADR-0072, ADR-0069 item 8, ADR-0070 item 9).
//!
//! Host pools have no snapshots yet (ADR-0073), so a host file is
//! `from_reset_only`: the recorder itself resets every instance when it
//! starts, with the caller's `Reseed`, and that reset is the first event.
//! Every `Reset` and `Step` event carries the outputs hash of the call; a
//! refusing host's `Step` also carries its refusal rows; a notifying host's
//! rejected call is a `Rejected` event. There are no keyframes or
//! checkpoints.

use std::io::Write;

use tessera_rl_core::HostEnvironment;
use tessera_rl_core::encoding::{HashSink, Sink, checksum};

use crate::error::EngineError;
use crate::host::HostPool;
use crate::outputs::Outputs;
use crate::persist::{host_identity, write_shared_header};
use crate::pool::Reseed;
use crate::validate::ActionInput;

use super::ReplayError;
use super::format::{
    END_BODY_LEN, EventTag, FLAG_FROM_RESET_ONLY, FLAG_OUTPUTS_HASH, IDS_ALL, REPLAY_MAGIC,
    push_ids, push_len,
};
use super::outputs_hash::{outputs_hash, refusal_rows};
use super::writer::{frame, push_reseed};

/// Records the calls on a [`HostPool`] into a `from_reset_only` replay
/// file written to `W`.
///
/// A call the pool refuses without poisoning is not logged, except a
/// whole-call rejection of a host with `NOTIFIES_REJECTION`, which is a
/// `Rejected` event. A call that poisons the pool is logged, followed by a
/// `Poisoned` event; the recorder then refuses every call.
pub struct HostRecorder<'p, E: HostEnvironment, const A: usize, W> {
    pool: &'p mut HostPool<E, A>,
    out: W,
    body: Vec<u8>,
    sum: HashSink,
    events: u64,
    closed: bool,
}

impl<E: HostEnvironment, const A: usize, W> core::fmt::Debug for HostRecorder<'_, E, A, W> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostRecorder")
            .field("events", &self.events)
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl<'p, E: HostEnvironment, const A: usize, W: Write> HostRecorder<'p, E, A, W> {
    /// Starts a recording: writes the header, then resets every instance with
    /// `reseed` and logs that reset as the first event. Recording therefore
    /// always starts from a reset; the pool's earlier state does not matter.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Engine`] if the reset fails (a poisoning reset is
    /// logged and closes the recording), or [`ReplayError::Io`].
    pub fn new(
        pool: &'p mut HostPool<E, A>,
        reseed: Reseed<'_>,
        out: W,
    ) -> Result<Self, ReplayError> {
        let header = header_bytes(pool)?;
        let mut recorder = Self {
            pool,
            out,
            body: Vec::new(),
            sum: HashSink::new(),
            events: 0,
            closed: false,
        };
        recorder.out.write_all(&header)?;
        recorder.reset_inner(None, reseed)?;
        Ok(recorder)
    }

    /// The recorded pool, read-only.
    #[must_use]
    pub fn pool(&self) -> &HostPool<E, A> {
        self.pool
    }

    /// [`HostPool::reset_all`], logged with its outputs hash.
    ///
    /// # Errors
    ///
    /// The pool's error as [`ReplayError::Engine`], [`ReplayError::Io`], or
    /// [`ReplayError::RecorderClosed`] after a poison.
    pub fn reset_all(&mut self, reseed: Reseed<'_>) -> Result<&Outputs, ReplayError> {
        self.reset_inner(None, reseed)
    }

    /// [`HostPool::reset`], logged with the outputs hash of `ids`.
    ///
    /// # Errors
    ///
    /// As [`HostRecorder::reset_all`].
    pub fn reset(&mut self, ids: &[u32], reseed: Reseed<'_>) -> Result<&Outputs, ReplayError> {
        self.reset_inner(Some(ids), reseed)
    }

    /// [`HostPool::step`], logged with its outputs hash and, for a refusing
    /// host, its refusal rows; a notifying host's rejection is logged as a
    /// `Rejected` event.
    ///
    /// # Errors
    ///
    /// As [`HostRecorder::reset_all`].
    pub fn step(&mut self, actions: ActionInput<'_>) -> Result<&Outputs, ReplayError> {
        self.open()?;
        let mut joint = Vec::new();
        push_len(&mut joint, actions.discrete.len())?;
        for a in actions.discrete {
            joint.extend_from_slice(&a.to_le_bytes());
        }
        push_len(&mut joint, actions.continuous.len())?;
        for a in actions.continuous {
            joint.extend_from_slice(&a.to_bits().to_le_bytes());
        }
        let result = self.pool.step(actions).map(|_| ());
        match result {
            Ok(()) => {
                let n = self.num_envs();
                let out = self.pool.outputs()?;
                let hash = outputs_hash(out, n, 0..n, E::REFUSES_AGENTS);
                let rows = if E::REFUSES_AGENTS {
                    refusal_rows(out, n)
                } else {
                    Vec::new()
                };
                self.body.clear();
                self.body.push(IDS_ALL);
                self.body.extend_from_slice(&joint);
                self.body.extend_from_slice(&hash.to_le_bytes());
                if E::REFUSES_AGENTS {
                    push_len(&mut self.body, rows.len())?;
                    for row in rows {
                        for word in [row.instance, row.slot, row.code] {
                            self.body.extend_from_slice(&word.to_le_bytes());
                        }
                    }
                }
                self.emit_body(EventTag::Step)?;
            }
            Err(e) if self.pool.is_poisoned() => {
                self.body.clear();
                self.body.push(IDS_ALL);
                self.body.extend_from_slice(&joint);
                // No outputs exist on a poisoned pool: a zero hash and no
                // rows, never compared (the replay expects the poison).
                self.body.extend_from_slice(&0u64.to_le_bytes());
                if E::REFUSES_AGENTS {
                    push_len(&mut self.body, 0)?;
                }
                return Err(self.poisoned(EventTag::Step, e));
            }
            Err(EngineError::ActionRejected {
                instance,
                code,
                reason,
            }) => {
                if E::NOTIFIES_REJECTION {
                    self.body.clear();
                    self.body.extend_from_slice(&joint);
                    self.body.extend_from_slice(&instance.to_le_bytes());
                    self.body.extend_from_slice(&code.to_le_bytes());
                    self.emit_body(EventTag::Rejected)?;
                }
                return Err(EngineError::ActionRejected {
                    instance,
                    code,
                    reason,
                }
                .into());
            }
            Err(e) => return Err(e.into()),
        }
        Ok(self.pool.outputs()?)
    }

    /// Writes the end event and returns the writer, flushed. There is no
    /// final checkpoint: host pools have no state hash yet.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Io`].
    pub fn finish(mut self) -> Result<W, ReplayError> {
        let mut end = [0u8; 16];
        let (count, sum) = end.split_at_mut(8);
        count.copy_from_slice(&self.events.to_le_bytes());
        sum.copy_from_slice(&self.sum.digest().to_le_bytes());
        self.out.write_all(&frame(EventTag::End, END_BODY_LEN))?;
        self.out.write_all(&end)?;
        self.out.flush()?;
        Ok(self.out)
    }

    fn num_envs(&self) -> usize {
        usize::try_from(self.pool.options().num_envs.get()).unwrap_or(0)
    }

    fn open(&self) -> Result<(), ReplayError> {
        if self.closed {
            Err(ReplayError::RecorderClosed)
        } else {
            Ok(())
        }
    }

    fn reset_inner(
        &mut self,
        ids: Option<&[u32]>,
        reseed: Reseed<'_>,
    ) -> Result<&Outputs, ReplayError> {
        self.open()?;
        self.body.clear();
        push_ids(&mut self.body, ids)?;
        push_reseed(&mut self.body, reseed)?;
        let result = match ids {
            None => self.pool.reset_all(reseed).map(|_| ()),
            Some(ids) => self.pool.reset(ids, reseed).map(|_| ()),
        };
        match result {
            Ok(()) => {
                let n = self.num_envs();
                let out = self.pool.outputs()?;
                let hash = match ids {
                    None => outputs_hash(out, n, 0..n, E::REFUSES_AGENTS),
                    Some(ids) => outputs_hash(
                        out,
                        n,
                        ids.iter().filter_map(|&id| usize::try_from(id).ok()),
                        E::REFUSES_AGENTS,
                    ),
                };
                self.body.extend_from_slice(&hash.to_le_bytes());
                self.emit_body(EventTag::Reset)?;
                Ok(self.pool.outputs()?)
            }
            Err(e) if self.pool.is_poisoned() => {
                self.body.extend_from_slice(&0u64.to_le_bytes());
                Err(self.poisoned(EventTag::Reset, e))
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Logs the poisoning call in `self.body` and the `Poisoned` marker, and
    /// closes the recording. A write error replaces the pool's error.
    fn poisoned(&mut self, tag: EventTag, e: EngineError) -> ReplayError {
        self.closed = true;
        if let Err(io) = self.emit_body(tag) {
            return io;
        }
        self.body.clear();
        if let Err(io) = self.emit_body(EventTag::Poisoned) {
            return io;
        }
        e.into()
    }

    fn emit_body(&mut self, tag: EventTag) -> Result<(), ReplayError> {
        let len = u32::try_from(self.body.len()).map_err(|_| ReplayError::Unsupported {
            feature: "an event body over u32::MAX bytes",
        })?;
        let frame = frame(tag, len);
        self.out.write_all(&frame)?;
        self.out.write_all(&self.body)?;
        self.sum.write(&frame);
        self.sum.write(&self.body);
        self.events = self.events.checked_add(1).ok_or(ReplayError::Unsupported {
            feature: "more than u64::MAX events",
        })?;
        Ok(())
    }
}

/// The header of a host file: the shared ADR-0011 fields with the host
/// identity of `crate::persist::host_identity`, an empty config, the action
/// layout, and the host fields.
fn header_bytes<E: HostEnvironment, const A: usize>(
    pool: &HostPool<E, A>,
) -> Result<Vec<u8>, ReplayError> {
    let id = host_identity(E::ENV_ID, E::ENV_VERSION, E::MASKS, pool.options());
    let mut out = Vec::new();
    write_shared_header(
        &mut out,
        REPLAY_MAGIC,
        &id,
        pool.options().num_envs.get(),
        env!("CARGO_PKG_VERSION"),
        &[],
    )?;
    push_len(&mut out, 0)?; // no canonical config for a host
    let (discrete, continuous) = pool.action_heads();
    push_len(&mut out, A)?;
    push_len(&mut out, discrete)?;
    push_len(&mut out, continuous)?;
    out.extend_from_slice(&[
        FLAG_FROM_RESET_ONLY | FLAG_OUTPUTS_HASH,
        if E::REFUSES_AGENTS { 2 } else { 1 },
        u8::from(E::NOTIFIES_REJECTION),
        u8::from(E::REFUSES_AGENTS),
    ]);
    let sum = checksum(&out);
    out.extend_from_slice(&sum.to_le_bytes());
    Ok(out)
}
