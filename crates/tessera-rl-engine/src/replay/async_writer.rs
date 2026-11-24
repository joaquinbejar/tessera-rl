// SPDX-License-Identifier: MIT OR Apache-2.0

//! The recorder of an [`AsyncPool`] (ADR-0072 section 3, tag 8; ADR-0047).
//!
//! Every `send` is a `Step` with its ids, every `recv` a `Recv` with the ids
//! the token holds, in order; `reset` and `reset_all` are `Reset` events;
//! `restore` and `restore_all` are `StateWrite` events followed by a
//! keyframe of the records written. The file's header has the async flag.
//!
//! # Quiescent points
//!
//! A keyframe or a checkpoint reads every instance, so it is taken only when
//! no id is outstanding (sent, reset or restored and not yet received): the
//! pool is quiescent. A periodic keyframe or checkpoint falls due every
//! `keyframe_every` or `checkpoint_every` logged calls (sends, receives,
//! resets and state writes) and is written at the next quiescent point after
//! that, possibly several calls later; a recording that is never quiescent
//! has no periodic keyframes or checkpoints. The final checkpoint of
//! [`AsyncRecorder::finish`] is written only if the pool is quiescent then.
//! The keyframe after a state write holds the records written, not a
//! snapshot, so it needs no quiescence.
//!
//! # Determinism
//!
//! Batch membership and order depend on worker timing; an instance's
//! trajectory does not (ADR-0006). The file records the batches this run
//! observed; replay waits for them (see `Replay::run_async`).

use std::io::Write;

use tessera_rl_core::config::CanonicalConfig;
use tessera_rl_core::encoding::{Canonical, HashSink, Sink, encode_to_vec};
use tessera_rl_core::{AdaptiveValues, Persist, RecordLayout};

use crate::buffer::batch::BatchToken;
use crate::error::EngineError;
use crate::masks::PersistMask;
use crate::modes::async_mode::AsyncPool;
use crate::pool::Reseed;
use crate::record::{DecodedRecord, Record};
use crate::validate::ActionInput;

use super::ReplayError;
use super::format::{
    END_BODY_LEN, EventTag, FLAG_ASYNC, FLAG_OUTPUTS_HASH, StateWriteOp, push_ids, push_len,
};
use super::writer::{ReplayWriterOptions, frame, header_from, push_reseed};

/// Appends the rows of `id` that an outputs hash covers: every observation
/// leaf, the rewards (`f32` bits), the agent terminations and truncations
/// (one byte each). Shared by the recorder and the replay.
pub(super) fn token_rows<E, M>(token: &BatchToken<E, M>, id: u32, out: &mut Vec<u8>)
where
    E: tessera_rl_core::Environment,
    M: crate::masks::MaskProvider<E>,
{
    let mut leaf = 0;
    while let Some(obs) = token.observation(leaf, id) {
        out.extend_from_slice(obs);
        leaf += 1;
    }
    for r in token.rewards(id).unwrap_or_default() {
        out.extend_from_slice(&r.to_bits().to_le_bytes());
    }
    for flags in [token.agent_terminated(id), token.agent_truncated(id)] {
        out.extend(flags.unwrap_or_default().iter().map(|f| u8::from(f.get())));
    }
}

/// Records the calls on an [`AsyncPool`] into a replay file written to `W`.
///
/// Every method mirrors the pool method of the same name. A call the pool
/// refuses without poisoning is not logged. A call that poisons the pool (a
/// `recv` that reports a panicked step) is logged with no ids, followed by a
/// `Poisoned` event; the recorder then refuses every call.
pub struct AsyncRecorder<'p, E, M, W>
where
    E: Persist,
    M: PersistMask<E>,
{
    pool: &'p mut AsyncPool<E, M>,
    out: W,
    options: ReplayWriterOptions,
    body: Vec<u8>,
    sum: HashSink,
    events: u64,
    calls: u64,
    closed: bool,
    checkpointed: bool,
    /// Ids sent, reset or restored and not yet received.
    outstanding: usize,
    keyframe_due: bool,
    checkpoint_due: bool,
}

impl<E, M, W> core::fmt::Debug for AsyncRecorder<'_, E, M, W>
where
    E: Persist,
    M: PersistMask<E>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AsyncRecorder")
            .field("events", &self.events)
            .field("calls", &self.calls)
            .field("outstanding", &self.outstanding)
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl<'p, E, M, W> AsyncRecorder<'p, E, M, W>
where
    E: Persist,
    E::Config: CanonicalConfig + Send + Sync,
    E::Header: Send + Sync,
    E::State: Send + Sync,
    M: PersistMask<E> + Send + Sync,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    W: Write,
{
    /// Starts a recording on a quiescent pool: writes the header (with the
    /// async flag) and a keyframe of every instance.
    ///
    /// # Errors
    ///
    /// [`ReplayError::AdaptiveUnsupported`], [`ReplayError::Engine`] with
    /// [`EngineError::InFlight`] while an id is outstanding (or another
    /// snapshot error), a [`ReplayError::Persist`] for a header the pool
    /// cannot describe, or [`ReplayError::Io`].
    pub fn new(
        pool: &'p mut AsyncPool<E, M>,
        out: W,
        options: ReplayWriterOptions,
    ) -> Result<Self, ReplayError> {
        if <<E::Header as RecordLayout>::Adaptive as Canonical>::ENCODED_LEN != 0 {
            return Err(ReplayError::AdaptiveUnsupported);
        }
        let mut flags = FLAG_ASYNC;
        if options.outputs_hash {
            flags |= FLAG_OUTPUTS_HASH;
        }
        let (cfg, pool_options, heads) = pool.replay_parts();
        let header = header_from::<E, M>(cfg, pool_options, heads, flags)?;
        let mut recorder = Self {
            pool,
            out,
            options,
            body: Vec::new(),
            sum: HashSink::new(),
            events: 0,
            calls: 0,
            closed: false,
            checkpointed: false,
            outstanding: 0,
            keyframe_due: false,
            checkpoint_due: false,
        };
        // The keyframe first: it refuses a pool with ids outstanding before
        // anything is written.
        let keyframe = recorder.snapshot_keyframe()?;
        recorder.out.write_all(&header)?;
        recorder.body = keyframe;
        recorder.emit_body(EventTag::Keyframe)?;
        Ok(recorder)
    }

    /// [`AsyncPool::snapshot`]: reads one idle instance and logs nothing,
    /// for a caller that restores the record later through
    /// [`AsyncRecorder::restore`].
    ///
    /// # Errors
    ///
    /// The pool's error as [`ReplayError::Engine`], or
    /// [`ReplayError::RecorderClosed`].
    pub fn snapshot(&mut self, id: u32) -> Result<Record<E>, ReplayError> {
        self.open()?;
        Ok(self.pool.snapshot(id)?)
    }

    /// [`AsyncPool::send`], logged as a `Step` with its ids.
    ///
    /// # Errors
    ///
    /// The pool's error as [`ReplayError::Engine`], [`ReplayError::Io`], or
    /// [`ReplayError::RecorderClosed`] after a poison.
    pub fn send(&mut self, ids: &[u32], actions: ActionInput<'_>) -> Result<(), ReplayError> {
        self.open()?;
        self.pool.send(ids, actions)?;
        self.body.clear();
        push_ids(&mut self.body, Some(ids))?;
        push_len(&mut self.body, actions.discrete.len())?;
        for a in actions.discrete {
            self.body.extend_from_slice(&a.to_le_bytes());
        }
        push_len(&mut self.body, actions.continuous.len())?;
        for a in actions.continuous {
            self.body.extend_from_slice(&a.to_bits().to_le_bytes());
        }
        self.emit_body(EventTag::Step)?;
        self.outstanding = self.outstanding.saturating_add(ids.len());
        self.after_call()
    }

    /// [`AsyncPool::recv`], logged as a `Recv` with the ids of the token, in
    /// order, and their outputs hash when the options ask for it.
    ///
    /// # Errors
    ///
    /// As [`AsyncRecorder::send`]; a panicked step poisons the pool and
    /// closes the recording.
    pub fn recv(&mut self) -> Result<BatchToken<E, M>, ReplayError> {
        self.open()?;
        let token = match self.pool.recv() {
            Ok(token) => token,
            Err(e) if self.pool.is_poisoned() => {
                self.body.clear();
                push_ids(&mut self.body, Some(&[][..]))?;
                if self.options.outputs_hash {
                    self.body.extend_from_slice(&0u64.to_le_bytes());
                }
                return Err(self.poisoned(EventTag::Recv, e));
            }
            Err(e) => return Err(e.into()),
        };
        let ids: Vec<u32> = token.ids().collect();
        self.body.clear();
        push_ids(&mut self.body, Some(&ids))?;
        if self.options.outputs_hash {
            let mut rows = Vec::new();
            let mut sink = HashSink::new();
            for &id in &ids {
                rows.clear();
                token_rows(&token, id, &mut rows);
                sink.write(&rows);
            }
            self.body.extend_from_slice(&sink.digest().to_le_bytes());
        }
        self.emit_body(EventTag::Recv)?;
        self.outstanding = self.outstanding.saturating_sub(ids.len());
        self.after_call()?;
        Ok(token)
    }

    /// [`AsyncPool::reset_all`], logged; the token is returned as the pool
    /// returns it.
    ///
    /// # Errors
    ///
    /// As [`AsyncRecorder::send`].
    pub fn reset_all(&mut self, reseed: Reseed<'_>) -> Result<BatchToken<E, M>, ReplayError> {
        self.open()?;
        self.body.clear();
        push_ids(&mut self.body, None)?;
        push_reseed(&mut self.body, reseed)?;
        let token = match self.pool.reset_all(reseed) {
            Ok(token) => token,
            Err(e) if self.pool.is_poisoned() => {
                return Err(self.poisoned(EventTag::Reset, e));
            }
            Err(e) => return Err(e.into()),
        };
        self.emit_body(EventTag::Reset)?;
        self.outstanding = 0;
        self.after_call()?;
        Ok(token)
    }

    /// [`AsyncPool::reset`], logged; the ids are delivered by a later `recv`.
    ///
    /// # Errors
    ///
    /// As [`AsyncRecorder::send`].
    pub fn reset(&mut self, ids: &[u32], reseed: Reseed<'_>) -> Result<(), ReplayError> {
        self.open()?;
        self.body.clear();
        push_ids(&mut self.body, Some(ids))?;
        push_reseed(&mut self.body, reseed)?;
        match self.pool.reset(ids, reseed) {
            Ok(()) => {}
            Err(e) if self.pool.is_poisoned() => {
                return Err(self.poisoned(EventTag::Reset, e));
            }
            Err(e) => return Err(e.into()),
        }
        self.emit_body(EventTag::Reset)?;
        self.outstanding = self.outstanding.saturating_add(ids.len());
        self.after_call()
    }

    /// [`AsyncPool::restore`], logged with a keyframe of the records written.
    ///
    /// # Errors
    ///
    /// As [`AsyncRecorder::send`].
    pub fn restore(&mut self, ids: &[u32], records: &[Record<E>]) -> Result<(), ReplayError> {
        self.open()?;
        let result = self.pool.restore(ids, records);
        self.state_write(StateWriteOp::Restore, Some(ids), records, result)?;
        self.outstanding = self.outstanding.saturating_add(ids.len());
        self.after_call()
    }

    /// [`AsyncPool::restore_all`], logged with a keyframe of the records.
    ///
    /// # Errors
    ///
    /// As [`AsyncRecorder::send`].
    pub fn restore_all(&mut self, records: &[Record<E>]) -> Result<(), ReplayError> {
        self.open()?;
        let result = self.pool.restore_all(records);
        self.state_write(StateWriteOp::RestoreAll, None, records, result)?;
        self.outstanding = 0;
        self.after_call()
    }

    /// Writes the trailer: a final checkpoint when the pool is quiescent
    /// (and not poisoned, and the last event is not already one), then the
    /// end event. Returns the writer, flushed.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Io`], or [`ReplayError::Engine`] if a snapshot fails.
    pub fn finish(mut self) -> Result<W, ReplayError> {
        if !self.closed && !self.checkpointed && self.outstanding == 0 {
            self.checkpoint()?;
        }
        let mut end = [0u8; 16];
        let (count, sum) = end.split_at_mut(8);
        count.copy_from_slice(&self.events.to_le_bytes());
        sum.copy_from_slice(&self.sum.digest().to_le_bytes());
        self.out.write_all(&frame(EventTag::End, END_BODY_LEN))?;
        self.out.write_all(&end)?;
        self.out.flush()?;
        Ok(self.out)
    }

    fn open(&self) -> Result<(), ReplayError> {
        if self.closed {
            Err(ReplayError::RecorderClosed)
        } else {
            Ok(())
        }
    }

    /// Logs a state write that succeeded (with a keyframe of `records`) or
    /// poisoned the pool; a plain refusal logs nothing.
    fn state_write(
        &mut self,
        op: StateWriteOp,
        ids: Option<&[u32]>,
        records: &[Record<E>],
        result: Result<(), EngineError>,
    ) -> Result<(), ReplayError> {
        if result.is_err() && !self.pool.is_poisoned() {
            return result.map_err(Into::into);
        }
        self.body.clear();
        self.body.push(op as u8);
        push_ids(&mut self.body, ids)?;
        // An Async pool has no `fork_branched`: its branch counter is 0.
        self.body.extend_from_slice(&0u64.to_le_bytes());
        if let Err(e) = result {
            return Err(self.poisoned(EventTag::StateWrite, e));
        }
        self.emit_body(EventTag::StateWrite)?;
        self.body.clear();
        self.body.extend_from_slice(&0u64.to_le_bytes());
        push_ids(&mut self.body, ids)?;
        for record in records {
            self.body
                .extend_from_slice(&encode_to_vec(&DecodedRecord::<E>::from(*record)));
        }
        self.emit_body(EventTag::Keyframe)
    }

    /// Counts a logged call, marks the periodic keyframe and checkpoint due,
    /// and writes what is due if the pool is quiescent.
    fn after_call(&mut self) -> Result<(), ReplayError> {
        self.calls = self.calls.checked_add(1).ok_or(ReplayError::Unsupported {
            feature: "more than u64::MAX calls",
        })?;
        let calls = self.calls;
        let every = |n: u64| n != 0 && calls.is_multiple_of(n);
        self.keyframe_due |= every(self.options.keyframe_every);
        self.checkpoint_due |= every(self.options.checkpoint_every);
        if self.outstanding != 0 {
            return Ok(());
        }
        if self.keyframe_due {
            self.body = self.snapshot_keyframe()?;
            self.emit_body(EventTag::Keyframe)?;
            self.keyframe_due = false;
        }
        if self.checkpoint_due {
            self.checkpoint()?;
            self.checkpoint_due = false;
        }
        Ok(())
    }

    /// The body of a keyframe of every instance, from snapshots.
    fn snapshot_keyframe(&mut self) -> Result<Vec<u8>, ReplayError> {
        let mut body = Vec::new();
        body.extend_from_slice(&0u64.to_le_bytes());
        push_ids(&mut body, None)?;
        for id in 0..self.pool.num_envs() {
            let record = DecodedRecord::<E>::from(self.pool.snapshot(id)?);
            body.extend_from_slice(&encode_to_vec(&record));
        }
        Ok(body)
    }

    /// A checkpoint of every instance; the pool is quiescent.
    fn checkpoint(&mut self) -> Result<(), ReplayError> {
        self.body.clear();
        push_ids(&mut self.body, None)?;
        for id in 0..self.pool.num_envs() {
            let hash = self.pool.snapshot(id)?.hash();
            self.body.extend_from_slice(&hash.get().to_le_bytes());
        }
        self.emit_body(EventTag::Checkpoint)
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
        self.checkpointed = tag == EventTag::Checkpoint;
        Ok(())
    }
}
