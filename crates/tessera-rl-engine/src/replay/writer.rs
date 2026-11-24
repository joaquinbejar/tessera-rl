// SPDX-License-Identifier: MIT OR Apache-2.0

//! The recorder: a wrapper around an [`EnvPool`] that logs every call that
//! changes state (ADR-0072 sections 1 to 4).
//!
//! The pool is never touched by recording: the recorder calls the pool's
//! public methods, then copies the call's inputs into its own buffer and
//! writes the event. A pool without a recorder pays nothing; recording
//! allocates, in the recorder only.

use std::io::Write;

use tessera_rl_core::config::{CanonicalConfig, canonical_bytes};
use tessera_rl_core::encoding::{Canonical, HashSink, Sink, encode_to_vec};
use tessera_rl_core::{AdaptiveValues, Persist, RecordLayout, Validate};

use crate::error::EngineError;
use crate::masks::PersistMask;
use crate::outputs::Outputs;
use crate::persist::{BUILD_ID, descriptor, identity, write_shared_header};
use crate::pool::{EnvPool, Reseed};
use crate::record::{DecodedRecord, Record};
use crate::validate::ActionInput;

use super::ReplayError;
use super::format::{EventTag, REPLAY_MAGIC, StateWriteOp, push_ids, push_len};

/// Options of a [`Recorder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayWriterOptions {
    /// A full keyframe every this many logged calls; 0 writes only the first.
    pub keyframe_every: u64,
    /// A checkpoint of every instance every this many logged calls; 0 writes
    /// only the last.
    pub checkpoint_every: u64,
}

impl Default for ReplayWriterOptions {
    fn default() -> Self {
        Self {
            keyframe_every: 0,
            checkpoint_every: 1024,
        }
    }
}

/// Records the calls on an [`EnvPool`] into a replay file written to `W`.
///
/// Every method mirrors the pool method of the same name. A call the pool
/// refuses without poisoning is not logged. A call that poisons the pool is
/// logged, followed by a `Poisoned` event; the recorder then refuses every
/// call, and [`Recorder::finish`] closes the file.
pub struct Recorder<'p, E, M, W>
where
    E: Persist,
    M: PersistMask<E>,
{
    pool: &'p mut EnvPool<E, M>,
    out: W,
    options: ReplayWriterOptions,
    body: Vec<u8>,
    sum: HashSink,
    events: u64,
    calls: u64,
    closed: bool,
    checkpointed: bool,
}

impl<E, M, W> core::fmt::Debug for Recorder<'_, E, M, W>
where
    E: Persist,
    M: PersistMask<E>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Recorder")
            .field("options", &self.options)
            .field("events", &self.events)
            .field("calls", &self.calls)
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl<'p, E, M, W> Recorder<'p, E, M, W>
where
    E: Persist,
    E::Config: CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    M: PersistMask<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    W: Write,
{
    /// Starts a recording: writes the header and a keyframe of every
    /// instance.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayError::AdaptiveUnsupported`] for a pool with adaptive
    /// config values, a [`ReplayError::Persist`] for a header the pool cannot
    /// describe (for example a `same-platform` environment built without
    /// `TESSERA_BUILD_ID`), [`ReplayError::Engine`] on a poisoned pool, or
    /// [`ReplayError::Io`].
    pub fn new(
        pool: &'p mut EnvPool<E, M>,
        out: W,
        options: ReplayWriterOptions,
    ) -> Result<Self, ReplayError> {
        if <<E::Header as RecordLayout>::Adaptive as Canonical>::ENCODED_LEN != 0 {
            return Err(ReplayError::AdaptiveUnsupported);
        }
        if pool.is_poisoned() {
            return Err(EngineError::Poisoned.into());
        }
        let header = header_bytes(pool)?;
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
        };
        recorder.out.write_all(&header)?;
        recorder.keyframe(None)?;
        Ok(recorder)
    }

    /// The recorded pool, read-only.
    #[must_use]
    pub fn pool(&self) -> &EnvPool<E, M> {
        self.pool
    }

    /// [`EnvPool::reset_all`], logged.
    ///
    /// # Errors
    ///
    /// The pool's error as [`ReplayError::Engine`], [`ReplayError::Io`], or
    /// [`ReplayError::RecorderClosed`] after a poison.
    pub fn reset_all(&mut self, reseed: Reseed<'_>) -> Result<&Outputs, ReplayError> {
        self.reset_inner(None, reseed)
    }

    /// [`EnvPool::reset`], logged.
    ///
    /// # Errors
    ///
    /// As [`Recorder::reset_all`].
    pub fn reset(&mut self, ids: &[u32], reseed: Reseed<'_>) -> Result<&Outputs, ReplayError> {
        self.reset_inner(Some(ids), reseed)
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
        self.logged(EventTag::Reset, result, KeyframeAfter::No)?;
        Ok(self.pool.outputs()?)
    }

    /// [`EnvPool::step`], logged.
    ///
    /// # Errors
    ///
    /// As [`Recorder::reset_all`].
    pub fn step(&mut self, actions: ActionInput<'_>) -> Result<&Outputs, ReplayError> {
        self.open()?;
        self.body.clear();
        push_ids(&mut self.body, None)?;
        push_len(&mut self.body, actions.discrete.len())?;
        for a in actions.discrete {
            self.body.extend_from_slice(&a.to_le_bytes());
        }
        push_len(&mut self.body, actions.continuous.len())?;
        for a in actions.continuous {
            self.body.extend_from_slice(&a.to_bits().to_le_bytes());
        }
        let result = self.pool.step(actions).map(|_| ());
        self.logged(EventTag::Step, result, KeyframeAfter::No)?;
        Ok(self.pool.outputs()?)
    }

    /// [`EnvPool::restore`], logged with a keyframe of `id`.
    ///
    /// # Errors
    ///
    /// As [`Recorder::reset_all`].
    pub fn restore(&mut self, id: u32, record: &Record<E>) -> Result<(), ReplayError> {
        self.state_write(StateWriteOp::Restore, Some(&[id]), |pool| {
            pool.restore(id, record)
        })
    }

    /// [`EnvPool::restore_all`], logged with a keyframe of every instance.
    ///
    /// # Errors
    ///
    /// As [`Recorder::reset_all`].
    pub fn restore_all(&mut self, records: &[Record<E>]) -> Result<(), ReplayError> {
        self.state_write(StateWriteOp::RestoreAll, None, |pool| {
            pool.restore_all(records)
        })
    }

    /// [`EnvPool::fork`], logged with a keyframe of `dst`.
    ///
    /// # Errors
    ///
    /// As [`Recorder::reset_all`].
    pub fn fork(&mut self, src: u32, dst: &[u32]) -> Result<(), ReplayError> {
        self.state_write(StateWriteOp::Fork, Some(dst), |pool| pool.fork(src, dst))
    }

    /// [`EnvPool::fork_branched`], logged with a keyframe of `dst` and the
    /// branch counter after the call.
    ///
    /// # Errors
    ///
    /// As [`Recorder::reset_all`].
    pub fn fork_branched(
        &mut self,
        src: u32,
        dst: &[u32],
        labels: Option<&[u64]>,
    ) -> Result<(), ReplayError> {
        self.state_write(StateWriteOp::ForkBranched, Some(dst), |pool| {
            pool.fork_branched(src, dst, labels)
        })
    }

    /// Writes the trailer: a final checkpoint (unless the pool was poisoned
    /// or the last event already is one), then the end event. Returns the
    /// writer, flushed.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Io`], or [`ReplayError::Engine`] if a hash cannot be
    /// taken.
    pub fn finish(mut self) -> Result<W, ReplayError> {
        if !self.closed && !self.checkpointed {
            self.checkpoint()?;
        }
        let mut end = [0u8; 16];
        let (count, sum) = end.split_at_mut(8);
        count.copy_from_slice(&self.events.to_le_bytes());
        sum.copy_from_slice(&self.sum.digest().to_le_bytes());
        self.out
            .write_all(&frame(EventTag::End, super::format::END_BODY_LEN))?;
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

    fn state_write(
        &mut self,
        op: StateWriteOp,
        ids: Option<&[u32]>,
        call: impl FnOnce(&mut EnvPool<E, M>) -> Result<(), EngineError>,
    ) -> Result<(), ReplayError> {
        self.open()?;
        let result = call(&mut *self.pool);
        if result.is_err() && !self.pool.is_poisoned() {
            return result.map_err(Into::into);
        }
        self.body.clear();
        self.body.push(op as u8);
        push_ids(&mut self.body, ids)?;
        // On a poison the counter cannot be read; the event is terminal.
        let counter = self.pool.pool_context().map_or(0, |c| c.branch_counter);
        self.body.extend_from_slice(&counter.to_le_bytes());
        self.logged(EventTag::StateWrite, result, KeyframeAfter::Of(ids))
    }

    /// Logs the call in `self.body` under `tag` if it succeeded or poisoned
    /// the pool. `keyframe` holds the written ids of a state write.
    fn logged(
        &mut self,
        tag: EventTag,
        result: Result<(), EngineError>,
        keyframe: KeyframeAfter<'_>,
    ) -> Result<(), ReplayError> {
        match result {
            Ok(()) => {
                self.emit_body(tag)?;
                if let KeyframeAfter::Of(ids) = keyframe {
                    self.keyframe(ids)?;
                }
                self.after_call()
            }
            Err(e) if self.pool.is_poisoned() => {
                self.emit_body(tag)?;
                self.body.clear();
                self.emit_body(EventTag::Poisoned)?;
                self.closed = true;
                Err(e.into())
            }
            Err(e) => Err(e.into()),
        }
    }

    fn after_call(&mut self) -> Result<(), ReplayError> {
        self.calls = self.calls.checked_add(1).ok_or(ReplayError::Unsupported {
            feature: "more than u64::MAX calls",
        })?;
        let calls = self.calls;
        let every = |n: u64| n != 0 && calls.is_multiple_of(n);
        let (keyframe, checkpoint) = (
            every(self.options.keyframe_every),
            every(self.options.checkpoint_every),
        );
        if keyframe {
            self.keyframe(None)?;
        }
        if checkpoint {
            self.checkpoint()?;
        }
        Ok(())
    }

    /// A keyframe of `ids` (`None`: every instance).
    fn keyframe(&mut self, ids: Option<&[u32]>) -> Result<(), ReplayError> {
        self.body.clear();
        let counter = self.pool.pool_context()?.branch_counter;
        self.body.extend_from_slice(&counter.to_le_bytes());
        push_ids(&mut self.body, ids)?;
        for id in self.ids(ids) {
            let record = DecodedRecord::<E>::from(self.pool.snapshot(id)?);
            self.body.extend_from_slice(&encode_to_vec(&record));
        }
        self.emit_body(EventTag::Keyframe)
    }

    /// A checkpoint of every instance.
    fn checkpoint(&mut self) -> Result<(), ReplayError> {
        self.body.clear();
        push_ids(&mut self.body, None)?;
        for id in self.ids(None) {
            let hash = self.pool.state_hash(id)?;
            self.body.extend_from_slice(&hash.get().to_le_bytes());
        }
        self.emit_body(EventTag::Checkpoint)
    }

    fn ids(&self, ids: Option<&[u32]>) -> Vec<u32> {
        ids.map_or_else(
            || (0..self.pool.options.num_envs.get()).collect(),
            <[u32]>::to_vec,
        )
    }

    /// Frames and writes `self.body` under `tag`.
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

impl<E, M, W> Recorder<'_, E, M, W>
where
    E: Persist + Validate,
    E::Config: CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    M: PersistMask<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    W: Write,
{
    /// [`EnvPool::import`], logged with a keyframe of `ids`.
    ///
    /// # Errors
    ///
    /// As [`Recorder::reset_all`].
    pub fn import(&mut self, ids: &[u32], records: &[DecodedRecord<E>]) -> Result<(), ReplayError> {
        self.state_write(StateWriteOp::Import, Some(ids), |pool| {
            pool.import(ids, records)
        })
    }
}

/// The header: the ADR-0011 fields under the replay magic, the replay
/// extension and the header checksum.
fn header_bytes<E, M>(pool: &EnvPool<E, M>) -> Result<Vec<u8>, ReplayError>
where
    E: Persist,
    E::Config: CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    M: PersistMask<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    let id = identity::<E, M>(&pool.cfg, &pool.options)?;
    let file_descriptor = descriptor(E::TIER, BUILD_ID)?;
    let mut out = Vec::new();
    write_shared_header(
        &mut out,
        REPLAY_MAGIC,
        &id,
        pool.options.num_envs.get(),
        env!("CARGO_PKG_VERSION"),
        &file_descriptor,
    )?;
    let config = canonical_bytes(&pool.cfg);
    if config.len() > super::format::MAX_CONFIG_BYTES {
        return Err(ReplayError::Unsupported {
            feature: "a canonical config over the replay cap",
        });
    }
    push_len(&mut out, config.len())?;
    out.extend_from_slice(&config);
    let (discrete, continuous) = pool.action_heads();
    push_len(&mut out, E::MAX_AGENTS)?;
    push_len(&mut out, discrete)?;
    push_len(&mut out, continuous)?;
    // flags, host_protocol, notifies, refuses: all 0 for a first-party pool.
    out.extend_from_slice(&[0, 0, 0, 0]);
    let sum = tessera_rl_core::encoding::checksum(&out);
    out.extend_from_slice(&sum.to_le_bytes());
    Ok(out)
}

/// An event's tag and length.
pub(super) const fn frame(tag: EventTag, len: u32) -> [u8; 5] {
    let [a, b, c, d] = len.to_le_bytes();
    [tag as u8, a, b, c, d]
}

/// Appends the `Reseed` variant and its seeds.
pub(super) fn push_reseed(out: &mut Vec<u8>, reseed: Reseed<'_>) -> Result<(), ReplayError> {
    match reseed {
        Reseed::Keep => out.push(0),
        Reseed::Base(seed) => {
            out.push(1);
            out.extend_from_slice(&seed.get().to_le_bytes());
        }
        Reseed::List(seeds) => {
            out.push(2);
            push_len(out, seeds.len())?;
            for seed in seeds {
                out.extend_from_slice(&seed.get().to_le_bytes());
            }
        }
        Reseed::Each(seeds) => {
            out.push(3);
            push_len(out, seeds.len())?;
            for seed in seeds {
                out.push(u8::from(seed.is_some()));
                out.extend_from_slice(
                    &seed
                        .map_or(0, tessera_rl_num::identity::Seed::get)
                        .to_le_bytes(),
                );
            }
        }
    }
    Ok(())
}

/// Whether a logged call is followed by a keyframe: a state write is, of the
/// ids it wrote (`None`: every instance).
#[derive(Clone, Copy)]
enum KeyframeAfter<'a> {
    /// No keyframe: a reset or a step.
    No,
    /// A keyframe of these ids.
    Of(Option<&'a [u32]>),
}
