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

use crate::archive::ResetSource;
use crate::error::EngineError;
use crate::masks::PersistMask;
use crate::outputs::Outputs;
use crate::persist::{BUILD_ID, descriptor, identity, write_shared_header};
use crate::pool::{EnvPool, Reseed};
use crate::record::{DecodedRecord, Record};
use crate::validate::ActionInput;

use super::format::{EventTag, REPLAY_MAGIC, StateWriteOp, push_ids, push_len};
use super::{ReplayError, ResetSourceRecord};

/// Options of a [`Recorder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayWriterOptions {
    /// A full keyframe every this many logged calls; 0 writes only the first.
    pub keyframe_every: u64,
    /// A checkpoint of every instance every this many logged calls; 0 writes
    /// only the last.
    pub checkpoint_every: u64,
    /// Whether `Reset` and `Step` events carry an outputs hash (ADR-0072
    /// section 4): off by default, since checkpoints cover state. Host
    /// recorders with snapshots only; [`Recorder`] refuses it.
    pub outputs_hash: bool,
}

impl Default for ReplayWriterOptions {
    fn default() -> Self {
        Self {
            keyframe_every: 0,
            checkpoint_every: 1024,
            outputs_hash: false,
        }
    }
}

/// A [`Recorder`] without its pool borrow, from [`Recorder::detach`].
///
/// [`DetachedRecorder::attach`] must be given the same pool, unchanged since
/// the detach: a call made on the pool while detached is not logged, and a
/// file recorded across another pool does not replay. Neither is undefined
/// behaviour; replay reports the divergence.
pub struct DetachedRecorder<E, M, W> {
    out: W,
    options: ReplayWriterOptions,
    body: Vec<u8>,
    sum: HashSink,
    events: u64,
    calls: u64,
    closed: bool,
    checkpointed: bool,
    _pool: core::marker::PhantomData<fn() -> (E, M)>,
}

impl<E, M, W> core::fmt::Debug for DetachedRecorder<E, M, W> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DetachedRecorder")
            .field("events", &self.events)
            .field("calls", &self.calls)
            .field("closed", &self.closed)
            .finish_non_exhaustive()
    }
}

impl<E, M, W> DetachedRecorder<E, M, W>
where
    E: Persist,
    M: PersistMask<E>,
{
    /// Resumes the recording on `pool`, which must be the pool it was
    /// detached from.
    #[must_use]
    pub fn attach(self, pool: &mut EnvPool<E, M>) -> Recorder<'_, E, M, W> {
        Recorder {
            pool,
            out: self.out,
            options: self.options,
            body: self.body,
            sum: self.sum,
            events: self.events,
            calls: self.calls,
            closed: self.closed,
            checkpointed: self.checkpointed,
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
        if options.outputs_hash {
            return Err(ReplayError::Unsupported {
                feature: "outputs hashes in first-party replay files",
            });
        }
        if pool.transform_identity() != 0 {
            return Err(ReplayError::Unsupported {
                feature: "observation transforms (ADR-0078 item 8)",
            });
        }
        // ADR-0079 item 6 (#347 stage B2): the header records the pool's
        // policy map (`header_bytes`).
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

    /// Releases the pool and keeps the recording, for a caller that cannot
    /// hold the borrow between calls (the Python bindings).
    /// [`DetachedRecorder::attach`] resumes it.
    #[must_use]
    pub fn detach(self) -> DetachedRecorder<E, M, W> {
        DetachedRecorder {
            out: self.out,
            options: self.options,
            body: self.body,
            sum: self.sum,
            events: self.events,
            calls: self.calls,
            closed: self.closed,
            checkpointed: self.checkpointed,
            _pool: core::marker::PhantomData,
        }
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

    /// [`EnvPool::set_lineage`], logged as a state write of `id` with a
    /// keyframe of its record, which holds the new RNG identity.
    ///
    /// # Errors
    ///
    /// As [`Recorder::reset_all`].
    pub fn set_lineage(
        &mut self,
        id: u32,
        lineage: tessera_rl_num::identity::LogicalId,
    ) -> Result<(), ReplayError> {
        self.state_write(StateWriteOp::SetLineage, Some(&[id]), |pool| {
            pool.set_lineage(id, lineage)
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

    /// [`EnvPool::set_reset_source`], logged as a `ResetSource` event
    /// (ADR-0075 item 7) after it succeeds; a refused source logs nothing.
    ///
    /// # Errors
    ///
    /// As [`Recorder::reset_all`].
    pub fn set_reset_source(&mut self, source: ResetSource<E>) -> Result<(), ReplayError> {
        self.open()?;
        self.body.clear();
        let record = match &source {
            ResetSource::Fresh => ResetSourceRecord::Fresh,
            ResetSource::Archive { archive, weights } => ResetSourceRecord::Archive {
                digest: archive.digest(),
                weights: weights.clone().unwrap_or_else(|| vec![1; archive.len()]),
            },
        };
        super::v1_fields::push_reset_source(&mut self.body, &record);
        let result = self.pool.set_reset_source(source);
        self.logged(EventTag::ResetSource, result, KeyframeAfter::No)
    }

    /// [`EnvPool::reset_from`], logged as a `ResetFrom` event (ADR-0075
    /// item 5).
    ///
    /// # Errors
    ///
    /// As [`Recorder::reset_all`].
    pub fn reset_from(&mut self, ids: &[u32], entries: &[u32]) -> Result<&Outputs, ReplayError> {
        self.open()?;
        self.body.clear();
        super::v1_fields::push_reset_from(&mut self.body, ids, entries);
        let result = self.pool.reset_from(ids, entries).map(|_| ());
        self.logged(EventTag::ResetFrom, result, KeyframeAfter::No)?;
        Ok(self.pool.outputs()?)
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
        let context = self.pool.pool_context()?;
        self.body
            .extend_from_slice(&context.branch_counter.to_le_bytes());
        // The pool context's reset source (ADR-0075 item 7).
        super::v1_fields::push_reset_source(&mut self.body, &context.reset_source);
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

    /// [`EnvPool::import_all`], logged with a keyframe of every instance.
    /// The recorder refuses every call on a poisoned pool, so this is never
    /// the recovery path here.
    ///
    /// # Errors
    ///
    /// As [`Recorder::reset_all`].
    pub fn import_all(&mut self, records: &[DecodedRecord<E>]) -> Result<(), ReplayError> {
        self.state_write(StateWriteOp::ImportAll, None, |pool| {
            pool.import_all(records)
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
    let source = pool.pool_context()?.reset_source;
    header_from::<E, M>(
        &pool.cfg,
        &pool.options,
        pool.action_heads(),
        pool.policy_map(),
        0,
        &source,
    )
}

/// The header of a first-party file from the pool's config, options,
/// action heads and policy map (ADR-0079 item 6: count 0 for the default
/// map, else one code per agent slot), with `flags` (0, or the async flags).
pub(super) fn header_from<E, M>(
    cfg: &E::Config,
    options: &crate::options::PoolOptions,
    (discrete, continuous): (usize, usize),
    policy_map: &crate::policy::PolicyMap,
    flags: u8,
    source: &super::ResetSourceRecord,
) -> Result<Vec<u8>, ReplayError>
where
    E: Persist,
    E::Config: CanonicalConfig,
    M: PersistMask<E>,
{
    super::policy_map::refuse_league(policy_map)?;
    let codes = super::policy_map::header_codes(policy_map, E::MAX_AGENTS)?;
    let id = identity::<E, M>(cfg, options)?;
    let file_descriptor = descriptor(E::TIER, BUILD_ID)?;
    let mut out = Vec::new();
    write_shared_header(
        &mut out,
        REPLAY_MAGIC,
        crate::persist::FORMAT_VERSION,
        &id,
        options.num_envs.get(),
        env!("CARGO_PKG_VERSION"),
        &file_descriptor,
    )?;
    let config = canonical_bytes(cfg);
    if config.len() > super::format::MAX_CONFIG_BYTES {
        return Err(ReplayError::Unsupported {
            feature: "a canonical config over the replay cap",
        });
    }
    push_len(&mut out, config.len())?;
    out.extend_from_slice(&config);
    push_len(&mut out, E::MAX_AGENTS)?;
    push_len(&mut out, discrete)?;
    push_len(&mut out, continuous)?;
    // flags, then host_protocol, notifies, refuses: 0 for a first-party pool.
    out.extend_from_slice(&[flags, 0, 0, 0]);
    let chunk = options.chunk_size.get();
    super::v1_fields::push_extension(&mut out, &codes, chunk, source);
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
