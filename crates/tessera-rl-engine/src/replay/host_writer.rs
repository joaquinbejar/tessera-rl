// SPDX-License-Identifier: MIT OR Apache-2.0

//! The recorder of a host pool (ADR-0072, ADR-0069 item 8, ADR-0070 item 9,
//! ADR-0073).
//!
//! Two kinds of file:
//!
//! - **With snapshots** ([`HostRecorder::with_snapshots`], for a host that
//!   implements `HostSnapshot` with a canonical config): as for an
//!   `EnvPool`, a keyframe of every instance's `HostRecord` first, periodic
//!   keyframes and checkpoints of the host `StateHash`, a final checkpoint,
//!   and state writes followed by a keyframe of the written ids. The config
//!   bytes and hash and the identity digest are in the header; the outputs
//!   hash is optional.
//! - **From reset only** ([`HostRecorder::new`], any host): the recorder
//!   resets every instance with the caller's `Reseed`, and that reset is the
//!   first event; every `Reset` and `Step` carries the outputs hash; there
//!   are no keyframes or checkpoints.
//!
//! In both, a refusing host's `Step` carries its refusal rows and a
//! notifying host's rejected call is a `Rejected` event.

use std::io::Write;

use tessera_rl_core::config::{CanonicalConfig, canonical_bytes, config_hash};
use tessera_rl_core::encoding::{HashSink, Sink, checksum};
use tessera_rl_core::{HostEnvironment, HostSnapshot, StateHash};

use crate::error::EngineError;
use crate::host::{HostPool, HostRecord};
use crate::outputs::Outputs;
use crate::persist::{host_identity, write_shared_header};
use crate::pool::Reseed;
use crate::validate::ActionInput;

use super::ReplayError;
use super::format::{
    END_BODY_LEN, EventTag, FLAG_CONFIG_CHECKED, FLAG_FROM_RESET_ONLY, FLAG_HOST_SNAPSHOTS,
    FLAG_OUTPUTS_HASH, IDS_ALL, MAX_CONFIG_BYTES, REPLAY_MAGIC, StateWriteOp, push_ids, push_len,
};
use super::outputs_hash::{outputs_hash, refusal_rows};
use super::writer::{ReplayWriterOptions, frame, push_reseed};

/// Appends `len:u32 record` for each id (`None`: every instance).
pub(super) type RecordsFn<E, const A: usize> =
    fn(&mut HostPool<E, A>, Option<&[u32]>, &mut Vec<u8>) -> Result<(), ReplayError>;

/// The snapshot operations of a host with `HostSnapshot`, as function
/// pointers, so the calls shared by both kinds of file need no bound.
pub(super) struct SnapOps<E: HostEnvironment, const A: usize> {
    /// Appends `len:u32 record` for each id (`None`: every instance).
    pub(super) records: RecordsFn<E, A>,
    /// The host `StateHash` of one instance.
    pub(super) hash: fn(&mut HostPool<E, A>, u32) -> Result<StateHash, EngineError>,
}

impl<E: HostEnvironment, const A: usize> Clone for SnapOps<E, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: HostEnvironment, const A: usize> Copy for SnapOps<E, A> {}

impl<E, const A: usize> SnapOps<E, A>
where
    E: HostSnapshot,
    E::Config: CanonicalConfig,
{
    pub(super) fn of() -> Self {
        Self {
            records: push_records::<E, A>,
            hash: HostPool::<E, A>::state_hash,
        }
    }
}

/// Appends the encoded records of `ids`, each after its `u32` length.
fn push_records<E, const A: usize>(
    pool: &mut HostPool<E, A>,
    ids: Option<&[u32]>,
    out: &mut Vec<u8>,
) -> Result<(), ReplayError>
where
    E: HostSnapshot,
    E::Config: CanonicalConfig,
{
    let ids: Vec<u32> = ids.map_or_else(
        || (0..pool.options().num_envs.get()).collect(),
        <[u32]>::to_vec,
    );
    let mut record = HostRecord::new();
    for id in ids {
        pool.snapshot_to(id, &mut record)?;
        let bytes = record.to_bytes().ok_or(ReplayError::Unsupported {
            feature: "a host record over u32::MAX bytes",
        })?;
        push_len(out, bytes.len())?;
        out.extend_from_slice(&bytes);
    }
    Ok(())
}

/// Records the calls on a [`HostPool`] into a replay file written to `W`.
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
    options: ReplayWriterOptions,
    /// Whether `Reset` and `Step` carry the outputs hash.
    hashes: bool,
    calls: u64,
    checkpointed: bool,
    /// `Some` for a file with snapshots.
    snap: Option<SnapOps<E, A>>,
}

impl<E: HostEnvironment, const A: usize, W> core::fmt::Debug for HostRecorder<'_, E, A, W> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostRecorder")
            .field("events", &self.events)
            .field("calls", &self.calls)
            .field("closed", &self.closed)
            .field("snapshots", &self.snap.is_some())
            .finish_non_exhaustive()
    }
}

impl<'p, E: HostEnvironment, const A: usize, W: Write> HostRecorder<'p, E, A, W> {
    /// Starts a `from_reset_only` recording: writes the header, then resets
    /// every instance with `reseed` and logs that reset as the first event.
    /// Recording therefore always starts from a reset; the pool's earlier
    /// state does not matter.
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
        let header = header_bytes(pool, FLAG_FROM_RESET_ONLY | FLAG_OUTPUTS_HASH, &[], 0, None)?;
        let mut recorder = Self::start(pool, out, ReplayWriterOptions::default(), true, None);
        recorder.out.write_all(&header)?;
        recorder.reset_inner(None, reseed)?;
        Ok(recorder)
    }

    fn start(
        pool: &'p mut HostPool<E, A>,
        out: W,
        options: ReplayWriterOptions,
        hashes: bool,
        snap: Option<SnapOps<E, A>>,
    ) -> Self {
        Self {
            pool,
            out,
            body: Vec::new(),
            sum: HashSink::new(),
            events: 0,
            closed: false,
            options,
            hashes,
            calls: 0,
            checkpointed: false,
            snap,
        }
    }

    /// The recorded pool, read-only.
    #[must_use]
    pub fn pool(&self) -> &HostPool<E, A> {
        self.pool
    }

    /// [`HostPool::reset_all`], logged.
    ///
    /// # Errors
    ///
    /// The pool's error as [`ReplayError::Engine`], [`ReplayError::Io`], or
    /// [`ReplayError::RecorderClosed`] after a poison.
    pub fn reset_all(&mut self, reseed: Reseed<'_>) -> Result<&Outputs, ReplayError> {
        self.reset_inner(None, reseed)
    }

    /// [`HostPool::reset`], logged.
    ///
    /// # Errors
    ///
    /// As [`HostRecorder::reset_all`].
    pub fn reset(&mut self, ids: &[u32], reseed: Reseed<'_>) -> Result<&Outputs, ReplayError> {
        self.reset_inner(Some(ids), reseed)
    }

    /// [`HostPool::step`], logged with, when the file has them, its outputs
    /// hash, and for a refusing host its refusal rows; a notifying host's
    /// rejection is logged as a `Rejected` event.
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
                if self.hashes {
                    self.body.extend_from_slice(&hash.to_le_bytes());
                }
                if E::REFUSES_AGENTS {
                    push_len(&mut self.body, rows.len())?;
                    for row in rows {
                        for word in [row.instance, row.slot, row.code] {
                            self.body.extend_from_slice(&word.to_le_bytes());
                        }
                    }
                }
                self.emit_body(EventTag::Step)?;
                self.after_call()?;
            }
            Err(e) if self.pool.is_poisoned() => {
                self.body.clear();
                self.body.push(IDS_ALL);
                self.body.extend_from_slice(&joint);
                // No outputs exist on a poisoned pool: a zero hash and no
                // rows, never compared (the replay expects the poison).
                if self.hashes {
                    self.body.extend_from_slice(&0u64.to_le_bytes());
                }
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
                    self.after_call()?;
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

    /// Writes the trailer: for a file with snapshots a final checkpoint
    /// (unless the pool was poisoned or the last event already is one), then
    /// the end event. Returns the writer, flushed.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Io`], or [`ReplayError::Engine`] if a hash cannot be
    /// taken.
    pub fn finish(mut self) -> Result<W, ReplayError> {
        if self.snap.is_some() && !self.closed && !self.checkpointed {
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
                if self.hashes {
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
                }
                self.emit_body(EventTag::Reset)?;
                self.after_call()?;
                Ok(self.pool.outputs()?)
            }
            Err(e) if self.pool.is_poisoned() => {
                if self.hashes {
                    self.body.extend_from_slice(&0u64.to_le_bytes());
                }
                Err(self.poisoned(EventTag::Reset, e))
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Logs a state write that succeeded or poisoned; then, on success, a
    /// keyframe of the written ids.
    fn state_write(
        &mut self,
        op: StateWriteOp,
        ids: Option<&[u32]>,
        result: Result<(), EngineError>,
    ) -> Result<(), ReplayError> {
        if result.is_err() && !self.pool.is_poisoned() {
            return result.map_err(Into::into);
        }
        self.body.clear();
        self.body.push(op as u8);
        push_ids(&mut self.body, ids)?;
        // Host pools have no branch counter (no `fork_branched`, ADR-0073).
        self.body.extend_from_slice(&0u64.to_le_bytes());
        if let Err(e) = result {
            return Err(self.poisoned(EventTag::StateWrite, e));
        }
        self.emit_body(EventTag::StateWrite)?;
        self.keyframe(ids)?;
        self.after_call()
    }

    /// Counts a logged call; adds the periodic keyframe and checkpoint of a
    /// file with snapshots.
    fn after_call(&mut self) -> Result<(), ReplayError> {
        self.calls = self.calls.checked_add(1).ok_or(ReplayError::Unsupported {
            feature: "more than u64::MAX calls",
        })?;
        if self.snap.is_none() {
            return Ok(());
        }
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

    /// A keyframe of `ids` (`None`: every instance): branch counter 0, the
    /// ids, then each `HostRecord` after its `u32` length.
    fn keyframe(&mut self, ids: Option<&[u32]>) -> Result<(), ReplayError> {
        let Some(snap) = self.snap else {
            return Ok(());
        };
        self.body.clear();
        self.body.extend_from_slice(&0u64.to_le_bytes());
        // The reset source (ADR-0075): host pools have no archives (item 8).
        self.body.push(0);
        push_ids(&mut self.body, ids)?;
        (snap.records)(self.pool, ids, &mut self.body)?;
        self.emit_body(EventTag::Keyframe)
    }

    /// A checkpoint of every instance's host `StateHash`.
    fn checkpoint(&mut self) -> Result<(), ReplayError> {
        let Some(snap) = self.snap else {
            return Ok(());
        };
        self.body.clear();
        push_ids(&mut self.body, None)?;
        for id in 0..self.pool.options().num_envs.get() {
            let hash = (snap.hash)(self.pool, id)?;
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

impl<'p, E, const A: usize, W> HostRecorder<'p, E, A, W>
where
    E: HostSnapshot,
    E::Config: CanonicalConfig,
    W: Write,
{
    /// Starts a recording with snapshots (ADR-0073): writes the header, with
    /// the canonical config, its hash and the identity digest, then a
    /// keyframe of every instance. The pool may be in any state.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Engine`] on a poisoned pool or when a snapshot fails,
    /// [`ReplayError::Unsupported`] for a config over the cap, or
    /// [`ReplayError::Io`].
    pub fn with_snapshots(
        pool: &'p mut HostPool<E, A>,
        out: W,
        options: ReplayWriterOptions,
    ) -> Result<Self, ReplayError> {
        let identity = pool.snapshot_identity()?;
        let config = canonical_bytes(pool.config());
        let hash =
            config_hash(E::ENV_ID, E::ENV_VERSION, pool.config()).map_err(EngineError::from)?;
        let mut flags = FLAG_HOST_SNAPSHOTS | FLAG_CONFIG_CHECKED;
        if options.outputs_hash {
            flags |= FLAG_OUTPUTS_HASH;
        }
        let header = header_bytes(pool, flags, &config, hash, Some(identity))?;
        let mut recorder = Self::start(
            pool,
            out,
            options,
            options.outputs_hash,
            Some(SnapOps::of()),
        );
        recorder.out.write_all(&header)?;
        recorder.keyframe(None)?;
        Ok(recorder)
    }

    /// [`HostPool::snapshot_all`], for a caller that restores the records
    /// later through [`HostRecorder::restore_all`]. It reads state and logs
    /// nothing. If `save` panics the pool is poisoned and the recording is
    /// closed without a `Poisoned` event: no logged call poisoned it.
    ///
    /// # Errors
    ///
    /// The pool's error as [`ReplayError::Engine`], or
    /// [`ReplayError::RecorderClosed`].
    pub fn snapshot_all(&mut self, out: &mut [HostRecord<A>]) -> Result<(), ReplayError> {
        self.open()?;
        let result = self.pool.snapshot_all(out);
        if self.pool.is_poisoned() {
            self.closed = true;
        }
        Ok(result?)
    }

    /// [`HostPool::restore`], logged with a keyframe of `id`.
    ///
    /// # Errors
    ///
    /// As [`HostRecorder::reset_all`].
    pub fn restore(&mut self, id: u32, record: &HostRecord<A>) -> Result<(), ReplayError> {
        self.open()?;
        let result = self.pool.restore(id, record);
        self.state_write(StateWriteOp::Restore, Some(&[id]), result)
    }

    /// [`HostPool::restore_all`], logged with a keyframe of every instance.
    ///
    /// # Errors
    ///
    /// As [`HostRecorder::reset_all`].
    pub fn restore_all(&mut self, records: &[HostRecord<A>]) -> Result<(), ReplayError> {
        self.open()?;
        let result = self.pool.restore_all(records);
        self.state_write(StateWriteOp::RestoreAll, None, result)
    }

    /// [`HostPool::import`], logged with a keyframe of `ids`.
    ///
    /// # Errors
    ///
    /// As [`HostRecorder::reset_all`].
    pub fn import(&mut self, ids: &[u32], records: &[HostRecord<A>]) -> Result<(), ReplayError> {
        self.open()?;
        let result = self.pool.import(ids, records);
        self.state_write(StateWriteOp::Import, Some(ids), result)
    }

    /// [`HostPool::fork`], logged with a keyframe of `dst`.
    ///
    /// # Errors
    ///
    /// As [`HostRecorder::reset_all`].
    pub fn fork(&mut self, src: u32, dst: &[u32]) -> Result<(), ReplayError> {
        self.open()?;
        let result = self.pool.fork(src, dst);
        self.state_write(StateWriteOp::Fork, Some(dst), result)
    }
}

/// The header of a host file: the shared ADR-0011 fields with the host
/// identity of `crate::persist::host_identity`, the config (empty unless
/// checked), the action layout, the host fields and, with snapshots, the
/// identity digest.
fn header_bytes<E: HostEnvironment, const A: usize>(
    pool: &HostPool<E, A>,
    flags: u8,
    config: &[u8],
    config_hash: u64,
    identity: Option<u64>,
) -> Result<Vec<u8>, ReplayError> {
    if config.len() > MAX_CONFIG_BYTES {
        return Err(ReplayError::Unsupported {
            feature: "a canonical config over the replay cap",
        });
    }
    // ADR-0077: host files hold no assets.
    let mut assets = Vec::new();
    tessera_rl_core::EnvConfig::asset_entries(pool.config(), &mut assets);
    if !assets.is_empty() {
        return Err(crate::persist::PersistError::Incompatible { field: "assets" }.into());
    }
    let id = host_identity(
        E::ENV_ID,
        E::ENV_VERSION,
        E::MASKS,
        pool.options(),
        config_hash,
    );
    let mut out = Vec::new();
    write_shared_header(
        &mut out,
        REPLAY_MAGIC,
        crate::persist::FORMAT_VERSION,
        &id,
        pool.options().num_envs.get(),
        env!("CARGO_PKG_VERSION"),
        &[],
    )?;
    push_len(&mut out, config.len())?;
    out.extend_from_slice(config);
    let (discrete, continuous) = pool.action_heads();
    push_len(&mut out, A)?;
    push_len(&mut out, discrete)?;
    push_len(&mut out, continuous)?;
    out.extend_from_slice(&[
        flags,
        if E::REFUSES_AGENTS { 2 } else { 1 },
        u8::from(E::NOTIFIES_REJECTION),
        u8::from(E::REFUSES_AGENTS),
    ]);
    if let Some(identity) = identity {
        out.extend_from_slice(&identity.to_le_bytes());
    }
    // ADR-0075 item 8: host pools have no archives.
    // Host pools have no policy map: the default map, count 0.
    super::v1_fields::push_extension(
        &mut out,
        &[],
        false,
        pool.options().chunk_size.get(),
        &super::ResetSourceRecord::Fresh,
    );
    let sum = checksum(&out);
    out.extend_from_slice(&sum.to_le_bytes());
    Ok(out)
}
