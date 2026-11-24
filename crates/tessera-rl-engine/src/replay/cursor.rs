// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stepping through a replay (issue #378): a cursor that applies a file's
//! events a few at a time, so the caller can look at the pool between
//! them, and seeking to a periodic keyframe.
//!
//! The cursor builds the byte offset of every event once; each advance
//! parses only the events it applies, so stepping through a file costs
//! O(events) in total, never a rescan from the start.

use core::ops::ControlFlow;

use tessera_rl_core::config::CanonicalConfig;
use tessera_rl_core::{AdaptiveValues, Persist, RecordLayout, Validate};

use crate::archive::Archive;
use crate::masks::PersistMask;
use crate::persist::u64_of;
use crate::pool::EnvPool;
use crate::record::PoolContext;

use super::format::EventTag;
use super::reader::{Event, Ids, Replay};
use super::run::{AnyArchive, Expect, State, archive_ref, check_source, restore_context};
use super::{DivergenceKind, ReplayError, ReplaySummary};

/// What kind of event a [`ReplayPoint`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointKind {
    /// A call: a `Reset`, a `Step`, or the keyframe that completes a state
    /// write (the write is applied with its keyframe).
    Call,
    /// A keyframe that is not part of a state write (the first one or a
    /// periodic one).
    Keyframe,
    /// A checkpoint.
    Checkpoint,
    /// Anything else: a state write waiting for its keyframe, `Poisoned`.
    Other,
}

/// An event a replay has just applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayPoint {
    /// The event's index in the file.
    pub event_index: u64,
    /// The event's kind.
    pub kind: PointKind,
}

/// A replay in progress: where it stands in the file and what it checked.
///
/// Built by [`Replay::cursor`], moved by [`Replay::advance`] and
/// [`Replay::seek`], ended by [`Replay::finish`]. It holds no borrow of the
/// file, so it can be kept between calls while the file bytes are owned
/// elsewhere (see [`Replay::reopen`]).
#[derive(Debug, Clone)]
pub struct ReplayCursor {
    /// The byte offset of every event's frame; event `i` is at position `i`.
    offsets: Vec<usize>,
    /// The position of the next event to apply.
    pos: usize,
    state: State,
    /// The last advance reached the end of the file.
    done: bool,
    /// The event bytes' length and the input length of the file the cursor
    /// was built from.
    events_len: usize,
    input_len: u64,
}

impl ReplayCursor {
    /// The index of the next event to apply.
    #[must_use]
    pub fn next_event(&self) -> u64 {
        u64_of(self.pos)
    }

    /// Whether every event has been applied.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.done || self.pos >= self.offsets.len()
    }

    /// The event length, event count and input length of the file the
    /// cursor was built from.
    pub(super) fn file_shape(&self) -> (usize, u64, u64) {
        (self.events_len, u64_of(self.offsets.len()), self.input_len)
    }

    /// The position after `pos`.
    fn after(pos: usize) -> Result<usize, ReplayError> {
        pos.checked_add(1).ok_or(ReplayError::Malformed {
            event_index: u64_of(pos),
            field: "event count",
        })
    }
}

/// The refusal of a seek target.
const fn not_seekable() -> ReplayError {
    ReplayError::Unsupported {
        feature: "seeking to an event that is not a keyframe of every instance outside a state write",
    }
}

/// The kind of `event`, given the state before it is applied.
fn kind_of(st: &State, event: &Event<'_>) -> PointKind {
    match event {
        Event::Reset { .. } | Event::Step { .. } => PointKind::Call,
        Event::Keyframe { .. } => {
            if matches!(st.expect, Expect::Keyframe(_)) {
                PointKind::Call
            } else {
                PointKind::Keyframe
            }
        }
        Event::Checkpoint { .. } => PointKind::Checkpoint,
        _ => PointKind::Other,
    }
}

impl Replay<'_> {
    /// A cursor at the start of the file, for `pool`: the pool is checked
    /// against the header as [`Replay::run`] does, and the offset of every
    /// event is found in one walk over the frames.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Persist`] if the pool does not match the header, and
    /// [`ReplayError::Allocation`] if the offsets cannot be allocated.
    pub fn cursor<E, M>(&self, pool: &EnvPool<E, M>) -> Result<ReplayCursor, ReplayError>
    where
        E: Persist,
        E::Config: CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        self.cursor_with_archive(pool, None)
    }

    /// [`Replay::cursor`] for a file whose pool resets from an archive
    /// (ADR-0075 item 7): the cursor keeps `archive`, which the first
    /// keyframe, `ResetSource` events and [`Replay::seek`] restore by digest.
    ///
    /// # Errors
    ///
    /// As [`Replay::cursor`].
    pub fn cursor_with_archive<E, M>(
        &self,
        pool: &EnvPool<E, M>,
        archive: Option<&Archive<E>>,
    ) -> Result<ReplayCursor, ReplayError>
    where
        E: Persist,
        E::Config: CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        self.check_pool(pool)?;
        self.check_archives(archive)?;
        let offsets = self.event_offsets()?;
        let (events_len, input_len) = self.shape();
        let mut state = State::new();
        state.archive = archive.map(AnyArchive::new);
        Ok(ReplayCursor {
            offsets,
            pos: 0,
            state,
            done: false,
            events_len,
            input_len,
        })
    }

    /// Applies events from the cursor's position, as [`Replay::run`] does,
    /// calling `stop` after each. Returns the point `stop` accepted, or
    /// `None` once the file ends (the cursor is then done). The cost is the
    /// events applied; nothing before the cursor is read again.
    ///
    /// # Errors
    ///
    /// The errors of [`Replay::run`] for the events applied. After an error
    /// the cursor stays at the failing event; finishing it reports a
    /// stopped replay.
    pub fn advance<E, M>(
        &self,
        pool: &mut EnvPool<E, M>,
        cur: &mut ReplayCursor,
        mut stop: impl FnMut(ReplayPoint) -> bool,
    ) -> Result<Option<ReplayPoint>, ReplayError>
    where
        E: Persist + Validate,
        E::Config: CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        let Some(&offset) = cur.offsets.get(cur.pos) else {
            cur.done = true;
            return Ok(None);
        };
        if cur.done {
            return Ok(None);
        }
        for item in self.events_at(offset, u64_of(cur.pos)) {
            let (index, event) = item?;
            let kind = kind_of(&cur.state, &event);
            self.apply(pool, &mut cur.state, index, event)?;
            cur.state.summary.events = index.checked_add(1).ok_or(ReplayError::Malformed {
                event_index: index,
                field: "event count",
            })?;
            cur.pos = ReplayCursor::after(cur.pos)?;
            let point = ReplayPoint {
                event_index: index,
                kind,
            };
            if stop(point) {
                return Ok(Some(point));
            }
        }
        cur.done = true;
        Ok(None)
    }

    /// Ends a replay. A cursor that applied every event gets the end-of-file
    /// checks of [`Replay::run`] and a verified summary; any other gets the
    /// summary so far with `stopped_at` set to the next event, which never
    /// reads as verified.
    ///
    /// # Errors
    ///
    /// For a finished file, the end-of-file errors of [`Replay::run`];
    /// [`ReplayError::Malformed`] for a cursor built from another file.
    pub fn finish(&self, cur: ReplayCursor) -> Result<ReplaySummary, ReplayError> {
        let (events_len, input_len) = self.shape();
        if events_len != cur.events_len || input_len != cur.input_len {
            return Err(ReplayError::Malformed {
                event_index: 0,
                field: "a cursor of another file",
            });
        }
        if cur.is_done() {
            return cur.state.end();
        }
        let next = cur.next_event();
        let mut summary = cur.state.summary;
        summary.stopped_at = Some(next);
        Ok(summary)
    }

    /// Refuses, before anything is replayed, a file whose archive sources
    /// cannot be restored with `archive` (ADR-0075 item 7): every digest the
    /// file names (the header's archive block, every keyframe's reset source
    /// and every `ResetSource` event) must be one archive, the one given.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Unsupported`] for two or more archives in one file;
    /// [`ReplayError::Engine`] with [`crate::EngineError::ArchiveMismatch`]
    /// when the file names an archive and `archive` is missing or another;
    /// the parse errors of [`Replay::events`].
    fn check_archives<E: tessera_rl_core::Environment>(
        &self,
        archive: Option<&Archive<E>>,
    ) -> Result<(), ReplayError> {
        let mut named: Option<u64> = self.header.archive.map(|a| a.digest);
        let mut note = |source: &super::ResetSourceRecord| -> Result<(), ReplayError> {
            if let super::ResetSourceRecord::Archive { digest, .. } = source {
                match named {
                    Some(d) if d != *digest => {
                        return Err(ReplayError::Unsupported {
                            feature: "more than one archive in one replay file: one archive per file (ADR-0075)",
                        });
                    }
                    _ => named = Some(*digest),
                }
            }
            Ok(())
        };
        for item in self.events() {
            match item? {
                (_, Event::Keyframe { reset_source, .. }) => note(&reset_source)?,
                (_, Event::ResetSource(source)) => note(&source)?,
                _ => {}
            }
        }
        match named {
            Some(digest) => check_source(
                &super::ResetSourceRecord::Archive {
                    digest,
                    weights: Vec::new(),
                },
                archive,
            ),
            None => Ok(()),
        }
    }

    /// The event indices of the keyframes [`Replay::seek`] accepts: the
    /// keyframes of every instance that do not complete a state write, in
    /// file order (event 0 included). A keyframe with an archive reset source
    /// needs a cursor made with [`Replay::cursor_with_archive`] (ADR-0075).
    ///
    /// # Errors
    ///
    /// The parse errors of [`Replay::events`].
    pub fn keyframes(&self) -> Result<Vec<u64>, ReplayError> {
        let mut out = Vec::new();
        let mut after_write = false;
        for item in self.events() {
            let (index, event) = item?;
            if let Event::Keyframe { ids: Ids::All, .. } = &event
                && !after_write
            {
                out.push(index);
            }
            after_write = matches!(event, Event::StateWrite { .. });
        }
        Ok(out)
    }

    /// Moves the cursor to the keyframe at event `keyframe_event`: its
    /// records are imported into every instance with `import_all` and the
    /// pool context (the branch counter and the reset source, with the
    /// cursor's archive for an archive source) is restored, so the pool holds the
    /// recorded state at that event. The cursor then stands just after the
    /// keyframe, with fresh counters: a summary after a seek counts from the
    /// keyframe.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Unsupported`] unless the event is one that
    /// [`Replay::keyframes`] lists (a keyframe of every instance, not part
    /// of a state write); [`ReplayError::Engine`] with
    /// [`crate::EngineError::ArchiveMismatch`], before any write, for an
    /// archive reset source when the cursor has no archive or another one;
    /// the parse errors of the keyframe; [`ReplayError::Diverged`] with
    /// `KeyframeRejected` if the pool refuses its records.
    pub fn seek<E, M>(
        &self,
        pool: &mut EnvPool<E, M>,
        cur: &mut ReplayCursor,
        keyframe_event: u64,
    ) -> Result<(), ReplayError>
    where
        E: Persist + Validate,
        E::Config: CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        let pos = usize::try_from(keyframe_event).map_err(|_| not_seekable())?;
        let offset = *cur.offsets.get(pos).ok_or_else(not_seekable)?;
        if let Some(prev) = pos.checked_sub(1) {
            let prev = *cur.offsets.get(prev).ok_or_else(not_seekable)?;
            if self.tag_at(prev) == Some(EventTag::StateWrite as u8) {
                return Err(not_seekable());
            }
        }
        let (index, event) = self
            .events_at(offset, keyframe_event)
            .next()
            .ok_or_else(not_seekable)??;
        let Event::Keyframe {
            branch_counter,
            reset_source,
            ids: Ids::All,
            records,
        } = event
        else {
            return Err(not_seekable());
        };
        // An archive source needs the cursor's archive: refused before any
        // write, so a refused seek leaves the pool untouched.
        let archive = cur.state.archive.clone();
        let mut st = State::new();
        st.archive = archive;
        let given = archive_ref::<E>(&st);
        // Unreachable after check_archives: kept as defence in depth (rev, #363 B r2).
        check_source(&reset_source, given)?;
        let records = self.decode_records::<E>(index, records)?;
        pool.import_all(&records)
            .map_err(|_| st.diverged(index, DivergenceKind::KeyframeRejected, None))?;
        // The whole pool context: the branch counter and the keyframe's reset
        // source (ADR-0075), whatever the pool held before the seek
        // (REVIEW-4c-design-r1 point 2).
        restore_context(
            pool,
            PoolContext {
                branch_counter,
                reset_source,
                // Replays refuse pools with transforms (ADR-0078 item 8).
                adaptive: crate::adaptive::AdaptiveContext::default(),
                // The recorded map's digest (ADR-0079 item 6): a pool with
                // another map is refused by `check_context`, before any write.
                policy_map: self.header.policy_map_digest()?,
            },
            archive_ref::<E>(&st),
        )?;
        st.summary.keyframes = 1;
        st.summary.started_at = Some(index);
        let next = ReplayCursor::after(pos)?;
        st.summary.events = u64_of(next);
        cur.state = st;
        cur.pos = next;
        cur.done = false;
        Ok(())
    }

    /// Replays the file into `pool` as [`Replay::run`] does, calling
    /// `after` with the pool after each event. A `Break` stops the replay:
    /// the summary then has `stopped_at` set and is not verified (unless the
    /// break came after the last event).
    ///
    /// # Errors
    ///
    /// The errors of [`Replay::run`].
    pub fn run_with<E, M>(
        &self,
        pool: &mut EnvPool<E, M>,
        mut after: impl FnMut(ReplayPoint, &EnvPool<E, M>) -> ControlFlow<()>,
    ) -> Result<ReplaySummary, ReplayError>
    where
        E: Persist + Validate,
        E::Config: CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        let mut cur = self.cursor(pool)?;
        // One event per advance, so `after` can see the pool; each advance
        // starts at the cursor's offset, so the walk stays linear.
        while let Some(point) = self.advance(pool, &mut cur, |_| true)? {
            if after(point, pool).is_break() {
                break;
            }
        }
        self.finish(cur)
    }
}
