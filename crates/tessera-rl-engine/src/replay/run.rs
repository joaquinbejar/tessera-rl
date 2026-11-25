// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replaying a file into a pool and the divergence check (ADR-0072 section
//! 5).

use core::any::Any;
use std::sync::Arc;
use tessera_rl_core::config::{CanonicalConfig, canonical_bytes};
use tessera_rl_core::encoding::{Canonical, decode_from_slice};

use tessera_rl_core::{AdaptiveValues, Environment, Persist, RecordLayout, StateHash, Validate};

use crate::archive::{Archive, ResetSource};
use crate::error::EngineError;
use crate::masks::PersistMask;
use crate::persist::{BUILD_ID, PersistError, identity};
use crate::pool::EnvPool;
use crate::record::{DecodedRecord, PoolContext};
use crate::validate::ActionInput;

use super::format::EventTag as Tag;
use super::reader::{Event, Ids, Replay};
use super::v1_fields::not_yet;
use super::{DivergenceKind, ReplayError, ResetSourceRecord};

/// What a replay checked.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReplaySummary {
    /// Events replayed, the end event excluded.
    pub events: u64,
    /// Calls re-issued (`Reset`, `Step` and `StateWrite` events).
    pub calls: u64,
    /// Checkpoints that matched.
    pub checkpoints: u64,
    /// Keyframes imported or checked.
    pub keyframes: u64,
    /// Whether the file ends with a `Poisoned` event that the replay
    /// reproduced.
    pub poisoned: bool,
    /// Whether the file ends with a state write that poisoned the recorded
    /// pool. Replay does not re-run the source of a state write, so this
    /// poison is not reproduced and the replay pool is left unpoisoned.
    pub unverified_poison: bool,
    /// Async replay only: ids that came back during the replay but that no
    /// `Recv` of the file claimed before it ended, in id order. Non-empty
    /// shows a file cut while ids were outstanding (for example a truncated
    /// recording); empty for every other replay.
    pub unclaimed: Vec<u32>,
    /// The index of the next event, for a replay stopped before the end of
    /// the file ([`Replay::run_with`] broke, or a cursor was finished
    /// early); `None` for a replay that reached the end and passed its
    /// end-of-file checks.
    pub stopped_at: Option<u64>,
    /// The keyframe a seek started from ([`Replay::seek`]); `None` for a
    /// replay that started at the beginning of the file.
    pub started_at: Option<u64>,
    /// Async replay only: events skipped because the replay observed a
    /// panic before the file recorded it (worker timing); the file's
    /// `Poisoned` was then reached and the panic reproduced, but these events
    /// were not checked. 0 for every other replay.
    pub unchecked_before_poison: u64,
}

impl ReplaySummary {
    /// Whether the whole file was replayed and passed every check. A
    /// stopped replay never reads as verified (the events after
    /// `stopped_at` were not checked), nor does one that started from a
    /// seek (the events before `started_at` were not replayed), nor one
    /// that skipped events before a panic it observed early
    /// (`unchecked_before_poison`).
    #[must_use]
    pub const fn verified(&self) -> bool {
        self.stopped_at.is_none() && self.started_at.is_none() && self.unchecked_before_poison == 0
    }
}

/// Where the replay stands between events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Expect {
    /// Any call, keyframe or checkpoint.
    Any,
    /// The keyframe of the state write at this event.
    Keyframe(u64),
    /// The `Poisoned` event: the call at this event poisoned the replay pool.
    Poison(u64),
    /// Nothing: the file ended with `Poisoned`.
    Nothing,
}

/// The replay's progress between events.
#[derive(Debug, Clone)]
pub(super) struct State {
    pub(super) summary: ReplaySummary,
    pub(super) expect: Expect,
    last_checkpoint: Option<u64>,
    last_call: Option<u64>,
    /// The ids and branch counter of the pending state write.
    write: Option<(Ids, u64)>,
    /// The archive the caller gave for the file's archive reset source
    /// (ADR-0075 item 7), type-erased: the cursor is not generic.
    pub(super) archive: Option<AnyArchive>,
}

/// An `Archive<E>` without its environment type, for the replay state.
#[derive(Clone)]
pub(super) struct AnyArchive(Arc<dyn Any + Send + Sync>);

impl AnyArchive {
    pub(super) fn new<E: Environment>(archive: &Archive<E>) -> Self {
        Self(Arc::new(archive.clone()))
    }
}

impl core::fmt::Debug for AnyArchive {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("AnyArchive")
    }
}

/// The caller's archive in `st`, if it has one of environment `E`.
pub(super) fn archive_ref<E: Environment>(st: &State) -> Option<&Archive<E>> {
    st.archive.as_ref()?.0.downcast_ref::<Archive<E>>()
}

/// Checks before any write that a keyframe's reset source can be restored:
/// an archive source needs the caller's archive with its digest.
pub(super) fn check_source<E: Environment>(
    source: &ResetSourceRecord,
    archive: Option<&Archive<E>>,
) -> Result<(), ReplayError> {
    match (source, archive) {
        (ResetSourceRecord::Fresh, _) => Ok(()),
        (ResetSourceRecord::Archive { digest, .. }, Some(a)) if a.digest() == *digest => Ok(()),
        (ResetSourceRecord::Archive { digest, .. }, other) => Err(EngineError::ArchiveMismatch {
            expected: *digest,
            found: other.map_or(0, Archive::digest),
        }
        .into()),
    }
}

/// Restores a keyframe's whole pool context, its reset source included,
/// with the caller's archive when the source names one.
pub(super) fn restore_context<E, M>(
    pool: &mut EnvPool<E, M>,
    context: PoolContext,
    archive: Option<&Archive<E>>,
) -> Result<(), ReplayError>
where
    E: Persist + Validate,
    E::Config: CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    M: PersistMask<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    let given = match context.reset_source {
        ResetSourceRecord::Fresh => None,
        ResetSourceRecord::Archive { .. } => archive,
    };
    pool.restore_pool_context(context, given)?;
    Ok(())
}

impl State {
    /// The state at the start of a file, or just after a seek.
    pub(super) fn new() -> Self {
        Self {
            summary: ReplaySummary::default(),
            expect: Expect::Any,
            last_checkpoint: None,
            last_call: None,
            write: None,
            archive: None,
        }
    }

    /// The end-of-file checks: no state write left without its keyframe and
    /// no poisoning call without its `Poisoned` event.
    pub(super) fn end(self) -> Result<ReplaySummary, ReplayError> {
        match self.expect {
            Expect::Any | Expect::Nothing => Ok(self.summary),
            Expect::Keyframe(index) => Err(ReplayError::Malformed {
                event_index: index,
                field: "a state write without its keyframe",
            }),
            Expect::Poison(index) => {
                Err(self.diverged(index, DivergenceKind::UnexpectedPoison, None))
            }
        }
    }

    pub(super) const fn diverged(
        &self,
        event_index: u64,
        kind: DivergenceKind,
        instance: Option<u32>,
    ) -> ReplayError {
        ReplayError::Diverged {
            event_index,
            kind,
            instance,
            last_matching_checkpoint: self.last_checkpoint,
        }
    }
}

impl Replay<'_> {
    /// Replays the file into `pool` and checks it (ADR-0072 section 5): the
    /// pool is checked against the header, the first keyframe is imported
    /// through the validated `import_all` path, every call is re-issued
    /// through the pool's public methods, state-write keyframes are imported
    /// through `import`, and checkpoints and periodic keyframes are compared
    /// with the replayed instances.
    ///
    /// The pool is checked against the header first: identity fields
    /// (ADR-0011), the canonical config bytes, the instance count and the
    /// action layout; the execution mode is free. To rebuild the pool from
    /// the header instead, as ADR-0072 describes, use
    /// [`Replay::run_rebuilt`] (or [`Replay::build_pool`]).
    ///
    /// Known gap: ADR-0072 section 8 says replay streams events. This reads
    /// them from the slice given to [`Replay::open`], so the whole file is
    /// held in memory.
    ///
    /// On a divergence the pool is left as the replay left it, for
    /// inspection.
    ///
    /// # Errors
    ///
    /// [`ReplayError::Persist`] if the pool does not match the header,
    /// [`ReplayError::Engine`] with [`EngineError::PolicyMapMismatch`],
    /// before any event, if the pool's policy map is not the recorded one
    /// ([`super::ReplayHeader::policy_map`]; ADR-0079 item 6, erratum B2:
    /// the caller builds the pool with that map),
    /// [`ReplayError::Diverged`] at the first failing check, and other typed
    /// errors for a malformed event.
    pub fn run<E, M>(&self, pool: &mut EnvPool<E, M>) -> Result<ReplaySummary, ReplayError>
    where
        E: Persist + Validate,
        E::Config: CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        self.run_with(pool, |_, _| core::ops::ControlFlow::Continue(()))
    }

    /// [`Replay::run`] for a file whose pool resets from an archive
    /// (ADR-0075 item 7): `archive` is the archive the file names by digest,
    /// in its header, keyframes and `ResetSource` events. `None` replays a
    /// file of fresh resets only.
    ///
    /// # Errors
    ///
    /// As [`Replay::run`], and [`ReplayError::Engine`] with
    /// [`EngineError::ArchiveMismatch`] when the file names an archive and
    /// `archive` is missing or has another digest.
    pub fn run_with_archive<E, M>(
        &self,
        pool: &mut EnvPool<E, M>,
        archive: Option<&Archive<E>>,
    ) -> Result<ReplaySummary, ReplayError>
    where
        E: Persist + Validate,
        E::Config: CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        let mut cur = self.cursor_with_archive(pool, archive)?;
        while self.advance(pool, &mut cur, |_| false)?.is_some() {}
        self.finish(cur)
    }

    pub(crate) fn check_pool<E, M>(&self, pool: &EnvPool<E, M>) -> Result<(), ReplayError>
    where
        E: Persist,
        E::Config: CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        if self.header.async_pool {
            return Err(PersistError::Incompatible {
                field: "async: use run_async",
            }
            .into());
        }
        self.check_parts::<E, M>(&pool.cfg, &pool.options, pool.action_heads())?;
        // ADR-0079 item 6 (erratum B2): the caller builds the pool with the
        // recorded map; replay verifies it before any event.
        self.check_policy_map(pool.policy_map())?;
        self.check_league(pool)
    }

    /// Checks a league file against the caller-built pool before any event
    /// (ADR-0088 section 8, erratum 3): the pool has a league exactly when
    /// the file does; its current view is the file's first `League` event
    /// (`LeagueMismatch` for another population, `League` for another
    /// table); and every ONNX policy of every `League` event is one the
    /// pool's ONNX opponents serve. Nothing is mutated.
    fn check_league<E, M>(&self, pool: &EnvPool<E, M>) -> Result<(), ReplayError>
    where
        E: Persist,
        E::Config: CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        let league = |reason| -> ReplayError { EngineError::League { reason }.into() };
        let Some(view) = pool.league_view() else {
            if self.header.has_league {
                return Err(league("the file has a league and the pool none"));
            }
            return Ok(());
        };
        if !self.header.has_league {
            return Err(league("the pool has a league and the file none"));
        }
        let mut first = true;
        for item in self.events() {
            let (index, event) = item?;
            let Event::League { body } = event else {
                continue;
            };
            let recorded = super::league::read_view(body, index)?;
            if first {
                first = false;
                if recorded.population_digest() != view.population_digest() {
                    return Err(EngineError::LeagueMismatch {
                        expected: view.population_digest(),
                        found: recorded.population_digest(),
                    }
                    .into());
                }
                if recorded != *view {
                    return Err(league("the pool's league table is not the file's"));
                }
            }
            for kind in recorded.kinds() {
                if let crate::league::OpponentKind::Onnx { content_hash } = *kind
                    && !pool.serves_onnx(&content_hash)
                {
                    return Err(
                        EngineError::OpponentPolicy(crate::league::OpponentPolicyError {
                            reason: "a recorded ONNX policy the pool's ONNX opponents do not serve"
                                .to_owned(),
                        })
                        .into(),
                    );
                }
            }
        }
        Ok(())
    }

    /// Checks a first-party pool's config, options and action heads against
    /// the header.
    pub(super) fn check_parts<E, M>(
        &self,
        cfg: &E::Config,
        options: &crate::options::PoolOptions,
        (discrete, continuous): (usize, usize),
    ) -> Result<(), ReplayError>
    where
        E: Persist,
        E::Config: CanonicalConfig,
        M: PersistMask<E>,
    {
        let incompatible = |field: &'static str| -> Result<(), ReplayError> {
            Err(PersistError::Incompatible { field }.into())
        };
        if self.header.host.is_some() {
            return incompatible("host_protocol");
        }
        self.check_v1(options.chunk_size.get())?;
        self.file
            .check_against(&identity::<E, M>(cfg, options)?, E::TIER, BUILD_ID)?;
        if self.header.num_envs != options.num_envs.get() {
            return incompatible("instance_count");
        }
        if self.header.config != canonical_bytes(cfg).as_slice() {
            return incompatible("config");
        }
        let same = |n: u32, m: usize| usize::try_from(n).is_ok_and(|n| n == m);
        if !same(self.header.agents, E::MAX_AGENTS)
            || !same(self.header.discrete_heads, discrete)
            || !same(self.header.continuous_heads, continuous)
        {
            return incompatible("action_layout");
        }
        if <<E::Header as RecordLayout>::Adaptive as Canonical>::ENCODED_LEN != 0 {
            return Err(ReplayError::AdaptiveUnsupported);
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one match arm per event kind (ADR-0072 section 3)"
    )]
    pub(super) fn apply<E, M>(
        &self,
        pool: &mut EnvPool<E, M>,
        st: &mut State,
        index: u64,
        event: Event<'_>,
    ) -> Result<(), ReplayError>
    where
        E: Persist + Validate,
        E::Config: CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        let malformed = |field| ReplayError::Malformed {
            event_index: index,
            field,
        };
        match (st.expect, &event) {
            (Expect::Nothing, _) => return Err(malformed("an event after Poisoned")),
            (Expect::Poison(_), Event::Poisoned) => {
                st.summary.poisoned = true;
                st.expect = Expect::Nothing;
                return Ok(());
            }
            (Expect::Poison(call), _) => {
                return Err(st.diverged(call, DivergenceKind::UnexpectedPoison, None));
            }
            (Expect::Keyframe(_), Event::Keyframe { .. }) | (Expect::Any, _) => {}
            (Expect::Keyframe(_), Event::Poisoned) => {
                // A state write that poisoned: its source is not re-run.
                st.summary.unverified_poison = true;
                st.expect = Expect::Nothing;
                return Ok(());
            }
            (Expect::Keyframe(_), _) => {
                return Err(malformed("a state write without its keyframe"));
            }
        }
        match event {
            Event::Reset { ids, reseed, .. } => {
                let result = match &ids {
                    Ids::All => pool.reset_all(reseed.as_reseed()).map(|_| ()),
                    Ids::List(ids) => pool.reset(ids, reseed.as_reseed()).map(|_| ()),
                };
                call_outcome(pool, st, index, result.is_ok())
            }
            Event::Step {
                ids,
                discrete,
                continuous_bits,
                ..
            } => {
                if ids != Ids::All {
                    return Err(ReplayError::Unsupported {
                        feature: "subset steps",
                    });
                }
                let continuous: Vec<f32> =
                    continuous_bits.iter().map(|&b| f32::from_bits(b)).collect();
                let result = pool
                    .step(ActionInput {
                        discrete: &discrete,
                        continuous: &continuous,
                    })
                    .map(|_| ());
                call_outcome(pool, st, index, result.is_ok())
            }
            Event::StateWrite {
                op,
                ids,
                branch_counter,
            } => {
                if op.writes_all() && ids != Ids::All {
                    return Err(malformed("restore_all ids"));
                }
                st.summary.calls = st.summary.calls.saturating_add(1);
                st.last_call = Some(index);
                st.expect = Expect::Keyframe(index);
                st.write = Some((ids, branch_counter));
                Ok(())
            }
            Event::Keyframe {
                branch_counter,
                reset_source,
                ids,
                records,
                assignments,
            } => self.apply_keyframe(
                pool,
                st,
                index,
                PoolContext {
                    branch_counter,
                    reset_source,
                    // Replays refuse pools with transforms (ADR-0078 item 8).
                    adaptive: crate::adaptive::AdaptiveContext::default(),
                    // The recorded map's digest (ADR-0079 item 6): the pool's
                    // map was checked against it before the first event.
                    policy_map: self.header.policy_map_digest()?,
                },
                &ids,
                (records, assignments),
            ),
            Event::Adaptive { .. } => Err(not_yet(Tag::Adaptive)),
            // The first view was checked against the pool before any event
            // (`check_league`); a later one replaces the view as the
            // recorded barrier did (ADR-0088 section 8).
            Event::League { .. } if st.summary.keyframes == 0 => Ok(()),
            Event::League { body } => {
                let view = super::league::read_view(body, index)?;
                let league = pool
                    .league
                    .as_mut()
                    .ok_or_else(|| malformed("a League event for a pool without a league"))?;
                league
                    .replace_view(view)
                    .map_err(|_| malformed("a League event whose population shrank"))
            }
            Event::ResetSource(record) => {
                let source = reset_source_of(&record, archive_ref::<E>(st))?;
                let result = pool.set_reset_source(source);
                call_outcome(pool, st, index, result.is_ok())
            }
            Event::ResetFrom { ids, entries } => {
                let result = pool.reset_from(&ids, &entries).map(|_| ());
                call_outcome(pool, st, index, result.is_ok())
            }
            Event::Checkpoint { ids, hashes } => {
                for (id, hash) in ids_of(&ids, self.header.num_envs).into_iter().zip(hashes) {
                    if pool.state_hash(id)? != hash {
                        return Err(st.diverged(index, DivergenceKind::Checkpoint, Some(id)));
                    }
                }
                st.last_checkpoint = Some(index);
                st.summary.checkpoints = st.summary.checkpoints.saturating_add(1);
                Ok(())
            }
            Event::Rejected { .. } => Err(malformed("a Rejected event in a first-party file")),
            Event::Recv { .. } => Err(malformed("a Recv event outside an async file")),
            Event::Poisoned => match st.last_call {
                Some(call) => Err(st.diverged(call, DivergenceKind::NotPoisoned, None)),
                None => Err(malformed("Poisoned before any call")),
            },
        }
    }

    /// The first keyframe is imported with `import_all`; a state write's
    /// keyframe with `import`; a periodic keyframe is compared with the pool.
    fn apply_keyframe<E, M>(
        &self,
        pool: &mut EnvPool<E, M>,
        st: &mut State,
        index: u64,
        mut context: PoolContext,
        ids: &Ids,
        (records, assignments): (&[u8], &[u8]),
    ) -> Result<(), ReplayError>
    where
        E: Persist + Validate,
        E::Config: CanonicalConfig + Sync,
        E::Header: Send,
        E::State: Send,
        M: PersistMask<E>,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        let malformed = |field| ReplayError::Malformed {
            event_index: index,
            field,
        };
        let records = self.decode_records::<E>(index, records)?;
        let assignments = read_assignments(assignments);
        let first = st.summary.keyframes == 0;
        st.summary.keyframes = st.summary.keyframes.saturating_add(1);
        let branch_counter = context.branch_counter;
        if first {
            if *ids != Ids::All {
                return Err(malformed("first keyframe ids"));
            }
            // The header's archive block is the source at recording start:
            // the first keyframe's context must name the same.
            let header_source = self
                .header
                .archive
                .map_or(ResetSourceRecord::Fresh, |a| a.to_record());
            if header_source != context.reset_source {
                return Err(malformed("first keyframe reset source"));
            }
            // Refused before any write: a missing or different archive.
            check_source(&context.reset_source, archive_ref::<E>(st))?;
            // A league file's first keyframe restores each instance's
            // assignment with its record (ADR-0088 erratum 3), as
            // `restore_with_league` does.
            context.adaptive.league = league_context(pool, assignments);
            pool.import_all(&records)
                .map_err(|_| st.diverged(index, DivergenceKind::KeyframeRejected, None))?;
            restore_context(pool, context, archive_ref::<E>(st))?;
            return Ok(());
        }
        if pool.pool_context()?.reset_source != context.reset_source {
            return Err(st.diverged(index, DivergenceKind::ResetSource, None));
        }
        if let Expect::Keyframe(write) = st.expect {
            st.expect = Expect::Any;
            if st.write.take() != Some((ids.clone(), branch_counter)) {
                return Err(malformed("a keyframe that differs from its state write"));
            }
            let id_list = ids_of(ids, self.header.num_envs);
            pool.import(&id_list, &records)
                .map_err(|_| st.diverged(write, DivergenceKind::KeyframeRejected, None))?;
            pool.set_branch_counter(branch_counter)?;
            // The written instances' opponents were redrawn or copied on
            // replay as live: the keyframe's assignments check them.
            return check_assignments(pool, st, write, &id_list, &assignments);
        }
        // A periodic keyframe: check it against the replayed pool.
        if pool.pool_context()?.branch_counter != branch_counter {
            return Err(st.diverged(index, DivergenceKind::BranchCounter, None));
        }
        let id_list = ids_of(ids, self.header.num_envs);
        for (&id, record) in id_list.iter().zip(&records) {
            if pool.state_hash(id)? != StateHash::of(record) {
                return Err(st.diverged(index, DivergenceKind::Keyframe, Some(id)));
            }
        }
        check_assignments(pool, st, index, &id_list, &assignments)
    }

    /// Decodes a keyframe's untrusted records; the import validates them.
    pub(super) fn decode_records<E: Persist>(
        &self,
        index: u64,
        bytes: &[u8],
    ) -> Result<Vec<DecodedRecord<E>>, ReplayError> {
        let len = usize::try_from(self.header.record_len)
            .ok()
            .filter(|&len| len == <DecodedRecord<E> as Canonical>::ENCODED_LEN && len > 0)
            .ok_or(PersistError::Incompatible {
                field: "record_len",
            })?;
        let n = bytes.len() / len;
        let mut out = Vec::new();
        out.try_reserve_exact(n)
            .map_err(|_| ReplayError::Allocation)?;
        for (i, chunk) in bytes.chunks_exact(len).enumerate() {
            let record = decode_from_slice(chunk).map_err(|source| ReplayError::Record {
                event_index: index,
                index: i,
                source,
            })?;
            out.push(record);
        }
        Ok(out)
    }
}

/// A keyframe's assignments (ADR-0088 erratum 3): empty outside a league
/// file. The reader took exactly four bytes per id.
pub(super) fn read_assignments(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&w| u32::from_le_bytes(w))
        .collect()
}

/// The league entry of a first keyframe's context: the pool's current
/// view (the file's latest `League` event) and the keyframe's assignments;
/// `None` for a pool without a league.
pub(super) fn league_context<E, M>(
    pool: &EnvPool<E, M>,
    assignments: Vec<u32>,
) -> Option<Box<crate::league::LeagueContext>>
where
    E: Persist,
    E::Config: CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    M: PersistMask<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    pool.league_view().map(|view| {
        Box::new(crate::league::LeagueContext {
            view: view.clone(),
            assignments,
        })
    })
}

/// Checks the replayed assignments of `ids` against a keyframe's: a
/// difference is a `Keyframe` divergence at `index` naming the instance.
fn check_assignments<E, M>(
    pool: &EnvPool<E, M>,
    st: &State,
    index: u64,
    ids: &[u32],
    recorded: &[u32],
) -> Result<(), ReplayError>
where
    E: Persist,
    E::Config: CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    M: PersistMask<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    let Some(live) = pool.assignment() else {
        return Ok(());
    };
    for (&id, &policy) in ids.iter().zip(recorded) {
        let replayed = usize::try_from(id).ok().and_then(|i| live.get(i)).copied();
        if replayed != Some(policy) {
            return Err(st.diverged(index, DivergenceKind::Keyframe, Some(id)));
        }
    }
    Ok(())
}

/// Records the outcome of a re-issued call: a success, a poison (the next
/// event must be `Poisoned`), or a divergence.
fn call_outcome<E, M>(
    pool: &EnvPool<E, M>,
    st: &mut State,
    index: u64,
    ok: bool,
) -> Result<(), ReplayError>
where
    E: Persist,
    E::Config: CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    M: PersistMask<E>,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    st.summary.calls = st.summary.calls.saturating_add(1);
    st.last_call = Some(index);
    if ok {
        return Ok(());
    }
    if pool.is_poisoned() {
        st.expect = Expect::Poison(index);
        return Ok(());
    }
    Err(st.diverged(index, DivergenceKind::CallFailed, None))
}

/// The pool's reset source for a `ResetSource` event: fresh, or `archive`
/// when its digest is the one recorded.
fn reset_source_of<E: Environment>(
    record: &ResetSourceRecord,
    archive: Option<&Archive<E>>,
) -> Result<ResetSource<E>, ReplayError> {
    // Unreachable after check_archives: kept as defence in depth (rev, #363 B r2).
    check_source(record, archive)?;
    match (record, archive) {
        (ResetSourceRecord::Archive { weights, .. }, Some(a)) => Ok(ResetSource::Archive {
            archive: a.clone(),
            weights: Some(weights.clone()),
        }),
        _ => Ok(ResetSource::Fresh),
    }
}

/// The ids of an event as a list.
fn ids_of(ids: &Ids, num_envs: u32) -> Vec<u32> {
    match ids {
        Ids::All => (0..num_envs).collect(),
        Ids::List(ids) => ids.clone(),
    }
}
