// SPDX-License-Identifier: MIT OR Apache-2.0

//! Replaying an async file into an [`AsyncPool`] (ADR-0072 section 5, `Recv`
//! events).
//!
//! # Waiting for the recorded ids
//!
//! Batch membership depends on worker timing (ADR-0006): a `recv` returns
//! the first `batch_size` completed ids, so a replay cannot ask the pool for
//! a recorded batch. Replay therefore needs `batch_size` 1 and waits: at a
//! `Recv` event it receives one id at a time until every recorded id has
//! come back. An id that comes back early is held until the `Recv` that
//! recorded it. This is sound because an id's trajectory depends only on its
//! own sends and resets (ADR-0006), and every id the file sends again was
//! received at a `Recv` before that send. A recorded id that never comes back
//! (it was not sent on replay) is a [`DivergenceKind::RecvMismatch`].
//!
//! # A panic observed early
//!
//! Once a step panicked, every later `recv` (and `send`) of the pool reports
//! the panic, so where a panic is observed depends on worker timing, on
//! record and on replay alike. When replay observes it before the file
//! does, the events up to the recorded `Poisoned` cannot be re-issued on the
//! poisoned pool. Replay then skips them, if each is a `Step`, `Reset`,
//! `Recv`, `Checkpoint` or periodic `Keyframe` (calls the poisoned pool
//! refuses, or checks that need a live pool), and counts them in
//! [`ReplaySummary::unchecked_before_poison`]; such a replay is not
//! [verified](ReplaySummary::verified). Any other event in between (a state
//! write, an adaptive or reset source event), or a file with no `Poisoned`,
//! is a [`DivergenceKind::UnexpectedPoison`]. The file does not record which
//! instance panicked, so replay checks that a panic occurred, not its cause.

use tessera_rl_core::config::CanonicalConfig;
use tessera_rl_core::encoding::{HashSink, Sink};
use tessera_rl_core::{AdaptiveValues, Persist, RecordLayout, StateHash};

use crate::error::EngineError;
use crate::masks::PersistMask;
use crate::modes::async_mode::AsyncPool;
use crate::persist::PersistError;
use crate::validate::ActionInput;

use super::async_writer::token_rows;
use super::format::EventTag as Tag;
use super::format::StateWriteOp;
use super::reader::{Event, Ids, Replay};
use super::run::ReplaySummary;
use super::v1_fields::{fresh_only, not_yet};
use super::{DivergenceKind, ReplayError};

/// Where the replay stands between events.
struct Run {
    summary: ReplaySummary,
    last_checkpoint: Option<u64>,
    last_call: Option<u64>,
    awaiting_poison: Option<u64>,
    /// The state write whose keyframe comes next.
    write: Option<(u64, StateWriteOp, Ids)>,
    /// Ids received ahead of their `Recv`, with their output rows (empty when
    /// the file has no outputs hash).
    early: Vec<Option<Vec<u8>>>,
}

impl Run {
    const fn diverged(
        &self,
        index: u64,
        kind: DivergenceKind,
        instance: Option<u32>,
    ) -> ReplayError {
        ReplayError::Diverged {
            event_index: index,
            kind,
            instance,
            last_matching_checkpoint: self.last_checkpoint,
        }
    }
}

impl Replay<'_> {
    /// Replays an async file into `pool` and checks it. The pool is checked
    /// against the header as by [`Replay::run`], and must have `batch_size`
    /// 1 (see the module notes): batch size changes how ids are grouped into
    /// batches, not their results, and only single-id batches let replay
    /// wait for any recorded batch. Its worker count is free. The first
    /// keyframe is imported through `import_all`; each `Step` is re-sent
    /// with its ids, each `Reset` re-issued (a `reset_all` token is dropped at
    /// once); each `Recv` waits for exactly its recorded ids and compares
    /// their outputs hash when the file has one; state-write keyframes go
    /// through the validated import; periodic keyframes and checkpoints,
    /// taken at quiescent points when recording, are compared there.
    ///
    /// The whole file is held in memory (ADR-0072 section 8, a known gap).
    ///
    /// # Errors
    ///
    /// [`ReplayError::Persist`] for a file that is not async or does not
    /// match the pool, [`ReplayError::Diverged`] at the first failing check
    /// ([`DivergenceKind::RecvMismatch`] for a `Recv` that cannot deliver its
    /// ids), and other typed errors for a malformed event.
    pub fn run_async<E, M>(&self, pool: &mut AsyncPool<E, M>) -> Result<ReplaySummary, ReplayError>
    where
        E: Persist,
        E::Config: CanonicalConfig + Send + Sync,
        E::Header: Send + Sync,
        E::State: Send + Sync,
        M: PersistMask<E> + Send + Sync,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        let incompatible = |field: &'static str| -> Result<ReplaySummary, ReplayError> {
            Err(PersistError::Incompatible { field }.into())
        };
        if !self.header.async_pool {
            return incompatible("async: the file was not recorded on an AsyncPool");
        }
        let (cfg, options, heads) = pool.replay_parts();
        self.check_parts::<E, M>(cfg, options, heads)?;
        if self.header.archive.is_some() {
            return Err(ReplayError::Unsupported {
                feature: "archive reset sources in async replay files (ADR-0075)",
            });
        }
        if pool.batch_size() != 1 {
            return incompatible("batch_size: an async replay needs batch_size 1");
        }
        let n = usize::try_from(pool.num_envs()).map_err(|_| ReplayError::Allocation)?;
        let mut run = Run {
            summary: ReplaySummary::default(),
            last_checkpoint: None,
            last_call: None,
            awaiting_poison: None,
            write: None,
            early: (0..n).map(|_| None).collect(),
        };
        for item in self.events() {
            let (index, event) = item?;
            if run.summary.poisoned {
                return Err(ReplayError::Malformed {
                    event_index: index,
                    field: "an event after Poisoned",
                });
            }
            if let Some(call) = run.awaiting_poison {
                match event {
                    Event::Poisoned => {
                        run.awaiting_poison = None;
                        run.summary.poisoned = true;
                    }
                    // Observed early: the recording saw the panic later.
                    Event::Step { .. }
                    | Event::Reset { .. }
                    | Event::Recv { .. }
                    | Event::Checkpoint { .. }
                    | Event::Keyframe { .. } => {
                        run.summary.unchecked_before_poison =
                            run.summary.unchecked_before_poison.saturating_add(1);
                    }
                    _ => {
                        return Err(run.diverged(call, DivergenceKind::UnexpectedPoison, None));
                    }
                }
            } else {
                self.apply_async(pool, &mut run, index, event)?;
            }
            run.summary.events = index.saturating_add(1);
        }
        if let Some(call) = run.awaiting_poison {
            return Err(run.diverged(call, DivergenceKind::UnexpectedPoison, None));
        }
        if let Some((write, _, _)) = run.write {
            return Err(ReplayError::Malformed {
                event_index: write,
                field: "a state write without its keyframe",
            });
        }
        // Receive what is still in flight, so ids the file sent but never
        // claimed show up in `unclaimed` whatever the timing (a file cut while
        // ids were outstanding).
        if !run.summary.poisoned {
            while let Ok(token) = pool.recv() {
                for id in token.ids() {
                    if let Some(held) = usize::try_from(id).ok().and_then(|i| run.early.get_mut(i))
                    {
                        *held = Some(Vec::new());
                    }
                }
            }
        }
        run.summary.unclaimed = run
            .early
            .iter()
            .zip(0u32..)
            .filter_map(|(held, id)| held.as_ref().map(|_| id))
            .collect();
        Ok(run.summary)
    }

    fn apply_async<E, M>(
        &self,
        pool: &mut AsyncPool<E, M>,
        run: &mut Run,
        index: u64,
        event: Event<'_>,
    ) -> Result<(), ReplayError>
    where
        E: Persist,
        E::Config: CanonicalConfig + Send + Sync,
        E::Header: Send + Sync,
        E::State: Send + Sync,
        M: PersistMask<E> + Send + Sync,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        let malformed = |field| ReplayError::Malformed {
            event_index: index,
            field,
        };
        if run.write.is_some() && !matches!(event, Event::Keyframe { .. }) {
            return Err(malformed("a state write without its keyframe"));
        }
        let result = match event {
            Event::Keyframe {
                ids,
                records,
                reset_source,
                ..
            } => {
                fresh_only(&reset_source)?;
                return self.async_keyframe(pool, run, index, &ids, records);
            }
            Event::Adaptive { .. } => return Err(not_yet(Tag::Adaptive)),
            Event::League { .. } => return Err(not_yet(Tag::League)),
            Event::ResetSource(_) => return Err(not_yet(Tag::ResetSource)),
            Event::ResetFrom { .. } => return Err(not_yet(Tag::ResetFrom)),
            Event::Checkpoint { ids, hashes } => {
                for (id, hash) in ids_of(&ids, run.early.len()).into_iter().zip(hashes) {
                    if pool.snapshot(id)?.hash() != hash {
                        return Err(run.diverged(index, DivergenceKind::Checkpoint, Some(id)));
                    }
                }
                run.last_checkpoint = Some(index);
                run.summary.checkpoints = run.summary.checkpoints.saturating_add(1);
                return Ok(());
            }
            Event::Poisoned => {
                return Err(match run.last_call {
                    Some(call) => run.diverged(call, DivergenceKind::NotPoisoned, None),
                    None => malformed("Poisoned before any call"),
                });
            }
            Event::Rejected { .. } => {
                return Err(malformed("a Rejected event in a first-party file"));
            }
            Event::StateWrite { op, ids, .. } => {
                run.write = Some((index, op, ids));
                Ok(())
            }
            Event::Reset { ids, reseed, .. } => match &ids {
                Ids::All => pool.reset_all(reseed.as_reseed()).map(drop),
                Ids::List(ids) => pool.reset(ids, reseed.as_reseed()),
            },
            Event::Step {
                ids,
                discrete,
                continuous_bits,
                ..
            } => {
                let Ids::List(ids) = ids else {
                    return Err(malformed("async step ids"));
                };
                let continuous: Vec<f32> =
                    continuous_bits.iter().map(|&b| f32::from_bits(b)).collect();
                pool.send(
                    &ids,
                    ActionInput {
                        discrete: &discrete,
                        continuous: &continuous,
                    },
                )
            }
            Event::Recv { ids, outputs_hash } => {
                return Self::async_recv(pool, run, index, &ids, outputs_hash);
            }
        };
        run.summary.calls = run.summary.calls.saturating_add(1);
        run.last_call = Some(index);
        match result {
            Ok(()) => Ok(()),
            Err(_) if pool.is_poisoned() => {
                run.awaiting_poison = Some(index);
                Ok(())
            }
            Err(_) => Err(run.diverged(index, DivergenceKind::CallFailed, None)),
        }
    }

    /// Receives one id at a time until every id of `ids` came back; checks
    /// their outputs hash. Empty `ids`: the recorded `recv` poisoned.
    fn async_recv<E, M>(
        pool: &mut AsyncPool<E, M>,
        run: &mut Run,
        index: u64,
        ids: &[u32],
        recorded: Option<u64>,
    ) -> Result<(), ReplayError>
    where
        E: Persist,
        E::Config: CanonicalConfig + Send + Sync,
        E::Header: Send + Sync,
        E::State: Send + Sync,
        M: PersistMask<E> + Send + Sync,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        run.summary.calls = run.summary.calls.saturating_add(1);
        run.last_call = Some(index);
        let missing = |run: &Run| {
            ids.iter()
                .copied()
                .find(|&id| slot(&run.early, id).is_none_or(Option::is_none))
        };
        // An empty recorded batch is a recv that reported a panic: receive
        // until the pool reports it.
        while ids.is_empty() || missing(run).is_some() {
            match pool.recv() {
                Ok(token) => {
                    for id in token.ids() {
                        let mut rows = Vec::new();
                        if recorded.is_some() {
                            token_rows(&token, id, &mut rows);
                        }
                        if let Some(held) =
                            usize::try_from(id).ok().and_then(|i| run.early.get_mut(i))
                        {
                            *held = Some(rows);
                        }
                    }
                }
                Err(_) if pool.is_poisoned() => {
                    run.awaiting_poison = Some(index);
                    return Ok(());
                }
                Err(_) => {
                    return Err(run.diverged(index, DivergenceKind::RecvMismatch, missing(run)));
                }
            }
        }
        let mut sink = HashSink::new();
        for &id in ids {
            let rows = usize::try_from(id)
                .ok()
                .and_then(|i| run.early.get_mut(i))
                .and_then(Option::take)
                .unwrap_or_default();
            sink.write(&rows);
        }
        if let Some(recorded) = recorded
            && sink.digest() != recorded
        {
            return Err(run.diverged(index, DivergenceKind::OutputsMismatch, None));
        }
        Ok(())
    }

    /// The first keyframe is imported with `import_all`; a state write's
    /// keyframe with the validated import of its ids; a periodic keyframe is
    /// compared with the pool.
    fn async_keyframe<E, M>(
        &self,
        pool: &mut AsyncPool<E, M>,
        run: &mut Run,
        index: u64,
        ids: &Ids,
        bytes: &[u8],
    ) -> Result<(), ReplayError>
    where
        E: Persist,
        E::Config: CanonicalConfig + Send + Sync,
        E::Header: Send + Sync,
        E::State: Send + Sync,
        M: PersistMask<E> + Send + Sync,
        <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
    {
        let records = self.decode_records::<E>(index, bytes)?;
        let id_list = ids_of(ids, run.early.len());
        run.summary.keyframes = run.summary.keyframes.saturating_add(1);
        if index == 0 {
            pool.import_all(&records)
                .map_err(|_| run.diverged(index, DivergenceKind::KeyframeRejected, None))?;
            return Ok(());
        }
        if let Some((write, op, written)) = run.write.take() {
            if written != *ids {
                return Err(ReplayError::Malformed {
                    event_index: index,
                    field: "a keyframe that differs from its state write",
                });
            }
            let result = if op.writes_all() {
                pool.import_all(&records)
            } else {
                pool.import(&id_list, &records)
            };
            return match result {
                Ok(()) => Ok(()),
                Err(EngineError::InvalidRecord { .. } | EngineError::Invariant { .. }) => {
                    Err(run.diverged(write, DivergenceKind::KeyframeRejected, None))
                }
                Err(_) => Err(run.diverged(write, DivergenceKind::CallFailed, None)),
            };
        }
        for (id, record) in id_list.into_iter().zip(&records) {
            if pool.snapshot(id)?.hash() != StateHash::of(record) {
                return Err(run.diverged(index, DivergenceKind::Keyframe, Some(id)));
            }
        }
        Ok(())
    }
}

/// The held rows of `id`, if `id` is in range.
fn slot(early: &[Option<Vec<u8>>], id: u32) -> Option<&Option<Vec<u8>>> {
    usize::try_from(id).ok().and_then(|i| early.get(i))
}

/// The ids of an event as a list.
fn ids_of(ids: &Ids, n: usize) -> Vec<u32> {
    match ids {
        Ids::All => (0..u32::try_from(n).unwrap_or(0)).collect(),
        Ids::List(ids) => ids.clone(),
    }
}
