// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Async side of an environment object (ADR-0047 section 9): the pool,
//! its stored tokens and the gather buffer behind a binding mutex, under the
//! rules of `split`. The binding always uses `send(ids)`; it has no use for
//! `send_batch`, which only saves a caller the ids a token already holds.
//!
//! A token covers one batch, so stored tokens have disjoint ids and never
//! outnumber the engine's token pool: `recv` cannot starve on them. A send
//! drops the tokens covering its ids; a writer drops them all, because the
//! engine's writers need no token alive at all (ADR-0047 section 3).

use core::any::Any;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tessera_rl_core::Persist;
use tessera_rl_core::config::CanonicalConfig;
use tessera_rl_engine::{DecodeLimits, PersistMask};

use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::ThreadId;

use tessera_rl_core::{AdaptiveValues, Bool8, Environment, RecordLayout};
use tessera_rl_engine::{
    ActionInput, AsyncPool, BatchRow, BatchToken, BatchWaiter, EngineError, IdGenerations,
    MaskProvider, OutputField, OutputStorage, Record, Reseed,
};
use tessera_rl_num::identity::Seed;

use crate::env::{BindError, Layout, Owned, PoolOps, ResetSeed};
use crate::ffi::gather::{GatherBuffer, GatherWriter};
use crate::split::Snapshot;
use crate::view::{ViewSet, view_set};

/// One id of a batch, with what ends its window, captured under the mutex.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BatchId {
    pub(crate) id: u32,
    pub(crate) generation: u64,
    pub(crate) released: u64,
}

/// Per id: how many successful writers dropped a token covering it. A writer
/// drops every stored token, so an indexed view of ids it did not write goes
/// stale here, with no engine generation moved for rows nobody wrote.
pub(crate) type ReleaseEpochs = Arc<[AtomicU64]>;

/// Whether an indexed view's window has ended.
pub(crate) fn ids_stale(
    generations: &IdGenerations,
    ids: &[BatchId],
    epochs: &[AtomicU64],
    unsettled: &AtomicBool,
) -> bool {
    // A row read before this call is not ordered by an Acquire load alone: this
    // fence pairs with the writer's fence after it sets the flag or moves the
    // generation (a seqlock reader). Python reads rows without Rust atomics, so
    // at that boundary this is best effort, not a guarantee.
    core::sync::atomic::fence(Ordering::Acquire);
    unsettled.load(Ordering::Acquire)
        || ids.iter().any(|b| {
            generations.id(b.id) != Some(b.generation)
                || epochs
                    .get(usize::try_from(b.id).unwrap_or(usize::MAX))
                    .is_none_or(|e| e.load(Ordering::Acquire) != b.released)
        })
}

/// Whether a gathered view written at `taken` has ended.
pub(crate) fn gather_stale(buffer: &GatherBuffer, taken: u64, unsettled: &AtomicBool) -> bool {
    // A row read before this call is not ordered by an Acquire load alone: this
    // fence pairs with the writer's fence after it sets the flag or moves the
    // generation (a seqlock reader). Python reads rows without Rust atomics, so
    // at that boundary this is best effort, not a guarantee.
    core::sync::atomic::fence(Ordering::Acquire);
    unsettled.load(Ordering::Acquire) || buffer.generation() != taken
}

/// One batch: its ids in completion order and how to read its rows.
#[derive(Debug, Clone)]
pub(crate) struct BatchOutput {
    /// The delivery number of a `recv` (0 for `reset_all`).
    pub(crate) seq: u64,
    pub(crate) ids: Arc<[BatchId]>,
    /// The gather buffer's generation when written; 0 for N-indexed views.
    pub(crate) gathered: u64,
    pub(crate) owned: Option<Arc<Owned>>,
}

/// What one non-waiting receive found.
#[derive(Debug)]
pub(crate) enum BatchPolled {
    Batch(BatchOutput),
    Running,
    /// Another thread has a delivery it has not acknowledged yet.
    Busy,
}

/// What an Async environment object can do, whatever the environment.
pub(crate) trait AsyncOps: Send + Sync {
    fn storage(&self) -> Arc<OutputStorage>;
    fn id_generations(&self) -> IdGenerations;
    fn release_epochs(&self) -> ReleaseEpochs;
    /// The gather generation and epoch total when the flag last settled.
    #[cfg(test)]
    fn settled_at(&self) -> (u64, u64);
    /// Set while a writer runs or the pool is poisoned; views read it.
    fn unsettled_flag(&self) -> Arc<AtomicBool>;
    fn gather(&self) -> Option<Arc<GatherBuffer>>;
    fn receiving(&self) -> &AtomicBool;
    fn waiter(&self) -> &BatchWaiter;
    fn reset_all(&self, seed: &ResetSeed) -> Result<BatchOutput, BindError>;
    fn reset(&self, ids: &[u32], seed: &ResetSeed, delivered: u64) -> Result<(), BindError>;
    fn send(&self, ids: &[u32], actions: ActionInput<'_>, delivered: u64) -> Result<(), BindError>;
    fn try_recv(&self, delivered: u64) -> Result<BatchPolled, BindError>;
    fn snapshot(&self, id: u32) -> Result<Snapshot, BindError>;
    fn restore(&self, ids: &[u32], snapshots: &[Snapshot], delivered: u64)
    -> Result<(), BindError>;
    fn snapshot_all(&self) -> Result<Snapshot, BindError>;
    fn restore_all(&self, snapshot: &Snapshot) -> Result<(), BindError>;
    /// The persisted file of `ids`; no sent id may be outstanding.
    fn encode(&self, ids: &[u32]) -> Result<Vec<u8>, BindError>;
    /// Loads a persisted file into every instance. A refused file leaves
    /// tokens, views and deliveries as they were.
    fn load(&self, bytes: &[u8], limits: DecodeLimits) -> Result<(), BindError>;
    fn is_poisoned(&self) -> bool;
    /// Drains and joins the workers; idempotent.
    fn close(&self);
}

const NO_SLOT: u32 = u32::MAX;

/// The stored tokens, with the slot of each covered id.
struct Stored<E: Environment, M: MaskProvider<E>> {
    slots: Vec<Option<BatchToken<E, M>>>,
    /// Per id: the slot holding its token, or [`NO_SLOT`].
    owner: Vec<u32>,
    /// Per id: a delivery's views depend on it until a send moves the id's
    /// generation or a successful writer bumps its epoch. Survives a token
    /// released by a refused writer, whose views stay fresh.
    covered: Vec<bool>,
}

impl<E: Environment, M: MaskProvider<E>> Stored<E, M> {
    fn put(&mut self, token: BatchToken<E, M>) {
        let index = self
            .slots
            .iter()
            .position(Option::is_none)
            .unwrap_or_else(|| {
                self.slots.push(None);
                self.slots.len() - 1
            });
        let slot = u32::try_from(index).unwrap_or(NO_SLOT);
        for id in token.ids() {
            let at = usize::try_from(id).unwrap_or(usize::MAX);
            if let Some(o) = self.owner.get_mut(at) {
                *o = slot;
            }
            if let Some(c) = self.covered.get_mut(at) {
                *c = true;
            }
        }
        if let Some(s) = self.slots.get_mut(index) {
            *s = Some(token);
        }
    }

    fn drop_slot(&mut self, slot: u32) {
        let token = self
            .slots
            .get_mut(usize::try_from(slot).unwrap_or(usize::MAX))
            .and_then(Option::take);
        if let Some(token) = token {
            for id in token.ids() {
                if let Some(o) = self
                    .owner
                    .get_mut(usize::try_from(id).unwrap_or(usize::MAX))
                {
                    *o = NO_SLOT;
                }
            }
        }
    }

    /// The FFI release for `ids` (ADR-0047 section 9).
    fn release(&mut self, ids: &[u32]) {
        for &id in ids {
            let slot = self
                .owner
                .get(usize::try_from(id).unwrap_or(usize::MAX))
                .copied()
                .unwrap_or(NO_SLOT);
            if slot != NO_SLOT {
                self.drop_slot(slot);
            }
        }
    }

    fn release_all(&mut self) {
        self.slots.clear();
        self.owner.fill(NO_SLOT);
    }

    /// After a send of `ids`: their generations moved.
    fn uncover(&mut self, ids: &[u32]) {
        for &id in ids {
            if let Some(c) = self
                .covered
                .get_mut(usize::try_from(id).unwrap_or(usize::MAX))
            {
                *c = false;
            }
        }
    }

    /// After a successful writer: every view that depended on a dropped
    /// token goes stale.
    fn bump_covered(&mut self, epochs: &[AtomicU64]) {
        for (c, e) in self.covered.iter_mut().zip(epochs) {
            if core::mem::take(c) {
                e.fetch_add(1, Ordering::AcqRel);
            }
        }
    }
}

struct State<E: Environment, M: MaskProvider<E>> {
    pool: Option<AsyncPool<E, M>>,
    stored: Stored<E, M>,
    /// The last delivery, until acknowledged, and the thread it went to.
    pending: Option<(BatchOutput, ThreadId)>,
    last_seq: u64,
    gather: Option<GatherWriter>,
}

impl<E: Environment, M: MaskProvider<E>> State<E, M> {
    fn acknowledge(&mut self, delivered: u64) {
        if self
            .pending
            .as_ref()
            .is_some_and(|(out, _)| out.seq <= delivered)
        {
            self.pending = None;
        }
    }

    /// A partial writer waits for any unacknowledged delivery: dropping its
    /// token would make the delivery repeat with views already stale.
    fn refuse_while_pending(&self) -> Result<(), BindError> {
        match self.pending.as_ref().and_then(|(out, _)| out.ids.first()) {
            Some(b) => Err(BindError::Engine(EngineError::InFlight { id: b.id })),
            None => Ok(()),
        }
    }

    /// Ids a caller may not know were delivered count as in flight.
    fn check_pending(&self, ids: &[u32]) -> Result<(), BindError> {
        if let Some((out, _)) = &self.pending
            && let Some(&id) = ids.iter().find(|i| out.ids.iter().any(|b| b.id == **i))
        {
            return Err(BindError::Engine(EngineError::InFlight { id }));
        }
        Ok(())
    }
}

/// Records taken from one pool.
struct Taken<E: Environment> {
    owner: Arc<OutputStorage>,
    records: Vec<Record<E>>,
}

struct BatchedPool<E: Environment, M: MaskProvider<E>> {
    state: Mutex<State<E, M>>,
    storage: Arc<OutputStorage>,
    generations: IdGenerations,
    epochs: ReleaseEpochs,
    unsettled: Arc<AtomicBool>,
    #[cfg(test)]
    settled_at: Mutex<(u64, u64)>,
    buffer: Option<Arc<GatherBuffer>>,
    receiving: AtomicBool,
    waiter: BatchWaiter,
    template: Record<E>,
    set: Option<ViewSet>,
    /// Owned mode: fresh copies of each gathered batch.
    owned: bool,
    num_envs: usize,
}

fn closed() -> BindError {
    BindError::Argument("the environment is closed".to_owned())
}

fn f32_bytes(row: &[f32], mut put: impl FnMut(usize, &[u8])) {
    for (i, r) in row.iter().enumerate() {
        put(i * 4, &r.to_le_bytes());
    }
}

fn flag_bytes(row: &[Bool8], mut put: impl FnMut(usize, &[u8])) {
    for (i, f) in row.iter().enumerate() {
        put(i, &[f.to_byte()]);
    }
}

/// Copies `row` in `field` through `put(offset in the row, bytes)`.
fn row_of<E: Environment, M: MaskProvider<E>>(
    row: &BatchRow<'_, E, M>,
    field: OutputField,
    mut put: impl FnMut(usize, &[u8]),
) {
    match field {
        OutputField::Observation(leaf) => {
            if let Some(bytes) = row.observation(leaf) {
                put(0, bytes);
            }
        }
        OutputField::TerminalObservation(leaf) => {
            if let Some(bytes) = row.terminal_observation(leaf) {
                put(0, bytes);
            }
        }
        OutputField::Rewards => f32_bytes(row.rewards().unwrap_or(&[]), put),
        OutputField::AgentTerminated => flag_bytes(row.agent_terminated().unwrap_or(&[]), put),
        OutputField::AgentTruncated => flag_bytes(row.agent_truncated().unwrap_or(&[]), put),
        OutputField::Alive => flag_bytes(row.alive().unwrap_or(&[]), put),
        OutputField::Acting => flag_bytes(row.acting().unwrap_or(&[]), put),
        OutputField::Masks => flag_bytes(row.masks().unwrap_or(&[]), put),
        OutputField::EpisodeTerminated => {
            put(0, &[row.episode_terminated().map_or(0, Bool8::to_byte)]);
        }
        OutputField::EpisodeTruncated => {
            put(0, &[row.episode_truncated().map_or(0, Bool8::to_byte)]);
        }
        _ => {}
    }
}

impl<E, M> BatchedPool<E, M>
where
    E: Environment,
    M: MaskProvider<E>,
    E::Config: Send + Sync,
    E::Header: Send + Sync,
    E::State: Send + Sync,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    fn lock(&self) -> Result<MutexGuard<'_, State<E, M>>, BindError> {
        self.state
            .lock()
            .map_err(|_| BindError::Engine(EngineError::Poisoned))
    }

    /// Views read stale from here until `settle`: a writer rewrites rows
    /// before the engine moves their generations (ADR-0047 section 5). The
    /// fence orders this store before the writer's row writes.
    fn unsettle(&self) {
        self.unsettled.store(true, Ordering::Release);
        core::sync::atomic::fence(Ordering::SeqCst);
    }

    /// Publishes the outcome before the call returns, after anything else a
    /// view reads has moved: the pool's poison keeps views stale.
    fn settle(&self, pool: &AsyncPool<E, M>) {
        #[cfg(test)]
        self.record_settle();
        self.unsettled.store(pool.is_poisoned(), Ordering::Release);
    }

    /// What views read at the moment the flag is cleared.
    #[cfg(test)]
    fn record_settle(&self) {
        let gather = self.buffer.as_ref().map_or(0, |b| b.generation());
        let epochs = self.epochs.iter().map(|e| e.load(Ordering::Acquire)).sum();
        if let Ok(mut at) = self.settled_at.lock() {
            *at = (gather, epochs);
        }
    }

    fn epoch(&self, id: u32) -> u64 {
        self.epochs
            .get(usize::try_from(id).unwrap_or(usize::MAX))
            .map_or(0, |e| e.load(Ordering::Acquire))
    }

    /// Gathers `token` and describes it; the buffer's generation moves first.
    fn output(
        &self,
        gather: &mut Option<GatherWriter>,
        seq: u64,
        token: &BatchToken<E, M>,
    ) -> BatchOutput {
        let ids: Arc<[BatchId]> = token
            .rows()
            .map(|row| BatchId {
                id: row.id(),
                generation: row.generation(),
                released: self.epoch(row.id()),
            })
            .collect();
        let (mut gathered, mut owned) = (0, None);
        if let (Some(writer), Some(set)) = (gather.as_mut(), self.set.as_ref()) {
            gathered = writer.begin();
            let specs = set
                .observations
                .iter()
                .chain(&set.terminal_observations)
                .chain(set.others.iter().map(|(_, spec)| spec));
            let mut sizes = Vec::new();
            for spec in specs {
                let row_bytes = spec
                    .shape
                    .iter()
                    .skip(1)
                    .fold(spec.elem.itemsize, |acc, &d| acc.saturating_mul(d));
                for (k, row) in token.rows().enumerate() {
                    let start = k.saturating_mul(row_bytes);
                    row_of(&row, spec.field, |at, bytes| {
                        let fits = writer.write(spec.field, start.saturating_add(at), bytes);
                        debug_assert!(fits, "a gathered row fits its region");
                    });
                }
                sizes.push((spec.field, row_bytes.saturating_mul(ids.len())));
            }
            if self.owned {
                owned = Some(Arc::new(copy_out(writer, &sizes)));
            }
        }
        BatchOutput {
            seq,
            ids,
            gathered,
            owned,
        }
    }
}

/// Fresh copies of the gathered rows, keyed as `Owned`.
fn copy_out(writer: &GatherWriter, sizes: &[(OutputField, usize)]) -> Owned {
    let mut owned = Owned::default();
    for &(field, len) in sizes {
        let bytes = writer.copy(field, len);
        match field {
            OutputField::Observation(_) => owned.observations.push(bytes),
            OutputField::TerminalObservation(_) => owned.terminal_observations.push(bytes),
            OutputField::Rewards => owned.rewards = bytes,
            OutputField::AgentTerminated => owned.agent_terminated = bytes,
            OutputField::AgentTruncated => owned.agent_truncated = bytes,
            OutputField::Alive => owned.alive = bytes,
            OutputField::Acting => owned.acting = bytes,
            OutputField::EpisodeTerminated => owned.episode_terminated = bytes,
            OutputField::EpisodeTruncated => owned.episode_truncated = bytes,
            OutputField::Masks => owned.masks = bytes,
            _ => {}
        }
    }
    owned
}

fn reseed_of<'a>(seed: &ResetSeed, seeds: &'a mut Vec<Option<Seed>>) -> Reseed<'a> {
    match seed {
        ResetSeed::Keep => Reseed::Keep,
        ResetSeed::Base(s) => Reseed::Base(Seed::new(*s)),
        ResetSeed::List(list) => {
            seeds.extend(list.iter().map(|s| s.map(Seed::new)));
            Reseed::Each(seeds)
        }
    }
}

impl<E, M> AsyncOps for BatchedPool<E, M>
where
    E: Persist + 'static,
    M: PersistMask<E> + Send + Sync + 'static,
    E::Config: CanonicalConfig + Send + Sync,
    E::Header: Send + Sync,
    E::State: Send + Sync,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    fn storage(&self) -> Arc<OutputStorage> {
        Arc::clone(&self.storage)
    }

    fn id_generations(&self) -> IdGenerations {
        self.generations.clone()
    }

    fn release_epochs(&self) -> ReleaseEpochs {
        Arc::clone(&self.epochs)
    }

    #[cfg(test)]
    fn settled_at(&self) -> (u64, u64) {
        self.settled_at.lock().map_or((0, 0), |at| *at)
    }

    fn unsettled_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.unsettled)
    }

    fn gather(&self) -> Option<Arc<GatherBuffer>> {
        self.buffer.clone()
    }

    fn receiving(&self) -> &AtomicBool {
        &self.receiving
    }

    fn waiter(&self) -> &BatchWaiter {
        &self.waiter
    }

    fn reset_all(&self, seed: &ResetSeed) -> Result<BatchOutput, BindError> {
        let mut guard = self.lock()?;
        let state = &mut *guard;
        let pool = state.pool.as_mut().ok_or_else(closed)?;
        state.stored.release_all();
        let mut seeds = Vec::new();
        self.unsettle();
        let token = match pool.reset_all(reseed_of(seed, &mut seeds)) {
            Ok(token) => token,
            Err(e) => {
                self.settle(pool);
                return Err(e.into());
            }
        };
        // Every generation moved, so nothing depends on the dropped tokens.
        state.stored.covered.fill(false);
        // Gathering moves the buffer's generation: old gathered views must
        // not read fresh in between.
        let out = self.output(&mut state.gather, 0, &token);
        self.settle(pool);
        state.stored.put(token);
        state.pending = None;
        Ok(out)
    }

    fn reset(&self, ids: &[u32], seed: &ResetSeed, delivered: u64) -> Result<(), BindError> {
        let mut guard = self.lock()?;
        let state = &mut *guard;
        state.acknowledge(delivered);
        state.refuse_while_pending()?;
        let pool = state.pool.as_mut().ok_or_else(closed)?;
        state.stored.release_all();
        let mut seeds = Vec::new();
        self.unsettle();
        let result = pool.reset(ids, reseed_of(seed, &mut seeds));
        if result.is_ok() {
            state.stored.bump_covered(&self.epochs);
        }
        self.settle(pool);
        result?;
        Ok(())
    }

    fn send(&self, ids: &[u32], actions: ActionInput<'_>, delivered: u64) -> Result<(), BindError> {
        let mut guard = self.lock()?;
        let state = &mut *guard;
        state.acknowledge(delivered);
        state.check_pending(ids)?;
        let pool = state.pool.as_mut().ok_or_else(closed)?;
        state.stored.release(ids);
        let result = pool.send(ids, actions);
        self.settle(pool);
        result?;
        state.stored.uncover(ids);
        Ok(())
    }

    fn try_recv(&self, delivered: u64) -> Result<BatchPolled, BindError> {
        let mut guard = self.lock()?;
        let state = &mut *guard;
        state.acknowledge(delivered);
        let pool = state.pool.as_mut().ok_or_else(closed)?;
        let me = std::thread::current().id();
        // Health first: a cached delivery is not returned after a panic. It is
        // lost; recovery clears it, and the next delivery has a new seq.
        if pool.is_poisoned() {
            let refused = pool.try_recv().map(|_| ());
            self.settle(pool);
            refused?;
            return Err(BindError::Engine(EngineError::Poisoned));
        }
        if let Some((out, owner)) = &state.pending {
            return Ok(if *owner == me {
                BatchPolled::Batch(out.clone())
            } else {
                BatchPolled::Busy
            });
        }
        let received = pool.try_recv();
        self.settle(pool);
        let Some(token) = received? else {
            return Ok(BatchPolled::Running);
        };
        state.last_seq = state.last_seq.saturating_add(1);
        let out = self.output(&mut state.gather, state.last_seq, &token);
        state.stored.put(token);
        state.pending = Some((out.clone(), me));
        Ok(BatchPolled::Batch(out))
    }

    fn snapshot(&self, id: u32) -> Result<Snapshot, BindError> {
        let mut guard = self.lock()?;
        let pool = guard.pool.as_mut().ok_or_else(closed)?;
        let record = pool.snapshot(id)?;
        Ok(Arc::new(Taken::<E> {
            owner: Arc::clone(&self.storage),
            records: vec![record],
        }))
    }

    fn restore(
        &self,
        ids: &[u32],
        snapshots: &[Snapshot],
        delivered: u64,
    ) -> Result<(), BindError> {
        let mut records = Vec::with_capacity(snapshots.len());
        for snapshot in snapshots {
            match self.mine(&**snapshot)?.records.as_slice() {
                [record] => records.push(*record),
                _ => {
                    return Err(BindError::Argument(
                        "restore takes snapshots of one instance".to_owned(),
                    ));
                }
            }
        }
        let mut guard = self.lock()?;
        let state = &mut *guard;
        state.acknowledge(delivered);
        state.refuse_while_pending()?;
        let pool = state.pool.as_mut().ok_or_else(closed)?;
        state.stored.release_all();
        self.unsettle();
        let result = pool.restore(ids, &records);
        if result.is_ok() {
            state.stored.bump_covered(&self.epochs);
        }
        self.settle(pool);
        result?;
        Ok(())
    }

    fn snapshot_all(&self) -> Result<Snapshot, BindError> {
        let guard = self.lock()?;
        let pool = guard.pool.as_ref().ok_or_else(closed)?;
        let mut records = vec![self.template; self.num_envs];
        pool.snapshot_all(&mut records)?;
        Ok(Arc::new(Taken::<E> {
            owner: Arc::clone(&self.storage),
            records,
        }))
    }

    fn restore_all(&self, snapshot: &Snapshot) -> Result<(), BindError> {
        let taken = self.mine(&**snapshot)?;
        let mut guard = self.lock()?;
        let state = &mut *guard;
        let pool = state.pool.as_mut().ok_or_else(closed)?;
        state.stored.release_all();
        let recovering = pool.is_poisoned();
        self.unsettle();
        let result = pool.restore_all(&taken.records);
        if result.is_ok() {
            state.stored.covered.fill(false);
            // This writes no batch: old gathered views must stay stale after
            // a recovery, and move before the flag clears.
            if recovering && let Some(writer) = state.gather.as_mut() {
                writer.begin();
            }
        }
        self.settle(pool);
        result?;
        state.pending = None;
        Ok(())
    }

    fn encode(&self, ids: &[u32]) -> Result<Vec<u8>, BindError> {
        let guard = self.lock()?;
        let pool = guard.pool.as_ref().ok_or_else(closed)?;
        Ok(pool.encode(ids)?)
    }

    fn load(&self, bytes: &[u8], limits: DecodeLimits) -> Result<(), BindError> {
        let mut guard = self.lock()?;
        let state = &mut *guard;
        let pool = state.pool.as_mut().ok_or_else(closed)?;
        // Everything that can refuse the file comes before the tokens go.
        let records = pool.decode(bytes, limits)?;
        pool.check_records(&records)?;
        state.stored.release_all();
        let recovering = pool.is_poisoned();
        self.unsettle();
        let result = pool.import_all(&records);
        if result.is_ok() {
            state.stored.covered.fill(false);
            // As `restore_all`: old gathered views stay stale after a recovery.
            if recovering && let Some(writer) = state.gather.as_mut() {
                writer.begin();
            }
        }
        self.settle(pool);
        result?;
        state.pending = None;
        Ok(())
    }

    fn is_poisoned(&self) -> bool {
        self.lock().map_or(true, |s| {
            s.pool.as_ref().is_some_and(AsyncPool::is_poisoned)
        })
    }

    fn close(&self) {
        let taken = self.lock().ok().map(|mut state| {
            state.pending = None;
            state.stored.release_all();
            state.pool.take()
        });
        // Outside the mutex: this drains and joins the workers.
        drop(taken);
    }
}

impl<E, M> BatchedPool<E, M>
where
    E: Environment + 'static,
    M: MaskProvider<E>,
{
    fn mine<'a>(&self, snapshot: &'a (dyn Any + Send + Sync)) -> Result<&'a Taken<E>, BindError> {
        snapshot
            .downcast_ref::<Taken<E>>()
            .filter(|t| Arc::ptr_eq(&t.owner, &self.storage))
            .ok_or_else(|| {
                BindError::Argument("a snapshot restores only the pool that took it".to_owned())
            })
    }
}

/// An Async environment object behind [`PoolOps`].
struct AsyncHandle {
    ops: Arc<dyn AsyncOps>,
    layout: Layout,
}

fn wrong_mode(call: &'static str) -> BindError {
    BindError::Engine(EngineError::WrongMode {
        call,
        mode: "async",
    })
}

impl PoolOps for AsyncHandle {
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn reset(&mut self, _ids: Option<&[u32]>, _seed: &ResetSeed) -> Result<(), BindError> {
        Err(wrong_mode("reset"))
    }

    fn step(&mut self, _actions: ActionInput<'_>) -> Result<(), BindError> {
        Err(wrong_mode("step"))
    }

    fn outputs(&self) -> Result<Owned, BindError> {
        Ok(Owned::default())
    }

    fn storage(&self) -> Arc<OutputStorage> {
        self.ops.storage()
    }

    fn snapshot(&self, _id: u32) -> Result<Vec<u8>, BindError> {
        Err(wrong_mode("snapshot"))
    }

    fn import(&mut self, _id: u32, _bytes: &[u8]) -> Result<(), BindError> {
        Err(wrong_mode("import_record"))
    }

    fn encode(&self, _ids: &[u32]) -> Result<Vec<u8>, BindError> {
        Err(wrong_mode("encode"))
    }

    fn load(
        &mut self,
        _bytes: &[u8],
        _ids: Option<&[u32]>,
        _limits: tessera_rl_engine::DecodeLimits,
    ) -> Result<(), BindError> {
        Err(wrong_mode("load"))
    }

    fn fork(&mut self, _src: u32, _dst: &[u32]) -> Result<(), BindError> {
        Err(wrong_mode("fork"))
    }

    fn is_poisoned(&self) -> bool {
        self.ops.is_poisoned()
    }

    fn batched(&self) -> Option<Arc<dyn AsyncOps>> {
        Some(Arc::clone(&self.ops))
    }
}

/// How an Async pool hands out outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchOutputs {
    /// Borrowed views of the gather buffer (the default).
    Gathered,
    /// Fresh copies of each gathered batch.
    Owned,
    /// Borrowed views over all N rows.
    Indexed,
}

/// The environment object of an Async `pool`.
pub(crate) fn batched<E, M>(
    pool: AsyncPool<E, M>,
    storage: Arc<OutputStorage>,
    template: Record<E>,
    layout: Layout,
    outputs: BatchOutputs,
) -> Result<Box<dyn PoolOps>, BindError>
where
    E: Persist + 'static,
    M: PersistMask<E> + Send + Sync + 'static,
    E::Config: CanonicalConfig + Send + Sync,
    E::Header: Send + Sync,
    E::State: Send + Sync,
    <E::Header as RecordLayout>::Adaptive: AdaptiveValues<E::Config>,
{
    let num_envs = layout.num_envs;
    let set = view_set(&layout);
    let writer = (outputs != BatchOutputs::Indexed)
        .then(|| {
            let set = set.as_ref().ok_or_else(|| {
                BindError::Argument("an observation dtype has no buffer format".to_owned())
            })?;
            let fields: Vec<(OutputField, usize)> = set
                .observations
                .iter()
                .chain(&set.terminal_observations)
                .chain(set.others.iter().map(|(_, spec)| spec))
                .map(|spec| {
                    let bytes = spec
                        .shape
                        .iter()
                        .fold(spec.elem.itemsize, |acc, &d| acc.saturating_mul(d));
                    (spec.field, bytes)
                })
                .collect();
            Ok::<_, BindError>(GatherWriter::new(&fields))
        })
        .transpose()?;
    let buffer = writer.as_ref().map(GatherWriter::buffer);
    let waiter = pool.waiter();
    let generations = pool.id_generations();
    let ops: Arc<dyn AsyncOps> = Arc::new(BatchedPool {
        state: Mutex::new(State {
            pool: Some(pool),
            stored: Stored {
                slots: Vec::new(),
                owner: vec![NO_SLOT; num_envs],
                covered: vec![false; num_envs],
            },
            pending: None,
            last_seq: 0,
            gather: writer,
        }),
        storage,
        generations,
        epochs: (0..num_envs).map(|_| AtomicU64::new(0)).collect(),
        unsettled: Arc::new(AtomicBool::new(false)),
        #[cfg(test)]
        settled_at: Mutex::new((0, 0)),
        buffer,
        receiving: AtomicBool::new(false),
        waiter,
        template,
        set,
        owned: outputs == BatchOutputs::Owned,
        num_envs,
    });
    Ok(Box::new(AsyncHandle { ops, layout }))
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU32;
    use core::time::Duration;

    use tessera_rl_engine::{EnvPool, NoMasks, PoolOptions};

    use super::*;
    use crate::env::{PoolArgs, finish};
    use crate::split::common::{CountConfig, Counter};

    const N: u32 = 8;

    fn args(pipelined: bool, asynced: Option<(usize, BatchOutputs)>) -> PoolArgs {
        PoolArgs {
            num_envs: N,
            seed: 1,
            max_episode_steps: None,
            auto_reset: "next_step".to_owned(),
            chunk_size: Some(2),
            workers: 2,
            pipelined,
            asynced,
        }
    }

    fn panicking(a: &PoolArgs) -> Box<dyn PoolOps> {
        let cfg = CountConfig {
            panic_at: 1,
            length: 9,
            ..CountConfig::default()
        };
        let mut options = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1));
        options.chunk_size = NonZeroU32::new(2).unwrap();
        let pool = EnvPool::<Counter, NoMasks>::new(cfg.clone(), options).unwrap();
        finish(&cfg, pool, a, 8, false).unwrap()
    }

    fn damaged(file: &[u8]) -> Vec<u8> {
        let mut bad = file.to_vec();
        if let Some(last) = bad.last_mut() {
            *last ^= 1;
        }
        bad
    }

    #[test]
    fn test_async_load_refuses_a_bad_file_and_recovers_from_poison() {
        let ops = panicking(&args(false, Some((2, BatchOutputs::Gathered))));
        let ops = ops.batched().unwrap();
        ops.reset_all(&ResetSeed::Keep).unwrap();
        let ids: Vec<u32> = (0..N).collect();
        let file = ops.encode(&ids).unwrap();
        let limits = tessera_rl_engine::DecodeLimits::default();
        let actions = vec![0; ids.len() * <Counter as Environment>::MAX_AGENTS];
        ops.send(&ids, crate::env::discrete(&actions), 0).unwrap();
        while !ops.is_poisoned() {
            let _ = ops.try_recv(0);
            std::thread::yield_now();
        }
        assert!(ops.encode(&[]).is_err());
        assert!(ops.load(&damaged(&file), limits).is_err());
        assert!(ops.is_poisoned());
        ops.load(&file, limits).unwrap();
        assert!(!ops.is_poisoned());
        assert_eq!(ops.encode(&ids).unwrap(), file);
        ops.close();
    }

    #[test]
    fn test_sync_load_with_every_id_recovers_from_poison_and_with_ids_does_not() {
        let mut ops = panicking(&args(false, None));
        ops.reset(None, &ResetSeed::Keep).unwrap();
        let ids: Vec<u32> = (0..N).collect();
        let file = ops.encode(&ids).unwrap();
        let limits = tessera_rl_engine::DecodeLimits::default();
        let actions = vec![0; ids.len() * <Counter as Environment>::MAX_AGENTS];
        assert!(ops.step(crate::env::discrete(&actions)).is_err());
        assert!(ops.is_poisoned());
        assert!(ops.encode(&[]).is_err());
        assert!(ops.load(&file, Some(&ids), limits).is_err());
        assert!(ops.is_poisoned());
        ops.load(&file, None, limits).unwrap();
        assert!(!ops.is_poisoned());
        assert_eq!(ops.encode(&ids).unwrap(), file);
    }

    #[test]
    fn test_async_panic_poisons_then_reset_all_recovers() {
        let ops = panicking(&args(false, Some((2, BatchOutputs::Gathered))));
        let ops = ops.batched().unwrap();
        ops.reset_all(&ResetSeed::Keep).unwrap();
        let ids: Vec<u32> = (0..N).collect();
        let actions = vec![0; ids.len() * <Counter as Environment>::MAX_AGENTS];
        ops.send(&ids, crate::env::discrete(&actions), 0).unwrap();
        let mut failed = None;
        let mut delivered = 0;
        for _ in 0..10_000 {
            match ops.try_recv(delivered) {
                Ok(BatchPolled::Batch(out)) => delivered = out.seq,
                Ok(BatchPolled::Running) => {
                    let _ = ops.waiter().wait_ready(Duration::from_millis(1));
                }
                Ok(BatchPolled::Busy) => panic!("one thread only"),
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        assert!(matches!(
            failed,
            Some(BindError::Engine(
                EngineError::Panicked | EngineError::Poisoned
            ))
        ));
        assert!(ops.is_poisoned());
        ops.reset_all(&ResetSeed::Keep).unwrap();
        assert!(!ops.is_poisoned());
        ops.close();
    }

    fn asynced(cfg: &CountConfig, outputs: BatchOutputs) -> Arc<dyn AsyncOps> {
        let mut options = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1));
        options.chunk_size = NonZeroU32::new(2).unwrap();
        let pool = EnvPool::<Counter, NoMasks>::new(cfg.clone(), options).unwrap();
        let a = args(false, Some((2, outputs)));
        finish(cfg, pool, &a, 8, false).unwrap().batched().unwrap()
    }

    /// As [`asynced`], with an environment that can panic in `observe`.
    fn asynced_observe_panic() -> Arc<dyn AsyncOps> {
        let cfg = CountConfig::default();
        let mut options = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1));
        options.chunk_size = NonZeroU32::new(2).unwrap();
        let pool = EnvPool::<crate::observe_panic::ObservePanicCounter, NoMasks>::new(
            cfg.clone(),
            options,
        )
        .unwrap();
        let a = args(false, Some((2, BatchOutputs::Indexed)));
        finish(&cfg, pool, &a, 8, false).unwrap().batched().unwrap()
    }

    /// A panic in a writer moves no generation, yet both kinds of view must
    /// report stale over rows it may have partly written.
    #[test]
    fn test_async_flag_is_set_inside_a_writer() {
        let ops = asynced(
            &CountConfig {
                panic_on_reset: true,
                ..CountConfig::default()
            },
            BatchOutputs::Indexed,
        );
        let flag = ops.unsettled_flag();
        let seen = crate::split::reset_probe::flag_at_reset_panic(&flag, || {
            assert!(ops.reset_all(&ResetSeed::Keep).is_err());
        });
        assert_eq!(seen, Some(true), "views must read stale during the write");
        ops.close();
    }

    #[test]
    fn test_async_flag_is_set_inside_restore_all_and_restore() {
        // One pool per writer: the first panic poisons, which keeps the flag.
        let ops = asynced_observe_panic();
        ops.reset_all(&ResetSeed::Keep).unwrap();
        let one = ops.snapshot(5).unwrap();
        let flag = ops.unsettled_flag();
        assert!(!flag.load(Ordering::Acquire));
        let seen = crate::split::reset_probe::flag_at_restore_panic(&flag, || {
            assert!(ops.restore(&[5], &[one], 0).is_err());
        });
        assert_eq!(seen, Some(true), "views must read stale during restore");
        ops.close();
        let ops = asynced_observe_panic();
        ops.reset_all(&ResetSeed::Keep).unwrap();
        let all = ops.snapshot_all().unwrap();
        let flag = ops.unsettled_flag();
        assert!(!flag.load(Ordering::Acquire));
        let seen = crate::split::reset_probe::flag_at_restore_panic(&flag, || {
            assert!(ops.restore_all(&all).is_err());
        });
        assert_eq!(seen, Some(true), "views must read stale during restore_all");
        ops.close();
    }

    #[test]
    fn test_async_panic_during_writer_makes_every_view_stale() {
        let _serial = crate::split::reset_probe::RESET_PANICS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for outputs in [BatchOutputs::Gathered, BatchOutputs::Indexed] {
            let ops = asynced(
                &CountConfig {
                    panic_on_reset: true,
                    ..CountConfig::default()
                },
                outputs,
            );
            let gens = ops.id_generations();
            let epochs = ops.release_epochs();
            let flag = ops.unsettled_flag();
            let ids = [BatchId {
                id: 0,
                generation: gens.id(0).unwrap(),
                released: 0,
            }];
            assert!(!ids_stale(&gens, &ids, &epochs, &flag));
            let buffer = ops.gather();
            let taken = buffer.as_ref().map(|b| b.generation());
            assert!(ops.reset_all(&ResetSeed::Keep).is_err());
            assert!(ops.is_poisoned());
            assert_eq!(gens.id(0), Some(ids[0].generation), "no generation moved");
            assert!(ids_stale(&gens, &ids, &epochs, &flag));
            if let (Some(b), Some(t)) = (&buffer, taken) {
                assert_eq!(b.generation(), t, "nothing gathered");
                assert!(gather_stale(b, t, &flag));
            }
            ops.close();
        }
    }

    /// A step's panic ends every view; each recovery leaves old views stale
    /// and new ones fresh.
    #[test]
    fn test_async_poison_ends_views_and_recovery_keeps_old_ones_stale() {
        for recovery_by_restore in [false, true] {
            for outputs in [BatchOutputs::Gathered, BatchOutputs::Indexed] {
                let ops = asynced(
                    &CountConfig {
                        panic_at: 2,
                        length: 9,
                        ..CountConfig::default()
                    },
                    outputs,
                );
                let first = ops.reset_all(&ResetSeed::Keep).unwrap();
                let taken = ops.snapshot_all().unwrap();
                let (gens, epochs, flag) = (
                    ops.id_generations(),
                    ops.release_epochs(),
                    ops.unsettled_flag(),
                );
                let buffer = ops.gather();
                assert!(!ids_stale(&gens, &first.ids, &epochs, &flag));
                let every: Vec<u32> = (0..N).collect();
                let actions = vec![0; every.len() * <Counter as Environment>::MAX_AGENTS];
                ops.send(&every, crate::env::discrete(&actions), 0).unwrap();
                let mut delivered = 0;
                let mut last = None;
                loop {
                    match ops.try_recv(delivered) {
                        Ok(BatchPolled::Batch(out)) => {
                            delivered = out.seq;
                            let ids: Vec<u32> = out.ids.iter().map(|b| b.id).collect();
                            let actions = vec![0; ids.len() * <Counter as Environment>::MAX_AGENTS];
                            last = Some(out);
                            if ops
                                .send(&ids, crate::env::discrete(&actions), delivered)
                                .is_err()
                            {
                                break;
                            }
                        }
                        Ok(BatchPolled::Running) => {
                            let _ = ops.waiter().wait_ready(Duration::from_millis(1));
                        }
                        Ok(BatchPolled::Busy) => panic!("one thread only"),
                        Err(_) => break,
                    }
                }
                assert!(ops.is_poisoned());
                let old = last.unwrap_or(first);
                let is_stale = |out: &BatchOutput| match &buffer {
                    Some(b) => gather_stale(b, out.gathered, &flag),
                    None => ids_stale(&gens, &out.ids, &epochs, &flag),
                };
                assert!(is_stale(&old));
                if recovery_by_restore {
                    ops.restore_all(&taken).unwrap();
                } else {
                    let fresh = ops.reset_all(&ResetSeed::Keep).unwrap();
                    assert!(!is_stale(&fresh), "new views are fresh");
                }
                assert!(!ops.is_poisoned());
                assert!(is_stale(&old), "old views stay stale");
                ops.close();
            }
        }
    }

    fn deliver(ops: &dyn AsyncOps, ids: &[u32], delivered: u64) -> BatchOutput {
        let actions = vec![0; ids.len() * <Counter as Environment>::MAX_AGENTS];
        ops.send(ids, crate::env::discrete(&actions), delivered)
            .unwrap();
        recv_batch(ops, delivered)
    }

    fn recv_batch(ops: &dyn AsyncOps, delivered: u64) -> BatchOutput {
        loop {
            match ops.try_recv(delivered).unwrap() {
                BatchPolled::Batch(out) => return out,
                BatchPolled::Running => {
                    let _ = ops.waiter().wait_ready(Duration::from_millis(1));
                }
                BatchPolled::Busy => panic!("one thread only"),
            }
        }
    }

    /// Stale views never revive: what ends them has moved before the flag
    /// clears, on every recovery and every writer that drops tokens.
    #[test]
    fn test_async_gather_and_epochs_move_before_the_flag_clears() {
        for by_restore in [false, true] {
            let ops = asynced(
                &CountConfig {
                    panic_at: 1,
                    length: 9,
                    ..CountConfig::default()
                },
                BatchOutputs::Gathered,
            );
            ops.reset_all(&ResetSeed::Keep).unwrap();
            let taken = ops.snapshot_all().unwrap();
            let every: Vec<u32> = (0..N).collect();
            let actions = vec![0; every.len() * <Counter as Environment>::MAX_AGENTS];
            ops.send(&every, crate::env::discrete(&actions), 0).unwrap();
            let mut delivered = 0;
            loop {
                match ops.try_recv(delivered) {
                    Ok(BatchPolled::Batch(out)) => delivered = out.seq,
                    Ok(BatchPolled::Running) => {
                        let _ = ops.waiter().wait_ready(Duration::from_millis(1));
                    }
                    Ok(BatchPolled::Busy) => panic!("one thread only"),
                    Err(_) => break,
                }
            }
            let buffer = ops.gather().unwrap();
            let old = buffer.generation();
            if by_restore {
                ops.restore_all(&taken).unwrap();
            } else {
                ops.reset_all(&ResetSeed::Keep).unwrap();
            }
            assert!(
                ops.settled_at().0 > old,
                "gather moved before the flag cleared"
            );
            ops.close();
        }
        let ops = asynced(&CountConfig::default(), BatchOutputs::Indexed);
        ops.reset_all(&ResetSeed::Keep).unwrap();
        let out = deliver(&*ops, &[0, 1], 0);
        let before = ops.settled_at().1;
        ops.reset(&[4, 5], &ResetSeed::Keep, out.seq).unwrap();
        assert!(
            ops.settled_at().1 > before,
            "epochs moved before the flag cleared"
        );
        // restore(ids), likewise. The reset ids are delivered by the next recv: they
        // fill that batch, so the next one is exactly the sent ids. Were one of them
        // still running, the snapshot would rightly be `NotQuiescent`.
        let out = recv_batch(&*ops, out.seq);
        let reset: Vec<u32> = out.ids.iter().map(|b| b.id).collect();
        assert_eq!(reset, [4, 5]);
        let out = deliver(&*ops, &[0, 1], out.seq);
        let mut sent: Vec<u32> = out.ids.iter().map(|b| b.id).collect();
        sent.sort_unstable();
        assert_eq!(sent, [0, 1]);
        let record = ops.snapshot(6).unwrap();
        let before = ops.settled_at().1;
        ops.restore(&[6], &[record], out.seq).unwrap();
        assert!(
            ops.settled_at().1 > before,
            "restore bumped before the flag cleared"
        );
        ops.close();
    }

    /// Dropping a token ends an indexed view of unwritten ids only when the
    /// writer succeeds, and moves no generation for those rows.
    #[test]
    fn test_async_writer_drop_stales_views_only_on_success() {
        let ops = asynced(&CountConfig::default(), BatchOutputs::Indexed);
        ops.reset_all(&ResetSeed::Keep).unwrap();
        let out = deliver(&*ops, &[0, 1], 0);
        let (gens, epochs, flag) = (
            ops.id_generations(),
            ops.release_epochs(),
            ops.unsettled_flag(),
        );
        assert!(!ids_stale(&gens, &out.ids, &epochs, &flag));
        // Refused after releasing the tokens: an id out of range.
        assert!(ops.reset(&[99], &ResetSeed::Keep, out.seq).is_err());
        assert!(
            !ids_stale(&gens, &out.ids, &epochs, &flag),
            "refused: fresh"
        );
        ops.reset(&[5], &ResetSeed::Keep, out.seq).unwrap();
        assert!(ids_stale(&gens, &out.ids, &epochs, &flag), "dropped: stale");
        assert_eq!(gens.id(0), Some(out.ids[0].generation), "row not written");
        ops.close();
    }

    /// A delivery cached and unacknowledged when another id panics is refused,
    /// not returned (Codex's repro).
    #[test]
    fn test_async_cached_delivery_refuses_poison() {
        let cfg = CountConfig {
            panic_at: 2,
            length: 9,
            ..CountConfig::default()
        };
        let mut options = PoolOptions::new(NonZeroU32::new(N).unwrap(), Seed::new(1));
        options.chunk_size = NonZeroU32::new(2).unwrap();
        let pool = EnvPool::<Counter, NoMasks>::new(cfg.clone(), options).unwrap();
        let a = args(false, Some((1, BatchOutputs::Gathered)));
        let ops = finish(&cfg, pool, &a, 8, false).unwrap().batched().unwrap();
        ops.reset_all(&ResetSeed::Keep).unwrap();
        let receive = |ack| loop {
            match ops.try_recv(ack).unwrap() {
                BatchPolled::Batch(out) => return out,
                BatchPolled::Running => std::thread::yield_now(),
                BatchPolled::Busy => panic!("one thread only"),
            }
        };
        ops.send(&[1], crate::env::discrete(&[0, 0]), 0).unwrap();
        let first = receive(0);
        ops.send(&[0], crate::env::discrete(&[0, 0]), first.seq)
            .unwrap();
        let second = receive(first.seq);
        assert!(second.seq > first.seq);
        ops.send(&[1], crate::env::discrete(&[0, 0]), first.seq)
            .unwrap();
        while !ops.is_poisoned() {
            std::thread::yield_now();
        }
        assert!(matches!(
            ops.try_recv(first.seq),
            Err(BindError::Engine(
                EngineError::Poisoned | EngineError::Panicked
            ))
        ));
        assert!(matches!(
            ops.try_recv(first.seq),
            Err(BindError::Engine(EngineError::Poisoned))
        ));
        ops.reset_all(&ResetSeed::Keep).unwrap();
        ops.send(&[0], crate::env::discrete(&[0, 0]), 0).unwrap();
        assert!(
            receive(0).seq > second.seq,
            "the lost delivery is not replayed"
        );
        ops.close();
    }
}
