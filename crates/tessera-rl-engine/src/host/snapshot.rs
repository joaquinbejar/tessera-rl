// SPDX-License-Identifier: MIT OR Apache-2.0

//! Snapshot, restore, fork and state hash of host pools (ADR-0073 item 4).
//!
//! Only for environments that implement [`HostSnapshot`]. A host pool of
//! any other environment has none of these methods (ADR-0062 item 1).
//!
//! - **Check, then apply.** Every record of a call is checked before any
//!   instance is touched: its identity digest, version and length, the
//!   host header rules of ADR-0068 (r2h in place of r2), then the
//!   environment's `check_snapshot`. A refused call changes nothing.
//! - **Poison.** A poisoned pool refuses every operation here, `restore_all`
//!   included: a panic may have left an instance half-mutated, and `load`
//!   on it is not known to be safe. The recovery is a new pool.
//! - **Pending instances** (ADR-0068 Follow-up). A restored instance whose
//!   header is pending is not observed: zero rows, `pending_reset` set, as
//!   after an aborted reset. Its next call is reset-only.
//! - **Guards.** `save`, `check_snapshot`, `load` and the re-observation run
//!   on the calling thread, in id order, under canonical floating-point
//!   controls and the panic boundary (a panic poisons). Under
//!   `debug-checks` their allocations count toward the host statistic.
//! - **No `fork_branched`.** The engine's branch salt keys `StepRng`, which
//!   a host does not use; it cannot make an author RNG diverge.

use tessera_rl_core::config::{CanonicalConfig, config_hash};
use tessera_rl_core::encoding::StateHasher;
use tessera_rl_core::{HostSnapshot, RecordLayout, StateHash};

use super::HostPool;
use super::kernels::{self, HostHeader};
use super::record::HostRecord;
use super::{MAX_HOST_SNAPSHOT_BYTES, SnapshotParams};
use crate::error::EngineError;
use crate::outputs::OutputsView;
use crate::pool::{clear_instance, clear_terminal, run_kernels, write_agent_masks};
use crate::snapshot::{RecordKind, check_header};
use crate::validate::check_ids;

/// The identity digest of `E` with `cfg` (ADR-0073 item 3): XXH3-64 over
/// the environment id (`u16` length, bytes), `ENV_VERSION`,
/// `SNAPSHOT_VERSION` and the config hash, in canonical encoding.
///
/// # Errors
///
/// [`EngineError::Config`] when the config hash cannot be computed.
pub(crate) fn identity_digest<E>(cfg: &E::Config) -> Result<u64, EngineError>
where
    E: HostSnapshot,
    E::Config: CanonicalConfig,
{
    let hash = config_hash(E::ENV_ID, E::ENV_VERSION, cfg)?;
    let id_len = u16::try_from(E::ENV_ID.len()).unwrap_or(u16::MAX);
    let mut hasher = StateHasher::new();
    hasher.add(&id_len);
    hasher.add_bytes(E::ENV_ID.as_bytes());
    hasher.add(&E::ENV_VERSION);
    hasher.add(&E::SNAPSHOT_VERSION);
    hasher.add(&hash);
    Ok(hasher.finish().get())
}

impl<E, const A: usize> HostPool<E, A>
where
    E: HostSnapshot,
    E::Config: CanonicalConfig,
{
    fn snapshot_index(&self, id: u32) -> Result<usize, EngineError> {
        usize::try_from(id)
            .ok()
            .filter(|&i| i < self.headers.len())
            .ok_or(EngineError::InstanceOutOfRange {
                id,
                num_envs: self.options.num_envs.get(),
            })
    }

    const fn refuse_poisoned(&self) -> Result<(), EngineError> {
        if self.poisoned {
            Err(EngineError::Poisoned)
        } else {
            Ok(())
        }
    }

    /// The pool's snapshot parameters, computed once and kept (review of
    /// #344): `E::max_snapshot_len` runs under the panic boundary (a panic
    /// poisons) and must be at most [`MAX_HOST_SNAPSHOT_BYTES`]; the
    /// identity digest is computed with it. `HostPool::new` cannot compute
    /// them, since it is generic over every `HostEnvironment`, so the first
    /// snapshot operation does.
    fn params(&mut self) -> Result<SnapshotParams, EngineError> {
        if let Some(p) = self.snapshot_params {
            return Ok(p);
        }
        let cfg: &E::Config = &self.cfg;
        let mut max = 0;
        run_kernels(&mut self.poisoned, || max = E::max_snapshot_len(cfg))?;
        if max > MAX_HOST_SNAPSHOT_BYTES {
            return Err(EngineError::SnapshotBound {
                max,
                cap: MAX_HOST_SNAPSHOT_BYTES,
            });
        }
        let params = SnapshotParams {
            identity: identity_digest::<E>(&*self.cfg)?,
            max,
        };
        self.snapshot_params = Some(params);
        Ok(params)
    }

    /// The identity digest records of this pool carry.
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`] on a poisoned pool, and the errors of the
    /// first computation: [`EngineError::Panicked`] if `max_snapshot_len`
    /// panics, [`EngineError::SnapshotBound`] for a bound over
    /// [`MAX_HOST_SNAPSHOT_BYTES`], [`EngineError::Config`] when the config
    /// hash cannot be computed.
    pub fn snapshot_identity(&mut self) -> Result<u64, EngineError> {
        self.refuse_poisoned()?;
        self.params().map(|p| p.identity)
    }

    /// `E::max_snapshot_len` for this pool's config, checked and kept.
    ///
    /// # Errors
    ///
    /// As [`HostPool::snapshot_identity`].
    pub fn max_snapshot_len(&mut self) -> Result<usize, EngineError> {
        self.refuse_poisoned()?;
        self.params().map(|p| p.max)
    }

    /// Writes instance `i`'s record into `out`, reusing its buffer: the
    /// engine reserves `max_snapshot_len` first, so a conforming `save`
    /// never reallocates it.
    fn save_into(
        &mut self,
        i: usize,
        identity: u64,
        max: usize,
        out: &mut HostRecord<A>,
    ) -> Result<(), EngineError> {
        let (Some(header), Some(inst)) = (self.headers.get(i), self.instances.get(i)) else {
            return Err(EngineError::InstanceOutOfRange {
                id: u32::try_from(i).unwrap_or(u32::MAX),
                num_envs: self.options.num_envs.get(),
            });
        };
        out.bytes.clear();
        out.bytes
            .try_reserve(max)
            .map_err(|_| EngineError::SnapshotAllocation { bytes: max })?;
        {
            #[cfg(feature = "debug-checks")]
            let _bracket = crate::sys::alloc_track::HostBracket::enter(&self.allocs);
            let bytes = &mut out.bytes;
            run_kernels(&mut self.poisoned, || inst.save(bytes))?;
        }
        if out.bytes.len() > max {
            self.poisoned = true;
            return Err(EngineError::SnapshotTooLong {
                instance: u32::try_from(i).unwrap_or(u32::MAX),
                len: out.bytes.len(),
                max,
            });
        }
        out.identity = identity;
        out.header = *header;
        out.version = E::SNAPSHOT_VERSION;
        Ok(())
    }

    /// The continuation record of instance `id` (ADR-0073).
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`] on a poisoned pool,
    /// [`EngineError::InstanceOutOfRange`] for an unknown id,
    /// [`EngineError::Panicked`] if `save` panics, or
    /// [`EngineError::SnapshotTooLong`] if it writes more than
    /// `max_snapshot_len`; both poison the pool.
    pub fn snapshot(&mut self, id: u32) -> Result<HostRecord<A>, EngineError> {
        let mut out = HostRecord::new();
        self.snapshot_to(id, &mut out)?;
        Ok(out)
    }

    /// [`HostPool::snapshot`] into an existing record, reusing its buffer.
    ///
    /// # Errors
    ///
    /// As [`HostPool::snapshot`].
    pub fn snapshot_to(&mut self, id: u32, out: &mut HostRecord<A>) -> Result<(), EngineError> {
        self.refuse_poisoned()?;
        let i = self.snapshot_index(id)?;
        let SnapshotParams { identity, max } = self.params()?;
        let result = self.save_into(i, identity, max, out);
        self.count_host_allocations();
        result
    }

    /// Writes the records of every instance into `out`, one per instance,
    /// reusing their buffers.
    ///
    /// # Errors
    ///
    /// [`EngineError::RecordCount`] if `out` has the wrong length, and the
    /// errors of [`HostPool::snapshot`].
    pub fn snapshot_all(&mut self, out: &mut [HostRecord<A>]) -> Result<(), EngineError> {
        self.refuse_poisoned()?;
        self.check_record_count(out.len())?;
        let SnapshotParams { identity, max } = self.params()?;
        let mut result = Ok(());
        for (i, slot) in out.iter_mut().enumerate() {
            result = self.save_into(i, identity, max, slot);
            if result.is_err() {
                break;
            }
        }
        self.count_host_allocations();
        result
    }

    /// The host `StateHash` of instance `id` ([`HostRecord::hash`]).
    ///
    /// # Errors
    ///
    /// As [`HostPool::snapshot`].
    pub fn state_hash(&mut self, id: u32) -> Result<StateHash, EngineError> {
        // Into the pool's scratch record: no allocation once it holds its
        // buffer.
        let mut scratch = core::mem::take(&mut self.snapshot_scratch);
        let result = self.snapshot_to(id, &mut scratch).map(|()| scratch.hash());
        self.snapshot_scratch = scratch;
        result
    }

    /// Restores `record` into instance `id`, after checking it.
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`], [`EngineError::InstanceOutOfRange`],
    /// [`EngineError::InvalidRecord`] or [`EngineError::SnapshotRejected`]
    /// (nothing is mutated then), or [`EngineError::Panicked`] if
    /// environment code panics, which poisons the pool.
    pub fn restore(&mut self, id: u32, record: &HostRecord<A>) -> Result<(), EngineError> {
        self.import(&[id], core::slice::from_ref(record))
    }

    /// Restores `records` into `ids`, in order, after checking every one.
    /// Atomic: on a refused record nothing is mutated.
    ///
    /// # Errors
    ///
    /// Unknown, duplicate or miscounted ids, and the errors of
    /// [`HostPool::restore`].
    pub fn import(&mut self, ids: &[u32], records: &[HostRecord<A>]) -> Result<(), EngineError> {
        self.refuse_poisoned()?;
        check_ids(ids, &mut self.seen)?;
        if ids.len() != records.len() {
            return Err(EngineError::RecordCount {
                records: records.len(),
                instances: ids.len(),
            });
        }
        let indices = ids.iter().filter_map(|&id| usize::try_from(id).ok());
        self.restore_checked(indices, records)
    }

    /// Restores every instance from `records`, one per instance. Unlike
    /// `EnvPool::restore_all` it is no recovery path: a poisoned host pool
    /// refuses it (ADR-0073 item 4).
    ///
    /// # Errors
    ///
    /// [`EngineError::RecordCount`] for the wrong number of records, and
    /// the errors of [`HostPool::restore`].
    pub fn restore_all(&mut self, records: &[HostRecord<A>]) -> Result<(), EngineError> {
        self.refuse_poisoned()?;
        self.check_record_count(records.len())?;
        self.restore_checked(0..records.len(), records)
    }

    /// Copies instance `src` into every instance of `dst`: one `save` into
    /// the pool's scratch record, then a `load` and a header copy for each
    /// destination. The bytes come from this pool, so `check_snapshot` is
    /// not run, but the `max_snapshot_len` bound is.
    ///
    /// # Errors
    ///
    /// [`EngineError::Poisoned`], unknown or duplicate ids (nothing is
    /// mutated then), or the poisoning errors of [`HostPool::snapshot`] and
    /// [`HostPool::restore`].
    pub fn fork(&mut self, src: u32, dst: &[u32]) -> Result<(), EngineError> {
        self.refuse_poisoned()?;
        let i = self.snapshot_index(src)?;
        check_ids(dst, &mut self.seen)?;
        let SnapshotParams { identity, max } = self.params()?;
        let mut scratch = core::mem::take(&mut self.snapshot_scratch);
        let mut result = self.save_into(i, identity, max, &mut scratch);
        if result.is_ok() {
            let indices = dst.iter().filter_map(|&id| usize::try_from(id).ok());
            result = self.apply(indices.map(|k| (k, &scratch)));
        }
        self.snapshot_scratch = scratch;
        self.count_host_allocations();
        result
    }

    fn check_record_count(&self, records: usize) -> Result<(), EngineError> {
        if records == self.headers.len() {
            Ok(())
        } else {
            Err(EngineError::RecordCount {
                records,
                instances: self.headers.len(),
            })
        }
    }

    /// Checks every record (ADR-0073 item 3), then applies them all.
    fn restore_checked(
        &mut self,
        indices: impl Iterator<Item = usize> + Clone,
        records: &[HostRecord<A>],
    ) -> Result<(), EngineError> {
        let SnapshotParams { identity, max } = self.params()?;
        for (index, record) in records.iter().enumerate() {
            self.check_one(index, identity, max, record)?;
        }
        let result = self.apply(indices.zip(records));
        self.count_host_allocations();
        result
    }

    /// The checks of one untrusted record, in order: identity, version,
    /// length, the host header rules, the environment's `check_snapshot`.
    fn check_one(
        &mut self,
        index: usize,
        identity: u64,
        max: usize,
        record: &HostRecord<A>,
    ) -> Result<(), EngineError> {
        let invalid = |reason| Err(EngineError::InvalidRecord { index, reason });
        if record.identity != identity {
            return invalid("the identity digest differs from this pool's");
        }
        if record.version != E::SNAPSHOT_VERSION {
            return invalid("another SNAPSHOT_VERSION");
        }
        if record.bytes.len() > max {
            return invalid("state bytes over max_snapshot_len");
        }
        check_header(
            &record.header,
            &self.options,
            self.layout.mask.count(),
            index,
            RecordKind::Host,
        )?;
        let cfg: &E::Config = &self.cfg;
        let mut verdict = Ok(());
        {
            #[cfg(feature = "debug-checks")]
            let _bracket = crate::sys::alloc_track::HostBracket::enter(&self.allocs);
            run_kernels(&mut self.poisoned, || {
                verdict = E::check_snapshot(cfg, &record.bytes);
            })?;
        }
        verdict.map_err(|reject| EngineError::SnapshotRejected {
            index,
            code: reject.code,
            reason: reject.reason,
        })
    }

    /// Writes checked records into their instances and re-observes them;
    /// then the turn-order check of a `Sequential` environment (a
    /// violation poisons, as after a step).
    fn apply<'r>(
        &mut self,
        pairs: impl Iterator<Item = (usize, &'r HostRecord<A>)>,
    ) -> Result<(), EngineError> {
        let Self {
            headers,
            instances,
            outputs,
            poisoned,
            #[cfg(feature = "debug-checks")]
            allocs,
            ..
        } = self;
        run_kernels(poisoned, || {
            #[cfg(feature = "debug-checks")]
            let _bracket = crate::sys::alloc_track::HostBracket::enter(allocs);
            let outputs = &mut outputs.view_all();
            for (i, record) in pairs {
                let (Some(header), Some(inst)) = (headers.get_mut(i), instances.get_mut(i)) else {
                    continue;
                };
                *header = record.header;
                inst.load(&record.bytes);
                reobserve_one::<E, A>(header, inst, outputs, i);
            }
        })?;
        self.check_turns()
    }

    /// Under `debug-checks`, moves the host allocations counted by the
    /// brackets of this call into the statistic.
    #[cfg_attr(
        not(feature = "debug-checks"),
        expect(
            clippy::unused_self,
            reason = "the statistic exists only under debug-checks"
        )
    )]
    fn count_host_allocations(&mut self) {
        #[cfg(feature = "debug-checks")]
        {
            self.host_allocations = self.host_allocations.saturating_add(self.allocs.take());
        }
    }
}

/// The outputs of a restored instance: per-step outputs, terminal and
/// abort rows cleared; a pending instance gets zero rows and is never
/// observed (ADR-0068 Follow-up); any other is observed, after its acting
/// flags are read from the environment for this state.
fn reobserve_one<E: HostSnapshot, const A: usize>(
    header: &mut HostHeader<A>,
    inst: &E,
    outputs: &mut OutputsView<'_>,
    i: usize,
) {
    clear_instance(outputs, i);
    clear_terminal(outputs, i);
    kernels::clear_abort_row(outputs, i);
    // ADR-0070: refusals are per-call outputs, not state.
    if let Some(row) = outputs.refused.get_mut(i * A..(i + 1) * A) {
        row.fill(tessera_rl_core::Bool8::FALSE);
    }
    if let Some(row) = outputs.refuse_code.get_mut(i * A..(i + 1) * A) {
        row.fill(0);
    }
    if header.engine().pending_reset.get() {
        write_agent_masks(outputs, i, header);
        kernels::zero_rows::<E, A>(outputs, i);
    } else {
        kernels::update_acting::<E, A>(header, inst);
        write_agent_masks(outputs, i, header);
        kernels::observe_instance::<E, A>(header, inst, outputs, i);
    }
    kernels::write_pending(outputs, i, header);
}
