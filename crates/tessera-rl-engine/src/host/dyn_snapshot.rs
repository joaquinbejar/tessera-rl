// SPDX-License-Identifier: MIT OR Apache-2.0

//! Snapshots through [`DynHostPool`](super::DynHostPool) (ADR-0073 items 4
//! and 6): a type-erased pool does not know whether its environment
//! implements `HostSnapshot`, so a pool built for one carries a table of
//! its snapshot functions, monomorphised where the bound is known
//! ([`HostPool::with_dyn_snapshots`]). Records cross as bytes
//! ([`HostRecord::to_bytes`]), which is what a host wheel and `tessera_rl`
//! can share.

use tessera_rl_core::HostSnapshot;
use tessera_rl_core::config::CanonicalConfig;

use super::HostPool;
use super::record::{HostRecord, RecordDecodeError};
use crate::error::EngineError;
use crate::persist::{DecodeLimits, PersistError};

/// A pool of `E`, for the function types below.
type Pool<E, const A: usize> = HostPool<E, A>;
type Records = Result<Vec<Vec<u8>>, EngineError>;
type Done = Result<(), EngineError>;
type SnapshotFn<E, const A: usize> = fn(&mut Pool<E, A>, &[u32]) -> Records;
type RestoreFn<E, const A: usize> = fn(&mut Pool<E, A>, &[u32], &[&[u8]]) -> Done;
type RestoreAllFn<E, const A: usize> = fn(&mut Pool<E, A>, &[&[u8]]) -> Done;
type ForkFn<E, const A: usize> = fn(&mut Pool<E, A>, u32, &[u32]) -> Done;
type HashFn<E, const A: usize> = fn(&mut Pool<E, A>, u32) -> Result<u64, EngineError>;
type EncodeFn<E, const A: usize> = fn(&mut Pool<E, A>, &[u32]) -> Result<Vec<u8>, PersistError>;
type ImportFn<E, const A: usize> =
    fn(&mut Pool<E, A>, &[u32], &[u8], DecodeLimits) -> Result<(), PersistError>;

/// The snapshot functions of one `HostPool<E, A>` whose `E` implements
/// `HostSnapshot`.
pub(crate) struct DynSnapshotOps<E: tessera_rl_core::HostEnvironment, const A: usize> {
    pub(crate) snapshot: SnapshotFn<E, A>,
    pub(crate) restore: RestoreFn<E, A>,
    pub(crate) restore_all: RestoreAllFn<E, A>,
    pub(crate) fork: ForkFn<E, A>,
    pub(crate) state_hash: HashFn<E, A>,
    pub(crate) encode: EncodeFn<E, A>,
    pub(crate) import_file: ImportFn<E, A>,
}

impl<E: tessera_rl_core::HostEnvironment, const A: usize> Clone for DynSnapshotOps<E, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: tessera_rl_core::HostEnvironment, const A: usize> Copy for DynSnapshotOps<E, A> {}

impl<E: tessera_rl_core::HostEnvironment, const A: usize> core::fmt::Debug
    for DynSnapshotOps<E, A>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("DynSnapshotOps")
    }
}

/// The reason a record's bytes do not decode, as a record error.
fn decode_reason(e: &RecordDecodeError) -> &'static str {
    match e {
        RecordDecodeError::TrailingBytes => "bytes follow the host record",
        RecordDecodeError::TooLong { .. } => "state bytes over max_snapshot_len",
        RecordDecodeError::Header(_) => "an invalid host record header field",
        RecordDecodeError::Allocation => "could not allocate the host record",
        _ => "a truncated host record",
    }
}

impl<E, const A: usize> HostPool<E, A>
where
    E: HostSnapshot,
    E::Config: CanonicalConfig,
{
    /// Makes this pool's snapshots reachable through
    /// [`DynHostPool`](super::DynHostPool) (ADR-0073 item 4): the
    /// type-erased pool's snapshot methods then work instead of returning
    /// [`EngineError::NoSnapshots`], and [`HostLayout::snapshots`](super::HostLayout)
    /// is true. `register_host_factory!(Env, snapshots)` and
    /// `export_host_env!(Env, .., snapshots)` call it.
    #[must_use]
    pub fn with_dyn_snapshots(mut self) -> Self {
        self.dyn_snapshots = Some(DynSnapshotOps {
            snapshot: |pool, ids| {
                ids.iter()
                    .map(|&id| {
                        pool.snapshot(id)?
                            .to_bytes()
                            .ok_or(EngineError::InvalidRecord {
                                index: 0,
                                reason: "state bytes longer than u32::MAX",
                            })
                    })
                    .collect()
            },
            restore: |pool, ids, records| {
                let records = decode_records::<A>(records, pool.max_snapshot_len()?)?;
                pool.import(ids, &records)
            },
            restore_all: |pool, records| {
                let records = decode_records::<A>(records, pool.max_snapshot_len()?)?;
                pool.restore_all(&records)
            },
            fork: |pool, src, dst| pool.fork(src, dst),
            state_hash: |pool, id| pool.state_hash(id).map(tessera_rl_core::StateHash::get),
            encode: |pool, ids| pool.encode(ids),
            import_file: |pool, ids, bytes, limits| {
                let records = pool.decode(bytes, limits)?;
                pool.import(ids, &records).map_err(PersistError::from)
            },
        });
        self.host_layout.snapshots = true;
        self
    }
}

/// Decodes records from bytes, each within `max` state bytes.
fn decode_records<const A: usize>(
    records: &[&[u8]],
    max: usize,
) -> Result<Vec<HostRecord<A>>, EngineError> {
    records
        .iter()
        .enumerate()
        .map(|(index, bytes)| {
            HostRecord::from_bytes(bytes, max).map_err(|e| EngineError::InvalidRecord {
                index,
                reason: decode_reason(&e),
            })
        })
        .collect()
}
