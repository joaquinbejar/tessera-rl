// SPDX-License-Identifier: MIT OR Apache-2.0

//! `DynHostPool`: a host pool behind a trait object, for make-by-id
//! (ADR-0065 item 3).
//!
//! It mirrors [`HostPool`]'s public methods one for one, with the same
//! outputs, errors and modes (Inline and Sync). Dispatch costs one virtual
//! call per method call, never per instance or agent: inside, the pool is
//! the monomorphised `HostPool<E, A>`. A method added to `HostPool` is
//! added here too (the mirror test lists both). Experimental until 1.0,
//! like the other host surfaces (ADR-0062 item 9).

use tessera_rl_core::HostEnvironment;
use tessera_rl_core::envs::HostRegistration;

use crate::error::EngineError;
use crate::options::PoolOptions;
use crate::outputs::Outputs;
use crate::pool::Reseed;
use crate::validate::ActionInput;

use super::{HostLayout, HostPool};
use crate::persist::{DecodeLimits, PersistError};

/// A host pool of some environment, without its type (ADR-0065 item 3).
/// Every method is [`HostPool`]'s method of the same name.
pub trait DynHostPool: Send {
    /// [`HostPool::registration`].
    fn registration(&self) -> &'static HostRegistration;
    /// [`HostPool::options`].
    fn options(&self) -> &PoolOptions;
    /// [`HostPool::layout`].
    fn layout(&self) -> &HostLayout;
    /// [`HostPool::workers`].
    fn workers(&self) -> usize;
    /// [`HostPool::pinned_workers`].
    fn pinned_workers(&self) -> bool;
    /// [`HostPool::action_heads`].
    fn action_heads(&self) -> (usize, usize);
    /// [`HostPool::is_poisoned`].
    fn is_poisoned(&self) -> bool;
    /// [`HostPool::aborts`].
    fn aborts(&self) -> u64;
    /// [`HostPool::host_fp_resets`].
    fn host_fp_resets(&self) -> u64;
    /// [`HostPool::allocations_in_host_steps`].
    #[cfg(feature = "debug-checks")]
    fn allocations_in_host_steps(&self) -> u64;
    /// [`HostPool::outputs`].
    ///
    /// # Errors
    ///
    /// As [`HostPool::outputs`].
    fn outputs(&self) -> Result<&Outputs, EngineError>;
    /// [`HostPool::reset_all`].
    ///
    /// # Errors
    ///
    /// As [`HostPool::reset_all`].
    fn reset_all(&mut self, reseed: Reseed<'_>) -> Result<&Outputs, EngineError>;
    /// [`HostPool::reset`].
    ///
    /// # Errors
    ///
    /// As [`HostPool::reset`].
    fn reset(&mut self, ids: &[u32], reseed: Reseed<'_>) -> Result<&Outputs, EngineError>;
    /// [`HostPool::step`].
    ///
    /// # Errors
    ///
    /// As [`HostPool::step`].
    fn step(&mut self, actions: ActionInput<'_>) -> Result<&Outputs, EngineError>;
    /// Whether the snapshot methods below work: the environment implements
    /// `HostSnapshot` and the pool was built with
    /// [`HostPool::with_dyn_snapshots`] (ADR-0073). Without, each returns
    /// [`EngineError::NoSnapshots`].
    fn snapshots(&self) -> bool;
    /// The records of `ids`, each in [`HostRecord`](super::HostRecord)'s
    /// encoding ([`HostPool::snapshot`]).
    ///
    /// # Errors
    ///
    /// As [`HostPool::snapshot`], or [`EngineError::NoSnapshots`].
    fn snapshot_records(&mut self, ids: &[u32]) -> Result<Vec<Vec<u8>>, EngineError>;
    /// Restores encoded records into `ids` ([`HostPool::import`]).
    ///
    /// # Errors
    ///
    /// As [`HostPool::import`]; [`EngineError::InvalidRecord`] for bytes
    /// that are not a record; or [`EngineError::NoSnapshots`].
    fn restore_records(&mut self, ids: &[u32], records: &[&[u8]]) -> Result<(), EngineError>;
    /// Restores every instance from encoded records
    /// ([`HostPool::restore_all`]).
    ///
    /// # Errors
    ///
    /// As [`DynHostPool::restore_records`].
    fn restore_all_records(&mut self, records: &[&[u8]]) -> Result<(), EngineError>;
    /// [`HostPool::fork`].
    ///
    /// # Errors
    ///
    /// As [`HostPool::fork`], or [`EngineError::NoSnapshots`].
    fn fork(&mut self, src: u32, dst: &[u32]) -> Result<(), EngineError>;
    /// [`HostPool::state_hash`], raw.
    ///
    /// # Errors
    ///
    /// As [`HostPool::state_hash`], or [`EngineError::NoSnapshots`].
    fn state_hash(&mut self, id: u32) -> Result<u64, EngineError>;
    /// [`HostPool::encode`]: a `TESHOST` file.
    ///
    /// # Errors
    ///
    /// As [`HostPool::encode`], or [`EngineError::NoSnapshots`].
    fn encode(&mut self, ids: &[u32]) -> Result<Vec<u8>, PersistError>;
    /// [`HostPool::decode`] then [`HostPool::import`] into `ids`.
    ///
    /// # Errors
    ///
    /// As those two, or [`EngineError::NoSnapshots`].
    fn import_file(
        &mut self,
        ids: &[u32],
        bytes: &[u8],
        limits: DecodeLimits,
    ) -> Result<(), PersistError>;
}

/// The pool keeps its configuration (ADR-0073), so a type-erased pool,
/// which is `Send`, needs a `Send` configuration.
impl<E: HostEnvironment, const A: usize> DynHostPool for HostPool<E, A>
where
    E::Config: Send,
{
    fn registration(&self) -> &'static HostRegistration {
        Self::registration(self)
    }
    fn options(&self) -> &PoolOptions {
        Self::options(self)
    }
    fn layout(&self) -> &HostLayout {
        Self::layout(self)
    }
    fn workers(&self) -> usize {
        Self::workers(self)
    }
    fn pinned_workers(&self) -> bool {
        Self::pinned_workers(self)
    }
    fn action_heads(&self) -> (usize, usize) {
        Self::action_heads(self)
    }
    fn is_poisoned(&self) -> bool {
        Self::is_poisoned(self)
    }
    fn aborts(&self) -> u64 {
        Self::aborts(self)
    }
    fn host_fp_resets(&self) -> u64 {
        Self::host_fp_resets(self)
    }
    #[cfg(feature = "debug-checks")]
    fn allocations_in_host_steps(&self) -> u64 {
        Self::allocations_in_host_steps(self)
    }
    fn outputs(&self) -> Result<&Outputs, EngineError> {
        Self::outputs(self)
    }
    fn reset_all(&mut self, reseed: Reseed<'_>) -> Result<&Outputs, EngineError> {
        Self::reset_all(self, reseed)
    }
    fn reset(&mut self, ids: &[u32], reseed: Reseed<'_>) -> Result<&Outputs, EngineError> {
        Self::reset(self, ids, reseed)
    }
    fn step(&mut self, actions: ActionInput<'_>) -> Result<&Outputs, EngineError> {
        Self::step(self, actions)
    }
    fn snapshots(&self) -> bool {
        self.dyn_snapshots.is_some()
    }
    fn snapshot_records(&mut self, ids: &[u32]) -> Result<Vec<Vec<u8>>, EngineError> {
        let ops = self.dyn_snapshots.ok_or(EngineError::NoSnapshots)?;
        (ops.snapshot)(self, ids)
    }
    fn restore_records(&mut self, ids: &[u32], records: &[&[u8]]) -> Result<(), EngineError> {
        let ops = self.dyn_snapshots.ok_or(EngineError::NoSnapshots)?;
        (ops.restore)(self, ids, records)
    }
    fn restore_all_records(&mut self, records: &[&[u8]]) -> Result<(), EngineError> {
        let ops = self.dyn_snapshots.ok_or(EngineError::NoSnapshots)?;
        (ops.restore_all)(self, records)
    }
    fn fork(&mut self, src: u32, dst: &[u32]) -> Result<(), EngineError> {
        let ops = self.dyn_snapshots.ok_or(EngineError::NoSnapshots)?;
        (ops.fork)(self, src, dst)
    }
    fn state_hash(&mut self, id: u32) -> Result<u64, EngineError> {
        let ops = self.dyn_snapshots.ok_or(EngineError::NoSnapshots)?;
        (ops.state_hash)(self, id)
    }
    fn encode(&mut self, ids: &[u32]) -> Result<Vec<u8>, PersistError> {
        let ops = self.dyn_snapshots.ok_or(EngineError::NoSnapshots)?;
        (ops.encode)(self, ids)
    }
    fn import_file(
        &mut self,
        ids: &[u32],
        bytes: &[u8],
        limits: DecodeLimits,
    ) -> Result<(), PersistError> {
        let ops = self.dyn_snapshots.ok_or(EngineError::NoSnapshots)?;
        (ops.import_file)(self, ids, bytes, limits)
    }
}
