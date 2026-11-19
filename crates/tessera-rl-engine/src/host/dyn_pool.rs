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
}

impl<E: HostEnvironment, const A: usize> DynHostPool for HostPool<E, A> {
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
}
