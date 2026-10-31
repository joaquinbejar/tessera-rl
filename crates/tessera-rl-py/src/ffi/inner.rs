// SPDX-License-Identifier: MIT OR Apache-2.0

//! An environment object's state and its reentrancy guard.
//!
//! The object is shared by every Python thread that holds it, and the
//! interpreter is released while the engine runs, so two threads can be in
//! entry points of the same object at once. Each entry therefore gets only
//! `&Inner`; the pool behind it is reachable as `&mut` only after the
//! `busy` flag was acquired, so there is never more than one `&mut` to it
//! (TESSERA-REVIEWER, R-PR17).

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, Ordering};

use super::capi::{CResult, Gil};

use crate::env::PoolOps;
use crate::errors::{Kind, error};

/// An environment object's state.
pub(crate) struct Inner {
    /// Written only while `busy` is held.
    ops: UnsafeCell<Option<Box<dyn PoolOps>>>,
    busy: AtomicBool,
    /// Output mode, fixed at construction: borrowed views or owned copies.
    borrowed: bool,
}

/// Releases the reentrancy guard on drop.
struct Busy<'a>(&'a AtomicBool);

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        // Release: publishes the pool's writes to the next holder.
        self.0.store(false, Ordering::Release);
    }
}

impl Inner {
    /// An open environment around `ops`.
    pub(crate) fn new(ops: Box<dyn PoolOps>, borrowed: bool) -> Self {
        Self {
            ops: UnsafeCell::new(Some(ops)),
            busy: AtomicBool::new(false),
            borrowed,
        }
    }

    /// Whether calls return borrowed views (else owned copies).
    pub(crate) const fn borrowed(&self) -> bool {
        self.borrowed
    }

    fn acquire(&self, g: Gil) -> CResult<Busy<'_>> {
        // Acquire: pairs with the Release in `Busy::drop`.
        if self
            .busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(error(
                g,
                Kind::Concurrency,
                "another call on this environment is running".to_owned(),
            ));
        }
        Ok(Busy(&self.busy))
    }

    /// Runs `f` on the pool under the guard: a concurrent or nested call
    /// raises `ConcurrencyError`, a closed environment `ArgumentError`.
    pub(crate) fn with<R>(
        &self,
        g: Gil,
        f: impl FnOnce(&mut dyn PoolOps) -> CResult<R>,
    ) -> CResult<R> {
        let _busy = self.acquire(g)?;
        // SAFETY: `busy` is held for the lifetime of `_busy`, and every
        // access to `ops` happens under it, so this is the only reference.
        let ops = unsafe { &mut *self.ops.get() };
        let ops = ops
            .as_deref_mut()
            .ok_or_else(|| error(g, Kind::Argument, "the environment is closed".to_owned()))?;
        f(ops)
    }

    /// Takes the pool out under the guard (closing); idempotent.
    pub(crate) fn take(&self, g: Gil) -> CResult<Option<Box<dyn PoolOps>>> {
        let _busy = self.acquire(g)?;
        // SAFETY: as in `with`: `busy` is held.
        Ok(unsafe { &mut *self.ops.get() }.take())
    }
}
