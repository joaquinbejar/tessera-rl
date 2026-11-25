// SPDX-License-Identifier: MIT OR Apache-2.0

//! An environment object's state and its reentrancy guard.
//!
//! The object is shared by every Python thread that holds it, and the
//! interpreter is released while the engine runs, so two threads can be in
//! entry points of the same object at once. Each entry therefore gets only
//! `&Inner`; the pool behind it is reachable as `&mut` only after the
//! `busy` flag was acquired, so there is never more than one `&mut` to it
//! (TESSERA-REVIEWER, R-PR17).

#![expect(
    clippy::missing_errors_doc,
    reason = "every `Err(Raised)` here means a Python exception is set: ConcurrencyError, ArgumentError or ForkError"
)]

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::capi::{CResult, Gil};

use crate::errors::{Kind, error};

/// Why the shared state refused a call made without the interpreter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// Another call holds the reentrancy guard.
    Busy,
    /// The pool was taken out (closed).
    Closed,
}

/// The guarded part of an object's state: the pool and its reentrancy
/// guard, shared between the Python object and the XLA FFI registry
/// (ADR-0095), which reaches the pool without the interpreter.
pub struct Shared<T: ?Sized> {
    /// Written only while `busy` is held.
    ops: UnsafeCell<Option<Box<T>>>,
    busy: AtomicBool,
}

// SAFETY: a lock. Every path to `T` (`try_with`, and `Inner`'s `with`,
// `peek` and `take`, which all go through `guard`) acquires `busy` first
// and releases it on return, and no other method reads or writes `ops`, so
// at most one thread reaches `T` at a time; sharing `&Shared` across
// threads then only moves `T` between them, which `T: Send` allows (as
// `Mutex<T>: Sync` for `T: Send`).
unsafe impl<T: ?Sized + Send> Sync for Shared<T> {}

/// Releases the reentrancy guard on drop.
struct Busy<'a>(&'a AtomicBool);

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        // Release: publishes the pool's writes to the next holder.
        self.0.store(false, Ordering::Release);
    }
}

impl<T: ?Sized> Shared<T> {
    fn guard(&self) -> Option<Busy<'_>> {
        // Acquire: pairs with the Release in `Busy::drop`.
        self.busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
            // Lazily: a `Busy` built for a failed acquire would release
            // the holder's guard when dropped.
            .then(|| Busy(&self.busy))
    }

    /// Runs `f` on the pool under the guard, without the interpreter: a
    /// concurrent call is refused with [`Refused::Busy`] (never waited
    /// for), a closed pool with [`Refused::Closed`].
    pub fn try_with<R>(&self, f: impl FnOnce(&mut T) -> R) -> Result<R, Refused> {
        let _busy = self.guard().ok_or(Refused::Busy)?;
        // SAFETY: `busy` is held for the lifetime of `_busy`, and every
        // access to `ops` happens under it, so this is the only reference.
        let ops = unsafe { &mut *self.ops.get() };
        ops.as_deref_mut().map(f).ok_or(Refused::Closed)
    }

    fn with_slot<R>(&self, f: impl FnOnce(&mut Option<Box<T>>) -> R) -> Result<R, Refused> {
        let _busy = self.guard().ok_or(Refused::Busy)?;
        // SAFETY: as in `try_with`: `busy` is held.
        Ok(f(unsafe { &mut *self.ops.get() }))
    }
}

/// An object's state, `T` behind a reentrancy guard: an environment pool,
/// a search or a host pool, as each extension defines it.
pub struct Inner<T: ?Sized> {
    shared: Arc<Shared<T>>,
    /// Output mode, fixed at construction: borrowed views or owned copies.
    borrowed: bool,
    /// The fork generation this object was created in.
    born: u64,
}

fn refusal(g: Gil, why: Refused) -> crate::ffi::capi::Raised {
    match why {
        Refused::Busy => error(
            g,
            Kind::Concurrency,
            "another call on this environment is running".to_owned(),
        ),
        Refused::Closed => error(g, Kind::Argument, "the environment is closed".to_owned()),
    }
}

impl<T: ?Sized> Inner<T> {
    /// An open object around `ops`.
    #[must_use]
    pub fn new(ops: Box<T>, borrowed: bool) -> Self {
        Self {
            shared: Arc::new(Shared {
                ops: UnsafeCell::new(Some(ops)),
                busy: AtomicBool::new(false),
            }),
            borrowed,
            born: super::fork::generation(),
        }
    }

    /// The guarded state, for a holder that runs without the interpreter
    /// (the XLA FFI registry, ADR-0095). It shares this object's guard.
    #[must_use]
    pub fn shared(&self) -> Arc<Shared<T>> {
        Arc::clone(&self.shared)
    }

    /// Whether calls return borrowed views (else owned copies).
    #[must_use]
    pub const fn borrowed(&self) -> bool {
        self.borrowed
    }

    /// Whether this object was created before a fork of this process: its
    /// workers do not exist here, and its locks may be held forever.
    #[must_use]
    pub fn inherited(&self) -> bool {
        self.born != super::fork::generation()
    }

    /// Refuses an inherited object, before touching any of its state.
    pub fn check_fork(&self, g: Gil) -> CResult<()> {
        if self.inherited() {
            return Err(error(g, Kind::Forked, super::fork::REFUSAL.to_owned()));
        }
        Ok(())
    }

    /// Runs `f` on the pool under the guard: a concurrent or nested call
    /// raises `ConcurrencyError`, a closed environment `ArgumentError`.
    pub fn with<R>(&self, g: Gil, f: impl FnOnce(&mut T) -> CResult<R>) -> CResult<R> {
        self.shared.try_with(f).map_err(|why| refusal(g, why))?
    }

    /// Runs `f` on the pool, if still open, under the guard.
    pub fn peek<R>(&self, g: Gil, f: impl FnOnce(Option<&T>) -> R) -> CResult<R> {
        self.shared
            .with_slot(|ops| f(ops.as_deref()))
            .map_err(|why| refusal(g, why))
    }

    /// Takes the pool out under the guard (closing); idempotent.
    pub fn take(&self, g: Gil) -> CResult<Option<Box<T>>> {
        self.shared
            .with_slot(Option::take)
            .map_err(|why| refusal(g, why))
    }
}

#[cfg(test)]
mod tests {
    use super::{Inner, Refused};

    #[test]
    fn test_shared_try_with_refuses_nested_and_closed() {
        let inner = Inner::new(Box::new(5u64), false);
        let shared = inner.shared();
        assert_eq!(shared.try_with(|v| *v + 1), Ok(6));
        // A nested call while the guard is held is refused, never waited for.
        let nested = shared.try_with(|_| inner.shared().try_with(|v| *v));
        assert_eq!(nested, Ok(Err(Refused::Busy)));
        // A refused call leaves the holder's guard held: a second nested
        // call is still refused.
        let twice = shared.try_with(|_| {
            let first = inner.shared().try_with(|v| *v);
            let second = inner.shared().try_with(|v| *v);
            (first, second)
        });
        assert_eq!(twice, Ok((Err(Refused::Busy), Err(Refused::Busy))));
        // Taking the pool out closes it for every holder.
        assert_eq!(
            shared.with_slot(Option::take).map(|b| b.map(|v| *v)),
            Ok(Some(5))
        );
        assert_eq!(shared.try_with(|v| *v), Err(Refused::Closed));
    }

    #[test]
    fn test_shared_outlives_its_inner_and_moves_across_threads() {
        let inner = Inner::new(Box::new(vec![1u8, 2, 3]), true);
        let shared = inner.shared();
        drop(inner);
        let total = std::thread::spawn(move || {
            shared.try_with(|v| v.iter().map(|&x| u32::from(x)).sum::<u32>())
        })
        .join();
        assert!(matches!(total, Ok(Ok(6))));
    }
}
