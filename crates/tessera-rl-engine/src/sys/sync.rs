// SPDX-License-Identifier: MIT OR Apache-2.0

//! Synchronization primitives: `std` normally, `loom` under `cfg(loom)` so
//! the epoch protocol can be model checked.

#[cfg(loom)]
pub(crate) use loom::cell::UnsafeCell;
#[cfg(loom)]
pub(crate) use loom::hint::spin_loop;
#[cfg(loom)]
pub(crate) use loom::sync::Arc;
#[cfg(loom)]
pub(crate) use loom::sync::atomic::{
    AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering,
};
#[cfg(loom)]
pub(crate) use loom::sync::{Condvar, Mutex};
#[cfg(loom)]
pub(crate) use loom::thread;

#[cfg(not(loom))]
pub(crate) use cell::UnsafeCell;
#[cfg(not(loom))]
pub(crate) use core::hint::spin_loop;
#[cfg(not(loom))]
pub(crate) use std::sync::Arc;
#[cfg(not(loom))]
pub(crate) use std::sync::atomic::{
    AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering,
};
#[cfg(not(loom))]
pub(crate) use std::thread;

/// `core::cell::UnsafeCell` behind loom's `with` and `with_mut` interface,
/// so code that loom checks for overlapping cell accesses runs unchanged
/// natively and under Miri.
#[cfg(not(loom))]
mod cell {
    #[derive(Debug, Default)]
    pub(crate) struct UnsafeCell<T>(core::cell::UnsafeCell<T>);

    impl<T> UnsafeCell<T> {
        pub(crate) const fn new(value: T) -> Self {
            Self(core::cell::UnsafeCell::new(value))
        }

        /// Calls `f` with a pointer for reading the value.
        pub(crate) fn with<R>(&self, f: impl FnOnce(*const T) -> R) -> R {
            f(self.0.get().cast_const())
        }

        /// Calls `f` with a pointer for writing the value.
        pub(crate) fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
            f(self.0.get())
        }
    }
}
