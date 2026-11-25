// SPDX-License-Identifier: MIT OR Apache-2.0

//! Synchronization primitives of the registry: `std` normally, `loom` under
//! `cfg(tessera_xla_loom)`, so the step-versus-close protocol can be model checked
//! (ADR-0095 point 3).

#[cfg(tessera_xla_loom)]
pub(crate) use loom::cell::UnsafeCell;
#[cfg(tessera_xla_loom)]
pub(crate) use loom::hint::spin_loop;
#[cfg(tessera_xla_loom)]
pub(crate) use loom::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(tessera_xla_loom)]
pub(crate) use loom::thread::yield_now;

#[cfg(not(tessera_xla_loom))]
pub(crate) use cell::UnsafeCell;
#[cfg(not(tessera_xla_loom))]
pub(crate) use core::hint::spin_loop;
#[cfg(not(tessera_xla_loom))]
pub(crate) use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(not(tessera_xla_loom))]
pub(crate) use std::thread::yield_now;

/// `core::cell::UnsafeCell` behind loom's `with_mut` interface, so the
/// registry runs unchanged natively and under loom.
#[cfg(not(tessera_xla_loom))]
mod cell {
    #[derive(Debug, Default)]
    pub(crate) struct UnsafeCell<T>(core::cell::UnsafeCell<T>);

    impl<T> UnsafeCell<T> {
        pub(crate) const fn new(value: T) -> Self {
            Self(core::cell::UnsafeCell::new(value))
        }

        pub(crate) fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
            f(self.0.get())
        }
    }
}
