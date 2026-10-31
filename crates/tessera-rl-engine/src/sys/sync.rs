// SPDX-License-Identifier: MIT OR Apache-2.0

//! Synchronization primitives: `std` normally, `loom` under `cfg(loom)` so
//! the epoch protocol can be model checked.

#[cfg(loom)]
pub(crate) use loom::hint::spin_loop;
#[cfg(loom)]
pub(crate) use loom::sync::Arc;
#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
#[cfg(loom)]
pub(crate) use loom::thread;

#[cfg(not(loom))]
pub(crate) use core::hint::spin_loop;
#[cfg(not(loom))]
pub(crate) use std::sync::Arc;
#[cfg(not(loom))]
pub(crate) use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
#[cfg(not(loom))]
pub(crate) use std::thread;
