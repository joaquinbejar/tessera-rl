// SPDX-License-Identifier: MIT OR Apache-2.0

//! Process-wide registry of engines with workers (ADR-0005).
//!
//! The registry is one active-engine count. A Rust-only binary uses this
//! crate's own static count. In a Python process every wheel links its own
//! copy of this crate, so the count must be shared across copies: the
//! Python layer attaches the counter it finds or creates at
//! `sys.modules["tessera_rl._registry"]` (#349, `tessera_rl_pyffi`'s
//! `ffi::registry`) with [`attach_shared_registry`], at module
//! initialization, before any engine of that wheel exists.
//!
//! Each registration records the counter it incremented and decrements
//! that one on drop, so a registration made before an attach never moves
//! the shared count. Only construction and drop touch the registry, never
//! the hot path.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The count of this binary, used until a shared counter is attached.
static LOCAL: AtomicUsize = AtomicUsize::new(0);

/// The counter shared by every wheel of the process, once attached.
static SHARED: OnceLock<&'static AtomicUsize> = OnceLock::new();

/// Attaches `counter` as the process-wide active-engine count shared
/// across wheels (ADR-0005, #349). Engines registered afterwards count in
/// it; engines already registered keep their own counter.
///
/// Returns whether `counter` is the attached counter: `true` for the first
/// attach and for a repeated attach of the same counter, `false` when a
/// different counter is already attached (this call is then ignored).
#[must_use]
pub fn attach_shared_registry(counter: &'static AtomicUsize) -> bool {
    core::ptr::eq(*SHARED.get_or_init(|| counter), counter)
}

/// The counter new registrations use.
fn current() -> &'static AtomicUsize {
    SHARED.get().copied().unwrap_or(&LOCAL)
}

/// Registration of one engine with workers; unregisters on drop.
#[derive(Debug)]
pub(crate) struct Registration {
    /// The counter this registration incremented.
    counter: &'static AtomicUsize,
}

impl Registration {
    /// Registers an engine and returns how many were already active.
    pub(crate) fn register() -> (Self, usize) {
        let counter = current();
        let before = counter.fetch_add(1, Ordering::AcqRel);
        (Self { counter }, before)
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}
