// SPDX-License-Identifier: MIT OR Apache-2.0

//! Process-wide registry of engines with workers (ADR-0005).
//!
//! Rust programs use one static registry per binary; the Python-level
//! registry shared across wheels comes in phase 1b. Only construction and
//! drop touch it, never the hot path.

use std::sync::atomic::{AtomicUsize, Ordering};

static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// Registration of one engine with workers; unregisters on drop.
#[derive(Debug)]
pub(crate) struct Registration;

impl Registration {
    /// Registers an engine and returns how many were already active.
    pub(crate) fn register() -> (Self, usize) {
        let before = ACTIVE.fetch_add(1, Ordering::AcqRel);
        (Self, before)
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        ACTIVE.fetch_sub(1, Ordering::AcqRel);
    }
}
