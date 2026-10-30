// SPDX-License-Identifier: MIT OR Apache-2.0

//! Best-effort pinning of worker threads (ADR-0005): a failure is logged
//! and never fatal.

/// Pins the current thread to logical core `core`, if the OS allows it.
#[cfg(not(loom))]
pub(crate) fn pin_current(core: usize) {
    let pinned = core_affinity::set_for_current(core_affinity::CoreId { id: core });
    if !pinned {
        tracing::warn!(core, "could not pin worker thread; it runs unpinned");
    }
}

#[cfg(loom)]
pub(crate) fn pin_current(_: usize) {}
