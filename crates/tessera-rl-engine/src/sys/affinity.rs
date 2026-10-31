// SPDX-License-Identifier: MIT OR Apache-2.0

//! Best-effort pinning of worker threads (ADR-0005), through the approved
//! `core_affinity` dependency: a failure, or a panic inside the library,
//! is logged and never fatal.

/// Logical core ids the OS reports, if it reports them. On Windows only
/// ids below 64 are usable (`core_affinity` builds a 64-bit mask).
#[cfg(not(loom))]
pub(crate) fn available_cores() -> Option<Vec<usize>> {
    let ids = core_affinity::get_core_ids()?;
    Some(
        ids.into_iter()
            .map(|c| c.id)
            .filter(|&id| !cfg!(windows) || id < 64)
            .collect(),
    )
}

#[cfg(loom)]
pub(crate) fn available_cores() -> Option<Vec<usize>> {
    None
}

/// Pins the current thread to logical core `core`, if the OS allows it.
#[cfg(not(loom))]
pub(crate) fn pin_current(core: usize) {
    let pinned = std::panic::catch_unwind(|| {
        core_affinity::set_for_current(core_affinity::CoreId { id: core })
    });
    if !matches!(pinned, Ok(true)) {
        tracing::warn!(core, "could not pin worker thread; it runs unpinned");
    }
}

#[cfg(loom)]
pub(crate) fn pin_current(_: usize) {}
