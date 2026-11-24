// SPDX-License-Identifier: MIT OR Apache-2.0

//! Author-provided snapshots of host environments (ADR-0073).
//!
//! A [`HostEnvironment`] owns its state, often on the heap, so the engine
//! cannot copy it. An environment that implements [`HostSnapshot`] writes
//! its continuation state as bytes, and the engine's host pool builds
//! snapshot, restore, fork, state hash and persisted files on those bytes.
//! The bytes are the only representation: one encoding serves every
//! operation, so they cannot disagree.
//!
//! **What the bytes hold** (ADR-0073 item 2): everything in `self` that
//! influences the instance's future outputs and that the engine does not
//! keep: the simulation state with any hidden information (the bytes are
//! privileged, ADR-0035), the environment's own RNG state, any notice
//! recorded by `on_rejected`. They do not hold the engine header, outputs,
//! or what the configuration already determines.
//!
//! **Canonical bytes.** Two instances whose continuation state is equal
//! must save equal bytes: no heap iteration order, no uninitialised bytes,
//! no pointer values. The state hash is computed over them.
//!
//! **Determinism.** Host pools keep no determinism tier (ADR-0062). If the
//! environment's `reset`, `step`, `observe`, `save` and `load` are
//! deterministic and its bytes canonical, an instance restored anywhere
//! continues exactly as the original; the engine guarantees its own half.
//! `save`, `check_snapshot` and `load` must not read the clock, do I/O,
//! depend on the thread or call into Python.
//!
//! Experimental until 1.0 (ADR-0062 item 9).

use crate::host::HostEnvironment;

/// Bytes a host environment refuses to load
/// ([`HostSnapshot::check_snapshot`]): a code and a static reason the
/// engine reports. Making one allocates nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotReject {
    /// The environment's own code.
    pub code: u32,
    /// Why, for the error message.
    pub reason: &'static str,
}

/// A host environment that can save and load its continuation state
/// (ADR-0073).
///
/// The engine calls these methods on the calling thread, one instance at a
/// time in id order, under canonical floating-point controls; a panic in
/// any of them poisons the pool. They are not on the step path and may
/// allocate.
///
/// Restoring is check-then-apply: the engine checks every record of a call
/// (its own header rules, then [`HostSnapshot::check_snapshot`]) before it
/// touches any instance, so a refused restore changes nothing. `load` is
/// infallible.
pub trait HostSnapshot: HostEnvironment {
    /// The version of the bytes [`HostSnapshot::save`] writes, bumped with
    /// any change to them. A record of another version is refused before
    /// [`HostSnapshot::check_snapshot`] sees it.
    const SNAPSHOT_VERSION: u32;

    /// The most bytes [`HostSnapshot::save`] writes for one instance of
    /// this configuration. The engine reserves this much before each
    /// `save`, refuses longer records before reading them, and poisons the
    /// pool if a `save` writes more.
    #[must_use]
    fn max_snapshot_len(cfg: &Self::Config) -> usize;

    /// Appends this instance's continuation state to `out`, which the
    /// engine empties and reserves first. It must be pure: the same state
    /// writes the same bytes.
    fn save(&self, out: &mut Vec<u8>);

    /// Checks `bytes` before any instance is touched. Pure: no `self`, no
    /// I/O, clock or randomness. Every byte string it accepts must be safe
    /// to [`HostSnapshot::load`].
    ///
    /// # Errors
    ///
    /// [`SnapshotReject`] for bytes the environment refuses.
    fn check_snapshot(cfg: &Self::Config, bytes: &[u8]) -> Result<(), SnapshotReject>;

    /// Replaces this instance's state with `bytes`, which passed
    /// [`HostSnapshot::check_snapshot`]. Keeps what the configuration
    /// determines (assets, handles, scratch) and replaces the rest.
    fn load(&mut self, bytes: &[u8]);
}
