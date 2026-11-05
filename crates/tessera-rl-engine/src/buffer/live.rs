// SPDX-License-Identifier: MIT OR Apache-2.0

//! The live flag of a split-phase token (ADR-0047 section 1, invariant 4).
//!
//! A token claims its rows by setting the flag and ends by clearing it; an
//! admission checks it. The orderings are the borrow-window handoff: every
//! read through the token happens before its Release in [`release`], and
//! the admission's Acquire in [`is_live`] (or [`claim`]) sees it before a
//! worker is given the rows. The functions are shared by the mode and its
//! loom models, so the models check the code the mode runs.

use crate::sys::sync::{AtomicBool, Ordering};

/// Claims the rows: `true` if no token covered them. Acquire, so the reads
/// through the new token see the writes of the job that completed before.
pub(crate) fn claim(flag: &AtomicBool) -> bool {
    !flag.swap(true, Ordering::AcqRel)
}

/// Ends the token: every read through it happens before the next claim or
/// admission that sees the flag clear.
pub(crate) fn release(flag: &AtomicBool) {
    flag.store(false, Ordering::Release);
}

/// Whether a token covers the rows (Acquire: pairs with [`release`]).
pub(crate) fn is_live(flag: &AtomicBool) -> bool {
    flag.load(Ordering::Acquire)
}
