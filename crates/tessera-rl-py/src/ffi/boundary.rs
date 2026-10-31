// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Rust side of the native floating-point entry boundary (ADR-0041,
//! ADR-0037).
//!
//! Every C entry goes through a thunk (`thunks`) that saves the caller's
//! floating-point controls, canonicalizes them, and calls
//! [`tessera_fp_enter`] before the first Rust frame of the entry, then
//! [`tessera_fp_leave`] after it returns and before restoring the caller's
//! controls. These two record the entry in a per-thread frame, so the entry
//! body knows whether its caller's controls were canonical and what they
//! were. Frames nest (a finalizer or a warning hook may re-enter).
//!
//! The policy applies in `ffi::entry`, under canonical controls:
//! `canonicalize` (the default) proceeds and warns once per process;
//! `reject` raises `FloatingPointEnvironmentError` before any work.
//! Cleanup entries (deallocation, capsule destructor, DLPack deleters)
//! never consult it: they always clean up.

use core::cell::Cell;
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use tessera_rl_num::fp;

use super::capi::{self, CResult, Gil};
use crate::errors::{Kind, error};

// A frame token: bit 0 marks a live frame, bit 1 a non-canonical caller,
// bits 2.. hold the caller's control register (32 bits on both targets:
// MXCSR is 32-bit, and FPCR's defined bits are all below bit 32).
const LIVE: u64 = 1;
const NON_CANONICAL: u64 = 1 << 1;
const SHIFT: u32 = 2;

thread_local! {
    /// The innermost boundary frame of this thread; 0 outside any entry.
    static FRAME: Cell<u64> = const { Cell::new(0) };
}

/// Called by every thunk, under canonical controls, before the entry body.
/// Records the frame and returns the previous one, which the thunk hands
/// back to [`tessera_fp_leave`].
pub(super) extern "C" fn tessera_fp_enter(caller: u64) -> u64 {
    let non_canonical = !fp::is_canonical_bits(caller);
    if non_canonical {
        fp::record_non_canonical_entry();
    }
    debug_assert!(caller >> 32 == 0, "the control register fits 32 bits");
    let token =
        LIVE | if non_canonical { NON_CANONICAL } else { 0 } | ((caller & 0xffff_ffff) << SHIFT);
    FRAME.with(|f| f.replace(token))
}

/// Called by every thunk, under canonical controls, after the entry body:
/// restores the previous frame.
pub(super) extern "C" fn tessera_fp_leave(previous: u64) {
    FRAME.with(|f| f.set(previous));
}

/// Called by every callout thunk, under canonical controls, before the
/// host call: the controls to run it under, those of the innermost
/// boundary frame's caller (or the current ones outside any frame).
pub(super) extern "C" fn tessera_fp_caller_controls() -> u64 {
    frame().map_or_else(fp::current_bits, |(_, caller)| caller)
}

/// Called by the callout thunk after the host code returned and the
/// controls were made canonical again: counts a callback that changed the
/// caller's controls.
pub(super) extern "C" fn tessera_fp_callout_returned(before: u64, after: u64) {
    if fp::controls_differ(before, after) {
        fp::record_non_canonical_entry();
    }
}

/// The raw frame token of this thread (tests check nesting with it).
#[cfg(test)]
pub(super) fn frame_token() -> u64 {
    FRAME.with(Cell::get)
}

/// The current frame: whether the caller's controls were canonical, and
/// their value. `None` outside a boundary frame (a test calling Rust
/// directly).
pub(super) fn frame() -> Option<(bool, u64)> {
    let token = FRAME.with(Cell::get);
    (token & LIVE != 0).then_some((token & NON_CANONICAL != 0, token >> SHIFT))
}

/// The process-wide policy for entries whose caller controls were not
/// canonical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Policy {
    /// Proceed under canonical controls, count, warn once per process.
    Canonicalize = 0,
    /// Raise `FloatingPointEnvironmentError` before any work.
    Reject = 1,
}

static POLICY: AtomicU8 = AtomicU8::new(Policy::Canonicalize as u8);
static WARNED: AtomicBool = AtomicBool::new(false);

pub(crate) fn set_policy(policy: Policy) {
    POLICY.store(policy as u8, Ordering::Relaxed);
}

pub(crate) fn policy() -> Policy {
    if POLICY.load(Ordering::Relaxed) == Policy::Reject as u8 {
        Policy::Reject
    } else {
        Policy::Canonicalize
    }
}

const WARNING: &core::ffi::CStr = c"tessera_rl: the caller's floating-point controls were not the default (for example flush-to-zero); Tessera computed under default controls and restored the caller's. This warning is shown once per process (ADR-0041).";

/// The policy check of a fallible entry, run first in `ffi::entry`, under
/// canonical controls. `Err` means the entry must fail before any work:
/// a rejection under the strict policy, or a warning that raised.
pub(super) fn check(g: Gil) -> CResult<()> {
    let Some((true, _caller)) = frame() else {
        return Ok(());
    };
    if policy() == Policy::Reject {
        return Err(error(
            g,
            Kind::FloatingPoint,
            "the caller's floating-point controls are not the default and the policy is \"reject\" (tessera_rl.set_fp_policy)".to_owned(),
        ));
    }
    if WARNED.swap(true, Ordering::Relaxed) {
        return Ok(());
    }
    // The warning runs host code (filters, hooks, or an exception under
    // warnings-as-errors): `capi::warn` is a callout, which runs it under
    // the caller's controls and re-canonicalizes afterwards. Failing
    // propagates before the entry runs.
    capi::warn(g, capi::runtime_warning(), WARNING)
}
