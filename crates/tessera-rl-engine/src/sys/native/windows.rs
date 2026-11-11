// SPDX-License-Identifier: MIT OR Apache-2.0

//! `WaitOnAddress` waits on Windows. Opt-in only until it has run on real
//! Windows in CI; the default backend there is park.

use std::sync::atomic::AtomicU32;

use windows_sys::Win32::System::Threading::{INFINITE, WaitOnAddress, WakeByAddressAll};

#[expect(unsafe_code, reason = "audited module: WaitOnAddress")]
pub(super) fn wait(word: &AtomicU32, expected: u32) {
    // SAFETY: both pointers are to live, aligned 4-byte values (the atomic
    // and a local copy of `expected`), and the size passed is 4.
    // `WaitOnAddress` only reads them. The result is ignored: callers
    // re-check after any return.
    unsafe {
        WaitOnAddress(
            word.as_ptr().cast(),
            core::ptr::from_ref(&expected).cast(),
            4,
            INFINITE,
        );
    }
}

/// Blocks while `word == expected`, at most `timeout` rounded up to whole
/// milliseconds (so a short wait never becomes a zero-time poll); may
/// return early.
#[expect(unsafe_code, reason = "audited module: WaitOnAddress")]
pub(super) fn wait_timeout(word: &AtomicU32, expected: u32, timeout: core::time::Duration) {
    // Below INFINITE (u32::MAX), which would wait without limit.
    let ms = timeout
        .as_nanos()
        .div_ceil(1_000_000)
        .min(u128::from(INFINITE - 1));
    let ms = u32::try_from(ms).unwrap_or(INFINITE - 1);
    // SAFETY: as in `wait`. A timeout returns FALSE with ERROR_TIMEOUT,
    // which is ignored: callers re-check their deadline.
    unsafe {
        WaitOnAddress(
            word.as_ptr().cast(),
            core::ptr::from_ref(&expected).cast(),
            4,
            ms,
        );
    }
}

#[expect(unsafe_code, reason = "audited module: WakeByAddressAll")]
pub(super) fn wake_all(word: &AtomicU32) {
    // SAFETY: the address is a live, aligned `AtomicU32`; it is only used
    // as a key to wake waiters.
    unsafe { WakeByAddressAll(word.as_ptr().cast()) };
}
