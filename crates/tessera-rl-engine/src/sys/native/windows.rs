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

#[expect(unsafe_code, reason = "audited module: WakeByAddressAll")]
pub(super) fn wake_all(word: &AtomicU32) {
    // SAFETY: the address is a live, aligned `AtomicU32`; it is only used
    // as a key to wake waiters.
    unsafe { WakeByAddressAll(word.as_ptr().cast()) };
}
