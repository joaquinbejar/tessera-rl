// SPDX-License-Identifier: MIT OR Apache-2.0

//! `futex` waits on Linux.

use std::sync::atomic::AtomicU32;

#[expect(unsafe_code, reason = "audited module: futex system call")]
pub(super) fn wait(word: &AtomicU32, expected: u32) {
    // SAFETY: FUTEX_WAIT with FUTEX_PRIVATE_FLAG reads the aligned 32-bit word at a valid
    // address (a live `AtomicU32` borrowed for the call) and compares it
    // with `expected` atomically in the kernel; a null timeout waits
    // without limit. The word lives in this process only, so the private
    // variant is correct. The result is ignored: callers re-check (EAGAIN,
    // EINTR and spurious wake-ups are all fine).
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            word.as_ptr(),
            libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
            expected,
            core::ptr::null::<libc::timespec>(),
        );
    }
}

#[expect(unsafe_code, reason = "audited module: futex system call")]
pub(super) fn wake_all(word: &AtomicU32) {
    // SAFETY: FUTEX_WAKE with FUTEX_PRIVATE_FLAG only uses the address as a key to wake
    // waiters; it is a valid, aligned, live `AtomicU32`.
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            word.as_ptr(),
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            i32::MAX,
        );
    }
}
