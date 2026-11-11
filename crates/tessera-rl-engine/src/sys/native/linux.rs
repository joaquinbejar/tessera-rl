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

/// Blocks while `word == expected`, at most `timeout`; may return early.
#[expect(unsafe_code, reason = "audited module: futex system call")]
pub(super) fn wait_timeout(word: &AtomicU32, expected: u32, timeout: core::time::Duration) {
    // FUTEX_WAIT takes a relative timeout, measured on CLOCK_MONOTONIC.
    // A timeout beyond `time_t` saturates at `time_t::MAX` seconds (the
    // caller rechecks its deadline after any return). The nanoseconds are
    // below 10^9, so they always fit an `i32`, which converts into `c_long`
    // infallibly whether it is 64-bit (every target built today) or 32-bit.
    let ts = libc::timespec {
        tv_sec: libc::time_t::try_from(timeout.as_secs()).unwrap_or(libc::time_t::MAX),
        tv_nsec: libc::c_long::from(i32::try_from(timeout.subsec_nanos()).unwrap_or(0)),
    };
    // SAFETY: as in `wait`, and `ts` is a valid `timespec` that lives for
    // the call (`tv_nsec` below one second). ETIMEDOUT, EAGAIN and EINTR
    // are all fine: callers re-check.
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            word.as_ptr(),
            libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
            expected,
            core::ptr::from_ref(&ts),
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
