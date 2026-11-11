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
    // Seconds beyond `i32::MAX` (about 68 years) saturate (the caller
    // rechecks its deadline after any return). Both parts are `i32`, which
    // convert infallibly into the fields whether `time_t` and `c_long` are
    // 64-bit or 32-bit, without naming `time_t` (deprecated on musl).
    let (secs, nanos) = super::timespec_parts(timeout);
    let ts = libc::timespec {
        tv_sec: secs.into(),
        tv_nsec: nanos.into(),
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
