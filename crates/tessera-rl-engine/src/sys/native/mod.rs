// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native address-based waits for the gang (ADR-0005). Audited `unsafe`
//! module (part of `engine::sys`).
//!
//! `wait(word, expected)` blocks while `word == expected` (it may return
//! spuriously; callers re-check), `wait_timeout` does the same for at most a
//! duration (ADR-0056 decision 4), and `wake_all(word)` wakes every thread
//! blocked on `word`. Backends: `futex` on Linux, `WaitOnAddress` on
//! Windows, `os_sync_wait_on_address` on macOS 14.4 or later (found at
//! run time; older systems have none and use the portable park backend).

use core::time::Duration;
use std::sync::atomic::AtomicU32;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(windows)]
mod windows;

/// A relative timeout as `timespec` parts `(seconds, nanoseconds)`, both
/// `i32`, so each converts infallibly into the platform's field types
/// whatever their width (`time_t` is 32-bit on some targets, and on musl the
/// libc crate deprecates naming it). Seconds beyond `i32::MAX` (about 68
/// years) saturate; every caller rechecks its deadline after any return.
/// The nanoseconds are below 10^9 and always fit.
#[cfg_attr(
    not(any(target_os = "linux", test)),
    expect(dead_code, reason = "only the Linux futex builds a timespec")
)]
#[must_use]
pub(crate) fn timespec_parts(timeout: Duration) -> (i32, i32) {
    (
        i32::try_from(timeout.as_secs()).unwrap_or(i32::MAX),
        i32::try_from(timeout.subsec_nanos()).unwrap_or(0),
    )
}

/// A native wait primitive.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Native {
    #[cfg(target_os = "macos")]
    os_sync: macos::OsSync,
}

impl Native {
    /// The native primitive of this platform, if it has one.
    #[cfg_attr(
        any(target_os = "linux", windows),
        expect(clippy::unnecessary_wraps, reason = "always present on this platform")
    )]
    pub(crate) fn detect() -> Option<Self> {
        #[cfg(target_os = "macos")]
        {
            macos::OsSync::resolve().map(|os_sync| Self { os_sync })
        }
        #[cfg(any(target_os = "linux", windows))]
        {
            Some(Self {})
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
        {
            None
        }
    }

    /// Blocks while `word == expected`; may return spuriously.
    #[cfg_attr(
        any(target_os = "linux", windows),
        expect(clippy::unused_self, reason = "the macOS backend carries state")
    )]
    pub(crate) fn wait(self, word: &AtomicU32, expected: u32) {
        #[cfg(target_os = "linux")]
        linux::wait(word, expected);
        #[cfg(target_os = "macos")]
        self.os_sync.wait(word, expected);
        #[cfg(windows)]
        windows::wait(word, expected);
        #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
        let _ = (self, word, expected);
    }

    /// Blocks while `word == expected`, at most `timeout`; may return early
    /// (callers recheck their deadline). A zero timeout returns at once
    /// (ADR-0056 decision 4).
    #[cfg_attr(
        any(target_os = "linux", windows),
        expect(clippy::unused_self, reason = "the macOS backend carries state")
    )]
    pub(crate) fn wait_timeout(self, word: &AtomicU32, expected: u32, timeout: Duration) {
        if timeout.is_zero() {
            return;
        }
        #[cfg(target_os = "linux")]
        linux::wait_timeout(word, expected, timeout);
        #[cfg(target_os = "macos")]
        self.os_sync.wait_timeout(word, expected, timeout);
        #[cfg(windows)]
        windows::wait_timeout(word, expected, timeout);
        #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
        let _ = (self, word, expected, timeout);
    }

    /// Wakes every thread blocked on `word`.
    #[cfg_attr(
        any(target_os = "linux", windows),
        expect(clippy::unused_self, reason = "the macOS backend carries state")
    )]
    pub(crate) fn wake_all(self, word: &AtomicU32) {
        #[cfg(target_os = "linux")]
        linux::wake_all(word);
        #[cfg(target_os = "macos")]
        self.os_sync.wake_all(word);
        #[cfg(windows)]
        windows::wake_all(word);
        #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
        let _ = (self, word);
    }
}

// FFI calls: native only, not under Miri.
#[cfg(test)]
#[cfg(not(miri))]
mod tests;

// Pure arithmetic: runs on every platform, Miri included.
#[cfg(test)]
mod timespec_tests {
    use super::{Duration, timespec_parts};

    /// Zero, sub-second, normal and huge timeouts; seconds saturate at
    /// `i32::MAX`, nanoseconds are kept.
    #[test]
    fn test_timespec_parts_saturates_seconds_and_keeps_nanos() {
        assert_eq!(timespec_parts(Duration::ZERO), (0, 0));
        assert_eq!(
            timespec_parts(Duration::from_nanos(999_999_999)),
            (0, 999_999_999)
        );
        assert_eq!(timespec_parts(Duration::new(5, 250)), (5, 250));
        assert_eq!(
            timespec_parts(Duration::from_secs(u64::from(i32::MAX.unsigned_abs()))),
            (i32::MAX, 0)
        );
        assert_eq!(timespec_parts(Duration::new(u64::MAX, 7)), (i32::MAX, 7));
        assert_eq!(timespec_parts(Duration::MAX), (i32::MAX, 999_999_999));
    }
}
