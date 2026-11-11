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
