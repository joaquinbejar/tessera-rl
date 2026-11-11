// SPDX-License-Identifier: MIT OR Apache-2.0

//! `os_sync_wait_on_address` waits on macOS 14.4 or later, resolved at run
//! time so older systems keep working (they have no native backend).

use std::sync::OnceLock;
use std::sync::atomic::AtomicU32;

type WaitFn = unsafe extern "C" fn(*mut core::ffi::c_void, u64, usize, u32) -> i32;
type WakeFn = unsafe extern "C" fn(*mut core::ffi::c_void, usize, u32) -> i32;
/// `os_sync_wait_on_address_with_timeout(addr, value, size, flags, clockid,
/// timeout_ns)`.
type WaitTimeoutFn = unsafe extern "C" fn(*mut core::ffi::c_void, u64, usize, u32, u32, u64) -> i32;

/// `OS_SYNC_WAIT_ON_ADDRESS_NONE` and `OS_SYNC_WAKE_BY_ADDRESS_NONE`.
const NO_FLAGS: u32 = 0;
/// `OS_CLOCK_MACH_ABSOLUTE_TIME` (`<os/clock.h>`): the clock the timeout is
/// anchored to; the timeout itself is in nanoseconds.
const OS_CLOCK_MACH_ABSOLUTE_TIME: u32 = 32;

#[derive(Debug, Clone, Copy)]
pub(super) struct OsSync {
    wait: WaitFn,
    wake_all: WakeFn,
    wait_timeout: WaitTimeoutFn,
}

impl OsSync {
    /// The three functions, if this macOS has them (all since 14.4); looked
    /// up once. Without any of them there is no native backend, and the pool
    /// uses park, whose timed wait is `park_timeout` (ADR-0056 decision 4).
    pub(super) fn resolve() -> Option<Self> {
        static FOUND: OnceLock<Option<OsSync>> = OnceLock::new();
        *FOUND.get_or_init(|| {
            let wait = lookup(c"os_sync_wait_on_address")?;
            let wake_all = lookup(c"os_sync_wake_by_address_all")?;
            let wait_timeout = lookup(c"os_sync_wait_on_address_with_timeout")?;
            // SAFETY: the symbol is the libSystem function of this name,
            // whose C signature is `WaitFn`
            // (`<os/os_sync_wait_on_address.h>`, macOS 14.4).
            #[expect(unsafe_code, reason = "audited module: typed dlsym result")]
            let wait = unsafe { core::mem::transmute::<*mut core::ffi::c_void, WaitFn>(wait) };
            // SAFETY: as above, for `os_sync_wake_by_address_all` and `WakeFn`.
            #[expect(unsafe_code, reason = "audited module: typed dlsym result")]
            let wake_all =
                unsafe { core::mem::transmute::<*mut core::ffi::c_void, WakeFn>(wake_all) };
            // SAFETY: as above, for `os_sync_wait_on_address_with_timeout`
            // and `WaitTimeoutFn` (`os_clockid_t` is a `uint32_t` enum).
            #[expect(unsafe_code, reason = "audited module: typed dlsym result")]
            let wait_timeout = unsafe {
                core::mem::transmute::<*mut core::ffi::c_void, WaitTimeoutFn>(wait_timeout)
            };
            let functions = Self {
                wait,
                wake_all,
                wait_timeout,
            };
            Some(functions)
        })
    }

    #[expect(unsafe_code, reason = "audited module: os_sync_wait_on_address")]
    pub(super) fn wait(self, word: &AtomicU32, expected: u32) {
        // SAFETY: the address is a live, aligned 4-byte atomic and the size
        // passed is 4; the call compares it with `expected` atomically. The
        // result is ignored: callers re-check after any return.
        unsafe {
            (self.wait)(word.as_ptr().cast(), u64::from(expected), 4, NO_FLAGS);
        }
    }

    /// Blocks while `word == expected`, at most `timeout`; may return early.
    #[expect(
        unsafe_code,
        reason = "audited module: os_sync_wait_on_address_with_timeout"
    )]
    pub(super) fn wait_timeout(
        self,
        word: &AtomicU32,
        expected: u32,
        timeout: core::time::Duration,
    ) {
        // A zero timeout is EINVAL: the smallest wait is one nanosecond.
        let ns = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX).max(1);
        // SAFETY: as in `wait`; the clock id is a valid `os_clockid_t` and
        // the timeout is nonzero. ETIMEDOUT and every other result are
        // ignored: callers re-check their deadline.
        unsafe {
            (self.wait_timeout)(
                word.as_ptr().cast(),
                u64::from(expected),
                4,
                NO_FLAGS,
                OS_CLOCK_MACH_ABSOLUTE_TIME,
                ns,
            );
        }
    }

    #[expect(unsafe_code, reason = "audited module: os_sync_wake_by_address_all")]
    pub(super) fn wake_all(self, word: &AtomicU32) {
        // SAFETY: the address is a live, aligned 4-byte atomic, used only as
        // a key to wake waiters.
        unsafe {
            (self.wake_all)(word.as_ptr().cast(), 4, NO_FLAGS);
        }
    }
}

#[expect(unsafe_code, reason = "audited module: dlsym")]
fn lookup(name: &core::ffi::CStr) -> Option<*mut core::ffi::c_void> {
    // SAFETY: `name` is a NUL-terminated C string and RTLD_DEFAULT searches
    // the images already loaded; dlsym has no other preconditions.
    let symbol = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
    (!symbol.is_null()).then_some(symbol)
}
