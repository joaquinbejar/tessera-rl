// SPDX-License-Identifier: MIT OR Apache-2.0

//! `os_sync_wait_on_address` waits on macOS 14.4 or later, resolved at run
//! time so older systems keep working (they have no native backend).

use std::sync::OnceLock;
use std::sync::atomic::AtomicU32;

type WaitFn = unsafe extern "C" fn(*mut core::ffi::c_void, u64, usize, u32) -> i32;
type WakeFn = unsafe extern "C" fn(*mut core::ffi::c_void, usize, u32) -> i32;

/// `OS_SYNC_WAIT_ON_ADDRESS_NONE` and `OS_SYNC_WAKE_BY_ADDRESS_NONE`.
const NO_FLAGS: u32 = 0;

#[derive(Debug, Clone, Copy)]
pub(super) struct OsSync {
    wait: WaitFn,
    wake_all: WakeFn,
}

impl OsSync {
    /// The two functions, if this macOS has them; looked up once.
    pub(super) fn resolve() -> Option<Self> {
        static FOUND: OnceLock<Option<OsSync>> = OnceLock::new();
        *FOUND.get_or_init(|| {
            let wait = lookup(c"os_sync_wait_on_address")?;
            let wake_all = lookup(c"os_sync_wake_by_address_all")?;
            // SAFETY: the symbol is the libSystem function of this name,
            // whose C signature is `WaitFn`
            // (`<os/os_sync_wait_on_address.h>`, macOS 14.4).
            #[expect(unsafe_code, reason = "audited module: typed dlsym result")]
            let wait = unsafe { core::mem::transmute::<*mut core::ffi::c_void, WaitFn>(wait) };
            // SAFETY: as above, for `os_sync_wake_by_address_all` and `WakeFn`.
            #[expect(unsafe_code, reason = "audited module: typed dlsym result")]
            let wake_all =
                unsafe { core::mem::transmute::<*mut core::ffi::c_void, WakeFn>(wake_all) };
            let functions = Self { wait, wake_all };
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
