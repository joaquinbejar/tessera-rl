// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fork awareness (issue #93). Worker threads do not survive `fork`, and a
//! mutex may be inherited locked, so an environment object made before a fork
//! must never touch its pool in the child. The objects are tagged with the
//! fork generation at creation, and the child's generation moves in an
//! `os.register_at_fork` hook, with no `getpid` per call. An object made in
//! the child has the new generation and works normally.

use core::sync::atomic::{AtomicU64, Ordering};

use pyo3::ffi;

use super::capi::{self, CResult, Gil, Ref};

/// Moved once in each forked child, before any other thread can exist there.
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// The current fork generation.
pub(crate) fn generation() -> u64 {
    GENERATION.load(Ordering::Acquire)
}

/// The message of the refusal.
pub(crate) const REFUSAL: &str = "this environment was created before os.fork() and cannot be \
     used in the child, where its worker threads do not exist: create a new environment in the \
     child, or use the spawn start method";

/// `_after_fork_child()`: the `after_in_child` hook. It takes no boundary
/// check: the caller's floating-point controls must not stop the guard.
pub(super) unsafe extern "C" fn after_in_child(
    _module: *mut ffi::PyObject,
    _args: *mut ffi::PyObject,
) -> *mut ffi::PyObject {
    GENERATION.fetch_add(1, Ordering::AcqRel);
    // SAFETY: CPython calls a method attached.
    let g = unsafe { Gil::assume() };
    capi::none(g).into_ptr()
}

/// Registers the hook once, at module initialization. Windows has no
/// `os.register_at_fork` and no fork.
pub(super) fn register(g: Gil, module: Ref<'_>) -> CResult<()> {
    #[cfg(not(windows))]
    capi::register_after_fork_in_child(g, module, c"_after_fork_child")?;
    #[cfg(windows)]
    let _ = (g, module);
    Ok(())
}
