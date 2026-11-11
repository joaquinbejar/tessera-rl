// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fork awareness (issue #93). Worker threads do not survive `fork`, and a
//! mutex may be inherited locked, so an environment object made before a fork
//! must never touch its pool in the child. The objects are tagged with the
//! fork generation at creation, and the child's generation moves in a
//! `pthread_atfork` handler, with no `getpid` per call. The handler runs inside
//! `fork`, before any Python at-fork hook, and also for a C extension calling
//! `fork` directly. An object made in the child has the new generation and works
//! normally.
//!
//! A child that first imports this module after the fork has no handler from
//! its parent: its generation is 0 and it takes the default wait backend.

use core::sync::atomic::{AtomicU64, Ordering};

use super::capi::{CResult, Gil};

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

/// The child handler. An atomic increment only: it runs inside `fork`, where
/// only async-signal-safe work is allowed, so it touches no Python and does not
/// allocate.
#[cfg(not(windows))]
extern "C" fn after_in_child() {
    GENERATION.fetch_add(1, Ordering::AcqRel);
}

/// Registers the handler once, at module initialization. Windows has no fork.
///
/// # Errors
///
/// A failed registration (ENOMEM) raises, so the import fails: a silently absent
/// guard would bring back the hangs and traps of a forked child.
#[cfg_attr(
    windows,
    expect(
        clippy::unnecessary_wraps,
        reason = "one signature on every platform: the registration can fail where fork exists"
    )
)]
pub(super) fn register(g: Gil) -> CResult<()> {
    #[cfg(not(windows))]
    {
        unsafe extern "C" {
            fn pthread_atfork(
                prepare: Option<extern "C" fn()>,
                parent: Option<extern "C" fn()>,
                child: Option<extern "C" fn()>,
            ) -> core::ffi::c_int;
        }
        // SAFETY: a libc function with this C signature; the handler is a
        // `'static` function that stays loaded, as the module is never unloaded
        // while the interpreter lives, and it is registered once, as the module
        // has one initialization (`m_size` is -1).
        #[expect(unsafe_code, reason = "audited module: one libc registration")]
        let status = unsafe { pthread_atfork(None, None, Some(after_in_child)) };
        if status != 0 {
            return Err(super::capi::raise(
                g,
                super::capi::runtime_error(),
                "pthread_atfork failed: the fork guard cannot be installed",
            ));
        }
    }
    #[cfg(windows)]
    let _ = g;
    Ok(())
}
