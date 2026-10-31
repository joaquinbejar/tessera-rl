// SPDX-License-Identifier: MIT OR Apache-2.0

//! The floating-point boundary of every public entry point (ADR-0037).
//!
//! Each public method that runs engine or environment code calls
//! [`canonical`] first, so its body runs in the canonical environment. The
//! glue before that call must stay free of floating-point operations.
//!
//! Callers must enter from Rust's default floating-point environment;
//! non-default caller controls are unsupported in phase 1a (ADR-0037). The
//! guard is a defensive mitigation against that misuse and the counter
//! makes it observable; neither is a guarantee. Foreign callers with
//! non-default controls need the native entry boundary that is a hard gate
//! for phase 1b.

use tessera_rl_num::fp;

/// Runs `f` in the canonical environment. Entries from a non-canonical
/// caller thread are counted by `fp::non_canonical_entries`, which is
/// process-wide and shared by every pool (and any other user of
/// `fp::with_canonical`). With `debug-checks` the first one is also logged,
/// once per process; the log runs inside the canonical environment, so
/// subscriber code never sees the caller's controls.
#[inline]
pub(crate) fn canonical<R>(f: impl FnOnce() -> R) -> R {
    // Reading the control register is not a floating-point operation.
    #[cfg(feature = "debug-checks")]
    let misuse = !fp::is_canonical();
    fp::with_canonical(|| {
        #[cfg(feature = "debug-checks")]
        if misuse {
            warn_once();
        }
        f()
    })
}

#[cfg(feature = "debug-checks")]
#[cold]
fn warn_once() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "entered Tessera from a thread in a non-canonical floating-point environment; \
             this is unsupported (ADR-0037): the guard is only a defensive mitigation"
        );
    });
}
