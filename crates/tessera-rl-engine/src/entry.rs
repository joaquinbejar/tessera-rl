// SPDX-License-Identifier: MIT OR Apache-2.0

//! The floating-point boundary of every public entry point (ADR-0037).
//!
//! Each public method that runs engine or environment code calls
//! [`canonical`] first, so its body runs in the canonical environment. The
//! glue before that call must stay free of floating-point operations: on a
//! caller thread left in a non-default mode (for example flush-to-zero by a
//! host library), that glue runs outside the guard.

use tessera_rl_num::fp;

/// Runs `f` in the canonical environment. Entries from a non-canonical
/// caller thread are counted by `fp::non_canonical_entries`, which is
/// process-wide and shared by every pool (and any other user of
/// `fp::with_canonical`); with `debug-checks` the first one is also
/// logged, once per process.
#[inline]
pub(crate) fn canonical<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(feature = "debug-checks")]
    warn_once_if_non_canonical();
    fp::with_canonical(f)
}

#[cfg(feature = "debug-checks")]
#[cold]
fn warn_once_if_non_canonical() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    if !fp::is_canonical() {
        ONCE.call_once(|| {
            tracing::warn!(
                "entered Tessera from a thread in a non-canonical floating-point environment; \
                 the body runs canonical, but this is a tested mitigation, not a guarantee (ADR-0037)"
            );
        });
    }
}
