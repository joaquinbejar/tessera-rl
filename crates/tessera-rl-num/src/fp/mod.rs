// SPDX-License-Identifier: MIT OR Apache-2.0

//! Floating-point scope guard (SPEC 8.2, ADR-0008). Audited `unsafe` module.
//!
//! [`FpGuard`] puts the current thread's floating-point control register in
//! the canonical state, round to nearest with flush-to-zero and
//! denormals-are-zero off, and restores the previous state when it is
//! dropped, on every exit path including unwinding. The engine takes a
//! guard on entry to every entry point that runs environment code.
//!
//! - x86-64: MXCSR. Clears RC (bits 13 and 14), FTZ (bit 15) and DAZ (bit 6).
//! - aarch64: FPCR. Clears `RMode` (bits 22 and 23), FZ (bit 24, which also
//!   covers denormal inputs), FZ16 (bit 19), DN (bit 25, default NaN) and the
//!   `FEAT_AFP` bits AH and FIZ (bits 1 and 0).
//!
//! Exception masks and status flags are left as the caller set them.
//!
//! Miri cannot run inline assembly, so this module is covered by native
//! tests on every CI platform.

use core::marker::PhantomData;

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
use aarch64 as arch;

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
use x86_64 as arch;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("tessera-rl-num::fp supports only x86_64 and aarch64");

/// Restores the caller's floating-point state on drop.
///
/// Not `Send`: the guard must be dropped on the thread that created it.
#[derive(Debug)]
#[must_use = "the canonical state lasts only while the guard is alive"]
pub struct FpGuard {
    saved: arch::Control,
    _not_send: PhantomData<*const ()>,
}

impl FpGuard {
    /// Saves the current state and enters the canonical state.
    #[inline]
    pub fn enter() -> Self {
        let saved = arch::read();
        arch::write(arch::canonical(saved));
        Self {
            saved,
            _not_send: PhantomData,
        }
    }
}

impl Drop for FpGuard {
    #[inline]
    fn drop(&mut self) {
        arch::write(self.saved);
    }
}

/// Returns whether the current thread is in the canonical state. Used by
/// `debug-checks` builds after each chunk.
#[must_use]
#[inline]
pub fn is_canonical() -> bool {
    let current = arch::read();
    arch::canonical(current) == current
}

#[cfg(test)]
mod tests {
    use core::hint::black_box;

    use super::{FpGuard, arch, is_canonical};

    fn half_min_positive() -> f64 {
        black_box(f64::MIN_POSITIVE) / black_box(2.0)
    }

    #[test]
    fn test_fp_guard_flush_mode_is_cleared_and_restored() {
        let original = arch::read();
        let flushing = arch::with_flush_to_zero(original);
        arch::write(flushing);
        assert!(!is_canonical());
        {
            let _guard = FpGuard::enter();
            assert!(is_canonical());
            assert!(half_min_positive() > 0.0, "subnormal flushed under guard");
        }
        assert_eq!(arch::read(), flushing);
        arch::write(original);
    }

    #[test]
    fn test_fp_guard_restores_on_panic() {
        let original = arch::read();
        let flushing = arch::with_flush_to_zero(original);
        arch::write(flushing);
        let result = std::panic::catch_unwind(|| {
            let _guard = FpGuard::enter();
            panic!("unwind through the guard");
        });
        assert!(result.is_err());
        assert_eq!(arch::read(), flushing);
        arch::write(original);
    }

    #[test]
    fn test_fp_guard_nested_restores_in_order() {
        let original = arch::read();
        {
            let _outer = FpGuard::enter();
            let _inner = FpGuard::enter();
            assert!(is_canonical());
        }
        assert_eq!(arch::read(), original);
    }
}
