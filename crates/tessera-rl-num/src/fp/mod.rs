// SPDX-License-Identifier: MIT OR Apache-2.0

//! Floating-point environment control (SPEC 8.2, ADR-0008). Audited
//! `unsafe` module.
//!
//! The canonical environment is round to nearest, ties to even, with
//! flush-to-zero and denormals-are-zero off, IEEE NaN propagation and every
//! floating-point exception masked (so no trap can fire inside a kernel).
//!
//! - x86-64 (MXCSR): RC, FTZ and DAZ cleared; all exception masks set.
//! - aarch64 (FPCR): `RMode`, FZ, FZ16, DN, AH, FIZ and every trap enable
//!   cleared. Each bit is documented in the `aarch64` module.
//!
//! # Boundary
//!
//! - **Engine worker threads** are created by Tessera and own their
//!   environment: they call [`establish_canonical`] when they start, never
//!   restore it, and check [`is_canonical`] per batch in `debug-checks`
//!   builds.
//! - **Caller threads** (Inline mode, construction, hashing) take an
//!   [`FpGuard`] at the entry point. Guards are scoped and strictly LIFO:
//!   drop them in reverse order of creation, on the thread that created
//!   them (`FpGuard` is not `Send`).
//!
//! On drop the guard restores the caller's control bits and keeps the
//! current sticky status flags (x86-64 MXCSR bits 0 to 5; aarch64 keeps them
//! in FPSR, which is never written), so exceptions raised inside remain
//! visible to the caller, as if the code had run in the caller's mode.
//!
//! The register accesses are full compiler barriers (no `nomem`,
//! `readonly` or `preserves_flags`), and [`FpGuard::enter`] and its `Drop`
//! are `#[inline(never)]`, so floating-point operations are not moved across
//! the boundary.
//!
//! # Residual caveat
//!
//! Rust's abstract machine assumes the default floating-point environment;
//! running Rust code under a different one is outside the language's
//! guarantees even when it is restored before any operation. Code that runs
//! between a host's non-default mode and [`FpGuard::enter`] (the entry
//! point's own prologue) is therefore not covered by the language, only by
//! the barriers above and by release-mode boundary tests.
//!
//! The barriers order memory, not register-only arithmetic: in release
//! builds the compiler may move a floating-point operation whose result is
//! only used after the guard is dropped to the other side of the boundary
//! (observed in this module's tests). Results computed under a guard must
//! therefore be stored to memory (in the engine, the output buffers) before
//! the guard is dropped. Running kernels
//! inside an assembly shim would close this; the decision is recorded in a
//! proposed ADR (FP environment boundary).
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

/// Puts the current thread in the canonical environment without saving the
/// previous one. For threads whose environment the engine owns.
#[inline(never)]
pub fn establish_canonical() {
    arch::write(arch::canonical(arch::read()));
}

/// Returns whether the current thread is in the canonical environment.
/// Status flags are ignored.
#[must_use]
#[inline]
pub fn is_canonical() -> bool {
    let current = arch::read();
    arch::canonical(current) == current
}

/// Enters the canonical environment and restores the caller's on drop.
///
/// Not `Send`: the guard must be dropped on the thread that created it.
#[derive(Debug)]
#[must_use = "the canonical environment lasts only while the guard is alive"]
pub struct FpGuard {
    saved: arch::Control,
    _not_send: PhantomData<*const ()>,
}

impl FpGuard {
    /// Saves the caller's environment and enters the canonical one.
    #[inline(never)]
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
    #[inline(never)]
    fn drop(&mut self) {
        debug_assert!(is_canonical(), "FpGuard dropped out of LIFO order");
        arch::write(arch::restored(self.saved, arch::read()));
    }
}

#[cfg(test)]
mod tests {
    use core::hint::black_box;

    use super::arch::{self, test_modes};
    use super::{FpGuard, establish_canonical, is_canonical};

    // 1/3 rounded to nearest.
    const ONE_THIRD_NEAREST: u64 = 0x3FD5_5555_5555_5555;

    /// Runs `f` with `mode` applied to the current state, then restores the
    /// original state before any further floating-point operation.
    fn with_mode(mode: fn(arch::Control) -> arch::Control, f: impl FnOnce(arch::Control)) {
        let original = arch::read();
        arch::write(mode(original));
        // Read back: some environments (Rosetta 2 for exception masks,
        // CPUs without trap support on aarch64) ignore part of the write.
        let entered = arch::read();
        f(entered);
        arch::write(original);
    }

    #[test]
    fn test_fp_guard_under_flush_to_zero_keeps_subnormals() {
        with_mode(test_modes::flush_to_zero, |entered| {
            let tiny = {
                let _guard = FpGuard::enter();
                assert!(is_canonical());
                black_box(black_box(f64::MIN_POSITIVE) / black_box(2.0))
            };
            assert_eq!(
                test_modes::control_bits(arch::read()),
                test_modes::control_bits(entered)
            );
            // Compare bits: under DAZ a float comparison would read the
            // subnormal as zero.
            assert_eq!(tiny.to_bits(), 0x0008_0000_0000_0000);
        });
    }

    #[test]
    fn test_fp_guard_under_toward_zero_rounds_to_nearest() {
        with_mode(test_modes::toward_zero, |entered| {
            let third = {
                let _guard = FpGuard::enter();
                black_box(black_box(1.0f64) / black_box(3.0))
            };
            assert_eq!(
                test_modes::control_bits(arch::read()),
                test_modes::control_bits(entered)
            );
            assert_eq!(third.to_bits(), ONE_THIRD_NEAREST);
        });
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_fp_guard_masks_unmasked_exceptions_inside() {
        with_mode(test_modes::unmask_divide_by_zero, |entered| {
            let inf = {
                let _guard = FpGuard::enter();
                assert!(test_modes::all_masked(arch::read()));
                black_box(black_box(1.0f64) / black_box(0.0))
            };
            assert_eq!(
                test_modes::control_bits(arch::read()),
                test_modes::control_bits(entered)
            );
            assert!(inf.is_infinite());
        });
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_fp_guard_restore_keeps_status_flags_raised_inside() {
        with_mode(test_modes::clear_status, |entered| {
            {
                let _guard = FpGuard::enter();
                black_box(black_box(1.0f64) / black_box(3.0));
            }
            let after = arch::read();
            assert_eq!(
                test_modes::control_bits(after),
                test_modes::control_bits(entered)
            );
            assert!(test_modes::inexact_raised(after));
        });
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_fp_guard_clears_dn_ah_and_fz16() {
        with_mode(test_modes::non_ieee_nan_and_half, |entered| {
            {
                let _guard = FpGuard::enter();
                assert!(is_canonical());
            }
            assert_eq!(arch::read(), entered);
        });
    }

    #[test]
    fn test_fp_guard_restores_on_panic() {
        with_mode(test_modes::flush_to_zero, |entered| {
            let result = std::panic::catch_unwind(|| {
                let _guard = FpGuard::enter();
                panic!("unwind through the guard");
            });
            assert!(result.is_err());
            assert_eq!(
                test_modes::control_bits(arch::read()),
                test_modes::control_bits(entered)
            );
        });
    }

    #[test]
    fn test_fp_guard_nested_restores_in_order() {
        let original = test_modes::control_bits(arch::read());
        {
            let _outer = FpGuard::enter();
            let _inner = FpGuard::enter();
            assert!(is_canonical());
        }
        assert_eq!(test_modes::control_bits(arch::read()), original);
    }

    #[test]
    fn test_establish_canonical_enters_without_guard() {
        with_mode(test_modes::flush_to_zero, |_| {
            establish_canonical();
            assert!(is_canonical());
        });
    }
}
