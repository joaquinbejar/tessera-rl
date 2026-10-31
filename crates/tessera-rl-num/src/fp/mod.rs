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
//! - **Caller threads** (Inline mode, construction, hashing) run
//!   environment code only through [`with_canonical`], at the outermost
//!   Rust frame of the entry point. It enters a scoped guard and calls the
//!   closure through an `#[inline(never)]` helper, to keep the closure's
//!   floating-point operations between the mode change and the restore,
//!   even when results stay in registers. The raw guard is not public;
//!   guards are strictly LIFO and not `Send`.
//!
//! # Precondition (ADR-0037, accepted)
//!
//! In phase 1a the API is Rust-only, and **a caller thread must be in Rust's
//! default floating-point environment**. Entering with non-default controls
//! (for example flush-to-zero left by a host library) is **unsupported**.
//! The guard is a defensive mitigation against that misuse, empirically
//! tested in release builds, not a language guarantee: `#[inline(never)]`
//! controls inlining, not LLVM's floating-point semantics.
//! [`non_canonical_entries`] makes such misuse observable; it is not proof
//! of correctness. Supporting foreign callers with non-default controls
//! requires an audited native entry boundary that canonicalizes before the
//! first Rust frame: a hard gate for phase 1b.
//!
//! On drop the guard restores the caller's control bits and keeps the
//! current sticky status flags (x86-64 MXCSR bits 0 to 5; aarch64 keeps them
//! in FPSR, which is never written), so exceptions raised inside remain
//! visible to the caller, as if the code had run in the caller's mode.
//!
//! The register accesses are full compiler barriers (no `nomem`,
//! `readonly` or `preserves_flags`), and the guard's `enter` and its `Drop`
//! are `#[inline(never)]`, so floating-point operations are not moved across
//! the boundary.
//!
//! # Residual caveat
//!
//! Rust's abstract machine assumes the default floating-point environment;
//! running Rust code under a different one is outside the language's
//! guarantees even when it is restored before any operation. Code that runs
//! between a host's non-default mode and the guard's `enter` (the entry
//! point's own prologue) is therefore not covered by the language, only by
//! the barriers above and by release-mode boundary tests.
//!
//! The barriers order memory, not register-only arithmetic: in release
//! builds the compiler moved a quotient whose result was only used after a
//! raw guard was dropped to the other side of the boundary. That is why
//! code runs only inside the non-inlined call of [`with_canonical`], which
//! release regression tests show to be effective on the supported targets;
//! it is not a language-level proof. The native entry boundary required
//! before phase 1b (ADR-0037) is what will give one.
//!
//! Miri cannot run inline assembly, so this module is covered by native
//! tests on every CI platform.

use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering};

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

/// Puts the current thread in the canonical environment if it is not
/// already, and returns whether a reset was needed. Engine workers call it
/// at the start of every batch and count resets (ADR-0037).
#[must_use = "workers count resets"]
#[inline(never)]
pub fn ensure_canonical() -> bool {
    let current = arch::read();
    let canonical = arch::canonical(current);
    if canonical == current {
        false
    } else {
        arch::write(canonical);
        true
    }
}

/// Runs `f` in the canonical environment and restores the caller's
/// environment afterwards, also when `f` unwinds. Control bits are
/// restored; sticky status flags raised inside are kept.
///
/// This is the only way environment code runs on a caller's thread.
///
/// The caller's thread must be in the default environment (the module's
/// precondition, ADR-0037). Keeping the closure's operations inside the
/// canonical environment when it is not is a defensive mitigation,
/// empirically tested in release builds on x86-64 and aarch64, not a Rust
/// language guarantee.
#[inline]
pub fn with_canonical<R>(f: impl FnOnce() -> R) -> R {
    let guard = FpGuard::enter();
    let result = call_opaque(f);
    drop(guard);
    result
}

/// A non-inlined call, so the closure's floating-point operations are not
/// merged into the caller and moved across the mode change. Tested, not
/// guaranteed by the language (ADR-0037).
///
/// Without the barriers LLVM infers `memory(none)` for this function when
/// the closure is pure, which would let it hoist or reschedule the call
/// outside the guard. The empty `asm!` blocks (without `nomem`) give it an
/// opaque memory effect, and `black_box` hides the closure and its result.
#[inline(never)]
fn call_opaque<R>(f: impl FnOnce() -> R) -> R {
    compiler_barrier();
    let result = core::hint::black_box(core::hint::black_box(f)());
    compiler_barrier();
    result
}

/// An empty assembly block with an opaque memory effect.
#[inline]
#[expect(unsafe_code, reason = "audited module: compiler barrier")]
fn compiler_barrier() {
    // SAFETY: the template is empty, so no instruction executes; it touches
    // no register, memory or flag. Omitting `nomem` makes the compiler
    // assume arbitrary memory effects, which is the only purpose.
    unsafe {
        core::arch::asm!("", options(nostack));
    }
}

/// Test-only: runs `f` with flush-to-zero (and, on x86-64,
/// denormals-are-zero) enabled, as a hostile host library might leave the
/// thread, then restores the previous state.
///
/// Running Rust code in a non-default FP mode is outside the language's
/// guarantees (ADR-0037), so this exists only behind the non-default
/// `test-hooks` feature, which only test dev-dependencies enable.
#[cfg(any(test, feature = "test-hooks"))]
#[inline(never)]
pub fn with_hostile_flush_to_zero<R>(f: impl FnOnce() -> R) -> R {
    let original = arch::read();
    arch::write(arch::flush_to_zero(original));
    let result = call_opaque(f);
    arch::write(original);
    result
}

/// Test-only hostile control modes and a scope that runs code under them,
/// for testing native entry boundaries (ADR-0041) from outside this crate.
/// Running Rust under these modes is outside the language's guarantees,
/// hence the non-default `test-hooks` feature.
#[cfg(feature = "test-hooks")]
pub mod hostile {
    use super::{Control, arch, call_opaque};

    /// Flush-to-zero (and denormals-are-zero on x86-64).
    #[must_use]
    pub const fn flush_to_zero(c: Control) -> Control {
        arch::test_modes::flush_to_zero(c)
    }

    /// Round toward zero.
    #[must_use]
    pub const fn toward_zero(c: Control) -> Control {
        arch::test_modes::toward_zero(c)
    }

    /// The divide-by-zero exception unmasked (x86-64).
    #[cfg(target_arch = "x86_64")]
    #[must_use]
    pub const fn unmask_divide_by_zero(c: Control) -> Control {
        arch::test_modes::unmask_divide_by_zero(c)
    }

    /// Default NaN, alternate handling and half-precision flush (aarch64).
    #[cfg(target_arch = "aarch64")]
    #[must_use]
    pub const fn non_ieee_nan_and_half(c: Control) -> Control {
        arch::test_modes::non_ieee_nan_and_half(c)
    }

    /// The control bits of a value (status flags removed).
    #[must_use]
    pub const fn control_bits(c: Control) -> Control {
        arch::test_modes::control_bits(c)
    }

    /// The current control register.
    #[must_use]
    pub fn read() -> Control {
        arch::read()
    }

    /// Runs `f` with the controls set to `mode(current)`; returns its
    /// result and the control register `f` left behind, then restores the
    /// original controls.
    #[inline(never)]
    pub fn run<R>(mode: impl FnOnce(Control) -> Control, f: impl FnOnce() -> R) -> (R, Control) {
        let original = arch::read();
        arch::write(mode(original));
        let result = call_opaque(f);
        let after = arch::read();
        arch::write(original);
        (result, after)
    }
}

/// Entries into [`with_canonical`] whose caller thread was not in the
/// canonical environment (ADR-0037 observability), process-wide.
static NON_CANONICAL_ENTRIES: AtomicU64 = AtomicU64::new(0);

/// How many times, process-wide and shared by every pool, a caller thread
/// entered [`with_canonical`] in a non-canonical floating-point environment
/// (for example after a host library enabled flush-to-zero).
///
/// Such entries are **unsupported** in phase 1a (ADR-0037): the guard only
/// mitigates them defensively, and the Rust code before it runs in the
/// caller's environment. This counter is observability of that misuse, not
/// proof that results are correct.
#[must_use]
pub fn non_canonical_entries() -> u64 {
    NON_CANONICAL_ENTRIES.load(Ordering::Relaxed)
}

/// The floating-point control register of this target: MXCSR on x86-64,
/// FPCR on aarch64.
pub type Control = arch::Control;

/// Control bits the canonical environment clears. A native boundary
/// computes the canonical value as `(c & !CANONICAL_CLEAR) | CANONICAL_SET`.
pub const CANONICAL_CLEAR: Control = arch::CLEAR;

/// Control bits the canonical environment sets (the exception masks on
/// x86-64; none on aarch64, where clear trap enables mask exceptions).
pub const CANONICAL_SET: Control = arch::SET;

/// The sticky status flags held in the control register (x86-64 MXCSR bits
/// 0 to 5; none on aarch64, whose flags live in FPSR).
pub const STATUS_FLAGS: Control = arch::STATUS_FLAGS;

/// The canonical value of a control register value; status flags are
/// kept.
#[must_use]
pub const fn canonical_control(control: Control) -> Control {
    arch::canonical(control)
}

/// Whether a control register value, passed zero-extended to 64 bits by a
/// native boundary, is canonical (status flags ignored).
#[must_use]
pub const fn is_canonical_bits(bits: u64) -> bool {
    let control = arch::from_bits(bits);
    arch::canonical(control) == control
}

/// Whether two control register values, zero-extended to 64 bits, differ
/// in a control bit (status flags ignored).
#[must_use]
pub const fn controls_differ(a: u64, b: u64) -> bool {
    let flags = arch::from_bits(u64::MAX) & STATUS_FLAGS;
    (arch::from_bits(a) & !flags) != (arch::from_bits(b) & !flags)
}

/// Counts one entry made in a non-canonical environment, for entry
/// boundaries outside this crate (the native thunks of the Python
/// bindings, ADR-0041). Observability only, like the counter itself.
pub fn record_non_canonical_entry() {
    NON_CANONICAL_ENTRIES.fetch_add(1, Ordering::Relaxed);
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
/// Use [`with_canonical`]; the raw guard does not keep register arithmetic
/// inside its scope.
///
/// Not `Send`: the guard must be dropped on the thread that created it.
#[derive(Debug)]
#[must_use = "the canonical environment lasts only while the guard is alive"]
pub(crate) struct FpGuard {
    saved: arch::Control,
    _not_send: PhantomData<*const ()>,
}

impl FpGuard {
    /// Saves the caller's environment and enters the canonical one.
    #[inline(never)]
    pub(crate) fn enter() -> Self {
        let saved = arch::read();
        let canonical = arch::canonical(saved);
        if canonical != saved {
            NON_CANONICAL_ENTRIES.fetch_add(1, Ordering::Relaxed);
        }
        arch::write(canonical);
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
    use super::with_hostile_flush_to_zero;
    use super::{
        FpGuard, ensure_canonical, establish_canonical, is_canonical, non_canonical_entries,
        with_canonical,
    };

    // 1/3 rounded to nearest.
    const ONE_THIRD_NEAREST: u64 = 0x3FD5_5555_5555_5555;

    #[test]
    fn test_with_canonical_counts_entries_from_a_non_canonical_thread() {
        let before = non_canonical_entries();
        with_hostile_flush_to_zero(|| with_canonical(|| ()));
        assert!(non_canonical_entries() > before);
    }

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

    // Regression for the hoist agy reproduced: a caller inlined into a loop
    // under flush-to-zero. Each quotient is a subnormal that must not be
    // flushed. Results are compared as bits: under DAZ a float comparison
    // would read them as zero.
    #[inline(always)]
    #[expect(
        clippy::inline_always,
        reason = "the regression needs the caller inlined"
    )]
    fn halve_all(xs: &[f64], out: &mut Vec<u64>) {
        for &x in xs {
            out.push(with_canonical(|| x / 2.0).to_bits());
        }
    }

    #[test]
    fn test_with_canonical_inlined_loop_under_ftz_keeps_subnormals() {
        with_mode(test_modes::flush_to_zero, |_| {
            let xs = black_box([f64::MIN_POSITIVE; 64]);
            let mut out = Vec::with_capacity(xs.len());
            halve_all(&xs, &mut out);
            assert!(out.iter().all(|&b| b == 0x0008_0000_0000_0000));
        });
    }

    // A loop-invariant closure under FTZ: the call must not be hoisted out
    // of the loop or merged across iterations outside the guard.
    #[test]
    fn test_with_canonical_loop_invariant_closure_under_ftz() {
        with_mode(test_modes::flush_to_zero, |_| {
            let (a, b) = (black_box(f64::MIN_POSITIVE), black_box(2.0));
            let mut out = [0u64; 32];
            for slot in &mut out {
                *slot = with_canonical(|| a / b).to_bits();
            }
            assert!(out.iter().all(|&bits| bits == 0x0008_0000_0000_0000));
        });
    }

    // The case observed in PR #2: a register-only result used after the
    // scope.
    #[test]
    fn test_with_canonical_register_result_used_after_scope_not_flushed() {
        with_mode(test_modes::flush_to_zero, |_| {
            let (a, b) = (black_box(f64::MIN_POSITIVE), black_box(2.0));
            let t = with_canonical(|| a / b);
            assert_eq!(t.to_bits(), 0x0008_0000_0000_0000);
        });
    }

    #[test]
    fn test_with_canonical_restores_on_unwind() {
        with_mode(test_modes::flush_to_zero, |entered| {
            let result = std::panic::catch_unwind(|| with_canonical(|| panic!("unwind")));
            assert!(result.is_err());
            assert_eq!(
                test_modes::control_bits(arch::read()),
                test_modes::control_bits(entered)
            );
        });
    }

    #[test]
    fn test_ensure_canonical_reports_reset() {
        with_mode(test_modes::flush_to_zero, |_| {
            assert!(ensure_canonical());
            assert!(!ensure_canonical());
        });
    }

    #[test]
    fn test_with_hostile_flush_to_zero_flushes_and_restores() {
        let before = arch::read();
        let flushed = with_hostile_flush_to_zero(|| {
            assert!(!is_canonical());
            black_box(black_box(f64::MIN_POSITIVE) / black_box(2.0)).to_bits()
        });
        assert_eq!(flushed, 0);
        assert_eq!(arch::read(), before);
    }

    #[test]
    fn test_establish_canonical_enters_without_guard() {
        with_mode(test_modes::flush_to_zero, |_| {
            establish_canonical();
            assert!(is_canonical());
        });
    }
}
