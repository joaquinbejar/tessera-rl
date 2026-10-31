// SPDX-License-Identifier: MIT OR Apache-2.0

//! The C boundary with CPython. Audited `unsafe` module (ADR-0041).
//!
//! - [`thunk!`] defines one boundary thunk per C entry point: a naked
//!   function with the exact C signature of the slot it fills, which
//!   transfers control to the Rust `extern "C"` implementation. Until B8 it
//!   is a pass-through (a single tail jump, which preserves every argument
//!   register, the stack and callee-saved registers on System V x86-64,
//!   Win64 and AAPCS64); B8 adds the floating-point save, canonicalize and
//!   restore around the call.
//! - Module and type tables ([`module`]) hold only thunk pointers.

#![expect(unsafe_code, reason = "audited module: the CPython C boundary")]

/// Defines the boundary thunk `$thunk` for the `extern "C"` function
/// `$target`, which must have exactly the signature given.
macro_rules! thunk {
    ($(#[$meta:meta])* $thunk:ident => $target:path : fn($($arg:ty),*) -> $ret:ty) => {
        $(#[$meta])*
        #[unsafe(naked)]
        pub(crate) unsafe extern "C" fn $thunk($(_: $arg),*) -> $ret {
            // SAFETY: a tail jump to a function with the same C signature:
            // arguments, return value, stack and callee-saved registers are
            // passed through untouched.
            #[cfg(target_arch = "x86_64")]
            core::arch::naked_asm!("jmp {f}", f = sym $target);
            #[cfg(target_arch = "aarch64")]
            core::arch::naked_asm!("b {f}", f = sym $target);
        }
        // The signature of the target must match the thunk's.
        const _: unsafe extern "C" fn($($arg),*) -> $ret = $target;
    };
}

/// A static C table holding raw pointers, shared read-only with CPython.
#[repr(transparent)]
pub(crate) struct Table<T>(pub(crate) T);

// SAFETY: the tables are written once at compile time and only read
// afterwards (by CPython, holding the interpreter lock).
unsafe impl<T> Sync for Table<T> {}

/// The body of every entry point: attached, panics contained (no unwind
/// crosses the C boundary), errors raised as Python exceptions.
pub(crate) fn entry(
    f: impl FnOnce(pyo3::Python<'_>) -> pyo3::PyResult<*mut pyo3::ffi::PyObject>,
) -> *mut pyo3::ffi::PyObject {
    // SAFETY: every entry point is called by CPython with the interpreter
    // attached to this thread.
    let py = unsafe { pyo3::Python::assume_attached() };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(py))) {
        Ok(Ok(obj)) => obj,
        Ok(Err(err)) => {
            err.restore(py);
            core::ptr::null_mut()
        }
        Err(payload) => {
            // The payload's own drop could panic: leak it instead.
            core::mem::forget(payload);
            pyo3::exceptions::PyRuntimeError::new_err("internal panic in Tessera").restore(py);
            core::ptr::null_mut()
        }
    }
}

mod env_type;
pub(crate) mod module;
