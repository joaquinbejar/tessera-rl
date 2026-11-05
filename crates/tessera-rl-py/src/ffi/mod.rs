// SPDX-License-Identifier: MIT OR Apache-2.0

//! The C boundary with CPython. Audited `unsafe` module (ADR-0041).
//!
//! - `thunks` defines every boundary thunk (ADR-0041) and is the only
//!   module that can make a `ThunkFn`.
//! - `tables` builds every C table row; function entries take a
//!   `ThunkFn`, so a raw function pointer cannot be registered.
//! - `inner` holds an environment object's state behind the reentrancy
//!   guard.

#![expect(unsafe_code, reason = "audited module: the CPython C boundary")]

/// A static C table holding raw pointers, shared read-only with CPython.
#[repr(transparent)]
pub(crate) struct Table<T>(pub(crate) T);

// SAFETY: the tables are written once at compile time and only read
// afterwards (by CPython, holding the interpreter lock).
unsafe impl<T> Sync for Table<T> {}

/// The body of every fallible entry point that returns an object: the
/// floating-point policy applied first (`boundary::check`: a rejection or a
/// failed warning returns before `f` runs), panics contained (no unwind
/// crosses the C boundary), and null returned with an exception set on any
/// failure.
pub(crate) fn entry(
    f: impl FnOnce(capi::Gil) -> capi::CResult<capi::Mine>,
) -> *mut pyo3::ffi::PyObject {
    // SAFETY: every entry point is called by CPython with the interpreter
    // attached to this thread.
    let g = unsafe { capi::Gil::assume() };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        boundary::check(g)?;
        f(g)
    })) {
        Ok(Ok(obj)) => obj.into_ptr(),
        Ok(Err(capi::Raised)) => core::ptr::null_mut(),
        Err(payload) => {
            dispose(payload);
            capi::raise(g, capi::runtime_error(), "internal panic in Tessera");
            core::ptr::null_mut()
        }
    }
}

/// As [`entry`], for slots returning a status: 0, or -1 with an exception
/// set (`getbuffer`).
pub(crate) fn entry_status(f: impl FnOnce(capi::Gil) -> capi::CResult<()>) -> core::ffi::c_int {
    // SAFETY: as in `entry`.
    let g = unsafe { capi::Gil::assume() };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        boundary::check(g)?;
        f(g)
    })) {
        Ok(Ok(())) => 0,
        Ok(Err(capi::Raised)) => -1,
        Err(payload) => {
            dispose(payload);
            capi::raise(g, capi::runtime_error(), "internal panic in Tessera");
            -1
        }
    }
}

/// Disposes of a panic payload without running its `Drop`, which could
/// itself panic across the C boundary: it is leaked.
pub(crate) fn dispose(payload: Box<dyn core::any::Any + Send>) {
    core::mem::forget(payload);
}

mod boundary;
pub(crate) mod capi;
mod dlpack;
mod env_type;
pub(crate) mod fork;
pub(crate) mod gather;
pub(crate) mod inner;
mod module;
mod search_group_type;
mod search_type;
pub(crate) mod snapshot_type;
mod tables;
mod thunks;
pub(crate) mod view_type;
