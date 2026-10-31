// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every boundary thunk of the crate (ADR-0041), and the only place that
//! can make a [`ThunkFn`].
//!
//! A thunk is a naked function with the exact C signature of the slot it
//! fills. Until B8 it is a pass-through: one tail jump to the Rust
//! `extern "C"` implementation, which preserves every argument register,
//! the stack and the callee-saved registers on System V x86-64, Win64 and
//! AAPCS64. B8 turns the template into a real frame that saves,
//! canonicalizes and restores the floating-point controls.
//!
//! [`ThunkFn`]'s field is private to this module, so a C table built from
//! `ThunkFn` values (`tables`) cannot be given a raw function pointer:
//! registering anything but a thunk is a compile error.

use pyo3::ffi;

use super::dlpack::{self, DLManagedTensor, DLManagedTensorVersioned};
use super::{env_type, module, view_type};

/// `PyCFunction`: `(self, args) -> object`.
pub(crate) type CFunction =
    unsafe extern "C" fn(*mut ffi::PyObject, *mut ffi::PyObject) -> *mut ffi::PyObject;
/// `PyCFunctionWithKeywords`: `(self, args, kwargs) -> object`.
pub(crate) type CFunctionKw = unsafe extern "C" fn(
    *mut ffi::PyObject,
    *mut ffi::PyObject,
    *mut ffi::PyObject,
) -> *mut ffi::PyObject;
/// `destructor`: `(object)`.
pub(crate) type Destructor = unsafe extern "C" fn(*mut ffi::PyObject);

/// `getbufferproc`: `(exporter, view, flags) -> 0 | -1`.
pub(crate) type GetBuffer = unsafe extern "C" fn(
    *mut ffi::PyObject,
    *mut ffi::Py_buffer,
    core::ffi::c_int,
) -> core::ffi::c_int;
/// `DLManagedTensor::deleter`.
pub(crate) type DlDeleter = unsafe extern "C" fn(*mut DLManagedTensor);
/// `DLManagedTensorVersioned::deleter`.
pub(crate) type DlDeleterVersioned = unsafe extern "C" fn(*mut DLManagedTensorVersioned);

/// A boundary thunk of C signature `F`. Only this module can make one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ThunkFn<F>(F);

impl<F: Copy> ThunkFn<F> {
    /// The thunk's function pointer, for a C table.
    pub(crate) const fn get(self) -> F {
        self.0
    }
}

/// Defines the naked thunk `$thunk` for `$target` and, with a name, the
/// `ThunkFn` constant tables use.
macro_rules! thunk {
    (@naked $(#[$meta:meta])* $thunk:ident => $target:path : fn($($arg:ty),*) -> $ret:ty) => {
        $(#[$meta])*
        #[unsafe(naked)]
        pub(crate) unsafe extern "C" fn $thunk($(_: $arg),*) -> $ret {
            // SAFETY: a tail jump to a function with the same C signature
            // (checked below): arguments, return value, stack and
            // callee-saved registers pass through untouched.
            #[cfg(target_arch = "x86_64")]
            core::arch::naked_asm!("jmp {f}", f = sym $target);
            #[cfg(target_arch = "aarch64")]
            core::arch::naked_asm!("b {f}", f = sym $target);
        }
        // The target's signature must match the thunk's.
        const _: unsafe extern "C" fn($($arg),*) -> $ret = $target;
    };
    ($name:ident: $ty:ident = $thunk:ident => $target:path : fn($($arg:ty),*) -> $ret:ty) => {
        thunk!(@naked $thunk => $target : fn($($arg),*) -> $ret);
        pub(crate) const $name: ThunkFn<$ty> = ThunkFn($thunk);
    };
}

type Obj = *mut ffi::PyObject;

thunk!(
    @naked
    /// The module initializer CPython looks up by name.
    #[unsafe(no_mangle)]
    PyInit__native => module::init : fn() -> Obj
);

thunk!(VERSION: CFunction = thunk_version => module::version : fn(Obj, Obj) -> Obj);
thunk!(BUILD_PROFILE: CFunction = thunk_build_profile => module::build_profile : fn(Obj, Obj) -> Obj);
thunk!(MAKE_CARTPOLE: CFunctionKw = thunk_make_cartpole => env_type::make_cartpole : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_TAG: CFunctionKw = thunk_make_tag => env_type::make_tag : fn(Obj, Obj, Obj) -> Obj);
thunk!(LAYOUT: CFunction = thunk_layout => env_type::layout : fn(Obj, Obj) -> Obj);
thunk!(OUTPUTS: CFunction = thunk_outputs => env_type::outputs : fn(Obj, Obj) -> Obj);
thunk!(IS_POISONED: CFunction = thunk_is_poisoned => env_type::is_poisoned : fn(Obj, Obj) -> Obj);
thunk!(CLOSE: CFunction = thunk_close => env_type::close : fn(Obj, Obj) -> Obj);
thunk!(RESET: CFunction = thunk_reset => env_type::reset : fn(Obj, Obj) -> Obj);
thunk!(STEP: CFunction = thunk_step => env_type::step : fn(Obj, Obj) -> Obj);
thunk!(SNAPSHOT: CFunction = thunk_snapshot => env_type::snapshot : fn(Obj, Obj) -> Obj);
thunk!(IMPORT_RECORD: CFunction = thunk_import_record => env_type::import_record : fn(Obj, Obj) -> Obj);
thunk!(FORK: CFunction = thunk_fork => env_type::fork : fn(Obj, Obj) -> Obj);
thunk!(DEALLOC: Destructor = thunk_dealloc => env_type::dealloc : fn(Obj) -> ());

// Borrowed output views (ADR-0003). The deleters and the capsule destructor
// are cleanup entries: they never fail or reject.
thunk!(VIEW_DEALLOC: Destructor = thunk_view_dealloc => view_type::dealloc : fn(Obj) -> ());
thunk!(VIEW_GETBUFFER: GetBuffer = thunk_view_getbuffer => view_type::getbuffer : fn(Obj, *mut ffi::Py_buffer, core::ffi::c_int) -> core::ffi::c_int);
thunk!(VIEW_DLPACK: CFunctionKw = thunk_view_dlpack => view_type::dlpack : fn(Obj, Obj, Obj) -> Obj);
thunk!(VIEW_DLPACK_DEVICE: CFunction = thunk_view_dlpack_device => view_type::dlpack_device : fn(Obj, Obj) -> Obj);
thunk!(VIEW_GENERATION: CFunction = thunk_view_generation => view_type::generation : fn(Obj, Obj) -> Obj);
thunk!(VIEW_IS_STALE: CFunction = thunk_view_is_stale => view_type::is_stale : fn(Obj, Obj) -> Obj);
thunk!(CAPSULE_DESTRUCTOR: Destructor = thunk_capsule_destructor => dlpack::destroy_capsule : fn(Obj) -> ());
thunk!(DL_DELETER: DlDeleter = thunk_dl_deleter => dlpack::delete_legacy : fn(*mut DLManagedTensor) -> ());
thunk!(DL_DELETER_VERSIONED: DlDeleterVersioned = thunk_dl_deleter_versioned => dlpack::delete_versioned : fn(*mut DLManagedTensorVersioned) -> ());
