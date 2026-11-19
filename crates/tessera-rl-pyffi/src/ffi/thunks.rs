// SPDX-License-Identifier: MIT OR Apache-2.0

//! The boundary thunk templates (ADR-0041) and the only constructor of a
//! [`ThunkFn`]: one generic naked template per C signature, instantiated
//! for a typed shim with `ThunkFn::of::<E>()` (D2-G, ADR-0062 item 8); the
//! callouts; and the fixed module initializers. Private to this crate: an
//! extension writes safe trait impls (`object`, `module_def`, `buffer`).
//!
//! A thunk is a naked function with the exact C signature of the slot it
//! fills, built from one audited template per ABI (System V x86-64 and
//! AAPCS64, Apple and Linux; Win64 on Windows x86-64, with its SEH unwind
//! data). It:
//!
//! 1. sets up a frame (stack aligned to 16 bytes, CFI for unwinders and
//!    debuggers) and saves every integer argument register;
//! 2. saves the caller's floating-point controls (MXCSR or FPCR) and loads
//!    their canonical value (`tessera_rl_num::fp`: round to nearest, no
//!    flush-to-zero or denormals-are-zero, all exceptions masked, IEEE NaN
//!    handling), so no Rust frame of the entry ever runs under the caller's
//!    controls;
//! 3. calls `boundary::tessera_fp_enter` with the caller's controls, then
//!    the `extern "C"` entry with the original arguments, then
//!    `boundary::tessera_fp_leave`;
//! 4. restores the caller's control bits, keeping every sticky status flag
//!    (raised before the call or inside it; on aarch64 the flags live in
//!    FPSR, which the thunk never writes), and returns the entry's result
//!    unchanged.
//!
//! The template is signature-independent: it preserves all integer
//! argument registers (rdi, rsi, rdx, rcx, r8, r9 on System V; rcx, rdx,
//! r8, r9 on Win64; x0 to x8, including the indirect-result register x8)
//! and the integer return register. It does
//! **not** preserve floating-point argument registers, so a thunk accepts
//! only signatures whose arguments and result are integers or pointers: a
//! type invariant ([`IntClass`]), checked when each thunk is defined.
//!
//! On x86-64 the thunks also preserve the caller's x87 control word
//! (ADR-0044): preserve only, never canonicalized, counted or checked, since
//! Rust on x86-64 computes with SSE only. An entry saves it with `fnstcw`
//! and restores it with `fldcw` on its single exit path; a callout does the
//! same around the host call, so a change the host makes never outlives
//! the callout. Rust never writes the control word, so whenever Rust runs
//! inside a frame it holds that frame's caller's value, and the callout's
//! save is that value. Only the control word: the x87 status word, tag word
//! and stack are untouched (no `fninit`, `fnclex` or `fldenv`). A pending
//! unmasked x87 exception at an entry or at a callback return is an
//! unsupported precondition (ADR-0044): `fldcw` then faults inside the
//! thunk.
//! Callee-saved registers are untouched; no unwind crosses a thunk (the
//! entries are `extern "C"` and contain panics in `ffi::entry`).
//!
//! [`ThunkFn`]'s field is private to this module, so a C table built from
//! `ThunkFn` values (`tables`) cannot be given a raw function pointer:
//! registering anything but a thunk is a compile error.

use core::ffi::{c_char, c_int};

use pyo3::ffi;

use super::boundary::{tessera_fp_caller_controls, tessera_fp_callout_returned};
use super::dlpack::{DLManagedTensor, DLManagedTensorVersioned};

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
/// `getter`: `(object, closure) -> object`, a read-only attribute.
pub(crate) type Getter =
    unsafe extern "C" fn(*mut ffi::PyObject, *mut core::ffi::c_void) -> *mut ffi::PyObject;

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

/// C types passed in integer registers: the only argument and result types
/// a thunk accepts (module docs). Sealed.
pub(crate) trait IntClass: sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
    impl<T> Sealed for *mut T {}
    impl<T> Sealed for *const T {}
    impl Sealed for core::ffi::c_int {}
    impl Sealed for isize {}
    impl Sealed for u64 {}
    impl Sealed for i64 {}
    impl Sealed for () {}
}
impl<T> IntClass for *mut T {}
impl<T> IntClass for *const T {}
impl IntClass for c_int {}
impl IntClass for isize {}
impl IntClass for u64 {}
impl IntClass for i64 {}
impl IntClass for () {}

/// Compile-time check that a type is [`IntClass`].
pub(crate) const fn int_class<T: IntClass>() {}

/// A boundary thunk of C signature `F`. Only this module can make one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ThunkFn<F>(F);

impl<F: Copy> ThunkFn<F> {
    /// The thunk's function pointer, for a C table.
    pub(crate) const fn get(self) -> F {
        self.0
    }
}

/// The audited template, System V x86-64 (Linux, macOS). Frame: rbp, then 80 bytes:
/// [rbp-8..-48] rdi, rsi, rdx, rcx, r8, r9; [rbp-52] caller MXCSR;
/// [rbp-56] scratch; [rbp-64] previous boundary frame; [rbp-66] caller x87
/// control word (ADR-0044); [rbp-80..-67] unused (alignment).
#[cfg(all(target_arch = "x86_64", not(windows)))]
macro_rules! boundary_template {
    ($target:path, $me:path) => {
        core::arch::naked_asm!(
            ".cfi_startproc",
            "push rbp",
            ".cfi_adjust_cfa_offset 8",
            ".cfi_rel_offset rbp, 0",
            "mov rbp, rsp",
            ".cfi_def_cfa_register rbp",
            "sub rsp, 80",
            "mov [rbp - 8], rdi",
            "mov [rbp - 16], rsi",
            "mov [rbp - 24], rdx",
            "mov [rbp - 32], rcx",
            "mov [rbp - 40], r8",
            "mov [rbp - 48], r9",
            // Save the caller's x87 control word (ADR-0044, no-wait form).
            "fnstcw [rbp - 66]",
            // Save the caller's MXCSR and load its canonical value
            // (status flags kept).
            "stmxcsr [rbp - 52]",
            "mov eax, [rbp - 52]",
            "and eax, {keep}",
            "or eax, {set}",
            "mov [rbp - 56], eax",
            "ldmxcsr [rbp - 56]",
            // Record the frame (Rust, canonical controls).
            "mov edi, [rbp - 52]",
            "call {enter}",
            "mov [rbp - 64], rax",
            // The entry, with the original arguments.
            "mov rdi, [rbp - 8]",
            "mov rsi, [rbp - 16]",
            "mov rdx, [rbp - 24]",
            "mov rcx, [rbp - 32]",
            "mov r8, [rbp - 40]",
            "mov r9, [rbp - 48]",
            "call {target}",
            "mov [rbp - 8], rax",
            "mov rdi, [rbp - 64]",
            "call {leave}",
            // Restore the caller's control bits with every sticky flag:
            // (caller & !STATUS) | ((caller | current) & STATUS).
            "stmxcsr [rbp - 56]",
            "mov eax, [rbp - 52]",
            "mov ecx, [rbp - 56]",
            "or ecx, eax",
            "and ecx, {status}",
            "and eax, {not_status}",
            "or eax, ecx",
            "mov [rbp - 56], eax",
            "ldmxcsr [rbp - 56]",
            // The caller's x87 control word back (ADR-0044).
            "fldcw [rbp - 66]",
            "mov rax, [rbp - 8]",
            "leave",
            ".cfi_def_cfa rsp, 8",
            "ret",
            ".cfi_endproc",
            keep = const !tessera_rl_num::fp::CANONICAL_CLEAR,
            set = const tessera_rl_num::fp::CANONICAL_SET,
            status = const tessera_rl_num::fp::STATUS_FLAGS,
            not_status = const !tessera_rl_num::fp::STATUS_FLAGS,
            enter = sym crate::ffi::boundary::tessera_fp_enter,
            leave = sym crate::ffi::boundary::tessera_fp_leave,
            target = sym $target,
        )
    };
}

/// The audited template, AAPCS64 (Apple and Linux). Frame of 112 bytes:
/// [sp] x29, x30; [sp+16..+80] x0 to x8; [sp+88] caller FPCR; [sp+96]
/// previous boundary frame. FPSR (the sticky flags) is never written.
#[cfg(all(target_arch = "aarch64", not(windows)))]
macro_rules! boundary_template {
    ($target:path, $me:path) => {
        core::arch::naked_asm!(
            ".cfi_startproc",
            "stp x29, x30, [sp, #-112]!",
            ".cfi_def_cfa_offset 112",
            ".cfi_offset x30, -104",
            ".cfi_offset x29, -112",
            "mov x29, sp",
            ".cfi_def_cfa x29, 112",
            "stp x0, x1, [sp, #16]",
            "stp x2, x3, [sp, #32]",
            "stp x4, x5, [sp, #48]",
            "stp x6, x7, [sp, #64]",
            "str x8, [sp, #80]",
            // Save the caller's FPCR and load its canonical value.
            "mrs x9, fpcr",
            "str x9, [sp, #88]",
            "movz x10, #{k0}",
            "movk x10, #{k1}, lsl #16",
            "movk x10, #{k2}, lsl #32",
            "movk x10, #{k3}, lsl #48",
            "and x10, x9, x10",
            "msr fpcr, x10",
            // Record the frame (Rust, canonical controls).
            "mov x0, x9",
            "bl {enter}",
            "str x0, [sp, #96]",
            // The entry, with the original arguments.
            "ldp x0, x1, [sp, #16]",
            "ldp x2, x3, [sp, #32]",
            "ldp x4, x5, [sp, #48]",
            "ldp x6, x7, [sp, #64]",
            "ldr x8, [sp, #80]",
            "bl {target}",
            "str x0, [sp, #16]",
            "ldr x0, [sp, #96]",
            "bl {leave}",
            // Restore the caller's FPCR (it holds no status flags).
            "ldr x9, [sp, #88]",
            "msr fpcr, x9",
            "ldr x0, [sp, #16]",
            "ldp x29, x30, [sp], #112",
            ".cfi_def_cfa sp, 0",
            ".cfi_restore x30",
            ".cfi_restore x29",
            "ret",
            ".cfi_endproc",
            k0 = const (!tessera_rl_num::fp::CANONICAL_CLEAR) & 0xffff,
            k1 = const ((!tessera_rl_num::fp::CANONICAL_CLEAR) >> 16) & 0xffff,
            k2 = const ((!tessera_rl_num::fp::CANONICAL_CLEAR) >> 32) & 0xffff,
            k3 = const ((!tessera_rl_num::fp::CANONICAL_CLEAR) >> 48) & 0xffff,
            enter = sym crate::ffi::boundary::tessera_fp_enter,
            leave = sym crate::ffi::boundary::tessera_fp_leave,
            target = sym $target,
        )
    };
}

/// The audited template, Win64 (Windows x86-64). Arguments in rcx, rdx,
/// r8, r9 (four at most: [`MAX_ARGS`]; a fifth would be on the caller's
/// stack, which the template does not forward). No frame pointer and no
/// callee-saved register is used (rsi, rdi, rbx, rbp, r12 to r15 and xmm6
/// to xmm15 are untouched), so the SEH unwind data is a single stack
/// allocation: `.seh_stackalloc 104`. On entry rsp is 8 mod 16, so after it
/// rsp is 16-byte aligned for the calls. Frame (rsp-relative): [rsp+0..32]
/// the shadow space of the calls; [rsp+32..64] rcx, rdx, r8, r9; [rsp+64]
/// caller MXCSR; [rsp+68] scratch; [rsp+72] previous boundary frame;
/// [rsp+80] caller x87 control word (ADR-0044; unverified on native
/// Windows until the Windows session checks it); [rsp+82..104] unused.
#[cfg(all(target_arch = "x86_64", windows))]
macro_rules! boundary_template {
    ($target:path, $me:path) => {
        core::arch::naked_asm!(
            ".seh_proc {me}",
            "sub rsp, 104",
            ".seh_stackalloc 104",
            ".seh_endprologue",
            "mov [rsp + 32], rcx",
            "mov [rsp + 40], rdx",
            "mov [rsp + 48], r8",
            "mov [rsp + 56], r9",
            // Save the caller's x87 control word (ADR-0044, no-wait form).
            "fnstcw [rsp + 80]",
            // Save the caller's MXCSR and load its canonical value
            // (status flags kept).
            "stmxcsr [rsp + 64]",
            "mov eax, [rsp + 64]",
            "and eax, {keep}",
            "or eax, {set}",
            "mov [rsp + 68], eax",
            "ldmxcsr [rsp + 68]",
            // Record the frame (Rust, canonical controls); the 32-bit move
            // zero-extends into the u64 argument.
            "mov ecx, [rsp + 64]",
            "call {enter}",
            "mov [rsp + 72], rax",
            // The entry, with the original arguments.
            "mov rcx, [rsp + 32]",
            "mov rdx, [rsp + 40]",
            "mov r8, [rsp + 48]",
            "mov r9, [rsp + 56]",
            "call {target}",
            "mov [rsp + 32], rax",
            "mov rcx, [rsp + 72]",
            "call {leave}",
            // Restore the caller's control bits with every sticky flag:
            // (caller & !STATUS) | ((caller | current) & STATUS).
            "stmxcsr [rsp + 68]",
            "mov eax, [rsp + 64]",
            "mov ecx, [rsp + 68]",
            "or ecx, eax",
            "and ecx, {status}",
            "and eax, {not_status}",
            "or eax, ecx",
            "mov [rsp + 68], eax",
            "ldmxcsr [rsp + 68]",
            // The caller's x87 control word back (ADR-0044).
            "fldcw [rsp + 80]",
            "mov rax, [rsp + 32]",
            // The epilogue form the Win64 unwinder recognizes.
            "add rsp, 104",
            "ret",
            ".seh_endproc",
            keep = const !tessera_rl_num::fp::CANONICAL_CLEAR,
            set = const tessera_rl_num::fp::CANONICAL_SET,
            status = const tessera_rl_num::fp::STATUS_FLAGS,
            not_status = const !tessera_rl_num::fp::STATUS_FLAGS,
            enter = sym crate::ffi::boundary::tessera_fp_enter,
            leave = sym crate::ffi::boundary::tessera_fp_leave,
            target = sym $target,
            me = sym $me,
        )
    };
}

#[cfg(all(windows, not(target_arch = "x86_64")))]
compile_error!("the floating-point boundary (ADR-0041) supports Windows on x86-64 only");

/// Which boundary this build carries (`tessera_rl._native.fp_boundary()`).
pub const BOUNDARY: &str = if cfg!(windows) {
    "Win64"
} else if cfg!(target_arch = "x86_64") {
    "System V x86-64"
} else {
    "AAPCS64"
};

/// Integer argument registers the templates preserve.
#[cfg(all(target_arch = "x86_64", not(windows)))]
pub(crate) const MAX_ARGS: usize = 6;
#[cfg(all(target_arch = "x86_64", windows))]
pub(crate) const MAX_ARGS: usize = 4;
#[cfg(target_arch = "aarch64")]
pub(crate) const MAX_ARGS: usize = 8;

/// The audited callout template, System V x86-64 (ADR-0041 callouts): a
/// call from Rust, under canonical controls, into C-API code that can run
/// Python code. It saves the integer argument registers, loads the
/// caller's control bits (the current boundary frame's, kept by
/// `boundary`) with the current sticky flags, calls the C function, then
/// reloads the canonical controls (keeping the flags the host raised) and
/// lets `tessera_fp_callout_returned` count a change before returning to
/// Rust. Around the C call it saves and reloads the x87 control word
/// (ADR-0044: a host change never outlives the callout; not counted).
/// Frame: [rbp-8..-48] rdi..r9; [rbp-52] canonical MXCSR; [rbp-56] the
/// value loaded; [rbp-60] the value after; [rbp-64] scratch; [rbp-66] the
/// x87 control word; [rbp-80..-67] unused (alignment).
#[cfg(all(target_arch = "x86_64", not(windows)))]
macro_rules! callout_template {
    ($target:path, $me:path) => {
        core::arch::naked_asm!(
            ".cfi_startproc",
            "push rbp",
            ".cfi_adjust_cfa_offset 8",
            ".cfi_rel_offset rbp, 0",
            "mov rbp, rsp",
            ".cfi_def_cfa_register rbp",
            "sub rsp, 80",
            "mov [rbp - 8], rdi",
            "mov [rbp - 16], rsi",
            "mov [rbp - 24], rdx",
            "mov [rbp - 32], rcx",
            "mov [rbp - 40], r8",
            "mov [rbp - 48], r9",
            // The frame caller's x87 control word (Rust never changes it).
            "fnstcw [rbp - 66]",
            "stmxcsr [rbp - 52]",
            "call {caller}",
            "mov ecx, [rbp - 52]",
            "and ecx, {status}",
            "and eax, {not_status}",
            "or eax, ecx",
            "mov [rbp - 56], eax",
            "ldmxcsr [rbp - 56]",
            "mov rdi, [rbp - 8]",
            "mov rsi, [rbp - 16]",
            "mov rdx, [rbp - 24]",
            "mov rcx, [rbp - 32]",
            "mov r8, [rbp - 40]",
            "mov r9, [rbp - 48]",
            "call {target}",
            // Undo any x87 control change by the host before Rust resumes.
            "fldcw [rbp - 66]",
            "mov [rbp - 8], rax",
            "stmxcsr [rbp - 60]",
            "mov eax, [rbp - 52]",
            "mov ecx, [rbp - 60]",
            "and ecx, {status}",
            "or eax, ecx",
            "mov [rbp - 64], eax",
            "ldmxcsr [rbp - 64]",
            "mov edi, [rbp - 56]",
            "mov esi, [rbp - 60]",
            "call {returned}",
            "mov rax, [rbp - 8]",
            "leave",
            ".cfi_def_cfa rsp, 8",
            "ret",
            ".cfi_endproc",
            status = const tessera_rl_num::fp::STATUS_FLAGS,
            not_status = const !tessera_rl_num::fp::STATUS_FLAGS,
            caller = sym tessera_fp_caller_controls,
            returned = sym tessera_fp_callout_returned,
            target = sym $target,
        )
    };
}

/// The audited callout template, AAPCS64. Frame of 112 bytes: [sp] x29,
/// x30; [sp+16..+80] x0 to x8; [sp+88] canonical FPCR; [sp+96] the
/// caller's FPCR loaded for the call.
#[cfg(all(target_arch = "aarch64", not(windows)))]
macro_rules! callout_template {
    ($target:path, $me:path) => {
        core::arch::naked_asm!(
            ".cfi_startproc",
            "stp x29, x30, [sp, #-112]!",
            ".cfi_def_cfa_offset 112",
            ".cfi_offset x30, -104",
            ".cfi_offset x29, -112",
            "mov x29, sp",
            ".cfi_def_cfa x29, 112",
            "stp x0, x1, [sp, #16]",
            "stp x2, x3, [sp, #32]",
            "stp x4, x5, [sp, #48]",
            "stp x6, x7, [sp, #64]",
            "str x8, [sp, #80]",
            "mrs x9, fpcr",
            "str x9, [sp, #88]",
            "bl {caller}",
            "str x0, [sp, #96]",
            "msr fpcr, x0",
            "ldp x0, x1, [sp, #16]",
            "ldp x2, x3, [sp, #32]",
            "ldp x4, x5, [sp, #48]",
            "ldp x6, x7, [sp, #64]",
            "ldr x8, [sp, #80]",
            "bl {target}",
            "str x0, [sp, #16]",
            "mrs x1, fpcr",
            "ldr x9, [sp, #88]",
            "msr fpcr, x9",
            "ldr x0, [sp, #96]",
            "bl {returned}",
            "ldr x0, [sp, #16]",
            "ldp x29, x30, [sp], #112",
            ".cfi_def_cfa sp, 0",
            ".cfi_restore x30",
            ".cfi_restore x29",
            "ret",
            ".cfi_endproc",
            caller = sym tessera_fp_caller_controls,
            returned = sym tessera_fp_callout_returned,
            target = sym $target,
        )
    };
}

/// The audited callout template, Win64 (the frame and SEH data of the
/// Win64 boundary template). Frame: [rsp+0..32] shadow space;
/// [rsp+32..64] rcx, rdx, r8, r9; [rsp+64] canonical MXCSR; [rsp+68] the
/// value loaded for the call; [rsp+72] the value after; [rsp+76] scratch;
/// [rsp+80] the x87 control word (ADR-0044, unverified on native Windows).
#[cfg(all(target_arch = "x86_64", windows))]
macro_rules! callout_template {
    ($target:path, $me:path) => {
        core::arch::naked_asm!(
            ".seh_proc {me}",
            "sub rsp, 104",
            ".seh_stackalloc 104",
            ".seh_endprologue",
            "mov [rsp + 32], rcx",
            "mov [rsp + 40], rdx",
            "mov [rsp + 48], r8",
            "mov [rsp + 56], r9",
            // The frame caller's x87 control word (Rust never changes it).
            "fnstcw [rsp + 80]",
            "stmxcsr [rsp + 64]",
            "call {caller}",
            "mov ecx, [rsp + 64]",
            "and ecx, {status}",
            "and eax, {not_status}",
            "or eax, ecx",
            "mov [rsp + 68], eax",
            "ldmxcsr [rsp + 68]",
            "mov rcx, [rsp + 32]",
            "mov rdx, [rsp + 40]",
            "mov r8, [rsp + 48]",
            "mov r9, [rsp + 56]",
            "call {target}",
            // Undo any x87 control change by the host before Rust resumes.
            "fldcw [rsp + 80]",
            "mov [rsp + 32], rax",
            "stmxcsr [rsp + 72]",
            "mov eax, [rsp + 64]",
            "mov ecx, [rsp + 72]",
            "and ecx, {status}",
            "or eax, ecx",
            "mov [rsp + 76], eax",
            "ldmxcsr [rsp + 76]",
            "mov ecx, [rsp + 68]",
            "mov edx, [rsp + 72]",
            "call {returned}",
            "mov rax, [rsp + 32]",
            "add rsp, 104",
            "ret",
            ".seh_endproc",
            status = const tessera_rl_num::fp::STATUS_FLAGS,
            not_status = const !tessera_rl_num::fp::STATUS_FLAGS,
            caller = sym tessera_fp_caller_controls,
            returned = sym tessera_fp_callout_returned,
            target = sym $target,
            me = sym $me,
        )
    };
}

/// Defines the callout `$name` for the C function `$target`: integer-class
/// arguments and result only, at most [`MAX_ARGS`] arguments.
macro_rules! callout {
    ($(#[$meta:meta])* $name:ident => $target:path : fn($($arg:ty),*) -> $ret:ty) => {
        $(#[$meta])*
        #[unsafe(naked)]
        pub(crate) unsafe extern "C" fn $name($(_: $arg),*) -> $ret {
            // SAFETY: the audited callout template (callout_template docs):
            // a full frame that preserves every integer argument register,
            // the stack alignment and the callee-saved registers, runs the
            // C function under the caller's controls and returns its integer
            // result under canonical controls; the target has the same C
            // signature (checked below) with integer-class types only.
            callout_template!($target, $name);
        }
        const _: unsafe extern "C" fn($($arg),*) -> $ret = $target;
        const _: () = { $( int_class::<$arg>(); )* int_class::<$ret>(); };
        const _: () = assert!(<[&str]>::len(&[$(stringify!($arg)),*]) <= MAX_ARGS, "too many arguments for the template");
    };
}

/// Defines the non-generic boundary thunk `$thunk` for `$target`: the
/// fixed module initializers, which carry `no_mangle` (a generic
/// instantiation cannot). Gated like its only users, the two
/// initializers below.
#[cfg(any(feature = "native-module", feature = "host"))]
macro_rules! thunk {
    (@naked $(#[$meta:meta])* $thunk:ident => $target:path : fn($($arg:ty),*) -> $ret:ty) => {
        $(#[$meta])*
        #[unsafe(naked)]
        pub(crate) unsafe extern "C" fn $thunk($(_: $arg),*) -> $ret {
            // SAFETY: the audited template (module docs): a full frame that
            // preserves every integer argument register, the stack
            // alignment and the callee-saved registers, canonicalizes and
            // restores the floating-point controls, and returns the
            // target's integer result; the target has the same C signature
            // (checked below) with integer-class types only.
            boundary_template!($target, $thunk);
        }
        // The target's signature must match the thunk's.
        const _: unsafe extern "C" fn($($arg),*) -> $ret = $target;
        // Integer-class arguments and result only (module docs), and no
        // more than the template preserves.
        const _: () = { $( int_class::<$arg>(); )* int_class::<$ret>(); };
        const _: () = assert!(<[&str]>::len(&[$(stringify!($arg)),*]) <= MAX_ARGS, "too many arguments for the template");
    };
}

type Obj = *mut ffi::PyObject;

/// Defines the entry trait `$entry` of one C signature and its generic
/// boundary thunk `$thunk::<E>` (D2-G, ADR-0062 item 8): the audited
/// template, byte for byte, with `E::call` as its target. A type implements
/// the trait to become an entry; [`ThunkFn::of`] is the only way to put an
/// instantiation in a C table.
macro_rules! generic_thunk {
    ($(#[$meta:meta])* $entry:ident, $thunk:ident : fn($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        $(#[$meta])*
        pub(crate) trait $entry {
            /// The entry the thunk calls, under canonical controls.
            ///
            /// # Safety
            ///
            /// Called only by the thunk, with the arguments CPython (or the
            /// DLPack consumer) passed to the slot.
            unsafe extern "C" fn call($($arg: $ty),*) -> $ret;
        }

        #[doc = concat!("The boundary thunk of an entry `E: ", stringify!($entry), "`.")]
        #[unsafe(naked)]
        pub(crate) unsafe extern "C" fn $thunk<E: $entry>($(_: $ty),*) -> $ret {
            // SAFETY: the audited template (module docs): a full frame that
            // preserves every integer argument register, the stack
            // alignment and the callee-saved registers, canonicalizes and
            // restores the floating-point controls, and returns the
            // target's integer result; the target, `E::call`, has the
            // thunk's C signature (the trait's) with integer-class types
            // only (checked below).
            boundary_template!(E::call, $thunk::<E>);
        }
        // Integer-class arguments and result only (module docs), and no
        // more than the template preserves.
        const _: () = { $( int_class::<$ty>(); )* int_class::<$ret>(); };
        const _: () = assert!(<[&str]>::len(&[$(stringify!($ty)),*]) <= MAX_ARGS, "too many arguments for the template");
    };
}

generic_thunk!(
    /// A `PyCFunction` entry: a method without keywords or a module function.
    CFunctionEntry, cfunction : fn(slf: Obj, args: Obj) -> Obj
);
generic_thunk!(
    /// A `PyCFunctionWithKeywords` entry.
    CFunctionKwEntry, cfunction_kw : fn(slf: Obj, args: Obj, kwargs: Obj) -> Obj
);
generic_thunk!(
    /// A `getter` entry: a read-only attribute (`Py_tp_getset`).
    GetterEntry, getter : fn(slf: Obj, closure: *mut core::ffi::c_void) -> Obj
);
generic_thunk!(
    /// A `destructor` entry: `tp_dealloc` or a capsule destructor.
    DestructorEntry, destructor : fn(obj: Obj) -> ()
);
generic_thunk!(
    /// A `getbufferproc` entry.
    GetBufferEntry, getbuffer : fn(obj: Obj, view: *mut ffi::Py_buffer, flags: c_int) -> c_int
);
generic_thunk!(
    /// A `releasebufferproc` entry.
    ReleaseBufferEntry, release_buffer : fn(obj: Obj, view: *mut ffi::Py_buffer) -> ()
);
generic_thunk!(
    /// A `DLManagedTensor::deleter` entry.
    DlDeleterEntry, dl_deleter : fn(managed: *mut DLManagedTensor) -> ()
);
generic_thunk!(
    /// A `DLManagedTensorVersioned::deleter` entry.
    DlDeleterVersionedEntry, dl_deleter_versioned : fn(managed: *mut DLManagedTensorVersioned) -> ()
);

// The fixed module initializers CPython looks up by name (P1 of the #272
// step 3 design): non-generic, so they can carry `no_mangle`.
#[cfg(feature = "native-module")]
thunk!(
    @naked
    /// `_native`'s initializer: the module registered with
    /// `export_module!(native: ...)`.
    #[unsafe(no_mangle)]
    PyInit__native => super::module_def::init_native : fn() -> *mut ffi::PyObject
);
#[cfg(feature = "host")]
thunk!(
    @naked
    /// A host environment wheel's initializer (ADR-0062 D3): the module
    /// registered with `export_module!(host: ...)`.
    #[unsafe(no_mangle)]
    PyInit__tessera_host => super::module_def::init_host : fn() -> *mut ffi::PyObject
);

impl ThunkFn<CFunction> {
    /// The boundary thunk of `E`.
    pub(crate) const fn of<E: CFunctionEntry>() -> Self {
        Self(cfunction::<E>)
    }
}

impl ThunkFn<CFunctionKw> {
    /// The boundary thunk of `E`.
    pub(crate) const fn of<E: CFunctionKwEntry>() -> Self {
        Self(cfunction_kw::<E>)
    }
}

impl ThunkFn<Getter> {
    /// The boundary thunk of `E`.
    pub(crate) const fn of<E: GetterEntry>() -> Self {
        Self(getter::<E>)
    }
}

impl ThunkFn<Destructor> {
    /// The boundary thunk of `E`.
    pub(crate) const fn of<E: DestructorEntry>() -> Self {
        Self(destructor::<E>)
    }
}

impl ThunkFn<GetBuffer> {
    /// The boundary thunk of `E`.
    pub(crate) const fn of<E: GetBufferEntry>() -> Self {
        Self(getbuffer::<E>)
    }
}

/// `releasebufferproc`: `(exporter, view)`.
pub(crate) type ReleaseBuffer = unsafe extern "C" fn(*mut ffi::PyObject, *mut ffi::Py_buffer);

impl ThunkFn<ReleaseBuffer> {
    /// The boundary thunk of `E`.
    pub(crate) const fn of<E: ReleaseBufferEntry>() -> Self {
        Self(release_buffer::<E>)
    }
}

impl ThunkFn<DlDeleter> {
    /// The boundary thunk of `E`.
    pub(crate) const fn of<E: DlDeleterEntry>() -> Self {
        Self(dl_deleter::<E>)
    }
}

impl ThunkFn<DlDeleterVersioned> {
    /// The boundary thunk of `E`.
    pub(crate) const fn of<E: DlDeleterVersionedEntry>() -> Self {
        Self(dl_deleter_versioned::<E>)
    }
}

#[cfg(test)]
mod tests {
    //! The template against hostile caller controls (ADR-0041 evidence,
    //! run in release builds on each ABI): canonical controls inside,
    //! arguments and result intact, the caller's controls restored with the
    //! sticky flags, the entry counted, nesting and panics contained.

    use core::cell::Cell;
    use core::hint::black_box;

    use tessera_rl_num::fp::{self, hostile};

    use super::super::boundary;
    use super::*;

    // The template tests run through generic instantiations (D2-G): one
    // entry trait per test signature, made by the same macro as the
    // CPython signatures'.

    #[cfg(not(windows))]
    mod six {
        use super::super::{MAX_ARGS, int_class};
        generic_thunk!(Args6, args6 : fn(a0: u64, a1: u64, a2: u64, a3: u64, a4: u64, a5: u64) -> u64);
    }
    #[cfg(not(windows))]
    use six::{Args6, args6};

    #[cfg(windows)]
    mod four {
        use super::super::{MAX_ARGS, int_class};
        generic_thunk!(Args4, args4 : fn(a0: u64, a1: u64, a2: u64, a3: u64) -> u64);
    }
    #[cfg(windows)]
    use four::{Args4, args4};

    generic_thunk!(Args1, args1 : fn(a0: u64) -> u64);

    // Per thread: the tests run in parallel threads.
    thread_local! {
        static INSIDE_CANONICAL: Cell<bool> = const { Cell::new(false) };
        static INSIDE_NON_CANONICAL_FRAME: Cell<bool> = const { Cell::new(false) };
        static INSIDE_DEPTH_TOKEN: Cell<u64> = const { Cell::new(0) };
    }

    extern "C" fn target6(a0: u64, a1: u64, a2: u64, a3: u64, a4: u64, a5: u64) -> u64 {
        target4(a0, a1, a2, a3) ^ a4.rotate_left(32) ^ a5.rotate_left(40)
    }

    /// The first `MAX_ARGS`-compatible part of `target6`: the four
    /// register arguments of every ABI.
    extern "C" fn target4(a0: u64, a1: u64, a2: u64, a3: u64) -> u64 {
        INSIDE_CANONICAL.with(|c| c.set(fp::is_canonical()));
        let non_canonical = boundary::frame().is_some_and(|(nc, _)| nc);
        INSIDE_NON_CANONICAL_FRAME.with(|c| c.set(non_canonical));
        // Raises the inexact flag inside the entry.
        black_box(black_box(1.0f64) / black_box(3.0f64));
        a0 ^ a1.rotate_left(8) ^ a2.rotate_left(16) ^ a3.rotate_left(24)
    }

    // System V and AAPCS64 preserve six or more integer registers: the
    // thunk takes all six arguments.
    #[cfg(not(windows))]
    struct Target6;
    #[cfg(not(windows))]
    impl Args6 for Target6 {
        unsafe extern "C" fn call(a0: u64, a1: u64, a2: u64, a3: u64, a4: u64, a5: u64) -> u64 {
            target6(a0, a1, a2, a3, a4, a5)
        }
    }
    #[cfg(not(windows))]
    const THUNK6: unsafe extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64 = args6::<Target6>;

    // Win64 preserves four (MAX_ARGS): the thunk takes the register
    // arguments, and the test helper folds in the last two outside it, so
    // every test checks the same value on every ABI.
    #[cfg(windows)]
    struct Target4;
    #[cfg(windows)]
    impl Args4 for Target4 {
        unsafe extern "C" fn call(a0: u64, a1: u64, a2: u64, a3: u64) -> u64 {
            target4(a0, a1, a2, a3)
        }
    }
    #[cfg(windows)]
    const THUNK4: unsafe extern "C" fn(u64, u64, u64, u64) -> u64 = args4::<Target4>;

    #[cfg(windows)]
    unsafe extern "C" fn thunk6_win64(a0: u64, a1: u64, a2: u64, a3: u64, a4: u64, a5: u64) -> u64 {
        // SAFETY: a thunk of a test target.
        unsafe { THUNK4(a0, a1, a2, a3) ^ a4.rotate_left(32) ^ a5.rotate_left(40) }
    }
    #[cfg(windows)]
    const THUNK6: unsafe extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64 = thunk6_win64;

    extern "C" fn nested(a: u64) -> u64 {
        // SAFETY: a thunk of a test target.
        let inner = unsafe { THUNK6(a, 1, 2, 3, 4, 5) };
        INSIDE_DEPTH_TOKEN.with(|c| c.set(boundary::frame_token()));
        inner
    }
    struct Nested;
    impl Args1 for Nested {
        unsafe extern "C" fn call(a0: u64) -> u64 {
            nested(a0)
        }
    }
    const THUNK_NESTED: unsafe extern "C" fn(u64) -> u64 = args1::<Nested>;

    extern "C" fn panicky(a: u64) -> u64 {
        match std::panic::catch_unwind(|| {
            assert!(a != 1, "a contained panic");
            a
        }) {
            Ok(v) => v,
            Err(payload) => {
                super::super::dispose(payload);
                u64::MAX
            }
        }
    }
    struct Panicky;
    impl Args1 for Panicky {
        unsafe extern "C" fn call(a0: u64) -> u64 {
            panicky(a0)
        }
    }
    const THUNK_PANICKY: unsafe extern "C" fn(u64) -> u64 = args1::<Panicky>;

    const ARGS: [u64; 6] = [
        0x0123_4567_89ab_cdef,
        0xfedc_ba98_7654_3210,
        0x1111_2222_3333_4444,
        0x5555_6666_7777_8888,
        0x9999_aaaa_bbbb_cccc,
        0xdddd_eeee_ffff_0001,
    ];

    fn expected() -> u64 {
        target6(ARGS[0], ARGS[1], ARGS[2], ARGS[3], ARGS[4], ARGS[5])
    }

    fn call6() -> u64 {
        // SAFETY: a thunk of a test target with integer arguments.
        unsafe { THUNK6(ARGS[0], ARGS[1], ARGS[2], ARGS[3], ARGS[4], ARGS[5]) }
    }

    fn check_mode(mode: fn(fp::Control) -> fp::Control) {
        let want = expected();
        let before = fp::non_canonical_entries();
        let (got, after) = hostile::run(mode, call6);
        assert_eq!(got, want, "arguments and result pass through");
        assert!(INSIDE_CANONICAL.with(Cell::get), "canonical inside");
        assert!(
            INSIDE_NON_CANONICAL_FRAME.with(Cell::get),
            "the frame records the caller"
        );
        // What the CPU actually latches for `mode` (no entry in between):
        // a bit the CPU does not implement reads as zero (for example
        // FPCR.AH without FEAT_AFP is RES0), so the restore is checked
        // against the written-and-read-back controls. Every writable bit is
        // still compared.
        let ((), latched) = hostile::run(mode, || ());
        assert_eq!(
            hostile::control_bits(after),
            hostile::control_bits(latched),
            "the caller's control bits are restored"
        );
        assert!(fp::non_canonical_entries() > before, "the entry is counted");
        assert_eq!(boundary::frame_token(), 0, "no frame outside the entry");
    }

    // The production templates, instantiated with test entries: each
    // records whether it ran under canonical controls and folds its
    // pointer arguments (never dereferenced) into its result.
    thread_local! {
        static SEEN: Cell<(bool, usize)> = const { Cell::new((false, 0)) };
    }

    fn seen(fold: usize) {
        SEEN.with(|c| c.set((fp::is_canonical(), fold)));
    }

    fn ptr(v: usize) -> Obj {
        core::ptr::without_provenance_mut(v)
    }

    struct PyFn;
    impl CFunctionEntry for PyFn {
        unsafe extern "C" fn call(slf: Obj, args: Obj) -> Obj {
            seen(slf.addr() ^ args.addr().rotate_left(8));
            ptr(slf.addr() ^ args.addr())
        }
    }

    struct PyFnKw;
    impl CFunctionKwEntry for PyFnKw {
        unsafe extern "C" fn call(slf: Obj, args: Obj, kwargs: Obj) -> Obj {
            seen(slf.addr() ^ args.addr().rotate_left(8) ^ kwargs.addr().rotate_left(16));
            ptr(slf.addr() ^ kwargs.addr())
        }
    }

    struct Get;
    impl GetterEntry for Get {
        unsafe extern "C" fn call(slf: Obj, closure: *mut core::ffi::c_void) -> Obj {
            seen(slf.addr() ^ closure.addr().rotate_left(8));
            ptr(closure.addr())
        }
    }

    struct Dealloc;
    impl DestructorEntry for Dealloc {
        unsafe extern "C" fn call(obj: Obj) {
            seen(obj.addr());
        }
    }

    struct Buffer;
    impl GetBufferEntry for Buffer {
        unsafe extern "C" fn call(obj: Obj, view: *mut ffi::Py_buffer, flags: c_int) -> c_int {
            seen(obj.addr() ^ view.addr().rotate_left(8) ^ usize::try_from(flags).unwrap_or(0));
            flags.wrapping_neg()
        }
    }

    struct Deleter;
    impl DlDeleterEntry for Deleter {
        unsafe extern "C" fn call(managed: *mut DLManagedTensor) {
            seen(managed.addr());
        }
    }

    struct DeleterVersioned;
    impl DlDeleterVersionedEntry for DeleterVersioned {
        unsafe extern "C" fn call(managed: *mut DLManagedTensorVersioned) {
            seen(managed.addr());
        }
    }

    /// Runs `f` under flush-to-zero caller controls: the instantiation ran
    /// canonical, folded `want` and left the caller's controls in place.
    fn check_instantiation<R: PartialEq + core::fmt::Debug>(
        want: usize,
        result: &R,
        f: impl FnOnce() -> R,
    ) {
        SEEN.with(|c| c.set((false, 0)));
        let (got, after) = hostile::run(hostile::flush_to_zero, f);
        let ((), written) = hostile::run(hostile::flush_to_zero, || ());
        assert_eq!(&got, result, "the result passes through");
        assert_eq!(
            SEEN.with(Cell::get),
            (true, want),
            "canonical inside, arguments intact"
        );
        assert_eq!(
            after & !fp::STATUS_FLAGS,
            written & !fp::STATUS_FLAGS,
            "the caller's controls are restored"
        );
    }

    #[test]
    fn test_thunk_generic_cfunction_runs_canonical_and_passes_arguments() {
        let f = ThunkFn::<CFunction>::of::<PyFn>().get();
        // SAFETY: a thunk of a test entry; the pointers are never dereferenced.
        check_instantiation(0x10 ^ 0x2000, &ptr(0x10 ^ 0x20), || unsafe {
            f(ptr(0x10), ptr(0x20))
        });
    }

    #[test]
    fn test_thunk_generic_cfunction_kw_runs_canonical_and_passes_arguments() {
        let f = ThunkFn::<CFunctionKw>::of::<PyFnKw>().get();
        // SAFETY: as above.
        check_instantiation(0x10 ^ 0x2000 ^ 0x30_0000, &ptr(0x10 ^ 0x30), || unsafe {
            f(ptr(0x10), ptr(0x20), ptr(0x30))
        });
    }

    #[test]
    fn test_thunk_generic_getter_runs_canonical_and_passes_arguments() {
        let f = ThunkFn::<Getter>::of::<Get>().get();
        // SAFETY: as above.
        check_instantiation(0x10 ^ 0x9000, &ptr(0x90), || unsafe {
            f(ptr(0x10), core::ptr::without_provenance_mut(0x90))
        });
    }

    #[test]
    fn test_thunk_generic_destructor_runs_canonical_and_passes_arguments() {
        let f = ThunkFn::<Destructor>::of::<Dealloc>().get();
        // SAFETY: as above.
        check_instantiation(0x40, &(), || unsafe { f(ptr(0x40)) });
    }

    #[test]
    fn test_thunk_generic_getbuffer_runs_canonical_and_passes_arguments() {
        let f = ThunkFn::<GetBuffer>::of::<Buffer>().get();
        let view: *mut ffi::Py_buffer = core::ptr::without_provenance_mut(0x50);
        // SAFETY: as above.
        check_instantiation(0x10 ^ 0x5000 ^ 7, &-7, || unsafe { f(ptr(0x10), view, 7) });
    }

    #[test]
    fn test_thunk_generic_dlpack_deleters_run_canonical_and_pass_arguments() {
        let legacy = ThunkFn::<DlDeleter>::of::<Deleter>().get();
        let versioned = ThunkFn::<DlDeleterVersioned>::of::<DeleterVersioned>().get();
        // SAFETY: as above.
        check_instantiation(0x60, &(), || unsafe {
            legacy(core::ptr::without_provenance_mut(0x60));
        });
        // SAFETY: as above.
        check_instantiation(0x70, &(), || unsafe {
            versioned(core::ptr::without_provenance_mut(0x70));
        });
    }

    #[test]
    fn test_thunk_canonical_caller_passes_through_uncounted() {
        let before = fp::non_canonical_entries();
        assert_eq!(call6(), expected());
        assert!(INSIDE_CANONICAL.with(Cell::get));
        assert!(!INSIDE_NON_CANONICAL_FRAME.with(Cell::get));
        assert!(fp::is_canonical());
        // Other tests may count concurrently; this entry itself does not.
        let _ = before;
    }

    #[test]
    fn test_thunk_flush_to_zero_caller_is_canonicalized_and_restored() {
        check_mode(hostile::flush_to_zero);
    }

    #[test]
    fn test_thunk_toward_zero_caller_is_canonicalized_and_restored() {
        check_mode(hostile::toward_zero);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_thunk_unmasked_exceptions_never_trap_inside() {
        // target6 divides; with divide-by-zero unmasked by the caller the
        // canonical controls inside keep every exception masked.
        check_mode(hostile::unmask_divide_by_zero);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_thunk_non_ieee_nan_and_half_caller_is_canonicalized_and_restored() {
        check_mode(hostile::non_ieee_nan_and_half);
    }

    /// FEAT_AFP's FIZ and AH (bits 0 and 1): where the CPU has them they
    /// stick when written, and the template clears them; where it does not
    /// they read as zero (RES0) and clearing them is a no-op.
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[expect(
        clippy::print_stdout,
        reason = "the test reports what the CPU implements"
    )]
    fn test_thunk_clears_afp_bits_and_traps_where_present() {
        let afp = |c: fp::Control| c | 0b11 | (1 << 8) | (0b11 << 22);
        let (got, after) = hostile::run(afp, call6);
        assert_eq!(got, expected());
        assert!(
            INSIDE_CANONICAL.with(Cell::get),
            "RMode RN, no traps, AH/FIZ clear inside"
        );
        let present = after & 0b11 == 0b11;
        let trap_enable_sticks = after & (1 << 8) != 0;
        std::println!(
            "FEAT_AFP bits present: {present}; trap enables writable: {trap_enable_sticks}"
        );
        assert_eq!(
            after & (0b11 << 22),
            0b11 << 22,
            "the caller's RMode is restored"
        );
    }

    /// The FPCR trap enables (IOE, DZE, OFE, UFE, IXE: bits 8 to 12; IDE:
    /// bit 15) are optional in AArch64: a core without trapped
    /// floating-point exceptions reads them as zero and ignores writes
    /// (RAZ/WI). Where they stick, a call through the boundary clears them
    /// inside and restores exactly the caller's enables; where they read
    /// back as zero, only that restoration assertion is skipped (issue
    /// #229).
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[expect(
        clippy::print_stdout,
        reason = "the test reports what the CPU implements"
    )]
    fn test_thunk_trap_enables_where_writable_are_restored_exactly() {
        const TRAP_ENABLES: fp::Control = (0b1_1111 << 8) | (1 << 15);
        let enable = |c: fp::Control| c | TRAP_ENABLES;
        // Written and read back with no call in between: what the CPU
        // latches.
        let ((), latched) = hostile::run(enable, || ());
        let writable = latched & TRAP_ENABLES;
        let (got, after) = hostile::run(enable, call6);
        assert_eq!(got, expected(), "arguments and result pass through");
        assert!(
            INSIDE_CANONICAL.with(Cell::get),
            "every trap enable is clear inside the call"
        );
        assert_eq!(
            hostile::control_bits(after) & !TRAP_ENABLES,
            hostile::control_bits(latched) & !TRAP_ENABLES,
            "the caller's other control bits are restored"
        );
        std::println!(
            "FPCR trap enables: requested {TRAP_ENABLES:#06x}, read back {writable:#06x}, after the call {:#06x}",
            after & TRAP_ENABLES
        );
        if writable == 0 {
            std::println!(
                "FPCR trap enables read back as zero (RAZ/WI): this CPU does not implement trapped floating-point exceptions; restoration assertion skipped"
            );
            return;
        }
        assert_eq!(
            after & TRAP_ENABLES,
            writable,
            "the caller's trap enables are restored exactly"
        );
    }

    /// The per-call cost of the boundary against a direct call (release,
    /// run with `--ignored --nocapture`).
    #[test]
    #[ignore = "a measurement, not a check"]
    #[expect(clippy::print_stdout, reason = "the measurement is printed")]
    fn bench_thunk_overhead_per_call() {
        const N: u32 = 10_000_000;
        let time = |f: &dyn Fn() -> u64| {
            let start = std::time::Instant::now();
            let mut acc = 0u64;
            for _ in 0..N {
                acc = acc.wrapping_add(f());
            }
            black_box(acc);
            start.elapsed().as_secs_f64() * 1e9 / f64::from(N)
        };
        let direct =
            time(&|| black_box(target6)(ARGS[0], ARGS[1], ARGS[2], ARGS[3], ARGS[4], ARGS[5]));
        let thunked = time(&call6);
        std::println!(
            "direct {direct:.2} ns, through the boundary {thunked:.2} ns, overhead {:.2} ns",
            thunked - direct
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_thunk_keeps_sticky_flags_from_before_and_inside() {
        // Divide-by-zero flag (bit 2) raised by the caller before the call,
        // inexact (bit 5) raised inside it: both survive the return.
        let mode = |c: fp::Control| (hostile::toward_zero(c) & !0x3f) | (1 << 2);
        let ((), after) = hostile::run(mode, || {
            let _ = call6();
        });
        assert!(
            after & (1 << 2) != 0,
            "a flag raised before the call survives"
        );
        assert!(
            after & (1 << 5) != 0,
            "a flag raised inside the call survives"
        );
    }

    #[test]
    fn test_thunk_nesting_restores_each_frame() {
        let (got, after) = hostile::run(hostile::flush_to_zero, || {
            // SAFETY: a thunk of a test target.
            unsafe { THUNK_NESTED(ARGS[0]) }
        });
        assert_eq!(got, target6(ARGS[0], 1, 2, 3, 4, 5));
        // After the inner entry returned, the outer frame is current again.
        let outer = INSIDE_DEPTH_TOKEN.with(Cell::get);
        assert!(
            outer & 1 == 1 && outer & 2 == 2,
            "the outer frame, non-canonical caller"
        );
        assert_eq!(
            hostile::control_bits(after),
            hostile::control_bits(hostile::flush_to_zero(hostile::read())),
        );
        assert_eq!(boundary::frame_token(), 0);
    }

    /// The raw x87 control word (ADR-0044 tests).
    #[cfg(target_arch = "x86_64")]
    fn x87_cw() -> u16 {
        let mut cw: u16 = 0;
        // SAFETY: `fnstcw` stores two bytes to a valid, aligned local and
        // has no other effect.
        unsafe {
            core::arch::asm!("fnstcw [{}]", in(reg) &raw mut cw, options(nostack, preserves_flags));
        }
        cw
    }

    /// Loads `cw` as the x87 control word (ADR-0044 tests). Every caller
    /// keeps all x87 exceptions masked, so no pending exception can fault.
    #[cfg(target_arch = "x86_64")]
    fn set_x87_cw(cw: u16) {
        // SAFETY: `fldcw` reads two bytes from a valid local; no x87
        // exception is pending or unmasked (the tests mask all six).
        unsafe {
            core::arch::asm!("fldcw [{}]", in(reg) &raw const cw, options(nostack, preserves_flags));
        }
    }

    /// An x87 control word unlike the default 0x037f: precision control
    /// double (bits 8-9 = 0b10) and rounding up (bits 10-11 = 0b10), with
    /// every exception still masked.
    #[cfg(target_arch = "x86_64")]
    const X87_HOSTILE: u16 = 0x0a7f;
    /// What a callback or entry body writes: extended precision, toward
    /// zero, all masked.
    #[cfg(target_arch = "x86_64")]
    const X87_CHANGED: u16 = 0x0f7f;

    #[cfg(target_arch = "x86_64")]
    thread_local! {
        static INSIDE_X87: Cell<u16> = const { Cell::new(0) };
        static AFTER_CALLOUT_X87: Cell<u16> = const { Cell::new(0) };
        static INSIDE_X87_NON_CANONICAL: Cell<bool> = const { Cell::new(true) };
    }

    /// An entry body that changes the x87 control word and leaves it.
    #[cfg(target_arch = "x86_64")]
    extern "C" fn x87_changer(a: u64) -> u64 {
        INSIDE_X87.with(|c| c.set(x87_cw()));
        INSIDE_X87_NON_CANONICAL.with(|c| c.set(boundary::frame().is_some_and(|(nc, _)| nc)));
        set_x87_cw(X87_CHANGED);
        a
    }
    #[cfg(target_arch = "x86_64")]
    struct X87Changer;
    #[cfg(target_arch = "x86_64")]
    impl Args1 for X87Changer {
        unsafe extern "C" fn call(a0: u64) -> u64 {
            x87_changer(a0)
        }
    }
    #[cfg(target_arch = "x86_64")]
    const THUNK_X87_CHANGER: unsafe extern "C" fn(u64) -> u64 = args1::<X87Changer>;

    /// A host function (the target of a test callout) that changes the x87
    /// control word and leaves it.
    #[cfg(target_arch = "x86_64")]
    unsafe extern "C" fn host_x87_changer(a: u64) -> u64 {
        set_x87_cw(X87_CHANGED);
        a
    }
    #[cfg(target_arch = "x86_64")]
    callout!(callout_x87_changer => host_x87_changer : fn(u64) -> u64);

    /// An entry body that calls the host twice through the callout.
    #[cfg(target_arch = "x86_64")]
    extern "C" fn calls_host(a: u64) -> u64 {
        // SAFETY: a callout of a test host function with an integer
        // argument and result.
        let r = unsafe { callout_x87_changer(a) };
        AFTER_CALLOUT_X87.with(|c| c.set(x87_cw()));
        // SAFETY: as above.
        unsafe { callout_x87_changer(r) }
    }
    #[cfg(target_arch = "x86_64")]
    struct CallsHost;
    #[cfg(target_arch = "x86_64")]
    impl Args1 for CallsHost {
        unsafe extern "C" fn call(a0: u64) -> u64 {
            calls_host(a0)
        }
    }
    #[cfg(target_arch = "x86_64")]
    const THUNK_CALLS_HOST: unsafe extern "C" fn(u64) -> u64 = args1::<CallsHost>;

    /// An entry body that enters again (a nested entry whose body changes
    /// the control word), as host code inside a callback would.
    #[cfg(target_arch = "x86_64")]
    extern "C" fn enters_again(a: u64) -> u64 {
        // SAFETY: a thunk of a test target.
        let r = unsafe { THUNK_X87_CHANGER(a) };
        AFTER_CALLOUT_X87.with(|c| c.set(x87_cw()));
        r
    }
    #[cfg(target_arch = "x86_64")]
    struct EntersAgain;
    #[cfg(target_arch = "x86_64")]
    impl Args1 for EntersAgain {
        unsafe extern "C" fn call(a0: u64) -> u64 {
            enters_again(a0)
        }
    }
    #[cfg(target_arch = "x86_64")]
    const THUNK_ENTERS_AGAIN: unsafe extern "C" fn(u64) -> u64 = args1::<EntersAgain>;

    /// Runs `f` with the x87 control word set to `cw`, and returns `f`'s
    /// result with the control word read right after it; restores the
    /// previous control word.
    #[cfg(target_arch = "x86_64")]
    fn with_x87<R>(cw: u16, f: impl FnOnce() -> R) -> (R, u16) {
        let saved = x87_cw();
        set_x87_cw(cw);
        let r = f();
        let after = x87_cw();
        set_x87_cw(saved);
        (r, after)
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_thunk_x87_entry_preserves_caller_word_uncounted() {
        // SAFETY: a thunk of a test target.
        let (got, after) = with_x87(X87_HOSTILE, || unsafe { THUNK_X87_CHANGER(7) });
        assert_eq!(got, 7);
        assert_eq!(
            INSIDE_X87.with(Cell::get),
            X87_HOSTILE,
            "not canonicalized inside"
        );
        assert!(
            !INSIDE_X87_NON_CANONICAL.with(Cell::get),
            "an x87-only change is not a non-canonical caller"
        );
        assert_eq!(
            after, X87_HOSTILE,
            "the body's change does not outlive the entry"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_thunk_x87_callout_change_never_outlives_the_callout() {
        // SAFETY: a thunk of a test target.
        let (got, after) = with_x87(X87_HOSTILE, || unsafe { THUNK_CALLS_HOST(9) });
        assert_eq!(got, 9);
        assert_eq!(
            AFTER_CALLOUT_X87.with(Cell::get),
            X87_HOSTILE,
            "Rust resumes under the caller's word after the host changed it"
        );
        assert_eq!(after, X87_HOSTILE);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_thunk_x87_nested_entry_restores_its_immediate_caller() {
        // SAFETY: a thunk of a test target.
        let (got, after) = with_x87(X87_HOSTILE, || unsafe { THUNK_ENTERS_AGAIN(3) });
        assert_eq!(got, 3);
        assert_eq!(
            AFTER_CALLOUT_X87.with(Cell::get),
            X87_HOSTILE,
            "the inner entry restored its caller"
        );
        assert_eq!(after, X87_HOSTILE);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_thunk_x87_preserved_with_hostile_mxcsr() {
        // Both registers hostile at once: each is restored independently.
        let ((got, cw), after) = hostile::run(hostile::toward_zero, || {
            // SAFETY: a thunk of a test target.
            with_x87(X87_HOSTILE, || unsafe { THUNK_CALLS_HOST(5) })
        });
        assert_eq!(got, 5);
        assert_eq!(cw, X87_HOSTILE);
        assert_eq!(
            hostile::control_bits(after),
            hostile::control_bits(hostile::toward_zero(hostile::read())),
        );
    }

    #[test]
    fn test_thunk_contained_panic_returns_and_restores() {
        let (got, after) = hostile::run(hostile::toward_zero, || {
            // SAFETY: a thunk of a test target.
            unsafe { THUNK_PANICKY(1) }
        });
        assert_eq!(got, u64::MAX);
        assert_eq!(
            hostile::control_bits(after),
            hostile::control_bits(hostile::toward_zero(hostile::read())),
        );
        assert_eq!(boundary::frame_token(), 0);
    }
}

// The callout table (ADR-0041 callouts): every C-API function that can run
// Python code, called from Rust only through `capi`. Rule: a GC-tracked
// allocation is a callout (it can run `gc.collect` and any `__del__`), as
// is setting an exception (it normalizes: an allocation), a conversion that
// may raise. (The bindings never own a user object, so no decref needs a
// callout: capi's decref invariant.)
type CStr = *const c_char;
callout!(c_err_set_string => ffi::PyErr_SetString : fn(Obj, CStr) -> ());
callout!(c_err_new_exception => ffi::PyErr_NewExceptionWithDoc : fn(CStr, CStr, Obj, Obj) -> Obj);
callout!(c_err_warn => ffi::PyErr_WarnEx : fn(Obj, CStr, ffi::Py_ssize_t) -> c_int);
callout!(c_module_create => ffi::PyModule_Create2 : fn(*mut ffi::PyModuleDef, c_int) -> Obj);
callout!(c_module_add => ffi::PyModule_AddObjectRef : fn(Obj, CStr, Obj) -> c_int);
callout!(c_type_from_spec => ffi::PyType_FromSpec : fn(*mut ffi::PyType_Spec) -> Obj);
callout!(c_list_new => ffi::PyList_New : fn(ffi::Py_ssize_t) -> Obj);
callout!(c_tuple_new => ffi::PyTuple_New : fn(ffi::Py_ssize_t) -> Obj);
callout!(c_dict_new => ffi::PyDict_New : fn() -> Obj);
callout!(c_list_append => ffi::PyList_Append : fn(Obj, Obj) -> c_int);
callout!(c_tuple_set_item => ffi::PyTuple_SetItem : fn(Obj, ffi::Py_ssize_t, Obj) -> c_int);
callout!(c_dict_set_item_string => ffi::PyDict_SetItemString : fn(Obj, CStr, Obj) -> c_int);
callout!(c_decref => ffi::Py_DecRef : fn(Obj) -> ());
callout!(c_object_call => ffi::PyObject_CallObject : fn(Obj, Obj) -> Obj);
callout!(c_err_clear => ffi::PyErr_Clear : fn() -> ());
callout!(c_err_restore => ffi::PyErr_Restore : fn(Obj, Obj, Obj) -> ());
callout!(c_check_signals => ffi::PyErr_CheckSignals : fn() -> c_int);
callout!(c_long_as_i64 => ffi::PyLong_AsLongLong : fn(Obj) -> i64);
callout!(c_long_as_u64 => ffi::PyLong_AsUnsignedLongLong : fn(Obj) -> u64);
callout!(c_unicode_as_utf8 => ffi::PyUnicode_AsUTF8AndSize : fn(Obj, *mut ffi::Py_ssize_t) -> CStr);
callout!(
    #[cfg(any(feature = "native-module", feature = "host"))]
    c_object_str => ffi::PyObject_Str : fn(Obj) -> Obj
);
callout!(
    #[cfg(feature = "host")]
    c_import_module => ffi::PyImport_ImportModule : fn(CStr) -> Obj
);
callout!(
    #[cfg(feature = "host")]
    c_get_attr_string => ffi::PyObject_GetAttrString : fn(Obj, CStr) -> Obj
);
