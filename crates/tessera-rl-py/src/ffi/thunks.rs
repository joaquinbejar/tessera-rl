// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every boundary thunk of the crate (ADR-0041), and the only place that
//! can make a [`ThunkFn`].
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

use super::boundary::{
    tessera_fp_caller_controls, tessera_fp_callout_returned, tessera_fp_enter, tessera_fp_leave,
};
use super::dlpack::{self, DLManagedTensor, DLManagedTensorVersioned};
use super::{env_type, module, search_group_type, search_type, snapshot_type, view_type};

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
            enter = sym tessera_fp_enter,
            leave = sym tessera_fp_leave,
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
            enter = sym tessera_fp_enter,
            leave = sym tessera_fp_leave,
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
            enter = sym tessera_fp_enter,
            leave = sym tessera_fp_leave,
            target = sym $target,
            me = sym $me,
        )
    };
}

#[cfg(all(windows, not(target_arch = "x86_64")))]
compile_error!("the floating-point boundary (ADR-0041) supports Windows on x86-64 only");

/// Which boundary this build carries (`tessera_rl._native.fp_boundary()`).
pub(crate) const BOUNDARY: &str = if cfg!(windows) {
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

/// Defines the boundary thunk `$thunk` for `$target` and, with a name, the
/// `ThunkFn` constant tables use.
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
thunk!(SET_FP_POLICY: CFunction = thunk_set_fp_policy => module::set_fp_policy : fn(Obj, Obj) -> Obj);
thunk!(GET_FP_POLICY: CFunction = thunk_get_fp_policy => module::get_fp_policy : fn(Obj, Obj) -> Obj);
thunk!(FP_NON_CANONICAL_ENTRIES: CFunction = thunk_fp_non_canonical_entries => module::fp_non_canonical_entries : fn(Obj, Obj) -> Obj);
#[cfg(feature = "test-hooks")]
thunk!(FAIL_AFTER: CFunction = thunk_fail_after => module::fail_after : fn(Obj, Obj) -> Obj);
thunk!(FP_BOUNDARY: CFunction = thunk_fp_boundary => module::fp_boundary : fn(Obj, Obj) -> Obj);
thunk!(BUILD_PROFILE: CFunction = thunk_build_profile => module::build_profile : fn(Obj, Obj) -> Obj);
thunk!(MAKE_CARTPOLE: CFunctionKw = thunk_make_cartpole => env_type::make_cartpole : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_GRID: CFunctionKw = thunk_make_grid => env_type::make_grid : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_TAG: CFunctionKw = thunk_make_tag => env_type::make_tag : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_CONNECT4: CFunctionKw = thunk_make_connect4 => env_type::make_connect4 : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_LEDUC: CFunctionKw = thunk_make_leduc => env_type::make_leduc : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_OTHELLO: CFunctionKw = thunk_make_othello => env_type::make_othello : fn(Obj, Obj, Obj) -> Obj);
thunk!(MAKE_GO9: CFunctionKw = thunk_make_go9 => env_type::make_go9 : fn(Obj, Obj, Obj) -> Obj);
thunk!(LAYOUT: CFunction = thunk_layout => env_type::layout : fn(Obj, Obj) -> Obj);
thunk!(OUTPUTS: CFunction = thunk_outputs => env_type::outputs : fn(Obj, Obj) -> Obj);
thunk!(IS_POISONED: CFunction = thunk_is_poisoned => env_type::is_poisoned : fn(Obj, Obj) -> Obj);
thunk!(CLOSE: CFunction = thunk_close => env_type::close : fn(Obj, Obj) -> Obj);
thunk!(RESET: CFunction = thunk_reset => env_type::reset : fn(Obj, Obj) -> Obj);
thunk!(STEP: CFunction = thunk_step => env_type::step : fn(Obj, Obj) -> Obj);
thunk!(SNAPSHOT: CFunction = thunk_snapshot => env_type::snapshot : fn(Obj, Obj) -> Obj);
thunk!(IMPORT_RECORD: CFunction = thunk_import_record => env_type::import_record : fn(Obj, Obj) -> Obj);
thunk!(FORK: CFunction = thunk_fork => env_type::fork : fn(Obj, Obj) -> Obj);
thunk!(SEND: CFunction = thunk_send => env_type::send : fn(Obj, Obj) -> Obj);
thunk!(RECV: CFunction = thunk_recv => env_type::recv : fn(Obj, Obj) -> Obj);
thunk!(RESTORE: CFunction = thunk_restore => env_type::restore : fn(Obj, Obj) -> Obj);
thunk!(SNAPSHOT_ALL: CFunction = thunk_snapshot_all => env_type::snapshot_all : fn(Obj, Obj) -> Obj);
thunk!(RESTORE_ALL: CFunction = thunk_restore_all => env_type::restore_all : fn(Obj, Obj) -> Obj);
thunk!(SNAPSHOT_DEALLOC: Destructor = thunk_snapshot_dealloc => snapshot_type::dealloc : fn(Obj) -> ());
thunk!(DEALLOC: Destructor = thunk_dealloc => env_type::dealloc : fn(Obj) -> ());
thunk!(MAKE_SEARCH: CFunctionKw = thunk_make_search => search_type::make_search : fn(Obj, Obj, Obj) -> Obj);
thunk!(SEARCH_LAYOUT: CFunction = thunk_search_layout => search_type::layout : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_SET_ROOT: CFunction = thunk_search_set_root => search_type::set_root : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_RUN: CFunction = thunk_search_run => search_type::run : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_ROOT_VISITS: CFunction = thunk_search_root_visits => search_type::root_visits : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_SELECT_ACTION: CFunction = thunk_search_select_action => search_type::select_action : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_CALLS: CFunction = thunk_search_calls => search_type::calls : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_TREE_HASH: CFunction = thunk_search_tree_hash => search_type::tree_hash : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_IS_PRIVILEGED: CFunction = thunk_search_is_privileged => search_type::is_privileged : fn(Obj, Obj) -> Obj);
thunk!(SEARCH_DEALLOC: Destructor = thunk_search_dealloc => search_type::dealloc : fn(Obj) -> ());
thunk!(MAKE_SEARCH_GROUP: CFunctionKw = thunk_make_search_group => search_group_type::make_search_group : fn(Obj, Obj, Obj) -> Obj);
thunk!(GROUP_LAYOUT: CFunction = thunk_group_layout => search_group_type::layout : fn(Obj, Obj) -> Obj);
thunk!(GROUP_LAST_FAILURE: CFunction = thunk_group_last_failure => search_group_type::last_failure : fn(Obj, Obj) -> Obj);
thunk!(GROUP_SET_ROOT: CFunction = thunk_group_set_root => search_group_type::set_root : fn(Obj, Obj) -> Obj);
thunk!(GROUP_SET_ACTIVE: CFunction = thunk_group_set_active => search_group_type::set_active : fn(Obj, Obj) -> Obj);
thunk!(GROUP_ACTIVE: CFunction = thunk_group_active => search_group_type::active : fn(Obj, Obj) -> Obj);
thunk!(GROUP_RUN: CFunction = thunk_group_run => search_group_type::run : fn(Obj, Obj) -> Obj);
thunk!(GROUP_ROOT_VISITS: CFunction = thunk_group_root_visits => search_group_type::root_visits : fn(Obj, Obj) -> Obj);
thunk!(GROUP_SELECT_ACTION: CFunction = thunk_group_select_action => search_group_type::select_action : fn(Obj, Obj) -> Obj);
thunk!(GROUP_CALLS: CFunction = thunk_group_calls => search_group_type::calls : fn(Obj, Obj) -> Obj);
thunk!(GROUP_TREE_HASH: CFunction = thunk_group_tree_hash => search_group_type::tree_hash : fn(Obj, Obj) -> Obj);
thunk!(GROUP_DEALLOC: Destructor = thunk_group_dealloc => search_group_type::dealloc : fn(Obj) -> ());

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

    // Per thread: the tests run in parallel threads.
    thread_local! {
        static INSIDE_CANONICAL: Cell<bool> = const { Cell::new(false) };
        static INSIDE_NON_CANONICAL_FRAME: Cell<bool> = const { Cell::new(false) };
        static INSIDE_DEPTH_TOKEN: Cell<u64> = const { Cell::new(0) };
    }

    extern "C" fn target6(a0: u64, a1: u64, a2: u64, a3: u64, a4: u64, a5: u64) -> u64 {
        target4(a0, a1, a2, a3) ^ a4.rotate_left(32) ^ a5.rotate_left(40)
    }

    /// The first [`MAX_ARGS`]-compatible part of `target6`: the four
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
    thunk!(@naked thunk6 => target6 : fn(u64, u64, u64, u64, u64, u64) -> u64);

    // Win64 preserves four (MAX_ARGS): the thunk takes the register
    // arguments, and the test helper folds in the last two outside it, so
    // every test checks the same value on every ABI.
    #[cfg(windows)]
    thunk!(@naked thunk4 => target4 : fn(u64, u64, u64, u64) -> u64);

    #[cfg(windows)]
    unsafe extern "C" fn thunk6(a0: u64, a1: u64, a2: u64, a3: u64, a4: u64, a5: u64) -> u64 {
        // SAFETY: a thunk of a test target.
        unsafe { thunk4(a0, a1, a2, a3) ^ a4.rotate_left(32) ^ a5.rotate_left(40) }
    }

    extern "C" fn nested(a: u64) -> u64 {
        // SAFETY: a thunk of a test target.
        let inner = unsafe { thunk6(a, 1, 2, 3, 4, 5) };
        INSIDE_DEPTH_TOKEN.with(|c| c.set(boundary::frame_token()));
        inner
    }
    thunk!(@naked thunk_nested => nested : fn(u64) -> u64);

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
    thunk!(@naked thunk_panicky => panicky : fn(u64) -> u64);

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
        unsafe { thunk6(ARGS[0], ARGS[1], ARGS[2], ARGS[3], ARGS[4], ARGS[5]) }
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
            unsafe { thunk_nested(ARGS[0]) }
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
    thunk!(@naked thunk_x87_changer => x87_changer : fn(u64) -> u64);

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
    thunk!(@naked thunk_calls_host => calls_host : fn(u64) -> u64);

    /// An entry body that enters again (a nested entry whose body changes
    /// the control word), as host code inside a callback would.
    #[cfg(target_arch = "x86_64")]
    extern "C" fn enters_again(a: u64) -> u64 {
        // SAFETY: a thunk of a test target.
        let r = unsafe { thunk_x87_changer(a) };
        AFTER_CALLOUT_X87.with(|c| c.set(x87_cw()));
        r
    }
    #[cfg(target_arch = "x86_64")]
    thunk!(@naked thunk_enters_again => enters_again : fn(u64) -> u64);

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
        let (got, after) = with_x87(X87_HOSTILE, || unsafe { thunk_x87_changer(7) });
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
        let (got, after) = with_x87(X87_HOSTILE, || unsafe { thunk_calls_host(9) });
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
        let (got, after) = with_x87(X87_HOSTILE, || unsafe { thunk_enters_again(3) });
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
            with_x87(X87_HOSTILE, || unsafe { thunk_calls_host(5) })
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
            unsafe { thunk_panicky(1) }
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
callout!(c_check_signals => ffi::PyErr_CheckSignals : fn() -> c_int);
callout!(c_long_as_i64 => ffi::PyLong_AsLongLong : fn(Obj) -> i64);
callout!(c_long_as_u64 => ffi::PyLong_AsUnsignedLongLong : fn(Obj) -> u64);
callout!(c_unicode_as_utf8 => ffi::PyUnicode_AsUTF8AndSize : fn(Obj, *mut ffi::Py_ssize_t) -> CStr);
