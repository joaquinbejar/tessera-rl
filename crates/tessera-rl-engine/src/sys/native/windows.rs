// SPDX-License-Identifier: MIT OR Apache-2.0

//! `WaitOnAddress` waits on Windows. Opt-in only until it has run on real
//! Windows in CI; the default backend there is park.

use std::sync::atomic::AtomicU32;

use windows_sys::Win32::System::Threading::{INFINITE, WaitOnAddress, WakeByAddressAll};

#[expect(unsafe_code, reason = "audited module: WaitOnAddress")]
pub(super) fn wait(word: &AtomicU32, expected: u32) {
    // SAFETY: both pointers are to live, aligned 4-byte values (the atomic
    // and a local copy of `expected`), and the size passed is 4.
    // `WaitOnAddress` only reads them. The result is ignored: callers
    // re-check after any return.
    unsafe {
        WaitOnAddress(
            word.as_ptr().cast(),
            core::ptr::from_ref(&expected).cast(),
            4,
            INFINITE,
        );
    }
}

/// Blocks while `word == expected`, at most `timeout` rounded up to whole
/// milliseconds (so a short wait never becomes a zero-time poll); may
/// return early.
#[expect(unsafe_code, reason = "audited module: WaitOnAddress")]
pub(super) fn wait_timeout(word: &AtomicU32, expected: u32, timeout: core::time::Duration) {
    // Below INFINITE (u32::MAX), which would wait without limit.
    let ms = timeout
        .as_nanos()
        .div_ceil(1_000_000)
        .min(u128::from(INFINITE - 1));
    let ms = u32::try_from(ms).unwrap_or(INFINITE - 1);
    // SAFETY: as in `wait`. A timeout returns FALSE with ERROR_TIMEOUT,
    // which is ignored: callers re-check their deadline.
    unsafe {
        WaitOnAddress(
            word.as_ptr().cast(),
            core::ptr::from_ref(&expected).cast(),
            4,
            ms,
        );
    }
}

#[expect(unsafe_code, reason = "audited module: WakeByAddressAll")]
pub(super) fn wake_all(word: &AtomicU32) {
    // SAFETY: the address is a live, aligned `AtomicU32`; it is only used
    // as a key to wake waiters.
    unsafe { WakeByAddressAll(word.as_ptr().cast()) };
}

/// Processor core records (`RelationProcessorCore`) of
/// `GetLogicalProcessorInformationEx`: physical cores, efficiency cores
/// included, all processor groups; `None` if the query fails. Not limited to
/// the process affinity mask: the caller caps it by `available_parallelism`.
#[expect(
    unsafe_code,
    reason = "audited module: GetLogicalProcessorInformationEx"
)]
pub(super) fn processor_cores() -> Option<core::num::NonZeroUsize> {
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationProcessorCore,
    };
    let mut len: u32 = 0;
    // SAFETY: a null buffer with length 0 only asks for the needed length,
    // which the call writes to `len` (it fails with
    // ERROR_INSUFFICIENT_BUFFER, which the length check below covers).
    unsafe {
        GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            core::ptr::null_mut(),
            &raw mut len,
        );
    }
    let size = usize::try_from(len).ok()?;
    if size == 0 {
        return None;
    }
    // `u64` words: the records hold 8-byte fields, so the buffer is 8-byte
    // aligned, and at least `len` bytes long.
    let mut words = vec![0_u64; size.div_ceil(8)];
    // SAFETY: `words` is a live, writable, 8-byte-aligned buffer of at least
    // `len` bytes; the call writes at most `len` bytes and updates `len` to
    // the bytes written.
    let ok = unsafe {
        GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            words.as_mut_ptr().cast(),
            &raw mut len,
        )
    };
    if ok == 0 {
        return None;
    }
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
    let written = bytes.get(..usize::try_from(len).ok()?)?;
    crate::sys::cores::count_records(written, RelationProcessorCore)
}
