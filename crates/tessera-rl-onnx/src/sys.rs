// SPDX-License-Identifier: MIT OR Apache-2.0

//! The exact version of an ONNX Runtime shared library (ADR-0057 section
//! 4). Audited `unsafe` module.
//!
//! `ort` accepts any library at or above its minimum and exposes no version
//! getter, so the version is read here from the library itself: open it,
//! look up `OrtGetApiBase`, call it, and call the `GetVersionString` entry
//! of the returned `OrtApiBase`. This is the only content of the module and
//! the only direct use of `libloading` in the workspace.
//!
//! Opening a library runs its initializers. The caller must therefore have
//! made every check that has to come before any load (the telemetry rule of
//! ADR-0057 section 4) before it calls [`RuntimeLibrary::open`].
//!
//! The handle is kept in the returned [`RuntimeLibrary`]. The loader keeps
//! that value for the life of the process and never drops it: unloading a
//! library that `ort` has initialized is not sound.
//!
//! Miri cannot load a C library, so these blocks are covered by native
//! tests (`sys_tests.rs`) on every platform the crate is tested on.

use std::ffi::{CStr, c_char};
use std::path::{Path, PathBuf};

use libloading::{Library, Symbol};

use crate::error::SysError;

/// The C name of the entry point the version is read through.
const ENTRY: &[u8] = b"OrtGetApiBase\0";

/// The longest version string accepted, in bytes.
const MAX_VERSION_LEN: usize = 32;

/// `OrtApiBase` of `onnxruntime_c_api.h`: two function pointers, in this
/// order, stable across every ONNX Runtime release.
#[repr(C)]
struct OrtApiBase {
    /// `const OrtApi* GetApi(uint32_t version)`; not called here.
    _get_api: unsafe extern "C" fn(u32) -> *const core::ffi::c_void,
    /// `const char* GetVersionString(void)`.
    get_version_string: unsafe extern "C" fn() -> *const c_char,
}

/// `const OrtApiBase* OrtGetApiBase(void)`.
type GetApiBase = unsafe extern "C" fn() -> *const OrtApiBase;

/// An opened ONNX Runtime library and the version it reports.
#[derive(Debug)]
pub(crate) struct RuntimeLibrary {
    /// Held, never read: keeps the library loaded for as long as this
    /// value lives.
    _library: Library,
    /// The path it was opened from.
    path: PathBuf,
    /// The version it reports, for example `1.31.1`.
    version: String,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "used by the loader (ADR-0057 section 4), added next"
    )
)]
impl RuntimeLibrary {
    /// Opens the library at `path` and reads its version.
    ///
    /// # Errors
    ///
    /// [`SysError::Open`] if it cannot be opened, [`SysError::Symbol`] if
    /// it does not export `OrtGetApiBase`, [`SysError::Null`] if a pointer
    /// it returns is null, [`SysError::Version`] if the string is not a
    /// `major.minor.patch` version.
    pub(crate) fn open(path: &Path) -> Result<Self, SysError> {
        #[expect(unsafe_code, reason = "audited module: loading a C library")]
        // SAFETY: loading runs the library's initializers. The caller has
        // made the checks that must precede any load (module docs); the
        // path is the one the user named explicitly (ADR-0026).
        let library = unsafe { Library::new(path) }.map_err(|e| SysError::Open {
            path: path.to_path_buf(),
            reason: describe(&e),
        })?;
        let version = {
            #[expect(unsafe_code, reason = "audited module: symbol lookup")]
            // SAFETY: `GetApiBase` is the C signature of `OrtGetApiBase` in
            // `onnxruntime_c_api.h`, unchanged since its first release.
            let entry: Symbol<'_, GetApiBase> =
                unsafe { library.get(ENTRY) }.map_err(|e| SysError::Symbol {
                    path: path.to_path_buf(),
                    reason: describe(&e),
                })?;
            #[expect(unsafe_code, reason = "audited module: C call")]
            // SAFETY: `OrtGetApiBase` takes no argument and returns a
            // pointer to a static table, or null; the library is open.
            let base = unsafe { entry() };
            #[expect(unsafe_code, reason = "audited module: pointer read")]
            // SAFETY: a non-null `base` points to the library's static
            // `OrtApiBase`, valid while the library is open.
            let base = unsafe { base.as_ref() }.ok_or(SysError::Null { what: "OrtApiBase" })?;
            #[expect(unsafe_code, reason = "audited module: C call")]
            // SAFETY: `GetVersionString` takes no argument and returns a
            // pointer to a static NUL-terminated string, or null.
            let raw = unsafe { (base.get_version_string)() };
            #[expect(unsafe_code, reason = "audited module: C string read")]
            // SAFETY: `raw` comes from `GetVersionString` of the open
            // library: null or a static NUL-terminated string.
            unsafe { decode(raw) }?
        };
        Ok(Self {
            _library: library,
            path: path.to_path_buf(),
            version,
        })
    }

    /// The version the library reports, for example `1.31.1`.
    #[must_use]
    pub(crate) fn version(&self) -> &str {
        &self.version
    }

    /// The path the library was opened from.
    #[must_use]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

/// The loader's message with its whole source chain (`libloading` puts
/// the system's text in the source, not in the top-level message).
fn describe(error: &libloading::Error) -> String {
    let mut text = error.to_string();
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}

/// Decodes the version string at `raw` and checks its form.
///
/// # Safety
///
/// `raw` is null or points to a NUL-terminated string that stays valid for
/// the duration of the call.
#[expect(unsafe_code, reason = "audited module: C string read")]
pub(crate) unsafe fn decode(raw: *const c_char) -> Result<String, SysError> {
    if raw.is_null() {
        return Err(SysError::Null {
            what: "version string",
        });
    }
    // SAFETY: non-null, and NUL-terminated and valid by the contract.
    let text = unsafe { CStr::from_ptr(raw) };
    let text = text.to_str().map_err(|_| SysError::Version {
        reason: "not UTF-8".to_owned(),
    })?;
    check(text)?;
    Ok(text.to_owned())
}

/// Checks that `text` is `major.minor.patch`, three non-empty runs of
/// ASCII digits, at most [`MAX_VERSION_LEN`] bytes.
fn check(text: &str) -> Result<(), SysError> {
    let bad = |reason: &str| SysError::Version {
        reason: format!("{reason}: {text:?}"),
    };
    if text.len() > MAX_VERSION_LEN {
        return Err(bad("too long"));
    }
    let parts: Vec<&str> = text.split('.').collect();
    if parts.len() != 3 {
        return Err(bad("not three dot-separated parts"));
    }
    if parts
        .iter()
        .any(|p| p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit()))
    {
        return Err(bad("a part is not a decimal number"));
    }
    Ok(())
}

#[cfg(test)]
#[path = "sys_tests.rs"]
mod tests;
