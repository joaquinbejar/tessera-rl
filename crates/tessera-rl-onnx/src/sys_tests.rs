// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native tests of the audited version read (Miri cannot load a library).
//!
//! `test_sys_real_library_reports_pinned_version` opens the real ONNX
//! Runtime. It is ignored by default and run by `make onnx-test`, which
//! sets `ORT_DISABLE_TELEMETRY=1` and a fresh scratch `HOME` under `target/`. The test itself
//! refuses to run without both, and checks afterwards that no telemetry
//! directory appeared: no test loads the runtime with telemetry enabled.

use std::ffi::c_char;
use std::path::{Path, PathBuf};

use super::{RuntimeLibrary, decode};
use crate::error::SysError;

/// The version pinned in `compat/targets.toml` (`[env.main] onnxruntime`).
const PINNED: &str = "1.31.1";

fn decode_bytes(bytes: &[u8]) -> Result<String, SysError> {
    assert_eq!(bytes.last(), Some(&0), "test strings are NUL-terminated");
    // SAFETY: `bytes` is NUL-terminated and outlives the call.
    #[expect(unsafe_code, reason = "audited module: test of the C string read")]
    unsafe {
        decode(bytes.as_ptr().cast::<c_char>())
    }
}

#[test]
fn test_sys_decode_null_is_typed_error() {
    // SAFETY: null is allowed by the contract.
    #[expect(unsafe_code, reason = "audited module: test of the C string read")]
    let got = unsafe { decode(std::ptr::null()) };
    assert_eq!(
        got,
        Err(SysError::Null {
            what: "version string"
        })
    );
}

#[test]
fn test_sys_decode_full_version_is_exact() {
    assert_eq!(decode_bytes(b"1.31.1\0").as_deref(), Ok(PINNED));
    assert_eq!(decode_bytes(b"10.0.123\0").as_deref(), Ok("10.0.123"));
}

#[test]
fn test_sys_decode_malformed_is_version_error() {
    for bad in [
        &b"1.31\0"[..],
        b"1.31.\0",
        b".31.1\0",
        b"1.31.1.0\0",
        b"1.31.1-rc1\0",
        b"v1.31.1\0",
        b"\0",
        b"\xff.1.1\0",
        b"1.1.111111111111111111111111111111\0",
    ] {
        assert!(
            matches!(decode_bytes(bad), Err(SysError::Version { .. })),
            "{bad:?} must be refused"
        );
    }
}

#[test]
fn test_sys_open_missing_path_is_open_error() {
    let path = Path::new("/nonexistent/tessera-onnx/libonnxruntime.so");
    assert!(matches!(
        RuntimeLibrary::open(path),
        Err(SysError::Open { .. })
    ));
}

#[test]
fn test_sys_open_non_library_is_open_error() {
    let dir = std::env::temp_dir().join(format!("tessera-onnx-sys-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("not-a-library.so");
    std::fs::write(&file, b"not a shared library").unwrap();
    let got = RuntimeLibrary::open(&file);
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(matches!(got, Err(SysError::Open { .. })), "{got:?}");
}

/// A system library that is certainly present and has no `OrtGetApiBase`.
#[cfg(any(target_os = "macos", all(target_os = "linux", target_env = "gnu")))]
fn library_without_entry() -> &'static Path {
    if cfg!(target_os = "macos") {
        Path::new("/usr/lib/libSystem.B.dylib")
    } else {
        Path::new("libc.so.6")
    }
}

#[cfg(any(target_os = "macos", all(target_os = "linux", target_env = "gnu")))]
#[test]
fn test_sys_open_library_without_entry_is_symbol_error() {
    let got = RuntimeLibrary::open(library_without_entry());
    assert!(matches!(got, Err(SysError::Symbol { .. })), "{got:?}");
}

/// Telemetry directories ONNX Runtime 1.31.1 creates under `HOME`
/// (spike S1): Linux, then macOS.
fn telemetry_dirs(home: &Path) -> [PathBuf; 2] {
    [
        home.join(".cache/Microsoft/DeveloperTools/.onnxruntime"),
        home.join("Library/Application Support/Microsoft/DeveloperTools/.onnxruntime"),
    ]
}

#[test]
#[ignore = "opens the real ONNX Runtime: run through `make onnx-test`"]
fn test_sys_real_library_reports_pinned_version() {
    assert_eq!(
        std::env::var("ORT_DISABLE_TELEMETRY").as_deref(),
        Ok("1"),
        "refusing to load ONNX Runtime with telemetry enabled: set ORT_DISABLE_TELEMETRY=1"
    );
    let home = PathBuf::from(std::env::var_os("HOME").expect("HOME is set"));
    assert!(
        home.ends_with("target/onnx-test-home"),
        "HOME must be the scratch directory `make onnx-test` creates, got {home:?}"
    );
    let lib = PathBuf::from(
        std::env::var_os("TESSERA_ONNXRUNTIME_LIB").expect("TESSERA_ONNXRUNTIME_LIB is set"),
    );

    let runtime = RuntimeLibrary::open(&lib).unwrap();
    assert_eq!(runtime.version(), PINNED);

    for dir in telemetry_dirs(&home) {
        assert!(!dir.exists(), "telemetry directory appeared: {dir:?}");
    }
}
