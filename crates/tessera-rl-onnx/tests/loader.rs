// SPDX-License-Identifier: MIT OR Apache-2.0

//! The process-wide loader (ADR-0057 section 4), one child process per
//! case: the loader keeps its result for the life of a process.
//!
//! Gate cases never load a runtime; they run by default. The telemetry
//! cases remove `ORT_DISABLE_TELEMETRY` from the child's environment.
//!
//! Real-library cases are ignored by default and run through
//! `make onnx-test` (rule: no test loads ONNX Runtime with telemetry
//! enabled). The parent refuses to run them unless `ORT_DISABLE_TELEMETRY=1`
//! and `HOME` is the scratch directory, passes both to the child, and checks
//! afterwards that no telemetry directory appeared.

#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test scenarios: a failure is the assertion"
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use tessera_rl_onnx::{
    OnnxError, PINNED_VERSION, TELEMETRY_VARIABLE, TelemetryPolicy, load_runtime,
};

const CHILD: &str = "TESSERA_ONNX_LOADER_CASE";
const MISSING: &str = "/nonexistent/tessera-onnx/libonnxruntime.so.1.31.1";

/// Runs the test `name` alone in a child process with `CHILD=name`.
fn run_child(name: &str, remove: &[&str], set: &[(&str, &str)]) {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([name, "--exact", "--include-ignored", "--test-threads=1"])
        .env(CHILD, name);
    for var in remove {
        command.env_remove(var);
    }
    for (var, value) in set {
        command.env(var, value);
    }
    let out = command.output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "child {name} failed:\n{text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        text.contains("1 passed"),
        "child {name} ran no test:\n{text}"
    );
}

fn in_child(name: &str) -> bool {
    std::env::var(CHILD).as_deref() == Ok(name)
}

/// Rule T, checked in the parent before a real-library case.
fn real_library() -> (PathBuf, PathBuf) {
    assert_eq!(
        std::env::var(TELEMETRY_VARIABLE).as_deref(),
        Ok("1"),
        "refusing to load ONNX Runtime with telemetry enabled: set ORT_DISABLE_TELEMETRY=1"
    );
    let home = PathBuf::from(std::env::var_os("HOME").expect("HOME is set"));
    assert!(
        home.ends_with("target/onnx-test-home"),
        "HOME must be the scratch directory `make onnx-test` creates, got {}",
        home.display()
    );
    (lib_from_env(), home)
}

fn lib_from_env() -> PathBuf {
    PathBuf::from(
        std::env::var_os("TESSERA_ONNXRUNTIME_LIB").expect("TESSERA_ONNXRUNTIME_LIB is set"),
    )
}

fn assert_no_telemetry(home: &Path) {
    for dir in [
        home.join(".cache/Microsoft/DeveloperTools/.onnxruntime"),
        home.join("Library/Application Support/Microsoft/DeveloperTools/.onnxruntime"),
    ] {
        assert!(
            !dir.exists(),
            "telemetry directory appeared: {}",
            dir.display()
        );
    }
}

fn scratch_file(tag: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("tessera-onnx-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("libonnxruntime.so.1.31.1");
    std::fs::write(&file, b"not a library").unwrap();
    (dir, file)
}

// Gate and failure cases: no runtime is loaded.

#[test]
fn test_loader_unset_without_opt_out_is_telemetry_refusal() {
    const NAME: &str = "test_loader_unset_without_opt_out_is_telemetry_refusal";
    if in_child(NAME) {
        let before: Vec<_> = std::env::vars_os().collect();
        // A non-library file that exists: if anything were opened before the
        // gate, the error would be MissingLibrary instead.
        let (dir, file) = scratch_file("gate");
        let got = load_runtime(&file, TelemetryPolicy::RequireDisabled);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(
            got.err(),
            Some(OnnxError::TelemetryNotDisabled {
                variable: "ORT_DISABLE_TELEMETRY"
            })
        );
        let after: Vec<_> = std::env::vars_os().collect();
        assert_eq!(before, after, "the environment changed");
        assert!(std::env::var_os(TELEMETRY_VARIABLE).is_none());
        return;
    }
    run_child(NAME, &[TELEMETRY_VARIABLE], &[]);
}

#[test]
fn test_loader_unset_with_opt_out_passes_the_gate_to_missing_library() {
    const NAME: &str = "test_loader_unset_with_opt_out_passes_the_gate_to_missing_library";
    if in_child(NAME) {
        let got = load_runtime(Path::new(MISSING), TelemetryPolicy::Accept);
        assert!(
            matches!(got, Err(OnnxError::MissingLibrary { .. })),
            "{got:?}"
        );
        assert!(std::env::var_os(TELEMETRY_VARIABLE).is_none());
        return;
    }
    run_child(NAME, &[TELEMETRY_VARIABLE], &[]);
}

#[test]
fn test_loader_refusal_message_names_the_variable() {
    let text = OnnxError::TelemetryNotDisabled {
        variable: TELEMETRY_VARIABLE,
    }
    .to_string();
    assert!(text.contains("ORT_DISABLE_TELEMETRY=1"), "{text}");
}

#[test]
fn test_loader_non_library_is_missing_then_every_load_fails_earlier() {
    const NAME: &str = "test_loader_non_library_is_missing_then_every_load_fails_earlier";
    if in_child(NAME) {
        let (dir, file) = scratch_file("fail");
        let first = load_runtime(&file, TelemetryPolicy::RequireDisabled);
        let second = load_runtime(&file, TelemetryPolicy::RequireDisabled);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(
            matches!(first, Err(OnnxError::MissingLibrary { .. })),
            "{first:?}"
        );
        assert!(
            matches!(second, Err(OnnxError::LoadFailedEarlier { .. })),
            "{second:?}"
        );
        return;
    }
    run_child(NAME, &[], &[(TELEMETRY_VARIABLE, "1")]);
}

// Real-library cases (make onnx-test).

#[test]
#[ignore = "opens the real ONNX Runtime: run through `make onnx-test`"]
fn test_loader_real_library_loads_once_at_the_pinned_version() {
    const NAME: &str = "test_loader_real_library_loads_once_at_the_pinned_version";
    if in_child(NAME) {
        let lib = lib_from_env();
        let a = load_runtime(&lib, TelemetryPolicy::RequireDisabled).unwrap();
        assert_eq!(a.version(), PINNED_VERSION);
        assert_eq!(a.path(), lib.canonicalize().unwrap());
        assert!(!a.telemetry_enabled());
        let b = load_runtime(&lib, TelemetryPolicy::RequireDisabled).unwrap();
        assert!(std::ptr::eq(a, b), "a second load returns the same runtime");
        return;
    }
    let (_, home) = real_library();
    run_child(NAME, &[], &[]);
    assert_no_telemetry(&home);
}

#[test]
#[ignore = "opens the real ONNX Runtime: run through `make onnx-test`"]
fn test_loader_ort_dylib_path_is_ignored() {
    const NAME: &str = "test_loader_ort_dylib_path_is_ignored";
    if in_child(NAME) {
        let lib = lib_from_env();
        let rt = load_runtime(&lib, TelemetryPolicy::RequireDisabled).unwrap();
        assert_eq!(rt.path(), lib.canonicalize().unwrap());
        return;
    }
    let (_, home) = real_library();
    run_child(NAME, &[], &[("ORT_DYLIB_PATH", MISSING)]);
    assert_no_telemetry(&home);
}

#[test]
#[ignore = "opens the real ONNX Runtime: run through `make onnx-test`"]
fn test_loader_failed_load_then_real_library_is_failed_earlier() {
    const NAME: &str = "test_loader_failed_load_then_real_library_is_failed_earlier";
    if in_child(NAME) {
        let lib = lib_from_env();
        let (dir, file) = scratch_file("then");
        let first = load_runtime(&file, TelemetryPolicy::RequireDisabled);
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(first.is_err());
        // ORT_DYLIB_PATH names the real library: no `ort` call is made after
        // a failed load, so it is never picked up.
        let second = load_runtime(&lib, TelemetryPolicy::RequireDisabled);
        assert!(
            matches!(second, Err(OnnxError::LoadFailedEarlier { .. })),
            "{second:?}"
        );
        return;
    }
    let (lib, home) = real_library();
    run_child(NAME, &[], &[("ORT_DYLIB_PATH", lib.to_str().unwrap())]);
    assert_no_telemetry(&home);
}

#[test]
#[ignore = "opens the real ONNX Runtime: run through `make onnx-test`"]
fn test_loader_second_library_is_refused() {
    const NAME: &str = "test_loader_second_library_is_refused";
    if in_child(NAME) {
        let lib = lib_from_env();
        let other = PathBuf::from(std::env::var_os("TESSERA_ONNX_COPY").unwrap());
        load_runtime(&lib, TelemetryPolicy::RequireDisabled).unwrap();
        let got = load_runtime(&other, TelemetryPolicy::RequireDisabled);
        assert!(
            matches!(got, Err(OnnxError::DifferentLibrary { .. })),
            "{got:?}"
        );
        return;
    }
    let (lib, home) = real_library();
    // A byte-identical copy under another path: another library by path,
    // refused before it is opened.
    let copy = home.join("copy").join(lib.file_name().unwrap());
    std::fs::create_dir_all(copy.parent().unwrap()).unwrap();
    std::fs::copy(&lib, &copy).unwrap();
    run_child(NAME, &[], &[("TESSERA_ONNX_COPY", copy.to_str().unwrap())]);
    std::fs::remove_dir_all(copy.parent().unwrap()).unwrap();
    assert_no_telemetry(&home);
}
