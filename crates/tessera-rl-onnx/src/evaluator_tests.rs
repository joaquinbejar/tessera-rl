// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests of the session builder (ADR-0057 section 2).
//!
//! The provider is verified by construction (the session is given exactly
//! the CPU device, [`single_cpu`], unit-tested here) and by the CPU-output
//! test, not by a post-commit query.
//!
//! The real-library tests run spike S1's models at one thread on S1's
//! 16-row batch, hashed as S1 hashed them (sha256 of the priors' then the
//! value's `f32` bits, little endian), against S1's recorded results:
//! - `mlp` gives the same bytes at every level, so it tests the provider:
//!   the CPU result is `12ecaf73…` on aarch64 (macOS and Linux) and
//!   `a1bfd079…` on x86-64; on macOS the default device policy chose
//!   `CoreML` and gave `36d39242…`;
//! - `fuse` differs between levels, so it pins the optimization level:
//!   basic is `699164d0…` on aarch64 and `94bfe861…` on x86-64, all is
//!   `8313fbd2…` and `2b7ee970…`.
//!
//! They open the real ONNX Runtime: ignored by default, run through
//! `make onnx-test` (telemetry off, scratch `HOME`, checked here).

use std::path::{Path, PathBuf};

use ort::value::TensorRef;

use super::{CPU_PROVIDER, OnnxOptions, build_session, single_cpu};
use crate::loader::load_runtime;
use crate::telemetry::{TELEMETRY_VARIABLE, TelemetryPolicy};

const OBS: usize = 32;
const ROWS: usize = 16;

/// A fixture and its sha256 (spike S1's generator).
const MLP: (&str, &str) = (
    "s1_mlp.onnx",
    "7bef92f98f77829bd3087ddaea355df05058b2aadd39b543b339b69f5abe4dfa",
);
const FUSE: (&str, &str) = (
    "s1_fuse.onnx",
    "ed7c4c1dcd099f8a8905eafb05dc5297f9811058d99555f2e58c3ab0ef875140",
);

/// S1's results per architecture: `(aarch64, x86-64)`.
const MLP_CPU: (&str, &str) = (
    "12ecaf73063ee3ed2dd99fc3ac63c4e3229d80101a208806f6a9376303885188",
    "a1bfd0790ae2ccbbe7de48aff86dae7ecb87b5a048ce59e2228ca435a80ab27e",
);
const FUSE_BASIC: (&str, &str) = (
    "699164d0c4cb8fffcf4c03d300f42f16a97ade579ab67f95379053d4d4f4881d",
    "94bfe861a4d11e71e53940d1477589a7f896c045242e7e857a8643e856a2d54a",
);
const FUSE_ALL: (&str, &str) = (
    "8313fbd233cd54d248a5d7471a772864189cecf16103632642a6a0d2f0af6451",
    "2b7ee970e17c3af025b1c890b01a8dcfba799e5b4fc356dbf02109036699cdcc",
);
/// `CoreML`'s `mlp` result on macOS (spike S1): never accepted.
const COREML_MACOS: &str = "36d39242";

fn for_arch(pair: (&'static str, &'static str)) -> &'static str {
    match std::env::consts::ARCH {
        "aarch64" => pair.0,
        "x86_64" => pair.1,
        other => panic!("no S1 result for {other}"),
    }
}

fn input() -> Vec<f32> {
    (0..ROWS * OBS)
        .map(|i| {
            let v = u16::try_from((i * 37) % 101).unwrap_or(0);
            (f32::from(v) - 50.0) / 25.0
        })
        .collect()
}

/// Rule T, then the library path; returns `(library, home)`.
fn rule_t() -> (PathBuf, PathBuf) {
    assert_eq!(
        std::env::var(TELEMETRY_VARIABLE).as_deref(),
        Ok("1"),
        "refusing to load ONNX Runtime with telemetry enabled: set ORT_DISABLE_TELEMETRY=1"
    );
    let home = PathBuf::from(std::env::var_os("HOME").expect("HOME is set"));
    assert!(
        home.ends_with("target/onnx-test-home"),
        "HOME must be the scratch directory"
    );
    let lib = PathBuf::from(std::env::var_os("TESSERA_ONNXRUNTIME_LIB").expect("library"));
    (lib, home)
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

/// S1's hash of `fixture` through a production session.
fn s1_hash(lib: &Path, fixture: (&str, &str)) -> String {
    let model = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture.0);
    let bytes = std::fs::read(model).unwrap();
    assert_eq!(crate::hash::sha256(&bytes), fixture.1, "{}", fixture.0);
    let runtime = load_runtime(lib, TelemetryPolicy::RequireDisabled).unwrap();
    let Ok(mut session) = build_session(runtime.environment(), &bytes, &OnnxOptions::new(lib))
    else {
        panic!("session");
    };
    let x = input();
    let t = TensorRef::from_array_view(([ROWS, OBS], x.as_slice())).unwrap();
    let out = session.run(ort::inputs!["obs" => t]).unwrap();
    let mut raw = Vec::new();
    for name in ["priors", "value"] {
        let (_, data) = out[name].try_extract_tensor::<f32>().unwrap();
        for f in data {
            raw.extend_from_slice(&f.to_bits().to_le_bytes());
        }
    }
    crate::hash::sha256(&raw)
}

#[test]
#[ignore = "opens the real ONNX Runtime: run through `make onnx-test`"]
fn test_evaluator_session_outputs_are_the_cpu_providers() {
    let (lib, home) = rule_t();
    let got = s1_hash(&lib, MLP);
    assert!(!got.starts_with(COREML_MACOS), "CoreML's result: {got}");
    assert_eq!(
        got,
        for_arch(MLP_CPU),
        "not the CPU execution provider's result"
    );
    assert_no_telemetry(&home);
}

#[test]
#[ignore = "opens the real ONNX Runtime: run through `make onnx-test`"]
fn test_evaluator_session_runs_at_the_basic_level() {
    let (lib, home) = rule_t();
    let got = s1_hash(&lib, FUSE);
    assert_ne!(got, for_arch(FUSE_ALL), "the `all` level's result");
    assert_eq!(got, for_arch(FUSE_BASIC), "not the basic level's result");
    assert_no_telemetry(&home);
}

fn named(eps: &[Option<&str>]) -> Vec<(usize, Option<String>)> {
    eps.iter()
        .enumerate()
        .map(|(i, e)| (i, e.map(str::to_owned)))
        .collect()
}

#[test]
fn test_evaluator_single_cpu_one_cpu_device_is_chosen() {
    assert_eq!(single_cpu(named(&[Some(CPU_PROVIDER)])), Ok(0));
    let mixed = named(&[Some("CoreMLExecutionProvider"), Some(CPU_PROVIDER), None]);
    assert_eq!(single_cpu(mixed), Ok(1));
}

#[test]
fn test_evaluator_single_cpu_no_cpu_device_is_refused() {
    assert!(single_cpu(named(&[])).is_err());
    assert!(single_cpu(named(&[Some("CoreMLExecutionProvider"), None])).is_err());
}

#[test]
fn test_evaluator_single_cpu_two_cpu_devices_are_refused() {
    assert!(single_cpu(named(&[Some(CPU_PROVIDER), Some(CPU_PROVIDER)])).is_err());
}
