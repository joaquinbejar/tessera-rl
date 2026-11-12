// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CPU-output test of the session builder (ADR-0057 section 2): the
//! provider is verified by construction (the session is given exactly the
//! CPU device) and by this test, not by a post-commit query.
//!
//! Spike S1's `mlp` model (`tests/fixtures/s1_mlp.onnx`) at the basic level
//! and one thread, on S1's 16-row batch, hashed as S1 hashed it (sha256 of
//! the priors' then the value's `f32` bits, little endian). With the CPU
//! execution provider S1 measured `12ecaf73…` on aarch64 (macOS and Linux)
//! and `a1bfd079…` on x86-64; on macOS the default device policy chose
//! `CoreML` and gave `36d39242…`. Removing the device restriction from
//! `build_session` makes this test fail on macOS.
//!
//! It opens the real ONNX Runtime: ignored by default, run through
//! `make onnx-test` (telemetry off, scratch `HOME`, checked here).

use std::path::{Path, PathBuf};

use ort::value::TensorRef;

use super::{OnnxOptions, build_session};
use crate::loader::load_runtime;
use crate::telemetry::{TELEMETRY_VARIABLE, TelemetryPolicy};

const OBS: usize = 32;
const ROWS: usize = 16;
/// sha256 of `s1_mlp.onnx` (spike S1's generator).
const MLP_SHA256: &str = "7bef92f98f77829bd3087ddaea355df05058b2aadd39b543b339b69f5abe4dfa";
/// The CPU execution provider's result, per architecture (spike S1).
const CPU_AARCH64: &str = "12ecaf73063ee3ed2dd99fc3ac63c4e3229d80101a208806f6a9376303885188";
const CPU_X86_64: &str = "a1bfd0790ae2ccbbe7de48aff86dae7ecb87b5a048ce59e2228ca435a80ab27e";
/// `CoreML`'s result on macOS (spike S1): never accepted.
const COREML_MACOS: &str = "36d39242";

fn input() -> Vec<f32> {
    (0..ROWS * OBS)
        .map(|i| {
            let v = u16::try_from((i * 37) % 101).unwrap_or(0);
            (f32::from(v) - 50.0) / 25.0
        })
        .collect()
}

#[test]
#[ignore = "opens the real ONNX Runtime: run through `make onnx-test`"]
fn test_evaluator_session_outputs_are_the_cpu_providers() {
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

    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/s1_mlp.onnx");
    let bytes = std::fs::read(model).unwrap();
    assert_eq!(crate::hash::sha256(&bytes), MLP_SHA256);
    let runtime = load_runtime(&lib, TelemetryPolicy::RequireDisabled).unwrap();
    let Ok(mut session) = build_session(runtime.environment(), &bytes, &OnnxOptions::new(&lib))
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
    let got = crate::hash::sha256(&raw);
    assert!(!got.starts_with(COREML_MACOS), "CoreML's result: {got}");
    let expected = match std::env::consts::ARCH {
        "aarch64" => CPU_AARCH64,
        "x86_64" => CPU_X86_64,
        other => panic!("no S1 CPU result for {other}"),
    };
    assert_eq!(got, expected, "not the CPU execution provider's result");
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
