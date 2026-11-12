// SPDX-License-Identifier: MIT OR Apache-2.0

//! What an ONNX evaluator's results depend on (ADR-0057 section 3).

use std::path::PathBuf;

use crate::evaluator::{CPU_PROVIDER, OPTIMIZATION_LEVEL, OnnxOptions};
use crate::loader::Runtime;

/// The provenance of an [`OnnxEvaluator`](crate::OnnxEvaluator)'s results.
///
/// Repeatability (ADR-0057 section 1) holds only between results with the
/// same model sha256, library sha256, provider, threads, optimization level
/// and CPU. The library is identified by its sha256, not its version: on
/// macOS the same version ships different library bytes in different
/// wheels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// sha256 of the model bytes, lowercase hex.
    pub model_sha256: String,
    /// The library's canonical path (provenance, not a class condition).
    pub library_path: PathBuf,
    /// sha256 of the library file, lowercase hex.
    pub library_sha256: String,
    /// The version the library reports.
    pub library_version: String,
    /// The execution provider (always the CPU one).
    pub provider: &'static str,
    /// Intra-op threads.
    pub intra_threads: usize,
    /// Inter-op threads.
    pub inter_threads: usize,
    /// The graph optimization level.
    pub optimization_level: &'static str,
    /// Whether the runtime was loaded with telemetry possibly enabled
    /// (through `TelemetryPolicy::Accept` without `ORT_DISABLE_TELEMETRY=1`).
    pub telemetry_enabled: bool,
    /// The CPU architecture (`std::env::consts::ARCH`).
    pub arch: &'static str,
    /// The CPU model as the operating system reports it, where it is read
    /// without running a program (Linux `/proc/cpuinfo`); `None` elsewhere.
    pub cpu_model: Option<String>,
}

impl Provenance {
    pub(crate) fn new(runtime: &Runtime, model_sha256: String, options: &OnnxOptions) -> Self {
        Self {
            model_sha256,
            library_path: runtime.path().to_path_buf(),
            library_sha256: runtime.sha256().to_owned(),
            library_version: runtime.version().to_owned(),
            provider: CPU_PROVIDER,
            intra_threads: options.intra_threads,
            inter_threads: options.inter_threads,
            optimization_level: OPTIMIZATION_LEVEL,
            telemetry_enabled: runtime.telemetry_enabled(),
            arch: std::env::consts::ARCH,
            cpu_model: cpu_model(),
        }
    }
}

fn cpu_model() -> Option<String> {
    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    text.lines()
        .find_map(|l| {
            let (key, value) = l.split_once(':')?;
            matches!(key.trim(), "model name" | "Model" | "CPU part")
                .then(|| value.trim().to_owned())
        })
        .filter(|s| !s.is_empty())
}
