// SPDX-License-Identifier: MIT OR Apache-2.0

//! Errors of the runtime library checks (ADR-0057 section 4).

use std::path::PathBuf;

use thiserror::Error;

/// Why the version of an ONNX Runtime library could not be read.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub(crate) enum SysError {
    /// The library could not be opened.
    #[error("cannot open the ONNX Runtime library {path:?}: {reason}")]
    Open {
        /// The path given.
        path: PathBuf,
        /// The loader's message.
        reason: String,
    },
    /// The library does not export `OrtGetApiBase`.
    #[error("{path:?} does not export OrtGetApiBase: {reason}")]
    Symbol {
        /// The path given.
        path: PathBuf,
        /// The loader's message.
        reason: String,
    },
    /// `OrtGetApiBase` or `GetVersionString` returned a null pointer.
    #[error("the ONNX Runtime library returned a null {what}")]
    Null {
        /// Which pointer was null.
        what: &'static str,
    },
    /// The version string is not of the form `major.minor.patch`.
    #[error("the ONNX Runtime library reports an invalid version: {reason}")]
    Version {
        /// What is wrong with it.
        reason: String,
    },
}

/// Why the ONNX Runtime library was not loaded (ADR-0057 section 4).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum OnnxError {
    /// `ORT_DISABLE_TELEMETRY=1` is not set and telemetry was not
    /// accepted explicitly; nothing was loaded and the environment was not
    /// changed.
    #[error(
        "refusing to load ONNX Runtime: set {variable}=1 before loading it, \
         or accept telemetry explicitly (TelemetryPolicy::Accept)"
    )]
    TelemetryNotDisabled {
        /// The variable that must be `1`.
        variable: &'static str,
    },
    /// No loadable library at the path.
    #[error("no ONNX Runtime library at {path:?}: {reason}")]
    MissingLibrary {
        /// The path given.
        path: PathBuf,
        /// What failed.
        reason: String,
    },
    /// The file is a library but not a usable ONNX Runtime.
    #[error("{path:?} is not a usable ONNX Runtime library: {reason}")]
    NotRuntime {
        /// The path given.
        path: PathBuf,
        /// What failed.
        reason: String,
    },
    /// The library's version is not the pinned one.
    #[error("ONNX Runtime {found} at {path:?}; this build requires exactly {pinned}")]
    Version {
        /// The path given.
        path: PathBuf,
        /// The version the library reports.
        found: String,
        /// The pinned version.
        pinned: &'static str,
    },
    /// `ort` failed to initialize the runtime from the library.
    #[error("initializing ONNX Runtime from {path:?} failed: {reason}")]
    Init {
        /// The path given.
        path: PathBuf,
        /// `ort`'s message.
        reason: String,
    },
    /// An earlier load in this process failed; no further load is tried.
    #[error("an earlier ONNX Runtime load in this process failed: {first}")]
    LoadFailedEarlier {
        /// The first failure.
        first: Box<OnnxError>,
    },
    /// Another library (by content sha256) is already loaded in this
    /// process.
    #[error(
        "ONNX Runtime is already loaded from {loaded:?} (sha256 {loaded_sha256}); \
         refusing {requested:?} (sha256 {requested_sha256})"
    )]
    DifferentLibrary {
        /// The library loaded first.
        loaded: PathBuf,
        /// Its sha256.
        loaded_sha256: String,
        /// The library asked for now.
        requested: PathBuf,
        /// Its sha256.
        requested_sha256: String,
    },
    /// The CPU execution provider could not be selected as the only one.
    #[error("cannot restrict the session to the CPU execution provider: {reason}")]
    Provider {
        /// What failed.
        reason: String,
    },
    /// The model was refused or could not be read or parsed.
    #[error("model {path:?}: {reason}")]
    Model {
        /// The model path.
        path: PathBuf,
        /// What failed.
        reason: String,
    },
    /// The model's inputs or outputs do not match the environment.
    #[error("the model does not match the environment: {what}")]
    Contract {
        /// What does not match.
        what: String,
    },
}
