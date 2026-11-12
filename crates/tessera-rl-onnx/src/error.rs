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
