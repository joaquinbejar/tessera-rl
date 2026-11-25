// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tool's errors (#351).

use std::path::PathBuf;

use crate::args::ArgsError;
use crate::names::NameError;
use crate::render::RenderError;

/// Why a `cargo tessera` command failed.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// The command line.
    #[error(transparent)]
    Args(#[from] ArgsError),
    /// A crate name, package name or id.
    #[error(transparent)]
    Name(#[from] NameError),
    /// An embedded template (a bug in the tool).
    #[error(transparent)]
    Render(#[from] RenderError),
    /// `new` never overwrites.
    #[error("{} already exists: cargo tessera new never overwrites", path.display())]
    Exists {
        /// The path.
        path: PathBuf,
    },
    /// A file system operation failed.
    #[error("{action} {}: {source}", path.display())]
    Io {
        /// What was being done.
        action: &'static str,
        /// The path.
        path: PathBuf,
        /// The error.
        source: std::io::Error,
    },
    /// `--tessera-path` is not usable.
    #[error("--tessera-path {}: {reason}", path.display())]
    TesseraPath {
        /// The path.
        path: PathBuf,
        /// Why.
        reason: &'static str,
    },
    /// A program the command needs is not on the PATH.
    #[error("`{program}` is not on the PATH: {hint}")]
    MissingProgram {
        /// The program.
        program: String,
        /// How to get it.
        hint: &'static str,
    },
    /// A program could not be started.
    #[error("could not run `{program}`: {source}")]
    Spawn {
        /// The program.
        program: String,
        /// The error.
        source: std::io::Error,
    },
    /// A command exited with failure.
    #[error("`{command}` failed ({status})")]
    Failed {
        /// The command line.
        command: String,
        /// Its exit status.
        status: String,
    },
    /// `check` or `bench` outside a scaffolded crate.
    #[error("{} is not a crate scaffolded by cargo tessera new: no {missing}", dir.display())]
    NotACrate {
        /// The directory.
        dir: PathBuf,
        /// The missing file.
        missing: &'static str,
    },
    /// `check --python` in a `--kind env` crate (#628).
    #[error(
        "{} is a Rust-only Environment crate (--kind env): it has no Python package. \
         Python access needs a host environment: cargo tessera new --kind host (ADR-0062)",
        dir.display()
    )]
    NoPython {
        /// The crate's directory.
        dir: PathBuf,
    },
    /// The bench left no estimates to summarize.
    #[error("no criterion estimates under {}: did the bench run?", dir.display())]
    NoEstimates {
        /// The criterion directory.
        dir: PathBuf,
    },
}
