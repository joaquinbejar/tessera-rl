// SPDX-License-Identifier: MIT OR Apache-2.0

//! The process-wide ONNX Runtime loader (ADR-0057 section 4).
//!
//! Order, on the first call in a process:
//! 1. the telemetry gate ([`TelemetryPolicy`]), before anything is opened;
//! 2. the explicit path, made canonical (`ORT_DYLIB_PATH` is never read);
//! 3. the exact version, read from the library itself (`sys`);
//! 4. `ort::init_from` with the same path, then one `ort` environment.
//!
//! The result, success or failure, is kept for the life of the process in
//! a `OnceLock`: after a failure no further `ort` call is made and every
//! later call returns [`OnnxError::LoadFailedEarlier`]; after a success a
//! call naming another library is refused. The handle the version read
//! opened is kept in [`Runtime`] and never closed: unloading a library that
//! `ort` has initialized is not sound. `ort` opens the same path again,
//! which the dynamic loader resolves to the image already loaded.
//!
//! Libraries are identified by their canonical path. Identity by content
//! (the library's sha256, which ADR-0057 records as provenance) comes with
//! the provenance record.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::error::{OnnxError, SysError};
use crate::sys::RuntimeLibrary;
use crate::telemetry::{self, TelemetryPolicy};

/// The ONNX Runtime version this build accepts, exactly. It equals
/// `onnxruntime` in `compat/targets.toml` (a test checks it).
pub const PINNED_VERSION: &str = "1.31.1";

/// The runtime loaded in this process.
#[derive(Debug)]
pub struct Runtime {
    /// Keeps the library loaded; never dropped (it lives in a static).
    library: RuntimeLibrary,
    /// The canonical path it was loaded from.
    canonical: PathBuf,
    /// Whether it was loaded with telemetry possibly enabled.
    telemetry_enabled: bool,
    /// The one `ort` environment of the process.
    _environment: ort::environment::Environment,
}

impl Runtime {
    /// The version the library reports (equal to [`PINNED_VERSION`]).
    #[must_use]
    pub fn version(&self) -> &str {
        self.library.version()
    }

    /// The canonical path the library was loaded from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.canonical
    }

    /// Whether the runtime was loaded without `ORT_DISABLE_TELEMETRY=1`,
    /// through [`TelemetryPolicy::Accept`].
    #[must_use]
    pub fn telemetry_enabled(&self) -> bool {
        self.telemetry_enabled
    }
}

static RUNTIME: OnceLock<Result<Runtime, OnnxError>> = OnceLock::new();

/// Loads ONNX Runtime from `path` once per process and returns it.
///
/// # Errors
///
/// [`OnnxError::TelemetryNotDisabled`] before anything else (nothing is
/// opened); [`OnnxError::MissingLibrary`] if `path` does not resolve or
/// does not open; [`OnnxError::NotRuntime`] if it is not an ONNX Runtime;
/// [`OnnxError::Version`] if its version is not [`PINNED_VERSION`];
/// [`OnnxError::Init`] if `ort` fails; [`OnnxError::LoadFailedEarlier`]
/// after any failed load in this process; [`OnnxError::DifferentLibrary`]
/// if another library is already loaded.
pub fn load_runtime(path: &Path, policy: TelemetryPolicy) -> Result<&'static Runtime, OnnxError> {
    let telemetry_enabled = telemetry::gate(policy)?;
    let requested = path.canonicalize().map_err(|e| OnnxError::MissingLibrary {
        path: path.to_path_buf(),
        reason: e.to_string(),
    })?;
    let mut fresh = false;
    let state = RUNTIME.get_or_init(|| {
        fresh = true;
        init(&requested, telemetry_enabled)
    });
    match state {
        Ok(runtime) if runtime.canonical == requested => Ok(runtime),
        Ok(runtime) => Err(OnnxError::DifferentLibrary {
            loaded: runtime.canonical.clone(),
            requested,
        }),
        Err(first) if fresh => Err(first.clone()),
        Err(first) => Err(OnnxError::LoadFailedEarlier {
            first: Box::new(first.clone()),
        }),
    }
}

/// The first load of the process.
fn init(path: &Path, telemetry_enabled: bool) -> Result<Runtime, OnnxError> {
    let library = RuntimeLibrary::open(path).map_err(|e| from_sys(path, e))?;
    check_version(path, library.version())?;
    let init_err = |reason: String| OnnxError::Init {
        path: path.to_path_buf(),
        reason,
    };
    let builder = ort::init_from(path).map_err(|e| init_err(e.to_string()))?;
    let environment = builder
        .with_name("tessera-rl")
        .with_telemetry(telemetry_enabled)
        .build()
        .map_err(|e| init_err(e.to_string()))?;
    Ok(Runtime {
        library,
        canonical: path.to_path_buf(),
        telemetry_enabled,
        _environment: environment,
    })
}

/// Requires exactly [`PINNED_VERSION`].
fn check_version(path: &Path, found: &str) -> Result<(), OnnxError> {
    if found == PINNED_VERSION {
        Ok(())
    } else {
        Err(OnnxError::Version {
            path: path.to_path_buf(),
            found: found.to_owned(),
            pinned: PINNED_VERSION,
        })
    }
}

fn from_sys(path: &Path, error: SysError) -> OnnxError {
    match error {
        SysError::Open { reason, .. } => OnnxError::MissingLibrary {
            path: path.to_path_buf(),
            reason,
        },
        other => OnnxError::NotRuntime {
            path: path.to_path_buf(),
            reason: other.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{PINNED_VERSION, check_version};
    use crate::error::OnnxError;

    #[test]
    fn test_loader_version_exact_is_accepted() {
        assert_eq!(check_version(Path::new("x"), "1.31.1"), Ok(()));
    }

    #[test]
    fn test_loader_version_other_is_refused() {
        for v in ["1.31.0", "1.31.2", "1.32.1", "2.31.1", "1.3.1"] {
            assert!(
                matches!(check_version(Path::new("x"), v), Err(OnnxError::Version { pinned, .. }) if pinned == PINNED_VERSION),
                "{v}"
            );
        }
    }

    #[test]
    fn test_loader_pinned_version_matches_compat_targets() {
        let toml = std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../compat/targets.toml"),
        )
        .unwrap();
        let line = toml
            .lines()
            .find(|l| l.trim_start().starts_with("onnxruntime ="))
            .expect("compat/targets.toml pins onnxruntime");
        assert_eq!(line.trim(), format!("onnxruntime = \"{PINNED_VERSION}\""));
    }
}
