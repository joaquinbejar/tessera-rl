// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ONNX evaluator's entry points (ADR-0057): `_native.onnx_load` and
//! the `attach_onnx` and `run_onnx` methods of `_native.Search`.
//!
//! `onnx_load(library=, accept_telemetry=)` loads ONNX Runtime once per
//! process. It refuses, before anything is opened, unless
//! `ORT_DISABLE_TELEMETRY=1` is already set or `accept_telemetry` is true;
//! the environment is only read. A search attaches its own session
//! (`attach_onnx`) and runs with it (`run_onnx`), polling for pending
//! signals before every batch as the Python callback does. Every failure is
//! `tessera_rl.OnnxError`, except failures of a search call, which are the
//! usual `SearchError` and `EvaluatorError` (the tree rolled back).

use std::path::PathBuf;

use crate::api::Kwargs;
use crate::env::BindError;
use crate::errors::{Kind, error, to_py};
use crate::ffi::capi::{self, CResult, Gil, Mine, Raised, Ref};
use crate::search::{OnnxArgs, OnnxInfo};
use crate::search_api::SearchInner;
use tessera_rl_search::EvaluatorError;

const LOAD_KEYS: [&str; 2] = ["library", "accept_telemetry"];
const ATTACH_KEYS: [&str; 4] = ["model", "library", "accept_telemetry", "max_model_bytes"];

fn required(g: Gil, kw: &Kwargs<'_>, key: &str) -> CResult<String> {
    kw.string(g, key)?
        .ok_or_else(|| error(g, Kind::Argument, format!("missing argument {key:?}")))
}

/// `onnx_load(library: str, accept_telemetry: bool = False) -> dict`.
pub(crate) fn onnx_load(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<Mine> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&LOAD_KEYS])?;
    let library = PathBuf::from(required(g, &kw, "library")?);
    let accept = kw.bool(g, "accept_telemetry")?.unwrap_or(false);
    load(g, &library, accept)
}

#[cfg(feature = "onnx")]
fn load(g: Gil, library: &std::path::Path, accept: bool) -> CResult<Mine> {
    use tessera_rl_onnx::{TelemetryPolicy, load_runtime};
    let policy = if accept {
        TelemetryPolicy::Accept
    } else {
        TelemetryPolicy::RequireDisabled
    };
    let runtime =
        load_runtime(library, policy).map_err(|e| to_py(g, BindError::Onnx(e.to_string())))?;
    let d = capi::dict(g)?;
    capi::dict_set(g, &d, c"version", &capi::string(g, runtime.version())?)?;
    capi::dict_set(g, &d, c"sha256", &capi::string(g, runtime.sha256())?)?;
    let path = runtime.path().display().to_string();
    capi::dict_set(g, &d, c"path", &capi::string(g, &path)?)?;
    capi::dict_set(
        g,
        &d,
        c"telemetry_enabled",
        &capi::boolean(g, runtime.telemetry_enabled()),
    )?;
    Ok(d)
}

#[cfg(not(feature = "onnx"))]
fn load(g: Gil, _: &std::path::Path, _: bool) -> CResult<Mine> {
    Err(to_py(
        g,
        BindError::Onnx(
            "this build of tessera-rl has no ONNX evaluator (the onnx feature)".to_owned(),
        ),
    ))
}

/// The provenance dict.
fn info_dict(g: Gil, info: &OnnxInfo) -> CResult<Mine> {
    let d = capi::dict(g)?;
    capi::dict_set(
        g,
        &d,
        c"model_sha256",
        &capi::string(g, &info.model_sha256)?,
    )?;
    capi::dict_set(
        g,
        &d,
        c"library_path",
        &capi::string(g, &info.library_path)?,
    )?;
    capi::dict_set(
        g,
        &d,
        c"library_sha256",
        &capi::string(g, &info.library_sha256)?,
    )?;
    capi::dict_set(
        g,
        &d,
        c"library_version",
        &capi::string(g, &info.library_version)?,
    )?;
    capi::dict_set(g, &d, c"provider", &capi::string(g, info.provider)?)?;
    let threads = |n: usize| capi::int_u64(g, u64::try_from(n).unwrap_or(u64::MAX));
    capi::dict_set(g, &d, c"intra_threads", &threads(info.intra_threads)?)?;
    capi::dict_set(g, &d, c"inter_threads", &threads(info.inter_threads)?)?;
    capi::dict_set(
        g,
        &d,
        c"optimization_level",
        &capi::string(g, info.optimization_level)?,
    )?;
    capi::dict_set(
        g,
        &d,
        c"telemetry_enabled",
        &capi::boolean(g, info.telemetry_enabled),
    )?;
    capi::dict_set(g, &d, c"arch", &capi::string(g, info.arch)?)?;
    let cpu = match &info.cpu_model {
        Some(m) => capi::string(g, m)?,
        None => capi::none(g),
    };
    capi::dict_set(g, &d, c"cpu_model", &cpu)?;
    Ok(d)
}

/// `attach_onnx(model=, library=, accept_telemetry=, max_model_bytes=)`:
/// builds the search's ONNX session and returns its provenance.
pub(crate) fn attach_onnx(g: Gil, inner: &SearchInner, kwargs: Option<Ref<'_>>) -> CResult<Mine> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&ATTACH_KEYS])?;
    let args = OnnxArgs {
        model: PathBuf::from(required(g, &kw, "model")?),
        library: PathBuf::from(required(g, &kw, "library")?),
        accept_telemetry: kw.bool(g, "accept_telemetry")?.unwrap_or(false),
        max_model_bytes: kw.u64(g, "max_model_bytes")?,
    };
    let info = inner.with(g, |ops| ops.attach_onnx(&args).map_err(|e| to_py(g, e)))?;
    info_dict(g, &info)
}

/// `run_onnx()`: one search call with the attached ONNX evaluator.
pub(crate) fn run_onnx(g: Gil, inner: &SearchInner) -> CResult<Mine> {
    let mut pending = None;
    let result = inner.with(g, |ops| {
        let mut check = || {
            // Pending signals (Ctrl-C) before every batch, as for the
            // Python callback: a handler that raises aborts the call.
            if capi::check_signals(g).is_err() {
                pending = Some(capi::fetch_pending());
                return Err(EvaluatorError::Failed {
                    reason: "a signal handler raised".to_owned(),
                });
            }
            Ok(())
        };
        // Inference releases the interpreter: other Python threads run
        // meanwhile. The search object stays held by this call.
        let mut detach = |f: &mut dyn FnMut()| g.detach(f);
        Ok(ops.run_onnx(&mut check, &mut detach))
    })?;
    match (result, pending.take()) {
        (Ok(()), _) => Ok(capi::none(g)),
        (Err(_), Some(p)) => {
            capi::restore_pending(p);
            Err(Raised)
        }
        (Err(e), None) => Err(to_py(g, e)),
    }
}
