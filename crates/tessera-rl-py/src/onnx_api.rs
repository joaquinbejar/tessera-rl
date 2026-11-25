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
use crate::ffi::capi::{self, CResult, Gil, Mine, Ref};
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
pub(crate) fn onnx_load<'py>(g: Gil<'py>, kwargs: Option<Ref<'_>>) -> CResult<Mine<'py>> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&LOAD_KEYS])?;
    let library = PathBuf::from(required(g, &kw, "library")?);
    let accept = kw.bool(g, "accept_telemetry")?.unwrap_or(false);
    load(g, &library, accept)
}

#[cfg(feature = "onnx")]
fn load<'py>(g: Gil<'py>, library: &std::path::Path, accept: bool) -> CResult<Mine<'py>> {
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
fn load<'py>(g: Gil<'py>, _: &std::path::Path, _: bool) -> CResult<Mine<'py>> {
    Err(to_py(
        g,
        BindError::Onnx(
            "this build of tessera-rl has no ONNX evaluator (the onnx feature)".to_owned(),
        ),
    ))
}

/// `onnx_model_provenance(model=, library=, accept_telemetry=False,
/// max_model_bytes=None) -> dict` (`model` is `bytes`, as `os.fsencode`): an ONNX opponent model's provenance
/// (ADR-0057 section 3) with its league `content_hash` (32 hex digits),
/// read without building a session (#619).
pub(crate) fn onnx_model_provenance<'py>(
    g: Gil<'py>,
    kwargs: Option<Ref<'_>>,
) -> CResult<Mine<'py>> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&ATTACH_KEYS])?;
    let model = kw
        .get("model")
        .ok_or_else(|| error(g, Kind::Argument, "missing argument \"model\"".to_owned()))?;
    let args = OnnxArgs {
        model: crate::api::path_of(g, model, "model")?,
        library: PathBuf::from(required(g, &kw, "library")?),
        accept_telemetry: kw.bool(g, "accept_telemetry")?.unwrap_or(false),
        max_model_bytes: kw.u64(g, "max_model_bytes")?,
    };
    model_provenance(g, &args)
}

#[cfg(feature = "onnx")]
fn model_provenance<'py>(g: Gil<'py>, args: &OnnxArgs) -> CResult<Mine<'py>> {
    use std::fmt::Write as _;
    let mut options = tessera_rl_onnx::OnnxOptions::new(&args.library);
    if args.accept_telemetry {
        options.telemetry = tessera_rl_onnx::TelemetryPolicy::Accept;
    }
    if let Some(cap) = args.max_model_bytes {
        options.max_model_bytes = cap;
    }
    let (hash, p) = tessera_rl_onnx::model_provenance(&args.model, &options)
        .map_err(|e| to_py(g, BindError::Onnx(e.to_string())))?;
    let d = info_dict(g, &info_of(&p))?;
    let hex = hash.iter().fold(String::with_capacity(32), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    });
    capi::dict_set(g, &d, c"content_hash", &capi::string(g, &hex)?)?;
    Ok(d)
}

/// The binding's view of an ONNX session's provenance.
#[cfg(feature = "onnx")]
pub(crate) fn info_of(p: &tessera_rl_onnx::Provenance) -> OnnxInfo {
    OnnxInfo {
        model_sha256: p.model_sha256.clone(),
        library_path: p.library_path.display().to_string(),
        library_sha256: p.library_sha256.clone(),
        library_version: p.library_version.clone(),
        provider: p.provider,
        intra_threads: p.intra_threads,
        inter_threads: p.inter_threads,
        optimization_level: p.optimization_level,
        telemetry_enabled: p.telemetry_enabled,
        arch: p.arch,
        cpu_model: p.cpu_model.clone(),
    }
}

/// `onnx_take_loaded() -> dict`: the provenance of the ONNX opponent
/// models the last pool made on this thread actually loaded, by content
/// hash, and forgets it (#619). `tessera_rl.make` calls it before and right
/// after the native maker, on the same thread, so the record describes the
/// bytes the pool runs, never an earlier read.
pub(crate) fn onnx_take_loaded<'py>(g: Gil<'py>) -> CResult<Mine<'py>> {
    let d = capi::dict(g)?;
    for (hash, info) in crate::policy::take_loaded() {
        let key = std::ffi::CString::new(hash)
            .map_err(|_| error(g, Kind::Argument, "a content hash with NUL".to_owned()))?;
        capi::dict_set(g, &d, &key, &info_dict(g, &info)?)?;
    }
    Ok(d)
}

#[cfg(not(feature = "onnx"))]
fn model_provenance<'py>(g: Gil<'py>, _: &OnnxArgs) -> CResult<Mine<'py>> {
    Err(to_py(
        g,
        BindError::Capability(
            "this build of tessera-rl has no ONNX opponents (the onnx feature)".to_owned(),
        ),
    ))
}

/// The provenance dict.
fn info_dict<'py>(g: Gil<'py>, info: &OnnxInfo) -> CResult<Mine<'py>> {
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
pub(crate) fn attach_onnx<'py>(
    g: Gil<'py>,
    inner: &SearchInner,
    kwargs: Option<Ref<'_>>,
) -> CResult<Mine<'py>> {
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
pub(crate) fn run_onnx<'py>(g: Gil<'py>, inner: &SearchInner) -> CResult<Mine<'py>> {
    let mut pending = None;
    let result = inner.with(g, |ops| {
        let mut check = || {
            // Pending signals (Ctrl-C) before every batch, as for the
            // Python callback: a handler that raises aborts the call.
            if let Err(raised) = capi::check_signals(g) {
                capi::stash_first(g, &mut pending, raised);
                return Err(EvaluatorError::Failed {
                    reason: "a signal handler raised".to_owned(),
                });
            }
            Ok(())
        };
        // Inference releases the interpreter: other Python threads run
        // meanwhile. The search object stays held by this call.
        let mut detach = |f: &mut (dyn FnMut() + Send)| g.detach(f);
        Ok(ops.run_onnx(&mut check, &mut detach))
    })?;
    match (result, pending.take()) {
        (Ok(()), _) => Ok(capi::none(g)),
        (Err(_), Some(p)) => Err(p.raise_again(g)),
        (Err(e), None) => Err(to_py(g, e)),
    }
}
