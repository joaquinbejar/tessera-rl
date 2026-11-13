// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ONNX search evaluator (ADR-0057): an [`Evaluator`] backed by one
//! ONNX Runtime session on the CPU execution provider.
//!
//! Construction loads the runtime ([`load_runtime`], telemetry gate
//! first), reads the model once under a size cap, hashes it, builds the
//! session from those bytes at the pinned settings (CPU execution provider
//! only, one intra-op and one inter-op thread, spinning off, the basic
//! optimization level), and checks the model's inputs and outputs against
//! the environment. Every refusal is a typed [`OnnxError`].
//!
//! Each batch copies the leaves' observations into typed buffers, runs the
//! session on the calling thread, widens the outputs to `f64`, and hands
//! them to the search, which validates them as for any evaluator. Runtime
//! failures are [`EvaluatorError::Failed`], wrong output shapes
//! [`EvaluatorError::Shape`]; either rolls the search call back. The
//! floating-point controls are made canonical again after every call. The
//! call allocates (the safe `ort` API): the evaluator boundary is outside
//! the zero-allocation guarantee, as the Python callback is.

use std::io::Read;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::session::builder::{AutoDevicePolicy, GraphOptimizationLevel};
use ort::value::{TensorElementType, TensorRef, ValueType};
use tessera_rl_core::Environment;
use tessera_rl_core::space::DType;
use tessera_rl_engine::Record;
use tessera_rl_search::{Evaluator, EvaluatorError, LeafBatch};

use crate::error::OnnxError;
use crate::layout::{LeafInput, action_count, leaf_inputs};
use crate::loader::load_runtime;
use crate::provenance::Provenance;
use crate::telemetry::TelemetryPolicy;

/// The default model size cap: 256 MiB (ADR-0057 section 5).
pub const DEFAULT_MAX_MODEL_BYTES: u64 = 256 * 1024 * 1024;

/// The execution provider every session is restricted to.
pub const CPU_PROVIDER: &str = "CPUExecutionProvider";

/// The pinned graph optimization level (ADR-0057 section 2).
pub const OPTIMIZATION_LEVEL: &str = "basic";

/// How to build an [`OnnxEvaluator`].
#[derive(Debug, Clone)]
pub struct OnnxOptions {
    /// The ONNX Runtime library, by explicit path (ADR-0026).
    pub library: PathBuf,
    /// Whether the runtime may be loaded with telemetry enabled.
    pub telemetry: TelemetryPolicy,
    /// The model size cap in bytes, checked while reading.
    pub max_model_bytes: u64,
    /// Intra-op threads; other than 1 is recorded and outside the class.
    pub intra_threads: usize,
    /// Inter-op threads; other than 1 is recorded and outside the class.
    pub inter_threads: usize,
}

impl OnnxOptions {
    /// The defaults: telemetry required off, 256 MiB, one thread each.
    #[must_use]
    pub fn new(library: impl Into<PathBuf>) -> Self {
        Self {
            library: library.into(),
            telemetry: TelemetryPolicy::RequireDisabled,
            max_model_bytes: DEFAULT_MAX_MODEL_BYTES,
            intra_threads: 1,
            inter_threads: 1,
        }
    }
}

/// The output element types accepted for `priors` and `value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Float {
    F32,
    F64,
}

/// One leaf's typed input buffer, reused across batches.
#[derive(Debug)]
enum Buffer {
    F32(Vec<f32>),
    F64(Vec<f64>),
    U8(Vec<u8>),
    I8(Vec<i8>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

/// An ONNX model as a search evaluator for environment `E`.
#[derive(Debug)]
pub struct OnnxEvaluator<E: Environment> {
    session: Session,
    leaves: Vec<LeafInput>,
    buffers: Vec<Buffer>,
    /// Bytes of one leaf's whole observation (all leaves).
    row_bytes: usize,
    actions: usize,
    priors_type: Float,
    value_type: Float,
    provenance: Provenance,
    _env: PhantomData<fn() -> E>,
}

impl<E: Environment> OnnxEvaluator<E> {
    /// Builds the evaluator for `cfg` from the model at `model`.
    ///
    /// # Errors
    ///
    /// Any loader error ([`load_runtime`]); [`OnnxError::Model`] if the
    /// model is too large, cannot be read or parsed, or uses external data;
    /// [`OnnxError::Provider`] if the CPU execution provider cannot be made
    /// the only one; [`OnnxError::Contract`] if its inputs or outputs do not
    /// match the environment.
    pub fn new(cfg: &E::Config, model: &Path, options: &OnnxOptions) -> Result<Self, OnnxError> {
        let runtime = load_runtime(&options.library, options.telemetry)?;
        let contract = |what: String| OnnxError::Contract { what };
        let leaves = leaf_inputs(&E::observation_space(cfg).map_err(|e| contract(e.to_string()))?)?;
        let actions = action_count(&E::action_space(cfg).map_err(|e| contract(e.to_string()))?)?;
        let bytes = read_model(model, options.max_model_bytes)?;
        let model_sha256 = crate::hash::sha256(&bytes);
        let session =
            build_session(runtime.environment(), &bytes, options).map_err(|e| match e {
                Built::Provider(reason) => OnnxError::Provider { reason },
                Built::Model(reason) => OnnxError::Model {
                    path: model.to_path_buf(),
                    reason,
                },
            })?;
        check_inputs(&session, &leaves)?;
        let priors_type = check_output(&session, "priors", Some(actions))?;
        let value_type = check_value(&session)?;
        let buffers = leaves
            .iter()
            .map(|l| empty_buffer(l.dtype))
            .collect::<Result<_, _>>()?;
        let row_bytes = leaves.iter().map(LeafInput::bytes).sum();
        let provenance = Provenance::new(runtime, model_sha256, options);
        Ok(Self {
            session,
            leaves,
            buffers,
            row_bytes,
            actions,
            priors_type,
            value_type,
            provenance,
            _env: PhantomData,
        })
    }

    /// What this evaluator's results depend on (ADR-0057 section 3).
    #[must_use]
    pub const fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    fn run(
        &mut self,
        observations: &[u8],
        n: usize,
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        let failed = |reason: String| EvaluatorError::Failed { reason };
        let shape = |what: String| EvaluatorError::Shape { what };
        if observations.len() != n.saturating_mul(self.row_bytes) {
            return Err(shape(format!(
                "observations hold {} bytes, expected {n} rows of {}",
                observations.len(),
                self.row_bytes
            )));
        }
        fill(
            &self.leaves,
            &mut self.buffers,
            observations,
            n,
            self.row_bytes,
        );
        let mut inputs: Vec<(String, ort::session::SessionInputValue<'_>)> =
            Vec::with_capacity(self.leaves.len());
        for (leaf, buffer) in self.leaves.iter().zip(&self.buffers) {
            let mut dims = Vec::with_capacity(leaf.shape.len() + 1);
            dims.push(n);
            dims.extend(
                leaf.shape
                    .iter()
                    .map(|&d| usize::try_from(d).unwrap_or(usize::MAX)),
            );
            let value = tensor(buffer, dims).map_err(|e| failed(e.to_string()))?;
            inputs.push((leaf.name.clone(), value));
        }
        let outputs = self.session.run(inputs).map_err(|e| failed(e.to_string()));
        // The runtime may change the controls; restore the canonical ones
        // before the search does any arithmetic (ADR-0037, ADR-0057).
        let _ = tessera_rl_num::fp::ensure_canonical();
        let outputs = outputs?;
        let priors_out = outputs
            .get("priors")
            .ok_or_else(|| failed("no priors output".to_owned()))?;
        widen(priors_out, self.priors_type, &[n, self.actions], priors).map_err(shape)?;
        let value_out = outputs
            .get("value")
            .ok_or_else(|| failed("no value output".to_owned()))?;
        widen_value(value_out, self.value_type, n, values).map_err(shape)?;
        Ok(())
    }
}

impl<E: Environment> Evaluator<E> for OnnxEvaluator<E> {
    /// The ONNX evaluator needs the observations: the search always calls
    /// [`Self::evaluate_batch`]; a direct call to this method fails.
    fn evaluate(
        &mut self,
        _: &[Record<E>],
        _: &mut [f64],
        _: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        Err(EvaluatorError::Failed {
            reason: "the ONNX evaluator evaluates through evaluate_batch".to_owned(),
        })
    }

    fn evaluate_batch(
        &mut self,
        batch: &mut LeafBatch<'_, E>,
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        let n = batch.len();
        if n == 0 {
            return Ok(());
        }
        if batch.obs_bytes() != self.row_bytes {
            return Err(EvaluatorError::Shape {
                what: format!(
                    "leaf observations have {} bytes, the model expects {}",
                    batch.obs_bytes(),
                    self.row_bytes
                ),
            });
        }
        let observations = batch.observations();
        self.run(observations, n, priors, values)
    }
}

/// Reads the model once, refusing more than `cap` bytes while reading.
fn read_model(path: &Path, cap: u64) -> Result<Vec<u8>, OnnxError> {
    let model = |reason: String| OnnxError::Model {
        path: path.to_path_buf(),
        reason,
    };
    let file = std::fs::File::open(path).map_err(|e| model(e.to_string()))?;
    let mut bytes = Vec::new();
    file.take(cap.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|e| model(e.to_string()))?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > cap {
        return Err(model(format!(
            "larger than the {cap}-byte cap (max_model_bytes)"
        )));
    }
    Ok(bytes)
}

pub(crate) enum Built {
    Provider(String),
    Model(String),
}

/// The session at the pinned settings, on the CPU execution provider only.
pub(crate) fn build_session(
    env: &ort::environment::Environment,
    bytes: &[u8],
    options: &OnnxOptions,
) -> Result<Session, Built> {
    let model = |e: &dyn std::fmt::Display| Built::Model(e.to_string());
    let cpu = single_cpu(env.devices().map(|d| {
        let ep = d.ep().ok().map(str::to_owned);
        (d, ep)
    }))
    .map_err(Built::Provider)?;
    let builder = Session::builder(env).map_err(|e| model(&e))?;
    let mut builder = builder
        .with_auto_device(AutoDevicePolicy::PreferCPU)
        .map_err(|e| Built::Provider(e.to_string()))?
        .with_devices([cpu], None)
        .map_err(|e| Built::Provider(e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .map_err(|e| model(&e))?
        .with_intra_threads(options.intra_threads)
        .map_err(|e| model(&e))?
        .with_inter_threads(options.inter_threads)
        .map_err(|e| model(&e))?
        .with_intra_op_spinning(false)
        .map_err(|e| model(&e))?
        .with_inter_op_spinning(false)
        .map_err(|e| model(&e))?;
    builder.commit_from_memory(bytes).map_err(|e| model(&e))
}

/// The one device whose execution provider is the CPU one, from
/// `(device, provider name)` pairs; none or several is an error.
pub(crate) fn single_cpu<T>(
    devices: impl IntoIterator<Item = (T, Option<String>)>,
) -> Result<T, String> {
    let mut cpu = devices
        .into_iter()
        .filter(|(_, ep)| ep.as_deref() == Some(CPU_PROVIDER))
        .map(|(d, _)| d);
    match (cpu.next(), cpu.next()) {
        (Some(device), None) => Ok(device),
        (None, _) => Err(format!("no device of {CPU_PROVIDER}")),
        (Some(_), Some(_)) => Err(format!("more than one device of {CPU_PROVIDER}")),
    }
}

fn element_type(dtype: DType) -> Option<TensorElementType> {
    Some(match dtype {
        DType::F32 => TensorElementType::Float32,
        DType::F64 => TensorElementType::Float64,
        DType::U8 => TensorElementType::Uint8,
        DType::I8 => TensorElementType::Int8,
        DType::I32 => TensorElementType::Int32,
        DType::I64 => TensorElementType::Int64,
        DType::Bool => TensorElementType::Bool,
        DType::U16 | DType::I16 | DType::U32 => return None,
    })
}

fn empty_buffer(dtype: DType) -> Result<Buffer, OnnxError> {
    Ok(match dtype {
        DType::F32 => Buffer::F32(Vec::new()),
        DType::F64 => Buffer::F64(Vec::new()),
        DType::U8 => Buffer::U8(Vec::new()),
        DType::I8 => Buffer::I8(Vec::new()),
        DType::I32 => Buffer::I32(Vec::new()),
        DType::I64 => Buffer::I64(Vec::new()),
        DType::Bool => Buffer::Bool(Vec::new()),
        other => {
            return Err(OnnxError::Contract {
                what: format!("observation dtype {other:?} is not supported by the ONNX evaluator"),
            });
        }
    })
}

/// A non-batch dimension matches when equal; the batch dimension must be
/// dynamic (`-1`) or a positive number (checked per batch by the runtime).
fn dims_match(model: &[i64], expected: &[u32]) -> bool {
    model.len() == expected.len() + 1
        && model
            .iter()
            .skip(1)
            .zip(expected)
            .all(|(&m, &e)| m == i64::from(e))
}

fn check_inputs(session: &Session, leaves: &[LeafInput]) -> Result<(), OnnxError> {
    let contract = |what: String| OnnxError::Contract { what };
    let inputs = session.inputs();
    if inputs.len() != leaves.len() {
        let names: Vec<_> = inputs.iter().map(|i| i.name().to_owned()).collect();
        return Err(contract(format!(
            "the model has inputs {names:?}, the environment's leaves are {:?}",
            leaves.iter().map(|l| l.name.as_str()).collect::<Vec<_>>()
        )));
    }
    for leaf in leaves {
        let input = inputs
            .iter()
            .find(|i| i.name() == leaf.name)
            .ok_or_else(|| contract(format!("the model has no input named {:?}", leaf.name)))?;
        let ValueType::Tensor { ty, shape, .. } = input.dtype() else {
            return Err(contract(format!("input {:?} is not a tensor", leaf.name)));
        };
        if Some(*ty) != element_type(leaf.dtype) {
            return Err(contract(format!(
                "input {:?} is {ty:?}, the leaf is {:?}",
                leaf.name, leaf.dtype
            )));
        }
        if !dims_match(shape, &leaf.shape) {
            return Err(contract(format!(
                "input {:?} has shape {:?}, expected [batch, {:?}]",
                leaf.name,
                &shape[..],
                leaf.shape
            )));
        }
    }
    Ok(())
}

fn float_of(ty: TensorElementType) -> Option<Float> {
    match ty {
        TensorElementType::Float32 => Some(Float::F32),
        TensorElementType::Float64 => Some(Float::F64),
        _ => None,
    }
}

/// `priors`: `[batch, actions]`, `float32` or `float64`.
fn check_output(session: &Session, name: &str, actions: Option<usize>) -> Result<Float, OnnxError> {
    let contract = |what: String| OnnxError::Contract { what };
    let output = session
        .outputs()
        .iter()
        .find(|o| o.name() == name)
        .ok_or_else(|| contract(format!("the model has no output named {name:?}")))?;
    let ValueType::Tensor { ty, shape, .. } = output.dtype() else {
        return Err(contract(format!("output {name:?} is not a tensor")));
    };
    let float = float_of(*ty)
        .ok_or_else(|| contract(format!("output {name:?} is {ty:?}, not float32 or float64")))?;
    if let Some(a) = actions {
        let a = u32::try_from(a).map_err(|_| contract("too many actions".to_owned()))?;
        if !dims_match(shape, &[a]) {
            return Err(contract(format!(
                "output {name:?} has shape {:?}, expected [batch, {a}]",
                &shape[..]
            )));
        }
    }
    Ok(float)
}

/// `value`: `[batch]` or `[batch, 1]`, `float32` or `float64`.
fn check_value(session: &Session) -> Result<Float, OnnxError> {
    let float = check_output(session, "value", None)?;
    let output = session.outputs().iter().find(|o| o.name() == "value");
    if let Some(ValueType::Tensor { shape, .. }) = output.map(ort::value::Outlet::dtype)
        && !(shape.len() == 1 || dims_match(shape, &[1]))
    {
        return Err(OnnxError::Contract {
            what: format!(
                "output \"value\" has shape {:?}, expected [batch] or [batch, 1]",
                &shape[..]
            ),
        });
    }
    Ok(float)
}

/// Splits each observation row into its leaves and decodes them into the
/// typed buffers (native byte order, as the engine writes them).
fn fill(
    leaves: &[LeafInput],
    buffers: &mut [Buffer],
    observations: &[u8],
    n: usize,
    row_bytes: usize,
) {
    let mut offset = 0usize;
    for (leaf, buffer) in leaves.iter().zip(buffers.iter_mut()) {
        let width = leaf.bytes();
        let rows = (0..n).filter_map(|r| {
            let start = r.checked_mul(row_bytes)?.checked_add(offset)?;
            observations.get(start..start.checked_add(width)?)
        });
        match buffer {
            Buffer::F32(v) => decode(v, rows, f32::from_ne_bytes),
            Buffer::F64(v) => decode(v, rows, f64::from_ne_bytes),
            Buffer::U8(v) => {
                v.clear();
                rows.for_each(|r| v.extend_from_slice(r));
            }
            Buffer::I8(v) => decode(v, rows, i8::from_ne_bytes),
            Buffer::I32(v) => decode(v, rows, i32::from_ne_bytes),
            Buffer::I64(v) => decode(v, rows, i64::from_ne_bytes),
            Buffer::Bool(v) => {
                v.clear();
                rows.for_each(|r| v.extend(r.iter().map(|&b| b != 0)));
            }
        }
        offset = offset.saturating_add(width);
    }
}

fn decode<'a, T, const N: usize>(
    out: &mut Vec<T>,
    rows: impl Iterator<Item = &'a [u8]>,
    from: fn([u8; N]) -> T,
) {
    out.clear();
    for row in rows {
        out.extend(row.as_chunks::<N>().0.iter().map(|&c| from(c)));
    }
}

fn tensor(buffer: &Buffer, dims: Vec<usize>) -> ort::Result<ort::session::SessionInputValue<'_>> {
    Ok(match buffer {
        Buffer::F32(v) => TensorRef::from_array_view((dims, v.as_slice()))?.into(),
        Buffer::F64(v) => TensorRef::from_array_view((dims, v.as_slice()))?.into(),
        Buffer::U8(v) => TensorRef::from_array_view((dims, v.as_slice()))?.into(),
        Buffer::I8(v) => TensorRef::from_array_view((dims, v.as_slice()))?.into(),
        Buffer::I32(v) => TensorRef::from_array_view((dims, v.as_slice()))?.into(),
        Buffer::I64(v) => TensorRef::from_array_view((dims, v.as_slice()))?.into(),
        Buffer::Bool(v) => TensorRef::from_array_view((dims, v.as_slice()))?.into(),
    })
}

fn shape_is(shape: &[i64], expected: &[usize]) -> bool {
    shape.len() == expected.len()
        && shape
            .iter()
            .zip(expected)
            .all(|(&s, &e)| usize::try_from(s).ok() == Some(e))
}

/// Copies an output of shape `expected`, widened exactly to `f64`.
fn widen(
    value: &ort::value::DynValue,
    ty: Float,
    expected: &[usize],
    out: &mut [f64],
) -> Result<(), String> {
    let copy = |shape: &[i64], len: usize| {
        if !shape_is(shape, expected) || len != out.len() {
            return Err(format!("output has shape {shape:?}, expected {expected:?}"));
        }
        Ok(())
    };
    match ty {
        Float::F32 => {
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            copy(shape, data.len())?;
            out.iter_mut()
                .zip(data)
                .for_each(|(o, &d)| *o = f64::from(d));
        }
        Float::F64 => {
            let (shape, data) = value
                .try_extract_tensor::<f64>()
                .map_err(|e| e.to_string())?;
            copy(shape, data.len())?;
            out.copy_from_slice(data);
        }
    }
    Ok(())
}

/// `value` is `[n]` or `[n, 1]`.
fn widen_value(
    value: &ort::value::DynValue,
    ty: Float,
    n: usize,
    out: &mut [f64],
) -> Result<(), String> {
    widen(value, ty, &[n], out).or_else(|_| widen(value, ty, &[n, 1], out))
}

#[cfg(test)]
#[path = "evaluator_tests.rs"]
mod tests;
