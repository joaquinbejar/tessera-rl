// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Python exception hierarchy and the mapping from engine errors.
//!
//! ```text
//! TesseraError(Exception)
//!   ConfigError        construction: config, spaces, options
//!   ArgumentError      ids, counts, labels, record sizes, bad arguments
//!   ActionError        actions that fail the validation boundary
//!   RecordError        records that fail decoding or import checks
//!   PendingResetError  stepping an instance that waits for its reset
//!   PoisonedError      the environment is poisoned (or a worker failed)
//!   ConcurrencyError   a concurrent or nested call on one environment
//!   ExhaustedError     a step, episode or branch counter would overflow
//! ```

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyModule, PyType};
use tessera_rl_engine::EngineError;

use crate::env::BindError;

/// The exception classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Base,
    Config,
    Argument,
    Action,
    Record,
    PendingReset,
    Poisoned,
    Concurrency,
    Exhausted,
}

const KINDS: [(Kind, &str, &std::ffi::CStr); 9] = [
    (
        Kind::Base,
        "TesseraError",
        c"Base class of every Tessera error.",
    ),
    (
        Kind::Config,
        "ConfigError",
        c"Invalid config, spaces or pool options.",
    ),
    (
        Kind::Argument,
        "ArgumentError",
        c"Invalid ids, counts, labels or arguments.",
    ),
    (
        Kind::Action,
        "ActionError",
        c"Actions rejected at the validation boundary.",
    ),
    (
        Kind::Record,
        "RecordError",
        c"A record that fails decoding or import checks.",
    ),
    (
        Kind::PendingReset,
        "PendingResetError",
        c"The instance waits for its reset.",
    ),
    (
        Kind::Poisoned,
        "PoisonedError",
        c"The environment is poisoned.",
    ),
    (
        Kind::Concurrency,
        "ConcurrencyError",
        c"A concurrent or nested call on one environment.",
    ),
    (
        Kind::Exhausted,
        "ExhaustedError",
        c"A step, episode or branch counter would overflow.",
    ),
];

static TYPES: [PyOnceLock<Py<PyType>>; 9] = [const { PyOnceLock::new() }; 9];

fn index(kind: Kind) -> usize {
    KINDS.iter().position(|(k, _, _)| *k == kind).unwrap_or(0)
}

/// Creates the classes and adds them to `module`.
pub(crate) fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let base = PyErr::new_type(
        py,
        c"tessera_rl.TesseraError",
        Some(KINDS[0].2),
        Some(&py.get_type::<PyException>()),
        None,
    )?;
    let _ = TYPES[0].set(py, base.clone_ref(py));
    module.add(KINDS[0].1, base.bind(py))?;
    for (i, (_, name, doc)) in KINDS.iter().enumerate().skip(1) {
        let qualified = std::ffi::CString::new(format!("tessera_rl.{name}"))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let ty = PyErr::new_type(py, &qualified, Some(*doc), Some(base.bind(py)), None)?;
        module.add(*name, ty.bind(py))?;
        if let Some(slot) = TYPES.get(i) {
            let _ = slot.set(py, ty);
        }
    }
    Ok(())
}

/// An error of class `kind` with `message`.
pub(crate) fn error(py: Python<'_>, kind: Kind, message: String) -> PyErr {
    match TYPES.get(index(kind)).and_then(|slot| slot.get(py)) {
        Some(ty) => PyErr::from_type(ty.bind(py).clone(), message),
        None => PyException::new_err(message),
    }
}

fn engine_kind(e: &EngineError) -> Kind {
    match e {
        EngineError::Config(_)
        | EngineError::Space(_)
        | EngineError::UnsupportedSpace { .. }
        | EngineError::Options { .. } => Kind::Config,
        EngineError::InstanceOutOfRange { .. }
        | EngineError::DuplicateInstance { .. }
        | EngineError::SeedCount { .. }
        | EngineError::RecordCount { .. }
        | EngineError::LabelCount { .. }
        | EngineError::DuplicateLabel { .. } => Kind::Argument,
        EngineError::ActionLength { .. }
        | EngineError::DiscreteOutOfRange { .. }
        | EngineError::ContinuousOutOfRange { .. }
        | EngineError::IllegalAction { .. }
        | EngineError::MaskBudgetExceeded { .. } => Kind::Action,
        EngineError::InvalidRecord { .. } | EngineError::Invariant { .. } => Kind::Record,
        EngineError::PendingReset { .. } => Kind::PendingReset,
        EngineError::Poisoned
        | EngineError::Panicked
        | EngineError::WorkerDied
        | EngineError::WorkerStartup { .. }
        | EngineError::InvalidAgentEvent { .. } => Kind::Poisoned,
        EngineError::NestedEngine => Kind::Concurrency,
        EngineError::CounterExhausted { .. } | EngineError::BranchCounterExhausted => {
            Kind::Exhausted
        }
    }
}

/// The Python exception for `e`.
pub(crate) fn to_py(py: Python<'_>, e: BindError) -> PyErr {
    match e {
        BindError::Engine(e) => error(py, engine_kind(&e), e.to_string()),
        BindError::Decode(e) => error(py, Kind::Record, e.to_string()),
        BindError::Argument(message) => error(py, Kind::Argument, message),
    }
}
