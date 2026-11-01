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
//!   FloatingPointEnvironmentError
//!                      a call rejected for non-default caller FP controls
//!                      (strict policy, ADR-0041)
//! ```

use core::sync::atomic::{AtomicPtr, Ordering};

use pyo3::ffi;
use tessera_rl_engine::EngineError;

use crate::env::BindError;
use crate::ffi::capi::{self, CResult, Gil, Raised, Ref};

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
    FloatingPoint,
}

const KINDS: [(Kind, &str, &std::ffi::CStr); 10] = [
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
    (
        Kind::FloatingPoint,
        "FloatingPointEnvironmentError",
        c"A call rejected because the caller's floating-point controls were not the default (strict policy).",
    ),
];

/// The classes, created once at module initialization; null before.
static TYPES: [AtomicPtr<ffi::PyObject>; 10] =
    [const { AtomicPtr::new(core::ptr::null_mut()) }; 10];

fn index(kind: Kind) -> usize {
    KINDS.iter().position(|(k, _, _)| *k == kind).unwrap_or(0)
}

/// Creates the classes and adds them to `module` (callouts: class creation
/// and module attributes allocate).
pub(crate) fn register(g: Gil, module: Ref<'_>) -> CResult<()> {
    let mut base = capi::exception_base();
    for (i, (_, name, doc)) in KINDS.iter().enumerate() {
        let qualified = std::ffi::CString::new(format!("tessera_rl.{name}"))
            .map_err(|_| capi::raise(g, capi::runtime_error(), "exception name"))?;
        let short = std::ffi::CString::new(*name)
            .map_err(|_| capi::raise(g, capi::runtime_error(), "exception name"))?;
        let ty = capi::new_exception(g, &qualified, doc, base)?;
        capi::module_add(g, module, &short, &ty)?;
        // The module keeps the class alive for the process; the stored
        // pointer is a borrowed reference to it.
        let ptr = ty.as_ref().as_ptr();
        if let Some(slot) = TYPES.get(i) {
            slot.store(ptr, Ordering::Release);
        }
        if i == 0 {
            base = ptr;
        }
    }
    Ok(())
}

/// Raises an error of class `kind` with `message`.
#[expect(
    clippy::needless_pass_by_value,
    reason = "callers build the message in place"
)]
pub(crate) fn error(g: Gil, kind: Kind, message: String) -> Raised {
    let ty = TYPES
        .get(index(kind))
        .map_or(core::ptr::null_mut(), |slot| slot.load(Ordering::Acquire));
    let ty = if ty.is_null() {
        capi::runtime_error()
    } else {
        ty
    };
    capi::raise(g, ty, &message)
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
        | EngineError::InvalidAgentEvent { .. }
        | EngineError::TurnOrder { .. } => Kind::Poisoned,
        EngineError::NestedEngine => Kind::Concurrency,
        EngineError::CounterExhausted { .. }
        | EngineError::BranchCounterExhausted
        | EngineError::StorageIdsExhausted => Kind::Exhausted,
    }
}

/// Raises the Python exception for `e`.
pub(crate) fn to_py(g: Gil, e: BindError) -> Raised {
    match e {
        BindError::Engine(e) => error(g, engine_kind(&e), e.to_string()),
        BindError::Decode(e) => error(g, Kind::Record, e.to_string()),
        BindError::Argument(message) => error(g, Kind::Argument, message),
    }
}
