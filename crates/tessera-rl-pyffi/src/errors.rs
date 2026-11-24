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
//!   PersistError       a persisted file that is damaged, malformed or mismatched
//!   ConcurrencyError   a concurrent or nested call, or a split-phase state
//!                      error (in flight, borrowed, nothing to receive)
//!   ExhaustedError     a step, episode or branch counter would overflow
//!   ForkError          an environment made before os.fork() used in the child
//!   SearchError        a search call rejected or failed
//!   EvaluatorError     the search evaluator failed or answered badly
//!   OnnxError          the ONNX evaluator could not load its runtime or model
//!   FloatingPointEnvironmentError
//!                      a call rejected for non-default caller FP controls
//!                      (strict policy, ADR-0041)
//! ```

use core::sync::atomic::{AtomicPtr, Ordering};

use pyo3::ffi;
use tessera_rl_engine::EngineError;

use crate::ffi::capi::{self, CResult, Gil, Raised, Ref};

/// The exception classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `TesseraError`, the base class.
    Base,
    /// `ConfigError`.
    Config,
    /// `ArgumentError`.
    Argument,
    /// `ActionError`.
    Action,
    /// `RecordError`.
    Record,
    /// `PendingResetError`.
    PendingReset,
    /// `PoisonedError`.
    Poisoned,
    /// `ConcurrencyError`.
    Concurrency,
    /// `ExhaustedError`.
    Exhausted,
    /// `FloatingPointEnvironmentError`.
    FloatingPoint,
    /// `SearchError`.
    Search,
    /// `EvaluatorError`.
    Evaluator,
    /// `ForkError`.
    Forked,
    /// `PersistError`.
    Persist,
    /// `OnnxError`.
    Onnx,
    /// `CapabilityError`.
    Capability,
}

const KINDS: [(Kind, &str, &std::ffi::CStr); 16] = [
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
    (
        Kind::Search,
        "SearchError",
        c"A search call rejected or failed (ADR-0043); the tree is unchanged.",
    ),
    (
        Kind::Evaluator,
        "EvaluatorError",
        c"The search evaluator failed or returned an invalid response; the tree is unchanged.",
    ),
    (
        Kind::Forked,
        "ForkError",
        c"An environment created before os.fork() was used in the child.",
    ),
    (
        Kind::Persist,
        "PersistError",
        c"A persisted file that is damaged, malformed or does not match this environment.",
    ),
    (
        Kind::Onnx,
        "OnnxError",
        c"The ONNX evaluator could not load ONNX Runtime or its model (ADR-0057); nothing was evaluated.",
    ),
    (
        Kind::Capability,
        "CapabilityError",
        c"A call this environment kind does not support (a host environment's snapshot, for example).",
    ),
];

/// The classes, created once at module initialization; null before.
static TYPES: [AtomicPtr<ffi::PyObject>; KINDS.len()] =
    [const { AtomicPtr::new(core::ptr::null_mut()) }; KINDS.len()];

fn index(kind: Kind) -> usize {
    KINDS.iter().position(|(k, _, _)| *k == kind).unwrap_or(0)
}

/// Creates the classes and adds them to `module` (callouts: class creation
/// and module attributes allocate). The classes are created once per
/// process and held by a strong reference for its lifetime; a second
/// `register` (a second initialization of the extension module) adds the
/// same classes to the new module, so an error raised later is always of a
/// live class, and `except tessera_rl.ArgumentError` keeps catching it.
///
/// # Errors
///
/// A Python exception is set if a class cannot be created or added.
pub fn register(g: Gil<'_>, module: Ref<'_>) -> CResult<()> {
    let mut base = capi::exception_base();
    for (i, (_, name, doc)) in KINDS.iter().enumerate() {
        let short = std::ffi::CString::new(*name)
            .map_err(|_| capi::raise(g, capi::runtime_error(), "exception name"))?;
        let Some(slot) = TYPES.get(i) else {
            return Err(capi::raise(g, capi::runtime_error(), "exception table"));
        };
        let existing = slot.load(Ordering::Acquire);
        let ty = if existing.is_null() {
            let qualified = std::ffi::CString::new(format!("tessera_rl.{name}"))
                .map_err(|_| capi::raise(g, capi::runtime_error(), "exception name"))?;
            let created = capi::new_exception(g, &qualified, doc, base)?;
            // The class is kept for the process: this strong reference is
            // never released, so the stored pointer can never dangle.
            let ptr = created.into_ptr();
            slot.store(ptr, Ordering::Release);
            ptr
        } else {
            existing
        };
        capi::module_add_class(g, module, &short, ty)?;
        if i == 0 {
            base = ty;
        }
    }
    Ok(())
}

/// The classes a host module binds from `tessera_rl` (SPEC 14.4.6).
#[cfg(feature = "host")]
const BOUND: [Kind; 13] = [
    Kind::Base,
    Kind::Config,
    Kind::Argument,
    Kind::Action,
    Kind::PendingReset,
    Kind::Poisoned,
    Kind::Concurrency,
    Kind::Exhausted,
    Kind::Forked,
    Kind::FloatingPoint,
    Kind::Capability,
    // ADR-0073: host snapshots raise these two.
    Kind::Record,
    Kind::Persist,
];

/// Host modules (SPEC 14.4.6): imports `tessera_rl` and binds its
/// exception classes by name, so this module raises `tessera_rl`'s own
/// classes and `except tessera_rl.ActionError` catches them across wheels.
/// Each class is held by a strong reference for the process, as
/// [`register`]'s are. Classes outside the list keep raising
/// `RuntimeError`.
///
/// # Errors
///
/// `ImportError` naming the class when `tessera_rl` lacks one or binds a
/// name to something that is not a `BaseException` subclass (the host
/// module's import fails); the import's own error, or any other exception
/// of an attribute read, unchanged.
#[cfg(feature = "host")]
pub fn bind(g: Gil<'_>) -> CResult<()> {
    let module = capi::import_module(g, c"tessera_rl")?;
    for kind in BOUND {
        let i = index(kind);
        let (Some(slot), Some((_, name, _))) = (TYPES.get(i), KINDS.get(i)) else {
            return Err(capi::raise(g, capi::runtime_error(), "exception table"));
        };
        let short = std::ffi::CString::new(*name)
            .map_err(|_| capi::raise(g, capi::runtime_error(), "exception name"))?;
        let class = capi::get_attr(g, module.as_ref(), &short).map_err(|raised| {
            // A missing attribute fails the import with its name; any other
            // exception of the read (KeyboardInterrupt, MemoryError, an error
            // of a module `__getattr__`) propagates unchanged.
            if capi::pending_is_attribute_error(g) {
                capi::clear_pending(g);
                capi::raise(
                    g,
                    capi::import_error(),
                    &format!(
                        "tessera_rl has no exception class {name} (host protocol 1, SPEC 14.4.6)"
                    ),
                )
            } else {
                raised
            }
        })?;
        if !capi::is_exception_class(class.as_ref()) {
            return Err(capi::raise(
                g,
                capi::import_error(),
                &format!(
                    "tessera_rl.{name} is not an exception class (host protocol 1, SPEC 14.4.6)"
                ),
            ));
        }
        if slot.load(Ordering::Acquire).is_null() {
            // Kept for the process: this strong reference is never released.
            slot.store(class.into_ptr(), Ordering::Release);
        }
    }
    Ok(())
}

/// Raises an error of class `kind` with `message`.
#[expect(
    clippy::needless_pass_by_value,
    reason = "callers build the message in place"
)]
pub fn error(g: Gil, kind: Kind, message: String) -> Raised {
    let ty = TYPES
        .get(index(kind))
        .map_or(core::ptr::null_mut(), |slot| slot.load(Ordering::Acquire));
    let ty = if ty.is_null() {
        capi::runtime_error()
    } else {
        // One of the classes `register` created, or `bind` bound after
        // checking it is an exception class; held for the process.
        capi::Exc::from_type(ty)
    };
    capi::raise(g, ty, &message)
}

/// The exception class of an engine error.
#[must_use]
pub fn engine_kind(e: &EngineError) -> Kind {
    match e {
        EngineError::Config(_)
        | EngineError::SnapshotBound { .. }
        | EngineError::Space(_)
        | EngineError::UnsupportedSpace { .. }
        | EngineError::Options { .. }
        | EngineError::HalfTooSmall { .. }
        | EngineError::LaneEnvUnsupported
        | EngineError::InvalidMaskLayout { .. } => Kind::Config,
        EngineError::InstanceOutOfRange { .. }
        | EngineError::DuplicateInstance { .. }
        | EngineError::SeedCount { .. }
        | EngineError::RecordCount { .. }
        | EngineError::LabelCount { .. }
        | EngineError::DuplicateLabel { .. }
        | EngineError::WrongPool
        | EngineError::InvalidHalf { .. }
        | EngineError::WrongMode { .. }
        | EngineError::EmptySend => Kind::Argument,
        EngineError::ActionLength { .. }
        | EngineError::DiscreteOutOfRange { .. }
        | EngineError::ContinuousOutOfRange { .. }
        | EngineError::IllegalAction { .. }
        | EngineError::MaskBudgetExceeded { .. }
        | EngineError::ActionRejected { .. } => Kind::Action,
        EngineError::InvalidRecord { .. }
        | EngineError::Invariant { .. }
        | EngineError::SnapshotRejected { .. } => Kind::Record,
        EngineError::NoSnapshots => Kind::Capability,
        EngineError::PendingReset { .. } => Kind::PendingReset,
        EngineError::Poisoned
        | EngineError::Panicked
        | EngineError::WorkerDied
        | EngineError::WorkerStartup { .. }
        | EngineError::InvalidAgentEvent { .. }
        | EngineError::NanDetected { .. }
        | EngineError::AllocationInChunkStep { .. }
        | EngineError::TurnOrder { .. }
        | EngineError::RefusedNotActing { .. }
        | EngineError::SnapshotTooLong { .. } => Kind::Poisoned,
        EngineError::NestedEngine
        | EngineError::GangQueue { .. }
        | EngineError::InFlight { .. }
        | EngineError::HalfInFlight { .. }
        | EngineError::Borrowed { .. }
        | EngineError::HalfBorrowed { .. }
        | EngineError::NothingInFlight
        | EngineError::BatchUnreachable { .. }
        | EngineError::NotQuiescent
        | EngineError::WatchAttached
        | EngineError::CycleAhead { .. } => Kind::Concurrency,
        EngineError::CounterExhausted { .. }
        | EngineError::BranchCounterExhausted
        | EngineError::StorageIdsExhausted
        | EngineError::SequenceExhausted => Kind::Exhausted,
        // `EngineError` is non-exhaustive (ADR-0055): a variant added later
        // raises the base class until it is mapped here.
        _ => Kind::Base,
    }
}

/// The class of a persisted-file error (ADR-0073 host files): an engine
/// error keeps its own class, every other one is `PersistError`.
#[cfg(feature = "host")]
#[must_use]
pub fn persist_kind(e: &tessera_rl_engine::PersistError) -> Kind {
    match e {
        tessera_rl_engine::PersistError::Engine(e) => engine_kind(e),
        _ => Kind::Persist,
    }
}
