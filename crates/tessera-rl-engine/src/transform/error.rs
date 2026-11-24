// SPDX-License-Identifier: MIT OR Apache-2.0

//! Errors of a transform list (ADR-0078).

/// A transform list the pool cannot run, or a transform operation it
/// refuses. Configuration errors are raised by `with_transforms`, before
/// the first reset.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum TransformError {
    /// No exported leaf has this name.
    #[error("no observation leaf named {leaf:?}")]
    UnknownLeaf {
        /// The name.
        leaf: String,
    },
    /// The leaf belongs to a `Set` (ADR-0082): transforms on set leaves are
    /// refused for now.
    #[error("leaf {leaf:?} belongs to a Set; transforms on Set leaves are not supported")]
    OnSetLeaf {
        /// The leaf.
        leaf: String,
    },
    /// The transform does not apply to this leaf.
    #[error("{transform} does not apply to leaf {leaf:?}: {reason}")]
    Unsupported {
        /// The transform.
        transform: &'static str,
        /// The leaf.
        leaf: String,
        /// Why.
        reason: &'static str,
    },
    /// A frame stack depth outside `2..=64`.
    #[error("frame stack depth {k} outside 2..=64")]
    StackDepth {
        /// The depth.
        k: u32,
    },
    /// A clip that is not finite and positive.
    #[error("normalize clip {clip} must be finite and positive")]
    Clip {
        /// The clip.
        clip: f32,
    },
    /// A second `Normalize` of the same leaf.
    #[error("leaf {leaf:?} is normalized twice")]
    NormalizeTwice {
        /// The leaf.
        leaf: String,
    },
    /// A `Normalize` after a `FrameStack` of the same leaf.
    #[error("leaf {leaf:?} is normalized after a FrameStack")]
    NormalizeAfterStack {
        /// The leaf.
        leaf: String,
    },
    /// The transformed observation of one agent exceeds 16 MiB.
    #[error("transformed observation of {bytes} bytes per agent exceeds 16 MiB")]
    RowTooLarge {
        /// Bytes per agent.
        bytes: usize,
    },
    /// The normalization accumulators exceed 256 MiB.
    #[error("normalization accumulators of {bytes} bytes exceed 256 MiB")]
    AccumulatorTooLarge {
        /// The bound, `ceil(N / chunk_size) * elements * 24`.
        bytes: usize,
    },
    /// An Async pool with an unfrozen `Normalize`: its workers do not step
    /// whole chunks, so chunk accumulation would race.
    #[error("Normalize is not supported in Async mode")]
    NormalizeInAsync,
    /// `with_transforms` after the first reset, or a second time.
    #[error("transforms are set before the first reset, once")]
    AfterReset,
    /// A context whose transform identity is not the pool's.
    #[error("transform identity {found:#018x} differs from the pool's {expected:#018x}")]
    IdentityMismatch {
        /// The pool's.
        expected: u64,
        /// The context's.
        found: u64,
    },
    /// A frame-stack slice of the wrong length.
    #[error("frame-stack slice of {found} bytes, expected {expected}")]
    StateLength {
        /// Bytes per instance.
        expected: usize,
        /// Bytes given.
        found: usize,
    },
    /// Persisting or recording a pool with transforms (ADR-0078 item 8).
    #[error("pools with transforms cannot be persisted or recorded yet")]
    NotPersistable,
    /// A plan invariant broke (a bug).
    #[error("internal transform plan error")]
    Internal,
}
