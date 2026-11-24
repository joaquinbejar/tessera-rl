// SPDX-License-Identifier: MIT OR Apache-2.0

//! How a host environment check failed (#351).

use crate::error::EngineError;

/// The first failure a check found. `call` counts pool calls: 0 is the
/// first reset, `k` the `k`-th step after it.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CheckError {
    /// A pool call failed (construction, a reset, a step, a snapshot or a
    /// restore), or the pool was poisoned.
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// Two runs that must agree gave different outputs.
    #[error("call {call}: the {what} differ between {left} and {right}")]
    Diverged {
        /// The call.
        call: usize,
        /// The first run.
        left: &'static str,
        /// The second run.
        right: &'static str,
        /// The first output that differs (`rewards`, `alive`, ...).
        what: &'static str,
    },
    /// An observation value lies outside the declared space.
    #[error(
        "call {call}, instance {instance}, agent {agent}: observation leaf `{leaf}` element {element} is {value}, outside its space ({what})"
    )]
    OutsideSpace {
        /// The call.
        call: usize,
        /// The instance.
        instance: usize,
        /// The agent slot.
        agent: usize,
        /// The leaf's path (empty for a single-leaf space).
        leaf: String,
        /// The element within one agent's leaf.
        element: usize,
        /// The value read, formatted.
        value: String,
        /// Which rule it broke.
        what: &'static str,
    },
    /// A reward is NaN or infinite.
    #[error("call {call}, instance {instance}, agent {agent}: reward {value} is not finite")]
    RewardNotFinite {
        /// The call.
        call: usize,
        /// The instance.
        instance: usize,
        /// The agent slot.
        agent: usize,
        /// The reward.
        value: f32,
    },
    /// The pool's layout does not match the declared spaces.
    #[error("the layout does not match the declared spaces: {what}")]
    Layout {
        /// What does not match.
        what: &'static str,
    },
    /// An output broke one of the host invariants.
    #[error("call {call}, instance {instance}: {what}")]
    Invariant {
        /// The call.
        call: usize,
        /// The instance.
        instance: usize,
        /// Which invariant.
        what: &'static str,
    },
}
