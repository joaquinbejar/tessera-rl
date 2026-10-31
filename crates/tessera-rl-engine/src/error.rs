// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed errors of the engine (ADR-0022, ADR-0028).

use tessera_rl_core::{ConfigError, SpaceError};
use thiserror::Error;

/// Errors returned by pool construction and the validation boundary. On
/// any error from `step` or `reset`, no instance was mutated.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum EngineError {
    /// The environment config is invalid.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// A space is invalid or not supported.
    #[error(transparent)]
    Space(#[from] SpaceError),
    /// A space kind the engine does not support yet.
    #[error("unsupported space: {what}")]
    UnsupportedSpace {
        /// What is not supported.
        what: &'static str,
    },
    /// Invalid pool options.
    #[error("invalid pool option {option}: {reason}")]
    Options {
        /// The option.
        option: &'static str,
        /// Why it is invalid.
        reason: &'static str,
    },
    /// An instance id is out of range.
    #[error("instance {id} out of range for a pool of {num_envs}")]
    InstanceOutOfRange {
        /// The id.
        id: u32,
        /// Pool size.
        num_envs: u32,
    },
    /// An instance id appears twice in one call.
    #[error("instance {id} appears more than once")]
    DuplicateInstance {
        /// The id.
        id: u32,
    },
    /// A seed list does not have one seed per instance.
    #[error("{seeds} seeds for {instances} instances")]
    SeedCount {
        /// Seeds given.
        seeds: usize,
        /// Instances to reseed.
        instances: usize,
    },
    /// An action buffer has the wrong length.
    #[error("{buffer} actions hold {actual} values, expected {expected}")]
    ActionLength {
        /// `discrete` or `continuous`.
        buffer: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// A discrete action is outside `0..n`.
    #[error("instance {instance}, agent {agent}, head {head}: action {value} outside 0..{n}")]
    DiscreteOutOfRange {
        /// Instance.
        instance: u32,
        /// Agent slot.
        agent: u16,
        /// Head.
        head: u32,
        /// The value.
        value: i32,
        /// Number of choices.
        n: u32,
    },
    /// A continuous action is NaN, infinite or outside its bounds.
    #[error(
        "instance {instance}, agent {agent}, head {head}: action {value} is not finite or outside its bounds"
    )]
    ContinuousOutOfRange {
        /// Instance.
        instance: u32,
        /// Agent slot.
        agent: u16,
        /// Head.
        head: u32,
        /// The value.
        value: f32,
    },
    /// The instance's episode ended and it has not been reset.
    #[error("instance {id} is pending reset")]
    PendingReset {
        /// The id.
        id: u32,
    },
    /// The instance's step or episode counter would overflow.
    #[error("instance {id}: {counter} exhausted")]
    CounterExhausted {
        /// The id.
        id: u32,
        /// `step_index` or `episode_index`.
        counter: &'static str,
    },
}
