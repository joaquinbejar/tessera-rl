// SPDX-License-Identifier: MIT OR Apache-2.0

//! League errors, raised at construction and at the barrier, never on the
//! step path.

/// An error of the league machinery.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LeagueError {
    /// The population would exceed `u32::MAX` entries.
    #[error("the population is full")]
    PopulationFull,
    /// A policy id names no entry of the population.
    #[error("unknown policy id {0}")]
    UnknownPolicy(u32),
    /// A result names an instance outside the result rings.
    #[error("instance {instance} is outside the {instances} result rings")]
    UnknownInstance {
        /// The instance.
        instance: usize,
        /// The number of rings.
        instances: usize,
    },
    /// A configuration value is out of range.
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),
    /// Matchmaking found no active policy to draw.
    #[error("no active policy to match against")]
    NoActivePolicy,
    /// The result rings could not be allocated.
    #[error("the result rings could not be allocated")]
    Allocation,
    /// A population with an ONNX policy given to a pool: ONNX opponents
    /// come with ADR-0088's head C.
    #[error("policy {0} is an ONNX policy: pools do not run ONNX opponents yet (ADR-0088 head C)")]
    OnnxUnsupported(u32),
    /// The engine refused the view built from this league.
    #[error("invalid league view: {0}")]
    View(#[from] tessera_rl_engine::league::LeagueViewError),
}
