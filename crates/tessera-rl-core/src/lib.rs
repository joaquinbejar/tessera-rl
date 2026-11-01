// SPDX-License-Identifier: MIT OR Apache-2.0

//! Environment traits, spaces, specs, errors and canonical encoding.
//!
//! Environment crates depend on this crate, never on the engine (ADR-0021).

// State building blocks (ADR-0029).
pub mod bool8;
pub mod pool;
pub mod reserved;
pub mod state_enum;
pub mod state_init;

// Encoding and hashing (SPEC 8.3, ADR-0029).
pub mod encoding;

// Authoring API (SPEC 10, ADR-0012, ADR-0014, ADR-0030).
pub mod agent;
pub mod config;
pub mod env;
pub mod error;
pub mod header;
pub mod header_view;
pub mod io;
pub mod leaf;
pub mod space;
pub mod step_out;

pub use agent::{AgentSlot, Tier, TurnStructure};
pub use bool8::Bool8;
pub use encoding::{Canonical, StateHash};
pub use env::{
    ActionMask, Determinize, EnvAssets, EnvConfig, EnvState, Environment, Heuristic,
    ImperfectInformation, Render, Validate,
};
pub use error::{ConfigError, CoreError, DecodeError, EncodeError, InvariantViolation, SpaceError};
pub use header::{AdaptiveValues, AgentRow, EngineHeader, NoAdaptive, RecordHeader, RecordLayout};
pub use header_view::{AgentView, HeaderView, Phase};
pub use leaf::{LeafDType, LeafId, ObsLeaf, Observation};
pub use pool::Pool;
pub use reserved::Reserved;
pub use state_enum::StateEnum;
pub use state_init::StateInit;
pub use step_out::{EpisodeEnd, StepOut};
