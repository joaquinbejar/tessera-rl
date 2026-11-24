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
pub mod adapters;
pub mod agent;
pub mod asset;
pub mod config;
pub mod config_canonical;
pub mod config_decode;
pub mod entity_heads;
pub mod env;
pub mod envs;
pub mod error;
pub mod header;
pub mod header_view;
pub mod host;
pub mod host_out;
pub mod host_snapshot;
pub mod io;
pub mod lane_field;
pub mod lane_step;
pub mod leaf;
pub mod policy;
pub mod refusals;
pub mod render;
#[doc(hidden)]
pub mod sealed;
pub mod set_leaf;
pub mod set_of;
pub mod space;
pub mod step_out;

pub use adapters::Adapters;
pub use agent::{AgentSlot, Tier, TurnStructure};
pub use asset::{Asset, AssetData, AssetEntry, AssetError, AssetHash};
pub use bool8::Bool8;
pub use config_canonical::{CanonicalDecode, CanonicalDecodeError, decode_canonical};
pub use config_decode::{ConfigDecode, ConfigDecodeError, ConfigItem, ConfigSource};
pub use encoding::{Canonical, StateHash};
pub use entity_heads::{EntityHeads, HeadSpec};
#[expect(
    deprecated,
    reason = "EnvAssets stays exported until its removal (#418)"
)]
pub use env::EnvAssets;
pub use env::{
    ActionMask, Adaptive, Determinize, EnvConfig, EnvState, Environment, Heuristic,
    ImperfectInformation, InfoHeuristic, Persist, PrivilegedState, Render, Validate,
};
pub use error::{ConfigError, CoreError, DecodeError, EncodeError, InvariantViolation, SpaceError};
pub use header::{AdaptiveValues, AgentRow, EngineHeader, NoAdaptive, RecordHeader, RecordLayout};
pub use header_view::{AgentView, HeaderView, Phase};
pub use host::{ActionReject, HostEnvironment, HostError, HostSeed};
pub use host_out::{AbortCode, HostOut};
pub use host_snapshot::{HostSnapshot, SnapshotReject};
pub use lane_field::{LaneField, LaneTuple, RawColumn};
pub use lane_step::{LANES, LaneHook, LaneOut, LaneStep, lane_hook};
pub use leaf::{LeafDType, LeafId, ObsLeaf, Observation, ObservationLayout};
pub use policy::PolicyGroup;
pub use pool::Pool;
pub use refusals::{Refusals, RefuseCode};
pub use reserved::Reserved;
pub use set_leaf::{ObsSet, SetId, SetLeaf};
pub use set_of::{ItemId, ObsSetOf, SetOf, SetOfLeaf};
pub use state_enum::StateEnum;
pub use state_init::StateInit;
pub use step_out::{EpisodeEnd, StepOut};
