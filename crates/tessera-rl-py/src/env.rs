// SPDX-License-Identifier: MIT OR Apache-2.0

//! The safe side of an environment object: one pool behind [`PoolOps`],
//! constructed from Python arguments. No
//! `unsafe` here; the C entry points live in `ffi`.

use core::num::{NonZeroU32, NonZeroUsize};

use crate::ffi::archive_type::AnyArchive;
use tessera_rl_classic::{
    Acrobot, AcrobotConfig, BrickWall, BrickWallConfig, CartPole, CartPoleConfig, Gather,
    GatherConfig, Grid, GridConfig, KeyDoor, KeyDoorConfig, MountainCar, MountainCarConfig,
    PaddleDuel, PaddleDuelConfig, Pendulum, PendulumConfig, Snake, SnakeConfig, Tag, TagConfig,
};
use tessera_rl_core::encoding::decode_from_slice;
use tessera_rl_core::space::{BoxBounds, DType, Space};
use tessera_rl_core::{DecodeError, Environment, TurnStructure, Validate};
use tessera_rl_engine::{
    ActionInput, AsyncOptions, AutoReset, DecodedRecord, EngineError, EnvPool, MAX_LEAVES,
    MaskProvider, Masked, NoMasks, PipelinedOptions, PoolOptions, Reseed, Schedule, SyncOptions,
    WaitBackend, WaitPolicy,
};
use tessera_rl_games::{
    Connect4, Connect4Config, Go9, Go9Config, Lanterns, LanternsConfig, Leduc, LeducConfig,
    Othello, OthelloConfig,
};
use tessera_rl_markets::{LimitOrderBook, LobConfig};
use tessera_rl_num::identity::Seed;
use tessera_rl_ops::{JobShop, JobShopConfig, Vrp, VrpConfig};
use tessera_rl_rts::{Rts, RtsConfig};

/// Errors of the bindings: the engine's, plus record decoding and argument
/// errors detected here.
#[derive(Debug)]
pub(crate) enum BindError {
    /// An engine error.
    Engine(EngineError),
    /// A record that does not decode.
    Decode(DecodeError),
    /// A persisted file refused, or an engine error around one.
    Persist(tessera_rl_engine::PersistError),
    /// An argument the bindings reject before reaching the engine.
    Argument(String),
    /// A pool that cannot be built in a forked child.
    Forked(String),
    /// A search error (ADR-0043), the evaluator's included.
    Search(tessera_rl_search::SearchError),
    /// The ONNX evaluator (ADR-0057): loading, model, contract, or a build
    /// without the `onnx` feature.
    Onnx(String),
    /// A replay file refused or a replay that diverged (ADR-0072).
    Replay(tessera_rl_engine::ReplayError),
    /// A call this pool does not support (`CapabilityError`), such as
    /// `render` without `Render` (ADR-0083).
    Capability(String),
}

impl From<tessera_rl_engine::ReplayError> for BindError {
    /// Engine errors keep their own class, so a recorded call raises what the
    /// same call raises unrecorded; a closed recording is a poisoned pool.
    fn from(e: tessera_rl_engine::ReplayError) -> Self {
        use tessera_rl_engine::ReplayError;
        match e {
            ReplayError::Engine(e) => Self::Engine(e),
            // The recorder closes only after it logged a call that poisoned
            // the pool, so a later call meets a poisoned pool: raise what that
            // call raises unrecorded (`PoisonedError`).
            ReplayError::RecorderClosed => Self::Engine(EngineError::Poisoned),
            ReplayError::Persist(e) => Self::Persist(e),
            e => Self::Replay(e),
        }
    }
}

/// The refusal of archive resets where they do not exist: host
/// environments (ADR-0075 item 8).
pub(crate) fn no_archives() -> BindError {
    BindError::Capability(
        "archive resets need a first-party environment (ADR-0075 item 8)".to_owned(),
    )
}

/// The engine reset source of `archive` (an `Archive<E>`, type-erased) and
/// `weights`, or fresh resets without an archive.
pub(crate) fn archive_source<E: Environment>(
    archive: Option<&AnyArchive>,
    weights: Option<Vec<u32>>,
) -> Result<tessera_rl_engine::ResetSource<E>, BindError> {
    match archive {
        None if weights.is_some() => Err(BindError::Argument("weights need an archive".to_owned())),
        None => Ok(tessera_rl_engine::ResetSource::Fresh),
        Some(any) => {
            let archive = any
                .downcast_ref::<tessera_rl_engine::Archive<E>>()
                .ok_or_else(|| {
                    BindError::Argument("this archive belongs to another environment".to_owned())
                })?;
            Ok(tessera_rl_engine::ResetSource::Archive {
                archive: archive.clone(),
                weights,
            })
        }
    }
}

/// The engine archive of `archive`, for replay, if it is one of `E`.
pub(crate) fn archive_ref<E: Environment>(
    archive: Option<&AnyArchive>,
) -> Result<Option<&tessera_rl_engine::Archive<E>>, BindError> {
    archive
        .map(|any| {
            any.downcast_ref::<tessera_rl_engine::Archive<E>>()
                .ok_or_else(|| {
                    BindError::Argument("this archive belongs to another environment".to_owned())
                })
        })
        .transpose()
}

/// Builds an `Archive<E>` from record bytes under `cfg`.
pub(crate) fn build_archive<E>(
    cfg: &E::Config,
    records: &[Vec<u8>],
) -> Result<(AnyArchive, u64, usize), BindError>
where
    E: tessera_rl_core::Persist,
    E::Config: tessera_rl_core::config::CanonicalConfig,
{
    let mut decoded = Vec::with_capacity(records.len());
    for bytes in records {
        decoded.push(decode_from_slice::<DecodedRecord<E>>(bytes).map_err(BindError::Decode)?);
    }
    let archive = tessera_rl_engine::Archive::<E>::from_decoded(cfg, &decoded)?;
    let (digest, len) = (archive.digest(), archive.len());
    Ok((std::sync::Arc::new(archive), digest, len))
}

/// The refusal of an operation a pool mode does not offer.
fn sync_only(what: &str) -> BindError {
    BindError::Argument(format!("{what} is available in sync mode only"))
}

impl From<tessera_rl_engine::PersistError> for BindError {
    fn from(e: tessera_rl_engine::PersistError) -> Self {
        Self::Persist(e)
    }
}

impl From<EngineError> for BindError {
    fn from(e: EngineError) -> Self {
        Self::Engine(e)
    }
}

/// One observation leaf, as Python sees it.
#[derive(Debug, Clone)]
pub(crate) struct Leaf {
    /// Leaf name (empty for a single unnamed box).
    pub(crate) name: String,
    /// NumPy dtype string (`"float32"`, `"uint8"`, ...).
    pub(crate) dtype: &'static str,
    /// Shape of one agent's observation.
    pub(crate) shape: Vec<u32>,
    /// Inclusive bounds per element, exact: integers for integer leaves,
    /// floats (infinities for unbounded elements) for float leaves.
    pub(crate) low: Vec<Bound>,
    pub(crate) high: Vec<Bound>,
}

/// One inclusive leaf bound, kept in the type the space declares it in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Bound {
    /// A signed integer bound.
    Int(i64),
    /// An unsigned integer bound.
    UInt(u64),
    /// A float bound (bits unchanged).
    Float(f64),
}

/// One observation `Set` (ADR-0082): its path, capacity and leaves (item
/// leaves, then the mask leaf).
pub(crate) type SetLayout = (String, u32, Vec<usize>);

/// How a reset chooses seeds.
#[derive(Debug, Clone)]
pub(crate) enum ResetSeed {
    /// Keep the seeds; the episode index advances.
    Keep,
    /// The k-th reset instance gets `base + k` (Gymnasium vector rule).
    Base(u64),
    /// The k-th reset instance gets `seeds[k]`; `None` keeps that
    /// instance's seed (Gymnasium's per-environment `None`).
    List(Vec<Option<u64>>),
}

/// Discrete actions as an engine input (unit tests).
#[cfg(test)]
pub(crate) fn discrete(actions: &[i32]) -> ActionInput<'_> {
    ActionInput {
        discrete: actions,
        continuous: &[],
    }
}

/// The fixed layout of a pool's outputs.
#[derive(Debug, Clone)]
pub(crate) struct Layout {
    pub(crate) num_envs: usize,
    pub(crate) agents: usize,
    /// `Environment::initial_agents(cfg)`: the initial population, not a
    /// bound on which slots can ever act (`agents` is the capacity).
    pub(crate) initial_agents: usize,
    pub(crate) leaves: Vec<Leaf>,
    /// The observation's sets (ADR-0082): `(path, max, leaf indices)`, the
    /// indices of the set's item leaves then of its mask leaf, in leaf
    /// order.
    pub(crate) sets: Vec<SetLayout>,
    /// Choices of the one discrete head; 0 for a continuous or a
    /// multi-head environment.
    pub(crate) action_choices: u32,
    /// Choices per discrete head of one agent (SPEC 11.1): `[n]` for
    /// `Discrete(n)`, `nvec` for `MultiDiscrete(nvec)`, empty for a
    /// continuous environment.
    pub(crate) action_nvec: Vec<u32>,
    /// Inclusive bounds of the one continuous `f32` head, if that is the
    /// action space.
    pub(crate) action_bounds: Option<(f64, f64)>,
    pub(crate) mask_len: usize,
    /// Each discrete head's block in an agent's mask row (SPEC 11.3):
    /// `(offset, parent sizes, choices)`; empty without masks.
    pub(crate) mask_heads: Vec<(usize, Vec<usize>, usize)>,
    /// Discrete heads per entity group (ADR-0051); 0 when none.
    pub(crate) entity_group_heads: usize,
    /// Turn structure (ADR-0013): `"simultaneous"`, `"sequential"` or
    /// `"mixed"`.
    pub(crate) turns: &'static str,
    /// Terminal observations are written (`same_step` auto-reset).
    pub(crate) terminal: bool,
    /// The policy group of each agent slot (ADR-0079 item 5.1):
    /// `"learner"`, `"external"`, `"heuristic"` or `"privileged_heuristic"`.
    pub(crate) policy_groups: Vec<&'static str>,
    /// The privileged state leaves (ADR-0079 item 7), one row per
    /// instance; empty without `privileged_state=True`.
    pub(crate) state_leaves: Vec<Leaf>,
}

/// Copies of one call's outputs (`owned` mode).
#[derive(Debug, Clone, Default)]
pub(crate) struct Owned {
    /// One buffer per leaf, `[N, A, leaf]`.
    pub(crate) observations: Vec<Vec<u8>>,
    /// `[N, A]`, little-endian f32.
    pub(crate) rewards: Vec<u8>,
    /// `[N, A]`, one byte per flag.
    pub(crate) agent_terminated: Vec<u8>,
    pub(crate) agent_truncated: Vec<u8>,
    pub(crate) alive: Vec<u8>,
    /// Agents whose action this step reads (ADR-0013).
    pub(crate) acting: Vec<u8>,
    /// `[N, A]`, little-endian u32: each slot's occupant generation
    /// (ADR-0013 identity).
    pub(crate) slot_generations: Vec<u8>,
    /// `[N]`.
    pub(crate) episode_terminated: Vec<u8>,
    pub(crate) episode_truncated: Vec<u8>,
    /// `[N, A, mask length]`; empty without masks.
    pub(crate) masks: Vec<u8>,
    /// One buffer per leaf in `same_step` mode, `[N, A, leaf]`: final
    /// observations of episodes that ended in the call (ADR-0039); rows of
    /// other instances are stale. Empty in other modes.
    pub(crate) terminal_observations: Vec<Vec<u8>>,
    /// One buffer per privileged state leaf, `[N, leaf]` (ADR-0079 item 7);
    /// empty without the opt-in.
    pub(crate) state: Vec<Vec<u8>>,
}

/// What an environment object can do, whatever the environment.
pub(crate) trait PoolOps: Send {
    fn layout(&self) -> &Layout;
    /// The view set of `layout`, built once at construction (the layout is
    /// fixed); `None` for a leaf dtype the protocols cannot name. It holds
    /// only regions and shapes: each call still makes new views, stamped
    /// with that call's storage generation. `None` by default: Pipelined
    /// mode exports only halves, through [`PoolOps::half_views`].
    fn views(&self) -> Option<&std::sync::Arc<crate::view::ViewSet>> {
        None
    }
    /// The view set of half `half`'s rows, built once (Pipelined only).
    fn half_views(&self, _half: usize) -> Option<std::sync::Arc<crate::view::ViewSet>> {
        None
    }
    /// Resets every instance, or only `ids`.
    fn reset(&mut self, ids: Option<&[u32]>, seed: &ResetSeed) -> Result<(), BindError>;
    fn step(&mut self, actions: ActionInput<'_>) -> Result<(), BindError>;
    /// The last call's outputs; refused on a poisoned pool.
    fn outputs(&self) -> Result<Owned, BindError>;
    /// Copies the last call's outputs into `dst` without allocating, for
    /// the XLA FFI path (ADR-0095 point 5). `dst` holds one buffer per
    /// observation leaf, then rewards (`f32` little-endian), agent
    /// terminated, agent truncated, alive, acting (one byte per flag),
    /// slot generations (`u32` little-endian), episode terminated, episode
    /// truncated and, with masks, the masks; each exactly as long as the
    /// matching [`Owned`] field. The copy happens inside the caller's
    /// admitted section, so within the outputs' borrow window. Sync mode
    /// only: the split-phase modes refuse it. A pool with privileged state
    /// (ADR-0079) is refused too: the XLA FFI path exports no privileged
    /// leaves.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "the XLA FFI handler of #369 head B calls it")
    )]
    fn outputs_into(&self, _dst: &mut [&mut [u8]]) -> Result<(), BindError> {
        Err(BindError::Argument(
            "outputs_into is available in sync mode only".to_owned(),
        ))
    }
    /// The shared output storage, for borrowed views.
    fn storage(&self) -> std::sync::Arc<tessera_rl_engine::OutputStorage>;
    /// The storage for new borrowed views: a view made on a poisoned pool would
    /// take the current generation and read rows a panic left partly written.
    fn readable_storage(
        &self,
    ) -> Result<std::sync::Arc<tessera_rl_engine::OutputStorage>, BindError> {
        if self.is_poisoned() {
            return Err(BindError::Engine(EngineError::Poisoned));
        }
        Ok(self.storage())
    }
    fn snapshot(&self, id: u32) -> Result<Vec<u8>, BindError>;
    /// The canvas `(width, height)` of a pool that renders (ADR-0083).
    fn canvas(&self) -> Option<(u16, u16)> {
        None
    }
    /// The frames of `ids`, `len(ids) * height * width * 3` RGB8 bytes, and
    /// whether each scene overflowed. Each frame is drawn from the
    /// instance's snapshot record, on the caller's thread.
    fn render(&self, _ids: &[u32]) -> Result<(Vec<u8>, Vec<bool>), BindError> {
        Err(crate::render_api::not_renderable())
    }
    /// The scene of instance `id` and whether it overflowed.
    fn scene(&self, _id: u32) -> Result<(Vec<tessera_rl_core::render::Prim>, bool), BindError> {
        Err(crate::render_api::not_renderable())
    }
    /// The epoch barrier of the pool's normalization (ADR-0078 item 5.3):
    /// sync mode only.
    fn epoch_barrier(&mut self) -> Result<(), BindError> {
        Err(transforms_sync_only())
    }
    /// Freezes or unfreezes normalization (ADR-0078 item 5.6): sync mode
    /// only.
    fn freeze_normalization(&mut self, _frozen: bool) -> Result<(), BindError> {
        Err(transforms_sync_only())
    }
    /// Illegal-but-valid actions replaced in instance `id`'s current
    /// episode, per agent slot (the record header's statistic). Sync mode
    /// only; the split-phase modes refuse it.
    fn illegal_actions(&self, _id: u32) -> Result<Vec<u32>, BindError> {
        Err(BindError::Argument(
            "illegal_actions is available in sync mode only".to_owned(),
        ))
    }
    /// Instance `id`'s state hash, the one Rust's goldens fold (sync mode).
    fn state_hash(&self, _id: u32) -> Result<u64, BindError> {
        Err(BindError::Argument(
            "state_hash is available in sync mode only".to_owned(),
        ))
    }
    /// The last step's heuristic actions (ADR-0079 item 3.5), `[N, A,
    /// discrete heads]` little-endian `i32`, copied; `None` without a
    /// heuristic slot. The split-phase handles read their wrapper's rows
    /// (#347 stage B1).
    fn heuristic_actions(&self) -> Result<Option<Vec<u8>>, BindError> {
        Ok(None)
    }
    fn import(&mut self, id: u32, bytes: &[u8]) -> Result<(), BindError>;
    /// The persisted file of `ids` (ADR-0011).
    fn encode(&self, ids: &[u32]) -> Result<Vec<u8>, BindError>;
    /// Loads a persisted file into `ids`, or into every instance.
    fn load(
        &mut self,
        bytes: &[u8],
        ids: Option<&[u32]>,
        limits: tessera_rl_engine::DecodeLimits,
    ) -> Result<(), BindError>;
    fn fork(&mut self, src: u32, dst: &[u32]) -> Result<(), BindError>;
    fn is_poisoned(&self) -> bool;
    /// Attaches a live watch (ADR-0083 item 7) and returns its single
    /// consumer handle. Host pools refuse it (`CapabilityError`).
    fn watch(&mut self) -> Result<tessera_rl_engine::WatchHandle, BindError> {
        Err(BindError::Capability(
            "watch is not available for this environment".to_owned(),
        ))
    }
    /// The split-phase operations, in Pipelined mode.
    fn split(&self) -> Option<std::sync::Arc<dyn crate::split::SplitOps>> {
        None
    }
    /// The Async operations, in Async mode.
    fn batched(&self) -> Option<std::sync::Arc<dyn crate::batched::AsyncOps>> {
        None
    }
    /// Starts recording a replay file (ADR-0072) in memory: from now on
    /// `reset`, `step` and `fork` are logged. Sync mode only.
    fn record_start(
        &mut self,
        _keyframe_every: u64,
        _checkpoint_every: u64,
    ) -> Result<(), BindError> {
        Err(sync_only("record"))
    }
    /// Ends the recording and returns the file.
    fn record_finish(&mut self) -> Result<Vec<u8>, BindError> {
        Err(sync_only("record"))
    }
    /// Drops the recording, if any; the pool keeps its state.
    fn record_abort(&mut self) {}
    /// Builds an archive of this environment from record bytes (ADR-0075),
    /// with its digest and entry count.
    fn make_archive(&self, _records: &[Vec<u8>]) -> Result<(AnyArchive, u64, usize), BindError> {
        Err(no_archives())
    }
    /// Sets what resets start from (ADR-0075 item 2): an archive of this
    /// environment with optional weights, or fresh resets (`None`).
    fn set_reset_source(
        &mut self,
        _archive: Option<&AnyArchive>,
        _weights: Option<Vec<u32>>,
    ) -> Result<(), BindError> {
        Err(no_archives())
    }
    /// Resets `ids` from the archive entries `entries` (ADR-0075 item 5).
    fn reset_from(&mut self, _ids: &[u32], _entries: &[u32]) -> Result<(), BindError> {
        Err(no_archives())
    }
    /// A whole-pool snapshot of a Sync or Inline pool: records, frame stacks
    /// and the pool context (#465), opaque.
    fn snapshot_pool(&self) -> Result<crate::split::Snapshot, BindError> {
        Err(no_archives())
    }
    /// Restores a [`PoolOps::snapshot_pool`] snapshot of this pool, with the
    /// archive its context names (ADR-0075 item 7); an error mutates nothing.
    fn restore_pool(
        &mut self,
        _snapshot: &crate::split::Snapshot,
        _archive: Option<&AnyArchive>,
    ) -> Result<(), BindError> {
        Err(no_archives())
    }
    /// Replays a file into this pool and checks it (ADR-0072 section 5),
    /// with the archive the file names (ADR-0075 item 7), if any.
    fn replay(
        &mut self,
        _data: &[u8],
        _archive: Option<&AnyArchive>,
    ) -> Result<tessera_rl_engine::ReplaySummary, BindError> {
        Err(sync_only("replay"))
    }
    /// Starts stepping through a replay file (issue #378): the pool is
    /// checked against the header and a cursor is kept, with a copy of the
    /// file, until [`PoolOps::replay_finish`]. Calls that change the pool
    /// are refused meanwhile.
    fn replay_begin(
        &mut self,
        _data: &[u8],
        _archive: Option<&AnyArchive>,
    ) -> Result<(), BindError> {
        Err(sync_only("replay"))
    }
    /// Applies events until `calls` more calls (resets, steps, state writes)
    /// were replayed: the index of the last event applied, or `None` at the
    /// end of the file.
    fn replay_advance(&mut self, _calls: u64) -> Result<Option<u64>, BindError> {
        Err(sync_only("replay"))
    }
    /// Imports the keyframe at event `keyframe` and moves the cursor after it.
    fn replay_seek(&mut self, _keyframe: u64) -> Result<(), BindError> {
        Err(sync_only("replay"))
    }
    /// The event indices of the keyframes `replay_seek` accepts.
    fn replay_keyframes(&self) -> Result<Vec<u64>, BindError> {
        Err(sync_only("replay"))
    }
    /// Ends the stepped replay and drops the cursor: a verified summary if
    /// every event was applied, else one with `stopped_at` set.
    fn replay_finish(&mut self) -> Result<tessera_rl_engine::ReplaySummary, BindError> {
        Err(sync_only("replay"))
    }
}

/// A pool with its active stepped replay: the pool, the file's bytes and
/// the cursor.
type ActiveCursor<'a, E, M> = (
    &'a mut EnvPool<E, M>,
    &'a [u8],
    &'a mut tessera_rl_engine::ReplayCursor,
);

/// A Sync or Inline pool snapshot (#465): the pool that took it, by its
/// output storage, and the engine's whole-pool state.
struct SyncTaken<E: Environment> {
    owner: std::sync::Arc<tessera_rl_engine::OutputStorage>,
    state: tessera_rl_engine::PoolState<E>,
}

struct Pooled<E: Environment, M: MaskProvider<E>> {
    pool: EnvPool<E, M>,
    /// The recording in progress, if any (ADR-0072): held without its pool
    /// borrow between calls.
    rec: Option<tessera_rl_engine::replay::DetachedRecorder<E, M, Vec<u8>>>,
    layout: Layout,
    views: Option<std::sync::Arc<crate::view::ViewSet>>,
    /// Captured while healthy: identity and regions for views, no row values.
    storage: std::sync::Arc<tessera_rl_engine::OutputStorage>,
    /// The configuration and render functions, for an environment that
    /// implements `Render` (ADR-0083).
    render: Option<(E::Config, crate::render_api::RenderFns<E>)>,
    /// The stepped replay in progress, if any (issue #378): the file bytes,
    /// owned, and the cursor. Each call reopens the bytes with
    /// `Replay::reopen`, which parses the header but does not rescan the
    /// event frames (the cursor holds their offsets), so an advance costs
    /// the header plus the events it applies, never the whole file.
    cursor: Option<(Vec<u8>, tessera_rl_engine::ReplayCursor)>,
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::I8 => "int8",
        DType::I16 => "int16",
        DType::I32 => "int32",
        DType::I64 => "int64",
        DType::U8 => "uint8",
        DType::U16 => "uint16",
        DType::U32 => "uint32",
        DType::F32 => "float32",
        DType::F64 => "float64",
    }
}

/// The bounds of a box, each in its declared type (no casts).
fn exact_bounds(bounds: &BoxBounds) -> (Vec<Bound>, Vec<Bound>) {
    match bounds {
        BoxBounds::Int { low, high } => (
            low.iter().copied().map(Bound::Int).collect(),
            high.iter().copied().map(Bound::Int).collect(),
        ),
        BoxBounds::UInt { low, high } => (
            low.iter().copied().map(Bound::UInt).collect(),
            high.iter().copied().map(Bound::UInt).collect(),
        ),
        BoxBounds::Float { low, high } => (
            low.iter().copied().map(Bound::Float).collect(),
            high.iter().copied().map(Bound::Float).collect(),
        ),
    }
}

fn leaves(
    space: &Space,
    name: &str,
    out: &mut Vec<Leaf>,
    sets: &mut Vec<SetLayout>,
) -> Result<(), BindError> {
    match space {
        Space::Box {
            shape,
            dtype,
            bounds,
        } => {
            let (low, high) = exact_bounds(bounds);
            out.push(Leaf {
                name: name.to_owned(),
                dtype: dtype_name(*dtype),
                shape: shape.clone(),
                low,
                high,
            });
        }
        Space::Dict(entries) => {
            for (key, inner) in entries {
                leaves(inner, key, out, sets)?;
            }
        }
        Space::Set { inner, max } => set_leaves(inner, *max, name, out, sets)?,
        _ => {
            return Err(BindError::Argument(
                "observation spaces other than boxes, dicts and sets are not exposed yet"
                    .to_owned(),
            ));
        }
    }
    Ok(())
}

/// A set `name` (ADR-0082 point 2): its item leaves at `name` or
/// `name/<key>`, shape `[max, ...]` with the bounds broadcast, then
/// `name_mask`, `bool` `[max]`, bounds 0..1.
fn set_leaves(
    inner: &Space,
    max: u32,
    name: &str,
    out: &mut Vec<Leaf>,
    sets: &mut Vec<SetLayout>,
) -> Result<(), BindError> {
    let mut items = Vec::new();
    let mut nested = Vec::new();
    leaves(inner, "", &mut items, &mut nested)?;
    if !nested.is_empty() {
        return Err(BindError::Argument("a set inside a set".to_owned()));
    }
    let copies = usize::try_from(max).unwrap_or(usize::MAX);
    let broadcast = |v: Vec<Bound>| if v.len() == 1 { v } else { v.repeat(copies) };
    let first = out.len();
    for item in items {
        let mut shape = vec![max];
        shape.extend(item.shape);
        out.push(Leaf {
            name: if item.name.is_empty() {
                name.to_owned()
            } else {
                format!("{name}/{}", item.name)
            },
            dtype: item.dtype,
            shape,
            low: broadcast(item.low),
            high: broadcast(item.high),
        });
    }
    out.push(Leaf {
        name: format!("{name}_mask"),
        dtype: "bool",
        shape: vec![max],
        low: vec![Bound::UInt(0)],
        high: vec![Bound::UInt(1)],
    });
    sets.push((name.to_owned(), max, (first..out.len()).collect()));
    Ok(())
}

impl<E, M> Pooled<E, M>
where
    E: Environment + Validate,
    M: MaskProvider<E>,
{
    fn new(
        cfg: &E::Config,
        pool: EnvPool<E, M>,
        num_envs: usize,
        terminal: bool,
        storage: std::sync::Arc<tessera_rl_engine::OutputStorage>,
    ) -> Result<Self, BindError> {
        let mut obs = Vec::new();
        let mut sets = Vec::new();
        // The exported observation: transformed when the pool has
        // transforms (ADR-0078 item 2.2).
        leaves(&pool.observation_space()?, "", &mut obs, &mut sets)?;
        let (action_choices, action_bounds, action_nvec) = action_head(
            &E::action_space(cfg).map_err(EngineError::from)?,
            E::MAX_AGENTS,
        )?;
        // Each head's block in an agent's mask row, as the engine lays them
        // out (SPEC 11.3); none without masks.
        let mut mask_heads: Vec<(usize, Vec<usize>, usize)> = Vec::new();
        if M::ENABLED {
            for h in 0..action_nvec.len() {
                // With masks, the engine lays out every discrete head.
                let m = pool.mask_head_shape(h).ok_or_else(|| {
                    BindError::Argument(format!("discrete head {h} has no mask block"))
                })?;
                mask_heads.push((
                    m.offset,
                    m.parent_sizes.iter().flatten().copied().collect(),
                    m.size,
                ));
            }
        }
        // One agent's mask row: the end of its last head block.
        let mask_len = mask_heads
            .iter()
            .map(|(offset, parents, size)| offset + parents.iter().product::<usize>() * size)
            .max()
            .unwrap_or(0);
        let entity_group_heads = if M::ENABLED {
            M::entity_group_heads(cfg).map_or(0, |g| usize::from(g.get()))
        } else {
            0
        };
        // The privileged state leaves, named as observation leaves are.
        let mut state_leaves = Vec::new();
        if let Some(space) = pool.privileged_state_space() {
            leaves(space, "", &mut state_leaves, &mut Vec::new())?;
        }
        let layout = Layout {
            num_envs,
            agents: E::MAX_AGENTS,
            initial_agents: E::initial_agents(cfg),
            leaves: obs,
            sets,
            action_choices,
            action_nvec,
            action_bounds,
            mask_len,
            mask_heads,
            entity_group_heads,
            turns: match E::TURNS {
                TurnStructure::Simultaneous => "simultaneous",
                TurnStructure::Sequential => "sequential",
                TurnStructure::Mixed => "mixed",
            },
            terminal,
            policy_groups: pool.policy_groups().iter().map(|g| g.name()).collect(),
            state_leaves,
        };
        Ok(Self {
            storage,
            pool,
            rec: None,
            cursor: None,
            views: crate::view::view_set(&layout).map(std::sync::Arc::new),
            layout,
            render: None,
        })
    }
}

fn flags(v: &[tessera_rl_core::Bool8]) -> Vec<u8> {
    v.iter().map(|f| f.to_byte()).collect()
}

/// Little-endian bytes of `v`.
pub(crate) fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl<E, M> Pooled<E, M>
where
    E: tessera_rl_core::Persist,
    E::Config: tessera_rl_core::config::CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
    M: tessera_rl_engine::PersistMask<E>,
{
    /// A reset with one optional seed per reset instance (`None` keeps
    /// that instance's seed), as one engine call: validated as a whole
    /// before anything is written (ADR-0028). A full reset goes through
    /// `reset_all`, which recovers a poisoned pool.
    fn reset_list(&mut self, ids: Option<&[u32]>, seeds: &[Option<u64>]) -> Result<(), BindError> {
        let seeds: Vec<Option<Seed>> = seeds.iter().map(|s| s.map(Seed::new)).collect();
        self.reset_with(ids, Reseed::Each(&seeds))
    }
}

impl<E, M> Pooled<E, M>
where
    E: tessera_rl_core::Persist,
    E::Config: tessera_rl_core::config::CanonicalConfig + Sync,
    E::Header: Send,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
    M: tessera_rl_engine::PersistMask<E>,
{
    /// One reset, through the recorder while recording.
    fn reset_with(&mut self, ids: Option<&[u32]>, reseed: Reseed<'_>) -> Result<(), BindError> {
        if let Some(detached) = self.rec.take() {
            let mut rec = detached.attach(&mut self.pool);
            let result = match ids {
                None => rec.reset_all(reseed).map(|_| ()),
                Some(ids) => rec.reset(ids, reseed).map(|_| ()),
            };
            self.rec = Some(rec.detach());
            return result.map_err(BindError::from);
        }
        match ids {
            None => self.pool.reset_all(reseed)?,
            Some(ids) => self.pool.reset(ids, reseed)?,
        };
        Ok(())
    }

    /// Refuses an unlogged state write while recording.
    fn not_recording(&self, what: &str) -> Result<(), BindError> {
        if self.rec.is_some() {
            return Err(BindError::Argument(format!(
                "{what} is not recorded: finish the recording first"
            )));
        }
        Ok(())
    }

    /// Refuses a call that changes the pool while a stepped replay holds it:
    /// the replay would no longer check what the file recorded.
    fn not_replaying(&self, what: &str) -> Result<(), BindError> {
        if self.cursor.is_some() {
            return Err(BindError::Argument(format!(
                "{what} is refused while a replay is in progress: finish the replay first"
            )));
        }
        Ok(())
    }

    /// The active cursor, or the refusal of `what` without one.
    fn active(&mut self, what: &str) -> Result<ActiveCursor<'_, E, M>, BindError> {
        let Self { pool, cursor, .. } = self;
        let (bytes, cur) = cursor
            .as_mut()
            .ok_or_else(|| BindError::Argument(format!("{what}: no replay in progress")))?;
        Ok((pool, bytes.as_slice(), cur))
    }
}

impl<E, M> PoolOps for Pooled<E, M>
where
    E: tessera_rl_core::Persist,
    E::Config: tessera_rl_core::config::CanonicalConfig + Send + Sync,
    E::Header: Send + Sync,
    E::State: Send,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
    M: tessera_rl_engine::PersistMask<E> + Send,
{
    fn layout(&self) -> &Layout {
        &self.layout
    }

    fn views(&self) -> Option<&std::sync::Arc<crate::view::ViewSet>> {
        self.views.as_ref()
    }

    fn reset(&mut self, ids: Option<&[u32]>, seed: &ResetSeed) -> Result<(), BindError> {
        self.not_replaying("reset")?;
        let reseed = match seed {
            ResetSeed::Keep => Reseed::Keep,
            ResetSeed::Base(s) => Reseed::Base(Seed::new(*s)),
            ResetSeed::List(seeds) => return self.reset_list(ids, seeds),
        };
        self.reset_with(ids, reseed)
    }

    fn step(&mut self, actions: ActionInput<'_>) -> Result<(), BindError> {
        self.not_replaying("step")?;
        if let Some(detached) = self.rec.take() {
            let mut rec = detached.attach(&mut self.pool);
            let result = rec.step(actions).map(|_| ());
            self.rec = Some(rec.detach());
            return result.map_err(BindError::from);
        }
        self.pool.step(actions)?;
        Ok(())
    }

    fn outputs(&self) -> Result<Owned, BindError> {
        let out = self.pool.outputs()?;
        let observations = (0..MAX_LEAVES)
            .map_while(|leaf| out.observation(leaf).map(<[u8]>::to_vec))
            .collect();
        let terminal_observations = if self.layout.terminal {
            (0..MAX_LEAVES)
                .map_while(|leaf| out.terminal_observation(leaf).map(<[u8]>::to_vec))
                .collect()
        } else {
            Vec::new()
        };
        Ok(Owned {
            terminal_observations,
            observations,
            rewards: out.rewards().iter().flat_map(|r| r.to_le_bytes()).collect(),
            agent_terminated: flags(out.agent_terminated()),
            agent_truncated: flags(out.agent_truncated()),
            alive: flags(out.alive()),
            acting: flags(out.acting()),
            slot_generations: u32_bytes(out.generations()),
            episode_terminated: flags(out.episode_terminated()),
            episode_truncated: flags(out.episode_truncated()),
            masks: flags(out.masks()),
            state: (0..MAX_LEAVES)
                .map_while(|leaf| out.privileged_state(leaf).map(<[u8]>::to_vec))
                .collect(),
        })
    }

    fn outputs_into(&self, dst: &mut [&mut [u8]]) -> Result<(), BindError> {
        let out = self.pool.outputs()?;
        if out.privileged_state(0).is_some() {
            return Err(BindError::Argument(
                "outputs_into does not export privileged state".to_owned(),
            ));
        }
        let leaves = (0..MAX_LEAVES)
            .map_while(|leaf| out.observation(leaf))
            .count();
        let masks = out.masks();
        let want = leaves + 8 + usize::from(!masks.is_empty());
        if dst.len() != want {
            return Err(BindError::Argument(format!(
                "outputs_into needs {want} buffers, got {}",
                dst.len()
            )));
        }
        let mut dst = dst.iter_mut();
        let mut next = |len: usize, what: &str| -> Result<&mut [u8], BindError> {
            let buf = dst.next().map(|b| &mut **b).unwrap_or_default();
            if buf.len() == len {
                Ok(buf)
            } else {
                Err(BindError::Argument(format!(
                    "outputs_into: {what} needs {len} bytes, got {}",
                    buf.len()
                )))
            }
        };
        for leaf in 0..leaves {
            let src = out.observation(leaf).unwrap_or_default();
            next(src.len(), "an observation leaf")?.copy_from_slice(src);
        }
        let rewards = out.rewards();
        let buf = next(rewards.len() * 4, "rewards")?;
        for (d, r) in buf.as_chunks_mut::<4>().0.iter_mut().zip(rewards) {
            *d = r.to_le_bytes();
        }
        let flags_into = |buf: &mut [u8], src: &[tessera_rl_core::Bool8]| {
            for (d, f) in buf.iter_mut().zip(src) {
                *d = f.to_byte();
            }
        };
        flags_into(
            next(out.agent_terminated().len(), "agent_terminated")?,
            out.agent_terminated(),
        );
        flags_into(
            next(out.agent_truncated().len(), "agent_truncated")?,
            out.agent_truncated(),
        );
        flags_into(next(out.alive().len(), "alive")?, out.alive());
        flags_into(next(out.acting().len(), "acting")?, out.acting());
        let generations = out.generations();
        let buf = next(generations.len() * 4, "slot_generations")?;
        for (d, g) in buf.as_chunks_mut::<4>().0.iter_mut().zip(generations) {
            *d = g.to_le_bytes();
        }
        flags_into(
            next(out.episode_terminated().len(), "episode_terminated")?,
            out.episode_terminated(),
        );
        flags_into(
            next(out.episode_truncated().len(), "episode_truncated")?,
            out.episode_truncated(),
        );
        if !masks.is_empty() {
            flags_into(next(masks.len(), "masks")?, masks);
        }
        Ok(())
    }

    fn storage(&self) -> std::sync::Arc<tessera_rl_engine::OutputStorage> {
        std::sync::Arc::clone(&self.storage)
    }

    fn snapshot(&self, id: u32) -> Result<Vec<u8>, BindError> {
        Ok(self.pool.snapshot(id)?.to_bytes())
    }

    fn canvas(&self) -> Option<(u16, u16)> {
        self.render.as_ref().map(|(_, fns)| fns.canvas)
    }

    fn epoch_barrier(&mut self) -> Result<(), BindError> {
        // The report reaches Python in #359 head D (ADR-0081 item 1.3).
        self.pool.epoch_barrier()?;
        Ok(())
    }

    fn freeze_normalization(&mut self, frozen: bool) -> Result<(), BindError> {
        self.pool.freeze_normalization(frozen);
        Ok(())
    }

    fn render(&self, ids: &[u32]) -> Result<(Vec<u8>, Vec<bool>), BindError> {
        let (cfg, fns) = self
            .render
            .as_ref()
            .ok_or_else(crate::render_api::not_renderable)?;
        let (w, h) = fns.canvas;
        let len = usize::from(w) * usize::from(h) * 3;
        let total = len
            .checked_mul(ids.len())
            .ok_or_else(|| BindError::Argument("too many frames".to_owned()))?;
        let mut out = vec![0; total];
        let mut overflowed = Vec::with_capacity(ids.len());
        for (&id, frame) in ids.iter().zip(out.chunks_exact_mut(len)) {
            let record = self.pool.snapshot(id)?;
            overflowed.push((fns.frame)(cfg, record.state(), frame)?);
        }
        Ok((out, overflowed))
    }

    fn scene(&self, id: u32) -> Result<(Vec<tessera_rl_core::render::Prim>, bool), BindError> {
        let (cfg, fns) = self
            .render
            .as_ref()
            .ok_or_else(crate::render_api::not_renderable)?;
        let record = self.pool.snapshot(id)?;
        (fns.scene)(cfg, record.state())
    }

    fn state_hash(&self, id: u32) -> Result<u64, BindError> {
        Ok(self.pool.state_hash(id)?.get())
    }

    fn heuristic_actions(&self) -> Result<Option<Vec<u8>>, BindError> {
        if self.pool.is_poisoned() {
            return Err(EngineError::Poisoned.into());
        }
        Ok(self
            .pool
            .heuristic_actions()
            .map(|rows| rows.iter().flat_map(|v| v.to_le_bytes()).collect()))
    }

    fn illegal_actions(&self, id: u32) -> Result<Vec<u32>, BindError> {
        use tessera_rl_core::RecordLayout as _;
        let record = self.pool.snapshot(id)?;
        Ok(record
            .header()
            .agents()
            .iter()
            .map(|a| a.illegal_actions)
            .collect())
    }

    fn import(&mut self, id: u32, bytes: &[u8]) -> Result<(), BindError> {
        self.not_recording("import_record")?;
        self.not_replaying("import_record")?;
        let record = decode_from_slice::<DecodedRecord<E>>(bytes).map_err(BindError::Decode)?;
        self.pool.import(&[id], &[record])?;
        Ok(())
    }

    fn encode(&self, ids: &[u32]) -> Result<Vec<u8>, BindError> {
        Ok(self.pool.encode(ids)?)
    }

    fn load(
        &mut self,
        bytes: &[u8],
        ids: Option<&[u32]>,
        limits: tessera_rl_engine::DecodeLimits,
    ) -> Result<(), BindError> {
        self.not_recording("load")?;
        self.not_replaying("load")?;
        let records = self.pool.decode(bytes, limits)?;
        match ids {
            Some(ids) => self.pool.import(ids, &records)?,
            // The recovery path on a poisoned pool.
            None => self.pool.import_all(&records)?,
        }
        Ok(())
    }

    fn fork(&mut self, src: u32, dst: &[u32]) -> Result<(), BindError> {
        self.not_replaying("fork")?;
        if let Some(detached) = self.rec.take() {
            let mut rec = detached.attach(&mut self.pool);
            let result = rec.fork(src, dst);
            self.rec = Some(rec.detach());
            return result.map_err(BindError::from);
        }
        self.pool.fork(src, dst)?;
        Ok(())
    }

    fn is_poisoned(&self) -> bool {
        self.pool.is_poisoned()
    }

    fn watch(&mut self) -> Result<tessera_rl_engine::WatchHandle, BindError> {
        Ok(self.pool.attach_watch()?)
    }

    fn record_start(
        &mut self,
        keyframe_every: u64,
        checkpoint_every: u64,
    ) -> Result<(), BindError> {
        self.not_replaying("record")?;
        if self.rec.is_some() {
            return Err(BindError::Argument("already recording".to_owned()));
        }
        let options = tessera_rl_engine::ReplayWriterOptions {
            keyframe_every,
            checkpoint_every,
            outputs_hash: false,
        };
        let rec = tessera_rl_engine::Recorder::new(&mut self.pool, Vec::new(), options)?;
        self.rec = Some(rec.detach());
        Ok(())
    }

    fn record_finish(&mut self) -> Result<Vec<u8>, BindError> {
        let detached = self
            .rec
            .take()
            .ok_or_else(|| BindError::Argument("not recording".to_owned()))?;
        Ok(detached.attach(&mut self.pool).finish()?)
    }

    fn record_abort(&mut self) {
        self.rec = None;
    }

    fn make_archive(&self, records: &[Vec<u8>]) -> Result<(AnyArchive, u64, usize), BindError> {
        build_archive::<E>(self.pool.config(), records)
    }

    fn set_reset_source(
        &mut self,
        archive: Option<&AnyArchive>,
        weights: Option<Vec<u32>>,
    ) -> Result<(), BindError> {
        self.not_replaying("set_reset_source")?;
        let source = archive_source::<E>(archive, weights)?;
        // Logged while recording (ADR-0075 item 7, tag 10).
        if let Some(detached) = self.rec.take() {
            let mut rec = detached.attach(&mut self.pool);
            let result = rec.set_reset_source(source);
            self.rec = Some(rec.detach());
            return result.map_err(BindError::from);
        }
        self.pool.set_reset_source(source)?;
        Ok(())
    }

    fn snapshot_pool(&self) -> Result<crate::split::Snapshot, BindError> {
        let state = self.pool.snapshot_pool()?;
        Ok(std::sync::Arc::new(SyncTaken::<E> {
            owner: std::sync::Arc::clone(&self.storage),
            state,
        }))
    }

    fn restore_pool(
        &mut self,
        snapshot: &crate::split::Snapshot,
        archive: Option<&AnyArchive>,
    ) -> Result<(), BindError> {
        self.not_recording("restore_pool")?;
        self.not_replaying("restore_pool")?;
        let taken = snapshot
            .downcast_ref::<SyncTaken<E>>()
            .filter(|t| std::sync::Arc::ptr_eq(&t.owner, &self.storage))
            .ok_or_else(|| {
                BindError::Argument("a snapshot restores only the pool that took it".to_owned())
            })?;
        let archive = archive_ref::<E>(archive)?;
        self.pool.restore_pool(&taken.state, archive)?;
        Ok(())
    }

    fn reset_from(&mut self, ids: &[u32], entries: &[u32]) -> Result<(), BindError> {
        self.not_replaying("reset_from")?;
        // Logged while recording (ADR-0075 item 5, tag 11).
        if let Some(detached) = self.rec.take() {
            let mut rec = detached.attach(&mut self.pool);
            let result = rec.reset_from(ids, entries).map(|_| ());
            self.rec = Some(rec.detach());
            return result.map_err(BindError::from);
        }
        self.pool.reset_from(ids, entries)?;
        Ok(())
    }

    fn replay(
        &mut self,
        data: &[u8],
        archive: Option<&AnyArchive>,
    ) -> Result<tessera_rl_engine::ReplaySummary, BindError> {
        self.not_recording("replay")?;
        self.not_replaying("replay")?;
        let archive = archive_ref::<E>(archive)?;
        let replay =
            tessera_rl_engine::Replay::open(data, tessera_rl_engine::DecodeLimits::default())
                .map_err(BindError::from)?;
        replay
            .run_with_archive(&mut self.pool, archive)
            .map_err(BindError::from)
    }

    fn replay_begin(&mut self, data: &[u8], archive: Option<&AnyArchive>) -> Result<(), BindError> {
        self.not_recording("replay")?;
        self.not_replaying("replay")?;
        let archive = archive_ref::<E>(archive)?;
        let replay =
            tessera_rl_engine::Replay::open(data, tessera_rl_engine::DecodeLimits::default())?;
        let cur = replay.cursor_with_archive(&self.pool, archive)?;
        self.cursor = Some((data.to_vec(), cur));
        Ok(())
    }

    fn replay_advance(&mut self, calls: u64) -> Result<Option<u64>, BindError> {
        use tessera_rl_engine::{PointKind, Replay};
        if calls == 0 {
            return Err(BindError::Argument("calls must be at least 1".to_owned()));
        }
        let (pool, bytes, cur) = self.active("replay_advance")?;
        let replay = Replay::reopen(bytes, tessera_rl_engine::DecodeLimits::default(), cur)?;
        let mut seen: u64 = 0;
        let point = replay.advance(pool, cur, |p| {
            if p.kind == PointKind::Call {
                // `seen < calls <= u64::MAX` before this add (the previous
                // call would have stopped otherwise), so it cannot overflow;
                // checked anyway, stopping if it ever did.
                match seen.checked_add(1) {
                    Some(next) => seen = next,
                    None => return true,
                }
            }
            seen >= calls
        })?;
        Ok(point.map(|p| p.event_index))
    }

    fn replay_seek(&mut self, keyframe: u64) -> Result<(), BindError> {
        let (pool, bytes, cur) = self.active("replay_seek")?;
        let replay = tessera_rl_engine::Replay::reopen(
            bytes,
            tessera_rl_engine::DecodeLimits::default(),
            cur,
        )?;
        replay.seek(pool, cur, keyframe)?;
        Ok(())
    }

    fn replay_keyframes(&self) -> Result<Vec<u64>, BindError> {
        let (bytes, cur) = self.cursor.as_ref().ok_or_else(|| {
            BindError::Argument("replay_keyframes: no replay in progress".to_owned())
        })?;
        let replay = tessera_rl_engine::Replay::reopen(
            bytes,
            tessera_rl_engine::DecodeLimits::default(),
            cur,
        )?;
        Ok(replay.keyframes()?)
    }

    fn replay_finish(&mut self) -> Result<tessera_rl_engine::ReplaySummary, BindError> {
        let (bytes, cur) = self.cursor.take().ok_or_else(|| {
            BindError::Argument("replay_finish: no replay in progress".to_owned())
        })?;
        let replay = tessera_rl_engine::Replay::reopen(
            &bytes,
            tessera_rl_engine::DecodeLimits::default(),
            &cur,
        )?;
        Ok(replay.finish(cur)?)
    }
}

/// Pool arguments shared by every environment.
#[derive(Debug, Clone)]
pub(crate) struct PoolArgs {
    pub(crate) num_envs: u32,
    pub(crate) seed: u64,
    pub(crate) max_episode_steps: Option<u32>,
    pub(crate) auto_reset: String,
    pub(crate) chunk_size: Option<u32>,
    /// How chunks are given to workers: `"static"` (the default) or
    /// `"dynamic"` (ADR-0076). Never changes what a step computes.
    pub(crate) schedule: String,
    /// Worker threads; `None` (not given) is the engine default,
    /// [`tessera_rl_engine::default_workers`] (SPEC 6.1).
    pub(crate) workers: Option<usize>,
    /// Spins before a waiting worker or caller blocks (`SyncOptions::spin`);
    /// `None` keeps the engine default. Needs workers.
    pub(crate) spin: Option<u32>,
    /// The worker wait policy (ADR-0074): `"fixed"` (the default) or
    /// `"adaptive"`; `None` keeps the default. Needs workers.
    pub(crate) wait: Option<String>,
    /// The adaptive cap in microseconds, 1 to 1,000,000 (default 10,000);
    /// only with `wait="adaptive"`.
    pub(crate) wait_cap_us: Option<u32>,
    /// Pipelined mode (ADR-0047) instead of Sync.
    pub(crate) pipelined: bool,
    /// Async mode (ADR-0047) with its batch size and outputs.
    pub(crate) asynced: Option<(usize, crate::batched::BatchOutputs)>,
    /// The observation transforms (ADR-0078); empty for none.
    pub(crate) transforms: tessera_rl_engine::Transforms,
    /// The policy map (ADR-0079), `None` for the default: every slot a
    /// learner.
    pub(crate) policy_map: Option<crate::policy::PolicySpec>,
    /// The privileged state buffer (ADR-0079 item 7): only for an
    /// environment that implements `PrivilegedState`, in Sync mode.
    pub(crate) privileged_state: bool,
}

/// The refusal of the normalization calls outside sync mode.
fn transforms_sync_only() -> BindError {
    BindError::Capability(
        "epoch_barrier and freeze_normalization need a sync-mode pool (ADR-0078)".to_owned(),
    )
}

/// The native form of a transform list (built by `tessera_rl.transforms`):
/// one transform per line, `kind<TAB>leaf<TAB>parameter`, kinds
/// `frame_stack` (`k`), `one_hot` (`n`) and `normalize` (`clip`).
///
/// # Errors
///
/// [`BindError::Argument`] for a line of another shape.
pub(crate) fn parse_transforms(
    spec: Option<&str>,
) -> Result<tessera_rl_engine::Transforms, BindError> {
    let mut out = tessera_rl_engine::Transforms::new();
    let bad = |line: &str| BindError::Argument(format!("invalid transform {line:?}"));
    for line in spec.unwrap_or("").lines().filter(|l| !l.is_empty()) {
        let mut parts = line.split('\t');
        let (Some(kind), Some(leaf), Some(param), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(bad(line));
        };
        out = match kind {
            "frame_stack" => out.frame_stack(leaf, param.parse().map_err(|_| bad(line))?),
            "one_hot" => out.one_hot(leaf, param.parse().map_err(|_| bad(line))?),
            "normalize" => out.normalize(leaf, param.parse().map_err(|_| bad(line))?),
            _ => return Err(bad(line)),
        };
    }
    Ok(out)
}

fn options(args: &PoolArgs) -> Result<(PoolOptions, usize), BindError> {
    let num_envs = NonZeroU32::new(args.num_envs)
        .ok_or_else(|| BindError::Argument("num_envs must be at least 1".to_owned()))?;
    let auto_reset = match args.auto_reset.as_str() {
        "next_step" => AutoReset::NextStep,
        "same_step" => AutoReset::SameStep,
        "disabled" => AutoReset::Disabled,
        other => {
            return Err(BindError::Argument(format!(
                "auto_reset must be next_step, same_step or disabled, not {other:?}"
            )));
        }
    };
    let mut options = PoolOptions::new(num_envs, Seed::new(args.seed));
    options.auto_reset = auto_reset;
    options.max_episode_steps = args.max_episode_steps.and_then(NonZeroU32::new);
    if let Some(chunk) = args.chunk_size {
        options.chunk_size = NonZeroU32::new(chunk)
            .ok_or_else(|| BindError::Argument("chunk_size must be at least 1".to_owned()))?;
    }
    options.schedule = match args.schedule.as_str() {
        "static" => Schedule::Static,
        "dynamic" => Schedule::Dynamic,
        other => {
            return Err(BindError::Argument(format!(
                "schedule must be static or dynamic, not {other:?}"
            )));
        }
    };
    let n = usize::try_from(args.num_envs)
        .map_err(|_| BindError::Argument("num_envs too large".to_owned()))?;
    Ok((options, n))
}

/// The wait backend for new workers. On macOS `thread::park` is a libdispatch
/// semaphore, which traps in the child of a multithreaded parent (`SIGTRAP`,
/// issue #93), so a pool made in a forked child waits on the address instead.
/// `None` where it must not be built at all: that wait is missing too, and it
/// would fall back to park (macOS before 14.4).
#[must_use]
fn pick_wait(macos: bool, forked: bool, native: bool) -> Option<WaitBackend> {
    match (macos && forked, native) {
        (false, _) => Some(WaitBackend::Auto),
        (true, true) => Some(WaitBackend::Native),
        (true, false) => None,
    }
}

fn wait_backend() -> Result<WaitBackend, BindError> {
    pick_wait(
        cfg!(target_os = "macos"),
        crate::ffi::fork::generation() != 0,
        tessera_rl_engine::native_wait_available(),
    )
    .ok_or_else(|| {
        BindError::Forked(
            "worker threads cannot wait safely in a forked child on this macOS, which has no \
             native address wait (needs 14.4): use workers=0 in Sync mode, or the spawn start \
             method"
                .to_owned(),
        )
    })
}

/// Refuses `spin` where nothing would spin: Sync mode without workers. The
/// split-phase modes always have at least one worker.
fn check_spin(args: &PoolArgs) -> Result<(), BindError> {
    let sync = !args.pipelined && args.asynced.is_none();
    if args.spin.is_some() && sync && args.workers == Some(0) {
        return Err(BindError::Argument(
            "spin needs workers >= 1 in Sync mode".to_owned(),
        ));
    }
    Ok(())
}

/// The worker count of `args`: the given count (`None` for 0: inline in
/// Sync mode, one worker in the split-phase modes), or the engine default
/// when none was given.
fn resolve_workers(args: &PoolArgs) -> Result<Option<NonZeroUsize>, BindError> {
    if let Some(w) = args.workers {
        return Ok(NonZeroUsize::new(w));
    }
    let num_envs = NonZeroU32::new(args.num_envs)
        .ok_or_else(|| BindError::Argument("num_envs must be at least 1".to_owned()))?;
    let chunk = args.chunk_size.map_or_else(
        || Ok(PoolOptions::new(num_envs, Seed::new(0)).chunk_size),
        |c| {
            NonZeroU32::new(c)
                .ok_or_else(|| BindError::Argument("chunk_size must be at least 1".to_owned()))
        },
    )?;
    Ok(Some(tessera_rl_engine::default_workers(num_envs, chunk)))
}

/// The default adaptive cap of the Python surface, in microseconds.
pub(crate) const DEFAULT_WAIT_CAP_US: u32 = 10_000;

/// The largest `wait_cap_us`: one second, `MAX_ADAPTIVE_CAP`.
pub(crate) const MAX_WAIT_CAP_US: u32 = 1_000_000;

/// The wait policy of `wait` and `wait_cap_us` (ADR-0074), checked like
/// `spin`: it needs workers in Sync mode; `wait_cap_us` needs
/// `wait="adaptive"` and lies in 1..=1,000,000.
fn wait_policy(args: &PoolArgs) -> Result<WaitPolicy, BindError> {
    let sync = !args.pipelined && args.asynced.is_none();
    if args.wait.is_some() && sync && args.workers == Some(0) {
        return Err(BindError::Argument(
            "wait needs workers >= 1 in Sync mode".to_owned(),
        ));
    }
    match (args.wait.as_deref(), args.wait_cap_us) {
        (None | Some("fixed"), None) => Ok(WaitPolicy::Fixed),
        (None | Some("fixed"), Some(_)) => Err(BindError::Argument(
            "wait_cap_us needs wait=\"adaptive\"".to_owned(),
        )),
        (Some("adaptive"), cap) => {
            let cap = cap.unwrap_or(DEFAULT_WAIT_CAP_US);
            if !(1..=MAX_WAIT_CAP_US).contains(&cap) {
                return Err(BindError::Argument(format!(
                    "wait_cap_us must be in 1..=1000000 (microseconds), got {cap}"
                )));
            }
            Ok(WaitPolicy::Adaptive {
                cap: core::time::Duration::from_micros(u64::from(cap)),
            })
        }
        (Some(other), _) => Err(BindError::Argument(format!(
            "wait must be \"fixed\" or \"adaptive\", not {other:?}"
        ))),
    }
}

/// Sets the spin budget when the caller gave one, and the wait policy.
fn apply_spin(options: &mut SyncOptions, spin: Option<u32>, policy: WaitPolicy) {
    if let Some(spin) = spin {
        options.spin = spin;
    }
    options.wait_policy = policy;
}

/// The environment object of `pool`, in the mode of `args`. Pipelined mode
/// needs workers, one by default.
pub(crate) fn finish<E, M>(
    cfg: &E::Config,
    pool: EnvPool<E, M>,
    args: &PoolArgs,
    n: usize,
    terminal: bool,
) -> Result<Box<dyn PoolOps>, BindError>
where
    E: tessera_rl_core::Persist + 'static,
    E::Config: tessera_rl_core::config::CanonicalConfig + Send + Sync,
    E::Header: Send + Sync,
    E::State: Send + Sync,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
    M: tessera_rl_engine::PersistMask<E> + Send + Sync + 'static,
{
    finish_with_render(cfg, pool, args, n, terminal, None)
}

/// [`finish`] for an environment that implements `Render`: the sync-mode
/// pool renders (ADR-0083).
pub(crate) fn finish_rendered<E, M>(
    cfg: &E::Config,
    pool: EnvPool<E, M>,
    args: &PoolArgs,
    n: usize,
    terminal: bool,
) -> Result<Box<dyn PoolOps>, BindError>
where
    E: tessera_rl_core::Persist + tessera_rl_core::Render + 'static,
    E::Config: tessera_rl_core::config::CanonicalConfig + Send + Sync,
    E::Header: Send + Sync,
    E::State: Send + Sync,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
    E::Config: Clone,
    M: tessera_rl_engine::PersistMask<E> + Send + Sync + 'static,
{
    let fns = crate::render_api::render_fns::<E>(cfg)?;
    finish_with_render(cfg, pool, args, n, terminal, Some((cfg.clone(), fns)))
}

fn finish_with_render<E, M>(
    cfg: &E::Config,
    pool: EnvPool<E, M>,
    args: &PoolArgs,
    n: usize,
    terminal: bool,
    render: Option<(E::Config, crate::render_api::RenderFns<E>)>,
) -> Result<Box<dyn PoolOps>, BindError>
where
    E: tessera_rl_core::Persist + 'static,
    E::Config: tessera_rl_core::config::CanonicalConfig + Send + Sync,
    E::Header: Send + Sync,
    E::State: Send + Sync,
    <E::Header as tessera_rl_core::RecordLayout>::Adaptive:
        tessera_rl_core::AdaptiveValues<E::Config>,
    M: tessera_rl_engine::PersistMask<E> + Send + Sync + 'static,
{
    let workers = resolve_workers(args)?;
    check_spin(args)?;
    let policy = wait_policy(args)?;
    // Before the first reset and before any mode (ADR-0078 item 2.1).
    let pool = pool.with_transforms(args.transforms.clone())?;
    // Learner and external groups for any environment; a maker whose
    // environment runs heuristics applied its map already (ADR-0079).
    let pool = crate::policy::plain(pool, args)?;
    // A maker whose environment has a privileged state applied it already.
    crate::privileged::check(&pool, args)?;
    let storage = pool.outputs()?.storage();
    let pooled = Pooled::new(cfg, pool, n, terminal, storage)?;
    if args.pipelined {
        let Pooled {
            pool,
            layout,
            storage,
            ..
        } = pooled;
        let mut options = PipelinedOptions::new(workers.unwrap_or(NonZeroUsize::MIN));
        options.sync.wait = wait_backend()?;
        apply_spin(&mut options.sync, args.spin, policy);
        let pool = pool
            .pipelined(options)
            .map_err(|e| BindError::Engine(e.error))?;
        return Ok(crate::split::pipelined(pool, storage, layout));
    }
    if let Some((batch, outputs)) = args.asynced {
        let batch = NonZeroUsize::new(batch)
            .ok_or_else(|| BindError::Argument("batch_size must be at least 1".to_owned()))?;
        let Pooled {
            pool,
            layout,
            storage,
            ..
        } = pooled;
        let mut options = AsyncOptions::new(workers.unwrap_or(NonZeroUsize::MIN), batch);
        options.sync.wait = wait_backend()?;
        apply_spin(&mut options.sync, args.spin, policy);
        let pool = pool
            .async_mode(options)
            .map_err(|e| BindError::Engine(e.error))?;
        return crate::batched::batched(pool, storage, layout, outputs);
    }
    let Pooled {
        pool,
        layout,
        views,
        storage,
        ..
    } = pooled;
    let pool = match workers {
        Some(w) => {
            let mut options = SyncOptions::new(w);
            options.wait = wait_backend()?;
            apply_spin(&mut options, args.spin, policy);
            pool.with_workers(options)?
        }
        None => pool,
    };
    Ok(Box::new(Pooled {
        pool,
        rec: None,
        layout,
        views,
        storage,
        render,
        cursor: None,
    }))
}

/// The action head the Python layer can drive: one `Discrete`, or, for a
/// single-agent environment, one `f32` box element (design section 4.3).
/// One agent's action head: `(choices, bounds, nvec)`.
type ActionHead = (u32, Option<(f64, f64)>, Vec<u32>);

/// The action head of one agent: `(choices, bounds, nvec)`. `choices` is
/// the one discrete head's (0 otherwise), `bounds` the one continuous
/// head's, `nvec` the choices per discrete head.
fn action_head(space: &Space, agents: usize) -> Result<ActionHead, BindError> {
    match space {
        Space::Discrete { n } => return Ok((*n, None, vec![*n])),
        // One head is Discrete(n) in all but name: report its choices.
        Space::MultiDiscrete { nvec } => {
            let choices = match nvec.as_slice() {
                [n] => *n,
                _ => 0,
            };
            return Ok((choices, None, nvec.clone()));
        }
        Space::Box {
            shape,
            dtype: DType::F32,
            bounds: BoxBounds::Float { low, high },
        } if agents == 1 && shape.as_slice() == [1] => {
            if let ([low], [high]) = (low.as_slice(), high.as_slice()) {
                return Ok((0, Some((*low, *high)), Vec::new()));
            }
        }
        _ => {}
    }
    Err(BindError::Argument(
        "only Discrete, MultiDiscrete and single-agent one-element f32 Box action spaces are exposed"
            .to_owned(),
    ))
}

/// A pool whose first step panics (`make py-test-hooks`). Never in a release wheel.
#[cfg(feature = "test-hooks")]
pub(crate) fn panic_counter(args: &PoolArgs) -> Result<Box<dyn PoolOps>, BindError> {
    use crate::split::common::{CountConfig, Counter};
    let cfg = CountConfig {
        panic_at: 1,
        length: 9,
        ..CountConfig::default()
    };
    let (options, n) = options(args)?;
    let pool = EnvPool::<Counter, NoMasks>::new(cfg.clone(), options)?;
    finish(&cfg, pool, args, n, false)
}

/// A pool with two discrete heads, `MultiDiscrete([3, 5])` (`make
/// py-test-hooks`). Never in a release wheel.
#[cfg(feature = "test-hooks")]
pub(crate) fn multi_probe(args: &PoolArgs) -> Result<Box<dyn PoolOps>, BindError> {
    use crate::multi_probe::MultiProbe;
    use crate::split::common::CountConfig;
    let cfg = CountConfig::default();
    let (options, n) = options(args)?;
    let pool = EnvPool::<MultiProbe, Masked>::with_masks(cfg.clone(), options)?;
    finish(&cfg, pool, args, n, false)
}

/// A pool with one continuous head (`make py-test-hooks`). Never in a release
/// wheel.
#[cfg(feature = "test-hooks")]
pub(crate) fn continuous_probe(args: &PoolArgs) -> Result<Box<dyn PoolOps>, BindError> {
    use crate::continuous_probe::ContinuousProbe;
    use crate::split::common::CountConfig;
    let cfg = CountConfig::default();
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<ContinuousProbe, NoMasks>::new(cfg.clone(), options)?;
    finish(&cfg, pool, args, n, same_step)
}

/// A pendulum pool (`tessera/Pendulum-v1`).
pub(crate) fn pendulum(args: &PoolArgs) -> Result<Box<dyn PoolOps>, BindError> {
    let cfg = PendulumConfig::default();
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<Pendulum, NoMasks>::new(cfg.clone(), options)?;
    finish(&cfg, pool, args, n, same_step)
}

/// A CartPole pool (`tessera/CartPole-v1`).
pub(crate) fn cartpole(args: &PoolArgs) -> Result<Box<dyn PoolOps>, BindError> {
    let cfg = CartPoleConfig::default();
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<CartPole, NoMasks>::new(cfg.clone(), options)?;
    finish_rendered(&cfg, pool, args, n, same_step)
}

/// An acrobot pool (`tessera/Acrobot-v1`).
pub(crate) fn acrobot(args: &PoolArgs) -> Result<Box<dyn PoolOps>, BindError> {
    let cfg = AcrobotConfig::default();
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<Acrobot, NoMasks>::new(cfg.clone(), options)?;
    finish(&cfg, pool, args, n, same_step)
}

/// A mountain car pool (`tessera/MountainCar-v1`).
pub(crate) fn mountaincar(args: &PoolArgs) -> Result<Box<dyn PoolOps>, BindError> {
    let cfg = MountainCarConfig::default();
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<MountainCar, NoMasks>::new(cfg.clone(), options)?;
    finish(&cfg, pool, args, n, same_step)
}

/// A reference grid world pool (`tessera/Grid-v1`). A value too large for the
/// config's type becomes the largest one, which its range rejects.
pub(crate) fn grid(
    args: &PoolArgs,
    size: Option<u32>,
    max_steps: Option<u32>,
    layout: Option<tessera_rl_core::Asset<tessera_rl_classic::GridLayout>>,
) -> Result<Box<dyn PoolOps>, BindError> {
    let d = GridConfig::default();
    let field =
        |v: Option<u32>, default: u16| v.map_or(default, |x| u16::try_from(x).unwrap_or(u16::MAX));
    let cfg = GridConfig {
        size: field(size, d.size),
        max_steps: field(max_steps, d.max_steps),
        layout: layout.unwrap_or(d.layout),
    };
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<Grid, NoMasks>::new(cfg.clone(), options)?;
    finish_rendered(&cfg, pool, args, n, same_step)
}

/// A key and door pool (`tessera/GridKeyDoor-v1`), with masks; heuristic
/// slots run its `Heuristic` (ADR-0079 item 4.1: a perfect-information
/// environment).
pub(crate) fn keydoor(
    args: &PoolArgs,
    size: Option<u32>,
    agents: Option<u32>,
) -> Result<Box<dyn PoolOps>, BindError> {
    let d = KeyDoorConfig::default();
    let cfg = KeyDoorConfig {
        size: size.unwrap_or(d.size),
        agents: agents.unwrap_or(d.agents),
    };
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<KeyDoor, Masked>::with_masks(cfg.clone(), options)?;
    let pool = crate::policy::perfect(pool, args)?;
    finish_rendered(&cfg, pool, args, n, same_step)
}

/// A Gather pool (`tessera/Gather-v1`, ADR-0082). A value too large for the
/// config's type becomes the largest one, which its range rejects.
pub(crate) fn gather(
    args: &PoolArgs,
    size: Option<u32>,
    items: Option<u32>,
    max_steps: Option<u32>,
) -> Result<Box<dyn PoolOps>, BindError> {
    let d = GatherConfig::default();
    let cfg = GatherConfig {
        size: size.map_or(d.size, |x| u16::try_from(x).unwrap_or(u16::MAX)),
        items: items.map_or(d.items, |x| u8::try_from(x).unwrap_or(u8::MAX)),
        max_steps: max_steps.map_or(d.max_steps, |x| u16::try_from(x).unwrap_or(u16::MAX)),
    };
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<Gather, NoMasks>::new(cfg.clone(), options)?;
    finish(&cfg, pool, args, n, same_step)
}

/// The paddle duel config fields, by name, as Python passes them.
#[derive(Debug, Clone, Default)]
pub(crate) struct PaddleDuelFields {
    pub(crate) agents: Option<u32>,
    pub(crate) frameskip: Option<u32>,
    pub(crate) max_score: Option<u32>,
    pub(crate) bounce: Option<u32>,
}

/// A paddle duel pool (`tessera/PaddleDuel-v1`). `obs` names the observation.
pub(crate) fn paddleduel(
    args: &PoolArgs,
    f: &PaddleDuelFields,
    obs: Option<&str>,
) -> Result<Box<dyn PoolOps>, BindError> {
    use tessera_rl_classic::paddleduel::{OBS_PIXELS, OBS_STATE};
    let d = PaddleDuelConfig::default();
    let obs = match obs {
        None | Some("pixels") => OBS_PIXELS,
        Some("state") => OBS_STATE,
        Some(other) => {
            return Err(BindError::Argument(format!(
                "obs must be \"pixels\" or \"state\", not {other:?}"
            )));
        }
    };
    let cfg = PaddleDuelConfig {
        agents: f.agents.unwrap_or(d.agents),
        frameskip: f.frameskip.unwrap_or(d.frameskip),
        max_score: f.max_score.unwrap_or(d.max_score),
        bounce: f.bounce.unwrap_or(d.bounce),
        obs,
    };
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<PaddleDuel, NoMasks>::new(cfg.clone(), options)?;
    finish_rendered(&cfg, pool, args, n, same_step)
}

/// A brick wall pool (`tessera/BrickWall-v1`). `obs` names the observation.
pub(crate) fn brickwall(
    args: &PoolArgs,
    frameskip: Option<u32>,
    lives: Option<u32>,
    obs: Option<&str>,
) -> Result<Box<dyn PoolOps>, BindError> {
    use tessera_rl_classic::brickwall::{OBS_PIXELS, OBS_STATE};
    let d = BrickWallConfig::default();
    let obs = match obs {
        None | Some("pixels") => OBS_PIXELS,
        Some("state") => OBS_STATE,
        Some(other) => {
            return Err(BindError::Argument(format!(
                "obs must be \"pixels\" or \"state\", not {other:?}"
            )));
        }
    };
    let cfg = BrickWallConfig {
        frameskip: frameskip.unwrap_or(d.frameskip),
        lives: lives.unwrap_or(d.lives),
        obs,
    };
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<BrickWall, NoMasks>::new(cfg.clone(), options)?;
    finish_rendered(&cfg, pool, args, n, same_step)
}

/// A snake pool (`tessera/Snake-v1`), with masks. `obs` names the observation.
pub(crate) fn snake(
    args: &PoolArgs,
    size: Option<u32>,
    agents: Option<u32>,
    obs: Option<&str>,
) -> Result<Box<dyn PoolOps>, BindError> {
    use tessera_rl_classic::snake::{OBS_PIXELS, OBS_STATE};
    let d = SnakeConfig::default();
    let obs = match obs {
        None | Some("pixels") => OBS_PIXELS,
        Some("state") => OBS_STATE,
        Some(other) => {
            return Err(BindError::Argument(format!(
                "obs must be \"pixels\" or \"state\", not {other:?}"
            )));
        }
    };
    let cfg = SnakeConfig {
        size: size.unwrap_or(d.size),
        agents: agents.unwrap_or(d.agents),
        obs,
    };
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<Snake, Masked>::with_masks(cfg.clone(), options)?;
    finish_rendered(&cfg, pool, args, n, same_step)
}

/// The Tag config fields, by name, as Python passes them.
#[derive(Debug, Clone, Default)]
pub(crate) struct TagFields {
    pub(crate) width: Option<u32>,
    pub(crate) height: Option<u32>,
    pub(crate) pursuers: Option<u32>,
    pub(crate) evaders: Option<u32>,
    pub(crate) slip: Option<u32>,
    pub(crate) radius: Option<u32>,
}

/// A Tag pool (`tessera/Tag-v1`), with masks.
pub(crate) fn tag(args: &PoolArgs, fields: &TagFields) -> Result<Box<dyn PoolOps>, BindError> {
    let d = TagConfig::default();
    let cfg = TagConfig {
        width: fields.width.unwrap_or(d.width),
        height: fields.height.unwrap_or(d.height),
        pursuers: fields.pursuers.unwrap_or(d.pursuers),
        evaders: fields.evaders.unwrap_or(d.evaders),
        slip: fields.slip.unwrap_or(d.slip),
        radius: fields.radius.unwrap_or(d.radius),
    };
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<Tag, Masked>::with_masks(cfg.clone(), options)?;
    let pool = crate::policy::info(pool, args)?;
    finish(&cfg, pool, args, n, same_step)
}

/// The RTS config fields, by name, as Python passes them.
#[derive(Debug, Default)]
pub(crate) struct RtsFields {
    pub(crate) size: Option<u32>,
    pub(crate) decision_ticks: Option<u32>,
    pub(crate) max_units: Option<u32>,
    pub(crate) step_limit: Option<u32>,
    pub(crate) seat_swap: Option<bool>,
    pub(crate) opponent: Option<u32>,
}

/// An RTS pool (`tessera/RTS-v1`), with masks: two agents, simultaneous
/// turns, three discrete heads per unit slot grouped per unit (ADR-0051).
/// An order on an empty or dead slot other than the no-op `(0, 0, 0)`
/// counts as illegal, per slot.
pub(crate) fn rts(args: &PoolArgs, fields: &RtsFields) -> Result<Box<dyn PoolOps>, BindError> {
    let d = RtsConfig::default();
    let small = |v: Option<u32>, default: u16, what: &str| {
        v.map_or(Ok(default), |v| {
            u16::try_from(v).map_err(|_| BindError::Argument(format!("{what} is out of range")))
        })
    };
    let cfg = RtsConfig {
        size: small(fields.size, d.size, "size")?,
        decision_ticks: small(fields.decision_ticks, d.decision_ticks, "decision_ticks")?,
        max_units: small(fields.max_units, d.max_units, "max_units")?,
        step_limit: fields.step_limit.unwrap_or(d.step_limit),
        seat_swap: fields.seat_swap.unwrap_or(d.seat_swap),
        opponent: fields.opponent.map_or(Ok(d.opponent), |v| {
            u8::try_from(v).map_err(|_| BindError::Argument("opponent is out of range".to_owned()))
        })?,
    };
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<Rts, Masked>::with_masks(cfg.clone(), options)?;
    // Heuristic slots play the fog-respecting Rush bot (#586).
    let pool = crate::policy::info(pool, args)?;
    finish(&cfg, pool, args, n, same_step)
}

/// The limit order book config fields, as Python passes them: the `u32`
/// fields in the order of `api::LOB_U32_KEYS` (its last entry, `outputs`,
/// is not a config field and stays `None`), then the others.
#[derive(Debug, Default)]
pub(crate) struct LobFields {
    pub(crate) u32s: [Option<u32>; 17],
    pub(crate) initial_cash: Option<i64>,
    pub(crate) allow_short_cash: Option<bool>,
    pub(crate) bankrupt: Option<u64>,
}

/// A limit order book pool (`tessera/LimitOrderBook-v1`, #381), with masks:
/// simultaneous turns, four conditional discrete heads, `Set` observations.
pub(crate) fn lob(args: &PoolArgs, fields: &LobFields) -> Result<Box<dyn PoolOps>, BindError> {
    let d = LobConfig::default();
    let u = |i: usize, default: u32| fields.u32s.get(i).copied().flatten().unwrap_or(default);
    let cfg = LobConfig {
        agents: u(0, d.agents),
        steps: u(1, d.steps),
        depth: u(2, d.depth),
        min_tick: u(3, d.min_tick),
        max_tick: u(4, d.max_tick),
        initial_price: u(5, d.initial_price),
        up_permille: u(6, d.up_permille),
        down_permille: u(7, d.down_permille),
        latency: u(8, d.latency),
        size_0: u(9, d.size_0),
        size_1: u(10, d.size_1),
        size_2: u(11, d.size_2),
        size_3: u(12, d.size_3),
        max_position: u(13, d.max_position),
        noise: u(14, d.noise),
        reward_shift: u(15, d.reward_shift),
        initial_cash: fields.initial_cash.unwrap_or(d.initial_cash),
        allow_short_cash: fields.allow_short_cash.unwrap_or(d.allow_short_cash),
        bankrupt: fields.bankrupt.unwrap_or(d.bankrupt),
    };
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<LimitOrderBook, Masked>::with_masks(cfg.clone(), options)?;
    // Heuristic slots from the information state (InfoHeuristic), as Tag (#470).
    let pool = crate::policy::info(pool, args)?;
    let pool = crate::privileged::apply(pool, args)?;
    finish(&cfg, pool, args, n, same_step)
}

/// A Connect Four pool (`tessera/Connect4-v1`), with masks and
/// Sequential turns.
pub(crate) fn connect4(args: &PoolArgs) -> Result<Box<dyn PoolOps>, BindError> {
    let cfg = Connect4Config {};
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<Connect4, Masked>::with_masks(cfg.clone(), options)?;
    let pool = crate::policy::perfect(pool, args)?;
    finish_rendered(&cfg, pool, args, n, same_step)
}

/// An Othello pool (`tessera/Othello-v1`), with masks and Sequential turns.
pub(crate) fn othello(args: &PoolArgs) -> Result<Box<dyn PoolOps>, BindError> {
    let cfg = OthelloConfig {};
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<Othello, Masked>::with_masks(cfg.clone(), options)?;
    let pool = crate::policy::perfect(pool, args)?;
    finish_rendered(&cfg, pool, args, n, same_step)
}

/// A Go 9x9 pool (`tessera/Go9-v1`), with masks and Sequential turns.
pub(crate) fn go9(args: &PoolArgs, komi: Option<i32>) -> Result<Box<dyn PoolOps>, BindError> {
    let d = Go9Config::default();
    let cfg = Go9Config {
        komi_half_points: komi.unwrap_or(d.komi_half_points),
    };
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<Go9, Masked>::with_masks(cfg.clone(), options)?;
    let pool = crate::policy::perfect(pool, args)?;
    finish_rendered(&cfg, pool, args, n, same_step)
}

/// A Leduc hold'em pool (`tessera/Leduc-v1`), with masks and Sequential
/// turns.
pub(crate) fn leduc(args: &PoolArgs) -> Result<Box<dyn PoolOps>, BindError> {
    let cfg = LeducConfig {};
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<Leduc, Masked>::with_masks(cfg.clone(), options)?;
    let pool = crate::policy::info(pool, args)?;
    let pool = crate::privileged::apply(pool, args)?;
    finish(&cfg, pool, args, n, same_step)
}

/// The Lanterns config fields, by name, as Python passes them.
#[derive(Debug, Default)]
pub(crate) struct LanternsFields {
    pub(crate) players: Option<u32>,
    pub(crate) zero_on_fuse: Option<bool>,
}

/// A Lanterns pool (`tessera/Lanterns-v1`), with masks and Sequential
/// turns.
pub(crate) fn lanterns(
    args: &PoolArgs,
    fields: &LanternsFields,
) -> Result<Box<dyn PoolOps>, BindError> {
    let d = LanternsConfig::default();
    let cfg = LanternsConfig {
        players: fields.players.unwrap_or(d.players),
        zero_on_fuse: fields.zero_on_fuse.unwrap_or(d.zero_on_fuse),
    };
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<Lanterns, Masked>::with_masks(cfg.clone(), options)?;
    let pool = crate::policy::info(pool, args)?;
    finish(&cfg, pool, args, n, same_step)
}

/// The JobShop config fields, by name, as Python passes them.
#[derive(Debug, Default)]
pub(crate) struct JobShopFields {
    pub(crate) jobs: Option<u32>,
    pub(crate) machines: Option<u32>,
    pub(crate) max_duration: Option<u32>,
}

/// A job shop pool (`tessera/JobShop-v1`), with masks; heuristic slots
/// run the SPT `Heuristic`.
pub(crate) fn jobshop(
    args: &PoolArgs,
    fields: &JobShopFields,
) -> Result<Box<dyn PoolOps>, BindError> {
    let d = JobShopConfig::default();
    let cfg = JobShopConfig {
        jobs: fields.jobs.unwrap_or(d.jobs),
        machines: fields.machines.unwrap_or(d.machines),
        max_duration: fields.max_duration.unwrap_or(d.max_duration),
    };
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<JobShop, Masked>::with_masks(cfg.clone(), options)?;
    let pool = crate::policy::perfect(pool, args)?;
    finish(&cfg, pool, args, n, same_step)
}

/// The Vrp config fields, by name, as Python passes them.
#[derive(Debug, Default)]
pub(crate) struct VrpFields {
    pub(crate) customers: Option<u32>,
    pub(crate) vehicles: Option<u32>,
    pub(crate) grid: Option<u32>,
    pub(crate) max_demand: Option<u32>,
    pub(crate) capacity: Option<u32>,
}

/// A vehicle routing pool (`tessera/Vrp-v1`), with masks and Sequential
/// turns; heuristic slots run the nearest-neighbour `Heuristic`.
pub(crate) fn vrp(args: &PoolArgs, fields: &VrpFields) -> Result<Box<dyn PoolOps>, BindError> {
    let d = VrpConfig::default();
    let cfg = VrpConfig {
        customers: fields.customers.unwrap_or(d.customers),
        vehicles: fields.vehicles.unwrap_or(d.vehicles),
        grid: fields.grid.unwrap_or(d.grid),
        max_demand: fields.max_demand.unwrap_or(d.max_demand),
        capacity: fields.capacity.unwrap_or(d.capacity),
    };
    let (options, n) = options(args)?;
    let same_step = options.auto_reset == AutoReset::SameStep;
    let pool = EnvPool::<Vrp, Masked>::with_masks(cfg.clone(), options)?;
    let pool = crate::policy::perfect(pool, args)?;
    finish(&cfg, pool, args, n, same_step)
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_action_head_discrete_multi_and_box() {
        use super::action_head;
        assert_eq!(
            action_head(&Space::Discrete { n: 4 }, 2).unwrap(),
            (4, None, vec![4])
        );
        assert_eq!(
            action_head(&Space::MultiDiscrete { nvec: vec![5] }, 2).unwrap(),
            (5, None, vec![5])
        );
        assert_eq!(
            action_head(&Space::MultiDiscrete { nvec: vec![3, 5] }, 2).unwrap(),
            (0, None, vec![3, 5])
        );
        let float = |shape: Vec<u32>| Space::Box {
            shape,
            dtype: DType::F32,
            bounds: BoxBounds::Float {
                low: vec![-1.0],
                high: vec![1.0],
            },
        };
        assert_eq!(
            action_head(&float(vec![1]), 1).unwrap(),
            (0, Some((-1.0, 1.0)), vec![])
        );
        assert!(action_head(&float(vec![1]), 2).is_err());
        assert!(action_head(&float(vec![2]), 1).is_err());
    }

    use super::*;
    use crate::split::common::{CountConfig, Counter};

    /// Counts this thread's allocations (`outputs_into`'s zero-allocation
    /// test, ADR-0095 point 5).
    #[global_allocator]
    static COUNTING: tessera_rl_pyffi::xla::testing::CountingAlloc =
        tessera_rl_pyffi::xla::testing::CountingAlloc;

    fn sync_args(n: u32) -> PoolArgs {
        PoolArgs {
            num_envs: n,
            seed: 3,
            max_episode_steps: None,
            auto_reset: "next_step".to_owned(),
            chunk_size: None,
            schedule: "static".to_owned(),
            workers: Some(0),
            spin: None,
            wait: None,
            wait_cap_us: None,
            transforms: tessera_rl_engine::Transforms::new(),
            policy_map: None,
            pipelined: false,
            asynced: None,
            privileged_state: false,
        }
    }

    /// The `Owned` fields, in `outputs_into`'s order.
    fn owned_parts(o: &Owned) -> Vec<Vec<u8>> {
        let mut parts = o.observations.clone();
        parts.extend([
            o.rewards.clone(),
            o.agent_terminated.clone(),
            o.agent_truncated.clone(),
            o.alive.clone(),
            o.acting.clone(),
            o.slot_generations.clone(),
            o.episode_terminated.clone(),
            o.episode_truncated.clone(),
        ]);
        if !o.masks.is_empty() {
            parts.push(o.masks.clone());
        }
        parts
    }

    #[test]
    fn test_outputs_into_equals_outputs_without_allocating() {
        let mut pool = tag(&sync_args(4), &TagFields::default()).unwrap();
        pool.reset(None, &ResetSeed::Keep).unwrap();
        let agents = pool.layout().agents;
        // Stay (0) is always legal under Tag's mask.
        let actions = vec![0i32; 4 * agents];
        for _ in 0..5 {
            pool.step(ActionInput {
                discrete: &actions,
                continuous: &[],
            })
            .unwrap();
            let want = owned_parts(&pool.outputs().unwrap());
            let mut bufs: Vec<Vec<u8>> = want.iter().map(|w| vec![0xAA; w.len()]).collect();
            let mut views: Vec<&mut [u8]> = bufs.iter_mut().map(Vec::as_mut_slice).collect();
            let mut result = Ok(());
            let n =
                tessera_rl_pyffi::xla::testing::count(|| result = pool.outputs_into(&mut views));
            result.unwrap();
            assert_eq!(n, 0, "allocations on the success path");
            assert_eq!(bufs, want);
        }
    }

    #[test]
    fn test_outputs_into_refuses_privileged_state() {
        let args = PoolArgs {
            privileged_state: true,
            ..sync_args(2)
        };
        let mut pool = leduc(&args).unwrap();
        pool.reset(None, &ResetSeed::Keep).unwrap();
        let want = owned_parts(&pool.outputs().unwrap());
        let mut bufs: Vec<Vec<u8>> = want.iter().map(|w| vec![0; w.len()]).collect();
        let mut views: Vec<&mut [u8]> = bufs.iter_mut().map(Vec::as_mut_slice).collect();
        assert!(pool.outputs_into(&mut views).is_err());
    }

    #[test]
    fn test_outputs_into_refuses_wrong_buffers() {
        let mut pool = cartpole(&sync_args(2)).unwrap();
        pool.reset(None, &ResetSeed::Keep).unwrap();
        let want = owned_parts(&pool.outputs().unwrap());
        let mut short: Vec<Vec<u8>> = want.iter().map(|w| vec![0; w.len()]).collect();
        short.pop();
        let mut views: Vec<&mut [u8]> = short.iter_mut().map(Vec::as_mut_slice).collect();
        assert!(pool.outputs_into(&mut views).is_err());
        let mut wrong: Vec<Vec<u8>> = want.iter().map(|w| vec![0; w.len() + 1]).collect();
        let mut views: Vec<&mut [u8]> = wrong.iter_mut().map(Vec::as_mut_slice).collect();
        assert!(pool.outputs_into(&mut views).is_err());
    }

    fn poisonable() -> Box<dyn PoolOps> {
        let cfg = CountConfig {
            panic_at: 1,
            length: 9,
            ..CountConfig::default()
        };
        let options = PoolOptions::new(NonZeroU32::new(8).unwrap(), Seed::new(1));
        let pool = EnvPool::<Counter, NoMasks>::new(cfg.clone(), options).unwrap();
        let args = PoolArgs {
            num_envs: 8,
            seed: 1,
            max_episode_steps: None,
            auto_reset: "next_step".to_owned(),
            chunk_size: None,
            schedule: "static".to_owned(),
            workers: Some(0),
            spin: None,
            wait: None,
            wait_cap_us: None,
            transforms: tessera_rl_engine::Transforms::new(),
            policy_map: None,
            privileged_state: false,
            pipelined: false,
            asynced: None,
        };
        finish(&cfg, pool, &args, 8, false).unwrap()
    }

    fn spin_args(workers: Option<usize>, spin: Option<u32>, pipelined: bool) -> PoolArgs {
        PoolArgs {
            num_envs: 2,
            seed: 0,
            max_episode_steps: None,
            auto_reset: "next_step".to_owned(),
            chunk_size: None,
            schedule: "static".to_owned(),
            workers,
            spin,
            wait: None,
            wait_cap_us: None,
            transforms: tessera_rl_engine::Transforms::new(),
            policy_map: None,
            privileged_state: false,
            pipelined,
            asynced: None,
        }
    }

    fn wait_args(workers: usize, wait: Option<&str>, cap: Option<u32>) -> PoolArgs {
        let mut a = spin_args(Some(workers), None, false);
        a.wait = wait.map(str::to_owned);
        a.wait_cap_us = cap;
        a
    }

    #[test]
    fn test_wait_policy_defaults_and_adaptive_cap_resolved() {
        assert_eq!(
            wait_policy(&wait_args(0, None, None)).unwrap(),
            WaitPolicy::Fixed
        );
        assert_eq!(
            wait_policy(&wait_args(1, Some("fixed"), None)).unwrap(),
            WaitPolicy::Fixed
        );
        assert_eq!(
            wait_policy(&wait_args(1, Some("adaptive"), None)).unwrap(),
            WaitPolicy::Adaptive {
                cap: core::time::Duration::from_micros(10_000)
            }
        );
        assert_eq!(
            wait_policy(&wait_args(1, Some("adaptive"), Some(MAX_WAIT_CAP_US))).unwrap(),
            WaitPolicy::Adaptive {
                cap: core::time::Duration::from_secs(1)
            }
        );
    }

    #[test]
    fn test_parse_transforms_accepts_three_fields_and_refuses_the_rest() {
        assert!(parse_transforms(None).is_ok());
        assert!(
            parse_transforms(Some(
                "frame_stack\tboard\t2\none_hot\tview\t4\nnormalize\t\t5.0"
            ))
            .is_ok()
        );
        for bad in [
            "frame_stack\tboard\t2\textra",
            "frame_stack\tboard",
            "stack\tboard\t2",
            "frame_stack\tboard\ttwo",
            "normalize\t\tfive",
        ] {
            assert!(
                matches!(parse_transforms(Some(bad)), Err(BindError::Argument(_))),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn test_wait_policy_bad_arguments_refused() {
        for args in [
            wait_args(0, Some("adaptive"), None),
            wait_args(1, Some("eager"), None),
            wait_args(1, Some("adaptive"), Some(0)),
            wait_args(1, Some("adaptive"), Some(MAX_WAIT_CAP_US + 1)),
            wait_args(1, None, Some(5)),
            wait_args(1, Some("fixed"), Some(5)),
        ] {
            assert!(
                matches!(wait_policy(&args), Err(BindError::Argument(_))),
                "{args:?}"
            );
        }
    }

    #[test]
    fn test_resolve_workers_omitted_uses_engine_default() {
        let mut args = spin_args(None, None, false);
        args.num_envs = 1000;
        args.chunk_size = Some(64);
        let want = tessera_rl_engine::default_workers(
            NonZeroU32::new(1000).unwrap(),
            NonZeroU32::new(64).unwrap(),
        );
        assert_eq!(resolve_workers(&args).unwrap(), Some(want));
        args.chunk_size = None;
        let default_chunk =
            PoolOptions::new(NonZeroU32::new(1000).unwrap(), Seed::new(0)).chunk_size;
        let want =
            tessera_rl_engine::default_workers(NonZeroU32::new(1000).unwrap(), default_chunk);
        assert_eq!(resolve_workers(&args).unwrap(), Some(want));
    }

    #[test]
    fn test_resolve_workers_explicit_kept_and_zero_inline() {
        assert_eq!(
            resolve_workers(&spin_args(Some(0), None, false)).unwrap(),
            None
        );
        assert_eq!(
            resolve_workers(&spin_args(Some(64), None, false)).unwrap(),
            NonZeroUsize::new(64)
        );
    }

    #[test]
    fn test_resolve_workers_omitted_with_bad_sizes_refused() {
        let mut args = spin_args(None, None, false);
        args.chunk_size = Some(0);
        assert!(matches!(
            resolve_workers(&args),
            Err(BindError::Argument(_))
        ));
        args.chunk_size = None;
        args.num_envs = 0;
        assert!(matches!(
            resolve_workers(&args),
            Err(BindError::Argument(_))
        ));
    }

    #[test]
    fn test_check_spin_sync_default_workers_accepted() {
        assert!(check_spin(&spin_args(None, Some(5), false)).is_ok());
    }

    #[test]
    fn test_check_spin_sync_without_workers_refused() {
        assert!(matches!(
            check_spin(&spin_args(Some(0), Some(0), false)),
            Err(BindError::Argument(_))
        ));
    }

    #[test]
    fn test_check_spin_allowed_cases_accepted() {
        assert!(check_spin(&spin_args(Some(0), None, false)).is_ok());
        assert!(check_spin(&spin_args(Some(1), Some(200_000), false)).is_ok());
        assert!(check_spin(&spin_args(Some(0), Some(5), true)).is_ok());
        let mut asynced = spin_args(Some(0), Some(5), false);
        asynced.asynced = Some((1, crate::batched::BatchOutputs::Gathered));
        assert!(check_spin(&asynced).is_ok());
    }

    #[test]
    fn test_apply_spin_some_and_none_set_or_keep() {
        let mut options = SyncOptions::new(NonZeroUsize::MIN);
        apply_spin(&mut options, None, WaitPolicy::Fixed);
        assert_eq!(options.spin, SyncOptions::new(NonZeroUsize::MIN).spin);
        apply_spin(&mut options, Some(7), WaitPolicy::Fixed);
        assert_eq!(options.spin, 7);
    }

    /// Owned copies from before the panic stay valid, a new read is refused,
    /// the captured storage stays usable for teardown, and old views go stale.
    #[test]
    fn test_outputs_refused_while_poisoned_and_recovered_by_reset() {
        let mut ops = poisonable();
        ops.reset(None, &ResetSeed::Keep).unwrap();
        let copy = ops.outputs().unwrap();
        let storage = ops.storage();
        let before = storage.generation();
        let actions = vec![0; copy.rewards.len() / 4];
        assert!(ops.step(discrete(&actions)).is_err());
        assert!(ops.is_poisoned());
        assert!(matches!(
            ops.outputs(),
            Err(BindError::Engine(EngineError::Poisoned))
        ));
        assert_ne!(storage.generation(), before, "an old view is stale");
        assert_eq!(ops.storage().bytes(), storage.bytes());
        assert!(matches!(
            ops.readable_storage(),
            Err(BindError::Engine(EngineError::Poisoned))
        ));
        assert!(!copy.rewards.is_empty(), "the owned copy is unchanged");
        ops.reset(None, &ResetSeed::Keep).unwrap();
        assert!(ops.outputs().is_ok());
    }

    #[test]
    fn test_u32_bytes_are_little_endian() {
        assert_eq!(
            u32_bytes(&[0, 1, 0x0102_0304, u32::MAX]),
            [0, 0, 0, 0, 1, 0, 0, 0, 4, 3, 2, 1, 255, 255, 255, 255]
        );
    }

    /// Issue #273: a respawn into a dead slot reaches the owned copy as
    /// generation 1, little-endian, at that slot's offset.
    #[test]
    fn test_owned_slot_generations_carry_a_respawn() {
        let cfg = CountConfig {
            kill_at: 1,
            respawn: true,
            length: 20,
            ..CountConfig::default()
        };
        let options = PoolOptions::new(NonZeroU32::new(2).unwrap(), Seed::new(1));
        let pool = EnvPool::<Counter, NoMasks>::new(cfg.clone(), options).unwrap();
        let args = PoolArgs {
            num_envs: 2,
            seed: 1,
            max_episode_steps: None,
            auto_reset: "next_step".to_owned(),
            chunk_size: None,
            schedule: "static".to_owned(),
            workers: Some(0),
            spin: None,
            wait: None,
            wait_cap_us: None,
            transforms: tessera_rl_engine::Transforms::new(),
            policy_map: None,
            privileged_state: false,
            pipelined: false,
            asynced: None,
        };
        let mut ops = finish(&cfg, pool, &args, 2, false).unwrap();
        ops.reset(None, &ResetSeed::Keep).unwrap();
        assert_eq!(ops.outputs().unwrap().slot_generations, [0u8; 16]);
        let mut respawned = false;
        for _ in 0..10 {
            ops.step(discrete(&[0, 2, 0, 2])).unwrap();
            let out = ops.outputs().unwrap();
            assert_eq!(out.slot_generations.len(), 16);
            // Instance 0's slot 1: bytes 4..8 (`[N, A]`, u32 little-endian).
            if out.slot_generations.get(4..8) == Some(&[1, 0, 0, 0][..]) {
                assert_eq!(out.slot_generations.get(0..4), Some(&[0u8; 4][..]));
                respawned = true;
                break;
            }
        }
        assert!(respawned, "instance 0's agent 1 never respawned");
    }

    #[test]
    fn test_pick_wait_refuses_a_forked_macos_child_without_native_wait() {
        assert_eq!(pick_wait(true, true, false), None);
        assert_eq!(pick_wait(true, true, true), Some(WaitBackend::Native));
        assert_eq!(pick_wait(true, false, false), Some(WaitBackend::Auto));
        assert_eq!(pick_wait(false, true, false), Some(WaitBackend::Auto));
    }

    fn box_leaf(bounds: BoxBounds, dtype: DType) -> super::Leaf {
        let mut out = Vec::new();
        let mut sets = Vec::new();
        super::leaves(
            &Space::Box {
                shape: vec![2],
                dtype,
                bounds,
            },
            "x",
            &mut out,
            &mut sets,
        )
        .unwrap();
        assert_eq!(sets.len(), 0);
        out.pop().unwrap()
    }

    #[test]
    fn test_leaves_signed_bounds_extremes_exact() {
        use super::Bound;
        let leaf = box_leaf(
            BoxBounds::Int {
                low: vec![i64::MIN, -(1 << 53) - 1],
                high: vec![i64::MAX, (1 << 53) + 1],
            },
            DType::I64,
        );
        assert_eq!(
            leaf.low,
            vec![Bound::Int(i64::MIN), Bound::Int(-(1 << 53) - 1)]
        );
        assert_eq!(
            leaf.high,
            vec![Bound::Int(i64::MAX), Bound::Int((1 << 53) + 1)]
        );
    }

    #[test]
    fn test_leaves_unsigned_bounds_extremes_exact() {
        use super::Bound;
        let leaf = box_leaf(
            BoxBounds::UInt {
                low: vec![0, (1 << 53) + 1],
                high: vec![u64::MAX, (1 << 53) + 1],
            },
            DType::U32,
        );
        assert_eq!(leaf.low, vec![Bound::UInt(0), Bound::UInt((1 << 53) + 1)]);
        assert_eq!(
            leaf.high,
            vec![Bound::UInt(u64::MAX), Bound::UInt((1 << 53) + 1)]
        );
    }

    #[test]
    fn test_leaves_float_bounds_bits_unchanged() {
        use super::Bound;
        let low = vec![f64::NEG_INFINITY, -0.0, f64::from_bits(1), -1.5];
        let high = vec![f64::INFINITY, 0.0, f64::MIN_POSITIVE / 4.0, 2.5];
        let leaf = box_leaf(
            BoxBounds::Float {
                low: low.clone(),
                high: high.clone(),
            },
            DType::F64,
        );
        let bits = |v: &[Bound]| -> Vec<u64> {
            v.iter()
                .map(|b| match b {
                    Bound::Float(f) => f.to_bits(),
                    other => panic!("not a float bound: {other:?}"),
                })
                .collect()
        };
        let want = |v: &[f64]| -> Vec<u64> { v.iter().map(|f| f.to_bits()).collect() };
        assert_eq!(bits(&leaf.low), want(&low));
        assert_eq!(bits(&leaf.high), want(&high));
    }

    #[test]
    fn test_set_leaves_bounds_broadcast_exact_and_mask_zero_one() {
        use super::Bound;
        let mut out = Vec::new();
        let mut sets = Vec::new();
        super::leaves(
            &Space::Set {
                inner: Box::new(Space::Box {
                    shape: vec![2],
                    dtype: DType::I64,
                    bounds: BoxBounds::Int {
                        low: vec![i64::MIN, 0],
                        high: vec![i64::MAX, (1 << 53) + 1],
                    },
                }),
                max: 2,
            },
            "s",
            &mut out,
            &mut sets,
        )
        .unwrap();
        let [item, mask] = out.as_slice() else {
            panic!("two leaves expected, got {out:?}");
        };
        assert_eq!(
            item.high,
            vec![
                Bound::Int(i64::MAX),
                Bound::Int((1 << 53) + 1),
                Bound::Int(i64::MAX),
                Bound::Int((1 << 53) + 1)
            ]
        );
        assert_eq!(item.low.first(), Some(&Bound::Int(i64::MIN)));
        assert_eq!(mask.low, vec![Bound::UInt(0)]);
        assert_eq!(mask.high, vec![Bound::UInt(1)]);
    }
}
