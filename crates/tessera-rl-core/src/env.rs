// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `Environment` trait and its capability traits (SPEC 10, ADR-0012,
//! ADR-0035).

use tessera_rl_num::rng::{ObsRng, StepRng};

use crate::adapters::Adapters;
use crate::agent::{AgentSlot, Tier, TurnStructure};
use crate::encoding::Canonical;
use crate::error::{ConfigError, InvariantViolation, SpaceError};
use crate::header::RecordLayout;
use crate::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use crate::render::SceneWriter;
use crate::space::Space;
use crate::state_init::StateInit;
use crate::step_out::StepOut;

/// Plain-data state (ADR-0029): `#[repr(C)]`, padding-free, a valid initial
/// value ([`StateInit`]) and a canonical encoding.
///
/// Implemented by `#[derive(EnvState)]`, which checks the layout rules at
/// compile time. Implementing it by hand must uphold the same rules.
///
/// `Send + Sync`: plain data holds no pointers and no interior mutability,
/// so it is always both; the bound lets worker threads read shared states,
/// such as the entries of an archive (ADR-0075).
///
/// **Migrating a hand-written implementation (ADR-0090):** add
/// `type Fields = ();`, `fn to_fields(&self) -> Self::Fields {}` and a
/// `from_fields` that returns the default state; `LANES` keeps its default,
/// `false`. The derive emits them for you.
pub trait EnvState: Canonical + StateInit + Copy + Send + Sync + 'static {
    /// Whether the state is stored as Struct-of-Arrays columns, one per
    /// field (`#[state(lanes)]`, ADR-0090). `false`: the default
    /// Array-of-Structs store, where workers get `&mut State`.
    const LANES: bool = false;

    /// The state's fields as a tuple, for the column store of a lane-wise
    /// state; `()` for every other state. The tuple's element order is the
    /// fields' declaration order, and it is the column order: the engine
    /// keeps one column per element (ADR-0090 decision 2). `Copy + Send +
    /// Sync + 'static` through [`LaneTuple`](crate::LaneTuple), which is
    /// sealed: `()` or a tuple of up to 12 [`LaneField`](crate::LaneField)s.
    type Fields: crate::lane_field::LaneTuple;

    /// The fields of `self`, in declaration order.
    #[must_use]
    fn to_fields(&self) -> Self::Fields;

    /// The state with these fields. For a state with `Fields = ()` the
    /// engine never calls it; the derive returns the default state.
    #[must_use]
    fn from_fields(fields: Self::Fields) -> Self;
}

/// An environment configuration, validated at construction.
pub trait EnvConfig: Sized + 'static {
    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] naming the invalid field.
    fn validate(&self) -> Result<(), ConfigError> {
        Ok(())
    }

    /// Appends the config's assets (ADR-0077), nested configs included.
    /// `#[derive(EnvConfig)]` generates it; configs without assets keep
    /// this default, which appends nothing.
    fn asset_entries(&self, out: &mut Vec<crate::asset::AssetEntry>) {
        let _ = out;
    }
}

/// The effective adaptive config values of an environment's instance: its
/// record header's `adaptive` (ADR-0030).
pub type Adaptive<E> = <<E as Environment>::Header as RecordLayout>::Adaptive;

/// An environment: one trait for single-agent and multi-agent games.
///
/// `reset`, `step` and `observe` are infallible: all fallible work happens
/// at construction and at the validation boundary (ADR-0028).
pub trait Environment: Sized + 'static {
    /// Author state of one instance.
    type State: EnvState;
    /// Configuration shared by all instances of a pool.
    type Config: EnvConfig;
    /// Observation type (a marker until the `Observation` derive exists).
    type Obs: 'static;
    /// Action type (a marker until typed action heads exist).
    type Action: 'static;
    /// Record header: `RecordHeader<MAX_AGENTS, Adaptive>` (ADR-0036). The
    /// engine checks at compile time that its agent count equals
    /// `MAX_AGENTS`; use `NoAdaptive` when the config has no adaptive
    /// fields.
    type Header: RecordLayout;

    /// Number of agent slots; 1 for single-agent environments.
    const MAX_AGENTS: usize;
    /// The declared determinism tier.
    const TIER: Tier;
    /// Whether every agent observes the full state (ADR-0035).
    const PERFECT_INFORMATION: bool = true;
    /// The turn structure (ADR-0013). `Sequential` and `Mixed` take the
    /// acting agents from [`Environment::acting`]; `Simultaneous` ignores it.
    const TURNS: TurnStructure = TurnStructure::Simultaneous;
    /// The PettingZoo interfaces the environment declares (SPEC 12,
    /// ADR-0080 item 2). Empty by default: an environment offers a
    /// PettingZoo interface only by declaring it. [`register!`](crate::register)
    /// refuses a set that is not [`Adapters::valid_for`] its `TURNS`.
    const ADAPTERS: Adapters = Adapters::NONE;

    /// The observation space for `cfg`.
    ///
    /// # Errors
    ///
    /// Returns a [`SpaceError`] if `cfg` yields an invalid space.
    fn observation_space(cfg: &Self::Config) -> Result<Space, SpaceError>;

    /// The action space for `cfg`.
    ///
    /// # Errors
    ///
    /// Returns a [`SpaceError`] if `cfg` yields an invalid space.
    fn action_space(cfg: &Self::Config) -> Result<Space, SpaceError>;

    /// Number of agent slots alive at the start of each episode: slots
    /// `0..n` start alive and acting, the rest start empty and may be
    /// filled later by `agent_spawned` (a refinement of ADR-0013). The
    /// engine checks `1..=MAX_AGENTS` at construction. Defaults to
    /// `MAX_AGENTS`.
    #[must_use]
    fn initial_agents(cfg: &Self::Config) -> usize {
        let _ = cfg;
        Self::MAX_AGENTS
    }

    /// The team of agent `slot` (ADR-0079 item 8): a configuration fact,
    /// constant for the pool's life. The engine reads it once per slot at
    /// construction, never while stepping. Agents with the same team share
    /// a side; defaults to 0 for every slot.
    #[must_use]
    fn team(cfg: &Self::Config, slot: AgentSlot) -> u16 {
        let _ = (cfg, slot);
        0
    }

    /// The role of agent `slot` (ADR-0079 item 8), such as `"pursuer"`: a
    /// configuration fact, constant for the pool's life, read once per
    /// slot at construction, never while stepping. Defaults to `""` (no
    /// role).
    #[must_use]
    fn role(cfg: &Self::Config, slot: AgentSlot) -> &'static str {
        let _ = (cfg, slot);
        ""
    }

    /// Whether alive `agent` acts in `state`: its action is required and
    /// read in the next step. The engine asks after every reset and every
    /// step (after the step's deaths and spawns), for alive agents only,
    /// when [`Environment::TURNS`] is not `Simultaneous`. `Sequential` and
    /// `Mixed` environments must override it: the default, `true`, makes
    /// every alive agent act. A `Sequential` environment returns `true` for
    /// exactly one alive agent whenever the episode goes on, or the pool is
    /// poisoned; a `Mixed` one may return `false` for every agent.
    #[must_use]
    fn acting(cfg: &Self::Config, state: &Self::State, agent: AgentSlot) -> bool {
        let _ = (cfg, state, agent);
        true
    }

    /// Starts an episode. `state` holds the initial value or the previous
    /// episode.
    fn reset(cfg: &Self::Config, rng: &mut StepRng, state: &mut Self::State);

    /// Whether [`Environment::reset_with`] or [`Environment::step_with`]
    /// reads the instance's adaptive values (ADR-0091). Pools refuse
    /// curricula and targets for an environment with adaptive fields that
    /// does not read them, and lane stepping requires it false. Set it
    /// together with an override; a wrapper environment forwards its inner
    /// environment's value and both methods.
    const READS_ADAPTIVE: bool = false;

    /// [`Environment::reset`] with the instance's effective adaptive config
    /// values (ADR-0030, ADR-0091), which the engine always calls. The
    /// default ignores them.
    fn reset_with(
        cfg: &Self::Config,
        adaptive: &Adaptive<Self>,
        rng: &mut StepRng,
        state: &mut Self::State,
    ) {
        let _ = adaptive;
        Self::reset(cfg, rng, state);
    }

    /// Advances one transition.
    fn step(
        cfg: &Self::Config,
        state: &mut Self::State,
        actions: &Actions<'_, Self::Action>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    );

    /// [`Environment::step`] with the instance's effective adaptive config
    /// values (ADR-0091), which the engine always calls. The default
    /// ignores them.
    fn step_with(
        cfg: &Self::Config,
        adaptive: &Adaptive<Self>,
        state: &mut Self::State,
        actions: &Actions<'_, Self::Action>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let _ = adaptive;
        Self::step(cfg, state, actions, rng, out);
    }

    /// Writes the observation of `agent`. Every leaf is zero on entry:
    /// bytes left unwritten are zero in the output, no earlier content is
    /// visible, and an implementation need not clear a leaf (ADR-0058).
    fn observe(
        cfg: &Self::Config,
        state: &Self::State,
        agent: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, Self::Obs>,
    );
}

/// Legal-action masks, written in the observation pass (ADR-0014), and
/// the no-op that replaces an illegal action.
pub trait ActionMask: Environment {
    /// Writes the mask of `agent`: one entry per choice of each discrete
    /// head, heads concatenated in order.
    fn action_mask(
        cfg: &Self::Config,
        state: &Self::State,
        agent: AgentSlot,
        out: &mut MaskWriter<'_>,
    );

    /// Writes the environment's no-op action for `agent`, used in place of
    /// an illegal action (ADR-0028). There is no default: the author
    /// defines it.
    fn noop(cfg: &Self::Config, state: &Self::State, agent: AgentSlot, out: &mut ActionWriter<'_>);

    /// Parents of discrete head `head`, earlier heads whose values select
    /// its mask row (SPEC 11.3, at most two). Called at construction only.
    #[must_use]
    fn head_parents(_cfg: &Self::Config, _head: usize) -> [Option<u16>; 2] {
        [None, None]
    }

    /// Discrete heads per entity group, for per-entity illegal-action
    /// fallback (ADR-0051); `None` keeps one agent-wide group. Called at
    /// construction only.
    #[must_use]
    fn entity_group_heads(_cfg: &Self::Config) -> Option<core::num::NonZeroU16> {
        None
    }

    /// Whether discrete head `head` is used when its parents
    /// ([`Self::head_parents`]) take the values `parents` (unused entries
    /// 0), ADR-0086. Config-only, never state; called at construction only.
    ///
    /// An environment that returns `false` for some head must ignore that
    /// head's value in its step: an action and its copies differing only in
    /// inactive heads are the same game action. Search keeps one of them,
    /// the canonical joint with every inactive head at 0.
    /// Without an `ActionMask` implementation every head is active, so every
    /// joint is canonical and non-canonical copies appear in search.
    /// `tessera_rl_search::check_head_activity` tests the obligation.
    #[must_use]
    fn head_active(_cfg: &Self::Config, _head: usize, _parents: [u32; 2]) -> bool {
        true
    }
}

/// Imperfect information: the per-agent information state (ADR-0035).
///
/// Required when [`Environment::PERFECT_INFORMATION`] is `false`.
pub trait ImperfectInformation: Environment {
    /// Fixed-size, plain-data information state of one agent.
    type InfoState: EnvState;

    /// Returns what `agent` knows, read from state.
    fn info_state(cfg: &Self::Config, state: &Self::State, agent: AgentSlot) -> Self::InfoState;
}

/// Samples full states consistent with an information state, for
/// player-perspective search (ADR-0035).
pub trait Determinize: ImperfectInformation {
    /// Writes into `state` a full author state consistent with `info`. It
    /// sees only the information state, never the hidden state.
    fn sample(
        cfg: &Self::Config,
        info: &Self::InfoState,
        rng: &mut StepRng,
        state: &mut Self::State,
    );
}

/// Rendering (SPEC 10.1, 13, ADR-0017, ADR-0083): draws a snapshot's state
/// into a vector scene, from which frames are rasterized.
///
/// `draw` is a pure, infallible function of `(cfg, state)`: no RNG, no
/// time. Nothing in `reset`, `step` or `observe` calls it; renderers call it
/// on states taken from snapshot records. A scene built without floats
/// rasterizes to bit-exact frames on every platform.
pub trait Render: Environment {
    /// The scene buffer's capacity in primitives (at most
    /// [`render::MAX_PRIMITIVES_LIMIT`](crate::render::MAX_PRIMITIVES_LIMIT)).
    /// A scene that needs more is truncated and reported as overflowed.
    const MAX_PRIMITIVES: usize = 1024;

    /// The canvas size in pixels, `(width, height)`, each in
    /// `1..=`[`render::MAX_CANVAS`](crate::render::MAX_CANVAS).
    fn canvas(cfg: &Self::Config) -> (u16, u16);

    /// Draws `state` into `scene`.
    fn draw(cfg: &Self::Config, state: &Self::State, scene: &mut SceneWriter<'_>);
}

/// A scripted policy for opponents and baselines.
pub trait Heuristic: Environment {
    /// Writes the action of `agent`.
    fn act(
        cfg: &Self::Config,
        state: &Self::State,
        agent: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    );
}

/// A player-only scripted policy of an imperfect-information environment
/// (ADR-0079 item 4.2): it sees the agent's information state, never the
/// hidden state, so it may drive a `Heuristic` slot of a policy map.
///
/// Its rules are those of `Environment` code: no allocation, no locks, and
/// randomness only from `rng`. Every action it writes must be in range and
/// legal under the mask, or the engine poisons the pool.
pub trait InfoHeuristic: ImperfectInformation {
    /// Writes the action of `agent` from its information state `info`.
    fn act_info(
        cfg: &Self::Config,
        info: &Self::InfoState,
        agent: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    );
}

/// The privileged state of one instance (ADR-0079 item 7): everything a
/// centralized critic needs, hidden information included, as one row per
/// instance. The engine writes it only when the pool opts in
/// (`EnvPool::with_privileged_state`), in the same calls that write the
/// observations; search never reads it.
///
/// Its rules are those of `Environment` code: no allocation, no locks, no
/// randomness, and the output must be a function of `cfg` and `state`.
pub trait PrivilegedState: Environment {
    /// The space of one instance's row: boxes, dicts and sets, laid out as
    /// an observation space. Called at construction only.
    ///
    /// # Errors
    ///
    /// [`SpaceError`] for an invalid space.
    fn state_space(cfg: &Self::Config) -> Result<Space, SpaceError>;

    /// Writes the privileged state of `state`. Every leaf is zero on entry,
    /// as for [`Environment::observe`] (ADR-0058).
    fn write_state(cfg: &Self::Config, state: &Self::State, out: &mut ObsWriter<'_, '_, ()>);
}

/// Identity for persisted snapshots (ADR-0011).
///
/// Format 1 covers only environments with no external immutable assets:
/// implementing this for one that has them is a contract violation, as
/// the file could not identify its assets.
pub trait Persist: Validate {
    /// The versioned environment id, `tessera/<Name>-v<N>`.
    const ENV_ID: &'static str;
    /// The environment version.
    const ENV_VERSION: u32;
}

/// Immutable shared data, content-addressed (SPEC 17).
///
/// Deprecated: assets are `Asset<T>` config fields (ADR-0077). An asset type
/// implements [`crate::AssetData`] (`bundled` names its bundled assets), and
/// `#[derive(EnvConfig)]` lists the config's assets for persisted headers.
/// Nothing uses this trait; it will be removed in a later minor release.
#[deprecated(
    since = "0.4.0",
    note = "use `Asset<T>` config fields and `AssetData` (ADR-0077); this trait is unused"
)]
pub trait EnvAssets: Environment {
    /// The asset bundle, shared read-only by every instance.
    type Assets: Send + Sync + 'static;
}

/// Game invariants checked when decoding persisted snapshots (ADR-0029).
pub trait Validate: Environment {
    /// Checks `state` against `cfg`.
    ///
    /// # Errors
    ///
    /// Returns the first [`InvariantViolation`] found.
    fn validate(cfg: &Self::Config, state: &Self::State) -> Result<(), InvariantViolation>;

    /// Checks an untrusted record: `state` together with its engine header
    /// (ADR-0040), for invariants that span both (for example, an agent
    /// slot is alive exactly when the state has that agent). The engine
    /// calls it on every imported record, after its own header checks and
    /// before writing anything; restore and fork of trusted records do not
    /// call it. Defaults to [`Validate::validate`].
    ///
    /// # Errors
    ///
    /// Returns the [`InvariantViolation`] found.
    fn validate_record(
        cfg: &Self::Config,
        header: &crate::header_view::HeaderView<'_>,
        state: &Self::State,
    ) -> Result<(), InvariantViolation> {
        let _ = header;
        Self::validate(cfg, state)
    }
}
