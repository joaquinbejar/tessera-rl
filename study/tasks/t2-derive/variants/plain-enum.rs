//! `study-t2-derive/StudyT2Derive-v1`: a Tessera environment on the `Environment` trait (SPEC
//! 10), scaffolded by `cargo tessera new --kind env`. Replace the counter
//! game with your environment; `cargo tessera check` keeps it
//! deterministic, continuable and within its spaces.
//!
//! Rust only: a separately compiled `Environment` cannot be loaded into the
//! prebuilt `tessera-rl` wheel (ADR-0032). The crate runs through
//! `tessera_rl_engine::EnvPool`, every engine mode, snapshot, restore and
//! fork, and search. For Python access, scaffold a host environment
//! (`cargo tessera new --kind host`, ADR-0062).
//!
//! The game: 1 agent(s) add 0, 1 or 2 to a shared counter. An
//! add is rewarded by its size and is legal only while the counter stays
//! at or below the target; reaching the target terminates the episode, and
//! the step limit truncates it.
//!
//! The rules this file follows, which the checks and clippy enforce:
//! - the state is plain data (ADR-0029): `#[repr(C)]`, no padding, no
//!   pointers or heap fields, `Bool8` and `#[derive(StateEnum)]` instead of
//!   `bool` and plain enums;
//! - randomness comes only from the `StepRng` and `ObsRng` the engine
//!   passes, which are keyed per instance (ADR-0007);
//! - `reset`, `step` and `observe` are infallible and allocate nothing;
//!   actions are validated by the engine before the step;
//! - no `HashMap` iteration, no wall-clock time, and none of the float
//!   methods `clippy.toml` bans (ADR-0008): transcendental math comes from
//!   `tessera_rl_num::math`.

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Bool8, Environment, EpisodeEnd, Heuristic, InvariantViolation,
    NoAdaptive, ObsLeaf, Observation, Persist, RecordHeader, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation, StateEnum};
use tessera_rl_num::rng::{ObsRng, StepRng};

/// The versioned environment id (ADR-0023).
pub const ENV_ID: &str = "study-t2-derive/StudyT2Derive-v1";

/// The environment version: bump it, and the id's, with any change to the
/// outputs, then re-bless the goldens.
pub const ENV_VERSION: u32 = 1;

/// Agent slots.
pub const AGENTS: usize = 1;

/// Choices per agent: add 0, 1 or 2.
pub const CHOICES: u32 = 3;

/// The bound of every observation value.
pub const BOUND: i32 = 1_000_000;

/// The configuration, shared by every instance of a pool.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct CounterConfig {
    /// Steps per episode; the episode is truncated at this step.
    #[config(default = 64, range = 1..=100_000)]
    pub episode: u32,
    /// The counter's target; reaching it terminates the episode.
    #[config(default = 40, range = 1..=100_000)]
    pub target: u32,
}

/// The phase of an episode: a plain-data enum (`#[derive(StateEnum)]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Phase {
    /// The counter is below half the target.
    #[default]
    Opening = 1,
    /// The counter is at or above half the target.
    Closing = 2,
}

/// One instance's state: 12 bytes, no padding (ADR-0029).
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct CounterState {
    /// The shared counter, at most the target.
    pub counter: u32,
    /// Steps taken in this episode.
    pub step: u32,
    /// Sequential: the slot that acts next. Mixed: the step's parity.
    pub turn: u16,
    /// The phase, from the counter.
    pub phase: Phase,
    /// Whether the counter reached the target.
    pub reached: Bool8,
}

/// The observation of one agent.
#[derive(Observation)]
pub struct CounterObs {
    /// The counter, the step, the turn and the agent's own slot.
    #[space(box(low = 0, high = 1_000_000, shape = [4]))]
    pub values: ObsLeaf<i32>,
}

/// The counter game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Counter;

/// Agent slot `i`.
fn slot(i: usize) -> Option<AgentSlot> {
    AgentSlot::new(u16::try_from(i).ok()?, AGENTS)
}

/// The phase of `counter`.
const fn phase_of(cfg: &CounterConfig, counter: u32) -> Phase {
    // The counter is at most the target (100_000), so doubling it cannot
    // overflow.
    if counter.saturating_mul(2) < cfg.target {
        Phase::Opening
    } else {
        Phase::Closing
    }
}

/// Whether adding `add` keeps the counter at or below the target.
const fn legal(cfg: &CounterConfig, s: &CounterState, add: u32) -> bool {
    // The counter is at most the target (100_000) and `add` at most 2.
    s.counter.saturating_add(add) <= cfg.target
}

impl Environment for Counter {
    type State = CounterState;
    type Config = CounterConfig;
    type Obs = CounterObs;
    type Action = ();
    type Header = RecordHeader<AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = AGENTS;
    const TIER: Tier = Tier::CrossPlatform;

    fn observation_space(cfg: &CounterConfig) -> Result<Space, SpaceError> {
        CounterObs::space(cfg)
    }

    /// One discrete head: the add.
    fn action_space(_: &CounterConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: CHOICES })
    }

    /// The opening counter is drawn from the instance's `StepRng`, never
    /// above the target.
    fn reset(cfg: &CounterConfig, rng: &mut StepRng, s: &mut CounterState) {
        let counter = (rng.next_u32() % CHOICES).min(cfg.target);
        *s = CounterState {
            counter,
            phase: phase_of(cfg, counter),
            ..CounterState::default()
        };
    }

    fn step(
        cfg: &CounterConfig,
        s: &mut CounterState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        // Bounded by the episode length (at most 100_000).
        s.step = s.step.saturating_add(1);
        for i in 0..AGENTS {
            let Some(agent) = slot(i) else {
                continue;
            };
            if !Self::acting(cfg, s, agent) {
                continue;
            }
            // The engine checked the range; the mask replaced an illegal
            // add with the no-op (ADR-0028).
            let add = actions
                .discrete(agent, 0)
                .and_then(|a| u32::try_from(a).ok())
                .filter(|&a| legal(cfg, s, a))
                .unwrap_or(0);
            // Bounded by the target: `legal` kept the sum at or below it.
            s.counter = s.counter.saturating_add(add);
            out.set_reward(agent, f32::from(u8::try_from(add).unwrap_or(0)));
        }
        s.phase = phase_of(cfg, s.counter);
        s.reached = Bool8::from(s.counter >= cfg.target);
        if s.reached.get() {
            out.end_episode(EpisodeEnd::Terminated);
        } else if s.step >= cfg.episode {
            out.end_episode(EpisodeEnd::Truncated);
        }
    }

    fn observe(
        _: &CounterConfig,
        s: &CounterState,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, CounterObs>,
    ) {
        let Some(mut leaf) = out.leaf_mut(CounterObs::VALUES) else {
            return;
        };
        let index = u32::try_from(agent.index()).unwrap_or(0);
        let values = [s.counter, s.step, u32::from(s.turn), index];
        for (k, v) in values.into_iter().enumerate() {
            leaf.set(k, i32::try_from(v).unwrap_or(BOUND).min(BOUND));
        }
    }
}

/// Legal adds keep the counter at or below the target; adding 0 is the
/// no-op that replaces an illegal action.
impl ActionMask for Counter {
    fn action_mask(
        cfg: &CounterConfig,
        s: &CounterState,
        _: AgentSlot,
        out: &mut MaskWriter<'_>,
    ) {
        for add in 0..CHOICES {
            out.set(usize::try_from(add).unwrap_or(0), legal(cfg, s, add));
        }
    }

    fn noop(_: &CounterConfig, _: &CounterState, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, 0);
    }
}

/// The scripted baseline: add 1 while it is legal.
impl Heuristic for Counter {
    fn act(
        cfg: &CounterConfig,
        s: &CounterState,
        _: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        out.set_discrete(0, i32::from(legal(cfg, s, 1)));
    }
}

/// The invariants an imported state must hold (ADR-0040): the engine checks
/// every record it decodes from bytes against them.
impl Validate for Counter {
    fn validate(cfg: &CounterConfig, s: &CounterState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        if s.counter > cfg.target {
            return invalid("a counter past the target");
        }
        if s.step > cfg.episode {
            return invalid("a step past the episode length");
        }
        if usize::from(s.turn) >= AGENTS.max(2) {
            return invalid("a turn past the agents");
        }
        if s.phase != phase_of(cfg, s.counter) {
            return invalid("a phase that disagrees with the counter");
        }
        if s.reached.get() != (s.step > 0 && s.counter >= cfg.target) {
            return invalid("a reached flag that disagrees with the counter");
        }
        Ok(())
    }
}

/// The id and version persisted records carry (SPEC 9).
impl Persist for Counter {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
