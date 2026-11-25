//! `study-ref-b/StudyRefB-v1`: the reference implementation of rule sheet
//! T1-B, two-player Nim (study/tasks/t1-b-nim.md, #639). It began as the
//! `cargo tessera new --kind env --agents 2 --turns sequential` scaffold.
//! Graders use it to record study/grading/golden/b.txt. It is not handed
//! to participants.
//!
//! The rules (the sheet is normative):
//! - three heaps, starting at 3, 4 and 5 items; no randomness;
//! - two players alternate, slot 0 first (sequential turns);
//! - action `a` in `Discrete(9)` takes `a % 3 + 1` items from heap `a / 3`.
//!   It is legal when that heap holds at least that many items;
//! - whoever takes the last item wins: reward 1 to the winner and -1 to
//!   the other, and the episode terminates. Every other step gives 0;
//! - step `episode` (default 64) truncates;
//! - the observation, the same for both players, is one `Box(i32, [4])`
//!   leaf: the three heap sizes, then the slot to move.

use tessera_rl_core::TurnStructure;
use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Environment, EpisodeEnd, Heuristic, InvariantViolation, NoAdaptive,
    ObsLeaf, Observation, Persist, RecordHeader, SpaceError, StepOut, Tier, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

/// The versioned environment id (ADR-0023).
pub const ENV_ID: &str = "study-ref-b/StudyRefB-v1";

/// The environment version.
pub const ENV_VERSION: u32 = 1;

/// Agent slots.
pub const AGENTS: usize = 2;

/// Heaps.
pub const HEAPS: usize = 3;

/// The most items one move takes.
pub const TAKE: u32 = 3;

/// Choices per agent: heap and count.
pub const CHOICES: u32 = 9;

/// The bound of every observation value.
pub const BOUND: i32 = 5;

/// The heaps at reset.
const START: [u8; HEAPS] = [3, 4, 5];

/// The configuration, shared by every instance of a pool.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct NimConfig {
    /// Steps per episode; the episode is truncated at this step.
    #[config(default = 64, range = 1..=100_000)]
    pub episode: u32,
}

/// One instance's state: 8 bytes, no padding (ADR-0029).
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct NimState {
    /// Steps taken in this episode.
    pub step: u32,
    /// The heap sizes.
    pub heaps: [u8; HEAPS],
    /// The slot to move.
    pub turn: u8,
}

/// The observation of one player.
#[derive(Observation)]
pub struct NimObs {
    /// The three heap sizes, then the slot to move.
    #[space(box(low = 0, high = 5, shape = [4]))]
    pub values: ObsLeaf<i32>,
}

/// Nim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Nim;

/// The environment the study's grader runs (the sheet's grading name).
pub type StudyEnv = Nim;

/// The heap and the count of action `a`.
fn decode(a: u32) -> (usize, u8) {
    let heap = usize::try_from(a / TAKE).unwrap_or(usize::MAX);
    let count = u8::try_from(a % TAKE + 1).unwrap_or(u8::MAX);
    (heap, count)
}

/// Whether action `a` is legal in `s`.
fn legal(s: &NimState, a: u32) -> bool {
    let (heap, count) = decode(a);
    s.heaps.get(heap).is_some_and(|&h| h >= count)
}

impl Environment for Nim {
    type State = NimState;
    type Config = NimConfig;
    type Obs = NimObs;
    type Action = ();
    type Header = RecordHeader<AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = AGENTS;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &NimConfig) -> Result<Space, SpaceError> {
        NimObs::space(cfg)
    }

    /// One discrete head: heap and count.
    fn action_space(_: &NimConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: CHOICES })
    }

    /// One player per step, slot 0 first.
    fn acting(_: &NimConfig, s: &NimState, agent: AgentSlot) -> bool {
        usize::from(s.turn) == agent.index()
    }

    /// The heaps start full; Nim draws nothing.
    fn reset(_: &NimConfig, _: &mut StepRng, s: &mut NimState) {
        *s = NimState {
            heaps: START,
            ..NimState::default()
        };
    }

    fn step(
        cfg: &NimConfig,
        s: &mut NimState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        // Bounded by the episode length (at most 100_000).
        s.step = s.step.saturating_add(1);
        let mover = s.turn;
        let (Some(me), Some(other)) = (
            AgentSlot::new(u16::from(mover), AGENTS),
            AgentSlot::new(u16::from(1 - mover.min(1)), AGENTS),
        ) else {
            return;
        };
        // The engine checked the range; the mask replaced an illegal move
        // with the no-op, which is legal whenever a heap is not empty.
        let a = actions
            .discrete(me, 0)
            .and_then(|a| u32::try_from(a).ok())
            .filter(|&a| legal(s, a));
        if let Some(a) = a {
            let (heap, count) = decode(a);
            if let Some(h) = s.heaps.get_mut(heap) {
                // `legal` checked that the heap holds at least `count`.
                *h = h.saturating_sub(count);
            }
        }
        s.turn = 1 - mover.min(1);
        if s.heaps.iter().all(|&h| h == 0) {
            out.set_reward(me, 1.0);
            out.set_reward(other, -1.0);
            out.end_episode(EpisodeEnd::Terminated);
        } else if s.step >= cfg.episode {
            out.end_episode(EpisodeEnd::Truncated);
        }
    }

    fn observe(
        _: &NimConfig,
        s: &NimState,
        _: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, NimObs>,
    ) {
        let Some(mut leaf) = out.leaf_mut(NimObs::VALUES) else {
            return;
        };
        let [h0, h1, h2] = s.heaps;
        let values = [h0, h1, h2, s.turn];
        for (k, v) in values.into_iter().enumerate() {
            leaf.set(k, i32::from(v));
        }
    }
}

/// Legal moves take at most what a heap holds. The no-op is the first
/// legal move, so it is always legal while the game runs.
impl ActionMask for Nim {
    fn action_mask(_: &NimConfig, s: &NimState, _: AgentSlot, out: &mut MaskWriter<'_>) {
        for a in 0..CHOICES {
            out.set(usize::try_from(a).unwrap_or(0), legal(s, a));
        }
    }

    fn noop(_: &NimConfig, s: &NimState, _: AgentSlot, out: &mut ActionWriter<'_>) {
        let first = (0..CHOICES).find(|&a| legal(s, a)).unwrap_or(0);
        out.set_discrete(0, i32::try_from(first).unwrap_or(0));
    }
}

/// The scripted baseline: the first legal move.
impl Heuristic for Nim {
    fn act(_: &NimConfig, s: &NimState, _: AgentSlot, _: &mut StepRng, out: &mut ActionWriter<'_>) {
        let first = (0..CHOICES).find(|&a| legal(s, a)).unwrap_or(0);
        out.set_discrete(0, i32::try_from(first).unwrap_or(0));
    }
}

/// The invariants an imported state must hold (ADR-0040).
impl Validate for Nim {
    fn validate(cfg: &NimConfig, s: &NimState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        if s.heaps.iter().zip(START).any(|(&h, start)| h > start) {
            return invalid("a heap above its start");
        }
        if usize::from(s.turn) >= AGENTS {
            return invalid("a turn past the players");
        }
        if s.step > cfg.episode {
            return invalid("a step past the episode length");
        }
        Ok(())
    }
}

/// The id and version persisted records carry (SPEC 9).
impl Persist for Nim {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
