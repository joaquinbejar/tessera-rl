// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hidden Dice: a two-player imperfect-information test game that draws
//! chance at step time from `StepRng` (a die) and from `ObsRng`
//! (observation noise).
//!
//! - Each player holds a private card in `0..CARDS`, dealt at reset.
//! - Players move in turns (`Sequential`), `Discrete(2)`; a move adds the
//!   action and a fresh die roll in `0..DIE` to the public pile.
//! - Each move pays the mover `card / 100`; the last move pays +1 to the
//!   higher card and -1 to the lower (0 each on a tie) instead.
//! - The observation of an agent is its own card, the last die, the pile
//!   and one noise value in `0..NOISE` drawn from `ObsRng`.
//! - The information state of an agent is its slot, its card, the moves,
//!   the last die and the pile: the opponent's card is hidden.

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Determinize, Environment, EpisodeEnd, ImperfectInformation,
    InvariantViolation, NoAdaptive, ObsLeaf, Observation, RecordHeader, SpaceError, StepOut, Tier,
    TurnStructure, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

pub const CARDS: u32 = 4;
pub const DIE: u32 = 6;
pub const NOISE: u32 = 4;
pub const MOVES: u32 = 6;

#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct DiceConfig {
    /// A broken sampler for the refusal tests: 0 none, 1 a sample with
    /// another information state, 2 a sample in which nobody acts.
    #[config(default = 0, range = 0..=2)]
    pub sabotage: u32,
}

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct DiceState {
    pub cards: [u32; 2],
    pub moves: u32,
    pub die: u32,
    pub pile: u32,
    /// Hidden; 1 stops every turn (only a sabotaged sampler sets it).
    pub stall: u32,
}

#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct DiceInfo {
    pub me: u32,
    pub card: u32,
    pub moves: u32,
    pub die: u32,
    pub pile: u32,
}

#[derive(Observation)]
pub struct DiceObs {
    /// Own card, last die, pile, noise.
    #[space(box(low = 0, high = 1000, dtype = u32, shape = [4]))]
    pub view: ObsLeaf<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Dice;

fn bound(n: u32) -> core::num::NonZeroU32 {
    core::num::NonZeroU32::new(n).unwrap_or(core::num::NonZeroU32::MIN)
}

fn slot(i: usize) -> Option<AgentSlot> {
    AgentSlot::new(u16::try_from(i).ok()?, 2)
}

impl Environment for Dice {
    type State = DiceState;
    type Config = DiceConfig;
    type Obs = DiceObs;
    type Action = ();
    type Header = RecordHeader<2, NoAdaptive>;

    const MAX_AGENTS: usize = 2;
    const TIER: Tier = Tier::CrossPlatform;
    const PERFECT_INFORMATION: bool = false;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &DiceConfig) -> Result<Space, SpaceError> {
        DiceObs::space(cfg)
    }

    fn action_space(_: &DiceConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: 2 })
    }

    fn acting(_: &DiceConfig, state: &DiceState, agent: AgentSlot) -> bool {
        state.stall == 0 && state.moves < MOVES && agent.index() == (state.moves % 2) as usize
    }

    fn reset(_: &DiceConfig, rng: &mut StepRng, state: &mut DiceState) {
        *state = DiceState {
            cards: [rng.below_u32(bound(CARDS)), rng.below_u32(bound(CARDS))],
            ..DiceState::default()
        };
    }

    fn step(
        _: &DiceConfig,
        state: &mut DiceState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let me = (state.moves % 2) as usize;
        let (Some(mover), Some(a0), Some(a1)) = (slot(me), slot(0), slot(1)) else {
            return;
        };
        let action = actions
            .discrete(mover, 0)
            .and_then(|a| u32::try_from(a).ok())
            .unwrap_or(0)
            .min(1);
        state.die = rng.below_u32(bound(DIE));
        state.pile += action + state.die;
        state.moves += 1;
        if state.moves >= MOVES {
            let [c0, c1] = state.cards;
            let r = match c0.cmp(&c1) {
                core::cmp::Ordering::Greater => 1.0,
                core::cmp::Ordering::Less => -1.0,
                core::cmp::Ordering::Equal => 0.0,
            };
            out.set_reward(a0, r);
            out.set_reward(a1, -r);
            out.end_episode(EpisodeEnd::Terminated);
        } else {
            let card = state.cards.get(me).copied().unwrap_or(0);
            out.set_reward(mover, f32::from(u16::try_from(card).unwrap_or(0)) / 100.0);
        }
    }

    fn observe(
        _: &DiceConfig,
        state: &DiceState,
        agent: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, DiceObs>,
    ) {
        let card = state.cards.get(agent.index()).copied().unwrap_or(0);
        let noise = rng.below_u32(bound(NOISE));
        let Some(mut leaf) = out.leaf_mut(DiceObs::VIEW) else {
            return;
        };
        for (k, v) in [card, state.die, state.pile, noise].into_iter().enumerate() {
            leaf.set(k, v);
        }
    }
}

impl ActionMask for Dice {
    fn action_mask(_: &DiceConfig, _: &DiceState, _: AgentSlot, out: &mut MaskWriter<'_>) {
        out.set(0, true);
        out.set(1, true);
    }

    fn noop(_: &DiceConfig, _: &DiceState, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, 0);
    }
}

impl Validate for Dice {
    fn validate(_: &DiceConfig, s: &DiceState) -> Result<(), InvariantViolation> {
        if s.cards.iter().any(|&c| c >= CARDS) || s.die >= DIE || s.moves > MOVES || s.stall > 1 {
            return Err(InvariantViolation {
                code: "a value out of range",
            });
        }
        Ok(())
    }
}

impl ImperfectInformation for Dice {
    type InfoState = DiceInfo;

    fn info_state(_: &DiceConfig, s: &DiceState, agent: AgentSlot) -> DiceInfo {
        let me = agent.index().min(1);
        DiceInfo {
            me: u32::try_from(me).unwrap_or(0),
            card: s.cards.get(me).copied().unwrap_or(0),
            moves: s.moves,
            die: s.die,
            pile: s.pile,
        }
    }
}

impl Determinize for Dice {
    fn sample(cfg: &DiceConfig, info: &DiceInfo, rng: &mut StepRng, state: &mut DiceState) {
        let me = (info.me as usize).min(1);
        let mut cards = [0; 2];
        if let Some(c) = cards.get_mut(me) {
            *c = info.card;
        }
        if let Some(c) = cards.get_mut(1 - me) {
            *c = rng.below_u32(bound(CARDS));
        }
        *state = DiceState {
            cards,
            moves: info.moves,
            die: info.die,
            pile: info.pile + u32::from(cfg.sabotage == 1),
            stall: u32::from(cfg.sabotage == 2),
        };
    }
}
