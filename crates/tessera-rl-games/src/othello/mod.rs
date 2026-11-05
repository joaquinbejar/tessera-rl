// SPDX-License-Identifier: MIT OR Apache-2.0

//! Othello (`tessera/Othello-v1`): two players place discs on an 8 x 8
//! board, flipping every straight line of opposing discs each placement
//! brackets; the side with more discs at the end wins. Cross-platform
//! tier, integer state, perfect information.
//!
//! - Start: white on `d4` and `e5`, black on `e4` and `d5`, with file `a`
//!   on the left and rank 1 on top.
//! - Agents: slot 0 is black and moves first, slot 1 is white; both are
//!   alive for the whole episode.
//! - Turns: `Sequential` (ADR-0013). The acting agent is slot
//!   `moves % 2`; only its action is read. Nobody acts on a finished
//!   board.
//! - Action: `Discrete(65)`: `row * 8 + col` places a disc (`a1` is 0 and
//!   `h8` is 63), and 64 is the pass. The mask (ADR-0014) marks the legal
//!   placements; the pass is legal exactly when there are none, so it is
//!   forced. The no-op is the lowest legal square, else the pass.
//! - Rewards: +1 to the side with more discs and -1 to the other in the
//!   final step; 0 to both on equal counts and on every other step.
//!   Empty squares are not awarded.
//! - Ends: terminated as soon as neither side can place, so no episode
//!   ends on a pass and two passes never follow each other. A game lasts
//!   at most [`MAX_PLIES`] plies: 60 placements and a pass in at most each
//!   of the 59 gaps between them.
//! - Observation: `Box(u8, [2, 8, 8])` from the observer's view: channel
//!   0 holds the observer's discs, channel 1 the opponent's; row 0 is the
//!   top rank, column 0 the left file.

pub mod board;

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Environment, EpisodeEnd, HeaderView, Heuristic, InvariantViolation,
    NoAdaptive, ObsLeaf, Observation, Phase, RecordHeader, Reserved, SpaceError, StepOut, Tier,
    TurnStructure, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

use board::{
    CENTRE, DIRECTIONS, SIDE, SQUARES, START_BLACK, START_WHITE, flips, is_connected,
    legal_squares, shift,
};

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/Othello-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;
/// Agent slots.
pub const MAX_AGENTS: usize = 2;
/// The pass action.
pub const PASS: u32 = SQUARES;
/// Choices of the action head: one per square and the pass.
pub const ACTIONS: u32 = SQUARES + 1;
/// Discs on the starting board.
pub const START_DISCS: u32 = 4;
/// The longest game in plies: 60 placements and at most 59 passes.
pub const MAX_PLIES: u32 = 119;

/// Othello has no parameters.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct OthelloConfig {}

/// Othello state: each player's discs and the plies played. The all-zero
/// default is the value before the first reset, not a board.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct OthelloState {
    /// Discs of slot 0 and slot 1, as bitboards (see [`board`]).
    pub stones: [u64; 2],
    /// Plies played in this episode, passes included; slot `moves % 2` is
    /// to move.
    pub moves: u32,
    /// Padding made explicit (ADR-0029).
    pub reserved: Reserved<4>,
}

impl OthelloState {
    /// The starting board.
    #[must_use]
    pub fn start() -> Self {
        Self {
            stones: [START_BLACK, START_WHITE],
            ..Self::default()
        }
    }

    /// The slot index to move.
    #[inline]
    #[must_use]
    pub const fn to_move(&self) -> usize {
        (self.moves % 2) as usize
    }

    /// The discs of the side to move, then those of the other side.
    #[inline]
    #[must_use]
    pub const fn sides(&self) -> (u64, u64) {
        let [black, white] = self.stones;
        if self.moves.is_multiple_of(2) {
            (black, white)
        } else {
            (white, black)
        }
    }

    /// The squares where the side to move may place.
    #[inline]
    #[must_use]
    pub const fn legal(&self) -> u64 {
        let (me, opp) = self.sides();
        legal_squares(me, opp)
    }

    /// Whether the game is over: neither side can place.
    #[inline]
    #[must_use]
    pub const fn is_over(&self) -> bool {
        let (me, opp) = self.sides();
        legal_squares(me, opp) == 0 && legal_squares(opp, me) == 0
    }

    /// Plays `action` for the side to move; the caller checked that it is
    /// legal. A pass only hands over the turn.
    pub fn play(&mut self, action: u32) {
        if action < SQUARES {
            let at = 1u64 << action;
            let (me, opp) = self.sides();
            let flipped = flips(me, opp, at);
            let (mine, theirs) = (me | at | flipped, opp & !flipped);
            self.stones = if self.to_move() == 0 {
                [mine, theirs]
            } else {
                [theirs, mine]
            };
        }
        self.moves += 1;
    }

    /// The lowest legal action of the side to move.
    #[inline]
    #[must_use]
    pub const fn first_legal(&self) -> u32 {
        let legal = self.legal();
        if legal == 0 {
            PASS
        } else {
            legal.trailing_zeros()
        }
    }

    /// Whether `action` is legal for the side to move.
    #[inline]
    #[must_use]
    pub const fn is_legal(&self, action: u32) -> bool {
        let legal = self.legal();
        if action < SQUARES {
            legal & (1u64 << action) != 0
        } else {
            action == PASS && legal == 0
        }
    }
}

/// Othello observation.
#[derive(Observation)]
pub struct OthelloObs {
    /// `[channel, row, column]`, row-major; channel 0 is the observer.
    #[space(box(low = 0, high = 1, dtype = u8, shape = [2, 8, 8]))]
    pub board: ObsLeaf<u8>,
}

/// The Othello environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Othello;

fn slot(index: usize) -> Option<AgentSlot> {
    AgentSlot::new(u16::try_from(index).ok()?, MAX_AGENTS)
}

impl Environment for Othello {
    type State = OthelloState;
    type Config = OthelloConfig;
    type Obs = OthelloObs;
    type Action = ();
    type Header = RecordHeader<MAX_AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = MAX_AGENTS;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &OthelloConfig) -> Result<Space, SpaceError> {
        OthelloObs::space(cfg)
    }

    fn action_space(_: &OthelloConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: ACTIONS })
    }

    /// Nobody acts on a finished board (the engine allows it while the
    /// instance waits for its reset).
    fn acting(_: &OthelloConfig, state: &OthelloState, agent: AgentSlot) -> bool {
        agent.index() == state.to_move() && !state.is_over()
    }

    fn reset(_: &OthelloConfig, _: &mut StepRng, state: &mut OthelloState) {
        *state = OthelloState::start();
    }

    fn step(
        _: &OthelloConfig,
        state: &mut OthelloState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let Some(mover) = slot(state.to_move()) else {
            return;
        };
        // The engine replaced an illegal action (ADR-0028); one cannot
        // reach the kernel, but the kernel stays total.
        let action = actions
            .discrete(mover, 0)
            .and_then(|a| u32::try_from(a).ok())
            .filter(|&a| state.is_legal(a))
            .unwrap_or_else(|| state.first_legal());
        state.play(action);
        if state.is_over() {
            let [black, white] = state.stones.map(u64::count_ones);
            if let (Some(b), Some(w)) = (slot(0), slot(1)) {
                let reward = f32::from(i8::from(black > white) - i8::from(black < white));
                out.set_reward(b, reward);
                out.set_reward(w, -reward);
            }
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &OthelloConfig,
        state: &OthelloState,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, OthelloObs>,
    ) {
        let me = agent.index().min(1);
        let planes = [
            state.stones.get(me).copied().unwrap_or(0),
            state.stones.get(1 - me).copied().unwrap_or(0),
        ];
        let Some(mut leaf) = out.leaf_mut(OthelloObs::BOARD) else {
            return;
        };
        for (channel, discs) in planes.iter().enumerate() {
            for sq in 0..SQUARES {
                let k = channel * SQUARES as usize + sq as usize;
                leaf.set(k, u8::from(discs & (1u64 << sq) != 0));
            }
        }
    }
}

/// The legal placements, and the pass exactly when there are none; the
/// no-op is the lowest legal action.
impl ActionMask for Othello {
    fn action_mask(
        _: &OthelloConfig,
        state: &OthelloState,
        _: AgentSlot,
        out: &mut MaskWriter<'_>,
    ) {
        let legal = state.legal();
        for sq in 0..SQUARES {
            out.set(sq as usize, legal & (1u64 << sq) != 0);
        }
        out.set(PASS as usize, legal == 0);
    }

    fn noop(_: &OthelloConfig, state: &OthelloState, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, i32::try_from(state.first_legal()).unwrap_or(0));
    }
}

/// Whether the side that just moved holds a disc it could have placed
/// last, with two of its discs in line next to it (the flipped disc and the
/// one that bracketed it). The centre need not be excluded: such a line has
/// two ends two squares apart, and the centre cannot hold both.
const fn could_have_placed(mover: u64) -> bool {
    let mut dir = 0;
    while dir < DIRECTIONS {
        let next = shift(mover, dir);
        if mover & next & shift(next, dir) != 0 {
            return true;
        }
        dir += 1;
    }
    false
}

/// Imported states: disjoint discs that hold the centre and form one
/// connected group, plies that match the discs (every ply a placement or a
/// pass, and at most one pass per placement), the exact starting board at
/// ply 0, and a last ply that the board can explain. That ply was a
/// placement by the side not to move, which leaves fewer passes than
/// placements (a pass only fills a gap between two placements), or its
/// pass, which needs no placement for it and one for the side to move. So
/// no accepted state exceeds [`MAX_PLIES`]. These
/// are necessary conditions, not a proof that a game reaches the state.
impl Validate for Othello {
    fn validate(_: &OthelloConfig, state: &OthelloState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let [a, b] = state.stones;
        let occupied = a | b;
        if a & b != 0 {
            return invalid("discs of both sides on one square");
        }
        if occupied & CENTRE != CENTRE {
            return invalid("an empty centre square");
        }
        if !is_connected(occupied) {
            return invalid("discs that are not one connected group");
        }
        let placements = occupied.count_ones() - START_DISCS;
        let Some(passes) = state.moves.checked_sub(placements) else {
            return invalid("fewer plies than placements");
        };
        if passes > placements {
            return invalid("more passes than placements");
        }
        if state.moves == 0 {
            return if state.stones == [START_BLACK, START_WHITE] {
                Ok(())
            } else {
                invalid("a board other than the start at ply 0")
            };
        }
        let (me, opp) = state.sides();
        let by_placement = passes < placements && could_have_placed(opp);
        let by_pass = passes > 0 && legal_squares(opp, me) == 0 && legal_squares(me, opp) != 0;
        if !by_placement && !by_pass {
            return invalid("a last ply that the board cannot explain");
        }
        Ok(())
    }

    /// ADR-0040: before the first reset the state is exactly the initial
    /// value; once reset, both slots are alive. A running episode on a
    /// finished board needs no check here: nobody acts on it, and the
    /// engine refuses a running sequential record without an actor.
    fn validate_record(
        cfg: &OthelloConfig,
        header: &HeaderView<'_>,
        state: &OthelloState,
    ) -> Result<(), InvariantViolation> {
        if header.phase() == Phase::NeverReset {
            if *state != OthelloState::default() {
                return Err(InvariantViolation {
                    code: "a state other than the initial value before the first reset",
                });
            }
            return Ok(());
        }
        Self::validate(cfg, state)?;
        if !(0..MAX_AGENTS).all(|k| header.agent(k).is_some_and(|v| v.alive)) {
            return Err(InvariantViolation {
                code: "a player that is not alive",
            });
        }
        Ok(())
    }
}

/// The baseline's value of placing on `sq`: corners are worth most, the
/// squares that give a corner away least.
#[must_use]
pub const fn square_weight(sq: u32) -> i32 {
    let (col, row) = (sq % SIDE, sq / SIDE);
    let c = if col < SIDE - col {
        col
    } else {
        SIDE - 1 - col
    };
    let r = if row < SIDE - row {
        row
    } else {
        SIDE - 1 - row
    };
    match (c, r) {
        (0, 0) => 100,
        (1, 1) => -50,
        (0, 1) | (1, 0) => -20,
        (0, _) | (_, 0) => 10,
        _ => 1,
    }
}

/// Scripted baseline (deterministic, one ply): the legal square with the
/// highest [`square_weight`], then the most discs flipped, then the lowest
/// index; the pass when forced.
impl Heuristic for Othello {
    fn act(
        _: &OthelloConfig,
        state: &OthelloState,
        _: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let (me, opp) = state.sides();
        let mut rest = legal_squares(me, opp);
        let mut best = PASS;
        let mut best_key = (i32::MIN, 0);
        while rest != 0 {
            let sq = rest.trailing_zeros();
            let at = rest & rest.wrapping_neg();
            rest &= rest - 1;
            let key = (square_weight(sq), flips(me, opp, at).count_ones());
            if key > best_key {
                best = sq;
                best_key = key;
            }
        }
        out.set_discrete(0, i32::try_from(best).unwrap_or(0));
    }
}
