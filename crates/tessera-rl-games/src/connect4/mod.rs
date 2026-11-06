// SPDX-License-Identifier: MIT OR Apache-2.0

//! Connect Four (`tessera/Connect4-v1`): two players drop stones into a
//! 7 x 6 board in turns; four in a row (vertical, horizontal or
//! diagonal) wins. Cross-platform tier, integer state, perfect
//! information.
//!
//! - Agents: slot 0 moves first, slot 1 second; both are alive for the
//!   whole episode.
//! - Turns: `Sequential` (ADR-0013). The acting agent is slot
//!   `moves % 2`; only its action is read. Nobody acts on a finished
//!   board.
//! - Action: `Discrete(7)`, the column (0 is the leftmost). The mask
//!   (ADR-0014) marks full columns illegal; the no-op is the first
//!   non-full column in centre-first order.
//! - Rewards: +1 to the winner and -1 to the loser in the step that
//!   completes four; 0 to both on a draw (a full board) and on every
//!   other step.
//! - Ends: terminated on a win or a full board. The game needs no step
//!   limit (at most 42 plies).
//! - Observation: `Box(u8, [2, 6, 7])` from the observer's view: channel
//!   0 holds the observer's stones, channel 1 the opponent's; row 0 is the
//!   top row, column 0 the leftmost.

pub mod board;
pub mod bot;

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Environment, EpisodeEnd, HeaderView, Heuristic, InvariantViolation,
    NoAdaptive, ObsLeaf, Observation, Phase, RecordHeader, Reserved, SpaceError, StepOut, Tier,
    TurnStructure, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

use board::{BOARD, CELLS, CENTRE_ORDER, HEIGHT, WIDTH, can_play, cell, has_four, landing};

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/Connect4-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;
/// Agent slots.
pub const MAX_AGENTS: usize = 2;
/// Choices of the action head: one per column.
pub const COLUMNS: u32 = WIDTH;

/// Connect Four has no parameters.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct Connect4Config {}

/// Connect Four state: each player's stones and the plies played.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Connect4State {
    /// Stones of slot 0 and slot 1, as bitboards (see [`board`]).
    pub stones: [u64; 2],
    /// Plies played in this episode; slot `moves % 2` is to move.
    pub moves: u32,
    /// Padding made explicit (ADR-0029).
    pub reserved: Reserved<4>,
}

impl Connect4State {
    /// Every occupied cell.
    #[must_use]
    pub const fn occupied(&self) -> u64 {
        self.stones[0] | self.stones[1]
    }

    /// The slot index to move.
    #[must_use]
    pub const fn to_move(&self) -> usize {
        (self.moves % 2) as usize
    }

    /// Whether the game is over: a player holds four or the board is full.
    #[must_use]
    pub const fn is_over(&self) -> bool {
        has_four(self.stones[0]) || has_four(self.stones[1]) || self.moves >= CELLS
    }

    /// Plays column `col` for the side to move; the caller checked that
    /// it has room. Returns the cell played.
    pub fn play(&mut self, col: u32) -> u64 {
        let at = landing(self.occupied(), col);
        let side = self.to_move();
        if let Some(s) = self.stones.get_mut(side) {
            *s |= at;
        }
        self.moves += 1;
        at
    }

    /// Whether `side` (a slot index) wins at once by playing `col`.
    #[must_use]
    pub fn wins_with(&self, col: u32, side: usize) -> bool {
        let occupied = self.occupied();
        can_play(occupied, col)
            && self
                .stones
                .get(side)
                .is_some_and(|&s| has_four(s | landing(occupied, col)))
    }
}

/// Connect Four observation.
#[derive(Observation)]
pub struct Connect4Obs {
    /// `[channel, row, column]`, row-major; channel 0 is the observer.
    #[space(box(low = 0, high = 1, dtype = u8, shape = [2, 6, 7]))]
    pub board: ObsLeaf<u8>,
}

/// The Connect Four environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Connect4;

fn slot(index: usize) -> Option<AgentSlot> {
    AgentSlot::new(u16::try_from(index).ok()?, MAX_AGENTS)
}

/// The first non-full column in centre-first order.
fn first_legal(occupied: u64) -> u32 {
    CENTRE_ORDER
        .iter()
        .copied()
        .find(|&c| can_play(occupied, c))
        .unwrap_or(0)
}

impl Environment for Connect4 {
    type State = Connect4State;
    type Config = Connect4Config;
    type Obs = Connect4Obs;
    type Action = ();
    type Header = RecordHeader<MAX_AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = MAX_AGENTS;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &Connect4Config) -> Result<Space, SpaceError> {
        Connect4Obs::space(cfg)
    }

    fn action_space(_: &Connect4Config) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: COLUMNS })
    }

    /// Nobody acts on a finished board (the engine allows it while the
    /// instance waits for its reset).
    fn acting(_: &Connect4Config, state: &Connect4State, agent: AgentSlot) -> bool {
        agent.index() == state.to_move() && !state.is_over()
    }

    fn reset(_: &Connect4Config, _: &mut StepRng, state: &mut Connect4State) {
        *state = Connect4State::default();
    }

    fn step(
        _: &Connect4Config,
        state: &mut Connect4State,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let me = state.to_move();
        let (Some(mover), Some(other)) = (slot(me), slot(1 - me)) else {
            return;
        };
        let occupied = state.occupied();
        // The engine replaced an illegal column (ADR-0028); a full column
        // cannot reach the kernel, but the kernel stays total.
        let col = actions
            .discrete(mover, 0)
            .and_then(|a| u32::try_from(a).ok())
            .filter(|&c| can_play(occupied, c))
            .unwrap_or_else(|| first_legal(occupied));
        state.play(col);
        if state.stones.get(me).is_some_and(|&s| has_four(s)) {
            out.set_reward(mover, 1.0);
            out.set_reward(other, -1.0);
            out.end_episode(EpisodeEnd::Terminated);
        } else if state.moves >= CELLS {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &Connect4Config,
        state: &Connect4State,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, Connect4Obs>,
    ) {
        let me = agent.index().min(1);
        let planes = [
            state.stones.get(me).copied().unwrap_or(0),
            state.stones.get(1 - me).copied().unwrap_or(0),
        ];
        let Some(mut leaf) = out.leaf_mut(Connect4Obs::BOARD) else {
            return;
        };
        let (w, h) = (WIDTH as usize, HEIGHT as usize);
        for (channel, stones) in planes.iter().enumerate() {
            for top_row in 0..HEIGHT {
                for col in 0..WIDTH {
                    let row = HEIGHT - 1 - top_row;
                    let k = channel * h * w + top_row as usize * w + col as usize;
                    leaf.set(k, u8::from(stones & cell(col, row) != 0));
                }
            }
        }
    }
}

/// Full columns are illegal; the no-op is the first non-full column in
/// centre-first order.
impl ActionMask for Connect4 {
    fn action_mask(
        _: &Connect4Config,
        state: &Connect4State,
        _: AgentSlot,
        out: &mut MaskWriter<'_>,
    ) {
        let occupied = state.occupied();
        for col in 0..WIDTH {
            out.set(col as usize, can_play(occupied, col));
        }
    }

    fn noop(_: &Connect4Config, state: &Connect4State, _: AgentSlot, out: &mut ActionWriter<'_>) {
        let col = first_legal(state.occupied());
        out.set_discrete(0, i32::try_from(col).unwrap_or(0));
    }
}

/// Imported states: disjoint stones that obey gravity, stone counts that
/// match the plies and the turn, and no four for the side to move. The
/// side that just moved may hold four only if its last stone made them
/// all: some stone on top of its column lies in every one of its fours.
/// Otherwise play went on after a win, which no game reaches.
impl Validate for Connect4 {
    fn validate(_: &Connect4Config, state: &Connect4State) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let [a, b] = state.stones;
        if a & b != 0 || (a | b) & !BOARD != 0 || !board::is_stacked(a | b) {
            return invalid("stones overlap, leave the board or float");
        }
        let (na, nb) = (a.count_ones(), b.count_ones());
        if state.moves != na + nb || (na != nb && na != nb + 1) {
            return invalid("stone counts that do not match the plies");
        }
        let to_move = state.stones.get(state.to_move()).copied().unwrap_or(0);
        if has_four(to_move) {
            return invalid("the side to move already holds four");
        }
        let mover = state.stones.get(1 - state.to_move()).copied().unwrap_or(0);
        if has_four(mover) && !ended_on_last_move(mover, a | b) {
            return invalid("play went on after a four");
        }
        Ok(())
    }

    /// ADR-0040: once reset, both slots are alive, and a running episode
    /// is not a finished game.
    fn validate_record(
        cfg: &Connect4Config,
        header: &HeaderView<'_>,
        state: &Connect4State,
    ) -> Result<(), InvariantViolation> {
        Self::validate(cfg, state)?;
        let phase = header.phase();
        if phase == Phase::NeverReset {
            return Ok(());
        }
        let both_alive = (0..MAX_AGENTS).all(|k| header.agent(k).is_some_and(|v| v.alive));
        if !both_alive {
            return Err(InvariantViolation {
                code: "a player that is not alive",
            });
        }
        if phase == Phase::Active && state.is_over() {
            return Err(InvariantViolation {
                code: "a running episode on a finished board",
            });
        }
        Ok(())
    }
}

/// Whether some stone of `mover` on top of its column lies in every four
/// of `mover` (removing it leaves none): the four were made by the last
/// stone played.
fn ended_on_last_move(mover: u64, occupied: u64) -> bool {
    let mut rest = mover;
    while rest != 0 {
        let stone = rest & rest.wrapping_neg();
        rest &= rest - 1;
        // The cell above is empty, or is the column's sentinel bit.
        let on_top = (stone << 1) & occupied == 0;
        if on_top && !has_four(mover & !stone) {
            return true;
        }
    }
    false
}

/// Scripted baseline (not the P2-G1 bot): win at once if possible, else
/// block the opponent's immediate win, else the first non-full column in
/// centre-first order.
impl Heuristic for Connect4 {
    fn act(
        _: &Connect4Config,
        state: &Connect4State,
        agent: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let me = agent.index().min(1);
        let pick = |side: usize| {
            CENTRE_ORDER
                .iter()
                .copied()
                .find(|&c| state.wins_with(c, side))
        };
        let col = pick(me)
            .or_else(|| pick(1 - me))
            .unwrap_or_else(|| first_legal(state.occupied()));
        out.set_discrete(0, i32::try_from(col).unwrap_or(0));
    }
}

impl tessera_rl_core::Persist for Connect4 {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
