// SPDX-License-Identifier: MIT OR Apache-2.0

//! Go on a 9 x 9 board (`tessera/Go9-v1`). Cross-platform tier, integer
//! state, perfect information.
//!
//! The rules are Tromp-Taylor (area scoring, no dead stone removal,
//! positional superko), with three documented deviations:
//!
//! 1. Suicide is illegal; Tromp-Taylor allows it.
//! 2. A game stops at [`MAX_PLIES`] plies and is scored as it stands, as a
//!    termination; Tromp-Taylor has no cap. The cap is what lets the
//!    superko history be fixed-size state.
//! 3. Komi is a configured value, in half points, default 7.5; Tromp-Taylor
//!    leaves it to the players.
//!
//! - Agents: slot 0 is black and moves first, slot 1 is white; both are
//!   alive for the whole episode.
//! - Turns: `Sequential` (ADR-0013). The acting agent is slot
//!   `moves % 2`; only its action is read. Nobody acts on a finished game.
//! - Action: `Discrete(82)`: `row * 9 + col` places a stone, and 81 is the
//!   pass. The mask (ADR-0014) marks empty points that are neither suicide
//!   nor a repeat of an earlier position; the pass is always legal, and it
//!   is the no-op. Every slot's mask row describes the side to move, as in
//!   Connect Four: read it together with the acting mask.
//! - Rewards: +1 to the side with the higher score and -1 to the other in
//!   the final step, 0 to both on an exact tie (only an even komi allows
//!   one) and on every other step.
//! - Ends: terminated by two passes in a row or at [`MAX_PLIES`].
//! - Observation: `Box(u8, [2, 9, 9])` from the observer's view: channel 0
//!   holds the observer's stones, channel 1 the opponent's. It omits the
//!   history, so superko shows only through the mask.
//!
//! The state holds exact past positions, not hashes, so legality is never
//! probabilistic. The price is a state of about 3.9 KB, which every
//! snapshot and fork copies.

pub mod board;

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Environment, EpisodeEnd, HeaderView, Heuristic, InvariantViolation,
    NoAdaptive, ObsLeaf, Observation, Phase, RecordHeader, SpaceError, StepOut, Tier,
    TurnStructure, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

use board::{BOARD, POINTS, SIDE, group, neighbours, place};

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/Go9-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;

// Registration under the versioned id (SPEC 10, ADR-0054).
tessera_rl_core::register!(Go9, id = "tessera/Go9-v1");
/// Agent slots.
pub const MAX_AGENTS: usize = 2;
/// The pass action.
pub const PASS: u32 = POINTS;
/// Choices of the action head: one per point and the pass.
pub const ACTIONS: u32 = POINTS + 1;
/// The ply cap: twice the points of the board.
pub const MAX_PLIES: u32 = 2 * POINTS;
/// Entries of the position history: one per placement at most.
pub const HISTORY: usize = MAX_PLIES as usize;

/// Go configuration.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct Go9Config {
    /// Komi in half points, added to white's score; 15 is 7.5.
    #[config(default = 15, range = -162..=162)]
    pub komi_half_points: i32,
}

/// Go state: the stones, the counters and every position so far. The
/// all-zero default is the empty board, both before and after a reset.
#[derive(EnvState, Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Go9State {
    /// Stones of slot 0 and slot 1: 81 bits each, low word first.
    pub stones: [[u64; 2]; 2],
    /// Plies played, passes included; slot `moves % 2` is to move.
    pub moves: u32,
    /// Passes since the last placement; two end the game.
    pub passes_in_row: u32,
    /// History entries in use.
    pub placements: u32,
    /// `placements` right after the last capture, 0 if none.
    pub last_capture: u32,
    /// The position after each placement, packed; unused entries are zero.
    pub history: [[u64; 3]; HISTORY],
}

impl Default for Go9State {
    fn default() -> Self {
        Self {
            stones: [[0; 2]; 2],
            moves: 0,
            passes_in_row: 0,
            placements: 0,
            last_capture: 0,
            history: [[0; 3]; HISTORY],
        }
    }
}

const fn join(words: [u64; 2]) -> u128 {
    (words[0] as u128) | ((words[1] as u128) << 64)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "keeping the low 64 bits is the point"
)]
const fn split(bits: u128) -> [u64; 2] {
    [bits as u64, (bits >> 64) as u64]
}

/// Packs a position into 192 bits: black in the low 81, white above.
#[expect(
    clippy::cast_possible_truncation,
    reason = "keeping the low 64 bits is the point"
)]
#[must_use]
pub const fn pack(black: u128, white: u128) -> [u64; 3] {
    [
        black as u64,
        ((black >> 64) as u64) | ((white as u64) << 17),
        (white >> 47) as u64,
    ]
}

/// The black and white stones of a packed position. Bits beyond the 162
/// that [`pack`] writes are kept in the white set, off the board.
#[must_use]
pub const fn unpack(p: [u64; 3]) -> (u128, u128) {
    let black = (p[0] as u128) | (((p[1] & 0x1_ffff) as u128) << 64);
    let white = ((p[1] >> 17) as u128) | ((p[2] as u128) << 47);
    (black, white)
}

impl Go9State {
    /// The stones of slot 0 and slot 1.
    #[inline]
    #[must_use]
    pub const fn sets(&self) -> (u128, u128) {
        (join(self.stones[0]), join(self.stones[1]))
    }

    /// The slot index to move.
    #[inline]
    #[must_use]
    pub const fn to_move(&self) -> usize {
        (self.moves % 2) as usize
    }

    /// The stones of the side to move, then those of the other side.
    #[inline]
    #[must_use]
    pub const fn sides(&self) -> (u128, u128) {
        let (black, white) = self.sets();
        if self.moves.is_multiple_of(2) {
            (black, white)
        } else {
            (white, black)
        }
    }

    /// Whether the game is over: two passes in a row or the ply cap.
    #[inline]
    #[must_use]
    pub const fn is_over(&self) -> bool {
        self.passes_in_row >= 2 || self.moves >= MAX_PLIES
    }

    /// Whether the position of black and white, reached by placing on an
    /// empty point, has appeared before. Stones leave only on a capture, so
    /// every entry from the last capture on is a subset of the board and
    /// lacks the placed stone: none can match, and those need no look.
    fn seen(&self, black: u128, white: u128) -> bool {
        let n = self.last_capture.saturating_sub(1);
        let packed = pack(black, white);
        self.history
            .get(..n as usize)
            .is_some_and(|h| h.contains(&packed))
    }

    /// The legal placement on the point `sq` for the side to move: its
    /// stones, the opponent's and whether it captures.
    fn placement(&self, sq: u32) -> Option<(u128, u128, bool)> {
        let at = 1u128 << sq;
        let (me, opp) = self.sides();
        if (me | opp) & at != 0 {
            return None;
        }
        let p = place(me, opp, at)?;
        let captured = p.opp != opp;
        let (black, white) = if self.to_move() == 0 {
            (p.me, p.opp)
        } else {
            (p.opp, p.me)
        };
        (!self.seen(black, white)).then_some((black, white, captured))
    }

    /// Whether `action` is legal for the side to move.
    #[must_use]
    pub fn is_legal(&self, action: u32) -> bool {
        action == PASS || (action < PASS && self.placement(action).is_some())
    }

    /// Plays `action` for the side to move; an illegal one is the pass.
    pub fn play(&mut self, action: u32) {
        let placed = if action < PASS {
            self.placement(action)
        } else {
            None
        };
        self.moves += 1;
        let Some((black, white, captured)) = placed else {
            self.passes_in_row += 1;
            return;
        };
        self.stones = [split(black), split(white)];
        if let Some(entry) = self.history.get_mut(self.placements as usize) {
            *entry = pack(black, white);
        }
        self.placements += 1;
        self.passes_in_row = 0;
        if captured {
            self.last_capture = self.placements;
        }
    }

    /// Black's and white's area: stones plus the empty regions that touch
    /// only that colour.
    #[must_use]
    pub const fn areas(&self) -> (u32, u32) {
        let (black, white) = self.sets();
        let empty = BOARD & !(black | white);
        let (mut b, mut w) = (black.count_ones(), white.count_ones());
        let mut rest = empty;
        while rest != 0 {
            let region = group(rest & rest.wrapping_neg(), empty);
            rest &= !region;
            let touching = neighbours(region);
            match (touching & black != 0, touching & white != 0) {
                (true, false) => b += region.count_ones(),
                (false, true) => w += region.count_ones(),
                _ => {}
            }
        }
        (b, w)
    }

    /// Black's score minus white's, in half points.
    #[must_use]
    pub const fn margin_half_points(&self, komi_half_points: i32) -> i32 {
        let (b, w) = self.areas();
        2 * (b.cast_signed() - w.cast_signed()) - komi_half_points
    }
}

/// Go observation.
#[derive(Observation)]
pub struct Go9Obs {
    /// `[channel, row, column]`, row-major; channel 0 is the observer.
    #[space(box(low = 0, high = 1, dtype = u8, shape = [2, 9, 9]))]
    pub board: ObsLeaf<u8>,
}

/// The Go 9x9 environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Go9;

fn slot(index: usize) -> Option<AgentSlot> {
    AgentSlot::new(u16::try_from(index).ok()?, MAX_AGENTS)
}

impl Environment for Go9 {
    type State = Go9State;
    type Config = Go9Config;
    type Obs = Go9Obs;
    type Action = ();
    type Header = RecordHeader<MAX_AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = MAX_AGENTS;
    const TIER: Tier = Tier::CrossPlatform;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &Go9Config) -> Result<Space, SpaceError> {
        Go9Obs::space(cfg)
    }

    fn action_space(_: &Go9Config) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: ACTIONS })
    }

    /// Nobody acts on a finished game (the engine allows it while the
    /// instance waits for its reset).
    fn acting(_: &Go9Config, state: &Go9State, agent: AgentSlot) -> bool {
        agent.index() == state.to_move() && !state.is_over()
    }

    fn reset(_: &Go9Config, _: &mut StepRng, state: &mut Go9State) {
        *state = Go9State::default();
    }

    fn step(
        cfg: &Go9Config,
        state: &mut Go9State,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let Some(mover) = slot(state.to_move()) else {
            return;
        };
        // The engine replaced an illegal action (ADR-0028); `play` keeps
        // the kernel total by treating one as the pass.
        let action = actions
            .discrete(mover, 0)
            .and_then(|a| u32::try_from(a).ok())
            .unwrap_or(PASS);
        state.play(action);
        if state.is_over() {
            if let (Some(b), Some(w)) = (slot(0), slot(1)) {
                let margin = state.margin_half_points(cfg.komi_half_points);
                let reward = f32::from(i8::from(margin > 0) - i8::from(margin < 0));
                out.set_reward(b, reward);
                out.set_reward(w, -reward);
            }
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &Go9Config,
        state: &Go9State,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, Go9Obs>,
    ) {
        let (black, white) = state.sets();
        let planes = if agent.index() == 0 {
            [black, white]
        } else {
            [white, black]
        };
        let Some(mut leaf) = out.leaf_mut(Go9Obs::BOARD) else {
            return;
        };
        for (channel, stones) in planes.iter().enumerate() {
            for sq in 0..POINTS {
                let k = channel * POINTS as usize + sq as usize;
                leaf.set(k, u8::from(stones & (1u128 << sq) != 0));
            }
        }
    }
}

/// Empty points that are neither suicide nor a repeat; the pass is always
/// legal and is the no-op.
impl ActionMask for Go9 {
    fn action_mask(_: &Go9Config, state: &Go9State, _: AgentSlot, out: &mut MaskWriter<'_>) {
        for sq in 0..POINTS {
            out.set(sq as usize, state.placement(sq).is_some());
        }
        out.set(PASS as usize, true);
    }

    fn noop(_: &Go9Config, _: &Go9State, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, PASS.cast_signed());
    }
}

/// One replayed placement: the mover (0 or 1) that turns `before` into
/// `after` by the rules, and whether it captured.
fn replayed_mover(before: (u128, u128), after: (u128, u128)) -> Option<(usize, bool)> {
    let (b0, w0) = before;
    let (b1, w1) = after;
    let (mover, at) = if b1 & !b0 != 0 {
        (0, b1 & !b0)
    } else {
        (1, w1 & !w0)
    };
    if !at.is_power_of_two() || (b0 | w0) & at != 0 {
        return None;
    }
    let (me, opp, me_after, opp_after) = if mover == 0 {
        (b0, w0, b1, w1)
    } else {
        (w0, b0, w1, b1)
    };
    let p = place(me, opp, at)?;
    (p.me == me_after && p.opp == opp_after).then_some((mover, p.opp != opp))
}

/// Imported states: the history replays from the empty board as legal
/// placements with no position repeated, at most one pass fits between two
/// placements (two would have ended the game), and the board, the plies
/// and every counter are what that replay gives. So every accepted state
/// is one a game reaches.
impl Validate for Go9 {
    fn validate(_: &Go9Config, state: &Go9State) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        if state.moves > MAX_PLIES || state.placements as usize > HISTORY {
            return invalid("more plies or placements than the cap");
        }
        if state.passes_in_row > 2 {
            return invalid("more than two passes in a row");
        }
        let n = state.placements as usize;
        let Some((used, unused)) = state.history.split_at_checked(n) else {
            return invalid("more placements than the history holds");
        };
        if unused.iter().any(|e| *e != [0; 3]) {
            return invalid("a non-zero unused history entry");
        }
        let mut board = (0u128, 0u128);
        let (mut plies, mut last_capture) = (0u32, 0u32);
        for (k, entry) in used.iter().enumerate() {
            let after = unpack(*entry);
            if (after.0 | after.1) & !BOARD != 0 || pack(after.0, after.1) != *entry {
                return invalid("a history entry off the board");
            }
            if used.get(..k).is_some_and(|earlier| earlier.contains(entry)) {
                return invalid("a position repeated in the history");
            }
            let Some((mover, captured)) = replayed_mover(board, after) else {
                return invalid("a history entry that is not one legal placement");
            };
            // A mover out of turn means the other side passed first.
            if (plies % 2) as usize != mover {
                plies += 1;
            }
            plies += 1;
            board = after;
            if captured {
                last_capture = u32::try_from(k).unwrap_or(0) + 1;
            }
        }
        if state.sets() != board {
            return invalid("stones other than the last history entry");
        }
        if state.last_capture != last_capture {
            return invalid("a last capture that the history does not give");
        }
        if state.moves != plies + state.passes_in_row {
            return invalid("plies that the history and the passes do not give");
        }
        Ok(())
    }

    /// ADR-0040: before the first reset the state is exactly the initial
    /// value; once reset, both slots are alive. A running episode on a
    /// finished game needs no check here: nobody acts on it, and the
    /// engine refuses a running sequential record without an actor.
    fn validate_record(
        cfg: &Go9Config,
        header: &HeaderView<'_>,
        state: &Go9State,
    ) -> Result<(), InvariantViolation> {
        if header.phase() == Phase::NeverReset {
            if *state != Go9State::default() {
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

/// The distance of `sq` from the centre point, in rows plus columns.
const fn centre_distance(sq: u32) -> u32 {
    let (col, row) = (sq % SIDE, sq / SIDE);
    col.abs_diff(SIDE / 2) + row.abs_diff(SIDE / 2)
}

/// Whether the mover's stones next to `at` include a group in atari that
/// placing there leaves with more than one liberty.
fn rescues(me: u128, opp: u128, at: u128, after_me: u128, after_opp: u128) -> bool {
    let empty = BOARD & !(me | opp);
    let mut touching = neighbours(at) & me;
    while touching != 0 {
        let g = group(touching & touching.wrapping_neg(), me);
        touching &= !g;
        if (neighbours(g) & empty).is_power_of_two() {
            let free = BOARD & !(after_me | after_opp);
            return (neighbours(group(at, after_me)) & free).count_ones() > 1;
        }
    }
    false
}

/// Scripted baseline (deterministic, one ply): the legal placement that
/// captures the most stones; else one that saves an own group in atari;
/// else the legal placement nearest the centre that does not fill a point
/// whose neighbours are all the mover's own; ties by lowest index; else
/// the pass.
impl Heuristic for Go9 {
    fn act(
        _: &Go9Config,
        state: &Go9State,
        _: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let (me, opp) = state.sides();
        let mut best = PASS;
        let mut best_key = (0, false, i32::MIN);
        for sq in 0..POINTS {
            let Some((black, white, _)) = state.placement(sq) else {
                continue;
            };
            let at = 1u128 << sq;
            let (after_me, after_opp) = if state.to_move() == 0 {
                (black, white)
            } else {
                (white, black)
            };
            let captured = (opp & !after_opp).count_ones();
            let rescue = rescues(me, opp, at, after_me, after_opp);
            let eye = neighbours(at) & !me == 0;
            if eye && captured == 0 && !rescue {
                continue;
            }
            let key = (captured, rescue, -centre_distance(sq).cast_signed());
            if key > best_key {
                best = sq;
                best_key = key;
            }
        }
        out.set_discrete(0, best.cast_signed());
    }
}

impl tessera_rl_core::Persist for Go9 {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
