// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lanterns (`tessera/Lanterns-v1`): a cooperative card game with hidden
//! hands, an original implementation of the public "hidden-hand fireworks"
//! mechanics (SPEC 16, ADR-0034). Cross-platform tier, integer state,
//! imperfect information.
//!
//! - **Cards:** 5 colours, ranks 1 to 5 (stored 0 to 4), three copies of
//!   rank 1, two of ranks 2 to 4 and one of rank 5: 50 cards. A card is
//!   `colour * 5 + rank`; [`NO_CARD`] marks an empty hand slot.
//! - **Players:** 2 to 5 (`players`, the alive slots `0..players`).
//!   Hands hold 5 cards with 2 or 3 players and 4 with 4 or 5.
//! - **Turns:** `Sequential` (ADR-0013): one player acts, then the next.
//! - **Actions:** `Discrete(50)`. `0..5` play the card in a slot, `5..10`
//!   discard it, `10..30` hint a colour and `30..50` hint a rank: for
//!   `10 + 5 * (o - 1) + c` the target is the player `o` seats after the
//!   actor (1 to 4) and `c` the colour or rank. The mask (ADR-0014) marks a
//!   play or a discard of an empty slot, a discard with all 8 hint tokens,
//!   and a hint without tokens, to an absent player, or touching no card
//!   as illegal; the no-op is the play of the first slot holding a card.
//! - **Hints** say which cards of the target's hand have that colour (or
//!   rank) and, by omission, which do not: the target's knowledge of each
//!   slot (the colours and ranks still possible) narrows both ways.
//! - **Play:** a card of the next rank of its colour joins the firework
//!   (completing a colour's rank 5 returns a hint token, up to 8); any
//!   other card is discarded and costs a fuse. The played or discarded
//!   card is replaced from the deck while cards remain.
//! - **End:** at the third fuse, at a perfect score of 25, or one full
//!   round after the last card is drawn (each player, starting with the
//!   next, gets one more turn).
//! - **Rewards:** cooperative and shared: every player gets the score
//!   increase of the step, in points. With `zero_on_fuse` the third fuse
//!   also takes back the whole score (the variant some benchmarks use);
//!   the default keeps it. Rewards are small integers, exact in `f32`.
//! - **Observation:** `Box(u8, [OBS_LEN])` from the observer's seat:
//!   the other players' hands in seat order (card plus one, 0 for none),
//!   every player's knowledge masks (own first), fireworks, discard
//!   counts per card, tokens, fuses, deck size, the final-round counter,
//!   whose turn it is, and the last action with its actor.
//! - **Information state:** everything but the observer's own cards and
//!   the deck order. [`Determinize`] deals the observer's hand and the
//!   deck uniformly among the deals consistent with its knowledge and the
//!   cards it can see (exact weighted sampling, [`deal`]).

pub mod baseline;
pub mod deal;

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Determinize, Environment, EpisodeEnd, HeaderView, ImperfectInformation,
    InvariantViolation, NoAdaptive, ObsLeaf, Observation, Phase, RecordHeader, SpaceError, StepOut,
    Tier, TurnStructure, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/Lanterns-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;

// Registration under the versioned id (SPEC 10, ADR-0054).
tessera_rl_core::register!(Lanterns, id = "tessera/Lanterns-v1");

/// Agent slots.
pub const MAX_AGENTS: usize = 5;
/// Colours.
pub const COLOURS: usize = 5;
/// Ranks.
pub const RANKS: usize = 5;
/// Card kinds: colour times rank.
pub const KINDS: usize = COLOURS * RANKS;
/// Cards in the deck.
pub const DECK: usize = 50;
/// The most cards a hand holds.
pub const HAND: usize = 5;
/// An empty hand slot.
pub const NO_CARD: u8 = 25;
/// Hint tokens at the start, and the most there can be.
pub const MAX_HINTS: u8 = 8;
/// Fuses at the start.
pub const FUSES: u8 = 3;
/// The perfect score.
pub const MAX_SCORE: u8 = 25;
const _: () = assert!(NO_CARD as usize == KINDS && MAX_SCORE as usize == COLOURS * RANKS);
/// Copies of each rank in one colour.
pub const COPIES: [u8; RANKS] = [3, 2, 2, 2, 1];
/// Choices of the action head.
pub const ACTIONS: u32 = 50;
/// First play action.
pub const PLAY: u32 = 0;
/// First discard action.
pub const DISCARD: u32 = 5;
/// First colour hint.
pub const HINT_COLOUR: u32 = 10;
/// First rank hint.
pub const HINT_RANK: u32 = 30;
/// The final-round counter before the deck runs out.
pub const NOT_FINAL: u8 = u8::MAX;
/// A knowledge mask allowing every colour or every rank.
pub const ALL: u8 = 0b1_1111;
/// Length of the observation vector.
pub const OBS_LEN: usize = (MAX_AGENTS - 1) * HAND + 2 * MAX_AGENTS * HAND + COLOURS + KINDS + 8;
const _: () = assert!(OBS_LEN == 108, "the observation leaf's declared shape");

/// Lanterns configuration.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct LanternsConfig {
    /// Players, 2 to 5.
    #[config(default = 2, range = 2..=5)]
    pub players: u32,
    /// Whether the third fuse also takes back the score.
    #[config(default = false)]
    pub zero_on_fuse: bool,
}

impl LanternsConfig {
    /// Players, as a count of slots.
    #[must_use]
    pub fn players(&self) -> usize {
        usize::try_from(self.players)
            .unwrap_or(2)
            .clamp(2, MAX_AGENTS)
    }

    /// Cards per hand.
    #[must_use]
    pub fn hand_size(&self) -> usize {
        if self.players() <= 3 { 5 } else { 4 }
    }
}

/// Lanterns state. Every field is a byte, so the layout has no padding.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct LanternsState {
    /// The deck in dealing order; `deck[..next]` has been drawn.
    pub deck: [u8; DECK],
    /// The next card to draw.
    pub next: u8,
    /// Each player's hand, by seat; [`NO_CARD`] for an empty slot.
    pub hands: [[u8; HAND]; MAX_AGENTS],
    /// What each player knows of each slot: the colours still possible.
    pub colours: [[u8; HAND]; MAX_AGENTS],
    /// What each player knows of each slot: the ranks still possible.
    pub ranks: [[u8; HAND]; MAX_AGENTS],
    /// Cards played per colour (the top rank plus one).
    pub fireworks: [u8; COLOURS],
    /// Discarded cards per kind.
    pub discards: [u8; KINDS],
    /// Hint tokens.
    pub hints: u8,
    /// Fuses left.
    pub fuses: u8,
    /// Turns left once the deck is empty, or [`NOT_FINAL`].
    pub final_turns: u8,
    /// The seat to move.
    pub to_move: u8,
    /// Players.
    pub players: u8,
    /// The last action plus one (0: none yet).
    pub last_action: u8,
    /// The seat that took the last action.
    pub last_actor: u8,
    /// 1 once the game is over.
    pub done: u8,
}

impl Default for LanternsState {
    fn default() -> Self {
        Self {
            deck: [0; DECK],
            next: 0,
            hands: [[0; HAND]; MAX_AGENTS],
            colours: [[0; HAND]; MAX_AGENTS],
            ranks: [[0; HAND]; MAX_AGENTS],
            fireworks: [0; COLOURS],
            discards: [0; KINDS],
            hints: 0,
            fuses: 0,
            final_turns: 0,
            to_move: 0,
            players: 0,
            last_action: 0,
            last_actor: 0,
            done: 0,
        }
    }
}

/// What one player knows: everything but its own cards and the deck
/// order.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct LanternsInfo {
    /// The observer's seat.
    pub me: u8,
    /// Cards left in the deck.
    pub deck_left: u8,
    /// Every hand, with the observer's own cards hidden: its slots hold
    /// [`NO_CARD`] when empty and [`HIDDEN`] when they hold a card.
    pub hands: [[u8; HAND]; MAX_AGENTS],
    /// Every player's colour knowledge.
    pub colours: [[u8; HAND]; MAX_AGENTS],
    /// Every player's rank knowledge.
    pub ranks: [[u8; HAND]; MAX_AGENTS],
    /// Fireworks.
    pub fireworks: [u8; COLOURS],
    /// Discards per kind.
    pub discards: [u8; KINDS],
    /// Hint tokens.
    pub hints: u8,
    /// Fuses left.
    pub fuses: u8,
    /// The final-round counter.
    pub final_turns: u8,
    /// The seat to move.
    pub to_move: u8,
    /// Players.
    pub players: u8,
    /// The last action plus one.
    pub last_action: u8,
    /// The seat of the last action.
    pub last_actor: u8,
    /// 1 once the game is over.
    pub done: u8,
}

impl Default for LanternsInfo {
    fn default() -> Self {
        Self {
            me: 0,
            deck_left: 0,
            hands: [[0; HAND]; MAX_AGENTS],
            colours: [[0; HAND]; MAX_AGENTS],
            ranks: [[0; HAND]; MAX_AGENTS],
            fireworks: [0; COLOURS],
            discards: [0; KINDS],
            hints: 0,
            fuses: 0,
            final_turns: 0,
            to_move: 0,
            players: 0,
            last_action: 0,
            last_actor: 0,
            done: 0,
        }
    }
}

/// An own card the observer cannot see, in [`LanternsInfo::hands`].
pub const HIDDEN: u8 = NO_CARD + 1;

/// Lanterns observation.
#[derive(Observation)]
pub struct LanternsObs {
    /// See the module documentation for the layout.
    #[space(box(low = 0, high = 255, dtype = u8, shape = [108]))]
    pub view: ObsLeaf<u8>,
}

/// The Lanterns environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Lanterns;

/// The colour of a card.
#[must_use]
pub const fn colour(card: u8) -> usize {
    card as usize / RANKS
}

/// The rank of a card, 0 to 4.
#[must_use]
pub const fn rank(card: u8) -> usize {
    card as usize % RANKS
}

/// Copies of a card kind in the full deck.
#[must_use]
pub fn copies(kind: usize) -> u8 {
    COPIES.get(kind % RANKS).copied().unwrap_or(0)
}

/// The full deck, by kind count.
#[must_use]
pub fn full_counts() -> [u8; KINDS] {
    core::array::from_fn(copies)
}

fn slot(index: usize) -> Option<AgentSlot> {
    AgentSlot::new(u16::try_from(index).ok()?, MAX_AGENTS)
}

fn bound(n: usize) -> core::num::NonZeroU32 {
    u32::try_from(n)
        .ok()
        .and_then(core::num::NonZeroU32::new)
        .unwrap_or(core::num::NonZeroU32::MIN)
}

/// Shuffles `cards` with Fisher-Yates, one `below_u32` draw per position
/// from the back.
pub(crate) fn shuffle(rng: &mut StepRng, cards: &mut [u8]) {
    for i in (1..cards.len()).rev() {
        let j = usize::try_from(rng.below_u32(bound(i + 1))).unwrap_or(0);
        cards.swap(i, j);
    }
}

/// A decoded action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    /// Play the card in a slot.
    Play(usize),
    /// Discard the card in a slot.
    Discard(usize),
    /// Hint a colour to the player `offset` seats on.
    HintColour {
        /// Seats after the actor, 1 to 4.
        offset: usize,
        /// The colour.
        colour: usize,
    },
    /// Hint a rank to the player `offset` seats on.
    HintRank {
        /// Seats after the actor, 1 to 4.
        offset: usize,
        /// The rank.
        rank: usize,
    },
}

impl Move {
    /// The move of action `a`, or `None` outside `0..ACTIONS`.
    #[must_use]
    pub const fn decode(a: u32) -> Option<Self> {
        let a = a as usize;
        match a {
            0..5 => Some(Self::Play(a)),
            5..10 => Some(Self::Discard(a - 5)),
            10..30 => Some(Self::HintColour {
                offset: (a - 10) / 5 + 1,
                colour: (a - 10) % 5,
            }),
            30..50 => Some(Self::HintRank {
                offset: (a - 30) / 5 + 1,
                rank: (a - 30) % 5,
            }),
            _ => None,
        }
    }

    /// The action of this move.
    #[must_use]
    pub fn encode(self) -> u32 {
        let code = match self {
            Self::Play(s) => s,
            Self::Discard(s) => 5 + s,
            Self::HintColour { offset, colour } => 10 + 5 * (offset - 1) + colour,
            Self::HintRank { offset, rank } => 30 + 5 * (offset - 1) + rank,
        };
        u32::try_from(code).unwrap_or(0)
    }
}

/// The public facts legality needs, shared by the state and the
/// information state.
pub(crate) struct Table<'a> {
    pub(crate) hands: &'a [[u8; HAND]; MAX_AGENTS],
    pub(crate) hints: u8,
    pub(crate) players: usize,
    pub(crate) to_move: usize,
}

impl Table<'_> {
    /// Whether seat `seat`'s slot `s` holds a card (own hidden cards
    /// count).
    fn holds(&self, seat: usize, s: usize) -> bool {
        self.hands
            .get(seat)
            .and_then(|h| h.get(s))
            .is_some_and(|&c| c != NO_CARD)
    }

    /// Whether `m` is legal for the player to move. A hint's target must
    /// be a seat whose cards are visible to the actor, which holds for
    /// every other seat.
    pub(crate) fn legal(&self, m: Move) -> bool {
        let me = self.to_move;
        match m {
            Move::Play(s) => self.holds(me, s),
            Move::Discard(s) => self.holds(me, s) && self.hints < MAX_HINTS,
            Move::HintColour { offset, colour: c } => {
                self.hint_touches(offset, |card| colour(card) == c)
            }
            Move::HintRank { offset, rank: r } => self.hint_touches(offset, |card| rank(card) == r),
        }
    }

    fn hint_touches(&self, offset: usize, hit: impl Fn(u8) -> bool) -> bool {
        if self.hints == 0 || offset == 0 || offset >= self.players {
            return false;
        }
        let seat = (self.to_move + offset) % self.players;
        self.hands
            .get(seat)
            .is_some_and(|h| h.iter().any(|&c| c < NO_CARD && hit(c)))
    }

    /// The no-op: the play of the first slot holding a card.
    pub(crate) fn noop(&self) -> u32 {
        (0..HAND)
            .find(|&s| self.holds(self.to_move, s))
            .map_or(PLAY, |s| Move::Play(s).encode())
    }
}

impl LanternsState {
    fn table(&self) -> Table<'_> {
        Table {
            hands: &self.hands,
            hints: self.hints,
            players: usize::from(self.players),
            to_move: usize::from(self.to_move),
        }
    }

    /// The score: cards on the fireworks.
    #[must_use]
    pub fn score(&self) -> u8 {
        self.fireworks.iter().sum()
    }

    /// Draws the next card into seat `seat`'s slot `s`, with fresh
    /// knowledge, or leaves it empty when the deck is out.
    fn refill(&mut self, seat: usize, s: usize) {
        let card = self
            .deck
            .get(usize::from(self.next))
            .copied()
            .filter(|_| usize::from(self.next) < DECK);
        let (Some(h), Some(c), Some(r)) = (
            self.hands.get_mut(seat).and_then(|h| h.get_mut(s)),
            self.colours.get_mut(seat).and_then(|h| h.get_mut(s)),
            self.ranks.get_mut(seat).and_then(|h| h.get_mut(s)),
        ) else {
            return;
        };
        if let Some(card) = card {
            *h = card;
            *c = ALL;
            *r = ALL;
            self.next += 1;
            if usize::from(self.next) == DECK {
                // Every player, the next one first, gets one more turn.
                self.final_turns = self.players;
            }
        } else {
            *h = NO_CARD;
            *c = 0;
            *r = 0;
        }
    }

    /// Applies a legal move of the player to move; returns the score
    /// change and whether the game ended on a fuse.
    fn apply(&mut self, m: Move) -> (i32, bool) {
        let me = usize::from(self.to_move);
        let before = i32::from(self.score());
        let mut fused = false;
        match m {
            Move::Play(s) | Move::Discard(s) => {
                let card = self
                    .hands
                    .get(me)
                    .and_then(|h| h.get(s))
                    .copied()
                    .unwrap_or(NO_CARD);
                if card < NO_CARD {
                    let (c, r) = (colour(card), rank(card));
                    let top = self.fireworks.get(c).copied().unwrap_or(0);
                    let played = matches!(m, Move::Play(_)) && usize::from(top) == r;
                    if played {
                        if let Some(f) = self.fireworks.get_mut(c) {
                            *f += 1;
                        }
                        if r == RANKS - 1 && self.hints < MAX_HINTS {
                            self.hints += 1;
                        }
                    } else {
                        if let Some(d) = self.discards.get_mut(usize::from(card)) {
                            *d += 1;
                        }
                        if matches!(m, Move::Play(_)) {
                            self.fuses = self.fuses.saturating_sub(1);
                            fused = self.fuses == 0;
                        } else {
                            self.hints = (self.hints + 1).min(MAX_HINTS);
                        }
                    }
                    self.refill(me, s);
                }
            }
            Move::HintColour { offset, colour: c } => {
                self.hint(offset, |card| colour(card) == c, |k| (c, k, true));
            }
            Move::HintRank { offset, rank: r } => {
                self.hint(offset, |card| rank(card) == r, |k| (r, k, false));
            }
        }
        (i32::from(self.score()) - before, fused)
    }

    /// Narrows the target's knowledge both ways and spends a token.
    fn hint(
        &mut self,
        offset: usize,
        hit: impl Fn(u8) -> bool,
        which: impl Fn(usize) -> (usize, usize, bool),
    ) {
        let players = usize::from(self.players).max(1);
        let seat = (usize::from(self.to_move) + offset) % players;
        self.hints = self.hints.saturating_sub(1);
        for k in 0..HAND {
            let card = self
                .hands
                .get(seat)
                .and_then(|h| h.get(k))
                .copied()
                .unwrap_or(NO_CARD);
            if card >= NO_CARD {
                continue;
            }
            let (value, _, is_colour) = which(k);
            let bit = 1u8 << value;
            let masks = if is_colour {
                &mut self.colours
            } else {
                &mut self.ranks
            };
            if let Some(m) = masks.get_mut(seat).and_then(|h| h.get_mut(k)) {
                *m = if hit(card) { bit } else { *m & !bit };
            }
        }
    }
}

impl Environment for Lanterns {
    /// AEC only: a sequential game (SPEC 12, ADR-0080).
    const ADAPTERS: tessera_rl_core::Adapters = tessera_rl_core::Adapters::PZ_AEC;
    type State = LanternsState;
    type Config = LanternsConfig;
    type Obs = LanternsObs;
    type Action = ();
    type Header = RecordHeader<MAX_AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = MAX_AGENTS;
    const TIER: Tier = Tier::CrossPlatform;
    const PERFECT_INFORMATION: bool = false;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &LanternsConfig) -> Result<Space, SpaceError> {
        LanternsObs::space(cfg)
    }

    fn action_space(_: &LanternsConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: ACTIONS })
    }

    fn initial_agents(cfg: &LanternsConfig) -> usize {
        cfg.players()
    }

    fn acting(_: &LanternsConfig, state: &LanternsState, agent: AgentSlot) -> bool {
        state.done == 0 && agent.index() == usize::from(state.to_move)
    }

    fn reset(cfg: &LanternsConfig, rng: &mut StepRng, state: &mut LanternsState) {
        let mut deck = [0u8; DECK];
        let mut i = 0;
        for (kind, n) in full_counts().iter().enumerate() {
            for _ in 0..*n {
                if let Some(d) = deck.get_mut(i) {
                    *d = u8::try_from(kind).unwrap_or(0);
                }
                i += 1;
            }
        }
        shuffle(rng, &mut deck);
        let players = cfg.players();
        *state = LanternsState {
            deck,
            hints: MAX_HINTS,
            fuses: FUSES,
            final_turns: NOT_FINAL,
            players: u8::try_from(players).unwrap_or(2),
            hands: [[NO_CARD; HAND]; MAX_AGENTS],
            ..LanternsState::default()
        };
        for seat in 0..players {
            for s in 0..cfg.hand_size() {
                state.refill(seat, s);
            }
        }
    }

    fn step(
        cfg: &LanternsConfig,
        state: &mut LanternsState,
        actions: &Actions<'_, ()>,
        _: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let me = usize::from(state.to_move);
        let Some(mover) = slot(me) else {
            return;
        };
        let table = state.table();
        // The engine replaced an illegal action (ADR-0028); the kernel
        // stays total and reads anything else as the no-op.
        let action = actions
            .discrete(mover, 0)
            .and_then(|a| u32::try_from(a).ok())
            .filter(|&a| Move::decode(a).is_some_and(|m| table.legal(m)))
            .unwrap_or_else(|| table.noop());
        let Some(m) = Move::decode(action) else {
            return;
        };
        let was_final = state.final_turns != NOT_FINAL;
        let (gain, fused) = state.apply(m);
        state.last_action = u8::try_from(action + 1).unwrap_or(0);
        state.last_actor = u8::try_from(me).unwrap_or(0);
        let players = usize::from(state.players).max(1);
        let mut reward = gain;
        let mut over = fused || state.score() == MAX_SCORE;
        if fused && cfg.zero_on_fuse {
            reward -= i32::from(state.score());
        }
        if was_final && !over {
            state.final_turns = state.final_turns.saturating_sub(1);
            over = state.final_turns == 0;
        }
        state.to_move = u8::try_from((me + 1) % players).unwrap_or(0);
        #[expect(clippy::cast_precision_loss, reason = "scores are at most 25")]
        let r = reward as f32;
        for seat in 0..players {
            if let Some(a) = slot(seat) {
                out.set_reward(a, r);
            }
        }
        if over {
            state.done = 1;
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        _: &LanternsConfig,
        state: &LanternsState,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, LanternsObs>,
    ) {
        let info = info_of(state, agent.index());
        let Some(mut leaf) = out.leaf_mut(LanternsObs::VIEW) else {
            return;
        };
        for (k, v) in view(&info).iter().enumerate() {
            leaf.set(k, *v);
        }
    }
}

/// The observation of an information state (module documentation).
#[must_use]
pub fn view(info: &LanternsInfo) -> [u8; OBS_LEN] {
    let mut v = [0u8; OBS_LEN];
    let players = usize::from(info.players).max(1);
    let me = usize::from(info.me);
    let mut i = 0;
    let mut put = |x: u8| {
        if let Some(e) = v.get_mut(i) {
            *e = x;
        }
        i += 1;
    };
    for offset in 1..MAX_AGENTS {
        for s in 0..HAND {
            let card = if offset < players {
                let seat = (me + offset) % players;
                info.hands
                    .get(seat)
                    .and_then(|h| h.get(s))
                    .copied()
                    .unwrap_or(NO_CARD)
            } else {
                NO_CARD
            };
            put(if card < NO_CARD { card + 1 } else { 0 });
        }
    }
    for offset in 0..MAX_AGENTS {
        let seat = (me + offset) % players;
        for s in 0..HAND {
            let (c, r) = if offset < players {
                (
                    info.colours
                        .get(seat)
                        .and_then(|h| h.get(s))
                        .copied()
                        .unwrap_or(0),
                    info.ranks
                        .get(seat)
                        .and_then(|h| h.get(s))
                        .copied()
                        .unwrap_or(0),
                )
            } else {
                (0, 0)
            };
            put(c);
            put(r);
        }
    }
    for &f in &info.fireworks {
        put(f);
    }
    for &d in &info.discards {
        put(d);
    }
    put(info.hints);
    put(info.fuses);
    put(info.deck_left);
    put(info.final_turns);
    put(u8::from(info.done == 0 && info.to_move == info.me));
    put(info.last_action);
    let actor = (usize::from(info.last_actor) + players - me % players) % players;
    put(u8::try_from(actor).unwrap_or(0));
    put(info.players);
    v
}

/// The information state of the player in seat `seat`.
#[must_use]
pub fn info_of(s: &LanternsState, seat: usize) -> LanternsInfo {
    let mut hands = s.hands;
    if let Some(own) = hands.get_mut(seat) {
        for c in own.iter_mut() {
            if *c != NO_CARD {
                *c = HIDDEN;
            }
        }
    }
    LanternsInfo {
        me: u8::try_from(seat).unwrap_or(0),
        deck_left: u8::try_from(DECK - usize::from(s.next).min(DECK)).unwrap_or(0),
        hands,
        colours: s.colours,
        ranks: s.ranks,
        fireworks: s.fireworks,
        discards: s.discards,
        hints: s.hints,
        fuses: s.fuses,
        final_turns: s.final_turns,
        to_move: s.to_move,
        players: s.players,
        last_action: s.last_action,
        last_actor: s.last_actor,
        done: s.done,
    }
}

impl ActionMask for Lanterns {
    fn action_mask(
        _: &LanternsConfig,
        state: &LanternsState,
        _: AgentSlot,
        out: &mut MaskWriter<'_>,
    ) {
        let table = state.table();
        for a in 0..ACTIONS {
            let legal = Move::decode(a).is_some_and(|m| table.legal(m));
            out.set(a as usize, legal);
        }
    }

    fn noop(_: &LanternsConfig, state: &LanternsState, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, i32::try_from(state.table().noop()).unwrap_or(0));
    }
}

/// Imported states: the deck is the full deck; the drawn cards are exactly
/// the cards in hands, discards and fireworks; while the deck lasts, each
/// active seat holds cards in its first slots only; once the deck is out, a
/// played or discarded slot stays empty and later slots keep their cards, so
/// empty slots may appear anywhere in an active hand (#728); knowledge allows
/// each held card's colour and rank and is empty for empty slots;
/// counters are in range.
impl Validate for Lanterns {
    fn validate(cfg: &LanternsConfig, s: &LanternsState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        let players = cfg.players();
        if usize::from(s.players) != players {
            return invalid("a player count other than the config's");
        }
        let mut deck = [0u8; KINDS];
        for &c in &s.deck {
            match deck.get_mut(usize::from(c)) {
                Some(n) => *n += 1,
                None => return invalid("a card out of range in the deck"),
            }
        }
        if deck != full_counts() {
            return invalid("a deck that is not the full deck");
        }
        if usize::from(s.next) > DECK
            || s.hints > MAX_HINTS
            || s.fuses > FUSES
            || usize::from(s.to_move) >= players
            || s.done > 1
            || usize::from(s.last_actor) >= MAX_AGENTS
            || u32::from(s.last_action) > ACTIONS
        {
            return invalid("a counter out of range");
        }
        if s.fireworks.iter().any(|&f| usize::from(f) > RANKS) {
            return invalid("a firework past rank 5");
        }
        let mut drawn = [0u8; KINDS];
        for &c in s.deck.get(..usize::from(s.next)).unwrap_or(&[]) {
            if let Some(n) = drawn.get_mut(usize::from(c)) {
                *n += 1;
            }
        }
        let mut seen = s.discards;
        for (c, &f) in s.fireworks.iter().enumerate() {
            for r in 0..usize::from(f) {
                if let Some(n) = seen.get_mut(c * RANKS + r) {
                    *n += 1;
                }
            }
        }
        for (seat, hand) in s.hands.iter().enumerate() {
            let mut empty = false;
            for (k, &card) in hand.iter().enumerate() {
                let (cm, rm) = (
                    s.colours
                        .get(seat)
                        .and_then(|h| h.get(k))
                        .copied()
                        .unwrap_or(0),
                    s.ranks
                        .get(seat)
                        .and_then(|h| h.get(k))
                        .copied()
                        .unwrap_or(0),
                );
                if card == NO_CARD {
                    empty = true;
                    if cm != 0 || rm != 0 {
                        return invalid("knowledge of an empty slot");
                    }
                    continue;
                }
                // A card after an empty slot is only possible once the deck
                // is out: refill leaves the slot empty then (#728).
                if seat >= players || k >= cfg.hand_size() || (empty && usize::from(s.next) < DECK)
                {
                    return invalid("a card in a slot that cannot hold one");
                }
                if card > NO_CARD
                    || cm & (1 << colour(card)) == 0
                    || rm & (1 << rank(card)) == 0
                    || cm > ALL
                    || rm > ALL
                {
                    return invalid("knowledge that excludes the card it describes");
                }
                if let Some(n) = seen.get_mut(usize::from(card)) {
                    *n += 1;
                }
            }
        }
        if seen != drawn {
            return invalid("cards in play that are not the drawn cards");
        }
        Ok(())
    }

    /// ADR-0040: before the first reset the state is exactly the initial
    /// value; once reset, the players' slots are alive.
    fn validate_record(
        cfg: &LanternsConfig,
        header: &HeaderView<'_>,
        state: &LanternsState,
    ) -> Result<(), InvariantViolation> {
        if header.phase() == Phase::NeverReset {
            if *state != LanternsState::default() {
                return Err(InvariantViolation {
                    code: "a state other than the initial value before the first reset",
                });
            }
            return Ok(());
        }
        Self::validate(cfg, state)?;
        if !(0..cfg.players()).all(|k| header.agent(k).is_some_and(|v| v.alive)) {
            return Err(InvariantViolation {
                code: "a player that is not alive",
            });
        }
        Ok(())
    }
}

impl ImperfectInformation for Lanterns {
    type InfoState = LanternsInfo;

    fn info_state(_: &LanternsConfig, s: &LanternsState, agent: AgentSlot) -> LanternsInfo {
        info_of(s, agent.index())
    }
}

/// Deals the observer's hand and the deck uniformly among the deals
/// consistent with its knowledge and the cards it sees ([`deal`]).
impl Determinize for Lanterns {
    fn sample(
        _: &LanternsConfig,
        info: &LanternsInfo,
        rng: &mut StepRng,
        state: &mut LanternsState,
    ) {
        *state = deal::sample(info, rng);
    }
}

impl tessera_rl_core::Persist for Lanterns {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}
