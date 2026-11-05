// SPDX-License-Identifier: MIT OR Apache-2.0

//! Leduc hold'em (`tessera/Leduc-v1`): a two-player poker game with
//! hidden cards. Cross-platform tier, integer state, imperfect
//! information.
//!
//! - Deck: six cards, two each of jack, queen and king (ranks 0, 1, 2).
//! - Reset: each player antes 1 and is dealt one private card.
//! - Rounds: two betting rounds; after the first, one public card is
//!   dealt from the four cards left, at step time, from `StepRng`. Each
//!   round allows at most two raises, of 2 in the first round and 4 in
//!   the second. Slot 0 acts first in each round.
//! - Turns: `Sequential` (ADR-0013); the acting agent is `to_move`, and
//!   nobody acts on a finished hand.
//! - Action: `Discrete(3)`: 0 fold, 1 check or call, 2 raise. The mask
//!   (ADR-0014) marks fold illegal when there is no bet to answer, and
//!   raise illegal after two raises in the round; the no-op is check or
//!   call, always legal.
//! - Rewards: at the end of the hand, the winner gets what the loser put
//!   in the pot and the loser its negation; a fold loses what the folder
//!   put in. At showdown a pair with the public card wins, then the
//!   higher rank; equal ranks split (0 to both). Every other step pays 0.
//!   Rewards and returns are normalized: they are in units of
//!   [`MAX_CONTRIBUTION`] (13) chips, the most a player can lose, so they
//!   lie in `[-1, 1]` as two-player zero-sum search requires (ADR-0043).
//!   Multiplying a reward or a return by 13 gives chips.
//! - Ends: terminated on a fold or a showdown; at most 8 actions.
//! - Observation: `Box(u8, [OBS_LEN])` from the observer's view: own
//!   rank, public rank (3 before the deal), round, whether the observer
//!   is to move, both contributions (observer first), raises in the
//!   round, and the two rounds' action histories with their lengths.
//! - Information state: the observer's slot and card and everything
//!   public; the opponent's card is hidden. [`Determinize`] draws it
//!   from the cards the observer cannot see.

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, AgentSlot, Determinize, Environment, EpisodeEnd, HeaderView, Heuristic,
    ImperfectInformation, InvariantViolation, NoAdaptive, ObsLeaf, Observation, Phase,
    RecordHeader, SpaceError, StepOut, Tier, TurnStructure, Validate,
};
use tessera_rl_macros::{EnvConfig, EnvState, Observation};
use tessera_rl_num::rng::{ObsRng, StepRng};

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/Leduc-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;
/// Agent slots.
pub const MAX_AGENTS: usize = 2;
/// Ranks: jack, queen, king.
pub const RANKS: u32 = 3;
/// Copies of each rank in the deck.
pub const COPIES: u32 = 2;
/// The public card before it is dealt.
pub const NO_CARD: u32 = RANKS;
/// Choices of the action head.
pub const ACTIONS: u32 = 3;
/// Fold.
pub const FOLD: u32 = 0;
/// Check or call.
pub const CALL: u32 = 1;
/// Raise.
pub const RAISE: u32 = 2;
/// Raises allowed per round.
pub const MAX_RAISES: u32 = 2;
/// Actions a round can hold: check, raise, raise, call.
pub const ROUND_ACTIONS: u32 = 4;
/// Length of the observation vector.
pub const OBS_LEN: usize = 11;
/// The largest contribution of one player: the ante and four raises.
pub const MAX_CONTRIBUTION: u32 = 1 + MAX_RAISES * 2 + MAX_RAISES * 4;

/// Leduc hold'em has no parameters.
#[derive(EnvConfig, Debug, Clone, PartialEq, Eq)]
pub struct LeducConfig {}

/// Leduc hold'em state.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct LeducState {
    /// Each player's private rank.
    pub cards: [u32; 2],
    /// The public rank, or [`NO_CARD`] before the deal.
    pub public: u32,
    /// The betting round, 0 or 1.
    pub round: u32,
    /// The slot index to move.
    pub to_move: u32,
    /// Raises in the current round.
    pub raises: u32,
    /// What each player has put in the pot.
    pub contributions: [u32; 2],
    /// Each round's actions, two bits each, oldest in the low bits; an
    /// entry holds the action plus one.
    pub history: [u32; 2],
    /// Actions in each round.
    pub history_len: [u32; 2],
    /// 1 once the hand is over.
    pub done: u32,
}

/// What one player knows: everything but the opponent's card.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct LeducInfo {
    /// The observer's slot index.
    pub me: u32,
    /// The observer's rank.
    pub card: u32,
    /// The public rank, or [`NO_CARD`].
    pub public: u32,
    /// The betting round.
    pub round: u32,
    /// The slot index to move.
    pub to_move: u32,
    /// Raises in the current round.
    pub raises: u32,
    /// Both contributions, by slot.
    pub contributions: [u32; 2],
    /// Both rounds' histories.
    pub history: [u32; 2],
    /// Both rounds' history lengths.
    pub history_len: [u32; 2],
    /// 1 once the hand is over.
    pub done: u32,
}

/// Leduc hold'em observation.
#[derive(Observation)]
pub struct LeducObs {
    /// See the module documentation for the layout.
    #[space(box(low = 0, high = 255, dtype = u8, shape = [11]))]
    pub view: ObsLeaf<u8>,
}

/// The Leduc hold'em environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Leduc;

fn slot(index: usize) -> Option<AgentSlot> {
    AgentSlot::new(u16::try_from(index).ok()?, MAX_AGENTS)
}

fn bound(n: u32) -> core::num::NonZeroU32 {
    core::num::NonZeroU32::new(n).unwrap_or(core::num::NonZeroU32::MIN)
}

/// Draws one card from the deck less the `seen` ranks ([`NO_CARD`] is
/// ignored): a uniform card, returned as its rank.
#[must_use]
pub fn draw(rng: &mut StepRng, seen: &[u32]) -> u32 {
    let mut left = [COPIES; RANKS as usize];
    for &r in seen {
        if let Some(n) = left.get_mut(r as usize)
            && *n > 0
        {
            *n -= 1;
        }
    }
    let total: u32 = left.iter().sum();
    let mut k = rng.below_u32(bound(total));
    for (rank, &n) in (0_u32..).zip(left.iter()) {
        if k < n {
            return rank;
        }
        k -= n;
    }
    0
}

impl LeducState {
    /// The side to move, as a slot index.
    #[must_use]
    pub const fn mover(&self) -> usize {
        (self.to_move & 1) as usize
    }

    /// Whether the mover faces a bet it has not matched.
    #[must_use]
    pub const fn facing_bet(&self) -> bool {
        let [c0, c1] = self.contributions;
        if self.mover() == 0 { c1 > c0 } else { c0 > c1 }
    }

    fn push(&mut self, action: u32) {
        let r = (self.round & 1) as usize;
        if let (Some(h), Some(n)) = (self.history.get_mut(r), self.history_len.get_mut(r)) {
            *h |= (action + 1) << (2 * *n);
            *n += 1;
        }
    }

    fn round_actions(&self) -> u32 {
        self.history_len
            .get((self.round & 1) as usize)
            .copied()
            .unwrap_or(0)
    }

    /// The showdown result for slot 0: 1 win, -1 loss, 0 split.
    #[must_use]
    pub const fn showdown(&self) -> i32 {
        let [a, b] = self.cards;
        let (pa, pb) = (a == self.public, b == self.public);
        if pa != pb {
            return if pa { 1 } else { -1 };
        }
        if a > b {
            1
        } else if a < b {
            -1
        } else {
            0
        }
    }
}

/// `n` chips in units of [`MAX_CONTRIBUTION`].
#[expect(clippy::cast_precision_loss, reason = "contributions are at most 13")]
fn chips(n: u32) -> f32 {
    n as f32 / MAX_CONTRIBUTION as f32
}

impl Environment for Leduc {
    type State = LeducState;
    type Config = LeducConfig;
    type Obs = LeducObs;
    type Action = ();
    type Header = RecordHeader<MAX_AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = MAX_AGENTS;
    const TIER: Tier = Tier::CrossPlatform;
    const PERFECT_INFORMATION: bool = false;
    const TURNS: TurnStructure = TurnStructure::Sequential;

    fn observation_space(cfg: &LeducConfig) -> Result<Space, SpaceError> {
        LeducObs::space(cfg)
    }

    fn action_space(_: &LeducConfig) -> Result<Space, SpaceError> {
        Ok(Space::Discrete { n: ACTIONS })
    }

    fn acting(_: &LeducConfig, state: &LeducState, agent: AgentSlot) -> bool {
        state.done == 0 && agent.index() == state.mover()
    }

    fn reset(_: &LeducConfig, rng: &mut StepRng, state: &mut LeducState) {
        let first = draw(rng, &[]);
        let second = draw(rng, &[first]);
        *state = LeducState {
            cards: [first, second],
            public: NO_CARD,
            contributions: [1, 1],
            ..LeducState::default()
        };
    }

    fn step(
        _: &LeducConfig,
        state: &mut LeducState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let me = state.mover();
        let (Some(mover), Some(other), Some(s0), Some(s1)) =
            (slot(me), slot(1 - me), slot(0), slot(1))
        else {
            return;
        };
        // The engine replaced an illegal action (ADR-0028); the kernel
        // stays total and reads anything else as check or call.
        let facing = state.facing_bet();
        let action = match actions
            .discrete(mover, 0)
            .and_then(|a| u32::try_from(a).ok())
        {
            Some(FOLD) if facing => FOLD,
            Some(RAISE) if state.raises < MAX_RAISES => RAISE,
            _ => CALL,
        };
        state.push(action);
        let [c0, c1] = state.contributions;
        let mine = state.contributions.get(me).copied().unwrap_or(0);
        let theirs = state.contributions.get(1 - me).copied().unwrap_or(0);
        match action {
            FOLD => {
                out.set_reward(mover, -chips(mine));
                out.set_reward(other, chips(mine));
                state.done = 1;
                out.end_episode(EpisodeEnd::Terminated);
            }
            RAISE => {
                let size = if state.round == 0 { 2 } else { 4 };
                if let Some(c) = state.contributions.get_mut(me) {
                    *c = theirs + size;
                }
                state.raises += 1;
                state.to_move = u32::from(me == 0);
            }
            _ => {
                if let Some(c) = state.contributions.get_mut(me) {
                    *c = theirs;
                }
                if state.round_actions() < 2 {
                    state.to_move = u32::from(me == 0);
                } else if state.round == 0 {
                    state.public = draw(rng, &state.cards);
                    state.round = 1;
                    state.raises = 0;
                    state.to_move = 0;
                } else {
                    let pot = c0.max(c1);
                    let r = chips(pot)
                        * match state.showdown() {
                            1 => 1.0,
                            -1 => -1.0,
                            _ => 0.0,
                        };
                    out.set_reward(s0, r);
                    out.set_reward(s1, -r);
                    state.done = 1;
                    out.end_episode(EpisodeEnd::Terminated);
                }
            }
        }
    }

    fn observe(
        _: &LeducConfig,
        state: &LeducState,
        agent: AgentSlot,
        _: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, LeducObs>,
    ) {
        let me = agent.index().min(1);
        let get = |a: &[u32; 2], i: usize| a.get(i).copied().unwrap_or(0);
        let values = [
            get(&state.cards, me),
            state.public,
            state.round,
            u32::from(state.done == 0 && state.mover() == me),
            get(&state.contributions, me),
            get(&state.contributions, 1 - me),
            state.raises,
            state.history[0],
            state.history[1],
            state.history_len[0],
            state.history_len[1],
        ];
        let Some(mut leaf) = out.leaf_mut(LeducObs::VIEW) else {
            return;
        };
        for (k, v) in values.into_iter().enumerate() {
            leaf.set(k, u8::try_from(v).unwrap_or(u8::MAX));
        }
    }
}

/// Fold is illegal without a bet to answer, raise after two raises; the
/// no-op is check or call.
impl ActionMask for Leduc {
    fn action_mask(_: &LeducConfig, state: &LeducState, _: AgentSlot, out: &mut MaskWriter<'_>) {
        out.set(FOLD as usize, state.facing_bet());
        out.set(CALL as usize, true);
        out.set(RAISE as usize, state.raises < MAX_RAISES);
    }

    fn noop(_: &LeducConfig, _: &LeducState, _: AgentSlot, out: &mut ActionWriter<'_>) {
        out.set_discrete(0, i32::try_from(CALL).unwrap_or(1));
    }
}

/// Imported states: ranks in range with at most two copies dealt, the
/// public card dealt exactly in the second round, bounded counters and
/// histories, and contributions in range.
impl Validate for Leduc {
    fn validate(_: &LeducConfig, s: &LeducState) -> Result<(), InvariantViolation> {
        let invalid = |code| Err(InvariantViolation { code });
        if s.cards.iter().any(|&c| c >= RANKS) || s.public > NO_CARD {
            return invalid("a rank out of range");
        }
        for rank in 0..RANKS {
            let n = s.cards.iter().filter(|&&c| c == rank).count() + usize::from(s.public == rank);
            if n > COPIES as usize {
                return invalid("more copies of a rank than the deck holds");
            }
        }
        if s.round > 1 || (s.public != NO_CARD) != (s.round == 1) {
            return invalid("the public card and the round disagree");
        }
        if s.to_move > 1 || s.done > 1 || s.raises > MAX_RAISES {
            return invalid("a counter out of range");
        }
        let fits = |h: u32, n: u32| n <= ROUND_ACTIONS && (n == ROUND_ACTIONS || h >> (2 * n) == 0);
        if !s
            .history
            .iter()
            .zip(s.history_len.iter())
            .all(|(&h, &n)| fits(h, n))
        {
            return invalid("a history out of range");
        }
        if s.contributions
            .iter()
            .any(|&c| c == 0 || c > MAX_CONTRIBUTION)
        {
            return invalid("a contribution out of range");
        }
        Ok(())
    }

    /// ADR-0040: once reset, both slots are alive, and a running episode
    /// is not a finished hand.
    fn validate_record(
        cfg: &LeducConfig,
        header: &HeaderView<'_>,
        state: &LeducState,
    ) -> Result<(), InvariantViolation> {
        Self::validate(cfg, state)?;
        let phase = header.phase();
        if phase == Phase::NeverReset {
            return Ok(());
        }
        if !(0..MAX_AGENTS).all(|k| header.agent(k).is_some_and(|v| v.alive)) {
            return Err(InvariantViolation {
                code: "a player that is not alive",
            });
        }
        if phase == Phase::Active && state.done != 0 {
            return Err(InvariantViolation {
                code: "a running episode on a finished hand",
            });
        }
        Ok(())
    }
}

impl ImperfectInformation for Leduc {
    type InfoState = LeducInfo;

    fn info_state(_: &LeducConfig, s: &LeducState, agent: AgentSlot) -> LeducInfo {
        let me = agent.index().min(1);
        LeducInfo {
            me: u32::from(me == 1),
            card: s.cards.get(me).copied().unwrap_or(0),
            public: s.public,
            round: s.round,
            to_move: s.to_move,
            raises: s.raises,
            contributions: s.contributions,
            history: s.history,
            history_len: s.history_len,
            done: s.done,
        }
    }
}

/// Draws the opponent's card from the four cards the observer cannot see
/// (the deck less its own card and the public card). The public card,
/// when not dealt yet, stays undealt: it is drawn at step time.
impl Determinize for Leduc {
    fn sample(_: &LeducConfig, info: &LeducInfo, rng: &mut StepRng, state: &mut LeducState) {
        let me = (info.me & 1) as usize;
        let opponent = draw(rng, &[info.card, info.public]);
        let mut cards = [0; 2];
        if let Some(c) = cards.get_mut(me) {
            *c = info.card;
        }
        if let Some(c) = cards.get_mut(1 - me) {
            *c = opponent;
        }
        *state = LeducState {
            cards,
            public: info.public,
            round: info.round,
            to_move: info.to_move,
            raises: info.raises,
            contributions: info.contributions,
            history: info.history,
            history_len: info.history_len,
            done: info.done,
        };
    }
}

/// Scripted baseline: raise with a king or a pair while raises remain,
/// fold a jack without a pair facing a bet, otherwise check or call.
impl Heuristic for Leduc {
    fn act(
        _: &LeducConfig,
        state: &LeducState,
        agent: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let card = state.cards.get(agent.index().min(1)).copied().unwrap_or(0);
        let strong = card == RANKS - 1 || card == state.public;
        let action = if strong && state.raises < MAX_RAISES {
            RAISE
        } else if card == 0 && !strong && state.facing_bet() {
            FOLD
        } else {
            CALL
        };
        out.set_discrete(0, i32::try_from(action).unwrap_or(1));
    }
}
