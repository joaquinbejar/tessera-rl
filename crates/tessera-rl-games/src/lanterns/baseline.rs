// SPDX-License-Identifier: MIT OR Apache-2.0

//! The scripted Lanterns baseline (SPEC 16.1): a rule-based player that
//! reads only the information state (ADR-0035, ADR-0079), in order:
//!
//! 1. play a card every possibility of which is playable now;
//! 2. with a hint token, hint a playable card to the first player after
//!    it who holds one and does not know its rank yet (a rank hint);
//! 3. below 8 tokens, discard the oldest card it has no information on,
//!    else its first card;
//! 4. otherwise give the first legal hint, else play the first card.
//!
//! [`act_info`] takes the information state; `Heuristic::act` derives it
//! from the state and calls it, so states with equal information states
//! give equal actions. It becomes `InfoHeuristic::act_info` once that
//! trait exists (ADR-0079).

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic};
use tessera_rl_num::rng::StepRng;

use super::{
    ACTIONS, ALL, COLOURS, HAND, HIDDEN, Lanterns, LanternsConfig, LanternsInfo, LanternsState,
    MAX_HINTS, Move, NO_CARD, RANKS, Table, colour, info_of, rank,
};

/// Whether every kind the knowledge `(cm, rm)` allows is playable on
/// `fireworks`.
fn surely_playable(fireworks: [u8; COLOURS], cm: u8, rm: u8) -> bool {
    let mut any = false;
    for c in 0..COLOURS {
        if cm & (1 << c) == 0 {
            continue;
        }
        for r in 0..RANKS {
            if rm & (1 << r) == 0 {
                continue;
            }
            any = true;
            if fireworks.get(c).is_none_or(|&f| usize::from(f) != r) {
                return false;
            }
        }
    }
    any
}

/// The baseline's action from the information state alone.
#[must_use]
pub fn act_info(info: &LanternsInfo) -> u32 {
    let me = usize::from(info.me);
    let players = usize::from(info.players).max(1);
    let table = Table {
        hands: &info.hands,
        hints: info.hints,
        players,
        to_move: usize::from(info.to_move),
    };
    let knows = |seat: usize, k: usize| {
        (
            info.colours
                .get(seat)
                .and_then(|h| h.get(k))
                .copied()
                .unwrap_or(0),
            info.ranks
                .get(seat)
                .and_then(|h| h.get(k))
                .copied()
                .unwrap_or(0),
        )
    };
    let own = info.hands.get(me).copied().unwrap_or([NO_CARD; HAND]);
    // 1. A card surely playable.
    for (k, &c) in own.iter().enumerate() {
        let (cm, rm) = knows(me, k);
        if c == HIDDEN && surely_playable(info.fireworks, cm, rm) {
            return Move::Play(k).encode();
        }
    }
    // 2. Hint a playable card whose holder does not know it fully.
    if info.hints > 0 {
        for offset in 1..players {
            let seat = (me + offset) % players;
            let hand = info.hands.get(seat).copied().unwrap_or([NO_CARD; HAND]);
            for (k, &card) in hand.iter().enumerate() {
                if card >= NO_CARD {
                    continue;
                }
                let playable = info
                    .fireworks
                    .get(colour(card))
                    .is_some_and(|&f| usize::from(f) == rank(card));
                if !playable {
                    continue;
                }
                // Rank first, then colour: once both are known the holder
                // plays it as surely playable.
                let (cm, rm) = knows(seat, k);
                let m = if rm != 1 << rank(card) {
                    Move::HintRank {
                        offset,
                        rank: rank(card),
                    }
                } else if cm != 1 << colour(card) {
                    Move::HintColour {
                        offset,
                        colour: colour(card),
                    }
                } else {
                    continue;
                };
                if table.legal(m) {
                    return m.encode();
                }
            }
        }
    }
    // 3. Discard, the oldest card without information first.
    if info.hints < MAX_HINTS {
        let unknown =
            (0..HAND).find(|&k| own.get(k) == Some(&HIDDEN) && knows(me, k) == (ALL, ALL));
        if let Some(k) = unknown.or_else(|| (0..HAND).find(|&k| own.get(k) == Some(&HIDDEN))) {
            return Move::Discard(k).encode();
        }
    }
    // 4. Any legal hint, else the no-op.
    (super::HINT_COLOUR..ACTIONS)
        .find(|&a| Move::decode(a).is_some_and(|m| table.legal(m)))
        .unwrap_or_else(|| table.noop())
}

/// The baseline through `Heuristic` (ADR-0079 until `InfoHeuristic`).
impl Heuristic for Lanterns {
    fn act(
        _: &LanternsConfig,
        state: &LanternsState,
        agent: AgentSlot,
        _: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let action = act_info(&info_of(state, agent.index()));
        out.set_discrete(0, i32::try_from(action).unwrap_or(0));
    }
}
