// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact determinization of Lanterns (SPEC 9.3, ADR-0035; #380 review r1).
//!
//! The observer sees every card but its own hand and the deck. Its own
//! slots are constrained by its knowledge (the colours and ranks each slot
//! may still be, narrowed by positive and negative hints). Among the
//! physical deals of the unseen cards that are consistent with that
//! knowledge, every deal is equally likely given the history, so the
//! sampler draws one **uniformly**: a deal is chosen with probability
//! proportional to the number of ways the unseen physical cards realise it.
//!
//! Filling slots one at a time, each from its allowed cards, is not
//! uniform when slots compete for the same cards. Instead a dynamic
//! program over card kinds and subsets of own slots counts, exactly, the
//! weighted assignments: `ways[k][filled]` is the number of ways to put
//! cards of the first `k` kinds into the slots of `filled`, a kind with
//! `n` unseen copies filling a subset `A` in `n (n-1) ... (n-|A|+1)` ways.
//! With at most 5 slots and 25 kinds this is `25 * 32 * 32` steps and the
//! counts stay below `50^5`, so a `u64` holds them and one `below_u32`
//! draw per kind samples backwards from the full subset. The remaining
//! unseen cards are then shuffled into the deck.

use tessera_rl_num::rng::StepRng;

use super::{
    DECK, HAND, HIDDEN, KINDS, LanternsInfo, LanternsState, MAX_AGENTS, NO_CARD, RANKS,
    full_counts, shuffle,
};

/// Subsets of own slots.
const SUBSETS: usize = 1 << HAND;

/// Unseen copies of each kind: the full deck less every card the observer
/// sees (other hands, discards, fireworks).
#[must_use]
pub fn unseen(info: &LanternsInfo) -> [u8; KINDS] {
    let mut left = full_counts();
    let mut take = |kind: usize| {
        if let Some(n) = left.get_mut(kind) {
            *n = n.saturating_sub(1);
        }
    };
    for hand in &info.hands {
        for &c in hand {
            if c < NO_CARD {
                take(usize::from(c));
            }
        }
    }
    for (kind, &n) in info.discards.iter().enumerate() {
        for _ in 0..n {
            take(kind);
        }
    }
    for (c, &f) in info.fireworks.iter().enumerate() {
        for r in 0..usize::from(f) {
            take(c * RANKS + r);
        }
    }
    left
}

/// The observer's slots that hold a card, as a bit set.
fn own_slots(info: &LanternsInfo) -> usize {
    let me = usize::from(info.me);
    info.hands.get(me).map_or(0, |h| {
        h.iter()
            .enumerate()
            .filter(|&(_, &c)| c == HIDDEN)
            .fold(0, |m, (k, _)| m | (1 << k))
    })
}

/// The own slots (of `slots`) whose knowledge allows `kind`.
fn allows(info: &LanternsInfo, slots: usize, kind: usize) -> usize {
    let me = usize::from(info.me);
    let (c, r) = (kind / RANKS, kind % RANKS);
    (0..HAND)
        .filter(|&k| slots & (1 << k) != 0)
        .filter(|&k| {
            let cm = info
                .colours
                .get(me)
                .and_then(|h| h.get(k))
                .copied()
                .unwrap_or(0);
            let rm = info
                .ranks
                .get(me)
                .and_then(|h| h.get(k))
                .copied()
                .unwrap_or(0);
            cm & (1 << c) != 0 && rm & (1 << r) != 0
        })
        .fold(0, |m, k| m | (1 << k))
}

/// `n (n-1) ... (n-k+1)`.
fn falling(n: u8, k: u32) -> u64 {
    (0..k).fold(1u64, |acc, i| {
        acc * u64::from(n).saturating_sub(u64::from(i))
    })
}

/// The weighted counts: `ways[k][filled]` over the first `k` kinds.
fn ways(info: &LanternsInfo, left: &[u8; KINDS], slots: usize) -> [[u64; SUBSETS]; KINDS + 1] {
    let mut w = [[0u64; SUBSETS]; KINDS + 1];
    if let Some(first) = w.first_mut().and_then(|row| row.first_mut()) {
        *first = 1;
    }
    for kind in 0..KINDS {
        let n = left.get(kind).copied().unwrap_or(0);
        let open = allows(info, slots, kind);
        for filled in 0..SUBSETS {
            let base = w
                .get(kind)
                .and_then(|row| row.get(filled))
                .copied()
                .unwrap_or(0);
            if base == 0 {
                continue;
            }
            // Every subset `a` of the open slots not yet filled.
            let free = open & !filled;
            let mut a = free;
            loop {
                let f = falling(n, a.count_ones());
                if f > 0
                    && let Some(cell) = w.get_mut(kind + 1).and_then(|row| row.get_mut(filled | a))
                {
                    *cell += base * f;
                }
                if a == 0 {
                    break;
                }
                a = (a - 1) & free;
            }
        }
    }
    w
}

/// A uniform deal consistent with `info`, as a full state (ADR-0035: only
/// the information state is read). With no consistent deal (an
/// information state no game reaches), the own slots get the first unseen
/// cards in kind order.
#[must_use]
pub fn sample(info: &LanternsInfo, rng: &mut StepRng) -> LanternsState {
    let mut left = unseen(info);
    let slots = own_slots(info);
    let table = ways(info, &left, slots);
    let me = usize::from(info.me);
    let mut own = [NO_CARD; HAND];
    let mut target = slots;
    for kind in (0..KINDS).rev() {
        let row_total = table
            .get(kind + 1)
            .and_then(|row| row.get(target))
            .copied()
            .unwrap_or(0);
        if row_total == 0 {
            break;
        }
        let n = left.get(kind).copied().unwrap_or(0);
        let open = allows(info, target, kind);
        // The subsets `a` of the open slots in `target`, each weighted by
        // the ways to fill the rest with earlier kinds.
        let mut pick = u64::from(rng.below_u32(bound(row_total)));
        let mut a = open;
        loop {
            let falls = falling(n, a.count_ones());
            let rest = table
                .get(kind)
                .and_then(|row| row.get(target & !a))
                .copied()
                .unwrap_or(0);
            let weight = rest * falls;
            if pick < weight {
                for s in 0..HAND {
                    if a & (1 << s) != 0
                        && let Some(c) = own.get_mut(s)
                    {
                        *c = u8::try_from(kind).unwrap_or(0);
                    }
                }
                if let Some(m) = left.get_mut(kind) {
                    *m = m.saturating_sub(u8::try_from(a.count_ones()).unwrap_or(0));
                }
                target &= !a;
                break;
            }
            pick -= weight;
            if a == 0 {
                break;
            }
            a = (a - 1) & open;
        }
    }
    fallback(&mut own, &mut left, slots);
    build(info, me, own, left, rng)
}

/// Fills own slots still empty from the first unseen kinds (only for an
/// inconsistent information state).
fn fallback(own: &mut [u8; HAND], left: &mut [u8; KINDS], slots: usize) {
    for (s, c) in own.iter_mut().enumerate() {
        if slots & (1 << s) == 0 || *c != NO_CARD {
            continue;
        }
        // Only an information state no game reaches gets here (review
        // headA r1): tests must not pass by this path silently.
        debug_assert!(*c != NO_CARD, "no consistent deal for own slot {s}");
        if let Some(kind) = left.iter().position(|&n| n > 0) {
            *c = u8::try_from(kind).unwrap_or(0);
            if let Some(n) = left.get_mut(kind) {
                *n -= 1;
            }
        }
    }
}

/// The state of the deal: the drawn prefix of the deck holds every card in
/// play (canonical order), the rest the shuffled unseen cards.
fn build(
    info: &LanternsInfo,
    me: usize,
    own: [u8; HAND],
    left: [u8; KINDS],
    rng: &mut StepRng,
) -> LanternsState {
    let mut hands = info.hands;
    if let Some(h) = hands.get_mut(me) {
        *h = own;
    }
    let mut deck = [0u8; DECK];
    let mut filled = 0usize;
    let mut push = |card: u8| {
        if let Some(d) = deck.get_mut(filled) {
            *d = card;
        }
        filled += 1;
        filled
    };
    let mut next = 0usize;
    for hand in hands.iter().take(MAX_AGENTS) {
        for &c in hand {
            if c < NO_CARD {
                next = push(c);
            }
        }
    }
    for (kind, &n) in info.discards.iter().enumerate() {
        for _ in 0..n {
            next = push(u8::try_from(kind).unwrap_or(0));
        }
    }
    for (c, &f) in info.fireworks.iter().enumerate() {
        for r in 0..usize::from(f) {
            next = push(u8::try_from(c * RANKS + r).unwrap_or(0));
        }
    }
    for (kind, &n) in left.iter().enumerate() {
        for _ in 0..n {
            push(u8::try_from(kind).unwrap_or(0));
        }
    }
    if let Some(rest) = deck.get_mut(next.min(DECK)..) {
        shuffle(rng, rest);
    }
    LanternsState {
        deck,
        next: u8::try_from(next.min(DECK)).unwrap_or(0),
        hands,
        colours: info.colours,
        ranks: info.ranks,
        fireworks: info.fireworks,
        discards: info.discards,
        hints: info.hints,
        fuses: info.fuses,
        final_turns: info.final_turns,
        to_move: info.to_move,
        players: info.players,
        last_action: info.last_action,
        last_actor: info.last_actor,
        done: info.done,
    }
}

/// The most weighted assignments a table can count: the falling factorial
/// of the deck over a full hand, `50 * 49 * 48 * 47 * 46`, below `2^32`,
/// so every total fits one `below_u32` draw.
const MAX_WAYS: u64 = 50 * 49 * 48 * 47 * 46;
const _: () = assert!(MAX_WAYS < 1 << 32 && DECK == 50 && HAND == 5);

/// The draw bound for a nonzero table total (at most [`MAX_WAYS`]).
fn bound(n: u64) -> core::num::NonZeroU32 {
    debug_assert!(n > 0 && n <= MAX_WAYS, "table total {n} out of range");
    u32::try_from(n)
        .ok()
        .and_then(core::num::NonZeroU32::new)
        .unwrap_or(core::num::NonZeroU32::MAX)
}

/// The exact probability of each own-hand assignment (kinds per slot,
/// [`NO_CARD`] for slots without a card), by brute-force enumeration of
/// the physical unseen cards: for tests of [`sample`].
#[must_use]
pub fn exact_distribution(info: &LanternsInfo) -> Vec<([u8; HAND], f64)> {
    let left = unseen(info);
    let slots = own_slots(info);
    let mut cards = Vec::new();
    for (kind, &n) in left.iter().enumerate() {
        for _ in 0..n {
            cards.push(u8::try_from(kind).unwrap_or(0));
        }
    }
    let own: Vec<usize> = (0..HAND).filter(|&k| slots & (1 << k) != 0).collect();
    let mut counts: Vec<([u8; HAND], u64)> = Vec::new();
    let mut used = vec![false; cards.len()];
    let mut hand = [NO_CARD; HAND];
    enumerate(info, &cards, &own, 0, &mut used, &mut hand, &mut counts);
    let total: u64 = counts.iter().map(|(_, n)| n).sum();
    #[expect(clippy::cast_precision_loss, reason = "test-scale counts")]
    counts
        .into_iter()
        .map(|(h, n)| (h, n as f64 / total.max(1) as f64))
        .collect()
}

fn enumerate(
    info: &LanternsInfo,
    cards: &[u8],
    own: &[usize],
    depth: usize,
    used: &mut [bool],
    hand: &mut [u8; HAND],
    out: &mut Vec<([u8; HAND], u64)>,
) {
    let Some(&s) = own.get(depth) else {
        match out.iter_mut().find(|(h, _)| h == hand) {
            Some((_, n)) => *n += 1,
            None => out.push((*hand, 1)),
        }
        return;
    };
    for i in 0..cards.len() {
        let (Some(&card), Some(&u)) = (cards.get(i), used.get(i)) else {
            continue;
        };
        if u || allows(info, 1 << s, usize::from(card)) == 0 {
            continue;
        }
        if let (Some(flag), Some(slot)) = (used.get_mut(i), hand.get_mut(s)) {
            *flag = true;
            *slot = card;
        }
        enumerate(info, cards, own, depth + 1, used, hand, out);
        if let (Some(flag), Some(slot)) = (used.get_mut(i), hand.get_mut(s)) {
            *flag = false;
            *slot = NO_CARD;
        }
    }
}
