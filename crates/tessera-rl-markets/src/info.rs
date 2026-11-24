// SPDX-License-Identifier: MIT OR Apache-2.0

//! The information boundary (ADR-0035, ADR-0046; design section 3, with
//! correction 3 of review r1 and the blocking point of the PR 3 review).
//!
//! An agent's information state ([`LobInfo`]) holds only what it knows:
//!
//! - its own account, last equity, signal and `quote_step`;
//! - its own resting orders and queued items, with their ids and arrival
//!   numbers replaced by their rank among its own (the absolute numbers
//!   would count the other agents' orders);
//! - the public part of the state: the step, `last_mid`, the mid
//!   history, the last trade price, the reset price, the alive flags, and
//!   this step's trade prints (the observation shows them);
//! - the book as the observation shows it: the best `depth` aggregated
//!   levels per side, price and total lots.
//!
//! Nothing else: no other agent's orders, owners, groups, queued items,
//! accounts or signals, and not the fundamental. No ring of past
//! observations is kept: everything a sample must reproduce is above, and
//! the mid history and `last_mid` are already part of the state.
//!
//! [`Determinize::sample`] keeps the information state exactly and
//! constructs the rest:
//!
//! 1. the fundamental uniformly in `signal +- noise`, within
//!    `[min_tick, max_tick]` (only the last signal is known; a
//!    history-aware posterior would be tighter);
//! 2. the other agents' resting orders: each visible level's lots minus
//!    the observer's own lots at that price are split into orders of at
//!    most `MAX_LOT` lots and assigned to the other alive agents, largest
//!    level first, each piece to the agent with the most remaining room
//!    (`2 x max_position` lots over both sides, at most `MAX_OPEN`
//!    orders). Levels beyond the visible depth are empty in a sample;
//! 3. the other agents' queued items are empty (a possible world, biased:
//!    a sample never holds hidden orders in flight, so search
//!    underestimates the book of the next `latency` steps);
//! 4. inventories for the other alive agents that sum to minus the
//!    observer's (dead agents hold none in a sample) and keep
//!    `inventory + buy lots <= max_position` and
//!    `inventory - sell lots >= -max_position`. Each interval
//!    `[sell lots - max, max - buy lots]` is non-empty (step 2), and their
//!    sum covers the target: the sampled buy and sell lots are at most the
//!    true others' (part of their true orders), and the true inventories
//!    meet the same bounds with the same sum;
//! 5. cash for the other alive agents: each its reserved buy cash plus an
//!    equal share of what remains of the conserved total
//!    (`agents x initial_cash`); signals around the sampled fundamental,
//!    last equities marked at the public mid, `quote_step` 0.
//!
//! **Order ids.** The book's time priority among hidden orders is not
//! observed, so a sample uses one convention: hidden orders get the ids
//! `0..h` in the order they are built (largest level first, then the
//! pieces of a level in order), and the observer's own orders the ids
//! `h..` by their rank. At any price, the hidden orders are ahead of the
//! observer's in time priority (the pessimistic queue position for it).
//!
//! **Step 2 is not proved to always fit.** It is a bin packing with
//! splitting and an order count per bin. A true state exists, so a fitting
//! assignment exists; the greedy finds it in every case the tests reach,
//! but no proof is claimed. On a failure it retries [`RETRIES`] times with
//! a random level order, and as a last resort assigns the remaining lots
//! with the order count only. The visible book is then still exact and
//! the sample still passes `Validate` (which bounds `|inventory|`, not the
//! open lots); only correction 3's open-lot bound may be exceeded for one
//! agent. `tests/determinize.rs` checks every sample against it.

#![deny(clippy::float_arithmetic)]

use core::num::NonZeroU32;

use tessera_rl_core::{AgentSlot, Bool8, Determinize, ImperfectInformation, Reserved};
use tessera_rl_macros::EnvState;
use tessera_rl_num::rng::StepRng;

use crate::account::Account;
use crate::book::{Book, Level};
use crate::config::LobConfig;
use crate::env::LimitOrderBook;
use crate::limits::{BOOK_CAP, MAX_AGENTS, MAX_LOT, MAX_OPEN};
use crate::order::{Order, Side};
use crate::pending::{Pending, PendingQueue};
use crate::state::{LobState, MAX_TRADES, MID_RING, Trade};

/// The deepest visible levels per side (the `depth` config bound).
pub const MAX_DEPTH: usize = 16;
/// Queued items one agent can have: one per step for at most
/// `MAX_LATENCY = 3` steps, plus the margin of the queue's sizing.
pub const MAX_OWN_PENDING: usize = 4;
/// The owner of the stand-in orders of [`public_state`].
pub const HIDDEN: u8 = u8::MAX;
/// Random level orders tried after the largest-first greedy fails.
pub const RETRIES: usize = 4;

/// What one agent knows. Fields past each count are zero.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct LobInfo {
    /// The observer's account.
    pub account: Account,
    /// The observer's last equity.
    pub last_equity: i64,
    /// The observer's resting orders by id, with `seq` replaced by rank.
    pub own_orders: [Order; MAX_OPEN],
    /// The observer's queued items by arrival, with `arrival` replaced by
    /// rank.
    pub own_pending: [Pending; MAX_OWN_PENDING],
    /// The best visible bid levels, `[price, total lots]`, best first.
    pub bids: [[u32; 2]; MAX_DEPTH],
    /// The best visible ask levels, as `bids`.
    pub asks: [[u32; 2]; MAX_DEPTH],
    /// This step's trade prints.
    pub trades: [Trade; MAX_TRADES],
    /// The mid history.
    pub mids: [u32; MID_RING],
    /// Steps played.
    pub step: u32,
    /// The mid at the end of the previous step.
    pub last_mid: u32,
    /// The last trade price.
    pub last_trade: u32,
    /// The fundamental at reset.
    pub reset_price: u32,
    /// Entries of `trades` in use.
    pub trade_count: u32,
    /// The observer's last limit order step.
    pub quote_step: u32,
    /// The observer's signal.
    pub signal: i32,
    /// Who is alive.
    pub alive: [Bool8; MAX_AGENTS],
    /// The observing slot.
    pub me: u8,
    /// Entries of `own_orders` in use.
    pub own_count: u8,
    /// Entries of `own_pending` in use.
    pub pending_count: u8,
    /// Zero.
    pub pad: Reserved<1>,
}

const _: () = assert!(size_of::<LobInfo>() == 776);

/// The visible levels of `side`, `[price, lots]`, best first.
fn visible(state: &LobState, side: Side, depth: usize) -> [[u32; 2]; MAX_DEPTH] {
    let mut levels = [Level::default(); MAX_DEPTH];
    let n = state.book.levels(
        side,
        levels.get_mut(..depth.min(MAX_DEPTH)).unwrap_or(&mut []),
    );
    let mut out = [[0; 2]; MAX_DEPTH];
    for (o, l) in out.iter_mut().zip(levels.iter().take(n)) {
        *o = [l.price, l.lots];
    }
    out
}

/// The information state of slot `me` in `state` under `cfg`.
#[must_use]
pub fn info_of(cfg: &LobConfig, state: &LobState, me: usize) -> LobInfo {
    let owner = u8::try_from(me).unwrap_or(HIDDEN);
    let depth = usize::try_from(cfg.depth).unwrap_or(0);
    let mut own_orders = [Order::default(); MAX_OPEN];
    let own_count = state.book.own_orders(owner, &mut own_orders);
    for (rank, o) in own_orders.iter_mut().take(own_count).enumerate() {
        o.seq = u32::try_from(rank).unwrap_or(0);
    }
    // Own queued items by arrival, as ranks.
    let mut own_pending = [Pending::default(); MAX_OWN_PENDING];
    let mut pending_count = 0;
    let mut after: Option<u32> = None;
    for slot in &mut own_pending {
        let next = state
            .pending
            .items()
            .filter(|p| p.owner == owner && after.is_none_or(|a| p.arrival > a))
            .min_by_key(|p| p.arrival)
            .copied();
        let Some(p) = next else {
            break;
        };
        after = Some(p.arrival);
        *slot = Pending {
            arrival: u32::try_from(pending_count).unwrap_or(0),
            ..p
        };
        pending_count += 1;
    }
    LobInfo {
        account: state.account(me),
        last_equity: state.last_equity.get(me).copied().unwrap_or(0),
        own_orders,
        own_pending,
        bids: visible(state, Side::Buy, depth),
        asks: visible(state, Side::Sell, depth),
        trades: state.trades,
        mids: state.mids,
        step: state.step,
        last_mid: state.last_mid,
        last_trade: state.last_trade,
        reset_price: state.reset_price,
        trade_count: state.trade_count,
        quote_step: state.quote_step.get(me).copied().unwrap_or(0),
        signal: state.signal.get(me).copied().unwrap_or(0),
        alive: state.alive,
        me: owner,
        own_count: u8::try_from(own_count).unwrap_or(0),
        pending_count: u8::try_from(pending_count).unwrap_or(0),
        pad: Reserved::ZERO,
    }
}

impl ImperfectInformation for LimitOrderBook {
    type InfoState = LobInfo;

    fn info_state(cfg: &LobConfig, state: &LobState, agent: AgentSlot) -> LobInfo {
        info_of(cfg, state, agent.index())
    }
}

impl LobInfo {
    /// The observer's open orders in use.
    #[must_use]
    pub fn own(&self) -> &[Order] {
        self.own_orders
            .get(..usize::from(self.own_count))
            .unwrap_or(&[])
    }

    /// The observer's queued items in use.
    #[must_use]
    pub fn own_queued(&self) -> &[Pending] {
        self.own_pending
            .get(..usize::from(self.pending_count))
            .unwrap_or(&[])
    }

    /// The visible levels of `side` with the lots of other owners
    /// (`total - own` at that price): `(side, price, hidden lots)`.
    fn hidden_levels(&self) -> impl Iterator<Item = (Side, u32, u32)> + '_ {
        Side::ALL.into_iter().flat_map(move |side| {
            let levels = match side {
                Side::Buy => &self.bids,
                Side::Sell => &self.asks,
            };
            levels
                .iter()
                .filter(|l| l[1] > 0)
                .map(move |&[price, lots]| {
                    let own: u32 = self
                        .own()
                        .iter()
                        .filter(|o| o.side == side && o.price == price)
                        .map(|o| o.qty)
                        .sum();
                    // `own` is part of `lots` in any information state
                    // taken from a state; 0 for an inconsistent one.
                    (side, price, lots.saturating_sub(own))
                })
                .filter(|&(_, _, hidden)| hidden > 0)
        })
    }
}

/// The public and own part of a state for `info`: every field the
/// information state fixes, the book holding `hidden` (the other agents'
/// orders, ids `0..`) then the observer's own orders, and its own queue.
fn base_state(info: &LobInfo, hidden: &[Order]) -> LobState {
    let me = usize::from(info.me);
    let h = u32::try_from(hidden.len()).unwrap_or(0);
    let own = info.own().iter().map(|o| Order {
        seq: h + o.seq,
        ..*o
    });
    let own_count = u32::from(info.own_count);
    let book = Book::from_orders(hidden.iter().copied().chain(own), h + own_count);
    let queued = u32::from(info.pending_count);
    let pending = PendingQueue::from_items(info.own_queued().iter().copied(), queued);
    let mut state = LobState {
        book: book.unwrap_or_default(),
        pending: pending.unwrap_or_default(),
        trades: info.trades,
        step: info.step,
        last_trade: info.last_trade,
        trade_count: info.trade_count,
        reset_price: info.reset_price,
        mids: info.mids,
        alive: info.alive,
        last_mid: info.last_mid,
        ..LobState::default()
    };
    if let (Some(a), Some(e), Some(s), Some(q)) = (
        state.accounts.get_mut(me),
        state.last_equity.get_mut(me),
        state.signal.get_mut(me),
        state.quote_step.get_mut(me),
    ) {
        *a = info.account;
        *e = info.last_equity;
        *s = info.signal;
        *q = info.quote_step;
    }
    state
}

/// A state the observer cannot tell from the true one for the purpose of
/// its own decisions: its own part, the public part, and one stand-in
/// order owned by [`HIDDEN`] per visible level for the other agents' lots.
/// Masks and the action decoder read the same values from it as from the
/// true state (touch, mid, visible levels, own orders and queue), so a
/// heuristic acting on it acts legally on the true state.
#[must_use]
pub fn public_state(info: &LobInfo) -> LobState {
    let mut hidden = [Order::default(); 2 * MAX_DEPTH];
    let mut n = 0;
    for (side, price, qty) in info.hidden_levels() {
        if let Some(slot) = hidden.get_mut(n) {
            *slot = Order {
                price,
                qty,
                seq: u32::try_from(n).unwrap_or(0),
                owner: HIDDEN,
                side,
                pad: Reserved::ZERO,
            };
            n += 1;
        }
    }
    base_state(info, hidden.get(..n).unwrap_or(&[]))
}

/// A uniform draw in `[0, n)`; 0 for `n = 0`.
fn below(rng: &mut StepRng, n: usize) -> usize {
    let bound = u32::try_from(n).ok().and_then(NonZeroU32::new);
    bound.map_or(0, |b| usize::try_from(rng.below_u32(b)).unwrap_or(0))
}

/// Per-agent open lots per side, reserved buy cash and orders.
#[derive(Debug, Clone, Copy, Default)]
struct Load {
    lots: [u64; 2],
    cash: u64,
    orders: usize,
}

/// The built hidden orders and the loads they put on each agent.
struct Built {
    orders: [Order; BOOK_CAP],
    count: usize,
    loads: [Load; MAX_AGENTS],
}

/// Step 2: splits the hidden lots of `levels` (in the order `order`) into
/// orders for `others`, each piece to the agent with the most remaining
/// room (`cap` lots, `MAX_OPEN` orders, first in `others` on a tie), at
/// most `MAX_LOT` and at most `limit` orders in all. `None` if a lot finds
/// no room.
fn build(
    levels: &[(Side, u32, u32)],
    order: &[usize],
    others: &[usize],
    cap: u64,
    limit: usize,
) -> Option<Built> {
    let mut built = Built {
        orders: [Order::default(); BOOK_CAP],
        count: 0,
        loads: [Load::default(); MAX_AGENTS],
    };
    for &i in order {
        let &(side, price, total) = levels.get(i)?;
        let mut left = u64::from(total);
        // Each piece fills the level, an agent's room, or MAX_LOT: bounded
        // by the order limit below.
        while left > 0 {
            if built.count >= limit {
                return None;
            }
            let room = |l: &Load| cap.saturating_sub(l.lots[0] + l.lots[1]);
            let j = others
                .iter()
                .copied()
                .filter(|&j| {
                    built
                        .loads
                        .get(j)
                        .is_some_and(|l| l.orders < MAX_OPEN && room(l) > 0)
                })
                .max_by_key(|&j| (built.loads.get(j).map_or(0, room), core::cmp::Reverse(j)))?;
            let load = built.loads.get_mut(j)?;
            let take = left.min(room(load)).min(u64::from(MAX_LOT));
            let qty = u32::try_from(take).ok()?;
            *load.lots.get_mut(side.index())? += take;
            load.orders += 1;
            if side == Side::Buy {
                load.cash += take * u64::from(price);
            }
            *built.orders.get_mut(built.count)? = Order {
                price,
                qty,
                seq: u32::try_from(built.count).ok()?,
                owner: u8::try_from(j).ok()?,
                side,
                pad: Reserved::ZERO,
            };
            built.count += 1;
            left -= take;
        }
    }
    Some(built)
}

/// Step 2 with its retries and last resort.
fn construct(cfg: &LobConfig, info: &LobInfo, others: &[usize], rng: &mut StepRng) -> Built {
    let mut levels = [(Side::Buy, 0u32, 0u32); 2 * MAX_DEPTH];
    let mut n = 0;
    for level in info.hidden_levels() {
        if let Some(slot) = levels.get_mut(n) {
            *slot = level;
            n += 1;
        }
    }
    let levels = levels.get(..n).unwrap_or(&[]);
    let mut order = [0usize; 2 * MAX_DEPTH];
    for (i, o) in order.iter_mut().enumerate() {
        *o = i;
    }
    let order = order.get_mut(..n).unwrap_or(&mut []);
    // Largest level first.
    order.sort_unstable_by_key(|&i| (core::cmp::Reverse(levels.get(i).map_or(0, |l| l.2)), i));
    let cap = 2 * u64::from(cfg.max_position);
    let limit = BOOK_CAP - usize::from(info.own_count);
    if let Some(b) = build(levels, order, others, cap, limit) {
        return b;
    }
    for _ in 0..RETRIES {
        for i in (1..order.len()).rev() {
            let j = below(rng, i + 1);
            order.swap(i, j);
        }
        if let Some(b) = build(levels, order, others, cap, limit) {
            return b;
        }
    }
    build(levels, order, others, u64::MAX, limit).unwrap_or(Built {
        orders: [Order::default(); BOOK_CAP],
        count: 0,
        loads: [Load::default(); MAX_AGENTS],
    })
}

/// Step 4: inventories of `others` that sum to `target`, each within
/// `[sell lots - max, max - buy lots]` (and `[-max, max]`).
fn inventories(
    cfg: &LobConfig,
    loads: &[Load; MAX_AGENTS],
    others: &[usize],
    mut target: i64,
    rng: &mut StepRng,
) -> [i64; MAX_AGENTS] {
    let max = i64::from(cfg.max_position);
    let mut inventory = [0i64; MAX_AGENTS];
    for &j in others {
        let l = loads.get(j).copied().unwrap_or_default();
        let low = (i64::try_from(l.lots[1]).unwrap_or(max) - max).clamp(-max, max);
        if let Some(v) = inventory.get_mut(j) {
            *v = low;
        }
        target -= low;
    }
    let start = below(rng, others.len());
    for k in 0..others.len() {
        let Some(&j) = others.get((start + k) % others.len().max(1)) else {
            continue;
        };
        let l = loads.get(j).copied().unwrap_or_default();
        let high = (max - i64::try_from(l.lots[0]).unwrap_or(max)).min(max);
        if let Some(v) = inventory.get_mut(j) {
            let add = target.clamp(0, (high - *v).max(0));
            *v += add;
            target -= add;
        }
    }
    inventory
}

impl Determinize for LimitOrderBook {
    fn sample(cfg: &LobConfig, info: &LobInfo, rng: &mut StepRng, state: &mut LobState) {
        let me = usize::from(info.me);
        let n = cfg.agent_count();

        // 1. The fundamental, within the signal's noise band.
        let signal = i64::from(info.signal);
        let noise = i64::from(cfg.noise);
        let lo = (signal - noise).max(i64::from(cfg.min_tick));
        let hi = (signal + noise).min(i64::from(cfg.max_tick)).max(lo);
        let width = usize::try_from(hi - lo + 1).unwrap_or(1);
        let fundamental = lo + i64::try_from(below(rng, width)).unwrap_or(0);

        // 2 and 3. The other agents' resting orders; their queues empty.
        let mut others = [0usize; MAX_AGENTS];
        let mut n_others = 0;
        for slot in (0..n).filter(|&s| s != me && info.alive.get(s).is_some_and(|a| a.get())) {
            if let Some(o) = others.get_mut(n_others) {
                *o = slot;
                n_others += 1;
            }
        }
        let others = others.get(..n_others).unwrap_or(&[]);
        let built = construct(cfg, info, others, rng);
        *state = base_state(info, built.orders.get(..built.count).unwrap_or(&[]));
        state.fundamental = u32::try_from(fundamental).unwrap_or(cfg.min_tick);
        let loads = built.loads;

        // 4. Inventories.
        let inventory = inventories(cfg, &loads, others, -i64::from(info.account.inventory), rng);

        // 5. Cash: reserved buy cash plus an equal share of the rest of
        // the conserved total; dead agents hold the initial cash.
        let agents = i64::try_from(n).unwrap_or(0);
        let dead = (0..n).filter(|&s| s != me && !state.is_alive(s)).count();
        let dead = i64::try_from(dead).unwrap_or(0);
        let total = agents * cfg.initial_cash - info.account.cash - dead * cfg.initial_cash;
        let reserved_of = |j: usize| {
            loads
                .get(j)
                .map_or(0, |l| i64::try_from(l.cash).unwrap_or(0))
        };
        let reserved: i64 = others.iter().map(|&j| reserved_of(j)).sum();
        let count = i64::try_from(others.len()).unwrap_or(1).max(1);
        let rest = total - reserved;
        let share = rest.div_euclid(count);
        let mut remainder = rest - share * count;

        // Accounts, signals and last equities (marked at the public mid,
        // as rewards are) of the other used slots.
        let mid = state.touch().mid;
        let width = cfg
            .noise
            .checked_mul(2)
            .and_then(|w| w.checked_add(1))
            .and_then(NonZeroU32::new);
        for slot in (0..n).filter(|&s| s != me) {
            let account = if state.is_alive(slot) {
                let cash = reserved_of(slot) + share + remainder;
                remainder = 0;
                let lots = inventory.get(slot).copied().unwrap_or(0);
                Account {
                    cash,
                    inventory: i32::try_from(lots).unwrap_or(0),
                    pad: Reserved::ZERO,
                }
            } else {
                Account::with_cash(cfg.initial_cash)
            };
            let noise = width.map_or(0, |w| i64::from(rng.below_u32(w)) - i64::from(cfg.noise));
            if let (Some(a), Some(e), Some(s)) = (
                state.accounts.get_mut(slot),
                state.last_equity.get_mut(slot),
                state.signal.get_mut(slot),
            ) {
                *a = account;
                *e = account.equity(mid);
                *s = i32::try_from(fundamental + noise).unwrap_or(0);
            }
        }
    }
}
