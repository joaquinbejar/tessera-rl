// SPDX-License-Identifier: MIT OR Apache-2.0

//! The information boundary (ADR-0035, ADR-0046; design section 3, with
//! correction 3 of review r1).
//!
//! An agent's information state is the state with everything private to
//! other agents erased: their accounts and last equities, their signals,
//! the fundamental, and the identity of the owner of every order and
//! queued item that is not the agent's own. What stays is public (the
//! book's prices, sizes and ids, the trades, the reference prices, the mid
//! and id histories, `last_mid`, the step, who is alive) or the agent's own
//! (its account, last equity, signal, orders and queued items).
//!
//! Other owners are replaced by anonymous group labels, [`ANON`] plus a
//! rank: orders that share a hidden owner share a label, and labels are
//! ranked by first appearance (resting orders by id, then queued items by
//! arrival), which says nothing about which slot owns a group. The
//! information state therefore holds which hidden orders belong together,
//! but not to whom. That is more than the observation shows (as are the
//! other agents' queued items, kept because the scope of #381 PR 3
//! reassigns them); it is what makes a valid sample always exist, below.
//!
//! [`Determinize::sample`] keeps the information state exactly and
//! samples the rest:
//!
//! 1. the fundamental uniformly in `signal +- noise`, within
//!    `[min_tick, max_tick]`; this uses only the last signal, which is all
//!    the information state holds (a history-aware posterior would be
//!    tighter);
//! 2. an owner for every group: a uniformly drawn injection of the groups
//!    into the other alive agents (a partial Fisher-Yates shuffle). Every
//!    group is the true order set of one alive agent, so it already holds
//!    at most `MAX_OPEN` resting plus pending limit orders and at most
//!    `2 x max_position` open and pending lots; there are never more
//!    groups than other alive agents. No rejection, no fallback: the
//!    per-agent limits of correction 3 hold by construction;
//! 3. inventories for the other alive agents that sum to minus the
//!    agent's own (lots are conserved; dead agents hold none in a sample)
//!    and keep `inventory + buy lots <= max_position` and
//!    `inventory - sell lots >= -max_position`. Each interval
//!    `[sell lots - max, max - buy lots]` is non-empty (point 2), and
//!    their sum covers the target: the total buy and sell lots of the
//!    others do not depend on who holds them, and the true inventories
//!    satisfy the same bounds with the same sum;
//! 4. cash for the other alive agents: each its reserved buy cash plus an
//!    equal share of what remains of the conserved total
//!    (`agents x initial_cash`);
//! 5. their signals around the sampled fundamental, and their last
//!    equities marked at the (public) mid.
//!
//! The sample passes [`Validate`](tessera_rl_core::Validate) and has the
//! same information state; `tests/determinize.rs` checks both on states
//! from play.

#![deny(clippy::float_arithmetic)]

use core::num::NonZeroU32;

use tessera_rl_core::{AgentSlot, Determinize, ImperfectInformation, Reserved};
use tessera_rl_macros::EnvState;
use tessera_rl_num::rng::StepRng;

use crate::account::Account;
use crate::config::LobConfig;
use crate::env::LimitOrderBook;
use crate::limits::MAX_AGENTS;
use crate::order::{Kind, Side};
use crate::state::LobState;

/// The first anonymous group label: the owner of an order or item that is
/// not the observer's becomes `ANON + rank` of its hidden owner.
pub const ANON: u8 = 0x80;

/// What one agent knows.
#[derive(EnvState, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct LobInfo {
    /// The state with every other agent's private part erased.
    pub view: LobState,
    /// The observing slot.
    pub me: u8,
    /// Zero.
    pub pad: Reserved<7>,
}

/// Whether `owner` is an anonymous group label.
#[must_use]
pub const fn is_anon(owner: u8) -> bool {
    owner >= ANON
}

/// The rank of each owner other than `me` by first appearance: resting
/// orders by id, then queued items by arrival. Keys are unique (ids and
/// arrivals are), so ranks are distinct; owners with nothing get none.
fn ranks(state: &LobState, me: u8) -> [Option<u8>; MAX_AGENTS] {
    // (class, value): 0 with the smallest resting id, 1 with the smallest
    // arrival, for owners that hold only queued items.
    let mut key: [Option<(u8, u32)>; MAX_AGENTS] = [None; MAX_AGENTS];
    let mut see = |owner: u8, k: (u8, u32)| {
        if owner != me
            && let Some(slot) = key.get_mut(usize::from(owner))
            && slot.is_none_or(|old| k < old)
        {
            *slot = Some(k);
        }
    };
    for o in state.book.orders() {
        see(o.owner, (0, o.seq));
    }
    for p in state.pending.items() {
        see(p.owner, (1, p.arrival));
    }
    let mut out = [None; MAX_AGENTS];
    for (slot, k) in key.iter().enumerate() {
        if let (Some(k), Some(r)) = (k, out.get_mut(slot)) {
            let rank = key.iter().filter(|o| o.is_some_and(|o| o < *k)).count();
            *r = u8::try_from(rank).ok();
        }
    }
    out
}

/// The information state of slot `me` in `state`.
#[must_use]
pub fn public_view(state: &LobState, me: usize) -> LobInfo {
    let owner = u8::try_from(me).unwrap_or(u8::MAX);
    let rank = ranks(state, owner);
    let label = |o: u8| {
        if o == owner {
            o
        } else {
            let r = rank.get(usize::from(o)).copied().flatten().unwrap_or(0);
            ANON + r
        }
    };
    let mut view = *state;
    view.book.set_owners(|o| label(o.owner));
    view.pending.set_owners(|p| label(p.owner));
    for (slot, ((a, e), sig)) in view
        .accounts
        .iter_mut()
        .zip(view.last_equity.iter_mut())
        .zip(view.signal.iter_mut())
        .enumerate()
    {
        if slot != me {
            *a = Account::default();
            *e = 0;
            *sig = 0;
        }
    }
    view.fundamental = 0;
    LobInfo {
        view,
        me: owner,
        pad: Reserved::ZERO,
    }
}

impl ImperfectInformation for LimitOrderBook {
    type InfoState = LobInfo;

    fn info_state(_: &LobConfig, state: &LobState, agent: AgentSlot) -> LobInfo {
        public_view(state, agent.index())
    }
}

/// A uniform draw in `[0, n)`; 0 for `n = 0`.
fn below(rng: &mut StepRng, n: usize) -> usize {
    let bound = u32::try_from(n).ok().and_then(NonZeroU32::new);
    bound.map_or(0, |b| usize::try_from(rng.below_u32(b)).unwrap_or(0))
}

/// Per-agent open and pending lots per side and reserved buy cash.
#[derive(Debug, Clone, Copy, Default)]
struct Load {
    lots: [u64; 2],
    cash: u64,
}

impl Load {
    fn add(&mut self, side: Side, qty: u32, price: u32) {
        if let Some(l) = self.lots.get_mut(side.index()) {
            *l += u64::from(qty);
        }
        self.cash += buy_cash(side, qty, price);
    }
}

/// Step 2: a uniformly drawn injection of the group labels into `others`,
/// written into the book and the queue; returns each agent's load.
/// `None` when there are more groups than other alive agents, which no
/// valid state has (every group is one alive agent's orders).
fn assign_groups(
    cfg: &LobConfig,
    state: &mut LobState,
    others: &[usize],
    rng: &mut StepRng,
) -> Option<[Load; MAX_AGENTS]> {
    let groups = state
        .book
        .orders()
        .map(|o| o.owner)
        .chain(state.pending.items().map(|p| p.owner))
        .filter(|&o| is_anon(o))
        .map(|o| usize::from(o - ANON) + 1)
        .max()
        .unwrap_or(0);
    if groups > others.len() {
        return None;
    }
    // Partial Fisher-Yates: the first `groups` entries of a shuffle.
    let mut pick = [0usize; MAX_AGENTS];
    for (dst, &src) in pick.iter_mut().zip(others) {
        *dst = src;
    }
    for i in 0..groups {
        let j = i + below(rng, others.len() - i);
        pick.swap(i, j);
    }
    let owner_of = |o: u8| -> u8 {
        if is_anon(o) {
            pick.get(usize::from(o - ANON))
                .and_then(|&j| u8::try_from(j).ok())
                .unwrap_or(o)
        } else {
            o
        }
    };
    state.book.set_owners(|o| owner_of(o.owner));
    state.pending.set_owners(|p| owner_of(p.owner));
    let mut loads = [Load::default(); MAX_AGENTS];
    for &j in others {
        let owner = u8::try_from(j).ok()?;
        let load = loads.get_mut(j)?;
        for o in state.book.orders().filter(|o| o.owner == owner) {
            load.add(o.side, o.qty, o.price);
        }
        for p in state.pending.items().filter(|p| p.owner == owner) {
            match p.kind {
                Kind::Cancel => {}
                Kind::Limit => load.add(p.side, p.qty, p.price),
                Kind::Market => {
                    let price = if p.price > 0 { p.price } else { cfg.max_tick };
                    load.add(p.side, p.qty, price);
                }
            }
        }
    }
    Some(loads)
}

/// Step 3: inventories of `others` that sum to `target`, each within
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
        *state = info.view;
        let own = state.account(me);

        // 1. The fundamental, within the signal's noise band.
        let signal = i64::from(info.view.signal.get(me).copied().unwrap_or(0));
        let noise = i64::from(cfg.noise);
        let lo = (signal - noise).max(i64::from(cfg.min_tick));
        let hi = (signal + noise).min(i64::from(cfg.max_tick)).max(lo);
        let width = usize::try_from(hi - lo + 1).unwrap_or(1);
        let fundamental = lo + i64::try_from(below(rng, width)).unwrap_or(0);
        state.fundamental = u32::try_from(fundamental).unwrap_or(cfg.min_tick);

        // 2. Owners of the anonymous orders and items.
        let mut others = [0usize; MAX_AGENTS];
        let mut n_others = 0;
        for slot in (0..n).filter(|&s| s != me && state.is_alive(s)) {
            if let Some(o) = others.get_mut(n_others) {
                *o = slot;
                n_others += 1;
            }
        }
        let others = others.get(..n_others).unwrap_or(&[]);
        // A valid information state never has more groups than other
        // alive agents; for an invalid one the labels stay and the sample
        // fails `Validate` (an owner out of range), as it should.
        let loads = assign_groups(cfg, state, others, rng).unwrap_or([Load::default(); MAX_AGENTS]);

        // 3. Inventories.
        let inventory = inventories(cfg, &loads, others, -i64::from(own.inventory), rng);

        // 4. Cash: reserved buy cash plus an equal share of the rest of
        // the conserved total; dead agents hold the initial cash.
        let agents = i64::try_from(n).unwrap_or(0);
        let dead = (0..n).filter(|&s| s != me && !state.is_alive(s)).count();
        let dead = i64::try_from(dead).unwrap_or(0);
        let total = agents * cfg.initial_cash - own.cash - dead * cfg.initial_cash;
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

        // 5. Accounts, signals and last equities (marked at the public
        // mid, as rewards are) of the other used slots.
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

/// The cash a buy of `qty` lots at `price` reserves; 0 for a sell.
fn buy_cash(side: Side, qty: u32, price: u32) -> u64 {
    match side {
        Side::Buy => u64::from(qty) * u64::from(price),
        Side::Sell => 0,
    }
}
