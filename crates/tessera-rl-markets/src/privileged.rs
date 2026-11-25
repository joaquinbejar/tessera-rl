// SPDX-License-Identifier: MIT OR Apache-2.0

//! The privileged state of ADR-0079 item 7: everything a centralized
//! critic needs and no agent observes. Every account, every resting order
//! and queued item with its owner, the fundamental and every signal.
//!
//! [`LimitOrderBook`] implements `PrivilegedState` by forwarding to the two
//! plain functions [`state_space`] and [`write_state`], which keep the same
//! `ObsWriter` contract as an observation (leaves zero on entry, ADR-0058).
//! Search never reads it (ADR-0079 item 7.3).
//!
//! All leaves are `i64`, one dtype; absent rows are zero and their `_mask`
//! entry is 0.

#![deny(clippy::float_arithmetic)]

use tessera_rl_core::io::ObsWriter;
use tessera_rl_core::space::Space;
use tessera_rl_core::{ObsLeaf, Observation, PrivilegedState, SpaceError};
use tessera_rl_macros::Observation;

use crate::config::LobConfig;
use crate::env::LimitOrderBook;
use crate::limits::{BOOK_CAP, MAX_AGENTS, PENDING_CAP};
use crate::order::{Kind, Side};
use crate::state::LobState;

/// Columns of a `orders` row.
pub const ORDER_COLUMNS: u32 = 5;
/// Columns of a `pending` row.
pub const PENDING_COLUMNS: u32 = 8;
/// Columns of an `agents` row.
pub const AGENT_COLUMNS: u32 = 5;

const _: () = assert!(BOOK_CAP == 128 && PENDING_CAP == 64 && MAX_AGENTS == 16);

/// The privileged state of one instance.
#[derive(Observation)]
pub struct LobPrivileged {
    /// Per slot: `[alive, cash, inventory, last equity, signal]`.
    #[space(box(low = i64::MIN, high = i64::MAX, dtype = i64, shape = |_: &LobConfig| [16, AGENT_COLUMNS]))]
    pub agents: ObsLeaf<i64>,
    /// `[fundamental, step, last trade, reset price, last mid]`.
    #[space(box(low = i64::MIN, high = i64::MAX, dtype = i64, shape = |_: &LobConfig| [5]))]
    pub market: ObsLeaf<i64>,
    /// Resting orders by id: `[owner, side (0 buy, 1 sell), price, lots, id]`.
    #[space(box(low = 0, high = i64::MAX, dtype = i64, shape = |_: &LobConfig| [128, ORDER_COLUMNS]))]
    pub orders: ObsLeaf<i64>,
    /// 1 for each row of `orders` in use.
    #[space(box(low = 0, high = 1, dtype = i64, shape = |_: &LobConfig| [128]))]
    pub orders_mask: ObsLeaf<i64>,
    /// Queued items by arrival: `[owner, kind (0 limit, 1 market, 2
    /// cancel), side, price, lots, target id, ready step, arrival]`.
    #[space(box(low = 0, high = i64::MAX, dtype = i64, shape = |_: &LobConfig| [64, PENDING_COLUMNS]))]
    pub pending: ObsLeaf<i64>,
    /// 1 for each row of `pending` in use.
    #[space(box(low = 0, high = 1, dtype = i64, shape = |_: &LobConfig| [64]))]
    pub pending_mask: ObsLeaf<i64>,
}

/// The space of the privileged state buffer for `cfg`.
///
/// # Errors
///
/// A [`SpaceError`] if the space is invalid (it never is for a valid
/// config).
pub fn state_space(cfg: &LobConfig) -> Result<Space, SpaceError> {
    LobPrivileged::space(cfg)
}

const fn side_code(side: Side) -> i64 {
    match side {
        Side::Buy => 0,
        Side::Sell => 1,
    }
}

const fn kind_code(kind: Kind) -> i64 {
    match kind {
        Kind::Limit => 0,
        Kind::Market => 1,
        Kind::Cancel => 2,
    }
}

/// Writes the privileged state of `state`. Rows are in a canonical order
/// (orders by id, items by arrival), never by storage index.
pub fn write_state<O>(_: &LobConfig, state: &LobState, out: &mut ObsWriter<'_, '_, O>) {
    if let Some(mut leaf) = out.leaf_mut(LobPrivileged::AGENTS) {
        let cols = 5;
        for slot in 0..MAX_AGENTS {
            let a = state.account(slot);
            let row = [
                i64::from(state.is_alive(slot)),
                a.cash,
                i64::from(a.inventory),
                state.last_equity.get(slot).copied().unwrap_or(0),
                i64::from(state.signal.get(slot).copied().unwrap_or(0)),
            ];
            for (c, v) in row.into_iter().enumerate() {
                leaf.set(slot * cols + c, v);
            }
        }
    }
    if let Some(mut leaf) = out.leaf_mut(LobPrivileged::MARKET) {
        let row = [
            state.fundamental,
            state.step,
            state.last_trade,
            state.reset_price,
            state.last_mid,
        ];
        for (c, v) in row.into_iter().enumerate() {
            leaf.set(c, i64::from(v));
        }
    }
    // Orders by id: repeatedly the smallest id above the last written.
    let mut after: Option<u32> = None;
    for written in 0..BOOK_CAP {
        let next = state
            .book
            .orders()
            .filter(|o| after.is_none_or(|a| o.seq > a))
            .min_by_key(|o| o.seq)
            .copied();
        let Some(o) = next else {
            break;
        };
        let row = [
            i64::from(o.owner),
            side_code(o.side),
            i64::from(o.price),
            i64::from(o.qty),
            i64::from(o.seq),
        ];
        if let Some(mut leaf) = out.leaf_mut(LobPrivileged::ORDERS) {
            for (c, v) in row.into_iter().enumerate() {
                leaf.set(written * 5 + c, v);
            }
        }
        if let Some(mut leaf) = out.leaf_mut(LobPrivileged::ORDERS_MASK) {
            leaf.set(written, 1);
        }
        after = Some(o.seq);
    }
    let mut after: Option<u32> = None;
    for written in 0..PENDING_CAP {
        let next = state
            .pending
            .items()
            .filter(|p| after.is_none_or(|a| p.arrival > a))
            .min_by_key(|p| p.arrival)
            .copied();
        let Some(p) = next else {
            break;
        };
        let row = [
            i64::from(p.owner),
            kind_code(p.kind),
            side_code(p.side),
            i64::from(p.price),
            i64::from(p.qty),
            i64::from(p.target),
            i64::from(p.ready),
            i64::from(p.arrival),
        ];
        if let Some(mut leaf) = out.leaf_mut(LobPrivileged::PENDING) {
            for (c, v) in row.into_iter().enumerate() {
                leaf.set(written * 8 + c, v);
            }
        }
        if let Some(mut leaf) = out.leaf_mut(LobPrivileged::PENDING_MASK) {
            leaf.set(written, 1);
        }
        after = Some(p.arrival);
    }
}

/// The privileged state buffer of ADR-0079 item 7: [`LobPrivileged`].
impl PrivilegedState for LimitOrderBook {
    fn state_space(cfg: &LobConfig) -> Result<Space, SpaceError> {
        state_space(cfg)
    }

    fn write_state(cfg: &LobConfig, state: &LobState, out: &mut ObsWriter<'_, '_, ()>) {
        write_state(cfg, state, out);
    }
}
