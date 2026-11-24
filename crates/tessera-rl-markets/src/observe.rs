// SPDX-License-Identifier: MIT OR Apache-2.0

//! The observation of one agent (`docs/reviews/381/DESIGN.md`, section 3).
//!
//! Every item and box leaf is `i32`; the two `Set` masks are `bool`
//! (ADR-0082). Prices are offsets in ticks from the mid
//! ([`crate::state::Touch::mid`]). No agent sees another's identity,
//! orders, inventory or the true fundamental.
//!
//! `own_orders` (at most 8) and `trades` (at most 16) are `Set` leaves
//! (ADR-0082, #450): each exports its item leaf `[MAX, 3]` and then a
//! `bool` mask leaf named `<set>_mask`. Present rows are packed from 0;
//! absent rows are zero (the engine zeroes them). The switch from the
//! earlier `Box` plus `i32` mask leaves kept the environment version
//! (coordinator decision).

#![deny(clippy::float_arithmetic)]

use tessera_rl_core::io::ObsWriter;
use tessera_rl_core::{AgentSlot, ObsLeaf, ObsSet};
use tessera_rl_macros::Observation;
use tessera_rl_num::rng::ObsRng;

use crate::book::Level;
use crate::config::LobConfig;
use crate::limits::MAX_OPEN;
use crate::order::{Order, Side};
use crate::state::{LobState, MAX_TRADES};

/// The largest observed price offset, in ticks.
pub const OFFSET_BOUND: i32 = 65_536;
/// The largest lots at one observed level: `BOOK_CAP x MAX_LOT`.
pub const LOTS_BOUND: i32 = 131_072;
/// Entries of the `own` leaf.
pub const OWN: u32 = 6;
/// The deepest book levels observed per side.
const MAX_DEPTH: usize = 16;

/// One agent's observation.
#[derive(Observation)]
pub struct LobObs {
    /// The best `depth` bid levels: `[price offset, lots]`; empty levels
    /// are zero.
    #[space(box(low = -LOTS_BOUND, high = LOTS_BOUND, dtype = i32, shape = |c: &LobConfig| [c.depth, 2]))]
    pub bids: ObsLeaf<i32>,
    /// The best `depth` ask levels, as `bids`.
    #[space(box(low = -LOTS_BOUND, high = LOTS_BOUND, dtype = i32, shape = |c: &LobConfig| [c.depth, 2]))]
    pub asks: ObsLeaf<i32>,
    /// Inventory; cash change since reset; equity change at the mid since
    /// reset (both clipped to `i32`, the observation only); open orders;
    /// pending items; steps remaining.
    #[space(box(low = i32::MIN, high = i32::MAX, dtype = i32, shape = |_: &LobConfig| [OWN]))]
    pub own: ObsLeaf<i32>,
    /// The agent's open orders by id: `[side (0 buy, 1 sell), price
    /// offset, remaining lots]`.
    #[space(set(max = MAX_OPEN, inner(box(low = -OFFSET_BOUND, high = OFFSET_BOUND, shape = [3]))))]
    pub own_orders: ObsSet<i32>,
    /// This step's first 16 fills: `[price offset, lots, aggressor side]`.
    #[space(set(max = MAX_TRADES, inner(box(low = -OFFSET_BOUND, high = OFFSET_BOUND, shape = [3]))))]
    pub trades: ObsSet<i32>,
    /// The private signal's offset from the mid: the fundamental plus a
    /// uniform noise in `[-noise, +noise]` ticks, drawn per agent each step
    /// from `StepRng` and kept in the state.
    #[space(box(low = -OFFSET_BOUND, high = OFFSET_BOUND, dtype = i32, shape = |_: &LobConfig| [1]))]
    pub signal: ObsLeaf<i32>,
}

const _: () = assert!(MAX_OPEN == 8 && MAX_TRADES == 16);

/// `value` clipped to `i32` (a documented observation clip; the state
/// keeps the full value).
fn clip(value: i64) -> i32 {
    i32::try_from(value).unwrap_or(if value < 0 { i32::MIN } else { i32::MAX })
}

fn side_code(side: Side) -> i32 {
    match side {
        Side::Buy => 0,
        Side::Sell => 1,
    }
}

fn write_levels(
    out: &mut ObsWriter<'_, '_, LobObs>,
    cfg: &LobConfig,
    state: &LobState,
    side: Side,
) {
    let touch = state.touch();
    let mut levels = [Level::default(); MAX_DEPTH];
    let depth = usize::try_from(cfg.depth).unwrap_or(0).min(MAX_DEPTH);
    let n = state
        .book
        .levels(side, levels.get_mut(..depth).unwrap_or(&mut []));
    let id = match side {
        Side::Buy => LobObs::BIDS,
        Side::Sell => LobObs::ASKS,
    };
    if let Some(mut leaf) = out.leaf_mut(id) {
        for (i, level) in levels.iter().take(n).enumerate() {
            leaf.set(2 * i, touch.offset(level.price));
            leaf.set(2 * i + 1, i32::try_from(level.lots).unwrap_or(LOTS_BOUND));
        }
    }
}

/// Writes the observation of `agent`. Leaves are zero on entry (ADR-0058).
pub fn observe(
    cfg: &LobConfig,
    state: &LobState,
    agent: AgentSlot,
    _: &mut ObsRng,
    out: &mut ObsWriter<'_, '_, LobObs>,
) {
    let slot = agent.index();
    let owner = u8::try_from(slot).unwrap_or(u8::MAX);
    let touch = state.touch();
    write_levels(out, cfg, state, Side::Buy);
    write_levels(out, cfg, state, Side::Sell);

    let account = state.account(slot);
    let mut own = [Order::default(); MAX_OPEN];
    let n_own = state.book.own_orders(owner, &mut own);
    let initial = cfg.initial_cash;
    let equity = account.equity(touch.mid);
    let values = [
        i64::from(account.inventory),
        account.cash - initial,
        equity - initial,
        i64::try_from(n_own).unwrap_or(0),
        i64::try_from(state.pending.count(owner)).unwrap_or(0),
        i64::from(cfg.steps) - i64::from(state.step),
    ];
    if let Some(mut leaf) = out.leaf_mut(LobObs::OWN) {
        for (i, v) in values.into_iter().enumerate() {
            leaf.set(i, clip(v));
        }
    }
    if let Some(mut set) = out.set_mut(LobObs::OWN_ORDERS) {
        for (i, o) in own.iter().take(n_own).enumerate() {
            set.set(i, true);
            if let Some(mut row) = set.item(i) {
                row.set(0, side_code(o.side));
                row.set(1, touch.offset(o.price));
                row.set(2, i32::try_from(o.qty).unwrap_or(0));
            }
        }
    }
    let trades = usize::try_from(state.trade_count)
        .unwrap_or(0)
        .min(MAX_TRADES);
    if let Some(mut set) = out.set_mut(LobObs::TRADES) {
        for (i, t) in state.trades.iter().take(trades).enumerate() {
            set.set(i, true);
            if let Some(mut row) = set.item(i) {
                row.set(0, touch.offset(t.price));
                row.set(1, i32::try_from(t.qty).unwrap_or(0));
                row.set(2, side_code(t.side));
            }
        }
    }
    // The slot's own signal, drawn in the step (`LobState::signal`).
    let own_signal = state.signal.get(slot).copied().unwrap_or(0);
    let signal = i64::from(own_signal) - i64::from(touch.mid);
    if let Some(mut leaf) = out.leaf_mut(LobObs::SIGNAL) {
        leaf.set(0, clip(signal));
    }
}
