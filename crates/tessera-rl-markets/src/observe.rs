// SPDX-License-Identifier: MIT OR Apache-2.0

//! The observation of one agent (`docs/reviews/381/DESIGN.md`, section 3).
//!
//! Every leaf is `i32`, so the PufferLib flat view is zero-copy. Prices are
//! offsets in ticks from the mid ([`crate::state::Touch::mid`]). No agent sees another's
//! identity, orders, inventory or the true fundamental.
//!
//! `own_orders` and `trades` are fixed-size item leaves with a 0/1 `_mask`
//! leaf after each, the layout ADR-0082 gives a `Set`; absent rows are
//! zero. They are plain `Box` leaves here because `Set` leaves are not on
//! this crate's base yet; moving to `Set` changes the mask leaves' dtype
//! to `bool` and is a new environment version.

#![deny(clippy::float_arithmetic)]

use core::num::NonZeroU32;

use tessera_rl_core::io::ObsWriter;
use tessera_rl_core::{AgentSlot, ObsLeaf};
use tessera_rl_macros::Observation;
use tessera_rl_num::rng::ObsRng;

use crate::book::Level;
use crate::config::LobConfig;
use crate::limits::{MAX_AGENTS, MAX_OPEN};
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
    #[space(box(low = -OFFSET_BOUND, high = OFFSET_BOUND, dtype = i32, shape = |_: &LobConfig| [8, 3]))]
    pub own_orders: ObsLeaf<i32>,
    /// 1 for each row of `own_orders` in use, else 0.
    #[space(box(low = 0, high = 1, dtype = i32, shape = |_: &LobConfig| [8]))]
    pub own_orders_mask: ObsLeaf<i32>,
    /// This step's first 16 fills: `[price offset, lots, aggressor side]`.
    #[space(box(low = -OFFSET_BOUND, high = OFFSET_BOUND, dtype = i32, shape = |_: &LobConfig| [16, 3]))]
    pub trades: ObsLeaf<i32>,
    /// 1 for each row of `trades` in use, else 0.
    #[space(box(low = 0, high = 1, dtype = i32, shape = |_: &LobConfig| [16]))]
    pub trades_mask: ObsLeaf<i32>,
    /// The private signal: the fundamental's offset from the mid plus a
    /// uniform noise in `[-noise, +noise]` ticks, drawn per agent.
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

/// The noise of `agent`: the `agent`-th uniform draw in `[-noise, noise]`
/// of the observation stream, so every agent's signal differs.
fn noise(cfg: &LobConfig, agent: usize, rng: &mut ObsRng) -> i64 {
    let Some(width) = cfg
        .noise
        .checked_mul(2)
        .and_then(|w| w.checked_add(1))
        .and_then(NonZeroU32::new)
    else {
        return 0;
    };
    let mut draw = 0;
    for _ in 0..=agent.min(MAX_AGENTS) {
        draw = rng.below_u32(width);
    }
    i64::from(draw) - i64::from(cfg.noise)
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
    rng: &mut ObsRng,
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
    if let Some(mut leaf) = out.leaf_mut(LobObs::OWN_ORDERS) {
        for (i, o) in own.iter().take(n_own).enumerate() {
            leaf.set(3 * i, side_code(o.side));
            leaf.set(3 * i + 1, touch.offset(o.price));
            leaf.set(3 * i + 2, i32::try_from(o.qty).unwrap_or(0));
        }
    }
    if let Some(mut leaf) = out.leaf_mut(LobObs::OWN_ORDERS_MASK) {
        for i in 0..n_own {
            leaf.set(i, 1);
        }
    }
    let trades = usize::try_from(state.trade_count)
        .unwrap_or(0)
        .min(MAX_TRADES);
    if let Some(mut leaf) = out.leaf_mut(LobObs::TRADES) {
        for (i, t) in state.trades.iter().take(trades).enumerate() {
            leaf.set(3 * i, touch.offset(t.price));
            leaf.set(3 * i + 1, i32::try_from(t.qty).unwrap_or(0));
            leaf.set(3 * i + 2, side_code(t.side));
        }
    }
    if let Some(mut leaf) = out.leaf_mut(LobObs::TRADES_MASK) {
        for i in 0..trades {
            leaf.set(i, 1);
        }
    }
    let signal = i64::from(touch.offset(state.fundamental)) + noise(cfg, slot, rng);
    if let Some(mut leaf) = out.leaf_mut(LobObs::SIGNAL) {
        leaf.set(0, clip(signal));
    }
}
