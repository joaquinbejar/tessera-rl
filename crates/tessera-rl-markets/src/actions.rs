// SPDX-License-Identifier: MIT OR Apache-2.0

//! Action decoding and legality (`docs/reviews/381/DESIGN.md`, section 4,
//! with the corrections of review r1).
//!
//! The action is `MultiDiscrete([6, 9, 4, 8])`: `kind`, `offset`, `size`
//! and `cancel`. `offset` and `cancel` have `kind` as parent; `size` has
//! `kind` and `offset` as parents, so its mask row can price each size at
//! the chosen limit price. One function, [`decide`], is both the mask (it
//! is asked for every combination) and the step's own check: an action the
//! mask forbids decodes to [`Decision::Noop`] in the step too, so the
//! limits below hold even in a pool without masks.
//!
//! Legality is worst case over every open and pending order of the agent
//! (correction 1):
//!
//! - buys: `inventory + open and pending buy lots + size <= max_position`;
//!   sells: `inventory - open and pending sell lots - size >= -max_position`;
//! - without `allow_short_cash`, buys:
//!   `reserved buy cash + size x price <= cash`, where a limit order is
//!   priced at its limit and a market order at its protection price;
//! - limit orders: fewer than `MAX_OPEN` resting plus pending limit orders,
//!   and a price in `[1, max_tick]`;
//! - market orders: some other agent's order on the opposite side.
//!
//! A market order is immediate or cancel with a protection price
//! (correction 2): the worst price its size reaches over the `depth`
//! visible levels of the current book ([`Book::sweep`]), or the bound when
//! those levels cannot fill it (`max_tick` for a buy, 1 for a sell). It
//! never fills beyond that price, so reserving `size x protection` cash is
//! an upper bound on what it can pay, even when the book changes before it
//! arrives (latency, or orders ahead of it in the same step).
//!
//! Fills only move lots from open orders into inventory, and cancels and
//! self-trade prevention only remove open orders, so the two sums above
//! never grow between steps: `|inventory| <= max_position` and, without
//! `allow_short_cash`, `cash >= reserved buy cash >= 0` hold at every step
//! before the settlement.
//!
//! [`Book::sweep`]: crate::book::Book::sweep

#![deny(clippy::float_arithmetic)]

use tessera_rl_core::AgentSlot;
use tessera_rl_core::io::MaskWriter;

use crate::config::LobConfig;
use crate::exposure::Exposure;
use crate::limits::{MAX_OPEN, MAX_PRICE};
use crate::order::{Order, Side};
use crate::state::{LobState, Touch};

/// Choices of the `kind` head.
pub const KINDS: u32 = 6;
/// Choices of the `offset` head: `-4..=+4` ticks.
pub const OFFSETS: u32 = 9;
/// Choices of the `size` head: the four size buckets.
pub const SIZES: u32 = 4;
/// Choices of the `cancel` head: an index into the agent's open orders.
pub const CANCELS: u32 = 8;
/// The `offset` value of 0 ticks.
pub const OFFSET_ZERO: i32 = 4;

/// Discrete heads.
pub const HEADS: usize = 4;
/// Length of one agent's mask row: `6 + 6 x 9 + 6 x 9 x 4 + 6 x 8`.
pub const MASK_LEN: usize = 324;
/// Start of the `offset` rows (one per `kind`).
pub const OFFSET_ROWS: usize = 6;
/// Start of the `size` rows (one per `kind` and `offset`).
pub const SIZE_ROWS: usize = 60;
/// Start of the `cancel` rows (one per `kind`).
pub const CANCEL_ROWS: usize = 276;

const _: () = assert!(CANCELS as usize == MAX_OPEN);
const _: () = assert!(OFFSET_ROWS == KINDS as usize);
const _: () = assert!(SIZE_ROWS == OFFSET_ROWS + (KINDS * OFFSETS) as usize);
const _: () = assert!(CANCEL_ROWS == SIZE_ROWS + (KINDS * OFFSETS * SIZES) as usize);
const _: () = assert!(MASK_LEN == CANCEL_ROWS + (KINDS * CANCELS) as usize);

/// The `kind` head.
pub mod kind {
    /// Nothing.
    pub const NOOP: i32 = 0;
    /// Market buy.
    pub const MARKET_BUY: i32 = 1;
    /// Market sell.
    pub const MARKET_SELL: i32 = 2;
    /// Limit buy.
    pub const LIMIT_BUY: i32 = 3;
    /// Limit sell.
    pub const LIMIT_SELL: i32 = 4;
    /// Cancel one of the agent's open orders.
    pub const CANCEL: i32 = 5;
}

/// A decoded, legal action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Decision {
    /// Nothing.
    #[default]
    Noop,
    /// A limit order.
    Limit {
        /// Side.
        side: Side,
        /// Limit price, in ticks.
        price: u32,
        /// Lots.
        qty: u32,
    },
    /// A market order with its protection price.
    Market {
        /// Side.
        side: Side,
        /// The worst price it may fill at.
        cap: u32,
        /// Lots.
        qty: u32,
    },
    /// A cancel of the open order with id `seq`.
    Cancel {
        /// The order's id.
        seq: u32,
    },
}

/// What legality needs to know about one agent, read once per agent.
#[derive(Debug, Clone)]
pub struct Context {
    inventory: i64,
    cash: i64,
    exposure: Option<Exposure>,
    touch: Touch,
    /// Protection price per side and size bucket; `None` when no other
    /// agent's order is on the opposite side.
    caps: [[Option<u32>; SIZES as usize]; 2],
    own: [Order; MAX_OPEN],
    own_count: usize,
}

impl Context {
    /// The context of `slot` in `state`.
    #[must_use]
    pub fn new(cfg: &LobConfig, state: &LobState, slot: usize) -> Self {
        let owner = u8::try_from(slot).unwrap_or(u8::MAX);
        let account = state.account(slot);
        let exposure = state.book.exposure(owner).and_then(|book| {
            let pending = state.pending.exposure(owner, cfg.max_tick)?;
            book.checked_add(&pending)
        });
        let depth = usize::try_from(cfg.depth).unwrap_or(0);
        let mut caps = [[None; SIZES as usize]; 2];
        for side in Side::ALL {
            let bound = match side {
                Side::Buy => cfg.max_tick,
                Side::Sell => 1,
            };
            if let Some(row) = caps.get_mut(side.index()) {
                for (cap, qty) in row.iter_mut().zip(cfg.sizes()) {
                    let sweep = state.book.sweep(side, owner, qty, depth);
                    *cap = (sweep.lots > 0).then(|| sweep.worst_bounded(qty, bound));
                }
            }
        }
        let mut own = [Order::default(); MAX_OPEN];
        let own_count = state.book.own_orders(owner, &mut own);
        Self {
            inventory: i64::from(account.inventory),
            cash: account.cash,
            exposure,
            touch: state.touch(),
            caps,
            own,
            own_count,
        }
    }

    /// The agent's open orders, by id.
    #[must_use]
    pub fn own_orders(&self) -> &[Order] {
        self.own.get(..self.own_count).unwrap_or(&[])
    }

    /// Whether an order of `qty` lots on `side`, costing `price` per lot
    /// if it is a buy, keeps the position and cash limits in the worst
    /// case.
    fn affordable(&self, cfg: &LobConfig, side: Side, qty: u32, price: u32) -> bool {
        let Some(e) = self.exposure else {
            return false;
        };
        let qty = i64::from(qty);
        let limit = i64::from(cfg.max_position);
        match side {
            Side::Buy => {
                let lots = self.inventory + i64::from(e.lots_on(Side::Buy)) + qty;
                let cost = i64::try_from(e.buy_cash)
                    .ok()
                    .and_then(|c| c.checked_add(qty.checked_mul(i64::from(price))?));
                lots <= limit && (cfg.allow_short_cash || cost.is_some_and(|c| c <= self.cash))
            }
            Side::Sell => self.inventory - i64::from(e.lots_on(Side::Sell)) - qty >= -limit,
        }
    }

    /// Whether the agent may add a resting order.
    fn can_rest(&self) -> bool {
        self.exposure
            .is_some_and(|e| usize::try_from(e.total_orders()).is_ok_and(|n| n < MAX_OPEN))
    }
}

/// The side of a market or limit `kind`.
const fn side_of(kind: i32) -> Option<Side> {
    match kind {
        kind::MARKET_BUY | kind::LIMIT_BUY => Some(Side::Buy),
        kind::MARKET_SELL | kind::LIMIT_SELL => Some(Side::Sell),
        _ => None,
    }
}

/// The limit price of `offset` on `side`, if it is in `[1, max_tick]`.
fn limit_price(cfg: &LobConfig, touch: &Touch, side: Side, offset: i32) -> Option<u32> {
    let price = i64::from(touch.base(side)) + i64::from(offset) - i64::from(OFFSET_ZERO);
    let price = u32::try_from(price).ok()?;
    (price >= 1 && price <= cfg.max_tick && price <= MAX_PRICE).then_some(price)
}

/// The decision for heads `[kind, offset, size, cancel]`, or `None` if the
/// action is illegal. Heads a kind does not use are ignored (any value is
/// legal for them).
#[must_use]
pub fn decide(cfg: &LobConfig, ctx: &Context, heads: [i32; HEADS]) -> Option<Decision> {
    let [order_kind, offset, size_head, cancel] = heads;
    let qty = usize::try_from(size_head)
        .ok()
        .and_then(|s| cfg.sizes().get(s).copied());
    match order_kind {
        kind::NOOP => Some(Decision::Noop),
        kind::MARKET_BUY | kind::MARKET_SELL => {
            let side = side_of(order_kind)?;
            let qty = qty?;
            let s = usize::try_from(size_head).ok()?;
            let cap = (*ctx.caps.get(side.index())?.get(s)?)?;
            ctx.affordable(cfg, side, qty, cap)
                .then_some(Decision::Market { side, cap, qty })
        }
        kind::LIMIT_BUY | kind::LIMIT_SELL => {
            let side = side_of(order_kind)?;
            let qty = qty?;
            if !(0..i32::try_from(OFFSETS).ok()?).contains(&offset) {
                return None;
            }
            let price = limit_price(cfg, &ctx.touch, side, offset)?;
            (ctx.can_rest() && ctx.affordable(cfg, side, qty, price)).then_some(Decision::Limit {
                side,
                price,
                qty,
            })
        }
        kind::CANCEL => {
            let order = ctx.own_orders().get(usize::try_from(cancel).ok()?)?;
            Some(Decision::Cancel { seq: order.seq })
        }
        _ => None,
    }
}

/// Writes the mask row of one agent ([`MASK_LEN`] entries, the layout of
/// SPEC 11.3 for the parents above). Rows of heads a kind does not use are
/// all legal.
pub fn write_mask(cfg: &LobConfig, ctx: &Context, out: &mut MaskWriter<'_>) {
    let kinds = i32::try_from(KINDS).unwrap_or(0);
    let offsets = i32::try_from(OFFSETS).unwrap_or(0);
    let sizes = i32::try_from(SIZES).unwrap_or(0);
    let cancels = i32::try_from(CANCELS).unwrap_or(0);
    let idx = |v: i32| usize::try_from(v).unwrap_or(usize::MAX);
    for k in 0..kinds {
        let mut kind_legal = false;
        for o in 0..offsets {
            let mut offset_legal = false;
            for s in 0..sizes {
                let legal = match k {
                    kind::CANCEL => true,
                    _ => decide(cfg, ctx, [k, o, s, 0]).is_some(),
                };
                let row = SIZE_ROWS + idx(k) * idx(offsets * sizes) + idx(o) * idx(sizes);
                out.set(row + idx(s), legal);
                offset_legal |= legal;
            }
            out.set(OFFSET_ROWS + idx(k) * idx(offsets) + idx(o), offset_legal);
            kind_legal |= offset_legal;
        }
        for c in 0..cancels {
            let legal = match k {
                kind::CANCEL => decide(cfg, ctx, [k, 0, 0, c]).is_some(),
                _ => true,
            };
            out.set(CANCEL_ROWS + idx(k) * idx(cancels) + idx(c), legal);
        }
        if k == kind::CANCEL {
            kind_legal = ctx.own_count > 0;
        }
        out.set(idx(k), kind_legal);
    }
}

/// The parents of discrete head `head` (SPEC 11.3).
#[must_use]
pub const fn head_parents(head: usize) -> [Option<u16>; 2] {
    match head {
        1 | 3 => [Some(0), None],
        2 => [Some(0), Some(1)],
        _ => [None, None],
    }
}

/// The decision of `agent` in `state` for the raw heads of the step,
/// [`Decision::Noop`] when illegal.
#[must_use]
pub fn decode(
    cfg: &LobConfig,
    state: &LobState,
    agent: AgentSlot,
    heads: [i32; HEADS],
) -> Decision {
    let ctx = Context::new(cfg, state, agent.index());
    decide(cfg, &ctx, heads).unwrap_or(Decision::Noop)
}

#[cfg(test)]
mod tests {
    use super::{Context, Decision, decide, kind};
    use crate::account::Account;
    use crate::config::LobConfig;
    use crate::fill::Fills;
    use crate::order::Side;
    use crate::pending::Pending;
    use crate::state::LobState;

    fn setup(cash: i64, inventory: i32) -> (LobConfig, LobState) {
        let cfg = LobConfig {
            agents: 2,
            max_position: 5,
            initial_cash: cash,
            ..LobConfig::default()
        };
        let mut s = LobState {
            fundamental: 512,
            reset_price: 512,
            ..LobState::default()
        };
        s.accounts = [Account::with_cash(cash); 16];
        if let Some(a) = s.accounts.get_mut(0) {
            a.inventory = inventory;
        }
        (cfg, s)
    }

    #[test]
    fn test_decide_limit_price_is_offset_from_reference() {
        let (cfg, s) = setup(100_000, 0);
        let ctx = Context::new(&cfg, &s, 0);
        // No bids: the mid (512) is the reference; offset 8 is +4 ticks.
        let d = decide(&cfg, &ctx, [kind::LIMIT_BUY, 8, 0, 0]);
        assert_eq!(
            d,
            Some(Decision::Limit {
                side: Side::Buy,
                price: 516,
                qty: 1
            })
        );
        assert_eq!(decide(&cfg, &ctx, [kind::LIMIT_BUY, 9, 0, 0]), None);
        assert_eq!(decide(&cfg, &ctx, [kind::CANCEL, 0, 0, 0]), None);
        assert_eq!(decide(&cfg, &ctx, [kind::MARKET_BUY, 0, 0, 0]), None);
    }

    #[test]
    fn test_decide_position_counts_open_and_pending_lots() {
        let (cfg, mut s) = setup(100_000, 2);
        let mut f = Fills::new();
        // 1 resting and 1 pending buy lot: 2 + 2 + size must stay <= 5.
        assert!(s.book.limit(0, Side::Buy, 500, 1, &mut f).is_ok());
        assert_eq!(
            s.pending.push(Pending::limit(0, Side::Buy, 500, 1), 3),
            Ok(())
        );
        let ctx = Context::new(&cfg, &s, 0);
        // Size bucket 0 is 1 lot: 5 <= 5; bucket 1 is 2 lots: 6 > 5.
        assert!(decide(&cfg, &ctx, [kind::LIMIT_BUY, 4, 0, 0]).is_some());
        assert_eq!(decide(&cfg, &ctx, [kind::LIMIT_BUY, 4, 1, 0]), None);
        // Sells are bounded by -5 only.
        assert!(decide(&cfg, &ctx, [kind::LIMIT_SELL, 4, 3, 0]).is_none());
        assert!(decide(&cfg, &ctx, [kind::LIMIT_SELL, 4, 2, 0]).is_some());
    }

    #[test]
    fn test_decide_cash_reserves_open_buys_and_market_protection() {
        let (cfg, mut s) = setup(1_000, 0);
        let mut f = Fills::new();
        // 500 reserved by a resting bid; an ask from slot 1 at 520.
        assert!(s.book.limit(0, Side::Buy, 500, 1, &mut f).is_ok());
        assert!(s.book.limit(1, Side::Sell, 520, 3, &mut f).is_ok());
        let ctx = Context::new(&cfg, &s, 0);
        // A market buy of 1 lot is protected at 520: 500 + 520 > 1000.
        assert_eq!(decide(&cfg, &ctx, [kind::MARKET_BUY, 0, 0, 0]), None);
        let (cfg, mut s) = setup(1_100, 0);
        assert!(s.book.limit(0, Side::Buy, 500, 1, &mut f).is_ok());
        assert!(s.book.limit(1, Side::Sell, 520, 3, &mut f).is_ok());
        let ctx = Context::new(&cfg, &s, 0);
        assert_eq!(
            decide(&cfg, &ctx, [kind::MARKET_BUY, 0, 0, 0]),
            Some(Decision::Market {
                side: Side::Buy,
                cap: 520,
                qty: 1
            })
        );
        // Two lots: the visible book holds 3 at 520, so the cap stays 520,
        // and 500 + 1040 > 1100.
        assert_eq!(decide(&cfg, &ctx, [kind::MARKET_BUY, 0, 1, 0]), None);
    }

    #[test]
    fn test_decide_limit_orders_stop_at_max_open_resting_plus_pending() {
        let (cfg, mut s) = setup(1_000_000, 0);
        let mut f = Fills::new();
        // 4 resting bids and 3 resting asks: 7 orders, within the position limit.
        for price in 400..404 {
            assert!(s.book.limit(0, Side::Buy, price, 1, &mut f).is_ok());
        }
        for price in 600..603 {
            assert!(s.book.limit(0, Side::Sell, price, 1, &mut f).is_ok());
        }
        let ctx = Context::new(&cfg, &s, 0);
        assert!(decide(&cfg, &ctx, [kind::LIMIT_SELL, 4, 0, 0]).is_some());
        // One pending limit order makes 8 (`MAX_OPEN`): no further limit order.
        assert_eq!(
            s.pending.push(Pending::limit(0, Side::Sell, 610, 1), 3),
            Ok(())
        );
        let ctx = Context::new(&cfg, &s, 0);
        assert_eq!(decide(&cfg, &ctx, [kind::LIMIT_SELL, 4, 0, 0]), None);
        assert_eq!(decide(&cfg, &ctx, [kind::LIMIT_BUY, 4, 0, 0]), None);
    }
}
