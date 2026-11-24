// SPDX-License-Identifier: MIT OR Apache-2.0

//! The fixed bounds of `tessera/LimitOrderBook-v1` (`docs/reviews/381/DESIGN.md`,
//! sections 1 and 2). Every capacity below is derived from them, and the
//! derivations are checked at compile time.

/// Agent slots.
pub const MAX_AGENTS: usize = 16;

/// Open (resting) orders one agent may hold at once.
pub const MAX_OPEN: usize = 8;

/// Resting orders in the book: `MAX_AGENTS x MAX_OPEN`, so the book can
/// never overflow while every agent respects [`MAX_OPEN`].
pub const BOOK_CAP: usize = MAX_AGENTS * MAX_OPEN;

/// The largest latency, in steps.
pub const MAX_LATENCY: u32 = 3;

/// Items in the latency queue: `MAX_AGENTS x (MAX_LATENCY + 1)`. One
/// action per agent and step, released `latency` steps later, keeps at
/// most `MAX_LATENCY` items per agent between steps; the extra one is the
/// margin of the step that submits before it releases.
pub const PENDING_CAP: usize = MAX_AGENTS * 4;

/// The largest price, in ticks (`2^16`). Prices are in `[1, MAX_PRICE]`.
pub const MAX_PRICE: u32 = 1 << 16;

/// The largest quantity of one order, in lots (`2^10`).
pub const MAX_LOT: u32 = 1 << 10;

/// The longest episode, in steps.
pub const MAX_STEPS: u32 = 4096;

/// Fills one incoming order can produce: each fill but the last consumes a
/// resting order, and the last touches one more, so at most one per
/// resting order.
pub const FILLS_CAP: usize = BOOK_CAP;

const _: () = assert!(BOOK_CAP == 128);
const _: () = assert!(PENDING_CAP == 64 && MAX_LATENCY + 1 == 4);
// `Pool<T, CAP>` needs `CAP` to be a multiple of `align_of::<T>()` (4 for
// the order and the pending item).
const _: () = assert!(BOOK_CAP.is_multiple_of(4) && PENDING_CAP.is_multiple_of(4));
// Agent indices fit the `u8` owner field.
const _: () = assert!(MAX_AGENTS <= 255);
