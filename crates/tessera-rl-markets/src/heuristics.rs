// SPDX-License-Identifier: MIT OR Apache-2.0

//! The three scripted baselines (design section 5). Each reads only the
//! agent's information state ([`LobInfo`]) and every action it writes is
//! legal under the mask: a candidate is kept only if
//! [`crate::actions::decide`] accepts it, otherwise the no-op.
//!
//! - **Market maker:** cancels its oldest quote once its last limit order
//!   is more than [`QUOTE_AGE`] steps old (`quote_step`, its own
//!   knowledge; the design's per-quote age would need order timestamps);
//!   otherwise quotes one lot at the best bid
//!   minus one tick when it has no bid, else at the best ask plus one when
//!   it has no ask; both skewed one tick down per 10 lots of long
//!   inventory (up per 10 short).
//! - **Noise trader:** with probability 1/4, a random legal action (up to
//!   [`NOISE_TRIES`] uniform draws of the four heads); otherwise nothing.
//! - **Momentum:** buys (sells) size bucket 1 at market when the mid rose
//!   (fell) by at least [`MOMENTUM_TICKS`] over the last
//!   [`crate::state::MID_RING`] - 1 steps.
//!
//! The default mix is the design's: slot 0 is the learner's seat (it gets
//! the market maker when a heuristic plays it), then per seven slots one
//! market maker, three noise traders and three momentum traders
//! ([`baseline_of`]).
//!
//! ADR-0079's policy map and `InfoHeuristic` are not on main yet; the
//! [`Heuristic`] impl computes the information state itself and calls
//! [`act_info`], so it reads nothing hidden, and `InfoHeuristic` can call
//! [`act_info`] directly once it exists.

#![deny(clippy::float_arithmetic)]

use core::num::NonZeroU32;

use tessera_rl_core::io::ActionWriter;
use tessera_rl_core::{AgentSlot, Heuristic};
use tessera_rl_num::rng::StepRng;

use crate::actions::{Context, HEADS, OFFSET_ZERO, decide, kind};
use crate::config::LobConfig;
use crate::env::{LimitOrderBook, ring};
use crate::info::{LobInfo, info_of, public_state};
use crate::order::Side;
use crate::state::{LobState, MID_RING};

/// A quote older than this many steps is cancelled by the market maker.
pub const QUOTE_AGE: u32 = 20;
/// Uniform draws the noise trader tries before giving up.
pub const NOISE_TRIES: usize = 8;
/// The mid move that triggers the momentum trader, in ticks.
pub const MOMENTUM_TICKS: i64 = 2;

/// One of the scripted baselines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Baseline {
    /// Quotes around the touch.
    MarketMaker,
    /// Random legal actions, one step in four.
    Noise,
    /// Trades with the recent mid move.
    Momentum,
}

impl Baseline {
    /// Every baseline.
    pub const ALL: [Self; 3] = [Self::MarketMaker, Self::Noise, Self::Momentum];
}

/// The baseline that plays `slot` in the default mix.
#[must_use]
pub const fn baseline_of(slot: usize) -> Baseline {
    if slot == 0 {
        return Baseline::MarketMaker;
    }
    match (slot - 1) % 7 {
        0 => Baseline::MarketMaker,
        1..=3 => Baseline::Noise,
        _ => Baseline::Momentum,
    }
}

/// The heads `[kind, offset, size bucket, cancel index]`.
const fn heads(k: i32, offset: i32, size: i32, cancel: i32) -> [i32; HEADS] {
    [k, offset, size, cancel]
}

/// The first candidate the mask allows, or the no-op.
fn first_legal(cfg: &LobConfig, ctx: &Context, candidates: &[[i32; HEADS]]) -> [i32; HEADS] {
    candidates
        .iter()
        .copied()
        .find(|&h| decide(cfg, ctx, h).is_some())
        .unwrap_or([kind::NOOP, 0, 0, 0])
}

fn market_maker(cfg: &LobConfig, info: &LobInfo, ctx: &Context) -> [i32; HEADS] {
    let own = ctx.own_orders();
    // Quotes untouched for more than QUOTE_AGE steps: cancel the oldest
    // (index 0 of the cancel head).
    if !own.is_empty()
        && info
            .step
            .checked_sub(info.quote_step)
            .is_some_and(|age| age > QUOTE_AGE)
    {
        return first_legal(cfg, ctx, &[heads(kind::CANCEL, 0, 0, 0)]);
    }
    let inventory = i64::from(info.account.inventory);
    let skew = i32::try_from(inventory / 10).unwrap_or(0);
    let has = |side: Side| own.iter().any(|o| o.side == side);
    let clamp = |o: i32| o.clamp(0, OFFSET_ZERO * 2);
    let mut candidates = [[kind::NOOP, 0, 0, 0]; 2];
    let mut n = 0;
    let quotes = [
        (
            Side::Buy,
            heads(kind::LIMIT_BUY, clamp(OFFSET_ZERO - 1 - skew), 0, 0),
        ),
        (
            Side::Sell,
            heads(kind::LIMIT_SELL, clamp(OFFSET_ZERO + 1 - skew), 0, 0),
        ),
    ];
    for (side, h) in quotes {
        if !has(side)
            && let Some(c) = candidates.get_mut(n)
        {
            *c = h;
            n += 1;
        }
    }
    first_legal(cfg, ctx, candidates.get(..n).unwrap_or(&[]))
}

fn noise(cfg: &LobConfig, ctx: &Context, rng: &mut StepRng) -> [i32; HEADS] {
    let draw = |rng: &mut StepRng, n: u32| {
        NonZeroU32::new(n).map_or(0, |b| i32::try_from(rng.below_u32(b)).unwrap_or(0))
    };
    if draw(rng, 4) != 0 {
        return [kind::NOOP, 0, 0, 0];
    }
    for _ in 0..NOISE_TRIES {
        let h = heads(draw(rng, 6), draw(rng, 9), draw(rng, 4), draw(rng, 8));
        if decide(cfg, ctx, h).is_some() {
            return h;
        }
    }
    [kind::NOOP, 0, 0, 0]
}

fn momentum(cfg: &LobConfig, info: &LobInfo, ctx: &Context) -> [i32; HEADS] {
    let ring_len = u32::try_from(MID_RING).unwrap_or(1);
    if info.step < ring_len {
        return [kind::NOOP, 0, 0, 0];
    }
    let now = info
        .mids
        .get(ring(info.step, MID_RING))
        .copied()
        .unwrap_or(0);
    // The oldest entry of the ring: the mid `MID_RING - 1` steps ago.
    let then = info
        .mids
        .get(ring(info.step + 1, MID_RING))
        .copied()
        .unwrap_or(0);
    let moved = i64::from(now) - i64::from(then);
    if moved >= MOMENTUM_TICKS {
        first_legal(cfg, ctx, &[heads(kind::MARKET_BUY, 0, 1, 0)])
    } else if moved <= -MOMENTUM_TICKS {
        first_legal(cfg, ctx, &[heads(kind::MARKET_SELL, 0, 1, 0)])
    } else {
        [kind::NOOP, 0, 0, 0]
    }
}

/// The action of `baseline` for the agent of `info`, written to `out`.
/// Reads only `info`; draws from `rng` only for the noise trader.
pub fn act_info(
    cfg: &LobConfig,
    info: &LobInfo,
    baseline: Baseline,
    rng: &mut StepRng,
    out: &mut ActionWriter<'_>,
) {
    let ctx = Context::new(cfg, &public_state(info), usize::from(info.me));
    let h = match baseline {
        Baseline::MarketMaker => market_maker(cfg, info, &ctx),
        Baseline::Noise => noise(cfg, &ctx, rng),
        Baseline::Momentum => momentum(cfg, info, &ctx),
    };
    for (head, v) in h.into_iter().enumerate() {
        out.set_discrete(head, v);
    }
}

/// The default mix ([`baseline_of`]), through the agent's information
/// state only.
impl Heuristic for LimitOrderBook {
    fn act(
        cfg: &LobConfig,
        state: &LobState,
        agent: AgentSlot,
        rng: &mut StepRng,
        out: &mut ActionWriter<'_>,
    ) {
        let info = info_of(cfg, state, agent.index());
        act_info(cfg, &info, baseline_of(agent.index()), rng, out);
    }
}
