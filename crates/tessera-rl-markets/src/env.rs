// SPDX-License-Identifier: MIT OR Apache-2.0

//! `tessera/LimitOrderBook-v1`: the `Environment` (`docs/reviews/381/DESIGN.md`,
//! sections 1 to 4).
//!
//! Rules per step, in this order:
//!
//! 1. Every alive agent's action is decoded against the state at the start
//!    of the step (the state its mask described); an illegal action is a
//!    no-op ([`crate::actions`]). A slot whose decision is a limit order
//!    records the step in `quote_step`.
//! 2. The fundamental takes one step of its random walk (`StepRng`), and
//!    every used slot draws its private signal of it (`StepRng`).
//! 3. Queued items whose delay has elapsed are applied, in arrival order.
//! 4. The alive slots are put in a Fisher-Yates permutation drawn from
//!    `StepRng`, never slot order; in that order each agent's order or
//!    cancel is applied now (`latency = 0`) or queued for `latency` steps.
//! 5. With bankruptcy enabled, an agent whose equity marked at the mid is
//!    below `-bankrupt` loses its orders and queued items and dies.
//! 6. On the last step (`steps`), the market closes: every resting order
//!    is cancelled and every queued item dropped, then every inventory of
//!    an agent alive at the start of the step is settled at the
//!    fundamental and the episode is truncated.
//! 7. Each agent alive at the start of the step is rewarded its equity
//!    change marked at the mid ([`LobState::mark`]), scaled by
//!    `2^-reward_shift`. Settlement comes first, so the last step's reward
//!    holds the difference between the mid and the fundamental.
//! 8. The mid at the end of the step is kept as `last_mid`, the mid of a
//!    one-sided or empty book in the next step.
//!
//! Neither rewards nor bankruptcy read the fundamental before settlement:
//! a reward minus the cash change would otherwise reveal it exactly.
//! Fewer than two alive agents terminate the episode.

#![deny(clippy::float_arithmetic)]

use core::num::NonZeroU32;

use tessera_rl_core::io::{ActionWriter, Actions, MaskWriter, ObsWriter};
use tessera_rl_core::space::Space;
use tessera_rl_core::{
    ActionMask, Adapters, AgentSlot, Bool8, Environment, EpisodeEnd, HeaderView,
    InvariantViolation, NoAdaptive, Observation, Phase, RecordHeader, SpaceError, StepOut, Tier,
    Validate,
};
use tessera_rl_num::rng::{ObsRng, StepRng};

use crate::account::{Account, apply_fill};
use crate::actions::{self, CANCELS, Context, Decision, HEADS, KINDS, OFFSETS, SIZES};
use crate::config::LobConfig;
use crate::fill::Fills;
use crate::fundamental::RandomWalk;
use crate::limits::{MAX_AGENTS, MAX_LOT, MAX_OPEN};
use crate::observe::{self, LobObs};
use crate::order::Kind;
use crate::pending::Pending;
use crate::reward::scaled;
use crate::state::{LobState, MAX_TRADES, MID_RING, Trade};

/// Versioned environment id (ADR-0023).
pub const ENV_ID: &str = "tessera/LimitOrderBook-v1";
/// Environment version, bumped with any change to its outputs.
pub const ENV_VERSION: u32 = 1;

// Registration under the versioned id (SPEC 10, ADR-0054).
tessera_rl_core::register!(LimitOrderBook, id = "tessera/LimitOrderBook-v1");

/// The limit order book environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LimitOrderBook;

fn agent_slot(index: usize) -> Option<AgentSlot> {
    AgentSlot::new(u16::try_from(index).ok()?, MAX_AGENTS)
}

/// The raw heads of `agent` in this step's actions; missing heads are 0.
fn heads_of(actions: &Actions<'_, ()>, agent: AgentSlot) -> [i32; HEADS] {
    let mut heads = [0; HEADS];
    for (h, v) in heads.iter_mut().enumerate() {
        *v = actions.discrete(agent, h).unwrap_or(0);
    }
    heads
}

/// Applies one order or cancel of `owner` to the book and the accounts,
/// and records its fills.
fn apply(state: &mut LobState, owner: u8, item: &Pending, fills: &mut Fills) {
    let result = match item.kind {
        Kind::Limit => state
            .book
            .limit(owner, item.side, item.price, item.qty, fills),
        Kind::Market => state
            .book
            .ioc(owner, item.side, item.price, item.qty, fills),
        Kind::Cancel => {
            let _ = state.book.cancel(owner, item.target);
            return;
        }
    };
    // Masks and `decide` exclude every refusal (owner, price, size, open
    // orders); a refusal here is a broken invariant and changes nothing.
    debug_assert!(result.is_ok(), "the book refused a legal order: {result:?}");
    if result.is_err() {
        return;
    }
    for fill in fills.as_slice() {
        let applied = apply_fill(&mut state.accounts, fill);
        debug_assert!(applied, "a fill between invalid owners");
        state.last_trade = fill.price;
        let n = usize::try_from(state.trade_count).unwrap_or(MAX_TRADES);
        if let Some(t) = state.trades.get_mut(n) {
            *t = Trade {
                price: fill.price,
                qty: fill.qty,
                side: fill.taker_side,
                ..Trade::default()
            };
            state.trade_count += 1;
        }
    }
}

/// The queue item of a decision; `None` for a no-op.
fn item_of(owner: u8, decision: Decision) -> Option<Pending> {
    match decision {
        Decision::Noop => None,
        Decision::Limit { side, price, qty } => Some(Pending::limit(owner, side, price, qty)),
        Decision::Market { side, cap, qty } => Some(Pending {
            price: cap,
            ..Pending::market(owner, side, qty)
        }),
        Decision::Cancel { seq } => Some(Pending::cancel(owner, seq)),
    }
}

/// The alive slots in a uniformly drawn order (Fisher-Yates over the
/// slots in index order, one bounded draw per position).
fn arrival_order(state: &LobState, n: usize, rng: &mut StepRng) -> ([u8; MAX_AGENTS], usize) {
    let mut order = [0u8; MAX_AGENTS];
    let mut count = 0;
    for slot in 0..n.min(MAX_AGENTS) {
        if state.is_alive(slot)
            && let (Some(dst), Ok(s)) = (order.get_mut(count), u8::try_from(slot))
        {
            *dst = s;
            count += 1;
        }
    }
    for i in (1..count).rev() {
        let bound = u32::try_from(i + 1).ok().and_then(NonZeroU32::new);
        let j = bound.map_or(0, |b| rng.below_u32(b));
        if let Ok(j) = usize::try_from(j) {
            order.swap(i, j.min(i));
        }
    }
    (order, count)
}

/// Step rule 5: with bankruptcy enabled, an agent whose equity marked at
/// the mid is below `-bankrupt` loses its orders and queued items and
/// dies.
fn bankruptcies(cfg: &LobConfig, state: &mut LobState, out: &mut StepOut<'_>) {
    if cfg.bankrupt == 0 {
        return;
    }
    let floor = i64::try_from(cfg.bankrupt).map_or(i64::MIN, |b| -b);
    // One mid for every slot: a bankrupt agent's cancels must not move the
    // mark of the slots after it.
    let mid = state.touch().mid;
    for slot in 0..cfg.agent_count() {
        if !state.is_alive(slot) || state.account(slot).equity(mid) >= floor {
            continue;
        }
        let owner = u8::try_from(slot).unwrap_or(u8::MAX);
        state.book.cancel_all(owner);
        state.pending.drop_owner(owner);
        if let (Some(alive), Some(agent)) = (state.alive.get_mut(slot), agent_slot(slot)) {
            *alive = Bool8::FALSE;
            out.agent_died(agent);
        }
    }
}

/// Step rule 6: the market closes on the last step: every resting order
/// is cancelled and every queued item dropped, then every inventory of an
/// agent alive at the start of the step is settled at the fundamental.
/// Closing first keeps the position limits: an agent short `k` lots may hold
/// up to `max_position + k` open buy lots, which would exceed the limit once
/// its inventory is settled to zero.
pub(crate) fn settle(state: &mut LobState, was_alive: &[bool; MAX_AGENTS]) {
    for owner in 0..MAX_AGENTS {
        let owner = u8::try_from(owner).unwrap_or(u8::MAX);
        state.book.cancel_all(owner);
        state.pending.drop_owner(owner);
    }
    let price = i64::from(state.fundamental);
    for (a, &alive) in state.accounts.iter_mut().zip(was_alive) {
        if alive {
            // |inventory| x max_tick fits i32 and cash stays below 2^62 by
            // config: plain arithmetic.
            a.cash += i64::from(a.inventory) * price;
            a.inventory = 0;
        }
    }
}

/// Step rule 7: each agent alive at the start of the step gets its equity
/// change marked at the mid, scaled by `2^-reward_shift`.
fn reward(
    cfg: &LobConfig,
    state: &mut LobState,
    was_alive: &[bool; MAX_AGENTS],
    out: &mut StepOut<'_>,
) {
    let mid = state.touch().mid;
    for (slot, &alive) in was_alive.iter().enumerate() {
        if !alive {
            continue;
        }
        // `state.mark(slot)`, with the touch read once.
        let equity = state.account(slot).equity(mid);
        if let (Some(prev), Some(agent)) = (state.last_equity.get_mut(slot), agent_slot(slot)) {
            out.set_reward(agent, scaled(equity - *prev, cfg.reward_shift));
            *prev = equity;
        }
    }
}

/// The index of `step` in a ring of `len` entries.
pub(crate) fn ring(step: u32, len: usize) -> usize {
    usize::try_from(step).map_or(0, |s| s % len.max(1))
}

/// Draws the private signal of every used slot: the fundamental plus a
/// uniform noise in `[-noise, +noise]`, one `StepRng` draw per slot in slot
/// order (dead slots too, so the draw count depends on the config only).
fn draw_signals(cfg: &LobConfig, state: &mut LobState, rng: &mut StepRng) {
    let width = cfg
        .noise
        .checked_mul(2)
        .and_then(|w| w.checked_add(1))
        .and_then(NonZeroU32::new);
    let fundamental = i64::from(state.fundamental);
    for signal in state.signal.iter_mut().take(cfg.agent_count()) {
        let noise = width.map_or(0, |w| i64::from(rng.below_u32(w)) - i64::from(cfg.noise));
        // At most 2^16 + 64: fits i32.
        *signal = i32::try_from(fundamental + noise).unwrap_or(0);
    }
}

impl Environment for LimitOrderBook {
    /// Parallel only (SPEC 12, ADR-0080), as the RTS: cycle buffering of a
    /// multi-head environment for AEC is not designed yet (#381, PR 4).
    const ADAPTERS: Adapters = Adapters::PZ_PARALLEL;
    type State = LobState;
    type Config = LobConfig;
    type Obs = LobObs;
    type Action = ();
    type Header = RecordHeader<MAX_AGENTS, NoAdaptive>;

    const MAX_AGENTS: usize = MAX_AGENTS;
    const TIER: Tier = Tier::CrossPlatform;
    /// No agent sees others' orders, inventories or the fundamental.
    const PERFECT_INFORMATION: bool = false;

    fn observation_space(cfg: &LobConfig) -> Result<Space, SpaceError> {
        LobObs::space(cfg)
    }

    fn action_space(_: &LobConfig) -> Result<Space, SpaceError> {
        Ok(Space::MultiDiscrete {
            nvec: vec![KINDS, OFFSETS, SIZES, CANCELS],
        })
    }

    fn initial_agents(cfg: &LobConfig) -> usize {
        cfg.agent_count()
    }

    fn reset(cfg: &LobConfig, rng: &mut StepRng, state: &mut LobState) {
        let n = cfg.agent_count();
        *state = LobState::default();
        for slot in 0..n {
            if let (Some(a), Some(e), Some(alive)) = (
                state.accounts.get_mut(slot),
                state.last_equity.get_mut(slot),
                state.alive.get_mut(slot),
            ) {
                *a = Account::with_cash(cfg.initial_cash);
                *e = cfg.initial_cash;
                *alive = Bool8::TRUE;
            }
        }
        state.fundamental = cfg.initial_price;
        state.reset_price = cfg.initial_price;
        state.last_mid = cfg.initial_price;
        state.mids = [cfg.initial_price; MID_RING];
        draw_signals(cfg, state, rng);
    }

    fn step(
        cfg: &LobConfig,
        state: &mut LobState,
        actions: &Actions<'_, ()>,
        rng: &mut StepRng,
        out: &mut StepOut<'_>,
    ) {
        let n = cfg.agent_count();
        // 1. Decisions against the state the masks described.
        let mut decisions = [Decision::Noop; MAX_AGENTS];
        let mut was_alive = [false; MAX_AGENTS];
        for (slot, (decision, alive)) in decisions.iter_mut().zip(&mut was_alive).enumerate() {
            let Some(agent) = agent_slot(slot).filter(|_| slot < n && state.is_alive(slot)) else {
                continue;
            };
            *alive = true;
            *decision = crate::actions::decode(cfg, state, agent, heads_of(actions, agent));
        }
        // The step being played, 1-based; `steps <= 4096` by config.
        let now = state.step + 1;
        for (decision, quoted) in decisions.iter().zip(state.quote_step.iter_mut()) {
            if matches!(decision, Decision::Limit { .. }) {
                *quoted = now;
            }
        }
        // 2. The fundamental, then every used slot's private signal of it.
        state.fundamental = RandomWalk::new(cfg).step(state.fundamental, rng);
        draw_signals(cfg, state, rng);
        state.trades = [Trade::default(); MAX_TRADES];
        state.trade_count = 0;
        let mut fills = Fills::new();
        // 3. Released items, in arrival order (at most PENDING_CAP).
        while let Some(item) = state.pending.pop_ready(now) {
            apply(state, item.owner, &item, &mut fills);
        }
        // 4. This step's submissions in a drawn arrival order.
        let (order, count) = arrival_order(state, n, rng);
        for &owner in order.iter().take(count) {
            let Some(item) = decisions
                .get(usize::from(owner))
                .and_then(|&d| item_of(owner, d))
            else {
                continue;
            };
            if cfg.latency == 0 {
                apply(state, owner, &item, &mut fills);
            } else {
                // One item per agent and step, kept at most `latency`
                // steps: the queue cannot fill (PENDING_CAP = 16 x 4).
                let queued = state.pending.push(item, now + cfg.latency);
                debug_assert!(queued.is_ok(), "the latency queue overflowed");
            }
        }
        bankruptcies(cfg, state, out);
        let last = now >= cfg.steps;
        if last {
            settle(state, &was_alive);
        }
        reward(cfg, state, &was_alive, out);
        // 8. The mid the next step falls back to, also kept in the
        // momentum history.
        let mid = state.touch().mid;
        state.last_mid = mid;
        state.step = now;
        if let Some(m) = state.mids.get_mut(ring(now, MID_RING)) {
            *m = mid;
        }
        let alive = (0..n).filter(|&s| state.is_alive(s)).count();
        if last {
            out.end_episode(EpisodeEnd::Truncated);
        } else if alive < 2 {
            out.end_episode(EpisodeEnd::Terminated);
        }
    }

    fn observe(
        cfg: &LobConfig,
        state: &LobState,
        agent: AgentSlot,
        rng: &mut ObsRng,
        out: &mut ObsWriter<'_, '_, LobObs>,
    ) {
        observe::observe(cfg, state, agent, rng, out);
    }
}

/// Masks per [`crate::actions`]; the no-op is every head 0.
impl ActionMask for LimitOrderBook {
    fn action_mask(cfg: &LobConfig, state: &LobState, agent: AgentSlot, out: &mut MaskWriter<'_>) {
        let ctx = Context::new(cfg, state, agent.index());
        actions::write_mask(cfg, &ctx, out);
    }

    fn noop(_: &LobConfig, _: &LobState, _: AgentSlot, out: &mut ActionWriter<'_>) {
        for head in 0..HEADS {
            out.set_discrete(head, 0);
        }
    }

    fn head_parents(_: &LobConfig, head: usize) -> [Option<u16>; 2] {
        actions::head_parents(head)
    }
}

fn violation(code: &'static str) -> Result<(), InvariantViolation> {
    Err(InvariantViolation { code })
}

/// Imported states: the book, the queue and the accounts as the rules
/// leave them.
impl Validate for LimitOrderBook {
    fn validate(cfg: &LobConfig, s: &LobState) -> Result<(), InvariantViolation> {
        let n = cfg.agent_count();
        if s.step > cfg.steps {
            return violation("more steps than the episode length");
        }
        // The all-zero state before the first reset has no prices yet.
        let reset = s.reset_price != 0;
        if reset && s.reset_price != cfg.initial_price {
            return violation("a reset price that is not the config's");
        }
        if reset && (s.fundamental < cfg.min_tick || s.fundamental > cfg.max_tick) {
            return violation("a fundamental outside [min_tick, max_tick]");
        }
        // Limit prices span `[1, max_tick]` (not `min_tick`, which bounds the
        // fundamental only), and so does any mid of the book.
        if reset && (s.last_mid == 0 || s.last_mid > cfg.max_tick) {
            return violation("a last mid outside [1, max_tick]");
        }
        let noise = i64::from(cfg.noise);
        for (slot, &signal) in s.signal.iter().enumerate() {
            let used = reset && slot < n;
            let off = i64::from(signal) - i64::from(s.fundamental);
            if (used && off.abs() > noise) || (!used && signal != 0) {
                return violation("a signal outside the fundamental's noise band");
            }
        }
        for (slot, &quoted) in s.quote_step.iter().enumerate() {
            if quoted > s.step || (slot >= n && quoted != 0) {
                return violation("a quote step after the current step, or for an unused slot");
            }
        }
        if s.last_trade > cfg.max_tick {
            return violation("a last trade above max_tick");
        }
        let trades = usize::try_from(s.trade_count).unwrap_or(usize::MAX);
        if trades > MAX_TRADES {
            return violation("more trades than MAX_TRADES");
        }
        if s.trades.iter().skip(trades).any(|t| *t != Trade::default()) {
            return violation("a trade entry past trade_count");
        }
        for o in s.book.orders() {
            if usize::from(o.owner) >= n
                || o.price == 0
                || o.price > cfg.max_tick
                || o.qty == 0
                || o.qty > MAX_LOT
                || o.seq >= s.book.next_seq()
            {
                return violation("a resting order out of range");
            }
        }
        if s.book.is_crossed() {
            return violation("a crossed book");
        }
        for p in s.pending.items() {
            if usize::from(p.owner) >= n || p.ready <= s.step || p.ready > s.step + cfg.latency {
                return violation("a queued item out of range");
            }
        }
        for slot in 0..MAX_AGENTS {
            let owner = u8::try_from(slot).unwrap_or(u8::MAX);
            let a = s.account(slot);
            let alive = s.is_alive(slot);
            if slot >= n {
                if alive || a != Account::default() || s.last_equity.get(slot) != Some(&0) {
                    return violation("an unused slot with an account");
                }
                continue;
            }
            if s.book.open_count(owner) > MAX_OPEN {
                return violation("more than MAX_OPEN resting orders");
            }
            if !alive && (s.book.open_count(owner) > 0 || s.pending.count(owner) > 0) {
                return violation("a dead agent with orders");
            }
            if alive && a.inventory.unsigned_abs() > cfg.max_position {
                return violation("an inventory beyond max_position");
            }
        }
        Ok(())
    }

    /// In a record that has been reset, a used slot is alive exactly when
    /// the state says so, and an unused slot is neither alive nor acting.
    fn validate_record(
        cfg: &LobConfig,
        header: &HeaderView<'_>,
        state: &LobState,
    ) -> Result<(), InvariantViolation> {
        Self::validate(cfg, state)?;
        if header.phase() == Phase::NeverReset {
            return Ok(());
        }
        for slot in 0..header.agents() {
            let Some(view) = header.agent(slot) else {
                continue;
            };
            if slot < cfg.agent_count() {
                if view.alive != state.is_alive(slot) {
                    return violation("a used slot alive differently from its state");
                }
            } else if view.alive || view.acting {
                return violation("an unused slot that is alive or acting");
            }
        }
        Ok(())
    }
}

impl tessera_rl_core::Persist for LimitOrderBook {
    const ENV_ID: &'static str = ENV_ID;
    const ENV_VERSION: u32 = ENV_VERSION;
}

#[cfg(test)]
mod tests {
    use super::settle;
    use crate::account::Account;
    use crate::fill::Fills;
    use crate::limits::MAX_AGENTS;
    use crate::order::Side;
    use crate::pending::Pending;
    use crate::state::LobState;

    #[test]
    fn test_settle_closes_the_market_before_settling_inventories() {
        let mut s = LobState {
            fundamental: 500,
            ..LobState::default()
        };
        s.accounts[0] = Account::with_cash(1_000);
        s.accounts[0].inventory = -5;
        s.accounts[1] = Account::with_cash(0);
        s.accounts[1].inventory = 5;
        let mut f = Fills::new();
        // Agent 0, short 5, rests 11 buy lots (legal: -5 + 11 <= 6) and queues one.
        assert!(s.book.limit(0, Side::Buy, 400, 11, &mut f).is_ok());
        assert_eq!(
            s.pending.push(Pending::limit(0, Side::Buy, 401, 1), 3),
            Ok(())
        );
        let mut alive = [false; MAX_AGENTS];
        alive[0] = true;
        alive[1] = true;
        settle(&mut s, &alive);
        assert!(s.book.is_empty());
        assert!(s.pending.is_empty());
        assert_eq!((s.accounts[0].cash, s.accounts[0].inventory), (-1_500, 0));
        assert_eq!((s.accounts[1].cash, s.accounts[1].inventory), (2_500, 0));
    }
}
