// SPDX-License-Identifier: MIT OR Apache-2.0

//! The rules of `tessera/LimitOrderBook-v1` through the engine: mask
//! layout, a crossing trade, latency, settlement with truncation, and
//! bankruptcy.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures: a failure is the assertion"
)]

mod support;

use support::{Pool, config, input, pool};
use tessera_rl_core::Bool8;
use tessera_rl_engine::Reseed;
use tessera_rl_markets::actions::{CANCEL_ROWS, HEADS, KINDS, MASK_LEN, kind};
use tessera_rl_markets::limits::MAX_AGENTS;
use tessera_rl_markets::{LobConfig, Side};

/// One instance's actions: every agent no-op except `acts`.
fn acts(list: &[(usize, [i32; HEADS])]) -> Vec<i32> {
    let mut a = vec![0; MAX_AGENTS * HEADS];
    for (slot, heads) in list {
        a[slot * HEADS..(slot + 1) * HEADS].copy_from_slice(heads);
    }
    a
}

fn at(k: i32) -> usize {
    usize::try_from(k).unwrap()
}

fn mask_row(p: &Pool, slot: usize) -> Vec<Bool8> {
    p.outputs().unwrap().masks()[slot * MASK_LEN..(slot + 1) * MASK_LEN].to_vec()
}

/// A limit order at `offset - 4` ticks from the side's reference, of size
/// bucket 0.
const fn limit(side: Side, offset: i32) -> [i32; HEADS] {
    let k = match side {
        Side::Buy => kind::LIMIT_BUY,
        Side::Sell => kind::LIMIT_SELL,
    };
    [k, offset, 0, 0]
}

/// A config whose fundamental never moves.
fn still(agents: u32, latency: u32) -> LobConfig {
    LobConfig {
        up_permille: 0,
        down_permille: 0,
        ..config(agents, latency)
    }
}

#[test]
fn test_env_reset_masks_have_layout_and_dead_slots_false() {
    let mut p = pool(still(2, 0), 1, 7, None);
    let out = p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(out.masks().len(), MASK_LEN * MAX_AGENTS);
    let row = mask_row(&p, 0);
    let kinds: Vec<bool> = row[..KINDS as usize].iter().map(|m| m.get()).collect();
    // Noop and limits only: the book is empty, nothing to cancel.
    assert_eq!(kinds, [true, false, false, true, true, false]);
    assert!(mask_row(&p, 2).iter().all(|m| !m.get()));
}

#[test]
fn test_env_limit_orders_cross_and_trade_zero_sum() {
    let mut p = pool(still(2, 0), 1, 7, None);
    p.reset_all(Reseed::Keep).unwrap();
    // Agent 0 asks 1 lot at the mid (512).
    p.step(input(&acts(&[(0, limit(Side::Sell, 4))]))).unwrap();
    assert_eq!(p.state(0).unwrap().book.best(Side::Sell), Some(512));
    // Agent 0 sees a cancel; agent 1 sees a market buy, agent 0 does not
    // (its own ask is the only one).
    assert!(mask_row(&p, 0)[at(kind::CANCEL)].get());
    assert!(mask_row(&p, 0)[CANCEL_ROWS + 5 * 8].get());
    assert!(!mask_row(&p, 0)[at(kind::MARKET_BUY)].get());
    assert!(mask_row(&p, 1)[at(kind::MARKET_BUY)].get());
    // Agent 1 bids at the mid: it crosses.
    let out = p.step(input(&acts(&[(1, limit(Side::Buy, 4))]))).unwrap();
    let r = out.rewards();
    assert_eq!((r[0] + r[1]).to_bits(), 0.0_f32.to_bits());
    let s = p.state(0).unwrap();
    assert!(s.book.is_empty());
    assert_eq!(s.accounts[1].inventory, 1);
    assert_eq!(s.accounts[0].inventory, -1);
    assert_eq!(s.accounts[1].cash, 20_000 - 512);
    assert_eq!(s.trade_count, 1);
    assert_eq!(s.last_trade, 512);
}

#[test]
fn test_env_latency_delays_orders_by_latency_steps() {
    let mut p = pool(still(2, 2), 1, 7, None);
    p.reset_all(Reseed::Keep).unwrap();
    p.step(input(&acts(&[(0, limit(Side::Sell, 4))]))).unwrap();
    assert!(p.state(0).unwrap().book.is_empty());
    assert_eq!(p.state(0).unwrap().pending.len(), 1);
    p.step(input(&acts(&[]))).unwrap();
    assert!(p.state(0).unwrap().book.is_empty());
    p.step(input(&acts(&[]))).unwrap();
    assert_eq!(p.state(0).unwrap().book.len(), 1);
    assert!(p.state(0).unwrap().pending.is_empty());
}

#[test]
fn test_env_last_step_settles_and_truncates() {
    let cfg = LobConfig {
        steps: 3,
        ..still(2, 0)
    };
    let mut p = pool(cfg, 1, 7, None);
    p.reset_all(Reseed::Keep).unwrap();
    p.step(input(&acts(&[(0, limit(Side::Sell, 5))]))).unwrap();
    p.step(input(&acts(&[(1, limit(Side::Buy, 5))]))).unwrap();
    assert_eq!(p.state(0).unwrap().accounts[1].inventory, 1);
    let out = p.step(input(&acts(&[]))).unwrap();
    assert!(out.episode_truncated()[0].get());
    let s = p.state(0).unwrap();
    assert_eq!(s.accounts[0].inventory, 0);
    assert_eq!(s.accounts[1].inventory, 0);
    // Settled at the fundamental (512) after a trade at 513.
    assert_eq!(s.accounts[1].cash, 20_000 - 513 + 512);
    assert_eq!(s.accounts[0].cash, 20_000 + 513 - 512);
}

#[test]
fn test_env_bankruptcy_kills_agent_and_terminates() {
    let cfg = LobConfig {
        initial_cash: 0,
        allow_short_cash: true,
        bankrupt: 1,
        ..still(2, 0)
    };
    let mut p = pool(cfg, 1, 7, None);
    p.reset_all(Reseed::Keep).unwrap();
    // Agent 0 asks at 516; agent 1 lifts it: its equity at 512 is -4.
    p.step(input(&acts(&[(0, limit(Side::Sell, 8))]))).unwrap();
    let out = p.step(input(&acts(&[(1, limit(Side::Buy, 8))]))).unwrap();
    assert_eq!(out.rewards()[1].to_bits(), (-4.0_f32).to_bits());
    assert_eq!(out.rewards()[0].to_bits(), 4.0_f32.to_bits());
    assert!(!out.alive()[1].get());
    assert!(out.episode_terminated()[0].get());
    assert!(!p.state(0).unwrap().alive[1].get());
}

#[test]
fn test_env_reward_fundamental_only_change_is_hidden_until_settlement() {
    let steps = 10;
    let flat = LobConfig {
        steps,
        ..still(2, 0)
    };
    // Every walk draw moves up: the same draws as `flat` (one per step).
    // The signal draws that follow are one bounded draw per used slot,
    // whose count and bound do not depend on the fundamental (only their
    // stored values do), so both pools consume the same stream and the
    // arrival permutations match too.
    let drift = LobConfig {
        up_permille: 1000,
        ..flat.clone()
    };
    let mut a = pool(flat, 1, 7, None);
    let mut b = pool(drift, 1, 7, None);
    a.reset_all(Reseed::Keep).unwrap();
    b.reset_all(Reseed::Keep).unwrap();
    // Agent 0 asks at the mid, agent 1 lifts it: agent 1 holds 1 lot and
    // the book is empty, so the mid stays at 512; then only no-ops.
    let first = acts(&[(0, limit(Side::Sell, 4))]);
    let second = acts(&[(1, limit(Side::Buy, 4))]);
    let noop = acts(&[]);
    for step in 1..=steps {
        let action = match step {
            1 => &first,
            2 => &second,
            _ => &noop,
        };
        let ra = a.step(input(action)).unwrap().rewards().to_vec();
        let rb = b.step(input(action)).unwrap().rewards().to_vec();
        let bits = |r: &[f32]| r.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        if step < steps {
            assert_eq!(bits(&ra), bits(&rb), "step {step}");
            let (sa, sb) = (a.state(0).unwrap(), b.state(0).unwrap());
            assert_eq!(sa.fundamental, 512);
            assert_eq!(sb.fundamental, 512 + step);
            assert_eq!(sa.accounts, sb.accounts);
            assert_eq!(sb.touch().mid, 512);
        } else {
            // Settlement at 512 against 512 + steps.
            assert_ne!(bits(&ra), bits(&rb));
            assert_eq!(bits(&ra[..2]), [0.0_f32.to_bits(); 2]);
            assert_eq!(rb[1].to_bits(), 10.0_f32.to_bits());
            assert_eq!(rb[0].to_bits(), (-10.0_f32).to_bits());
        }
    }
    assert_eq!(a.state(0).unwrap().accounts[1].inventory, 0);
}
