// SPDX-License-Identifier: MIT OR Apache-2.0

//! The observation's `Set` leaves (ADR-0082, #450): `own_orders` and
//! `trades` export an `i32` item leaf `[MAX, 3]` and then a `bool` mask
//! leaf; present rows are packed from 0, absent rows are zero.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test fixtures: a failure is the assertion"
)]

mod support;

use support::{Pool, config, input, pool};
use tessera_rl_core::Environment;
use tessera_rl_core::space::Space;
use tessera_rl_engine::Reseed;
use tessera_rl_markets::actions::{HEADS, kind};
use tessera_rl_markets::limits::MAX_AGENTS;
use tessera_rl_markets::{LimitOrderBook, LobConfig};

/// Leaf indices in export order.
const OWN_ORDERS: usize = 3;
const OWN_ORDERS_MASK: usize = 4;
const TRADES: usize = 5;
const TRADES_MASK: usize = 6;

fn still() -> LobConfig {
    LobConfig {
        up_permille: 0,
        down_permille: 0,
        ..config(2, 0)
    }
}

/// The `i32` values of `agent`'s row of leaf `leaf`.
fn ints(p: &Pool, leaf: usize, agent: usize) -> Vec<i32> {
    let bytes = p.outputs().unwrap().observation(leaf).unwrap();
    let per = bytes.len() / MAX_AGENTS;
    bytes[agent * per..(agent + 1) * per]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}

/// The mask bytes of `agent`'s row of leaf `leaf`.
fn mask(p: &Pool, leaf: usize, agent: usize) -> Vec<u8> {
    let bytes = p.outputs().unwrap().observation(leaf).unwrap();
    let per = bytes.len() / MAX_AGENTS;
    bytes[agent * per..(agent + 1) * per].to_vec()
}

fn acts(slot: usize, heads: [i32; HEADS]) -> Vec<i32> {
    let mut a = vec![0; MAX_AGENTS * HEADS];
    a[slot * HEADS..(slot + 1) * HEADS].copy_from_slice(&heads);
    a
}

#[test]
fn test_observe_space_has_two_sets() {
    let space = LimitOrderBook::observation_space(&still()).unwrap();
    assert!(space.contains_set());
    assert!(matches!(space, Space::Dict(ref entries) if entries.len() == 6));
}

#[test]
fn test_observe_own_orders_and_trades_are_packed_set_rows() {
    let mut p = pool(still(), 1, 7, None);
    p.reset_all(Reseed::Keep).unwrap();
    assert_eq!(mask(&p, OWN_ORDERS_MASK, 0), vec![0; 8]);
    // Agent 0 asks 2 lots (size bucket 1) at the mid.
    p.step(input(&acts(0, [kind::LIMIT_SELL, 4, 1, 0])))
        .unwrap();
    let rows = ints(&p, OWN_ORDERS, 0);
    assert_eq!(rows.len(), 8 * 3);
    // Side 1 (sell), offset from the new mid, 2 lots; the rest zero.
    assert_eq!(rows[0], 1);
    assert_eq!(rows[2], 2);
    assert!(rows[3..].iter().all(|&v| v == 0));
    assert_eq!(mask(&p, OWN_ORDERS_MASK, 0), [1, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(mask(&p, OWN_ORDERS_MASK, 1), vec![0; 8]);
    // Agent 1 lifts 1 lot: one trade print for everyone, aggressor buy.
    p.step(input(&acts(1, [kind::MARKET_BUY, 0, 0, 0])))
        .unwrap();
    for agent in 0..2 {
        let m = mask(&p, TRADES_MASK, agent);
        assert_eq!(m[0], 1);
        assert!(m[1..].iter().all(|&b| b == 0));
        let t = ints(&p, TRADES, agent);
        assert_eq!(t.len(), 16 * 3);
        assert_eq!((t[1], t[2]), (1, 0));
        assert!(t[3..].iter().all(|&v| v == 0));
    }
    // Agent 0's remaining order: 1 lot.
    assert_eq!(ints(&p, OWN_ORDERS, 0)[2], 1);
}
