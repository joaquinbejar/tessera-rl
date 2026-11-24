// SPDX-License-Identifier: MIT OR Apache-2.0

//! The privileged state (ADR-0079 item 7): its space has the documented
//! leaves, and a written buffer holds every account, every resting order
//! and queued item with its owner, and the fundamental, in canonical
//! order and independent of storage order.

#![expect(
    clippy::unwrap_used,
    reason = "test fixtures: a failure is the assertion"
)]

mod support;

use support::{Rng, config, input, legal_actions, pool};
use tessera_rl_core::io::ObsWriter;
use tessera_rl_core::space::Space;
use tessera_rl_engine::Reseed;
use tessera_rl_markets::privileged::{
    AGENT_COLUMNS, LobPrivileged, ORDER_COLUMNS, PENDING_COLUMNS, state_space, write_state,
};
use tessera_rl_markets::{LobConfig, LobState};

/// Leaf sizes in bytes: `agents`, `market`, `orders`, `orders_mask`,
/// `pending`, `pending_mask` (all i64).
fn sizes() -> [usize; 6] {
    let a = usize::try_from(AGENT_COLUMNS).unwrap();
    let o = usize::try_from(ORDER_COLUMNS).unwrap();
    let p = usize::try_from(PENDING_COLUMNS).unwrap();
    [16 * a * 8, 5 * 8, 128 * o * 8, 128 * 8, 64 * p * 8, 64 * 8]
}

fn written(cfg: &LobConfig, s: &LobState) -> Vec<Vec<i64>> {
    let mut bufs: Vec<Vec<u8>> = sizes().iter().map(|&n| vec![0u8; n]).collect();
    {
        let mut leaves: Vec<&mut [u8]> = bufs.iter_mut().map(Vec::as_mut_slice).collect();
        let mut out: ObsWriter<'_, '_, LobPrivileged> = ObsWriter::new(&mut leaves);
        write_state(cfg, s, &mut out);
    }
    bufs.iter()
        .map(|b| {
            let (chunks, _) = b.as_chunks::<8>();
            chunks.iter().map(|c| i64::from_le_bytes(*c)).collect()
        })
        .collect()
}

#[test]
fn test_privileged_space_is_a_dict_of_six_leaves() {
    let space = state_space(&config(4, 0)).unwrap();
    assert!(matches!(space, Space::Dict(ref entries) if entries.len() == 6));
}

#[test]
fn test_privileged_state_holds_accounts_orders_queue_and_fundamental() {
    let cfg = config(4, 2);
    let mut p = pool(cfg.clone(), 1, 5, None);
    let mut masks = p.reset_all(Reseed::Keep).unwrap().masks().to_vec();
    let mut rng = Rng(77);
    for _ in 0..25 {
        let a = legal_actions(&mut rng, &masks, 1);
        masks = p.step(input(&a)).unwrap().masks().to_vec();
    }
    let s = *p.state(0).unwrap();
    let leaves = written(&cfg, &s);
    let (agents, market, orders, orders_mask, pending, pending_mask) = (
        &leaves[0], &leaves[1], &leaves[2], &leaves[3], &leaves[4], &leaves[5],
    );
    for slot in 0..16 {
        let row = &agents[slot * 5..slot * 5 + 5];
        assert_eq!(row[1], s.accounts[slot].cash);
        assert_eq!(row[2], i64::from(s.accounts[slot].inventory));
    }
    assert_eq!(market[0], i64::from(s.fundamental));
    assert_eq!(market[1], i64::from(s.step));
    let n = s.book.len();
    assert_eq!(orders_mask.iter().sum::<i64>(), i64::try_from(n).unwrap());
    let ids: Vec<i64> = (0..n).map(|i| orders[i * 5 + 4]).collect();
    let sorted = {
        let mut v = ids.clone();
        v.sort_unstable();
        v
    };
    assert_eq!(ids, sorted);
    for i in 0..n {
        let seq = u32::try_from(orders[i * 5 + 4]).unwrap();
        let o = s.book.order(seq).unwrap();
        assert_eq!(orders[i * 5], i64::from(o.owner));
        assert_eq!(orders[i * 5 + 3], i64::from(o.qty));
    }
    let q = s.pending.len();
    assert_eq!(pending_mask.iter().sum::<i64>(), i64::try_from(q).unwrap());
    for i in 0..q {
        let owner = pending[i * 8];
        assert!((0..4).contains(&owner));
    }
    assert!(orders[n * 5..].iter().all(|&v| v == 0));
}
